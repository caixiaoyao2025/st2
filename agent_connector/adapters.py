"""Adapters connect discovered tools to a DOWNSTREAM bio-agent and let the
agent's own planner / tool-loop drive them.

Design rules (per project architecture):
  * The adapter ONLY does: agent creation (where the agent has no single
    go()/run() entry), tool injection, MCP/config/prompt/wrapper conversion,
    and environment preparation.
  * The adapter NEVER implements a planner, tool selection, a multi-turn agent
    loop, or its own LLM calls. That is the downstream agent's job.
  * After injection, the caller drives the agent with run_agent(agent, prompt).
    The entry contract is NOT hardcoded: run_agent defers to the capability-
    selected adapter (e.g. BioChatterAdapter uses conversation.query), and only
    falls back to probing go/run/invoke when no dedicated adapter is present.
    Our system never substitutes its own loop.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Default probe order for agents whose entry contract was NOT discovered as a
# dedicated adapter. This is a FALLBACK, not the canonical contract -- agents
# like BioChatter that expose conversation.query(prompt, tools=[...]) are
# served by their own adapter instead.
_NATIVE_ENTRIES = ("go", "run", "execute", "predict", "forward", "invoke", "kickoff", "__call__")

# Extensions that likely denote data files the downstream agent may need.
# Used to build the pre-go() environment context so the agent does NOT have to
# discover files itself via `find`/shell (which is exactly what destabilised
# A1's code-execution loop in testing).
_DATA_EXTS = {
    ".fasta", ".fa", ".fna", ".ffn", ".frn", ".fq", ".fastq", ".fastq.gz",
    ".gb", ".gbk", ".gbff", ".gff", ".gtf", ".gff3", ".vcf", ".vcf.gz",
    ".bam", ".sam", ".bed", ".wig", ".bedgraph", ".pdb", ".cif",
    ".msa", ".aln", ".phy", ".nex", ".newick", ".nwk", ".tree",
    ".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".h5", ".hdf5",
    ".bcf", ".cram", ".maf", ".clustal",
}


def build_environment_context(roots: list[str] | None = None,
                               tool_names: list[str] | None = None,
                               max_files: int = 300) -> str:
    """Build a concise environment-context block handed to the agent BEFORE it
    runs, so it never has to probe the filesystem itself.

    Includes: the working directory, the available data files (absolute paths,
    filtered to bioinformatics-relevant extensions), and the names of the
    injected tools. Returned as ready-to-prepend text (empty string if nothing
    to report).
    """
    import os
    roots = roots or [".", "uploads", "data"]
    found: list[str] = []
    cwd = os.getcwd()
    for root in roots:
        r = root if os.path.isabs(root) else os.path.join(cwd, root)
        if not os.path.isdir(r):
            continue
        for dirpath, _dirs, files in os.walk(r):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in _DATA_EXTS:
                    found.append(os.path.join(dirpath, fn))
                    if len(found) >= max_files:
                        break
            if len(found) >= max_files:
                break
        if len(found) >= max_files:
            break
    if not found and not tool_names:
        return ""
    lines = ["[Environment context -- provided by the runtime, do NOT rediscover "
             "these files with shell/file commands]:",
             f"- working_dir: {cwd}"]
    if found:
        lines.append("- available data files (absolute paths):")
        for p in found:
            lines.append(f"    {os.path.basename(p)} -> {p}")
    if tool_names:
        lines.append("- injected tools callable in your namespace:")
        for t in tool_names:
            lines.append(f"    {t}")
    lines.append("")
    return "\n".join(lines)


def native_entry(agent: Any, candidates: list[str] | None = None) -> Any:
    """Return the downstream agent's native run method.

    ``candidates`` lets a discovered entry contract override the default probe
    order (e.g. an agent whose entry is ``query`` rather than ``go``/``run``).
    """
    entries = list(candidates) if candidates else list(_NATIVE_ENTRIES)
    for m in entries:
        if m == "__call__":
            if callable(agent):
                return agent
            continue
        fn = getattr(agent, m, None)
        if callable(fn):
            return fn
    return None


def run_agent(agent: Any, prompt: str, adapter: Any = None,
              env_context: str | None = None) -> Any:
    """Drive the DOWNSTREAM agent's own planner/tool-loop.

    The entry contract is NOT hardcoded here. We prefer the adapter that was
    selected by the capability registry (it knows how THIS agent runs -- e.g.
    BioChatter's ``conversation.query``); otherwise we probe the agent for a
    native entry method. Our system never substitutes its own loop.

    Before driving the agent we prepend a small environment-context block
    (working dir + data-file paths + injected tool names) so code-execution
    agents like Biomni A1 start with the facts they need (file locations, tool
    namespace) instead of shelling out to `find` and destabilising their own
    execute loop.
    """
    ad = adapter or getattr(agent, "_st2_adapter", None)
    if env_context is None and ad is not None:
        try:
            tool_names = [t.get("name") for t in getattr(ad, "tools", []) or []]
        except Exception:
            tool_names = None
        env_context = build_environment_context(tool_names=tool_names)
    effective_prompt = (env_context + "\n\n" + prompt) if env_context else prompt
    if ad is not None and hasattr(ad, "run"):
        return ad.run(agent, effective_prompt)
    fn = native_entry(agent)
    if fn is None:
        raise RuntimeError(
            "Downstream agent exposes no recognised native entry. Its entry "
            "contract must be discovered by the scanner and served by a dedicated "
            "adapter (e.g. BioChatterAdapter uses conversation.query), rather than "
            "assuming go/run/invoke."
        )
    return fn(effective_prompt)


def _write_mcp_config() -> str:
    """Ensure mcp_config_cluster.yaml exists, pointing server.py at THIS repo."""
    _cwd = os.getcwd()
    cfg_path = os.path.join(_cwd, "mcp_config_cluster.yaml")
    if os.path.exists(cfg_path):
        return cfg_path
    env = {
        "MCP_APP_ROOT": _cwd,
        "MCP_DATA_ROOT": _cwd,
        "MCP_REGISTRY_PATH": os.path.join(_cwd, "registry.yaml"),
        "MCP_USER_REGISTRY_PATH": os.path.join(_cwd, "data", "mcp_registry.yaml"),
        "MCP_TRANSPORT": "stdio",
    }
    lines = ["mcp_servers:\n", "  bio-mcp:\n", "    enabled: true\n", "    command:\n",
             "      - python\n", "      - server.py\n", "    env:\n"]
    for k, v in env.items():
        lines.append(f"      {k}: {v}\n")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return cfg_path


class BaseAdapter:
    def __init__(self, tools, schema, *, model=None, base_url=None, api_key=None,
                 openai_client=None):
        self.tools = tools or []
        self.schema = schema or {}
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.openai_client = openai_client

    def create_agent(self, **kwargs) -> Any:
        """Optional: build the downstream agent object for this interface.

        Most adapters expect the notebook to have instantiated the agent
        already; interfaces without a single go()/run() entry (e.g. BioChatter)
        override this to construct their Conversation/backend."""
        raise NotImplementedError(
            f"{self.name} does not create an agent; pass an instantiated agent "
            "to build_runtime, or implement create_agent()."
        )

    def connect(self, agent: Any) -> None:
        raise NotImplementedError

    def run(self, agent: Any, prompt: str) -> Any:
        """Default entry contract: call the agent's discovered native entry.

        Prefers the scanner-discovered ``execution_method`` (so an agent whose
        real entry is e.g. ``query`` is honoured) and only falls back to the
        generic go/run/invoke probe order otherwise."""
        ti = (self.schema or {}).get("tool_interface") or {}
        entry = ti.get("execution_method")
        candidates = [entry] if entry else None
        fn = native_entry(agent, candidates)
        if fn is None:
            raise RuntimeError(
                f"{self.name}: agent exposes no native entry "
                f"(execution_method={entry!r}, tried {_NATIVE_ENTRIES})."
            )
        return fn(prompt)

    @property
    def name(self) -> str:
        return type(self).__name__


class MCPAdapter(BaseAdapter):
    """Expose our FastMCP server (server.py) to an MCP-capable agent via
    add_mcp. The agent's OWN planner then selects and calls the MCP tools."""

    def connect(self, agent: Any) -> None:
        cfg = _write_mcp_config()
        if not hasattr(agent, "add_mcp"):
            raise RuntimeError("Agent reports MCP support but has no add_mcp().")
        # Tool registration over MCP makes ZERO LLM calls: the MCP server
        # already exposes our canonical schemas (data/mcp_registry.yaml). Only
        # the agent's own planner (agent.go) needs the LLM later.
        errors = []
        for _call in (lambda: agent.add_mcp(config_path=cfg),
                      lambda: agent.add_mcp(config=cfg)):
            try:
                _call()
                return
            except TypeError as exc:
                errors.append(str(exc))
                continue
        raise RuntimeError(
            "agent.add_mcp(...) accepted none of the expected signatures "
            f"(config_path=, config=). Errors: {errors}"
        )


class NativeToolAdapter(BaseAdapter):
    """Inject wrapper functions into an agent that supports tool registration
    (add_tool / tools list / register / bind_tools). The agent's own loop
    drives them. The exact wrapper *shape* follows the agent's ``schema_format``
    (interface contract), not the framework name."""

    def connect(self, agent: Any) -> None:
        from agent_connector.generator import generate_wiring, load_wrappers

        fmt = (self.schema.get("tool_interface") or {}).get("schema_format") or "function"
        wiring_dir = os.path.join(os.getcwd(), "wiring")
        generate_wiring(self.tools, self.schema, out_dir=wiring_dir)
        if wiring_dir not in sys.path:
            sys.path.insert(0, wiring_dir)
        wrappers = load_wrappers(package_name="generated_tools",
                                registration_style="function")

        # A1 (biomni) parses a function's source/docstring itself via
        # function_to_api_schema to build the schema dict. It may be detected
        # as `mcp`/`function` format. _maybe_patch_a1 only patches when BOTH
        # (a) this is Biomni A1 and (b) our wrappers carry canonical
        # _TOOL_SPEC -- so we never override the official add_tool() behaviour
        # for unrecognised tools. The patch makes add_tool() build the schema
        # from _TOOL_SPEC with ZERO LLM calls (fixes
        # "Generated schema is not a dictionary" on quota failure).
        self._maybe_patch_a1(agent, wrappers)

        # Reshape wrappers to the interface's expected tool schema where possible.
        wrapped = self._adapt_wrappers(wrappers, fmt)

        # Per-tool injection report: the success criterion is NOT "N tools
        # injected" but "N attempted / N accepted by agent.add_tool / N appear
        # in the agent's registry". The downstream planner (A1.go) then calls
        # them by name in its generated <execute> blocks.
        report = {"attempted": 0, "accepted": 0, "rejected": 0,
                  "by_method": {}, "schema_format": fmt, "details": []}
        for w in wrapped:
            report["attempted"] += 1
            name = getattr(w, "__name__", "tool")
            ok = False
            for method, call in (
                ("add_tool", lambda: agent.add_tool(w)),
                ("bind_tools", lambda: agent.bind_tools([w])),
            ):
                if not hasattr(agent, method):
                    continue
                try:
                    call()
                    ok = True
                    report["by_method"][method] = report["by_method"].get(method, 0) + 1
                    break
                except Exception as e:  # noqa: BLE001
                    report["details"].append({"tool": name, "method": method, "error": str(e)})
            if not ok and hasattr(agent, "tools"):
                try:
                    if agent.tools is None:
                        agent.tools = []
                    agent.tools.append(w)
                    ok = True
                    report["by_method"]["tools_append"] = (
                        report["by_method"].get("tools_append", 0) + 1)
                except Exception as e:  # noqa: BLE001
                    report["details"].append({"tool": name, "method": "tools_append", "error": str(e)})
            if ok:
                report["accepted"] += 1
            else:
                report["rejected"] += 1
        self._inject_report = report
        if report["accepted"] == 0:
            raise RuntimeError(
                "NativeToolAdapter: no tools could be injected into the agent. "
                f"report={report}"
            )

    @staticmethod
    def _adapt_wrappers(wrappers, schema_format: str) -> list:
        """Convert the canonical function wrappers into the shape the target
        agent expects. Falls back to the raw functions when the required
        library (e.g. langchain) is unavailable."""
        if schema_format in ("structured_tool", "langchain_tool", "openai_function"):
            try:
                from langchain_core.tools import StructuredTool
            except Exception:
                return wrappers
            out = []
            for w in wrappers:
                spec = getattr(w, "_TOOL_SPEC", {}) or {}
                try:
                    out.append(StructuredTool.from_function(
                        func=w,
                        name=spec.get("name") or getattr(w, "__name__", "tool"),
                        description=spec.get("description") or (getattr(w, "__doc__", "") or ""),
                    ))
                except Exception:
                    out.append(w)
            return out
        return wrappers

    def _maybe_patch_a1(self, agent: Any, wrappers) -> None:
        cls = (self.schema.get("agent_class") or type(agent).__name__).lower()
        mod_path = (self.schema.get("module_path") or type(agent).__module__ or "").lower()
        if "a1" not in cls and "biomni" not in mod_path:
            return
        # Resolve Biomni's modules from the ALREADY-imported process image (the
        # agent object exists, so biomni is loaded). We deliberately avoid any
        # literal `import biomni...` so the adapter stays framework-agnostic and
        # never triggers a fresh biomni import outside an A1 session.
        import sys, importlib
        agent_mod_name = type(agent).__module__ or ""
        candidate_mods = [agent_mod_name, "biomni.utils", "biomni.agent.a1"]
        resolved = {}
        for mname in candidate_mods:
            m = sys.modules.get(mname)
            if m is None:
                try:
                    m = importlib.import_module(mname)
                except Exception:
                    m = None
            if m is not None and hasattr(m, "function_to_api_schema"):
                resolved[mname] = m
        if not resolved:
            return
        import ast as _ast
        # Build a name->spec index. Prefer a live _TOOL_SPEC attribute on the
        # wrapper; if that is missing after reload, also record by wrapper name so
        # the source-parse below can still match. We intentionally do NOT bail
        # when this is empty -- the patch is applied regardless so that
        # function_to_api_schema never calls the (broken) LLM schema generator.
        spec_by_name = {}
        for w in wrappers:
            s = getattr(w, "_TOOL_SPEC", None)
            if s:
                spec_by_name[getattr(w, "__name__", s.get("name"))] = s
                if s.get("name"):
                    spec_by_name[s["name"]] = s

        # ROBUST EXTRACTION: A1 calls function_to_api_schema(inspect.getsource(func)),
        # and getsource(func) returns ONLY the `def` block -- the module-level
        # `_TOOL_SPEC = {...}` assignment is NOT included. So parsing that string
        # alone finds nothing. Instead, parse the FULL source of the generated
        # wrapper module(s) (which DO contain every `_TOOL_SPEC = {...}`), and
        # index by tool name. This works regardless of whether the wrapper object
        # still carries the _TOOL_SPEC attribute after reload.
        import inspect as _inspect
        module_src = {}
        for w in wrappers:
            mod = _inspect.getmodule(w)
            if mod is None or not getattr(mod, "__name__", ""):
                continue
            if mod.__name__ in module_src:
                continue
            try:
                module_src[mod.__name__] = _inspect.getsource(mod)
            except Exception:
                continue
        for src in module_src.values():
            try:
                mtree = _ast.parse(src)
            except Exception:
                continue
            for node in mtree.body:
                if isinstance(node, _ast.Assign):
                    for t in node.targets:
                        if isinstance(t, _ast.Name) and t.id == "_TOOL_SPEC":
                            try:
                                s = _ast.literal_eval(node.value)
                            except Exception:
                                s = None
                            if isinstance(s, dict) and s.get("name"):
                                spec_by_name[s["name"]] = s

        # Capture the ORIGINAL implementation before overriding every binding.
        _orig = next(iter(resolved.values())).function_to_api_schema

        def _ast_get_doc(node):
            """Extract a function/class docstring from an ast node, if any."""
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], _ast.Expr):
                try:
                    return _ast.literal_eval(body[0].value)
                except Exception:
                    return None
            return None

        def _args_from_def(node):
            """LLM-free fallback: build a parameter-name schema from the
            function signature when no _TOOL_SPEC is available."""
            params = []
            arguments = getattr(node, "args", None)
            for a in getattr(arguments, "args", []) or []:
                if getattr(a, "arg", None):
                    params.append(a.arg)
            return params

        def _reliable(function_string, llm):
            spec = None
            try:
                tree = _ast.parse(function_string)
                for node in tree.body:
                    if isinstance(node, _ast.Assign):
                        for t in node.targets:
                            if isinstance(t, _ast.Name) and t.id == "_TOOL_SPEC":
                                try:
                                    spec = _ast.literal_eval(node.value)
                                except Exception:
                                    spec = None
                    elif isinstance(node, (_ast.FunctionDef, _ast.ClassDef)) and spec is None:
                        # The wrapper source always carries `_TOOL_SPEC = {...}`
                        # (see generator.py), but fall back to the name index or
                        # the raw signature so we never hit the LLM path.
                        spec = spec_by_name.get(node.name)
                        if spec is None:
                            params = _args_from_def(node)
                            if params:
                                spec = {
                                    "name": node.name,
                                    "description": (
                                        (_ast_get_doc(node) or "").strip()
                                        or f"Auto-discovered tool {node.name}"
                                    ),
                                    "inputs": {p: {"required": True,
                                                    "description": p} for p in params
                                               if p not in ("kwargs",)},
                                }
            except Exception:
                spec = None
            if spec is None:
                # Truly unknown function: fall back to official behaviour (last
                # resort). This keeps the patch safe for non-our tools.
                return _orig(function_string, llm)
            req, opt = [], []
            for p, m in (spec.get("inputs") or {}).items():
                if p == "subcommand":
                    continue
                mm = m or {}
                entry = {"name": p, "type": "str", "description": mm.get("description", ""),
                         "default": None}
                (req if mm.get("required") else opt).append(entry)
            return {"name": spec.get("name"), "description": spec.get("description", ""),
                    "required_parameters": req, "optional_parameters": opt}

        # Patch every module binding of function_to_api_schema: a1.py may call
        # it as a module global (from `from biomni.utils import
        # function_to_api_schema`) OR qualified as `biomni.utils....`. Cover all
        # loaded modules that expose the attribute.
        for m in resolved.values():
            m.function_to_api_schema = _reliable


class CodeExecutionAdapter(BaseAdapter):
    """Place wrapper/tool functions into the agent's execution namespace and give
    it tool docs. The agent generates & executes its own code. No loop from us."""

    def connect(self, agent: Any) -> None:
        from agent_connector.generator import generate_wiring, load_wrappers

        wiring_dir = os.path.join(os.getcwd(), "wiring")
        generate_wiring(self.tools, self.schema, out_dir=wiring_dir)
        if wiring_dir not in sys.path:
            sys.path.insert(0, wiring_dir)
        wrappers = load_wrappers(package_name="generated_tools",
                                 registration_style="function")
        ns = getattr(agent, "namespace", None) or getattr(agent, "globals", None) or agent
        for w in wrappers:
            setattr(ns, getattr(w, "__name__", "tool"), w)
        try:
            agent.tool_catalog = [(getattr(w, "__name__", ""), getattr(w, "__doc__", ""))
                                  for w in wrappers]
        except Exception:
            pass


class ConfigAdapter(BaseAdapter):
    """Write a config file the agent loads at init. Injection is declarative."""

    def connect(self, agent: Any) -> None:
        pass


class PromptAdapter(BaseAdapter):
    """No programmatic injection possible: hand the agent tool descriptions.
    Still driven by agent.go() -- we never run a loop."""

    def connect(self, agent: Any) -> None:
        try:
            agent.discovered_tool_descriptions = [
                (t.get("name"), t.get("description")) for t in self.tools
            ]
        except Exception:
            pass


class BioChatterAdapter(BaseAdapter):
    """BioChatter is a Conversation/LLM backend, NOT an Agent with a ``go()``
    method. Its entry method differs by version: legacy ``Conversation`` uses
    ``query``, while modern ``DynamicAgent`` does not. We therefore DISCOVER
    the agent's real callable entry (query/run/ask/invoke/answer/chat) instead
    of hardcoding one, and pass the discovered tool specs where the signature
    allows.

    This adapter builds the agent, registers the discovered tools, and drives it
    via its own entry -- it never assumes ``agent.go()`` and the runtime never
    substitutes its own planner/loop.
    """

    def create_agent(self, **kwargs) -> Any:
        # Build an OpenAI-compatible chat model and a BioChatter Conversation.
        # Adjust the import paths to your installed BioChatter version.
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "BioChatterAdapter needs langchain_openai to build the LLM "
                f"backend: {exc}"
            )
        llm = ChatOpenAI(
            model=self.model or "minimax-m3",
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
        )
        try:
            from biochatter.dynamic_agent import DynamicAgent
        except Exception:
            try:
                from biochatter.conversation import Conversation
            except Exception:
                try:
                    from biochat import Conversation  # older package layout
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Could not import BioChatter's agent class "
                        "(DynamicAgent / Conversation). Install biochatter and "
                        f"align the import with your version: {exc}"
                    )
        # Modern BioChatter exposes `DynamicAgent`; older versions `Conversation`.
        # Both are constructed with the LLM backend. Some versions also accept a
        # `tools=` keyword at construction -- pass our specs when supported.
        specs = [
            {
                "name": t.get("name"),
                "description": t.get("description"),
                "parameters": t.get("parameters") or {},
            }
            for t in self.tools
        ]
        try:
            agent_cls = DynamicAgent
        except NameError:
            agent_cls = Conversation
        try:
            conv = agent_cls(llm, tools=specs)
        except TypeError:
            try:
                conv = agent_cls(llm)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to construct BioChatter agent ({agent_cls.__name__}): "
                    f"{exc}"
                )
        return conv

    def connect(self, agent: Any) -> None:
        # Store the canonical tool specs; BioChatter receives them either at
        # construction (handled in create_agent) or via a tool-registration
        # method. Try every known registration entry; ignore the rest.
        self._tool_specs = [
            {
                "name": t.get("name"),
                "description": t.get("description"),
                "parameters": t.get("parameters") or {},
            }
            for t in self.tools
        ]
        for attr in ("set_tools", "add_tools", "register_tools", "set_functions"):
            fn = getattr(agent, attr, None)
            if callable(fn):
                try:
                    fn(self._tool_specs)
                    break
                except Exception:
                    continue

    def run(self, agent: Any, prompt: str) -> Any:
        """Drive the BioChatter agent WITHOUT assuming a fixed method name.

        BioChatter's entry method differs by version: legacy `Conversation` uses
        ``query``, the modern ``DynamicAgent`` does not. We discover the agent's
        real callable entry (query/run/ask/invoke/answer/chat/__call__) and call
        it, optionally passing our tool specs where the signature allows. This is
        the same "discover the entry contract, don't hardcode it" principle as
        the rest of the runtime.
        """
        specs = getattr(self, "_tool_specs", [])
        candidates = ("query", "run", "ask", "invoke", "answer", "chat", "__call__")
        for name in candidates:
            fn = getattr(agent, name, None)
            if not callable(fn):
                continue
            # Try tool-carrying signatures first, then a plain call.
            for call in (
                lambda: fn(prompt, tools=specs),
                lambda: fn(prompt, tool=specs),
                lambda: fn(prompt, tools=specs, functions=specs),
                lambda: fn(prompt),
            ):
                try:
                    return call()
                except TypeError:
                    continue
            # If it's callable but none of the above signatures matched, the
            # plain call would have run above; fall through to next candidate.
        raise RuntimeError(
            "BioChatter agent has no recognised callable entry method "
            f"(tried {candidates}). Its installed version's interface is "
            "unexpected -- align BioChatterAdapter to it."
        )


def resolve_adapter(schema: dict, caps: dict):
    """Deprecated: kept for backwards compatibility. Use
    ``agent_connector.adapter_registry.find_adapter`` which raises on unknown
    interfaces instead of silently guessing."""
    from agent_connector.adapter_registry import find_adapter, UnsupportedAgentInterface
    try:
        return find_adapter(schema, caps)
    except UnsupportedAgentInterface:
        # Older fallback behaviour; new callers should expect the exception.
        return PromptAdapter, "prompt"


def build_runtime(agent, schema, tools, agent_dir, *, model=None, base_url=None,
                  api_key=None, openai_client=None):
    """Select the adapter from the curated interface registry, inject tools into
    the (already instantiated) downstream agent, return (agent, info).

    If the detected agent interface is not in the supported set, ``find_adapter``
    raises ``UnsupportedAgentInterface`` -- we do NOT guess a schema. The caller
    then drives the agent with ``run_agent``; our system never substitutes its
    own planner/tool-loop."""
    from agent_connector.adapter_registry import find_adapter, UnsupportedAgentInterface

    caps = (schema or {}).get("capabilities") or {}
    try:
        adapter_cls, contract = find_adapter(schema, caps)
    except UnsupportedAgentInterface:
        # Propagate clearly so the notebook / caller sees exactly what to fix.
        raise

    info = {
        "supports_mcp": bool((caps.get("mcp") or {}).get("supported")),
        "tool_interface": (schema or {}).get("tool_interface"),
        "schema_format": ((schema or {}).get("tool_interface") or {}).get("schema_format"),
        "adapter_contract": contract,
        "agent_dir": agent_dir,
    }
    if agent is None:
        # Some interfaces (e.g. BioChatter) are created by their adapter rather
        # than pre-instantiated in the notebook. Try that before giving up.
        try:
            agent = adapter.create_agent(
                model=model, base_url=base_url, api_key=api_key,
                openai_client=openai_client,
            )
            info["agent_created_by"] = adapter.name
        except Exception as exc:
            info["error"] = (
                "No downstream agent was instantiated and the adapter could not "
                f"create one ({type(exc).__name__}: {exc})."
            )
            info["driver"] = "none"
            return agent, info
    adapter = adapter_cls(tools, schema, model=model, base_url=base_url,
                          api_key=api_key, openai_client=openai_client)
    info["adapter"] = adapter.name
    # Attach the selected adapter to the agent so run_agent() can drive THIS
    # agent via its own entry contract (e.g. BioChatterAdapter -> query),
    # instead of assuming a hardcoded go/run/invoke method.
    try:
        setattr(agent, "_st2_adapter", adapter)
        info["entry_contract"] = "adapter:%s" % adapter.name
    except Exception:
        info["entry_contract"] = "probe:%s" % ", ".join(_NATIVE_ENTRIES)

    # SETUP-TIME tool dependency provisioning: install each tool's declared /
    # curated runtime deps into the execution environment (sys.executable, which
    # is also the interpreter A1's own <execute> blocks run in) BEFORE the agent
    # ever calls a tool. This is NOT done by the agent at runtime.
    from agent_connector.tool_runner import install_tool_runtime_dependencies
    info["tool_dependencies"] = install_tool_runtime_dependencies(tools)

    try:
        adapter.connect(agent)
        info["injected"] = True
        info["inject_report"] = getattr(adapter, "_inject_report", None)
    except Exception as exc:
        info["inject_error"] = f"{type(exc).__name__}: {exc}"
        info["driver"] = "error"
        return agent, info
    info["driver"] = "downstream-agent"
    return agent, info
