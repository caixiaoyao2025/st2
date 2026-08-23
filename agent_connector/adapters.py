"""Adapters connect discovered tools to a DOWNSTREAM bio-agent and let the
agent's own planner / tool-loop drive them.

Design rules (per project architecture):
  * The adapter ONLY does: tool injection, MCP/config/prompt/wrapper conversion,
    and environment preparation.
  * The adapter NEVER implements a planner, tool selection, a multi-turn agent
    loop, or its own LLM calls. That is the downstream agent's job.
  * After injection, the caller drives the agent with run_agent(agent, prompt)
    which simply calls the agent's native entry (agent.go(...) / run(...) /
    invoke(...) / ...). Our system never substitutes its own loop.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_NATIVE_ENTRIES = ("go", "run", "execute", "predict", "forward", "invoke", "kickoff", "__call__")


def native_entry(agent: Any) -> Any:
    """Return the downstream agent's native run method (go/run/invoke/...)."""
    for m in _NATIVE_ENTRIES:
        if m == "__call__":
            if callable(agent):
                return agent
            continue
        fn = getattr(agent, m, None)
        if callable(fn):
            return fn
    return None


def run_agent(agent: Any, prompt: str) -> Any:
    """Single unified entry point: drive the DOWNSTREAM agent's own planner/loop.
    Our system never runs its own planner here."""
    fn = native_entry(agent)
    if fn is None:
        raise RuntimeError(
            "Downstream agent exposes no native entry (go/run/invoke/...). "
            "Connect a real agent instead of falling back to our own loop."
        )
    return fn(prompt)


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

    def connect(self, agent: Any) -> None:
        raise NotImplementedError

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
        if "a1" not in cls and "biomni" not in (self.schema.get("module_path") or "").lower():
            return
        try:
            import ast as _ast
            import biomni.agent.a1 as _a1mod
            import biomni.utils as _utilmod
        except Exception:
            return
        spec_by_name = {}
        for w in wrappers:
            s = getattr(w, "_TOOL_SPEC", None)
            if s:
                spec_by_name[getattr(w, "__name__", s.get("name"))] = s
                if s.get("name"):
                    spec_by_name[s["name"]] = s
        # CAUTION: only override Biomni's official function_to_api_schema when
        # our OWN canonical wrappers are present. If a caller hands A1 ordinary
        # functions (no _TOOL_SPEC), we leave the official behaviour intact so
        # a future A1 version change cannot be silently broken by our patch.
        if not spec_by_name:
            return
        # Capture the ORIGINAL implementation before overriding either binding.
        _orig = _a1mod.function_to_api_schema

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
                        spec = spec_by_name.get(node.name)
            except Exception:
                spec = None
            if spec is None:
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

        # Patch BOTH bindings: a1.py may call `function_to_api_schema(...)` as a
        # module global (from `from biomni.utils import function_to_api_schema`)
        # OR `biomni.utils.function_to_api_schema(...)` qualified. Overriding the
        # module attribute on biomni.utils covers the qualified call path; the
        # a1 module attribute covers the unqualified one.
        _utilmod.function_to_api_schema = _reliable
        _a1mod.function_to_api_schema = _reliable


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
        info["error"] = ("No downstream agent was instantiated (create_agent failed); "
                         "cannot drive tools.")
        info["driver"] = "none"
        return agent, info
    adapter = adapter_cls(tools, schema, model=model, base_url=base_url,
                          api_key=api_key, openai_client=openai_client)
    info["adapter"] = adapter.name
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
