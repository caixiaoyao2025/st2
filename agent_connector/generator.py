"""Code generation: turn ToolSpec entries into wrapper classes and produce
an Adapter that injects them into an arbitrary agent framework.

Wrapper: one class per tool, exposing a method named after the agent's
detected execution interface (default: run). Execution is delegated to
tool_runner.run_tool_spec, so wrappers never reimplement tool logic.

Adapter: calls the agent's detected registration method for each tool.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_EXECUTION_METHOD = "run"

WRAPPER_TEMPLATE = '''"""Auto-generated wrapper for tool: {name}"""
import json as _json
from pathlib import Path as _Path

from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {spec_repr}


class {class_name}:
    """Tool wrapper backed by a ToolSpec (registry.yaml entry)."""

    def __init__(self):
        self.name = {name_json}
        self.description = {desc_json}

    def __repr__(self):
        return f"<{{self.__class__.__name__}} name={{self.name}}>"

    def {method}(self, **kwargs):
        return format_result(run_tool_spec(_TOOL_SPEC, kwargs))

    def call_with_args(self, tool_input):
        if isinstance(tool_input, str):
            tool_input = _json.loads(tool_input)
        return self.{method}(**tool_input)

    __call__ = {method}
    func = {method}
'''

FUNCTION_WRAPPER_TEMPLATE = '''"""Auto-generated function wrapper for tool: {name}"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {spec_repr}


def {func_name}({signature}) -> str:
    """{desc}"""
    return format_result(run_tool_spec(_TOOL_SPEC, {kwargs_dict}))
'''


def make_class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) + "Wrapper"


def generate_wrappers(
    tools: list[dict[str, Any]],
    out_dir: str = "generated_tools",
    execution_method: str = DEFAULT_EXECUTION_METHOD,
    registration_style: str = "object",
) -> list[str]:
    """Write one _wrapper.py per tool; return the list of module paths.

    registration_style='function' emits a plain Python function (for
    frameworks that introspect functions, e.g. Biomni's add_tool).
    Otherwise a class exposing run/invoke/__call__/func is emitted.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    init_path = Path(out_dir) / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    written: list[str] = []
    for tool in tools:
        # --- expand subcommand CLIs into one leaf wrapper per subcommand ---
        # e.g. bqtools -> bqtools_encode(input, output), bqtools_decode(input)
        # The leaf wrapper sets _active_subcommand so the unified runner
        # dispatches to the right subcommand's params. Inputs are scoped via
        # the canonical make_leaf_spec (same source the runner + agent test use).
        if tool.get("arg_style") == "subcommand" and tool.get("subcommand_details"):
            from agent_connector.tool_spec import make_leaf_spec  # noqa: PLC0415

            for sub in (tool.get("subcommand_details") or {}):
                leaf = make_leaf_spec(tool, sub)
                code = _emit_wrapper(leaf, registration_style, execution_method)
                filename = Path(out_dir) / f"{leaf['name']}_wrapper.py"
                filename.write_text(code, encoding="utf-8")
                written.append(str(filename))
            continue
        code = _emit_wrapper(tool, registration_style, execution_method)
        filename = Path(out_dir) / f"{tool['name']}_wrapper.py"
        filename.write_text(code, encoding="utf-8")
        written.append(str(filename))
    return written


def _function_signature(tool: dict) -> tuple[str, str]:
    """Build (python_signature, kwargs_dict_literal) for a function wrapper.

    Real parameter names are emitted from ToolSpec.inputs so that agents which
    synthesize a schema from the wrapper SOURCE (e.g. Biomni's A1.add_tool ->
    function_to_api_schema) see concrete variables and produce a parseable
    schema. The ToolSpec remains the single source of truth at runtime: the
    signature is only a hint for schema generation, and run_tool_spec still
    coerces every value itself.

    Required parameters (no default) MUST precede optional ones, or Python
    raises SyntaxError ("parameter without a default follows parameter with a
    default") -- required params are kept default-less so A1's schema generator
    lists them under required_parameters.
    """
    inputs = tool.get("inputs") or {}
    required: list[str] = []
    optional: list[str] = []
    assigns: list[str] = []
    for raw, meta in inputs.items():
        if raw in ("subcommand",):
            continue
        pid = _safe_identifier(raw)
        if (meta or {}).get("required"):
            required.append(f"{pid}: str")
        else:
            optional.append(f"{pid}: str = None")
        assigns.append(f"{repr(raw)}: {pid}")
    parts = required + optional
    sig = ", ".join(parts) if parts else ""
    kd = "{" + ", ".join(assigns) + "}" if assigns else "{}"
    return sig, kd


def _emit_wrapper(tool: dict, registration_style: str, execution_method: str) -> str:
    class_name = make_class_name(tool["name"])
    if registration_style == "function":
        # Real parameter names (from ToolSpec.inputs) so A1's function_to_api_schema
        # can synthesize a parseable schema; the ToolSpec stays the single source
        # of truth at runtime (run_tool_spec coerces values itself).
        sig, kd = _function_signature(tool)
        return FUNCTION_WRAPPER_TEMPLATE.format(
            name=tool["name"],
            func_name=_safe_identifier(tool["name"]),
            signature=sig,
            desc=tool.get("description", ""),
            spec_repr=repr(tool),
            kwargs_dict=kd,
        )
    return WRAPPER_TEMPLATE.format(
        class_name=class_name,
        name=tool["name"],
        name_json=json.dumps(tool["name"], ensure_ascii=False),
        desc_json=json.dumps(tool.get("description", ""), ensure_ascii=False),
        spec_repr=repr(tool),
        method=_safe_identifier(execution_method),
    )


def _safe_identifier(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not cleaned:
        cleaned = DEFAULT_EXECUTION_METHOD
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


from string import Template


ADAPTER_TEMPLATE = Template('''"""Auto-generated adapter for agent class: $agent_class"""
import importlib as _importlib


class $adapter_class:
    """Unified adapter: create_agent / register_tools / run.

    Auto-generated from the agent's source code.  The execution entrypoint
    ($execution_method) was detected by scanner.py.
    """

    def __init__(self, agent=None, agent_class_path="$module_path", agent_class_name="$agent_class"):
        self._agent_class_path = agent_class_path
        self._agent_class_name = agent_class_name
        self._reg_method = "$registration_method"
        self._exec_method = "$execution_method"
        self._init_defaults = $init_defaults
        self.agent = agent

    # -- lifecycle -----------------------------------------------------------

    def create_agent(self, **overrides):
        """Import the agent class and instantiate it with detected defaults."""
        import os, inspect
        _model = os.environ.get("OPENAI_MODEL", "")
        _api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("WESTLAKE_API_KEY")
        _base_url = os.environ.get("OPENAI_BASE_URL", "")

        # --- Set BIOMNI env vars BEFORE importing so default_config picks them up ---
        if _model and not os.environ.get("BIOMNI_LLM"):
            os.environ["BIOMNI_LLM"] = _model
        if _api_key and not os.environ.get("BIOMNI_CUSTOM_API_KEY"):
            os.environ["BIOMNI_CUSTOM_API_KEY"] = _api_key
        if _base_url and not os.environ.get("BIOMNI_CUSTOM_BASE_URL"):
            os.environ["BIOMNI_CUSTOM_BASE_URL"] = _base_url
        if not os.environ.get("BIOMNI_SOURCE"):
            os.environ["BIOMNI_SOURCE"] = "Custom"

        mod = _importlib.import_module(self._agent_class_path)
        cls = getattr(mod, self._agent_class_name)

        # --- Monkey-patch default_config in case it was created before env vars ---
        try:
            from biomni.config import default_config as _dc
            if _model:
                _dc.llm = _model
            if _base_url:
                _dc.base_url = _base_url
            if _api_key:
                _dc.api_key = _api_key
            _dc.source = "Custom"
        except Exception:
            pass

        kwargs = dict(self._init_defaults)
        for k in list(kwargs):
            kl = k.lower()
            if kl in ("llm", "model") and not kwargs[k]:
                kwargs[k] = _model or "gpt-4o"
            elif kl == "api_key" and not kwargs[k]:
                kwargs[k] = _api_key
            elif kl == "base_url" and not kwargs[k]:
                kwargs[k] = _base_url
            elif kl == "source" and not kwargs[k]:
                kwargs[k] = "Custom"
        try:
            sig = inspect.signature(cls.__init__)
            valid = set(sig.parameters.keys()) - {"self"}
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except (ValueError, TypeError):
            pass
        kwargs.update(overrides)
        self.agent = cls(**kwargs) if kwargs else cls()
        return self.agent

    def register_tools(self, agent, tools):
        """Inject tools: try agent's native method, fallback to direct append."""
        target = agent or self.agent
        # Try agent's native registration method
        reg_fn = getattr(target, self._reg_method, None)
        if reg_fn:
            for t in tools:
                try:
                    reg_fn(t)
                except Exception:
                    # Fallback: direct append to tools list
                    if not hasattr(target, 'tools') or target.tools is None:
                        target.tools = []
                    target.tools.append(t)
        else:
            if not hasattr(target, 'tools') or target.tools is None:
                target.tools = []
            for t in tools:
                target.tools.append(t)
        return len(tools)

    def install_tools(self, tools):
        """Legacy: inject tools into self.agent (backward compat)."""
        return self.register_tools(self.agent, tools)

    def run(self, agent=None, prompt=""):
        """Execute the agent with a prompt via its detected entrypoint."""
        target = agent or self.agent
        return getattr(target, self._exec_method)(prompt)
''')


def make_adapter_class_name(agent_class: str) -> str:
    return agent_class + "Adapter"


def generate_adapter(
    agent_class: str,
    registration_method: str,
    execution_method: str = "run",
    module_path: str = "",
    init_defaults: dict | None = None,
    out_path: str = "adapter.py",
) -> str:
    """Write the adapter module and return its path."""
    code = ADAPTER_TEMPLATE.safe_substitute(
        agent_class=agent_class,
        adapter_class=make_adapter_class_name(agent_class),
        registration_method=_safe_identifier(registration_method),
        execution_method=_safe_identifier(execution_method),
        module_path=module_path or "",
        init_defaults=repr(init_defaults or {}),
    )
    Path(out_path).write_text(code, encoding="utf-8")
    return out_path


def load_wrappers(package_name: str = "generated_tools", registration_style: str = "object") -> list[Any]:
    """Import every *_wrapper.py module and collect its wrapper callables.

    For registration_style='function', plain functions defined in the
    wrapper modules are collected. Otherwise Wrapper class instances are
    instantiated.
    """
    instances: list[Any] = []
    try:
        package = importlib.import_module(package_name)
    except (ImportError, ModuleNotFoundError):
        # No wrappers were generated (e.g. MCP mode serves tools via server.py),
        # or the package path isn't importable yet. Return empty so callers that
        # guard on the result keep working.
        return instances
    package_path = Path(package.__file__).parent
    for filename in sorted(os.listdir(package_path)):
        if not filename.endswith("_wrapper.py"):
            continue
        module = importlib.import_module(f"{package_name}.{filename[:-3]}")
        if registration_style == "function":
            for _name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__ and not _name.startswith("_"):
                    instances.append(obj)
        else:
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if _name.endswith("Wrapper") and obj.__module__ == module.__name__:
                    instances.append(obj())
    return instances


def load_adapter(agent_class: str, adapter_path: str = "adapter.py") -> Any:
    """Import the generated adapter module and return the Adapter class.

    Loads from the file path directly, so it works regardless of sys.path
    or whether adapter.py lives inside a package directory.
    """
    module_name = Path(adapter_path).stem
    spec = importlib.util.spec_from_file_location(module_name, Path(adapter_path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, make_adapter_class_name(agent_class))()


# --- fallback wiring for agents with no register method -------------------


def _tool_to_function_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-style function schema entry for a ToolSpec.

    Built EXCLUSIVELY from ToolSpec.inputs via the canonical
    function_property (positional/flag metadata rides along), so this can
    never produce a parameter name the runner doesn't know. The outputs
    contract is surfaced too: the caller sees what "success" produces."""
    from agent_connector.tool_spec import function_property  # noqa: PLC0415

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, meta in (tool.get("inputs") or {}).items():
        properties[name] = function_property(meta)
        # ONLY an explicit required: true is required -- matches
        # to_function_schemas in tool_agent_test.py (forcing every param
        # required hands the LLM a fake schema).
        if (meta or {}).get("required") is True:
            required.append(name)
    # surface the install / environment contract so the caller knows what to
    # set up before invoking the tool (see discovery_to_registry.py 'install')
    install = tool.get("install") or {}
    fn = {
        "name": tool["name"],
        "description": tool.get("description", "") or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    if tool.get("arg_style"):
        fn["arg_style"] = tool["arg_style"]
    if install:
        fn["install"] = install
    return {"type": "function", "function": fn}


def expand_subcommand_leaves(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand subcommand CLIs into one entry per subcommand.

    Same contract as tool_agent_test.to_function_schemas: the agent sees
    bqtools_encode(input, output) instead of a bare bqtools() with no params.
    Leaf inputs come from the SAME canonical make_leaf_spec every other
    consumer uses (single source of truth -- no per-file re-derivation).
    """
    from agent_connector.tool_spec import make_leaf_spec  # noqa: PLC0415

    expanded: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("arg_style") == "subcommand" and tool.get("subcommand_details"):
            for sub in (tool.get("subcommand_details") or {}):
                expanded.append(make_leaf_spec(tool, sub))
        else:
            expanded.append(tool)
    return expanded


def generate_tools_manifest(tools: list[dict[str, Any]], out_path: str = "tools_manifest.json") -> str:
    """Wiring style 'manifest': write a `tools=[...]` list (OpenAI function
    calling / LangChain compatible) and return the output path.

    Consume it with:
      with open(out_path) as f:
          tools_list = json.load(f)
      llm_kwargs = {"tools": tools_list, "tool_choice": "auto"}
    """
    manifest = [_tool_to_function_schema(t) for t in expand_subcommand_leaves(tools)]
    Path(out_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def generate_config_fragment(tools: list[dict[str, Any]], out_path: str = "tools_config.yaml") -> str:
    """Wiring style 'config': write a canonical tool-catalog config snippet that
    can be merged into an agent that reads tools from a config file.

    YAML by default; pass a path ending in .json to emit JSON instead. The
    shape is deliberately generic -- adjust key names to match the target
    agent's own config schema.
    """
    entries: dict[str, Any] = {}
    for tool in expand_subcommand_leaves(tools):
        inputs = {
            name: {"type": meta.get("type", "string"), "description": meta.get("description", "")}
            for name, meta in (tool.get("inputs") or {}).items()
        }
        entries[tool["name"]] = {
            "description": tool.get("description", "") or "",
            "execution": tool.get("execution") or {"type": tool.get("type"), "command": tool.get("command")},
            "inputs": inputs,
        }
    if str(out_path).endswith(".json"):
        Path(out_path).write_text(
            json.dumps({"tools": entries}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        import yaml

        Path(out_path).write_text(
            yaml.safe_dump({"tools": entries}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return out_path


def generate_prompt_block(tools: list[dict[str, Any]]) -> str:
    """Wiring style 'prompt': return a text block describing every tool that
    can be appended to the agent's system prompt.

    The agent reads the block and emits calls like:
      <tool name="samtools_flagstat" args='{"bam_path": "/data/x.bam"}'/>
    """
    lines = ["## Available tools", "Call a tool exactly as: "
             '<tool name="TOOL_NAME" args=\'{"param": "value"}\'/>', ""]
    for tool in expand_subcommand_leaves(tools):
        lines.append(f"### {tool['name']}")
        if tool.get("description"):
            lines.append(f"Description: {tool['description']}")
        inputs = tool.get("inputs") or {}
        if inputs:
            lines.append("Parameters:")
            for name, meta in inputs.items():
                lines.append(f"- {name} ({meta.get('type', 'string')}): {meta.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


WIRING_INSTRUCTIONS = {
    "manifest": (
        "No register method found. Pass the manifest to each LLM invocation: "
        "tools_list = json.load(open('{manifest}')); llm(..., tools=tools_list). "
        "Each entry is an OpenAI function-calling schema."
    ),
    "config": (
        "No register method found. Merge the emitted tool catalog into the "
        "agent's config file (key names may need renaming to match its schema)."
    ),
    "prompt": (
        "No register method found. Append the prompt block to the agent's "
        "system prompt; the agent calls tools by emitting <tool name=... args=.../>."
    ),
}


def _is_mcp_supported(schema: dict[str, Any], caps: dict[str, Any] | None) -> bool:
    """MCP is the highest-priority capability.

    Honours the structured ``capabilities`` block first, then falls back to a
    token heuristic for callers that haven't populated capabilities yet.
    """
    if caps and isinstance(caps.get("mcp"), dict):
        if caps["mcp"].get("supported"):
            return True
    if caps and caps.get("mcp") is True:
        return True
    hay = " ".join(str(x or "") for x in (
        schema.get("module_path"), schema.get("agent_class"),
        schema.get("registration_method"),
    )).lower()
    return any(t in hay for t in ("biochatter", "biomni", "langchain", "mcp"))


def generate_mcp_config(
    tools: list[dict[str, Any]],
    out_dir: str = "wiring",
    registry_path: str | None = None,
) -> str:
    """MCP mode artifact: how to launch the repo's FastMCP server (server.py).

    No per-agent wrapper/prompt generation is produced -- the registry tools are
    served over MCP, so the agent only needs an MCP client pointing here.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cfg = {
        "transport": "stdio",
        "command": ["python", "server.py"],
        "registry": registry_path or "data/mcp_registry.yaml",
        "tools_served": [t.get("name") for t in tools],
        "note": (
            "Tools are served by the repo's FastMCP server (server.py) from the "
            "registry. Point the agent's MCP client at this server -- no per-agent "
            "wrapper/prompt generation is required."
        ),
    }
    path = Path(out_dir) / "mcp_server_config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def generate_wiring(
    tools: list[dict[str, Any]],
    schema: dict[str, Any],
    out_dir: str = "wiring",
    registration_style: str | None = None,
) -> dict[str, Any]:
    """Top-level wiring dispatch based on the agent_schema capabilities.

    Priority (the architecture-adapter decision):

        1. mcp                  -> MCP server config only (tools served by server.py)
        2. native_tool_calling  -> wrappers + adapter (inject path)
        3. code_execution       -> prompt block (agent executes described tools)
        4. config_wiring        -> tool-catalog config fragment
        5. prompt_wiring        -> system-prompt tool block

    MCP is checked first because it is a cross-framework standard interface: when
    the agent supports MCP we never regenerate per-agent wrappers/prompts.
    """
    artifacts: dict[str, Any] = {}
    caps = schema.get("capabilities")
    mode: str

    # ---- 1) MCP (first-class, highest priority) -----------------------------
    if _is_mcp_supported(schema, caps):
        mode = "mcp"
        artifacts["mcp"] = generate_mcp_config(tools, out_dir=out_dir)
        instructions = (
            "MCP mode: tools are served by the repo's FastMCP server (server.py) from "
            "the registry. Point the agent's MCP client at the generated config -- no "
            "per-agent wrapper/prompt generation is needed. The runtime drives the "
            "tools over MCP (mcp-native when the agent is itself an MCP client, otherwise "
            "via our MCP tool-calling loop)."
        )
        return {
            "mode": mode,
            "artifacts": artifacts,
            "instructions": instructions,
            "capabilities": caps,
        }

    # ---- 2) native tool calling (adapter / inject path) ---------------------
    if schema.get("registration_method"):
        mode = "adapter"
        generated = generate_wrappers(
            tools,
            out_dir=os.path.join(out_dir, "generated_tools"),
            execution_method=schema.get("execution_method") or DEFAULT_EXECUTION_METHOD,
            registration_style=registration_style or schema.get("registration_style") or "object",
        )
        artifacts["wrappers"] = generated
        adapter_path = generate_adapter(
            schema.get("agent_class") or "Agent",
            schema["registration_method"],
            execution_method=schema.get("execution_method") or DEFAULT_EXECUTION_METHOD,
            module_path=schema.get("module_path") or "",
            init_defaults=schema.get("init_signature", {}).get("defaults") if schema.get("init_signature") else None,
            out_path=os.path.join(out_dir, "adapter.py"),
        )
        artifacts["adapter"] = adapter_path
        instructions = (
            f"Adapter = load_adapter('{schema.get('agent_class') or 'Agent'}', adapter_path=...); "
            f"agent = adapter.create_agent(); "
            f"adapter.register_tools(agent, wrappers); "
            f"result = adapter.run(agent, prompt)"
        )
    else:
        # ---- 3/4/5) code_execution / config / prompt -----------------------
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        caps_d = caps or {}
        # Capability priority: code_execution > config_wiring > prompt_wiring >
        # fall back to the legacy wiring_style heuristic (manifest/config/prompt).
        if caps_d.get("code_execution"):
            mode = "code"
        elif caps_d.get("config_wiring"):
            mode = "config"
        elif caps_d.get("prompt_wiring"):
            mode = "prompt"
        else:
            mode = schema.get("wiring_style") or "prompt"
        if mode == "manifest":
            artifacts["manifest"] = generate_tools_manifest(tools, out_path=os.path.join(out_dir, "tools_manifest.json"))
        elif mode == "config":
            artifacts["config"] = generate_config_fragment(tools, out_path=os.path.join(out_dir, "tools_config.yaml"))
        elif mode == "code":
            artifacts["prompt_block"] = generate_prompt_block(tools)
        else:
            artifacts["prompt_block"] = generate_prompt_block(tools)
        instructions = WIRING_INSTRUCTIONS.get(mode, WIRING_INSTRUCTIONS["prompt"]).format(**artifacts)

    return {"mode": mode, "artifacts": artifacts, "instructions": instructions, "capabilities": caps}
