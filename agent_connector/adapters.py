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
        agent.add_mcp(config_path=cfg)


class NativeToolAdapter(BaseAdapter):
    """Inject wrapper functions into an agent that supports tool registration
    (add_tool / tools list / register). The agent's own loop drives them."""

    def connect(self, agent: Any) -> None:
        from agent_connector.generator import generate_wiring, load_wrappers

        wiring_dir = os.path.join(os.getcwd(), "wiring")
        generate_wiring(self.tools, self.schema, out_dir=wiring_dir)
        if wiring_dir not in sys.path:
            sys.path.insert(0, wiring_dir)
        wrappers = load_wrappers(package_name="generated_tools",
                                 registration_style="function")
        self._maybe_patch_a1(agent, wrappers)

        registered = 0
        for w in wrappers:
            if hasattr(agent, "add_tool"):
                try:
                    agent.add_tool(w)
                    registered += 1
                    continue
                except Exception:
                    pass
            if hasattr(agent, "tools"):
                if agent.tools is None:
                    agent.tools = []
                agent.tools.append(w)
                registered += 1
        if registered == 0:
            raise RuntimeError("NativeToolAdapter: no tools could be injected.")

    def _maybe_patch_a1(self, agent: Any, wrappers) -> None:
        cls = (self.schema.get("agent_class") or type(agent).__name__).lower()
        if "a1" not in cls and "biomni" not in (self.schema.get("module_path") or "").lower():
            return
        try:
            import ast as _ast
            import biomni.agent.a1 as _a1mod
        except Exception:
            return
        spec_by_name = {}
        for w in wrappers:
            s = getattr(w, "_TOOL_SPEC", None)
            if s:
                spec_by_name[getattr(w, "__name__", s.get("name"))] = s
                if s.get("name"):
                    spec_by_name[s["name"]] = s
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


def build_runtime(agent, schema, tools, agent_dir, *, model=None, base_url=None,
                  api_key=None, openai_client=None):
    """Select the right Adapter from capabilities, inject tools into the
    (already instantiated) downstream agent, return (agent, info). The caller
    then drives it with run_agent. Our system never substitutes its own loop."""
    caps = (schema or {}).get("capabilities") or {}
    mcp = caps.get("mcp") or {}
    native = caps.get("native_tool_calling") or {}
    code = caps.get("code_execution") or {}
    config = caps.get("config_wiring") or {}

    info = {"supports_mcp": bool(mcp.get("supported")), "agent_dir": agent_dir}
    if mcp.get("supported") and agent is not None and hasattr(agent, "add_mcp"):
        adapter = MCPAdapter(tools, schema, model=model, base_url=base_url,
                             api_key=api_key, openai_client=openai_client)
    elif native.get("supported") and agent is not None:
        adapter = NativeToolAdapter(tools, schema, model=model, base_url=base_url,
                                    api_key=api_key, openai_client=openai_client)
    elif code.get("supported") and agent is not None:
        adapter = CodeExecutionAdapter(tools, schema, model=model, base_url=base_url,
                                       api_key=api_key, openai_client=openai_client)
    elif config.get("supported") and agent is not None:
        adapter = ConfigAdapter(tools, schema, model=model, base_url=base_url,
                                api_key=api_key, openai_client=openai_client)
    else:
        adapter = PromptAdapter(tools, schema, model=model, base_url=base_url,
                               api_key=api_key, openai_client=openai_client)

    info["adapter"] = adapter.name
    if agent is None:
        info["error"] = ("No downstream agent was instantiated (create_agent failed); "
                         "cannot drive tools.")
        info["driver"] = "none"
        return agent, info
    try:
        adapter.connect(agent)
        info["injected"] = True
    except Exception as exc:
        info["inject_error"] = f"{type(exc).__name__}: {exc}"
        info["driver"] = "error"
        return agent, info
    info["driver"] = "downstream-agent"
    return agent, info
