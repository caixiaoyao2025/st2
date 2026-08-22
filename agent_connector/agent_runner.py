"""MCP-aware runtime selection for discovered bio-agents.

The harness supports a wide range of downstream bio-agents (Biomni, BioChatter,
CellAgent, GeneAgent, langchain-based agents, smolagents, ...). Their tool
integration contracts differ, so we centralise the "how do we actually drive the
tools?" decision here.

Two execution strategies are available:

* MCP (preferred whenever the agent *supports* MCP): tools are served through the repo's own FastMCP
  server (``server.py``) and driven via genuine MCP tool-calling. If the agent
  itself is an MCP client (it exposes ``add_mcp``), we attach it as a client and
  let *it* drive the tools with its own LLM -- the most native integration
  possible, and the exact server Biomni/BioChatter talk to. If the agent cannot
  be instantiated as an MCP client (e.g. BioChatter's base class is abstract and
  ``create_agent()`` fails), we fall back to driving the same MCP tools with our
  own OpenAI-compatible LLM through ``MCPToolBridge``. This keeps MCP the
  transport uniformly.

* Unified Tool Runner (recommended default / fallback): a retrieval -> LLM
  tool-call -> execute -> feedback loop that is fully agent-agnostic. The
  notebook keeps its own mature implementation of this, so ``build_runtime``
  returns ``None`` for the non-MCP case and lets the notebook drive.

The module is intentionally import-safe: importing it never triggers the heavy
MCP / langchain imports, so it can be used in environments where those
dependencies are absent.
"""

from __future__ import annotations

import os
from typing import Any, Callable


# Tokens that indicate an agent framework is capable of speaking MCP (either via
# an explicit ``add_mcp`` method or because the framework is known to support
# MCP tool servers). This is what makes the "support various bioagents" promise:
# detection is data-driven rather than hard-coded to a single agent.
MCP_CAPABLE_TOKENS = ("biochatter", "biomni", "langchain", "mcp")


def agent_supports_mcp(agent: Any, agent_dir: str, schema: dict | None) -> bool:
    """Heuristic: does this agent know how to talk to an MCP server?

    Prefers the structured ``capabilities`` block produced by the scanner
    (MCP is a first-class capability there); falls back to a token heuristic for
    callers that haven't populated capabilities yet, and to the live ``add_mcp``
    attribute if the agent is already instantiated.
    """
    if agent is not None and hasattr(agent, "add_mcp"):
        return True
    caps = (schema or {}).get("capabilities")
    if caps:
        mcp = caps.get("mcp")
        if isinstance(mcp, dict):
            if mcp.get("supported"):
                return True
        elif mcp is True:
            return True
    hay = " ".join(
        str(x or "")
        for x in (
            agent_dir,
            (schema or {}).get("module_path"),
            (schema or {}).get("agent_class"),
            (schema or {}).get("registration_method"),
        )
    ).lower()
    return any(token in hay for token in MCP_CAPABLE_TOKENS)


def build_runtime(
    tools: list[dict[str, Any]],
    schema: dict | None,
    agent: Any,
    agent_dir: str,
    exec_mode: str,
    *,
    tool_retriever: Any = None,
    openai_client: Any = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[Callable[[str], str] | None, dict[str, Any]]:
    """Return ``(runner, info)``.

    ``runner`` is a ``query -> answer`` callable to use in MCP mode, or ``None``
    when MCP should not be used (the caller keeps its own loop). ``info`` carries
    diagnostics for display.
    """
    model = model or os.environ.get("OPENAI_MODEL") or "hy3"
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("WESTLAKE_API_KEY")

    supports_mcp = agent_supports_mcp(agent, agent_dir, schema)
    want_native = (exec_mode == "Native tool calling")

    info: dict[str, Any] = {
        "exec_mode": exec_mode,
        "supports_mcp": supports_mcp,
        "driver": None,
        "n_tools": len(tools),
    }

    # ---- MCP-first: whenever the agent supports MCP, drive tools over MCP ----
    if supports_mcp:
        # If the agent is alive and an MCP client, attach it and let it drive.
        if agent is not None and hasattr(agent, "add_mcp"):
            try:
                agent.add_mcp(config_path=os.path.join(os.getcwd(), "mcp_config_cluster.yaml"))
                method = (
                    getattr(agent, "run", None)
                    or getattr(agent, "go", None)
                    or getattr(agent, "invoke", None)
                )
                if callable(method):
                    info["driver"] = "mcp-native"
                    info["note"] = (
                        "Agent connected to server.py as an MCP client; it drives the tools."
                    )
                    return (lambda q: method(q)), info
            except Exception as exc:  # pragma: no cover - defensive
                info["mcp_native_error"] = f"{type(exc).__name__}: {exc}"

        # Otherwise drive via our own MCP tool-calling loop over the MCP server.
        try:
            from agent_connector.mcp_bridge import MCPToolBridge, _HAVE_MCP

            if _HAVE_MCP:
                bridge = MCPToolBridge(model=model, base_url=base_url, api_key=api_key)
                info["driver"] = "mcp"
                info["note"] = "MCP tool-calling loop (server.py tools over MCP)."
                return bridge.run, info
            info["mcp_error"] = "langchain-mcp-adapters not installed"
        except Exception as exc:  # pragma: no cover - defensive
            info["mcp_error"] = f"{type(exc).__name__}: {exc}"

    # ---- Not MCP: let the notebook keep its own unified loop ----
    info["driver"] = "unified"
    info["note"] = "Unified Tool Runner loop (notebook-driven)."
    return None, info
