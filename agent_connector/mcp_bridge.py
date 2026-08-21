"""MCP-native tool-calling bridge.

For agents that speak MCP (e.g. BioChatter, which supports tool calling through
``langchain_mcp_adapters``), we do NOT take over the loop ourselves. Instead we
expose our registry tools through the repo's own FastMCP server (``server.py``)
and let the foreign agent connect to it as an MCP client, driving the tools with
its own (LLM-native) tool-calling. This is the most "native" integration
possible and reuses the exact same server Biomni talks to.

The bridge:

* launches our ``server.py`` over stdio (via ``StdioServerParameters``),
* loads its tools with ``langchain_mcp_adapters.load_mcp_tools``,
* runs a standard LangChain tool-calling loop with an OpenAI-compatible model,
  executing tool calls through the MCP session.

This keeps the loop agent-agnostic while still being genuine MCP tool calling:
BioChatter's own ``LangChainConversation`` uses the same LangChain primitives
under the hood, so wiring it through this bridge is equivalent to wiring it
through BioChatter directly.
"""

import asyncio
import os
import sys

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    _HAVE_MCP = True
except Exception:  # pragma: no cover - import guard for environments w/o deps
    _HAVE_MCP = False


def default_server_params(server_py=None):
    """Build StdioServerParameters that spawn the repo's FastMCP server.

    The server reads its tools from data/mcp_registry.yaml; we point its env at
    the repo so the subprocess inherits the right registry path.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    server_py = server_py or os.path.join(repo_root, "server.py")
    env = dict(os.environ)
    env.setdefault("MCP_APP_ROOT", repo_root)
    env.setdefault("MCP_DATA_ROOT", os.path.join(repo_root, "data"))
    env.setdefault(
        "MCP_USER_REGISTRY_PATH",
        os.path.join(repo_root, "data", "mcp_registry.yaml"),
    )
    env.setdefault("MCP_TRANSPORT", "stdio")
    return StdioServerParameters(
        command=sys.executable, args=[server_py], env=env
    )


class MCPToolBridge:
    """Drive an OpenAI-compatible LLM with our MCP tools via LangChain."""

    def __init__(self, server_params=None, model=None, base_url=None,
                 api_key=None, max_iter=6, temperature=0.3):
        if not _HAVE_MCP:
            raise RuntimeError(
                "langchain-mcp-adapters / langchain-openai not installed; "
                "run `pip install langchain-mcp-adapters langchain-openai`"
            )
        self.server_params = server_params or default_server_params()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_iter = max_iter
        self.temperature = temperature
        self.tools = []

    async def _run(self, query):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                self.tools = tools
                if not tools:
                    return "[error] MCP server exposed no tools"
                llm = init_chat_model(
                    "openai",
                    model=self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    temperature=self.temperature,
                )
                llm_with_tools = llm.bind_tools(tools)
                messages = [HumanMessage(content=query)]
                last = ""
                for _ in range(self.max_iter):
                    ai = await llm_with_tools.ainvoke(messages)
                    messages.append(ai)
                    last = ai.content if isinstance(ai.content, str) else str(ai.content)
                    if not getattr(ai, "tool_calls", None):
                        return last
                    for tc in ai.tool_calls:
                        name = tc.get("name")
                        args = tc.get("args", {}) or {}
                        tool = next((t for t in tools if t.name == name), None)
                        if tool is None:
                            messages.append(ToolMessage(
                                content=f"[error] tool {name} not found",
                                tool_call_id=tc.get("id")))
                            continue
                        try:
                            result = await tool.ainvoke(args)
                        except Exception as e:  # pragma: no cover - defensive
                            result = f"[error] {type(e).__name__}: {e}"
                        messages.append(ToolMessage(
                            content=str(result), tool_call_id=tc.get("id")))
                return last or "(max iterations reached)"

    def run(self, query):
        import nest_asyncio
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run(query))
