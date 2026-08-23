"""MCP-aware runtime selection for discovered bio-agents.

The architecture is adapter-based: ``build_runtime`` inspects the detected
agent capabilities and selects an :mod:`agent_connector.adapters` adapter that
injects the discovered tools into the (already instantiated) downstream agent.
The caller then drives the agent with ``run_agent(agent, prompt)``, which simply
calls the agent's native entry (``agent.go()`` / ``run()`` / ...). Our system
never substitutes its own planner/tool-loop for the downstream agent's.
"""

from agent_connector.adapters import (
    BaseAdapter,
    MCPAdapter,
    NativeToolAdapter,
    CodeExecutionAdapter,
    ConfigAdapter,
    PromptAdapter,
    build_runtime,
    run_agent,
    native_entry,
)

__all__ = [
    "BaseAdapter",
    "MCPAdapter",
    "NativeToolAdapter",
    "CodeExecutionAdapter",
    "ConfigAdapter",
    "PromptAdapter",
    "build_runtime",
    "run_agent",
    "native_entry",
]
