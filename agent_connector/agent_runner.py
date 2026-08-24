"""MCP-aware runtime selection for discovered bio-agents.

The architecture is adapter-based: ``build_runtime`` inspects the detected
agent capabilities and selects an :mod:`agent_connector.adapters` adapter that
injects the discovered tools into the (already instantiated) downstream agent.
The caller then drives the agent with ``run_agent(agent, prompt)``. The entry
contract is NOT hardcoded: ``run_agent`` defers to the selected adapter (e.g.
``BioChatterAdapter`` drives ``conversation.query``), falling back to probing
``go()``/``run()``/``invoke()`` only when no dedicated adapter exists. Our
system never substitutes its own planner/tool-loop for the downstream agent's.
"""

from agent_connector.adapters import (
    BaseAdapter,
    MCPAdapter,
    NativeToolAdapter,
    CodeExecutionAdapter,
    BioChatterAdapter,
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
    "BioChatterAdapter",
    "ConfigAdapter",
    "PromptAdapter",
    "build_runtime",
    "run_agent",
    "native_entry",
]
