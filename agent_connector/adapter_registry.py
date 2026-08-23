"""Curated registry of SUPPORTED agent interfaces and their adapters.

Strategy (deliberately NOT "auto-adapt to every agent"):

    Tool layer : auto-discovery + semantic audit  -> Canonical Registry
    Agent layer: known interface -> auto-adapter
                 unknown interface -> explicit error, user supplies an adapter

We maintain a SMALL, explicit mapping of mainstream agent interfaces to adapter
classes. If a detected agent matches a known interface we wire it directly; if
it does not, we FAIL CLEARLY instead of guessing a schema (mirroring the
active/needs_review/rejected governance stance: never feed the LLM a guessed
tool interface).

Adding support for a new mainstream agent = append ONE rule here; the tool
discovery pipeline is untouched.
"""

from __future__ import annotations

import os
from typing import Any

# Default supported interfaces. A match rule matches when EVERY key in ``match``
# is satisfied by the agent's detected interface (value may be a scalar or a
# list of acceptable values; omitted key = wildcard).
DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "name": "mcp",
        "match": {"mcp": True},
        "adapter": "MCPAdapter",
        "note": "MCP-capable agents (e.g. Biomni A1, any agent with add_mcp).",
    },
    {
        "name": "native-tool",
        "match": {"native_tool_calling": True},
        "adapter": "NativeToolAdapter",
        "note": "Agents that register callables (add_tool / tools.append / "
                "bind_tools / StructuredTool), e.g. Biomni react, LangChain.",
    },
    {
        "name": "code-execution",
        "match": {"code_execution": True},
        "adapter": "CodeExecutionAdapter",
        "note": "Agents that execute generated code in a namespace (e.g. BioChatter).",
    },
]

_ADAPTER_REGISTRY_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "adapter_registry.yaml"
)


class UnsupportedAgentInterface(Exception):
    """Raised when no known adapter matches the detected agent interface.

    The message tells the user exactly what to implement, so a custom agent is
    trivially supported by adding one rule + one adapter class.
    """


def _adapter_class(name: str):
    from agent_connector.adapters import (  # lazy import avoids cycle
        MCPAdapter, NativeToolAdapter, CodeExecutionAdapter,
        ConfigAdapter, PromptAdapter, BaseAdapter,
    )
    table = {
        "MCPAdapter": MCPAdapter,
        "NativeToolAdapter": NativeToolAdapter,
        "CodeExecutionAdapter": CodeExecutionAdapter,
        "ConfigAdapter": ConfigAdapter,
        "PromptAdapter": PromptAdapter,
        "BaseAdapter": BaseAdapter,
    }
    if name not in table:
        raise UnsupportedAgentInterface(
            f"Adapter class {name!r} is not known. Available: {sorted(table)}"
        )
    return table[name]


def _cap_bool(caps: dict | None, key: str) -> bool:
    """Read a capability flag that may be a bool OR a {supported: bool} dict."""
    v = (caps or {}).get(key)
    if isinstance(v, dict):
        return bool(v.get("supported"))
    return bool(v)


def build_interface(schema: dict, caps: dict) -> dict:
    """Flatten the detected agent interface into matchable fields."""
    ti = (schema or {}).get("tool_interface") or {}
    return {
        "framework": ti.get("framework"),
        "schema_format": ti.get("schema_format"),
        "registration_method": ti.get("registration_method"),
        "execution_method": ti.get("execution_method"),
        "mcp": _cap_bool(caps, "mcp"),
        "native_tool_calling": _cap_bool(caps, "native_tool_calling"),
        "code_execution": _cap_bool(caps, "code_execution"),
    }


def _rule_matches(rule: dict, interface: dict) -> bool:
    for key, expected in (rule.get("match") or {}).items():
        actual = interface.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def load_adapter_registry(path: str | None = None) -> list[dict[str, Any]]:
    """Load match rules. Prefer the YAML registry if present, else defaults."""
    import yaml
    p = path or _ADAPTER_REGISTRY_YAML
    if p and os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            rules = data.get("adapters") or data.get("rules")
            if isinstance(rules, list) and rules:
                return rules
        except Exception:
            pass
    return [dict(r) for r in DEFAULT_RULES]


def find_adapter(schema: dict, caps: dict):
    """Return (adapter_class, rule_name) for a detected agent interface.

    Raises UnsupportedAgentInterface if no known rule matches -- we never guess.
    """
    interface = build_interface(schema, caps)
    rules = load_adapter_registry()
    for rule in rules:
        if _rule_matches(rule, interface):
            return _adapter_class(rule["adapter"]), rule["name"]

    supported = ", ".join(sorted({r["name"] for r in rules})) or "(none)"
    raise UnsupportedAgentInterface(
        f"Unsupported agent interface: framework={interface['framework']!r}, "
        f"schema_format={interface['schema_format']!r}, "
        f"execution={interface['execution_method']!r}.\n"
        f"Cannot automatically determine how this agent registers/invokes tools.\n"
        f"Supported interfaces: {supported}.\n\n"
        f"To support this agent, implement an adapter and register it:\n"
        f"  # agent_connector/adapter_registry.yaml\n"
        f"  adapters:\n"
        f"    - name: <my_agent>\n"
        f"      match: {{framework: <my_agent>}}   # or schema_format/registration_method\n"
        f"      adapter: MyAgentAdapter\n"
        f"  # and define:\n"
        f"  class MyAgentAdapter(BaseAdapter):\n"
        f"      def connect(self, agent):  # inject discovered tools\n"
        f"          ...\n"
    )


def describe_supported(path: str | None = None) -> str:
    rules = load_adapter_registry(path)
    lines = []
    for r in rules:
        lines.append(f"  - {r['name']}: match={r.get('match')} -> {r['adapter']}"
                     + (f"  ({r['note']})" if r.get("note") else ""))
    return "Supported agent interfaces:\n" + "\n".join(lines)
