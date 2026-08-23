"""Write the semantic-audit verdict back into the registry as observable state.

This closes the governance loop the user described:

    Discovery -> artifact inference -> schema + confidence + provenance
              -> semantic_audit (active / needs_review / rejected)
              -> preflight Gate 4 -> server.is_active() -> LLM never sees bad tools

The registry stays the SINGLE SOURCE OF TRUTH: every tool gets a ``status``
field plus the evidence that produced it, so "why isn't this tool in the
agent?" is answered by reading the file, not by re-running discovery.

The persisted ``status`` is documentation/observability. Runtime enforcement
remains ``server.is_active()`` (which recomputes from the schema), so a stale
persisted status can never admit a bad tool.
"""

from __future__ import annotations

import sys

import yaml


def _ensure_provenance(tool_name: str, sub: str | None, pname: str,
                       param: dict, command: str) -> None:
    """Back-fill artifact_confidence / artifact_source on a param if missing
    (e.g. for registries generated before inference carried provenance)."""
    if "artifact_confidence" in param and "artifact_source" in param:
        return
    try:
        from discovery_to_registry import infer_artifact_contract
    except Exception:
        return
    contract = infer_artifact_contract(tool_name, pname,
                                        param.get("description", "") or "",
                                        command, sub_name=sub or "")
    if "artifact_confidence" in contract:
        param["artifact_confidence"] = contract["artifact_confidence"]
        param["artifact_source"] = contract["artifact_source"]


def _iter_artifact_inputs(tool: dict):
    """Yield (subcommand_or_None, param_name, param_dict) for inputs with an
    artifact_type."""
    for pname, p in (tool.get("inputs") or {}).items():
        if isinstance(p, dict) and p.get("artifact_type"):
            yield (None, pname, p)
    for sub, detail in (tool.get("subcommand_details") or {}).items():
        for p in (detail.get("params") or []):
            if isinstance(p, dict) and p.get("artifact_type"):
                yield (sub, p.get("name", ""), p)


def compute_status(tool: dict) -> dict:
    """Return the observable status block for one tool spec."""
    from agent_connector.semantic_audit import (
        audit_tool, classify_tool, execution_contract_check,
    )

    tool_name = tool.get("name", "")
    command = tool.get("command", "") or ""
    reasons: list[str] = []

    issues = audit_tool(tool)
    for i in issues:
        if i["severity"] in ("rejected", "needs_review"):
            reasons.append(f"[{i['severity']}] {i.get('param', '')}: {i['reason']}")
    exec_issues = execution_contract_check(tool)
    for i in exec_issues:
        if i["severity"] == "needs_review":
            reasons.append(f"[execution] {i['reason']}")

    status = classify_tool(tool)
    if status == "active" and reasons:
        status = "needs_review"

    # artifact_contract / confidence / source summaries
    contract_inputs: dict[str, str] = {}
    conf_inputs: dict[str, float] = {}
    src_inputs: dict[str, str] = {}
    for sub, pname, p in _iter_artifact_inputs(tool):
        _ensure_provenance(tool_name, sub, pname, p, command)
        key = f"{sub}.{pname}" if sub else pname
        contract_inputs[key] = p.get("artifact_type")
        if "artifact_confidence" in p:
            conf_inputs[key] = p["artifact_confidence"]
        if "artifact_source" in p:
            src_inputs[key] = p["artifact_source"]

    contract_outputs: dict[str, str] = {}
    for oname, o in (tool.get("outputs") or {}).items():
        if isinstance(o, dict) and o.get("artifact_type"):
            contract_outputs[oname] = o["artifact_type"]
    for sub, detail in (tool.get("subcommand_details") or {}).items():
        for oname, o in (detail.get("outputs") or {}).items():
            if isinstance(o, dict) and o.get("artifact_type"):
                contract_outputs[f"{sub}.{oname}"] = o["artifact_type"]

    block: dict = {"status": status}
    if reasons:
        block["status_reason"] = reasons
    if contract_inputs or contract_outputs:
        block["artifact_contract"] = {
            "inputs": contract_inputs,
            "outputs": contract_outputs,
        }
    if conf_inputs:
        block["artifact_confidence"] = {"inputs": conf_inputs}
    if src_inputs:
        block["artifact_source"] = {"inputs": src_inputs}
    return block


def annotate_registry_file(path: str) -> dict:
    """Load a registry, compute + attach status to every tool, write it back.

    Returns a summary {total, active, needs_review, rejected}."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = data.get("tools", [])
    counts = {"active": 0, "needs_review": 0, "rejected": 0}
    for t in tools:
        if not isinstance(t, dict):
            continue
        block = compute_status(t)
        # clean any previous annotation, then attach fresh
        for k in ("status", "status_reason", "artifact_contract",
                  "artifact_confidence", "artifact_source"):
            t.pop(k, None)
        t.update(block)
        counts[block["status"]] = counts.get(block["status"], 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=4096)
    counts["total"] = len(tools)
    return counts


def get_active_tools(registry: dict) -> list[dict]:
    return [t for t in registry.get("tools", []) if t.get("status") == "active"]


def get_quarantine_tools(registry: dict) -> list[dict]:
    return [t for t in registry.get("tools", [])
            if t.get("status") in ("needs_review", "rejected")]


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    path = argv[0] if argv else "data/mcp_registry.yaml"
    counts = annotate_registry_file(path)
    print(f"[registry-status] {path}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
