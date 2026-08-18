"""Tool-selection agent with argument validation + result validation + replan.

Architecture:
  User Query
    → Retrieval (graph-tool-call)
    → Tool Selection (LLM text selector)
    → Argument Extraction (LLM)
    → Input Validation
      ├─ missing required → NEED_USER_INPUT
      └─ complete → Execute
    → Result Validation (LLM)
      ├─ task satisfied → DONE
      └─ not satisfied → REPLAN (exclude failed tool, retry)
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from agent_connector.graph_retrieval import build_graph_from_registry, retrieve_tools

# -- Status codes -----------------------------------------------------------

DONE = "DONE"
NEED_USER_INPUT = "NEED_USER_INPUT"
TOOL_NOT_APPLICABLE = "TOOL_NOT_APPLICABLE"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
MAX_RETRIES = 2


# -- LLM helpers ------------------------------------------------------------

def _llm_json(client: OpenAI, model: str, system: str, user: str) -> dict:
    """Call LLM and parse JSON response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    # extract JSON from markdown code blocks if present
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


# -- Argument extraction ----------------------------------------------------

EXTRACT_SYSTEM = """Extract tool arguments from the user query.

Return a JSON object mapping parameter names to their values.
Only include parameters the user has explicitly provided or that are
clearly implied. Do NOT invent values for file paths, database paths,
or other specific resources the user hasn't mentioned.

Return: {"param_name": "value", ...}
If no arguments can be extracted: {}"""


def extract_arguments(
    query: str, spec: dict, client: OpenAI, model: str
) -> dict[str, str]:
    """Use LLM to extract tool arguments from user query.

    Returns a dict of {param_name: value} for arguments the user provided.
    """
    inputs = spec.get("inputs") or {}
    param_desc = []
    for name, meta in inputs.items():
        desc = (meta or {}).get("description", "")
        required = "REQUIRED" if (meta or {}).get("required") else "optional"
        param_desc.append(f"  {name} ({required}): {desc}")
    params_block = "\n".join(param_desc) if param_desc else "  (no parameters)"
    user_msg = f"User query: {query}\n\nTool parameters:\n{params_block}\n\nExtract arguments:"
    try:
        return _llm_json(client, model, EXTRACT_SYSTEM, user_msg)
    except (json.JSONDecodeError, ValueError):
        return {}


# -- Input validation -------------------------------------------------------

def validate_arguments(
    spec: dict, args: dict[str, str]
) -> tuple[bool, list[str]]:
    """Check if all required arguments are present.

    Returns (is_valid, list_of_missing_required_param_names).
    """
    inputs = spec.get("inputs") or {}
    missing = []
    for name, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("required") is True and name not in args:
            missing.append(name)
    return len(missing) == 0, missing


# -- Result validation ------------------------------------------------------

VALIDATE_SYSTEM = """You are a task result validator. Check if the tool output
satisfies the user's original goal.

Return JSON:
{"satisfied": true}  — result completes the user's task
{"satisfied": false, "reason": "brief explanation"}  — result does NOT complete the task

Be strict: the result must actually address what the user asked for,
not just be a successful tool execution."""


def validate_result(
    query: str, tool_name: str, result: str, client: OpenAI, model: str
) -> tuple[bool, str]:
    """Ask LLM if the tool result satisfies the user's original query.

    Returns (satisfied: bool, reason: str).
    """
    user_msg = (
        f"User goal: {query}\n\n"
        f"Tool used: {tool_name}\n\n"
        f"Tool result:\n{result[:1500]}\n\n"
        f"Does this result satisfy the user's goal?"
    )
    try:
        out = _llm_json(client, model, VALIDATE_SYSTEM, user_msg)
        satisfied = out.get("satisfied", False)
        reason = out.get("reason", "")
        return satisfied, reason
    except (json.JSONDecodeError, ValueError):
        return True, ""  # on parse error, assume satisfied (don't block)


# -- Main agent loop --------------------------------------------------------

def agent_loop(
    query: str,
    graph,
    schemas: list[dict],
    fnmap: dict[str, dict],
    client: OpenAI,
    model: str,
    runner_fn=None,
    selector_fn=None,
    extractor_fn=None,
    validator_fn=None,
) -> dict:
    """Run the full agent loop: select → extract → validate → execute → check.

    Args:
        query: User's natural language request.
        graph: ToolGraph for retrieval.
        schemas: All function schemas (for LLM selector context).
        fnmap: fn_name → ToolSpec leaf (for execution).
        client: OpenAI client.
        model: Model name.
        runner_fn: callable(spec, args, timeout) → raw result dict.
                   If None, skips execution (selection-only mode).
        selector_fn: callable(query, candidates) → tool_name|None.
                     If None, uses _select_tool (LLM-based).
        extractor_fn: callable(query, spec) → dict[str, str].
                       If None, uses extract_arguments (LLM-based).
        validator_fn: callable(query, tool_name, result_text) → (bool, reason).
                       If None, uses validate_result (LLM-based).

    Returns:
        {
            "status": str,          # DONE / NEED_USER_INPUT / TOOL_NOT_APPLICABLE / ...
            "tool": str | None,     # selected tool name
            "args": dict,           # extracted arguments
            "missing": list,        # missing required args (if NEED_USER_INPUT)
            "result": str,          # tool output (if executed)
            "reason": str,          # explanation
            "attempts": list,       # [{tool, args, result, status}, ...]
        }
    """
    candidates_all = [s["function"]["name"] for s in schemas]
    attempts = []
    excluded = set()

    for attempt in range(MAX_RETRIES + 1):
        # 1. Retrieve candidates
        if graph is not None:
            results = retrieve_tools(graph, query, top_k=5)
            candidate_names = [r[0] for r in results if r[0] not in excluded]
        else:
            candidate_names = []
        if not candidate_names:
            # fallback: use all tools minus excluded
            candidate_names = [n for n in candidates_all if n not in excluded]
        if not candidate_names:
            return {
                "status": TOOL_NOT_APPLICABLE,
                "tool": None, "args": {}, "missing": [],
                "result": "", "reason": "No candidate tools available",
                "attempts": attempts,
            }

        # 2. Select tool (text-based)
        _sel = selector_fn or (lambda q, c: _select_tool(q, c, client, model))
        tool_name = _sel(query, candidate_names)
        if tool_name is None:
            return {
                "status": TOOL_NOT_APPLICABLE,
                "tool": None, "args": {}, "missing": [],
                "result": "", "reason": "Selector returned NO_MATCHING_TOOL",
                "attempts": attempts,
            }

        # 3. Get tool spec
        spec = fnmap.get(tool_name)
        if spec is None:
            excluded.add(tool_name)
            attempts.append({"tool": tool_name, "args": {}, "result": "",
                             "status": "SPEC_NOT_FOUND"})
            continue

        # 4. Extract arguments
        _ext = extractor_fn or (lambda q, s: extract_arguments(q, s, client, model))
        args = _ext(query, spec)

        # 5. Input validation
        is_valid, missing = validate_arguments(spec, args)
        if not is_valid:
            return {
                "status": NEED_USER_INPUT,
                "tool": tool_name, "args": args, "missing": missing,
                "result": "",
                "reason": f"Missing required: {', '.join(missing)}",
                "attempts": attempts,
            }

        # 6. Execute (if runner provided)
        if runner_fn is None:
            return {
                "status": DONE,
                "tool": tool_name, "args": args, "missing": [],
                "result": "(execution skipped — selection-only mode)",
                "reason": "",
                "attempts": attempts,
            }

        try:
            raw = runner_fn(spec, args, timeout=300)
        except Exception as e:
            attempts.append({"tool": tool_name, "args": args,
                             "result": "", "status": "EXEC_ERROR",
                             "error": str(e)})
            return {
                "status": TOOL_EXECUTION_ERROR,
                "tool": tool_name, "args": args, "missing": [],
                "result": str(e),
                "reason": f"Execution failed: {e}",
                "attempts": attempts,
            }

        from agent_connector.tool_runner import format_result
        result_text = format_result(raw)
        exec_ok = raw.get("return_code") == 0 and raw.get("status") == "ok"

        if not exec_ok:
            attempts.append({"tool": tool_name, "args": args,
                             "result": result_text[:500],
                             "status": "EXEC_FAILED"})
            excluded.add(tool_name)
            continue

        # 7. Result validation
        if validator_fn:
            satisfied, reason = validator_fn(query, tool_name, result_text)
        else:
            satisfied, reason = validate_result(
                query, tool_name, result_text, client, model
            )

        attempts.append({"tool": tool_name, "args": args,
                         "result": result_text[:500],
                         "status": "VALIDATED" if satisfied else "NOT_SATISFIED"})

        if satisfied:
            return {
                "status": DONE,
                "tool": tool_name, "args": args, "missing": [],
                "result": result_text,
                "reason": "",
                "attempts": attempts,
            }

        # not satisfied → replan
        excluded.add(tool_name)

    return {
        "status": TOOL_NOT_APPLICABLE,
        "tool": None, "args": {}, "missing": [],
        "result": "",
        "reason": f"Exhausted {MAX_RETRIES + 1} attempts",
        "attempts": attempts,
    }


# -- Tool selector (text-based, same as test_tool_selection.py) -------------

SELECTOR_SYSTEM = """You are a tool selector. Your ONLY job is to pick the best tool from a list.

Rules:
- Output ONLY the exact tool name. Nothing else.
- Do NOT execute the tool.
- Do NOT ask the user for missing parameters.
- Do NOT explain your choice.
- Even if required arguments are missing, still select the best tool.
- If no tool matches, output: NO_MATCHING_TOOL"""


def _select_tool(
    query: str, candidates: list[str], client: OpenAI, model: str
) -> str | None:
    """Select the best tool from candidates. Returns tool name or None."""
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_msg = f"User request:\n{query}\n\nCandidate tools:\n{numbered}\n\nSelect the best tool:"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text == "NO_MATCHING_TOOL" or text == "":
        return None
    if ". " in text:
        text = text.split(". ", 1)[1].strip()
    text = text.strip("`\"'")
    if text in candidates:
        return text
    for c in candidates:
        if c in text:
            return c
    return None
