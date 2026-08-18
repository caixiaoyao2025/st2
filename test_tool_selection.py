"""Tool Selection Test — uses graph-tool-call for retrieval + LLM for selection.

Three-layer test:
  Layer 1 (retrieval): does the expected tool appear in top-k candidates?
  Layer 2 (LLM): does the LLM pick the right tool from candidates?
  Layer 3 (all_tools): LLM sees ALL schemas, picks without retrieval.

Metrics:
  retrieval_hit@1, @3, @5
  llm_first_choice_accuracy
  no_match_correct (negative test)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from openai import OpenAI

import yaml

from tool_agent_test import to_function_schemas
from agent_connector.graph_retrieval import build_graph_from_registry, retrieve_tools

BASE_URL = (
    os.environ.get("WESTLAKE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("DEEPSEEK_BASE_URL")
    or "https://ark.cn-beijing.volces.com/api/v3"
)
API_KEY = (
    os.environ.get("WESTLAKE_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or ""
)
MODEL = os.environ.get("WESTLAKE_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash-ga-260731"
MAX_RETRIES = 3
RETRY_DELAY = 5

# --- selection tasks: (user_task, expected_fn_name) ---
# expected_fn=None means no tool should match (negative test)
SELECTION_TASKS = [
    # bqtools subcommands
    ("Encode my FASTA file into BINSEQ format", "bqtools_encode"),
    ("Decode this BINSEQ file back to FASTA", "bqtools_decode"),
    ("Search for ACGT pattern in my BINSEQ file", "bqtools_grep"),
    ("Split this BINSEQ file by pattern into separate files", "bqtools_split"),
    ("Randomly sample 10% of records from a BINSEQ file", "bqtools_sample"),
    ("Reverse complement all sequences in this BINSEQ file", "bqtools_revcomp"),
    ("Concatenate multiple BINSEQ files into one", "bqtools_cat"),
    ("Show me the metadata and record count of this BINSEQ file", "bqtools_info"),
    ("Verify the integrity checksum of this BINSEQ file", "bqtools_verify"),
    # cross-tool selection
    ("Generate protein conformational samples from an amino-acid sequence", "bioemu"),
    ("Identify metagenomic species from ONT reads", "kaptain"),
    # negative: no matching tool in registry
    ("Predict RNA secondary structure from a sequence file", None),
]


def _load_schemas(registry_path: str):
    """Load registry and build function schemas for all tools."""
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    tools = reg.get("tools", [])
    schemas = []
    fnmap = {}
    for t in tools:
        sch, fm = to_function_schemas(t)
        schemas.extend(sch)
        fnmap.update(fm)
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    return tools, schemas, schema_by_name, fnmap


SELECTOR_SYSTEM = """You are a tool selector. Your ONLY job is to pick the best tool from a list.

Rules:
- Output ONLY the exact tool name. Nothing else.
- Do NOT execute the tool.
- Do NOT ask the user for missing parameters.
- Do NOT explain your choice.
- Do NOT describe how to use the tool.
- Even if required arguments are missing, still select the best tool.
- If no tool matches, output: NO_MATCHING_TOOL"""


def _llm_select(task: str, candidates: list[str], retries: int = MAX_RETRIES):
    """Send task + candidate list to LLM, return selected tool name or None.

    Uses plain text completion (not tool_call) to enforce strict output.
    Returns None for NO_MATCHING_TOOL.
    """
    if not API_KEY:
        raise RuntimeError("Missing API key")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_msg = f"User request:\n{task}\n\nCandidate tools:\n{numbered}\n\nSelect the best tool:"
    messages = [
        {"role": "system", "content": SELECTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text == "NO_MATCHING_TOOL" or text == "":
                return None
            # extract tool name: handle "1. bqtools_encode" or just "bqtools_encode"
            if ". " in text:
                text = text.split(". ", 1)[1].strip()
            text = text.strip("`\"'")
            if text in candidates:
                return text
            # fuzzy: check if any candidate is a substring
            for c in candidates:
                if c in text:
                    return c
            print(f"  [DEBUG] LLM output not in candidates: {text!r}")
            return text  # return raw for caller to see
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def test_retrieval_only():
    """Layer 1: test retrieval hit rate without LLM."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    graph = build_graph_from_registry(registry_path)
    print(f"== Retrieval-only test ==")
    hit1 = hit3 = hit5 = 0
    neg_correct = 0
    neg_total = 0
    total = len(SELECTION_TASKS)
    for task, expected in SELECTION_TASKS:
        if expected is None:
            neg_total += 1
            results = retrieve_tools(graph, task, top_k=3)
            top_name = results[0][0] if results else None
            ok = results == []
            if ok:
                neg_correct += 1
            tag = "NO_MATCH" if ok else f"WRONG: {top_name}"
            print(f"  [{'PASS' if ok else 'FAIL':4}] [NEG] {task[:50]}... -> {tag}")
            continue
        results = retrieve_tools(graph, task, top_k=5)
        names = [r[0] for r in results]
        in1 = expected == names[0] if names else False
        in3 = expected in names[:3]
        in5 = expected in names[:5]
        if in1: hit1 += 1
        if in3: hit3 += 1
        if in5: hit5 += 1
        mark = "OK" if in1 else ("HIT@3" if in3 else ("HIT@5" if in5 else "MISS"))
        print(f"  [{mark:6}] {task[:50]}... top3={names[:3]}")
    n = total - neg_total
    print(f"\n  retrieval_hit@1 = {hit1}/{n}")
    print(f"  retrieval_hit@3 = {hit3}/{n}")
    print(f"  retrieval_hit@5 = {hit5}/{n}")
    print(f"  no_match_correct = {neg_correct}/{neg_total}")
    neg_pass = neg_correct == neg_total
    all_pass = hit1 == n and neg_pass
    return 0 if all_pass else 1


def test_selection_retrieval_first():
    """Layer 1+2: graph-tool-call retrieval + LLM selection."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    graph = build_graph_from_registry(registry_path)
    _, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    print(f"== Selection test (graph-tool-call retrieval + LLM): {len(all_schemas)} total schemas ==")
    passed = 0
    failed = 0
    skipped = 0
    llm_correct = 0
    llm_total = 0
    neg_correct = 0
    neg_total = 0
    for task, expected in SELECTION_TASKS:
        if expected is None:
            neg_total += 1
            results = retrieve_tools(graph, task, top_k=3)
            if not results:
                neg_correct += 1
                print(f"  [PASS] [NEG] {task[:50]}... -> NO_MATCHING_TOOL")
            else:
                print(f"  [FAIL] [NEG] {task[:50]}... -> WRONG: {results[0][0]}")
            continue
        # retrieve candidates via graph-tool-call
        results = retrieve_tools(graph, task, top_k=5)
        candidate_names = [r[0] for r in results]
        if not candidate_names:
            print(f"  [SKIP] {task[:50]}... -> no retrieval candidates")
            skipped += 1
            continue
        # check retrieval hit
        retrieval_hit = expected in candidate_names
        # LLM picks from candidate names (strict text selection)
        llm_total += 1
        got = _llm_select(task, candidate_names)
        llm_correct_task = got == expected
        if llm_correct_task:
            llm_correct += 1
        if llm_correct_task:
            passed += 1
            print(f"  [PASS] {task[:50]}... -> {got} (retrieval={'HIT' if retrieval_hit else 'MISS'})")
        else:
            failed += 1
            print(f"  [FAIL] {task[:50]}...")
            print(f"         expected={expected}  got={got}  candidates={candidate_names[:5]}")
    print(f"\n  LLM accuracy: {llm_correct}/{llm_total}")
    print(f"  no_match_correct: {neg_correct}/{neg_total}")
    print(f"  Passed: {passed}/{len(SELECTION_TASKS)}, Failed: {failed}, Skipped: {skipped}")
    return failed


def test_selection_all_tools():
    """Layer 3: LLM sees ALL tool names, no retrieval."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    _, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    all_names = list(schema_by_name.keys())
    print(f"== Selection test (all_tools): {len(all_names)} total tools ==")
    passed = 0
    failed = 0
    neg_correct = 0
    neg_total = 0
    for task, expected in SELECTION_TASKS:
        got = _llm_select(task, all_names)
        if expected is None:
            neg_total += 1
            if got is None:
                neg_correct += 1
                print(f"  [PASS] [NEG] {task[:50]}... -> NO_MATCHING_TOOL")
            else:
                print(f"  [FAIL] [NEG] {task[:50]}... -> WRONG: {got}")
            continue
        if got == expected:
            passed += 1
            print(f"  [PASS] {task[:50]}... -> {got}")
        else:
            failed += 1
            print(f"  [FAIL] {task[:50]}...")
            print(f"         expected={expected}  got={got}")
    print(f"\n  Passed: {passed}/{len(SELECTION_TASKS)}, Failed: {failed}")
    print(f"  no_match_correct: {neg_correct}/{neg_total}")
    return failed


def test_agent_loop():
    """Full agent loop: select → extract → validate → execute → result check."""
    from agent_connector.agent import agent_loop, DONE, NEED_USER_INPUT, TOOL_NOT_APPLICABLE
    from openai import OpenAI as OAI
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    graph = build_graph_from_registry(registry_path)
    _, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    client = OAI(base_url=BASE_URL, api_key=API_KEY)
    print(f"== Agent loop test (selection only, no execution): {len(all_schemas)} schemas ==")
    passed = 0
    failed = 0
    neg_correct = 0
    neg_total = 0
    for task, expected in SELECTION_TASKS:
        result = agent_loop(task, graph, all_schemas, fnmap, client, MODEL, runner_fn=None)
        got = result["tool"]
        status = result["status"]
        if expected is None:
            neg_total += 1
            if status == TOOL_NOT_APPLICABLE and got is None:
                neg_correct += 1
                print(f"  [PASS] [NEG] {task[:50]}... -> NO_MATCHING_TOOL")
            else:
                print(f"  [FAIL] [NEG] {task[:50]}... -> got={got} status={status}")
            continue
        if got == expected:
            passed += 1
            print(f"  [PASS] {task[:50]}... -> {got} (status={status})")
        else:
            failed += 1
            print(f"  [FAIL] {task[:50]}...")
            print(f"         expected={expected}  got={got}  status={status}")
            print(f"         attempts={[a['tool'] for a in result['attempts']]}")
    print(f"\n  Passed: {passed}/{len(SELECTION_TASKS)}, Failed: {failed}")
    print(f"  no_match_correct: {neg_correct}/{neg_total}")
    return failed


def test_agent_replan():
    """Test replan loop with fully deterministic fixtures (no LLM, no real registry).

    Uses fake tools to isolate the replan logic:
    - Scenario 1: wrong tool → not satisfied → replan → right tool → satisfied → DONE
    - Scenario 2: all tools fail → exhaust retries → TOOL_NOT_APPLICABLE
    - Scenario 3: missing required args → NEED_USER_INPUT
    """
    from agent_connector.agent import (
        agent_loop, DONE, NEED_USER_INPUT, TOOL_NOT_APPLICABLE,
    )

    print(f"== Agent replan test (deterministic fixtures) ==")

    # -- Fake tool schemas (no retrieval, no LLM) --
    _mk_schema = lambda name, desc: {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input file"},
                },
                "required": ["input"],
            },
        },
    }
    fake_schemas = [
        _mk_schema("fake_wrong", "A tool that does metadata inspection"),
        _mk_schema("fake_right", "A tool that identifies metagenomic species"),
    ]
    fake_fnmap = {}
    for s in fake_schemas:
        fn = s["function"]
        fake_fnmap[fn["name"]] = {
            "function": fn,
            "inputs": {"input": {"required": True, "description": "Input file"}},
        }

    query = "Identify species from /data/reads.fastq"

    # Scenario 1: replan — wrong tool first, then right tool
    # Both selector and validator track call counts to control flow
    sel_count = [0]
    val_count = [0]

    def fake_selector(q, candidates):
        sel_count[0] += 1
        # Always pick fake_wrong if available, otherwise fake_right
        if "fake_wrong" in candidates:
            return "fake_wrong"
        if "fake_right" in candidates:
            return "fake_right"
        return None

    def fake_extractor(q, spec):
        return {"input": "/data/reads.fastq"}

    def fake_runner(spec, args, timeout=300):
        name = spec["function"]["name"]
        if name == "fake_wrong":
            return {"return_code": 0, "status": "ok",
                    "stdout": "metadata: record_count=100\nformat: BINSEQ\n"}
        return {"return_code": 0, "status": "ok",
                "stdout": "species: Lactobacillus, Bifidobacterium, Escherichia\n"}

    def fake_validator(q, tool_name, result):
        val_count[0] += 1
        if tool_name == "fake_wrong":
            return False, "Tool only provided metadata, not species identification"
        return True, ""

    sel_count[0] = 0
    val_count[0] = 0
    r1 = agent_loop(query, None, fake_schemas, fake_fnmap, None, None,
                     runner_fn=fake_runner, selector_fn=fake_selector,
                     extractor_fn=fake_extractor, validator_fn=fake_validator)
    tools_tried_1 = [a["tool"] for a in r1["attempts"]]
    print(f"  Scenario 1 (replan): status={r1['status']} "
          f"attempts={len(r1['attempts'])} tools_tried={tools_tried_1}")
    s1_ok = (r1["status"] == DONE
             and len(r1["attempts"]) >= 2
             and tools_tried_1[0] == "fake_wrong"
             and tools_tried_1[-1] == "fake_right")
    print(f"    [{'PASS' if s1_ok else 'FAIL'}] wrong→right replan")

    # Scenario 2: all tools fail → exhaust retries → TOOL_NOT_APPLICABLE
    def fake_validator_always_fail(q, tool_name, result):
        return False, "Never satisfies"

    sel_count[0] = 0
    r2 = agent_loop(query, None, fake_schemas, fake_fnmap, None, None,
                     runner_fn=fake_runner, selector_fn=fake_selector,
                     extractor_fn=fake_extractor,
                     validator_fn=fake_validator_always_fail)
    tools_tried_2 = [a["tool"] for a in r2["attempts"]]
    print(f"\n  Scenario 2 (all fail): status={r2['status']} "
          f"attempts={len(r2['attempts'])} tools_tried={tools_tried_2}")
    s2_ok = (r2["status"] == TOOL_NOT_APPLICABLE
             and len(r2["attempts"]) >= 2
             and "fake_wrong" in tools_tried_2
             and "fake_right" in tools_tried_2)
    print(f"    [{'PASS' if s2_ok else 'FAIL'}] exhaust → TOOL_NOT_APPLICABLE")

    # Scenario 3: missing required args → NEED_USER_INPUT (no runner needed)
    schema_missing = [{
        "type": "function",
        "function": {
            "name": "fake_needs_args",
            "description": "A tool that needs arguments",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input file"},
                },
                "required": ["input"],
            },
        },
    }]
    fnmap_missing = {
        "fake_needs_args": {
            "function": schema_missing[0]["function"],
            "inputs": {"input": {"required": True, "description": "Input file"}},
        },
    }

    def fake_extractor_empty(q, spec):
        return {}  # extracts nothing

    def fake_selector_always(q, candidates):
        return candidates[0] if candidates else None

    r3 = agent_loop("Do something vague", None, schema_missing, fnmap_missing,
                     None, None, runner_fn=None, selector_fn=fake_selector_always,
                     extractor_fn=fake_extractor_empty)
    print(f"\n  Scenario 3 (missing args): status={r3['status']} "
          f"tool={r3['tool']} missing={r3.get('missing', [])}")
    s3_ok = (r3["status"] == NEED_USER_INPUT
             and r3["tool"] == "fake_needs_args"
             and "input" in r3.get("missing", []))
    print(f"    [{'PASS' if s3_ok else 'FAIL'}] NEED_USER_INPUT with missing args")

    total = 3
    passed = sum([s1_ok, s2_ok, s3_ok])
    print(f"\n  Replan test: {passed}/{total}")
    return 0 if passed == total else 1


def main():
    parser = argparse.ArgumentParser(description="Tool selection test")
    parser.add_argument("--all-tools", action="store_true",
                        help="LLM sees ALL schemas (no retrieval)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Test retrieval hit rate only (no LLM)")
    parser.add_argument("--agent", action="store_true",
                        help="Full agent loop (select → extract → validate)")
    parser.add_argument("--replan", action="store_true",
                        help="Agent replan test (fake runner, recovery loop)")
    args = parser.parse_args()
    if args.retrieval_only:
        return test_retrieval_only()
    if args.replan:
        return test_agent_replan()
    if args.agent:
        return test_agent_loop()
    if args.all_tools:
        return test_selection_all_tools()
    return test_selection_retrieval_first()


if __name__ == "__main__":
    sys.exit(main())
