"""End-to-end test: LLM API direct call + tool execution loop.

Simulates the notebook Step 6 flow without Jupyter widgets.
Verifies: API key, model, retrieval, prompt, parse, execute, feed-back.
"""
import json
import os
import re
import sys
import time

# ── Config ──
os.environ["OPENAI_API_KEY"] = "sk-CXVEKD43upOkHWTdq1RJP3SMC4OyspQOkqB4ymqw6IJazWyB"
os.environ["OPENAI_BASE_URL"] = "https://tokenhub.tencentmaas.com/v1"
MODEL = "deepseek-v4-flash"
BASE_URL = "https://tokenhub.tencentmaas.com/v1"
API_KEY = os.environ["OPENAI_API_KEY"]

print(f"Model: {MODEL}")
print(f"Base URL: {BASE_URL}")
print(f"API Key: {API_KEY[:12]}...")
print()

# ── Step 1: Test LLM API connection ──
print("=" * 60)
print("Step 1: Test LLM API connection")
print("=" * 60)
from openai import OpenAI

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=10,
        temperature=0,
    )
    print(f"  [OK] LLM connected: {resp.choices[0].message.content}")
except Exception as e:
    print(f"  [FAIL] LLM connection failed: {e}")
    sys.exit(1)

# ── Step 2: Build retrieval graph ──
print()
print("=" * 60)
print("Step 2: Build retrieval graph")
print("=" * 60)
from agent_connector.graph_retrieval import build_graph_from_tools, retrieve_tools
import yaml

with open("data/mcp_registry.yaml", encoding="utf-8") as f:
    reg = yaml.safe_load(f)

graph = build_graph_from_tools(reg["tools"])
print(f"  [OK] Graph built: {len(reg['tools'])} tools")

# Build spec_lookup
spec_lookup = {}
for t in reg["tools"]:
    spec_lookup[t["name"]] = t
    for sub in t.get("subcommands", []):
        leaf_name = f"{t['name']}_{sub}"
        leaf = dict(t)
        leaf["name"] = leaf_name
        leaf["_active_subcommand"] = sub
        leaf["description"] = t.get("subcommand_details", {}).get(sub, {}).get("description", t.get("description", ""))
        # Rebuild inputs from subcommand details
        sub_inputs = {}
        for p in t.get("subcommand_details", {}).get(sub, {}).get("params", []):
            pname = p["name"].lstrip("-").replace("-", "_")
            sub_inputs[pname] = {
                "type": p.get("type", "string"),
                "description": p.get("description", ""),
                "required": p.get("required", False),
            }
        leaf["inputs"] = sub_inputs
        spec_lookup[leaf_name] = leaf

print(f"  [OK] spec_lookup: {len(spec_lookup)} entries")

# ── Step 3: Retrieval ──
print()
print("=" * 60)
print("Step 3: Retrieval")
print("=" * 60)
query = "I have a FASTA file reads.fasta. Encode it to BINSEQ format, then compute the reverse complement of the encoded sequences."
results = retrieve_tools(graph, query, top_k=5)
print(f"  Query: {query[:80]}...")
for name, score, conf in (results or []):
    print(f"  → {name} (score={score:.4f}, confidence={conf})")

# ── Step 4: Build prompt ──
print()
print("=" * 60)
print("Step 4: Build prompt")
print("=" * 60)

def build_tool_prompt(specs):
    if not specs:
        return ""
    lines = ["You have access to the following tools:", ""]
    for s in specs:
        lines.append(f'[Tool: {s["name"]}]')
        lines.append(f'Description: {s.get("description", "")}')
        inputs = s.get("inputs") or {}
        if inputs:
            lines.append("Arguments:")
            for pname, pmeta in inputs.items():
                if pname == "subcommand":
                    continue
                ptype = (pmeta or {}).get("type", "string")
                req = "required" if (pmeta or {}).get("required") else "optional"
                pdesc = (pmeta or {}).get("description", "")
                lines.append(f"  - {pname} ({ptype}, {req}): {pdesc}")
        lines.append("")
    lines.append("IMPORTANT: You MUST use these tools. Do NOT write your own Python code.")
    lines.append("To call a tool, use <tool_call> JSON format:")
    lines.append("")
    lines.append("<tool_call>")
    lines.append('{"name": "bqtools_encode", "arguments": {"input": "reads.fasta", "output": "reads.vbq"}}')
    lines.append("</tool_call>")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use the exact tool name from the list above.")
    lines.append("- Arguments must match the schema (required/optional, types).")
    lines.append("- Call ONE tool per <tool_call> block.")
    lines.append("- After receiving <tool_result>, continue solving the task.")
    return "\n".join(lines)

top_names = [n for n, s, c in (results or [])]
specs = [spec_lookup[n] for n in top_names if n in spec_lookup]
tool_block = build_tool_prompt(specs)
enhanced_query = f"{tool_block}\n\nTask: {query}"
print(f"  [OK] Prompt built ({len(enhanced_query)} chars)")
print(f"  Tools in prompt: {top_names}")

# ── Step 5: Parse tool calls ──
print()
print("=" * 60)
print("Step 5: Parse tool calls (self-test)")
print("=" * 60)

def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            calls.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            pass
    return calls

# Self-test
test_text = '<tool_call>{"name": "bqtools_encode", "arguments": {"input": "reads.fasta", "output": "reads.vbq"}}</tool_call>'
test_calls = parse_tool_calls(test_text)
assert len(test_calls) == 1
assert test_calls[0]["name"] == "bqtools_encode"
print(f"  [OK] Parser works: {test_calls}")

# ── Step 6: LLM call + tool execution loop ──
print()
print("=" * 60)
print("Step 6: LLM call + tool execution loop")
print("=" * 60)

from agent_connector.tool_runner import run_tool_spec, format_result

messages = [{"role": "user", "content": enhanced_query}]
MAX_ITER = 5
final = ""

for iteration in range(MAX_ITER):
    print(f"\n--- Iteration {iteration + 1} ---")
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.3
    )
    raw_str = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": raw_str})
    print(f"  LLM response ({len(raw_str)} chars): {raw_str[:200]}...")

    calls = parse_tool_calls(raw_str)
    if not calls:
        final = raw_str
        print("  No tool calls → final answer")
        break

    print(f"  Parsed {len(calls)} tool call(s)")
    tool_results = []
    for c in calls:
        name = c.get("name", "?")
        args = c.get("arguments", {})
        print(f"  → Executing {name}({args})...")
        spec = spec_lookup.get(name)
        if not spec:
            result_str = f'[error] Tool "{name}" not found in registry'
        else:
            result = run_tool_spec(spec, args)
            result_str = format_result(result)
        print(f"    Result: {result_str[:200]}")
        tool_results.append(f"<tool_result>\n{result_str}\n</tool_result>")

    feedback = "\n".join(tool_results)
    messages.append({
        "role": "user",
        "content": f"Tool execution results:\n{feedback}\n\nContinue with the next step or provide the final answer.",
    })
    print("  Feeding results back...")
else:
    final = "(max iterations reached)"

print()
print("=" * 60)
print("Final result")
print("=" * 60)
print(final[:500])
print()
print("[OK] E2E test completed")
