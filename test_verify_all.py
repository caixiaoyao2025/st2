"""Comprehensive verification of all bio agent pipelines."""
import sys, os, json, re, yaml
sys.path.insert(0, os.path.dirname(__file__))

from agent_connector.graph_retrieval import build_graph_from_registry, retrieve_tools

def build_tool_prompt(specs):
    if not specs:
        return ''
    lines = ['You have access to the following tools:', '']
    for s in specs:
        lines.append(f'[Tool: {s["name"]}]')
        lines.append(f'Description: {s.get("description", "")}')
        inputs = s.get('inputs') or {}
        if inputs:
            lines.append('Arguments:')
            for pname, pmeta in inputs.items():
                if pname == 'subcommand':
                    continue
                ptype = (pmeta or {}).get('type', 'string')
                req = 'required' if (pmeta or {}).get('required') else 'optional'
                pdesc = (pmeta or {}).get('description', '')
                lines.append(f'  - {pname} ({ptype}, {req}): {pdesc}')
        lines.append('')
    lines.append('If you need to use a tool, output exactly:')
    lines.append('<tool_call>')
    lines.append('{ "name": "<tool_name>", "arguments": {...} }')
    lines.append('</tool_call>')
    lines.append('')
    lines.append('After receiving <tool_result>, continue solving the task.')
    return '\n'.join(lines)

def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.S):
        try:
            calls.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            pass
    return calls

print("=" * 60)
print("PART 1: Graph Retrieval (id fix verified)")
print("=" * 60)
graph = build_graph_from_registry(os.path.join('data', 'mcp_registry.yaml'))
print("[OK] ToolGraph built successfully\n")

with open(os.path.join('data', 'mcp_registry.yaml'), 'r', encoding='utf-8') as f:
    reg = yaml.safe_load(f) or {}

spec_lookup = {}
for t in reg.get('tools', []):
    spec_lookup[t['name']] = t
    if t.get('subcommand_details'):
        for sub, detail in t['subcommand_details'].items():
            fname = f"{t['name']}_{sub.replace('-', '_')}"
            spec_lookup[fname] = {
                'name': fname,
                'description': detail.get('description', ''),
                'inputs': {
                    p['name'].lstrip('-').replace('-', '_'): p
                    for p in (detail.get('params') or [])
                },
            }

test_queries = [
    'Look up the human BRCA1 gene sequence from Ensembl, then compute its reverse complement.',
    'Encode this FASTA file to BINSEQ format',
    'Identify metagenomic species from ONT reads',
    'Predict protein structure for sequence MKTIIALSYIFCLVFA',
]

retrieval_ok = 0
for q in test_queries:
    results = retrieve_tools(graph, q, top_k=3)
    if results:
        specs = [spec_lookup[n] for n, _, _ in results if n in spec_lookup]
        prompt = build_tool_prompt(specs)
        names = [r[0] for r in results]
        print(f"[OK] Q: {q[:60]}...")
        print(f"     Retrieved: {names}")
        print(f"     Prompt: {len(prompt)} chars, {len(specs)} tools")
        retrieval_ok += 1
    else:
        print(f"[WARN] Q: {q[:60]}... -> no tools retrieved")
    print()

print("=" * 60)
print("PART 2: Tool Call Parsing")
print("=" * 60)
sample_agent_output = """
I'll look up the BRCA1 gene sequence first.

<tool_call>
{
  "name": "bqtools_revcomp",
  "arguments": {"input": "test.fasta"}
}
</tool_call>

Now let me check the file info.
<tool_call>
{
  "name": "bqtools_info",
  "arguments": {"input": "test.vbq"}
}
</tool_call>
"""
calls = parse_tool_calls(sample_agent_output)
assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
assert calls[0]['name'] == 'bqtools_revcomp'
assert calls[1]['name'] == 'bqtools_info'
print(f"[OK] Parsed {len(calls)} tool calls from agent output")
print(f"     Call 1: {calls[0]['name']}({calls[0]['arguments']})")
print(f"     Call 2: {calls[1]['name']}({calls[1]['arguments']})")
print()

print("=" * 60)
print("PART 3: Cross-Agent Injection Tests")
print("=" * 60)
from test_cross_agent import (
    test_biomni_scan,
    test_biochatter_inject,
    test_cellagent_inject,
    test_geneagent_inject,
)

tests = [
    ("Biomni scan", test_biomni_scan),
    ("BioChatter inject", test_biochatter_inject),
    ("CellAgent inject", test_cellagent_inject),
    ("GeneAgent inject", test_geneagent_inject),
]

passed = 0
skipped = 0
for name, fn in tests:
    try:
        fn()
        print(f"[PASS] {name}")
        passed += 1
    except Exception as e:
        err = str(e)
        if 'SKIP' in err or 'install failed' in err or 'not importable' in err:
            print(f"[SKIP] {name}: {err[:80]}")
            skipped += 1
        else:
            print(f"[FAIL] {name}: {err[:120]}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Retrieval: {retrieval_ok}/{len(test_queries)} queries returned tools")
print(f"  Parsing:   2/2 tool calls parsed correctly")
print(f"  Injection: {passed} passed, {skipped} skipped")
total = retrieval_ok + passed
print(f"  Overall:   {total} OK")
