"""End-to-end test: scan Biomni → generate adapter → create agent → inject tools → run task."""
import os, sys, json

os.environ["OPENAI_API_KEY"] = os.environ.get("WESTLAKE_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/v3"
os.environ["OPENAI_MODEL"] = "deepseek-v4-flash-ga-260731"
os.environ["WESTLAKE_API_KEY"] = os.environ["OPENAI_API_KEY"]
os.environ["BIOMNI_SOURCE"] = "Custom"
os.environ["BIOMNI_LLM"] = "deepseek-v4-flash-ga-260731"
os.environ["BIOMNI_CUSTOM_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
os.environ["BIOMNI_CUSTOM_API_KEY"] = os.environ["OPENAI_API_KEY"]

sys.path.insert(0, os.path.dirname(__file__))

# Find Biomni repo
biomni_dir = None
for candidate in [
    os.path.expanduser("~/AppData/Local/Programs/Python/Python311/Lib/site-packages/biomni"),
    "/content/_agent_Biomni",
    "../biomni",
]:
    if os.path.isdir(candidate):
        biomni_dir = candidate
        break

if not biomni_dir:
    print("ERROR: Biomni not found"); sys.exit(1)

print(f"[1] Biomni dir: {biomni_dir}")

# Step 1: Scan
from agent_connector.scanner import build_schema
schema = build_schema(biomni_dir, include_evidence=False)
print(f"[2] agent_class = {schema.get('agent_class')}")
print(f"    module_path = {schema.get('module_path')}")
print(f"    registration_method = {schema.get('registration_method')}")
print(f"    init_signature = {schema.get('init_signature')}")

# Step 2: Load registry tools
import yaml
reg_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
tools = yaml.safe_load(open(reg_path, encoding="utf-8"))["tools"]
print(f"[3] Registry: {len(tools)} tools")

# Step 3: Resolve execution method
from agent_connector.scanner import KNOWN_AGENT_EXECUTION, _PROBE_ORDER
exec_method = None
agent_class_lower = (schema.get("agent_class") or "").lower()
for pat, method in KNOWN_AGENT_EXECUTION.items():
    if pat in agent_class_lower or pat in biomni_dir.lower():
        exec_method = method
        print(f"[4] Known agent '{pat}' -> method '{exec_method}'")
        break
if not exec_method:
    exec_method = "run"
    print(f"[4] Default -> method '{exec_method}'")

schema["execution_method"] = exec_method

# Step 4: Generate wiring
from agent_connector.generator import generate_wiring, load_wrappers, load_adapter
wiring_dir = os.path.join(os.path.dirname(__file__), "_test_wiring")
wiring = generate_wiring(tools, schema, out_dir=wiring_dir)
print(f"[5] Wiring mode: {wiring['mode']}")

# Show generated adapter
adapter_path = wiring["artifacts"].get("adapter")
if adapter_path:
    print(f"    Adapter: {adapter_path}")
    print(open(adapter_path, encoding="utf-8").read()[:500])

# Step 5: Load wrappers
sys.path.insert(0, wiring_dir)
reg_style = "function"
wrappers = load_wrappers(package_name="generated_tools", registration_style=reg_style)
for w in wrappers:
    if not hasattr(w, "name"):
        w.name = getattr(w, "__name__", None)
print(f"[6] {len(wrappers)} wrappers loaded")
if wrappers:
    print(f"    First wrapper: {wrappers[0].name}")

# Step 6: Create agent via adapter
print("[7] Creating agent via adapter...")
try:
    adapter = load_adapter(schema.get("agent_class") or "Agent",
                           adapter_path=wiring["artifacts"]["adapter"])
    print(f"    Adapter type: {type(adapter).__name__}")
    # Try create_agent, fallback to A1 if react fails due to missing deps
    try:
        agent = adapter.create_agent(use_tool_retriever=False)
    except Exception as e:
        print(f"    Primary create failed: {e}")
        print("    Trying A1 class directly...")
        from biomni.agent.a1 import A1
        agent = A1(
            path="./_test_data",
            llm="deepseek-v4-flash-ga-260731",
            source="Custom",
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            use_tool_retriever=False,
            expected_data_lake_files=[],
        )
        adapter.agent = agent
    print(f"    Agent type: {type(agent).__name__}")
    print(f"    Agent has go: {hasattr(agent, 'go')}")
    print(f"    Agent has run: {hasattr(agent, 'run')}")
    print(f"    Agent.llm: {getattr(agent, 'llm', 'N/A')}")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 7: Register tools - convert our specs to LangChain tools
try:
    from langchain_core.tools import tool as lc_tool
    from agent_connector.tool_runner import run_tool_spec, format_result

    lc_tools = []
    for w in wrappers:
        if hasattr(w, '_TOOL_SPEC'):
            tool_spec = w._TOOL_SPEC
            def _make_runner(ts=tool_spec):
                desc = ts.get('description', ts.get('name', 'unknown tool'))
                @lc_tool(name=ts['name'], description=desc)
                def fn(**kwargs):
                    return format_result(run_tool_spec(ts, dict(kwargs)))
                return fn
            lc_tools.append(_make_runner())

    if not hasattr(agent, 'tools') or agent.tools is None:
        agent.tools = []
    agent.tools.extend(lc_tools)
    print(f"[8] Registered {len(lc_tools)} tools directly to agent.tools")
    print(f"    Agent tools count: {len(agent.tools)}")
    for t in agent.tools:
        print(f"    - {getattr(t, 'name', type(t).__name__)}")
except Exception as e:
    print(f"    Register FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 8: Run a simple query
print("[9] Running query...")
try:
    result = agent.go("What tools are available? List them briefly.")
    print(f"    Result: {str(result)[:500]}")
except Exception as e:
    print(f"    Run FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

print("\n[DONE]")
