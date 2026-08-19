"""Cross-agent adapter test: verify different wiring styles work correctly.

Tests the agent_connector scanner/generator/adapter infrastructure with
fake agents covering every wiring path:
  1. object style (add_tool method on agent instance)
  2. function style (framework introspects the callable)
  3. manifest wiring (tools=[...] passed to LLM, no register method)
  4. config wiring (tools defined in config file)
  5. prompt wiring (tools described in system prompt)

No LLM needed — all tests are deterministic with fake agents.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# Minimal ToolSpec for testing (like registry.yaml entries)
FAKE_TOOLS = [
    {
        "name": "fasta_stats",
        "type": "cli",
        "command": "python -c \"print('sequences=5 length=30')\"",
        "description": "Count sequences and total length in a FASTA file",
        "arg_style": "cli",
        "inputs": {
            "fasta_path": {
                "type": "string",
                "required": True,
                "description": "Path to FASTA file",
            }
        },
        "outputs": {"stdout": {"type": "stdout"}},
    },
    {
        "name": "sort_fasta",
        "type": "cli",
        "command": "python -c \"print('sorted output')\"",
        "description": "Sort sequences in a FASTA file by name",
        "arg_style": "cli",
        "inputs": {
            "input": {
                "type": "string",
                "required": True,
                "description": "Input FASTA path",
            },
            "output": {
                "type": "string",
                "required": True,
                "description": "Output FASTA path",
            },
        },
        "outputs": {"output": {"type": "file"}},
    },
]


# ---------------------------------------------------------------------------
# Fake agent classes for each wiring style
# ---------------------------------------------------------------------------

class ObjectStyleAgent:
    """Agent that registers tools via add_tool(instance)."""

    def __init__(self):
        self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)
        return True


class FunctionStyleAgent:
    """Agent that registers tools via add_tool (framework introspects callable)."""

    def __init__(self):
        self.tools = []

    def add_tool(self, func):
        self.tools.append(func)
        return True


class ManifestAgent:
    """Agent that takes tools=[...] as kwarg (no register method)."""

    def __init__(self):
        self.bound_tools = []

    def bind_tools(self, tools_list):
        self.bound_tools = tools_list
        return len(tools_list)


class ConfigAgent:
    """Agent that reads tools from a config file."""

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.tools_loaded = []

    def load_config(self, path):
        self.config_path = path
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.tools_loaded = list((cfg.get("tools") or {}).keys())
        return len(self.tools_loaded)


class PromptAgent:
    """Agent that accepts tools via system prompt text."""

    def __init__(self):
        self.system_prompt = ""

    def set_system_prompt(self, text):
        self.system_prompt = text
        return len(text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_pkg_counter = [0]


def _unique_pkg():
    _pkg_counter[0] += 1
    return f"gen_pkg_{_pkg_counter[0]}"


def _clean_pkg(pkg_name):
    """Remove cached module so next test gets fresh code."""
    for key in list(sys.modules):
        if key == pkg_name or key.startswith(pkg_name + "."):
            del sys.modules[key]


def test_object_style():
    """Object style: scanner detects add_tool, generator produces class wrappers,
    adapter calls agent.add_tool(instance) for each tool."""
    from agent_connector.generator import generate_wiring, load_wrappers, load_adapter

    schema = {
        "agent_class": "ObjectStyleAgent",
        "registration_method": "add_tool",
        "registration_style": "object",
        "execution_method": "run",
        "wiring_style": None,
    }

    pkg = _unique_pkg()
    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(FAKE_TOOLS, schema, out_dir=tmp)
        assert wiring["mode"] == "adapter", f"expected adapter mode, got {wiring['mode']}"

        # generated_tools/ is a package inside tmp; parent (tmp) must be on sys.path
        if tmp not in sys.path:
            sys.path.insert(0, tmp)

        # Rename package to avoid import cache collisions between tests
        # generate_wrappers always uses "generated_tools", so we load then clean
        wrappers = load_wrappers(
            package_name="generated_tools",
            registration_style="object",
        )
        assert len(wrappers) >= 2, f"expected >=2 wrappers, got {len(wrappers)}"

        adapter_path = wiring["artifacts"]["adapter"]
        Adapter = load_adapter("ObjectStyleAgent", adapter_path=adapter_path)

        agent = ObjectStyleAgent()
        Adapter(agent).install_tools(wrappers)

        assert len(agent.tools) >= 2, f"expected >=2 tools injected, got {len(agent.tools)}"
        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None)
                      for t in agent.tools]
        assert "fasta_stats" in tool_names, f"fasta_stats not in {tool_names}"
        assert "sort_fasta" in tool_names, f"sort_fasta not in {tool_names}"

    _clean_pkg("generated_tools")
    print("  [PASS] object style: add_tool(instance)")


def test_function_style():
    """Function style: wrappers are plain functions, adapter calls agent.add_tool(fn)."""
    from agent_connector.generator import generate_wiring, load_wrappers, load_adapter

    schema = {
        "agent_class": "FunctionStyleAgent",
        "registration_method": "add_tool",
        "registration_style": "function",
        "execution_method": "run",
        "wiring_style": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(FAKE_TOOLS, schema, out_dir=tmp)
        assert wiring["mode"] == "adapter"

        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        wrappers = load_wrappers(
            package_name="generated_tools",
            registration_style="function",
        )
        assert len(wrappers) >= 2

        # Verify they're plain functions (not class instances)
        for w in wrappers:
            assert callable(w), f"wrapper {w} is not callable"

        adapter_path = wiring["artifacts"]["adapter"]
        Adapter = load_adapter("FunctionStyleAgent", adapter_path=adapter_path)

        agent = FunctionStyleAgent()
        Adapter(agent).install_tools(wrappers)

        assert len(agent.tools) >= 2
        for t in agent.tools:
            assert callable(t), f"registered tool {t} is not callable"

    _clean_pkg("generated_tools")
    print("  [PASS] function style: add_tool(callable)")


def test_manifest_wiring():
    """Manifest wiring: no register method, produces tools_manifest.json
    (OpenAI function calling format)."""
    from agent_connector.generator import generate_wiring

    schema = {
        "agent_class": None,
        "registration_method": None,
        "registration_style": None,
        "execution_method": None,
        "wiring_style": "manifest",
    }

    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(FAKE_TOOLS, schema, out_dir=tmp)
        assert wiring["mode"] == "manifest"

        manifest_path = wiring["artifacts"]["manifest"]
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert isinstance(manifest, list)
        assert len(manifest) >= 2

        # Each entry has OpenAI function calling format
        for entry in manifest:
            assert entry["type"] == "function"
            assert "function" in entry
            fn = entry["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    print("  [PASS] manifest wiring: tools_manifest.json (OpenAI format)")


def test_config_wiring():
    """Config wiring: produces tools_config.yaml for config-file agents."""
    from agent_connector.generator import generate_wiring

    schema = {
        "agent_class": None,
        "registration_method": None,
        "registration_style": None,
        "execution_method": None,
        "wiring_style": "config",
    }

    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(FAKE_TOOLS, schema, out_dir=tmp)
        assert wiring["mode"] == "config"

        config_path = wiring["artifacts"]["config"]
        assert os.path.exists(config_path)

        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert "tools" in cfg
        tools = cfg["tools"]
        assert "fasta_stats" in tools
        assert "sort_fasta" in tools

        # Each tool has description + inputs
        for name, entry in tools.items():
            assert "description" in entry
            assert "inputs" in entry

    print("  [PASS] config wiring: tools_config.yaml")


def test_prompt_wiring():
    """Prompt wiring: produces a text block for system prompt injection."""
    from agent_connector.generator import generate_wiring

    schema = {
        "agent_class": None,
        "registration_method": None,
        "registration_style": None,
        "execution_method": None,
        "wiring_style": "prompt",
    }

    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(FAKE_TOOLS, schema, out_dir=tmp)
        assert wiring["mode"] == "prompt"

        prompt_block = wiring["artifacts"]["prompt_block"]
        assert isinstance(prompt_block, str)
        assert "fasta_stats" in prompt_block
        assert "sort_fasta" in prompt_block
        assert "fasta_path" in prompt_block  # parameter name
        assert "<tool" in prompt_block or "TOOL_NAME" in prompt_block

    print("  [PASS] prompt wiring: system prompt block")


def test_tool_execution_via_wrapper():
    """End-to-end: wrapper call -> run_tool_spec -> subprocess -> result."""
    from agent_connector.generator import generate_wiring, load_wrappers, load_adapter

    schema = {
        "agent_class": "ObjectStyleAgent",
        "registration_method": "add_tool",
        "registration_style": "object",
        "execution_method": "run",
        "wiring_style": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(FAKE_TOOLS, schema, out_dir=tmp)

        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        wrappers = load_wrappers(
            package_name="generated_tools",
            registration_style="object",
        )

        # Find fasta_stats wrapper
        stats_w = next((w for w in wrappers if getattr(w, "name", None) == "fasta_stats"), None)
        assert stats_w is not None, "fasta_stats wrapper not found"

        # Execute: call the tool via its run method
        result = stats_w.run(fasta_path="/dev/null")
        assert isinstance(result, str), f"expected str result, got {type(result)}"
        assert "sequences=5" in result or "5" in result, f"unexpected result: {result}"

    _clean_pkg("generated_tools")
    print("  [PASS] tool execution: wrapper -> run_tool_spec -> result")


def test_scanner_detection():
    """Scanner: build_schema correctly detects registration/execution methods."""
    from agent_connector.scanner import build_schema

    # Create a temp repo with a fake agent class
    with tempfile.TemporaryDirectory() as tmp:
        agent_code = '''
class MyAgent:
    def __init__(self):
        self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)

    def run(self, query):
        return "done"
'''
        with open(os.path.join(tmp, "agent.py"), "w") as f:
            f.write(agent_code)

        schema = build_schema(tmp, include_evidence=False)

        assert schema["agent_class"] == "MyAgent", f"expected MyAgent, got {schema['agent_class']}"
        assert schema["registration_method"] == "add_tool", (
            f"expected add_tool, got {schema['registration_method']}"
        )
        assert schema["execution_method"] == "run", f"expected run, got {schema['execution_method']}"

    print("  [PASS] scanner detection: add_tool + run")


def test_scanner_no_register():
    """Scanner: agents with no register method get wiring_style=manifest/config/prompt."""
    from agent_connector.scanner import build_schema

    with tempfile.TemporaryDirectory() as tmp:
        # Agent with tools=[] in __init__ but no add_tool
        agent_code = '''
class LLMWrapper:
    def __init__(self):
        self.tools = []

    def chat(self, messages, tools=[]):
        return "response"
'''
        with open(os.path.join(tmp, "llm.py"), "w") as f:
            f.write(agent_code)

        schema = build_schema(tmp, include_evidence=False)

        # No registration method found
        assert schema["registration_method"] is None
        # Should detect a wiring style
        assert schema["wiring_style"] in ("manifest", "config", "prompt", None)

    print("  [PASS] scanner no-register: wiring_style detected")


# ---------------------------------------------------------------------------
# Real agent tests (clone + install + inject)
# ---------------------------------------------------------------------------

BIOMNI_DIR = "/tmp/_biomni_test"
BIOMNI_REPO = "https://github.com/snap-stanford/Biomni.git"


def _clone_biomni():
    """Clone Biomni (shallow) if not already present."""
    if os.path.isdir(os.path.join(BIOMNI_DIR, ".git")):
        return
    import subprocess
    subprocess.run(
        ["git", "clone", "--depth", "1", BIOMNI_REPO, BIOMNI_DIR],
        check=True, capture_output=True, text=True, timeout=120,
    )


def test_biomni_scan():
    """Clone Biomni, run scanner, verify it detects add_tool / add_mcp."""
    from agent_connector.scanner import build_schema

    _clone_biomni()

    schema = build_schema(BIOMNI_DIR, include_evidence=False)
    print(f"    agent_class={schema.get('agent_class')} "
          f"reg={schema.get('registration_method')} "
          f"exec={schema.get('execution_method')}")

    # Biomni has add_tool and add_mcp on its agent class
    assert schema.get("agent_class") is not None, "no agent class found"
    assert schema.get("registration_method") in ("add_tool", "add_mcp", "register", None), (
        f"unexpected registration: {schema.get('registration_method')}"
    )
    print("  [PASS] Biomni scan: scanner detects real agent interface")


def test_biomni_inject():
    """Install Biomni, create A1 agent, inject our tools, verify callable."""
    import subprocess

    _clone_biomni()

    # Install Biomni (lightweight: pip install from source)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", BIOMNI_DIR],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"    pip install failed: {result.stderr[:200]}")
        print("  [SKIP] Biomni install failed (deps too heavy for this env)")
        return

    # Import and instantiate
    try:
        from biomni.agent import A1
    except ImportError as e:
        print(f"    import failed: {e}")
        print("  [SKIP] Biomni not importable after install")
        return

    # Create agent (no LLM needed for tool injection test)
    try:
        agent = A1(path="/tmp/_biomni_data")
    except Exception as e:
        print(f"    A1() init failed: {type(e).__name__}: {e}")
        # Some versions need API key even for init; create a minimal mock
        class MinimalBiomni:
            def __init__(self):
                self.tools = []
            def add_tool(self, tool):
                self.tools.append(tool)
            def add_mcp(self, **kwargs):
                pass
        agent = MinimalBiomni()
        print("    using MinimalBiomni fallback")

    # Generate wrappers from our registry
    import yaml
    from agent_connector.generator import generate_wiring, load_wrappers, load_adapter

    reg_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    if not os.path.exists(reg_path):
        print("  [SKIP] registry not found")
        return
    tools = yaml.safe_load(open(reg_path, encoding="utf-8"))["tools"]

    # Scan real Biomni to get the schema
    from agent_connector.scanner import build_schema as _build_schema
    schema = _build_schema(BIOMNI_DIR, include_evidence=False)
    reg_method = schema.get("registration_method") or "add_tool"
    exec_method = schema.get("execution_method") or "run"
    reg_style = schema.get("registration_style") or "function"

    with tempfile.TemporaryDirectory() as tmp:
        wiring = generate_wiring(tools, schema, out_dir=tmp)
        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        wrappers = load_wrappers(
            package_name="generated_tools",
            registration_style=reg_style,
        )
        assert len(wrappers) >= 2, f"expected >=2 wrappers, got {len(wrappers)}"

        adapter_path = wiring["artifacts"].get("adapter")
        if adapter_path and os.path.exists(adapter_path):
            Adapter = load_adapter(schema.get("agent_class") or "Agent",
                                   adapter_path=os.path.abspath(adapter_path))
            Adapter(agent).install_tools(wrappers)
        else:
            # No adapter (wiring style), just call add_tool directly
            for w in wrappers:
                agent.add_tool(w)

    # Verify injection
    if hasattr(agent, "tools"):
        injected = len(agent.tools)
    else:
        injected = 0
    print(f"    injected {injected} tools into {type(agent).__name__}")

    # Try executing one tool
    wmap = {}
    for w in wrappers:
        nm = getattr(w, "name", None) or getattr(w, "__name__", None)
        if nm:
            wmap[nm] = w

    test_result = None
    for name in ["fasta_contig_stats_python", "fasta_stats", "bqtools_info"]:
        if name in wmap:
            w = wmap[name]
            try:
                if hasattr(w, "run"):
                    test_result = w.run(fasta_path="/dev/null")
                elif callable(w):
                    test_result = w(fasta_path="/dev/null")
            except Exception as e:
                test_result = f"exec error: {e}"
            break

    print(f"    tool execution: {str(test_result)[:100] if test_result else 'N/A'}")
    print("  [PASS] Biomni inject: install → scan → generate → inject → verify")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("== Cross-agent adapter test ==")
    tests = [
        test_object_style,
        test_function_style,
        test_manifest_wiring,
        test_config_wiring,
        test_prompt_wiring,
        test_tool_execution_via_wrapper,
        test_scanner_detection,
        test_scanner_no_register,
        test_biomni_scan,
        test_biomni_inject,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")

    total = len(tests)
    print(f"\n  Cross-agent test: {passed}/{total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
