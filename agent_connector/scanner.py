"""AST-based discovery of how an arbitrary agent framework registers and
executes tools. Rule-based by default; optionally refined by an LLM
(see resolver.py).

The scanner does not need the target framework installed: it only reads
source code and produces a structured agent_schema.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent-connector.scanner")

# --- keyword heuristics -------------------------------------------------
REGISTRATION_HINTS = ("register", "add_tool", "add_mcp", "install", "attach", "append")
TOOL_NAME_HINTS = ("tool", "function", "action", "skill", "api", "mcp", "plugin")
EXECUTION_METHOD_HINTS = ("run", "execute", "call", "invoke", "use", "apply", "go", "__call__")
TOOL_SCHEMA_FIELDS = ("name", "description", "command", "parameters", "input_schema", "schema")

# Known agent frameworks → execution method (key = lowercase substring in class/module path)
KNOWN_AGENT_EXECUTION: dict[str, str] = {
    "biomni": "go",
    "cellagent": "run",
    "geneagent": "run",
    "crispr": "run",
    "biochatter": "run",
    "langchain": "invoke",
    "autogpt": "run",
    "crewai": "kickoff",
    "metagpt": "run",
    "camel": "step",
    "openai": "run",
    "smolagents": "run",
    "dspy": "forward",
}

# Probe order for unknown agents (try these method names in order)
_PROBE_ORDER = ("go", "run", "execute", "predict", "forward", "invoke")
SKIP_DIR_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    "__pycache__",
    ".tox",
    ".eggs",
    ".idea",
}


class FunctionInfo:
    """Lightweight wrapper around an AST function/class-node with extra info."""

    __slots__ = ("node", "class_name", "file", "line")

    def __init__(self, node: ast.AST, class_name: str, file: str, line: int) -> None:
        self.node = node
        self.class_name = class_name
        self.file = file
        self.line = line

    @property
    def name(self) -> str:
        return getattr(self.node, "name", "")

    def body_text(self) -> str:
        try:
            return "\n".join(ast.unparse(stmt) for stmt in self.node.body)
        except Exception:
            return ""

    @property
    def params(self) -> list[str]:
        args = self.node.args.args  # type: ignore[attr-defined]
        return [a.arg for a in args]

    @property
    def decorators(self) -> list[str]:
        out = []
        for dec in getattr(self.node, "decorator_list", []):
            try:
                out.append(ast.unparse(dec))
            except Exception:
                pass
        return out


class ClassInfo:
    __slots__ = ("node", "file", "line")

    def __init__(self, node: ast.ClassDef, file: str, line: int) -> None:
        self.node = node
        self.file = file
        self.line = line

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def bases(self) -> list[str]:
        out = []
        for base in self.node.bases:
            try:
                out.append(ast.unparse(base))
            except Exception:
                pass
        return out

    def attributes(self) -> list[str]:
        out = []
        for stmt in self.node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        out.append(target.id)
        return out


class RepoModel:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.files: list[str] = []
        self.functions: list[FunctionInfo] = []
        self.methods: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []

    def parse_errors(self) -> list[dict[str, Any]]:
        return getattr(self, "_parse_errors", [])


def _skip_dir(parts: tuple[str, ...]) -> bool:
    return any(part in SKIP_DIR_PARTS or part.startswith(".") for part in parts)


def scan_python_files(repo_path: str) -> list[str]:
    root = Path(repo_path).resolve()
    files: list[str] = []
    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if not _skip_dir((Path(current_root).name, d))]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(str(Path(current_root) / filename))
    return files


def _collect(repo_path: str) -> RepoModel:
    model = RepoModel(repo_path)
    model.files = scan_python_files(repo_path)
    model._parse_errors = []  # type: ignore[attr-defined]

    for file in model.files:
        try:
            tree = ast.parse(Path(file).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            model._parse_errors.append({"file": file, "error": str(exc)})  # type: ignore[attr-defined]
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.AsyncFunctionDef):
                    continue
                info = FunctionInfo(node, "", file, getattr(node, "lineno", 0))
                model.functions.append(info)
            elif isinstance(node, ast.ClassDef):
                model.classes.append(ClassInfo(node, file, getattr(node, "lineno", 0)))

    # Map methods into their class.
    for cls in model.classes:
        for stmt in cls.node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(stmt, ast.AsyncFunctionDef):
                    continue
                model.methods.append(FunctionInfo(stmt, cls.name, cls.file, stmt.lineno))

    return model


# --- classification rules ------------------------------------------------
def _decorator_is_tool_registration(decorators: list[str]) -> bool:
    for dec in decorators:
        cleaned = dec.replace(" ", "")
        if cleaned in {"@tool", "@tool_use", "@register_tool", "@mcp.tool()", "@mcp.tool"}:
            return True
        if "@tool(" in cleaned or "mcp.tool(" in cleaned:
            return True
    return False


def _detect_registration_style(name: str, body_lower: str, params: list[str]) -> str:
    """Classify how the framework expects the 'tool' argument to look:
    function (a callable it introspects), dict (schema mapping), or object
    (a tool class instance)."""
    if "inspect.getsource" in body_lower or "function_to_api_schema" in body_lower or "function_to_schema" in body_lower:
        return "function"
    if "required_parameters" in body_lower or "required_keys" in body_lower or "validate_tool" in body_lower:
        return "dict"
    tool_params = [p for p in params if p not in {"self", "cls", "this"}]
    if tool_params and any("tool" in p.lower() or "api" in p.lower() for p in tool_params):
        if ".append(" in body_lower:
            return "object"
    return "object"


def find_registration_candidates(model: RepoModel) -> list[dict[str, Any]]:
    """Return functions/methods that look like 'register a tool into the agent'."""
    candidates: list[dict[str, Any]] = []
    for info in model.functions + model.methods:
        name_lower = info.name.lower()
        body_lower = info.body_text().lower()
        decs = info.decorators

        is_decorator_reg = _decorator_is_tool_registration(decs)
        has_reg_hint = any(h in name_lower for h in REGISTRATION_HINTS)
        has_tool_hint = any(h in name_lower for h in TOOL_NAME_HINTS)
        body_manages_tools = any(
            token in body_lower for token in ("tools", "tool", "functions", "mcp")
        )

        if not (is_decorator_reg or (has_reg_hint and (has_tool_hint or body_manages_tools))):
            continue

        params = info.params
        tool_params = [p for p in params if p not in {"self", "cls", "this"}]
        storage = None
        if ".tools.append" in body_lower:
            storage = "<obj>.tools"
        elif "self.tools" in body_lower:
            storage = "self.tools"

        candidates.append(
            {
                "name": info.name,
                "class_name": info.class_name,
                "file": info.file,
                "line": info.line,
                "params": params,
                "tool_param_candidates": tool_params,
                "registration_style": _detect_registration_style(info.name, body_lower, params),
                "storage": storage,
                "decorator": is_decorator_reg,
                "decorators": decs,
                "evidence_snippet": info.body_text()[:300],
            }
        )
    return candidates


def find_execution_candidates(model: RepoModel) -> list[dict[str, Any]]:
    """Return methods that execute a tool object (name in EXECUTION_METHOD_HINTS)."""
    candidates: list[dict[str, Any]] = []
    for info in model.methods:
        name_lower = info.name.lower()
        if not any(name_lower.startswith(h) or name_lower == h for h in EXECUTION_METHOD_HINTS):
            continue
        body_lower = info.body_text().lower()
        # A tool-execution method usually touches tool inputs/outputs or dispatches.
        dispatches = any(token in body_lower for token in ("dispatch", "kwargs", "arguments", "inputs"))
        candidates.append(
            {
                "name": info.name,
                "class_name": info.class_name,
                "file": info.file,
                "line": info.line,
                "params": info.params,
                "dispatches_args": dispatches,
                "evidence_snippet": info.body_text()[:300],
            }
        )
    return candidates


def find_tool_class_candidates(model: RepoModel) -> list[dict[str, Any]]:
    """Return classes that look like the 'tool object' (has name/description/command fields)."""
    candidates: list[dict[str, Any]] = []
    for cls in model.classes:
        attributes = cls.attributes()
        name_lower = cls.name.lower()
        has_tool_hint = any(h in name_lower for h in TOOL_NAME_HINTS)
        has_schema_field = any(field in attributes for field in TOOL_SCHEMA_FIELDS)
        subclass_of_tool = any("tool" in base.lower() or "base" in base.lower() for base in cls.bases)
        if (has_tool_hint or has_schema_field or subclass_of_tool) and not name_lower.startswith("_"):
            candidates.append(
                {
                    "name": cls.name,
                    "file": cls.file,
                    "line": cls.line,
                    "bases": cls.bases,
                    "attributes": attributes,
                }
            )
    return candidates


_INVOCATION_RE = re.compile(r"\.(invoke|run|execute|go|__call__|call)\s*\(")


def detect_execution_method(files: list[str]) -> tuple[str | None, int, str | None]:
    """Scan source for `.<method>(` calls on tool-like objects and return the
    most common one. Useful when the framework relies on an external tool
    protocol (e.g. LangChain `.invoke`) not defined inside its own source."""
    counts: dict[str, int] = {}
    example_file: dict[str, str] = {}
    for file in files:
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _INVOCATION_RE.finditer(text):
            method = match.group(1)
            counts[method] = counts.get(method, 0) + 1
            example_file.setdefault(method, file)
    if not counts:
        return None, 0, None
    method, count = max(counts.items(), key=lambda item: item[1])
    return method, count, example_file.get(method)


def resolve_execution_method(
    agent_class: str | None,
    module_path: str | None,
    scanned_source: str = "",
) -> str:
    """Determine execution method: known mapping → source hint → probe order.

    Priority:
      1. KNOWN_AGENT_EXECUTION match (class/module name substring)
      2. Source code evidence (`.go(`, `.run(` etc.)
      3. Fallback: first of _PROBE_ORDER
    """
    lookup = f"{(agent_class or '').lower()} {(module_path or '').lower()}"
    for pattern, method in KNOWN_AGENT_EXECUTION.items():
        if pattern in lookup:
            return method
    if scanned_source:
        for m in ("go", "run", "execute", "predict", "forward", "invoke"):
            if f".{m}(" in scanned_source:
                return m
    return _PROBE_ORDER[0]


def probe_execution_method(agent_obj: object, candidates: list[str] | None = None) -> str | None:
    """Runtime probe: check which execution method the agent object actually has.

    Tries candidates in order, returns the first method name that is callable.
    Returns None if nothing works.
    """
    methods = candidates or list(_PROBE_ORDER)
    for name in methods:
        method = getattr(agent_obj, name, None)
        if callable(method):
            return name
    return None


# --- import path + constructor extraction --------------------------------

_IMPORT_RE = re.compile(
    r"^\s*from\s+([\w.]+)\s+import\s+(\w+)", re.MULTILINE
)


def find_import_paths(model: RepoModel) -> dict[str, str]:
    """Scan all source for 'from X import Y' and return {ClassName: module_path}.

    Only returns entries where Y matches a class found in the AST.
    E.g. {'A1': 'biomni.agent.A1'} from 'from biomni.agent.A1 import A1'.
    """
    class_names = {cls.name for cls in model.classes}
    result: dict[str, str] = {}
    for file in model.files:
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _IMPORT_RE.finditer(text):
            module_path, class_name = match.group(1), match.group(2)
            if class_name in class_names:
                result[class_name] = module_path
    return result


def find_init_signatures(model: RepoModel) -> dict[str, dict[str, Any]]:
    """Extract __init__ signatures for agent classes.

    Returns {class_name: {'params': [...], 'defaults': {...}, 'file': str}}.
    """
    result: dict[str, dict[str, Any]] = {}
    for info in model.methods:
        if info.name != "__init__":
            continue
        if not info.class_name:
            continue
        node = info.node
        defaults: dict[str, Any] = {}
        params: list[str] = []
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            params.append(arg.arg)
        # defaults are aligned to the end of args
        num_defaults = len(node.args.defaults)
        if num_defaults > 0:
            defaulted_params = params[-num_defaults:]
            for pname, dnode in zip(defaulted_params, node.args.defaults):
                try:
                    defaults[pname] = ast.literal_eval(dnode)
                except (ValueError, TypeError):
                    defaults[pname] = None
        result[info.class_name] = {
            "params": params,
            "defaults": defaults,
            "file": info.file,
        }
    return result


MCP_STRONG_EVIDENCE = (
    "langchain_mcp_adapters",
    "MultiServerMCPClient",
    "FastMCP",
    "from mcp import ClientSession",
    "from mcp import StdioServerParameters",
    "mcp.server",
    "mcp.client",
    "add_mcp",
)
MCP_CONFIG_NAMES = (".mcp.json", "mcp.json", "mcp_config.yaml",
                    "mcp_config.yml", "mcp_config.json")
NATIVE_REG_METHODS = ("add_tool", "register_tool", "register", "add_mcp",
                      "with_tools", "bind_tools")
CODE_EXEC_HINT_RE = re.compile(
    r"(<execute>|exec\(|__import__|code_execution|repl|ipython|run_code|execute_code|jupyter)",
    re.I,
)


def detect_capabilities(repo_path: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Classify the agent by *how* it can accept tools, with MCP as the
    highest-priority, most standard capability.

    Returns a structured ``capabilities`` dict that ``generate_wiring`` uses to
    pick an integration mode by the priority:

        mcp  >  native_tool_calling  >  code_execution  >  config_wiring  >  prompt_wiring

    ``mcp`` carries an ``evidence`` list so the caller knows *why* MCP was
    detected (avoids false positives from a bare "mcp" substring).
    """
    root = Path(repo_path).resolve()
    mcp_evidence: list[str] = []

    for file in scan_python_files(repo_path):
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for token in MCP_STRONG_EVIDENCE:
            if token in text and token not in mcp_evidence:
                mcp_evidence.append(token)

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if not _skip_dir((Path(current_root).name, d))]
        for filename in filenames:
            low = filename.lower()
            if low in MCP_CONFIG_NAMES or (
                low.startswith("mcp_config") and low.endswith((".yaml", ".yml", ".json"))
            ):
                if "config:" + filename not in mcp_evidence:
                    mcp_evidence.append("config:" + filename)

    reg_method = schema.get("registration_method")
    if reg_method == "add_mcp" and "registration_method=add_mcp" not in mcp_evidence:
        mcp_evidence.append("registration_method=add_mcp")

    mcp_supported = bool(mcp_evidence)

    native_tool_calling = bool(reg_method in NATIVE_REG_METHODS) if reg_method else False

    code_execution = False
    for file in scan_python_files(repo_path):
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if CODE_EXEC_HINT_RE.search(text):
            code_execution = True
            break

    wiring = detect_wiring_style(repo_path, reg_method)
    config_wiring = wiring == "config"
    prompt_wiring = wiring == "prompt"

    return {
        "mcp": {"supported": mcp_supported, "evidence": sorted(set(mcp_evidence))},
        "native_tool_calling": bool(native_tool_calling),
        "code_execution": bool(code_execution),
        "config_wiring": bool(config_wiring),
        "prompt_wiring": bool(prompt_wiring),
        "wiring_style": wiring,
    }


# --- interface contract -------------------------------------------------
# The resolver keys off the *interface contract* (registration method,
# registration style, schema format, execution method), NOT the framework
# name. Two agents can share a framework (e.g. both "langchain") yet require
# different tool schemas (StructuredTool vs OpenAI function vs bind_tools).

def _derive_framework(schema: dict[str, Any]) -> str | None:
    """Best-effort framework hint (informational only; NOT used for adapter
    selection). Derived from module_path/agent_class, never authoritative."""
    text = " ".join(str(schema.get(k) or "") for k in ("module_path", "agent_class"))
    text = text.lower()
    for name in ("biomni", "cellagent", "geneagent", "biochatter", "langchain",
                 "crewai", "autogpt", "metagpt", "camel", "smolagents", "dspy",
                 "openai", "huggingface", "transformers"):
        if name in text:
            return name
    return None


def _cap_bool(caps: dict[str, Any] | None, key: str) -> bool:
    """Read a capability flag that may be a bool OR a {supported: bool} dict
    (detect_capabilities returns booleans for most keys, a dict for 'mcp')."""
    v = (caps or {}).get(key)
    if isinstance(v, dict):
        return bool(v.get("supported"))
    return bool(v)


def detect_tool_interface(schema: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    """Produce the explicit tool interface contract for this agent.

    fields:
      framework           -- informational hint only (NOT the selector)
      registration_method -- how a tool is registered (add_tool/add_mcp/...)
      registration_style  -- function | dict | object
      schema_format       -- mcp | function | structured_tool | langchain_tool
                             | openai_function | namespace | config | prompt
      execution_method    -- go | run | invoke | ...
    """
    reg = schema.get("registration_method")
    style = schema.get("registration_style")
    via_dec = schema.get("registration_via_decorator")
    mcp_ok = _cap_bool(caps, "mcp")
    native_ok = _cap_bool(caps, "native_tool_calling")
    code_ok = _cap_bool(caps, "code_execution")
    config_ok = _cap_bool(caps, "config_wiring")
    prompt_ok = _cap_bool(caps, "prompt_wiring")

    if mcp_ok or reg == "add_mcp":
        schema_format = "mcp"
    elif reg in ("bind_tools", "with_tools"):
        schema_format = "openai_function"
    elif reg == "add_tool" and via_dec:
        schema_format = "structured_tool"
    elif reg == "add_tool" and style == "function":
        schema_format = "function"
    elif reg in ("add_tool", "register_tool", "register") or style in ("function", "object", "dict"):
        schema_format = "langchain_tool"
    elif code_ok:
        schema_format = "namespace"
    elif config_ok:
        schema_format = "config"
    elif prompt_ok:
        schema_format = "prompt"
    else:
        schema_format = "unknown"

    return {
        "framework": _derive_framework(schema),
        "registration_method": reg,
        "registration_style": style,
        "schema_format": schema_format,
        "execution_method": schema.get("execution_method"),
    }


def build_schema(
    repo_path: str,
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Run the full rule-based analysis and produce an agent_schema dict.

    Fields filled by heuristics; 'confidence' tells the caller how sure we
    are. An LLM pass (resolver.py) can confirm/override these.
    """
    model = _collect(repo_path)
    registrations = find_registration_candidates(model)
    executions = find_execution_candidates(model)
    tool_classes = find_tool_class_candidates(model)

    # Prefer methods/classes defined on a class whose name contains Agent/Executor,
    # then standalone functions.
    def rank_score(entry: dict[str, Any]) -> int:
        score = 0
        cls = entry.get("class_name") or ""
        name = entry.get("name") or ""
        if any(token in cls.lower() for token in ("agent", "executor", "react", "runner")):
            score += 3
        if entry.get("storage"):
            score += 2
        if entry.get("decorator"):
            score += 2
        if name in {"register_tool", "add_tool", "add_mcp"}:
            score += 2
        if name in {"execute", "run", "_run"}:
            score += 2
        return score

    best_reg = max(registrations, key=rank_score, default=None)
    best_exec = max(executions, key=rank_score, default=None)

    invoked_method, invoked_count, invoked_file = detect_execution_method(model.files)
    if best_exec is None and invoked_method:
        best_exec = {
            "name": invoked_method,
            "class_name": None,
            "file": invoked_file or "",
            "line": 0,
            "params": [],
            "dispatches_args": False,
            "evidence_snippet": "",
        }
    if best_exec and best_exec.get("file"):
        best_exec["execution_evidence"] = f"{best_exec['file']}:{best_exec['line']}"
        best_exec["invocation_count"] = invoked_count

    # Determine the agent/executor class name (where best candidates live).
    agent_class = best_reg.get("class_name") if best_reg else (best_exec.get("class_name") if best_exec else None)

    # Extract import paths and init signatures
    import_paths = find_import_paths(model)
    init_sigs = find_init_signatures(model)

    # Fallback: derive module_path from agent class file location
    def _file_to_module(filepath: str) -> str | None:
        """Convert e.g. /repo/biomni/agent/react.py -> biomni.agent.react"""
        # Find the real package root: walk up until we find a dir without __init__.py
        base = Path(model.repo_path).resolve()
        while (base / "__init__.py").exists() and base.parent != base:
            base = base.parent
        try:
            rel = os.path.relpath(filepath, str(base))
        except ValueError:
            return None
        parts = Path(rel).with_suffix("").parts
        if len(parts) < 2:
            return None
        parts = [p for p in parts if p != "__init__"]
        return ".".join(parts) if parts else None

    module_path = import_paths.get(agent_class) if agent_class else None
    if not module_path and agent_class:
        # Try from registration file
        if best_reg and best_reg.get("file"):
            module_path = _file_to_module(best_reg["file"])
        # Try from init signature file
        if not module_path and agent_class in init_sigs:
            module_path = _file_to_module(init_sigs[agent_class].get("file", ""))

    schema: dict[str, Any] = {
        "repo_path": str(Path(repo_path).resolve()),
        "wiring_style": detect_wiring_style(repo_path, best_reg["name"] if best_reg else None),
        "agent_class": agent_class,
        "module_path": module_path,
        "init_signature": init_sigs.get(agent_class) if agent_class else None,
        "registration_method": best_reg["name"] if best_reg else None,
        "registration_argument": (
            best_reg["tool_param_candidates"][0] if best_reg and best_reg["tool_param_candidates"] else None
        ),
        "registration_style": best_reg.get("registration_style") if best_reg else None,
        "registration_storage": best_reg.get("storage") if best_reg else None,
        "registration_via_decorator": bool(best_reg and best_reg.get("decorator")),
        "execution_method": best_exec["name"] if best_exec else None,
        "execution_class": best_exec.get("class_name") if best_exec else None,
        "execution_candidates": [e["name"] for e in executions] if executions else [],
        "tool_class": tool_classes[0]["name"] if tool_classes else None,
        "tool_class_fields": tool_classes[0]["attributes"] if tool_classes else [],
        "confidence": 0.7 if (best_reg and best_exec) else 0.3,
        "scanned_files": len(model.files),
        "parse_errors": model.parse_errors()[:10],
        "registrations": registrations[:10] if include_evidence else None,
        "executions": executions[:10] if include_evidence else None,
        "tool_classes": tool_classes[:10] if include_evidence else None,
    }
    # First-class capability classification (MCP is the highest-priority,
    # most standard integration path and is checked before wiring_style).
    schema["capabilities"] = detect_capabilities(repo_path, schema)
    # Explicit interface contract (selection key), derived from capabilities +
    # registration evidence. Framework is only a hint inside this block.
    schema["framework"] = _derive_framework(schema)
    schema["tool_interface"] = detect_tool_interface(schema, schema["capabilities"])
    return schema


CONFIG_FILE_HINTS = ("tool", "registry", "config", "mcp", "plugin")
WIRING_TOOLS_KWARG_RE = re.compile(r"(?<!\.)\btools\s*=\s*(\[|\w)", re.MULTILINE)


def detect_wiring_style(repo_path: str, registration_method: str | None = None) -> str | None:
    """Choose how tools can be wired into an agent that has no register method.

    Returns one of:
      None     -- a normal registration method exists (adapter path)
      manifest -- tools are passed as a `tools=[...]` list to each LLM call
      config   -- tools are defined in a config file the agent reads
      prompt   -- tools must be described as text in the system prompt
    """
    if registration_method:
        return None

    root = Path(repo_path).resolve()

    for file in scan_python_files(repo_path):
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if WIRING_TOOLS_KWARG_RE.search(text):
            return "manifest"

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if not _skip_dir((Path(current_root).name, d))]
        for filename in filenames:
            if filename.endswith((".yaml", ".yml", ".json", ".toml")):
                stem = Path(filename).stem.lower()
                if any(hint in stem for hint in CONFIG_FILE_HINTS):
                    return "config"

    return "prompt"
