"""Agent Preflight: check and bootstrap a downstream agent's environment
before tool injection.

Two-layer dependency model:
  Layer 1 — Agent runtime deps: imports needed to *instantiate* the agent
            class (traced from the agent entry point's import closure).
  Layer 2 — Tool runtime deps:  imports needed when a specific tool is
            invoked (scanned from tool subdirectories).  Reported but NOT
            blocking — installed lazily when the tool is actually selected.

Status codes:
  READY                  - all agent runtime deps satisfied
  SETUP_REQUIRED         - missing agent runtime deps that can be auto-installed
  USER_ACTION_REQUIRED   - needs API keys, models, data, etc.
  UNSUPPORTED            - cannot determine or resolve dependencies
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent-connector.preflight")

# Status constants
READY = "READY"
SETUP_REQUIRED = "SETUP_REQUIRED"
USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
UNSUPPORTED = "UNSUPPORTED"

# Heavy packages that should NOT be auto-installed (too slow / needs GPU)
HEAVY_PACKAGES = frozenset({
    "torch", "torchvision", "torchaudio", "tensorflow", "jax",
    "cupy", "paddle", "triton", "transformers", "diffusers",
    "deepchem", "rdkit", "openfold", "esm", "alphafold",
})

# Packages that are importable under a different name than the pip name
IMPORT_TO_PIP = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "Bio": "biopython",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "magic": "python-magic",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "dotenv": "python-dotenv",
    "magic": "python-magic",
}

# Standard library modules (Python 3.9+)
_STDLIB: set[str] = set()

def _stdlib() -> set[str]:
    global _STDLIB
    if not _STDLIB:
        _STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
    return _STDLIB


@dataclass
class DepCheck:
    name: str
    installed: bool = False
    import_name: str = ""
    pip_name: str = ""
    source: str = ""        # "agent_runtime", "tool", "manifest", "requirements"
    layer: str = ""         # "runtime" or "tool"
    heavy: bool = False
    auto_installable: bool = True


@dataclass
class EnvCheck:
    name: str
    set: bool = False
    value_preview: str = ""  # first 8 chars, masked


@dataclass
class PreflightResult:
    status: str = UNSUPPORTED
    python_ok: bool = False
    python_version: str = ""

    # Layer 1: agent runtime deps (blocking)
    deps: list[DepCheck] = field(default_factory=list)
    # Layer 2: tool-specific deps (informational)
    tool_deps: list[DepCheck] = field(default_factory=list)

    env_vars: list[EnvCheck] = field(default_factory=list)
    pip_installable: list[str] = field(default_factory=list)
    pip_heavy: list[str] = field(default_factory=list)
    user_action_needed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_found: bool = False
    inferred: bool = False
    agent_entry: str = ""   # e.g. "biomni.agent.react"
    tool_dirs_scanned: list[str] = field(default_factory=list)


def _pip_name_for(import_name: str) -> str:
    return IMPORT_TO_PIP.get(import_name, import_name)


def _is_installed(pkg: str) -> bool:
    import_name = pkg.replace("-", "_").split("[")[0]
    try:
        cp = subprocess.run(
            [sys.executable, "-c", f"import {import_name}"],
            capture_output=True, text=True, timeout=10,
        )
        return cp.returncode == 0
    except Exception:
        return False


def _check_installed_batch(pkgs: list[str]) -> dict[str, bool]:
    if not pkgs:
        return {}
    import_names = []
    for pkg in pkgs:
        imp = pkg.replace("-", "_").split("[")[0]
        import_names.append((pkg, imp))
    lines = ["import importlib, json, sys"]
    lines.append("results = {}")
    for orig, imp in import_names:
        safe_imp = imp.replace("'", "\\'")
        lines.append(f"try:")
        lines.append(f"    importlib.import_module('{safe_imp}')")
        lines.append(f"    results['{orig}'] = True")
        lines.append(f"except Exception:")
        lines.append(f"    results['{orig}'] = False")
    lines.append("print(json.dumps(results))")
    script = "\n".join(lines)
    try:
        cp = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            return json.loads(cp.stdout.strip())
    except Exception:
        pass
    return {pkg: False for pkg in pkgs}


# ---------------------------------------------------------------------------
# Import closure tracing  (replaces whole-repo _infer_imports_from_source)
# ---------------------------------------------------------------------------

def _parse_file_imports(filepath: str) -> list[tuple[str, str]]:
    """Parse a Python file, return [(top_level_import, full_module_or_empty)].

    Returns tuples like:
      ("os", "")           — import os
      ("langchain_core", "langchain_core.tools")  — from langchain_core.tools import tool
    """
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imports.append((top, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            imports.append((top, node.module))
    return imports


def _resolve_module_file(module_path: str, package_root: str) -> str | None:
    """Given e.g. 'biomni.config' and repo root, find the .py file."""
    parts = module_path.split(".")
    candidate = os.path.join(package_root, *parts) + ".py"
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(package_root, *parts, "__init__.py")
    if os.path.isfile(candidate):
        return candidate
    return None


def trace_agent_imports(
    agent_module_path: str,
    repo_root: str,
    max_depth: int = 8,
) -> tuple[set[str], set[str]]:
    """Trace the import closure starting from an agent entry point.

    Returns (external_top_level_packages, internal_module_paths).

    External = anything outside the agent's own package (e.g. langgraph, langchain_core).
    Internal = modules within the same top-level package (e.g. biomni.config, biomni.llm).
    """
    if not agent_module_path or not repo_root:
        return set(), set()

    # Determine the top-level package name
    top_pkg = agent_module_path.split(".")[0]

    # Find the actual package root directory.
    # Two cases:
    #   1. Cloned repo: repo_root/biomni/agent/react.py  → search_start = repo_root
    #   2. Site-packages: repo_root IS biomni/ → search_start = parent of repo_root
    search_start = repo_root
    direct_pkg = os.path.join(repo_root, top_pkg)
    if not os.path.isdir(direct_pkg) or not os.path.exists(os.path.join(direct_pkg, "__init__.py")):
        # repo_root itself might be the package dir
        if os.path.exists(os.path.join(repo_root, "__init__.py")):
            search_start = os.path.dirname(repo_root)

    external: set[str] = set()
    internal: set[str] = set()
    visited: set[str] = set()

    def _trace(module: str, depth: int) -> None:
        if depth > max_depth or module in visited:
            return
        visited.add(module)

        filepath = _resolve_module_file(module, search_start)
        if filepath is None:
            return

        for top_name, full_mod in _parse_file_imports(filepath):
            if top_name in _stdlib() or top_name.startswith("_"):
                continue

            if top_name == top_pkg or full_mod.startswith(top_pkg + "."):
                # Internal import — recurse
                if full_mod not in visited:
                    internal.add(full_mod)
                    _trace(full_mod, depth + 1)
            else:
                # External import
                external.add(top_name)

    _trace(agent_module_path, 0)
    return external, internal


def trace_tool_imports(
    tool_dirs: list[str],
    agent_package: str,
    max_files: int = 200,
) -> set[str]:
    """Scan tool-specific subdirectories for external imports.

    These are deps that tools need when invoked, NOT at agent startup.
    """
    external: set[str] = set()
    count = 0
    for tool_dir in tool_dirs:
        if not os.path.isdir(tool_dir):
            continue
        for root, dirs, files in os.walk(tool_dir):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "venv")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                count += 1
                if count > max_files:
                    return external
                filepath = os.path.join(root, f)
                for top_name, _full_mod in _parse_file_imports(filepath):
                    if top_name in _stdlib() or top_name.startswith("_"):
                        continue
                    if top_name != agent_package:
                        external.add(top_name)
    return external


# ---------------------------------------------------------------------------
# Manifest parsers (unchanged)
# ---------------------------------------------------------------------------

def _parse_requirements_txt(path: str) -> list[str]:
    pkgs = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pkgs
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        pkg = re.split(r"[>=<!\[\];@]", line)[0].strip()
        if pkg:
            pkgs.append(pkg)
    return pkgs


def _parse_setup_py(path: str) -> list[str]:
    pkgs = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception:
        return pkgs
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "install_requires":
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        pkg = re.split(r"[>=<!\[\];@]", elt.value)[0].strip()
                        if pkg:
                            pkgs.append(pkg)
    return pkgs


def _parse_pyproject_toml(path: str) -> list[str]:
    pkgs = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pkgs
    in_deps = False
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if stripped.startswith("[project.") or stripped.startswith("["):
            in_project = False
        if stripped in ("dependencies = [", "dependencies=["):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("]"):
                in_deps = False
                continue
            m = re.search(r'["\']([^"\']+)["\']', stripped)
            if m:
                pkg = re.split(r"[>=<!\[\];@]", m.group(1))[0].strip()
                if pkg:
                    pkgs.append(pkg)
    return pkgs


def _parse_environment_yml(path: str) -> tuple[list[str], list[str]]:
    pip_pkgs = []
    conda_pkgs = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pip_pkgs, conda_pkgs
    in_pip = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("pip:"):
            in_pip = True
            continue
        if stripped and not stripped.startswith("-") and not stripped.startswith("#"):
            in_pip = False
        if in_pip and stripped.startswith("- "):
            pkg = re.split(r"[>=<!\[\];@]", stripped[2:])[0].strip()
            if pkg:
                pip_pkgs.append(pkg)
        elif not in_pip and stripped.startswith("- ") and not stripped.startswith("- pip"):
            pkg = re.split(r"[>=<!\[\];@]", stripped[2:])[0].strip()
            if pkg and not pkg.startswith("pip"):
                conda_pkgs.append(pkg)
    return pip_pkgs, conda_pkgs


def _detect_env_vars_from_source(repo_dir: str, max_files: int = 200) -> list[str]:
    env_vars = set()
    count = 0
    skip = {".git", "__pycache__", "node_modules", "venv", ".venv"}
    pattern = re.compile(r"os\.(?:getenv|environ)\s*\(\s*['\"](\w+)['\"]")
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if not f.endswith(".py"):
                continue
            count += 1
            if count > max_files:
                return list(env_vars)
            try:
                text = Path(os.path.join(root, f)).read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for m in pattern.finditer(text):
                env_vars.add(m.group(1))
    return list(env_vars)


# ---------------------------------------------------------------------------
# Tool directory detection
# ---------------------------------------------------------------------------

def _find_tool_dirs(repo_dir: str, agent_package: str) -> list[str]:
    """Find subdirectories that likely contain tool implementations."""
    tool_dirs = []
    # Handle both cloned repo and site-packages layouts
    pkg_root = os.path.join(repo_dir, agent_package)
    if not os.path.isdir(pkg_root) or not os.path.exists(os.path.join(pkg_root, "__init__.py")):
        if os.path.exists(os.path.join(repo_dir, "__init__.py")):
            pkg_root = repo_dir
        else:
            return tool_dirs
    for entry in os.scandir(pkg_root):
        if entry.is_dir() and not entry.name.startswith("."):
            if entry.name in ("tool", "tools", "plugins"):
                tool_dirs.append(entry.path)
            elif entry.name not in ("agent", "__pycache__", "tests", "test"):
                # Could be tool subdirectories like biomni/tool/
                init_file = os.path.join(entry.path, "__init__.py")
                if os.path.isfile(init_file):
                    # Check if it contains tool-like names
                    tool_dirs.append(entry.path)
    return tool_dirs


# ---------------------------------------------------------------------------
# Main preflight
# ---------------------------------------------------------------------------

def preflight(
    repo_dir: str,
    *,
    agent_module_path: str | None = None,
    install_missing: bool = False,
    skip_heavy: bool = True,
    env_required: list[str] | None = None,
    manifest_path: str | None = None,
) -> PreflightResult:
    """Run preflight check on a downstream agent repo.

    Two-layer dependency analysis:
      Layer 1: Agent runtime deps — traced from agent_module_path import closure
      Layer 2: Tool deps — scanned from tool subdirectories (informational only)

    Args:
        repo_dir: Path to the cloned agent repo.
        agent_module_path: e.g. "biomni.agent.react" — the agent entry point.
        install_missing: If True, auto-install missing agent runtime deps.
        skip_heavy: If True, don't auto-install heavy packages.
        env_required: List of env var names that must be set.
        manifest_path: Optional path to agent manifest YAML.

    Returns:
        PreflightResult with two-layer dependency info.
    """
    result = PreflightResult()
    repo = Path(repo_dir)

    # --- Python version ---
    result.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    result.python_ok = sys.version_info >= (3, 9)
    result.agent_entry = agent_module_path or ""

    # --- Collect agent runtime deps ---
    runtime_deps: set[str] = set()
    tool_dep_set: set[str] = set()

    # 1. Manifest (highest priority)
    if manifest_path and os.path.isfile(manifest_path):
        result.manifest_found = True
        try:
            import yaml
            manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
            deps = manifest.get("dependencies", {})
            runtime_deps.update(deps.get("pip", []))
            env_required = env_required or []
            env_required.extend(deps.get("env", []))
        except Exception as e:
            result.errors.append(f"manifest parse error: {e}")

    # 2. Standard dependency files (treated as agent runtime deps)
    if not runtime_deps:
        for name, parser in [
            ("requirements.txt", lambda p: _parse_requirements_txt(str(p))),
            ("setup.py", lambda p: _parse_setup_py(str(p))),
            ("pyproject.toml", lambda p: _parse_pyproject_toml(str(p))),
            ("environment.yml", lambda p: _parse_environment_yml(str(p))[0]),
        ]:
            fpath = repo / name
            if fpath.exists():
                found = parser(fpath)
                if found:
                    runtime_deps.update(found)

    # 3. Import closure tracing from agent entry point (the key improvement)
    agent_package = agent_module_path.split(".")[0] if agent_module_path else ""
    if agent_module_path:
        try:
            ext_imports, _int_imports = trace_agent_imports(
                agent_module_path, str(repo), max_depth=8,
            )
            for imp in ext_imports:
                pip_name = _pip_name_for(imp)
                runtime_deps.add(pip_name)

            # Tool deps: scan tool subdirectories
            tool_dirs = _find_tool_dirs(str(repo), agent_package)
            result.tool_dirs_scanned = tool_dirs
            if tool_dirs:
                raw_tool_deps = trace_tool_imports(tool_dirs, agent_package)
                tool_dep_set = {_pip_name_for(imp) for imp in raw_tool_deps}
        except Exception as e:
            result.errors.append(f"import tracing error: {e}")

    # --- Deduplicate ---
    seen = set()
    unique_deps = []
    for pkg in runtime_deps:
        normalized = pkg.lower().replace("-", "_").split("[")[0]
        if normalized not in seen:
            seen.add(normalized)
            unique_deps.append(pkg)

    # Remove tool deps that are also runtime deps (they belong in layer 1)
    tool_dep_set -= runtime_deps

    # --- Check runtime deps (blocking) ---
    installed_map = _check_installed_batch(unique_deps)
    for pkg in unique_deps:
        is_heavy = pkg.lower() in HEAVY_PACKAGES
        dep = DepCheck(
            name=pkg,
            import_name=pkg.replace("-", "_").split("[")[0],
            pip_name=pkg,
            source="agent_runtime",
            layer="runtime",
            heavy=is_heavy,
            auto_installable=not is_heavy,
        )
        dep.installed = installed_map.get(pkg, False)
        result.deps.append(dep)

    # --- Check tool deps (informational, non-blocking) ---
    tool_pkgs = sorted(tool_dep_set)
    if tool_pkgs:
        tool_installed_map = _check_installed_batch(tool_pkgs)
        for pkg in tool_pkgs:
            is_heavy = pkg.lower() in HEAVY_PACKAGES
            dep = DepCheck(
                name=pkg,
                import_name=pkg.replace("-", "_").split("[")[0],
                pip_name=pkg,
                source="tool",
                layer="tool",
                heavy=is_heavy,
                auto_installable=not is_heavy,
            )
            dep.installed = tool_installed_map.get(pkg, False)
            result.tool_deps.append(dep)

    # --- Check env vars ---
    required_env = env_required or []
    source_env = _detect_env_vars_from_source(str(repo))
    all_env = list(dict.fromkeys(required_env + source_env))
    for var in all_env:
        val = os.environ.get(var, "")
        ec = EnvCheck(
            name=var,
            set=bool(val),
            value_preview=val[:8] + "..." if len(val) > 8 else val,
        )
        result.env_vars.append(ec)

    # --- Auto-install missing runtime deps only ---
    missing_pip = [d for d in result.deps if not d.installed and d.auto_installable]
    heavy_missing = [d for d in result.deps if not d.installed and d.heavy]

    if install_missing and missing_pip:
        pkg_names = [d.pip_name for d in missing_pip]
        print(f"  Installing {len(pkg_names)} runtime packages ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + pkg_names,
                capture_output=True, text=True, timeout=600,
            )
            recheck_map = _check_installed_batch([d.name for d in missing_pip])
            for d in missing_pip:
                d.installed = recheck_map.get(d.name, False)
        except Exception as e:
            result.errors.append(f"pip install failed: {e}")

    # --- Build action lists (runtime deps only) ---
    still_missing = [d for d in result.deps if not d.installed and d.auto_installable]
    result.pip_installable = [d.pip_name for d in still_missing]
    result.pip_heavy = [d.pip_name for d in heavy_missing]
    unset_env = [e.name for e in result.env_vars if not e.set]
    result.user_action_needed = unset_env

    # --- Determine status (based ONLY on runtime deps) ---
    if not result.python_ok:
        result.status = UNSUPPORTED
    elif still_missing or heavy_missing:
        result.status = SETUP_REQUIRED
    elif unset_env:
        result.status = USER_ACTION_REQUIRED
    else:
        result.status = READY

    return result


def preflight_report(result: PreflightResult) -> str:
    """Format preflight result as readable text."""
    lines = [f"Agent Preflight: {result.status}",
             f"  Python {result.python_version} ({'OK' if result.python_ok else 'TOO OLD'})"]

    if result.agent_entry:
        lines.append(f"  Entry point: `{result.agent_entry}`")

    if result.manifest_found:
        lines.append("  Manifest: found")
    else:
        lines.append("  Manifest: not found")

    # --- Layer 1: Agent runtime deps (blocking) ---
    installed = [d for d in result.deps if d.installed]
    missing = [d for d in result.deps if not d.installed and d.auto_installable]
    heavy = [d for d in result.deps if not d.installed and d.heavy]

    lines.append(f"")
    lines.append(f"  [Layer 1] Agent runtime deps: {len(installed)} OK, {len(missing)} missing, {len(heavy)} heavy")
    for d in result.deps:
        icon = "OK" if d.installed else ("HEAVY" if d.heavy else "MISSING")
        lines.append(f"    [{icon:7s}] {d.name} ({d.source})")

    # --- Layer 2: Tool deps (informational) ---
    if result.tool_deps:
        t_installed = [d for d in result.tool_deps if d.installed]
        t_missing = [d for d in result.tool_deps if not d.installed]
        lines.append(f"")
        lines.append(f"  [Layer 2] Tool-specific deps (non-blocking): {len(t_installed)} OK, {len(t_missing)} missing")
        for d in result.tool_deps[:20]:  # cap at 20 to keep report readable
            icon = "OK" if d.installed else "MISSING"
            lines.append(f"    [{icon:7s}] {d.name}")
        if len(result.tool_deps) > 20:
            lines.append(f"    ... and {len(result.tool_deps) - 20} more")

    # --- Env vars ---
    if result.env_vars:
        lines.append(f"")
        lines.append(f"  Environment variables:")
        for e in result.env_vars:
            icon = "OK" if e.set else "MISSING"
            lines.append(f"    [{icon:7s}] {e.name}")

    # --- Errors ---
    if result.errors:
        lines.append(f"")
        lines.append(f"  Errors:")
        for err in result.errors:
            lines.append(f"    {err}")

    # --- Action items ---
    if result.pip_installable:
        lines.append(f"")
        lines.append(f"  Install (agent runtime): {', '.join(result.pip_installable)}")
    if result.pip_heavy:
        lines.append(f"  Heavy (manual): {', '.join(result.pip_heavy)}")
    if result.user_action_needed:
        lines.append(f"  User action needed: {', '.join(result.user_action_needed)}")

    return "\n".join(lines)
