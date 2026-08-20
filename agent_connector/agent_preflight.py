"""Agent Preflight: check and bootstrap a downstream agent's environment
before tool injection.

Status codes:
  READY                  - all dependencies satisfied
  SETUP_REQUIRED         - missing deps that can be auto-installed
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
}


@dataclass
class DepCheck:
    name: str
    installed: bool = False
    import_name: str = ""
    pip_name: str = ""
    source: str = ""  # where this dep was found (requirements.txt, imports, etc.)
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
    deps: list[DepCheck] = field(default_factory=list)
    env_vars: list[EnvCheck] = field(default_factory=list)
    pip_installable: list[str] = field(default_factory=list)
    pip_heavy: list[str] = field(default_factory=list)
    user_action_needed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_found: bool = False
    inferred: bool = False  # True if deps were inferred (not from manifest)


def _pip_name_for(import_name: str) -> str:
    return IMPORT_TO_PIP.get(import_name, import_name)


def _is_installed(pkg: str) -> bool:
    """Check if a package is importable (via subprocess to avoid side-effects)."""
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
    """Check multiple packages in one subprocess call (fast)."""
    if not pkgs:
        return {}
    import_names = []
    for pkg in pkgs:
        imp = pkg.replace("-", "_").split("[")[0]
        import_names.append((pkg, imp))
    # Build a single script that tries each import
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
    # Fallback: return all False
    return {pkg: False for pkg in pkgs}


def _parse_requirements_txt(path: str) -> list[str]:
    """Parse requirements.txt, return list of package names."""
    pkgs = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pkgs
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # strip version specifiers
        pkg = re.split(r"[>=<!\[\];@]", line)[0].strip()
        if pkg:
            pkgs.append(pkg)
    return pkgs


def _parse_setup_py(path: str) -> list[str]:
    """Extract install_requires from setup.py via AST."""
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
    """Minimal TOML parser for pyproject.toml dependencies."""
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
            # extract quoted string
            m = re.search(r'["\']([^"\']+)["\']', stripped)
            if m:
                pkg = re.split(r"[>=<!\[\];@]", m.group(1))[0].strip()
                if pkg:
                    pkgs.append(pkg)
    return pkgs


def _parse_environment_yml(path: str) -> tuple[list[str], list[str]]:
    """Parse conda environment.yml. Returns (pip_pkgs, conda_pkgs)."""
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


def _infer_imports_from_source(repo_dir: str, max_files: int = 200) -> list[str]:
    """Scan Python files for import statements, return top-level import names."""
    imports = set()
    count = 0
    skip = {".git", "__pycache__", "node_modules", "venv", ".venv", "env",
            "build", "dist", "site-packages"}
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if not f.endswith(".py"):
                continue
            count += 1
            if count > max_files:
                return list(imports)
            try:
                text = Path(os.path.join(root, f)).read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(text)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in stdlib and not top.startswith("_"):
                            imports.add(top)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if top not in stdlib and not top.startswith("_"):
                        imports.add(top)
    return list(imports)


def _detect_env_vars_from_source(repo_dir: str, max_files: int = 200) -> list[str]:
    """Scan Python files for os.getenv / os.environ references."""
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


def _detect_cli_commands_from_source(repo_dir: str, max_files: int = 200) -> list[str]:
    """Scan Python files for subprocess calls to detect CLI dependencies."""
    cmds = set()
    count = 0
    skip = {".git", "__pycache__", "node_modules", "venv", ".venv"}
    pattern = re.compile(r"subprocess\.\w+\(\s*\[?\s*['\"](\w+)['\"]")

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if not f.endswith(".py"):
                continue
            count += 1
            if count > max_files:
                return list(cmds)
            try:
                text = Path(os.path.join(root, f)).read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for m in pattern.finditer(text):
                cmd = m.group(1)
                if cmd not in ("python", "python3", "pip", "git", "ls", "cat", "echo"):
                    cmds.add(cmd)
    return list(cmds)


def preflight(
    repo_dir: str,
    *,
    install_missing: bool = False,
    skip_heavy: bool = True,
    env_required: list[str] | None = None,
    launch_command: str | None = None,
    manifest_path: str | None = None,
) -> PreflightResult:
    """Run preflight check on a downstream agent repo.

    Args:
        repo_dir: Path to the cloned agent repo.
        install_missing: If True, auto-install missing pip packages.
        skip_heavy: If True, don't auto-install heavy packages (torch etc).
        env_required: List of env var names that must be set.
        launch_command: Optional command to test-launch the agent.
        manifest_path: Optional path to agent manifest YAML.

    Returns:
        PreflightResult with status, dependency list, and action items.
    """
    result = PreflightResult()
    repo = Path(repo_dir)

    # --- Python version ---
    result.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    result.python_ok = sys.version_info >= (3, 9)

    # --- Collect pip dependencies from all sources ---
    all_pip_deps: list[str] = []

    # 1. Check for manifest (highest priority)
    if manifest_path and os.path.isfile(manifest_path):
        result.manifest_found = True
        try:
            import yaml
            manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
            deps = manifest.get("dependencies", {})
            all_pip_deps.extend(deps.get("pip", []))
            env_required = env_required or []
            env_required.extend(deps.get("env", []))
            launch_command = launch_command or (manifest.get("launch") or {}).get("command")
        except Exception as e:
            result.errors.append(f"manifest parse error: {e}")

    # 2. Standard dependency files
    req_txt = repo / "requirements.txt"
    if req_txt.exists():
        all_pip_deps.extend(_parse_requirements_txt(str(req_txt)))

    setup_py = repo / "setup.py"
    if setup_py.exists():
        all_pip_deps.extend(_parse_setup_py(str(setup_py)))

    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        all_pip_deps.extend(_parse_pyproject_toml(str(pyproject)))

    env_yml = repo / "environment.yml"
    if env_yml.exists():
        pip_pkgs, _ = _parse_environment_yml(str(env_yml))
        all_pip_deps.extend(pip_pkgs)

    # 3. AST import inference (fallback)
    if not all_pip_deps:
        result.inferred = True
        imports = _infer_imports_from_source(str(repo))
        for imp in imports:
            pip_name = _pip_name_for(imp)
            if pip_name not in all_pip_deps:
                all_pip_deps.append(pip_name)

    # --- Deduplicate ---
    seen = set()
    unique_deps = []
    for pkg in all_pip_deps:
        normalized = pkg.lower().replace("-", "_").split("[")[0]
        if normalized not in seen:
            seen.add(normalized)
            unique_deps.append(pkg)

    # --- Check each dependency (batch) ---
    installed_map = _check_installed_batch(unique_deps)
    for pkg in unique_deps:
        is_heavy = pkg.lower() in HEAVY_PACKAGES
        dep = DepCheck(
            name=pkg,
            import_name=pkg.replace("-", "_").split("[")[0],
            pip_name=pkg,
            source="manifest" if result.manifest_found else "inferred" if result.inferred else "requirements",
            heavy=is_heavy,
            auto_installable=not is_heavy,
        )
        dep.installed = installed_map.get(pkg, False)
        result.deps.append(dep)

    # --- Check env vars ---
    required_env = env_required or []
    source_env = _detect_env_vars_from_source(str(repo))
    all_env = list(dict.fromkeys(required_env + source_env))  # dedupe, keep order

    for var in all_env:
        val = os.environ.get(var, "")
        ec = EnvCheck(
            name=var,
            set=bool(val),
            value_preview=val[:8] + "..." if len(val) > 8 else val,
        )
        result.env_vars.append(ec)

    # --- Auto-install missing pip deps ---
    missing_pip = [d for d in result.deps if not d.installed and d.auto_installable]
    heavy_missing = [d for d in result.deps if not d.installed and d.heavy]

    if install_missing and missing_pip:
        pkg_names = [d.pip_name for d in missing_pip]
        display(f"  Installing {len(pkg_names)} packages ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + pkg_names,
                capture_output=True, text=True, timeout=600,
            )
            # Re-check
            recheck_map = _check_installed_batch([d.name for d in missing_pip])
            for d in missing_pip:
                d.installed = recheck_map.get(d.name, False)
        except Exception as e:
            result.errors.append(f"pip install failed: {e}")

    # --- Build action lists ---
    still_missing = [d for d in result.deps if not d.installed and d.auto_installable]
    result.pip_installable = [d.pip_name for d in still_missing]
    result.pip_heavy = [d.pip_name for d in heavy_missing]
    unset_env = [e.name for e in result.env_vars if not e.set]
    result.user_action_needed = unset_env

    # --- Determine status ---
    if not result.python_ok:
        result.status = UNSUPPORTED
    elif still_missing or heavy_missing:
        result.status = SETUP_REQUIRED
    elif unset_env:
        result.status = USER_ACTION_REQUIRED
    else:
        result.status = READY

    return result


def display(result: PreflightResult) -> str:
    """Format preflight result as readable text."""
    lines = [f"Agent Preflight: {result.status}", f"  Python {result.python_version} ({'OK' if result.python_ok else 'TOO OLD'})"]

    if result.manifest_found:
        lines.append("  Manifest: found")
    elif result.inferred:
        lines.append("  Manifest: not found (deps inferred from imports)")
    else:
        lines.append("  Manifest: not found (deps from requirements files)")

    # Deps summary
    installed = [d for d in result.deps if d.installed]
    missing = [d for d in result.deps if not d.installed and d.auto_installable]
    heavy = [d for d in result.deps if not d.installed and d.heavy]

    lines.append(f"  Dependencies: {len(installed)} installed, {len(missing)} missing, {len(heavy)} heavy (skipped)")
    for d in result.deps:
        icon = "OK" if d.installed else ("HEAVY" if d.heavy else "MISSING")
        lines.append(f"    [{icon:7s}] {d.name} ({d.source})")

    # Env vars
    if result.env_vars:
        lines.append("  Environment variables:")
        for e in result.env_vars:
            icon = "OK" if e.set else "MISSING"
            lines.append(f"    [{icon:7s}] {e.name}")

    # Errors
    if result.errors:
        lines.append("  Errors:")
        for err in result.errors:
            lines.append(f"    {err}")

    # Action items
    if result.pip_installable:
        lines.append(f"  Auto-installable: {', '.join(result.pip_installable)}")
    if result.pip_heavy:
        lines.append(f"  Heavy (manual): {', '.join(result.pip_heavy)}")
    if result.user_action_needed:
        lines.append(f"  User action needed: {', '.join(result.user_action_needed)}")

    return "\n".join(lines)
