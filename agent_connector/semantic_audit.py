"""Semantic artifact-contract audit for discovered tool schemas.

Goal: upgrade "schema looks structurally valid" to "schema is evidence-backed,
self-consistent, and executable". This is the layer the structural preflight
(validate_spec) CANNOT catch -- e.g. an input inferred as ``artifact_type:
binseq`` while the tool's own --help says it consumes FASTA/FASTQ.

Three checks, each producing an issue with a severity:

  * ``rejected``     -- explicit contradiction (tool says X, schema says Y that
                        is provably wrong). Never reaches the active registry;
                        preflight exits 1.
  * ``needs_review`` -- ambiguous / under-evidenced (low confidence, missing
                        provenance, or declared format token absent while
                        another single format is clearly mentioned). Quarantined:
                        the agent never sees it, but it does not hard-fail CI.
  * ``active``       -- evidence-backed and self-consistent.

The audit is conservative on purpose: it prefers ``needs_review`` over
``rejected`` when in doubt, so a typo in a description never silently breaks a
working tool -- but a wrong artifact inference is still kept OUT of the active
registry (the server skips anything that is not ``active``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

# --- artifact keyword model -------------------------------------------------
# STRONG tokens are explicit format names; their presence in evidence text is a
# reliable signal that a given artifact type is involved.
STRONG_TOKENS: dict[str, list[str]] = {
    "fasta": ["fasta", "fastq", ".fa", ".fna"],
    "fastq": ["fastq", ".fq", ".fastq"],
    "binseq": ["binseq", ".vbq", ".cbq", ".bq", "vbq", "cbq"],
    "cool_matrix": ["cool", "mcool", ".cool", ".mcool"],
    "genomic_features": ["bed", "gtf", "gff", ".bed"],
    "bam": ["bam", ".bam"],
    "sam": ["sam", ".sam"],
    "vcf": ["vcf", ".vcf"],
    "pdb": ["pdb", ".pdb"],
    "bigwig": ["bigwig", ".bw", "wiggle"],
    "bedpe": ["bedpe", ".bedpe"],
}

# Weaker contextual phrases (lower weight, used only to reinforce STRONG hits).
WEAK_TOKENS: dict[str, list[str]] = {
    "fasta": ["sequence file", "amino acid sequence", "reads"],
    "fastq": ["sequencing read", "sequencing reads", "reads file"],
    "binseq": ["compressed sequence archive", "sequence archive"],
    "cool_matrix": ["contact matrix", "hic", "hi-c"],
    "genomic_features": ["genomic feature", "feature file"],
    "bam": ["read alignment", "alignment file"],
    "vcf": ["variant call", "variant file"],
    "pdb": ["protein structure"],
}

# Confidence threshold below which an artifact inference is quarantined.
CONFIDENCE_THRESHOLD = 0.5


def _count_hits(text: str, tokens: list[str]) -> int:
    return sum(1 for t in tokens if t in text)


_INPUT_SIGNALS = ("encode", "from", "input", "consume", "accept", "takes",
                  "reads", "given", "convert", "of ")


def _input_format_context(text: str, fmts: tuple[str, ...]) -> bool:
    """True if a format token appears in an INPUT context (preceded within a
    short window by an input signal such as 'encode'/'from'/'input')."""
    for m in re.finditer(r"(fasta|fastq)", text):
        window = text[max(0, m.start() - 30):m.start()]
        if any(s in window for s in _INPUT_SIGNALS):
            return True
    return False


def _format_tokens_present(text: str) -> set[str]:
    """Return the set of artifact types whose STRONG token appears in text."""
    present = set()
    for artifact, tokens in STRONG_TOKENS.items():
        if _count_hits(text, tokens) > 0:
            present.add(artifact)
    return present


def _evidence_text_for_input(tool: dict, sub: str | None, param: dict) -> str:
    """Collect the textual evidence relevant to one input parameter."""
    chunks: list[str] = []
    chunks.append((tool.get("description") or ""))
    if sub:
        detail = (tool.get("subcommand_details") or {}).get(sub) or {}
        chunks.append(detail.get("description") or "")
        chunks.append(detail.get("usage") or "")
        for o in (detail.get("outputs") or {}).values():
            chunks.append((o or {}).get("description") or "")
    chunks.append(param.get("description") or "")
    return " ".join(c for c in chunks if c).lower()


def _audit_one_input(tool: dict, sub: str | None, pname: str, param: dict) -> list[dict]:
    declared = param.get("artifact_type") or param.get("artifact")
    if not declared:
        return []
    declared = str(declared).lower()
    issues: list[dict] = []

    confidence = param.get("artifact_confidence")
    source = param.get("artifact_source") or param.get("source")
    evidence = _evidence_text_for_input(tool, sub, param)

    # Layer A: provenance / confidence.
    if isinstance(confidence, (int, float)) and confidence < CONFIDENCE_THRESHOLD:
        issues.append({
            "param": pname, "subcommand": sub, "artifact": declared,
            "severity": "needs_review",
            "reason": f"artifact confidence {confidence} < {CONFIDENCE_THRESHOLD} "
                      f"(source={source or 'unknown'}) -- treated as a guess",
            "evidence": (param.get("description") or "")[:160],
        })

    # Layer B0: binseq-as-input confusion. BINSEQ is an OUTPUT archive format; an
    # *input* declared binseq whose evidence describes FASTA/FASTQ as something
    # being encoded/consumed is almost certainly fasta/fastq input (e.g. the
    # bqtools base rule ``("bqtools","input") -> binseq`` wrongly winning over the
    # encode-specific ``fasta`` rule). Quarantine it -- the agent must not see a
    # wrong input contract. Correct tools (decode/cat) describe FASTA/FASTQ as
    # the *output* ("back to FASTA"), which this rule does NOT flag.
    if declared == "binseq" and ("fasta" in evidence or "fastq" in evidence) \
            and _input_format_context(evidence, ("fasta", "fastq")):
        issues.append({
            "param": pname, "subcommand": sub, "artifact": declared,
            "severity": "needs_review",
            "reason": "input declared 'binseq' but evidence describes FASTA/FASTQ "
                      "as the input format (BINSEQ is normally an output archive)",
            "evidence": evidence[:200],
        })
        return issues  # quarantined; no further checks needed

    # Layer B: explicit contradiction.
    # declared format token absent, but another format clearly present (>=2 hits).
    declared_hits = _count_hits(evidence, STRONG_TOKENS.get(declared, []))
    alt_hits = 0
    alt_type = None
    for artifact, tokens in STRONG_TOKENS.items():
        if artifact == declared:
            continue
        h = _count_hits(evidence, tokens)
        if h > alt_hits:
            alt_hits = h
            alt_type = artifact
    if declared_hits == 0 and alt_hits >= 2:
        issues.append({
            "param": pname, "subcommand": sub, "artifact": declared,
            "severity": "rejected",
            "reason": f"schema declares '{declared}' but evidence describes "
                      f"'{alt_type}' (e.g. {evidence[:120]!r})",
            "evidence": evidence[:200],
        })
        return issues  # explicit contradiction already decides rejected

    # Layer C: format ambiguity (conservative needs_review).
    present = _format_tokens_present(evidence)
    if declared not in present and present:
        # exactly one other format clearly present -> likely the real input type
        others = present - {declared}
        if len(others) == 1:
            issues.append({
                "param": pname, "subcommand": sub, "artifact": declared,
                "severity": "needs_review",
                "reason": f"declared '{declared}' but evidence only mentions "
                          f"'{sorted(others)[0]}'; possible artifact mismatch",
                "evidence": evidence[:200],
            })
        elif len(others) > 1:
            issues.append({
                "param": pname, "subcommand": sub, "artifact": declared,
                "severity": "needs_review",
                "reason": f"declared '{declared}' not found in evidence; multiple "
                          f"formats mentioned {sorted(others)}",
                "evidence": evidence[:200],
            })

    # Layer D: under-evidenced guess (no provenance, no textual support).
    if declared_hits == 0 and not (isinstance(confidence, (int, float))):
        issues.append({
            "param": pname, "subcommand": sub, "artifact": declared,
            "severity": "needs_review",
            "reason": f"artifact '{declared}' has no confidence/source and no "
                      f"supporting evidence in the tool description",
            "evidence": evidence[:200],
        })

    return issues


def audit_tool(tool: dict) -> list[dict]:
    """Return all semantic-artifact issues for a single tool spec."""
    if not isinstance(tool, dict):
        return []
    issues: list[dict] = []
    inputs = tool.get("inputs") or {}
    for pname, param in (inputs or {}).items():
        if isinstance(param, dict):
            issues.extend(_audit_one_input(tool, None, pname, param))
    # subcommand tools keep inputs under subcommand_details[sub].params
    subs = tool.get("subcommand_details") or {}
    for sub, detail in (subs or {}).items():
        for p in (detail.get("params") or []):
            pname = p.get("name", "")
            if isinstance(p, dict):
                issues.extend(_audit_one_input(tool, sub, pname, p))
    return issues


def classify_tool(tool: dict) -> str:
    """Return 'active' | 'needs_review' | 'rejected' for a tool spec."""
    issues = audit_tool(tool)
    if any(i["severity"] == "rejected" for i in issues):
        return "rejected"
    if any(i["severity"] == "needs_review" for i in issues):
        return "needs_review"
    return "active"


def is_active(tool: dict) -> bool:
    return classify_tool(tool) == "active"


# --- Gate 7: best-effort execution / contract audit -------------------------
# If the tool's executable is available, run `--help` (and `<sub> --help`) and
# confirm every declared flag appears. Missing executable => skipped (no CI box
# has every bio-tool installed). Missing declared flags => needs_review (naming
# drift), never a hard reject (help formats vary).
_FLAG_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_-]*", re.MULTILINE)


def execution_contract_check(tool: dict) -> list[dict]:
    """Best-effort: verify declared flags exist in the tool's --help output.

    Only meaningful when the tool is actually installed. If ``--help`` is
    unavailable (binary missing, or it errors), we SKIP -- we never downgrade a
    tool's status on an un-runnable audit. A failed/empty help is not evidence
    of a bad schema.
    """
    issues: list[dict] = []
    command = tool.get("command") or ""
    if not command:
        return issues
    # Runnable base command: drop {placeholders} (e.g. `python -m bioemu.sample`).
    base = [b for b in re.sub(r"\{[^}]*\}", "", command).split() if b]
    if not base:
        return issues
    exe = base[0]
    if not shutil.which(exe):
        return issues  # binary not installed here -> cannot audit; skip

    def _help_text(*extra: str) -> str | None:
        try:
            proc = subprocess.run([*base, *extra, "--help"], capture_output=True,
                                  text=True, errors="replace", timeout=30)
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "") + (proc.stderr or "")
        return out if out.strip() else None

    help_text = _help_text() or ""
    subs = tool.get("subcommands") or list((tool.get("subcommand_details") or {}).keys())
    for sub in subs:
        h = _help_text(sub)
        if h:
            help_text += "\n" + h
    if not help_text:
        return issues  # --help unavailable -> skip, do not quarantine

    declared_flags = set()
    for p in (tool.get("inputs") or {}).values():
        f = (p or {}).get("flag")
        if f:
            declared_flags.add(f.lstrip("-").split("=")[0])
    for sub, detail in (tool.get("subcommand_details") or {}).items():
        for p in (detail.get("params") or []):
            f = (p or {}).get("flag")
            if f:
                declared_flags.add(f.lstrip("-").split("=")[0])

    help_flags = {m.group(0).lstrip("-").split("=")[0] for m in _FLAG_RE.finditer(help_text)}
    missing = sorted(f for f in declared_flags if f and f not in help_flags)
    if missing:
        issues.append({
            "severity": "needs_review",
            "reason": f"declared flags not found in --help output: {missing}",
            "evidence": " ".join([*base, "--help"]),
        })
    return issues


def audit_tool_full(tool: dict) -> dict:
    """Run both artifact-contract and execution-contract checks."""
    artifact_issues = audit_tool(tool)
    exec_issues = execution_contract_check(tool)
    status = classify_tool(tool)
    if status == "active" and any(i["severity"] == "needs_review" for i in exec_issues):
        status = "needs_review"
    return {
        "status": status,
        "artifact_issues": artifact_issues,
        "execution_issues": exec_issues,
    }
