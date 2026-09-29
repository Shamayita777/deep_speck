"""
Output write-protection for audit runs.

Historical CE evidence is immutable. The historical adapter defaulted its
save paths straight into `evidence/ce1/`, so a re-run could have
overwritten the very artifacts the audit cites. The current audit writes only to an
explicit, versioned location and refuses anything else.
"""

from __future__ import annotations

from pathlib import Path

#: Repository-relative roots that hold historical evidence.
PROTECTED_ROOTS = ("evidence", "audit/cryptography/evidence")
#: Path components that always indicate historical/immutable material.
PROTECTED_COMPONENTS = frozenset({"evidence", "frozen", "evidence_bundle"})
#: Where current-audit output belongs.
AUDIT_OUTPUT_ROOT = "evidence_current"


class HistoricalWriteError(RuntimeError):
    pass


def assert_audit_output_path(path, *, repo_root=None) -> Path:
    """
    Refuse any output path that could touch historical evidence.

    Protection is by RESOLVED CONTAINMENT, not by string inspection: the
    target is fully resolved (symlinks and `..` collapsed) and must lie
    inside the resolved `evidence_current/` root. A component check alone is
    bypassable - `evidence_current/../evidence/ce1/x.json` contains the token
    `evidence_current` yet resolves into historical evidence.

    Returns the ORIGINAL path object when acceptable (callers write to it
    as given); the resolved form is used only for the safety decision.
    """
    p = Path(path)
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    target = p.resolve() if p.is_absolute() else (base / p).resolve()

    # 1. the resolved target must be inside the resolved evidence_current root
    audit_root = (base / AUDIT_OUTPUT_ROOT).resolve()
    if target != audit_root and audit_root not in target.parents:
        raise HistoricalWriteError(
            f"refusing to write {path}: resolved target {target} is not inside the current-audit "
            f"output root {audit_root}. current-audit output must be explicitly versioned and cannot "
            "be confused with historical evidence.")

    # 2. belt-and-braces: never inside a historical root, even if someone
    #    nests one under evidence_current/
    for protected in PROTECTED_ROOTS:
        pr = (base / protected).resolve()
        if target == pr or pr in target.parents:
            raise HistoricalWriteError(
                f"refusing to write {path}: resolved target {target} is inside historical "
                f"root {protected}.")
    hit = PROTECTED_COMPONENTS & set(target.relative_to(audit_root).parts) if (
        target != audit_root) else set()
    if hit:
        raise HistoricalWriteError(
            f"refusing to write {path}: resolved target contains protected component(s) "
            f"{sorted(hit)} beneath the audit output root.")
    return p
