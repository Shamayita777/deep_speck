from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from ....core import evidence as EV
from . import design as D


class ProvenanceError(RuntimeError):
    pass


def _load(path: Path) -> dict:
    v = D.parse_json_strict(Path(path).read_text())
    if not isinstance(v, dict):
        raise ProvenanceError(f"{path} is not a JSON object")
    return v


def _inside(root: Path, rel: str) -> Path:
    r = PurePosixPath(rel)
    if r.is_absolute() or ".." in r.parts:
        raise ProvenanceError(f"unsafe evidence path {rel!r}")
    p = root.joinpath(*r.parts).resolve()
    if root.resolve() not in p.parents:
        raise ProvenanceError(f"evidence path {rel!r} escapes {root}")
    return p


def dependency_problems(crypto_root: Path) -> list[str]:
    out = []
    for rel, sha in D.DEPENDENCIES.items():
        p = Path(crypto_root) / rel
        got = EV.sha256_file(p) if p.is_file() else None
        if got != sha:
            out.append(f"{rel}: {got} != pinned {sha}")
    return out


def verify_ce2_source_run(crypto_root: Path, *, hash_all_evidence: bool = True) -> dict:
    s = D.CE2_SOURCE_RUN
    root = (Path(crypto_root) / s["relpath"]).resolve()
    cert_p, man_p, ver_p = root / "certificate.json", root / "run_manifest.json", root / s["verification_report"]
    for p, key in ((cert_p, "certificate_sha256"), (man_p, "run_manifest_sha256"), (ver_p, "verification_report_sha256")):
        if not p.is_file():
            raise ProvenanceError(f"CE2 source file missing: {p}")
        if EV.sha256_file(p) != s[key]:
            raise ProvenanceError(f"CE2 {p.name} hash differs from the pinned {key}")
    cert, man, ver = _load(cert_p), _load(man_p), _load(ver_p)
    checks = {
        "production_evidentiary": man.get("mode") == cert.get("mode") == "production"
                                  and man.get("evidentiary") is True and cert.get("evidentiary") is True,
        "decision": cert.get("decision") == s["decision"],
        "production_complete": cert.get("status", {}).get("production") == "COMPLETE",
        "design": cert.get("design_sha256") == man.get("design_sha256") == s["design_sha256"]
                  and cert.get("design_version") == s["design_version"],
        "preregistration": (cert.get("preregistration") or {}).get("sha256") == s["preregistration_sha256"]
                           == (man.get("preregistration") or {}).get("sha256"),
        "approval": ((cert.get("preregistration") or {}).get("approval") or {}).get("status") == "APPROVED"
                    and cert["preregistration"]["approval"].get("approved_design_sha256") == s["design_sha256"],
        "checkpoint": cert.get("reference_model", {}).get("checkpoint_sha256") == D.PUBLISHED_CHECKPOINT_SHA256,
        "upstream_commit": cert.get("reference_model", {}).get("upstream_commit") == man.get("upstream_commit")
                           == D.UPSTREAM_COMMIT,
        "twin_record": all(isinstance(t, dict) and t.get("trained") is False and t.get("file") == s["twin_file"]
                           and t.get("file_sha256") == s["twin_file_sha256"]
                           and t.get("weights_sha256") == s["twin_weights_sha256"] and t.get("seed") == s["twin_seed"]
                           for t in (cert.get("untrained_twin"), man.get("untrained_twin"))),
        "twin_in_evidence_manifest": cert.get("evidence_manifest", {}).get(s["twin_file"]) == s["twin_file_sha256"],
        "verification_bound": ver.get("certificate_sha256") == s["certificate_sha256"] and ver.get("mode") == "production",
        "verification_passed": ver.get("passed") is True and ver.get("failures") == [] and int(ver.get("n_checks", 0)) > 0,
        "verification_recomputed_exact": ver.get("model_scores_recomputed") is True
                                         and ver.get("numerical_status") == "EXACT",
    }
    twin = _inside(root, s["twin_file"])
    checks["twin_file"] = twin.is_file() and EV.sha256_file(twin) == s["twin_file_sha256"]
    mismatched = []
    if hash_all_evidence:
        for rel, sha in cert.get("evidence_manifest", {}).items():
            p = _inside(root, rel)
            if not p.is_file() or EV.sha256_file(p) != sha:
                mismatched.append(rel)
        checks["evidence_manifest_intact"] = not mismatched
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise ProvenanceError(f"CE2 source run checks failed: {bad} {mismatched[:5]}")
    return {"run_dir": str(root), "relpath": s["relpath"], "checks": checks,
            "evidence_files_hashed": len(cert.get("evidence_manifest", {})) if hash_all_evidence else 0,
            "certificate_sha256": s["certificate_sha256"], "verification_report_sha256": s["verification_report_sha256"],
            "verification_n_checks": int(ver["n_checks"]), "twin_path": str(twin),
            "twin_file_sha256": s["twin_file_sha256"], "twin_weights_sha256": s["twin_weights_sha256"]}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
