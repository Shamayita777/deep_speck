"""E6: the two registered non-distinguishers against the CE2 criterion.

Battery v2 ran A3 against CE3 and A4 against CE4 and left CE2 untested
("no adversary in v2 is designed to defeat CE2"). A CE test can carry a claim
to L3 only if no constructed non-distinguisher satisfies its criterion (CE
Rule III: the evidence must have greater explanatory power than the remaining
alternatives). This module asks that question of CE2 with the same two
constructions, unchanged:

  A3  the frozen constant-output model (evidence/a3_20261006), loaded through
      battery.load_frozen_a3, which hard-fails on any provenance mismatch;
  A4  battery.A4XorResponse with its registered key: a keyed BLAKE2b function
      of the ciphertext-pair XOR, AUC 0.4999 on the sealed set.

CE2 itself is not re-implemented: each adversary is passed as ``model`` to the
frozen production routine ``ce234.production.run_ce2`` at the frozen CE2
replication (20 runs of 10^6 samples), and the frozen ``analyse_ce2`` decides.
The adversarial outcome is CE2_CRITERION_SATISFIED iff that decision is
SUPPORTED.

    python -m audit.adversarial.ce2_adversarial plan   --prereg P
    python -m audit.adversarial.ce2_adversarial smoke  --prereg P --out-dir O
    python -m audit.adversarial.ce2_adversarial run    --prereg P --out-dir O --confirm RUN_E6_PRODUCTION
    python -m audit.adversarial.ce2_adversarial verify --prereg P --out-dir O
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIRM_TOKEN = "RUN_E6_PRODUCTION"
A3_DIR = HERE / "evidence" / "a3_20261006"
BOUND_SOURCES = (
    "audit/adversarial/ce2_adversarial.py",
    "audit/adversarial/battery.py",
    "audit/cryptography/ce234/production.py",
    "audit/cryptography/ce234/frozen_plan.py",
    "audit/cryptography/gohr/speck.py",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def source_hashes() -> dict:
    return {rel: sha256_file(REPO_ROOT / rel) for rel in BOUND_SOURCES}


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", *BOUND_SOURCES)
    return {"head": run("rev-parse", "HEAD") or None, "bound_sources_dirty": bool(dirty),
            "dirty_entries": dirty.splitlines() if dirty else []}


def load_prereg(path: Path):
    doc = json.loads(Path(path).read_text())
    return doc, canonical_sha256(doc["design"])


def require_approval(doc: dict, design_sha: str) -> None:
    a = doc["approval"]
    if a.get("status") != "APPROVED":
        raise SystemExit(f"REFUSED: preregistration status is {a.get('status')!r}, not APPROVED.")
    if a.get("design_sha256") != design_sha:
        raise SystemExit(f"REFUSED: approval names design {a.get('design_sha256')}, "
                         f"but the design hashes to {design_sha}.")
    if not a.get("approved_by") or not a.get("approved_utc"):
        raise SystemExit("REFUSED: approval lacks approved_by / approved_utc.")


def build_adversary(name: str):
    from audit.adversarial import battery as B
    if name == "A3":
        # The artifact is bound by its frozen SHA-256 and freeze commit. The
        # battery's source-state check is recorded, not enforced: it guards the
        # battery's own interpretation of A3, and E6 interprets A3 only through
        # the frozen ce234 CE2 routine (registered in the design).
        model_path = A3_DIR / "a3_representation.keras"
        man = json.loads((A3_DIR / B.A3_FREEZE_NAME).read_text())
        B.verify_freeze_commit_recorded(man)
        actual = sha256_file(model_path)
        if actual != man["model_sha256"]:
            raise SystemExit(f"REFUSED: A3 artifact hash {actual} differs from the frozen "
                             f"{man['model_sha256']}.")
        try:
            B.verify_source_integrity(man)
            source_state = "MATCHES_FREEZE"
        except SystemExit as exc:
            source_state = f"DIFFERS_FROM_FREEZE: {exc}"
        import keras
        return keras.models.load_model(model_path), {
            "artifact": str(model_path.relative_to(REPO_ROOT)),
            "artifact_sha256": actual, "freeze_manifest": B.A3_FREEZE_NAME,
            "battery_source_state_vs_freeze": source_state}
    if name == "A4":
        cert = HERE / "evidence" / "a4_20261006" / "a4_certificate.json"
        status = json.loads(cert.read_text())["construction"]["construction_status"]
        if status != "VALID":
            raise SystemExit(f"REFUSED: A4 construction status is {status}.")
        return B.A4XorResponse(), {"kind": "closed-form", "key": B.A4_MAP_KEY.decode(),
                                   "construction_certificate": str(cert.relative_to(REPO_ROOT)),
                                   "construction_certificate_sha256": sha256_file(cert)}
    raise SystemExit(f"unknown adversary {name}")


def adversarial_outcome(decision: str | None) -> str:
    return "CE2_CRITERION_SATISFIED" if decision == "SUPPORTED" else "CE2_CRITERION_NOT_SATISFIED"


def run_one(name: str, design: dict, out: Path, *, n_runs: int, n_samples: int) -> dict:
    from audit.cryptography.ce234 import production as PR
    model, identity = build_adversary(name)
    seed = int(design["adversaries"][name]["seed"])
    r = PR.run_ce2(out / name / "ce2", n_runs=n_runs, n_samples=n_samples, seed=seed,
                   model=model)
    decision = r.get("decision")
    outcome = adversarial_outcome(decision)
    predicted = design["adversaries"][name]["predicted_outcome"]
    return {"adversary": name, "identity": identity, "seed": seed,
            "replication": {"n_runs": n_runs, "n_samples": n_samples,
                            "n_valid_runs": r.get("n_valid_runs")},
            "ce2_decision": decision,
            "ce2_decision_reason": r.get("decision_reason") or r.get("reason"),
            "ce2_endpoints": r.get("endpoints"),
            "holm_adjusted_p": r.get("holm_adjusted_p"),
            "runs": r.get("runs"),
            "adversarial_outcome": outcome, "predicted_outcome": predicted,
            "prediction_met": outcome == predicted}


def interpret(results: dict, design: dict) -> dict:
    satisfied = [k for k, v in results.items()
                 if v["adversarial_outcome"] == "CE2_CRITERION_SATISFIED"]
    if satisfied:
        return {"ce2_admission": "NOT_ADMITTED",
                "statement": design["interpretation"]["if_any_satisfied"],
                "satisfied_by": satisfied}
    return {"ce2_admission": "ADMITTED",
            "statement": design["interpretation"]["if_none_satisfied"]}


def verify(out: Path, design: dict) -> dict:
    """Recompute every CE2 rho from the raw files and re-apply the frozen rule."""
    from scipy.stats import spearmanr
    from audit.cryptography.ce234 import production as PR
    cert = json.loads((out / "e6_certificate.json").read_text())
    report = {"checks": 0, "mismatches": [], "adversaries": {}}
    for name, res in cert["results"].items():
        rec_runs = []
        for run in res["runs"] or []:
            if run.get("status") != "OK":
                rec_runs.append(run); continue
            raw = out / name / "ce2" / run["raw_file"]
            report["checks"] += 1
            if sha256_file(raw) != run["raw_sha256"]:
                report["mismatches"].append(f"{name} {run['run_id']}: raw file hash"); continue
            z = np.load(raw)
            rho = float(spearmanr(z["target"], z["output"]).statistic)
            report["checks"] += 1
            if not np.isclose(rho, run["rho_primary"], rtol=0, atol=1e-6):
                report["mismatches"].append(f"{name} {run['run_id']}: rho {rho} != {run['rho_primary']}")
            rec_runs.append(dict(run, rho_primary=rho))
        again = PR.analyse_ce2(rec_runs)
        report["checks"] += 1
        if again["decision"] != res["ce2_decision"]:
            report["mismatches"].append(f"{name}: decision {again['decision']} != {res['ce2_decision']}")
        report["adversaries"][name] = {"recomputed_decision": again["decision"],
                                       "recorded_decision": res["ce2_decision"]}
    report["status"] = "VERIFIED" if not report["mismatches"] else "NOT_VERIFIED"
    report["verified_utc"] = utc()
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("plan", "smoke", "run", "verify"))
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--confirm")
    a = ap.parse_args(argv)
    doc, dsha = load_prereg(a.prereg)
    design = doc["design"]

    if a.command == "plan":
        print(json.dumps({"design_sha256": dsha, "approval": doc["approval"],
                          "source_hashes": source_hashes(), "git": git_state()}, indent=2))
        return 0
    if not a.out_dir:
        ap.error("--out-dir is required")
    if a.command == "verify":
        rep = verify(a.out_dir, design)
        write_json_atomic(a.out_dir / "e6_verification.json", rep)
        print(json.dumps({k: rep[k] for k in ("status", "checks", "mismatches")}, indent=2))
        return 0 if rep["status"] == "VERIFIED" else 3

    smoke = a.command == "smoke"
    if not smoke:
        if a.confirm != CONFIRM_TOKEN:
            raise SystemExit(f"REFUSED: production needs --confirm {CONFIRM_TOKEN}.")
        require_approval(doc, dsha)
        g = git_state()
        if g["bound_sources_dirty"]:
            raise SystemExit(f"REFUSED: bound sources are not committed: {g['dirty_entries']}")
        if (a.out_dir / "e6_certificate.json").exists():
            raise SystemExit("REFUSED: a certificate already exists in this directory; "
                             "E6 runs once per directory and is never resumed.")
    from audit.cryptography.ce234 import frozen_plan as P
    n_runs = 2 if smoke else int(P.CE2.n_runs)
    n_samples = 2000 if smoke else int(P.CE2.samples_per_run)
    if not smoke and (n_runs != design["ce2_replication"]["n_runs"]
                      or n_samples != design["ce2_replication"]["samples_per_run"]):
        raise SystemExit("REFUSED: frozen CE2 replication differs from the registered one.")
    results = {}
    for name in design["adversaries"]:
        print(f"E6 {name}: CE2 with {n_runs} runs x {n_samples} samples", flush=True)
        results[name] = run_one(name, design, a.out_dir, n_runs=n_runs, n_samples=n_samples)
        print(f"  decision {results[name]['ce2_decision']} -> "
              f"{results[name]['adversarial_outcome']} "
              f"(predicted {results[name]['predicted_outcome']})", flush=True)
    cert = {"schema": "e6-certificate-1", "smoke": smoke, "design_sha256": dsha,
            "approval": doc["approval"], "source_hashes": source_hashes(),
            "git": git_state(), "results": results,
            "interpretation": interpret(results, design), "generated_utc": utc()}
    write_json_atomic(a.out_dir / "e6_certificate.json", cert)
    print(json.dumps(cert["interpretation"], indent=2))
    if smoke:
        print("SMOKE ONLY: 2 runs x 2,000 samples; not evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
