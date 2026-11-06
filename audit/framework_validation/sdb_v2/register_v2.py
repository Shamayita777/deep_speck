"""
Freeze the SDB-V2 manifest and registration (refuses to overwrite either). Run before execution.

  python -B audit/framework_validation/sdb_v2/register_v2.py

manifest_sdb_v2.json binds, by SHA-256: every V2 source and test, the design draft and allowlist,
every project module the benchmark imports (transitive closure measured in a fresh interpreter),
the released generator and reference cipher, every evidence file (checkpoints, certificates, the
H2 bundle), every fixture, and the pip freeze. registration_sdb_v2.json holds the case registry,
the expected outcomes, the declared-input pointers, the specification gaps and the manifest hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

COMPONENT_MODULES = [
    "audit.framework_validation.sdb_v2.cases_v2", "audit.framework_validation.sdb_v2.baselines_v2",
    "audit.framework_validation.sdb_v2.engine_v2", "audit.framework_validation.sdb_v2.oracle_v2",
    "audit.cryptography.ce234.production", "audit.cryptography.gohr.speck", "audit.implementation.gohr_adapter",
    "audit.implementation.ii2_independent.verify_m1", "audit.dataset.d1_duplicate_detection",
    "audit.dataset.d2_sample_dependence", "audit.dataset.experiments.d2n_fresh_draw_null",
    "audit.dataset.adapters.autond_d4", "audit.dataset.experiments.d4_autond_label_shuffle",
    "audit.dataset.d4_controlled_perturbation", "audit.implementation.ii4_parallel.verify",
    "audit.implementation.ii4_parallel.scheduler", "audit.implementation.ii4_parallel.worker",
    "audit.implementation.ii4_parallel.tests.ii4_parallel_fixtures",
    "framework.certificate", "framework.experiment", "framework.failures", "framework.replication",
    "framework.seeds", "framework.statistics",
]
EXTRA_SOURCES = ["speck.py", "train_nets.py", "audit/implementation/reference/speck.py",
                 "audit/framework_validation/__init__.py",
                 "audit/framework_validation/SDB_V2_DESIGN_CLEAN_DRAFT.md",
                 "audit/framework_validation/SDB_V2_H2_ALLOWLIST_DRAFT.json"]
EVIDENCE = ["audit/cryptography/evidence/ce1/best5depth10 (10).h5", "audit/cryptography/Archive/best5depth10.h5",
            "audit/cryptography/evidence/ce1/signal_destroyed.h5",
            "audit/adversarial/evidence/a3_20261006/a3_certificate.json",
            "audit/adversarial/evidence/e6_ce2_20261105/e6_certificate.json",
            "audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/result/d4_certificate.json",
            "audit/implementation/evidence/implementation-certificate-V5.json"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO))


def transitive_imports() -> list[str]:
    code = ("import importlib, json, sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "for m in %r: importlib.import_module(m)\n"
            "print('MODS' + json.dumps(sorted({getattr(v, '__file__', None) or '' for v in list(sys.modules.values())})))"
            % (str(REPO), str(REPO / "audit/experimental_validity"), COMPONENT_MODULES))
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    files = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("MODS"))[4:])
    return sorted(rel(f) for f in files if f and Path(f).resolve().is_relative_to(REPO)
                  and ".venv" not in f and "site-packages" not in f)


def main() -> int:
    reg_p, man_p = HERE / "registration_sdb_v2.json", HERE / "manifest_sdb_v2.json"
    if reg_p.exists() or man_p.exists():
        print("REFUSED: a registration or manifest already exists")
        return 1
    from audit.framework_validation.sdb_v2 import baselines_v2 as B
    from audit.framework_validation.sdb_v2 import cases_v2 as C
    from audit.framework_validation.sdb_v2 import fixtures_v2 as F
    v2 = sorted(rel(p) for p in list(HERE.glob("*.py")) + list((HERE / "tests").glob("*.py")))
    imports = transitive_imports()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (HERE / "environment_pip_freeze.txt").write_text(freeze)
    sources = {p: sha(REPO / p) for p in sorted(set(v2 + imports + EXTRA_SOURCES + [rel(HERE / "environment_pip_freeze.txt")]))}
    bundle_files = sorted(rel(p) for p in (HERE / "evidence/h2_ad4").rglob("*") if p.is_file())
    evidence = {p: sha(REPO / p) for p in EVIDENCE + bundle_files}
    fixtures = {rel(p): sha(p) for p in sorted((HERE / "fixtures").rglob("*")) if p.is_file()}
    bound = list(sources) + list(evidence) + list(fixtures)
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all", "--", *bound],
                            capture_output=True, text=True).stdout.splitlines()
    import h5py
    import keras
    import numpy
    import scipy
    import tensorflow
    manifest = {
        "artifact": "sdb-v2-manifest-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_state": {"repo": str(REPO), "branch": subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"],
                                                                    capture_output=True, text=True).stdout.strip(),
                         "head_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                       capture_output=True, text=True).stdout.strip(),
                         "bound_paths_git_status": status,
                         "note": "bound files may be untracked or modified locally; each is bound by its hash below"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": numpy.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
                        "keras": keras.__version__, "tensorflow": tensorflow.__version__,
                        "pip_freeze_file": rel(HERE / "environment_pip_freeze.txt")},
        "transitive_project_imports": imports,
        "sources": sources, "evidence": evidence, "fixtures": fixtures,
        "configuration": {"fixture_seeds": F.SEEDS, "b1_n": B.B1_N, "b1_threshold_rule": "> 0.5 (ce234 binding)",
                          "b1s_threshold_rule": ">= 0.5 (d4_controlled_perturbation.py:423)", "b1_z": B.Z95,
                          "reported_accuracy": F.REPORTED_ACCURACY, "published_architecture": F.PUBLISHED_ARCHITECTURE},
    }
    man_p.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    bm = json.loads((HERE / "evidence/h2_ad4/BUNDLE_MANIFEST.json").read_text())
    registration = {
        "artifact": "sdb-v2-registration-1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(man_p),
        "design": {"path": "audit/framework_validation/SDB_V2_DESIGN_CLEAN_DRAFT.md",
                   "sha256": sha(REPO / "audit/framework_validation/SDB_V2_DESIGN_CLEAN_DRAFT.md")},
        "v1_unchanged": {p: sha(REPO / "audit/framework_validation" / p) for p in
                         ("registration_sdb_v1.json", "seeded_defects.py", "claim_engine.py")},
        "case_order": C.ORDER, "case_groups": C.GROUPS, "case_levels": C.LEVELS,
        "counts": {"total": 23, "historical_defect": 1, "historical_replay": 1, "synthetic_defect": 6,
                   "clean_control": 7, "positive_decision_rule_control": 1, "abstention_control": 7},
        "excluded": {"U8": "no conflict-resolution rule in the manuscript"},
        "expected_outcomes": C.expected_outcomes(),
        "fixture_claim_template": C.FIXTURE_CLAIM,
        "a1_recorded_dimensions": C.A1_RECORDED,
        "asserted_inputs": {c: rel(HERE / "fixtures" / c / "declared.json") for c in ("P1", "U2", "U3", "U5", "U7")},
        "b1_status": {c: B.b1_status(c) for c in C.ORDER},
        "h2": {"bundle_manifest_sha256": sha(HERE / "evidence/h2_ad4/BUNDLE_MANIFEST.json"),
               "source_archive_sha256": bm["source_archive"]["sha256"],
               "checkpoint_hash_records_all_match": bm["checkpoint_hash_records_all_match"],
               "claim_mapping": ("A1. Under v1 every AutoND DI experiment bears on A1 and A2; the A1-only assignment "
                                 "of retraining experiments was first recorded after the A-D4 result (D2-N "
                                 "registration 5fb47d3, 2026-11-05 04:12 UTC; A-D4 evidence ad8c5d8, 2026-11-04). "
                                 "Not preregistered. No historical assignment or verdict is changed."),
               "scored_as": "replay MATCH/MISMATCH, separate from detection, false alarms and claim outcomes"},
        "specification_gaps_unspecified": {
            "X1": "SUBSTANTIAL reproduction for a component that is not reproduction-only",
            "X2": "PARTIAL reproduction",
            "X3": "a required dimension with an inadmissible FAIL when L* <= L1",
            "X4": "a required CE test that is INCONCLUSIVE (identified during implementation; not in the design draft)",
            "conflict": "no rule; U8 excluded", "invalid_evidence": "no rule; U4 detector-level only",
            "d4_progression_to_di": "no rule; S5 detector-level only"},
        "baselines": {"B1": "fresh-data inference on H1, S2, K3; descriptive; no decision threshold",
                      "B1S": "H2 stored-prediction rescoring; separate", "B1_naive": "illustrative only; not executed",
                      "B2": "executed checklist C1-C8 with per-bundle applicability"},
        "engine_tests": {"E1": 1960000, "E2": 134, "E3": 121, "P": "pairwise, seed 20261106",
                         "R": "100000 draws, seed 20261107", "B": "boundary literals"},
    }
    reg_p.write_text(json.dumps(registration, indent=1, sort_keys=True) + "\n")
    print(f"manifest {sha(man_p)}  ({len(sources)} sources, {len(evidence)} evidence, {len(fixtures)} fixtures)")
    print(f"registration {sha(reg_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
