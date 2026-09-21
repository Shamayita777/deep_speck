"""
D2 Section 16 reproducibility/validation test suite — v6.4 API.

NON-EVIDENTIARY. Structural/plumbing checks only.
This suite validates properties exposed by the frozen v6.4 calibration API.
It does NOT assert detector sensitivity, calibration success, or D2 PASS/FAIL.

Important v6.4 API fact:
    calibrate_fault() returns aggregate calibration results. It does not expose
    per-replicate records, spawn keys, or failure records. Therefore tests that
    previously depended on `replicate_records` are intentionally replaced by
    tests of the actual v6.4 contract rather than by modifying production code.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


# Avoid stale imports if this script is rerun in-process.
for module_name in list(sys.modules):
    if module_name.startswith("audit."):
        del sys.modules[module_name]

from audit.dataset.d2_sample_dependence import (  # noqa: E402
    CALIBRATION_MIN_REPLICATES,
    calibrate_fault,
)
import audit.dataset.experiments.d2_gohr_sample_dependence as driver  # noqa: E402

import audit.dataset.experiments.d2_gohr_calibration as final_mod


# ---------------------------------------------------------------------------
# 1. Stable configuration serialization.
# ---------------------------------------------------------------------------
cfg = {"a": 1, "b": [1, 2, 3], "c": {"x": 0.001}}
h1 = final_mod._sha256_json(cfg)
h2 = final_mod._sha256_json(cfg)
check(
    "config serialization is stable across repeated calls",
    h1 == h2,
    f"{h1} vs {h2}",
)


# ---------------------------------------------------------------------------
# 2. Configuration hash is sensitive to configuration changes.
# ---------------------------------------------------------------------------
cfg2 = dict(cfg)
cfg2["a"] = 2
check(
    "config hash changes when config changes",
    final_mod._sha256_json(cfg) != final_mod._sha256_json(cfg2),
)


# ---------------------------------------------------------------------------
# 3. Source-file hashing matches independent SHA-256.
# ---------------------------------------------------------------------------
engine_path = REPO_ROOT / "audit" / "dataset" / "d2_sample_dependence.py"
h_manual = hashlib.sha256(engine_path.read_bytes()).hexdigest()
h_fn = final_mod._sha256_file(str(engine_path))
check(
    "source file hash matches manual sha256",
    h_manual == h_fn,
)


# ---------------------------------------------------------------------------
# 4–6. Calibration-engine provenance.
# ---------------------------------------------------------------------------
prov = final_mod.verify_calibration_engine_provenance()

check(
    "provenance check VERIFIED against real engine",
    prov["verification_status"] == "VERIFIED",
)
check(
    "provenance records run_calibration module",
    prov["run_calibration"]["module"]
    == "audit.dataset.experiments.d2_gohr_sample_dependence",
)
check(
    "provenance records engine schema_version 6.4",
    prov["calibrate_fault_engine"]["schema_version"] == "6.4",
)


# ---------------------------------------------------------------------------
# 7. Provenance fails closed if the engine schema is tampered.
# ---------------------------------------------------------------------------
import audit.dataset.d2_sample_dependence as engine_mod  # noqa: E402

orig_schema = engine_mod.SCHEMA_VERSION
failed_closed = False
try:
    engine_mod.SCHEMA_VERSION = "bogus"
    final_mod.verify_calibration_engine_provenance()
except RuntimeError:
    failed_closed = True
finally:
    engine_mod.SCHEMA_VERSION = orig_schema

check(
    "provenance fails closed on tampered SCHEMA_VERSION",
    failed_closed,
)


# ---------------------------------------------------------------------------
# 8. Same seed -> identical AGGREGATE v6.4 result.
#
# v6.4 exposes aggregate detections/rate/CI, not per-replicate records.
# ---------------------------------------------------------------------------
def inj(x, rng):
    return x, {"note": "x"}


def det(x, rng):
    return {"detected": bool(rng.random() < 0.5)}


out_a = calibrate_fault(
    clean_features=np.zeros((5, 4), dtype=np.uint8),
    injector=inj,
    detector=det,
    replicates=100,
    seed=123,
)
out_b = calibrate_fault(
    clean_features=np.zeros((5, 4), dtype=np.uint8),
    injector=inj,
    detector=det,
    replicates=100,
    seed=123,
)

stable_fields = (
    "detections",
    "detection_rate",
    "detection_rate_95_ci",
    "point_estimate_target_met",
    "confidence_lower_bound_target_met",
    "target",
    "replicates",
    "trial_semantics",
)

same_aggregate = all(out_a[k] == out_b[k] for k in stable_fields)
check(
    "identical seed -> identical aggregate calibration result",
    same_aggregate,
)


# ---------------------------------------------------------------------------
# 9. Different seed changes the aggregate for this deterministic test harness.
#
# This is a seed-sensitivity sanity check, not a scientific result.
# ---------------------------------------------------------------------------
out_c = calibrate_fault(
    clean_features=np.zeros((5, 4), dtype=np.uint8),
    injector=inj,
    detector=det,
    replicates=100,
    seed=999,
)
different_aggregate = any(out_a[k] != out_c[k] for k in stable_fields)
check(
    "different seed -> different aggregate result in seed-sensitivity harness",
    different_aggregate,
)


# ---------------------------------------------------------------------------
# 10. v6.4 minimum-replicate guard fails closed.
# ---------------------------------------------------------------------------
raised = False
try:
    calibrate_fault(
        clean_features=np.zeros((5, 4), dtype=np.uint8),
        injector=inj,
        detector=det,
        replicates=CALIBRATION_MIN_REPLICATES - 1,
        seed=123,
    )
except ValueError:
    raised = True

check(
    "calibration rejects fewer than the required minimum replicates",
    raised,
    f"minimum={CALIBRATION_MIN_REPLICATES}",
)


# ---------------------------------------------------------------------------
# 11. v6.4 aggregate output schema is complete.
# ---------------------------------------------------------------------------
required_keys = {
    "confidence_lower_bound_target_met",
    "detection_rate",
    "detection_rate_95_ci",
    "detections",
    "injected_fault_metadata_summary",
    "point_estimate_target_met",
    "replicates",
    "target",
    "trial_semantics",
}
check(
    "v6.4 calibrate_fault output contains required aggregate schema keys",
    required_keys.issubset(out_a.keys()),
    f"missing: {sorted(required_keys - set(out_a.keys()))}",
)


# ---------------------------------------------------------------------------
# 12. Calibration/production isolation.
# ---------------------------------------------------------------------------
sig = inspect.signature(driver.run_calibration)
param_names = set(sig.parameters.keys())
check(
    "run_calibration has no production-certificate/decision parameters",
    not ({"decision", "certificate", "output_path"} & param_names),
)


# ---------------------------------------------------------------------------
# 13. Explicitly verify that this suite does not claim unavailable
# per-replicate provenance. This is a contract check, not a scientific result.
# ---------------------------------------------------------------------------
check(
    "v6.4 contract is aggregate-only (no replicate_records field assumed)",
    "replicate_records" not in out_a,
)

print()
passed = sum(1 for _, status, _ in results if status == "PASS")
print(f"=== {passed}/{len(results)} checks passed ===")
print(
    "NON-EVIDENTIARY: validates v6.4 plumbing/reproducibility only; "
    "does not establish detector sensitivity, dataset independence, "
    "or a D2 scientific PASS."
)
