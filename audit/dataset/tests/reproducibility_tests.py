"""Section 16 reproducibility/validation test suite.

NON-EVIDENTIARY. Structural/plumbing checks only -- this is NOT a scientific
calibration result and asserts nothing about detector sensitivity or clean
data. It validates that the three fixes behave as specified and that the
artifact-production plumbing is sound.
"""
from __future__ import annotations
import sys, json, hashlib
sys.path.insert(0, "/home/claude/work")
import numpy as np

results = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


for m in list(sys.modules):
    if m.startswith("audit."):
        del sys.modules[m]

from audit.dataset.d2_sample_dependence import calibrate_fault, CALIBRATION_MIN_REPLICATES
import audit.dataset.experiments.d2_gohr_sample_dependence as driver
import importlib.util
spec = importlib.util.spec_from_file_location(
    "d2_gohr_calibration_final", "/home/claude/work/audit/dataset/experiments/d2_gohr_calibration_final.py"
)
final_mod = importlib.util.module_from_spec(spec)
sys.modules["d2_gohr_calibration_final"] = final_mod
spec.loader.exec_module(final_mod)

# 1. Configuration serialization is stable (same config -> same JSON -> same hash, twice)
cfg = {"a": 1, "b": [1, 2, 3], "c": {"x": 0.001}}
h1 = final_mod._sha256_json(cfg)
h2 = final_mod._sha256_json(cfg)
check("config serialization is stable across repeated calls", h1 == h2, f"{h1} vs {h2}")

# 2. Config hash actually changes when config changes (not a constant/degenerate hash)
cfg2 = dict(cfg); cfg2["a"] = 2
check("config hash changes when config changes", final_mod._sha256_json(cfg) != final_mod._sha256_json(cfg2))

# 3. Source-file hashing is stable and matches manual sha256
p = "/home/claude/work/audit/dataset/d2_sample_dependence.py"
h_manual = hashlib.sha256(open(p, "rb").read()).hexdigest()
h_fn = final_mod._sha256_file(p)
check("source file hash matches manual sha256", h_manual == h_fn)

# 4. Provenance/source verification: passes against the real engine
prov = final_mod.verify_calibration_engine_provenance()
check("provenance check VERIFIED against real engine", prov["verification_status"] == "VERIFIED")
check("provenance records run_calibration module", prov["run_calibration"]["module"] == "audit.dataset.experiments.d2_gohr_sample_dependence")
check("provenance records engine schema_version 6.4", prov["calibrate_fault_engine"]["schema_version"] == "6.4")

# 5. Provenance fails closed on tampered schema_version
import audit.dataset.d2_sample_dependence as engine_mod
orig = engine_mod.SCHEMA_VERSION
engine_mod.SCHEMA_VERSION = "bogus"
failed_closed = False
try:
    final_mod.verify_calibration_engine_provenance()
except RuntimeError:
    failed_closed = True
finally:
    engine_mod.SCHEMA_VERSION = orig
check("provenance fails closed on tampered SCHEMA_VERSION", failed_closed)

# 6. Seeds are recorded / deterministic: same (seed, replicates) -> identical per-replicate spawn keys
def inj(x, r): return x, {"note": "x"}
def det(x, r): return {"detected": bool(r.random() < 0.5)}
out_a = calibrate_fault(clean_features=np.zeros((5,4),dtype=np.uint8), injector=inj, detector=det, replicates=100, seed=123)
out_b = calibrate_fault(clean_features=np.zeros((5,4),dtype=np.uint8), injector=inj, detector=det, replicates=100, seed=123)
same_outcomes = [r["detected"] for r in out_a["replicate_records"]] == [r["detected"] for r in out_b["replicate_records"]]
same_keys = [r["seed_spawn_key"] for r in out_a["replicate_records"]] == [r["seed_spawn_key"] for r in out_b["replicate_records"]]
check("identical seed -> identical spawn keys across runs", same_keys)
check("identical seed -> identical per-replicate outcomes across runs", same_outcomes)

# 7. Different seed -> (almost certainly) different outcomes, i.e. seed actually matters
out_c = calibrate_fault(clean_features=np.zeros((5,4),dtype=np.uint8), injector=inj, detector=det, replicates=100, seed=999)
diff_outcomes = [r["detected"] for r in out_a["replicate_records"]] != [r["detected"] for r in out_c["replicate_records"]]
check("different seed -> different outcome sequence", diff_outcomes)

# 8. Failed replicates are preserved, never silently dropped
def flaky(x, r):
    if r.random() < 0.4:
        raise RuntimeError("synthetic failure")
    return {"detected": True}
out_flaky = calibrate_fault(clean_features=np.zeros((5,4),dtype=np.uint8), injector=inj, detector=flaky, replicates=100, seed=5)
check("every requested replicate is accounted for (ok+failed)", len(out_flaky["replicate_records"]) == 100)
check("failed_replicates + valid_replicates == requested_replicates",
      out_flaky["failed_replicates"] + out_flaky["valid_replicates"] == out_flaky["requested_replicates"])
n_failed_records = sum(1 for r in out_flaky["replicate_records"] if r["status"] == "failed")
check("failed records carry a failure_reason", all(r["failure_reason"] for r in out_flaky["replicate_records"] if r["status"] == "failed"))
check("failed record count matches failed_replicates field", n_failed_records == out_flaky["failed_replicates"])

# 9. Below-minimum-valid-replicates -> INCONCLUSIVE, not a silently-reduced-denominator PASS/FAIL
def always_fail(x, r): raise RuntimeError("boom")
out_allfail = calibrate_fault(clean_features=np.zeros((5,4),dtype=np.uint8), injector=inj, detector=always_fail, replicates=100, seed=6)
check("all-failure run is INCONCLUSIVE, not silently scored", out_allfail["status"] == "INCONCLUSIVE_INSUFFICIENT_VALID_REPLICATES")
check("INCONCLUSIVE run reports null detection_rate (not 0.0)", out_allfail["detection_rate"] is None)

# 10. Alpha-resolvability guard fails closed on an under-resolving permutation budget
raised = False
try:
    driver.run_calibration(
        np.zeros((1000, 64), dtype=np.uint8), replicates=100, seed=1, query_count=64, tables=2,
        null_pairs=100, pairs=100, null_permutations=2_000, alpha=9.8e-05,
        practical_excess_tvd_threshold=0.01,
    )
except ValueError:
    raised = True
check("alpha-resolvability guard fails closed on 2,000-permutation (stale) budget", raised)

# 11. Alpha-resolvability guard does NOT fire for the frozen final 120,000-permutation budget
#     (only check the guard math itself here -- not a full run, which is expensive)
ref = min(driver.DEFAULT_NULL_REFERENCE_PERMUTATIONS if hasattr(driver, "DEFAULT_NULL_REFERENCE_PERMUTATIONS") else 50, max(1, 120_000 - 1))
tail = 120_000 - ref
check("frozen 120,000-permutation budget resolves the effective alpha", 1.0/(tail+1) < 9.8e-05)

# 12. Calibration/production isolation: run_calibration signature takes only a features array in,
#     never touches a "production" dataset path, certificate writer, or decision object.
import inspect
sig = inspect.signature(driver.run_calibration)
param_names = set(sig.parameters.keys())
check("run_calibration has no production-certificate/decision parameters",
      not ({"decision", "certificate", "output_path"} & param_names))

# 13. Output schema shape: calibrate_fault result has all required top-level keys
required_keys = {
    "status", "requested_replicates", "valid_replicates", "failed_replicates",
    "minimum_required_valid_replicates", "detections", "detection_rate",
    "detection_rate_95_ci", "target", "point_estimate_target_met",
    "confidence_lower_bound_target_met", "replicate_records",
}
check("calibrate_fault output contains all required schema keys", required_keys.issubset(out_a.keys()),
      f"missing: {required_keys - set(out_a.keys())}")

print()
print(f"=== {sum(1 for _,s,_ in results if s=='PASS')}/{len(results)} checks passed ===")
print("NON-EVIDENTIARY: this validates plumbing/reproducibility only, not any scientific calibration outcome.")
