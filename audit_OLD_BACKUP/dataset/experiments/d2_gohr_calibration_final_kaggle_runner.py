"""Kaggle-ready FINAL calibration driver for the Gohr/Speck D2 audit.

Identical frozen design, identical fail-closed provenance verification, and
identical output schema to d2_gohr_calibration_final.py. The only difference
is HOW the 25 independent calibration conditions are executed:

  * --max-workers 1 (default): conditions run one at a time in this process,
    via the SAME calibrate_fault/_near_detector/_lag_detector calls as the
    reference driver. Proven byte-for-byte equivalent to
    d2_gohr_calibration_final.py's output (see
    test_parallel_matches_serial.py) -- this flag changes nothing about the
    statistics, only adds checkpointing.

  * --max-workers N>1: the same 25 conditions are distributed across N
    worker processes. Each condition's seed is a fixed, deterministic
    function of (audit_seed, radius-or-None, fraction) -- copied verbatim
    from run_calibration() -- so which process computes a condition cannot
    change its result. This is proven empirically, not just argued: see
    test_parallel_matches_serial.py, which asserts max_workers=1 and
    max_workers=4 produce identical per-replicate outcomes and identical
    seed_spawn_key sequences for every condition.

  * Every condition is checkpointed to --checkpoint-dir as soon as it
    completes. Re-running with the same --checkpoint-dir and --audit-seed
    skips already-completed conditions and reproduces the identical final
    combined result -- this is how a run that would exceed a single
    scheduler time slot (e.g. Kaggle's 12-hour session limit) can be split
    across multiple sessions without deviating from the predeclared
    100-replicate-per-condition design: each condition is still computed in
    full (100 replicates, real frozen parameters) exactly once, regardless
    of how many process launches it took to get there.

This does not change the null construction, the alpha, the permutation
budget, the replicate count, or the decision logic.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from audit.dataset.adapters.gohr_d2 import GohrD2Adapter
from audit.dataset.d2_sample_dependence import (
    CALIBRATION_DETECTION_TARGET,
    CALIBRATION_MIN_REPLICATES,
    DEFAULT_LAGS,
    FAMILYWISE_ALPHA,
    NEAR_DUPLICATE_RADII,
    PRACTICAL_EXCESS_TVD_THRESHOLD,
    familywise_test_count,
)
from audit.dataset.experiments.d2_calibration_parallel_runner import run_calibration_checkpointed
from audit.dataset.experiments.d2_gohr_calibration import (
    CALIBRATION_DESIGN_VERSION,
    CALIBRATION_FRACTIONS,
    CALIBRATION_LAG,
    CALIBRATION_NEAR_RADII,
    DEFAULT_CALIBRATION_NULL_PAIRS,
    DEFAULT_CALIBRATION_NULL_PERMUTATIONS,
    DEFAULT_CALIBRATION_PAIRS,
    DEFAULT_CALIBRATION_QUERY_COUNT,
    DEFAULT_CALIBRATION_REPLICATES,
    DEFAULT_CALIBRATION_SAMPLES,
    DEFAULT_CALIBRATION_TABLES,
    _monotonicity_diagnostic,
    _production_effective_alpha,
    _sha256_json,
    verify_calibration_engine_provenance,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Kaggle-ready checkpointed/parallel execution of the frozen FINAL Gohr D2 calibration."
    )
    p.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    p.add_argument("--calibration-replicates", type=int, default=DEFAULT_CALIBRATION_REPLICATES)
    p.add_argument("--audit-seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("d2_calibration_final_v1.json"))
    p.add_argument(
        "--checkpoint-dir", type=Path, default=Path("d2_calibration_checkpoints"),
        help="Per-condition checkpoint files live here. Re-run with the same directory + audit-seed to resume.",
    )
    p.add_argument(
        "--max-workers", type=int, default=1,
        help="Number of worker processes across the 25 independent conditions. "
             "1 (default) = byte-identical to d2_gohr_calibration_final.py. "
             ">1 = same results, computed in parallel (proven equivalent; see test_parallel_matches_serial.py).",
    )
    return p.parse_args()


def validate_args(a):
    if a.calibration_samples < 2:
        raise ValueError("--calibration-samples must be >= 2")
    if a.calibration_replicates < CALIBRATION_MIN_REPLICATES:
        raise ValueError(f"--calibration-replicates must be >= {CALIBRATION_MIN_REPLICATES}")
    if a.audit_seed < 0:
        raise ValueError("--audit-seed must be >= 0")
    if a.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")


def main():
    a = parse_args()
    validate_args(a)

    print("Verifying calibration engine provenance (fail-closed check)...")
    engine_provenance = verify_calibration_engine_provenance()
    print(f"  run_calibration  : {engine_provenance['run_calibration']['module']}")
    print(f"  sha256           : {engine_provenance['run_calibration']['file_sha256']}")
    print("  Provenance check: VERIFIED")

    test_count, effective_alpha = _production_effective_alpha()

    adapter = GohrD2Adapter(num_rounds=5)
    print(f"Generating FINAL calibration dataset: {a.calibration_samples:,} samples...")
    clean, _, _ = adapter.generate_partition(a.calibration_samples)
    print(f"  shape={clean.shape}, dtype={clean.dtype}")

    print(f"Running FINAL calibration via checkpointed runner (max_workers={a.max_workers}, "
          f"checkpoint_dir={a.checkpoint_dir})...")

    def progress(msg):
        print(f"  {datetime.now(timezone.utc).isoformat()}  {msg}", flush=True)

    calibration = run_calibration_checkpointed(
        clean,
        replicates=a.calibration_replicates,
        seed=a.audit_seed + 50_000,
        query_count=DEFAULT_CALIBRATION_QUERY_COUNT,
        tables=DEFAULT_CALIBRATION_TABLES,
        null_pairs=DEFAULT_CALIBRATION_NULL_PAIRS,
        pairs=DEFAULT_CALIBRATION_PAIRS,
        null_permutations=DEFAULT_CALIBRATION_NULL_PERMUTATIONS,
        alpha=effective_alpha,
        practical_excess_tvd_threshold=PRACTICAL_EXCESS_TVD_THRESHOLD,
        checkpoint_dir=str(a.checkpoint_dir),
        max_workers=a.max_workers,
        progress_cb=progress,
    )
    calibration["lag_copy_injection_monotonicity_diagnostic"] = _monotonicity_diagnostic(calibration["lag_copy_injection"])

    frozen_configuration = {
        "familywise_alpha": FAMILYWISE_ALPHA,
        "effective_test_alpha": effective_alpha,
        "familywise_test_count": test_count,
        "calibration_samples": a.calibration_samples,
        "calibration_replicates": a.calibration_replicates,
        "calibration_fractions": list(CALIBRATION_FRACTIONS),
        "near_duplicate_radii": list(CALIBRATION_NEAR_RADII),
        "lag": CALIBRATION_LAG,
        "near_duplicate_query_count": DEFAULT_CALIBRATION_QUERY_COUNT,
        "near_duplicate_projection_tables": DEFAULT_CALIBRATION_TABLES,
        "near_duplicate_null_pairs": DEFAULT_CALIBRATION_NULL_PAIRS,
        "lag_pairs": DEFAULT_CALIBRATION_PAIRS,
        "lag_null_permutations": DEFAULT_CALIBRATION_NULL_PERMUTATIONS,
        "practical_excess_tvd_threshold": PRACTICAL_EXCESS_TVD_THRESHOLD,
        "audit_seed": a.audit_seed,
        "production_detector_configuration_matched": True,
    }

    out = {
        "schema_version": CALIBRATION_DESIGN_VERSION,
        "status": "COMPLETED",
        "execution_mode": "checkpointed_kaggle_runner",
        "execution_mode_note": (
            "Executed via d2_gohr_calibration_final_kaggle_runner.py, not the single-process reference "
            "driver. max_workers=1 is proven byte-identical to d2_gohr_calibration_final.py's output; "
            "max_workers>1 is proven to produce identical per-condition results, just computed in "
            "parallel -- see test_parallel_matches_serial.py."
        ),
        "calibration_is_not_clean_evidence": True,
        "purpose": "Controlled fault-injection sensitivity calibration; does not audit clean data.",
        "dataset": {
            "dataset_id": adapter.DATASET_ID,
            "dataset_version": adapter.DATASET_VERSION,
            "num_rounds": adapter.num_rounds,
            "samples": int(a.calibration_samples),
            "feature_bits": int(adapter.FEATURE_BITS),
            "dtype": str(clean.dtype),
            "randomness_source": "os.urandom",
            "exact_replay_available": False,
        },
        "configuration": frozen_configuration,
        "calibration": calibration,
        "execution": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "utc_completed_at": datetime.now(timezone.utc).isoformat(),
            "command_seed": a.audit_seed,
            "calibration_seed_offset": 50_000,
            "max_workers": a.max_workers,
            "checkpoint_dir": str(a.checkpoint_dir),
            "git_commit": engine_provenance["git"],
        },
        "calibration_engine_provenance": engine_provenance,
        "integrity": {
            "frozen_configuration_sha256": _sha256_json(frozen_configuration),
            "calibration_engine_verification_status": engine_provenance["verification_status"],
            "production_clean_logic_invoked": False,
            "production_clean_logic_modified": False,
            "production_dataset_generated": False,
        },
        "limitations": [
            "This is sensitivity calibration, not evidence that the clean production dataset is independent.",
            "Trials are independent fault-injection trials conditional on one fixed calibration dataset instance.",
            "The calibration dataset is smaller than the 12M-row production audit dataset; sensitivity is therefore demonstrated at the calibration scale and is not assumed to be scale-invariant.",
            "The near-duplicate component is screening-only and does not receive a formal Type-I error interpretation.",
            "The lag-copy calibration characterizes the frozen detector under the specified injected fault model; it does not establish sensitivity to every possible form of serial dependence.",
            "Gohr dataset generation uses os.urandom, so this dataset is a fresh calibration instance rather than an exact replay of historical data.",
            "The monotonicity diagnostic is descriptive and does not alter calibration decisions.",
        ],
        "scope_statement": (
            "This artifact calibrates detector sensitivity only. It must not be used as "
            "evidence that the clean production dataset passes D2. Clean-data evidence "
            "comes only from the separately executed production D2 audit."
        ),
    }

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Final calibration result: {a.output}")


if __name__ == "__main__":
    main()
