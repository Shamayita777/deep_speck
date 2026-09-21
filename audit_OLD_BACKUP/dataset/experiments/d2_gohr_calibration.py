"""Final calibration-only driver for the Gohr/Speck D2 audit.

This module is deliberately separate from the production D2 driver.
It does not modify or invoke the production audit entry point.  It imports
only the already-reviewed calibration implementation and runs controlled
fault-injection sensitivity calibration on a separate calibration dataset.

The default detector parameters are frozen to the production D2 configuration
used for the Gohr audit.  The calibration dataset size is intentionally
separate from the production 10M/1M/1M audit datasets; therefore calibration
results are sensitivity evidence for the detector, not evidence about the
clean production data and not automatically scale-invariant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
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
from audit.dataset.experiments.d2_gohr_sample_dependence import run_calibration

# ---------------------------------------------------------------------------
# Frozen final-calibration design.
# These values match the production D2 detector configuration wherever the
# calibration driver exercises that detector.  Only the calibration dataset
# size is intentionally smaller than the production audit dataset.
# ---------------------------------------------------------------------------

CALIBRATION_DESIGN_VERSION = "d2-calibration-final-v1"
DEFAULT_CALIBRATION_SAMPLES = 100_000
DEFAULT_CALIBRATION_REPLICATES = 100
DEFAULT_CALIBRATION_QUERY_COUNT = 2_048
DEFAULT_CALIBRATION_TABLES = 8
DEFAULT_CALIBRATION_NULL_PAIRS = 100_000
DEFAULT_CALIBRATION_PAIRS = 20_000
DEFAULT_CALIBRATION_NULL_PERMUTATIONS = 120_000

# These are inherited from the reviewed production D2 implementation rather
# than independently chosen by this calibration driver.
CALIBRATION_FRACTIONS = (0.001, 0.005, 0.01, 0.025, 0.05)
CALIBRATION_NEAR_RADII = tuple(NEAR_DUPLICATE_RADII)
CALIBRATION_LAG = 1


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run the frozen final Gohr D2 sensitivity calibration only. "
            "This does not run or modify the production clean-data audit."
        )
    )
    p.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Rows in the separate calibration dataset (default: 100000).",
    )
    p.add_argument(
        "--calibration-replicates",
        type=int,
        default=DEFAULT_CALIBRATION_REPLICATES,
        help="Independent fault-injection trials per calibration condition (default: 100).",
    )
    p.add_argument("--audit-seed", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("d2_calibration_final_v1.json"),
    )
    return p.parse_args()


def validate_args(a):
    if a.calibration_samples < 2:
        raise ValueError("--calibration-samples must be >= 2")
    if a.calibration_replicates < CALIBRATION_MIN_REPLICATES:
        raise ValueError(
            f"--calibration-replicates must be >= {CALIBRATION_MIN_REPLICATES}"
        )
    if a.audit_seed < 0:
        raise ValueError("--audit-seed must be >= 0")


def _sha256_json(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


EXPECTED_RUN_CALIBRATION_MODULE = "audit.dataset.experiments.d2_gohr_sample_dependence"
EXPECTED_ENGINE_MODULE = "audit.dataset.d2_sample_dependence"
EXPECTED_SCHEMA_VERSION = "6.4"


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> dict:
    """Best-effort git commit capture. Absence is a WARNING, not a fail-closed
    condition: a source tree shipped without .git (e.g. a Kaggle dataset
    upload of the repo) does not by itself indicate the wrong implementation
    is running -- that is what the module/hash checks below establish.
    """
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parent
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parent
        ) != 0
        return {"status": "captured", "commit": commit, "working_tree_dirty": dirty}
    except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
        return {"status": "unavailable", "warning": f"{type(exc).__name__}: {exc}"}


def verify_calibration_engine_provenance() -> dict:
    """Fail-closed check that `run_calibration` / `calibrate_fault` resolve to
    the expected, already-reviewed implementation, and record exactly what
    was actually imported so the output artifact is self-describing about
    which code produced it -- rather than trusting the import statement's
    text alone, which stale modules earlier on sys.path or a shadowing
    package could silently defeat.
    """
    import inspect

    import audit.dataset.d2_sample_dependence as engine_module
    from audit.dataset.d2_sample_dependence import calibrate_fault as engine_calibrate_fault
    from audit.dataset.experiments.d2_gohr_sample_dependence import run_calibration as driver_run_calibration

    problems = []

    run_calibration_module = driver_run_calibration.__module__
    if run_calibration_module != EXPECTED_RUN_CALIBRATION_MODULE:
        problems.append(
            f"run_calibration resolved to module '{run_calibration_module}', "
            f"expected '{EXPECTED_RUN_CALIBRATION_MODULE}'."
        )

    calibrate_fault_module = engine_calibrate_fault.__module__
    if calibrate_fault_module != EXPECTED_ENGINE_MODULE:
        problems.append(
            f"calibrate_fault resolved to module '{calibrate_fault_module}', "
            f"expected '{EXPECTED_ENGINE_MODULE}'."
        )

    schema_version = getattr(engine_module, "SCHEMA_VERSION", None)
    if str(schema_version) != EXPECTED_SCHEMA_VERSION:
        problems.append(
            f"audit.dataset.d2_sample_dependence.SCHEMA_VERSION={schema_version!r}, "
            f"expected {EXPECTED_SCHEMA_VERSION!r}."
        )

    try:
        run_calibration_file = inspect.getfile(driver_run_calibration)
        engine_file = inspect.getfile(engine_calibrate_fault)
        this_driver_file = inspect.getfile(sys.modules[__name__])
    except (TypeError, OSError) as exc:
        raise RuntimeError(
            f"CALIBRATION PROVENANCE CHECK FAILED (fail-closed): could not locate the source file "
            f"backing run_calibration/calibrate_fault ({exc}). Refusing to run an unverifiable "
            f"calibration implementation."
        ) from exc

    if problems:
        detail = " ".join(problems)
        raise RuntimeError(
            "CALIBRATION PROVENANCE CHECK FAILED (fail-closed): the calibration engine actually "
            f"resolved at runtime does not match the expected reviewed implementation. {detail} "
            "Refusing to run. Check for a stale/duplicate module earlier on sys.path or an "
            "unexpected package installation."
        )

    return {
        "verification_status": "VERIFIED",
        "run_calibration": {
            "module": run_calibration_module,
            "qualname": driver_run_calibration.__qualname__,
            "file": run_calibration_file,
            "file_sha256": _sha256_file(run_calibration_file),
        },
        "calibrate_fault_engine": {
            "module": calibrate_fault_module,
            "qualname": engine_calibrate_fault.__qualname__,
            "file": engine_file,
            "file_sha256": _sha256_file(engine_file),
            "schema_version": str(schema_version),
        },
        "this_driver": {
            "file": this_driver_file,
            "file_sha256": _sha256_file(this_driver_file),
        },
        "git": _git_commit(),
        "expected": {
            "run_calibration_module": EXPECTED_RUN_CALIBRATION_MODULE,
            "engine_module": EXPECTED_ENGINE_MODULE,
            "schema_version": EXPECTED_SCHEMA_VERSION,
        },
    }


def _production_effective_alpha() -> tuple[int, float]:
    """Reproduce the production D2 family count and effective alpha exactly."""
    total = familywise_test_count(
        partition_count=3,
        lag_count=3 * len(DEFAULT_LAGS),
        near_duplicate_radius_count=len(NEAR_DUPLICATE_RADII),
        structured_view_count=15,
        multivariate_count=3,
    )
    return total, FAMILYWISE_ALPHA / total


def _monotonicity_diagnostic(series: dict[str, dict]) -> dict:
    """Descriptive monotonicity diagnostic; never changes a calibration decision."""
    fractions = sorted(float(k) for k in series)
    rates = [float(series[f"{f:.6g}"]["detection_rate"]) for f in fractions]
    decreases = []
    for i in range(1, len(rates)):
        if rates[i] < rates[i - 1]:
            decreases.append(
                {
                    "from_fraction": fractions[i - 1],
                    "to_fraction": fractions[i],
                    "from_detection_rate": rates[i - 1],
                    "to_detection_rate": rates[i],
                    "decrease": rates[i - 1] - rates[i],
                }
            )
    return {
        "fractions": fractions,
        "detection_rates": rates,
        "nondecreasing_point_estimate": not decreases,
        "decreases": decreases,
        "interpretation": (
            "Descriptive diagnostic only. A decrease does not constitute a statistical "
            "failure of the calibration and does not alter any detector decision."
        ),
    }


def main():
    a = parse_args()
    validate_args(a)

    # Fail closed FIRST, before any dataset generation or computation, if the
    # calibration engine actually resolved at runtime is not the expected,
    # already-reviewed implementation.
    print("Verifying calibration engine provenance (fail-closed check)...")
    engine_provenance = verify_calibration_engine_provenance()
    print(f"  run_calibration  : {engine_provenance['run_calibration']['module']}")
    print(f"    file           : {engine_provenance['run_calibration']['file']}")
    print(f"    sha256         : {engine_provenance['run_calibration']['file_sha256']}")
    print(f"  calibrate_fault  : {engine_provenance['calibrate_fault_engine']['module']}"
          f" (schema_version={engine_provenance['calibrate_fault_engine']['schema_version']})")
    print(f"    sha256         : {engine_provenance['calibrate_fault_engine']['file_sha256']}")
    git = engine_provenance["git"]
    if git["status"] == "captured":
        print(f"  git commit       : {git['commit']}"
              f"{' (DIRTY working tree)' if git['working_tree_dirty'] else ''}")
    else:
        print(f"  git commit       : unavailable ({git.get('warning')}) -- not fail-closed, recorded as a warning")
    print("  Provenance check: VERIFIED")

    test_count, effective_alpha = _production_effective_alpha()

    # IMPORTANT: this is a separate adapter instance and a separate dataset.
    # The production driver is never called.
    adapter = GohrD2Adapter(num_rounds=5)

    print(
        f"Generating FINAL calibration dataset: {a.calibration_samples:,} samples..."
    )
    clean, _, _ = adapter.generate_partition(a.calibration_samples)
    print(f"  shape={clean.shape}, dtype={clean.dtype}")

    print("Running FINAL controlled D2 sensitivity calibration only...")
    print(f"  design={CALIBRATION_DESIGN_VERSION}")
    print(f"  effective alpha={effective_alpha:.12g} (FWER={FAMILYWISE_ALPHA})")
    print(f"  replicates={a.calibration_replicates}")
    print(f"  calibration samples={a.calibration_samples:,}")
    print(f"  near-duplicate null pairs={DEFAULT_CALIBRATION_NULL_PAIRS:,}")
    print(f"  lag pairs={DEFAULT_CALIBRATION_PAIRS:,}")
    print(f"  lag null permutations={DEFAULT_CALIBRATION_NULL_PERMUTATIONS:,}")

    calibration = run_calibration(
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
    )

    # Add a purely descriptive monotonicity diagnostic.  It is intentionally
    # not fed back into the calibration decision and cannot alter PASS/FAIL.
    calibration["lag_copy_injection_monotonicity_diagnostic"] = (
        _monotonicity_diagnostic(calibration["lag_copy_injection"])
    )

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
        "calibration_is_not_clean_evidence": True,
        "purpose": "Controlled fault-injection sensitivity calibration; does not audit clean data.",
        "protocol": {
            "name": "component-specific controlled fault-injection sensitivity calibration",
            "detection_target": CALIBRATION_DETECTION_TARGET,
            "target_semantics": (
                "Wilson lower bound is the only confidence-supported target criterion; "
                "point estimate is reported separately."
            ),
            "trial_semantics": (
                "Independent fault-injection trials conditional on one fixed clean "
                "calibration dataset instance."
            ),
            "near_duplicate_interpretation": "screening sensitivity only; no formal Type-I interpretation",
            "lag_interpretation": "sensitivity calibration of the frozen serial-dependence detector",
        },
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
