"""
Gohr-specific D2 experiment driver.

This is the only layer that knows the Gohr/Speck case-study details:
    * 5-round configuration
    * 64-bit feature representation
    * Gohr generator instrumentation
    * the case-study Hamming reference
    * calibration fault strengths

The statistical engine remains in d2_sample_dependence.py and is generic.
The existing audit.dataset.adapters.gohr.GohrAdapter is retained for the
project's dataset/model/training functionality. D2 uses GohrD2Adapter only
for its dataset-integrity observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from audit.dataset.adapters.gohr_d2 import GohrD2Adapter
from audit.dataset.d2_sample_dependence import (
    DEFAULT_AUDIT_REPLICATES,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_LAGS,
    DEFAULT_MULTIVARIATE_PAIRS,
    DEFAULT_MULTIVARIATE_PERMUTATIONS,
    DEFAULT_PAIRS_PER_TEST,
    NEAR_DUPLICATE_RADII,
    FAMILYWISE_ALPHA,
    TVD_THRESHOLD,
    familywise_test_count,
    build_d2_certificate,
    calibrate_fault,
    inject_duplicates,
    inject_lag_copy,
    binomial_reference_distribution,
    empirical_independent_hamming_pmf,
    sample_within_distances,
    sample_lagged_distances,
    near_duplicate_summary,
    summarize_distances,
    print_report,
    run_d2,
)

DEFAULT_TRAIN_SAMPLES = 10_000_000
DEFAULT_VALIDATION_SAMPLES = 1_000_000
DEFAULT_TEST_SAMPLES = 1_000_000
DEFAULT_CALIBRATION_REPLICATES = 20
DEFAULT_CALIBRATION_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
DEFAULT_CALIBRATION_PAIRS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gohr D2 Sample Dependence Audit.")
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--pairs-per-test", type=int, default=DEFAULT_PAIRS_PER_TEST)
    parser.add_argument("--audit-replicates", type=int, default=DEFAULT_AUDIT_REPLICATES)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--multivariate-pairs", type=int, default=DEFAULT_MULTIVARIATE_PAIRS)
    parser.add_argument("--multivariate-permutations", type=int, default=DEFAULT_MULTIVARIATE_PERMUTATIONS)
    parser.add_argument("--audit-seed", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true", help="Run controlled duplicate/lag-copy sensitivity calibration.")
    parser.add_argument("--calibration-replicates", type=int, default=DEFAULT_CALIBRATION_REPLICATES, help="Independent sensitivity-calibration replicates (default: 20).")
    parser.add_argument("--calibration-pairs", type=int, default=DEFAULT_CALIBRATION_PAIRS)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = {
        "train-samples": args.train_samples,
        "validation-samples": args.validation_samples,
        "test-samples": args.test_samples,
        "pairs-per-test": args.pairs_per_test,
        "audit-replicates": args.audit_replicates,
        "bootstrap-replicates": args.bootstrap_replicates,
        "multivariate-pairs": args.multivariate_pairs,
        "multivariate-permutations": args.multivariate_permutations,
        "calibration-replicates": args.calibration_replicates,
        "calibration-pairs": args.calibration_pairs,
        "calibration-bootstrap-replicates": args.calibration_bootstrap_replicates,
    }
    for name, value in values.items():
        if value < 1:
            raise ValueError(f"--{name} must be >= 1.")
    if args.train_samples < 2 or args.validation_samples < 2 or args.test_samples < 2:
        raise ValueError("All partitions must contain at least two samples.")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be >= 100.")
    if args.audit_seed < 0:
        raise ValueError("--audit-seed must be >= 0.")
    if args.calibrate and args.calibration_replicates < 10:
        raise ValueError("Calibration requires at least 10 independent replicates.")


def default_output_path(args: argparse.Namespace) -> Path:
    return Path("audit/dataset/evidence/d2") / (
        f"d2_gohr_sample_dependence_{args.train_samples}_{args.validation_samples}_"
        f"{args.test_samples}_{args.pairs_per_test}pairs_{args.audit_replicates}replicates_"
        f"seed{args.audit_seed}.json"
    )


def _calibration_detector(*, target_component: str, pairs: int, bootstrap_replicates: int, alpha: float):
    """Return a detector using the same component-level rule as production D2.

    Calibration intentionally targets one D2 component at a time. This avoids
    allowing an unrelated detector to satisfy the calibration criterion and
    makes the resulting detection rate interpretable for the injected fault.
    The confirmatory Hamming null is the same empirical independent-row
    Poisson-binomial null used by production D2; the Gohr Binomial reference
    is retained only as a diagnostic.
    """
    if target_component == "d2_1_near_duplicate_structure":
        def detector(features: np.ndarray, rng: np.random.Generator) -> bool:
            reference = empirical_independent_hamming_pmf(features, features)
            distance_replicates = [sample_within_distances(features, pairs, rng)]
            result = near_duplicate_summary(
                distance_replicates,
                NEAR_DUPLICATE_RADII,
                reference,
                bootstrap_replicates,
                rng,
                alpha=alpha,
            )
            return any(v["practically_excessive"] for v in result.values())
        return detector

    if target_component == "d2_2_serial_dependence":
        def detector(features: np.ndarray, rng: np.random.Generator) -> bool:
            reference = empirical_independent_hamming_pmf(features, features)
            for lag in DEFAULT_LAGS:
                if lag >= len(features):
                    continue
                distances = sample_lagged_distances(features, lag, pairs, rng)
                result = summarize_distances(
                    distances,
                    reference,
                    rng,
                    alpha,
                    bootstrap_replicates,
                )
                if not result["tvd_uncertainty"]["within_threshold"]:
                    return True
            return False
        return detector

    raise ValueError(f"Unsupported calibration target: {target_component}")


def run_calibration(
    clean_features: np.ndarray,
    *,
    replicates: int,
    pairs: int,
    bootstrap_replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    fractions = DEFAULT_CALIBRATION_FRACTIONS
    out: dict[str, Any] = {
        "protocol": "controlled fault-injection sensitivity calibration",
        "replicates": replicates,
        "detection_target": 0.95,
        "production_null": "empirical independent-row Poisson-binomial Hamming null preserving observed bit marginals",
        "nominal_gohr_reference": "Binomial(64, 0.5) diagnostic only; not used for confirmatory detection",
        "calibration_sampling": {
            "pairs_per_test": pairs,
            "bootstrap_replicates": bootstrap_replicates,
            "lags": list(DEFAULT_LAGS),
            "near_duplicate_radii": list(NEAR_DUPLICATE_RADII),
            "tvd_threshold": TVD_THRESHOLD,
        },
        "duplicate_injection": {},
        "lag_copy_injection": {},
    }
    detector_dup = _calibration_detector(
        target_component="d2_1_near_duplicate_structure",
        pairs=pairs,
        bootstrap_replicates=bootstrap_replicates,
        alpha=alpha,
    )
    detector_lag = _calibration_detector(
        target_component="d2_2_serial_dependence",
        pairs=pairs,
        bootstrap_replicates=bootstrap_replicates,
        alpha=alpha,
    )
    for fraction in fractions:
        key = f"{fraction:.6g}"
        out["duplicate_injection"][key] = calibrate_fault(
            clean_features=clean_features,
            injector=lambda x, r, f=fraction: inject_duplicates(x, f, r),
            detector=detector_dup,
            replicates=replicates,
            seed=seed + int(fraction * 1_000_000) + 10,
        )
        out["duplicate_injection"][key]["target_component"] = "d2_1_near_duplicate_structure"
        out["lag_copy_injection"][key] = calibrate_fault(
            clean_features=clean_features,
            injector=lambda x, r, f=fraction: inject_lag_copy(x, f, 1, r),
            detector=detector_lag,
            replicates=replicates,
            seed=seed + int(fraction * 1_000_000) + 20,
        )
        out["lag_copy_injection"][key]["target_component"] = "d2_2_serial_dependence"
    return out


def main() -> None:
    args = parse_args()
    validate_args(args)

    adapter = GohrD2Adapter(num_rounds=5)
    partitions: dict[str, np.ndarray] = {}
    structured_views: dict[str, dict[str, Any]] = {}
    for name, count in (
        ("train", args.train_samples),
        ("validation", args.validation_samples),
        ("test", args.test_samples),
    ):
        print(f"Generating {name}: {count:,} samples...")
        x, y, views = adapter.generate_partition(count)
        partitions[name] = x
        for view_name, spec in views.items():
            structured_views[f"{name}:{view_name}"] = spec
        print(f"  shape={x.shape}, dtype={x.dtype}")

    reference = binomial_reference_distribution(adapter.FEATURE_BITS, 0.5)
    results, decision = run_d2(
        partitions=partitions,
        reference_pmf=reference,
        feature_bits=adapter.FEATURE_BITS,
        structured_views=structured_views,
        pairs_per_test=args.pairs_per_test,
        audit_replicates=args.audit_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        lags=DEFAULT_LAGS,
        near_duplicate_radii=NEAR_DUPLICATE_RADII,
        multivariate_partitions=("train", "validation", "test"),
        multivariate_pairs=args.multivariate_pairs,
        multivariate_permutations=args.multivariate_permutations,
        audit_seed=args.audit_seed,
    )

    # Calibration uses one clean partition and is intentionally recorded as a
    # sensitivity result, never as evidence that the clean dataset is faulty.
    if args.calibrate:
        print("Running controlled D2 sensitivity calibration...")
        results["d2_5_detection_calibration"] = run_calibration(
            partitions["train"],
            replicates=args.calibration_replicates,
            pairs=args.calibration_pairs,
            bootstrap_replicates=args.calibration_bootstrap_replicates,
            seed=args.audit_seed + 50_000,
            alpha=FAMILYWISE_ALPHA / familywise_test_count(
                partition_count=3,
                lag_count=3 * len(DEFAULT_LAGS),
                near_duplicate_radius_count=len(NEAR_DUPLICATE_RADII),
                structured_view_count=15,
                multivariate_count=3,
            ),
        )

    provenance = adapter.reference_specification()
    output_path = Path(args.output) if args.output else default_output_path(args)
    certificate = build_d2_certificate(
        results=results,
        decision=decision,
        partitions=partitions,
        dataset_id=adapter.DATASET_ID,
        dataset_version=adapter.DATASET_VERSION,
        generation_procedure="GohrAdapter.generate_partition -> speck.make_train_data",
        generation_parameters={
            "num_rounds": adapter.num_rounds,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "randomness_source": "os.urandom",
        },
        generation_random_seed=None,
        reference_description=provenance["pairwise_hamming_reference"],
        reference_model_description=provenance["structured_collision_reference"],
        audit_seed=args.audit_seed,
        output_path=str(output_path),
    )
    print_report(results, certificate)
    print(f"Certificate: {output_path}")


if __name__ == "__main__":
    main()
