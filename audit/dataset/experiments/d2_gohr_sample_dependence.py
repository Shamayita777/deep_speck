"""Gohr/Speck driver for the generic D2 sample-dependence audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from audit.dataset.adapters.gohr_d2 import GohrD2Adapter
from audit.dataset.d2_sample_dependence import (
    CALIBRATION_DETECTION_TARGET,
    CALIBRATION_MIN_REPLICATES,
    DEFAULT_AUDIT_REPLICATES,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_MULTIVARIATE_PAIRS,
    DEFAULT_MULTIVARIATE_PERMUTATIONS,
    DEFAULT_NULL_PERMUTATIONS,
    DEFAULT_NULL_REFERENCE_PERMUTATIONS,
    DEFAULT_PAIRS_PER_TEST,
    DEFAULT_LAGS,
    FAMILYWISE_ALPHA,
    NEAR_DUPLICATE_RADII,
    PRACTICAL_EXCESS_TVD_THRESHOLD,
    binomial_reference_distribution,
    build_d2_certificate,
    calibrate_fault,
    familywise_test_count,
    inject_lag_copy,
    inject_near_duplicates,
    multivariate_pair_discrimination,
    near_neighbor_summary,
    print_report,
    run_d2,
    summarize_randomization_test,
    _null_pair_histogram,
    _sample_disjoint_lag_pairs,
    hamming_distances,
)

DEFAULT_TRAIN_SAMPLES = 10_000_000
DEFAULT_VALIDATION_SAMPLES = 1_000_000
DEFAULT_TEST_SAMPLES = 1_000_000
DEFAULT_CALIBRATION_REPLICATES = 100
DEFAULT_CALIBRATION_FRACTIONS = (0.001, 0.005, 0.01, 0.025, 0.05)
DEFAULT_CALIBRATION_QUERY_COUNT = 2048
DEFAULT_CALIBRATION_TABLES = 8
DEFAULT_CALIBRATION_NULL_PAIRS = 100_000


def parse_args():
    p = argparse.ArgumentParser(description="Run Gohr D2 Sample Dependence Audit")
    p.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    p.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    p.add_argument("--pairs-per-test", type=int, default=DEFAULT_PAIRS_PER_TEST)
    p.add_argument("--audit-replicates", type=int, default=DEFAULT_AUDIT_REPLICATES)
    p.add_argument("--null-permutations", type=int, default=DEFAULT_NULL_PERMUTATIONS, help="Maximum complete-row randomization draws including the separate reference draw; adaptive stopping may use fewer.")
    p.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    p.add_argument("--multivariate-pairs", type=int, default=DEFAULT_MULTIVARIATE_PAIRS)
    p.add_argument("--multivariate-permutations", type=int, default=DEFAULT_MULTIVARIATE_PERMUTATIONS)
    p.add_argument("--audit-seed", type=int, default=0)
    p.add_argument("--near-duplicate-query-count", type=int, default=2048)
    p.add_argument("--near-duplicate-projection-tables", type=int, default=8)
    p.add_argument("--near-duplicate-null-pairs", type=int, default=DEFAULT_CALIBRATION_NULL_PAIRS)
    p.add_argument(
        "--practical-excess-tvd-threshold",
        type=float,
        default=PRACTICAL_EXCESS_TVD_THRESHOLD,
        help="Pre-specified practical-effect threshold for null-centered excess TVD (default: 0.01).",
    )
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibration-replicates", type=int, default=DEFAULT_CALIBRATION_REPLICATES)
    p.add_argument("--calibration-query-count", type=int, default=DEFAULT_CALIBRATION_QUERY_COUNT)
    p.add_argument("--calibration-projection-tables", type=int, default=DEFAULT_CALIBRATION_TABLES)
    p.add_argument("--output", default=None)
    return p.parse_args()


def validate_args(a):
    positive = (
        "train_samples", "validation_samples", "test_samples", "pairs_per_test",
        "audit_replicates", "null_permutations", "bootstrap_replicates",
        "multivariate_pairs", "multivariate_permutations", "near_duplicate_query_count",
        "near_duplicate_projection_tables", "near_duplicate_null_pairs",
        "calibration_replicates", "calibration_query_count", "calibration_projection_tables",
    )
    for name in positive:
        if getattr(a, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if min(a.train_samples, a.validation_samples, a.test_samples) < 2:
        raise ValueError("Partitions must have >=2 rows")
    if a.null_permutations < 100 or a.bootstrap_replicates < 100:
        raise ValueError("--null-permutations and --bootstrap-replicates must be >=100")
    if a.calibrate and a.calibration_replicates < CALIBRATION_MIN_REPLICATES:
        raise ValueError(f"Calibration requires >= {CALIBRATION_MIN_REPLICATES} independent trials")
    if a.audit_seed < 0:
        raise ValueError("--audit-seed must be >=0")
    if not 0 <= a.practical_excess_tvd_threshold <= 1:
        raise ValueError("--practical-excess-tvd-threshold must be in [0, 1]")


def default_output_path(a):
    return Path("audit/dataset/evidence/d2") / (
        f"d2_gohr_sample_dependence_{a.train_samples}_{a.validation_samples}_{a.test_samples}_"
        f"{a.pairs_per_test}pairs_{a.audit_replicates}replicates_seed{a.audit_seed}.json"
    )


def _near_detector(radius, query_count, tables, null_pairs, alpha):
    def detector(x, rng):
        result = near_neighbor_summary(
            x,
            (radius,),
            rng=rng,
            query_count=query_count,
            projection_tables=tables,
            null_pairs=null_pairs,
            alpha=alpha,
        )[str(radius)]
        return {"detected": bool(result["practically_excessive"] or result["statistically_excessive"]), "result": result}
    return detector


def _lag_detector(query_pairs, null_permutations, alpha, practical_excess_tvd_threshold):
    def detector(x, rng):
        lag = 1
        count = min(query_pairs, (len(x) - lag) // 2)
        i, j = _sample_disjoint_lag_pairs(len(x), lag, count, rng)
        left, right = x[i], x[j]
        observed = hamming_distances(left, right)
        null_pmf, null_stats, null_meta = _null_pair_histogram(
        left,
        right,
        permutations=null_permutations,
        rng=rng,
        within=False,
        reference_permutations=min(DEFAULT_NULL_REFERENCE_PERMUTATIONS, max(1, null_permutations - 1)),
        alpha=alpha,
        observed_distances=observed,
    )
        result = summarize_randomization_test(
            observed,
            null_pmf,
            null_stats,
            alpha=alpha,
            rng=rng,
            bootstrap_replicates=100,
            practical_excess_tvd_threshold=practical_excess_tvd_threshold,
            null_sampling_metadata=null_meta,
        )
        return {"detected": bool(result["practically_excessive"] or result["randomization_test"]["statistically_excessive"]), "result": result}
    return detector


def run_calibration(clean, *, replicates, seed, query_count, tables, null_pairs, pairs, null_permutations, alpha, practical_excess_tvd_threshold):
    out = {
        "protocol": "component-specific controlled fault-injection sensitivity calibration",
        "replicates": int(replicates),
        "detection_target": CALIBRATION_DETECTION_TARGET,
        "target_semantics": "Wilson lower bound is the only confidence-supported target criterion; point estimate is reported separately",
        "calibration_is_not_clean_evidence": True,
        "near_duplicate_injection": {},
        "lag_copy_injection": {},
    }

    for radius in NEAR_DUPLICATE_RADII:
        out["near_duplicate_injection"][str(radius)] = {}
        for fraction in DEFAULT_CALIBRATION_FRACTIONS:
            seed_here = seed + 10_000 + radius * 100 + int(fraction * 1_000_000)
            result = calibrate_fault(
                clean_features=clean,
                injector=lambda x, r, f=fraction, rad=radius: inject_near_duplicates(x, f, rad, r),
                detector=_near_detector(radius, query_count, tables, null_pairs, alpha),
                replicates=replicates,
                seed=seed_here,
            )
            result["target_component"] = "d2_1_near_duplicate_structure"
            out["near_duplicate_injection"][str(radius)][f"{fraction:.6g}"] = result

    for fraction in DEFAULT_CALIBRATION_FRACTIONS:
        seed_here = seed + 50_000 + int(fraction * 1_000_000)
        result = calibrate_fault(
            clean_features=clean,
            injector=lambda x, r, f=fraction: inject_lag_copy(x, f, 1, r),
            detector=_lag_detector(pairs, null_permutations, alpha, practical_excess_tvd_threshold),
            replicates=replicates,
            seed=seed_here,
        )
        result["target_component"] = "d2_2_serial_dependence"
        out["lag_copy_injection"][f"{fraction:.6g}"] = result
    return out


def main():
    a = parse_args()
    validate_args(a)
    adapter = GohrD2Adapter(num_rounds=5)
    parts, views = {}, {}

    for name, n in (
        ("train", a.train_samples),
        ("validation", a.validation_samples),
        ("test", a.test_samples),
    ):
        print(f"Generating {name}: {n:,} samples...")
        x, y, v = adapter.generate_partition(n)
        parts[name] = x
        for view_name, spec in v.items():
            views[f"{name}:{view_name}"] = spec
        print(f"  shape={x.shape}, dtype={x.dtype}, labels={y.shape}")

    nominal_reference = binomial_reference_distribution(adapter.FEATURE_BITS, 0.5)
    results, decision = run_d2(
        partitions=parts,
        feature_bits=adapter.FEATURE_BITS,
        reference_pmf=nominal_reference,
        structured_views=views,
        pairs_per_test=a.pairs_per_test,
        audit_replicates=a.audit_replicates,
        null_permutations=a.null_permutations,
        bootstrap_replicates=a.bootstrap_replicates,
        lags=DEFAULT_LAGS,
        near_duplicate_radii=NEAR_DUPLICATE_RADII,
        multivariate_partitions=("train", "validation", "test"),
        multivariate_pairs=a.multivariate_pairs,
        multivariate_permutations=a.multivariate_permutations,
        audit_seed=a.audit_seed,
        near_duplicate_query_count=a.near_duplicate_query_count,
        near_duplicate_projection_tables=a.near_duplicate_projection_tables,
        near_duplicate_null_pairs=a.near_duplicate_null_pairs,
        practical_excess_tvd_threshold=a.practical_excess_tvd_threshold,
    )

    if a.calibrate:
        print("Running controlled D2 sensitivity calibration...")
        total = familywise_test_count(
            partition_count=3,
            lag_count=3 * len(DEFAULT_LAGS),
            near_duplicate_radius_count=len(NEAR_DUPLICATE_RADII),
            structured_view_count=15,
            multivariate_count=3,
        )
        results["d2_5_detection_calibration"] = run_calibration(
            parts["train"],
            replicates=a.calibration_replicates,
            seed=a.audit_seed + 50_000,
            query_count=a.calibration_query_count,
            tables=a.calibration_projection_tables,
            null_pairs=a.near_duplicate_null_pairs,
            pairs=a.pairs_per_test,
            null_permutations=a.null_permutations,
            alpha=FAMILYWISE_ALPHA / total,
            practical_excess_tvd_threshold=a.practical_excess_tvd_threshold,
        )
    else:
        results["d2_5_detection_calibration"] = {
            "status": "NOT_RUN",
            "reason": "Controlled fault-injection calibration was not requested; no empirical sensitivity claim is made.",
        }

    out = Path(a.output) if a.output else default_output_path(a)
    spec = adapter.reference_specification()
    cert = build_d2_certificate(
        results=results,
        decision=decision,
        partitions=parts,
        dataset_id=adapter.DATASET_ID,
        dataset_version=adapter.DATASET_VERSION,
        generation_procedure="GohrAdapter.generate_partition -> speck.make_train_data",
        generation_parameters={
            "num_rounds": adapter.num_rounds,
            "train_samples": a.train_samples,
            "validation_samples": a.validation_samples,
            "test_samples": a.test_samples,
            "randomness_source": "os.urandom",
            "exact_replay_available": False,
            "practical_excess_tvd_threshold": a.practical_excess_tvd_threshold,
        },
        generation_random_seed=None,
        reference_description=spec["pairwise_hamming_reference"],
        reference_model_description=spec["structured_collision_reference"],
        audit_seed=a.audit_seed,
        output_path=str(out),
    )
    print_report(results, cert)
    print(f"Certificate: {out}")


if __name__ == "__main__":
    main()
