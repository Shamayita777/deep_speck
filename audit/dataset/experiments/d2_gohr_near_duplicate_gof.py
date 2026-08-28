"""
Gohr D2 — Pairwise Hamming-Distance Goodness-of-Fit Experiment.

Dataset-specific experiment driver.

Responsibilities of this module:
    1. instantiate the Gohr adapter;
    2. generate the concrete Gohr partitions;
    3. construct the Gohr/Speck reference distribution;
    4. call the generic D2 audit;
    5. construct provenance and certificate output.

Responsibilities deliberately NOT owned here:
    - pair sampling;
    - chi-square calculation;
    - TVD calculation;
    - bootstrap calculation;
    - replicate aggregation;
    - D2 decision logic.

Those belong to audit.dataset.d2_near_duplicate_gof.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audit.dataset.adapters.gohr import GohrAdapter
from audit.dataset.d2_near_duplicate_gof import (
    DEFAULT_AUDIT_REPLICATES,
    DEFAULT_PAIRS_PER_COMPARISON,
    DEFAULT_TVD_BOOTSTRAP_REPLICATES,
    binomial_reference_distribution,
    build_d2_certificate,
    print_report,
    run_d2,
)


# ============================================================
# Gohr case-study configuration
# ============================================================

NUM_ROUNDS = 5
FEATURE_BITS = 64

DEFAULT_TRAIN_SAMPLES = 10**6
DEFAULT_VALIDATION_SAMPLES = 10**6
DEFAULT_TEST_SAMPLES = 10**6


def parse_args() -> argparse.Namespace:
    """Parse experiment configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the generic D2 audit on Gohr/Speck dataset "
            "partitions generated through GohrAdapter."
        )
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=DEFAULT_VALIDATION_SAMPLES,
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
    )

    parser.add_argument(
        "--pairs-per-comparison",
        type=int,
        default=DEFAULT_PAIRS_PER_COMPARISON,
    )

    parser.add_argument(
        "--audit-replicates",
        type=int,
        default=DEFAULT_AUDIT_REPLICATES,
        help=(
            "Number of independent pair-sampling replicates "
            "per comparison."
        ),
    )

    parser.add_argument(
        "--tvd-bootstrap-replicates",
        type=int,
        default=DEFAULT_TVD_BOOTSTRAP_REPLICATES,
        help=(
            "Number of multinomial bootstrap replicates used "
            "for each empirical TVD uncertainty interval."
        ),
    )

    parser.add_argument(
        "--audit-seed",
        type=int,
        default=0,
        help=(
            "Seed for D2 pair sampling and bootstrap streams. "
            "It does not seed Gohr dataset generation."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output certificate path.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line parameters."""

    if args.train_samples < 2:
        raise ValueError("--train-samples must be >= 2.")

    if args.validation_samples < 2:
        raise ValueError("--validation-samples must be >= 2.")

    if args.test_samples < 2:
        raise ValueError("--test-samples must be >= 2.")

    if args.pairs_per_comparison < 1:
        raise ValueError("--pairs-per-comparison must be >= 1.")

    if args.audit_replicates < 1:
        raise ValueError("--audit-replicates must be >= 1.")

    if args.tvd_bootstrap_replicates < 100:
        raise ValueError(
            "--tvd-bootstrap-replicates must be >= 100."
        )


def default_output_path(args) -> Path:
    """Return a collision-resistant default certificate path."""

    filename = (
        f"d2_gohr_"
        f"{args.train_samples}_"
        f"{args.validation_samples}_"
        f"{args.test_samples}_"
        f"5r_"
        f"{args.pairs_per_comparison}pairs_"
        f"{args.audit_replicates}replicates_"
        f"{args.tvd_bootstrap_replicates}bootstrap_"
        f"seed{args.audit_seed}.json"
    )

    return (
        Path("audit/dataset/evidence/d2")
        / filename
    )


def main() -> None:
    """Run the Gohr D2 case study."""

    args = parse_args()
    validate_args(args)

    adapter = GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=NUM_ROUNDS,
    )

    print("=" * 72)
    print("Gohr D2 Dataset Integrity Experiment")
    print("=" * 72)
    print()
    print(f"Speck rounds         : {NUM_ROUNDS}")
    print(f"Feature bits         : {FEATURE_BITS}")
    print(f"Training samples     : {args.train_samples}")
    print(f"Validation samples   : {args.validation_samples}")
    print(f"Test samples         : {args.test_samples}")
    print(f"Pairs/comparison     : {args.pairs_per_comparison}")
    print(f"Audit replicates     : {args.audit_replicates}")
    print(
        f"TVD bootstrap reps   : "
        f"{args.tvd_bootstrap_replicates}"
    )
    print(f"D2 audit seed        : {args.audit_seed}")
    print("Generator randomness : os.urandom")
    print()

    print("Generating training partition...")
    train_x, _train_y = adapter.generate_partition(
        args.train_samples
    )
    print(
        f"  shape={train_x.shape}, "
        f"dtype={train_x.dtype}"
    )

    print("Generating validation partition...")
    validation_x, _validation_y = adapter.generate_partition(
        args.validation_samples
    )
    print(
        f"  shape={validation_x.shape}, "
        f"dtype={validation_x.dtype}"
    )

    print("Generating test partition...")
    test_x, _test_y = adapter.generate_partition(
        args.test_samples
    )
    print(
        f"  shape={test_x.shape}, "
        f"dtype={test_x.dtype}"
    )
    print()

    reference_pmf = binomial_reference_distribution(
        n_bits=FEATURE_BITS,
        p=0.5,
    )

    comparisons, decision = run_d2(
        partitions={
            "train": train_x,
            "validation": validation_x,
            "test": test_x,
        },
        reference_pmf=reference_pmf,
        feature_bits=FEATURE_BITS,
        pairs_per_comparison=args.pairs_per_comparison,
        audit_replicates=args.audit_replicates,
        tvd_bootstrap_replicates=args.tvd_bootstrap_replicates,
        audit_seed=args.audit_seed,
    )

    provenance = adapter.dataset_provenance(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
    )

    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(args)
    )

    certificate = build_d2_certificate(
        comparisons=comparisons,
        decision=decision,
        partitions={
            "train": train_x,
            "validation": validation_x,
            "test": test_x,
        },
        dataset_id=provenance["dataset_id"],
        dataset_version=provenance["dataset_version"],
        generation_procedure=provenance["generation_procedure"],
        generation_parameters=provenance[
            "generation_parameters"
        ],
        generation_random_seed=provenance[
            "generation_random_seed"
        ],
        feature_bits=FEATURE_BITS,
        reference_description="Binomial(64, 0.5)",
        reference_model_description=(
            "Independent random binary feature vectors"
        ),
        pairs_per_comparison=args.pairs_per_comparison,
        audit_replicates=args.audit_replicates,
        tvd_bootstrap_replicates=args.tvd_bootstrap_replicates,
        audit_seed=args.audit_seed,
        output_path=output_path,
    )

    print()
    print_report(comparisons, certificate)

    print()
    print("=" * 72)
    print("D2 COMPLETE")
    print("=" * 72)
    print(f"Outcome     : {certificate['decision']['outcome']}")
    print(f"Certificate : {output_path}")
    print(
        "Scope       : pairwise Hamming-distance structure under "
        "the supplied Binomial(64, 0.5) reference"
    )


if __name__ == "__main__":
    main()