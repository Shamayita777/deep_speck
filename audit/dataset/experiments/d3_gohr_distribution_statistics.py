"""
Gohr D3 Dataset Integrity Experiment.

This experiment generates Gohr-compatible Speck datasets through
the original Gohr data-generation function and passes the resulting
arrays to the generic D3 distribution audit.

The generic D3 implementation contains no Gohr-specific generation
logic.

Gohr reference
--------------
The generator used here is:

    speck.make_train_data(n, num_rounds)

The generator itself uses os.urandom(), so the generated arrays
constitute a concrete dataset instance for the current audit run.

Scientific scope
----------------
This experiment characterizes the generated train, validation, and
test partitions and reports pairwise consistency statistics.

D3 remains descriptive. A numerically close set of summary
statistics must not be interpreted as proof that the complete
distributions are identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import speck as sp

from ..d3_distribution_statistics import (
    audit_distribution,
    print_report,
)


DEFAULT_ROUNDS = 5

DEFAULT_TRAIN_SAMPLES = 10_000_000
DEFAULT_VALIDATION_SAMPLES = 1_000_000
DEFAULT_TEST_SAMPLES = 1_000_000

DEFAULT_OUTPUT = (
    "audit/dataset/evidence/d3/"
    "d3_gohr_10000000_1000000_1000000_5r.json"
)


def generate_gohr_partition(
    samples: int,
    num_rounds: int,
):
    """
    Generate one partition using Gohr's original generator.
    """

    if samples <= 0:
        raise ValueError(
            "samples must be positive."
        )

    if num_rounds <= 0:
        raise ValueError(
            "num_rounds must be positive."
        )

    return sp.make_train_data(
        samples,
        num_rounds,
    )


def generate_gohr_dataset(
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_rounds: int,
):
    """
    Generate the three audited partitions using Gohr's
    make_train_data() independently for each partition.
    """

    print("Generating training partition...")
    X_train, Y_train = generate_gohr_partition(
        train_samples,
        num_rounds,
    )

    print(
        f"  shape={X_train.shape}, "
        f"dtype={X_train.dtype}"
    )

    print("Generating validation partition...")
    X_validation, Y_validation = generate_gohr_partition(
        validation_samples,
        num_rounds,
    )

    print(
        f"  shape={X_validation.shape}, "
        f"dtype={X_validation.dtype}"
    )

    print("Generating test partition...")
    X_test, Y_test = generate_gohr_partition(
        test_samples,
        num_rounds,
    )

    print(
        f"  shape={X_test.shape}, "
        f"dtype={X_test.dtype}"
    )

    return (
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        X_test,
        Y_test,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run D3 distribution statistics on "
            "Gohr-generated Speck data."
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
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
    )

    return parser


def main() -> None:
    """
    Run the Gohr D3 experiment.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    print("=" * 72)
    print("Gohr D3 Dataset Integrity Experiment")
    print("=" * 72)
    print()

    print(
        f"Speck rounds         : {args.rounds}"
    )

    print(
        f"Training samples     : "
        f"{args.train_samples}"
    )

    print(
        f"Validation samples   : "
        f"{args.validation_samples}"
    )

    print(
        f"Test samples         : "
        f"{args.test_samples}"
    )

    print(
        "Generator             : "
        "speck.make_train_data"
    )

    print(
        "Generator randomness  : "
        "os.urandom"
    )

    print()

    (
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        X_test,
        Y_test,
    ) = generate_gohr_dataset(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        num_rounds=args.rounds,
    )

    print()
    print("=" * 72)
    print("Running D3 Distribution Audit")
    print("=" * 72)
    print()

    results = audit_distribution(
        train=X_train,
        train_labels=Y_train,
        validation=X_validation,
        validation_labels=Y_validation,
        test=X_test,
        test_labels=Y_test,
    )

    print_report(results)

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    certificate = {
        "audit": {
            "id": "D3",
            "name": (
                "Distribution Statistics and "
                "Partition Consistency"
            ),
            "claim": (
                "The supplied dataset partitions are "
                "characterized by the reported empirical "
                "distribution statistics and pairwise "
                "effect-size comparisons."
            ),
        },
        "decision": {
            "status": "DESCRIPTIVE_ONLY",
            "interpretation": (
                "D3 does not establish equality of the "
                "full underlying distributions."
            ),
        },
        "provenance": {
            "dataset": {
                "dataset_id": "gohr-speck",
                "dataset_version": (
                    "original-make-train-data"
                ),
                "generation_procedure": (
                    "speck.make_train_data(n, nr)"
                ),
                "generator": (
                    "speck.make_train_data"
                ),
                "num_rounds": args.rounds,
                "randomness_source": "os.urandom",
                "train_samples": args.train_samples,
                "validation_samples": (
                    args.validation_samples
                ),
                "test_samples": (
                    args.test_samples
                ),
                "input_difference": {
                    "left_word": "0x0040",
                    "right_word": "0x0000",
                },
            }
        },
        "methodology": {
            "partition_statistics": [
                "class balance",
                "global feature mean",
                "global feature variance",
                "representation-specific marginal statistics",
            ],
            "binary_statistics": [
                "per-feature bit density",
                "per-feature binary entropy",
                "mean absolute bit-density difference",
                "RMS bit-density difference",
                "maximum bit-density difference",
            ],
            "byte_statistics": [
                "byte histogram",
                "byte Shannon entropy",
                "byte total variation distance",
            ],
            "comparison_scope": (
                "all supplied partition pairs"
            ),
        },
        "limitations": [
            (
                "D3 is descriptive and does not constitute "
                "a formal proof of distributional equality."
            ),
            (
                "Aggregate moments and marginal statistics "
                "do not characterize higher-order dependencies "
                "or the complete joint distribution."
            ),
            (
                "The reported statistics describe the concrete "
                "generated dataset instance used in this run."
            ),
            (
                "The Gohr generator uses os.urandom(), so a "
                "subsequent generation run produces another "
                "random dataset instance."
            ),
        ],
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            certificate,
            file,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 72)
    print("D3 COMPLETE")
    print("=" * 72)
    print(
        "Status                : DESCRIPTIVE_ONLY"
    )
    print(
        f"Certificate           : {output_path}"
    )


if __name__ == "__main__":
    main()