"""
Gohr Dataset Integrity Audit D1
Exact Duplicate and Partition Overlap Census.

This module applies the generic D1 audit to datasets generated
through Gohr's original Speck training-data generator.

The runner is intentionally separate from:

    audit/dataset/d1_duplicate_detection.py

which contains the generic D1 methodology.

This module is responsible only for:

    1. generating the Gohr dataset partitions;
    2. invoking the generic D1 audit;
    3. recording Gohr-specific provenance;
    4. writing the resulting D1 certificate.

Scientific scope
----------------
The audit evaluates the concrete generated dataset instance
for exact feature duplication and exact cross-partition feature
overlap.

It does NOT claim:

    - absence of near duplicates;
    - statistical independence;
    - absence of generation-order effects;
    - absence of metadata leakage;
    - distributional integrity;
    - validity of the neural distinguisher itself.

Those properties require separate audits.

Randomness
----------
Gohr's make_train_data() uses os.urandom() internally.

Therefore this runner does NOT claim that a numerical random seed
reproduces the generated dataset. The certificate explicitly
records the randomness source as OS-level randomness.

Dataset protocol
----------------
The historical Gohr training pipeline generates:

    training   : 10^7 samples
    evaluation : 10^6 samples

The audit framework additionally supports an independent test
partition of 10^6 samples, matching the three-partition protocol
already used by the controlled perturbation experiments.

For smoke testing, much smaller partitions can be requested.

Examples
--------
Smoke test:

    python -m audit.dataset.experiments.d1_gohr_exact_census \
        --train-samples 1000 \
        --validation-samples 1000 \
        --test-samples 1000

Full audit:

    python -m audit.dataset.experiments.d1_gohr_exact_census

The default full audit uses:

    train      = 10^7
    validation = 10^6
    test       = 10^6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import speck as sp

from audit.dataset.d1_duplicate_detection import (
    audit_duplicates,
    build_d1_certificate,
    print_report,
)


# ============================================================
# Gohr dataset configuration
# ============================================================

NUM_ROUNDS = 5

DEFAULT_TRAIN_SAMPLES = 10 ** 7
DEFAULT_VALIDATION_SAMPLES = 10 ** 6
DEFAULT_TEST_SAMPLES = 10 ** 6

DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"

RANDOMNESS_SOURCE = "os.urandom"

OUTPUT_DIRECTORY = Path(
    "audit/dataset/evidence/d1"
)


# ============================================================
# Dataset generation
# ============================================================

def generate_partition(
    *,
    samples: int,
    num_rounds: int,
):
    """
    Generate one partition using the existing Gohr generator.

    No alternate implementation of make_train_data() is used.

    Returns
    -------
    tuple
        (features, labels)
    """

    return sp.make_train_data(
        samples,
        num_rounds,
    )


def generate_gohr_dataset(
    *,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_rounds: int,
) -> Dict[str, Any]:
    """
    Generate the train, validation, and test partitions using
    Gohr's existing make_train_data() implementation.

    Each invocation of make_train_data() independently generates
    a partition using os.urandom().
    """

    print("=" * 64)
    print("Gohr D1 Dataset Generation")
    print("=" * 64)
    print()

    print(
        f"Speck rounds       : {num_rounds}"
    )

    print(
        f"Training samples   : {train_samples}"
    )

    print(
        f"Validation samples : {validation_samples}"
    )

    print(
        f"Test samples       : {test_samples}"
    )

    print(
        f"Randomness source  : {RANDOMNESS_SOURCE}"
    )

    print()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("Generating training partition...")

    train_x, train_y = generate_partition(
        samples=train_samples,
        num_rounds=num_rounds,
    )

    print(
        f"  shape={train_x.shape}, "
        f"dtype={train_x.dtype}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("Generating validation partition...")

    validation_x, validation_y = generate_partition(
        samples=validation_samples,
        num_rounds=num_rounds,
    )

    print(
        f"  shape={validation_x.shape}, "
        f"dtype={validation_x.dtype}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("Generating test partition...")

    test_x, test_y = generate_partition(
        samples=test_samples,
        num_rounds=num_rounds,
    )

    print(
        f"  shape={test_x.shape}, "
        f"dtype={test_x.dtype}"
    )

    print()

    return {
        "train_x": train_x,
        "train_y": train_y,
        "validation_x": validation_x,
        "validation_y": validation_y,
        "test_x": test_x,
        "test_y": test_y,
    }


# ============================================================
# Certificate metadata
# ============================================================

def build_generation_parameters(
    *,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_rounds: int,
) -> Dict[str, Any]:
    """
    Build the generation configuration recorded in the D1
    certificate.

    This describes the procedure used to construct the audited
    dataset instance; it does not claim deterministic replay.
    """

    return {
        "generator": (
            "speck.make_train_data"
        ),
        "num_rounds": num_rounds,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "input_difference": {
            "left_word": "0x0040",
            "right_word": "0x0000",
        },
        "randomness_source": RANDOMNESS_SOURCE,
        "deterministic_seed": None,
    }


# ============================================================
# D1 execution
# ============================================================

def run_d1(
    *,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_rounds: int,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Generate the Gohr dataset instance and execute D1.
    """

    dataset = generate_gohr_dataset(
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        num_rounds=num_rounds,
    )

    print("=" * 64)
    print("Running D1 Exact Census")
    print("=" * 64)
    print()

    results = audit_duplicates(
        train=dataset["train_x"],
        validation=dataset["validation_x"],
        test=dataset["test_x"],
        train_labels=dataset["train_y"],
        validation_labels=dataset["validation_y"],
        test_labels=dataset["test_y"],
    )

    generation_parameters = (
        build_generation_parameters(
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            num_rounds=num_rounds,
        )
    )

    certificate = build_d1_certificate(
        results,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        generation_procedure=(
            "speck.make_train_data(n, nr)"
        ),
        generation_parameters=generation_parameters,
        random_seed=None,
        output_path=str(output_path),
    )

    print_report(
        results,
        certificate,
    )

    return certificate


# ============================================================
# Command-line interface
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Dataset Integrity Audit D1 on a "
            "Gohr-generated Speck dataset."
        )
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help=(
            "Number of training samples. "
            f"Default: {DEFAULT_TRAIN_SAMPLES}"
        ),
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=DEFAULT_VALIDATION_SAMPLES,
        help=(
            "Number of validation samples. "
            f"Default: {DEFAULT_VALIDATION_SAMPLES}"
        ),
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help=(
            "Number of test samples. "
            f"Default: {DEFAULT_TEST_SAMPLES}"
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help=(
            "Number of Speck rounds. "
            f"Default: {NUM_ROUNDS}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path for the D1 certificate. "
            "If omitted, a filename is generated from the "
            "dataset configuration."
        ),
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.train_samples <= 0:
        parser.error(
            "--train-samples must be greater than zero."
        )

    if args.validation_samples <= 0:
        parser.error(
            "--validation-samples must be greater than zero."
        )

    if args.test_samples <= 0:
        parser.error(
            "--test-samples must be greater than zero."
        )

    if args.rounds <= 0:
        parser.error(
            "--rounds must be greater than zero."
        )

    # --------------------------------------------------------
    # Determine output path
    # --------------------------------------------------------

    if args.output is None:

        output_name = (
            f"d1_gohr_"
            f"{args.train_samples}_"
            f"{args.validation_samples}_"
            f"{args.test_samples}_"
            f"{args.rounds}r.json"
        )

        output_path = (
            OUTPUT_DIRECTORY / output_name
        )

    else:

        output_path = args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Run audit
    # --------------------------------------------------------

    certificate = run_d1(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        num_rounds=args.rounds,
        output_path=output_path,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("D1 COMPLETE")
    print("=" * 64)
    print(
        f"Outcome              : "
        f"{certificate['decision']['outcome']}"
    )
    print(
        f"Certificate          : "
        f"{output_path}"
    )
    print(
        f"Randomness           : "
        f"{RANDOMNESS_SOURCE}"
    )
    print()


if __name__ == "__main__":
    main()