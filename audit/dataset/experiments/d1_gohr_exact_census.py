"""
Gohr Dataset Integrity Audit D1
Exact Duplicate and Partition Overlap Census.

This is the Gohr-specific experiment driver for the generic
D1 Dataset Integrity audit.

Architecture
------------
Generic D1 methodology:
    audit.dataset.d1_duplicate_detection

Gohr-specific dataset generation:
    audit.dataset.adapters.gohr.GohrAdapter

This module is responsible only for:

    1. configuring the Gohr case study;
    2. generating Gohr partitions through GohrAdapter;
    3. invoking the generic D1 audit;
    4. recording Gohr-specific provenance;
    5. writing the resulting D1 certificate.

Scientific scope
----------------
D1 evaluates the concrete generated dataset instance for:

    - exact feature duplicates within partitions;
    - exact feature overlap across partitions;
    - optional exact-feature label conflicts.

D1 does NOT establish:

    - absence of near duplicates;
    - statistical independence;
    - absence of generation-order effects;
    - absence of metadata leakage;
    - distributional equivalence;
    - validity of the neural distinguisher.

Randomness
----------
Gohr's make_train_data() uses os.urandom() internally.

Therefore this experiment does not claim deterministic
reproduction from a numerical seed.

The certificate explicitly records the randomness source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from audit.dataset.adapters.gohr import GohrAdapter

from audit.dataset.d1_duplicate_detection import (
    audit_duplicates,
    build_d1_certificate,
    print_report,
)


# ============================================================
# Gohr configuration
# ============================================================

NUM_ROUNDS = 5

DEFAULT_TRAIN_SAMPLES = 10**7
DEFAULT_VALIDATION_SAMPLES = 10**6
DEFAULT_TEST_SAMPLES = 10**6

DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"

RANDOMNESS_SOURCE = "os.urandom"

OUTPUT_DIRECTORY = Path(
    "audit/dataset/evidence/d1"
)


# ============================================================
# Dataset generation
# ============================================================

def generate_gohr_dataset(
    *,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_rounds: int,
) -> Dict[str, Any]:
    """
    Generate the Gohr dataset partitions through GohrAdapter.

    No Gohr/Speck generation logic is implemented in this module.
    The adapter is the single dataset-specific generation boundary.
    """

    adapter = GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=num_rounds,
    )

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

    train_x, train_y = adapter.generate_partition(
        train_samples,
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

    validation_x, validation_y = (
        adapter.generate_partition(
            validation_samples,
            num_rounds=num_rounds,
        )
    )

    print(
        f"  shape={validation_x.shape}, "
        f"dtype={validation_x.dtype}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("Generating test partition...")

    test_x, test_y = adapter.generate_partition(
        test_samples,
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
    Build the Gohr generation configuration recorded in
    the D1 certificate.

    This describes the generation procedure used to construct
    the audited dataset instance. It does not claim deterministic
    replay because Gohr uses os.urandom().
    """

    return {
        "generator": (
            "speck.make_train_data"
        ),
        "adapter": (
            "audit.dataset.adapters.gohr.GohrAdapter"
        ),
        "num_rounds": int(num_rounds),
        "train_samples": int(train_samples),
        "validation_samples": int(
            validation_samples
        ),
        "test_samples": int(test_samples),
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
    Generate one concrete Gohr dataset instance and execute
    the generic D1 exact census.
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
        validation_labels=dataset[
            "validation_y"
        ],
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
        generation_parameters=(
            generation_parameters
        ),
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
            "Gohr-generated Speck dataset through "
            "GohrAdapter."
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
            "If omitted, a filename is generated from "
            "the dataset configuration."
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