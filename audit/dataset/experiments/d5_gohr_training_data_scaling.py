"""
Gohr D5 - Training Data Scaling Experiment.

Gohr-specific driver for the generic D5 Training Data Scaling Audit.

All Gohr dataset generation and model construction pass through
GohrAdapter. The generic D5 module contains the experiment
machinery, persistence, checkpointing, resumption, and reporting.

The experiment varies only training-set size:

    2,500,000
    5,000,000
    7,500,000
    10,000,000

For every condition:

    - Gohr's speck.make_train_data() is used;
    - GohrAdapter supplies the Gohr model;
    - Gohr's original training configuration is used;
    - the same fixed held-out test partition is used;
    - the generic D5 runner provides resumable checkpoints.

This is a training-data-size scaling experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit.dataset.adapters.gohr import GohrAdapter
from audit.dataset.d5_training_data_scaling import (
    generate_certificate,
    print_report,
    run_d5,
)


# ============================================================
# Gohr configuration
# ============================================================

NUM_ROUNDS = 5
DEPTH = 10

EPOCHS = 200
BATCH_SIZE = 5000

TEST_SAMPLES = 1_000_000

TRAINING_SIZES = (
    2_500_000,
    5_000_000,
    7_500_000,
    10_000_000,
)

DEFAULT_AUDIT_SEED = 0

DATASET_ID = "gohr-speck"

DATASET_VERSION = (
    "original-make-train-data"
)

OUTPUT_DIRECTORY = Path(
    "audit/dataset/evidence/d5"
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the Gohr training-data scaling audit."
        )
    )

    parser.add_argument(
        "--audit-seed",
        type=int,
        default=DEFAULT_AUDIT_SEED,
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=TEST_SAMPLES,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:

    if args.audit_seed < 0:
        raise ValueError(
            "--audit-seed must be non-negative."
        )

    if args.test_samples < 1:
        raise ValueError(
            "--test-samples must be >= 1."
        )

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1."
        )


# ============================================================
# Gohr adapter
# ============================================================

def make_adapter(
    *,
    seed: int,
) -> GohrAdapter:
    """
    Construct the dataset-specific Gohr adapter.

    No Gohr model or training logic is duplicated here.
    """

    return GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=NUM_ROUNDS,
        depth=DEPTH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=seed,
    )


# ============================================================
# Gohr dataset generation
# ============================================================

def generate_gohr_partition(
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data exclusively through GohrAdapter.

    GohrAdapter delegates directly to:

        speck.make_train_data()
    """

    adapter = make_adapter(
        seed=0,
    )

    return adapter.generate_partition(
        samples,
    )


def generate_gohr_test_partition(
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the fixed held-out Gohr test partition.

    Gohr's original generator uses os.urandom(), so the audit
    seed does not deterministically control generation.
    """

    del seed

    return generate_gohr_partition(
        samples,
    )


def generate_gohr_training_partition(
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one Gohr training partition.

    The seed is retained by the generic D5 interface for
    provenance, but is not passed to make_train_data(), because
    Gohr's generator uses os.urandom().
    """

    del seed

    return generate_gohr_partition(
        samples,
    )


# ============================================================
# Gohr model
# ============================================================

def make_gohr_model(
    seed: int,
):
    """
    Construct the Gohr model through GohrAdapter.

    This deliberately does not duplicate tn.make_resnet(),
    compilation, or other Gohr training logic here.
    """

    adapter = make_adapter(
        seed=seed,
    )

    return adapter.build_model()


def make_gohr_training_callbacks(
    seed: int,
):
    """
    Return Gohr's original training callbacks.

    The generic D5 runner supplies the checkpoint callback.
    The Gohr adapter supplies the Gohr-specific learning-rate
    schedule.
    """

    adapter = make_adapter(
        seed=seed,
    )

    return adapter.training_callbacks()


# ============================================================
# Gohr evaluation
# ============================================================

def evaluate_gohr_model(
    model,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[float, float]:
    """
    Evaluate a trained Gohr model on the fixed test partition.
    """

    loss, accuracy = model.evaluate(
        test_x,
        test_y,
        verbose=0,
    )

    return (
        float(loss),
        float(accuracy),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()
    validate_args(args)

    root = (
        OUTPUT_DIRECTORY
        / "training_data_scaling"
    )

    print("=" * 72)
    print("Gohr D5 - Training Data Scaling Audit")
    print("=" * 72)
    print()

    print(
        f"Dataset                    : "
        f"{DATASET_ID}"
    )

    print(
        f"Dataset version            : "
        f"{DATASET_VERSION}"
    )

    print(
        f"Speck rounds               : "
        f"{NUM_ROUNDS}"
    )

    print(
        f"ResNet depth               : "
        f"{DEPTH}"
    )

    print(
        f"Epochs                     : "
        f"{args.epochs}"
    )

    print(
        f"Batch size                 : "
        f"{args.batch_size}"
    )

    print(
        f"Fixed test samples         : "
        f"{args.test_samples:,}"
    )

    print(
        "Training sizes             : "
        + ", ".join(
            f"{size:,}"
            for size in TRAINING_SIZES
        )
    )

    print(
        "Generator                  : "
        "speck.make_train_data"
    )

    print(
        "Generator randomness       : "
        "os.urandom"
    )

    print()

    # --------------------------------------------------------
    # Run generic D5 machinery
    # --------------------------------------------------------

    result = run_d5(
        root=root,
        training_sizes=TRAINING_SIZES,
        test_samples=args.test_samples,
        audit_seed=args.audit_seed,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        train_dataset_factory=(
            generate_gohr_training_partition
        ),
        test_dataset_factory=(
            generate_gohr_test_partition
        ),
        model_factory=(
            make_gohr_model
        ),
        training_callbacks_factory=(
            make_gohr_training_callbacks
        ),
        evaluate_model=(
            evaluate_gohr_model
        ),
        experiment_name=(
            "Gohr Speck training-data scaling"
        ),
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "generation_procedure": (
            "speck.make_train_data(n, nr)"
        ),
        "generation_parameters": {
            "generator": (
                "speck.make_train_data"
            ),
            "num_rounds": NUM_ROUNDS,
            "depth": DEPTH,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "training_sizes": list(
                TRAINING_SIZES
            ),
            "test_samples": args.test_samples,
            "randomness_source": (
                "os.urandom"
            ),
            "deterministic_seed": None,
            "input_difference": {
                "left_word": "0x0040",
                "right_word": "0x0000",
            },
        },
    }

    certificate = generate_certificate(
        result,
        dataset_id=provenance["dataset_id"],
        dataset_version=provenance["dataset_version"],
        generation_procedure=(
            provenance["generation_procedure"]
        ),
        generation_parameters=(
            provenance["generation_parameters"]
        ),
        audit_seed=args.audit_seed,
    )

    print()

    print_report(
        result
    )

    # --------------------------------------------------------
    # Save certificate
    # --------------------------------------------------------

    output_path = (
        Path(args.output)
        if args.output is not None
        else (
            OUTPUT_DIRECTORY
            / (
                "d5_gohr_training_data_scaling_"
                f"seed{args.audit_seed}.json"
            )
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            certificate,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print()

    print("=" * 72)
    print("D5 COMPLETE")
    print("=" * 72)

    print(
        f"Certificate                : "
        f"{output_path}"
    )

    print(
        "Interpretation             : "
        "training-data-size scaling "
        "under the specified Gohr "
        "dataset/model/training procedure"
    )


if __name__ == "__main__":
    main()