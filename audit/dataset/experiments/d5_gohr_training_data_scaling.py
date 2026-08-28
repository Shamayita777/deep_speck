"""
Gohr D5 - Training Data Scaling Experiment.

This module is the Gohr-specific experiment driver for the generic
D5 Training Data Scaling Audit.

Dataset provenance
------------------
All datasets are generated exclusively through GohrAdapter.

GohrAdapter.generate_partition() delegates to:

    speck.make_train_data()

No alternate dataset generator is used.

Experimental design
-------------------
Training-set sizes:

    2,500,000
    5,000,000
    7,500,000
    10,000,000

For every condition:

    - Gohr's dataset generator is used;
    - the Gohr model architecture is used;
    - the same training procedure is used;
    - the same number of epochs is used;
    - a fixed held-out Gohr test partition is used;
    - the condition has its own resumable checkpoint.

The experiment is therefore a training-data-size scaling experiment,
not a label perturbation or synthetic-data experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_nets as tn

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

OUTPUT_DIRECTORY = Path(
    "audit/dataset/evidence/d5"
)

DATASET_ID = "gohr-speck"

DATASET_VERSION = (
    "original-make-train-data"
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
# Gohr adapter helpers
# ============================================================

def generate_gohr_partition(
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data exclusively through GohrAdapter.

    GohrAdapter delegates directly to speck.make_train_data().
    """

    adapter = GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=NUM_ROUNDS,
        depth=DEPTH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    return adapter.generate_partition(
        samples
    )


def generate_gohr_test_partition(
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the fixed held-out Gohr test partition.

    The seed argument is retained by the generic interface for
    provenance/stream separation. Gohr's original generator uses
    os.urandom internally, so this seed does not control the
    generator itself.
    """

    del seed

    return generate_gohr_partition(
        samples
    )


def generate_gohr_training_partition(
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one Gohr training partition.

    The seed is not passed to the Gohr generator because the
    original Gohr generator uses os.urandom internally.
    """

    del seed

    return generate_gohr_partition(
        samples
    )


# ============================================================
# Gohr model
# ============================================================

def make_gohr_model(
    seed: int,
):
    """
    Construct the Gohr neural distinguisher.

    This follows Gohr's existing model construction:

        tn.make_resnet(
            depth=10,
            reg_param=1e-5,
        )

    and the original Adam/MSE/accuracy compilation.
    """

    tn.set_seed(
        seed
    )

    model = tn.make_resnet(
        depth=DEPTH,
        reg_param=1e-5,
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# Gohr evaluation
# ============================================================

def evaluate_gohr_model(
    model,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[float, float]:
    """
    Evaluate a trained Gohr model on the fixed held-out test set.

    Returns
    -------
    test_loss, test_accuracy
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
        f"Dataset                    : {DATASET_ID}"
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
    # Run D5
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
        model_factory=make_gohr_model,
        evaluate_model=evaluate_gohr_model,
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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

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