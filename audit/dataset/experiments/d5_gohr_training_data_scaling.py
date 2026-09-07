
"""
Gohr D5 — strengthened training-data scaling experiment.

Design:
    - 2.5M, 5M, 7.5M, 10M training sizes;
    - default 5 independent replicates;
    - one independently generated 10M dataset per replicate;
    - nested prefixes form the four size conditions within each replicate;
    - the same model seed is reused across sizes within a replicate;
    - independent replicate seeds quantify dataset/model stochasticity;
    - one fixed 1M held-out test partition is shared by every condition;
    - 200 epochs and batch size 5000 are fixed;
    - resumable per-condition checkpoints and immutable run manifest.
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
    write_learning_curve_plot,
)

NUM_ROUNDS = 5
DEPTH = 10
EPOCHS = 200
BATCH_SIZE = 5000
TEST_SAMPLES = 1_000_000
TRAINING_SIZES = (2_500_000, 5_000_000, 7_500_000, 10_000_000)
REPLICATES = 5
BOOTSTRAP_REPLICATES = 5000
DEFAULT_AUDIT_SEED = 0

DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"
OUTPUT_DIRECTORY = Path("audit/dataset/evidence/d5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the strengthened Gohr D5 scaling audit.")
    p.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    p.add_argument("--replicates", type=int, default=REPLICATES)
    p.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    p.add_argument("--test-samples", type=int, default=TEST_SAMPLES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.audit_seed < 0:
        raise ValueError("--audit-seed must be non-negative.")
    if args.replicates < 2:
        raise ValueError("--replicates must be >= 2.")
    if args.bootstrap_replicates < 1000:
        raise ValueError("--bootstrap-replicates must be >= 1000.")
    if args.test_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("test-samples, epochs and batch-size must be positive.")


def make_adapter(seed: int) -> GohrAdapter:
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


def generate_gohr_partition(samples: int, seed: int):
    # Gohr's original generator uses os.urandom(); seed is retained only
    # as provenance for the independent replicate/model RNG stream.
    del seed
    return make_adapter(0).generate_partition(samples)


def generate_gohr_test_partition(samples: int, seed: int):
    return generate_gohr_partition(samples, seed)


def generate_gohr_training_partition(samples: int, seed: int):
    return generate_gohr_partition(samples, seed)


def make_gohr_model(seed: int):
    return make_adapter(seed).build_model()


def make_gohr_training_callbacks(seed: int):
    return make_adapter(seed).training_callbacks()


def evaluate_gohr_model(model, test_x, test_y):
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    return float(loss), float(accuracy)


def main() -> None:
    args = parse_args()
    validate_args(args)

    root = OUTPUT_DIRECTORY / "training_data_scaling_nested_replicates"

    print("=" * 72)
    print("Gohr D5 — Strengthened Training-Data Scaling Audit")
    print("=" * 72)
    print(f"Dataset                    : {DATASET_ID}")
    print(f"Dataset version            : {DATASET_VERSION}")
    print(f"Speck rounds               : {NUM_ROUNDS}")
    print(f"ResNet depth               : {DEPTH}")
    print(f"Epochs                     : {args.epochs}")
    print(f"Batch size                 : {args.batch_size}")
    print(f"Fixed test samples         : {args.test_samples:,}")
    print(f"Training sizes             : {', '.join(f'{n:,}' for n in TRAINING_SIZES)}")
    print(f"Independent replicates     : {args.replicates}")
    print(f"Bootstrap replicates       : {args.bootstrap_replicates}")
    print("Dataset design             : independent 10M replicate + nested prefixes")
    print("Model-seed design          : same seed across sizes within replicate")
    print("Generator                  : speck.make_train_data")
    print("Generator randomness       : os.urandom")
    print()

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "model": {"architecture": "Gohr tn.make_resnet", "depth": DEPTH, "reg_param": 1e-5},
        "optimizer": "adam",
        "loss": "mse",
        "training_procedure": "Gohr original cyclic learning-rate schedule",
        "generation": {
            "procedure": "speck.make_train_data(n, nr)",
            "num_rounds": NUM_ROUNDS,
            "randomness_source": "os.urandom",
            "deterministic_seed": None,
            "input_difference": {"left_word": "0x0040", "right_word": "0x0000"},
        },
    }

    result = run_d5(
        root=root,
        training_sizes=TRAINING_SIZES,
        test_samples=args.test_samples,
        replicates=args.replicates,
        audit_seed=args.audit_seed,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        train_dataset_factory=generate_gohr_training_partition,
        test_dataset_factory=generate_gohr_test_partition,
        model_factory=make_gohr_model,
        training_callbacks_factory=make_gohr_training_callbacks,
        evaluate_model=evaluate_gohr_model,
        experiment_name="Gohr Speck training-data scaling with nested independent replicates",
        bootstrap_replicates=args.bootstrap_replicates,
        manifest=manifest,
    )

    print_report(result)

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "generation_procedure": "speck.make_train_data(n, nr)",
        "generation_parameters": {
            "generator": "speck.make_train_data",
            "num_rounds": NUM_ROUNDS,
            "training_sizes": list(TRAINING_SIZES),
            "replicates": args.replicates,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "randomness_source": "os.urandom",
            "deterministic_seed": None,
            "input_difference": {"left_word": "0x0040", "right_word": "0x0000"},
        },
    }
    certificate = generate_certificate(
        result,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        generation_procedure=provenance["generation_procedure"],
        generation_parameters=provenance["generation_parameters"],
        audit_seed=args.audit_seed,
    )

    output_path = (
        Path(args.output) if args.output else
        OUTPUT_DIRECTORY / "d5_gohr_training_data_scaling_nested_replicates.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_path = output_path.with_suffix(".png")
    write_learning_curve_plot(result, plot_path)

    print()
    print("=" * 72)
    print("D5 COMPLETE")
    print("=" * 72)
    print(f"Certificate                : {output_path}")
    print(f"Learning-curve plot        : {plot_path}")
    print("Interpretation             : sample-size scaling under the declared Gohr protocol")


if __name__ == "__main__":
    main()
