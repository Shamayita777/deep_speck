"""
Full independent Gohr 5-round training protocol for CipherMind II-2.

This is the expensive stage.

Default protocol:
    rounds            = 5
    differential      = (0x0040, 0x0000)
    residual depth    = 10
    filters           = 32
    L2                = 1e-5
    training samples  = 10^7
    validation        = 10^6
    test              = 10^6
    epochs            = 200
    batch size        = 5000
    loss              = MSE
    optimizer         = Adam
    LR cycle          = 0.002 -> 0.0001
    checkpoint        = best validation loss

No historical CE checkpoints are loaded.

A fresh model is trained from its own initialization.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

from .dataset_independent import (
    GOHR_DIFFERENTIAL,
    generate_dataset,
)
from .model_independent import build_gohr_model


ROUNDS = 5
TRAIN_SAMPLES = 10**7
VALIDATION_SAMPLES = 10**6
TEST_SAMPLES = 10**6
EPOCHS = 200
BATCH_SIZE = 5000

HIGH_LR = 0.002
LOW_LR = 0.0001
LR_CYCLE_LENGTH = 10


ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    ROOT
    / "audit"
    / "implementation"
    / "evidence"
    / "m1"
)


def sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            check=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip()


def gpu_information() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return []

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def make_learning_rate_schedule():
    """
    Reproduce Gohr's cyclic schedule.

    The original public code calls:

        cyclic_lr(10, 0.002, 0.0001)

    which gives:
        epoch 0 -> 0.002
        epoch 9 -> 0.0001
        epoch 10 -> 0.002
    """
    from keras.callbacks import LearningRateScheduler

    def schedule(epoch: int, current_lr=None) -> float:
        phase = epoch % LR_CYCLE_LENGTH

        # Equivalent to the public implementation's
        # cyclic_lr(10, HIGH_LR, LOW_LR).
        value = LOW_LR + (
            (
                (LR_CYCLE_LENGTH - 1) - phase
            )
            / (LR_CYCLE_LENGTH - 1)
        ) * (HIGH_LR - LOW_LR)

        return float(value)

    return LearningRateScheduler(
        schedule,
        verbose=0,
    )


def train(
    *,
    rounds: int = ROUNDS,
    train_samples: int = TRAIN_SAMPLES,
    validation_samples: int = VALIDATION_SAMPLES,
    test_samples: int = TEST_SAMPLES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int | None = None,
):
    """
    Execute the independent 5-round training experiment.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # If explicitly requested, seed framework/model initialization.
    # Gohr's original data itself was drawn from /dev/urandom, so the
    # seed does NOT control dataset generation.
    if seed is not None:
        import os
        import random

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            import keras

            keras.utils.set_random_seed(seed)
        except Exception:
            pass

    print("Generating independent training data...")
    train_start = time.time()

    X_train, y_train = generate_dataset(
        train_samples,
        rounds,
        GOHR_DIFFERENTIAL,
    )

    X_val, y_val = generate_dataset(
        validation_samples,
        rounds,
        GOHR_DIFFERENTIAL,
    )

    data_seconds = time.time() - train_start

    print(
        "Training data generation completed in "
        f"{data_seconds:.3f} seconds."
    )

    model = build_gohr_model(
        depth=10,
        filters=32,
        dense_1=64,
        dense_2=64,
        kernel_size=3,
        word_size=16,
        input_words=4,
        l2_strength=1e-5,
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"],
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "m1_independent_best.keras"
    )

    from keras.callbacks import ModelCheckpoint

    callbacks = [
        make_learning_rate_schedule(),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
    ]

    print("Starting independent 200-epoch training...")

    training_start = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    training_seconds = time.time() - training_start

    print(
        "Independent training completed in "
        f"{training_seconds:.3f} seconds."
    )

    # Generate a completely fresh test set.
    print("Generating fresh independent test data...")

    X_test, y_test = generate_dataset(
        test_samples,
        rounds,
        GOHR_DIFFERENTIAL,
    )

    from keras.models import load_model

    best_model = load_model(
        checkpoint_path,
    )

    test_loss, test_accuracy = best_model.evaluate(
        X_test,
        y_test,
        batch_size=batch_size,
        verbose=1,
    )

    validation_losses = np.asarray(
        history.history["val_loss"],
        dtype=np.float64,
    )

    validation_accuracy = np.asarray(
        history.history["val_accuracy"],
        dtype=np.float64,
    )

    best_epoch = int(
        np.argmin(validation_losses)
    )

    report = {
        "experiment_id": "II2-INDEPENDENT-GOHR-5R",
        "status": "COMPLETE",
        "scientific_status": "TRAINED_INDEPENDENT_IMPLEMENTATION",
        "scope": {
            "cipher": "Speck32/64",
            "rounds": rounds,
            "differential": [
                hex(GOHR_DIFFERENTIAL[0]),
                hex(GOHR_DIFFERENTIAL[1]),
            ],
            "task": "real-vs-random differential classification",
        },
        "protocol": {
            "depth": 10,
            "filters": 32,
            "kernel_size": 3,
            "dense_1": 64,
            "dense_2": 64,
            "l2": 1e-5,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adam",
            "loss": "mse",
            "high_learning_rate": HIGH_LR,
            "low_learning_rate": LOW_LR,
            "lr_cycle_length": LR_CYCLE_LENGTH,
            "checkpoint_selection": "minimum_validation_loss",
        },
        "data": {
            "generation_random_source": "operating-system random source",
            "test_generated_after_training": True,
            "test_used_during_training": False,
            "test_used_for_checkpoint_selection": False,
        },
        "results": {
            "best_validation_epoch_zero_based": best_epoch,
            "best_validation_loss": float(
                validation_losses[best_epoch]
            ),
            "best_validation_accuracy": float(
                validation_accuracy[best_epoch]
            ),
            "fresh_test_loss": float(test_loss),
            "fresh_test_accuracy": float(test_accuracy),
        },
        "timing": {
            "data_generation_seconds": data_seconds,
            "training_seconds": training_seconds,
            "total_seconds": (
                data_seconds
                + training_seconds
            ),
        },
        "seed": seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gpu_names": gpu_information(),
        },
        "provenance": {
            "git_commit": git_commit(),
            "independent_speck_sha256": sha256_file(
                Path(__file__).with_name(
                    "speck_independent.py"
                )
            ),
            "independent_dataset_sha256": sha256_file(
                Path(__file__).with_name(
                    "dataset_independent.py"
                )
            ),
            "independent_model_sha256": sha256_file(
                Path(__file__).with_name(
                    "model_independent.py"
                )
            ),
            "independent_train_sha256": sha256_file(
                Path(__file__)
            ),
        },
        "source_discrepancy": {
            "paper_vs_public_code_dense2_batchnorm": True,
            "resolution": (
                "Independent implementation follows the public "
                "supplementary train_nets.py executable path, "
                "which includes BatchNormalization after the "
                "second dense layer. The paper prose says the "
                "second hidden layer does not use batch normalization. "
                "This discrepancy is recorded rather than hidden."
            ),
        },
        "note": (
            "This is an independently implemented neural reproduction. "
            "Numerical identity with the original training run is not "
            "expected because the original uses OS randomness and the "
            "software/hardware environment may differ."
        ),
    }

    report_path = (
        OUTPUT_DIR
        / "m1_independent_training_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print("=" * 72)
    print("M1 INDEPENDENT TRAINING")
    print("=" * 72)
    print(
        "Best validation accuracy:",
        report["results"]["best_validation_accuracy"],
    )
    print(
        "Fresh test accuracy:",
        report["results"]["fresh_test_accuracy"],
    )
    print(
        "Best validation epoch:",
        best_epoch + 1,
    )
    print(
        "Evidence:",
        report_path,
    )
    print("=" * 72)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=VALIDATION_SAMPLES,
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
        "--seed",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    train(
        rounds=args.rounds,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()