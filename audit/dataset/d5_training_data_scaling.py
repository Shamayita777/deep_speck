"""
D5 - Training Data Scaling Audit.

Generic framework for measuring how predictive performance changes
as the amount of training data is varied.

Scientific design
-----------------
D5 evaluates a sequence of training-set sizes while holding the
following fixed:

    - dataset-generation mechanism;
    - model architecture;
    - optimization procedure;
    - number of training epochs;
    - validation procedure;
    - held-out test partition.

Each training-data condition is an independent experiment.

The framework is dataset-agnostic. Dataset-specific generation and
model construction must be supplied by an adapter/factory.

Checkpointing
-------------
Each condition has:

    - a persisted training dataset;
    - a resumable Keras checkpoint;
    - a JSON state file.

The checkpoint is overwritten after every completed epoch.

A completed condition is recorded in its state file and skipped on
subsequent runs.

The dataset is generated once per condition and persisted. Therefore
a resumed run never silently regenerates a different training set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json
import os

import numpy as np
from keras.callbacks import Callback
from keras.models import load_model


# ============================================================
# Result structures
# ============================================================

@dataclass
class D5ConditionResult:
    """Result for one training-data-size condition."""

    condition: str
    training_samples: int
    test_samples: int

    seed: int

    test_accuracy: float
    test_loss: float

    completed_epochs: int
    total_epochs: int


@dataclass
class D5Result:
    """Aggregated D5 scaling experiment."""

    experiment: str

    conditions: list[D5ConditionResult]

    best_condition: str
    best_training_samples: int
    best_test_accuracy: float


# ============================================================
# Persistent paths
# ============================================================

def dataset_paths(
    root: Path,
    condition: str,
) -> dict[str, Path]:
    """Return paths for one persisted training dataset."""

    dataset_dir = (
        root
        / "datasets"
        / condition
    )

    return {
        "train_x": dataset_dir / "train_x.npy",
        "train_y": dataset_dir / "train_y.npy",
    }


def test_dataset_paths(
    root: Path,
) -> dict[str, Path]:
    """Return paths for the single fixed held-out test dataset."""

    dataset_dir = (
        root
        / "datasets"
        / "test"
    )

    return {
        "test_x": dataset_dir / "test_x.npy",
        "test_y": dataset_dir / "test_y.npy",
    }


def dataset_exists(
    root: Path,
    condition: str,
) -> bool:
    """Return whether the training dataset is completely persisted."""

    paths = dataset_paths(
        root,
        condition,
    )

    return all(
        path.exists()
        for path in paths.values()
    )


def test_dataset_exists(
    root: Path,
) -> bool:
    """Return whether the fixed test dataset is completely persisted."""

    paths = test_dataset_paths(root)

    return all(
        path.exists()
        for path in paths.values()
    )


def save_dataset(
    *,
    root: Path,
    condition: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> None:
    """Persist one complete training dataset."""

    paths = dataset_paths(
        root,
        condition,
    )

    paths["train_x"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        paths["train_x"],
        train_x,
    )

    np.save(
        paths["train_y"],
        train_y,
    )


def load_dataset(
    root: Path,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a persisted training dataset using memory mapping."""

    paths = dataset_paths(
        root,
        condition,
    )

    return (
        np.load(
            paths["train_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["train_y"],
            mmap_mode="r",
        ),
    )


def save_test_dataset(
    *,
    root: Path,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> None:
    """Persist the fixed held-out test dataset."""

    paths = test_dataset_paths(root)

    paths["test_x"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        paths["test_x"],
        test_x,
    )

    np.save(
        paths["test_y"],
        test_y,
    )


def load_test_dataset(
    root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the fixed held-out test dataset."""

    paths = test_dataset_paths(root)

    return (
        np.load(
            paths["test_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["test_y"],
            mmap_mode="r",
        ),
    )


# ============================================================
# State / checkpoint paths
# ============================================================

def state_path(
    root: Path,
    condition: str,
) -> Path:
    """Return the state-file path for one condition."""

    return (
        root
        / "state"
        / f"{condition}.json"
    )


def checkpoint_path(
    root: Path,
    condition: str,
) -> Path:
    """Return the latest resumable model checkpoint."""

    return (
        root
        / "checkpoints"
        / condition
        / "latest.keras"
    )


def save_state(
    path: Path,
    state: Mapping[str, Any],
) -> None:
    """Atomically save a JSON state file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            dict(state),
            handle,
            indent=2,
            sort_keys=True,
        )

    os.replace(
        temporary,
        path,
    )


def load_state(
    path: Path,
) -> dict[str, Any]:
    """Load a persisted state file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


# ============================================================
# Epoch checkpoint callback
# ============================================================

class EpochCheckpointCallback(Callback):
    """
    Save a resumable Keras model after every completed epoch.

    The checkpoint contains the model and optimizer state.

    A JSON sidecar records the completed epoch and latest metrics.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        state_path: Path,
        condition: str,
        training_samples: int,
        seed: int,
        total_epochs: int,
    ) -> None:

        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.state_path = state_path
        self.condition = condition
        self.training_samples = training_samples
        self.seed = seed
        self.total_epochs = total_epochs

    def on_epoch_end(
        self,
        epoch: int,
        logs=None,
    ) -> None:

        logs = logs or {}

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_checkpoint = (
            self.checkpoint_path.with_suffix(
                ".tmp.keras"
            )
        )

        self.model.save(
            temporary_checkpoint,
        )

        os.replace(
            temporary_checkpoint,
            self.checkpoint_path,
        )

        state = {
            "schema_version": "1.0",
            "condition": self.condition,
            "training_samples": self.training_samples,
            "seed": self.seed,
            "total_epochs": self.total_epochs,
            "completed_epochs": int(epoch + 1),
            "status": (
                "complete"
                if epoch + 1 >= self.total_epochs
                else "in_progress"
            ),
            "last_epoch_logs": {
                key: float(value)
                for key, value in logs.items()
                if np.isscalar(value)
            },
        }

        save_state(
            self.state_path,
            state,
        )


# ============================================================
# Condition execution
# ============================================================

def run_condition(
    *,
    root: Path,
    condition: str,
    training_samples: int,
    test_samples: int,
    seed: int,
    total_epochs: int,
    batch_size: int,
    train_dataset_factory: Callable[
        [int, int],
        tuple[np.ndarray, np.ndarray],
    ],
    model_factory: Callable[
        [int],
        Any,
    ],
    evaluate_model: Callable[
        [Any, np.ndarray, np.ndarray],
        tuple[float, float],
    ],
    training_callbacks_factory: Callable[
        [int],
        list[Any],
    ],
) -> D5ConditionResult:
    """
    Execute or resume one D5 training-data condition.

    Parameters
    ----------
    train_dataset_factory:
        Callable receiving (training_samples, seed) and returning
        (train_x, train_y).

    model_factory:
        Callable receiving seed and returning a fresh compiled model.

    evaluate_model:
        Callable receiving (model, test_x, test_y) and returning
        (loss, accuracy).
    """

    if training_samples < 1:
        raise ValueError(
            "training_samples must be >= 1."
        )

    if test_samples < 1:
        raise ValueError(
            "test_samples must be >= 1."
        )

    if total_epochs < 1:
        raise ValueError(
            "total_epochs must be >= 1."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1."
        )

    condition_state_path = state_path(
        root,
        condition,
    )

    condition_checkpoint_path = checkpoint_path(
        root,
        condition,
    )

    # --------------------------------------------------------
    # Already complete
    # --------------------------------------------------------

    if condition_state_path.exists():

        state = load_state(
            condition_state_path
        )

        if (
            state.get("status") == "complete"
            and condition_checkpoint_path.exists()
        ):
            print(
                f"Skipping completed condition: "
                f"{condition}"
            )

            test_x, test_y = load_test_dataset(
                root
            )

            model = load_model(
                condition_checkpoint_path,
                compile=True,
            )

            test_loss, test_accuracy = evaluate_model(
                model,
                test_x,
                test_y,
            )

            del model

            return D5ConditionResult(
                condition=condition,
                training_samples=training_samples,
                test_samples=test_samples,
                seed=seed,
                test_accuracy=test_accuracy,
                test_loss=test_loss,
                completed_epochs=total_epochs,
                total_epochs=total_epochs,
            )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    if dataset_exists(
        root,
        condition,
    ):
        print(
            f"Loading persisted training dataset: "
            f"{condition}"
        )

        train_x, train_y = load_dataset(
            root,
            condition,
        )

    else:
        print(
            f"Generating training dataset: "
            f"{condition}"
        )

        train_x, train_y = train_dataset_factory(
            training_samples,
            seed,
        )

        save_dataset(
            root=root,
            condition=condition,
            train_x=train_x,
            train_y=train_y,
        )

        print(
            f"Saved training dataset: "
            f"{condition}"
        )

    # --------------------------------------------------------
    # Fixed test dataset
    # --------------------------------------------------------

    test_x, test_y = load_test_dataset(
        root
    )

    # --------------------------------------------------------
    # Resume or create model
    # --------------------------------------------------------

    completed_epochs = 0

    if condition_checkpoint_path.exists():

        print(
            f"Resuming checkpoint: "
            f"{condition_checkpoint_path}"
        )

        model = load_model(
            condition_checkpoint_path,
            compile=True,
        )

        if condition_state_path.exists():

            state = load_state(
                condition_state_path
            )

            completed_epochs = int(
                state.get(
                    "completed_epochs",
                    0,
                )
            )

    else:

        print(
            f"Starting new model: "
            f"{condition}"
        )

        model = model_factory(
            seed
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    if completed_epochs < total_epochs:

        callback = EpochCheckpointCallback(
            checkpoint_path=condition_checkpoint_path,
            state_path=condition_state_path,
            condition=condition,
            training_samples=training_samples,
            seed=seed,
            total_epochs=total_epochs,
        )

        gohr_callbacks = (
            training_callbacks_factory(seed)
            if training_callbacks_factory is not None
            else []
        )

        model.fit(
            train_x,
            train_y,
            initial_epoch=completed_epochs,
            epochs=total_epochs,
            batch_size=batch_size,
            callbacks=[
                callback,
                *gohr_callbacks,
            ],
            verbose=1,
        )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    test_loss, test_accuracy = evaluate_model(
        model,
        test_x,
        test_y,
    )

    final_state = {
        "schema_version": "1.0",
        "condition": condition,
        "training_samples": training_samples,
        "seed": seed,
        "total_epochs": total_epochs,
        "completed_epochs": total_epochs,
        "status": "complete",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }

    save_state(
        condition_state_path,
        final_state,
    )

    del model

    return D5ConditionResult(
        condition=condition,
        training_samples=training_samples,
        test_samples=test_samples,
        seed=seed,
        test_accuracy=float(test_accuracy),
        test_loss=float(test_loss),
        completed_epochs=total_epochs,
        total_epochs=total_epochs,
    )


# ============================================================
# Complete D5 experiment
# ============================================================

def run_d5(
    *,
    root: Path,
    training_sizes: Sequence[int],
    test_samples: int,
    audit_seed: int,
    total_epochs: int,
    batch_size: int,
    train_dataset_factory: Callable[
        [int, int],
        tuple[np.ndarray, np.ndarray],
    ],
    test_dataset_factory: Callable[
        [int, int],
        tuple[np.ndarray, np.ndarray],
    ],
    model_factory: Callable[
        [int],
        Any,
    ],
    evaluate_model: Callable[
        [Any, np.ndarray, np.ndarray],
        tuple[float, float],
    ],
    experiment_name: str,
) -> D5Result:
    """
    Run all D5 training-data-size conditions.

    The test partition is generated exactly once and then reused
    across every condition.
    """

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(training_sizes) < 2:
        raise ValueError(
            "At least two training-data sizes are required."
        )

    # --------------------------------------------------------
    # Fixed held-out test partition
    # --------------------------------------------------------

    if test_dataset_exists(root):

        print(
            "Loading existing fixed test dataset..."
        )

    else:

        print(
            "Generating fixed held-out test dataset..."
        )

        test_x, test_y = test_dataset_factory(
            test_samples,
            audit_seed,
        )

        save_test_dataset(
            root=root,
            test_x=test_x,
            test_y=test_y,
        )

        print(
            "Fixed test dataset saved."
        )

        del test_x
        del test_y

    # --------------------------------------------------------
    # Independent condition seeds
    # --------------------------------------------------------

    seed_sequence = np.random.SeedSequence(
        audit_seed
    )

    child_sequences = seed_sequence.spawn(
        len(training_sizes)
    )

    condition_results: list[D5ConditionResult] = []

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    for index, (
        training_samples,
        child_sequence,
    ) in enumerate(
        zip(
            training_sizes,
            child_sequences,
        ),
        start=1,
    ):

        seed = int(
            child_sequence.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )

        condition = str(
            training_samples
        )

        print()
        print("=" * 72)
        print(
            f"D5 condition {index}/{len(training_sizes)}"
        )
        print(
            f"Training samples : {training_samples}"
        )
        print(
            f"Seed             : {seed}"
        )
        print("=" * 72)

        result = run_condition(
            root=root,
            condition=condition,
            training_samples=training_samples,
            test_samples=test_samples,
            seed=seed,
            total_epochs=total_epochs,
            batch_size=batch_size,
            train_dataset_factory=train_dataset_factory,
            model_factory=model_factory,
            evaluate_model=evaluate_model,
        )

        condition_results.append(
            result
        )

        print(
            f"Completed {condition}: "
            f"test_accuracy={result.test_accuracy:.8f}"
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    best = max(
        condition_results,
        key=lambda result: result.test_accuracy,
    )

    return D5Result(
        experiment=experiment_name,
        conditions=condition_results,
        best_condition=best.condition,
        best_training_samples=best.training_samples,
        best_test_accuracy=best.test_accuracy,
    )


# ============================================================
# Certificate
# ============================================================

def generate_certificate(
    result: D5Result,
    *,
    dataset_id: str,
    dataset_version: str,
    generation_procedure: str,
    generation_parameters: Mapping[str, Any],
    audit_seed: int,
) -> dict[str, Any]:
    """Construct a machine-readable D5 certificate."""

    return {
        "audit": {
            "id": "D5",
            "name": "Training Data Scaling Audit",
            "claim": (
                "Whether predictive performance changes as the "
                "amount of training data is varied under a fixed "
                "dataset-generation mechanism, model, training "
                "procedure, and held-out test partition."
            ),
        },
        "decision": {
            "outcome": "COMPLETED",
            "best_training_samples": (
                result.best_training_samples
            ),
            "best_test_accuracy": (
                result.best_test_accuracy
            ),
        },
        "findings": {
            "conditions": [
                asdict(condition)
                for condition in result.conditions
            ],
        },
        "methodology": {
            "unit_of_comparison": (
                "training-data-size condition"
            ),
            "evaluation_design": (
                "same fixed held-out test partition across "
                "all training-data-size conditions"
            ),
            "training_data": (
                "dataset generated by the dataset-specific "
                "adapter"
            ),
            "checkpoint_strategy": (
                "latest.keras overwritten after every "
                "completed epoch"
            ),
            "resume_strategy": (
                "resume from the latest model checkpoint "
                "and persisted completed-epoch state"
            ),
        },
        "provenance": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_procedure": generation_procedure,
            "generation_parameters": dict(
                generation_parameters
            ),
            "audit_seed": audit_seed,
        },
        "limitations": [
            (
                "The experiment measures performance scaling "
                "under the specified model and optimization "
                "procedure; it does not establish a universal "
                "law relating dataset size to performance."
            ),
            (
                "Different training-data-size conditions use "
                "independently generated training datasets "
                "unless the dataset-specific experiment "
                "explicitly specifies another design."
            ),
            (
                "The result is conditional on the specified "
                "dataset-generation procedure, model, "
                "optimization procedure, and test partition."
            ),
        ],
    }


# ============================================================
# Reporting
# ============================================================

def print_report(
    result: D5Result,
) -> None:
    """Print a concise D5 report."""

    print()
    print("=" * 72)
    print("Dataset Integrity Audit")
    print("D5 - Training Data Scaling Audit")
    print("=" * 72)
    print()

    print(
        f"Experiment                  : "
        f"{result.experiment}"
    )

    print()

    print("Training-data-size results")
    print("-" * 72)

    for condition in result.conditions:

        print(
            f"{condition.training_samples:>12,d} samples | "
            f"test accuracy={condition.test_accuracy:.8f} | "
            f"test loss={condition.test_loss:.8f}"
        )

    print()

    print(
        f"Best condition              : "
        f"{result.best_condition}"
    )

    print(
        f"Best training samples       : "
        f"{result.best_training_samples:,}"
    )

    print(
        f"Best test accuracy          : "
        f"{result.best_test_accuracy:.8f}"
    )

    print()
    print("=" * 72)
