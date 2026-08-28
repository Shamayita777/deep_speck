"""
Gohr D4 - Label Shuffle Controlled Perturbation Experiment.

This is the Gohr-specific experiment driver.

Dataset generation
------------------
All Gohr partitions are generated through GohrAdapter, which
delegates directly to speck.make_train_data(). No alternate
dataset-generation implementation is used.

Experimental design
-------------------
For each independent replicate:

    1. train the Gohr distinguisher on the original labels;
    2. train the same distinguisher on a random permutation
       of the training labels;
    3. evaluate both models on the SAME fixed held-out test
       partition;
    4. perform paired McNemar inference on per-example
       correctness.

The test partition is generated once and then reused for every
training replicate. It is never used to fit the models.

Interpretation
--------------
A reproducible performance collapse under label shuffling is
evidence that predictive performance depends on the original
feature/label correspondence.

It is NOT, by itself, evidence that the model specifically learned
cryptographic structure. Alternative explanations such as
dataset-construction artifacts or implementation artifacts remain
possible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from audit.dataset.adapters.gohr import GohrAdapter
from audit.dataset.d4_controlled_perturbation import (
    generate_certificate,
    print_report,
    run_d4,
)
from audit.dataset.perturbations.label_shuffle import (
    LabelShufflePerturbation,
)

# ============================================================
# Persistent dataset storage
# ============================================================

def dataset_paths(
    root: Path,
) -> dict[str, Path]:

    dataset_dir = root / "dataset"

    return {
        "train_x": dataset_dir / "train_x.npy",
        "train_y": dataset_dir / "train_y.npy",
        "validation_x": dataset_dir / "validation_x.npy",
        "validation_y": dataset_dir / "validation_y.npy",
        "test_x": dataset_dir / "test_x.npy",
        "test_y": dataset_dir / "test_y.npy",
        "metadata": dataset_dir / "metadata.json",
    }

def dataset_exists(
    root: Path,
) -> bool:

    paths = dataset_paths(root)

    return all(
        path.exists()
        for path in paths.values()
    )
def save_dataset(
    *,
    root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    metadata: dict,
) -> None:

    paths = dataset_paths(root)

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

    np.save(
        paths["validation_x"],
        validation_x,
    )

    np.save(
        paths["validation_y"],
        validation_y,
    )

    np.save(
        paths["test_x"],
        test_x,
    )

    np.save(
        paths["test_y"],
        test_y,
    )

    save_state(
        paths["metadata"],
        metadata,
    )
def load_dataset(
    root: Path,
):
    paths = dataset_paths(root)

    return (
        np.load(
            paths["train_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["train_y"],
            mmap_mode="r",
        ),
        np.load(
            paths["validation_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["validation_y"],
            mmap_mode="r",
        ),
        np.load(
            paths["test_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["test_y"],
            mmap_mode="r",
        ),
    )

def permutation_path(
    root: Path,
    replicate: int,
) -> Path:

    return (
        root
        / "perturbations"
        / f"replicate_{replicate:02d}_labels.npy"
    )


def load_or_create_label_permutation(
    *,
    root: Path,
    replicate: int,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    path = permutation_path(
        root,
        replicate,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        permutation = np.load(
            path,
            mmap_mode="r",
        )

        if permutation.shape != labels.shape:
            raise ValueError(
                "Persisted label permutation has "
                "an incompatible shape."
            )

        return permutation

    permutation = rng.permutation(
        len(labels)
    ).astype(
        np.int64,
        copy=False,
    )

    temporary = path.with_suffix(
        ".tmp.npy"
    )

    np.save(
        temporary,
        permutation,
    )

    temporary.replace(
        path
    )

    return np.load(
        path,
        mmap_mode="r",
    )

def experiment_metadata(
    *,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> dict:

    return {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "generator": "speck.make_train_data",
        "randomness_source": "os.urandom",
        "num_rounds": NUM_ROUNDS,
        "depth": DEPTH,
        "epochs": EPOCHS,
        "batch_size": 5000,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "input_difference": {
            "left_word": "0x0040",
            "right_word": "0x0000",
        },
    }

def state_path(
    root: Path,
    replicate: int,
    condition: str,
) -> Path:

    return (
        root
        / "state"
        / f"replicate_{replicate:02d}_{condition}.json"
    )


def checkpoint_path(
    root: Path,
    replicate: int,
    condition: str,
) -> Path:

    return (
        root
        / "checkpoints"
        / f"replicate_{replicate:02d}"
        / condition
        / "latest.keras"
    )


def save_state(
    path: Path,
    state: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            state,
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
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)

class EpochStateCallback(Callback):
    """
    Persist the completed epoch in a sidecar state file.

    Epoch numbering stored here is one-based:
        epoch 1 means the first epoch has completed.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        replicate: int,
        condition: str,
        seed: int,
        total_epochs: int,
    ) -> None:

        super().__init__()

        self.state_path = state_path
        self.replicate = replicate
        self.condition = condition
        self.seed = seed
        self.total_epochs = total_epochs

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):

        state = {
            "schema_version": "1.0",
            "replicate": self.replicate,
            "condition": self.condition,
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
                for key, value in (logs or {}).items()
                if np.isscalar(value)
            },
        }

        save_state(
            self.state_path,
            state,
        )


# ============================================================
# Gohr configuration
# ============================================================

NUM_ROUNDS = 5

DEPTH = 10
EPOCHS = 200

DEFAULT_TRAIN_SAMPLES = 10**7
DEFAULT_TEST_SAMPLES = 10**6

DEFAULT_REPLICATES = 5
DEFAULT_BOOTSTRAP_REPLICATES = 5000

DEFAULT_EFFECT_THRESHOLD = 0.01
DEFAULT_ALPHA = 0.05
DEFAULT_AUDIT_SEED = 0

DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"

OUTPUT_DIRECTORY = Path(
    "audit/dataset/evidence/d4"
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repeated Gohr label-shuffle "
            "controlled perturbation audit."
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
        default=10**6,
        help=(
            "Number of samples in the fixed Gohr validation "
            "partition used during training."
        ),
    )
    
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help=(
            "Number of independent baseline/perturbed "
            "training replicates."
        ),
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )

    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=DEFAULT_EFFECT_THRESHOLD,
        help=(
            "Minimum absolute accuracy difference "
            "in fraction-of-one units regarded as "
            "practically meaningful."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    parser.add_argument(
        "--audit-seed",
        type=int,
        default=DEFAULT_AUDIT_SEED,
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

    if args.train_samples < 1:
        raise ValueError(
            "--train-samples must be >= 1."
        )

    if args.validation_samples < 1:
        raise ValueError(
            "--validation-samples must be >= 1."
        )

    if args.test_samples < 1:
        raise ValueError(
            "--test-samples must be >= 1."
        )

    if args.replicates < 2:
        raise ValueError(
            "--replicates must be >= 2."
        )

    if args.bootstrap_replicates < 1000:
        raise ValueError(
            "--bootstrap-replicates must be >= 1000."
        )

    if args.effect_threshold < 0:
        raise ValueError(
            "--effect-threshold must be non-negative."
        )

    if not 0.0 < args.alpha < 1.0:
        raise ValueError(
            "--alpha must lie in (0, 1)."
        )

def dataset_cache_paths(
    root: Path,
) -> dict[str, Path]:

    directory = root / "dataset"

    return {
        "train_x": directory / "train_x.npy",
        "train_y": directory / "train_y.npy",
        "validation_x": directory / "validation_x.npy",
        "validation_y": directory / "validation_y.npy",
        "test_x": directory / "test_x.npy",
        "test_y": directory / "test_y.npy",
        "metadata": directory / "metadata.json",
    }

def cached_dataset_exists(
    root: Path,
) -> bool:

    paths = dataset_cache_paths(root)

    return all(
        path.exists()
        for path in paths.values()
    )

def save_dataset_cache(
    *,
    root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> None:

    paths = dataset_cache_paths(root)

    paths["train_x"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = {
        "train_x": train_x,
        "train_y": train_y,
        "validation_x": validation_x,
        "validation_y": validation_y,
        "test_x": test_x,
        "test_y": test_y,
    }

    for name, array in arrays.items():

        final_path = paths[name]

        temporary_path = final_path.with_suffix(
            ".tmp.npy"
        )

        np.save(
            temporary_path,
            array,
        )

        temporary_path.replace(
            final_path,
        )

def load_dataset_cache(
    root: Path,
):
    paths = dataset_cache_paths(root)

    return (
        np.load(
            paths["train_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["train_y"],
            mmap_mode="r",
        ),
        np.load(
            paths["validation_x"],
            mmap_mode="r",
        ),
        np.load(
            paths["validation_y"],
            mmap_mode="r",
        ),
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
# Gohr data generation
# ============================================================
def generate_gohr_partitions(
    *,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate the Gohr training, validation, and held-out test
    partitions.

    All partitions are generated directly through GohrAdapter,
    which delegates to speck.make_train_data().

    The validation partition is generated independently from the
    training and test partitions and is supplied to every model
    fit. The test partition is generated once and remains fixed
    across all D4 replicates and perturbation conditions.

    Returns
    -------
    train_x, train_y,
    validation_x, validation_y,
    test_x, test_y
    """

    if train_samples < 1:
        raise ValueError(
            "train_samples must be >= 1."
        )

    if validation_samples < 1:
        raise ValueError(
            "validation_samples must be >= 1."
        )

    if test_samples < 1:
        raise ValueError(
            "test_samples must be >= 1."
        )

    generator_adapter = GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=NUM_ROUNDS,
        depth=DEPTH,
        epochs=EPOCHS,
    )

    print("=" * 72)
    print("Gohr D4 Dataset Generation")
    print("=" * 72)
    print()

    print(
        f"Speck rounds             : {NUM_ROUNDS}"
    )

    print(
        f"Training samples         : {train_samples}"
    )

    print(
        f"Validation samples       : {validation_samples}"
    )

    print(
        f"Test samples             : {test_samples}"
    )

    print(
        "Generator                : "
        "speck.make_train_data"
    )

    print(
        "Generator randomness     : os.urandom"
    )

    print()

    print(
        "Generating training partition..."
    )

    train_x, train_y = (
        generator_adapter.generate_partition(
            train_samples
        )
    )

    print(
        f"  shape={train_x.shape}, "
        f"dtype={train_x.dtype}"
    )

    print(
        "Generating validation partition..."
    )

    validation_x, validation_y = (
        generator_adapter.generate_partition(
            validation_samples
        )
    )

    print(
        f"  shape={validation_x.shape}, "
        f"dtype={validation_x.dtype}"
    )

    print(
        "Generating fixed held-out test partition..."
    )

    test_x, test_y = (
        generator_adapter.generate_partition(
            test_samples
        )
    )

    print(
        f"  shape={test_x.shape}, "
        f"dtype={test_x.dtype}"
    )

    print()

    return (
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x,
        test_y,
    )

# ============================================================
# Adapter factory
# ============================================================
def make_adapter(
    *,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    model_directory: Path,
) -> GohrAdapter:
    """
    Construct a fresh Gohr adapter for one independent fit.

    The validation and test partitions are fixed across the
    baseline/perturbed comparison for a given D4 experiment.

    Each fit receives its own model directory so checkpoint files
    cannot collide between baseline and perturbed conditions.
    """

    return GohrAdapter(
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        num_rounds=NUM_ROUNDS,
        depth=DEPTH,
        epochs=EPOCHS,
        seed=seed,
        model_directory=str(model_directory),
    )
# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()
    validate_args(args)

    dataset_root = (
        OUTPUT_DIRECTORY
        / "persistent_data"
    )

    if cached_dataset_exists(
        dataset_root
    ):

        print("=" * 72)
        print("Loading cached D4 dataset")
        print("=" * 72)

        (
            train_x,
            train_y,
            validation_x,
            validation_y,
            test_x,
            test_y,
        ) = load_dataset_cache(
            dataset_root
        )

    else:

        (
            train_x,
            train_y,
            validation_x,
            validation_y,
            test_x,
            test_y,
        ) = generate_gohr_partitions(
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            test_samples=args.test_samples,
        )

        save_dataset_cache(
            root=dataset_root,
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            test_x=test_x,
            test_y=test_y,
        )

        print(
            "D4 dataset cached for future resumes."
        )

    perturbation = (
        LabelShufflePerturbation()
    )

    def adapter_factory(seed: int):
        model_directory = (
            OUTPUT_DIRECTORY
            / "models"
            / f"seed{seed}"
        )

        return make_adapter(
            validation_x=validation_x,
            validation_y=validation_y,
            test_x=test_x,
            test_y=test_y,
            seed=seed,
            model_directory=model_directory,
        )

    print("=" * 72)
    print("Running D4 Controlled Perturbation Audit")
    print("=" * 72)

    result = run_d4(
        perturbation=perturbation,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=adapter_factory,
        checkpoint_root=(
            OUTPUT_DIRECTORY
            / "checkpoints"
        ),
        replicates=args.replicates,
        audit_seed=args.audit_seed,
        effect_threshold=args.effect_threshold,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=0.95,
        alpha=args.alpha,
    )

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "generation_procedure": (
            "speck.make_train_data(n, nr)"
        ),
        "generation_parameters": {
            "generator": "speck.make_train_data",
            "num_rounds": NUM_ROUNDS,
            "depth": DEPTH,
            "epochs": EPOCHS,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "randomness_source": "os.urandom",
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
        confidence_level=0.95,
        alpha=args.alpha,
    )

    print()

    print_report(result)

    output_path = (
        Path(args.output)
        if args.output is not None
        else (
            OUTPUT_DIRECTORY
            / (
                "d4_gohr_label_shuffle_"
                f"{args.train_samples}_"
                f"{args.test_samples}_"
                f"{args.replicates}replicates_"
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
        handle.write("\n")

    print()
    print("=" * 72)
    print("D4 COMPLETE")
    print("=" * 72)
    print(
        f"Outcome                    : "
        f"{certificate['decision']['outcome']}"
    )
    print(
        f"Certificate                : "
        f"{output_path}"
    )
    print(
        "Interpretation             : "
        "label-shuffle dependence; "
        "not cryptographic attribution"
    )


if __name__ == "__main__":
    main()