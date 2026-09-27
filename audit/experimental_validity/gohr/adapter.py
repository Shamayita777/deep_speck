"""
GohrAdapter: executes one replicate of any of the four frozen EV
experiments, tying together dataset generation/reuse, the
representation transform, model construction, training, and
confirmatory evaluation.

This is the ONLY place Gohr-specific orchestration lives. The generic
framework (framework/*) knows nothing about Speck, bits, or Keras.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from framework.failures import FailureReason, ReplicateOutcome, ReplicateStatus
from framework.provenance import config_hash, software_provenance, utc_timestamp
from gohr import dataset as gohr_dataset
from gohr import evaluate as gohr_evaluate
from gohr import model as gohr_model
from gohr import train as gohr_train
from gohr.baseline import BASELINE, CONFIRMATORY_EVALUATION_PROTOCOL


def hash_initial_weights(model) -> str:
    """
    sha256 over the model's weight tensors in layer order, taken before
    training. Identity of this hash across a matched pair is the actual
    evidence that both arms started from the same initialization.
    """
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for w in model.get_weights():
        arr = np.ascontiguousarray(w)
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()
from gohr.representation import Candidate1Permutation, apply_candidate1, identity_permutation


@dataclass(frozen=True)
class MatchedDatasets:
    """
    One set of (train, validation, confirmatory-test) dataset instances,
    generated once and intended to be reused, unmodified, across both
    arms of a matched-paired replicate (H-EV-SHUFFLE, H-EV-REPRESENTATION).
    """
    train: gohr_dataset.DatasetBundle
    validation: gohr_dataset.DatasetBundle
    confirmatory_test: gohr_dataset.DatasetBundle


def generate_matched_datasets(
    *,
    rounds: int,
    differential: tuple[int, int],
    train_size: int,
    val_size: int,
    confirmatory_test_size: int,
) -> MatchedDatasets:
    """
    Generate one fresh instance of each partition. Because
    gohr.speck.make_train_data is os.urandom-based, these three calls
    produce three independent, non-overlapping-by-construction samples
    (Gohr's generator does not draw from a shared pool that could
    overlap) - each is independently generated, not split from one
    larger draw.
    """
    train = gohr_dataset.generate_dataset(n=train_size, rounds=rounds, differential=differential, role="train")
    validation = gohr_dataset.generate_dataset(n=val_size, rounds=rounds, differential=differential, role="validation")
    confirmatory_test = gohr_dataset.generate_dataset(
        n=confirmatory_test_size, rounds=rounds, differential=differential, role="confirmatory_test"
    )
    return MatchedDatasets(train=train, validation=validation, confirmatory_test=confirmatory_test)


def _apply_representation(X: np.ndarray, permutation: Candidate1Permutation) -> np.ndarray:
    return apply_candidate1(X, permutation)


@dataclass(frozen=True)
class ReplicateConfig:
    """Full, explicit configuration for one replicate. No hidden defaults."""

    experiment_id: str
    hypothesis_id: Optional[str]
    condition_id: str
    replicate_id: str
    run_id: str
    run_mode: str  # "smoke" | "production"

    rounds: int
    differential: tuple[int, int]
    depth: int
    epochs: int
    batch_size: int
    shuffle: bool
    optimizer: str
    lr_high: float
    lr_low: float
    lr_period: int
    model_seed: int

    representation: Candidate1Permutation

    output_dir: Path

    matched_datasets: Optional[MatchedDatasets] = None
    # If None, fresh independent datasets are generated for this replicate
    # (EV-BASELINE). If supplied, the exact same datasets are reused
    # (H-EV-SHUFFLE / H-EV-REPRESENTATION matched arms).
    independent_train_size: Optional[int] = None
    independent_val_size: Optional[int] = None
    independent_confirmatory_test_size: Optional[int] = None

    same_dataset: bool = False
    same_model_initialization: bool = False
    same_training_shuffle_stream: bool = False
    same_evaluation_data: bool = False


@dataclass(frozen=True)
class ReplicateExecutionResult:
    outcome: ReplicateOutcome
    manifest_fields: dict[str, Any]


class GohrAdapter:
    def run_replicate(self, config: ReplicateConfig) -> ReplicateExecutionResult:
        run_started_at = utc_timestamp()

        # --- Resolve datasets (matched vs independent) ---
        if config.matched_datasets is not None:
            datasets = config.matched_datasets
            dataset_mode = "matched"
        else:
            if config.independent_train_size is None:
                raise ValueError("independent_train_size required when matched_datasets is not supplied")
            datasets = generate_matched_datasets(
                rounds=config.rounds,
                differential=config.differential,
                train_size=config.independent_train_size,
                val_size=config.independent_val_size,
                confirmatory_test_size=config.independent_confirmatory_test_size,
            )
            dataset_mode = "independent"

        # --- Apply representation identically to train/val/confirmatory test ---
        X_train = _apply_representation(datasets.train.X, config.representation)
        X_val = _apply_representation(datasets.validation.X, config.representation)
        X_conf = _apply_representation(datasets.confirmatory_test.X, config.representation)

        # --- Seed and build model (seed MUST precede construction) ---
        gohr_train.set_seed(config.model_seed)
        model = gohr_model.make_resnet(depth=config.depth, reg_param=BASELINE.reg_param)
        model_config_hash = config_hash(
            {"depth": config.depth, "architecture": BASELINE.architecture}
        )
        # Hash the ACTUAL initial weights, immediately after construction and
        # BEFORE any training. Within a matched pair the two arms declare
        # same_model_initialization=True; the seed integer alone is not proof
        # of that (different RNG consumption, framework version or layer order
        # can diverge). This records the realized initialization so identity
        # is verifiable after the fact rather than assumed.
        initial_weight_hash = hash_initial_weights(model)

        checkpoint_path = str(
            Path(config.output_dir) / f"{config.condition_id}_{config.replicate_id}_checkpoint.weights.h5"
        )

        training_result = gohr_train.train_model(
            model, X_train, datasets.train.Y, X_val, datasets.validation.Y,
            epochs=config.epochs,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            lr_high=config.lr_high,
            lr_low=config.lr_low,
            lr_period=config.lr_period,
            optimizer=config.optimizer,
            checkpoint_path=checkpoint_path,
        )

        software = software_provenance()

        base_manifest_fields = {
            "experiment_id": config.experiment_id,
            "hypothesis_id": config.hypothesis_id,
            "experiment_version": "1.0",
            "methodology_version": "ciphermind-methodology-v1",
            "cipher": BASELINE.cipher,
            "task": "neural_ciphertext_distinguisher",
            "rounds": config.rounds,
            "differential": list(config.differential),
            "architecture": BASELINE.architecture,
            "depth": config.depth,
            "dataset_id": {
                "train": datasets.train.dataset_id,
                "validation": datasets.validation.dataset_id,
                "confirmatory_test": datasets.confirmatory_test.dataset_id,
            },
            "dataset_role": dataset_mode,
            "dataset_size": {
                "train": datasets.train.n,
                "validation": datasets.validation.n,
                "confirmatory_test": datasets.confirmatory_test.n,
            },
            "dataset_hash": {
                "train": datasets.train.combined_hash,
                "validation": datasets.validation.combined_hash,
                "confirmatory_test": datasets.confirmatory_test.combined_hash,
            },
            "dataset_generation_method": BASELINE.dataset_generation_method,
            "exact_replay_available": False,
            "model_seed": config.model_seed,
            "dataset_seed": None,
            "shuffle_seed_if_available": None,
            "evaluation_seed_if_available": None,
            "statistics_seed": None,
            "software_versions": software,
            "condition": config.condition_id,
            "independent_variable": None,
            "dependent_variable": "confirmatory_test_accuracy",
            "controls": [],
            "replicate_id": config.replicate_id,
            "run_id": config.run_id,
            "representation_permutation_hash": config.representation.hash,
            "pairing": {
                "same_dataset": config.same_dataset,
                "same_model_initialization": config.same_model_initialization,
                "same_training_shuffle_stream": config.same_training_shuffle_stream,
                "same_evaluation_data": config.same_evaluation_data,
            },
            "run_mode": config.run_mode,
            "timestamps": {"started": run_started_at, "finished": utc_timestamp()},
            "model_config_hash": model_config_hash,
            "checkpoint_hash": training_result.checkpoint_hash,
            "initial_weight_hash": initial_weight_hash,
            "confirmatory_evaluation_protocol": CONFIRMATORY_EVALUATION_PROTOCOL,
            "historical_reporting_convention_value": training_result.max_val_acc,
        }

        if training_result.failure_reason is not None:
            outcome = ReplicateOutcome(
                replicate_id=config.replicate_id,
                condition_id=config.condition_id,
                run_id=config.run_id,
                status=ReplicateStatus.FAILED,
                metric_value=None,
                failure_reason=training_result.failure_reason,
                failure_detail=f"Training failure during replicate {config.replicate_id}",
                provenance=base_manifest_fields,
            )
            return ReplicateExecutionResult(outcome=outcome, manifest_fields={
                **base_manifest_fields,
                "raw_metrics": {},
                "alpha": None, "practical_threshold": None,
                "practical_threshold_status": "NOT_AVAILABLE",
                "decision": "NOT_RUN", "warnings": [], "limitations": [],
                "failure_status": "failed",
                "failure_reason": training_result.failure_reason.value,
            })

        evaluation_result = gohr_evaluate.evaluate_model(model, X_conf, datasets.confirmatory_test.Y)

        outcome = ReplicateOutcome(
            replicate_id=config.replicate_id,
            condition_id=config.condition_id,
            run_id=config.run_id,
            status=ReplicateStatus.VALID,
            metric_value=evaluation_result.accuracy,
            provenance=base_manifest_fields,
        )
        manifest_fields = {
            **base_manifest_fields,
            "raw_metrics": {
                "confirmatory_test_accuracy": evaluation_result.accuracy,
                "confirmatory_test_n": evaluation_result.n,
                "max_val_acc_over_history": training_result.max_val_acc,
                "final_val_acc": training_result.final_val_acc,
            },
            "alpha": None,
            "practical_threshold": None,
            "practical_threshold_status": "NOT_AVAILABLE",
            "decision": "NOT_RUN",  # per-replicate manifests do not carry an experiment-level decision
            "warnings": [],
            "limitations": [
                "confirmatory_test_accuracy is computed on a dedicated sealed test set, "
                "distinct from the training-time validation set used for checkpoint/LR "
                "selection; historical_reporting_convention_value is preserved separately "
                "for comparability with the documented baseline and is not the EV outcome metric."
            ],
            "failure_status": "valid",
        }
        return ReplicateExecutionResult(outcome=outcome, manifest_fields=manifest_fields)
