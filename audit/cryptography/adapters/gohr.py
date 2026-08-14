"""
Gohr Adapter
============

Adapter connecting the Gohr neural distinguisher to the
Cryptographic Evidence framework.

This adapter orchestrates the Gohr implementation by delegating
dataset generation, model construction, training, evaluation,
and persistence to their respective components.

It contains no cryptographic logic, machine learning logic, or
experimental logic. Its sole responsibility is to translate the
framework interface into calls to the Gohr implementation.
"""

from __future__ import annotations
from pathlib import Path
from statistics import quantiles
from typing import Any
from keras.models import Model, load_model
from audit.cryptography.gohr import speck
from scipy.stats import spearmanr
from audit.cryptography.adapters.base import CryptographicAdapter
from audit.cryptography.gohr import dataset
from audit.cryptography.gohr.dataset import (
    CalibrationDataset,
    DatasetBundle,
    GohrDataset,
    TheoryDataset,
)
from audit.cryptography.test.ce3.types import (
    RepresentationTask,
    TargetSpecification,
    TargetType,
)
from audit.cryptography.gohr.model import GohrModel
from audit.cryptography.gohr.trainer import GohrTrainer
from audit.cryptography.gohr.evaluate import GohrEvaluator

import dataclasses
from audit.cryptography.test.ce4.types import InterventionTask
import numpy as np

class GohrAdapter(CryptographicAdapter):
    """
    Adapter for the Gohr neural distinguisher.
    """

    def __init__(
        self,
        dataset: GohrDataset,
        model: GohrModel | None = None,
        trainer: GohrTrainer | None = None,
        evaluator: GohrEvaluator | None = None,
        model_path: str | Path | None = None,
        theory_num_samples: int = 10**5,
        control_model_path: str | Path | None = None,
        baseline_model_path: str | Path | None = (
            "audit/cryptography/evidence/ce1/best5depth10 (10).h5"
        ),
        signal_destroyed_model_path: str | Path | None = (
            "audit/cryptography/evidence/ce1/signal_destroyed.h5"
        ),
        n_intervention_bits: int = 8,
        intervention_seed: int = 0,
    ) -> None:
        if dataset is None:
            raise ValueError(
                "GohrAdapter requires an explicit GohrDataset. "
                "Specify rounds and differential explicitly."
            )

        self._dataset = dataset
        self._model_factory = model or GohrModel()
        self._trainer = trainer or GohrTrainer()
        self._evaluator = evaluator or GohrEvaluator()
        self._model_path = model_path
        self._control_model_path = control_model_path
        self._theory_num_samples = theory_num_samples
        self._baseline_model_path = baseline_model_path
        self._signal_destroyed_model_path = signal_destroyed_model_path
        self._n_intervention_bits = n_intervention_bits
        self._intervention_seed = intervention_seed

        self._last_bundle: DatasetBundle | None = None

    # ---------------------------------------------------------
    # Dataset Generation
    # ---------------------------------------------------------

    def generate_baseline_dataset(
        self,
    ) -> DatasetBundle:
        """
        Generate the baseline experimental dataset.
        """

        bundle = self._dataset.generate_baseline_dataset()

        self._last_bundle = bundle

        return bundle

    def generate_signal_destroyed_dataset(
        self,
    ) -> DatasetBundle:
        """
        Generate the signal-destroyed experimental dataset.
        """

        bundle = (
            self._dataset.generate_signal_destroyed_dataset()
        )

        self._last_bundle = bundle

        return bundle

    def extract_representations(
        self,
        model: Model,
        dataset: Any,
    ) -> np.ndarray:
        """
        Extract frozen hidden representations for CE3.

        The representation is taken from the final hidden layer,
        immediately before the output classifier.
        """

        representation_model = Model(
            inputs=model.input,
            outputs=model.layers[-2].output,
        )

        representations = representation_model.predict(
            dataset.X,
            verbose=0,
        )

        return np.asarray(representations)
    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(
        self,
        dataset: DatasetBundle,
    ) -> Model:
        """
        Train a Gohr neural distinguisher.
        """

        model = self._model_factory.build()

        model, _ = self._trainer.train(
            model=model,
            train_dataset=dataset.train,
            validation_dataset=dataset.validation,
        )

        return model

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        model: Model,
    ) -> float:
        """
        Evaluate a trained model using the validation dataset.
        """

        if self._last_bundle is None:
            raise RuntimeError(
                "No dataset has been generated."
            )

        return self._evaluator.evaluate(
            model,
            self._last_bundle.validation,
        )

    def generate_primary_representation_task(
        self
    ) -> RepresentationTask:

        theory_dataset = self.generate_theory_dataset()

        probabilities = theory_dataset.theoretical_probabilities

        quantiles = np.quantile(
            probabilities, [0.2, 0.4, 0.6, 0.8],
        )
        probability_labels = np.digitize(
            probabilities, bins=quantiles, right=False,
        )

        target = TargetSpecification(
            name="Analytical Trail Probability",
            description="Discretized analytical trail probability.",
            target_type=TargetType.MULTICLASS,
            labels=probability_labels.astype(np.int64),
            theoretical_interpretation=(
                "Represents the analytical trail probability "
                "predicted by differential cryptanalysis. "
                "Successful decoding indicates that the learned "
                "representation encodes information correlated "
                "with the theoretical attack model."
            ),
        )

        return RepresentationTask(
            dataset=theory_dataset, target=target, is_primary=True,
        )


    def generate_calibration_representation_task(
        self
    ) -> RepresentationTask:

        calibration_dataset = (
            self._dataset.generate_calibration_dataset(
                num_samples=self._theory_num_samples,
            )
        )

        target = TargetSpecification(
            name="Differential Class",
            description=(
                "Binary differential-class labels used to verify "
                "the probing pipeline."
            ),
            target_type=TargetType.BINARY,
            labels=calibration_dataset.differential_labels,
            theoretical_interpretation=(
                "Indicates whether each sample belongs to the true "
                "differential distribution or the random "
                "distribution. Serves as a calibration task to "
                "verify the probing pipeline recovers a quantity "
                "known to be represented."
            ),
        )

        return RepresentationTask(
            dataset=calibration_dataset, target=target, is_primary=False,
        )


    def generate_representation_tasks(
        self,
    ) -> list[RepresentationTask]:
        """
        Retained for CE4's dependency on a single, fixed
        (dataset, target) pair. NOTE: this now returns ONE arbitrary
        instance among what CE3 treats as many independent
        replicates -- CE4 should be revisited to either call
        generate_primary_representation_task() directly for its own
        fresh draw, or explicitly document that it audits one
        representative instance rather than "the" CE3 dataset.
        """
        return [
            self.generate_calibration_representation_task(),
            self.generate_primary_representation_task(),
        ]

    def generate_theory_dataset(self):
        """
        Generate the CE2 theory dataset.

        Reuses this adapter's explicitly configured `rounds` and
        `differential` rather than accepting them as free parameters,
        so all cryptographic audit stages operate on the same declared
        configuration as the frozen model.
        """

        return self._dataset.generate_theory_dataset(
            num_samples=self._theory_num_samples,
            rounds=self._dataset.rounds,
        )

    def compute_theoretical_reference(
        self,
        dataset,
    ):
        return dataset.theoretical_probabilities

    def compute_model_predictions(self, model, dataset):
        prediction_dataset = self._dataset.generate_prediction_dataset(
            dataset
        )

        predictions = self._evaluator.predict(
            model, prediction_dataset.X,
        )

        return predictions.ravel()

    def compute_theory_consistency(
        self, theoretical_reference, model_predictions,
    ):
        from audit.cryptography.results import CorrelationStatistic

        if (
            np.std(theoretical_reference) == 0.0
            or np.std(model_predictions) == 0.0
        ):
            raise ValueError(
                "Theory Consistency Score is undefined: zero "
                "variance in the theoretical reference or the "
                "model predictions (all evaluated samples "
                "received an identical value)."
            )
        print("Theory reference:")
        print(theoretical_reference[:20])
        print()

        print("Model predictions:")
        print(model_predictions[:20])
        print()

        print("Theory unique:", len(set(theoretical_reference)))
        print("Prediction unique:", len(set(model_predictions)))
        score, p_value = spearmanr(
            theoretical_reference, model_predictions,
        )

        return CorrelationStatistic(
            statistic=float(score),
            p_value=float(p_value),
            n=int(len(theoretical_reference)),
        )

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(
        self,
        model: Model,
        path: str | Path,
    ) -> None:
        """
        Save a trained model.
        """

        model.save(path)

    def load(
        self,
        path: str | Path | None = None,
    ) -> Model:
        """
        Load a previously trained model.
        """

        if path is None:
            path = self._model_path

        if path is None:
            raise RuntimeError(
                "No model path provided."
            )

        return load_model(
            path,
            compile=False,
        )

    def provide_control_model(self) -> Model:
        """
        Load the CE1 signal-destroyed control model.

        Returns
        -------
        Model
            Frozen signal-destroyed model used as the CE3 baseline.
        """

        if self._signal_destroyed_model_path is None:
            raise RuntimeError(
                "No signal-destroyed model path provided."
            )

        return load_model(
            self._signal_destroyed_model_path,
            compile=False,
        )

    # ---------------------------------------------------------
    # CE4 — Causal Intervention
    # ---------------------------------------------------------

    @staticmethod
    def _mirrored_pairs() -> np.ndarray:
        """
        Column-index pairs (c0_bit, c1_bit) referring to the
        same bit position within the same 16-bit Speck word,
        under the exact column layout produced by
        speck.convert_to_binary([c0l, c0r, c1l, c1r]):
        columns 0-15 = c0l, 16-31 = c0r, 32-47 = c1l, 48-63 = c1r.

        Flipping both columns of a pair simultaneously changes
        both ciphertexts' absolute values at that bit position
        while leaving their XOR difference at that position
        provably unchanged: (a^1)^(b^1) == a^b.
        """

        word = 16
        left_pairs = [(j, 2 * word + j) for j in range(word)]
        right_pairs = [(word + j, 3 * word + j) for j in range(word)]
        return np.array(left_pairs + right_pairs)

    def generate_intervention_tasks(self) -> list[InterventionTask]:
        """
        Generate CE4's intervention task by reusing the exact
        (dataset, target) object CE3 marked as primary, rather
        than regenerating -- guaranteeing CE4 audits the same
        rounds, differential, and target CE2/CE3 already audited.
        """

        representation_tasks = self.generate_representation_tasks()

        primary = next(
            t for t in representation_tasks if t.is_primary
        )

        return [
            InterventionTask(
                dataset=primary.dataset,
                target=primary.target,
                is_primary=True,
            ),
        ]

    def apply_structural_intervention(self, dataset, target):
        """
        Flip `n_intervention_bits` independently random single
        columns per sample. Each single-sided flip alters the
        realized ciphertext-pair difference at that bit
        position -- this directly targets the theoretically
        relevant structure the analytical trail probability
        (declared by `target`) is defined over.
        """

        rng = np.random.default_rng(self._intervention_seed)

        X = dataset.X.copy()
        n_samples, n_bits = X.shape
        k = self._n_intervention_bits

        random_vals = rng.random((n_samples, n_bits))
        columns = np.argpartition(random_vals, k, axis=1)[:, :k]

        mask = np.zeros_like(X, dtype=bool)
        rows = np.arange(n_samples)[:, None]
        mask[rows, columns] = True

        X_perturbed = X ^ mask.astype(X.dtype)

        return dataclasses.replace(dataset, X=X_perturbed)

    def apply_control_intervention(self, dataset, target):
        """
        Flip `n_intervention_bits // 2` independently random
        mirrored bit-position PAIRS per sample (both ciphertexts'
        bit at the same position, flipped together). This changes
        the same total number of raw input bits as
        `apply_structural_intervention`, but leaves the realized
        ciphertext-pair difference completely unchanged at every
        flipped position -- a magnitude-matched perturbation that
        is, by construction, invisible to the declared target.
        """

        rng = np.random.default_rng(self._intervention_seed + 1)

        X = dataset.X.copy()
        n_samples, _ = X.shape
        k_pairs = self._n_intervention_bits // 2

        pairs = self._mirrored_pairs()
        n_pairs = pairs.shape[0]

        random_vals = rng.random((n_samples, n_pairs))
        pair_idx = np.argpartition(
            random_vals, k_pairs, axis=1,
        )[:, :k_pairs]

        selected_pairs = pairs[pair_idx]

        mask = np.zeros_like(X, dtype=bool)
        rows = np.arange(n_samples)[:, None, None]
        mask[rows, selected_pairs] = True

        X_perturbed = X ^ mask.astype(X.dtype)

        return dataclasses.replace(dataset, X=X_perturbed)

    def compute_intervention_magnitude(
        self, original_dataset, perturbed_dataset,
    ) -> np.ndarray:
        """
        Per-sample count of changed input bits, generic over any
        dataset exposing `.X` with matching shape. By
        construction, both interventions above return exactly
        `n_intervention_bits` here for every sample.
        """

        return np.sum(
            perturbed_dataset.X != original_dataset.X, axis=1,
        ).astype(np.float64)
    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "Gohr Adapter"

    @property
    def description(self) -> str:
        return (
            "Cryptographic adapter for the Gohr neural "
            "distinguisher."
        )