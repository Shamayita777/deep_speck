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
import numpy as np

class GohrAdapter(CryptographicAdapter):
    """
    Adapter for the Gohr neural distinguisher.
    """

    def __init__(
        self,
        dataset: GohrDataset | None = None,
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
    ) -> None:

        self._dataset = dataset or GohrDataset()
        self._model_factory = model or GohrModel()
        self._trainer = trainer or GohrTrainer()
        self._evaluator = evaluator or GohrEvaluator()
        self._model_path = model_path
        self._control_model_path = control_model_path
        self._theory_num_samples = theory_num_samples
        self._baseline_model_path = baseline_model_path
        self._signal_destroyed_model_path = signal_destroyed_model_path

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

    def generate_representation_tasks(
        self,
    ) -> list[RepresentationTask]:
        """
        Generate the representation probing tasks required
        for CE3.

        Returns
        -------
        list[RepresentationTask]
            Calibration and primary probing tasks.
        """

        # ---------------------------------------------
        # Calibration task
        # ---------------------------------------------

        calibration_dataset = (
            self._dataset.generate_calibration_dataset(
                num_samples=self._theory_num_samples,
            )
        )

        calibration_target = TargetSpecification(
            name="Differential Class",
            description=(
                "Binary differential-class labels used to "
                "verify the probing pipeline."
            ),
            target_type=TargetType.BINARY,
            labels=calibration_dataset.differential_labels,
        )

        # ---------------------------------------------
        # Primary task
        # ---------------------------------------------

        theory_dataset = self.generate_theory_dataset()

        probabilities = (
            theory_dataset.theoretical_probabilities
        )

        quantiles = np.quantile(
            probabilities,
            [0.2, 0.4, 0.6, 0.8],
        )

        probability_labels = np.digitize(
            probabilities,
            bins=quantiles,
            right=False,
        )

        primary_target = TargetSpecification(
            name="Analytical Trail Probability",
            description=(
                "Discretized analytical trail probability."
            ),
            target_type=TargetType.MULTICLASS,
            labels=probability_labels.astype(np.int64),
        )

        return [

            RepresentationTask(
                dataset=calibration_dataset,
                target=calibration_target,
                is_primary=False,
            ),

            RepresentationTask(
                dataset=theory_dataset,
                target=primary_target,
                is_primary=True,
            ),
        ]    

    def generate_theory_dataset(self):
        """
        Generate the CE2 theory dataset.

        Reuses this adapter's own `rounds` and `differential` —
        the same values the model was trained with — rather than
        accepting them as free parameters, so CE2 cannot
        accidentally be run against a round-count or differential
        that mismatches the frozen model.
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