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
from audit.cryptography.gohr.dataset import DatasetBundle, GohrDataset
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
    ) -> None:

        self._dataset = dataset or GohrDataset()
        self._model_factory = model or GohrModel()
        self._trainer = trainer or GohrTrainer()
        self._evaluator = evaluator or GohrEvaluator()
        self._model_path = model_path
        self._theory_num_samples = theory_num_samples

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