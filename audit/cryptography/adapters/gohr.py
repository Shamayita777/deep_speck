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

from keras.models import Model, load_model

from audit.cryptography.adapters.base import CryptographicAdapter

from audit.cryptography.gohr.dataset import DatasetBundle, GohrDataset
from audit.cryptography.gohr.model import GohrModel
from audit.cryptography.gohr.trainer import GohrTrainer
from audit.cryptography.gohr.evaluate import GohrEvaluator


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
    ) -> None:

        self._dataset = dataset or GohrDataset()
        self._model_factory = model or GohrModel()
        self._trainer = trainer or GohrTrainer()
        self._evaluator = evaluator or GohrEvaluator()

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
        path: str | Path,
    ) -> Model:
        """
        Load a previously trained model.
        """

        return load_model(path)

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