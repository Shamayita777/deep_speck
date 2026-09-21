"""
Gohr Evaluator
==============

Evaluation utilities for the Gohr neural distinguisher.

This module evaluates a trained neural network on a supplied
dataset. It performs no model construction, dataset generation,
training, or persistence.

Responsibilities
----------------
• Evaluate a trained model
• Return the evaluation metric(s)
• Remain independent of the Cryptographic Evidence framework
"""

from __future__ import annotations

from typing import Any

import numpy as np
from keras.models import Model


class GohrEvaluator:
    """
    Evaluator for the Gohr neural distinguisher.
    """

    def __init__(
        self,
        *,
        batch_size: int = 5000,
    ) -> None:
        """
        Initialise the evaluator.

        Parameters
        ----------
        batch_size
            Batch size used during evaluation.
        """

        self.batch_size = batch_size

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        model: Model,
        dataset: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """
        Evaluate a trained model.

        Parameters
        ----------
        model
            Trained Keras model.

        dataset
            Tuple of (X, Y).

        Returns
        -------
        float
            Classification accuracy.
        """

        X, Y = dataset

        _, accuracy = model.evaluate(
            X,
            Y,
            batch_size=self.batch_size,
            verbose=0,
        )

        return float(accuracy)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        model: Model,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Generate predictions for input samples.

        Parameters
        ----------
        model
            Trained Keras model.

        X
            Input samples.

        Returns
        -------
        numpy.ndarray
            Model predictions.
        """

        predictions = model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0,
        )

        return predictions

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Human-readable evaluator name.
        """
        return "Gohr Evaluator"

    @property
    def description(self) -> str:
        """
        Short evaluator description.
        """
        return (
            "Evaluation component for the Gohr neural "
            "distinguisher."
        )