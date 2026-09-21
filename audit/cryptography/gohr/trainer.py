"""
Gohr Trainer
============

Training utilities for the Gohr neural distinguisher.

This module is responsible solely for model training.
It is intentionally independent of dataset generation,
evaluation, persistence, and the Cryptographic Evidence
framework.

Responsibilities
----------------
• Train a supplied neural network
• Configure callbacks
• Manage learning-rate scheduling
• Return the trained model and training history
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
)
from keras.models import Model


class GohrTrainer:
    """
    Trainer for the Gohr neural distinguisher.
    """

    def __init__(
        self,
        *,
        batch_size: int = 5000,
        epochs: int = 200,
        checkpoint_dir: str | Path = "./checkpoints",
        save_best_only: bool = True,
        high_learning_rate: float = 0.002,
        low_learning_rate: float = 0.0001,
    ) -> None:

        self.batch_size = batch_size
        self.epochs = epochs

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.save_best_only = save_best_only

        self.high_learning_rate = high_learning_rate
        self.low_learning_rate = low_learning_rate

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set random seeds for reproducible training.
        """

        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)

        np.random.seed(seed)

        tf.random.set_seed(seed)

    # ---------------------------------------------------------
    # Learning Rate Schedule
    # ---------------------------------------------------------

    def learning_rate(
        self,
        epoch: int,
    ) -> float:
        """
        Cyclic learning-rate schedule.
        """

        return (
            self.low_learning_rate
            + (
                (self.epochs - 1)
                - (epoch % self.epochs)
            )
            / (self.epochs - 1)
            * (
                self.high_learning_rate
                - self.low_learning_rate
            )
        )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    def callbacks(
        self,
        checkpoint_name: str,
    ) -> list[Any]:
        """
        Construct Keras callbacks.
        """

        checkpoint = ModelCheckpoint(

            filepath=str(
                self.checkpoint_dir / checkpoint_name
            ),

            monitor="val_loss",

            save_best_only=self.save_best_only,

            verbose=0,
        )

        scheduler = LearningRateScheduler(
            self.learning_rate
        )

        return [
            checkpoint,
            scheduler,
        ]

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(
        self,
        model: Model,
        train_dataset: tuple[np.ndarray, np.ndarray],
        validation_dataset: tuple[np.ndarray, np.ndarray],
        *,
        seed: int = 0,
        checkpoint_name: str = "best_model.keras",
    ) -> tuple[Model, Any]:
        """
        Train the supplied model.

        Parameters
        ----------
        model
            Compiled Keras model.

        train_dataset
            Training data (X, Y).

        validation_dataset
            Validation data (X, Y).

        seed
            Random seed.

        checkpoint_name
            Checkpoint filename.

        Returns
        -------
        tuple
            (trained_model, history)
        """

        self.set_seed(seed)

        X_train, Y_train = train_dataset

        X_val, Y_val = validation_dataset

        history = model.fit(

            X_train,

            Y_train,

            validation_data=(
                X_val,
                Y_val,
            ),

            epochs=self.epochs,

            batch_size=self.batch_size,

            callbacks=self.callbacks(
                checkpoint_name,
            ),

            verbose=1,
        )

        return model, history

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "Gohr Trainer"

    @property
    def description(self) -> str:
        return (
            "Training component for the Gohr neural "
            "distinguisher."
        )