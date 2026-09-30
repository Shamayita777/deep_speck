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
        lr_cycle_length: int = 10,
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

        # Reference protocol: train_nets.py uses cyclic_lr(10, 0.002, 0.0001),
        # i.e. a CYCLE LENGTH of 10 epochs (20 sawtooth cycles across a
        # 200-epoch run). This was previously taken from self.epochs, which
        # collapsed the schedule into a single 200-epoch ramp - a silent
        # deviation from the reference training protocol.
        if lr_cycle_length < 2:
            raise ValueError(
                f"lr_cycle_length must be >= 2 (the reference uses 10); got {lr_cycle_length}")
        self.lr_cycle_length = lr_cycle_length

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

        Identical to the reference `cyclic_lr(num_epochs, high_lr, low_lr)`
        in train_nets.py, with `num_epochs` being the CYCLE LENGTH (10 in
        the reference), not the total number of training epochs.
        """

        cycle = self.lr_cycle_length

        return (
            self.low_learning_rate
            + (
                (cycle - 1)
                - (epoch % cycle)
            )
            / (cycle - 1)
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
        initial_best_val_loss: float | None = None,
    ) -> list[Any]:
        """
        Construct Keras callbacks.

        `initial_best_val_loss` restores the best-so-far threshold of the
        DEBUG-ONLY best-val checkpoint after a resume, so the first resumed
        epoch does not overwrite it merely because a fresh callback starts at
        +inf. It has no effect on the FINAL_EPOCH estimand.
        """

        checkpoint = ModelCheckpoint(

            initial_value_threshold=initial_best_val_loss,

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
        initial_epoch: int = 0,
        terminal_epoch: int | None = None,
        extra_callbacks: list | None = None,
        initial_best_val_loss: float | None = None,
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

        # RESUME SUPPORT. `initial_epoch` is the number of epochs ALREADY
        # completed, so training continues at initial_epoch + 1 in 1-based
        # terms and epochs 1..initial_epoch are never retrained. The
        # terminal epoch is the frozen scientific endpoint (200), not a
        # count of epochs executed in this process.
        epochs = self.epochs if terminal_epoch is None else terminal_epoch
        if initial_epoch < 0:
            raise ValueError("initial_epoch must be >= 0")
        if initial_epoch >= epochs:
            raise ValueError(
                f"initial_epoch={initial_epoch} is already at or beyond the terminal "
                f"epoch {epochs}; nothing to train. The caller must skip or evaluate "
                "this arm rather than re-entering training.")

        X_train, Y_train = train_dataset

        X_val, Y_val = validation_dataset

        history = model.fit(

            X_train,

            Y_train,

            validation_data=(
                X_val,
                Y_val,
            ),

            epochs=epochs,

            initial_epoch=initial_epoch,

            batch_size=self.batch_size,

            callbacks=self.callbacks(checkpoint_name, initial_best_val_loss)
            + list(extra_callbacks or []),

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