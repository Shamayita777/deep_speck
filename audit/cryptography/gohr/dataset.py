"""
Gohr Dataset Utilities
======================

Dataset generation utilities for the Gohr neural distinguisher.

This module provides the datasets required by the Cryptographic
Evidence framework while remaining completely independent of the
framework itself.

Responsibilities
----------------
• Generate baseline training and validation datasets
• Generate signal-destroyed training and validation datasets

The baseline dataset reproduces the original Gohr training
procedure.

The signal-destroyed dataset preserves the data format while
destroying the cryptographic relationship between samples.

References
----------
Gohr, A. (2019)
Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import speck as sp


@dataclass(frozen=True)
class DatasetBundle:
    """
    Container for the datasets required by one experiment.
    """

    train: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]


class GohrDataset:
    """
    Dataset generator for the Gohr neural distinguisher.
    """

    def __init__(
        self,
        rounds: int = 7,
        train_samples: int = 10**7,
        validation_samples: int = 10**6,
        differential: tuple[int, int] = (0x0040, 0x0000),
    ) -> None:

        self.rounds = rounds
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.differential = differential

    # ---------------------------------------------------------
    # Internal Dataset Generation
    # ---------------------------------------------------------

    def _generate_dataset(
        self,
        samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a Gohr dataset.
        """

        X, Y = sp.make_train_data(
            samples,
            self.rounds,
            diff=self.differential,
        )

        return X, Y

    # ---------------------------------------------------------
    # Baseline Dataset
    # ---------------------------------------------------------

    def generate_baseline_dataset(
        self,
    ) -> DatasetBundle:
        """
        Generate the baseline experimental dataset.
        """

        train_dataset = self._generate_dataset(
            self.train_samples,
        )

        validation_dataset = self._generate_dataset(
            self.validation_samples,
        )

        return DatasetBundle(
            train=train_dataset,
            validation=validation_dataset,
        )

    # ---------------------------------------------------------
    # Signal Destruction Dataset
    # ---------------------------------------------------------

    def generate_signal_destroyed_dataset(
        self,
    ) -> DatasetBundle:
        """
        Generate a signal-destroyed experimental dataset.

        The feature representation is preserved while the
        cryptographic relationship between inputs and labels is
        intentionally destroyed.
        """

        baseline = self.generate_baseline_dataset()

        rng = np.random.default_rng()

        X_train, Y_train = baseline.train
        X_val, Y_val = baseline.validation

        destroyed_train = (
            X_train,
            rng.permutation(Y_train),
        )

        destroyed_validation = (
            X_val,
            rng.permutation(Y_val),
        )

        return DatasetBundle(
            train=destroyed_train,
            validation=destroyed_validation,
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "Gohr Dataset"

    @property
    def description(self) -> str:
        return (
            "Dataset generator for the Gohr neural "
            "distinguisher experiments."
        )