"""
Cryptographic Probe Interface
=============================

Abstract interface for representation probes used by the
Cryptographic Evidence framework.

A cryptographic probe is a lightweight predictive model trained
on frozen internal representations in order to determine whether
a specific cryptographic quantity is encoded within the learned
representation.

The framework intentionally separates

    representation extraction

from

    probing

and

    evidence evaluation.

Consequently, probes know nothing about

    • cryptographic primitives
    • neural-network architectures
    • ciphertexts
    • plaintexts
    • Gohr
    • Speck
    • CE1
    • CE2

They receive only

    • representation vectors

and

    • target labels.

Concrete implementations may include

    • Logistic Regression
    • Ridge Regression
    • Linear SVM
    • k-Nearest Neighbours

or future probing algorithms.

Responsibilities
----------------
• Learn a mapping from representations to targets
• Evaluate predictive performance
• Remain completely independent of cryptographic
  implementations
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np


class CryptographicProbe(ABC):
    """
    Abstract base class for all representation probes.

    A probe receives frozen representation vectors together
    with an independently defined cryptographic target and
    evaluates whether that quantity is decodable from the
    representation.

    The framework intentionally places no restrictions on the
    underlying probing algorithm.
    """

    # ---------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        *,
        random_state: int | None = None,
    ) -> None:
        """
        Initialise the probe.

        Parameters
        ----------
        random_state
            Random seed supplied by the framework.

        Notes
        -----
        The framework supplies identical seeds to the probes
        trained on the real and control representations in
        order to isolate representation quality from probe
        randomness.
        """
        raise NotImplementedError
    
    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        train_representations: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        """
        Fit the probe.

        Parameters
        ----------
        train_representations
            Training representation matrix.

        train_labels
            Ground-truth labels corresponding to the training
            representations.

        Notes
        -----
        The Cryptographic Evidence framework performs all
        dataset partitioning. Probes receive only the training
        partition and therefore remain independent of the
        evaluation protocol.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    @abstractmethod
    def predict(
        self,
        test_representations: np.ndarray,
    ) -> np.ndarray:
        """
        Predict target values.

        Parameters
        ----------
        test_representations
            Test representation matrix.

        Returns
        -------
        np.ndarray
            Probe predictions.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable probe name.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Brief description of the probing algorithm.
        """
        raise NotImplementedError