"""
Cryptographic Adapter Interface
===============================

Abstract adapter interface for the Cryptographic Evidence (CE)
framework.

The adapter isolates paper-specific implementations from the
framework. Each cryptographic case study (e.g., Gohr's neural
distinguisher, side-channel analysis, or future cryptographic
learning systems) must implement this interface.

The framework never interacts directly with cipher-specific code,
datasets, or machine-learning pipelines. Instead, it communicates
exclusively through this abstraction.

Responsibilities
----------------
A concrete adapter is responsible for

    • generating the baseline experiment

    • generating the signal-destroyed experiment

    • training a model

    • evaluating a trained model

    • saving and loading trained models

The framework intentionally remains unaware of how these tasks are
performed.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class CryptographicAdapter(ABC):
    """
    Abstract interface implemented by every cryptographic
    case-study adapter.

    The adapter encapsulates all implementation-specific
    knowledge required to execute a cryptographic audit.

    Examples
    --------
    • Gohr neural distinguisher
    • Side-channel analysis framework
    • Learned cryptanalysis of another block cipher
    • Future cryptographic ML systems
    """

    # ---------------------------------------------------------
    # Dataset Generation
    # ---------------------------------------------------------

    @abstractmethod
    def generate_baseline_dataset(self) -> Any:
        """
        Generate the baseline experiment.

        Returns
        -------
        Any
            Dataset or experiment representation required
            for model training.

        Notes
        -----
        The returned object is intentionally unspecified.
        Different cryptographic systems may use entirely
        different internal representations.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_signal_destroyed_dataset(self) -> Any:
        """
        Generate a signal-destroyed version of the experiment.

        The adapter is responsible for removing the
        cryptographic relationship while preserving all
        remaining experimental conditions.

        Returns
        -------
        Any
            Signal-destroyed experiment.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------

    @abstractmethod
    def train(
        self,
        dataset: Any,
    ) -> Any:
        """
        Train a model on the supplied experiment.

        Parameters
        ----------
        dataset
            Experiment produced by one of the dataset
            generation methods.

        Returns
        -------
        Any
            Trained model.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        model: Any,
    ) -> float:
        """
        Evaluate a trained model.

        Parameters
        ----------
        model
            Trained machine-learning model.

        Returns
        -------
        float
            Evaluation metric (e.g., accuracy).
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    @abstractmethod
    def save(
        self,
        model: Any,
        path: str,
    ) -> None:
        """
        Persist a trained model.

        Parameters
        ----------
        model
            Trained model.

        path
            Destination path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        path: str,
    ) -> Any:
        """
        Load a previously saved model.

        Parameters
        ----------
        path
            Model location.

        Returns
        -------
        Any
            Loaded model.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the cryptographic
        implementation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Brief description of the implementation.
        """
        raise NotImplementedError