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

    • generating audit-specific datasets

    • training a model

    • evaluating a trained model

    • computing theory-derived reference values

    • computing theory consistency metrics

    • saving and loading trained models

The framework intentionally remains unaware of how these tasks are
performed.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from ..results import CorrelationStatistic

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
    # Theory Consistency (CE2)
    # ---------------------------------------------------------

    @abstractmethod
    def generate_theory_dataset(self) -> Any:
        """
        Generate the evaluation dataset used for theory
        consistency analysis.

        Returns
        -------
        Any
            Dataset suitable for comparing theoretical
            expectations with model behaviour.
        """
        raise NotImplementedError


    @abstractmethod
    def compute_theoretical_reference(
        self,
        dataset: Any,
    ) -> Any:
        """
        Compute the independent theoretical reference for the
        supplied dataset.

        The theoretical reference must be derived from
        established cryptographic analysis rather than from
        the learned model itself.

        Parameters
        ----------
        dataset
            Evaluation dataset.

        Returns
        -------
        Any
            Theory-derived reference values.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_model_predictions(
        self,
        model: Any,
        dataset: Any,
    ) -> Any:
        """
        Compute the model outputs used for theory comparison.

        Parameters
        ----------
        model
            Trained model.

        dataset
            Evaluation dataset.

        Returns
        -------
        Any
            Model predictions corresponding to the supplied
            dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_theory_consistency(
        self,
        theoretical_reference: Any,
        model_predictions: Any,
    ) -> "CorrelationStatistic":
        """
        Compute the Theory Consistency Score (TCS).

        Returns
        -------
        CorrelationStatistic
            Correlation coefficient, its p-value, and the
            sample size it was computed over.
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
        path: str | None = None,
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