"""
Cryptographic Test Interface
============================

Abstract interface for all Cryptographic Evidence (CE) audit tests.

The Cryptographic Evidence framework evaluates whether a machine
learning model derives its predictive performance from meaningful
cryptographic structure rather than unrelated statistical artefacts.

Each audit procedure (e.g., Signal Destruction, Theory Consistency,
Representation Interpretation) is implemented as a concrete subclass
of ``CryptographicTest``.

This module intentionally contains no implementation logic. It defines
only the contract that every cryptographic audit test must satisfy,
allowing individual tests to remain independent of any specific
cryptographic primitive, machine-learning model, or research paper.

Design Principles
-----------------
• Framework-level abstraction only.
• Independent of Gohr, Speck, AES, Simon, side-channel traces, etc.
• Stable API shared by all Cryptographic Evidence tests.
• Compatible with future CE2, CE3, CE4, and additional audit procedures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from enum import Enum

class EvidenceDirection(Enum):
    """
    Indicates whether larger or smaller observed effects
    provide stronger support for the experimental hypothesis.
    """

    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"

    LOWER_IS_BETTER = "LOWER_IS_BETTER"

class CryptographicTest(ABC):
    """
    Abstract base class for Cryptographic Evidence audit tests.

    A CryptographicTest represents a hypothesis-driven audit procedure
    that evaluates whether a machine-learning model relies upon a
    specific cryptographic property.

    Concrete subclasses should implement only the logic required for
    a particular audit procedure (e.g., signal destruction). They must
    not assume any specific cipher, dataset, representation, or model.

    Every cryptographic test exposes:

    name
        Human-readable name of the audit procedure.

    description
        Short explanation of the purpose of the test.

    hypothesis
        Scientific hypothesis evaluated by the audit.

    validate(adapter)
        Verify that the supplied adapter satisfies the interface
        required by this audit.

    run(adapter)
        Execute the audit and return a framework-defined result object.
        The exact return type is intentionally unspecified here because
        it belongs to later stages of the framework.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        hypothesis: str,
        evidence_direction: EvidenceDirection,
    ) -> None:
        """
        Initialise a cryptographic audit test.

        Parameters
        ----------
        name
            Name of the audit procedure.

        description
            Brief description of the audit.

        hypothesis
            Scientific hypothesis evaluated by the audit.
        """

        self.name = name
        self.description = description
        self.hypothesis = hypothesis
        self.evidence_direction = evidence_direction

    # ---------------------------------------------------------
    # Scientific Interpretation
    # ---------------------------------------------------------

    @property
    def supported_rationale(self) -> str:
        """
        Scientific interpretation when the hypothesis is
        supported.
        """
        return (
            "Observed evidence exceeds the support threshold."
        )

    @property
    def inconclusive_rationale(self) -> str:
        """
        Scientific interpretation when the evidence is
        inconclusive.
        """
        return (
            "Observed evidence is insufficient for strong "
            "support."
        )

    @property
    def unsupported_rationale(self) -> str:
        """
        Scientific interpretation when the hypothesis is not
        supported.
        """
        return (
            "Observed evidence is too weak to support the "
            "hypothesis."
        )

    # ---------------------------------------------------------
    # Abstract Interface
    # ---------------------------------------------------------

    @abstractmethod
    def validate(
        self,
        adapter: Any,
    ) -> None:
        """
        Validate that the supplied adapter supports this audit.

        Each audit may require different capabilities from an adapter.
        Implementations should verify those requirements before the
        experiment begins and raise an informative exception if the
        adapter is incompatible.

        Parameters
        ----------
        adapter
            Paper-specific cryptographic adapter.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        TypeError
            If the supplied adapter does not satisfy the required
            interface.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        adapter: Any,
    ) -> Any:
        """
        Execute the cryptographic audit.

        Implementations should perform the complete audit procedure,
        including any required validation, experiment execution, and
        result generation.

        Parameters
        ----------
        adapter
            Paper-specific cryptographic adapter.

        Returns
        -------
        Any
            Framework-defined result object produced by the audit.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return the audit test name."""
        return self.name

    def __repr__(self) -> str:
        """Return an unambiguous string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"hypothesis={self.hypothesis!r})"
        )