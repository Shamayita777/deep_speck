"""
Cryptographic Evidence Framework
================================

Framework-level result objects shared by all Cryptographic
Evidence (CE) audit procedures.

This module defines immutable result structures representing
the outcome of cryptographic audit experiments. These classes
are intentionally independent of any particular cryptographic
primitive, machine-learning model, dataset, or research paper.

The framework separates:

    • Experimental measurements
        (CryptographicTestResult)

from

    • Scientific interpretation
        (EvaluationResult)

This separation allows different evaluation policies to be
applied without modifying the underlying experimental record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from audit.cryptography.test.base import EvidenceDirection

# ============================================================
# Evaluation Decision
# ============================================================


class EvaluationDecision(Enum):
    """
    Scientific decision produced after evaluating a
    cryptographic audit result.
    """

    SUPPORTED = "SUPPORTED"

    NOT_SUPPORTED = "NOT_SUPPORTED"

    INCONCLUSIVE = "INCONCLUSIVE"


# ============================================================
# Experimental Result
# ============================================================


@dataclass(frozen=True, slots=True)
class CryptographicTestResult:
    """
    Immutable result produced by a cryptographic audit
    experiment.

    The result records a reference measurement, an observed
    measurement, and their quantitative relationship.

    The semantic interpretation of these measurements is
    defined by the corresponding audit procedure. For example,
    Signal Destruction interprets them as baseline and
    signal-destroyed performance, whereas Theory Consistency
    interprets them as theoretical and observed behaviour.

    This class stores only experimentally observed values.
    No scientific interpretation is performed here.
    """

    test_name: str

    evidence_direction: EvidenceDirection
    # Reference measurement defined by the audit.
    baseline_score: float
    # Observed measurement produced by the audit.
    test_score: float
    # Absolute deviation between the reference and observation.
    performance_drop: float
    # Normalized deviation.
    relative_difference: float
    runtime: float
    p_value: float | None = None
    sample_size: int | None = None
    notes: str = ""
    # Optional test-specific information.
    # Generic framework code does not interpret these values.
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    timestamp: datetime = field(
        default_factory=datetime.utcnow,
    )
    @property
    def changed(self) -> bool:
        """
        Whether any measurable performance change
        occurred.
        """

        return self.performance_drop != 0.0

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result into a serializable
        dictionary.
        """

        return {

            "test_name": self.test_name,

            "evidence_direction": self.evidence_direction.value,

            "p_value": self.p_value,

            "sample_size": self.sample_size,

            "baseline_score": self.baseline_score,

            "test_score": self.test_score,

            "performance_drop": self.performance_drop,

            "relative_difference": self.relative_difference,

            "runtime": self.runtime,

            "notes": self.notes,

            "metadata": self.metadata,

            "timestamp": self.timestamp.isoformat(),
        }

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def __str__(self) -> str:

        return (

            f"{self.test_name}"

            f" | Baseline={self.baseline_score:.6f}"

            f" | Test={self.test_score:.6f}"

            f" | Δ={self.relative_difference * 100:.2f}%"

        )


# ============================================================
# Scientific Evaluation
# ============================================================


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Scientific interpretation of a cryptographic audit.

    Unlike CryptographicTestResult, this class contains
    conclusions rather than raw measurements.
    """

    decision: EvaluationDecision

    rationale: str

    threshold: float

    observed_effect: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the evaluation into a serializable
        dictionary.
        """

        return {

            "decision": self.decision.value,

            "rationale": self.rationale,

            "threshold": self.threshold,

            "observed_effect": self.observed_effect,

        }

    @property
    def supported(self) -> bool:
        """
        Whether the hypothesis is supported.
        """

        return (
            self.decision
            is EvaluationDecision.SUPPORTED
        )

    @property
    def inconclusive(self) -> bool:
        """
        Whether the experiment is inconclusive.
        """

        return (
            self.decision
            is EvaluationDecision.INCONCLUSIVE
        )

    @property
    def rejected(self) -> bool:
        """
        Whether the hypothesis is not supported.
        """

        return (
            self.decision
            is EvaluationDecision.NOT_SUPPORTED
        )

    def __str__(self) -> str:

        return (

            f"{self.decision.value}"

            f" ({self.rationale})"

        )

@dataclass(frozen=True, slots=True)
class CorrelationStatistic:
    """
    Generic correlation-based statistical result, reusable by
    any Cryptographic Evidence test that reports a rank or
    linear agreement measure (CE2 today; potentially others).
    """

    statistic: float
    p_value: float
    n: int