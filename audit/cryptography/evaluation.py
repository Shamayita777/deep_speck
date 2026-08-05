"""
Cryptographic Evidence Evaluation
=================================

Evaluation policy for Cryptographic Evidence tests.

This module interprets the numerical results produced by
cryptographic experiments and converts them into scientific
evaluation decisions.

Responsibilities
----------------
• Evaluate CryptographicTestResult objects
• Produce EvaluationResult objects
• Apply configurable decision thresholds

This module contains no knowledge of any specific cryptographic
test or implementation.
"""

from __future__ import annotations

from audit.cryptography.test.base import EvidenceDirection
from audit.cryptography.test.base import CryptographicTest
from .results import (
    CryptographicTestResult,
    EvaluationDecision,
    EvaluationResult,
)


class CryptographicEvaluator:
    """
    Evaluates the outcome of cryptographic evidence tests.
    """

    def __init__(
        self,
        *,
        supported_threshold: float = 0.20,
        inconclusive_threshold: float = 0.05,
    ) -> None:
        """
        Parameters
        ----------
        supported_threshold
            Minimum relative performance drop required to
            support the hypothesis.

        inconclusive_threshold
            Minimum relative performance drop below which the
            hypothesis is considered unsupported.
        """

        if supported_threshold <= inconclusive_threshold:
            raise ValueError(
                "supported_threshold must be greater than "
                "inconclusive_threshold."
            )

        self.supported_threshold = supported_threshold
        self.inconclusive_threshold = inconclusive_threshold

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        test: CryptographicTest,
        result: CryptographicTestResult,
    ) -> EvaluationResult:
        """
        Evaluate a cryptographic test result.

        Parameters
        ----------
        test
            The cryptographic test that was executed.
        result
            Experimental result.

        Returns
        -------
        EvaluationResult
            Scientific interpretation of the result.
        """

        effect = result.relative_difference

        if (
            result.evidence_direction
            is EvidenceDirection.LOWER_IS_BETTER
        ):
            effect = -effect

        if effect >= self.supported_threshold:

            decision = EvaluationDecision.SUPPORTED

            rationale = test.supported_rationale

            threshold = self.supported_threshold

        elif effect >= self.inconclusive_threshold:

            decision = EvaluationDecision.INCONCLUSIVE

            rationale = test.inconclusive_rationale

            threshold = self.inconclusive_threshold

        else:

            decision = EvaluationDecision.NOT_SUPPORTED

            rationale = test.unsupported_rationale

            threshold = self.inconclusive_threshold

        return EvaluationResult(
            decision=decision,
            rationale=rationale,
            threshold=threshold,
            observed_effect=effect,
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "Cryptographic Evaluator"

    @property
    def description(self) -> str:
        return (
            "Scientific evaluation policy for "
            "Cryptographic Evidence experiments."
        )