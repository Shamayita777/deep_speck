"""
Cryptographic Evidence Certificate
==================================

Machine-readable audit certificate.

This module converts the outcome of a Cryptographic Evidence
evaluation into a structured certificate suitable for storage,
serialization, or downstream processing.

Responsibilities
----------------
• Generate audit certificates
• Produce machine-readable output
• Remain independent of reporting and presentation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .results import (
    CryptographicTestResult,
    EvaluationResult,
)


@dataclass(frozen=True)
class AuditCertificate:
    """
    Machine-readable audit certificate.
    """

    framework: str
    test: str
    decision: str
    rationale: str
    baseline_score: float
    test_score: float
    performance_drop: float
    relative_difference: float
    threshold: float
    observed_effect: float
    runtime: float
    timestamp: str


class CertificateGenerator:
    """
    Generates audit certificates.
    """

    FRAMEWORK_NAME = "Cryptographic Evidence Framework"

    def generate(
        self,
        result: CryptographicTestResult,
        evaluation: EvaluationResult,
    ) -> AuditCertificate:
        """
        Generate an immutable audit certificate.
        """

        return AuditCertificate(
            framework=self.FRAMEWORK_NAME,
            test=result.test_name,
            decision=evaluation.decision.value,
            rationale=evaluation.rationale,
            baseline_score=result.baseline_score,
            test_score=result.test_score,
            performance_drop=result.performance_drop,
            relative_difference=result.relative_difference,
            threshold=evaluation.threshold,
            observed_effect=evaluation.observed_effect,
            runtime=result.runtime,
            timestamp=result.timestamp.isoformat(),
        )

    @staticmethod
    def to_dict(
        certificate: AuditCertificate,
    ) -> dict:
        """
        Convert a certificate into a dictionary.
        """

        return asdict(certificate)

    @property
    def name(self) -> str:
        return "Certificate Generator"

    @property
    def description(self) -> str:
        return (
            "Produces machine-readable audit certificates."
        )