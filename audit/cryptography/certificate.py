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
from typing import Any

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
    threshold: float | None
    observed_effect: float
    runtime: float
    timestamp: str
    metadata: dict[str, Any]
    p_value: float | None
    sample_size: int | None


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
            threshold=(
                None
                if result.test_name == "Representation Interpretation"
                else evaluation.threshold
            ),
            observed_effect=evaluation.observed_effect,
            runtime=result.runtime,
            timestamp=result.timestamp.isoformat(),
            metadata=result.metadata,
            p_value=result.p_value,
            sample_size=result.sample_size,
        )

    @staticmethod
    def to_dict(certificate: AuditCertificate) -> dict:
        data = asdict(certificate)

        # Add CE2-specific alias for clarity.
        if certificate.test == "Theory Consistency":
            data["correlation"] = certificate.test_score

        # Add CE3-specific fields, sourced from metadata rather
        # than reusing baseline_score/test_score/relative_difference
        # for quantities those fields were never meant to carry.
        elif certificate.test == "Representation Interpretation":
            m = certificate.metadata
            data["real_probe_score"] = m.get("real_probe_score")
            data["control_probe_score"] = m.get("control_probe_score")
            data["selectivity"] = certificate.test_score
            data["effect_size"] = m.get("effect_size")
            data["confidence_interval"] = [
                m.get("ci_low"), m.get("ci_high"),
            ]
            data["statistical_test"] = m.get("statistical_test")
            data["metric"] = m.get("metric_name")
            data["n_folds"] = m.get("n_folds")
            data["target"] = m.get("target_name")
            data["calibration_selectivity"] = m.get(
                "calibration_selectivity"
            )

            data["calibration_p_value"] = m.get(
                "calibration_p_value"
            )

            data["calibration_validated"] = m.get(
                "calibration_validated"
            )

            data["primary_supported"] = m.get(
                "primary_supported"
            )

        return data

    @property
    def name(self) -> str:
        return "Certificate Generator"

    @property
    def description(self) -> str:
        return (
            "Produces machine-readable audit certificates."
        )