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
            data["n_folds"] = m.get("n_splits_per_replicate")
            data["n_replicates"] = m.get("n_replicates")
            data["selectivity_replicates"] = m.get("selectivity_replicates")
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

# =====================================================================
# Audit certificate schema, validation and safe writing
# (current audit implementation; see provenance.EXPERIMENT_DESIGN_VERSION)
# =====================================================================

import json as _json
import math as _math
from pathlib import Path as _Path

from audit.cryptography.audit_config import REFERENCE as _REFERENCE
from audit.cryptography.output_policy import assert_audit_output_path as _assert_output
from audit.cryptography.provenance import AUDIT_SCHEMA_VERSION

CERTIFICATE_SCHEMA_VERSION = AUDIT_SCHEMA_VERSION

REQUIRED_FIELDS = ("certificate_schema_version", "experiment_id",
                   "experiment_design_version", "reference_configuration",
                   "provenance", "results", "claim_scope")


class CertificateSchemaError(RuntimeError):
    pass


def _assert_json_clean(obj, path="root"):
    """Reject NaN/Infinity: not valid JSON, and they corrupt downstream parsing."""
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            raise CertificateSchemaError(f"non-finite float at {path}: {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise CertificateSchemaError(f"non-string key at {path}: {k!r}")
            _assert_json_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_json_clean(v, f"{path}[{i}]")


def validate_certificate(cert: dict) -> dict:
    """Schema gate applied before every write."""
    missing = [f for f in REQUIRED_FIELDS if f not in cert]
    if missing:
        raise CertificateSchemaError(f"certificate missing required fields: {missing}")
    if cert["certificate_schema_version"] != CERTIFICATE_SCHEMA_VERSION:
        raise CertificateSchemaError(
            f"unexpected schema version {cert['certificate_schema_version']!r}")
    blob = _json.dumps(cert)
    if "CE1" in str(cert.get("experiment_id", "")) and "primary_test" in blob:
        if '"H0: E[Delta] = 0' in blob or '"H0: E[\\u0394] = 0' in blob:
            raise CertificateSchemaError(
                "CE1 certificate states the superseded weak null 'H0: E[Delta] = 0' for an "
                "exact sign-flip test. The sign-flip is exact under the SHARP null with "
                "within-block arm exchangeability; see frozen_design.CE1.primary_hypothesis.")
    if "CE1" in str(cert.get("experiment_id", "")) and "exact_paired_sign_flip" in blob:
        if '"arm_assignment_randomized": true' not in blob.lower():
            raise CertificateSchemaError(
                "CE1 certificate reports an exact sign-flip test without recording that the "
                "arm assignment was randomized. Without that randomization the 2^K sign "
                "patterns are not a randomization distribution and the test is not exact.")
    if "n_folds" in _json.dumps(cert):
        raise CertificateSchemaError(
            "stale field 'n_folds'; report n_replicates and n_splits_per_replicate")
    if cert["reference_configuration"] != _REFERENCE.to_dict():
        raise CertificateSchemaError(
            "reference_configuration does not match the frozen reference")
    _assert_json_clean(cert)

    def _scan(o, path="root"):
        if isinstance(o, dict):
            if o.get("p_value") == 0.0:
                raise CertificateSchemaError(
                    f"{path}: p_value is the literal number 0.0. Underflowed p-values must "
                    "be reported as p_value=null with p_underflow=true and p_upper_bound.")
            if o.get("p_underflow") is True and o.get("p_value") is not None:
                raise CertificateSchemaError(f"{path}: p_underflow=true but p_value is not null.")
            for k, v in o.items():
                _scan(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                _scan(v, f"{path}[{i}]")
    _scan(cert)
    return cert


def write_certificate(cert: dict, path, *, repo_root=None) -> _Path:
    """Validate, then write ONLY to an approved new-evidence location."""
    validate_certificate(cert)
    target = _assert_output(path, repo_root=repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json.dumps(cert, indent=2, sort_keys=True, allow_nan=False))
    return target
