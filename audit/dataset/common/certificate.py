"""
Dataset Audit Certificate Infrastructure.

This module provides generic certificate construction for
Dataset Integrity audits.

D0 is infrastructure, not an audit itself.

The certificate format is intentionally independent of any
specific cryptographic primitive, model, dataset generator,
or audit such as D1/D2/D3/D4.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


CERTIFICATE_SCHEMA_VERSION = "1.0"


def utc_timestamp() -> str:
    """
    Return the current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def make_certificate(
    *,
    audit_id: str,
    audit_name: str,
    claim: str,
    outcome: str,
    findings: Mapping[str, Any],
    methodology: Mapping[str, Any],
    provenance: Mapping[str, Any],
    limitations: list[str],
    evidence_level: Optional[str] = None,
    certificate_version: str = CERTIFICATE_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """
    Construct a standardized Dataset Integrity certificate.

    Parameters
    ----------
    audit_id:
        Stable identifier such as "D1".

    audit_name:
        Human-readable audit name.

    claim:
        Scientific property being evaluated.

    outcome:
        Audit outcome, e.g. PASS, FAIL, or INCONCLUSIVE.

    findings:
        Machine-readable empirical findings.

    methodology:
        Description of the method, thresholds, and scope.

    provenance:
        Dataset and execution provenance.

    limitations:
        Explicit limitations of the audit.

    evidence_level:
        Optional evidence-level classification.

    certificate_version:
        Version of the certificate schema.

    Returns
    -------
    dict
        JSON-serializable certificate.
    """

    allowed_outcomes = {
        "PASS",
        "CONDITIONAL_PASS",
        "FAIL",
        "INCONCLUSIVE",
    }

    if outcome not in allowed_outcomes:
        raise ValueError(
            f"Unsupported audit outcome: {outcome!r}. "
            f"Expected one of {sorted(allowed_outcomes)}."
        )

    return {
        "certificate_schema_version": certificate_version,
        "audit": {
            "id": audit_id,
            "name": audit_name,
            "claim": claim,
        },
        "decision": {
            "outcome": outcome,
            "evidence_level": evidence_level,
        },
        "findings": dict(findings),
        "methodology": dict(methodology),
        "provenance": dict(provenance),
        "limitations": list(limitations),
        "generated_at_utc": utc_timestamp(),
    }


def write_certificate(
    certificate: Mapping[str, Any],
    output_path: str,
) -> None:
    """
    Write a certificate to a JSON file.

    Parameters
    ----------
    certificate:
        Certificate returned by make_certificate().

    output_path:
        Destination JSON path.
    """

    import json
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            certificate,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")