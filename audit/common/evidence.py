"""
Generic Evidence object - the atomic unit consumed by the evidence
ledger and, ultimately, the integration decision engine.

Knows nothing about Gohr, Speck, datasets, or models: an Evidence
object is produced by SOME pillar (identified generically by
Dimension) about SOME claim, referencing SOME provenance, carrying
SOME statistics - all of those are opaque payloads to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from audit.common.ids import Dimension, validate_identifier
from audit.common.outcomes import EvidenceTag
from audit.common.provenance import utc_timestamp
from audit.common.strict_json import find_non_finite_paths


class EvidenceValidationError(ValueError):
    pass


REQUIRED_EVIDENCE_FIELDS = [
    "evidence_id", "claim_id", "dimension", "experiment_id", "run_ids",
    "input_artifacts", "artifact_hashes", "observation", "statistics",
    "decision", "limitations", "provenance", "tag",
]

# Fields that may legitimately be null (e.g. a descriptive-only pillar
# result has no p_value) but must still be PRESENT as keys, so that
# "field omitted" is never confused with "field explicitly recorded as
# not applicable".
OPTIONAL_BUT_PRESENT_FIELDS = [
    "effect_size", "confidence_interval", "p_value", "adjusted_p_value", "practical_threshold",
]


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    claim_id: str
    dimension: Dimension
    experiment_id: str
    run_ids: list[str]
    input_artifacts: list[str]
    artifact_hashes: dict[str, str]
    observation: dict[str, Any]
    statistics: dict[str, Any]
    decision: str  # the pillar's own outcome string - see audit.common.outcomes module docstring
    limitations: list[str]
    provenance: dict[str, Any]
    tag: EvidenceTag
    effect_size: Optional[float] = None
    confidence_interval: Optional[dict[str, float]] = None
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    practical_threshold: Optional[float] = None
    recorded_at_utc: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        validate_identifier(self.evidence_id, context="evidence_id")
        validate_identifier(self.claim_id, context="claim_id")
        validate_identifier(self.experiment_id, context="experiment_id")
        if not self.run_ids:
            raise EvidenceValidationError(
                f"Evidence {self.evidence_id} has no run_ids. Evidence must be traceable to "
                "at least one concrete run, even a SKIPPED/FAILED one - not an abstract claim."
            )
        problems = find_non_finite_paths(self.to_dict())
        if problems:
            raise EvidenceValidationError(
                f"Evidence {self.evidence_id} contains non-finite value(s) at {problems}. "
                "A NaN/Infinity statistic must be reported as a limitation/failure, not "
                "embedded silently in an evidence record."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "dimension": self.dimension.value if isinstance(self.dimension, Dimension) else self.dimension,
            "experiment_id": self.experiment_id,
            "run_ids": list(self.run_ids),
            "input_artifacts": list(self.input_artifacts),
            "artifact_hashes": dict(self.artifact_hashes),
            "observation": self.observation,
            "statistics": self.statistics,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "practical_threshold": self.practical_threshold,
            "decision": self.decision,
            "limitations": list(self.limitations),
            "provenance": self.provenance,
            "tag": self.tag.value if isinstance(self.tag, EvidenceTag) else self.tag,
            "recorded_at_utc": self.recorded_at_utc,
        }



def validate_evidence_provenance(
    evidence: Evidence,
    *,
    known_run_manifests: Mapping[str, Any],
    provenance_graph: Optional[Any] = None,
) -> list[str]:
    """
    Verify that an Evidence object's claimed traceability is real.

    Checks, per the framework's traceability requirement that a reviewer
    can walk decision -> evidence -> run -> dataset -> code:

      1. Every run_id corresponds to an actual RunManifest. A run_id
         with no manifest behind it is an untraceable citation, which
         defeats the purpose of recording run_ids at all.
      2. CONFIRMATORY evidence must carry the provenance that justifies
         that label: a non-empty provenance record, and every cited
         manifest must reference a preregistration. Evidence that
         cannot demonstrate it was preregistered is EXPLORATORY,
         whatever it calls itself.
      3. If a provenance graph is supplied, it must itself be
         structurally valid.

    Returns a list of problems; empty means valid. Kept as a function
    rather than a constructor check so that Evidence remains
    constructible in unit tests without a full manifest registry, while
    production assembly paths can enforce it.
    """
    problems: list[str] = []

    for run_id in evidence.run_ids:
        if run_id not in known_run_manifests:
            problems.append(
                f"Evidence {evidence.evidence_id} cites run_id {run_id!r}, which has no "
                "corresponding RunManifest. Every cited run must be traceable to a real manifest."
            )

    if evidence.tag is EvidenceTag.CONFIRMATORY:
        if not evidence.provenance:
            problems.append(
                f"Evidence {evidence.evidence_id} is labelled CONFIRMATORY but carries no "
                "provenance. Confirmatory evidence must be traceable; otherwise it is EXPLORATORY."
            )
        for run_id in evidence.run_ids:
            manifest = known_run_manifests.get(run_id)
            if manifest is None:
                continue
            prereg = getattr(manifest, "preregistration_hash", None)
            if prereg is None and isinstance(manifest, dict):
                prereg = manifest.get("preregistration_hash")
            if not prereg:
                problems.append(
                    f"Evidence {evidence.evidence_id} is labelled CONFIRMATORY but its run "
                    f"{run_id!r} records no preregistration_hash. A confirmatory claim requires "
                    "a preregistered specification; without one the evidence is EXPLORATORY."
                )

    if provenance_graph is not None and hasattr(provenance_graph, "validate"):
        problems.extend(provenance_graph.validate())

    return problems
