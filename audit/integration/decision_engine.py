"""
Deterministic cross-pillar decision engine (methodology Sections 3.8.4,
3.8.7, 3.10).

Integration is NOT a fifth audit pillar: this module runs no
experiments, alters no hypotheses, generates no new statistics, and
never selects favourable results. It consumes already-produced pillar
findings plus retained conflicts and applies the methodology's decision
rules deterministically.

EVIDENCE LEVEL (3.8.4) is assigned from what the evidence ACTUALLY
shows, not from a fixed cap:

    LEVEL_0  no reproducible empirical support
    LEVEL_1  predictive performance above baseline, nothing more
    LEVEL_2  predictive performance surviving Implementation, Dataset
             and Experimental scrutiny
    LEVEL_3  observations consistent with cryptographic reasoning after
             identified confounders are eliminated
    LEVEL_4  multiple INDEPENDENT experiments across different
             configurations/architectures/datasets CONVERGING on the
             same conclusion

LEVEL_4 requires actual convergence evidence. An unperformed
independent reimplementation removes one possible convergence source;
it is not itself a cap (see audit.implementation.stages).

A predictive-performance result is never promoted to "cryptographic
evidence" merely for being statistically significant - Rule I of 3.8.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from audit.common.outcomes import (
    CryptographicOutcome,
    DimensionOutcome,
    EvidenceLevel,
    FinalAuditOutcome,
)
from audit.integration.conflict_resolution import Conflict, blocking_conflicts

DECISION_POLICY_VERSION = "ciphermind-decision-policy-v1"


@dataclass(frozen=True)
class ConvergenceSource:
    """One qualifying independent source toward the LEVEL_4 criterion."""
    source_kind: str      # e.g. "independent_implementation", "independent_dataset"
    evidence_reference: str
    converges_on: str

    def to_dict(self) -> dict:
        return {"source_kind": self.source_kind,
                "evidence_reference": self.evidence_reference,
                "converges_on": self.converges_on}


@dataclass(frozen=True)
class ClaimDecision:
    claim_id: str
    policy_version: str
    implementation: DimensionOutcome
    dataset: DimensionOutcome
    experimental: DimensionOutcome
    cryptographic: CryptographicOutcome
    evidence_level: EvidenceLevel
    final_outcome: FinalAuditOutcome
    rationale: str
    blocking_conflicts: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "policy_version": self.policy_version,
            "dimension_outcomes": {
                "implementation_integrity": self.implementation.value,
                "dataset_integrity": self.dataset.value,
                "experimental_validity": self.experimental.value,
                "cryptographic_evidence": self.cryptographic.value,
            },
            "evidence_level": self.evidence_level.value,
            "final_outcome": self.final_outcome.value,
            "rationale": self.rationale,
            "blocking_conflicts": list(self.blocking_conflicts),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "limitations": list(self.limitations),
        }


def assign_evidence_level(
    *,
    implementation: DimensionOutcome,
    dataset: DimensionOutcome,
    experimental: DimensionOutcome,
    cryptographic: CryptographicOutcome,
    convergence_sources: list[ConvergenceSource],
    has_blocking_conflict: bool,
) -> tuple[EvidenceLevel, str]:
    """Assign an evidence level per 3.8.4. Deterministic."""
    dimension_outcomes = (implementation, dataset, experimental)

    if any(o is DimensionOutcome.FAIL for o in dimension_outcomes):
        return EvidenceLevel.LEVEL_1_PREDICTIVE, (
            "A dimension FAILED, so alternative explanations arising from that dimension are "
            "supported. Evidence cannot rise above predictive performance."
        )

    if has_blocking_conflict:
        return EvidenceLevel.LEVEL_1_PREDICTIVE, (
            "An unresolved conflict among cryptographic observations on the same target "
            "quantity prevents any claim that observations are consistent with cryptographic "
            "reasoning. Evidence remains at the predictive level."
        )

    if any(o is DimensionOutcome.INCONCLUSIVE for o in dimension_outcomes):
        return EvidenceLevel.LEVEL_1_PREDICTIVE, (
            "At least one of Implementation, Dataset or Experimental validity is INCONCLUSIVE, "
            "so predictive performance has not been shown to survive that scrutiny. LEVEL_2 "
            "requires all three to be established, not merely un-refuted."
        )

    # All three are PASS or CONDITIONAL_PASS from here.
    if cryptographic in (CryptographicOutcome.NOT_SUPPORTED, CryptographicOutcome.INCONCLUSIVE):
        return EvidenceLevel.LEVEL_2_ROBUST_EXPERIMENTAL, (
            "Predictive performance survives implementation, dataset and experimental scrutiny, "
            "but the cryptographic interpretation is not supported. Robustness alone does not "
            "constitute cryptographic evidence (3.8.4 Level 2)."
        )

    qualifying = [s for s in convergence_sources if s.source_kind]
    distinct_kinds = {s.source_kind for s in qualifying}
    if cryptographic is CryptographicOutcome.SUPPORTED and len(distinct_kinds) >= 2:
        return EvidenceLevel.LEVEL_4_STRONG_CRYPTOGRAPHIC, (
            f"Multiple independent convergence sources ({sorted(distinct_kinds)}) converge on "
            "the same cryptographic conclusion, satisfying 3.8.4 Level 4."
        )

    if cryptographic is CryptographicOutcome.SUPPORTED:
        return EvidenceLevel.LEVEL_3_CRYPTOGRAPHIC, (
            "Observations are consistent with cryptographic reasoning after identified "
            "confounders were addressed, but fewer than two distinct independent convergence "
            "sources exist, so Level 4 is not met."
        )

    return EvidenceLevel.LEVEL_2_ROBUST_EXPERIMENTAL, (
        "Cryptographic interpretation is only partially supported."
    )


def decide_claim(
    *,
    claim_id: str,
    implementation: DimensionOutcome,
    dataset: DimensionOutcome,
    experimental: DimensionOutcome,
    cryptographic: CryptographicOutcome,
    conflicts: Optional[list[Conflict]] = None,
    convergence_sources: Optional[list[ConvergenceSource]] = None,
    supporting_evidence_ids: Optional[list[str]] = None,
    limitations: Optional[list[str]] = None,
) -> ClaimDecision:
    """
    Produce a claim-level decision per the 3.10.5 decision matrix.
    Deterministic: identical inputs always yield an identical decision.
    """
    conflicts = conflicts or []
    convergence_sources = convergence_sources or []
    blocking = blocking_conflicts(conflicts)
    blocking_ids = [c.conflict_id for c in blocking]

    level, level_rationale = assign_evidence_level(
        implementation=implementation, dataset=dataset, experimental=experimental,
        cryptographic=cryptographic, convergence_sources=convergence_sources,
        has_blocking_conflict=bool(blocking),
    )

    dimension_outcomes = (implementation, dataset, experimental)

    # 3.10.5: any dimension FAIL -> Not Supported for the affected claim.
    if any(o is DimensionOutcome.FAIL for o in dimension_outcomes):
        final = FinalAuditOutcome.NOT_SUPPORTED
        rationale = ("A dimension returned FAIL, supplying an experimentally supported "
                     "alternative explanation for the claim. " + level_rationale)
    elif cryptographic is CryptographicOutcome.NOT_SUPPORTED:
        final = FinalAuditOutcome.NOT_SUPPORTED
        rationale = ("Cryptographic Evidence is NOT_SUPPORTED. " + level_rationale)
    elif blocking:
        final = FinalAuditOutcome.INCONCLUSIVE
        rationale = (f"Unresolved conflict(s) {blocking_ids} persist after the 3.8.8 "
                     "procedure; the affected claim is INCONCLUSIVE. " + level_rationale)
    elif any(o is DimensionOutcome.INCONCLUSIVE for o in dimension_outcomes) or \
            cryptographic is CryptographicOutcome.INCONCLUSIVE:
        final = FinalAuditOutcome.INCONCLUSIVE
        rationale = ("At least one dimension necessary for the claim is INCONCLUSIVE. "
                     + level_rationale)
    elif cryptographic is CryptographicOutcome.PARTIALLY_SUPPORTED:
        final = FinalAuditOutcome.PARTIALLY_SUPPORTED
        rationale = ("Only a subset of the reported interpretation is supported. " + level_rationale)
    elif any(o is DimensionOutcome.CONDITIONAL_PASS for o in dimension_outcomes):
        final = FinalAuditOutcome.SUPPORTED_WITH_LIMITATIONS
        rationale = ("All dimensions are established and the cryptographic interpretation is "
                     "supported, with documented minor limitations. " + level_rationale)
    else:
        final = FinalAuditOutcome.SUPPORTED
        rationale = ("All dimensions PASS and the cryptographic interpretation is supported. "
                     + level_rationale)

    return ClaimDecision(
        claim_id=claim_id, policy_version=DECISION_POLICY_VERSION,
        implementation=implementation, dataset=dataset, experimental=experimental,
        cryptographic=cryptographic, evidence_level=level, final_outcome=final,
        rationale=rationale, blocking_conflicts=blocking_ids,
        supporting_evidence_ids=list(supporting_evidence_ids or []),
        limitations=list(limitations or []),
    )
