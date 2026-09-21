"""
Conflict resolution (methodology Section 3.8.8).

Generic and dimension-agnostic: operates on already-produced evidence
records. It does NOT run experiments, does not recompute statistics,
and - critically - does NOT resolve conflicts by majority vote or by
preferring the more convenient result.

Methodology 3.8.8 procedure, implemented literally:
    1. identify the conflicting observations;
    2. retain BOTH (neither is discarded);
    3. evaluate the methodological quality of each supporting experiment;
    4. determine whether a known confounder explains the discrepancy;
    5. assess whether additional controlled experimentation is required.
If contradictory evidence persists after these steps, the affected
claim is INCONCLUSIVE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConflictStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    EXPLAINED_BY_KNOWN_CONFOUNDER = "EXPLAINED_BY_KNOWN_CONFOUNDER"
    RESOLVED_BY_METHODOLOGICAL_QUALITY = "RESOLVED_BY_METHODOLOGICAL_QUALITY"
    REQUIRES_ADDITIONAL_EXPERIMENTATION = "REQUIRES_ADDITIONAL_EXPERIMENTATION"


@dataclass(frozen=True)
class ConflictingObservation:
    evidence_reference: str
    source_id: str
    observation: str
    decision: str
    methodological_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evidence_reference": self.evidence_reference, "source_id": self.source_id,
            "observation": self.observation, "decision": self.decision,
            "methodological_notes": list(self.methodological_notes),
        }


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    shared_subject: str
    observations: list[ConflictingObservation]
    status: ConflictStatus
    candidate_confounders: list[str] = field(default_factory=list)
    required_experimentation: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if len(self.observations) < 2:
            raise ValueError("A conflict requires at least two retained observations.")

    @property
    def blocks_claim(self) -> bool:
        """Per 3.8.8, a conflict that is not genuinely resolved forces INCONCLUSIVE."""
        return self.status in (
            ConflictStatus.UNRESOLVED,
            ConflictStatus.REQUIRES_ADDITIONAL_EXPERIMENTATION,
        )

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id, "shared_subject": self.shared_subject,
            "observations": [o.to_dict() for o in self.observations],
            "status": self.status.value,
            "candidate_confounders": list(self.candidate_confounders),
            "required_experimentation": list(self.required_experimentation),
            "rationale": self.rationale, "blocks_claim": self.blocks_claim,
        }


# ---------------------------------------------------------------------
# The verified CE2 vs CE3/CE4 conflict.
#
# Retained as a genuine scientific conflict, NOT resolved by preferring
# the two SUPPORTED results over the one NOT_SUPPORTED result. Majority
# voting is explicitly prohibited by 3.8.8, and would here amount to
# discarding the only observation that contradicts the desired
# conclusion.
# ---------------------------------------------------------------------

CE2_VS_CE3_CE4_CONFLICT = Conflict(
    conflict_id="CONFLICT-CE2-CE34-V1",
    shared_subject=(
        "The same analytical target quantity (Lipmaa-Moriai chained single-trail "
        "differential probability), verified identical across CE2/CE3/CE4 by source: CE4 reuses "
        "CE3's primary task object literally, and CE3's target and CE2's reference both read "
        "TheoryDataset.theoretical_probabilities."
    ),
    observations=[
        ConflictingObservation(
            evidence_reference="evidence/ce2/ce2_certificate_{1..5}.json",
            source_id="CE2",
            observation=(
                "Model output does NOT positively rank-correlate with the analytical target; "
                "the observed correlation is negative."
            ),
            decision="NOT_SUPPORTED",
            methodological_notes=[
                "Observational, no control arm.",
                "Per-sample unit (n=1e5) within one model instance.",
                "Five independent runs, which strengthens it against being a one-off.",
                "Measures the model's OUTPUT against the target.",
            ],
        ),
        ConflictingObservation(
            evidence_reference="evidence/ce3/ce3_certificate.json",
            source_id="CE3",
            observation=(
                "The analytical target IS selectively decodable from the trained model's hidden "
                "representation relative to a control representation."
            ),
            decision="SUPPORTED",
            methodological_notes=[
                "Replicate-level unit (n=20 independent evaluation replicates).",
                "Calibration-gated and Bonferroni-corrected - methodologically the strongest "
                "of the three designs.",
                "Measures the model's INTERNAL REPRESENTATION, not its output.",
            ],
        ),
        ConflictingObservation(
            evidence_reference="evidence/ce4/ce4_certificate.json",
            source_id="CE4",
            observation=(
                "Perturbing the analytical structure changes the model's output more than a "
                "magnitude-matched, target-preserving control perturbation."
            ),
            decision="SUPPORTED",
            methodological_notes=[
                "Per-sample unit within one model instance; p underflowed at n~1e5, d_z=0.545.",
                "Control provably preserves the declared target (XOR identity).",
                "Measures output SENSITIVITY to perturbation, not output AGREEMENT with the target.",
            ],
        ),
    ],
    status=ConflictStatus.REQUIRES_ADDITIONAL_EXPERIMENTATION,
    candidate_confounders=[
        "All three tests were run against the SAME depth-5 checkpoint "
        "(II-FINDING-DEPTH-V1), which diverges from the declared reference protocol's depth=10. "
        "A shared upstream artifact defect cannot by itself explain a DISAGREEMENT between "
        "tests run on that same artifact, so it does not resolve this conflict - but it does "
        "mean all three observations are scoped to a non-conformant model instance.",
        "The tests measure genuinely different things: CE2 tests OUTPUT AGREEMENT with the "
        "target; CE3 tests REPRESENTATIONAL DECODABILITY; CE4 tests OUTPUT SENSITIVITY to "
        "perturbing the target. These are not logically equivalent, and it is coherent for a "
        "model to encode and be sensitive to a quantity whose monotone ordering its output "
        "nonetheless fails to track. This is a plausible reconciliation, but it is a "
        "HYPOTHESIS, not an established finding, and it is not adopted as a resolution here.",
    ],
    required_experimentation=[
        "Re-execute CE2, CE3 and CE4 against a reference-conformant (depth=10) model so that "
        "all three observations describe the declared protocol rather than a divergent artifact.",
        "Test the measurement-target hypothesis directly: determine whether the "
        "output-agreement/representational-decodability divergence persists under a conformant "
        "model, which would distinguish a genuine dissociation from an artifact of this instance.",
    ],
    rationale=(
        "Both the contradicting and the supporting observations are RETAINED. The conflict is "
        "NOT resolved by majority vote (two SUPPORTED against one NOT_SUPPORTED), which "
        "methodology 3.8.8 prohibits and which would here amount to discarding the only "
        "observation contradicting the desired conclusion. It is also not resolved by "
        "methodological quality alone: although CE3 has the strongest design (replicate-level "
        "unit, calibration gate, multiplicity correction), a stronger design for a DIFFERENT "
        "measurement does not overturn a weaker design's finding about its own measurement. "
        "Per 3.8.8 the conflict therefore stands as requiring additional controlled "
        "experimentation, and the affected claim cannot be certified while it stands."
    ),
)

ALL_CONFLICTS = [CE2_VS_CE3_CE4_CONFLICT]


def blocking_conflicts(conflicts: list[Conflict]) -> list[Conflict]:
    return [c for c in conflicts if c.blocks_claim]
