"""
Implementation Integrity stage definitions (Pillar 1).

METHODOLOGY REVISION (applied): independent reimplementation is an
ENHANCED assessment, NOT a prerequisite for a core Implementation
Integrity finding. Per methodology Section 3.5.4 as revised for this
project, the CORE stages are:

    II-1  Baseline Reconstruction
    II-3  Controlled Verification
    II-4  Comparative Evaluation
    II-5  Statistical Assessment

and II-2 Independent Reimplementation is retained as an ENHANCED stage.
The original stage numbering is preserved (II-2 is not renumbered) so
that historical references remain valid.

EVIDENTIAL CONSEQUENCE OF THE CORE/ENHANCED DISTINCTION
-------------------------------------------------------
This is not a labelling change, but it is also NOT a universal cap.

Methodology Section 3.8.4 defines LEVEL_4 (Strong Cryptographic
Evidence) by convergence: "multiple independent experiments employing
different experimental configurations, architectures, or datasets
[that] converge upon the same cryptographic conclusion." Independent
reimplementation is ONE possible source of such convergence - it is
not the only one, and it is not a mandatory prerequisite. Convergence
could equally arise from independent datasets or independent
architectures evaluated elsewhere in the audit.

CORRECTED: this module therefore does NOT cap the attainable evidence
level. It reports, via `convergence_contribution()`, whether this
pillar contributes an independent-implementation convergence source.
The integration decision engine determines LEVEL_4 from the ACTUAL
qualifying convergence evidence available across all pillars. An
unperformed II-2 simply means this pillar contributes no convergence
source - not that LEVEL_4 is unreachable by any route.

This module contains no knowledge of Gohr, Speck, or any case study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from audit.common.outcomes import AssessmentTier, EnhancedAssessmentStatus


class StageId(str, Enum):
    II_1_BASELINE_RECONSTRUCTION = "II-1_BASELINE_RECONSTRUCTION"
    II_2_INDEPENDENT_REIMPLEMENTATION = "II-2_INDEPENDENT_REIMPLEMENTATION"
    II_3_CONTROLLED_VERIFICATION = "II-3_CONTROLLED_VERIFICATION"
    II_4_COMPARATIVE_EVALUATION = "II-4_COMPARATIVE_EVALUATION"
    II_5_STATISTICAL_ASSESSMENT = "II-5_STATISTICAL_ASSESSMENT"


STAGE_TIERS: dict[StageId, AssessmentTier] = {
    StageId.II_1_BASELINE_RECONSTRUCTION: AssessmentTier.CORE,
    StageId.II_2_INDEPENDENT_REIMPLEMENTATION: AssessmentTier.ENHANCED,
    StageId.II_3_CONTROLLED_VERIFICATION: AssessmentTier.CORE,
    StageId.II_4_COMPARATIVE_EVALUATION: AssessmentTier.CORE,
    StageId.II_5_STATISTICAL_ASSESSMENT: AssessmentTier.CORE,
}

CORE_STAGES = [s for s, t in STAGE_TIERS.items() if t is AssessmentTier.CORE]
ENHANCED_STAGES = [s for s, t in STAGE_TIERS.items() if t is AssessmentTier.ENHANCED]


class StageOutcome(str, Enum):
    """
    Explicit stage-completion vocabulary. CORRECTED: observing a
    deterministic configuration discrepancy does NOT make every
    downstream stage COMPLETED. The distinctions that matter:

    CONFORMANCE_VERIFIED
        A deterministic declared-vs-realized configuration check was
        performed and is conclusive on its own terms. This is what
        II-3 achieves by reading an artifact's realized architecture.
        It is NOT a comparative experimental evaluation.

    COMPLETED
        The stage's full intended work was performed, including any
        experimental comparison the stage calls for.

    NOT_PERFORMED
        The stage's work was not carried out in this execution (e.g.
        no comparative re-execution exists to evaluate). Distinct from
        NOT_APPLICABLE: the work is meaningful and still owed.

    NOT_APPLICABLE
        The stage's instrument genuinely does not apply to the evidence
        at hand (e.g. inferential statistics on a deterministic,
        exactly-verified configuration mismatch, where there is no
        sampling variability to reason about).

    NOT_ASSESSED
        An ENHANCED stage deliberately not undertaken.

    PARTIAL / FAILED
        Partially completed, or attempted and failed.
    """
    COMPLETED = "COMPLETED"
    CONFORMANCE_VERIFIED = "CONFORMANCE_VERIFIED"
    NOT_PERFORMED = "NOT_PERFORMED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_ASSESSED = "NOT_ASSESSED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


# Stage outcomes that mean "this CORE stage's work is genuinely done".
# CONFORMANCE_VERIFIED counts only for stages whose work IS a
# deterministic conformance check (II-1, II-3); it does not stand in
# for an unperformed comparative evaluation.
SATISFYING_OUTCOMES = {StageOutcome.COMPLETED, StageOutcome.CONFORMANCE_VERIFIED,
                       StageOutcome.NOT_APPLICABLE}


@dataclass(frozen=True)
class StageStatus:
    stage_id: StageId
    tier: AssessmentTier
    status: StageOutcome
    summary: str
    findings: list[str] = field(default_factory=list)

    @property
    def is_satisfied(self) -> bool:
        return self.status in SATISFYING_OUTCOMES

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id.value, "tier": self.tier.value,
            "status": self.status.value, "summary": self.summary, "findings": list(self.findings),
        }


@dataclass(frozen=True)
class ConvergenceContribution:
    """
    Whether this pillar supplies an independent-implementation
    convergence source toward methodology 3.8.4's LEVEL_4 criterion.

    This is an INPUT to the integration decision engine, not a verdict.
    `contributes=False` means only "no convergence source from THIS
    pillar"; it makes no claim about convergence sources available
    elsewhere (independent datasets, independent architectures), and
    must never be read as a cap on the attainable evidence level.
    """
    contributes: bool
    source_kind: Optional[str]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "contributes": self.contributes,
            "source_kind": self.source_kind,
            "rationale": self.rationale,
        }


def convergence_contribution(
    *, independent_reimplementation_status: EnhancedAssessmentStatus,
) -> ConvergenceContribution:
    """
    Report this pillar's contribution toward LEVEL_4 convergence. Does
    NOT decide the evidence level - see module docstring.
    """
    if independent_reimplementation_status is EnhancedAssessmentStatus.ASSESSED:
        return ConvergenceContribution(
            contributes=True,
            source_kind="independent_implementation",
            rationale=(
                "Independent reimplementation was performed, supplying one independent "
                "implementation path whose agreement (or disagreement) with the reference "
                "implementation is available to the integration engine as convergence evidence."
            ),
        )
    return ConvergenceContribution(
        contributes=False,
        source_kind=None,
        rationale=(
            "Independent reimplementation was NOT assessed, so Implementation Integrity "
            "contributes no independent-implementation convergence source. This says nothing "
            "about convergence sources from independent datasets or architectures elsewhere in "
            "the audit; LEVEL_4 remains determinable by the integration engine from whatever "
            "qualifying convergence evidence actually exists."
        ),
    )


def not_assessed_stage(stage_id: StageId, *, reason: str) -> StageStatus:
    """
    Construct an explicit NOT_ASSESSED record. Used so that an
    unperformed ENHANCED stage is visibly recorded rather than silently
    omitted, and is never described as having been performed at reduced
    rigor.
    """
    return StageStatus(
        stage_id=stage_id, tier=STAGE_TIERS[stage_id],
        status=StageOutcome.NOT_ASSESSED, summary=reason, findings=[],
    )
