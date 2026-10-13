"""
Implementation Integrity stage definitions (Pillar 1).

STAGE MODEL (methodology Section 3.5.4): five sequential CORE stages,

    II-1  Baseline Reconstruction
    II-2  Independent Reimplementation
    II-3  Controlled Verification
    II-4  Comparative Evaluation
    II-5  Statistical Assessment

HISTORY: certificates V1-V5 recorded II-2 as an ENHANCED (optional)
stage under a project-local revision note that existed only in this
module's docstring; no methodology document in the repository carries
that revision. II-2 is now CORE, as Section 3.5.4 specifies. Historical
certificates are preserved unchanged and still say ENHANCED; the
AssessmentTier.ENHANCED value is retained so they remain readable.

STAGE COMPONENTS
----------------
A stage may carry StageComponents: individually evidenced pieces of
work, each answering one stated question. Components exist so that
DIFFERENT QUESTIONS are never merged under one stage status. In
particular, II-4 as specified compares the REFERENCE and INDEPENDENT
implementations under identical experimental protocols. An experiment
that varies a single conformance factor (e.g. declared vs realized
depth) within one implementation answers a different question; it is
recorded as a CONFORMANCE_IMPACT_EXPERIMENT component and can never by
itself make II-4 or II-5 COMPLETED (see validate_stage_record).

CONVERGENCE (methodology Section 3.8.4)
---------------------------------------
LEVEL_4 is defined by convergence of independent experiments. This
module does NOT cap the attainable evidence level; it reports, via
`convergence_contribution()`, whether this pillar contributes an
independent-implementation convergence source. The integration engine
decides LEVEL_4 from whatever qualifying convergence evidence exists.

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
    StageId.II_2_INDEPENDENT_REIMPLEMENTATION: AssessmentTier.CORE,
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
        A stage deliberately not undertaken. For a CORE stage this is
        NOT satisfying.

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


class ComponentKind(str, Enum):
    """
    What question a StageComponent answers. The kinds that matter for
    II-4/II-5 are deliberately distinct:

    CONFORMANCE_IMPACT_EXPERIMENT
        Varies one declared-vs-realized factor within ONE implementation
        (e.g. depth 10 vs depth 5). Answers "does this discrepancy
        matter?" - NOT "do the reference and independent
        implementations agree?".
    FIXED_CHECKPOINT_IMPLEMENTATION_COMPARISON
        Evaluates one trained reference artifact and one trained
        independent artifact on common data. Scoped to those two
        checkpoints; says nothing about variation across training runs.
    FORMAL_IMPLEMENTATION_COMPARISON
        The comparison II-4 specifies: reference vs independent
        implementation under identical experimental protocols, with
        replicated training. Only this kind can make II-4/II-5 COMPLETED.
    """
    REFERENCE_VERIFICATION = "REFERENCE_VERIFICATION"
    INDEPENDENT_IMPLEMENTATION = "INDEPENDENT_IMPLEMENTATION"
    DETERMINISTIC_EQUIVALENCE = "DETERMINISTIC_EQUIVALENCE"
    ARCHITECTURE_CENSUS = "ARCHITECTURE_CENSUS"
    CONFORMANCE_IMPACT_EXPERIMENT = "CONFORMANCE_IMPACT_EXPERIMENT"
    FIXED_CHECKPOINT_IMPLEMENTATION_COMPARISON = "FIXED_CHECKPOINT_IMPLEMENTATION_COMPARISON"
    FORMAL_IMPLEMENTATION_COMPARISON = "FORMAL_IMPLEMENTATION_COMPARISON"


@dataclass(frozen=True)
class StageComponent:
    """
    One evidenced piece of a stage's work. `evidence` maps a
    repository-relative path (or other stable label) to the sha256 of
    the artifact actually inspected; `limitations` states what the
    component does NOT establish.
    """
    component_id: str
    kind: ComponentKind
    status: "StageOutcome"
    question: str
    summary: str
    evidence: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id, "kind": self.kind.value,
            "status": self.status.value, "question": self.question, "summary": self.summary,
            "evidence": dict(self.evidence), "limitations": list(self.limitations),
        }


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
    components: tuple[StageComponent, ...] = ()

    @property
    def is_satisfied(self) -> bool:
        return self.status in SATISFYING_OUTCOMES

    def to_dict(self) -> dict:
        out = {
            "stage_id": self.stage_id.value, "tier": self.tier.value,
            "status": self.status.value, "summary": self.summary, "findings": list(self.findings),
        }
        if self.components:
            out["components"] = [c.to_dict() for c in self.components]
        return out


def formal_comparison_completed(stage: StageStatus) -> bool:
    """True iff the stage carries a COMPLETED formal reference-vs-independent comparison."""
    return any(
        c.kind is ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON
        and c.status is StageOutcome.COMPLETED
        for c in stage.components
    )


def validate_stage_record(stage: StageStatus) -> list[str]:
    """
    Fail-closed semantic checks on a stage record. Returns problems;
    empty means the record does not overclaim.

    - II-4 / II-5 may be COMPLETED only on the strength of a COMPLETED
      FORMAL_IMPLEMENTATION_COMPARISON component. A conformance-impact
      experiment or a fixed-checkpoint comparison answers a different
      question and cannot stand in for it.
    - II-2 may be COMPLETED only if at least one component binds actual
      evidence (path -> sha256). Completion is a statement that the
      reimplementation exists and was exercised; it is NOT a statement
      that it agrees with the reference.
    """
    problems: list[str] = []
    sid = stage.stage_id.value
    if stage.tier is not STAGE_TIERS[stage.stage_id]:
        problems.append(
            f"{sid}: tier {stage.tier.value} disagrees with the stage registry "
            f"({STAGE_TIERS[stage.stage_id].value}).")
    if stage.stage_id in (StageId.II_4_COMPARATIVE_EVALUATION, StageId.II_5_STATISTICAL_ASSESSMENT):
        if stage.status is StageOutcome.COMPLETED and not formal_comparison_completed(stage):
            problems.append(
                f"{sid} is COMPLETED but carries no COMPLETED FORMAL_IMPLEMENTATION_COMPARISON "
                "component; other comparisons answer different questions and cannot complete it.")
    if stage.stage_id is StageId.II_2_INDEPENDENT_REIMPLEMENTATION:
        if stage.status is StageOutcome.COMPLETED and not any(c.evidence for c in stage.components):
            problems.append(f"{sid} is COMPLETED but binds no evidence artifacts.")
    for c in stage.components:
        for path, digest in c.evidence.items():
            if not (isinstance(digest, str) and len(digest) == 64):
                problems.append(f"{sid}/{c.component_id}: evidence {path!r} has no sha256.")
    return problems


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
    implementation_comparison_completed: Optional[bool] = None,
) -> ConvergenceContribution:
    """
    Report this pillar's contribution toward LEVEL_4 convergence. Does
    NOT decide the evidence level - see module docstring.

    `implementation_comparison_completed=False` states explicitly that
    the formal reference-vs-independent comparison (II-4) is still
    outstanding: an independent implementation then exists but its
    agreement with the reference at the level of experimental outcomes
    is not established, so it is not offered as a convergence source.
    `None` (legacy callers) does not make that distinction.
    """
    if (independent_reimplementation_status is EnhancedAssessmentStatus.ASSESSED
            and implementation_comparison_completed is False):
        return ConvergenceContribution(
            contributes=False,
            source_kind=None,
            rationale=(
                "Independent reimplementation was performed (II-2), but the formal comparison of "
                "the reference and independent implementations under identical experimental "
                "protocols (II-4) has not been completed. Agreement at the level of experimental "
                "outcomes is therefore not established, and this pillar does not yet offer an "
                "independent-implementation convergence source. This imposes no cap: LEVEL_4 "
                "remains determinable by the integration engine from other qualifying evidence."
            ),
        )
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
    Construct an explicit NOT_ASSESSED record, so that an unperformed
    stage is visibly recorded rather than silently omitted, and is never
    described as having been performed at reduced rigor. A NOT_ASSESSED
    CORE stage is not satisfied.
    """
    return StageStatus(
        stage_id=stage_id, tier=STAGE_TIERS[stage_id],
        status=StageOutcome.NOT_ASSESSED, summary=reason, findings=[],
    )
