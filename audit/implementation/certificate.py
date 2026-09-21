"""
Implementation Integrity certificate assembly.

Converts stage statuses and findings into (a) a DimensionOutcome and
(b) a generic audit.common.evidence.Evidence object that the
integration layer can consume without knowing anything about
Implementation Integrity's internals.

DECISION RULE (methodology Section 3.5.6/3.5.8, applied literally):
    Rule III -> FAIL: an implementation difference produces practically
        meaningful changes in reported scientific conclusions.
    Rule II  -> CONDITIONAL_PASS: differences exist but are practically
        negligible / do not alter conclusions.
    Rule I   -> PASS: no implementation-specific behaviour is supported
        as an alternative explanation.
    Rule IV  -> INCONCLUSIVE: reproduction could not be completed
        because essential information is unavailable. Rule IV
        explicitly prevents missing artifacts from being read as
        evidence AGAINST the original study.

A MATERIAL finding does NOT by itself map to FAIL. Rule III requires
that the difference "produce practically meaningful changes in reported
scientific conclusions", so the severity of a finding is combined with
its ScientificImpactStatus:

    MATERIAL + DEMONSTRATED   -> FAIL (Rule III satisfied)
    MATERIAL + UNDEMONSTRATED -> INCONCLUSIVE (discrepancy confirmed,
                                 impact not established; downstream
                                 evidence marked REQUIRES_REEVALUATION)
    MATERIAL + REFUTED        -> CONDITIONAL_PASS (Rule II)
    SCOPE_LIMITING            -> CONDITIONAL_PASS (Rule II)
    CLARITY only              -> PASS (Rule I)

Where a FAIL IS issued, note carefully what it does and does not mean:
implementation-specific behaviour IS a supported alternative
explanation for the evidence produced with that artifact - not that the
underlying cryptographic hypothesis is false.
"""

from __future__ import annotations

from typing import Optional

from audit.common.evidence import Evidence
from audit.common.ids import Dimension
from audit.common.outcomes import (
    AssessmentTier,
    DimensionOutcome,
    EnhancedAssessmentStatus,
    EvidenceTag,
)
from audit.implementation.findings import (
    FindingSeverity,
    ImplementationFinding,
    ScientificImpactStatus,
)
from audit.implementation.stages import (
    StageId,
    StageStatus,
    convergence_contribution,
)


def decide_implementation_outcome(
    *,
    findings: list[ImplementationFinding],
    core_stages_satisfied: bool,
    unavailable_essential_information: Optional[list[str]] = None,
) -> tuple[DimensionOutcome, str]:
    """
    Apply methodology Section 3.5.6 decision rules. Returns
    (outcome, rationale).

    CORRECTED SEMANTICS:

    Rule III (FAIL) requires that implementation differences "produce
    practically meaningful changes in reported scientific conclusions".
    A MATERIAL finding whose scientific impact is UNDEMONSTRATED does
    NOT satisfy that test: the divergence is confirmed, but its effect
    on any conclusion has not been established. Such a finding yields
    INCONCLUSIVE - the discrepancy is recorded, downstream evidence is
    marked for re-evaluation, and no verdict about implementation
    integrity is issued until the controlled comparison exists.

    Rule IV (INCONCLUSIVE) now also correctly governs unavailable
    essential information. Previously this path returned
    CONDITIONAL_PASS, which was wrong: CONDITIONAL_PASS asserts that
    differences are practically negligible, a positive claim that
    missing evidence cannot support. Missing essential evidence means
    no conclusion is drawn - in EITHER direction.
    """
    unavailable = unavailable_essential_information or []

    if not core_stages_satisfied:
        return (
            DimensionOutcome.INCONCLUSIVE,
            "Rule IV: the CORE Implementation Integrity stages were not all satisfied, so no "
            "conclusion regarding implementation integrity is drawn. Missing or unperformed "
            "work is not interpreted as evidence against the original study.",
        )

    if unavailable:
        return (
            DimensionOutcome.INCONCLUSIVE,
            f"Rule IV: essential information required for evaluation was unavailable "
            f"({unavailable}), so implementation integrity is indeterminate. No conclusion is "
            "drawn in either direction; in particular, missing evidence is NOT treated as "
            "evidence that differences are practically negligible.",
        )

    material_demonstrated = [
        f for f in findings
        if f.severity is FindingSeverity.MATERIAL
        and f.scientific_impact is ScientificImpactStatus.DEMONSTRATED
    ]
    material_undemonstrated = [
        f for f in findings
        if f.severity is FindingSeverity.MATERIAL
        and f.scientific_impact is ScientificImpactStatus.UNDEMONSTRATED
    ]
    material_refuted = [
        f for f in findings
        if f.severity is FindingSeverity.MATERIAL
        and f.scientific_impact is ScientificImpactStatus.REFUTED
    ]
    scope_limiting = [f for f in findings if f.severity is FindingSeverity.SCOPE_LIMITING]

    if material_demonstrated:
        titles = "; ".join(f.finding_id for f in material_demonstrated)
        return (
            DimensionOutcome.FAIL,
            f"Rule III: implementation discrepancy/discrepancies ({titles}) are DEMONSTRATED to "
            "produce practically meaningful changes in reported scientific conclusions, making "
            "implementation-specific behaviour a supported alternative explanation. This "
            "concerns evidential support, not the truth of the underlying cryptographic hypothesis.",
        )

    if material_undemonstrated:
        titles = "; ".join(f.finding_id for f in material_undemonstrated)
        return (
            DimensionOutcome.INCONCLUSIVE,
            f"CONFIRMED implementation discrepancy ({titles}) against the declared reference "
            "protocol, with downstream evidence marked REQUIRES_REEVALUATION. However, Rule III "
            "is NOT satisfied: the discrepancy's effect on any reported scientific conclusion is "
            "UNDEMONSTRATED, because the controlled comparison that would establish it has not "
            "been run. The outcome is therefore INCONCLUSIVE - neither a FAIL (which would assert "
            "unproven scientific impact) nor a PASS (which would ignore a confirmed divergence).",
        )

    if scope_limiting or material_refuted:
        titles = "; ".join(f.finding_id for f in [*scope_limiting, *material_refuted])
        return (
            DimensionOutcome.CONDITIONAL_PASS,
            f"Rule II: implementation differences were observed ({titles}) but are established "
            "as practically negligible or scope-limiting rather than conclusion-altering.",
        )

    return (
        DimensionOutcome.PASS,
        "Rule I: no implementation-specific behaviour is supported as an alternative "
        "explanation for the reported results.",
    )


def _verify_preregistration_reference(
    reference: Optional[dict], *, verify_artifact: bool = True,
) -> tuple[bool, list[str]]:
    """
    Verify that a supplied preregistration reference is genuine.

    CORRECTED: previously any (artifact_path, content_hash) pair was
    accepted at face value, so CONFIRMATORY status could be obtained by
    supplying a fabricated path and an arbitrary hash. Since the whole
    point of the CONFIRMATORY tag is that a preregistration genuinely
    existed BEFORE execution, accepting an unverifiable reference
    defeats it entirely.

    This now requires that the referenced artifact EXISTS on disk and
    that its actual content hash MATCHES the declared one. A reference
    that fails either check does not confer CONFIRMATORY status, and
    the reason is recorded as a limitation rather than silently
    downgrading.

    `verify_artifact=False` exists only for callers that have already
    verified the artifact through another trusted channel; it does not
    weaken the shape checks.
    """
    if not reference:
        return False, []

    problems: list[str] = []
    path = reference.get("artifact_path")
    declared_hash = reference.get("content_hash")

    if not path:
        problems.append("no artifact_path supplied.")
    if not declared_hash:
        problems.append("no content_hash supplied.")
    if problems:
        return False, problems

    if verify_artifact:
        from pathlib import Path as _Path

        from audit.common.provenance import sha256_file

        artifact = _Path(path)
        if not artifact.exists():
            return False, [
                f"artifact {path!r} does not exist, so it cannot be evidence of a "
                "pre-execution preregistration."
            ]
        actual = sha256_file(artifact)
        if actual != declared_hash:
            return False, [
                f"artifact {path!r} hash mismatch: declared {declared_hash}, actual {actual}."
            ]

    return True, []


def build_implementation_evidence(
    *,
    evidence_id: str,
    claim_id: str,
    experiment_id: str,
    run_ids: list[str],
    stages: list[StageStatus],
    findings: list[ImplementationFinding],
    independent_reimplementation_status: EnhancedAssessmentStatus,
    provenance: dict,
    run_manifest_references: Optional[list[dict]] = None,
    preregistration_reference: Optional[dict] = None,
    additional_artifact_hashes: Optional[dict[str, str]] = None,
    verify_preregistration_artifact: bool = True,
    unavailable_essential_information: Optional[list[str]] = None,
) -> Evidence:
    """
    Assemble the Implementation Integrity Evidence object.

    Reports the pillar's convergence CONTRIBUTION (an input to the
    integration engine) rather than imposing an evidence-level cap -
    see audit.implementation.stages.convergence_contribution.

    EVIDENCE TAG IS DERIVED, NOT ASSERTED (corrected): the evidence is
    tagged CONFIRMATORY only if a genuine PRE-EXECUTION preregistration
    artifact is supplied via `preregistration_reference` (with a real
    path and content hash). Otherwise it is tagged EXPLORATORY.
    Confirmatory status cannot be manufactured after the fact by
    hashing a runtime-constructed description of what was done.

    TRACEABILITY (corrected): `run_manifest_references` embeds, in the
    evidence itself, the path and content hash of every RunManifest
    cited by run_ids - so a reviewer can walk Evidence -> RunManifest
    from the artifact alone, without access to an in-memory mapping.
    """
    manifest_refs = list(run_manifest_references or [])
    has_real_preregistration, preregistration_problems = _verify_preregistration_reference(
        preregistration_reference, verify_artifact=verify_preregistration_artifact,
    )
    tag = EvidenceTag.CONFIRMATORY if has_real_preregistration else EvidenceTag.EXPLORATORY
    core_stages = [s for s in stages if s.tier is AssessmentTier.CORE]
    core_satisfied = bool(core_stages) and all(s.is_satisfied for s in core_stages)
    unsatisfied = [s.stage_id.value for s in core_stages if not s.is_satisfied]

    outcome, rationale = decide_implementation_outcome(
        findings=findings, core_stages_satisfied=core_satisfied,
        unavailable_essential_information=unavailable_essential_information,
    )

    convergence = convergence_contribution(
        independent_reimplementation_status=independent_reimplementation_status,
    )

    limitations = [
        f"Independent reimplementation (ENHANCED stage II-2) is "
        f"{independent_reimplementation_status.value}. {convergence.rationale}",
    ]
    if unsatisfied:
        limitations.append(
            f"CORE stages not satisfied in this execution: {unsatisfied}. No implementation-"
            "integrity verdict is issued on their behalf."
        )
    for finding in findings:
        if finding.scientific_impact is ScientificImpactStatus.UNDEMONSTRATED:
            limitations.append(
                f"{finding.finding_id}: discrepancy CONFIRMED, scientific impact UNDEMONSTRATED. "
                "A controlled re-execution under the declared protocol is required before any "
                "claim that this discrepancy changed a scientific conclusion."
            )
        limitations.extend(finding.open_questions)
    if unavailable_essential_information:
        limitations.append(
            f"Essential information unavailable: {unavailable_essential_information}."
        )
    for problem in preregistration_problems:
        limitations.append(f"Preregistration reference rejected: {problem}")
    if not has_real_preregistration:
        limitations.append(
            "No pre-execution preregistration artifact exists for this execution, so the "
            "evidence is tagged EXPLORATORY. It records a confirmed, reproducible "
            "artifact-vs-declared-protocol conformance check, but it is not preregistered "
            "confirmatory evidence and must not be presented as such."
        )

    artifact_hashes: dict[str, str] = {}
    for finding in findings:
        artifact_hashes.update(finding.supporting_artifacts)
    # Embed manifest (and preregistration, if genuine) hashes so the
    # traceability chain is self-contained within the evidence artifact.
    for name, digest in (additional_artifact_hashes or {}).items():
        artifact_hashes[name] = digest
    for ref in manifest_refs:
        artifact_hashes[ref["artifact_path"]] = ref["content_hash"]
    if has_real_preregistration:
        artifact_hashes[preregistration_reference["artifact_path"]] = (
            preregistration_reference["content_hash"]
        )

    requires_reevaluation: list[str] = []
    for finding in findings:
        requires_reevaluation.extend(finding.requires_reevaluation())

    return Evidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        dimension=Dimension.IMPLEMENTATION,
        experiment_id=experiment_id,
        run_ids=run_ids,
        input_artifacts=sorted(artifact_hashes.keys()),
        artifact_hashes=artifact_hashes,
        observation={
            "stages": [s.to_dict() for s in stages],
            "findings": [f.to_dict() for f in findings],
            "independent_reimplementation_status": independent_reimplementation_status.value,
            "convergence_contribution": convergence.to_dict(),
            "run_manifest_references": manifest_refs,
            "preregistration_reference": preregistration_reference,
            "core_stages_satisfied": core_satisfied,
            "core_stages_unsatisfied": unsatisfied,
            "discrepancy_confirmed": any(
                f.severity is FindingSeverity.MATERIAL for f in findings
            ),
            "scientific_impact_demonstrated": any(
                f.scientific_impact is ScientificImpactStatus.DEMONSTRATED for f in findings
            ),
            "downstream_requires_reevaluation": sorted(set(requires_reevaluation)),
            "rationale": rationale,
        },
        statistics={
            "note": (
                "II-5 Statistical Assessment is NOT_APPLICABLE in this execution: the only "
                "completed verification was a deterministic declared-vs-realized configuration "
                "check, for which inferential statistics are not the appropriate instrument "
                "(there is no sampling variability to reason about). II-5 becomes applicable "
                "once II-4 supplies comparable populations of training runs."
            ),
        },
        decision=outcome.value,
        limitations=limitations,
        provenance=provenance,
        tag=tag,
    )
