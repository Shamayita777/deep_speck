"""
Versioned Implementation Integrity findings, with downstream dependency
tracking.

A Finding records a discrepancy between the DECLARED reference protocol
and what an artifact/implementation ACTUALLY realizes, together with
the downstream evidence whose validity depends on it. Findings are
immutable and versioned: a corrected re-execution creates a NEW
finding version and NEVER edits or overwrites the historical one, nor
the historical downstream evidence it refers to.

The dependency model implemented here follows the project's stated
rule: a change at stage X preserves historical X evidence, creates a
new version of X, identifies downstream evidence dependent on X, and
marks ONLY the affected downstream claim evidence for
re-evaluation - never silently altering upstream historical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from audit.common.ids import Dimension, validate_identifier
from audit.common.provenance import utc_timestamp


class FindingSeverity(str, Enum):
    """
    Severity is about EVIDENTIAL CONSEQUENCE, not code quality.

    MATERIAL        - plausibly changes a scientific conclusion; downstream
                      evidence depending on it requires re-evaluation.
    SCOPE_LIMITING  - does not invalidate the result but narrows what it
                      can be said to establish.
    CLARITY         - misleading naming/documentation/dead parameters with
                      no effect on any computed result.
    """
    MATERIAL = "MATERIAL"
    SCOPE_LIMITING = "SCOPE_LIMITING"
    CLARITY = "CLARITY"


class ScientificImpactStatus(str, Enum):
    """
    CORRECTED, and central to this module's semantics.

    Methodology Rule III requires that implementation differences
    "produce practically meaningful changes in reported scientific
    conclusions" before Implementation Integrity is judged FAIL.
    Confirming that an artifact DIVERGES from the declared protocol is
    a different and weaker statement than demonstrating that the
    divergence CHANGED a scientific conclusion.

    DEMONSTRATED
        Controlled evidence shows the discrepancy changes a reported
        scientific conclusion. Satisfies Rule III -> FAIL.

    UNDEMONSTRATED
        The discrepancy is confirmed, but its effect on any scientific
        conclusion has not been established (typically because the
        controlled comparison has not been run). Does NOT satisfy
        Rule III. Downstream evidence requires re-evaluation, but no
        FAIL verdict is warranted yet.

    REFUTED
        Controlled evidence shows the discrepancy does NOT change the
        conclusion -> Rule II territory.

    NOT_APPLICABLE
        For findings (e.g. CLARITY) that cannot affect a conclusion by
        construction.
    """
    DEMONSTRATED = "DEMONSTRATED"
    UNDEMONSTRATED = "UNDEMONSTRATED"
    REFUTED = "REFUTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DownstreamStatus(str, Enum):
    REQUIRES_REEVALUATION = "REQUIRES_REEVALUATION"
    UNAFFECTED = "UNAFFECTED"
    OPEN_PENDING_EVIDENCE = "OPEN_PENDING_EVIDENCE"


@dataclass(frozen=True)
class DownstreamDependency:
    """One piece of downstream evidence whose validity depends on a finding."""
    evidence_reference: str
    dimension: Dimension
    status: DownstreamStatus
    rationale: str
    historical_artifact_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_reference": self.evidence_reference,
            "dimension": self.dimension.value,
            "status": self.status.value,
            "rationale": self.rationale,
            "historical_artifact_preserved": self.historical_artifact_preserved,
        }


@dataclass(frozen=True)
class ImplementationFinding:
    finding_id: str            # versioned, e.g. "II-FINDING-DEPTH-V1"
    stage_id: str
    severity: FindingSeverity
    scientific_impact: ScientificImpactStatus
    title: str
    declared: dict[str, Any]
    actual: dict[str, Any]
    verification_method: str
    supporting_artifacts: dict[str, str]   # name -> sha256
    downstream: list[DownstreamDependency] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recorded_at_utc: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        validate_identifier(self.finding_id, context="finding_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id, "stage_id": self.stage_id,
            "severity": self.severity.value,
            "scientific_impact": self.scientific_impact.value, "title": self.title,
            "declared": self.declared, "actual": self.actual,
            "verification_method": self.verification_method,
            "supporting_artifacts": dict(self.supporting_artifacts),
            "downstream": [d.to_dict() for d in self.downstream],
            "open_questions": list(self.open_questions),
            "recorded_at_utc": self.recorded_at_utc,
        }

    def requires_reevaluation(self) -> list[str]:
        return [d.evidence_reference for d in self.downstream
                if d.status is DownstreamStatus.REQUIRES_REEVALUATION]


# ---------------------------------------------------------------------
# The depth finding, as actually verified during this audit.
#
# Verification method (reproducible): h5py layer-group census on the
# real checkpoint files, deriving depth = (n_conv1d - 1) / 2. Both CE1
# checkpoints returned 11 Conv1D groups -> depth 5, while the reference
# protocol (train_5_rounds.py) declares depth=10.
#
# What is NOT asserted here: that depth alone quantitatively accounts
# for the observed 0.9291 -> 0.6108 accuracy gap. That requires a
# controlled depth-10 re-execution and is recorded as an open question,
# not stated as established.
# ---------------------------------------------------------------------

DEPTH_FINDING_V1 = ImplementationFinding(
    finding_id="II-FINDING-DEPTH-V1",
    stage_id="II-3_CONTROLLED_VERIFICATION",
    severity=FindingSeverity.MATERIAL,
    scientific_impact=ScientificImpactStatus.UNDEMONSTRATED,
    title=(
        "Realized residual depth of the audited checkpoint is 5, while the reference "
        "protocol (train_5_rounds.py) declares depth=10"
    ),
    declared={
        "depth": 10,
        "source": "reference/train_5_rounds.py: train_speck_distinguisher(200, num_rounds=5, depth=10)",
        "reference_reported_best_val_accuracy": 0.9291,
    },
    actual={
        "depth": 5,
        "n_conv1d_layers": 11,
        "derivation": "depth = (n_conv1d_layers - 1) / 2 = (11 - 1) / 2 = 5",
        "mechanism": (
            "The CE GohrModel constructor defaults to depth=5 and no CE driver script "
            "passes an explicit depth override, so the reference protocol's depth=10 is "
            "never applied."
        ),
        "observed_ce1_baseline_score": 0.6108,
    },
    verification_method=(
        "h5py layer-group census on the actual .h5 weight files (no TensorFlow, no model "
        "deserialization), cross-checked against the reference architecture's implied "
        "Conv1D count of 1 + 2*depth. Architecture read from the artifact itself, not "
        "inferred from its filename."
    ),
    supporting_artifacts={
        "best5depth10 (10).h5": "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea",
        "signal_destroyed.h5": "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308",
        "reference/train_5_rounds.py": "63b8bec9779a6b9a95cda4abdbd763b2b8aef384df45a40109c926ff8864ff24",
        "reference/train_nets.py": "b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669",
    },
    downstream=[
        DownstreamDependency(
            evidence_reference="evidence/ce1/ce1_certificate.json",
            dimension=Dimension.CRYPTOGRAPHIC,
            status=DownstreamStatus.REQUIRES_REEVALUATION,
            rationale=(
                "CE1 trains its own models via adapter.train() with no depth override, so its "
                "baseline and signal-destroyed models are both depth-5. Its INCONCLUSIVE verdict "
                "was produced at an architecture that does not match the reference protocol."
            ),
        ),
        DownstreamDependency(
            evidence_reference="evidence/ce2/ce2_certificate_{1..5}.json",
            dimension=Dimension.CRYPTOGRAPHIC,
            status=DownstreamStatus.REQUIRES_REEVALUATION,
            rationale=(
                "CE2 loads the depth-5 checkpoint. Its NOT_SUPPORTED verdict is a property of "
                "that model instance and is not established for a reference-conformant depth-10 model."
            ),
        ),
        DownstreamDependency(
            evidence_reference="evidence/ce3/ce3_certificate.json",
            dimension=Dimension.CRYPTOGRAPHIC,
            status=DownstreamStatus.REQUIRES_REEVALUATION,
            rationale=(
                "CE3 loads the same depth-5 checkpoint. Its statistical design is sound "
                "(replicate-level unit, calibration-gated, Bonferroni-corrected), so this is a "
                "scope/applicability issue rather than a methodological defect: the SUPPORTED "
                "verdict holds for the depth-5 model, not necessarily for the declared baseline."
            ),
        ),
        DownstreamDependency(
            evidence_reference="evidence/ce4/ce4_certificate.json",
            dimension=Dimension.CRYPTOGRAPHIC,
            status=DownstreamStatus.REQUIRES_REEVALUATION,
            rationale=(
                "CE4 loads the same depth-5 checkpoint and reuses CE3's task object. Its "
                "per-example causal finding is scoped to that one model instance."
            ),
        ),
        DownstreamDependency(
            evidence_reference="dataset/evidence/d1..d5",
            dimension=Dimension.DATASET,
            status=DownstreamStatus.UNAFFECTED,
            rationale=(
                "The finding is downstream of the dataset: it concerns model construction, not "
                "dataset generation, serialization, preprocessing, or storage. Per the project's "
                "dependency rule, Dataset Integrity is therefore NOT reopened and D1-D5 are "
                "neither rerun nor modified."
            ),
        ),
        DownstreamDependency(
            evidence_reference="experimental_validity (EV-BASELINE, EV-NOISE, H-EV-SHUFFLE, H-EV-REPRESENTATION)",
            dimension=Dimension.EXPERIMENTAL,
            status=DownstreamStatus.UNAFFECTED,
            rationale=(
                "EV constructs its own models through its own independent adapter with depth "
                "supplied explicitly from its frozen baseline, and does not load the CE "
                "checkpoint. No EV evidence depends on this artifact. (EV production evidence "
                "does not yet exist in any case.)"
            ),
        ),
    ],
    open_questions=[
        "Whether depth alone quantitatively accounts for the 0.9291 -> 0.6108 gap is NOT "
        "established; confirming it requires a controlled depth-10 re-execution under the "
        "reference protocol. Recorded as OPEN, not asserted.",
        "Whether the depth-5 default was intentional, accidental, or a filename mismatch is "
        "not determinable from the available artifacts: no run manifest exists for the "
        "historical CE executions. Recorded as OPEN.",
        "CE2 (NOT_SUPPORTED) vs CE3/CE4 (SUPPORTED) on the same verified-identical target "
        "quantity is an unresolved scientific conflict, independent of this finding; it is "
        "deferred to audit/integration/conflict_resolution.py.",
    ],
)


CE3_CLARITY_FINDING_V1 = ImplementationFinding(
    finding_id="II-FINDING-CE3-DEADPARAMS-V1",
    stage_id="II-3_CONTROLLED_VERIFICATION",
    severity=FindingSeverity.CLARITY,
    scientific_impact=ScientificImpactStatus.NOT_APPLICABLE,
    title="CE3 driver passes supported/inconclusive thresholds that its evaluator branch never reads",
    declared={
        "driver_passes": "CryptographicEvaluator(supported_threshold=0.20, inconclusive_threshold=0.05)",
    },
    actual={
        "effective_gate": (
            "evaluation.py's `test.name == 'Representation Interpretation'` branch decides "
            "SUPPORTED from metadata flags calibration_validated and primary_supported, which are "
            "computed inside the CE3 test from a Bonferroni-corrected alpha of 0.025 (m=2). The "
            "0.20/0.05 constructor arguments are never consulted on this path."
        ),
        "scientific_effect": "None - the historical CE3 decision is unaffected.",
    },
    verification_method="Source inspection of experiments/ce3 driver and evaluation.py branch logic.",
    supporting_artifacts={},
    downstream=[
        DownstreamDependency(
            evidence_reference="evidence/ce3/ce3_certificate.json",
            dimension=Dimension.CRYPTOGRAPHIC,
            status=DownstreamStatus.UNAFFECTED,
            rationale=(
                "The unused parameters had no effect on the computed decision, so the historical "
                "CE3 evidence is not invalidated by this finding. Documentation-level correction only; "
                "the historical artifact is not edited."
            ),
        ),
    ],
    open_questions=[
        "A dimension-agnostic evaluator that branches on a hard-coded test name contradicts its "
        "own module docstring ('contains no knowledge of any specific cryptographic test'). "
        "Recommended cleanup, not a scientific defect.",
    ],
)


ALL_FINDINGS = [DEPTH_FINDING_V1, CE3_CLARITY_FINDING_V1]


class FindingObservationMismatch(RuntimeError):
    """
    Raised when a stored finding's recorded artifact hashes or observed
    values do not match what was ACTUALLY inspected in this execution.

    This is the guard against a certificate citing a hash or a measured
    value that differs from the artifact the runner really opened -
    which would make the evidence untraceable at best and wrong at
    worst.
    """


def verify_finding_against_observations(
    finding: ImplementationFinding,
    *,
    observed_artifact_hashes: dict,
    observed_values: dict,
    verifiable_keys: tuple = (),
) -> list[str]:
    """
    Verify a stored finding against this execution's actual
    observations. Returns a list of mismatches; empty means the stored
    finding faithfully describes what was inspected.

    `observed_artifact_hashes` maps artifact name -> sha256 actually
    computed this run. `observed_values` carries measured quantities
    (e.g. {"depth": 5}) to check against the finding's `actual` block.
    Artifacts recorded in the finding but NOT inspected this run are
    reported, so a certificate can never silently carry forward a hash
    for something it did not open.
    """
    problems: list[str] = []

    for name, recorded_hash in finding.supporting_artifacts.items():
        if name.startswith("reference/"):
            continue  # reference sources are verified separately by II-1
        if name not in observed_artifact_hashes:
            problems.append(
                f"{finding.finding_id}: records artifact {name!r} that was not inspected in "
                "this execution; the certificate would cite an unverified hash."
            )
            continue
        if observed_artifact_hashes[name] != recorded_hash:
            problems.append(
                f"{finding.finding_id}: artifact {name!r} hash mismatch - finding records "
                f"{recorded_hash}, actual artifact is {observed_artifact_hashes[name]}."
            )

    for key, observed in observed_values.items():
        if key in finding.actual and finding.actual[key] != observed:
            problems.append(
                f"{finding.finding_id}: recorded actual[{key!r}]={finding.actual[key]!r} "
                f"does not match observed value {observed!r}."
            )

    # FAIL CLOSED: every factual key the finding asserts under `actual`
    # and which is declared verifiable must be positively supported by an
    # observation from this execution. Previously, a key that was simply
    # ABSENT from observed_values (e.g. because inspected checkpoints
    # disagreed and the runner therefore omitted the aggregate) passed
    # silently - meaning a stored factual claim could be published
    # unverified precisely in the case where the artifacts conflicted.
    for key in verifiable_keys or ():
        if key not in finding.actual:
            continue
        if key not in observed_values:
            problems.append(
                f"{finding.finding_id}: recorded actual[{key!r}]={finding.actual[key]!r} "
                "was NOT verified against this execution's observations (no unambiguous "
                "observed value was available - e.g. inspected artifacts disagreed). "
                "Refusing to publish an unverified factual claim."
            )

    return problems
