"""
Pillar interpretation layer.

Converts each pillar's NATIVE evidence vocabulary into the standardized
dimension-level input the decision engine consumes, recording for every
transformation whether the rule applied is defined by the methodology
or introduced by this integration layer.

WHY THIS LAYER EXISTS
---------------------
The methodology (Section 3.10.5) takes ONE outcome per dimension as
already given and specifies how the four dimensions combine into a
claim decision. It does NOT specify:

  (a) how multiple sub-audits within a dimension (D1..D5, CE1..CE4)
      combine into that dimension's single outcome; or
  (b) how a pillar's native non-standard status maps onto the
      standardized PASS/CONDITIONAL_PASS/FAIL/INCONCLUSIVE vocabulary.

Verified by inspection: dataset/dataset/common/certificate.py enforces
the four-value vocabulary PER SUB-AUDIT, and no roll-up module exists
anywhere in the dataset pillar. There is therefore no authoritative
dimension-level certificate to ingest.

Every rule below is consequently tagged:

  SOURCE_DEFINED
      Stated by the methodology. Cited to its section.

  INTEGRATION_OPERATIONALIZATION
      Introduced here because the methodology is silent. Carries an
      explicit rationale and is versioned. These are METHODOLOGICAL
      CHOICES BY THIS LAYER, not findings of the methodology, and must
      never be reported as the latter.

NON-COERCION RULE
-----------------
Native states with no methodology-defined mapping are NEVER forced into
PASS or FAIL. They are marked UNMAPPABLE, and the dimension-level
consequence of an UNMAPPABLE input is itself an explicitly declared
operationalization (see AGGREGATION_RULE_* below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from audit.common.ids import Dimension
from audit.common.outcomes import CryptographicOutcome, DimensionOutcome

INTERPRETATION_VERSION = "pillar-interpretation-v1"


class RuleProvenance(str, Enum):
    SOURCE_DEFINED = "SOURCE_DEFINED"
    INTEGRATION_OPERATIONALIZATION = "INTEGRATION_OPERATIONALIZATION"


class Mappability(str, Enum):
    DIRECTLY_MAPPABLE = "DIRECTLY_MAPPABLE"
    UNMAPPABLE = "UNMAPPABLE"
    NOT_PRODUCED = "NOT_PRODUCED"


@dataclass(frozen=True)
class InterpretationRecord:
    """One auditable native-status -> standardized-outcome transformation."""
    source_evidence_id: str
    source_pillar_id: str
    source_dimension: Dimension
    source_native_status: str
    mappability: Mappability
    normalized_outcome: Optional[str]
    rule_id: str
    rule_provenance: RuleProvenance
    rationale: str
    supporting_evidence_references: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    decision_bearing: bool = True
    version: str = INTERPRETATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_evidence_id": self.source_evidence_id,
            "source_pillar_id": self.source_pillar_id,
            "source_dimension": self.source_dimension.value,
            "source_native_status": self.source_native_status,
            "mappability": self.mappability.value,
            "normalized_outcome": self.normalized_outcome,
            "rule_id": self.rule_id,
            "rule_provenance": self.rule_provenance.value,
            "rationale": self.rationale,
            "supporting_evidence_references": list(self.supporting_evidence_references),
            "limitations": list(self.limitations),
            "decision_bearing": self.decision_bearing,
            "version": self.version,
        }


# ---------------------------------------------------------------------
# Native-status mapping table.
#
# Statuses already in the methodology's own vocabularies map directly
# (SOURCE_DEFINED). Everything else is UNMAPPABLE - deliberately not
# coerced.
# ---------------------------------------------------------------------

_STANDARD_DIMENSION_STATES = {"PASS", "CONDITIONAL_PASS", "FAIL", "INCONCLUSIVE"}
_STANDARD_CRYPTO_STATES = {"SUPPORTED", "PARTIALLY_SUPPORTED", "INCONCLUSIVE", "NOT_SUPPORTED"}

# Native statuses observed in this project's real artifacts that have NO
# methodology-defined mapping. Each records WHY it is not coerced.
UNMAPPABLE_NATIVE_STATES: dict[str, str] = {
    "DESCRIPTIVE_ONLY": (
        "D3's own artifact states it 'does not establish equality of the full underlying "
        "distributions'. A sub-audit that explicitly disclaims establishing its conclusion "
        "cannot be read as PASS, and it reports no failure, so it cannot be read as FAIL."
    ),
    "EFFECT_DETECTED": (
        "D4 reports that a controlled perturbation produced a detectable effect. Whether a "
        "detected effect is reassuring or alarming depends on the perturbation's purpose, "
        "which the status string alone does not encode; mapping it either way would add an "
        "interpretation the artifact does not make."
    ),
    "NOT_PERFORMED": (
        "Work that was not carried out yields no evidence in either direction."
    ),
    "NOT_ASSESSED": (
        "An ENHANCED assessment deliberately not undertaken contributes no evidence."
    ),
    "NOT_LOCATED": (
        "An artifact whose hash could not be re-verified cannot support any outcome."
    ),
    "NOT_PRODUCED": (
        "No production evidence exists for this component."
    ),
    "NOT_RECORDED_IN_ARTIFACT": (
        "The artifact records no outcome field at all."
    ),
}


def interpret_native_status(
    *,
    source_evidence_id: str,
    source_pillar_id: str,
    dimension: Dimension,
    native_status: str,
    supporting_references: Optional[list[str]] = None,
    limitations: Optional[list[str]] = None,
    decision_bearing: bool = True,
) -> InterpretationRecord:
    """Interpret one native status. Never coerces an unmapped state."""
    refs = list(supporting_references or [])
    lims = list(limitations or [])

    standard = (_STANDARD_CRYPTO_STATES if dimension is Dimension.CRYPTOGRAPHIC
                else _STANDARD_DIMENSION_STATES)

    if native_status in standard:
        return InterpretationRecord(
            source_evidence_id=source_evidence_id, source_pillar_id=source_pillar_id,
            source_dimension=dimension, source_native_status=native_status,
            mappability=Mappability.DIRECTLY_MAPPABLE, normalized_outcome=native_status,
            rule_id="RULE-DIRECT-VOCABULARY",
            rule_provenance=RuleProvenance.SOURCE_DEFINED,
            rationale=(
                "The native status is already a member of the vocabulary the methodology "
                "defines for this dimension (Sections 3.5.8/3.6.8/3.7.8 for the audit "
                "dimensions, 3.8.9 for Cryptographic Evidence), so it is carried through "
                "unchanged with no interpretation added."
            ),
            supporting_evidence_references=refs, limitations=lims,
            decision_bearing=decision_bearing,
        )

    reason = UNMAPPABLE_NATIVE_STATES.get(
        native_status,
        "No methodology-defined mapping exists for this native status, and this layer does "
        "not invent one.",
    )
    return InterpretationRecord(
        source_evidence_id=source_evidence_id, source_pillar_id=source_pillar_id,
        source_dimension=dimension, source_native_status=native_status,
        mappability=(Mappability.NOT_PRODUCED if native_status in
                     ("NOT_PRODUCED", "NOT_RECORDED_IN_ARTIFACT")
                     else Mappability.UNMAPPABLE),
        normalized_outcome=None,
        rule_id="RULE-NO-COERCION",
        rule_provenance=RuleProvenance.INTEGRATION_OPERATIONALIZATION,
        rationale=(
            f"Refusing to coerce native status {native_status!r} into the standardized "
            f"vocabulary. {reason} Marking it UNMAPPABLE preserves the information the "
            "pillar deliberately encoded; forcing a PASS/FAIL would discard it."
        ),
        supporting_evidence_references=refs, limitations=lims,
        decision_bearing=decision_bearing,
    )


# ---------------------------------------------------------------------
# Dimension aggregation.
#
# EXPLICITLY AN INTEGRATION OPERATIONALIZATION. Verified by inspection:
# the methodology does not define within-dimension aggregation, and no
# pillar in this repository produces a dimension-level roll-up.
# ---------------------------------------------------------------------

AGGREGATION_RULE_ID = "RULE-AGG-CONSERVATIVE-V1"
AGGREGATION_RULE_RATIONALE = (
    "Within-dimension aggregation is NOT defined by the methodology (Section 3.10.5 takes one "
    "outcome per dimension as already given) and no pillar in this repository produces a "
    "dimension-level roll-up. This layer therefore introduces an explicit conservative rule:\n"
    "  1. any decision-bearing FAIL          -> dimension FAIL\n"
    "  2. else any decision-bearing evidence that is UNMAPPABLE, NOT_PRODUCED or INCONCLUSIVE "
    "-> dimension INCONCLUSIVE\n"
    "  3. else any CONDITIONAL_PASS           -> dimension CONDITIONAL_PASS\n"
    "  4. else all PASS and at least one present -> dimension PASS\n"
    "  5. no decision-bearing evidence at all -> dimension INCONCLUSIVE\n"
    "Rule 2 is conservative because the methodology's own Rule IV logic (3.5.6) treats absent "
    "or insufficient evidence as grounds for drawing NO conclusion rather than a favourable "
    "one; extending that posture to unmapped states keeps the layer from manufacturing "
    "support the evidence does not provide. It is NOT claimed to be methodology-mandated."
)


@dataclass(frozen=True)
class DimensionInterpretation:
    dimension: Dimension
    outcome: str
    records: list[InterpretationRecord]
    rule_id: str
    rule_provenance: RuleProvenance
    rationale: str
    driving_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value, "outcome": self.outcome,
            "rule_id": self.rule_id, "rule_provenance": self.rule_provenance.value,
            "rationale": self.rationale,
            "driving_evidence_ids": list(self.driving_evidence_ids),
            "records": [r.to_dict() for r in self.records],
        }


CONFLICT_PRECEDENCE_RULE_ID = "RULE-CONFLICT-PRECEDENCE"
CONFLICT_PRECEDENCE_RATIONALE = (
    "Methodology Section 3.8.8 states that where contradictory evidence persists after the "
    "conflict-resolution procedure, the affected claim SHALL be classified as inconclusive, "
    "and that conflicting evidence shall not be discarded merely because it contradicts other "
    "observations. That rule is SOURCE_DEFINED.\n\n"
    "RULE-AGG-CONSERVATIVE-V1 is an INTEGRATION_OPERATIONALIZATION introduced by this layer "
    "because the methodology does not define within-dimension aggregation. An integration "
    "operationalization must never override a source-defined rule.\n\n"
    "Therefore: when a dimension's evidence includes a party to an UNRESOLVED conflict, the "
    "conflict rule takes precedence and the dimension is INCONCLUSIVE - even where the "
    "aggregation rule alone would have resolved it to NOT_SUPPORTED or SUPPORTED. Without this "
    "precedence, one arm of an unresolved contradiction would silently decide the dimension, "
    "which is exactly the outcome 3.8.8 prohibits."
)


def aggregate_dimension(
    dimension: Dimension,
    records: list[InterpretationRecord],
    conflicted_evidence_ids: Optional[set] = None,
) -> DimensionInterpretation:
    """
    Apply RULE-AGG-CONSERVATIVE-V1, subject to RULE-CONFLICT-PRECEDENCE.

    `conflicted_evidence_ids` names evidence that is party to an
    UNRESOLVED conflict. If any decision-bearing record is in that set,
    the source-defined conflict rule (3.8.8) takes precedence over this
    layer's aggregation operationalization and the dimension is
    INCONCLUSIVE. See CONFLICT_PRECEDENCE_RATIONALE.
    """
    bearing = [r for r in records if r.decision_bearing]
    conflicted_evidence_ids = conflicted_evidence_ids or set()

    inconclusive_like = [
        r for r in bearing
        if r.mappability in (Mappability.UNMAPPABLE, Mappability.NOT_PRODUCED)
        or r.normalized_outcome == "INCONCLUSIVE"
    ]
    failed = [r for r in bearing if r.normalized_outcome in ("FAIL", "NOT_SUPPORTED")]
    conditional = [r for r in bearing if r.normalized_outcome == "CONDITIONAL_PASS"]
    partial = [r for r in bearing if r.normalized_outcome == "PARTIALLY_SUPPORTED"]

    def build(outcome: str, drivers: list[InterpretationRecord], step: str) -> DimensionInterpretation:
        return DimensionInterpretation(
            dimension=dimension, outcome=outcome, records=records,
            rule_id=AGGREGATION_RULE_ID,
            rule_provenance=RuleProvenance.INTEGRATION_OPERATIONALIZATION,
            rationale=f"{step}\n\n{AGGREGATION_RULE_RATIONALE}",
            driving_evidence_ids=[r.source_evidence_id for r in drivers],
        )

    # SOURCE_DEFINED rule takes precedence over this layer's operationalization.
    conflicted = [r for r in bearing
                  if r.source_evidence_id in conflicted_evidence_ids
                  or r.source_pillar_id in conflicted_evidence_ids]
    if conflicted:
        return DimensionInterpretation(
            dimension=dimension, outcome="INCONCLUSIVE", records=records,
            rule_id=CONFLICT_PRECEDENCE_RULE_ID,
            rule_provenance=RuleProvenance.SOURCE_DEFINED,
            rationale=(
                f"Methodology 3.8.8 applied: decision-bearing evidence from "
                f"{sorted({r.source_pillar_id for r in conflicted})} is party to an unresolved "
                f"conflict, so the dimension is INCONCLUSIVE regardless of what the "
                f"aggregation operationalization alone would have produced.\n\n"
                f"{CONFLICT_PRECEDENCE_RATIONALE}"
            ),
            driving_evidence_ids=[r.source_evidence_id for r in conflicted],
        )

    if not bearing:
        return build(
            "INCONCLUSIVE" if dimension is not Dimension.CRYPTOGRAPHIC else "INCONCLUSIVE",
            [], "Rule 5 applied: no decision-bearing evidence exists for this dimension.")
    if failed:
        outcome = "NOT_SUPPORTED" if dimension is Dimension.CRYPTOGRAPHIC else "FAIL"
        return build(outcome, failed,
                     f"Rule 1 applied: decision-bearing evidence reports "
                     f"{[r.source_pillar_id for r in failed]} as failing/not-supported.")
    if inconclusive_like:
        return build("INCONCLUSIVE", inconclusive_like,
                     f"Rule 2 applied: decision-bearing evidence from "
                     f"{[r.source_pillar_id for r in inconclusive_like]} is INCONCLUSIVE, "
                     "UNMAPPABLE or NOT_PRODUCED.")
    if partial:
        return build("PARTIALLY_SUPPORTED", partial, "Rule 3 applied (cryptographic partial).")
    if conditional:
        return build("CONDITIONAL_PASS", conditional,
                     f"Rule 3 applied: {[r.source_pillar_id for r in conditional]} "
                     "conditional.")
    outcome = "SUPPORTED" if dimension is Dimension.CRYPTOGRAPHIC else "PASS"
    return build(outcome, bearing, "Rule 4 applied: all decision-bearing evidence passes.")


def to_dimension_outcome(interpretation: DimensionInterpretation) -> DimensionOutcome:
    return DimensionOutcome(interpretation.outcome)


def to_cryptographic_outcome(interpretation: DimensionInterpretation) -> CryptographicOutcome:
    return CryptographicOutcome(interpretation.outcome)
