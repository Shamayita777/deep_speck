"""
Decision semantics and certificate generation for Experimental Validity.

Decision vocabulary (fixed, do not extend ad hoc; matches the frozen
production certificate semantics):

    SUPPORTED       - the primary hypothesis (a material effect of the
                       manipulated factor) is supported by the
                       preregistered difference-detection test, AFTER
                       multiplicity correction, with the required valid
                       replicate count achieved.
    NOT_SUPPORTED   - formal practical equivalence (TOST, within the
                       predeclared +/-epsilon) is established. Only
                       reachable via assess_practical_equivalence();
                       "not significant" alone is never sufficient.
    INCONCLUSIVE    - insufficient valid replicates; non-significant
                       difference test WITHOUT formal equivalence;
                       insufficient statistical resolution/power; or
                       unresolved required provenance short of an
                       outright INVALID failure.
    NOT_RUN         - experiment not executed.
    INVALID         - provenance failure, firewall violation, dataset
                       mismatch, configuration mismatch, or other
                       predefined validity failure.

REVISION NOTE: assess_practical_equivalence previously approximated
equivalence via CI-containment using the primary (1-alpha) CI. Per
Schuirmann (1987) and Lakens (2017), correct TOST-based CI-containment
requires a (1-2*alpha) CI, not (1-alpha) - using the wider (1-alpha) CI
is overly conservative. This is now implemented directly via the two
one-sided t-tests (framework.statistics.tost_equivalence), which avoids
needing a second, differently-calibrated CI object entirely.

CRITICAL RULE (unchanged): a non-significant difference-detection test
NEVER by itself yields NOT_SUPPORTED. "p > alpha" is never interpreted
as evidence of no effect / equivalence - NOT_SUPPORTED requires a
formal, predeclared-epsilon TOST conclusion.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from framework.experiment import PracticalSignificance
from framework.manifest import build_manifest
from framework.provenance import utc_timestamp
from framework.statistics import PairedAnalysisResult, tost_equivalence


class DecisionState(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_RUN = "NOT_RUN"
    INVALID = "INVALID"


def assess_practical_equivalence(
    paired_result: PairedAnalysisResult,
    practical_significance: PracticalSignificance,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Formal practical-equivalence assessment via TOST. This is the ONLY
    place in the framework permitted to issue an equivalence-flavored
    verdict, and it categorically refuses to do so without a
    predeclared, justified epsilon.
    """
    if not practical_significance.is_available():
        return {
            "formal_practical_equivalence": "NOT_AVAILABLE",
            "reason": "no predeclared practical-significance threshold",
        }

    eps = practical_significance.threshold
    assert eps is not None
    try:
        tost = tost_equivalence(paired_result, epsilon=eps, alpha=alpha)
    except ValueError as exc:
        return {
            "formal_practical_equivalence": "NOT_AVAILABLE",
            "reason": f"TOST could not be computed: {exc}",
        }

    status = "EQUIVALENT_WITHIN_THRESHOLD" if tost.equivalent else "INCONCLUSIVE"

    return {
        "formal_practical_equivalence": status,
        "threshold": eps,
        "justification": practical_significance.justification,
        "tost": tost.to_dict(),
    }


def decide_difference_detection(
    *,
    adjusted_p_value: Optional[float],
    alpha: float,
    sufficient_replicates: bool,
) -> DecisionState:
    """
    Preregistered decision rule for the primary difference-detection
    objective (paired t-test, Holm-adjusted p-value).

        - insufficient valid replicates            -> INCONCLUSIVE
        - adjusted_p_value is None (test unavailable) -> INCONCLUSIVE
        - adjusted_p_value < alpha                 -> SUPPORTED
        - adjusted_p_value >= alpha                -> INCONCLUSIVE
                                                       (NEVER NOT_SUPPORTED)
    """
    if not sufficient_replicates:
        return DecisionState.INCONCLUSIVE
    if adjusted_p_value is None:
        return DecisionState.INCONCLUSIVE
    if adjusted_p_value < alpha:
        return DecisionState.SUPPORTED
    return DecisionState.INCONCLUSIVE


def decide_final(
    *,
    difference_decision: DecisionState,
    practical_equivalence: dict[str, Any],
) -> DecisionState:
    """
    Combine the (multiplicity-corrected) difference-detection decision
    with the equivalence assessment into the final certificate decision.

        - difference_decision == SUPPORTED           -> SUPPORTED
          (a detected difference is reported regardless of what TOST
          says; TOST is only consulted when no difference was detected)
        - difference_decision == INCONCLUSIVE and
              formal_practical_equivalence == EQUIVALENT_WITHIN_THRESHOLD
                                                        -> NOT_SUPPORTED
        - otherwise                                    -> INCONCLUSIVE
    """
    if difference_decision == DecisionState.SUPPORTED:
        return DecisionState.SUPPORTED
    if practical_equivalence.get("formal_practical_equivalence") == "EQUIVALENT_WITHIN_THRESHOLD":
        return DecisionState.NOT_SUPPORTED
    return DecisionState.INCONCLUSIVE


def build_certificate(
    *,
    manifest_fields: dict[str, Any],
    decision: DecisionState,
    what_was_tested: str,
    hypothesis_text: str,
    unit_of_replication: str,
    controls: list[str],
    what_changed: str,
    dataset_summary: dict[str, Any],
    exact_replay_available: bool,
    requested_replicates: int,
    valid_replicates: int,
    raw_replicate_results: list[dict[str, Any]],
    effect_size: Optional[float],
    confidence_interval: Optional[dict[str, float]],
    raw_p_value: Optional[float],
    adjusted_p_value: Optional[float],
    practical_significance_predeclared: bool,
    practical_equivalence: dict[str, Any],
    limitations: list[str],
    is_evidentiary: bool,
) -> dict[str, Any]:
    """
    Build the full human-and-machine-readable certificate. Fails closed:
    if manifest_fields is missing required provenance, build_manifest()
    raises rather than allowing an INVALID-but-otherwise-normal-looking
    certificate to be silently emitted with the wrong decision.
    """
    try:
        manifest = build_manifest(manifest_fields)
    except Exception as exc:
        # Fail closed: missing/malformed provenance forces INVALID,
        # overriding whatever decision the caller computed.
        return {
            "certificate_schema_version": "1.1",
            "decision": DecisionState.INVALID.value,
            "invalid_reason": str(exc),
            "is_evidentiary": False,
            "generated_at_utc": utc_timestamp(),
        }

    return {
        "certificate_schema_version": "1.1",
        "manifest": manifest,
        "narrative": {
            "what_was_tested": what_was_tested,
            "hypothesis": hypothesis_text,
            "unit_of_replication": unit_of_replication,
            "controls": controls,
            "what_changed": what_changed,
            "dataset": dataset_summary,
            "exact_replay_available": exact_replay_available,
        },
        "replication": {
            "requested_replicates": requested_replicates,
            "valid_replicates": valid_replicates,
            "raw_replicate_results": raw_replicate_results,
        },
        "statistics": {
            "effect_size": effect_size,
            "confidence_interval": confidence_interval,
            "raw_p_value": raw_p_value,
            "adjusted_p_value": adjusted_p_value,
        },
        "practical_significance": {
            "predeclared": practical_significance_predeclared,
            "assessment": practical_equivalence,
        },
        "decision": decision.value,
        "limitations": limitations,
        "is_evidentiary": is_evidentiary,
        "generated_at_utc": utc_timestamp(),
    }
