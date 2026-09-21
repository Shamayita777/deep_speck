from framework.certificate import (
    DecisionState,
    assess_practical_equivalence,
    build_certificate,
    decide_difference_detection,
    decide_final,
)
from framework.experiment import PracticalSignificance
from framework.manifest import REQUIRED_MANIFEST_FIELDS
from framework.statistics import paired_analysis, tost_equivalence
from framework.seeds import statistics_rng
import numpy as np


NO_EPS = PracticalSignificance(threshold=None, predeclared=False, justification=None)


def test_decide_difference_detection_insufficient_replicates_is_inconclusive():
    assert decide_difference_detection(adjusted_p_value=0.001, alpha=0.05, sufficient_replicates=False) == DecisionState.INCONCLUSIVE


def test_decide_difference_detection_significant_is_supported():
    assert decide_difference_detection(adjusted_p_value=0.01, alpha=0.05, sufficient_replicates=True) == DecisionState.SUPPORTED


def test_decide_difference_detection_nonsignificant_is_inconclusive_never_not_supported():
    """
    CRITICAL: p >= alpha must NEVER produce NOT_SUPPORTED (that would be
    exactly the forbidden "p > alpha => no effect / equivalence" logic).
    """
    decision = decide_difference_detection(adjusted_p_value=0.5, alpha=0.05, sufficient_replicates=True)
    assert decision == DecisionState.INCONCLUSIVE
    assert decision != DecisionState.NOT_SUPPORTED


def test_practical_equivalence_unavailable_without_epsilon():
    a = np.array([0.90, 0.91, 0.89, 0.905])
    b = np.array([0.901, 0.905, 0.895, 0.90])
    result = paired_analysis(a, b, rng=statistics_rng(1))
    assessment = assess_practical_equivalence(result, NO_EPS)
    assert assessment["formal_practical_equivalence"] == "NOT_AVAILABLE"
    assert "no predeclared" in assessment["reason"]


def test_practical_equivalence_available_with_epsilon():
    eps_available = PracticalSignificance(threshold=0.05, predeclared=True, justification="test")
    a = np.array([0.90, 0.91, 0.89, 0.905, 0.902, 0.898])
    b = np.array([0.901, 0.905, 0.895, 0.90, 0.903, 0.899])
    result = paired_analysis(a, b, rng=statistics_rng(2))
    assessment = assess_practical_equivalence(result, eps_available)
    assert assessment["formal_practical_equivalence"] in (
        "EQUIVALENT_WITHIN_THRESHOLD", "NOT_EQUIVALENT", "INCONCLUSIVE"
    )
    assert assessment["threshold"] == 0.05


def test_practical_equivalence_matches_manual_tost_computation():
    """
    Regression test for the CI-width bug found in the second audit round:
    the assessment must match a direct TOST computation exactly (not an
    approximation via a differently-calibrated CI).
    """
    eps_available = PracticalSignificance(threshold=0.01, predeclared=True, justification="test")
    a = np.array([0.90, 0.905, 0.895, 0.902, 0.898, 0.901, 0.899, 0.903])
    b = np.array([0.901, 0.904, 0.897, 0.900, 0.899, 0.902, 0.898, 0.901])
    result = paired_analysis(a, b, rng=statistics_rng(3))
    assessment = assess_practical_equivalence(result, eps_available)
    manual_tost = tost_equivalence(result, epsilon=0.01, alpha=0.05)
    expected_status = "EQUIVALENT_WITHIN_THRESHOLD" if manual_tost.equivalent else "INCONCLUSIVE"
    assert assessment["formal_practical_equivalence"] == expected_status
    assert abs(assessment["tost"]["p_tost"] - manual_tost.p_tost) < 1e-9


def test_decide_final_supported_overrides_equivalence():
    decision = decide_final(
        difference_decision=DecisionState.SUPPORTED,
        practical_equivalence={"formal_practical_equivalence": "EQUIVALENT_WITHIN_THRESHOLD"},
    )
    assert decision == DecisionState.SUPPORTED


def test_decide_final_not_supported_requires_formal_equivalence():
    decision = decide_final(
        difference_decision=DecisionState.INCONCLUSIVE,
        practical_equivalence={"formal_practical_equivalence": "EQUIVALENT_WITHIN_THRESHOLD"},
    )
    assert decision == DecisionState.NOT_SUPPORTED


def test_decide_final_inconclusive_when_neither_condition_met():
    decision = decide_final(
        difference_decision=DecisionState.INCONCLUSIVE,
        practical_equivalence={"formal_practical_equivalence": "NOT_AVAILABLE"},
    )
    assert decision == DecisionState.INCONCLUSIVE
    decision2 = decide_final(
        difference_decision=DecisionState.INCONCLUSIVE,
        practical_equivalence={"formal_practical_equivalence": "INCONCLUSIVE"},
    )
    assert decision2 == DecisionState.INCONCLUSIVE


def test_certificate_fails_closed_on_missing_provenance():
    cert = build_certificate(
        manifest_fields={"experiment_id": "X"},  # deliberately incomplete
        decision=DecisionState.SUPPORTED,  # caller claims SUPPORTED...
        what_was_tested="x", hypothesis_text="x", unit_of_replication="x", controls=[],
        what_changed="x", dataset_summary={}, exact_replay_available=False,
        requested_replicates=1, valid_replicates=1, raw_replicate_results=[],
        effect_size=None, confidence_interval=None, raw_p_value=None, adjusted_p_value=None,
        practical_significance_predeclared=False, practical_equivalence={}, limitations=[],
        is_evidentiary=True,
    )
    # ...but missing provenance must force INVALID regardless of the claimed decision.
    assert cert["decision"] == DecisionState.INVALID.value
    assert cert["is_evidentiary"] is False


def test_certificate_succeeds_with_complete_provenance():
    fields = {f: None for f in REQUIRED_MANIFEST_FIELDS}
    fields.update({"experiment_id": "H-EV-SHUFFLE", "decision": "SUPPORTED"})
    cert = build_certificate(
        manifest_fields=fields, decision=DecisionState.SUPPORTED,
        what_was_tested="x", hypothesis_text="x", unit_of_replication="x", controls=[],
        what_changed="x", dataset_summary={}, exact_replay_available=False,
        requested_replicates=5, valid_replicates=5, raw_replicate_results=[],
        effect_size=0.1, confidence_interval={"low": 0.01, "high": 0.2},
        raw_p_value=0.01, adjusted_p_value=0.02,
        practical_significance_predeclared=False,
        practical_equivalence={"formal_practical_equivalence": "NOT_AVAILABLE"},
        limitations=[], is_evidentiary=True,
    )
    assert cert["decision"] == "SUPPORTED"
    assert cert["is_evidentiary"] is True
