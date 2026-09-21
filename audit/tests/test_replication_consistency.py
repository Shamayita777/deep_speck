import math

import pytest

from audit.common.replication_consistency import (
    ConsistencyMethod,
    EstimateDependence as _ED,
    EstimateDependence,
    SuppliedConsistencyVerdict,
    InputRealization,
    ProtocolIdentity,
    ReplicationComparisonClass,
    ReplicationConsistencyError,
    classify_replication_consistency,
    compute_prediction_interval,
)


def ident(**overrides):
    base = dict(
        experiment_id="EXP1", config_hash="cfg-abc", preregistration_hash="prereg-1",
    )
    base.update(overrides)
    return ProtocolIdentity(**base)


# --- protocol identity ---

def test_refuses_comparison_across_different_config_hash():
    with pytest.raises(ReplicationConsistencyError, match="config_hash"):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(config_hash="cfg-different"),
            prior_decision="SUPPORTED", new_decision="SUPPORTED",
            prior_point_estimate=0.9, new_point_estimate=0.9,
        )


def test_refuses_comparison_across_different_preregistration():
    with pytest.raises(ReplicationConsistencyError, match="preregistration_hash"):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(preregistration_hash="prereg-2"),
            prior_decision="SUPPORTED", new_decision="SUPPORTED",
            prior_point_estimate=0.9, new_point_estimate=0.9,
        )


def test_different_dataset_snapshot_does_not_block_comparison():
    """
    CORRECTED: a dataset snapshot identifies an input realization, not
    the protocol. A genuine independent replication draws FRESH data
    and therefore has a different snapshot id - blocking on that would
    forbid the very case this module serves.
    """
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.905,
        prior_standard_error=0.02, new_standard_error=0.02,
        dependence=_ED.INDEPENDENT,
        input_realization=InputRealization("DATASET-AUDIT-V1", "DATASET-AUDIT-V2"),
    )
    assert result.comparison_class is ReplicationComparisonClass.ORDINARY_VARIATION_SAME_DECISION
    assert result.input_realization_kind == "DIFFERENT_INPUT_REALIZATION"


def test_identical_dataset_snapshot_uses_neutral_same_input_label():
    """
    A SAME-input comparison must route through CALLER_SUPPLIED (the
    independent-estimate formula is invalid there), and the label
    reported must be the neutral input-level one.
    """
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.905,
        consistency_method=ConsistencyMethod.CALLER_SUPPLIED,
        supplied_consistency_verdict=SuppliedConsistencyVerdict(
            statistically_consistent=True,
            method_description="Paired-difference SE accounting for shared input realization.",
        ),
        input_realization=InputRealization("DATASET-AUDIT-V1", "DATASET-AUDIT-V1"),
    )
    assert result.input_realization_kind == "SAME_INPUT_REALIZATION"
    assert "REPRODUCTION" not in result.input_realization_kind


def test_same_input_realization_cannot_reach_independent_formula():
    """
    REGRESSION (verified issue 3): previously a SAME_INPUT_REALIZATION
    comparison silently used the independent-estimate prediction
    interval. It must now be refused.
    """
    with pytest.raises(ReplicationConsistencyError, match="SAME input realization"):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(),
            prior_decision="SUPPORTED", new_decision="SUPPORTED",
            prior_point_estimate=0.90, new_point_estimate=0.905,
            prior_standard_error=0.02, new_standard_error=0.02,
            dependence=_ED.INDEPENDENT,
            input_realization=InputRealization("SNAP-1", "SNAP-1"),
        )


def test_dependence_defaults_to_unknown_and_refuses_prediction_interval():
    """
    REGRESSION: omitting `dependence` must NOT silently apply the
    independent formula. The default is UNKNOWN, which is refused.
    """
    with pytest.raises(ReplicationConsistencyError, match="INDEPENDENT"):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(),
            prior_decision="SUPPORTED", new_decision="SUPPORTED",
            prior_point_estimate=0.90, new_point_estimate=0.905,
            prior_standard_error=0.02, new_standard_error=0.02,
        )


def test_dependent_declaration_refused_for_prediction_interval():
    with pytest.raises(ReplicationConsistencyError):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(),
            prior_decision="SUPPORTED", new_decision="SUPPORTED",
            prior_point_estimate=0.90, new_point_estimate=0.905,
            prior_standard_error=0.02, new_standard_error=0.02,
            dependence=_ED.DEPENDENT,
        )


# --- the three conceptual outcomes plus insufficiency ---

def test_decision_flip_is_materially_contradictory_regardless_of_tiny_delta():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="INCONCLUSIVE",
        prior_point_estimate=0.90, new_point_estimate=0.9001,
        prior_standard_error=0.01, new_standard_error=0.01,
        dependence=_ED.INDEPENDENT, practical_threshold=0.05,
    )
    assert result.comparison_class is ReplicationComparisonClass.MATERIALLY_CONTRADICTORY
    assert result.decision_changed is True


def test_ordinary_variation_when_within_pi_and_threshold():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.905,
        prior_standard_error=0.02, new_standard_error=0.02,
        dependence=_ED.INDEPENDENT, practical_threshold=0.05,
    )
    assert result.comparison_class is ReplicationComparisonClass.ORDINARY_VARIATION_SAME_DECISION
    assert result.within_prediction_interval is True
    assert result.exceeds_practical_threshold is False


def test_magnitude_shift_when_outside_pi_but_decision_held():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.80,
        prior_standard_error=0.01, new_standard_error=0.01,
        dependence=_ED.INDEPENDENT,
    )
    assert result.comparison_class is ReplicationComparisonClass.MAGNITUDE_SHIFT_SAME_DECISION
    assert result.within_prediction_interval is False


def test_magnitude_shift_when_exceeds_threshold_only():
    # Inside a wide prediction interval, but beyond the practical threshold.
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.82,
        prior_standard_error=0.10, new_standard_error=0.10,
        dependence=_ED.INDEPENDENT, practical_threshold=0.05,
    )
    assert result.within_prediction_interval is True
    assert result.exceeds_practical_threshold is True
    assert result.comparison_class is ReplicationComparisonClass.MAGNITUDE_SHIFT_SAME_DECISION


def test_materially_contradictory_when_both_axes_fire_without_decision_flip():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.70,
        prior_standard_error=0.01, new_standard_error=0.01,
        dependence=_ED.INDEPENDENT, practical_threshold=0.05,
    )
    assert result.comparison_class is ReplicationComparisonClass.MATERIALLY_CONTRADICTORY
    assert result.decision_changed is False


def test_insufficient_evidence_when_no_uncertainty_and_no_threshold():
    """Matching decisions alone must NOT be certified as ordinary variation."""
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.10,  # huge shift, no uncertainty info
    )
    assert result.comparison_class is ReplicationComparisonClass.INSUFFICIENT_EVIDENCE
    assert result.within_prediction_interval is None
    assert result.exceeds_practical_threshold is None


def test_insufficient_evidence_when_only_one_standard_error_supplied():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.91,
        prior_standard_error=0.01,  # new_standard_error missing
    )
    assert result.comparison_class is ReplicationComparisonClass.INSUFFICIENT_EVIDENCE


def test_threshold_alone_within_bound_yields_insufficient_evidence_not_ordinary():
    """
    CORRECTED: a practical threshold alone cannot establish ordinary
    STATISTICAL variation. "Smaller than a threshold I chose" is a
    statement about importance, not about sampling error.
    """
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.905, practical_threshold=0.05,
    )
    assert result.comparison_class is ReplicationComparisonClass.INSUFFICIENT_EVIDENCE


def test_threshold_alone_when_exceeded_still_yields_magnitude_shift():
    """Exceeding a threshold needs no sampling-error argument, so this direction remains valid."""
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.70, practical_threshold=0.05,
    )
    assert result.comparison_class is ReplicationComparisonClass.MAGNITUDE_SHIFT_SAME_DECISION


def test_caller_supplied_consistency_method_supports_non_normal_metrics():
    """
    Metrics that violate the normal approximation (bounded proportions
    near 0/1, raw correlations) must be assessable via a caller-computed
    verdict rather than silently pushed through a prediction interval.
    """
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.998, new_point_estimate=0.9985,
        consistency_method=ConsistencyMethod.CALLER_SUPPLIED,
        supplied_consistency_verdict=SuppliedConsistencyVerdict(
            statistically_consistent=True,
            method_description="Bootstrap predictive distribution for a bounded proportion.",
            evidence_reference="analysis/bootstrap_v1.json",
        ),
    )
    assert result.comparison_class is ReplicationComparisonClass.ORDINARY_VARIATION_SAME_DECISION
    assert result.consistency_method == "CALLER_SUPPLIED"


def test_caller_supplied_method_without_verdict_is_insufficient_evidence():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.998, new_point_estimate=0.9985,
        consistency_method=ConsistencyMethod.CALLER_SUPPLIED,
    )
    assert result.comparison_class is ReplicationComparisonClass.INSUFFICIENT_EVIDENCE


def test_caller_supplied_inconsistent_verdict_yields_magnitude_shift():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.998, new_point_estimate=0.80,
        consistency_method=ConsistencyMethod.CALLER_SUPPLIED,
        supplied_consistency_verdict=SuppliedConsistencyVerdict(
            statistically_consistent=False,
            method_description="Bootstrap predictive distribution for a bounded proportion.",
        ),
    )
    assert result.comparison_class is ReplicationComparisonClass.MAGNITUDE_SHIFT_SAME_DECISION


# --- CI overlap is descriptive only ---

def test_ci_overlap_is_descriptive_and_does_not_drive_classification():
    """
    Overlapping CIs must NOT rescue a result that is outside the
    prediction interval - this is the CI-overlap fallacy the module
    docstring rejects (Greenland et al. 2016).
    """
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.90, new_point_estimate=0.80,
        prior_standard_error=0.01, new_standard_error=0.01,
        dependence=_ED.INDEPENDENT,
        prior_confidence_interval=(0.60, 0.95),  # deliberately wide, overlapping
        new_confidence_interval=(0.70, 0.99),
    )
    assert result.descriptive_confidence_intervals_overlap is True
    # Classification is still driven by the prediction interval, not overlap.
    assert result.comparison_class is ReplicationComparisonClass.MAGNITUDE_SHIFT_SAME_DECISION


def test_result_serializes_and_records_methodology():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="PASS", new_decision="PASS",
        prior_point_estimate=0.5, new_point_estimate=0.5,
        prior_standard_error=0.01, new_standard_error=0.01,
        dependence=_ED.INDEPENDENT,
    )
    d = result.to_dict()
    assert "Prediction interval" in d["methodology"]
    assert "descriptive only" in d["methodology"]
    assert d["prediction_interval_width"] is not None


def test_rejects_non_finite_point_estimates():
    with pytest.raises(ReplicationConsistencyError):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(),
            prior_decision="PASS", new_decision="PASS",
            prior_point_estimate=float("inf"), new_point_estimate=0.5,
        )


def test_rejects_negative_practical_threshold():
    with pytest.raises(ReplicationConsistencyError):
        classify_replication_consistency(
            prior_identity=ident(), new_identity=ident(),
            prior_decision="PASS", new_decision="PASS",
            prior_point_estimate=0.5, new_point_estimate=0.5, practical_threshold=-0.1,
        )


# --- correction 6: prediction-interval hardening ---

def test_prediction_interval_rejects_non_positive_critical_value():
    for bad in (0.0, -1.96):
        with pytest.raises(ReplicationConsistencyError, match="critical_value must be positive"):
            compute_prediction_interval(
                prior_estimate=0.5, prior_se=0.01, new_se=0.01, critical_value=bad)


def test_prediction_interval_refuses_dependent_estimates():
    """Adding variances is invalid under correlated sampling errors."""
    with pytest.raises(ReplicationConsistencyError, match="INDEPENDENT"):
        compute_prediction_interval(
            prior_estimate=0.5, prior_se=0.01, new_se=0.01,
            dependence=EstimateDependence.DEPENDENT)


def test_prediction_interval_refuses_unknown_dependence():
    """UNKNOWN is treated as dependent for safety, never silently assumed independent."""
    with pytest.raises(ReplicationConsistencyError):
        compute_prediction_interval(
            prior_estimate=0.5, prior_se=0.01, new_se=0.01,
            dependence=EstimateDependence.UNKNOWN)


def test_prediction_interval_accepts_explicit_independence():
    low, high = compute_prediction_interval(
        prior_estimate=0.5, prior_se=0.01, new_se=0.01,
        dependence=EstimateDependence.INDEPENDENT)
    assert low < 0.5 < high


# --- correction 7: supplied-verdict provenance ---

def test_supplied_verdict_requires_method_description():
    with pytest.raises(ReplicationConsistencyError, match="method_description"):
        SuppliedConsistencyVerdict(statistically_consistent=True, method_description="")


def test_supplied_verdict_rejects_whitespace_only_description():
    with pytest.raises(ReplicationConsistencyError):
        SuppliedConsistencyVerdict(statistically_consistent=True, method_description="   ")


def test_supplied_verdict_is_recorded_in_result_for_audit():
    result = classify_replication_consistency(
        prior_identity=ident(), new_identity=ident(),
        prior_decision="SUPPORTED", new_decision="SUPPORTED",
        prior_point_estimate=0.99, new_point_estimate=0.991,
        consistency_method=ConsistencyMethod.CALLER_SUPPLIED,
        supplied_consistency_verdict=SuppliedConsistencyVerdict(
            statistically_consistent=True,
            method_description="Exact binomial predictive interval.",
            evidence_reference="analysis/exact_v2.json",
        ),
    )
    recorded = result.to_dict()["supplied_consistency_verdict"]
    assert recorded["method_description"] == "Exact binomial predictive interval."
    assert recorded["evidence_reference"] == "analysis/exact_v2.json"
