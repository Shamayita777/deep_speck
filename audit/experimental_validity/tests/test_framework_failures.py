import pytest

from framework.failures import (
    FailureReason,
    ReplicateOutcome,
    ReplicateStatus,
    detect_failure_from_history,
    enforce_minimum_valid_replicates,
)


def test_valid_replicate_requires_metric_value():
    with pytest.raises(ValueError):
        ReplicateOutcome(
            replicate_id="r1", condition_id="c1", run_id="run1",
            status=ReplicateStatus.VALID, metric_value=None,
        )


def test_failed_replicate_requires_reason():
    with pytest.raises(ValueError):
        ReplicateOutcome(
            replicate_id="r1", condition_id="c1", run_id="run1",
            status=ReplicateStatus.FAILED, failure_reason=None,
        )


def test_detect_failure_from_history_nan():
    assert detect_failure_from_history({"loss": [0.5, float("nan")]}) == FailureReason.NAN


def test_detect_failure_from_history_inf():
    assert detect_failure_from_history({"loss": [0.5, float("inf")]}) == FailureReason.DIVERGENCE


def test_detect_failure_from_history_clean():
    assert detect_failure_from_history({"loss": [0.5, 0.3, 0.1], "val_acc": [0.6, 0.7, 0.9]}) is None


def test_minimum_valid_replicates_never_reduces_denominator():
    """CASE 9: intentional failed replicate must remain visible, not dropped."""
    outcomes = [
        ReplicateOutcome("r1", "c1", "run1", ReplicateStatus.VALID, metric_value=0.9),
        ReplicateOutcome("r2", "c1", "run2", ReplicateStatus.FAILED, failure_reason=FailureReason.NAN),
        ReplicateOutcome("r3", "c1", "run3", ReplicateStatus.VALID, metric_value=0.91),
    ]
    sufficient, summary = enforce_minimum_valid_replicates(outcomes, minimum_required=3)
    assert sufficient is False
    assert summary["requested_replicates"] == 3
    assert summary["valid_replicates"] == 2
    assert summary["failed_replicates"] == 1
    assert summary["failure_reasons"][0]["replicate_id"] == "r2"
    assert summary["failure_reasons"][0]["reason"] == "nan"


def test_minimum_valid_replicates_sufficient_case():
    outcomes = [
        ReplicateOutcome("r1", "c1", "run1", ReplicateStatus.VALID, metric_value=0.9),
        ReplicateOutcome("r2", "c1", "run2", ReplicateStatus.VALID, metric_value=0.91),
    ]
    sufficient, summary = enforce_minimum_valid_replicates(outcomes, minimum_required=2)
    assert sufficient is True
    assert summary["valid_replicates"] == 2
