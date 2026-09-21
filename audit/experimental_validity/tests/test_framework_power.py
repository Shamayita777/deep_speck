import numpy as np
import pytest

from framework.power import (
    UnderpoweredReplicatePlanError,
    power_unavailable,
    required_replicates_for_power,
    simulate_power,
    validate_replicate_plan_against_power,
)
from framework.seeds import statistics_rng


def test_power_forbids_observed_production_source():
    with pytest.raises(ValueError):
        simulate_power(
            target_effect=0.02, target_effect_source="observed_production",
            noise_sd=0.01, n_replicates=10, rng=statistics_rng(1),
        )


def test_power_rejects_unknown_source():
    with pytest.raises(ValueError):
        simulate_power(
            target_effect=0.02, target_effect_source="made_up_source",
            noise_sd=0.01, n_replicates=10, rng=statistics_rng(1),
        )


def test_power_accepts_predeclared_source():
    result = simulate_power(
        target_effect=0.05, target_effect_source="predeclared",
        noise_sd=0.01, n_replicates=10, n_simulations=200, rng=statistics_rng(2),
    )
    assert 0.0 <= result.empirical_power <= 1.0
    assert result.prospective is True


def test_power_large_effect_yields_high_power():
    """A large target effect relative to noise should yield high empirical power."""
    result = simulate_power(
        target_effect=0.10, target_effect_source="predeclared",
        noise_sd=0.01, n_replicates=15, n_simulations=500, rng=statistics_rng(3),
    )
    assert result.empirical_power > 0.9


def test_power_tiny_effect_yields_low_power():
    result = simulate_power(
        target_effect=0.001, target_effect_source="predeclared",
        noise_sd=0.05, n_replicates=5, n_simulations=500, rng=statistics_rng(4),
    )
    assert result.empirical_power < 0.3


def test_power_unavailable_explicit_record():
    record = power_unavailable("no defensible target effect exists")
    assert record["prospective_power"] == "UNAVAILABLE"
    assert "reason" in record


def test_required_replicates_returns_none_when_infeasible():
    result = required_replicates_for_power(
        target_effect=0.0001, target_effect_source="predeclared", noise_sd=0.05,
        target_power=0.99, n_simulations=100, max_replicates=5, rng=statistics_rng(5),
    )
    assert result is None


def test_power_simulates_ttest_not_wilcoxon():
    """
    Regression test: power.py must simulate the ACTUAL final procedure
    (paired t-test), not scipy.stats.wilcoxon. Verified indirectly by
    confirming the default procedure name and that requesting an
    unknown procedure raises rather than silently falling back.
    """
    result = simulate_power(
        procedure="difference_ttest", target_effect=0.05, target_effect_source="predeclared",
        noise_sd=0.01, n_replicates=10, n_simulations=200, rng=statistics_rng(6),
    )
    assert result.procedure == "difference_ttest"
    with pytest.raises(ValueError):
        simulate_power(
            procedure="wilcoxon", target_effect=0.05, target_effect_source="predeclared",
            noise_sd=0.01, n_replicates=10, n_simulations=200, rng=statistics_rng(6),
        )


def test_power_equivalence_tost_requires_epsilon():
    with pytest.raises(ValueError):
        simulate_power(
            procedure="equivalence_tost", target_effect=0.0, target_effect_source="pilot_variance_only",
            noise_sd=0.01, n_replicates=10, n_simulations=200, rng=statistics_rng(7),
        )


def test_power_equivalence_tost_high_power_when_truly_equivalent_and_low_noise():
    """Power to correctly conclude equivalence when the true effect is 0 and noise is small relative to epsilon."""
    result = simulate_power(
        procedure="equivalence_tost", target_effect=0.0, target_effect_source="pilot_variance_only",
        noise_sd=0.002, epsilon=0.01, n_replicates=20, n_simulations=500, rng=statistics_rng(8),
    )
    assert result.empirical_power > 0.8


def test_power_equivalence_tost_low_power_when_noise_comparable_to_epsilon():
    result = simulate_power(
        procedure="equivalence_tost", target_effect=0.0, target_effect_source="pilot_variance_only",
        noise_sd=0.01, epsilon=0.01, n_replicates=5, n_simulations=500, rng=statistics_rng(9),
    )
    assert result.empirical_power < 0.5


def test_validate_replicate_plan_no_op_when_no_power_analysis_declared():
    """required_n=None (not yet computed) must not raise - the separate placeholder check handles that case."""
    validate_replicate_plan_against_power(
        minimum_valid_replicates=5, power_analysis_required_n=None,
    )  # must not raise


def test_validate_replicate_plan_passes_when_minimum_meets_required_n():
    validate_replicate_plan_against_power(
        minimum_valid_replicates=24, power_analysis_required_n=24,
    )  # must not raise: exactly meets required_n
    validate_replicate_plan_against_power(
        minimum_valid_replicates=30, power_analysis_required_n=24,
    )  # must not raise: exceeds required_n


def test_validate_replicate_plan_rejects_underpowered_without_justification():
    """The exact scenario named in the audit: power-sized n=24, minimum_valid_replicates=20, no justification."""
    with pytest.raises(UnderpoweredReplicatePlanError):
        validate_replicate_plan_against_power(
            minimum_valid_replicates=20, power_analysis_required_n=24,
        )


def test_validate_replicate_plan_rejects_empty_justification():
    with pytest.raises(UnderpoweredReplicatePlanError):
        validate_replicate_plan_against_power(
            minimum_valid_replicates=20, power_analysis_required_n=24,
            underpowered_justification="   ",  # whitespace-only, not a real justification
        )


def test_validate_replicate_plan_allows_underpowered_with_explicit_justification():
    validate_replicate_plan_against_power(
        minimum_valid_replicates=20, power_analysis_required_n=24,
        underpowered_justification="Explicit, recorded decision to accept reduced power due to X.",
    )  # must not raise
