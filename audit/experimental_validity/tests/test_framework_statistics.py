import numpy as np
import pytest

from framework.statistics import descriptive_statistics, paired_analysis, tost_equivalence
from framework.seeds import statistics_rng


def test_descriptive_statistics_basic():
    rng = statistics_rng(1)
    values = np.array([0.5, 0.6, 0.55, 0.58])
    stats = descriptive_statistics(values, rng=rng)
    assert stats.n == 4
    assert abs(stats.mean - np.mean(values)) < 1e-9
    assert stats.minimum == 0.5
    assert stats.maximum == 0.6
    assert stats.ci_low <= stats.mean <= stats.ci_high


def test_paired_analysis_no_effect():
    """CASE 1: no experimental effect - diffs centered near zero."""
    rng = statistics_rng(2)
    a = rng.normal(0.9, 0.01, size=30)
    b = a + rng.normal(0, 0.001, size=30)
    result = paired_analysis(a, b, rng=statistics_rng(3))
    assert abs(result.mean_diff) < 0.01
    assert result.p_value is not None


def test_paired_analysis_strong_effect():
    """CASE 2: known strong effect - large, clearly detectable difference."""
    rng = statistics_rng(4)
    a = rng.normal(0.9, 0.005, size=30)
    b = a - 0.3  # large, consistent decrease
    result = paired_analysis(a, b, rng=statistics_rng(5))
    assert result.mean_diff < -0.25
    assert result.p_value is not None
    assert result.p_value < 0.01
    assert result.ci_high < 0


def test_paired_analysis_requires_equal_length():
    with pytest.raises(ValueError):
        paired_analysis(np.array([1.0, 2.0]), np.array([1.0]), rng=statistics_rng(6))


def test_paired_analysis_requires_minimum_pairs():
    with pytest.raises(ValueError):
        paired_analysis(np.array([1.0]), np.array([2.0]), rng=statistics_rng(7))


def test_paired_analysis_degenerate_zero_diffs():
    """CASE: all-zero differences must not crash and must be flagged."""
    a = np.array([0.9, 0.9, 0.9, 0.9])
    b = np.array([0.9, 0.9, 0.9, 0.9])
    result = paired_analysis(a, b, rng=statistics_rng(8))
    assert result.test_name == "paired_t_test_degenerate"
    assert any("exactly zero" in w for w in result.warnings)


def test_paired_analysis_primary_test_is_t_test_targeting_the_mean():
    """
    Regression test for the estimand/test mismatch defect: the primary
    p-value must come from a test whose null is mu_D=0 (paired t-test),
    not from Wilcoxon (whose null is symmetry about zero). Verified by
    checking the reported test_statistic against an independent
    scipy.stats.ttest_1samp computation on the same diffs.
    """
    from scipy import stats as scipy_stats

    rng = statistics_rng(20)
    a = rng.normal(0.90, 0.01, size=12)
    b = rng.normal(0.89, 0.01, size=12)
    result = paired_analysis(a, b, rng=statistics_rng(21))
    diffs = b - a
    expected = scipy_stats.ttest_1samp(diffs, popmean=0.0)
    assert result.test_name == "paired_t_test"
    assert abs(result.test_statistic - expected.statistic) < 1e-9
    assert abs(result.p_value - expected.pvalue) < 1e-9
    assert result.degrees_of_freedom == len(diffs) - 1


def test_paired_analysis_primary_ci_is_t_distribution_based():
    """The primary CI must match a hand-computed t-distribution CI, not the bootstrap CI."""
    from scipy import stats as scipy_stats

    rng = statistics_rng(22)
    a = rng.normal(0.90, 0.01, size=15)
    b = rng.normal(0.88, 0.01, size=15)
    result = paired_analysis(a, b, rng=statistics_rng(23))
    diffs = b - a
    n = len(diffs)
    mean_diff = np.mean(diffs)
    se = np.std(diffs, ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    expected_low, expected_high = mean_diff - t_crit * se, mean_diff + t_crit * se
    assert abs(result.ci_low - expected_low) < 1e-9
    assert abs(result.ci_high - expected_high) < 1e-9
    # Secondary bootstrap CI is reported but is a DIFFERENT quantity (resampled).
    assert result.secondary_bootstrap_ci_low != result.ci_low or result.secondary_bootstrap_ci_high != result.ci_high


def test_tost_equivalence_declares_equivalent_for_tiny_effect_large_n():
    """CASE 3: small effect below practical threshold, enough replicates to resolve it."""
    rng = statistics_rng(30)
    a = rng.normal(0.90, 0.003, size=40)
    b = a + rng.normal(0.0005, 0.0005, size=40)  # tiny, consistent difference
    result = paired_analysis(a, b, rng=statistics_rng(31))
    tost = tost_equivalence(result, epsilon=0.01, alpha=0.05)
    assert tost.equivalent is True


def test_tost_equivalence_not_equivalent_for_large_effect():
    rng = statistics_rng(32)
    a = rng.normal(0.90, 0.005, size=20)
    b = a - 0.05 + rng.normal(0, 0.002, size=20)  # far larger than epsilon=0.01, with genuine variance
    result = paired_analysis(a, b, rng=statistics_rng(33))
    tost = tost_equivalence(result, epsilon=0.01, alpha=0.05)
    assert tost.equivalent is False


def test_tost_equivalence_inconclusive_case():
    """CASE 4: insufficient power/resolution - small n, moderate noise, small n gives wide bounds."""
    rng = statistics_rng(34)
    a = rng.normal(0.90, 0.02, size=3)
    b = a + rng.normal(0.005, 0.02, size=3)
    result = paired_analysis(a, b, rng=statistics_rng(35))
    tost = tost_equivalence(result, epsilon=0.01, alpha=0.05)
    # With n=3 and this much noise relative to epsilon, equivalence should
    # not be concluded (neither direction has enough power to resolve it).
    assert tost.equivalent is False


def test_tost_requires_positive_epsilon():
    rng = statistics_rng(36)
    a = rng.normal(0.9, 0.01, size=10)
    b = rng.normal(0.9, 0.01, size=10)
    result = paired_analysis(a, b, rng=statistics_rng(37))
    with pytest.raises(ValueError):
        tost_equivalence(result, epsilon=0.0, alpha=0.05)


def test_no_pseudo_replication_is_structural():
    """
    The statistical unit must be the replicate, not individual examples.
    This test documents the structural guarantee: descriptive_statistics
    and paired_analysis both operate on 1-D arrays whose length is the
    number of replicates supplied by the caller - there is no code path
    in this module that accepts per-example predictions or expands an
    array using anything other than what the caller explicitly passed
    as replicate-level values.
    """
    import inspect
    from framework import statistics as stats_module

    src = inspect.getsource(stats_module)
    # Guards against accidentally introducing per-example expansion logic.
    assert "np.repeat" not in src
    assert "flatten()" not in src
