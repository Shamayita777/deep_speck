import numpy as np

from framework.confidence_intervals import percentile_bootstrap_ci
from framework.effect_sizes import cohens_dz, relative_to_baseline_advantage
from framework.seeds import statistics_rng


def test_cohens_dz_known_value():
    diffs = np.array([1.0, 1.0, 1.0, 1.0])  # zero variance
    value, warning = cohens_dz(diffs)
    assert np.isnan(value)
    assert warning is not None


def test_cohens_dz_normal_case():
    diffs = np.array([1.0, 2.0, 3.0, 4.0])
    value, warning = cohens_dz(diffs)
    expected = np.mean(diffs) / np.std(diffs, ddof=1)
    assert warning is None
    assert abs(value - expected) < 1e-9


def test_relative_to_baseline_advantage():
    # baseline accuracy 0.9291 -> advantage = 0.8582
    ratio = relative_to_baseline_advantage(0.05, 0.9291)
    assert ratio is not None
    assert 0 < ratio < 1


def test_relative_to_baseline_advantage_zero_advantage():
    assert relative_to_baseline_advantage(0.05, 0.5) is None


def test_bootstrap_ci_contains_true_mean_reasonably_often():
    rng = statistics_rng(10)
    values = rng.normal(0.9, 0.02, size=50)
    low, high = percentile_bootstrap_ci(values, rng=statistics_rng(11), n_boot=2000)
    assert low < np.mean(values) < high
