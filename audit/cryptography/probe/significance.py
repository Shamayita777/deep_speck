"""
Shared Paired Significance Testing
====================================

Statistical core shared by CE3 (probe.evaluation) and CE4
(probe.intervention). Both stages reduce to the same question:
is a set of paired differences significantly greater than zero?
CE3's pairing unit is per-fold selectivity; CE4's is per-sample
necessity gap. The significance procedure itself does not
depend on which.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from audit.cryptography.test.ce3.types import PairedComparisonStatistic


def paired_significance(
    diffs: np.ndarray,
    *,
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedComparisonStatistic:
    """
    Wilcoxon signed-rank test, Cohen's d_z, and a percentile
    bootstrap CI over an array of paired differences.
    """

    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.shape[0]

    if n < 2:
        raise ValueError(
            "At least 2 paired observations are required to "
            "compute statistical significance."
        )

    if np.all(diffs == diffs[0]):
        raise ValueError(
            "Wilcoxon signed-rank test is undefined: all paired "
            "differences are identical (zero variance)."
        )

    statistic, p_value = wilcoxon(diffs, alternative=alternative)

    std = np.std(diffs, ddof=1)
    effect_size = float(np.mean(diffs) / std) if std > 0 else float("nan")

    rng = np.random.default_rng(seed)
    resamples = rng.choice(diffs, size=(n_bootstrap, n), replace=True)
    bootstrap_means = resamples.mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    return PairedComparisonStatistic(
        test_name="wilcoxon_signed_rank",
        alternative=alternative,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=effect_size,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_pairs=int(n),
    )