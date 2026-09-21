"""
Confidence-interval calculations for Experimental Validity.

Single authoritative implementation - other modules import from here.
Percentile bootstrap is used throughout rather than a normal-theory CI,
because replicate-level accuracy is bounded in [0, 1] and, for small
replicate counts, poorly approximated by a normal distribution
(this is a documented methodological choice, not a methodology
requirement - see docs/statistical_plan.md).
"""

from __future__ import annotations

import numpy as np


def percentile_bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    statistic=np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Generic percentile bootstrap CI for an arbitrary statistic computed
    over `values`, resampling the replicate index with replacement.

    `statistic` must be a function array -> float (e.g. np.mean,
    np.std). Requires len(values) >= 2.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("Bootstrap CI requires at least 2 replicates.")
    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)
    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper
