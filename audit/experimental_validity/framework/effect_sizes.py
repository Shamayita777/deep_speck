"""
Effect-size calculations for Experimental Validity.

Single authoritative implementation - other modules import from here
rather than re-deriving effect sizes locally.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def cohens_dz(diffs: np.ndarray) -> tuple[float, Optional[str]]:
    """
    Cohen's d_z for paired differences: mean(diff) / sd(diff).

    Returns (value, warning). value is NaN with a warning if the
    differences have zero variance (d_z undefined in that case).
    """
    diffs = np.asarray(diffs, dtype=float)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan"), "Zero or undefined variance in paired differences; Cohen's d_z is undefined."
    return float(np.mean(diffs)) / sd, None


def relative_to_baseline_advantage(effect: float, baseline_accuracy: float) -> Optional[float]:
    """
    Express an effect (accuracy difference) as a fraction of the total
    observed baseline distinguishing advantage (2*accuracy - 1).

    This is a *derived descriptive scale*, not a practical-significance
    threshold: it lets a reader see "this effect is X% of the total
    signal", without asserting that any particular X% is scientifically
    "small enough". It must never be used, by itself, to issue a
    practical-equivalence or robustness decision - see
    framework.certificate for why that requires a predeclared epsilon.

    Returns None if the baseline advantage is zero or undefined
    (cannot express a meaningful ratio).
    """
    advantage = 2.0 * baseline_accuracy - 1.0
    if advantage == 0.0 or not np.isfinite(advantage):
        return None
    return float(effect) / advantage
