"""
Multiplicity correction for Experimental Validity.

The frozen primary EV hypothesis family is:
    H-EV-SHUFFLE
    H-EV-REPRESENTATION
corrected via Holm step-down at family alpha = 0.05.

Architecture or other exploratory experiments must never be silently
folded into this family - each family is identified explicitly by
`family_id` and correction is only ever applied within one declared
family at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MultiplicityResult:
    family_id: str
    correction_method: str
    alpha: float
    hypothesis_ids: list[str]
    raw_p_values: list[Optional[float]]
    adjusted_p_values: list[Optional[float]]
    reject_null: list[Optional[bool]]

    def for_hypothesis(self, hypothesis_id: str) -> dict:
        idx = self.hypothesis_ids.index(hypothesis_id)
        return {
            "family_id": self.family_id,
            "correction_method": self.correction_method,
            "alpha": self.alpha,
            "raw_p_value": self.raw_p_values[idx],
            "adjusted_p_value": self.adjusted_p_values[idx],
            "reject_null": self.reject_null[idx],
        }


def holm_correction(
    *,
    family_id: str,
    hypothesis_ids: list[str],
    p_values: list[Optional[float]],
    alpha: float = 0.05,
) -> MultiplicityResult:
    """
    Holm-Bonferroni step-down correction.

    p_values may contain None entries (e.g. a degenerate or unavailable
    test for one hypothesis in the family); these are excluded from the
    step-down ordering but preserved in the output, with reject_null=None
    (undetermined) rather than silently treated as non-significant.
    """
    if len(hypothesis_ids) != len(p_values):
        raise ValueError("hypothesis_ids and p_values must have equal length.")

    m = len(p_values)
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda t: t[1])

    adjusted: list[Optional[float]] = [None] * m
    reject: list[Optional[bool]] = [None] * m

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        k = rank + 1  # 1-indexed rank among tested hypotheses
        n_tested = len(indexed)
        holm_adjusted = min(1.0, (n_tested - k + 1) * p)
        running_max = max(running_max, holm_adjusted)
        adjusted[orig_idx] = running_max
        reject[orig_idx] = running_max < alpha

    return MultiplicityResult(
        family_id=family_id,
        correction_method="holm",
        alpha=alpha,
        hypothesis_ids=list(hypothesis_ids),
        raw_p_values=list(p_values),
        adjusted_p_values=adjusted,
        reject_null=reject,
    )
