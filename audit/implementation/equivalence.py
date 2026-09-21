"""
Generic equivalence primitives for Implementation Integrity (II-3
Controlled Verification, II-4 Comparative Evaluation).

Per methodology Section 3.5: integer/bit representations are compared
with EXACT equality; floating-point representations are compared with
PREREGISTERED tolerances. Bitwise identity across different hardware or
frameworks is NOT required unless the scientific question demands it.

No knowledge of Gohr, Speck, ciphertexts, or neural networks lives
here - these functions operate on arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ExactEquivalenceResult:
    equivalent: bool
    n_compared: int
    n_mismatched: int
    first_mismatch_index: Optional[int]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent, "n_compared": self.n_compared,
            "n_mismatched": self.n_mismatched, "first_mismatch_index": self.first_mismatch_index,
            "comparison": "exact_equality", "detail": self.detail,
        }


@dataclass(frozen=True)
class NumericEquivalenceResult:
    within_tolerance: bool
    max_absolute_difference: float
    mean_absolute_difference: float
    max_relative_difference: Optional[float]
    correlation: Optional[float]
    agreement_rate: Optional[float]
    n_compared: int
    absolute_tolerance: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "within_tolerance": self.within_tolerance,
            "max_absolute_difference": self.max_absolute_difference,
            "mean_absolute_difference": self.mean_absolute_difference,
            "max_relative_difference": self.max_relative_difference,
            "correlation": self.correlation,
            "agreement_rate": self.agreement_rate,
            "n_compared": self.n_compared,
            "absolute_tolerance": self.absolute_tolerance,
            "comparison": "preregistered_tolerance",
            "detail": self.detail,
        }


def check_exact_equivalence(a: np.ndarray, b: np.ndarray, *, label: str = "arrays") -> ExactEquivalenceResult:
    """
    Exact element-wise equality, for integer/bit representations where
    any difference is a genuine discrepancy (methodology 3.5: byte
    order, bit order, tensor layout, serialization).
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return ExactEquivalenceResult(
            equivalent=False, n_compared=0, n_mismatched=-1, first_mismatch_index=None,
            detail=f"{label}: shape mismatch {a.shape} vs {b.shape} - not comparable element-wise.",
        )
    mismatches = a != b
    n_mismatched = int(np.count_nonzero(mismatches))
    first_idx = int(np.argmax(mismatches)) if n_mismatched else None
    return ExactEquivalenceResult(
        equivalent=(n_mismatched == 0), n_compared=int(a.size), n_mismatched=n_mismatched,
        first_mismatch_index=first_idx,
        detail=(f"{label}: exact match across {a.size} elements."
                if n_mismatched == 0 else
                f"{label}: {n_mismatched}/{a.size} elements differ (first at flat index {first_idx})."),
    )


def check_numeric_equivalence(
    a: np.ndarray, b: np.ndarray, *,
    absolute_tolerance: float,
    decision_threshold: Optional[float] = None,
    label: str = "outputs",
) -> NumericEquivalenceResult:
    """
    Tolerance-based comparison for floating-point outputs (II-4 forward
    equivalence). `absolute_tolerance` must be PREREGISTERED by the
    caller - this function never picks one.

    If `decision_threshold` is supplied (e.g. 0.5 for a binary
    classifier), an agreement_rate is also computed: the fraction of
    samples on which both implementations produce the same DECISION,
    which can matter scientifically even when raw outputs differ
    slightly.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return NumericEquivalenceResult(
            within_tolerance=False, max_absolute_difference=float("nan"),
            mean_absolute_difference=float("nan"), max_relative_difference=None,
            correlation=None, agreement_rate=None, n_compared=0,
            absolute_tolerance=absolute_tolerance,
            detail=f"{label}: shape mismatch {a.shape} vs {b.shape} - not comparable.",
        )

    diff = np.abs(a - b)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0

    denom = np.maximum(np.abs(a), np.abs(b))
    nonzero = denom > 0
    max_rel = float(np.max(diff[nonzero] / denom[nonzero])) if np.any(nonzero) else None

    correlation: Optional[float] = None
    if a.size > 1 and np.std(a) > 0 and np.std(b) > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])

    agreement_rate: Optional[float] = None
    if decision_threshold is not None:
        agreement_rate = float(np.mean((a > decision_threshold) == (b > decision_threshold)))

    within = bool(max_abs <= absolute_tolerance)
    return NumericEquivalenceResult(
        within_tolerance=within, max_absolute_difference=max_abs,
        mean_absolute_difference=mean_abs, max_relative_difference=max_rel,
        correlation=correlation, agreement_rate=agreement_rate, n_compared=int(a.size),
        absolute_tolerance=absolute_tolerance,
        detail=(f"{label}: max|diff|={max_abs:.3e} within preregistered tolerance "
                f"{absolute_tolerance:.3e}." if within else
                f"{label}: max|diff|={max_abs:.3e} EXCEEDS preregistered tolerance "
                f"{absolute_tolerance:.3e}."),
    )


@dataclass(frozen=True)
class ArchitectureComparison:
    """
    Comparison of a DECLARED architectural configuration against the
    configuration ACTUALLY realized in a produced artifact. Generic:
    "parameters" is any flat mapping of named configuration values.
    """
    declared: dict[str, Any]
    actual: dict[str, Any]
    mismatched_keys: list[str]
    equivalent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared, "actual": self.actual,
            "mismatched_keys": list(self.mismatched_keys), "equivalent": self.equivalent,
        }


def compare_declared_vs_actual(declared: dict[str, Any], actual: dict[str, Any]) -> ArchitectureComparison:
    """
    Compare a declared configuration against what was actually realized.
    Keys absent from `actual` are reported as mismatches (unverifiable
    is not the same as verified-equal, and is never treated as passing).
    """
    mismatched = [
        key for key, declared_value in declared.items()
        if key not in actual or actual[key] != declared_value
    ]
    return ArchitectureComparison(
        declared=dict(declared), actual=dict(actual),
        mismatched_keys=mismatched, equivalent=(len(mismatched) == 0),
    )
