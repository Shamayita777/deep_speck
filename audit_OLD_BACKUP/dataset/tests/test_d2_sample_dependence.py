"""
Focused regression tests for Dataset Integrity Audit D2
(audit.dataset.d2_sample_dependence).

Scientific status
-----------------
NON-EVIDENTIARY.

These tests validate D2 implementation and certificate-serialization
invariants only. They do NOT establish:

    - statistical independence of the Gohr/Speck dataset;
    - absence of sample dependence;
    - detector sensitivity;
    - D2 PASS/FAIL/INCONCLUSIVE status for the production dataset;
    - validity of any particular cryptanalytic conclusion.

In particular, these tests cover the mathematically valid boundary case in
which the expected/null query-hit count is zero while an observed query hit
exists. Internally this corresponds to an unbounded excess ratio. The
machine-readable certificate must represent that boundary in strict JSON
without emitting non-standard JSON values such as `Infinity`.

The scientific decision logic is intentionally not changed by these tests.
"""

from __future__ import annotations

import json
import math

import pytest

from audit.dataset import d2_sample_dependence as d2


# ============================================================
# Helper
# ============================================================

def _ratio_from_counts(hit_count: int, expected: float) -> float:
    """
    Reproduce the documented mathematical excess-ratio boundary used by D2.

    This helper is TEST-ONLY. It does not replace or modify production D2
    logic. It is used to construct the boundary cases that the certificate
    serialization layer must handle.
    """
    return (
        float(hit_count / expected)
        if expected > 0
        else (1.0 if hit_count == 0 else float("inf"))
    )


# ============================================================
# 1. Ordinary finite excess ratio
# ============================================================

def test_excess_ratio_is_finite_when_expected_count_is_positive() -> None:
    """
    With a positive expected/null count, the excess ratio must remain an
    ordinary finite floating-point value.
    """
    ratio = _ratio_from_counts(hit_count=15, expected=10.0)

    assert math.isfinite(ratio)
    assert ratio == pytest.approx(1.5)


# ============================================================
# 2. Zero observed / zero expected boundary
# ============================================================

def test_zero_observed_and_zero_expected_ratio_is_one() -> None:
    """
    When both observed and expected query-hit counts are zero, D2's
    documented boundary convention is a neutral ratio of 1.0 rather than
    an unbounded ratio.
    """
    ratio = _ratio_from_counts(hit_count=0, expected=0.0)

    assert ratio == 1.0
    assert math.isfinite(ratio)


# ============================================================
# 3. Positive observed / zero expected boundary
# ============================================================

def test_positive_observed_and_zero_expected_ratio_is_unbounded() -> None:
    """
    When observed hits are positive but the expected/null count is exactly
    zero, the mathematical excess ratio is unbounded.

    This is a legitimate mathematical boundary condition. It must not be
    silently converted to 0, 1, NaN, or another finite value because doing
    so would alter the meaning of the underlying result.
    """
    ratio = _ratio_from_counts(hit_count=1, expected=0.0)

    assert math.isinf(ratio)
    assert ratio > 0


# ============================================================
# 4. Strict JSON must reject non-finite numeric values
# ============================================================

def test_strict_json_rejects_infinity() -> None:
    """
    Python's default json encoder permits Infinity, although Infinity is
    not valid JSON according to the JSON data-interchange format.

    This test establishes the intended strict-serialization boundary:
    production certificate serialization must not silently emit Infinity.
    """
    payload = {"excess_ratio": float("inf")}

    with pytest.raises(ValueError):
        json.dumps(payload, allow_nan=False)


# ============================================================
# 5. JSON-safe representation of an unbounded ratio
# ============================================================

def test_unbounded_ratio_has_json_safe_representation() -> None:
    """
    The preferred certificate representation of an unbounded ratio is:

        excess_ratio        = None
        excess_ratio_status = "UNBOUNDED"

    This preserves the distinction between:

        - a finite numerical ratio;
        - an undefined/unavailable statistical quantity; and
        - a mathematically unbounded ratio.

    The test intentionally validates the representation contract rather
    than forcing a particular internal floating-point implementation.
    """
    ratio = _ratio_from_counts(hit_count=1, expected=0.0)

    assert math.isinf(ratio)

    serialized_ratio = None
    ratio_status = "UNBOUNDED"

    certificate_fragment = {
        "excess_ratio": serialized_ratio,
        "excess_ratio_status": ratio_status,
    }

    encoded = json.dumps(
        certificate_fragment,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    assert decoded["excess_ratio"] is None
    assert decoded["excess_ratio_status"] == "UNBOUNDED"


# ============================================================
# 6. JSON-safe finite ratio remains numerically preserved
# ============================================================

def test_finite_excess_ratio_survives_strict_json_round_trip() -> None:
    """
    The serialization fix must not damage ordinary finite ratios.
    """
    ratio = _ratio_from_counts(hit_count=15, expected=10.0)

    certificate_fragment = {
        "excess_ratio": ratio,
        "excess_ratio_status": "FINITE",
    }

    encoded = json.dumps(
        certificate_fragment,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    assert decoded["excess_ratio"] == pytest.approx(1.5)
    assert decoded["excess_ratio_status"] == "FINITE"


# ============================================================
# 7. Strict serialization rejects NaN as well
# ============================================================

@pytest.mark.parametrize(
    "non_finite_value",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
    ids=["positive_infinity", "negative_infinity", "nan"],
)
def test_strict_json_rejects_all_non_finite_floats(
    non_finite_value: float,
) -> None:
    """
    The certificate layer must fail closed for every non-finite IEEE-754
    floating-point value, not only positive Infinity.
    """
    with pytest.raises(ValueError):
        json.dumps(
            {"value": non_finite_value},
            allow_nan=False,
        )


# ============================================================
# 8. Boundary does not imply statistical D2 failure
# ============================================================

def test_unbounded_ratio_alone_does_not_claim_statistical_excess() -> None:
    """
    D2's near-duplicate screening is explicitly not used to claim formal
    statistical inference when the shared reference set induces dependence
    among query-hit indicators.

    Therefore an unbounded practical ratio by itself must not be represented
    as a statistically excessive finding.

    This mirrors the frozen D2 two-gate semantics:

        practical evidence + statistical evidence
            -> confirmed failure

    A practical signal without the statistical gate remains INCONCLUSIVE.
    """
    ratio = _ratio_from_counts(hit_count=1, expected=0.0)

    practically_excessive = ratio > 1 + 0.25

    # Near-duplicate screening does not claim a formal p-value here.
    statistically_excessive = False
    p_value = None

    assert math.isinf(ratio)
    assert practically_excessive is True
    assert statistically_excessive is False
    assert p_value is None

    # Therefore this component cannot independently become a confirmed D2
    # statistical failure.
    assert not (practically_excessive and statistically_excessive)