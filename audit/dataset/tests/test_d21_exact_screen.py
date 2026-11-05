"""Regression tests for D2.1 near-duplicate screening (schema 7.0).

Two jobs.

1. RETAINED DEFECT REPRODUCTIONS.  The schema <= 6.5 implementation
   ``near_neighbor_summary`` had four defects (``D21_SUPERSEDED_METHOD_DEFECTS``).
   The minimal reproductions are kept here so the defects cannot return
   unnoticed and so the containment markers stay in place.

2. CORRECTNESS OF THE REPLACEMENT.  ``near_neighbor_exact_screen`` is checked
   against O(n^2) brute force, against closed-form combinatorics, and against
   direct enumeration, and its gates are checked for reachability in both
   directions.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from audit.dataset.d2_sample_dependence import (
    D21_SUPERSEDED_METHOD_DEFECTS,
    _clopper_pearson_lower,
    _empirical_pair_distance_probability,
    declared_pair_probability,
    exact_duplicate_pair_count,
    exact_min_distances,
    minimum_detectable_near_duplicate_ratio,
    near_neighbor_exact_screen,
    near_neighbor_summary,
    reevaluate_legacy_d21_record,
    near_duplicate_regime,
    required_queries_for_power,
)

UNIFORM = {"kind": "uniform"}


def _rows(n, w, seed):
    return np.random.default_rng(seed).integers(0, 2, size=(n, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Retained reproductions of the superseded method's defects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [1, 2, 4, 8])
def test_legacy_null_estimator_underflows_to_exactly_zero(radius):
    """Defect B, the dominant one.

    The reference probability was a Monte Carlo estimate of a quantity between
    1e-18 and 1e-10 drawn with 1e5 samples, whose smallest representable
    non-zero value is 1e-05.  It therefore evaluated to exactly 0.0 and the
    divisor was always zero.
    """
    x = _rows(20_000, 64, 11)
    q = _empirical_pair_distance_probability(
        x, radius, pairs=100_000, rng=np.random.default_rng(3))
    assert q == 0.0
    # The quantity it was trying to estimate is non-zero but unreachable.
    true_p = declared_pair_probability(64, radius)
    assert 0.0 < true_p < 1e-5


def test_legacy_zero_hit_branch_reported_a_vacuous_result_as_a_pass():
    """Defect B's serialisation consequence, plus the containment marker.

    With the divisor zero and no hits, the legacy code hardcoded
    ``ratio = 1.0`` and ``excess_ratio_status = "FINITE"`` -- indistinguishable
    in a certificate from a genuine comparison against a resolved
    expectation.  The containment fields added at schema 7.0 must be present
    so such a record can never again be read as a valid D2.1 result.
    """
    x = _rows(20_000, 64, 12)
    out = near_neighbor_summary(
        x, (1, 2, 4), rng=np.random.default_rng(5),
        query_count=256, null_pairs=100_000)
    for radius in ("1", "2", "4"):
        rec = out[radius]
        assert rec["null_pair_probability"] == 0.0
        assert rec["expected_query_hits"] == 0.0
        if rec["rows_with_verified_neighbor_within_radius"] == 0:
            assert rec["excess_ratio"] == 1.0
            assert rec["excess_ratio_status"] == "FINITE"
        assert rec["method_status"] == "SUPERSEDED_METHOD"
        detail = rec["method_status_detail"]
        # The detail must name the defect, refuse inference in BOTH
        # directions, and point at the replacement.
        assert "do not support an inference in either direction" in detail
        assert "near_neighbor_exact_screen" in detail
        assert "0.0" in detail


def test_legacy_statistical_gate_was_unreachable():
    """Defect C: the gate was the literal False, so D2.1 could never FAIL."""
    x = _rows(20_000, 64, 13)
    out = near_neighbor_summary(
        x, (1, 8), rng=np.random.default_rng(6),
        query_count=256, null_pairs=100_000)
    assert all(not rec["statistically_excessive"] for rec in out.values())


def test_superseded_defects_are_enumerated():
    joined = " ".join(D21_SUPERSEDED_METHOD_DEFECTS)
    assert len(D21_SUPERSEDED_METHOD_DEFECTS) >= 4
    for marker in ("Monte Carlo", "statistically_excessive", "under audit",
                   "retrieval recall"):
        assert marker in joined


# ---------------------------------------------------------------------------
# 2. The replacement: exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "w,n,chunk,block",
    [(64, 300, 97, 7), (64, 300, 1_000_000, 64), (37, 220, 64, 3),
     (128, 180, 53, 16), (64, 300, 1, 1)],
)
def test_exact_min_distances_matches_brute_force(w, n, chunk, block):
    x = _rows(n, w, w * n)
    x[5] = x[4]                              # an exact duplicate
    x[9] = x[8].copy(); x[9, 0] ^= 1         # a distance-1 pair
    full = np.array([[int(np.count_nonzero(x[i] != x[j])) for j in range(n)]
                     for i in range(n)])
    np.fill_diagonal(full, 10 ** 6)
    want = full.min(axis=1)
    got = exact_min_distances(x, np.arange(n), chunk_rows=chunk, query_block=block)
    assert np.array_equal(want, got)


def test_exact_scan_is_invariant_to_chunk_and_block():
    """Chunking and query blocking are performance choices only."""
    x = _rows(2_000, 64, 21)
    q = np.random.default_rng(22).choice(2_000, 200, replace=False)
    ref = exact_min_distances(x, q, chunk_rows=64, query_block=1)
    for chunk, block in ((7, 3), (128, 16), (4096, 64), (1_000_000, 1024)):
        assert np.array_equal(
            ref, exact_min_distances(x, q, chunk_rows=chunk, query_block=block))


@pytest.mark.parametrize("w", [8, 37, 64, 128])
@pytest.mark.parametrize("r", [0, 1, 2, 4])
def test_declared_pair_probability_is_exact_closed_form(w, r):
    want = sum(math.comb(w, k) for k in range(r + 1)) / float(2 ** w)
    assert declared_pair_probability(w, r) == want
    assert declared_pair_probability(w, r) > 0.0


@pytest.mark.parametrize("w,r", [(8, 1), (16, 3), (64, 0), (64, 3)])
def test_bernoulli_product_reduces_to_uniform_at_half(w, r):
    uniform = declared_pair_probability(w, r)
    bern = declared_pair_probability(
        w, r, bit_probabilities=[0.5] * w, query_row=np.zeros(w, dtype=np.uint8))
    assert bern == pytest.approx(uniform, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("r", [0, 1, 2, 5])
def test_bernoulli_product_matches_direct_enumeration(r):
    w = 12
    rng = np.random.default_rng(4)
    p = rng.uniform(0.2, 0.8, size=w)
    query = rng.integers(0, 2, size=w, dtype=np.uint8)
    total = 0.0
    for bits in itertools.product((0, 1), repeat=w):
        b = np.array(bits, dtype=np.uint8)
        if int(np.count_nonzero(b != query)) <= r:
            total += float(np.prod(np.where(b == 1, p, 1.0 - p)))
    got = declared_pair_probability(w, r, bit_probabilities=p, query_row=query)
    assert got == pytest.approx(total, rel=1e-12)


def test_exact_duplicate_pair_count_is_exact():
    x = _rows(500, 64, 5)
    x[1] = x[0]
    x[2] = x[0]
    x[7] = x[6]
    out = exact_duplicate_pair_count(x)
    assert out["exact_duplicate_pairs"] == 4      # C(3,2) + C(2,2)
    assert out["largest_multiplicity"] == 3
    assert out["distinct_rows"] == 497


# ---------------------------------------------------------------------------
# 3. The replacement: declared null, live gates, honest power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [{}, {"kind": "empirical"}, {"kind": "resampled"}])
def test_null_model_must_be_declared(bad):
    x = _rows(500, 64, 31)
    with pytest.raises(ValueError, match="declared"):
        near_neighbor_exact_screen(
            x, (1,), rng=np.random.default_rng(0), null_model=bad)


def test_bernoulli_product_null_requires_bit_probabilities():
    x = _rows(500, 64, 32)
    with pytest.raises(ValueError, match="bit_probabilities"):
        near_neighbor_exact_screen(
            x, (1,), rng=np.random.default_rng(0),
            null_model={"kind": "bernoulli_product"})


@pytest.mark.parametrize("radius", [1, 2, 4, 8])
def test_divisor_is_positive_and_ratio_is_finite(radius):
    """The direct regression against defect B: no zero divisor, ever."""
    x = _rows(20_000, 64, 33)
    out = near_neighbor_exact_screen(
        x, (radius,), rng=np.random.default_rng(34),
        null_model=UNIFORM, query_count=256)
    rec = out["by_radius"][str(radius)]
    assert rec["declared_query_probability"] > 0.0
    assert math.isfinite(rec["excess_ratio"])
    assert rec["excess_ratio_status"] == "FINITE"
    assert rec["exhaustive"] is True
    assert "resolution_status" not in rec


def test_clean_data_is_not_flagged():
    x = _rows(20_000, 64, 35)
    out = near_neighbor_exact_screen(
        x, (1, 2, 4, 8), rng=np.random.default_rng(36),
        null_model=UNIFORM, query_count=512)
    for rec in out["by_radius"].values():
        assert rec["decision"] != "EXCESS_DETECTED"
        assert rec["rows_with_neighbor_within_radius"] == 0


def test_planted_near_duplicates_are_detected():
    """The statistical gate must be reachable -- the fix for defect C."""
    n, w, m, radius, planted = 20_000, 64, 2_000, 8, 20
    x = _rows(n, w, 37)
    qidx = np.random.default_rng(38).choice(n, m, replace=False)
    partners = np.random.default_rng(39).choice(
        np.setdiff1d(np.arange(n), qidx), planted, replace=False)
    for target, partner in zip(qidx[:planted], partners):
        x[partner] = x[target]
        x[partner, np.random.default_rng(int(partner)).choice(w, radius, replace=False)] ^= 1

    out = near_neighbor_exact_screen(
        x, (radius,), rng=np.random.default_rng(38),
        null_model=UNIFORM, query_count=m)
    rec = out["by_radius"][str(radius)]
    assert rec["rows_with_neighbor_within_radius"] >= planted
    assert rec["statistically_excessive"] is True
    assert rec["practically_excessive"] is True
    assert rec["decision"] == "EXCESS_DETECTED"


def test_exact_duplicates_are_counted_exhaustively_by_the_screen():
    x = _rows(5_000, 64, 41)
    x[3] = x[2]
    out = near_neighbor_exact_screen(
        x, (1,), rng=np.random.default_rng(42), null_model=UNIFORM,
        query_count=100)
    assert out["exact_duplicates"]["exact_duplicate_pairs"] == 1


def test_gate_is_equivalent_to_the_exact_binomial_tail_test():
    """``statistically_excessive`` must agree with the binomial p-value."""
    x = _rows(20_000, 64, 43)
    x[7] = x[6]
    out = near_neighbor_exact_screen(
        x, (1, 4, 8), rng=np.random.default_rng(44),
        null_model=UNIFORM, query_count=1_000)
    for rec in out["by_radius"].values():
        expected = rec["exact_binomial_upper_tail_p_value"] < rec["per_radius_alpha"]
        assert rec["statistically_excessive"] is expected


# ---------------------------------------------------------------------------
# 4. Per-radius budgets and the non-monotone sensitivity surface
# ---------------------------------------------------------------------------


def test_per_radius_budgets_use_prefixes_of_one_scan():
    x = _rows(5_000, 64, 51)
    budgets = {1: 200, 4: 500, 8: 1_000}
    out = near_neighbor_exact_screen(
        x, tuple(budgets), rng=np.random.default_rng(52),
        null_model=UNIFORM, query_count=budgets)
    assert out["queries_scanned"] == max(budgets.values())
    for r, m in budgets.items():
        assert out["by_radius"][str(r)]["queried_rows"] == m


def test_per_radius_budget_mapping_must_cover_every_radius():
    x = _rows(1_000, 64, 53)
    with pytest.raises(ValueError, match="lacks budgets"):
        near_neighbor_exact_screen(
            x, (1, 4), rng=np.random.default_rng(0),
            null_model=UNIFORM, query_count={1: 100})


def test_required_queries_measures_power_not_mere_attainability():
    """Guards a real defect in an earlier version of this preflight.

    That version asked whether the gate would reject IF the observed count
    equalled the alternative's expected count.  Where that expected count is
    far below one the question is vacuous: at n = 1e7, r = 4, m = 6,000 it
    reported a 1.25x target "attainable" while the true power was 0.276 per
    cent.  The replacement must report real binomial power.
    """
    from scipy.stats import binom

    n, alpha = 10_000_000, 0.0025
    q4 = -math.expm1((n - 1) * math.log1p(-declared_pair_probability(64, 4)))

    # The old criterion is satisfied at m=6,000 ...
    assert _clopper_pearson_lower(1, 6_000, alpha) > q4
    # ... while the actual power there is negligible.
    assert binom.sf(0, 6_000, 1.25 * q4) < 0.01

    # The replacement must therefore NOT return a small budget here.
    out = required_queries_for_power(
        1.25, null_query_probability=q4, alpha=alpha, power=0.80)
    assert out["required_queries"] is None or out["required_queries"] > 1_000_000
    if out["required_queries"] is None:
        assert "DETECTION" in out["note"]


def test_required_queries_for_power_is_verified_against_binomial_power():
    from scipy.stats import binom

    n, alpha, target, power = 10_000_000, 0.0025, 1.25, 0.80
    q8 = -math.expm1((n - 1) * math.log1p(-declared_pair_probability(64, 8)))
    out = required_queries_for_power(
        target, null_query_probability=q8, alpha=alpha, power=power)
    m = out["required_queries"]
    assert m is not None

    def achieved(mm):
        k = next((k for k in range(1, 10_000)
                  if binom.sf(k - 1, mm, q8) < alpha), None)
        return 0.0 if k is None else float(binom.sf(k - 1, mm, target * q8))

    assert achieved(m) >= power            # the returned budget suffices
    assert achieved(m - 1) < power         # and it is the smallest such


def test_detection_and_estimation_regimes_are_separated():
    n = 10_000_000
    q4 = -math.expm1((n - 1) * math.log1p(-declared_pair_probability(64, 4)))
    q8 = -math.expm1((n - 1) * math.log1p(-declared_pair_probability(64, 8)))
    assert near_duplicate_regime(queries=20_480, null_query_probability=q4) == "DETECTION"
    assert near_duplicate_regime(queries=20_480, null_query_probability=q8) == "ESTIMATION"


def test_detection_regime_marks_the_ratio_as_an_unstable_summary():
    x = _rows(20_000, 64, 63)
    out = near_neighbor_exact_screen(
        x, (1, 4), rng=np.random.default_rng(64), null_model=UNIFORM,
        query_count=256)
    for rec in out["by_radius"].values():
        assert rec["regime"] == "DETECTION"
        assert "UNSTABLE SUMMARY" in rec["excess_ratio_interpretation"]
        assert rec["power_status"] == "DETECTION_SINGLE_OCCURRENCE"


def test_power_block_reports_single_occurrence_sensitivity():
    x = _rows(20_000, 64, 61)
    out = near_neighbor_exact_screen(
        x, (1,), rng=np.random.default_rng(62), null_model=UNIFORM,
        query_count=256)
    rec = out["by_radius"]["1"]
    assert rec["power"]["minimum_rejecting_hits"] == 1
    assert rec["power_status"] == "DETECTION_SINGLE_OCCURRENCE"
    assert rec["regime"] == "DETECTION"


# ---------------------------------------------------------------------------
# 5. Re-evaluation of existing certificates
# ---------------------------------------------------------------------------


def test_reevaluation_never_reports_a_pass():
    """A legacy count is a LOWER bound, so absence can never be concluded."""
    record = {"queried_rows": 20_480,
              "rows_with_verified_neighbor_within_radius": 0,
              "excess_ratio": 1.0, "excess_ratio_status": "FINITE",
              "null_pair_probability": 0.0}
    out = reevaluate_legacy_d21_record(
        record, rows=10_000_000, bits=64, radius=8,
        alpha=0.01, radius_family_size=4)
    assert out["status"] == "INCONCLUSIVE_LOWER_BOUND"
    assert out["one_sided"] is True
    assert "not evidence of absence" in out["interpretation"].lower()


def test_reevaluation_can_confirm_an_excess_one_sidedly():
    record = {"queried_rows": 20_480,
              "rows_with_verified_neighbor_within_radius": 5_000}
    out = reevaluate_legacy_d21_record(
        record, rows=10_000_000, bits=64, radius=8,
        alpha=0.01, radius_family_size=4)
    assert out["status"] == "EXCESS_DETECTED"
    assert out["recomputed"]["excess_ratio_lower_bound"] > 1.25


def test_reevaluation_marks_incomplete_records_unverified():
    out = reevaluate_legacy_d21_record(
        {"queried_rows": 100}, rows=1_000, bits=64, radius=4)
    assert out["status"] == "UNVERIFIED"
    assert out["recomputed"] is None


def test_reevaluation_rejects_implausible_counts():
    out = reevaluate_legacy_d21_record(
        {"queried_rows": 100, "rows_with_verified_neighbor_within_radius": 101},
        rows=1_000, bits=64, radius=4)
    assert out["status"] == "UNVERIFIED"
