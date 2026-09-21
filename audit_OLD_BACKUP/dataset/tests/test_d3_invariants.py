"""
D3 invariant / regression test suite
(audit.dataset.d3_distribution_statistics, with light wiring checks against
audit.dataset.experiments.d3_gohr_distribution_statistics).

Scientific status
------------------
This suite validates that the D3 measurement machinery itself behaves
correctly under controlled synthetic inputs. It answers five implementation
questions:

    1. Does D3 avoid false alarms on identical/matched distributions?
    2. Does D3 detect a sufficiently strong, predeclared distribution shift?
    3. Is D3 reproducible under identical inputs/configuration?
    4. Does D3 preserve mathematically justified invariance to feature
       ordering (and correctly relabel, rather than falsely claim
       invariance, where reordering changes identity)?
    5. Does D3 behave correctly on finite-sample, degenerate, boundary, and
       invalid inputs?

These are implementation/regression properties. They are NOT evidence that
the real Gohr dataset passes D3, and passing this suite must never be read
that way. The production D3 audit is a separate execution, on the real
generated dataset, by d3_gohr_distribution_statistics.py.

Scope discipline
-----------------
This suite validates D3's declared scope only: class distribution,
first-order binary feature marginals, global mean/variance and entropy as
descriptive diagnostics, second-order feature-pair joint distributions, and
partition-to-partition consistency -- using the existing exact-binomial /
2x4-chi-square / Bonferroni / total-variation-distance machinery exactly as
implemented. It does not test, and must never be read as testing: exact
duplicates or cross-partition overlap (D1), sample-order or pairwise
Hamming dependence (D2), or equality of the complete 64-dimensional joint
distribution (explicitly excluded by D3's own docstring).

Determinism and floating-point policy
--------------------------------------
D3's generic module has no randomness; every function is a pure function of
its inputs. Fixtures here are constructed directly (explicit arrays or
simple deterministic index formulas), never from numpy.random.

Where a test compares a reduction (e.g. a per-feature or per-pair value)
computed twice on the SAME underlying numbers, exact equality (`==`) is
used. Where a test compares an AGGREGATE that involves summing a REORDERED
list of already-computed floats (e.g. mean_one_probability after permuting
columns), a tight relative tolerance (1e-12) is used instead, because
floating-point summation is not strictly order-independent at the level of
individual rounding bits -- this is not a weakened assertion, it is the
mathematically correct way to compare such a value. Every such use is
called out at the point it is used.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d3_gohr_distribution_statistics as gohr_d3


TIGHT_REL_TOL = 1e-12


def _approx(a: float, b: float, rel: float = TIGHT_REL_TOL) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-15)


# ============================================================
# Shared synthetic fixtures.
# ============================================================

def _balanced_d4_dataset() -> np.ndarray:
    """8 rows x 4 features, each column summing to exactly 4 (p=0.5 exactly
    for every feature by construction, verifiable by counting):

        row0: 0 0 0 0        row4: 1 0 0 1
        row1: 0 0 1 1        row5: 1 0 1 0
        row2: 0 1 0 1        row6: 1 1 0 0
        row3: 0 1 1 0        row7: 1 1 1 1
    """
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def _balanced_d4_labels() -> np.ndarray:
    """4 negative, 4 positive: an exactly balanced label vector."""
    return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)


def _intended_d4() -> dict:
    return {
        "expected_class_ratio": 0.5,
        "expected_bit_probability": 0.5,
        "class_ratio_tolerance": 0.05,
        "bit_probability_tolerance": 0.05,
        "pairwise_tvd_tolerance": 0.2,
    }


def _permutation_test_dataset() -> np.ndarray:
    """D=6, n=20, built from explicit deterministic index formulas (no
    randomness). Feature 3 is deliberately built as a function of features
    0 and 1 so at least one genuinely correlated feature pair exists,
    giving the permutation-equivariance test something non-trivial to
    verify (not just an already-symmetric case where an indexing bug could
    accidentally cancel out).

        col0: alternating          (idx % 2)
        col1: every third row      (idx % 3 == 0)
        col2: first five rows      (idx < 5)
        col3: col0 AND col1        -- correlated with 0 and 1 by construction
        col4: last five rows       (idx >= 15)
        col5: pattern of period 4  (idx % 4 < 2)
    """
    idx = np.arange(20)
    x = np.zeros((20, 6), dtype=np.uint8)
    x[:, 0] = (idx % 2).astype(np.uint8)
    x[:, 1] = (idx % 3 == 0).astype(np.uint8)
    x[:, 2] = (idx < 5).astype(np.uint8)
    x[:, 3] = ((idx % 2 == 1) & (idx % 3 == 0)).astype(np.uint8)
    x[:, 4] = (idx >= 15).astype(np.uint8)
    x[:, 5] = ((idx % 4) < 2).astype(np.uint8)
    return x


# ============================================================
# Invariant A -- identical / matched distributions (no false alarm).
# ============================================================

def test_identical_partition_compared_to_itself_is_exactly_null() -> None:
    """Comparing a dataset against ITSELF (not merely a same-distribution
    resample) has an independently derivable exact-zero answer for every
    second-order pair: row totals are equal, so the chi-square expected
    counts equal the observed counts exactly, giving chi_square==0.0,
    p_value==1.0, tvd==0.0, delta_phi==0.0 for every single pair -- not
    approximately, exactly, by construction of the homogeneity test."""
    x = _balanced_d4_dataset()
    result = d3.compare_pairwise_joint_distributions(
        x, x, practical_tvd_tolerance=0.01, familywise_alpha=0.01,
    )
    for record in result["pair_summaries"]:
        assert record["chi_square"] == 0.0
        assert record["p_value"] == 1.0
        assert record["tvd"] == 0.0
        assert record["delta_phi"] == 0.0
        assert record["statistically_inconsistent"] is False
        assert record["practically_inconsistent"] is False
    assert result["decision"]["family_pass"] is True


def test_identical_distribution_has_no_false_alarm() -> None:
    """End-to-end: train/validation/test all built from the SAME exactly-
    balanced construction (not merely close) must produce PASS with zero
    confirmed, zero statistical-only, and zero practical-only findings."""
    x = _balanced_d4_dataset()
    y = _balanced_d4_labels()
    results = d3.audit_distribution(
        x, y, x, y, x, y,
        intended_distribution=_intended_d4(),
        familywise_alpha=0.05,
    )
    decision = d3.evaluate_decision(results)
    assert decision["outcome"] == "PASS"
    assert decision["counts"] == {
        "confirmed_failures": 0, "statistical_only": 0, "practical_only": 0,
    }


def test_same_distribution_different_sample_still_has_no_false_alarm() -> None:
    """train and validation are NOT the same array -- validation is train's
    row-reversal -- but share exactly the same per-feature marginals
    (guaranteed by construction: reversing row order cannot change a
    column's sum). This is deliberately distinct from the exact-identical-
    array case above: it tests that D3 does not manufacture a spurious
    difference from sample identity/order alone, while still allowing
    (not asserting) that per-pair statistics need not be bit-identical the
    way the self-comparison case is."""
    x = _balanced_d4_dataset()
    y = _balanced_d4_labels()
    x_reordered = x[::-1].copy()
    y_reordered = y[::-1].copy()

    results = d3.audit_distribution(
        x, y, x_reordered, y_reordered,
        intended_distribution=_intended_d4(),
        familywise_alpha=0.05,
    )
    decision = d3.evaluate_decision(results)
    assert decision["outcome"] == "PASS"
    assert decision["counts"]["confirmed_failures"] == 0

    # The per-feature marginals themselves are exactly equal by construction
    # (row order cannot change a column sum) -- this IS an exact invariant,
    # independent of D3's internals: it follows from how the fixture itself
    # was built.
    marg_a = results["partitions"]["train"]["binary_marginal_distribution"]["per_feature_one_probability"]
    marg_b = results["partitions"]["validation"]["binary_marginal_distribution"]["per_feature_one_probability"]
    assert marg_a == marg_b


# ============================================================
# Invariant B -- controlled, predeclared distribution shift.
# ============================================================
#
# Every expected value below is computed BEFORE inspecting any D3 output,
# from the fixture's own construction or by hand-derived arithmetic on the
# 2x2/2x4 contingency tables involved. No threshold, seed, or fixture was
# adjusted after seeing a result.

def test_strong_class_balance_shift_is_detected() -> None:
    """n=200: reference partition exactly balanced (100/200 = 0.5);
    shifted partition 170/200 = 0.85. Predeclared: absolute_difference =
    0.35, which must exceed a test tolerance of 0.05, and the exact
    binomial two-sided p-value at n=200 for 170 successes against p=0.5
    must be astronomically small (this is a textbook-extreme binomial
    tail; it is not being tuned to produce significance, it IS a huge
    reference-null-to-alternative gap chosen in advance for exactly that
    reason)."""
    shifted_labels = np.array([0] * 30 + [1] * 170, dtype=np.uint8)
    observed = d3.class_distribution(shifted_labels)
    assert observed["positive_ratio"] == pytest.approx(0.85)

    result = d3.class_balance_test(
        observed, expected_positive_ratio=0.5,
        practical_tolerance=0.05, alpha=0.01,
    )
    assert result["absolute_difference"] == pytest.approx(0.35)
    assert result["practically_consistent"] is False
    assert result["p_value"] < 1e-10
    assert result["statistically_inconsistent"] is True


def test_strong_single_feature_marginal_shift_is_detected_and_isolated() -> None:
    """n=200, D=4. Reference: every feature exactly p=0.5 (100/200 ones,
    contiguous block). Shifted: ONLY feature 0 moves to p=0.9 (180/200);
    features 1-3 and class balance are held at the reference level so the
    shift is mechanistically isolated to one feature. Predeclared: feature
    0's absolute_difference = 0.4, far exceeding a 0.05 test tolerance;
    features 1-3 must show NO deviation (difference == 0.0 exactly, since
    they are constructed identically to the reference)."""
    n = 200
    shifted = np.zeros((n, 4), dtype=np.uint8)
    shifted[:180, 0] = 1  # p = 0.9 -- the shift
    shifted[50:150, 1] = 1  # p = 0.5 -- unchanged
    shifted[25:125, 2] = 1  # p = 0.5 -- unchanged
    shifted[75:175, 3] = 1  # p = 0.5 -- unchanged
    labels = np.array([0] * 100 + [1] * 100, dtype=np.uint8)  # class balance unchanged

    partition = d3.summarize_partition(shifted, labels)
    comparison = d3.compare_with_intended_distribution(
        partition,
        expected_class_ratio=0.5, expected_bit_probability=0.5,
        class_ratio_tolerance=0.05, bit_probability_tolerance=0.05,
        familywise_alpha=0.01,
    )

    per_feature = comparison["binary_marginals"]["per_feature"]
    assert per_feature[0]["absolute_difference"] == pytest.approx(0.4)
    assert per_feature[0]["statistically_inconsistent"] is True
    assert per_feature[0]["p_value"] < 1e-10
    for idx in (1, 2, 3):
        assert per_feature[idx]["absolute_difference"] == 0.0
        assert per_feature[idx]["statistically_inconsistent"] is False

    assert comparison["binary_marginals"]["practically_consistent"] is False
    assert comparison["class_distribution"]["statistically_inconsistent"] is False


def test_strong_shift_propagates_to_fail_outcome_end_to_end() -> None:
    """Same isolated single-feature shift as above, run through the full
    audit_distribution -> evaluate_decision pipeline (train vs. validation)
    to validate the pipeline's wiring, not just the underlying function.
    Predeclared: outcome must be FAIL, and the failing key must name
    train's binary_marginals specifically (not e.g. class_distribution,
    which was deliberately left unshifted)."""
    n = 200
    reference = np.zeros((n, 4), dtype=np.uint8)
    reference[:100, 0] = 1
    reference[50:150, 1] = 1
    reference[25:125, 2] = 1
    reference[75:175, 3] = 1
    shifted = reference.copy()
    shifted[:, 0] = 0
    shifted[:180, 0] = 1  # feature 0: 0.5 -> 0.9
    labels = np.array([0] * 100 + [1] * 100, dtype=np.uint8)

    results = d3.audit_distribution(
        shifted, labels, reference, labels,
        intended_distribution={
            "expected_class_ratio": 0.5, "expected_bit_probability": 0.5,
            "class_ratio_tolerance": 0.05, "bit_probability_tolerance": 0.05,
            "pairwise_tvd_tolerance": 0.5,
        },
        familywise_alpha=0.01,
    )
    decision = d3.evaluate_decision(results)
    assert decision["outcome"] == "FAIL"
    assert "train:binary_marginals" in decision["confirmed_statistical_and_practical_failures"]
    assert "train:class_distribution" not in decision["confirmed_statistical_and_practical_failures"]
    assert "train:class_distribution" not in decision["statistical_only_sensitivities"]


def test_strong_second_order_shift_is_detected_while_untouched_pairs_stay_clean() -> None:
    """D=3, n=200, two partitions A (reference) and B (shifted).

    Construction (verified by hand below, not by running the code first):
    A: rows0-49=(0,0), rows50-99=(0,1), rows100-149=(1,0), rows150-199=(1,1)
       for features 0,1 -- an EXACTLY uniform 4-way split (50 each).
       Feature 2 alternates 0,1,0,1,... across all 200 rows; because each of
       the four 50-row blocks above starts at an even offset (0,50,100,150)
       with even length (50), feature 2 is split exactly 25/25 within every
       block, making pairs (0,2) and (1,2) ALSO exactly uniform (50/50/50/50).
       => every pair in A has table [50,50,50,50]: phi = (50*50-50*50)/denom
          = 0.0 exactly; TVD from uniform = 0.0 exactly.

    B: rows0-99=(0,0), rows100-199=(1,1) for features 0,1 -- feature0==
       feature1 always. Table for pair(0,1) = [100,0,0,100]:
          phi = (100*100 - 0*0) / sqrt(100*100*100*100) = 10000/10000 = 1.0
          TVD from uniform = 0.5*(|0.5-0.25|+|0-0.25|+|0-0.25|+|0.5-0.25|)
                            = 0.5*(0.25*4) = 0.5
       Feature 2 again alternates 0,1,... across all 200 rows; feature0 (and
       feature1, identical here) is constant within each contiguous 100-row
       half (even offset, even length), so pairs (0,2) and (1,2) are AGAIN
       exactly [50,50,50,50] in B: phi=0.0, TVD=0.0 -- UNCHANGED from A.

    Predeclared expectation: pair (0,1) shows delta_phi == 1.0 exactly and
    tvd == 0.5 exactly, with a large chi-square statistic and a vanishing
    p-value; pairs (0,2) and (1,2) show delta_phi == 0.0 and tvd == 0.0
    exactly, with chi_square == 0.0 and p_value == 1.0 (identical tables in
    both partitions -- the same exact-null property as the self-comparison
    test above).
    """
    n = 200
    a = np.zeros((n, 3), dtype=np.uint8)
    a[0:50, 0] = 0; a[0:50, 1] = 0
    a[50:100, 0] = 0; a[50:100, 1] = 1
    a[100:150, 0] = 1; a[100:150, 1] = 0
    a[150:200, 0] = 1; a[150:200, 1] = 1
    a[:, 2] = np.tile([0, 1], 100)

    b = np.zeros((n, 3), dtype=np.uint8)
    b[0:100, 0] = 0; b[0:100, 1] = 0
    b[100:200, 0] = 1; b[100:200, 1] = 1
    b[:, 2] = np.tile([0, 1], 100)

    result = d3.compare_pairwise_joint_distributions(
        a, b, practical_tvd_tolerance=0.1, familywise_alpha=0.01,
    )
    by_pair = {(r["feature_i"], r["feature_j"]): r for r in result["pair_summaries"]}

    shifted_pair = by_pair[(0, 1)]
    assert shifted_pair["delta_phi"] == pytest.approx(1.0)
    assert shifted_pair["tvd"] == pytest.approx(0.5)
    assert shifted_pair["chi_square"] > 100.0
    assert shifted_pair["p_value"] < 1e-10
    assert shifted_pair["statistically_inconsistent"] is True
    assert shifted_pair["practically_inconsistent"] is True

    for key in [(0, 2), (1, 2)]:
        clean_pair = by_pair[key]
        assert clean_pair["delta_phi"] == 0.0
        assert clean_pair["tvd"] == 0.0
        assert clean_pair["chi_square"] == 0.0
        assert clean_pair["p_value"] == 1.0
        assert clean_pair["statistically_inconsistent"] is False
        assert clean_pair["practically_inconsistent"] is False


def test_pairwise_joint_table_cells_match_hand_counted_ground_truth() -> None:
    """Directly verifies a01 and a10 are assigned to their correct semantic
    meaning -- count(feature_i=0,feature_j=1) and count(feature_i=1,
    feature_j=0) respectively -- against a small, hand-countable,
    deliberately ASYMMETRIC fixture (a01 != a10).

    This test exists because adversarial mutation testing (see suite
    report) showed that a bug swapping which formula computes a10 vs a01
    is NOT detectable through phi, TVD, or chi-square: all three are
    provably symmetric under that specific relabeling (phi's a*d-b*c is
    unchanged by b<->c; TVD-from-uniform and chi-square treat all four
    cells symmetrically). Ground-truth cell inspection is the only way to
    catch that specific class of regression, so it is tested directly here
    rather than only being inferred from downstream statistics.

    Fixture (5 rows, columns = feature_i, feature_j):
        [0,0] [0,1] [1,0] [1,0] [1,0]
    By direct count: (i=0,j=0)=1, (i=0,j=1)=1, (i=1,j=0)=3, (i=1,j=1)=0.
    """
    x = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 0], [1, 0]],
        dtype=np.uint8,
    )
    data = d3.pairwise_joint_tables(x)
    i, j, table = data["pairs"][0]
    assert (i, j) == (0, 1)
    a00, a01, a10, a11 = (int(v) for v in table)
    assert (a00, a01, a10, a11) == (1, 1, 3, 0)


def test_second_order_bonferroni_alpha_is_correctly_computed() -> None:
    """Directly verifies the second-order Bonferroni arithmetic
    (familywise_alpha / pair_count) as an exact value, independent of
    whether any particular fixture's effect is strong enough to be
    significant either way.

    This test exists because adversarial mutation testing showed that
    disabling this correction (using familywise_alpha directly as the
    per-pair alpha) is invisible to a very-strong-effect fixture: an
    effect large enough to be significant after correction remains
    significant before correction too, so the correction's VALUE must be
    checked directly rather than only inferred from a detection outcome."""
    x = _permutation_test_dataset()  # D=6 -> C(6,2) = 15 pairs
    familywise_alpha = 0.05
    result = d3.compare_pairwise_joint_distributions(
        x, x, practical_tvd_tolerance=0.5, familywise_alpha=familywise_alpha,
    )
    expected_pair_count = 6 * 5 // 2
    expected_alpha = familywise_alpha / expected_pair_count

    assert result["pair_count"] == expected_pair_count
    assert result["multiple_comparison"]["familywise_alpha"] == familywise_alpha
    assert result["multiple_comparison"]["per_pair_alpha"] == pytest.approx(expected_alpha)
    for record in result["pair_summaries"]:
        assert record["alpha_bonferroni"] == pytest.approx(expected_alpha)


def test_first_order_bonferroni_alpha_is_correctly_computed() -> None:
    """Same direct-arithmetic check as above, for the first-order family
    (familywise_alpha / (n_features + 1)), covering class_distribution and
    every per-feature binomial test."""
    x = _balanced_d4_dataset()
    y = _balanced_d4_labels()
    partition = d3.summarize_partition(x, y)
    familywise_alpha = 0.05
    n_features = 4
    expected_alpha = familywise_alpha / (n_features + 1)

    comparison = d3.compare_with_intended_distribution(
        partition,
        expected_class_ratio=0.5, expected_bit_probability=0.5,
        class_ratio_tolerance=0.05, bit_probability_tolerance=0.05,
        familywise_alpha=familywise_alpha,
    )
    assert comparison["multiple_comparison"]["per_test_alpha"] == pytest.approx(expected_alpha)
    assert comparison["class_distribution"]["alpha"] == pytest.approx(expected_alpha)
    for record in comparison["binary_marginals"]["per_feature"]:
        assert record["alpha_bonferroni"] == pytest.approx(expected_alpha)


# ============================================================
# Invariant C -- reproducibility.
# ============================================================

def test_d3_statistics_are_reproducible() -> None:
    """D3's generic module has no randomness (confirmed by source
    inspection: no numpy.random, no seed parameter anywhere). Two calls on
    the identical input arrays and configuration must return an identical
    (deep-equal) result dict, and evaluate_decision on each must match."""
    x = _balanced_d4_dataset()
    y = _balanced_d4_labels()
    kwargs = dict(intended_distribution=_intended_d4(), familywise_alpha=0.05)

    results_1 = d3.audit_distribution(x, y, x, y, x, y, **kwargs)
    results_2 = d3.audit_distribution(x, y, x, y, x, y, **kwargs)

    assert results_1 == results_2
    assert d3.evaluate_decision(results_1) == d3.evaluate_decision(results_2)


def test_pairwise_joint_summary_is_reproducible_on_larger_input() -> None:
    """Reproducibility check on the permutation-test-sized fixture (n=20,
    d=6, 15 pairs) as a second, independent data point beyond the small
    8x4 fixture above."""
    x = _permutation_test_dataset()
    r1 = d3.pairwise_joint_summary(x)
    r2 = d3.pairwise_joint_summary(x)
    assert r1 == r2


# ============================================================
# Invariant D -- feature ordering (invariance vs. equivariance).
# ============================================================

def test_marginal_statistics_are_permutation_invariant() -> None:
    """Aggregate marginal statistics (min/max: exact; mean-type aggregates
    that resum already-computed floats: tight tolerance -- see module
    docstring) must not depend on feature order. Per-feature outputs must
    be a correctly-relabeled permutation of the originals, not identical
    positional order (asserting positional equality would be a FALSE
    invariant)."""
    x = _permutation_test_dataset()
    perm = np.array([5, 3, 1, 4, 0, 2])  # fixed, predeclared, arbitrary
    permuted = x[:, perm]

    original = d3.binary_marginal_distribution(x)
    shuffled = d3.binary_marginal_distribution(permuted)

    # min/max: order-independent selection, no arithmetic combination -> exact.
    assert shuffled["minimum_one_probability"] == original["minimum_one_probability"]
    assert shuffled["maximum_one_probability"] == original["maximum_one_probability"]
    # mean: sums 6 already-computed floats in a different order -> tight tolerance.
    assert _approx(shuffled["mean_one_probability"], original["mean_one_probability"])

    orig_probs = original["per_feature_one_probability"]
    shuf_probs = shuffled["per_feature_one_probability"]
    for k in range(6):
        # Each per-feature probability is itself a reduction over the SAME
        # 20 rows regardless of how OTHER columns are ordered, so this is
        # exact, not approximate.
        assert shuf_probs[k] == orig_probs[perm[k]]

    # feature_statistics: global mean/variance over the fully flattened
    # array. Permuting columns changes flatten order (row-major), so this
    # again sums the same 120 numbers in a different order -> tight tolerance
    # for the mean; variance similarly.
    orig_stats = d3.feature_statistics(x)
    shuf_stats = d3.feature_statistics(permuted)
    assert _approx(shuf_stats["feature_mean"], orig_stats["feature_mean"])
    assert _approx(shuf_stats["feature_variance"], orig_stats["feature_variance"])


def test_compare_binary_marginals_aggregates_are_permutation_invariant() -> None:
    """Same invariance/equivariance split as above, for
    compare_binary_marginals's aggregate vs. per-feature outputs."""
    x = _permutation_test_dataset()
    perm = np.array([5, 3, 1, 4, 0, 2])
    permuted = x[:, perm]

    original = d3.compare_binary_marginals(
        d3.binary_marginal_distribution(x), expected_probability=0.5, practical_tolerance=0.1,
    )
    shuffled = d3.compare_binary_marginals(
        d3.binary_marginal_distribution(permuted), expected_probability=0.5, practical_tolerance=0.1,
    )

    assert shuffled["maximum_absolute_difference"] == original["maximum_absolute_difference"]
    assert shuffled["practically_consistent"] == original["practically_consistent"]
    assert _approx(shuffled["mean_absolute_difference"], original["mean_absolute_difference"])
    assert _approx(shuffled["rms_difference"], original["rms_difference"])

    orig_diff = original["per_feature_absolute_difference"]
    shuf_diff = shuffled["per_feature_absolute_difference"]
    for k in range(6):
        assert shuf_diff[k] == orig_diff[perm[k]]


def test_pairwise_statistics_are_permutation_equivariant() -> None:
    """The precise, index-aware equivariance check: for every original pair
    (i, j), find its new position under the permutation, and confirm the
    table matches -- with the (0,1)/(1,0) cells correctly swapped whenever
    the pair's feature order flips (i.e. i comes after j in the permuted
    array). This directly targets an accidental pair-indexing regression,
    which a purely aggregate-level check could miss."""
    x = _permutation_test_dataset()
    perm = np.array([5, 3, 1, 4, 0, 2])
    permuted = x[:, perm]
    inverse_perm = np.argsort(perm)  # inverse_perm[old_feature] = new position

    original = d3.pairwise_joint_tables(x)
    shuffled = d3.pairwise_joint_tables(permuted)
    shuffled_lookup = {(row[0], row[1]): tuple(int(v) for v in row[2]) for row in shuffled["pairs"]}

    for i, j, table in original["pairs"]:
        new_i, new_j = int(inverse_perm[i]), int(inverse_perm[j])
        swapped = new_i > new_j
        key = (min(new_i, new_j), max(new_i, new_j))
        a00, a01, a10, a11 = (int(v) for v in table)
        expected = (a00, a10, a01, a11) if swapped else (a00, a01, a10, a11)
        assert shuffled_lookup[key] == expected, (i, j, key, swapped)

    # Independent cross-check: phi (a*d - b*c, symmetric under b<->c swap)
    # and TVD-from-uniform (symmetric in all four cells) are provably
    # invariant to the swap itself, so the MULTISET of per-pair values must
    # match exactly regardless of relabeling -- a second, algebraically
    # independent way of catching the same class of indexing bug.
    orig_summary = d3.pairwise_joint_summary(x)
    shuf_summary = d3.pairwise_joint_summary(permuted)
    orig_phi = sorted(r["phi"] for r in orig_summary["pair_summaries"])
    shuf_phi = sorted(r["phi"] for r in shuf_summary["pair_summaries"])
    assert shuf_phi == orig_phi
    orig_tvd = sorted(r["tvd_from_uniform_4cell"] for r in orig_summary["pair_summaries"])
    shuf_tvd = sorted(r["tvd_from_uniform_4cell"] for r in shuf_summary["pair_summaries"])
    assert shuf_tvd == orig_tvd


def test_compare_pairwise_joint_distributions_is_equivariant_under_consistent_relabeling() -> None:
    """compare_pairwise_joint_distributions zips its two inputs' pair lists
    BY POSITION (see review item 4/8) -- it assumes both partitions share
    feature semantics. Applying the SAME permutation to both partitions
    therefore must leave every aggregate exactly (min/max/counts) or
    approximately (mean-type) unchanged, since each pairwise comparison is
    still between the same two underlying tables, just relabeled."""
    a = _permutation_test_dataset()
    b = _permutation_test_dataset()[::-1].copy()  # a different but same-shape partition
    perm = np.array([5, 3, 1, 4, 0, 2])

    original = d3.compare_pairwise_joint_distributions(
        a, b, practical_tvd_tolerance=0.2, familywise_alpha=0.05,
    )
    shuffled = d3.compare_pairwise_joint_distributions(
        a[:, perm], b[:, perm], practical_tvd_tolerance=0.2, familywise_alpha=0.05,
    )

    assert shuffled["practical_effect"]["maximum_tvd"] == original["practical_effect"]["maximum_tvd"]
    assert (
        shuffled["practical_effect"]["practically_inconsistent_pair_count"]
        == original["practical_effect"]["practically_inconsistent_pair_count"]
    )
    assert (
        shuffled["statistical_effect"]["statistically_inconsistent_pair_count"]
        == original["statistical_effect"]["statistically_inconsistent_pair_count"]
    )
    assert shuffled["decision"] == original["decision"]
    assert _approx(shuffled["practical_effect"]["mean_tvd"], original["practical_effect"]["mean_tvd"])


# ============================================================
# Invariant E -- finite-sample, degenerate, and invalid inputs.
# ============================================================

def test_smallest_valid_dataset_is_accepted() -> None:
    """n=1 is accepted by D3's own validation contract (no additional
    minimum-n check exists beyond n>=1; confirmed by reading
    _validate_dataset). A single sample makes every feature fully
    concentrated, so every feature-pair table is one-hot and phi must fall
    back to its documented zero-denominator value."""
    x = np.array([[0, 1, 1]], dtype=np.uint8)
    y = np.array([1], dtype=np.uint8)
    summary = d3.summarize_partition(x, y)
    assert summary["samples"] == 1
    assert summary["class_distribution"]["positive_ratio"] == 1.0
    for pair in summary["second_order_structure"]["pair_summaries"]:
        assert pair["phi"] == 0.0


def test_zero_probability_feature_is_handled() -> None:
    """A feature that is always 0: binary_entropy must clip before taking
    log2, producing a tiny FINITE strictly-positive entropy -- NOT exactly
    0.0 (that would indicate the clip was removed, which would instead
    crash toward -inf/NaN on a true zero probability) and NOT NaN/Inf."""
    x = np.zeros((10, 2), dtype=np.uint8)
    x[:, 1] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    entropy = d3.binary_entropy(x)
    h0 = entropy["per_feature_entropy"][0]
    assert np.isfinite(h0)
    assert 0.0 < h0 < 1e-6


def test_unit_probability_feature_is_handled() -> None:
    """Symmetric case: a feature that is always 1."""
    x = np.ones((10, 2), dtype=np.uint8)
    x[:, 1] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    entropy = d3.binary_entropy(x)
    h0 = entropy["per_feature_entropy"][0]
    assert np.isfinite(h0)
    assert 0.0 < h0 < 1e-6


def test_degenerate_pair_table_has_defined_phi() -> None:
    """A feature pair where one feature is fully concentrated makes at
    least one of phi's four marginal-product denominator terms exactly
    zero. The documented boundary behavior (see _phi_from_table) is
    phi == 0.0 exactly -- not NaN, not Inf, not a crash."""
    x = np.zeros((10, 2), dtype=np.uint8)
    x[:, 1] = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    summary = d3.pairwise_joint_summary(x)
    phi = summary["pair_summaries"][0]["phi"]
    assert phi == 0.0
    assert np.isfinite(phi)


def test_all_identical_rows_are_handled_without_crash() -> None:
    """N copies of the same row: every feature is fully concentrated. D3
    does not test for duplicate ROWS (D1's scope) -- this only checks that
    D3's OWN marginal/entropy/second-order statistics stay finite and
    well-defined on such data, which is a real code path D3 must survive
    even though it isn't the thing D3 is measuring."""
    row = np.array([1, 0, 1], dtype=np.uint8)
    x = np.tile(row, (12, 1))
    y = np.zeros(12, dtype=np.uint8)
    summary = d3.summarize_partition(x, y)
    assert np.all(np.isfinite(summary["binary_entropy"]["per_feature_entropy"]))
    for pair in summary["second_order_structure"]["pair_summaries"]:
        assert pair["phi"] == 0.0
        assert np.isfinite(pair["tvd_from_uniform_4cell"])


def test_chi_square_handles_zero_mass_columns_without_crash() -> None:
    """Two single-sample partitions whose combined 2x4 table leaves at
    least one of the four columns with zero total mass: the
    implementation's `expected > 0` masking must avoid a division by zero,
    and must still yield a finite chi-square statistic and a valid
    p-value in [0, 1] (see review item on zero-mass cells)."""
    a = np.array([[0, 1]], dtype=np.uint8)  # (feature_i, feature_j) = (0, 1)
    b = np.array([[1, 1]], dtype=np.uint8)  # (feature_i, feature_j) = (1, 1)
    result = d3.compare_pairwise_joint_distributions(
        a, b, practical_tvd_tolerance=0.5, familywise_alpha=0.05,
    )
    record = result["pair_summaries"][0]
    assert np.isfinite(record["chi_square"])
    assert 0.0 <= record["p_value"] <= 1.0


def test_class_ratio_confidence_interval_at_zero_and_full_boundary() -> None:
    """Clopper-Pearson's explicit zero-successes (lo=0.0) and all-successes
    (hi=1.0) boundary branches must be hit exactly, without raising or
    producing NaN."""
    all_negative = d3.class_distribution(np.zeros(10, dtype=np.uint8))
    assert all_negative["positive_ratio_ci"]["lower"] == 0.0
    assert np.isfinite(all_negative["positive_ratio_ci"]["upper"])

    all_positive = d3.class_distribution(np.ones(10, dtype=np.uint8))
    assert all_positive["positive_ratio_ci"]["upper"] == 1.0
    assert np.isfinite(all_positive["positive_ratio_ci"]["lower"])


def test_wilson_interval_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        d3.wilson_interval(0, 0)


def test_empty_dataset_is_rejected() -> None:
    x = np.empty((0, 4), dtype=np.uint8)
    y = np.empty((0,), dtype=np.uint8)
    with pytest.raises(ValueError):
        d3.summarize_partition(x, y)


def test_nonbinary_dataset_is_rejected() -> None:
    x = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        d3.binary_marginal_distribution(x)


def test_label_length_mismatch_is_rejected() -> None:
    x = np.zeros((5, 3), dtype=np.uint8)
    y = np.zeros((4,), dtype=np.uint8)
    with pytest.raises(ValueError):
        d3.summarize_partition(x, y)


def test_nonbinary_labels_are_rejected() -> None:
    x = np.zeros((3, 2), dtype=np.uint8)
    y = np.array([0, 1, 2], dtype=np.uint8)
    with pytest.raises(ValueError):
        d3.summarize_partition(x, y)


def test_mismatched_feature_dimensions_are_rejected() -> None:
    a = np.zeros((5, 3), dtype=np.uint8)
    b = np.zeros((5, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        d3.compare_pairwise_joint_distributions(
            a, b, practical_tvd_tolerance=0.1, familywise_alpha=0.05,
        )


def test_audit_distribution_supports_two_partitions_without_test_set() -> None:
    """test/test_labels are optional (Optional[np.ndarray] = None in the
    signature). Only train+validation must still produce a valid,
    evaluable result, with exactly one partition-comparison pair."""
    x = _balanced_d4_dataset()
    y = _balanced_d4_labels()
    results = d3.audit_distribution(
        x, y, x[::-1].copy(), y[::-1].copy(),
        intended_distribution=_intended_d4(), familywise_alpha=0.05,
    )
    assert "test" not in results["partitions"]
    assert set(results["partition_comparisons"].keys()) == {"train_validation"}
    decision = d3.evaluate_decision(results)
    assert decision["outcome"] in {"PASS", "CONDITIONAL_PASS", "INCONCLUSIVE", "FAIL"}


# ============================================================
# Gohr driver -- configuration/wiring only (not the statistical machinery,
# not dataset generation, not the model).
# ============================================================

def _valid_gohr_args(**overrides) -> argparse.Namespace:
    base = dict(
        train_samples=10, validation_samples=10, test_samples=10, rounds=5,
        class_ratio_tolerance=0.005, bit_probability_tolerance=0.01,
        pairwise_tvd_tolerance=0.01, familywise_alpha=0.01,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_gohr_driver_accepts_valid_configuration() -> None:
    gohr_d3.validate_args(_valid_gohr_args())  # must not raise


def test_gohr_driver_rejects_out_of_range_tolerance() -> None:
    with pytest.raises(ValueError):
        gohr_d3.validate_args(_valid_gohr_args(class_ratio_tolerance=1.5))


def test_gohr_driver_rejects_nonpositive_familywise_alpha() -> None:
    with pytest.raises(ValueError):
        gohr_d3.validate_args(_valid_gohr_args(familywise_alpha=0.0))


def test_gohr_driver_rejects_nonpositive_sample_counts() -> None:
    with pytest.raises(ValueError):
        gohr_d3.validate_args(_valid_gohr_args(train_samples=0))


def test_gohr_driver_rejects_nonpositive_rounds() -> None:
    with pytest.raises(ValueError):
        gohr_d3.validate_args(_valid_gohr_args(rounds=0))


def test_gohr_driver_default_output_path_reflects_configuration() -> None:
    args = _valid_gohr_args(train_samples=123, validation_samples=45, test_samples=67, rounds=5)
    path = gohr_d3.default_output(args)
    assert "123_45_67_5r" in str(path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
