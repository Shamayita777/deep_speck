"""
D3 method-validity suite (audit.dataset.d3_distribution_statistics).

What this suite is
------------------
Evidence that the D3 statistical methods do what their certificates claim,
under declared, controlled operating conditions. It is organised by the
specification sections it answers:

    §1.1  first-order tests: size, power, class skew, dependence stress
    §1.2  second-order 2x4 homogeneity: degrees of freedom, sparse and
          degenerate tables, the Monte Carlo fallback
    §1.3  multiplicity: what each correction actually controls
    §1.4  complete-family (64 features, 2,016 pairs, 3 comparisons) size
          and power
    §1.5  count reconstruction from stored proportions
    §1.6  which fields the decision consumes; legacy PASS without precision
    §1.7  four-way decision coverage of the confirmatory evaluator

What it is NOT
--------------
It is not evidence about any real dataset, and a finite simulation is not
proof of universal calibration. Every rate below is an estimate for the
stated sample sizes, alphas and generating laws, reported with a Wilson
interval. Assertions are written as "not detectably above nominal" (the
interval's lower bound is <= alpha) rather than "below nominal", because a
point estimate under a finite budget cannot establish the latter.

The larger-budget runs whose numbers the method-validity report quotes are
produced by audit/dataset/tools/d3_method_validity_calibration.py; this
suite uses smaller budgets so it stays a unit-test run.

Randomness
----------
numpy.random.Generator(PCG64) with a fixed seed per test. Injected effects
are placed at a feature / pair drawn from an RNG stream separate from the
data stream, so no fixture is built around a detector's known blind spots or
favourite positions.

Independence
------------
The exact binomial and homogeneity tests assume independent rows. §1.1
includes a controlled-dependence stress test showing that assumption is
load-bearing: D3 is NOT calibrated under dependent rows, and D2 is the
dimension that tests for dependence.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, multivariate_hypergeom

from audit.dataset import d3_distribution_statistics as d3


def _rate_ci(hits: int, reps: int) -> tuple[float, float, float]:
    lo, hi = d3.wilson_interval(int(hits), int(reps), confidence=0.95)
    return hits / reps, lo, hi


def _not_above(hits: int, reps: int, alpha: float) -> bool:
    """Empirical size is not detectably above alpha (95% Wilson lower <= alpha)."""
    return _rate_ci(hits, reps)[1] <= alpha


def _bits(rng: np.random.Generator, n: int, d: int = 64, p: float = 0.5) -> np.ndarray:
    return (rng.random((n, d)) < p).astype(np.uint8)


# ===========================================================================
# §1.1  First-order tests
# ===========================================================================

def test_binomial_conformity_size_under_independent_null() -> None:
    """Independent Bernoulli(0.5) rows at the declared reference: the exact
    binomial test is not detectably above nominal at alpha = 0.05 and 0.01.
    Exact tests are conservative for discrete data, so a lower bound is
    also checked: size is not below alpha/3 (the test is not vacuous)."""
    rng = np.random.default_rng(11001)
    reps, n = 3000, 801
    k = rng.binomial(n, 0.5, size=reps)
    for alpha in (0.05, 0.01):
        fam = [d3.binomial_reference_family(
            [{"label": "b", "successes": int(ki), "trials": n,
              "expected_probability": 0.5}],
            family_id="x", description="", familywise_alpha=alpha, tolerance=0.01)
            for ki in k]
        hits = sum(f["records"]["significant"][0] for f in fam)
        rate, lo, hi = _rate_ci(hits, reps)
        assert lo <= alpha, (alpha, rate, lo, hi)
        assert hi >= alpha / 3, (alpha, rate, lo, hi)


def test_class_ratio_skew_is_classified_by_statistical_and_practical_criteria_separately() -> None:
    """Deterministic counts, so the categories are exact consequences of the
    rule, not sampling luck.

    * 0.520 at n = 200,000: significant and above tolerance 0.005 -> FAIL
    * 0.5015 at n = 4,000,000: significant, bound inside 0.005 -> STATISTICAL_ONLY
    * 0.5000 at n = 4,000,000: not significant, bound inside -> EQUIVALENT
    * 0.5000 at n = 2,000: not significant, bound OUTSIDE -> INSUFFICIENT_PRECISION
    """
    cases = [("skewed", 104_000, 200_000, "CONFIRMED_DEVIATION"),
             ("small_shift", 2_006_000, 4_000_000, "STATISTICAL_ONLY"),
             ("balanced", 2_000_000, 4_000_000, "EQUIVALENT_WITHIN_TOLERANCE"),
             ("too_small", 1_000, 2_000, "INSUFFICIENT_PRECISION")]
    fam = d3.binomial_reference_family(
        [{"label": c, "successes": k, "trials": n, "expected_probability": 0.5}
         for c, k, n, _ in cases],
        family_id="R1", description="class balance", familywise_alpha=0.01, tolerance=0.005)
    assert fam["records"]["category"] == [c[3] for c in cases]
    assert fam["outcome"] == "FAIL"


@pytest.mark.parametrize("delta_over_tol,minimum_power,maximum_power", [
    (0.5, 0.0, 0.15), (1.0, 0.30, 0.70), (2.0, 0.95, 1.0)])
def test_marginal_shift_power_at_prespecified_effect_sizes(delta_over_tol, minimum_power, maximum_power) -> None:
    """Power of the 64-feature first-order family (Bonferroni at 0.01) to flag
    a single shifted feature, at shifts below, at, and above tolerance 0.01,
    n = 36,000. The shifted feature is drawn from a separate RNG stream.

    Analytic expectation: SE = 0.00264, z_crit(0.01/64) = 3.78, so power is
    ~0.03 at delta = 0.005, ~0.50 at 0.01, ~1.00 at 0.02. The bands are wide
    enough to be stable under 200 replicates and narrow enough that a broken
    test or correction would leave them.

    Statistical and practical criteria are tallied separately: at half the
    tolerance a detection should almost never be a CONFIRMED_DEVIATION.
    """
    tol, n, reps = 0.01, 36_000, 200
    data_rng = np.random.default_rng(11100 + int(delta_over_tol * 10))
    pos_rng = np.random.default_rng(91100 + int(delta_over_tol * 10))
    p = np.full(64, 0.5)
    detected = confirmed = 0
    for _ in range(reps):
        j = int(pos_rng.integers(64))
        pj = p.copy()
        pj[j] += delta_over_tol * tol
        k = data_rng.binomial(n, pj)
        fam = d3.binomial_reference_family(
            [{"label": f"b{i}", "successes": int(k[i]), "trials": n,
              "expected_probability": 0.5} for i in range(64)],
            family_id="R2", description="", familywise_alpha=0.01, tolerance=tol)
        detected += bool(fam["records"]["significant"][j])
        confirmed += fam["records"]["category"][j] == "CONFIRMED_DEVIATION"
    power, lo, hi = _rate_ci(detected, reps)
    assert minimum_power <= power <= maximum_power, (power, lo, hi)
    if delta_over_tol < 1:
        assert confirmed / reps <= 0.05


def test_first_order_fixtures_are_reproducible() -> None:
    rng = np.random.default_rng(11200)
    x = _bits(rng, 3000, 8)
    items = [{"label": str(i), "successes": int(x[:, i].sum()), "trials": 3000,
              "expected_probability": 0.5} for i in range(8)]
    a = d3.binomial_reference_family(items, family_id="f", description="", familywise_alpha=0.01, tolerance=0.02)
    b = d3.binomial_reference_family(items, family_id="f", description="", familywise_alpha=0.01, tolerance=0.02)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_dependent_rows_break_calibration_stress_condition() -> None:
    """Controlled dependence: every independent draw is repeated 4 times, so
    the effective sample size is n/4. The binomial test, which assumes n
    independent rows, is then grossly anti-conservative. This is recorded as
    a limitation, not a defect to be fixed in D3: row independence is an
    assumption of D3 and is tested by D2."""
    rng = np.random.default_rng(11300)
    reps, n_eff, block = 4000, 500, 4
    k = rng.binomial(n_eff, 0.5, size=reps) * block
    alpha = 0.05
    hits = sum(
        d3.binomial_reference_family(
            [{"label": "b", "successes": int(ki), "trials": n_eff * block,
              "expected_probability": 0.5}],
            family_id="x", description="", familywise_alpha=alpha, tolerance=None)["records"]["significant"][0]
        for ki in k)
    rate, lo, _ = _rate_ci(hits, reps)
    assert lo > 4 * alpha, rate


# ===========================================================================
# §1.2  Second-order homogeneity
# ===========================================================================

@pytest.mark.parametrize("row_a,row_b,df", [
    ([10, 12, 9, 11], [8, 13, 10, 9], 3),     # nondegenerate
    ([0, 12, 9, 11], [0, 13, 10, 9], 2),      # one zero-mass column
    ([0, 0, 20, 22], [0, 0, 18, 24], 1),      # feature i constant: two empty columns
    ([0, 0, 0, 42], [0, 0, 0, 30], 0),        # both features constant
])
def test_observed_table_degrees_of_freedom(row_a, row_b, df) -> None:
    r = d3.homogeneity_chi_square([row_a, row_b])
    assert r["degrees_of_freedom"] == df
    if df == 0:
        assert r["asymptotic_p_value"] == 1.0
    else:
        assert math.isclose(r["asymptotic_p_value"], chi2.sf(r["chi_square"], df))


def test_legacy_pair_comparison_keeps_fixed_df_while_confirmatory_uses_observed_df() -> None:
    """The legacy path is the c603522 method (fixed df = 3, kept for the
    historical Gohr certificate); the confirmatory test on the SAME table
    uses the observed df = 1. Full legacy output is pinned in
    test_d3_legacy_golden.py."""
    a = np.zeros((40, 2), dtype=np.uint8)
    b = np.zeros((40, 2), dtype=np.uint8)
    a[::2, 1] = 1
    b[::3, 1] = 1
    rec = d3.compare_pairwise_joint_distributions(
        a, b, practical_tvd_tolerance=0.5, familywise_alpha=0.05)["pair_summaries"][0]
    assert rec["degrees_of_freedom"] == 3
    conf = d3.homogeneity_test_batch(d3._pair_cell_counts(a), d3._pair_cell_counts(b),
                                     per_hypothesis_alpha=0.05, mc_seed=0)
    assert int(conf["degrees_of_freedom"][0]) == 1


def test_fixed_three_df_was_conservative_and_observed_df_is_calibrated_with_constant_feature() -> None:
    """Defect evidence for the df change. One feature constant (two empty
    columns), n = 1000 per partition, 40,000 null replicates at alpha 0.05.
    The legacy fixed df=3 rejects ~0.5% of the time; the observed df ~5%."""
    rng = np.random.default_rng(12100)
    reps = 40_000
    a = rng.multinomial(1000, [0, 0, 0.5, 0.5], reps)
    b = rng.multinomial(1000, [0, 0, 0.5, 0.5], reps)
    t = d3.homogeneity_test_batch(a, b, per_hypothesis_alpha=0.05, mc_seed=0)
    assert set(t["method"]) == {"asymptotic"}
    fixed = int(np.sum(chi2.sf(t["chi_square"], 3) < 0.05))
    observed = int(np.sum(t["significant"]))
    assert _rate_ci(fixed, reps)[2] < 0.02
    rate, lo, hi = _rate_ci(observed, reps)
    assert lo <= 0.05 <= hi, (rate, lo, hi)


def test_dense_regime_asymptotic_test_is_calibrated() -> None:
    rng = np.random.default_rng(12200)
    reps = 100_000
    p = [0.25] * 4
    t = d3.homogeneity_test_batch(rng.multinomial(1000, p, reps), rng.multinomial(1000, p, reps),
                                  per_hypothesis_alpha=0.01, mc_seed=0)
    assert set(t["method"]) == {"asymptotic"}
    rate, lo, hi = _rate_ci(int(t["significant"].sum()), reps)
    assert lo <= 0.01 <= hi, (rate, lo, hi)


SPARSE_UNBALANCED = (100_000, 20, [0.01, 0.09, 0.45, 0.45])


def test_asymptotic_chi_square_is_anticonservative_in_sparse_unbalanced_tables() -> None:
    """Defect evidence for the Monte Carlo fallback: with a rare cell and a
    20-row partition against a 100,000-row one, the asymptotic test rejects
    a true null far more often than nominal. (At alpha 1e-5 the calibration
    tool measures ~100x inflation.)"""
    rng = np.random.default_rng(12300)
    na, nb, p = SPARSE_UNBALANCED
    reps = 200_000
    h = d3._homogeneity_batch(rng.multinomial(na, p, reps), rng.multinomial(nb, p, reps))
    rate, lo, _ = _rate_ci(int(np.sum(h["asymptotic_p_value"] < 1e-3)), reps)
    assert lo > 5e-3, rate
    assert np.all(h["minimum_expected_count"] < d3.MINIMUM_EXPECTED_COUNT)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_monte_carlo_fallback_is_calibrated_in_the_sparse_unbalanced_regime(alpha) -> None:
    rng = np.random.default_rng(12400 + int(alpha * 1000))
    na, nb, p = SPARSE_UNBALANCED
    reps = 1500
    t = d3.homogeneity_test_batch(rng.multinomial(na, p, reps), rng.multinomial(nb, p, reps),
                                  per_hypothesis_alpha=alpha, mc_seed=5)
    assert set(t["method"]) <= {"monte_carlo", "degenerate"}
    hits = int(t["significant"].sum())
    assert _not_above(hits, reps, alpha), _rate_ci(hits, reps)


def _exact_conditional_p(row_a, row_b) -> float:
    """Freeman-Halton-style exact conditional p-value by full enumeration."""
    col = np.asarray(row_a) + np.asarray(row_b)
    n_a = int(np.sum(row_a))
    live = col > 0
    colv = col[live]
    tot = int(colv.sum())
    ea, eb = n_a * colv / tot, (tot - n_a) * colv / tot

    def stat(x):
        x = np.asarray(x, dtype=float)
        y = colv - x
        return float(((x - ea) ** 2 / ea).sum() + ((y - eb) ** 2 / eb).sum())

    t_obs = stat(np.asarray(row_a)[live])
    dist = multivariate_hypergeom(m=colv, n=n_a)
    p = 0.0
    for x in itertools.product(*[range(int(c) + 1) for c in colv]):
        if sum(x) == n_a and stat(x) >= t_obs - 1e-9:
            p += float(dist.pmf(x))
    return p


@pytest.mark.parametrize("row_a,row_b", [([3, 0, 1, 2], [0, 4, 1, 0]), ([5, 1, 0, 0], [1, 2, 2, 1]),
                                         ([0, 2, 0, 7], [0, 1, 3, 1])])
def test_monte_carlo_p_value_agrees_with_exact_enumeration(row_a, row_b) -> None:
    exact = _exact_conditional_p(row_a, row_b)
    mc = d3.monte_carlo_homogeneity_p_value(row_a, row_b, draws=200_000,
                                            rng=np.random.default_rng(12500))["p_value"]
    se = math.sqrt(max(exact * (1 - exact), 1e-6) / 200_000)
    assert abs(mc - exact) < 5 * se + 1e-5, (mc, exact)


def test_unattainable_monte_carlo_rejection_is_reported_unresolved() -> None:
    """If the budget cannot reach the alpha, the hypothesis is UNRESOLVED_TEST
    (INCONCLUSIVE), not a silent non-rejection that could support PASS."""
    t = d3.homogeneity_test_batch(np.array([[1, 0, 0, 3]]), np.array([[0, 3, 1, 0]]),
                                  per_hypothesis_alpha=1e-6, mc_seed=1, mc_max_draws=1000)
    assert t["method"][0] == "monte_carlo"
    assert not t["rejection_attainable"][0]
    assert not t["significant"][0]
    # The Monte Carlo test is a DIAGNOSTIC: the gating exact test on the same table is always resolvable.
    ex = d3.exact_homogeneity_test_batch(np.array([[1, 0, 0, 3]]), np.array([[0, 3, 1, 0]]),
                                         per_hypothesis_alpha=1e-6)
    assert 0.0 < ex["p_value"][0] <= 1.0


def test_two_sample_proportion_test_is_calibrated() -> None:
    rng = np.random.default_rng(12600)
    reps = 100_000
    k1 = rng.binomial(5000, 0.5, reps)
    k2 = rng.binomial(4000, 0.5, reps)
    t = d3.homogeneity_test_batch(np.stack([5000 - k1, k1], 1), np.stack([4000 - k2, k2], 1),
                                  per_hypothesis_alpha=0.01, mc_seed=0)
    rate, lo, hi = _rate_ci(int(t["significant"].sum()), reps)
    assert lo <= 0.01 <= hi, (rate, lo, hi)


# ---------------------------------------------------------------------------
# §1.2b  Finite-sample effect bounds (TVD and proportion differences)
# ---------------------------------------------------------------------------
#
# Construction (d3.BOUND_CONSTRUCTION): Clopper-Pearson intervals for every
# cell of every sample, Bonferroni over all of them, then interval arithmetic.
# The guarantee is a theorem given the Clopper-Pearson guarantee
# P_p(p not in CP(K)) <= alpha for K ~ Binomial(n, p), every n, every p.
# These tests check (a) that guarantee EXACTLY, by summing binomial pmfs, at
# the per-cell alphas the confirmatory design actually uses, and (b) the
# implementation's coverage by simulation, with an acceptance rule fixed here
# before any run: the one-sided 99% Clopper-Pearson LOWER bound on empirical
# coverage must be >= 1 - alpha.

#: Per-cell alphas of the equal six-family design at global alpha 0.01:
#: pair TVD (A/B/C): 0.01/6/6048/8; per proportion in D: 0.01/6/192/2;
#: R2: 0.01/6/384; R1: 0.01/6/3.
PRODUCTION_CELL_ALPHAS = (0.01 / 6 / 6048 / 8, 0.01 / 6 / 192 / 2, 0.01 / 6 / 384, 0.01 / 6 / 3)


def _exact_cp_noncoverage(n: int, p: float, alpha: float) -> float:
    from scipy.stats import binom
    sd = math.sqrt(n * p * (1 - p)) if 0 < p < 1 else 0.0
    lo_k = max(0, int(n * p - 40 * sd - 40))
    hi_k = min(n, int(n * p + 40 * sd + 40))
    k = np.arange(lo_k, hi_k + 1, dtype=np.int64)
    lo, hi = d3.clopper_pearson_bounds(k, np.full_like(k, n), alpha=alpha)
    miss = (p < lo) | (p > hi)
    outside = binom.cdf(lo_k - 1, n, p) + binom.sf(hi_k, n, p)
    return float(np.sum(binom.pmf(k[miss], n, p)) + outside)


@pytest.mark.parametrize("alpha", PRODUCTION_CELL_ALPHAS)
@pytest.mark.parametrize("n,p", [(50, 0.5), (1000, 0.25), (1000, 1e-3), (500_000, 0.25),
                                 (500_000, 0.5), (5_000_000, 0.25), (500_000, 1e-5)])
def test_clopper_pearson_exact_noncoverage_at_production_alpha(n, p, alpha) -> None:
    """Exact (not simulated) non-coverage of the implemented interval at the
    per-cell alphas actually used. It must not exceed alpha."""
    assert _exact_cp_noncoverage(n, p, alpha) <= alpha * (1 + 1e-9)


def _coverage_lower_bound(hits: int, reps: int) -> float:
    lo, _ = d3.clopper_pearson_bounds(np.int64(hits), np.int64(reps), alpha=0.02)
    return float(lo)        # two-sided 98% => one-sided 99% lower bound


TVD_REGIMES = {
    "identical_uniform": (4000, 4000, [0.25] * 4, [0.25] * 4),
    "shifted": (5000, 3000, [0.25] * 4, [0.27, 0.24, 0.24, 0.25]),
    "rare_cells": (3000, 3000, [0.001, 0.009, 0.49, 0.5], [0.002, 0.008, 0.49, 0.5]),
    "unequal_sizes": (20000, 300, [0.25] * 4, [0.3, 0.2, 0.25, 0.25]),
    "zero_probability_cell": (2000, 2000, [0.0, 0.3, 0.3, 0.4], [0.0, 0.35, 0.25, 0.4]),
}


@pytest.mark.parametrize("regime", sorted(TVD_REGIMES))
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_tvd_bounds_cover_true_tvd_with_confidence(regime, alpha) -> None:
    na, nb, pa, pb = TVD_REGIMES[regime]
    reps = 20_000
    rng = np.random.default_rng(12700 + sorted(TVD_REGIMES).index(regime) * 10 + int(alpha * 100))
    true_tvd = 0.5 * np.abs(np.subtract(pa, pb)).sum()
    bd = d3.tvd_confidence_bounds(rng.multinomial(na, pa, reps), rng.multinomial(nb, pb, reps), alpha=alpha)
    covered = int(np.sum((bd["lower"] <= true_tvd + 1e-15) & (true_tvd <= bd["upper"] + 1e-15)))
    assert _coverage_lower_bound(covered, reps) >= 1 - alpha, (covered, reps)


@pytest.mark.parametrize("p1,p2,n1,n2", [(0.5, 0.5, 5000, 5000), (0.3, 0.32, 2000, 1500),
                                         (0.001, 0.003, 4000, 4000), (0.5, 0.6, 20000, 200)])
def test_proportion_difference_bounds_cover_with_confidence(p1, p2, n1, n2) -> None:
    reps, alpha = 20_000, 0.01
    rng = np.random.default_rng(int(1e6 * p1 + 1e3 * p2 + n2))
    bd = d3.proportion_difference_bounds(rng.binomial(n1, p1, reps), n1, rng.binomial(n2, p2, reps), n2,
                                         alpha=alpha)
    d = abs(p1 - p2)
    covered = int(np.sum((bd["abs_lower"] <= d + 1e-15) & (d <= bd["abs_upper"] + 1e-15)
                         & (bd["difference_lower"] <= p1 - p2 + 1e-15)
                         & (p1 - p2 <= bd["difference_upper"] + 1e-15)))
    assert _coverage_lower_bound(covered, reps) >= 1 - alpha, (covered, reps)


def test_tvd_bounds_boundary_cases() -> None:
    a = np.array([[0, 0, 5, 5], [10, 0, 0, 0], [3, 3, 3, 3]])
    b = np.array([[0, 0, 5, 5], [0, 0, 0, 10], [3, 3, 3, 3]])
    bd = d3.tvd_confidence_bounds(a, b, alpha=1e-6)
    assert np.all((0 <= bd["lower"]) & (bd["lower"] <= bd["upper"]) & (bd["upper"] <= 1))
    assert bd["lower"][0] == 0.0 and bd["lower"][2] == 0.0
    big = d3.tvd_confidence_bounds(np.array([10**6, 0, 0, 0]), np.array([0, 0, 0, 10**6]), alpha=1e-6)
    assert big["lower"] > 0.99 and big["upper"] == 1.0


def test_empirical_tvd_always_lies_inside_its_bounds() -> None:
    rng = np.random.default_rng(12900)
    a = rng.multinomial(300, [0.1, 0.2, 0.3, 0.4], 5000)
    b = rng.multinomial(90, [0.25] * 4, 5000)
    tvd = 0.5 * np.abs(a / 300 - b / 90).sum(axis=1)
    bd = d3.tvd_confidence_bounds(a, b, alpha=1e-4)
    assert np.all((bd["lower"] <= tvd + 1e-12) & (tvd <= bd["upper"] + 1e-12))


def test_clopper_pearson_limits_are_rounded_outward_never_inward() -> None:
    from scipy.stats import beta
    k, n, a = np.int64(37), np.int64(1000), 1e-7
    lo, hi = d3.clopper_pearson_bounds(k, n, alpha=a)
    assert lo <= beta.ppf(a / 2, 37, 964) and hi >= beta.ppf(1 - a / 2, 38, 963)


@pytest.mark.parametrize("bad", [
    dict(counts_a=np.array([1.0, 2.0]), counts_b=np.array([1, 2])),
    dict(counts_a=np.array([-1, 2]), counts_b=np.array([1, 2])),
    dict(counts_a=np.array([0, 0]), counts_b=np.array([1, 2])),
    dict(counts_a=np.array([1, 2, 3]), counts_b=np.array([1, 2])),
])
def test_tvd_bounds_refuse_invalid_counts(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        d3.tvd_confidence_bounds(alpha=0.05, **bad)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan"), float("inf"), True, "0.05"])
def test_bounds_refuse_invalid_alpha(alpha) -> None:
    with pytest.raises((TypeError, ValueError)):
        d3.tvd_confidence_bounds(np.array([1, 2]), np.array([2, 1]), alpha=alpha)


# ===========================================================================
# §1.3  Multiplicity
# ===========================================================================

def test_legacy_correction_is_per_partition_and_per_comparison_not_global() -> None:
    """Documents what the legacy `familywise_alpha` controls: each partition's
    65 first-order tests and each comparison's 2,016 pairs is its own family,
    so with three partitions the union bound is 6 x alpha, not alpha."""
    rng = np.random.default_rng(13100)
    parts = [_bits(rng, 300) for _ in range(3)]
    y = [np.arange(300) % 2 for _ in range(3)]
    res = d3.audit_distribution(parts[0], y[0], parts[1], y[1], parts[2], y[2],
                                intended_distribution={"expected_class_ratio": 0.5,
                                                       "expected_bit_probability": 0.5,
                                                       "class_ratio_tolerance": 0.01,
                                                       "bit_probability_tolerance": 0.01,
                                                       "pairwise_tvd_tolerance": 0.01},
                                familywise_alpha=0.01)
    first = [v["multiple_comparison"] for v in res["intended_distribution"].values()]
    second = [v["second_order_feature_pairs"]["multiple_comparison"]
              for v in res["partition_comparisons"].values()]
    union = sum(f["per_test_alpha"] * f["tests_in_first_order_family"] for f in first) + \
        sum(s["per_pair_alpha"] * 2016 for s in second)
    assert math.isclose(union, 0.06)


def test_weighted_bonferroni_allocation_sums_to_global_alpha() -> None:
    alloc = d3.allocate_familywise_alpha(0.01, ["R1", "R2", "A", "B", "C", "D"])
    assert math.isclose(sum(alloc.values()), 0.01)


def test_preregistration_refuses_family_alphas_above_the_global_alpha() -> None:
    rec = {"schema_version": d3.PREREGISTRATION_SCHEMA_VERSION, "approved": True, "approved_by": "Approver",
           "approved_utc": "2026-10-01T00:00:00+00:00", "approval_statement": d3.APPROVAL_STATEMENT,
           "data_examined_before_approval": "none", "global_familywise_alpha": 0.01,
           "family_alpha_allocation": {"R1": 0.02}, "family_roles": {"R1": "confirmatory"},
           "tolerances": {"class_ratio": None}, "tolerance_justification": {"class_ratio": ""}}
    with pytest.raises(d3.PreregistrationRejected):
        d3.validate_preregistration_content(rec)


def test_pairwise_family_divides_alpha_by_pairs_times_comparisons() -> None:
    rng = np.random.default_rng(13400)
    x = {n: _bits(rng, 500) for n in ("t", "v", "s")}
    fam = d3.pairwise_homogeneity_family(
        {"t_v": (x["t"], x["v"]), "t_s": (x["t"], x["s"]), "v_s": (x["v"], x["s"])},
        family_id="A", description="", familywise_alpha=0.002, tvd_tolerance=0.01, mc_seed=1)
    assert fam["hypothesis_count"] == 3 * 2016
    assert math.isclose(fam["per_hypothesis_alpha"], 0.002 / 6048)


# ===========================================================================
# §1.4  Complete-family size and power
# ===========================================================================

def _family_null_rejections(rng, reps, sizes, alpha, *, shift=None, pos_rng=None):
    """Run the full 64-feature / 2016-pair / 3-comparison family `reps` times.

    Returns (family_rejections, injected_pair_detections)."""
    rej = hit = 0
    iu, ju = np.triu_indices(64, k=1)
    for _ in range(reps):
        parts = {n: _bits(rng, s) for n, s in zip(("train", "validation", "test"), sizes)}
        target = None
        if shift is not None:
            k = int(pos_rng.integers(len(iu)))
            i, j = int(iu[k]), int(ju[k])
            v = parts["validation"]
            copy = rng.random(len(v)) < shift
            v[copy, j] = v[copy, i]
            target = k
        fam = d3.pairwise_homogeneity_family(
            {"train_validation": (parts["train"], parts["validation"]),
             "train_test": (parts["train"], parts["test"]),
             "validation_test": (parts["validation"], parts["test"])},
            family_id="A", description="", familywise_alpha=alpha, tvd_tolerance=None, mc_seed=0)
        sig = [np.asarray(c["pairs"]["significant"]) for c in fam["records"]["comparisons"].values()]
        rej += any(s.any() for s in sig)
        if target is not None:
            hit += bool(sig[0][target])
    return rej, hit


def test_complete_family_null_rejection_rate_not_above_alpha() -> None:
    """64 features, 2,016 pairs x 3 comparisons = 6,048 hypotheses, Bonferroni
    at familywise alpha 0.10 (a large alpha so a 300-replicate budget has
    events to count). The calibration tool repeats this at 0.05 and 0.01 with
    2,000 replicates."""
    rej, _ = _family_null_rejections(np.random.default_rng(14100), 300, (4000, 2000, 2000), 0.10)
    assert _not_above(rej, 300, 0.10), _rate_ci(rej, 300)


def test_complete_family_power_for_injected_pair_correlation() -> None:
    """Inject x_j <- x_i on a fraction q = 0.15 of validation rows at a pair
    drawn from a separate stream (each cell shifts by 0.0375; TVD 0.075).
    The gating test is exact (Fisher per cell, Bonferroni over 4 cells): per
    cell z = 0.0375 / sqrt(0.1875 (1/20000 + 1/10000)) = 7.07 against a
    critical 5.06 at alpha 0.01/6048/4, so per-cell power ~0.98. (At q = 0.10
    the exact test's power is ~0.85 where the former asymptotic chi-square
    had ~0.99: the measured price of an exact guarantee, reported by the
    calibration tool.)"""
    rej, hit = _family_null_rejections(np.random.default_rng(14200), 40, (20000, 10000, 10000),
                                       0.01, shift=0.15, pos_rng=np.random.default_rng(94200))
    power, lo, _ = _rate_ci(hit, 40)
    assert lo >= 0.8, power


# ===========================================================================
# §1.5  Count reconstruction
# ===========================================================================

def test_float_proportion_reconstructs_every_count_exactly_at_n_1e7() -> None:
    """k = int(round(p * n)) with p = fl(k / n), for EVERY k in [0, 1e7]."""
    n = 10_000_000
    k = np.arange(n + 1, dtype=np.int64)
    p = k / n
    assert np.array_equal(np.rint(p * n).astype(np.int64), k)


@pytest.mark.parametrize("k", [0, 1, 3_333_333, 4_999_999, 5_000_001, 9_999_999, 10_000_000])
def test_engine_mean_path_yields_correctly_rounded_proportion_at_n_1e7(k) -> None:
    """The engine's stored proportion comes from np.mean(uint8, axis=0,
    dtype=float64). With integer partial sums < 2**53 that is fl(k/n), so the
    reconstruction above applies. JSON round-trip preserves it exactly."""
    n = 10_000_000
    x = np.zeros((n, 1), dtype=np.uint8)
    x[:k, 0] = 1
    p = d3.binary_marginal_distribution(x)["per_feature_one_probability"][0]
    p = json.loads(json.dumps(p))
    assert p == k / n
    assert int(round(p * n)) == k


def test_confirmatory_families_refuse_float_counts() -> None:
    with pytest.raises(TypeError):
        d3.binomial_reference_family(
            [{"label": "b", "successes": 0.5 * 100, "trials": 100, "expected_probability": 0.5}],
            family_id="R", description="", familywise_alpha=0.01, tolerance=0.1)


# ===========================================================================
# §1.6  Decision-path tracing
# ===========================================================================

def _legacy(n: int, rng) -> dict:
    x = [_bits(rng, n, 8) for _ in range(3)]
    y = [(np.arange(n) % 2).astype(np.uint8) for _ in range(3)]
    return d3.audit_distribution(x[0], y[0], x[1], y[1], x[2], y[2],
                                 intended_distribution={"expected_class_ratio": 0.5,
                                                        "expected_bit_probability": 0.5,
                                                        "class_ratio_tolerance": 0.005,
                                                        "bit_probability_tolerance": 0.2,
                                                        "pairwise_tvd_tolerance": 0.3},
                                 familywise_alpha=0.01)


def test_partition_level_first_order_differences_are_descriptive_and_not_read() -> None:
    res = _legacy(400, np.random.default_rng(16100))
    before = d3.evaluate_decision(res)
    for comp in res["partition_comparisons"].values():
        comp["positive_ratio_difference"] = 0.9
        comp["binary_marginal_difference"]["maximum_absolute_difference"] = 0.9
    assert d3.evaluate_decision(res) == before


def test_legacy_pass_carries_no_precision_while_confirmatory_path_is_inconclusive() -> None:
    """Demonstrated defect in the legacy semantics: with exactly balanced
    labels at n = 400, the legacy rule records the class family as
    practically consistent and the outcome can be PASS, although the design
    cannot resolve a 0.005 tolerance. The confirmatory path says so."""
    res = _legacy(400, np.random.default_rng(16200))
    assert res["intended_distribution"]["train"]["class_distribution"]["practically_consistent"]
    fam = d3.binomial_reference_family(
        [{"label": "train", "successes": 200, "trials": 400, "expected_probability": 0.5}],
        family_id="R1", description="", familywise_alpha=0.01, tolerance=0.005)
    assert fam["records"]["category"] == ["INSUFFICIENT_PRECISION"]
    res_ = d3.design_resolution(400, per_hypothesis_alpha=0.01, cell_probabilities=[0.5])
    assert res_["minimum_certifiable_tolerance"] > 0.005


# ===========================================================================
# §1.7  Four-way decision coverage
# ===========================================================================

def test_pairwise_family_reaches_pass_and_fail_end_to_end() -> None:
    """End-to-end through the second-order family: identical partitions PASS
    when the tolerance is resolvable; a strong injected pair shift FAILs."""
    rng = np.random.default_rng(17600)
    x = _bits(rng, 60_000, 8)
    fam = d3.pairwise_homogeneity_family({"a_b": (x, x.copy())}, family_id="A", description="",
                                         familywise_alpha=0.01, tvd_tolerance=0.05, mc_seed=0)
    assert fam["outcome"] == "PASS"
    y = x.copy()
    y[: 30_000, 5] = y[: 30_000, 2]
    fam = d3.pairwise_homogeneity_family({"a_b": (x, y)}, family_id="A", description="",
                                         familywise_alpha=0.01, tvd_tolerance=0.05, mc_seed=0)
    assert fam["outcome"] == "FAIL"


# ===========================================================================
# Workstream 4  Strict input validation and Monte Carlo record-keeping
# ===========================================================================

def _item(**over):
    it = {"label": "x", "successes": 50, "trials": 100, "expected_probability": 0.5}
    it.update(over)
    return it


@pytest.mark.parametrize("over", [
    dict(successes=50.0), dict(successes=True), dict(successes=np.float64(50)), dict(successes=-1),
    dict(successes=101), dict(trials=0), dict(trials=100.0), dict(expected_probability=0.0),
    dict(expected_probability=1.0), dict(expected_probability=float("nan")), dict(tolerance=0.1),
])
def test_binomial_family_refuses_invalid_items(over) -> None:
    with pytest.raises((TypeError, ValueError)):
        d3.binomial_reference_family([_item(**over)], family_id="R", description="",
                                     familywise_alpha=0.01, tolerance=0.1)


@pytest.mark.parametrize("tol", [0.0, -0.01, 1.5, float("nan"), float("inf"), True, "0.1"])
def test_invalid_tolerances_are_refused(tol) -> None:
    with pytest.raises((TypeError, ValueError)):
        d3.binomial_reference_family([_item()], family_id="R", description="", familywise_alpha=0.01,
                                     tolerance=tol)


@pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan"), True, None])
def test_invalid_family_alpha_is_refused(alpha) -> None:
    with pytest.raises((TypeError, ValueError)):
        d3.binomial_reference_family([_item()], family_id="R", description="", familywise_alpha=alpha,
                                     tolerance=0.1)


@pytest.mark.parametrize("over", [dict(k1=10.5), dict(k1=True), dict(k1=-1), dict(k1=200),
                                  dict(n2=0), dict(tolerance=0.1)])
def test_two_sample_family_refuses_invalid_items(over) -> None:
    it = {"label": "x", "k1": 10, "n1": 100, "k2": 12, "n2": 100}
    it.update(over)
    with pytest.raises((TypeError, ValueError)):
        d3.two_sample_proportion_family([it], family_id="D", description="", familywise_alpha=0.01,
                                        tolerance=0.1, mc_seed=0)


@pytest.mark.parametrize("x", [np.array([[0, 2], [1, 0]], dtype=np.uint8),
                               np.array([[0, 1], [1, 0]], dtype=np.int64),
                               np.array([[0.0, 1.0], [1.0, 0.0]]),
                               np.zeros((0, 2), dtype=np.uint8), np.zeros(3, dtype=np.uint8)])
def test_confirmatory_binary_matrix_refuses_invalid_features(x) -> None:
    with pytest.raises((TypeError, ValueError)):
        d3._binary_matrix(x)


def _strat_parts(seed=1, n=600):
    rng = np.random.default_rng(seed)
    return {p: ((rng.random((n, 6)) < 0.5).astype(np.uint8), rng.integers(0, 2, n).astype(np.uint8))
            for p in ("train", "validation", "test")}


STRAT_ALLOC = {f: 0.01 / 6 for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
STRAT_TOL = {k: 0.2 for k in d3.TOLERANCE_KEYS}


def test_class_stratified_audit_carries_exact_counts_and_declared_tolerances() -> None:
    parts = _strat_parts()
    r = d3.class_stratified_audit(parts, expected_class_ratio=0.5, expected_bit_probability={0: 0.5, 1: 0.5},
                                  tolerances=STRAT_TOL, family_alpha_allocation=STRAT_ALLOC, mc_seed=3).payload()
    fams = {f["family_id"]: f for f in r["families"]}
    assert set(fams) == set(d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES)
    assert fams["R1"]["records"]["successes"] == [int(parts[p][1].sum()) for p in parts]
    y = parts["test"][1]
    assert fams["D"]["records"]["k1"][12:18] == parts["test"][0][y == 1].sum(axis=0).tolist()
    assert r["design"]["partition_sizes"] == {p: 600 for p in parts}
    assert r["input_provenance"]["test"]["labels_sha256"] == _array_sha256(parts["test"][1])
    for fid, f in fams.items():
        assert f["tolerance_key"] == d3.FAMILY_TOLERANCE_KEY[fid] and f["tolerance"] == 0.2
        assert math.isclose(f["familywise_alpha"], STRAT_ALLOC[fid])


@pytest.mark.parametrize("kw", [
    dict(family_alpha_allocation={"R1": 0.001}), dict(tolerances={"class_ratio": 0.1}),
    dict(tolerances={**STRAT_TOL, "invented": 0.1}),
    dict(expected_class_ratio=1.0), dict(expected_bit_probability={1: 0.5}), dict(mc_seed=-1),
    dict(mc_seed=1.5),
])
def test_class_stratified_audit_refuses_incomplete_or_invalid_configuration(kw) -> None:
    args = dict(expected_class_ratio=0.5, expected_bit_probability={0: 0.5, 1: 0.5},
                tolerances=STRAT_TOL, family_alpha_allocation=STRAT_ALLOC, mc_seed=3)
    args.update(kw)
    with pytest.raises((TypeError, ValueError)):
        d3.class_stratified_audit(_strat_parts(), **args)


def test_class_stratified_audit_refuses_single_class_partition() -> None:
    parts = _strat_parts()
    parts["test"] = (parts["test"][0], np.ones(600, dtype=np.uint8))
    with pytest.raises(ValueError):
        d3.class_stratified_audit(parts, expected_class_ratio=0.5, expected_bit_probability={0: 0.5, 1: 0.5},
                                  tolerances=STRAT_TOL, family_alpha_allocation=STRAT_ALLOC, mc_seed=3)


def test_allocation_refuses_invalid_inputs() -> None:
    for g, ids in ((0.0, ["A"]), (0.01, []), (0.01, ["A", "A"]), (float("nan"), ["A"])):
        with pytest.raises((TypeError, ValueError)):
            d3.allocate_familywise_alpha(g, ids)


def test_monte_carlo_records_exceedances_uncertainty_and_is_seed_deterministic() -> None:
    a, b = np.array([[1, 0, 0, 3]]), np.array([[0, 3, 1, 0]])
    t1 = d3.homogeneity_test_batch(a, b, per_hypothesis_alpha=0.05, mc_seed=7, mc_stream=(1,))
    t2 = d3.homogeneity_test_batch(a, b, per_hypothesis_alpha=0.05, mc_seed=7, mc_stream=(1,))
    t3 = d3.homogeneity_test_batch(a, b, per_hypothesis_alpha=0.05, mc_seed=7, mc_stream=(2,))
    assert t1["method"][0] == "monte_carlo" and t1["monte_carlo_draws"][0] == d3.MC_MIN_DRAWS
    assert t1["p_value"][0] == (1 + t1["monte_carlo_exceedances"][0]) / (d3.MC_MIN_DRAWS + 1)
    assert t1["p_value"][0] == t2["p_value"][0]
    for t, stream in ((t1, 1), (t3, 2)):
        ref = d3.monte_carlo_homogeneity_p_value(a[0], b[0], draws=d3.MC_MIN_DRAWS,
                                                 rng=np.random.default_rng([7, stream, 0]))
        assert t["monte_carlo_exceedances"][0] == ref["exceedances"]
    mc = d3.monte_carlo_homogeneity_p_value(a[0], b[0], draws=5000, rng=np.random.default_rng(0))
    lo, hi = mc["tail_probability_ci99"]
    assert lo <= mc["exceedances"] / 5000 <= hi


def test_homogeneity_refuses_float_or_negative_tables() -> None:
    for a in (np.array([[1.0, 2.0]]), np.array([[-1, 2]]), np.array([[True, False]])):
        with pytest.raises((TypeError, ValueError)):
            d3.homogeneity_test_batch(a, np.array([[1, 2]]), per_hypothesis_alpha=0.05, mc_seed=0)


# ===========================================================================
# Decision gates (defect regressions for the 2026-10-22 gate review)
# ===========================================================================
#
# Every confirmatory decision is evaluated through the real sealed path:
# class_stratified_audit -> AuditResult; verify_calibration_evidence ->
# CalibrationBinding; verify_preregistration -> PreregistrationBinding.
# FIXTURES: calibration evidence is fabricated in tmp_path to exercise the
# gate (the binding checks completeness and identity, not authenticity), and
# preregistrations are committed in a throwaway git repository in tmp_path.
# Input provenance is recomputed in the test, independently of the engine.

import copy as _copy
import datetime as _dt
import subprocess as _sp

from audit.dataset.common.provenance import array_sha256 as _array_sha256

TOOL_PATH = Path(d3.__file__).parent / "tools" / "d3_method_validity_calibration.py"
SUITE_PATH = Path(__file__)
# The global alpha (0.01) is joint over G1 and G2, so the allocation sums to 0.01 / GUARANTEE_COUNT.
GATE_ALLOC = {f: 0.01 / d3.GUARANTEE_COUNT / 6 for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
FULL_SPLIT_ALLOC = {f: 0.01 / 6 for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}   # pre-joint design
ROLES = {f: "confirmatory" for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
WIDE_TOL = {"class_ratio": 0.1, "class_conditional_bit": 0.15, "pairwise_tvd": 0.3, "cross_class_bit": 0.2}


def _gate_parts(seed=31, n=3000, d=4, test_ratio=0.5):
    """Deterministic partitions; exact label counts; independent uniform bits."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("train", "validation", "test"):
        k = int(round((test_ratio if name == "test" else 0.5) * n))
        y = np.zeros(n, dtype=np.uint8)
        y[:k] = 1
        rng.shuffle(y)
        out[name] = ((rng.random((n, d)) < 0.5).astype(np.uint8), y)
    return out


def _provenance(parts):
    return {n: {"features_sha256": _array_sha256(np.ascontiguousarray(x)),
                "labels_sha256": _array_sha256(np.ascontiguousarray(y))} for n, (x, y) in parts.items()}


def _git_repo(tmp_path):
    repo = tmp_path / "prereg_repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "approver@example.org"],
                 ["config", "user.name", "Fixture Approver"]):
        _sp.run(["git", *args], cwd=repo, check=True)
    return repo


def _prereg_record(**over):
    rec = {"schema_version": d3.PREREGISTRATION_SCHEMA_VERSION, "approved": True,
           "approved_by": "Fixture Approver", "approval_statement": d3.APPROVAL_STATEMENT,
           "approved_utc": (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=1)).isoformat(),
           "data_examined_before_approval": "none (test fixture)",
           "global_familywise_alpha": 0.01, "family_alpha_allocation": dict(GATE_ALLOC),
           "family_roles": dict(ROLES), "tolerances": dict(WIDE_TOL),
           "tolerance_justification": {k: "test fixture justification text, not a scientific claim"
                                       for k in WIDE_TOL}}
    rec.update(over)
    return rec


def _commit_prereg(repo, rec, name="prereg.json", commit=True):
    path = repo / name
    data = json.dumps(rec, indent=1, sort_keys=True).encode()
    path.write_bytes(data)
    if commit:
        _sp.run(["git", "add", name], cwd=repo, check=True)
        _sp.run(["git", "commit", "-q", "-m", "preregister D3"], cwd=repo, check=True)
    return path, hashlib.sha256(data).hexdigest()


def _evidence(audit_payload, *, supported=(0.45, 0.55), mutate=None, manifest_mutate=None):
    des = audit_payload["design"]
    alloc = audit_payload["configuration"]["family_alpha_allocation"]
    used = d3._alphas_used(alloc, d3.expected_hypothesis_counts(len(des["partition_sizes"]), des["feature_dimension"]))
    suite_sha = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
    ev = {"schema_version": d3.CALIBRATION_SCHEMA_VERSION, "status": d3.CALIBRATION_STATUS, "run_id": "fixture-run",
          "checks": {c: True for c in d3.REQUIRED_CALIBRATION_CHECKS},
          "identity": {"engine_sha256": d3.engine_sha256(),
                       "tool_sha256": hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest(),
                       "test_suite_sha256": suite_sha, "method_version": d3.D3_METHOD_VERSION,
                       "decision_rule_version": d3.DECISION_RULE_VERSION,
                       "bound_construction": d3.BOUND_CONSTRUCTION, "detection_test": d3.DETECTION_TEST},
          "design": {"global_alpha": 0.01, "family_alpha_allocation": dict(alloc),
                     "partition_sizes": dict(des["partition_sizes"]),
                     "feature_dimension": des["feature_dimension"], "supported_class_fraction": list(supported)},
          "guarantee": {"basis": "fixture"},
          "S0_unit_suite": {"exit_code": 0, "suite_sha256": suite_sha},
          "S1_single_test_size": {}, "S2_complete_path_null": {},
          "S6_exact_clopper_pearson": {"rows": [{"alpha": a, "valid": True} for a in used["clopper_pearson"]]},
          "S7_bound_coverage": {"results": [{"valid": True}]},
          "S8_exact_fisher": {"rows": [{"alpha": a, "valid": True} for a in used["fisher_cell"]]}}
    if mutate:
        mutate(ev)
    return ev


def _write_cal(dir_, ev, manifest_mutate=None):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / d3.CALIBRATION_EVIDENCE_NAME).write_text(json.dumps(ev))
    man = {"run_id": ev.get("run_id"), "run_status": "COMPLETE",
           "files": {d3.CALIBRATION_EVIDENCE_NAME: hashlib.sha256((dir_ / d3.CALIBRATION_EVIDENCE_NAME).read_bytes()).hexdigest()}}
    if manifest_mutate:
        manifest_mutate(man)
    (dir_ / d3.CALIBRATION_MANIFEST_NAME).write_text(json.dumps(man))
    return dir_


def _bind(dir_):
    return d3.verify_calibration_evidence(dir_, tool_path=TOOL_PATH, test_suite_path=SUITE_PATH)


def _audit(parts, tol=WIDE_TOL, alloc=GATE_ALLOC):
    return d3.class_stratified_audit(parts, expected_class_ratio=0.5, expected_bit_probability={0: 0.5, 1: 0.5},
                                     tolerances=tol, family_alpha_allocation=alloc, mc_seed=7)


def _full_gate(tmp_path, parts, *, tol=WIDE_TOL, supported=(0.45, 0.55)):
    """Preregister (commit) -> audit -> calibration evidence -> bindings."""
    repo = _git_repo(tmp_path)
    path, sha = _commit_prereg(repo, _prereg_record(tolerances=dict(tol)))
    pr = d3.verify_preregistration(path, expected_sha256=sha)
    audit = _audit(parts, tol=tol)
    cal = _bind(_write_cal(tmp_path / "cal", _evidence(audit.payload(), supported=supported)))
    return audit, cal, pr, _provenance(parts), (repo, path, sha)


def _decide(audit, cal, pr, prov, **kw):
    return d3.evaluate_confirmatory_decision(audit, calibration=cal, preregistration=pr,
                                             expected_input_provenance=prov, **kw)


# --- four-way outcomes through the complete gated path ----------------------

@pytest.mark.parametrize("test_ratio,tol,supported,expected", [
    (0.5, WIDE_TOL, (0.45, 0.55), "PASS"),
    (0.56, WIDE_TOL, (0.05, 0.95), "CONDITIONAL_PASS"),     # R1 detected, bounded inside 0.1
    (0.8, WIDE_TOL, (0.05, 0.95), "FAIL"),                  # R1 lower bound > 0.1
    (0.5, {k: 0.001 for k in WIDE_TOL}, (0.45, 0.55), "INCONCLUSIVE"),   # insufficient precision
])
def test_four_way_outcomes_through_the_complete_gated_path(tmp_path, test_ratio, tol, supported, expected) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts(test_ratio=test_ratio), tol=tol, supported=supported)
    d = _decide(audit, cal, pr, prov)
    assert d["outcome_overridden_because"] == [] and d["outcome"] == expected, d["outcome_overridden_because"]


# --- Defect 1: preregistration integrity ---------------------------------------

def test_defect1_reproduction_unapproved_dict_with_fabricated_hash_cannot_pass(tmp_path) -> None:
    audit, cal, _, prov, _ = _full_gate(tmp_path, _gate_parts())
    fake = {"record_sha256": "f" * 64, "approved": False, "global_familywise_alpha": 0.01,
            "family_alpha_allocation": GATE_ALLOC, "family_roles": ROLES, "tolerances": WIDE_TOL}
    d = _decide(audit, cal, fake, prov)
    assert d["outcome"] == "INCONCLUSIVE" and d["provisional_outcome"] == "PASS"


@pytest.mark.parametrize("over", [
    dict(approved=False), dict(approved="true"), dict(approved=1), dict(approval_statement="I approve."),
    dict(approved_by=""), dict(data_examined_before_approval=""), dict(schema_version="d3-preregistration-1"),
    dict(approved_utc="yesterday"), dict(approved_utc="2026-10-22T00:00:00"), dict(approved_utc="2999-01-01T00:00:00+00:00"),
    dict(global_familywise_alpha=0.001), dict(family_roles={**ROLES, "A": "maybe"}),
    dict(tolerances={**WIDE_TOL, "pairwise_tvd": -0.1}), dict(tolerance_justification={k: "ok" for k in WIDE_TOL}),
    dict(record_sha256="f" * 64),
    dict(data_examined_before_approval=f"manifest examined; Gohr results: {d3.PENDING_MARKER}"),
    dict(family_alpha_allocation=dict(FULL_SPLIT_ALLOC)),
])
def test_unapproved_or_malformed_preregistration_is_refused(tmp_path, over) -> None:
    path, sha = _commit_prereg(_git_repo(tmp_path), _prereg_record(**over))
    with pytest.raises(d3.PreregistrationRejected):
        d3.verify_preregistration(path, expected_sha256=sha)


def test_preregistration_missing_required_field_is_refused(tmp_path) -> None:
    rec = _prereg_record()
    rec.pop("data_examined_before_approval")
    path, sha = _commit_prereg(_git_repo(tmp_path), rec)
    with pytest.raises(d3.PreregistrationRejected):
        d3.verify_preregistration(path, expected_sha256=sha)


def test_forged_or_stale_anchor_is_refused(tmp_path) -> None:
    path, sha = _commit_prereg(_git_repo(tmp_path), _prereg_record())
    for bad in ("0" * 64, sha[:-1] + ("0" if sha[-1] != "0" else "1"), "short"):
        with pytest.raises(d3.PreregistrationRejected):
            d3.verify_preregistration(path, expected_sha256=bad)


def test_uncommitted_or_untracked_preregistration_is_refused(tmp_path) -> None:
    repo = _git_repo(tmp_path)
    path, sha = _commit_prereg(repo, _prereg_record(), name="untracked.json", commit=False)
    with pytest.raises(d3.PreregistrationRejected):
        d3.verify_preregistration(path, expected_sha256=sha)
    path, sha = _commit_prereg(repo, _prereg_record(), name="tracked.json")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(d3.PreregistrationRejected):
        d3.verify_preregistration(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_preregistration_modified_after_binding_blocks_the_decision(tmp_path) -> None:
    audit, cal, pr, prov, (repo, path, sha) = _full_gate(tmp_path, _gate_parts())
    assert _decide(audit, cal, pr, prov)["outcome"] == "PASS"
    rec = json.loads(path.read_text())
    rec["tolerances"]["pairwise_tvd"] = 0.9
    path.write_text(json.dumps(rec))
    d = _decide(audit, cal, pr, prov)
    assert d["outcome"] == "INCONCLUSIVE" and any("no longer verifies" in r for r in d["outcome_overridden_because"])


def test_altered_or_hand_built_preregistration_binding_is_refused(tmp_path) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts())
    altered = _copy.copy(pr)
    object.__setattr__(altered, "record_bytes", pr.record_bytes.replace(b"0.3", b"0.9"))
    assert _decide(audit, cal, altered, prov)["outcome"] == "INCONCLUSIVE"
    forged = d3.PreregistrationBinding(path=pr.path, record_bytes=pr.record_bytes, record_sha256=pr.record_sha256,
                                       commit=pr.commit, commit_author=pr.commit_author, commit_utc=pr.commit_utc)
    assert _decide(audit, cal, forged, prov)["outcome"] == "INCONCLUSIVE"


def test_preregistration_committed_after_the_audit_is_refused(tmp_path) -> None:
    parts = _gate_parts()
    audit = _audit(parts)
    repo = _git_repo(tmp_path)
    path, sha = _commit_prereg(repo, _prereg_record(approved_utc=_dt.datetime.now(_dt.timezone.utc).isoformat()))
    pr = d3.verify_preregistration(path, expected_sha256=sha)
    cal = _bind(_write_cal(tmp_path / "cal", _evidence(audit.payload())))
    d = _decide(audit, cal, pr, _provenance(parts))
    assert d["outcome"] == "INCONCLUSIVE"
    assert any("before the audit" in r for r in d["outcome_overridden_because"])


def test_preregistered_values_must_equal_the_audit_configuration(tmp_path) -> None:
    parts = _gate_parts()
    repo = _git_repo(tmp_path)
    for i, over in enumerate((dict(tolerances={**WIDE_TOL, "class_ratio": 0.2}),
                              dict(family_alpha_allocation={**GATE_ALLOC, "A": 0.0005}),
                              dict(family_roles={**ROLES, "E": "exploratory"}))):
        path, sha = _commit_prereg(repo, _prereg_record(**over), name=f"p{i}.json")
        pr = d3.verify_preregistration(path, expected_sha256=sha)
        audit = _audit(parts)
        cal = _bind(_write_cal(tmp_path / f"cal{i}", _evidence(audit.payload())))
        assert _decide(audit, cal, pr, _provenance(parts))["outcome"] == "INCONCLUSIVE"


# --- Defect 2: calibration completeness ------------------------------------------

@pytest.fixture(scope="module")
def gate_payload():
    return _audit(_gate_parts()).payload()


@pytest.mark.parametrize("missing", sorted(d3.REQUIRED_CALIBRATION_CHECKS))
def test_every_omitted_required_check_is_rejected(tmp_path, gate_payload, missing) -> None:
    ev = _evidence(gate_payload, mutate=lambda e: e["checks"].pop(missing))
    with pytest.raises(d3.CalibrationRejected, match="missing"):
        _bind(_write_cal(tmp_path / "c", ev))


@pytest.mark.parametrize("missing", sorted(d3.REQUIRED_CALIBRATION_SECTIONS - {"schema_version", "run_id"}))
def test_every_omitted_required_section_is_rejected(tmp_path, gate_payload, missing) -> None:
    ev = _evidence(gate_payload, mutate=lambda e: e.pop(missing))
    with pytest.raises(d3.CalibrationRejected):
        _bind(_write_cal(tmp_path / "c", ev))


@pytest.mark.parametrize("mutate", [
    lambda e: e["checks"].__setitem__("S9_invented", True),
    lambda e: e["checks"].__setitem__("S7_bound_coverage", 1),
    lambda e: e["checks"].__setitem__("S7_bound_coverage", "true"),
    lambda e: e["checks"].__setitem__("S7_bound_coverage", False),
    lambda e: e.__setitem__("checks", {}),
    lambda e: e.__setitem__("checks", ["S0_unit_suite"]),
    lambda e: e.__setitem__("unexpected_section", {}),
    lambda e: e.__setitem__("status", "VALIDATED"),
    lambda e: e.__setitem__("schema_version", "d3-method-validity-2"),
    lambda e: e["S0_unit_suite"].__setitem__("exit_code", 1),
    lambda e: e["S0_unit_suite"].__setitem__("suite_sha256", "0" * 64),
    lambda e: e["S6_exact_clopper_pearson"]["rows"].pop(),
    lambda e: e["S6_exact_clopper_pearson"]["rows"][0].__setitem__("valid", False),
    lambda e: e["S8_exact_fisher"].__setitem__("rows", []),
    lambda e: e["S7_bound_coverage"]["results"][0].__setitem__("valid", False),
    lambda e: e["identity"].__setitem__("engine_sha256", "0" * 64),
    lambda e: e["identity"].__setitem__("tool_sha256", "0" * 64),
    lambda e: e["identity"].__setitem__("detection_test", "chi-square"),
    lambda e: e["design"].__setitem__("partition_sizes", {"train": 3000.0, "validation": 3000, "test": 3000}),
    lambda e: e["design"].__setitem__("family_alpha_allocation", {"R1": 0.01}),
    lambda e: e["design"].__setitem__("supported_class_fraction", [0.6, 0.7]),
    lambda e: e["design"].pop("feature_dimension"),
])
def test_incomplete_malformed_or_stale_calibration_is_rejected(tmp_path, gate_payload, mutate) -> None:
    with pytest.raises(d3.CalibrationRejected):
        _bind(_write_cal(tmp_path / "c", _evidence(gate_payload, mutate=mutate)))


@pytest.mark.parametrize("manifest_mutate", [
    lambda m: m.__setitem__("run_status", "STAGING"),
    lambda m: m["files"].__setitem__(d3.CALIBRATION_EVIDENCE_NAME, "0" * 64),
    lambda m: m.__setitem__("run_id", "other"),
])
def test_inconsistent_calibration_manifest_is_rejected(tmp_path, gate_payload, manifest_mutate) -> None:
    with pytest.raises(d3.CalibrationRejected):
        _bind(_write_cal(tmp_path / "c", _evidence(gate_payload), manifest_mutate=manifest_mutate))


def test_defect2_reproduction_real_prior_evidence_without_s7_is_rejected(tmp_path) -> None:
    src = Path(d3.__file__).parent / "evidence" / "d3_method_validity_20261022_6d81cf20"
    if not src.is_dir():
        pytest.skip("prior calibration evidence not present")
    ev = json.loads((src / d3.CALIBRATION_EVIDENCE_NAME).read_text())
    ev["checks"].pop("S7_bound_coverage")
    with pytest.raises(d3.CalibrationRejected):
        _bind(_write_cal(tmp_path / "c", ev))


def test_missing_or_unreadable_calibration_is_rejected(tmp_path) -> None:
    with pytest.raises(d3.CalibrationRejected):
        _bind(tmp_path / "absent")
    d = tmp_path / "bad"
    d.mkdir()
    (d / d3.CALIBRATION_EVIDENCE_NAME).write_text("{not json")
    (d / d3.CALIBRATION_MANIFEST_NAME).write_text("{}")
    with pytest.raises(d3.CalibrationRejected):
        _bind(d)


def test_calibration_binding_must_be_verified_not_hand_built(tmp_path) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts())
    forged = d3.CalibrationBinding(run_id=cal.run_id, evidence_sha256=cal.evidence_sha256,
                                   manifest_sha256=cal.manifest_sha256, engine_sha256=cal.engine_sha256,
                                   tool_sha256=cal.tool_sha256, test_suite_sha256=cal.test_suite_sha256,
                                   design_json=cal.design_json)
    for bad in (None, {"status": "VALIDATED"}, "VALIDATED", forged):
        d = _decide(audit, bad, pr, prov)
        assert d["outcome"] == "INCONCLUSIVE" and d["provisional_outcome"] == "PASS"


def test_calibration_goes_stale_when_the_engine_changes(tmp_path, monkeypatch) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts())
    monkeypatch.setattr(d3, "engine_sha256", lambda: "0" * 64)
    assert _decide(audit, cal, pr, prov)["outcome"] == "INCONCLUSIVE"


# --- Defect 3: the decision is bound to the audited input --------------------------

def test_defect3_reproduction_calibrated_design_cannot_be_substituted(tmp_path) -> None:
    """Audit 3,000-row partitions while the calibration covers 10^7 / 10^6 / 10^6."""
    parts = _gate_parts()
    audit, _, pr, prov, _ = _full_gate(tmp_path, parts)
    prod = _evidence(audit.payload(), mutate=lambda e: e["design"].__setitem__(
        "partition_sizes", {"train": 10_000_000, "validation": 1_000_000, "test": 1_000_000}))
    cal = _bind(_write_cal(tmp_path / "prod", prod))
    d = _decide(audit, cal, pr, prov)
    assert d["outcome"] == "INCONCLUSIVE"
    assert any("partition sizes" in r for r in d["outcome_overridden_because"])


@pytest.mark.parametrize("mutate,why", [
    (lambda e: e["design"].__setitem__("feature_dimension", 64), "feature dimension"),
    (lambda e: e["design"].__setitem__("supported_class_fraction", [0.499, 0.501]), "class fraction"),
])
def test_unsupported_dimension_or_class_strata_are_not_calibrated(tmp_path, mutate, why) -> None:
    parts = _gate_parts(test_ratio=0.4)
    audit, _, pr, prov, _ = _full_gate(tmp_path, parts, supported=(0.05, 0.95))
    ev = _evidence(audit.payload(), mutate=mutate)
    if why == "feature dimension":
        ev["S6_exact_clopper_pearson"]["rows"] += [{"alpha": a, "valid": True} for a in d3._alphas_used(
            ev["design"]["family_alpha_allocation"], d3.expected_hypothesis_counts(3, 64))["clopper_pearson"]]
        ev["S8_exact_fisher"]["rows"] += [{"alpha": a, "valid": True} for a in d3._alphas_used(
            ev["design"]["family_alpha_allocation"], d3.expected_hypothesis_counts(3, 64))["fisher_cell"]]
    cal = _bind(_write_cal(tmp_path / "c", ev))
    d = _decide(audit, cal, pr, prov)
    assert d["outcome"] == "INCONCLUSIVE" and any(why in r for r in d["outcome_overridden_because"])


def test_input_provenance_must_match_the_audited_arrays(tmp_path) -> None:
    parts = _gate_parts()
    audit, cal, pr, prov, _ = _full_gate(tmp_path, parts)
    assert _decide(audit, cal, pr, prov)["outcome"] == "PASS"
    assert _decide(audit, cal, pr, None)["outcome"] == "INCONCLUSIVE"
    other = _provenance(_gate_parts(seed=99))
    assert _decide(audit, cal, pr, other)["outcome"] == "INCONCLUSIVE"
    bad = _copy.deepcopy(prov)
    bad["test"]["labels_sha256"] = "0" * 64
    assert _decide(audit, cal, pr, bad)["outcome"] == "INCONCLUSIVE"


# --- Defect 4: family-result provenance ---------------------------------------------

def _reseal(payload):
    """Simulate an in-process forger who can re-seal: content checks must still catch edits."""
    blob = d3._canonical(payload)
    return d3.AuditResult(sealed_json=blob, seal_sha256=hashlib.sha256(blob).hexdigest(), _token=d3._SEAL_TOKEN)


def _fam(p, fid):
    return next(f for f in p["families"] if f["family_id"] == fid)


TAMPER = {
    "outcome_upgraded": lambda p: _fam(p, "R1").__setitem__("outcome", "FAIL"),
    "category_edited": lambda p: _fam(p, "R2")["records"]["category"].__setitem__(0, "STATISTICAL_ONLY"),
    "significance_flipped_consistently": lambda p: (
        _fam(p, "R1")["records"]["significant"].__setitem__(0, True),
        _fam(p, "R1")["records"]["category"].__setitem__(0, "STATISTICAL_ONLY"),
        _fam(p, "R1").__setitem__("outcome", "CONDITIONAL_PASS")),
    "bound_tightened": lambda p: _fam(p, "D")["records"]["absolute_difference_upper_bound"].__setitem__(0, 0.0),
    "p_value_edited": lambda p: next(iter(_fam(p, "A")["records"]["comparisons"].values()))["pairs"]["p_value"].__setitem__(0, 1e-12),
    "counts_edited": lambda p: _fam(p, "R1")["records"]["successes"].__setitem__(0, 1),
    "family_omitted": lambda p: p.__setitem__("families", [f for f in p["families"] if f["family_id"] != "C"]),
    "family_duplicated": lambda p: p["families"].append(_copy.deepcopy(_fam(p, "R1"))),
    "family_fabricated": lambda p: p["families"].append(dict(_copy.deepcopy(_fam(p, "R1")), family_id="Z")),
    "role_downgraded": lambda p: _fam(p, "B").__setitem__("role", "exploratory"),
    "alpha_edited": lambda p: _fam(p, "A").__setitem__("familywise_alpha", 0.5),
    "hypothesis_count_edited": lambda p: _fam(p, "D").__setitem__("hypothesis_count", 1),
    "tolerance_edited": lambda p: _fam(p, "A").__setitem__("tolerance", 0.9),
    "design_sizes_edited": lambda p: p["design"]["partition_sizes"].__setitem__("train", 10_000_000),
    "class_counts_edited": lambda p: p["design"]["class_counts"]["test"].__setitem__("positive", 1),
    "method_version_edited": lambda p: p.__setitem__("method_version", "d3-confirmatory-3"),
    "recorded_role_flipped": lambda p: p["configuration"]["family_roles"].__setitem__("R1", "exploratory"),
    "recorded_roles_removed": lambda p: p["configuration"].pop("family_roles"),
    "family_and_recorded_role_downgraded": lambda p: (
        p["configuration"]["family_roles"].__setitem__("B", "exploratory"),
        _fam(p, "B").__setitem__("role", "exploratory")),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_tampered_family_results_are_rejected_even_if_resealed(gate_payload, name) -> None:
    p = _copy.deepcopy(gate_payload)
    TAMPER[name](p)
    with pytest.raises(d3.AuditIntegrityError):
        d3.verify_audit_result(_reseal(p))


def test_genuine_resealed_payload_still_verifies(gate_payload) -> None:
    """Control for the tamper tests: re-sealing alone is not what they detect."""
    d3.verify_audit_result(_reseal(_copy.deepcopy(gate_payload)))


def test_unsealed_or_seal_mismatched_results_are_rejected(gate_payload) -> None:
    for bad in (gate_payload, [gate_payload], None):
        with pytest.raises(d3.AuditIntegrityError):
            d3.verify_audit_result(bad)
    a = _audit(_gate_parts())
    broken = _copy.copy(a)
    object.__setattr__(broken, "sealed_json", a.sealed_json.replace(b'"PASS"', b'"FAIL"', 1))
    with pytest.raises(d3.AuditIntegrityError):
        d3.verify_audit_result(broken)
    with pytest.raises(d3.AuditIntegrityError):
        d3.verify_audit_result(d3.AuditResult(sealed_json=a.sealed_json, seal_sha256=a.seal_sha256))


def test_defect4_reproduction_edited_fail_family_cannot_become_pass(tmp_path) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts(test_ratio=0.8), supported=(0.05, 0.95))
    assert _decide(audit, cal, pr, prov)["outcome"] == "FAIL"
    p = audit.payload()
    _fam(p, "R1")["outcome"] = "PASS"
    with pytest.raises(d3.AuditIntegrityError):
        d3.evaluate_confirmatory_decision(_reseal(p), calibration=cal, preregistration=pr,
                                          expected_input_provenance=prov)


def test_editing_a_payload_copy_does_not_affect_the_decision(tmp_path) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts(test_ratio=0.8), supported=(0.05, 0.95))
    p = audit.payload()
    for f in p["families"]:
        f["outcome"] = "PASS"
    assert _decide(audit, cal, pr, prov)["outcome"] == "FAIL"


# --- Defect 5: decisions rest only on exact methods -------------------------------

def test_diagnostic_chi_square_values_never_change_a_decision(gate_payload) -> None:
    p = _copy.deepcopy(gate_payload)
    for f in p["families"]:
        if f["hypothesis_kind"] == "pairwise_homogeneity":
            for c in f["records"]["comparisons"].values():
                c["pairs"]["diagnostic_chi_square_p_value"] = [0.0] * len(c["pairs"]["diagnostic_chi_square_p_value"])
        elif f["hypothesis_kind"] == "two_sample_proportion":
            f["records"]["diagnostic_chi_square_p_value"] = [0.0] * f["hypothesis_count"]
    q = d3.verify_audit_result(_reseal(p))
    assert [f["outcome"] for f in q["families"]] == [f["outcome"] for f in gate_payload["families"]]


def _exact_fisher_conditional_size(n_a, n_b, t, alpha):
    """Exact P(p < alpha) of the doubled one-sided hypergeometric test, given margins."""
    from scipy.stats import hypergeom
    total = n_a + n_b
    lo_x, hi_x = max(0, t - n_b), min(t, n_a)
    mean = t * n_a / total
    sd = math.sqrt(max(t * (n_a / total) * (n_b / total) * (total - t) / max(total - 1, 1), 1e-12))
    xs = np.arange(max(lo_x, int(mean - 40 * sd - 40)), min(hi_x, int(mean + 40 * sd + 40)) + 1)
    pc = d3.exact_cell_pvalues(np.stack([xs, n_a - xs], 1), np.stack([t - xs, n_b - t + xs], 1))[:, 0]
    pmf = hypergeom.pmf(xs, total, t, n_a)
    outside = hypergeom.cdf(xs[0] - 1, total, t, n_a) + hypergeom.sf(xs[-1], total, t, n_a)
    return float(pmf[pc < alpha].sum() + outside)


@pytest.mark.parametrize("alpha", [0.005 / 5 / 6048 / 4, 0.005 / 5 / 192])     # production: 0.001 per family
@pytest.mark.parametrize("n_a,n_b,frac", [(10_000_000, 1_000_000, 0.25), (1_000_000, 1_000_000, 0.5),
                                          (450_000, 550_000, 0.25), (5_000_000, 500_000, 0.5),
                                          (50, 40, 0.3), (1000, 10, 0.01)])
def test_exact_fisher_cell_test_size_at_production_alpha(n_a, n_b, frac, alpha) -> None:
    """The exact cell test's conditional size, computed exactly (no simulation), is <= alpha."""
    t = max(1, int(frac * (n_a + n_b)))
    assert _exact_fisher_conditional_size(n_a, n_b, t, alpha) <= alpha * (1 + 1e-9)


def test_exact_homogeneity_matches_fisher_exact_on_small_tables() -> None:
    from scipy.stats import fisher_exact
    rng = np.random.default_rng(5150)
    for _ in range(200):
        a = rng.integers(0, 30, 2)
        b = rng.integers(0, 30, 2)
        if a.sum() == 0 or b.sum() == 0:
            continue
        p = d3.exact_homogeneity_test_batch(a[None], b[None], per_hypothesis_alpha=0.05)["p_value"][0]
        one = min(fisher_exact([[a[1], a[0]], [b[1], b[0]]], alternative="greater")[1],
                  fisher_exact([[a[1], a[0]], [b[1], b[0]]], alternative="less")[1])
        assert math.isclose(p, min(1.0, 2 * one), rel_tol=1e-9, abs_tol=1e-15)


def test_classification_truth_table() -> None:
    c = d3.classify_hypothesis
    kw = dict(tolerance=0.1)
    assert c(significant=True, effect=0.2, lower_bound=0.15, upper_bound=0.3, **kw) == "CONFIRMED_DEVIATION"
    assert c(significant=False, effect=0.2, lower_bound=0.15, upper_bound=0.3, **kw) == "CONFIRMED_DEVIATION"
    assert c(significant=True, effect=0.2, lower_bound=0.05, upper_bound=0.3, **kw) == "SIGNIFICANT_PRACTICALITY_UNRESOLVED"
    assert c(significant=True, effect=0.05, lower_bound=0.01, upper_bound=0.08, **kw) == "STATISTICAL_ONLY"
    assert c(significant=False, effect=0.2, lower_bound=0.0, upper_bound=0.4, **kw) == "PRACTICAL_UNSUPPORTED"
    assert c(significant=False, effect=0.01, lower_bound=0.0, upper_bound=0.05, **kw) == "EQUIVALENT_WITHIN_TOLERANCE"
    assert c(significant=False, effect=0.01, lower_bound=0.0, upper_bound=0.2, **kw) == "INSUFFICIENT_PRECISION"
    assert c(significant=False, effect=0.0, lower_bound=0.0, upper_bound=0.0, tolerance=None) == "NO_TOLERANCE_DECLARED"
    assert set(d3.CATEGORY_OUTCOME.values()) == {"PASS", "CONDITIONAL_PASS", "INCONCLUSIVE", "FAIL"}
    for bad in (dict(lower_bound=0.3, upper_bound=0.2), dict(lower_bound=float("nan"), upper_bound=0.2),
                dict(lower_bound=-0.1, upper_bound=0.2), dict(lower_bound=0.0, upper_bound=1.5)):
        with pytest.raises(ValueError):
            c(significant=False, effect=0.1, tolerance=0.1, **bad)


def test_fail_requires_lower_bound_above_tolerance_not_point_estimate() -> None:
    fam = d3.binomial_reference_family(
        [{"label": "x", "successes": 101_300, "trials": 200_000, "expected_probability": 0.5}],
        family_id="R1", description="", familywise_alpha=0.005, tolerance=0.005, tolerance_key="class_ratio")
    rec = fam["records"]
    assert rec["significant"][0] and rec["absolute_difference"][0] > 0.005
    assert rec["absolute_difference_lower_bound"][0] <= 0.005
    assert rec["category"] == ["SIGNIFICANT_PRACTICALITY_UNRESOLVED"] and fam["outcome"] == "INCONCLUSIVE"


# ===========================================================================
# Joint alpha: one rule (d3.joint_alpha_problem) at every verification stage
# ===========================================================================

def test_joint_alpha_rule() -> None:
    assert d3.GUARANTEE_COUNT == 2
    assert d3.joint_alpha_problem(0.01, GATE_ALLOC) is None
    assert d3.joint_alpha_problem(0.01, {f: 0.001 for f in ("R2", "A", "B", "C", "D")}) is None
    assert "joint alpha" in d3.joint_alpha_problem(0.01, FULL_SPLIT_ALLOC)
    assert d3.joint_alpha_problem(0.01, {"A": 0.005 * (1 + 1e-9)}) is not None
    assert d3.joint_alpha_problem(0.02, FULL_SPLIT_ALLOC) is None       # the same split under a 0.02 joint alpha


def test_preregistration_with_a_per_guarantee_split_is_refused(tmp_path) -> None:
    path, sha = _commit_prereg(_git_repo(tmp_path), _prereg_record(family_alpha_allocation=dict(FULL_SPLIT_ALLOC)))
    with pytest.raises(d3.PreregistrationRejected, match="joint"):
        d3.verify_preregistration(path, expected_sha256=sha)


def test_calibration_design_with_a_per_guarantee_split_is_rejected(tmp_path) -> None:
    payload = _audit(_gate_parts(), alloc=FULL_SPLIT_ALLOC).payload()
    with pytest.raises(d3.CalibrationRejected, match="joint"):
        _bind(_write_cal(tmp_path / "c", _evidence(payload)))


def test_mismatched_allocation_cannot_pass_the_complete_verification_path(tmp_path) -> None:
    """An audit run at the full per-guarantee split is refused at every stage that sees it."""
    parts = _gate_parts()
    _, cal, pr, prov, _ = _full_gate(tmp_path, parts)          # valid joint prereg and calibration
    audit_full = _audit(parts, alloc=FULL_SPLIT_ALLOC)
    d = _decide(audit_full, cal, pr, prov)
    assert d["outcome"] == "INCONCLUSIVE" and d["provisional_outcome"] == "PASS"
    reasons = " | ".join(d["outcome_overridden_because"])
    assert "calibrated allocation differs" in reasons
    assert "preregistered allocation differs" in reasons
    assert "does not cover the audit jointly" in reasons
    assert d["guarantees"]["joint_error_bound"] == pytest.approx(0.02)


def test_decision_reports_the_joint_error_bound(tmp_path) -> None:
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts())
    d = _decide(audit, cal, pr, prov)
    assert d["outcome"] == "PASS"
    assert d["guarantees"]["alpha_total"] == pytest.approx(0.005)
    assert d["guarantees"]["joint_error_bound"] == pytest.approx(0.01)


# ===========================================================================
# Family roles: exploratory class-stratified families never gate
# ===========================================================================

R1_EXPLORATORY = {**ROLES, "R1": "exploratory"}
FIVE_ALLOC = {f: 0.01 / d3.GUARANTEE_COUNT / 5 for f in ("R2", "A", "B", "C", "D")}


def _roles_audit(parts, roles=R1_EXPLORATORY, alloc=FIVE_ALLOC, e_alpha=0.01, tol=WIDE_TOL):
    return d3.class_stratified_audit(parts, expected_class_ratio=0.5, expected_bit_probability={0: 0.5, 1: 0.5},
                                     tolerances=tol, family_alpha_allocation=alloc, mc_seed=7,
                                     exploratory_alpha=e_alpha, family_roles=roles)


def test_default_roles_are_recorded() -> None:
    p = _audit(_gate_parts()).payload()
    assert p["configuration"]["family_roles"] == ROLES
    assert {f["family_id"]: f["role"] for f in p["families"]} == ROLES


def test_exploratory_r1_is_reported_but_cannot_gate(tmp_path) -> None:
    """R1 would FAIL (test class ratio 0.8); as an exploratory family it is reported, never gating.
    n = 12,000 so the smallest class stratum (2,400 rows) resolves the other families' fixture thresholds."""
    parts = _gate_parts(n=12_000, test_ratio=0.8)
    repo = _git_repo(tmp_path)
    path, sha = _commit_prereg(repo, _prereg_record(family_alpha_allocation=dict(FIVE_ALLOC),
                                                    family_roles=dict(R1_EXPLORATORY)))
    pr = d3.verify_preregistration(path, expected_sha256=sha)
    audit = _roles_audit(parts)
    cal = _bind(_write_cal(tmp_path / "cal", _evidence(audit.payload(), supported=(0.05, 0.95))))
    d = _decide(audit, cal, pr, _provenance(parts))
    fams = {f["family_id"]: f for f in audit.payload()["families"]}
    assert fams["R1"]["role"] == "exploratory" and fams["R1"]["outcome"] == "NOT_GATED"
    assert fams["R1"]["familywise_alpha"] == 0.01
    assert "CONFIRMED_DEVIATION" in fams["R1"]["records"]["category"]      # computed and reported
    assert d["outcome_overridden_because"] == []
    assert d["family_outcomes"] == {"R1": "NOT_GATED", **{f: "PASS" for f in FIVE_ALLOC}}
    assert d["outcome"] == "PASS"
    assert "R1" in d["non_gating_families"] and "R1" not in d["gating_families"]
    assert d["guarantees"]["alpha_total"] == pytest.approx(0.005)


def test_same_data_with_r1_confirmatory_fails(tmp_path) -> None:
    """Control for the test above: the R1 deviation is real and gates when R1 is confirmatory."""
    audit, cal, pr, prov, _ = _full_gate(tmp_path, _gate_parts(n=12_000, test_ratio=0.8), supported=(0.05, 0.95))
    d = _decide(audit, cal, pr, prov)
    assert d["outcome"] == "FAIL" and d["family_outcomes"]["R1"] == "FAIL"


def test_preregistered_roles_must_equal_the_audit_roles(tmp_path) -> None:
    parts = _gate_parts()
    path, sha = _commit_prereg(_git_repo(tmp_path), _prereg_record())          # R1 confirmatory
    pr = d3.verify_preregistration(path, expected_sha256=sha)
    audit = _roles_audit(parts)                                               # R1 exploratory
    cal = _bind(_write_cal(tmp_path / "cal", _evidence(audit.payload())))
    d = _decide(audit, cal, pr, _provenance(parts))
    assert d["outcome"] == "INCONCLUSIVE"
    assert any("roles" in r for r in d["outcome_overridden_because"])


@pytest.mark.parametrize("kw", [
    dict(alloc=GATE_ALLOC),                                   # allocation still covers exploratory R1
    dict(e_alpha=None),                                       # exploratory family without an alpha
    dict(roles={**ROLES, "E": "confirmatory"}),               # E can only be exploratory
    dict(roles={f: r for f, r in R1_EXPLORATORY.items() if f != "D"}),   # D has no role
    dict(roles={**R1_EXPLORATORY, "Z": "confirmatory"}),      # unknown family
    dict(roles={**R1_EXPLORATORY, "A": "maybe"}),
    dict(roles={f: "exploratory" for f in ROLES}, alloc={}),  # nothing confirmatory
])
def test_invalid_roles_or_allocation_are_refused(kw) -> None:
    with pytest.raises((TypeError, ValueError)):
        _roles_audit(_gate_parts(), **kw)


def test_e_is_built_only_when_listed_in_roles() -> None:
    p = _roles_audit(_gate_parts()).payload()
    assert "E" not in {f["family_id"] for f in p["families"]}
    p = _roles_audit(_gate_parts(), roles={**R1_EXPLORATORY, "E": "exploratory"}).payload()
    fams = {f["family_id"]: f for f in p["families"]}
    assert fams["E"]["role"] == "exploratory" and fams["E"]["outcome"] == "NOT_GATED"
    d3.verify_audit_result(_reseal(p))


def test_resealed_roles_payload_verifies_and_r1_alpha_tamper_is_caught() -> None:
    p = _roles_audit(_gate_parts()).payload()
    d3.verify_audit_result(_reseal(_copy.deepcopy(p)))
    q = _copy.deepcopy(p)
    _fam(q, "R1")["familywise_alpha"] = FIVE_ALLOC["A"]
    with pytest.raises(d3.AuditIntegrityError):
        d3.verify_audit_result(_reseal(q))
    q = _copy.deepcopy(p)
    q["configuration"]["family_alpha_allocation"]["R1"] = 0.001
    with pytest.raises(d3.AuditIntegrityError):
        d3.verify_audit_result(_reseal(q))
