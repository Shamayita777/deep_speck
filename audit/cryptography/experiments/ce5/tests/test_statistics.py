"""Software-correctness tests of the statistics, each checked against the
verifier's independent implementation or a closed form."""

import math

import numpy as np
import pytest

from audit.cryptography.ce234 import frozen_plan as P
from audit.cryptography.experiments.ce5 import analysis as AN
from audit.cryptography.experiments.ce5 import verify as V


def test_spearman_with_ties_agrees_with_verifier():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=500), rng.normal(size=500)
    b[:50] = b[0]
    assert AN.spearman(a, b) == pytest.approx(V._rho(a, b), abs=1e-12)


def test_fisher_z_closed_form():
    for r in (-0.9, -0.3, 0.0, 0.5, 0.807):
        assert AN.fisher_z(r) == pytest.approx(0.5 * math.log((1 + r) / (1 - r)), abs=1e-14)


def test_resolution_matched_has_T_ties_and_D_order():
    rng = np.random.default_rng(2)
    D = rng.normal(size=300)
    T = rng.integers(0, 5, size=300).astype(float)
    m = AN.resolution_matched(D, T)
    assert np.array_equal(np.sort(m), np.sort(__import__("scipy").stats.rankdata(T)))
    assert np.all(np.diff(m[np.argsort(D)]) >= 0)
    assert np.allclose(m, V._matched(D, T))


def test_resolution_matched_removes_the_resolution_advantage():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=20000)
    N = latent + 0.3 * rng.normal(size=20000)
    T = np.round(latent)
    D = latent.copy()
    raw = AN.fisher_z(AN.spearman(N, D)) - AN.fisher_z(AN.spearman(N, T))
    matched = AN.fisher_z(AN.spearman(N, AN.resolution_matched(D, T))) - AN.fisher_z(AN.spearman(N, T))
    assert raw > 0.1 and abs(matched) < 0.02


def test_incremental_value_agrees_with_verifier_and_detects_the_informative_predictor():
    rng = np.random.default_rng(4)
    D, T = rng.normal(size=4000), rng.normal(size=4000)
    N = np.tanh(D) + 0.1 * rng.normal(size=4000)
    iv = AN.incremental_value(N, T, D)
    assert iv["delta_D"] > 0.5 and abs(iv["delta_T"]) < 0.01
    dD, dT = V._incremental(N, T, D)
    assert iv["delta_D"] == pytest.approx(dD, abs=1e-10) and iv["delta_T"] == pytest.approx(dT, abs=1e-10)


def test_incremental_value_has_no_leakage_from_evaluation_half():
    rng = np.random.default_rng(5)
    n = 4000
    D, T = rng.normal(size=n), rng.normal(size=n)
    N = rng.normal(size=n)
    N[n // 2:] = np.tanh(D[n // 2:])
    iv = AN.incremental_value(N, T, D)
    assert iv["delta_D"] < 0.05


def test_median_interval_order_statistics():
    ci = AN.median_interval(np.arange(1, 21, dtype=float))
    assert (ci["k"], ci["lower"], ci["upper"]) == (6, 6.0, 15.0)
    assert ci["coverage"] == pytest.approx(0.958611, abs=1e-6)
    for R in (6, 20, 25, 40, 60):
        x = list(np.random.default_rng(R).normal(size=R))
        ci = AN.median_interval(x)
        assert V._interval(x) == (ci["lower"], ci["upper"])
        assert ci["coverage"] >= 0.95
    assert AN.median_interval([1.0, 2.0, 3.0])["lower"] is None


def test_outcomes_exclusive_exhaustive_and_match_verifier():
    m = 0.1
    grid = np.linspace(-0.3, 0.3, 61)
    for lo in grid:
        for hi in grid[grid >= lo]:
            got = AN.classify(lo, hi, m)
            assert got in AN.OUTCOMES and got == V._decide(lo, hi, m)
    cases = {(0.01, 0.05): "EQUIVALENT", (0.01, 0.2): "INCONCLUSIVE", (0.1, 0.3): "DISTRIBUTION_GREATER",
             (0.0999, 0.3): "INCONCLUSIVE", (-0.3, -0.1): "TRAIL_GREATER", (-0.2, -0.01): "INCONCLUSIVE",
             (-0.01, 0.2): "INCONCLUSIVE", (-0.0999, 0.0999): "EQUIVALENT", (-0.1, 0.05): "INCONCLUSIVE",
             (0.05, 0.1): "INCONCLUSIVE"}
    for (lo, hi), want in cases.items():
        assert AN.classify(lo, hi, m) == want, (lo, hi)
    assert AN.qualifier(0.01, 0.2, m) == "DIRECTION_D"
    assert AN.qualifier(-0.2, -0.01, m) == "DIRECTION_T"
    assert AN.qualifier(-0.01, 0.2, m) == "UNDETERMINED"
    assert AN.qualifier(0.2, 0.3, m) is None and AN.qualifier(0.01, 0.05, m) is None
    assert AN.classify(None, None, m) == "INCONCLUSIVE"


def _reps(deltas):
    return [{"status": "OK", "delta": d, "delta_matched": d, "delta_LR": d, "delta_D": d, "delta_T": -d,
             "rho_N_D_real": 0.8, "rho_N_T_real": 0.8, "rho_N_LR_real": 0.5, "rho_N_Dmatched_real": 0.8,
             "rho_N_D_random": 0.1} for d in deltas]


def test_analyse_on_known_distributions():
    rng = np.random.default_rng(6)
    assert AN.analyse(_reps(rng.normal(0.3, 0.02, 40)), margin=0.1, minimum_replicates=40)["decision"] == \
        "DISTRIBUTION_GREATER"
    assert AN.analyse(_reps(rng.normal(-0.3, 0.02, 40)), margin=0.1, minimum_replicates=40)["decision"] == \
        "TRAIL_GREATER"
    assert AN.analyse(_reps(rng.normal(0.0, 0.02, 40)), margin=0.1, minimum_replicates=40)["decision"] == \
        "EQUIVALENT"
    assert AN.analyse(_reps(rng.normal(0.0, 0.5, 40)), margin=0.1, minimum_replicates=40)["decision"] == \
        "INCONCLUSIVE"


def test_non_significant_difference_is_not_equivalence():
    r = AN.analyse(_reps(np.linspace(-0.3, 0.3, 20)), margin=0.1, minimum_replicates=20)
    lo, hi = r["primary"]["interval"]["lower"], r["primary"]["interval"]["upper"]
    assert lo < 0 < hi and r["primary"]["p_value"] > 0.05
    assert r["decision"] == "INCONCLUSIVE"


def test_insufficient_or_invalid_replicates_are_inconclusive():
    reps = _reps([0.3] * 19) + [{"status": "FAILED"}]
    r = AN.analyse(reps, margin=0.1, minimum_replicates=20)
    assert r["decision"] == "INCONCLUSIVE" and r["n_valid"] == 19


def test_sign_test_and_holm_agree_with_verifier():
    rng = np.random.default_rng(7)
    for _ in range(20):
        v = list(rng.normal(0.1, 1, size=25))
        assert P.exact_sign_test(v)["p_value"] == pytest.approx(V._sign_p(v), abs=1e-12)
    p = {"a": 0.01, "b": 0.04}
    assert P.holm(p) == pytest.approx(V._holm(p))


SPEC = {"accuracy": (0.9248, 0.9348), "tpr": (0.8976, 0.9176), "tnr": (0.9422, 0.9622),
        "separation_lower": (0.9298 + 0.911) / 2}


def test_clopper_pearson_and_gate_components():
    lo, hi = AN.clopper_pearson(92_980, 100_000)
    assert lo < 0.9298 < hi and hi - lo < 0.004
    g = AN.gate(90_760, 9_240, 95_220, 4_780, SPEC)
    assert g["reproduced"] and g["separated"] and g["decision"] == "PASS"
    assert g["confusion"] == {"tp": 90_760, "fn": 9_240, "tn": 95_220, "fp": 4_780}
    shifted = AN.gate(92_500, 7_500, 92_100, 7_900, SPEC)
    assert shifted["separated"] and not shifted["reproduced"] and shifted["decision"] == "FAIL"
    ddt = AN.gate(87_700, 12_300, 94_700, 5_300, SPEC)
    assert not ddt["separated"] and ddt["decision"] == "FAIL"
    low = AN.gate(90_000, 10_000, 94_400, 5_600, SPEC)
    assert low["decision"] == "FAIL"


def test_gate_spec_from_design():
    from audit.cryptography.experiments.ce5 import design as DS
    d = DS.design()
    d["gate"]["tolerance"] = {"accuracy": 0.005, "rates": 0.01}
    spec = DS.gate_spec(d)
    assert spec["accuracy"] == pytest.approx((0.9248, 0.9348))
    assert spec["tpr"] == pytest.approx((0.8976, 0.9176)) and spec["tnr"] == pytest.approx((0.9422, 0.9622))
    assert spec["separation_lower"] == pytest.approx(0.9204)


def test_robust_sd_is_the_scaled_mad():
    v = np.array([0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.50])
    assert AN.robust_sd(v) == pytest.approx(1.4826 * np.median(np.abs(v - np.median(v))))
    ps = AN.planning_sd(v)
    assert ps["sigma_plan"] == pytest.approx(max(np.std(v, ddof=1), AN.robust_sd(v)) * ps["factor"])


def test_outcome_probabilities_sum_to_one_report_mc_error_and_power_grows():
    p = AN.outcome_probabilities(0.05, 20, 0.1, 0.0, sims=20_000)
    assert sum(p[k]["p"] for k in AN.OUTCOMES) == pytest.approx(1.0)
    for k in AN.OUTCOMES:
        assert p[k]["cp_lower"] <= p[k]["p"] and p[k]["mc_se"] <= 0.0036
    a = AN.outcome_probabilities(0.08, 20, 0.1, 0.0, sims=20_000)["EQUIVALENT"]["p"]
    b = AN.outcome_probabilities(0.08, 60, 0.1, 0.0, sims=20_000)["EQUIVALENT"]["p"]
    assert b > a


def test_cell_bounds_are_bonferroni_one_sided_clopper_pearson():
    assert AN.CELL_ALPHA == pytest.approx(0.01 / 3)
    p = AN.outcome_probabilities(0.05, 20, 0.1, 0.0, sims=10_000)
    c = p["EQUIVALENT"]["count"]
    assert p["EQUIVALENT"]["cp_lower"] == pytest.approx(__import__("scipy").stats.beta.ppf(0.01 / 3, c, 10_000 - c + 1))


def test_common_random_numbers_are_shared_across_conditions(monkeypatch):
    calls = []
    real = AN.planning_draws

    def spy(R, sims=AN.PLANNING_SIMS, seed=0):
        z = real(R, sims, seed)
        calls.append((R, id(z)))
        return z
    seen = []
    real_op = AN.outcome_probabilities

    def op(sd, R, margin, true_delta, *, sims=AN.PLANNING_SIMS, seed=0, draws=None):
        seen.append((R, id(draws)))
        return real_op(sd, R, margin, true_delta, sims=sims, seed=seed, draws=draws)
    monkeypatch.setattr(AN, "planning_draws", spy)
    monkeypatch.setattr(AN, "outcome_probabilities", op)
    AN.choose_replicates(0.05, 0.1, grid=(20, 25), sims=2_000)
    assert [R for R, _ in calls] == [20, 25]
    for R, zid in calls:
        ids = {i for r, i in seen if r == R}
        assert ids == {zid} and sum(1 for r, _ in seen if r == R) == 5


def test_shared_draws_give_the_shift_relation_exactly():
    z = AN.planning_draws(20, 3_000)
    m, sd = 0.1, 0.05
    k = AN.median_interval(np.zeros(20))["k"]
    for delta in (0.0, 0.2, -0.2):
        x = np.sort(delta + sd * z, axis=1)
        lo, hi = x[:, k - 1], x[:, 20 - k]
        manual = int(((lo > -m) & (hi < m)).sum())
        assert AN.outcome_probabilities(sd, 20, m, delta, draws=z)["EQUIVALENT"]["count"] == manual


def test_replicate_rule_covers_all_three_requirements():
    r = AN.choose_replicates(0.03, 0.1, grid=range(20, 41, 10), sims=20_000)
    assert r["met_target"] and r["chosen"] == 20
    t = r["outcome_probabilities"][20]
    assert t["delta=0"]["EQUIVALENT"]["cp_lower"] >= 0.8
    assert t["delta=+2m"]["DISTRIBUTION_GREATER"]["cp_lower"] >= 0.8
    big = AN.choose_replicates(1.0, 0.1, grid=range(20, 41, 10), sims=5_000)
    assert not big["met_target"] and big["chosen"] == 40


def test_borderline_point_estimate_is_not_enough(monkeypatch):
    def fake(sd, R, margin, true_delta, sims=None, seed=0, draws=None):
        p = 0.801 if R == 20 else 0.86
        lo = 0.795 if R == 20 else 0.855
        cell = {"count": 0, "p": p, "mc_se": 0.002, "cp_lower": lo}
        return {k: cell for k in AN.OUTCOMES} | {"sims": 50_000}
    monkeypatch.setattr(AN, "outcome_probabilities", fake)
    r = AN.choose_replicates(0.05, 0.1, grid=(20, 25))
    assert r["chosen"] == 25 and r["met_target"]


def test_planning_sd_is_never_below_the_exact_normal_bound_and_covers():
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(4000):
        v = rng.normal(0, 1.0, size=10)
        ps = AN.planning_sd(v)
        assert ps["sigma_plan"] >= ps["u_sd"] - 1e-15
        hits += 1.0 <= ps["sigma_plan"]
    assert hits / 4000 >= 0.79
    assert AN.sd_upper_bound([0.1, 0.12, 0.08, 0.11, 0.09]) > np.std([0.1, 0.12, 0.08, 0.11, 0.09], ddof=1)


def test_pairs_rule():
    assert AN.choose_pairs(0.01, 60)["chosen"] == 30_000
    assert AN.choose_pairs(10.0, 60)["chosen"] is None


@pytest.mark.parametrize("R,k,coverage", [(20, 6, 0.9586), (25, 8, 0.9567), (30, 10, 0.9572), (35, 12, 0.9590),
                                          (40, 14, 0.9615), (45, 16, 0.9643), (50, 18, 0.9672), (55, 20, 0.9700),
                                          (60, 22, 0.9727)])
def test_interval_indices_and_coverage_on_the_registered_grid(R, k, coverage):
    x = np.arange(1, R + 1, dtype=float)
    ci = AN.median_interval(x)
    assert (ci["k"], ci["lower"], ci["upper"]) == (k, float(k), float(R - k + 1))
    assert ci["coverage"] == pytest.approx(coverage, abs=5e-5)
    assert V._interval(list(x)) == (float(k), float(R - k + 1))


def test_exact_boundary_equalities():
    m = 0.1
    assert AN.classify(0.1, 0.5, m) == "DISTRIBUTION_GREATER" == V._decide(0.1, 0.5, m)
    assert AN.classify(-0.5, -0.1, m) == "TRAIL_GREATER" == V._decide(-0.5, -0.1, m)
    assert AN.classify(-0.1, 0.0, m) == "INCONCLUSIVE" == V._decide(-0.1, 0.0, m)
    assert AN.classify(0.0, 0.1, m) == "INCONCLUSIVE" == V._decide(0.0, 0.1, m)
    assert AN.classify(-0.3, 0.3, m) == "INCONCLUSIVE" and AN.qualifier(-0.3, 0.3, m) == "UNDETERMINED"
    for lo, hi in ((0.1, 0.5), (-0.5, -0.1), (0.0, 0.05)):
        assert V._qualifier(lo, hi, m) == AN.qualifier(lo, hi, m)
