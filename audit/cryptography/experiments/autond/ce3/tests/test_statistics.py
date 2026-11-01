import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from audit.cryptography.experiments.autond.ce3 import statistics as ST
from audit.cryptography.experiments.autond.ce3 import verify as V

pytest.importorskip("sklearn")

from audit.cryptography.experiments.autond.ce3 import probes as PR


def brute_upper_tail(pos, n):
    return Fraction(sum(1 for signs in itertools.product((0, 1), repeat=n) if sum(signs) >= pos), 2 ** n)


@pytest.mark.parametrize("n", range(1, 13))
def test_sign_test_equals_brute_force_enumeration(n):
    for pos in range(n + 1):
        t = ST.sign_test([1.0] * pos + [-1.0] * (n - pos))
        assert Fraction(t["p_numerator"], t["p_denominator"]) == brute_upper_tail(pos, n)


def test_zero_and_negative_zero_count_as_nonpositive_and_stay_in_the_denominator():
    t = ST.sign_test([0.1] * 17 + [-0.2, -0.3] + [-0.0])
    assert (t["n_valid"], t["n_positive"], t["n_nonpositive"], t["n_negative"], t["n_zero"]) == (20, 17, 3, 2, 1)
    assert Fraction(t["p_numerator"], t["p_denominator"]) == Fraction(1351, 2 ** 20)
    assert ST.sign_test([0.1] * 15 + [0.0] * 5)["p_value"] == ST.sign_test([0.1] * 15 + [-0.1] * 5)["p_value"]


def test_zeros_never_make_a_test_easier_to_reject():
    rng = np.random.default_rng(1)
    for _ in range(300):
        x = rng.choice([-1.0, 0.0, 1.0], 20)
        removed = x[x != 0]
        p_kept = Fraction(ST.sign_test(x)["p_numerator"], ST.sign_test(x)["p_denominator"])
        if removed.size:
            r = ST.sign_test(removed)
            assert p_kept >= Fraction(r["p_numerator"], r["p_denominator"])


def test_all_zero_endpoint_is_tested_and_never_rejects():
    t = ST.sign_test([0.0] * 20)
    assert t["status"] == "TESTED" and t["n_positive"] == 0 and t["p_value"] == 1.0
    assert not ST.rejects(t, 0.05)


def test_no_valid_observations_is_untestable():
    t = ST.sign_test([])
    assert t["status"] == ST.NO_EFFECTIVE and t["p_value"] is None and not ST.rejects(t, 0.05)


def test_non_finite_is_refused_not_coerced():
    with pytest.raises(ValueError):
        ST.sign_test([0.1, float("nan")])


def test_twenty_replicate_cutoff_is_fifteen():
    assert ST.rejects(ST.sign_test([1.0] * 15 + [-1.0] * 5), 0.05)
    assert not ST.rejects(ST.sign_test([1.0] * 14 + [-1.0] * 6), 0.05)
    assert not ST.rejects(ST.sign_test([1.0] * 14 + [0.0] * 6), 0.05)


def test_level_is_at_most_alpha_for_every_null_q():
    crit = 15
    for q in np.linspace(0.0, 0.5, 51):
        size = sum(math.comb(20, k) * q ** k * (1 - q) ** (20 - k) for k in range(crit, 21))
        assert size <= 0.05


def test_clopper_pearson_matches_independent_bisection():
    for n in (1, 5, 19, 20):
        for k in range(n + 1):
            assert np.allclose(ST.clopper_pearson(k, n), V.cp(k, n), atol=1e-9)


def _rank_upper(scores, tb):
    return PR.s1_rank(scores[0], list(scores[1:]), tb)["upper_half"]


def _exact_upper_fraction(multiset, b):
    m = b + 1
    arrangements = np.array(sorted(set(itertools.permutations(multiset))), dtype=np.int64)
    tiebreaks = np.array(list(itertools.permutations(range(m))), dtype=np.int64)
    s, t = arrangements[:, None, :], tiebreaks[None, :, :]
    below = ((s[..., 1:] < s[..., :1]) | ((s[..., 1:] == s[..., :1]) & (t[..., 1:] < t[..., :1]))).sum(-1)
    rank = below + 1
    counts = [int(np.count_nonzero(rank == r)) for r in range(1, m + 1)]
    return Fraction(int(np.count_nonzero(rank > m // 2)), rank.size), counts


@pytest.mark.parametrize("b", [1, 3, 5])
def test_s1_rank_is_exactly_uniform_for_every_exchangeable_law_with_ties(b):
    m = b + 1
    for multiset in itertools.combinations_with_replacement(range(3), m):
        frac, counts = _exact_upper_fraction(multiset, b)
        assert frac == Fraction(1, 2), multiset
        assert len(set(counts)) == 1, (multiset, counts)


def test_s1_rank_brute_force_through_the_producer_and_verifier_functions():
    b = 3
    for multiset in itertools.combinations_with_replacement([Fraction(1, 5), Fraction(1, 4), Fraction(1, 3)], b + 1):
        up = total = 0
        for arr in set(itertools.permutations(multiset)):
            for tb in itertools.permutations(range(b + 1)):
                p = PR.s1_rank(arr[0], list(arr[1:]), tb)
                assert p["rank"] == V.observed_rank(arr[0], list(arr[1:]), tb)
                up += p["upper_half"]
                total += 1
        assert Fraction(up, total) == Fraction(1, 2)


def test_superseded_median_rule_is_not_exact_under_ties():
    b = 5

    def old_conditional(p_one):
        p = {0: 1 - Fraction(p_one), 1: Fraction(p_one)}
        pos = neg = Fraction(0)
        for s in itertools.product((0, 1), repeat=b + 1):
            w = math.prod(p[v] for v in s)
            med = sorted(s[1:])[b // 2]
            pos += w * (s[0] > med)
            neg += w * (s[0] < med)
        return pos / (pos + neg)
    assert old_conditional(Fraction(3, 10)) > Fraction(68, 100)
    assert old_conditional(Fraction(7, 10)) < Fraction(32, 100)
    for p_one in (Fraction(3, 10), Fraction(7, 10)):
        prob = {0: 1 - p_one, 1: p_one}
        up = Fraction(0)
        for s in itertools.product((0, 1), repeat=b + 1):
            w = math.prod(prob[v] for v in s)
            frac, _ = _exact_upper_fraction(s, b)
            up += w * frac
        assert up == Fraction(1, 2)


@pytest.mark.parametrize("kind", ["untied_continuous", "heavily_tied_discrete", "all_tied"])
def test_s1_rank_simulation_under_exchangeability(kind):
    rng = np.random.default_rng({"untied_continuous": 1, "heavily_tied_discrete": 2, "all_tied": 3}[kind])
    b, n = 5, 200_000
    if kind == "untied_continuous":
        s = rng.standard_normal((n, b + 1))
    elif kind == "heavily_tied_discrete":
        s = rng.choice(np.array([0.2, 0.2, 0.2, 0.21, 0.25]), size=(n, b + 1))
    else:
        s = np.full((n, b + 1), 0.2)
    t = np.argsort(rng.random((n, b + 1)), axis=1)
    below = ((s[:, 1:] < s[:, :1]) | ((s[:, 1:] == s[:, :1]) & (t[:, 1:] < t[:, :1]))).sum(1)
    rank = below + 1
    se = math.sqrt(0.25 / n)
    assert abs(np.mean(rank > (b + 1) // 2) - 0.5) < 4.5 * se
    counts = np.bincount(rank, minlength=b + 2)[1:]
    chi2 = float(np.sum((counts - n / (b + 1)) ** 2 / (n / (b + 1))))
    assert chi2 < 20.5


def test_s1_rank_breaks_ties_only_through_the_recorded_values():
    a = Fraction(1, 5)
    r_low = PR.s1_rank(a, [a] * 5, [0, 1, 2, 3, 4, 5])
    r_high = PR.s1_rank(a, [a] * 5, [5, 0, 1, 2, 3, 4])
    assert (r_low["rank"], r_high["rank"]) == (1, 6) and r_low["ties_with_observed"] == 5
    assert r_low["centred_rank"] == Fraction(-5, 2) and r_high["centred_rank"] == Fraction(5, 2)
    assert PR.s1_rank(Fraction(1, 3), [Fraction(1, 5)] * 5, [0, 1, 2, 3, 4, 5])["rank"] == 6
    with pytest.raises(PR.ProbeError):
        PR.s1_rank(a, [a] * 5, [0, 1, 2])


def test_tiebreak_draw_is_a_uniform_permutation_and_validated():
    for _ in range(50):
        t = PR.draw_tiebreak(5)
        assert PR.tiebreak_ok(t, 5) and t.dtype == np.int16
    assert not PR.tiebreak_ok(np.array([0, 1, 1, 3, 4, 5]), 5)
    assert not PR.tiebreak_ok(np.arange(5), 5)
    firsts = np.bincount([int(PR.draw_tiebreak(5)[0]) for _ in range(6000)], minlength=6)
    assert np.all(np.abs(firsts - 1000) < 150)


def _tests_for(values):
    return {k: ST.sign_test(v) for k, v in values.items()}


def test_fixed_sequence_outcomes():
    pos, neg = [1.0] * 20, [1.0] * 13 + [-1.0] * 7
    base = {"C_task": pos, "C_twin": pos, "S1": pos, "S2": pos}
    cases = [({}, "SUPPORTED"), ({"S2": neg}, "TARGET_DECODABLE_TWIN_UNRESOLVED"), ({"S1": neg}, "INCONCLUSIVE"),
             ({"C_task": neg}, "POSITIVE_CONTROL_FAILED"), ({"C_twin": neg}, "POSITIVE_CONTROL_FAILED"),
             ({"S1": [0.0] * 20}, "INCONCLUSIVE"), ({"C_twin": [0.0] * 20}, "POSITIVE_CONTROL_FAILED"),
             ({"S2": [1.0] * 14 + [0.0] * 6}, "TARGET_DECODABLE_TWIN_UNRESOLVED")]
    for change, expected in cases:
        d = ST.decide(_tests_for({**base, **change}), alpha=0.05, n_valid=20, n_required=20, precheck_ok=True)
        assert d["decision"] == expected, (change, d)


def test_c_requires_both_task_recovery_and_twin_separation():
    weak = [1.0] * 10 + [-1.0] * 10
    for part in ("C_task", "C_twin"):
        vals = {"C_task": [1.0] * 20, "C_twin": [1.0] * 20, "S1": [1.0] * 20, "S2": [1.0] * 20, part: weak}
        d = ST.decide(_tests_for(vals), alpha=0.05, n_valid=20, n_required=20, precheck_ok=True)
        assert d["decision"] == "POSITIVE_CONTROL_FAILED" and d["sequence"][0]["rejected_components"][part] is False


def test_intersection_union_gate_level_is_alpha_without_adjustment():
    crit = 15
    for q_task in np.linspace(0.0, 0.5, 11):
        for q_twin in (0.5, 0.8, 1.0):
            pt = sum(math.comb(20, k) * q_task ** k * (1 - q_task) ** (20 - k) for k in range(crit, 21))
            pw = sum(math.comb(20, k) * q_twin ** k * (1 - q_twin) ** (20 - k) for k in range(crit, 21))
            assert min(pt, pw) <= 0.05


def test_incomplete_and_precheck():
    vals = {k: [1.0] * 19 for k in ("C_task", "C_twin", "S1", "S2")}
    assert ST.decide(_tests_for(vals), alpha=0.05, n_valid=19, n_required=20, precheck_ok=True)["decision"] == "INCONCLUSIVE"
    assert ST.decide(_tests_for(vals), alpha=0.05, n_valid=20, n_required=20, precheck_ok=False)["decision"] == "PRECHECK_FAILED"


def test_worked_example_from_methodology():
    vals = {"C_task": [1.0] * 20, "C_twin": [1.0] * 20, "S1": [2.5] * 9 + [1.5] * 5 + [0.5] * 3 + [-0.5, -1.5, -2.5],
            "S2": [1.0] * 13 + [-1.0] * 7}
    t = _tests_for(vals)
    assert Fraction(t["S1"]["p_numerator"], t["S1"]["p_denominator"]) == Fraction(1351, 2 ** 20)
    assert Fraction(t["S2"]["p_numerator"], t["S2"]["p_denominator"]) == Fraction(137980, 2 ** 20)
    d = ST.decide(t, alpha=0.05, n_valid=20, n_required=20, precheck_ok=True)
    assert d["decision"] == "TARGET_DECODABLE_TWIN_UNRESOLVED"


def test_producer_and_verifier_agree_on_random_cases():
    rng = np.random.default_rng(7)
    for _ in range(500):
        vals = {k: list(np.round(rng.normal(rng.uniform(-0.5, 1.0), 1.0, 20), 0)) for k in ("C_task", "C_twin", "S1", "S2")}
        a = ST.decide(_tests_for(vals), alpha=0.05, n_valid=20, n_required=20, precheck_ok=True)["decision"]
        vt = {k: V.sign_test(v) for k, v in vals.items()}
        assert a == V.decide(vt, 0.05, 20, 20, True)
        for k in vals:
            p, q = ST.sign_test(vals[k]), vt[k]
            assert all(p[f] == q[f] for f in ("n_valid", "n_positive", "n_nonpositive", "n_negative", "n_zero"))
            assert Fraction(p["p_numerator"], p["p_denominator"]) == q["p"]


def test_analyse_marks_untested_steps():
    states = {f"r{i}": "VALID" for i in range(20)}
    eps = {r: {"C_task": -1.0, "C_twin": 1.0, "S1": 2.5, "S2": 1.0, "raw_gap": 0.0, "S1_margin": 0.0} for r in states}
    a = ST.analyse(eps, states, alpha=0.05, n_required=20, precheck_ok=True)
    assert a["decision"] == "POSITIVE_CONTROL_FAILED"
    assert a["endpoints"]["S1"]["tested_in_sequence"] is False and a["endpoints"]["C_task"]["tested_in_sequence"] is True
    assert a["S1_margin"]["summary"]["n"] == 20


@pytest.mark.parametrize("states,expected", [
    ({**{f"r{i}": "VALID" for i in range(19)}, "r19": "FAILED"}, "INCONCLUSIVE"),
    ({**{f"r{i}": "VALID" for i in range(19)}, "r19": "EXCLUDED"}, "INCONCLUSIVE"),
    ({f"r{i}": "VALID" for i in range(20)}, "SUPPORTED"),
])
def test_analyse_failed_and_excluded_replicates_are_never_replaced(states, expected):
    eps = {r: {"C_task": 1.0, "C_twin": 1.0, "S1": 2.5, "S2": 1.0, "raw_gap": 0.0, "S1_margin": 0.0}
           for r, s in states.items() if s == "VALID"}
    a = ST.analyse(eps, states, alpha=0.05, n_required=20, precheck_ok=True)
    assert a["decision"] == expected and a["n_valid"] == len(eps)


def test_median_interval_for_twenty_uses_order_statistics_six_and_fifteen():
    m = ST.median_ci(np.arange(20.0))
    assert (m["lower"], m["upper"]) == (5.0, 14.0) and m["coverage"] >= 0.95
    assert math.isclose(m["coverage"], 1 - 2 * sum(math.comb(20, j) for j in range(6)) / 2 ** 20)
