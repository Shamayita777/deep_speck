"""Does each generic D2 test actually test what its certificate claims?

Scientific status
-----------------
METHOD-VALIDITY EVIDENCE for the generic D2 engine. These tests establish
behaviour of the AUDIT INSTRUMENT on synthetic data with known structure.
They establish nothing about the Gohr or AutoND datasets.

Three defects found at schema 6.4 are pinned here so they cannot return:

1. The within-partition and cross-partition pairwise randomization tests had
   no power. Their observed pairing was an arbitrary matching of randomly
   sampled rows and their null re-paired that same row pool, so the observed
   statistic was one draw from its own null: the null was true by
   construction and the p-value uniform under every alternative. They are now
   DESCRIPTIVE_ONLY and excluded from the decision.

2. The statistical gate was unreachable at the configured permutation
   budgets. It fires on a simultaneous Clopper-Pearson upper bound, but the
   guard checked 1/(max_tail+1) -- the point-estimate resolution, roughly an
   order of magnitude smaller. A budget could pass the guard while the gate
   stayed out of reach, so a maximal violation was silently downgraded to
   INCONCLUSIVE and the engine could never return FAIL from a distributional
   component.

3. The multivariate detector's positive class was an arbitrary matching, not
   the generation-order lag-1 pairing its interpretation described, so it too
   had no power. Its null also permuted labels freely although each row
   appears in one positive and one negative example.

The controls below are deliberately extreme: every second row is an exact or
2-bit copy of its predecessor. Any instrument claiming to test row dependence
must detect that. A test that cannot is not conservative, it is uninformative.
"""

from __future__ import annotations

import numpy as np
import pytest

from audit.dataset import d2_sample_dependence as d2

BITS = 64


# ---------------------------------------------------------------------
# Synthetic controls
# ---------------------------------------------------------------------

def _independent(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(n, BITS), dtype=np.uint8)


def _exact_copy_at_lag1(n: int, rng: np.random.Generator) -> np.ndarray:
    x = _independent(n, rng)
    x[1::2] = x[0::2]
    return x


def _near_copy_at_lag1(n: int, rng: np.random.Generator,
                       radius: int = 2) -> np.ndarray:
    x = _independent(n, rng)
    x[1::2] = x[0::2]
    for k in range(0, n - 1, 2):
        x[k + 1, rng.choice(BITS, radius, replace=False)] ^= 1
    return x


def _run_one(left, right, within, rng, *, alpha, permutations):
    observed = d2.hamming_distances(left, right)
    pmf, stats, meta = d2._null_pair_histogram(
        left, right, permutations=permutations, rng=rng, within=within,
        reference_permutations=d2.DEFAULT_NULL_REFERENCE_PERMUTATIONS,
        batch_size=d2.DEFAULT_NULL_BATCH_SIZE,
        stopping_error=d2.DEFAULT_NULL_STOPPING_ERROR,
        alpha=alpha, observed_distances=observed)
    return d2.summarize_randomization_test(
        observed, pmf, stats, alpha=alpha, rng=rng, bootstrap_replicates=100,
        practical_excess_tvd_threshold=0.01, null_sampling_metadata=meta)


# ---------------------------------------------------------------------
# 1. The pairwise within/cross construction is vacuous
# ---------------------------------------------------------------------

def test_within_observed_pairing_is_a_draw_from_its_own_null():
    """The construction itself, with no statistics involved.

    `_randomize_pairing(within=True)` splits one random 2*count subset into
    halves. `_null_pair_histogram(within=True)` pools those same rows and
    re-splits them. The two procedures are the same procedure, so the observed
    statistic cannot be extreme relative to the null for any dataset.
    """
    rng = np.random.default_rng(0)
    x = _exact_copy_at_lag1(2000, rng)
    left, right, i, j = d2._randomize_pairing(x, None, 300, rng, within=True)
    # The endpoints are disjoint and carry no order relation to the source.
    assert len(set(i.tolist()) & set(j.tolist())) == 0
    gaps = np.abs(j - i)
    assert gaps.min() > 1 or gaps.max() > 1, (
        "endpoint gaps look like a fixed lag; the pairing would then be "
        "source-defined and this test's premise would be wrong")
    # The null pool is exactly the observed endpoint rows.
    pool = np.concatenate([left, right], axis=0)
    assert pool.shape[0] == 2 * 300
    assert np.array_equal(np.sort(pool, axis=0),
                          np.sort(np.concatenate([x[i], x[j]], axis=0), axis=0))


@pytest.mark.parametrize("within", [True, False])
def test_pairwise_p_values_are_uniform_under_maximal_dependence(within):
    """Zero power: a KS test cannot distinguish the p-values from uniform.

    Replaces an eyeball check with a calibration statement. If this ever
    fails it means the pairing construction changed -- which would be good
    news, but the claim in the certificate must then change with it.
    """
    from scipy.stats import kstest
    alpha, perms, trials = 0.05, 1500, 25
    ps = []
    for t in range(trials):
        rng = np.random.default_rng(100 + t)
        x = _exact_copy_at_lag1(2500, rng)
        if within:
            left, right, _, _ = d2._randomize_pairing(x, None, 200, rng,
                                                      within=True)
        else:
            y = _exact_copy_at_lag1(2500, np.random.default_rng(900 + t))
            left, right, _, _ = d2._randomize_pairing(x, y, 200, rng,
                                                      within=False)
        ps.append(_run_one(left, right, within, rng, alpha=alpha,
                           permutations=perms)["randomization_test"]["p_value"])
    ks = kstest(ps, "uniform")
    assert ks.pvalue > 0.01, (
        f"p-values departed from uniform (KS p={ks.pvalue:.4g}) on maximally "
        f"dependent data; the within/cross construction may have changed")
    assert np.mean(np.asarray(ps) <= alpha) < 0.25, (
        "rejection rate is far above nominal; investigate before trusting")


def test_lag_test_detects_what_the_pairwise_test_cannot():
    """The same data, the valid instrument: separation must be total."""
    rng = np.random.default_rng(7)
    x = _exact_copy_at_lag1(2500, rng)
    i, j = d2._sample_disjoint_lag_pairs(len(x), 1, 200, rng)
    assert np.array_equal(j - i, np.ones(200, dtype=np.int64)), (
        "lag pairs must be (k, k+1) in SOURCE order; that is the whole "
        "reason this test has power")
    r = _run_one(x[i], x[j], False, rng, alpha=0.05, permutations=1500)
    assert r["mean_hamming_distance"] == 0.0
    assert r["tvd"] > 0.9
    assert r["excess_tvd"] > 0.5


def test_near_duplicate_dependence_is_also_caught_by_the_lag_test():
    rng = np.random.default_rng(8)
    x = _near_copy_at_lag1(2500, rng, radius=2)
    i, j = d2._sample_disjoint_lag_pairs(len(x), 1, 200, rng)
    r = _run_one(x[i], x[j], False, rng, alpha=0.05, permutations=1500)
    assert r["mean_hamming_distance"] == pytest.approx(2.0, abs=0.001)
    assert r["excess_tvd"] > 0.5


# ---------------------------------------------------------------------
# 2. The statistical gate must be reachable, and the guard must say so
# ---------------------------------------------------------------------

def test_the_gate_bound_is_far_above_the_point_estimate_resolution():
    """The two quantities are not interchangeable; that was the bug."""
    b = d2.minimum_attainable_decision_bound(120_000)
    assert b["minimum_attainable_upper_p"] > 10 * b[
        "minimum_attainable_monte_carlo_p"], (
        "if these were close, the old guard would have been adequate")
    assert "NOT the operative quantity" in b["gate_semantics"]


@pytest.mark.parametrize("total,alpha,reachable", [
    (120_000, 0.01 / 102, False),   # Gohr production family of 102
    (120_000, 0.01 / 105, False),   # AutoND production analysis A
    (150_000, 0.01 / 105, True),    # the raised default
    (10_000, 0.01 / 45, False),     # the former smoke budget
    (60_000, 0.01 / 45, True),      # the raised smoke budget
])
def test_known_configurations_are_classified_correctly(total, alpha,
                                                       reachable):
    b = d2.minimum_attainable_decision_bound(total)
    assert (b["minimum_attainable_upper_p"] < alpha) is reachable


def test_required_permutations_is_the_smallest_sufficient_budget():
    for alpha in (0.01 / 27, 0.01 / 45, 0.01 / 105):
        need = d2.required_permutations_for_alpha(alpha)
        assert d2.minimum_attainable_decision_bound(
            need)["minimum_attainable_upper_p"] < alpha
        assert d2.minimum_attainable_decision_bound(
            need - 1)["minimum_attainable_upper_p"] >= alpha, (
            f"{need} is not minimal for alpha={alpha}")


def test_the_shipped_default_clears_every_declared_configuration():
    b = d2.minimum_attainable_decision_bound(d2.DEFAULT_NULL_PERMUTATIONS)
    for family in (102, 105, 45, 27):
        assert b["minimum_attainable_upper_p"] < d2.FAMILYWISE_ALPHA / family, (
            f"the default budget cannot reach the gate at a family of "
            f"{family}")


def _run_d2(x_dict, *, permutations, **over):
    kw = dict(
        partitions=x_dict, feature_bits=BITS,
        reference_pmf=d2.binomial_reference_distribution(BITS, 0.5),
        structured_views={}, pairs_per_test=200, audit_replicates=1,
        null_permutations=permutations, bootstrap_replicates=100, lags=(1,),
        near_duplicate_radii=(1,), multivariate_partitions=("train",),
        multivariate_pairs=200, multivariate_permutations=500, audit_seed=1,
        practical_excess_tvd_threshold=0.01)
    kw.update(over)
    return d2.run_d2(**kw)


def test_run_d2_refuses_an_unreachable_permutation_budget():
    rng = np.random.default_rng(21)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    with pytest.raises(ValueError, match="cannot reach the statistical gate"):
        _run_d2(parts, permutations=3000)


def test_the_refusal_names_the_bound_the_point_estimate_and_the_remedy():
    rng = np.random.default_rng(22)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    with pytest.raises(ValueError) as e:
        _run_d2(parts, permutations=3000)
    msg = str(e.value)
    assert "minimum attainable upper bound" in msg
    assert "NOT the operative quantity" in msg
    assert "required total permutations" in msg
    assert "do not lower the familywise alpha" in msg


def test_maximal_dependence_now_reaches_FAIL_not_merely_INCONCLUSIVE():
    """The decisive regression.

    At schema 6.4 this same data produced INCONCLUSIVE: excess TVD 0.895
    against a 0.01 threshold, yet statistically_excessive False because the
    upper bound stalled at 3.3e-3. A two-gate rule whose statistical gate
    cannot fire is a one-gate rule that can only ever warn.
    """
    rng = np.random.default_rng(23)
    parts = {n: _exact_copy_at_lag1(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    assert dec["outcome"] == "FAIL"
    lag_failures = [f for f in dec["failures"] if f["component"].endswith("lag1")]
    assert len(lag_failures) == 3, dec["failures"]
    gate = res["configuration"]["statistical_gate_reachability"]
    assert gate["minimum_attainable_upper_p"] < res["configuration"][
        "effective_test_alpha"]


def test_independent_data_does_not_fail_at_a_reachable_budget():
    """Size control: the corrected engine must not invent findings."""
    rng = np.random.default_rng(24)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    assert dec["outcome"] != "FAIL", dec["failures"]
    assert not [f for f in dec["failures"]
                if f["type"] in ("distributional", "multivariate_dependence")]


# ---------------------------------------------------------------------
# 3. Pairwise components are reported but excluded from the decision
# ---------------------------------------------------------------------

def test_pairwise_components_are_marked_descriptive_only():
    rng = np.random.default_rng(31)
    parts = {n: _exact_copy_at_lag1(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    pw = res["d2_2_pairwise_structure"]
    assert pw, "pairwise components must still be computed and reported"
    for name, r in pw.items():
        assert r["inference"]["status"] == "DESCRIPTIVE_ONLY", name
        assert r["inference"]["excluded_from_decision"] is True, name
        assert "true by construction" in r["inference"]["reason"], name
        assert "serial" in r["inference"][
            "valid_instrument_for_this_question"], name


def test_no_failure_or_warning_comes_from_a_pairwise_component():
    rng = np.random.default_rng(32)
    parts = {n: _exact_copy_at_lag1(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    pairwise_names = set(res["d2_2_pairwise_structure"])
    for item in dec["failures"] + dec["warnings"]:
        assert item["component"] not in pairwise_names, item


def test_excluded_components_are_recorded_with_what_they_would_have_said():
    """Excluded, not hidden. A reader can audit the suppression."""
    rng = np.random.default_rng(33)
    parts = {n: _exact_copy_at_lag1(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    excl = dec["excluded_from_decision"]
    assert set(e["component"] for e in excl) == set(
        res["d2_2_pairwise_structure"])
    for e in excl:
        assert e["inference_status"] == "DESCRIPTIVE_ONLY"
        assert "would_have_flagged_statistical" in e
        assert "would_have_flagged_practical" in e
    assert "not inferential" in dec["excluded_from_decision_note"]


def test_the_certificate_does_not_claim_pairwise_evidence():
    rng = np.random.default_rng(34)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, _ = _run_d2(parts, permutations=40_000)
    cfg = res["configuration"]
    assert cfg["pairwise_inference_status"].startswith("DESCRIPTIVE_ONLY")
    assert cfg["confirmatory_pairwise_components"] == (
        "d2_2_serial_dependence only")
    assert "NOT CONFIRMATORY" in cfg["confirmatory_pairwise_null"]


# ---------------------------------------------------------------------
# 4. Multivariate detector semantics
# ---------------------------------------------------------------------

def test_detector_positives_are_source_order_lag_pairs():
    """The interpretation and the implementation must agree."""
    rng = np.random.default_rng(41)
    x = _independent(4000, rng)
    r = d2.multivariate_pair_discrimination(
        x, pairs=200, permutations=200, rng=rng, alpha=0.05)
    assert r["lag"] == 1
    assert "generation-order lag-1 pairs" in r["pair_definition"]
    assert "row k+1" in r["pair_definition"]
    assert r["row_disjoint_train_test"] is True


def test_detector_null_is_matched_not_free_permutation():
    rng = np.random.default_rng(42)
    r = d2.multivariate_pair_discrimination(
        _independent(4000, rng), pairs=200, permutations=200, rng=rng,
        alpha=0.05)
    assert r["null_model"] == (
        "matched-couple label permutation within shared-endpoint couples")
    assert "free label permutation is not valid" in r["null_model_reason"]


def test_detector_has_power_against_source_order_dependence():
    rng = np.random.default_rng(43)
    r = d2.multivariate_pair_discrimination(
        _exact_copy_at_lag1(4000, rng), pairs=300, permutations=1000,
        rng=rng, alpha=0.05)
    assert r["auc"] > 0.95, r["auc"]
    assert r["statistically_excessive"] is True
    assert r["practically_excessive"] is True


def test_detector_has_power_against_near_duplicate_source_order_dependence():
    rng = np.random.default_rng(44)
    r = d2.multivariate_pair_discrimination(
        _near_copy_at_lag1(4000, rng, radius=2), pairs=300,
        permutations=1000, rng=rng, alpha=0.05)
    assert r["auc"] > 0.95, r["auc"]
    assert r["statistically_excessive"] is True


def test_detector_size_is_controlled_on_independent_rows():
    rejections, trials = 0, 12
    for t in range(trials):
        rng = np.random.default_rng(500 + t)
        r = d2.multivariate_pair_discrimination(
            _independent(4000, np.random.default_rng(500 + t)), pairs=250,
            permutations=600, rng=rng, alpha=0.05)
        rejections += bool(r["statistically_excessive"])
    assert rejections <= 3, (
        f"{rejections}/{trials} rejections at alpha=0.05 on independent rows")


def test_detector_null_is_centred_on_chance():
    rng = np.random.default_rng(45)
    r = d2.multivariate_pair_discrimination(
        _independent(4000, rng), pairs=250, permutations=800, rng=rng,
        alpha=0.05)
    assert abs(r["null_auc_mean"] - 0.5) < 0.02, r["null_auc_mean"]


def test_detector_reports_its_own_permutation_resolution():
    rng = np.random.default_rng(46)
    r = d2.multivariate_pair_discrimination(
        _independent(4000, rng), pairs=200, permutations=99, rng=rng,
        alpha=1e-4)
    assert r["minimum_attainable_permutation_p"] == pytest.approx(1 / 100)
    assert r["permutation_budget_resolves_alpha"] is False
    r2 = d2.multivariate_pair_discrimination(
        _independent(4000, rng), pairs=200, permutations=200, rng=rng,
        alpha=0.05)
    assert r2["permutation_budget_resolves_alpha"] is True


def test_no_negative_example_is_accidentally_a_true_pair():
    """A cyclic derangement has no fixed point, so no negative is a positive."""
    n = 50
    rolled = np.roll(np.arange(n), 1)
    assert not np.any(rolled == np.arange(n))


# ---------------------------------------------------------------------
# 5. The two-gate rule, and the hypothesis inventory
# ---------------------------------------------------------------------

def test_both_gates_are_required_for_a_multivariate_failure():
    """One gate alone is INCONCLUSIVE, per the documented rule."""
    rng = np.random.default_rng(51)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    for name, r in res["d2_3_multivariate_dependence"].items():
        both = r["statistically_excessive"] and r["practically_excessive"]
        in_failures = any(f["component"] == name for f in dec["failures"])
        assert in_failures == both, (name, r["statistically_excessive"],
                                     r["practically_excessive"])


def test_inventory_separates_inferential_tests_from_denominator_slots():
    rng = np.random.default_rng(52)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, _ = _run_d2(parts, permutations=40_000)
    inv = res["configuration"]["inferential_test_inventory"]
    # 3 lag tests + 3 near-duplicate radii + 0 structured + 1 multivariate
    assert inv["inferential_tests"]["serial_dependence_lag"] == 3
    assert inv["inferential_tests"]["multivariate"] == 1
    assert inv["non_inferential_slots"][
        "pairwise_within_cross_descriptive_only"] == 6
    assert inv["familywise_denominator"] == res["configuration"][
        "familywise_test_count"]
    # The denominator must be at least the number of real tests, and the
    # engine must say that it counts slots rather than executed tests.
    assert inv["familywise_denominator"] >= inv["inferential_tests"]["total"]
    assert "CONSERVATIVE DENOMINATOR" in res["configuration"][
        "familywise_denominator_semantics"]
    assert "NOT_TESTED" in res["configuration"][
        "familywise_denominator_semantics"]


def test_structured_not_tested_views_count_as_slots_not_tests():
    rng = np.random.default_rng(53)
    parts = {n: _independent(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    vals = rng.integers(0, 2**32, size=3000, dtype=np.uint64)
    views = {
        "declared": {"values": vals, "domain_size": 2**32,
                     "collision_null": "uniform", "description": "d"},
        "undeclared": {"values": vals, "domain_size": 2**32,
                       "description": "u"},
    }
    res, _ = _run_d2(parts, permutations=40_000, structured_views=views)
    inv = res["configuration"]["inferential_test_inventory"]
    assert inv["inferential_tests"]["structured_views_tested"] == 1
    assert inv["non_inferential_slots"]["structured_views_not_tested"] == 1


def test_schema_version_records_that_the_method_changed():
    assert d2.SCHEMA_VERSION == "7.0", (
        "a method change must bump the schema so earlier certificates are "
        "not presented as directly comparable")
    # Bumping the string is not the point; the method must actually differ.
    # At 7.0 D2.1 verifies exhaustively against a declared closed-form null.
    import numpy as _np
    out = d2.near_neighbor_exact_screen(
        _np.random.default_rng(0).integers(0, 2, size=(2000, 64), dtype=_np.uint8),
        (1, 4), rng=_np.random.default_rng(1),
        null_model={"kind": "uniform"}, query_count=128)
    for rec in out["by_radius"].values():
        assert rec["exhaustive"] is True
        assert rec["declared_query_probability"] > 0.0
        assert rec["declared_null_method"].startswith("closed form")


# ---------------------------------------------------------------------
# 6. What can and cannot produce a D2 failure
#
#    Established by reading run_d2's actual call graph, because it decides
#    how much inferential weight each invocation really carries.
# ---------------------------------------------------------------------

def test_the_superseded_near_duplicate_screen_never_crossed_the_gate():
    """Retained record of the SUPERSEDED schema <= 6.5 method.

    ``near_neighbor_summary`` is no longer called by run_d2; it is kept so
    existing certificates stay readable, and its numerical behaviour is
    deliberately unchanged.  This test pins that behaviour and checks that
    the schema 7.0 containment markers are attached, so such a record can
    never again be read as a valid D2.1 result.

    `near_neighbor_summary` hardcodes
    statistically_excessive=False and records statistical_inference as
    "not_claimed", because the query-hit indicators share one reference set
    and are therefore dependent. Under the two-gate rule it can raise a
    warning but never a failure. That is conservative and declared, not a
    defect, but it means the near-duplicate component cannot be cited as
    statistical evidence of dependence.
    """
    rng = np.random.default_rng(61)
    x = _near_copy_at_lag1(4000, rng, radius=2)   # heavy near-duplicate load
    out = d2.near_neighbor_summary(x, (1, 2, 4, 8), rng=rng, query_count=500,
                                   projection_tables=4, null_pairs=20_000,
                                   alpha=0.05)
    assert out, "no radii reported"
    for radius, r in out.items():
        assert r["statistically_excessive"] is False, radius
        assert r["statistical_inference"].startswith("not_claimed"), radius
        assert r["exhaustive"] is False, radius
        assert r["method_status"] == "SUPERSEDED_METHOD", radius
    # And at least one radius is practically excessive on this data, so the
    # component is not simply inert.
    assert any(r["practically_excessive"] for r in out.values())


def test_a_near_duplicate_component_can_now_reach_failure():
    """The inverse of what this asserted before schema 7.0.

    Defect C was a hardcoded ``statistically_excessive: False``, which left
    D2.1 unable to contribute a FAIL at any effect size.  With the exact
    method the gate is live, so heavy near-duplicate contamination must now
    be able to reach FAILURE, not merely warn.

    This test also guards a vacuity trap: the record is nested under
    "by_radius" at schema 7.0, and iterating the record directly yields
    schema metadata keys instead of radii, which made the previous version of
    this assertion pass against names that could never match.
    """
    rng = np.random.default_rng(62)
    parts = {n: _near_copy_at_lag1(k, rng, radius=2) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    # The fixture plants pairs at distance EXACTLY 2, so radius 2 must detect
    # them and radius 1 must not.  Testing both directions is the point: the
    # default _run_d2 radii are (1,), under which this fixture is correctly
    # invisible -- which is the second reason the pre-7.0 version of this
    # assertion could never have failed.
    res, dec = _run_d2(parts, permutations=40_000, near_duplicate_radii=(1, 2))
    by_radius = {n: res["d2_1_near_duplicate_structure"][f"within:{n}"]["by_radius"]
                 for n in parts}
    near_names = {f"within:{n}:radius{r}" for n, rr in by_radius.items() for r in rr}
    # The names must be real radius components, not schema metadata keys.
    assert near_names
    assert all(nm.split("radius")[-1].isdigit() for nm in near_names)

    for n, rr in by_radius.items():
        assert rr["1"]["rows_with_neighbor_within_radius"] == 0, (n, "radius 1")
        assert rr["1"]["decision"] != "EXCESS_DETECTED", (n, "radius 1")
        assert rr["2"]["decision"] == "EXCESS_DETECTED", (n, "radius 2")

    reached = [f["component"] for f in dec["failures"]
               if f["component"] in near_names]
    assert reached, (
        "D2.1 did not reach FAILURE on heavy near-duplicate contamination, "
        "so its statistical gate is still unreachable")
    assert all(c.endswith("radius2") for c in reached), reached


def test_which_components_can_produce_a_failure_at_all():
    """Documents the real inferential surface of one invocation.

    Failure-capable: serial lag tests, structured collision views, the
    multivariate detector, and -- since schema 7.0 -- the near-duplicate
    screen, whose statistical gate was a hardcoded False before the exact
    method replaced it.  Not failure-capable: the pairwise components
    (descriptive only, excluded).
    """
    rng = np.random.default_rng(63)
    parts = {n: _exact_copy_at_lag1(k, rng) for n, k in
             (("train", 4000), ("validation", 2000), ("test", 2000))}
    res, dec = _run_d2(parts, permutations=40_000)
    assert dec["outcome"] == "FAIL"
    failure_types = {f["type"] for f in dec["failures"]}
    assert failure_types <= {"distributional", "structured_collision",
                             "multivariate_dependence",
                             "near_duplicate_structure"}, failure_types
    # The pairwise components are excluded, so "distributional" failures can
    # only have come from the lag tests.
    pairwise = set(res["d2_2_pairwise_structure"])
    for f in dec["failures"]:
        assert f["component"] not in pairwise
