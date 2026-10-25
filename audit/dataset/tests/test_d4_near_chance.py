"""
D4 rule near_chance_baseline_gated_v4: near-chance shuffled-label endpoint,
intact-label baseline-adequacy gate, composite outcome, secondary paired endpoint.

SOFTWARE VALIDATION, NOT SCIENTIFIC CALIBRATION.  These tests check that the
implementation computes the registered exact procedure: against an
independent rational-arithmetic reference on small enumerated cases, for the
exact level of each component test, on the decision semantics, and for
agreement between the runtime decision and independent certificate
recomputation.  The error guarantees themselves follow from exactness of the
binomial tails under the stated sampling assumptions (test rows i.i.d. from
the generator, independent of training).  test_generator_samples_labels_iid
checks the generator's construction; it cannot prove os.urandom's quality.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from math import comb

import numpy as np
import pytest

from audit.dataset import d4_controlled_perturbation as d4
from audit.dataset.integrity import REPOSITORY_ROOT, IntegrityError
from audit.dataset.tests.test_d4_engine_contract import CONFIG, ThresholdAdapter, data
from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

ALPHA, K = 0.05, 5
N, MARGIN, FLOOR = 1_000_000, 0.005, 0.90   # Gohr design size; candidate margin and baseline floor
# Acceptance region of the engine at (N, MARGIN, ALPHA, K), from the OC tool.
FIRST_EQ, LAST_EQ = 495823, 504177
FIRST_ADEQUATE, LAST_INADEQUATE = 900494, 899301
GOOD = 929_000                               # an adequate baseline (published 5-round level)


def state(x, n=N, margin=MARGIN, k=K):
    return d4.near_chance_replicate(x, n, chance=0.5, margin=margin, alpha=ALPHA, replicates=k)["state"]


def primary(xs, n=N, margin=MARGIN, planned=None):
    rows = [{"test_examples": n, "perturbed_correct": x} for x in xs]
    return d4.evaluate_near_chance(rows, planned_replicates=planned or len(xs), chance=0.5, margin=margin,
                                   alpha=ALPHA)


def bstate(x, n=N, floor=FLOOR, k=K):
    return d4.baseline_adequacy_replicate(x, n, threshold=floor, alpha=ALPHA, replicates=k)["state"]


def decision(xs, pairs, bases=None, n=N, margin=MARGIN, planned=None):
    bases = bases or [GOOD] * len(xs)
    rows = [{"test_examples": n, "perturbed_correct": x, "baseline_correct": base, "mcnemar_b": b,
             "mcnemar_c": c} for x, base, (b, c) in zip(xs, bases, pairs)]
    return d4.evaluate_near_chance_decision(rows, planned_replicates=planned or len(xs), chance=0.5,
                                            margin=margin, alpha=ALPHA, effect_threshold=0.01,
                                            baseline_threshold=FLOOR, paired_endpoint_role="secondary")


DEGRADED = [(10_000, 439_000)] * 5           # baseline 0.929 -> shuffled 0.50: paired degradation holds


# ============================================================ independent exact reference


def ref_upper(x, n, p):            # P(X >= x), exact rational
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x, n + 1))


def ref_lower(x, n, p):            # P(X <= x), exact rational
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))


def ref_state(x, n, margin: Fraction, alpha: Fraction, k):
    low, high = Fraction(1, 2) - margin, Fraction(1, 2) + margin
    if ref_upper(x, n, low) < alpha and ref_lower(x, n, high) < alpha:
        return "EQUIVALENT"
    if ref_upper(x, n, high) < alpha / (2 * k):
        return "ABOVE_BAND"
    if ref_lower(x, n, low) < alpha / (2 * k):
        return "BELOW_BAND"
    return "INDETERMINATE"


def near_tie(x, n, margin: Fraction, alpha: Fraction, k):
    """True if any reference p-value is within 1e-9 of its level (float/rational may then differ)."""
    low, high = Fraction(1, 2) - margin, Fraction(1, 2) + margin
    values = [(ref_upper(x, n, low), alpha), (ref_lower(x, n, high), alpha),
              (ref_upper(x, n, high), alpha / (2 * k)), (ref_lower(x, n, low), alpha / (2 * k))]
    return any(abs(float(v - level)) < 1e-9 for v, level in values)


@pytest.mark.parametrize("n, margin", [(40, Fraction(1, 5)), (100, Fraction(1, 10)), (101, Fraction(3, 20)),
                                       (60, Fraction(1, 20))])
def test_states_match_exact_rational_reference_for_every_outcome(n, margin):
    checked = 0
    for x in range(n + 1):
        if near_tie(x, n, margin, Fraction(1, 20), K):
            continue
        assert state(x, n=n, margin=float(margin)) == ref_state(x, n, margin, Fraction(1, 20), K), x
        checked += 1
    assert checked >= n - 2


@pytest.mark.parametrize("n, margin", [(40, 0.2), (100, 0.1), (1000, 0.05)])
def test_equivalence_is_dual_to_clopper_pearson_interval_inside_band(n, margin):
    for x in range(n + 1):
        row = d4.near_chance_replicate(x, n, chance=0.5, margin=margin, alpha=ALPHA, replicates=K)
        lo, hi = row["equivalence_interval"]
        inside = (0.5 - margin) < lo and hi < (0.5 + margin)
        if min(abs(lo - (0.5 - margin)), abs(hi - (0.5 + margin))) > 1e-9:
            assert (row["state"] == "EQUIVALENT") == inside, x


@pytest.mark.parametrize("n, margin", [(40, 0.2), (100, 0.1), (400, 0.05)])
def test_exact_level_of_each_component_test(n, margin):
    """Size of EQUIVALENT at the band edges <= alpha; departure size inside the band <= alpha/k."""
    states = [state(x, n=n, margin=margin) for x in range(n + 1)]
    pmf = lambda p: np.array([comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])
    eq = np.array([s == "EQUIVALENT" for s in states])
    depart = np.array([s in ("ABOVE_BAND", "BELOW_BAND") for s in states])
    for p in (0.5 - margin, 0.5 + margin):
        assert float(pmf(p)[eq].sum()) <= ALPHA + 1e-12
    for p in np.linspace(0.5 - margin, 0.5 + margin, 41):
        assert float(pmf(p)[depart].sum()) <= ALPHA / K + 1e-12


@pytest.mark.parametrize("n, margin", [(40, 0.2), (100, 0.1), (1000, 0.01)])
def test_equivalence_and_departure_are_mutually_exclusive_for_every_outcome(n, margin):
    for x in range(n + 1):
        assert state(x, n=n, margin=margin) in ("EQUIVALENT", "ABOVE_BAND", "BELOW_BAND", "INDETERMINATE")


def ref_bstate(x, n, floor: Fraction, alpha: Fraction, k):
    if ref_upper(x, n, floor) < alpha:
        return "ADEQUATE"
    if ref_lower(x, n, floor) < alpha / k:
        return "INADEQUATE"
    return "INDETERMINATE"


@pytest.mark.parametrize("n, floor", [(40, Fraction(9, 10)), (100, Fraction(4, 5)), (101, Fraction(3, 4))])
def test_baseline_states_match_exact_rational_reference(n, floor):
    alpha = Fraction(1, 20)
    for x in range(n + 1):
        tails = [(ref_upper(x, n, floor), alpha), (ref_lower(x, n, floor), alpha / K)]
        if any(abs(float(v - level)) < 1e-9 for v, level in tails):
            continue
        assert bstate(x, n=n, floor=float(floor)) == ref_bstate(x, n, floor, alpha, K), x


@pytest.mark.parametrize("n, floor", [(40, 0.9), (100, 0.8), (400, 0.9)])
def test_exact_level_of_the_baseline_tests(n, floor):
    states = [bstate(x, n=n, floor=floor) for x in range(n + 1)]
    pmf = lambda q: np.array([comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(n + 1)])
    adequate = np.array([s == "ADEQUATE" for s in states])
    inadequate = np.array([s == "INADEQUATE" for s in states])
    for q in np.linspace(floor - 0.2, floor, 21):          # H0 of adequacy: q <= floor
        assert float(pmf(q)[adequate].sum()) <= ALPHA + 1e-12
    for q in np.linspace(floor, min(0.999, floor + 0.09), 10):   # H0 of inadequacy: q >= floor
        assert float(pmf(q)[inadequate].sum()) <= ALPHA / K + 1e-12


@pytest.mark.parametrize("n", [40, 100, 1000])
def test_baseline_lower_bound_is_dual_to_the_adequacy_test(n):
    for x in range(n + 1):
        row = d4.baseline_adequacy_replicate(x, n, threshold=0.9, alpha=ALPHA, replicates=K)
        if abs(row["one_sided_lower_bound"] - 0.9) > 1e-9:
            assert (row["state"] == "ADEQUATE") == (row["one_sided_lower_bound"] > 0.9), x


# ============================================================ sampling design


def test_generator_samples_labels_iid_and_does_not_fix_class_counts():
    """speck.make_train_data draws each label as an independent urandom bit (no fixed class counts)."""
    import inspect
    import speck

    source = inspect.getsource(speck.make_train_data)
    assert "Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;" in source
    assert "keys = np.frombuffer(urandom(8*n)" in source        # a fresh key per row
    counts = {int(np.sum(speck.make_train_data(64, 5)[1])) for _ in range(30)}
    assert len(counts) > 1                                       # class count varies between calls


def test_class_conditional_diagnostics_are_descriptive_counts():
    labels = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
    correct = np.array([1, 0, 1, 1, 0], dtype=bool)
    assert d4.class_diagnostics(correct, labels) == {
        "label_0": {"rows": 3, "correct": 2, "true_negative_rate": 2 / 3},
        "label_1": {"rows": 2, "correct": 1, "true_positive_rate": 0.5}}


# ============================================================ required cases at the Gohr design size


def test_exactly_chance_is_equivalent_and_all_five_support():
    assert state(N // 2) == "EQUIVALENT"
    assert primary([N // 2] * 5)["outcome"] == "SUPPORTED"


def test_well_inside_band_is_equivalent():
    assert state(502_000) == "EQUIVALENT" and state(498_000) == "EQUIVALENT"


def test_acceptance_edges_near_each_band_boundary():
    assert state(LAST_EQ) == "EQUIVALENT" and state(LAST_EQ + 1) == "INDETERMINATE"
    assert state(FIRST_EQ) == "EQUIVALENT" and state(FIRST_EQ - 1) == "INDETERMINATE"


def test_interval_crossing_the_boundary_is_inconclusive():
    row = d4.near_chance_replicate(505_000, N, chance=0.5, margin=MARGIN, alpha=ALPHA, replicates=K)
    assert row["equivalence_interval"][0] < 0.505 < row["equivalence_interval"][1]
    assert row["state"] == "INDETERMINATE"
    assert primary([N // 2] * 4 + [505_000])["outcome"] == "INCONCLUSIVE"


def test_meaningfully_above_or_below_band_is_not_supported():
    assert state(520_000) == "ABOVE_BAND" and state(480_000) == "BELOW_BAND"
    assert primary([N // 2] * 4 + [520_000])["outcome"] == "NOT_SUPPORTED"
    assert primary([N // 2] * 4 + [480_000])["outcome"] == "NOT_SUPPORTED"
    assert state(514_000) == "ABOVE_BAND"          # Gohr 8-round strength is outside the band


def test_worse_than_baseline_but_far_from_chance_does_not_establish_the_primary_claim():
    # baseline 0.929, shuffled 0.70: b=10_000, c=239_000 -> delta = -0.229 on n = 10^6
    result = decision([700_000] * 5, [(10_000, 239_000)] * 5)
    assert result["supporting"]["established"] is True
    assert result["primary"]["outcome"] == "NOT_SUPPORTED"
    assert result["outcome"] == "NOT_SUPPORTED" and result["overall_supported"] is False


def test_at_chance_without_paired_degradation_shows_the_endpoints_are_distinct():
    # adequate baselines, shuffled at chance, but the paired endpoint is not established (b = c)
    result = decision([N // 2] * 5, [(100_000, 100_000)] * 5)
    assert result["primary"]["outcome"] == "SUPPORTED" and result["baseline"]["outcome"] == "PASSED"
    assert result["supporting"]["established"] is False
    assert result["outcome"] == "SUPPORTED"          # paired endpoint is secondary and never gates


def test_supported_needs_near_chance_and_adequate_baselines():
    result = decision([N // 2] * 5, DEGRADED)
    assert result["outcome"] == "SUPPORTED" and result["overall_supported"] is True


@pytest.mark.parametrize("bases, gate, overall", [
    ([GOOD] * 4 + [LAST_INADEQUATE], "FAILED", "BASELINE_INADEQUATE"),         # demonstrably below the floor
    ([GOOD] * 4 + [FIRST_ADEQUATE - 1], "INDETERMINATE", "INCONCLUSIVE"),      # adequacy not established
    ([N // 2] * 5, "FAILED", "BASELINE_INADEQUATE"),                           # baseline also at chance
])
def test_near_chance_primary_cannot_be_supported_without_adequate_baselines(bases, gate, overall):
    result = decision([N // 2] * 5, DEGRADED, bases=bases)
    assert result["primary"]["outcome"] == "SUPPORTED"
    assert result["baseline"]["outcome"] == gate
    assert result["outcome"] == overall and result["overall_supported"] is False


def test_shuffled_departure_is_not_supported_whatever_the_baseline():
    for bases in ([GOOD] * 5, [N // 2] * 5, [GOOD] * 4 + [FIRST_ADEQUATE - 1]):
        assert decision([N // 2] * 4 + [520_000], DEGRADED, bases=bases)["outcome"] == "NOT_SUPPORTED"


def test_indeterminate_primary_with_adequate_baselines_is_inconclusive():
    assert decision([N // 2] * 4 + [505_000], DEGRADED)["outcome"] == "INCONCLUSIVE"


def test_baseline_states_at_the_gohr_design_size():
    assert bstate(GOOD) == "ADEQUATE"
    assert bstate(FIRST_ADEQUATE) == "ADEQUATE" and bstate(FIRST_ADEQUATE - 1) == "INDETERMINATE"
    assert bstate(LAST_INADEQUATE) == "INADEQUATE" and bstate(LAST_INADEQUATE + 1) == "INDETERMINATE"
    assert bstate(900_000) == "INDETERMINATE"      # at the floor: neither established
    row = d4.baseline_adequacy_replicate(FIRST_ADEQUATE, N, threshold=FLOOR, alpha=ALPHA, replicates=K)
    assert row["one_sided_lower_bound"] > FLOOR
    row = d4.baseline_adequacy_replicate(FIRST_ADEQUATE - 1, N, threshold=FLOOR, alpha=ALPHA, replicates=K)
    assert row["one_sided_lower_bound"] <= FLOOR + 1e-12


def test_baseline_gate_needs_every_fit():
    rows = [{"test_examples": N, "baseline_correct": GOOD}] * 5
    assert d4.evaluate_baseline_adequacy(rows, planned_replicates=5, threshold=FLOOR, alpha=ALPHA)["outcome"] == "PASSED"
    rows = rows[:4] + [{"test_examples": N, "baseline_correct": FIRST_ADEQUATE - 1}]
    assert d4.evaluate_baseline_adequacy(rows, planned_replicates=5, threshold=FLOOR,
                                         alpha=ALPHA)["outcome"] == "INDETERMINATE"


def test_missing_fit_is_not_completed_and_never_a_pass():
    result = primary([N // 2] * 4, planned=5)
    assert result["outcome"] == "NOT_COMPLETED"
    full = decision([N // 2] * 4, DEGRADED[:4], planned=5)
    assert full["outcome"] == "NOT_COMPLETED" and full["overall_supported"] is False
    assert full["baseline"]["outcome"] == "NOT_COMPLETED" and full["supporting"] is None


@pytest.mark.parametrize("primary_outcome, gate, overall", [
    ("SUPPORTED", "PASSED", "SUPPORTED"), ("SUPPORTED", "FAILED", "BASELINE_INADEQUATE"),
    ("SUPPORTED", "INDETERMINATE", "INCONCLUSIVE"), ("INCONCLUSIVE", "PASSED", "INCONCLUSIVE"),
    ("INCONCLUSIVE", "FAILED", "BASELINE_INADEQUATE"), ("INCONCLUSIVE", "INDETERMINATE", "INCONCLUSIVE"),
    ("NOT_SUPPORTED", "PASSED", "NOT_SUPPORTED"), ("NOT_SUPPORTED", "FAILED", "NOT_SUPPORTED"),
    ("NOT_SUPPORTED", "INDETERMINATE", "NOT_SUPPORTED"), ("NOT_COMPLETED", "PASSED", "NOT_COMPLETED"),
    ("SUPPORTED", "NOT_COMPLETED", "NOT_COMPLETED"),
])
def test_composite_rule_truth_table(primary_outcome, gate, overall):
    assert d4.combine_outcome(primary_outcome, gate) == overall
    assert (overall == "SUPPORTED") is (primary_outcome == "SUPPORTED" and gate == "PASSED")


@pytest.mark.parametrize("xs, bases", [([N // 2] * 4 + [505_000], None), ([N // 2] * 4 + [520_000], None),
                                       ([N // 2] * 5, [GOOD] * 4 + [FIRST_ADEQUATE - 1]),
                                       ([N // 2] * 5, [N // 2] * 5)])
def test_inconclusive_failed_and_inadequate_are_never_a_pass(xs, bases):
    assert decision(xs, DEGRADED, bases=bases)["overall_supported"] is False


def test_non_rejection_of_p_equal_half_is_not_used_as_equivalence():
    # n = 40, x = 20: a test of p = 0.5 would not reject, yet the interval is far too wide for a 0.05 band.
    assert state(20, n=40, margin=0.05) == "INDETERMINATE"


# ============================================================ McNemar orientation and edge cases


def test_mcnemar_discordance_orientation_and_direction():
    baseline = np.array([1] * 30 + [0] * 5 + [1] * 10 + [0] * 5, dtype=bool)
    shuffled = np.array([0] * 30 + [1] * 5 + [1] * 10 + [0] * 5, dtype=bool)
    b, c, _ = d4.mcnemar_exact_pvalue(baseline, shuffled)
    assert (b, c) == (5, 30)                     # b: baseline wrong & perturbed right
    assert shuffled.mean() - baseline.mean() == pytest.approx((b - c) / len(baseline))
    assert d4.mcnemar_one_sided_pvalue(b, c) < 1e-4           # perturbed worse: significant
    assert d4.mcnemar_one_sided_pvalue(c, b) > 0.99           # reversed: perturbed better: not
    expected = sum(comb(35, i) for i in range(0, 6)) / 2 ** 35
    assert d4.mcnemar_one_sided_pvalue(b, c) == pytest.approx(expected, rel=1e-12)


def test_all_concordant_zero_discordance():
    paired = d4.evaluate_paired_degradation([{"mcnemar_b": 0, "mcnemar_c": 0, "test_examples": 100}],
                                            alpha=ALPHA, effect_threshold=0.01)
    row = paired["replicates"][0]
    assert row["one_sided_pvalue"] == 1.0 and row["two_sided_pvalue"] == 1.0
    assert row["delta"] == 0.0 and row["conditional_delta_interval"] == [0.0, 0.0]
    assert paired["established"] is False


def test_significant_but_below_practical_threshold_is_not_established():
    # n = 10^6, b = 0, c = 5000: p tiny but delta = -0.005 > -0.01
    paired = d4.evaluate_paired_degradation([{"mcnemar_b": 0, "mcnemar_c": 5000, "test_examples": N}],
                                            alpha=ALPHA, effect_threshold=0.01)
    assert paired["replicates"][0]["significant"] and not paired["established"]


# ============================================================ runtime versus independent recomputation


def run_v4(tmp_path, margin=0.15, floor=0.9, replicates=3):
    x, y = data()
    tx, ty = data(seed=4)
    return d4.run_d4(perturbation=LabelShufflePerturbation(), train_features=x, train_labels=y,
                     test_features=tx, test_labels=ty, adapter_factory=ThresholdAdapter,
                     checkpoint_root=tmp_path / "fits", replicates=replicates, audit_seed=0,
                     effect_threshold=0.01, bootstrap_replicates=1000, alpha=ALPHA, run_config_hash=CONFIG,
                     decision_rule=d4.NEAR_CHANCE_RULE, chance_margin=margin, paired_endpoint_role="secondary",
                     baseline_adequacy_threshold=floor)


def certificate_v4(result):
    certificate = d4.generate_certificate(result, dataset_id="synthetic", dataset_version="1",
                                          generation_procedure="numpy", generation_parameters={},
                                          audit_seed=0, confidence_level=0.95, alpha=ALPHA)
    return json.loads(json.dumps(certificate))      # exactly what is persisted and re-read


@pytest.mark.parametrize("margin, floor, overall", [(0.15, 0.9, "SUPPORTED"), (0.01, 0.9, "INCONCLUSIVE"),
                                                    (0.15, 0.999, "INCONCLUSIVE")])
def test_runtime_decision_equals_independent_certificate_recomputation(tmp_path, margin, floor, overall):
    result = run_v4(tmp_path, margin=margin, floor=floor)
    certificate = certificate_v4(result)
    recomputed = d4.recompute_certificate_decision(certificate)
    assert recomputed["outcome"] == result.decision["outcome"] == certificate["decision"]["outcome"] == overall
    assert recomputed["primary"] == result.decision["primary"]
    assert recomputed["baseline"] == result.decision["baseline"]
    decision = certificate["decision"]
    assert decision["overall_d4_outcome"] == overall and decision["overall_supported"] is (overall == "SUPPORTED")
    assert decision["paired_degradation_endpoint"]["role"] == "secondary_non_gating"
    assert {"shuffled_near_chance_endpoint", "baseline_adequacy_gate"} <= set(decision)
    for row in certificate["findings"]["baseline_adequacy"]["replicates"]:
        assert set(row) >= {"accuracy", "one_sided_lower_bound", "pvalue_h0_at_or_below_threshold", "state"}
    assert certificate["schema_version"] == d4.NEAR_CHANCE_CERTIFICATE_SCHEMA
    for row in certificate["findings"]["primary_endpoint"]["replicates"]:
        assert set(row) >= {"accuracy", "equivalence_interval", "departure_interval", "state", "correct"}
    for row, fit in zip(certificate["findings"]["replicates"], result.replicates):
        assert row["perturbed_accuracy"] == row["perturbed_correct"] / row["test_examples"]
        assert fit.perturbed_predictions["sha256"] and fit.baseline_predictions["sha256"]
        counts = row["baseline_predictions"]["class_counts"]
        assert counts["label_0"]["correct"] + counts["label_1"]["correct"] == row["baseline_correct"]


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["decision"].update(outcome="SUPPORTED", overall_d4_outcome="SUPPORTED", overall_supported=True,
                                    shuffled_near_chance_endpoint="SUPPORTED"), "does not match the recomputed"),
    (lambda c: c["decision"].update(baseline_adequacy_gate="FAILED"), "does not match the recomputed"),
    (lambda c: c["findings"]["baseline_adequacy"]["replicates"][0].update(state="INADEQUATE"), "endpoint"),
    (lambda c: (c["settings"].update(baseline_adequacy_threshold=0.6),
                c["methodology"].update(baseline_adequacy_threshold=0.6)), "endpoint"),
    (lambda c: c["methodology"]["overall_rule"].update(SUPPORTED="primary SUPPORTED only"), "procedure"),
    (lambda c: c["findings"]["replicates"][0]["baseline_predictions"]["class_counts"]["label_0"].update(correct=0),
     "class counts|true_negative_rate"),
    (lambda c: c["findings"]["replicates"][0].update(perturbed_correct=c["findings"]["replicates"][0]
                                                     ["perturbed_correct"] + 1), "2x2|counts"),
    (lambda c: (c["settings"].update(chance_margin=0.3), c["methodology"].update(chance_margin=0.3)), "endpoint"),
    (lambda c: c["settings"].update(chance_margin=0.3), "methodology chance_margin"),
    (lambda c: c["methodology"]["primary_procedure"].update(departure_test="two_sided_at_alpha"), "procedure"),
    (lambda c: c["findings"]["primary_endpoint"]["replicates"][0].update(state="EQUIVALENT"), "endpoint"),
    (lambda c: c["findings"]["replicates"].pop(), "replicate rows"),
    (lambda c: c["settings"].update(decision_rule="directional_iut_v2"), "schema and decision rule"),
    (lambda c: c.update(interpretation=d4.NEAR_CHANCE_INTERPRETATION["SUPPORTED"]), "interpretation"),
])
def test_v4_certificate_tampering_is_detected(tmp_path, mutate, message):
    certificate = certificate_v4(run_v4(tmp_path, margin=0.01))     # tiny margin: INCONCLUSIVE at n = 600
    assert certificate["decision"]["outcome"] != "SUPPORTED"
    assert certificate["decision"]["baseline_adequacy_gate"] == "PASSED"
    before = json.dumps(certificate, sort_keys=True)
    mutate(certificate)
    assert json.dumps(certificate, sort_keys=True) != before        # every case is a real tamper
    with pytest.raises(IntegrityError, match=message):
        d4.recompute_certificate_decision(certificate)


@pytest.mark.parametrize("kwargs, match", [
    ({"chance_margin": None, "paired_endpoint_role": "secondary"}, "chance_margin"),
    ({"chance_margin": 0.0, "paired_endpoint_role": "secondary"}, "chance_margin"),
    ({"chance_margin": 0.6, "paired_endpoint_role": "secondary"}, "chance_margin"),
    ({"chance_margin": 0.01, "paired_endpoint_role": None}, "paired_endpoint_role"),
    ({"chance_margin": 0.01, "paired_endpoint_role": "co_primary"}, "paired_endpoint_role"),
    ({"chance_margin": 0.01, "paired_endpoint_role": "secondary", "baseline_adequacy_threshold": None},
     "baseline_adequacy_threshold"),
    ({"chance_margin": 0.01, "paired_endpoint_role": "secondary", "baseline_adequacy_threshold": 0.5},
     "baseline_adequacy_threshold"),
])
def test_run_refuses_unregistered_margin_floor_or_role(tmp_path, kwargs, match):
    kwargs = {"baseline_adequacy_threshold": 0.9, **kwargs}
    x, y = data()
    with pytest.raises(ValueError, match=match):
        d4.run_d4(perturbation=LabelShufflePerturbation(), train_features=x, train_labels=y, test_features=x,
                  test_labels=y, adapter_factory=ThresholdAdapter, checkpoint_root=tmp_path, replicates=2,
                  audit_seed=0, effect_threshold=0.01, run_config_hash=CONFIG,
                  decision_rule=d4.NEAR_CHANCE_RULE, **kwargs)


def test_legacy_evaluator_refuses_the_near_chance_rule():
    with pytest.raises(ValueError, match="evaluate_near_chance_decision"):
        d4.evaluate_decision([], rule=d4.NEAR_CHANCE_RULE, alpha=ALPHA, effect_threshold=0.01, audit_seed=0,
                             bootstrap_replicates=1000, confidence_level=0.95)


def test_resumed_run_reuses_agreeing_predictions_and_refuses_disagreeing_ones(tmp_path):
    first = run_v4(tmp_path)
    again = run_v4(tmp_path)
    assert [r.perturbed_predictions for r in again.replicates] == [r.perturbed_predictions for r in first.replicates]
    path = d4.prediction_path(tmp_path / "fits", 1, "baseline")
    np.save(path, 1.0 - np.load(path))
    with pytest.raises(IntegrityError, match="disagree"):
        run_v4(tmp_path)


# ============================================================ preregistration gate


def shipped():
    return json.loads((REPOSITORY_ROOT / "audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json").read_text())


def approve_all_but(record, *left_pending):
    record = copy.deepcopy(record)
    for name, entry in record["decisions"].items():
        if name not in left_pending:
            entry.update(status="APPROVED", value=entry["recommended"])
    record.update(approved=True, approved_by="tester", approved_utc="2026-01-01T00:00:00Z",
                  approval_statement="s", data_examined_before_approval="d")
    return record


def test_shipped_draft_registers_v4_and_leaves_every_scientific_choice_pending():
    record = shipped()
    review = d4.review_preregistration(record)
    assert review["problems"] == []
    assert {"decisions.chance_margin", "decisions.baseline_adequacy_threshold", "decisions.paired_endpoint_role",
            "decisions.planning_source_attestation"} <= set(review["pending"])
    assert record["decisions"]["baseline_adequacy_threshold"]["value"] is None
    assert record["decisions"]["baseline_adequacy_threshold"]["recommended"] == 0.90
    assert record["decisions"]["chance_margin"]["recommended"] == 0.005
    assert record["decisions"]["paired_endpoint_role"]["options"] == ["secondary"]
    assert record["fixed_design"]["baseline_adequacy_endpoint"]["procedure"] == d4.BASELINE_ADEQUACY_PROCEDURE
    assert record["fixed_design"]["overall_rule"] == d4.OVERALL_RULE
    assert record["decisions"]["decision_rule"]["options"] == [d4.NEAR_CHANCE_RULE]
    assert record["decisions"]["chance_margin"]["value"] is None
    assert record["decisions"]["chance_margin"]["units"] == "accuracy_fraction"
    assert record["fixed_design"]["primary_endpoint"]["procedure"] == d4.NEAR_CHANCE_PROCEDURE
    assert record["fixed_design"]["supporting_endpoint"]["procedure"] == d4.PAIRED_PROCEDURE


@pytest.mark.parametrize("left", ["chance_margin", "baseline_adequacy_threshold", "paired_endpoint_role",
                                  "planning_source_attestation", "decision_rule", "alpha"])
def test_require_approved_blocks_while_any_required_decision_is_pending(left):
    with pytest.raises(IntegrityError, match=f"decisions.{left}"):
        d4.require_approved_preregistration(approve_all_but(shipped(), left))


def test_fully_approved_draft_passes_the_engine_and_runner_checks():
    from audit.dataset.experiments import d4_gohr_label_shuffle as driver
    record = approve_all_but(shipped())
    values = d4.require_approved_preregistration(record)
    assert driver.check_target_design(record) == [] and driver.check_decision_values(values) == []


@pytest.mark.parametrize("edit, match", [
    (lambda r: r["decisions"].pop("chance_margin"), "chance_margin' is missing"),
    (lambda r: r["decisions"]["chance_margin"].pop("units"), "units"),
    (lambda r: r["decisions"]["chance_margin"].update(value=0.7), "chance_margin"),
    (lambda r: r["decisions"]["chance_margin"].update(value=5), "chance_margin"),
    (lambda r: r["decisions"]["paired_endpoint_role"].update(value="co_primary"), "paired_endpoint_role"),
    (lambda r: r["decisions"].pop("baseline_adequacy_threshold"), "baseline_adequacy_threshold' is missing"),
    (lambda r: r["decisions"]["baseline_adequacy_threshold"].pop("units"), "units"),
    (lambda r: r["decisions"]["baseline_adequacy_threshold"].update(value=0.4), "baseline_adequacy_threshold"),
    (lambda r: r["decisions"]["baseline_adequacy_threshold"].update(value=1), "baseline_adequacy_threshold"),
    (lambda r: r["decisions"]["decision_rule"].update(value="directional_iut_v2"), "decision_rule"),
])
def test_malformed_margin_role_or_rule_is_refused(edit, match):
    record = approve_all_but(shipped())
    edit(record)
    review = d4.review_preregistration(record)
    from audit.dataset.experiments import d4_gohr_label_shuffle as driver
    problems = review["problems"] + driver.check_decision_values(review["values"])
    assert any(match.split("|")[0] in p for p in problems), problems


@pytest.mark.parametrize("section, key, value", [
    ("primary_endpoint", "equivalence_test", "exact_binomial_tost_each_one_sided_test_at_2alpha"),
    ("primary_endpoint", "departure_test", "exact_two_sided_binomial_at_alpha"),
    ("primary_endpoint", "chance_reference", 0.55),
    ("supporting_endpoint", "test", "exact_mcnemar_two_sided"),
    ("supporting_endpoint", "aggregation", "any_replicate_degrades"),
    ("supporting_endpoint", "role", "gating"),
    ("baseline_adequacy_endpoint", "adequacy_test", "exact_two_sided_binomial"),
    ("baseline_adequacy_endpoint", "adequacy_aggregation", "majority_of_baselines_adequate"),
])
def test_runner_refuses_a_changed_endpoint_procedure(section, key, value):
    from audit.dataset.experiments import d4_gohr_label_shuffle as driver
    record = approve_all_but(shipped())
    record["fixed_design"][section]["procedure"][key] = value
    assert any(section in p for p in driver.check_target_design(record))
