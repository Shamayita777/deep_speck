"""
D4 rule near_chance_tost_v3: near-chance primary endpoint, paired supporting endpoint.

SOFTWARE VALIDATION, NOT SCIENTIFIC CALIBRATION.  These tests check that the
implementation computes the registered exact procedure: against an
independent rational-arithmetic reference on small enumerated cases, for the
exact level of each component test, on the decision semantics, and for
agreement between the runtime decision and independent certificate
recomputation.  The error guarantees themselves follow from exactness of the
binomial tails under the stated sampling assumptions (test rows i.i.d. from
the generator, independent of training); no test here establishes those
assumptions.
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
N, MARGIN = 1_000_000, 0.005          # Gohr design size; the candidate margin
# Acceptance region of the engine at (N, MARGIN, ALPHA, K), from the OC tool.
FIRST_EQ, LAST_EQ = 495823, 504177


def state(x, n=N, margin=MARGIN, k=K):
    return d4.near_chance_replicate(x, n, chance=0.5, margin=margin, alpha=ALPHA, replicates=k)["state"]


def primary(xs, n=N, margin=MARGIN, planned=None):
    rows = [{"test_examples": n, "perturbed_correct": x} for x in xs]
    return d4.evaluate_near_chance(rows, planned_replicates=planned or len(xs), chance=0.5, margin=margin,
                                   alpha=ALPHA)


def decision(xs, pairs, role, n=N, margin=MARGIN):
    rows = [{"test_examples": n, "perturbed_correct": x, "mcnemar_b": b, "mcnemar_c": c}
            for x, (b, c) in zip(xs, pairs)]
    return d4.evaluate_near_chance_decision(rows, planned_replicates=len(xs), chance=0.5, margin=margin,
                                            alpha=ALPHA, effect_threshold=0.01, paired_endpoint_role=role)


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
    # baseline 0.93, shuffled 0.70: b=10_000, c=240_000 -> delta = -0.23 on n = 10^6
    pairs = [(10_000, 240_000)] * 5
    for role in d4.PAIRED_ENDPOINT_ROLES:
        result = decision([700_000] * 5, pairs, role)
        assert result["supporting"]["established"] is True
        assert result["primary"]["outcome"] == "NOT_SUPPORTED"
        assert result["outcome"] == "NOT_SUPPORTED" and result["primary_claim_supported"] is False


def test_at_chance_without_paired_degradation_shows_the_endpoints_are_distinct():
    pairs = [(100_000, 100_000)] * 5           # baseline also at chance: no degradation
    secondary = decision([N // 2] * 5, pairs, "secondary")
    assert secondary["primary"]["outcome"] == "SUPPORTED" and secondary["supporting"]["established"] is False
    assert secondary["outcome"] == "SUPPORTED"
    co = decision([N // 2] * 5, pairs, "co_primary")
    assert co["outcome"] == "INCONCLUSIVE" and co["primary_claim_supported"] is False


def test_co_primary_supported_needs_both_endpoints():
    pairs = [(10_000, 440_000)] * 5            # baseline 0.93 -> shuffled 0.50
    assert decision([N // 2] * 5, pairs, "co_primary")["outcome"] == "SUPPORTED"


def test_every_one_of_five_fits_is_required():
    assert primary([N // 2] * 4 + [LAST_EQ + 1])["outcome"] == "INCONCLUSIVE"
    assert primary([N // 2] * 5)["outcome"] == "SUPPORTED"


def test_missing_fit_is_not_completed_and_never_a_pass():
    result = primary([N // 2] * 4, planned=5)
    assert result["outcome"] == "NOT_COMPLETED"
    rows = [{"test_examples": N, "perturbed_correct": N // 2, "mcnemar_b": 1, "mcnemar_c": 400_000}] * 4
    full = d4.evaluate_near_chance_decision(rows, planned_replicates=5, chance=0.5, margin=MARGIN, alpha=ALPHA,
                                            effect_threshold=0.01, paired_endpoint_role="secondary")
    assert full["outcome"] == "NOT_COMPLETED" and full["primary_claim_supported"] is False


@pytest.mark.parametrize("xs", [[N // 2] * 4 + [505_000], [N // 2] * 4 + [520_000]])
def test_inconclusive_and_not_supported_are_never_a_pass(xs):
    for role in d4.PAIRED_ENDPOINT_ROLES:
        assert decision(xs, [(10_000, 440_000)] * 5, role)["primary_claim_supported"] is False


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


def run_v3(tmp_path, margin=0.15, role="secondary", replicates=3):
    x, y = data()
    tx, ty = data(seed=4)
    return d4.run_d4(perturbation=LabelShufflePerturbation(), train_features=x, train_labels=y,
                     test_features=tx, test_labels=ty, adapter_factory=ThresholdAdapter,
                     checkpoint_root=tmp_path / "fits", replicates=replicates, audit_seed=0,
                     effect_threshold=0.01, bootstrap_replicates=1000, alpha=ALPHA, run_config_hash=CONFIG,
                     decision_rule=d4.NEAR_CHANCE_RULE, chance_margin=margin, paired_endpoint_role=role)


def certificate_v3(result):
    certificate = d4.generate_certificate(result, dataset_id="synthetic", dataset_version="1",
                                          generation_procedure="numpy", generation_parameters={},
                                          audit_seed=0, confidence_level=0.95, alpha=ALPHA)
    return json.loads(json.dumps(certificate))      # exactly what is persisted and re-read


@pytest.mark.parametrize("role", d4.PAIRED_ENDPOINT_ROLES)
def test_runtime_decision_equals_independent_certificate_recomputation(tmp_path, role):
    result = run_v3(tmp_path, role=role)
    certificate = certificate_v3(result)
    recomputed = d4.recompute_certificate_decision(certificate)
    assert recomputed["outcome"] == result.decision["outcome"] == certificate["decision"]["outcome"]
    assert recomputed["primary"] == result.decision["primary"]
    assert certificate["schema_version"] == d4.NEAR_CHANCE_CERTIFICATE_SCHEMA
    for row in certificate["findings"]["primary_endpoint"]["replicates"]:
        assert set(row) >= {"accuracy", "equivalence_interval", "departure_interval", "state", "correct"}
    for row, fit in zip(certificate["findings"]["replicates"], result.replicates):
        assert row["perturbed_accuracy"] == row["perturbed_correct"] / row["test_examples"]
        assert fit.perturbed_predictions["sha256"] and fit.baseline_predictions["sha256"]


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["decision"].update(outcome="SUPPORTED", primary_claim_supported=True, primary_outcome="SUPPORTED"),
     "does not match the recomputed"),
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
def test_v3_certificate_tampering_is_detected(tmp_path, mutate, message):
    certificate = certificate_v3(run_v3(tmp_path, margin=0.01))     # tiny margin: INCONCLUSIVE at n = 600
    assert certificate["decision"]["outcome"] != "SUPPORTED"
    mutate(certificate)
    with pytest.raises(IntegrityError, match=message):
        d4.recompute_certificate_decision(certificate)


@pytest.mark.parametrize("kwargs, match", [
    ({"chance_margin": None, "paired_endpoint_role": "secondary"}, "chance_margin"),
    ({"chance_margin": 0.0, "paired_endpoint_role": "secondary"}, "chance_margin"),
    ({"chance_margin": 0.6, "paired_endpoint_role": "secondary"}, "chance_margin"),
    ({"chance_margin": 0.01, "paired_endpoint_role": None}, "paired_endpoint_role"),
])
def test_run_refuses_unregistered_margin_or_role(tmp_path, kwargs, match):
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
    first = run_v3(tmp_path)
    again = run_v3(tmp_path)
    assert [r.perturbed_predictions for r in again.replicates] == [r.perturbed_predictions for r in first.replicates]
    path = d4.prediction_path(tmp_path / "fits", 1, "baseline")
    np.save(path, 1.0 - np.load(path))
    with pytest.raises(IntegrityError, match="disagree"):
        run_v3(tmp_path)


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


def test_shipped_draft_registers_v3_and_leaves_the_margin_pending():
    record = shipped()
    review = d4.review_preregistration(record)
    assert review["problems"] == []
    assert {"decisions.chance_margin", "decisions.paired_endpoint_role"} <= set(review["pending"])
    assert record["decisions"]["decision_rule"]["options"] == [d4.NEAR_CHANCE_RULE]
    assert record["decisions"]["chance_margin"]["value"] is None
    assert record["decisions"]["chance_margin"]["units"] == "accuracy_fraction"
    assert record["fixed_design"]["primary_endpoint"]["procedure"] == d4.NEAR_CHANCE_PROCEDURE
    assert record["fixed_design"]["supporting_endpoint"]["procedure"] == d4.PAIRED_PROCEDURE


@pytest.mark.parametrize("left", ["chance_margin", "paired_endpoint_role", "decision_rule", "alpha"])
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
    (lambda r: r["decisions"]["paired_endpoint_role"].update(value="primary_only"), "paired_endpoint_role"),
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
])
def test_runner_refuses_a_changed_endpoint_procedure(section, key, value):
    from audit.dataset.experiments import d4_gohr_label_shuffle as driver
    record = approve_all_but(shipped())
    record["fixed_design"][section]["procedure"][key] = value
    assert any(section in p for p in driver.check_target_design(record))
