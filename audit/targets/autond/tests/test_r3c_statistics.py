"""
Regression tests for the two statistical defects corrected in r3c.

BLOCKER A - the power calculation treated the SD as known.
    `2*Phi(delta/se - t_crit) - 1` with a fixed se ignored the sampling
    variability of s (four degrees of freedom at n = 5) and went negative
    below the critical value, which was clamped to zero and then reported
    as a computed power of zero. Replaced by an exact integration over
    the chi distribution of s, cross-checked against a seeded Monte Carlo
    simulation of the actual rule.

BLOCKER B - DIFFERENT could fire for a mean inside the margin.
    The rule used a point-null test of d = 0. A discrepancy can be
    significantly non-zero while sitting inside the equivalence interval,
    and the old rule called that DIFFERENT. DIFFERENT now requires
    rejecting H0: |d| <= delta.

Everything here is synthetic. No confirmatory result is used to tune any
threshold, nothing is trained, searched or evaluated, and the margin is
not reopened: MARGIN_FRACTION stays at the approved 0.10 and a test
asserts it.
"""

from __future__ import annotations

import math
import statistics
import unittest

from audit.common import equivalence as eq
from audit.targets.autond.preregistrations import autond_ii_v2 as plan


def synth(reference: float, difference: float, se: float,
          n: int = plan.N_TEST_SETS) -> list[float]:
    """
    n accuracies with an exact sample mean and standard error.

    Built from fixed offsets rather than sampled, so every test below is
    deterministic. The offsets are symmetric about zero and evenly
    spaced - j = i - (n-1)/2 for i in 0..n-1 - so the sample mean is
    exactly `reference + difference`, and the sample SD is
    k*sqrt(sum(j^2)/(n-1)), from which k is solved to hit the requested
    se exactly.

    The default n tracks the registered design (plan.N_TEST_SETS) rather
    than being hardcoded. An earlier version of this helper accepted an
    `n` argument and then ignored it, always returning five values; when
    the registered sample size changed, every fixture built on it
    silently became a short sample that decide() correctly rejected under
    the missing-data rule. The assertions below pin both properties for
    several n so that cannot recur.
    """
    if n < 2:
        raise ValueError("a sample SD needs at least two observations")
    offsets = [i - (n - 1) / 2.0 for i in range(n)]
    scale = math.sqrt(sum(j * j for j in offsets) / (n - 1))
    k = (se * math.sqrt(n)) / scale
    base = reference + difference
    return [base + j * k for j in offsets]


class TestSynthHelper(unittest.TestCase):
    def test_helper_produces_the_requested_mean_and_se(self):
        xs = synth(0.5144, 0.0012, 0.0003)
        self.assertEqual(len(xs), plan.N_TEST_SETS)
        self.assertAlmostEqual(statistics.fmean(xs), 0.5144 + 0.0012, places=9)
        se = statistics.stdev(xs) / math.sqrt(len(xs))
        self.assertAlmostEqual(se, 0.0003, places=9)

    def test_helper_honours_n_rather_than_ignoring_it(self):
        """The defect that made this generalisation necessary."""
        for n in (2, 3, 5, 7, 10, 11, 20):
            xs = synth(0.5144, 0.0012, 0.0003, n=n)
            self.assertEqual(len(xs), n, f"n={n} was ignored")
            self.assertAlmostEqual(statistics.fmean(xs), 0.5156, places=9)
            self.assertAlmostEqual(
                statistics.stdev(xs) / math.sqrt(n), 0.0003, places=9,
                msg=f"n={n} did not hit the requested se")

    def test_default_tracks_the_registered_design(self):
        self.assertEqual(len(synth(0.5144, 0.0, 0.0003)),
                         plan.DESIGN["n_test_sets"])


# ===========================================================================
# Blocker A: power
# ===========================================================================

class TestStudentT(unittest.TestCase):
    def test_critical_values_match_published_tables(self):
        for df, want in {1: 12.706, 2: 4.303, 4: 2.776, 10: 2.228,
                         30: 2.042}.items():
            self.assertAlmostEqual(eq.student_t_ppf(0.975, df), want, places=2,
                                   msg=f"df={df}")

    def test_cdf_and_ppf_are_inverses(self):
        for df in (1, 4, 9, 40):
            for p in (0.005, 0.1, 0.5, 0.9, 0.995):
                t = eq.student_t_ppf(p, df)
                self.assertAlmostEqual(eq.student_t_cdf(t, df), p, places=6)

    def test_symmetry(self):
        for df in (2, 4, 7):
            self.assertAlmostEqual(eq.student_t_cdf(0.0, df), 0.5, places=12)
            self.assertAlmostEqual(eq.student_t_cdf(-1.7, df),
                                   1.0 - eq.student_t_cdf(1.7, df), places=10)


class TestPowerIsExactNotApproximated(unittest.TestCase):
    SD, N, ALPHA = 0.00067, 5, 0.05

    def test_exact_agrees_with_monte_carlo_at_every_fraction(self):
        """
        The independent check. The exact integral and a simulation of the
        actual decision rule must agree to Monte Carlo error.
        """
        for frac in (0.02, 0.05, 0.10, 0.15, 0.20):
            with self.subTest(frac=frac):
                d = frac * (0.5144 - 0.5)
                ex = eq.tost_power_exact(d, self.SD, self.N, self.ALPHA)["power"]
                mc = eq.tost_power_mc(d, self.SD, self.N, self.ALPHA,
                                      n_sim=40_000)
                lo, hi = mc["mc_interval"]
                self.assertTrue(
                    lo - 0.006 <= ex <= hi + 0.006,
                    f"exact {ex:.4f} outside MC 99% interval "
                    f"[{lo:.4f}, {hi:.4f}] at frac {frac}")

    def test_no_fraction_is_reported_as_exactly_zero_power(self):
        """
        The specific artifact: the old formula clamped negative values to
        0.000 and those were reported as computed powers. Every admissible
        margin has strictly positive power.
        """
        for frac in (0.02, 0.05, 0.10, 0.15, 0.20):
            with self.subTest(frac=frac):
                d = frac * (0.5144 - 0.5)
                for a in (self.ALPHA, self.ALPHA / 3):
                    pw = eq.tost_power_exact(d, self.SD, self.N, a)["power"]
                    self.assertGreater(pw, 0.0)
                    self.assertLessEqual(pw, 1.0)

    def test_power_is_monotone_in_the_margin(self):
        prev = -1.0
        for frac in (0.02, 0.05, 0.10, 0.15, 0.20):
            d = frac * (0.5144 - 0.5)
            pw = eq.tost_power_exact(d, self.SD, self.N, self.ALPHA)["power"]
            self.assertGreater(pw, prev, f"not monotone at frac {frac}")
            prev = pw

    def test_power_falls_as_the_sd_grows(self):
        d = 0.10 * (0.5144 - 0.5)
        a = eq.tost_power_exact(d, 0.0003, 5, self.ALPHA)["power"]
        b = eq.tost_power_exact(d, 0.0012, 5, self.ALPHA)["power"]
        self.assertGreater(a, b)

    def test_power_rises_with_more_replicates(self):
        d = 0.10 * (0.5144 - 0.5)
        five = eq.tost_power_exact(d, self.SD, 5, self.ALPHA)["power"]
        twenty = eq.tost_power_exact(d, self.SD, 20, self.ALPHA)["power"]
        self.assertGreater(twenty, five)

    def test_power_falls_as_the_true_difference_approaches_the_margin(self):
        d = 0.10 * (0.5144 - 0.5)
        centred = eq.tost_power_exact(d, self.SD, 5, self.ALPHA,
                                      true_difference=0.0)["power"]
        edge = eq.tost_power_exact(d, self.SD, 5, self.ALPHA,
                                   true_difference=d * 0.9)["power"]
        self.assertGreater(centred, edge)

    def test_conservative_holm_step_lowers_power(self):
        d = 0.10 * (0.5144 - 0.5)
        self.assertLess(
            eq.tost_power_exact(d, self.SD, 5, self.ALPHA / 3)["power"],
            eq.tost_power_exact(d, self.SD, 5, self.ALPHA)["power"])

    def test_monte_carlo_is_reproducible(self):
        d = 0.10 * (0.5144 - 0.5)
        a = eq.tost_power_mc(d, self.SD, 5, self.ALPHA, n_sim=5_000, seed=7)
        b = eq.tost_power_mc(d, self.SD, 5, self.ALPHA, n_sim=5_000, seed=7)
        self.assertEqual(a["power"], b["power"])
        c = eq.tost_power_mc(d, self.SD, 5, self.ALPHA, n_sim=5_000, seed=8)
        self.assertNotEqual(a["power"], c["power"],
                            "a different seed should give a different draw")

    def test_monte_carlo_reports_its_uncertainty(self):
        d = 0.10 * (0.5144 - 0.5)
        mc = eq.tost_power_mc(d, self.SD, 5, self.ALPHA, n_sim=5_000)
        lo, hi = mc["mc_interval"]
        self.assertLess(lo, mc["power"])
        self.assertGreater(hi, mc["power"])
        self.assertIn("seed", mc)
        self.assertGreater(mc["mc_half_width"], 0.0)

    def test_invalid_inputs_are_refused(self):
        for kwargs in ({"n": 1}, {"sd": 0.0}, {"sd": -1.0}, {"alpha": 0.0},
                       {"alpha": 0.9}):
            with self.subTest(**kwargs):
                args = {"delta": 0.001, "sd": self.SD, "n": 5,
                        "alpha": self.ALPHA, **kwargs}
                with self.assertRaises(ValueError):
                    eq.tost_power_exact(**args)
        with self.assertRaises(ValueError):
            eq.tost_power_exact(-0.001, self.SD, 5, self.ALPHA)

    def test_method_string_states_the_variance_is_estimated(self):
        r = eq.tost_power_exact(0.001, self.SD, 5, self.ALPHA)
        self.assertIn("estimated from the sample", r["method"])
        self.assertNotIn("normal approximation", r["method"])


class TestPlanPowerReporting(unittest.TestCase):
    def test_power_at_delegates_to_the_exact_calculation(self):
        pw = plan.power_at(plan.equivalence_margin(0.5144), m=3)
        self.assertIn("estimated from the sample", pw["method"])
        self.assertNotIn("approx_power", pw,
                         "the clamped approximation must be gone")
        self.assertGreater(pw["power"], 0.0)

    def test_margin_sensitivity_is_computed_not_transcribed(self):
        rows = plan.margin_sensitivity()
        self.assertEqual([r["fraction"] for r in rows],
                         [0.02, 0.05, 0.10, 0.15, 0.20])
        for r in rows:
            d = r["fraction"] * (0.5144 - 0.5)
            self.assertAlmostEqual(r["delta"], d, places=12)
            self.assertAlmostEqual(
                r["power_at_alpha"],
                eq.tost_power_exact(d, plan.PLANNING_SD, plan.N_TEST_SETS,
                                    plan.ALPHA)["power"], places=9)
            self.assertAlmostEqual(
                r["power_at_conservative_holm_step"],
                eq.tost_power_exact(d, plan.PLANNING_SD, plan.N_TEST_SETS,
                                    plan.ALPHA / 3)["power"], places=9)
        self.assertEqual([r["is_registered_margin"] for r in rows],
                         [False, False, True, False, False])

    def test_planning_sd_provenance_is_declared_and_uncertain(self):
        prov = plan.PLANNING_SD_PROVENANCE
        self.assertIn("ASSUMED", prov)
        self.assertIn("not a known population parameter", prov)
        self.assertIn("4 df", prov)
        self.assertIn("DESIGN only", prov)
        self.assertIs(plan.POWER["planning_sd_provenance"], prov)

    def test_superseded_method_is_recorded_not_deleted(self):
        self.assertIn("superseded_method", plan.POWER)
        self.assertIn("clamped", plan.POWER["superseded_method"])

    def test_alpha_over_m_is_labelled_a_planning_approximation(self):
        pw = plan.power_at(plan.equivalence_margin(0.5144), m=3)
        self.assertIn("planning", pw["alpha_m_is_planning_only"].lower())
        self.assertAlmostEqual(pw["alpha_adjusted"], plan.ALPHA / 3)

    def test_power_does_not_justify_the_margin(self):
        self.assertIn("power_does_not_justify_a_margin", plan.POWER)
        self.assertIn("does NOT",
                      plan.POWER["power_does_not_justify_a_margin"])

    def test_the_approved_margin_is_unchanged(self):
        self.assertEqual(plan.MARGIN_FRACTION, 0.10)
        self.assertAlmostEqual(plan.equivalence_margin(0.5144), 0.00144, places=9)
        self.assertAlmostEqual(plan.equivalence_margin(0.5179), 0.00179, places=9)
        self.assertAlmostEqual(plan.equivalence_margin(0.5182), 0.00182, places=9)
        self.assertEqual(plan.ALPHA, 0.05)
        self.assertEqual(len(plan.COMPARISONS), 3)


# ===========================================================================
# Blocker B: DIFFERENT must mean outside the margin
# ===========================================================================

class TestDifferentRequiresOutsideMargin(unittest.TestCase):
    REF = 0.5144

    def _one(self, difference, se):
        accs = synth(self.REF, difference, se)
        res = plan.decide({"speck3264": accs, "simon3264": accs})
        return res["per_comparison"]["CMP-SPECK-PAPER"]

    def test_the_exact_r3b_counterexample_is_now_indeterminate(self):
        """
        THE NAMED REGRESSION. d = +0.00120 inside delta = 0.00144, with
        p_tost = 0.234 and p_diff = 0.016. Revision 3b returned DIFFERENT.
        """
        r = self._one(0.0012, 0.0003)
        self.assertTrue(r["mean_inside_margin"])
        self.assertLess(r["p_diff"], 0.05,
                        "fixture must keep the point-null significant")
        self.assertGreater(r["p_tost"], 0.05,
                           "fixture must keep equivalence unestablished")
        self.assertEqual(r["outcome"], eq.INDETERMINATE)
        self.assertNotEqual(r["outcome"], eq.DIFFERENT)

    def test_no_inside_margin_mean_is_ever_different(self):
        """Swept across the whole inside-margin range and many precisions."""
        delta = plan.equivalence_margin(self.REF)
        for frac in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            for se in (0.00002, 0.00005, 0.0001, 0.0003, 0.0006, 0.001):
                with self.subTest(frac=frac, se=se):
                    r = self._one(delta * frac, se)
                    self.assertTrue(r["mean_inside_margin"])
                    self.assertNotEqual(
                        r["outcome"], eq.DIFFERENT,
                        f"inside-margin mean d={delta*frac:.6f} se={se} "
                        f"classified DIFFERENT (p_diff={r['p_diff']:.4g})")

    def test_near_zero_and_precise_is_equivalent(self):
        r = self._one(0.00002, 0.00005)
        self.assertEqual(r["outcome"], eq.EQUIVALENT)

    def test_clearly_outside_above_is_different(self):
        r = self._one(0.0040, 0.0002)
        self.assertEqual(r["outcome"], eq.DIFFERENT)
        self.assertEqual(r["outside_direction"], "above")

    def test_clearly_outside_below_is_different(self):
        r = self._one(-0.0040, 0.0002)
        self.assertEqual(r["outcome"], eq.DIFFERENT)
        self.assertEqual(r["outside_direction"], "below")

    def test_too_imprecise_is_indeterminate(self):
        r = self._one(0.0005, 0.0030)
        self.assertEqual(r["outcome"], eq.INDETERMINATE)
        self.assertGreater(r["p_tost"], 0.05)
        self.assertGreater(r["p_outside"], 0.05)

    def test_outside_the_margin_but_imprecise_is_indeterminate(self):
        """Being outside is not enough; the evidence must support it."""
        r = self._one(0.0020, 0.0040)
        self.assertFalse(r["mean_inside_margin"])
        self.assertEqual(r["outcome"], eq.INDETERMINATE)

    def test_point_null_is_recorded_but_flagged_diagnostic(self):
        r = self._one(0.0012, 0.0003)
        self.assertIn("p_diff", r)
        self.assertIn("diagnostic", r["p_diff_role"])

    def test_decision_never_reads_the_point_null(self):
        """Structural: classify() takes no point-null argument."""
        import inspect

        params = set(inspect.signature(eq.classify).parameters)
        self.assertEqual(params, {"equivalence_rejected", "outside_rejected"})

    def test_equivalent_and_different_are_mutually_exclusive(self):
        with self.assertRaises(ArithmeticError):
            eq.classify(equivalence_rejected=True, outside_rejected=True)

    def test_outcome_vocabulary_comes_from_the_framework(self):
        self.assertIs(plan.EQUIVALENT, eq.EQUIVALENT)
        self.assertIs(plan.DIFFERENT, eq.DIFFERENT)
        self.assertIs(plan.INDETERMINATE, eq.INDETERMINATE)


class TestOutsideMarginTest(unittest.TestCase):
    def test_rejection_regions_are_disjoint(self):
        """Why min() needs no correction between the two directions."""
        for d in (-0.01, -0.001, 0.0, 0.001, 0.01):
            r = eq.outside_margin(d, 0.0002, 4, 0.00144)
            rejects = [r["p_above"] < 0.05, r["p_below"] < 0.05]
            self.assertLessEqual(sum(rejects), 1,
                                 f"both directions rejected at d={d}")

    def test_direction_is_reported(self):
        self.assertEqual(eq.outside_margin(0.01, 2e-4, 4, 1.44e-3)["direction"],
                         "above")
        self.assertEqual(eq.outside_margin(-0.01, 2e-4, 4, 1.44e-3)["direction"],
                         "below")

    def test_tost_and_outside_cannot_both_be_small(self):
        for d in (-0.006, -0.002, -0.0005, 0.0, 0.0005, 0.002, 0.006):
            for se in (0.00005, 0.0002, 0.001):
                t = eq.tost(d, se, 4, 0.00144)
                o = eq.outside_margin(d, se, 4, 0.00144)
                self.assertFalse(t["p_tost"] < 0.05 and o["p_outside"] < 0.05,
                                 f"both significant at d={d}, se={se}")

    def test_invalid_inputs_are_refused(self):
        for fn in (eq.tost, eq.outside_margin):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(0.001, 0.0, 4, 0.001)
                with self.assertRaises(ValueError):
                    fn(0.001, 0.0002, 4, 0.0)
                with self.assertRaises(ValueError):
                    fn(0.001, 0.0002, 0, 0.001)
                with self.assertRaises(ValueError):
                    fn(float("nan"), 0.0002, 4, 0.001)


class TestHolmFamily(unittest.TestCase):
    def test_thresholds(self):
        r = eq.holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
        self.assertAlmostEqual(r["a"]["threshold"], 0.05 / 3)
        self.assertAlmostEqual(r["b"]["threshold"], 0.05 / 2)
        self.assertAlmostEqual(r["c"]["threshold"], 0.05)

    def test_step_down_stops_at_the_first_failure(self):
        r = eq.holm({"a": 0.001, "b": 0.03, "c": 0.04}, alpha=0.05)
        self.assertTrue(r["a"]["reject"])
        self.assertFalse(r["b"]["reject"])
        self.assertFalse(r["c"]["reject"],
                         "0.04 < 0.05 but must not be rejected after a failure")

    def test_ordering_is_by_pvalue(self):
        r = eq.holm({"z": 0.001, "a": 0.4}, alpha=0.05)
        self.assertEqual(r["z"]["rank"], 1)
        self.assertEqual(r["a"]["rank"], 2)

    def test_invalid_pvalues_refused(self):
        for bad in ({"a": -0.1}, {"a": 1.1}, {"a": float("nan")},
                    {"a": "x"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    eq.holm(bad)
        with self.assertRaises(ValueError):
            eq.holm({"a": 0.01}, alpha=0.0)

    def test_the_plan_delegates_to_the_framework_holm(self):
        a = plan.holm({"x": 0.01, "y": 0.4})
        b = eq.holm({"x": 0.01, "y": 0.4})
        self.assertEqual(a, b)

    def test_two_families_are_corrected_independently(self):
        accs = synth(0.5144, 0.0040, 0.0002)
        res = plan.decide({"speck3264": accs, "simon3264": accs})
        self.assertIn("holm_equivalence", res)
        self.assertIn("holm_outside_margin", res)
        self.assertEqual(set(res["holm_equivalence"]),
                         set(res["holm_outside_margin"]))
        for cid in res["holm_equivalence"]:
            self.assertEqual(res["holm_equivalence"][cid]["m"], 3)
            self.assertEqual(res["holm_outside_margin"][cid]["m"], 3)


class TestMissingAndInvalidData(unittest.TestCase):
    def test_short_sample_is_indeterminate(self):
        res = plan.decide({"speck3264": [0.5144, 0.5143, 0.5145],
                           "simon3264": []})
        for cid, r in res["per_comparison"].items():
            self.assertEqual(r["outcome"], eq.INDETERMINATE, cid)
            self.assertIn("missing-data rule", r["reason"])

    def test_non_numeric_accuracies_are_refused(self):
        res = plan.decide({"speck3264": [0.5144, "x", 0.5145, 0.5143, 0.5146],
                           "simon3264": [0.5179] * plan.N_TEST_SETS})
        r = res["per_comparison"]["CMP-SPECK-PAPER"]
        self.assertEqual(r["outcome"], eq.INDETERMINATE)
        self.assertIn("non-numeric", r["reason"])

    def test_non_finite_accuracies_are_refused(self):
        res = plan.decide({"speck3264": [0.5144, float("inf"), 0.5145,
                                         0.5143, 0.5146],
                           "simon3264": [0.5179] * plan.N_TEST_SETS})
        self.assertEqual(res["per_comparison"]["CMP-SPECK-PAPER"]["outcome"],
                         eq.INDETERMINATE)

    def test_zero_variance_is_indeterminate_not_a_crash(self):
        res = plan.decide({"speck3264": [0.5144] * plan.N_TEST_SETS,
                           "simon3264": [0.5179] * plan.N_TEST_SETS})
        r = res["per_comparison"]["CMP-SPECK-PAPER"]
        self.assertEqual(r["outcome"], eq.INDETERMINATE)
        self.assertIn("degenerate variance", r["reason"])
        self.assertTrue(r["degenerate_variance"])

    def test_degenerate_variance_does_not_depend_on_exact_float_zero(self):
        """
        REGRESSION, interpreter-dependent. The guard was `if not se:`.
        Whether n identical accuracies sum to exactly n*x is a floating-
        point accident: on CPython 3.13 the SD of [0.5144]*10 is exactly
        0.0, but on 3.11 - the REGISTERED environment - it is 1.17e-16,
        so the exact-zero guard did not fire and the TOST returned
        EQUIVALENT with SE = 3.7e-17. A maximally confident verdict from
        a sample carrying no information.

        Asserted over SDs spanning the float-noise band rather than
        relying on this interpreter reproducing either accident.
        """
        for sd in (0.0, 1e-18, 1.17e-16, 1e-15, 1e-14, 1e-13):
            with self.subTest(sd=sd):
                accs = synth(0.5144, 0.0, sd / math.sqrt(plan.N_TEST_SETS))
                r = plan.decide({"speck3264": accs, "simon3264": accs}
                                )["per_comparison"]["CMP-SPECK-PAPER"]
                self.assertEqual(
                    r["outcome"], eq.INDETERMINATE,
                    f"SD {sd!r} is float noise, not a measurement")
                self.assertTrue(r.get("degenerate_variance"))

    def test_a_real_sd_is_not_treated_as_degenerate(self):
        """
        The floor must not reclassify anything physically possible. The
        cross-set SD of independent draws is of order 5e-4; the floor is
        nine orders below that.
        """
        self.assertLess(plan.DEGENERATE_SD_RELATIVE_FLOOR, 1e-9)
        for sd in (1e-8, 1e-6, 1e-5, 5e-4, 1e-3):
            with self.subTest(sd=sd):
                accs = synth(0.5144, 0.0, sd / math.sqrt(plan.N_TEST_SETS))
                r = plan.decide({"speck3264": accs, "simon3264": accs}
                                )["per_comparison"]["CMP-SPECK-PAPER"]
                self.assertFalse(r.get("degenerate_variance"),
                                 f"SD {sd!r} is a real measurement")
                self.assertIsNotNone(r.get("df"))

    def test_the_floor_is_relative_to_the_measurement_scale(self):
        self.assertAlmostEqual(plan._degenerate_sd_floor(0.5144),
                               0.5144 * plan.DEGENERATE_SD_RELATIVE_FLOOR)
        # a zero or missing mean must not collapse the floor to zero
        self.assertGreater(plan._degenerate_sd_floor(0.0), 0.0)
        self.assertGreater(plan._degenerate_sd_floor(None), 0.0)
        self.assertTrue(plan._variance_is_degenerate(0.5144, None))


class TestWrittenRuleMatchesTheCode(unittest.TestCase):
    def test_the_prose_describes_the_outside_margin_test(self):
        dr = plan.DECISION_RULE
        self.assertIn("outside_margin_test", dr)
        self.assertNotIn("difference_test", dr,
                         "the old point-null-based clause must be gone")
        self.assertIn("point_null_is_diagnostic_only", dr)
        self.assertIn("mutual_exclusivity", dr)

    def test_the_prose_records_the_correction_with_the_counterexample(self):
        text = plan.DECISION_RULE["point_null_is_diagnostic_only"]
        self.assertIn("0.00120", text)
        self.assertIn("0.00144", text)
        self.assertIn("test_r3c_statistics", text)

    def test_outcome_descriptions_match_the_implemented_rule(self):
        d = plan.DECISION_RULE["per_comparison_outcomes"]
        self.assertIn("p_tost", d[eq.EQUIVALENT])
        self.assertIn("p_outside", d[eq.DIFFERENT])
        self.assertIn("point-null", d[eq.DIFFERENT])

    def test_hypothesis_conditions_use_the_outcome_vocabulary(self):
        for name, h in plan.HYPOTHESES.items():
            with self.subTest(h=name):
                self.assertIn("EQUIVALENT", h["supported_if"])
                self.assertIn("DIFFERENT", h["not_supported_if"])

    def test_simon_limitation_is_narrow_not_absolute(self):
        just = " ".join(plan.EQUIVALENCE_MARGIN["justification"])
        self.assertIn("overlap", just)
        self.assertIn("not expected to resolve", just)
        self.assertIn("NOT claimed", just)
        self.assertNotIn("CANNOT identify", just)

    def test_both_simon_references_are_preserved(self):
        ids = {c["comparison_id"] for c in plan.COMPARISONS}
        self.assertIn("CMP-SIMON-PAPER", ids)
        self.assertIn("CMP-SIMON-README", ids)
        refs = {c["reference"] for c in plan.COMPARISONS}
        self.assertIn(0.5179, refs)
        self.assertIn(0.5182, refs)

    def test_rule_version_is_recorded_in_the_output(self):
        res = plan.decide({"speck3264": synth(0.5144, 0.0, 0.0001),
                           "simon3264": synth(0.5179, 0.0, 0.0001)})
        self.assertEqual(res["rule_version"], "autond-ii-v2-decision-r3c")
        self.assertIn("never a point-null rejection", res["note"])


class TestNoDuplicatedGenericLogic(unittest.TestCase):
    def test_the_plan_defines_no_private_t_distribution(self):
        import inspect

        src = inspect.getsource(plan)
        for name in ("def student_t_cdf", "def student_t_sf",
                     "def student_t_ppf", "def _betacf",
                     "def regularised_incomplete_beta"):
            self.assertNotIn(name, src, f"{name} must live in the framework")

    def test_the_adapter_reexports_rather_than_redefines(self):
        from audit.targets.autond import evaluation_results as evres

        self.assertIs(evres.student_t_cdf, eq.student_t_cdf)
        self.assertIs(evres.student_t_ppf, eq.student_t_ppf)
        self.assertIs(evres.student_t_sf, eq.student_t_sf)

    def test_the_generic_module_knows_nothing_about_autond(self):
        import inspect

        src = inspect.getsource(eq).lower()
        for word in ("autond", "speck", "simon", "dbitnet", "0.5144"):
            self.assertNotIn(word, src,
                             f"the generic module must not mention {word!r}")

    def test_nothing_in_this_task_runs_an_experiment(self):
        import ast
        from pathlib import Path

        for mod in (eq, plan):
            tree = ast.parse(Path(mod.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and \
                        isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr, {"fit", "predict", "evo", "optimize"},
                        f"{mod.__name__} must not execute an experiment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
