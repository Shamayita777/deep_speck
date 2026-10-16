"""
Tests for the r3c -> r3d design change: n = 5 -> n = 10, margin unchanged.

The change is NOT a doubled constant. The target's nEval = 5 is a
module-level constant at eval_nets.py:22 and is not a command-line
argument, so reaching ten sets without editing the pinned repository
requires invoking eval_nets.py TWICE per checkpoint with distinct
--add_str values and pooling the two NPZs. These tests pin all of that:

  * the registered design states n = 10 and still Delta = 10%
  * the plan's OWN power calculation, at the registered multiplicity,
    reports the n = 10 figures (nothing is transcribed from chat)
  * the n = 5 figures are retained, not deleted
  * the gate emits two invocations per checkpoint, with distinct
    suffixes and therefore distinct NPZ paths
  * the journal and log filenames are per-invocation, so the second
    invocation cannot overwrite the first's record
  * the validator's duplicated design constants agree with the plan
  * one surviving invocation is REFUSED, not analysed as an n = 5 result
  * pooling is rejected when the two invocations are not independent
    draws of the same weights

Synthetic fixtures only. No real experimental result is used here, and
no threshold is tuned against one.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from audit.common import equivalence as eq
from audit.targets.autond import evaluation_gate as gate
from audit.targets.autond import evaluation_results as evres
from audit.targets.autond import upstream_interface as ui
from audit.targets.autond.preregistrations import autond_ii_v2 as plan
from audit.targets.autond.tests import portable


class TestRegisteredDesign(unittest.TestCase):
    def test_n_is_ten_and_derived_not_hardcoded(self):
        self.assertEqual(plan.SETS_PER_INVOCATION, 5)
        self.assertEqual(plan.INVOCATIONS_PER_CHECKPOINT, 2)
        self.assertEqual(plan.N_TEST_SETS, 10)
        self.assertEqual(
            plan.N_TEST_SETS,
            plan.SETS_PER_INVOCATION * plan.INVOCATIONS_PER_CHECKPOINT,
            "N_TEST_SETS must be derived, so the three cannot disagree")
        self.assertEqual(plan.DESIGN["n_test_sets"], plan.N_TEST_SETS)
        self.assertEqual(plan.DESIGN["sets_per_invocation"],
                         plan.SETS_PER_INVOCATION)
        self.assertEqual(plan.DESIGN["invocations_per_checkpoint"],
                         plan.INVOCATIONS_PER_CHECKPOINT)

    def test_the_margin_did_not_move(self):
        """The whole point: more samples, not a looser criterion."""
        self.assertEqual(plan.MARGIN_FRACTION, 0.10)
        self.assertAlmostEqual(plan.equivalence_margin(0.5144), 0.00144,
                               places=12)
        self.assertAlmostEqual(plan.equivalence_margin(0.5179), 0.00179,
                               places=12)
        self.assertAlmostEqual(plan.equivalence_margin(0.5182), 0.00182,
                               places=12)
        self.assertEqual(plan.ALPHA, 0.05)

    def test_suffixes_are_registered_and_distinct(self):
        self.assertEqual(len(plan.INVOCATION_SUFFIXES),
                         plan.INVOCATIONS_PER_CHECKPOINT)
        self.assertEqual(len(set(plan.INVOCATION_SUFFIXES)),
                         len(plan.INVOCATION_SUFFIXES))
        for s in plan.INVOCATION_SUFFIXES:
            self.assertTrue(s, "an empty suffix collides with the default name")

    def test_the_design_records_why_a_constant_was_not_edited(self):
        why = plan.DESIGN["why_two_invocations_and_not_a_changed_constant"]
        self.assertIn("nEval", why)
        self.assertIn("not", why.lower())
        self.assertIn("add_str", why)
        for phrase in ("modify", "pinned"):
            self.assertIn(phrase, why.lower())

    def test_pooling_rule_and_its_assumption_are_both_recorded(self):
        rule = plan.DESIGN["pooling_rule"]
        self.assertIn("sqrt(10)", rule)
        self.assertIn("df = 9", rule)
        assumption = plan.DESIGN["pooling_assumption"]
        self.assertIn("exchangeable", assumption)
        self.assertIn("RECORDED ASSUMPTION", assumption)
        self.assertIn("eval_nets.py:208-209", assumption)

    def test_sd_and_df_prose_match_the_registered_n(self):
        rule = plan.DECISION_RULE["test"]
        self.assertIn("sqrt(10)", rule)
        self.assertIn("df = 9", rule)
        self.assertNotIn("sqrt(5)", rule)
        self.assertIn("df = 9", plan.DECISION_RULE["outside_margin_test"])


class TestPowerIsComputedAtTheRegisteredN(unittest.TestCase):
    """
    The operator's condition for approving n = 10 was that the figures
    come from the plan's own registered procedure, including the
    multiplicity correction - not from a number quoted in conversation.
    """

    def test_margin_sensitivity_uses_the_registered_n(self):
        m = len(plan.COMPARISONS)
        for row in plan.margin_sensitivity():
            d = row["delta"]
            self.assertAlmostEqual(
                row["power_at_alpha"],
                eq.tost_power_exact(d, plan.PLANNING_SD, plan.N_TEST_SETS,
                                    plan.ALPHA)["power"],
                places=9)
            self.assertAlmostEqual(
                row["power_at_conservative_holm_step"],
                eq.tost_power_exact(d, plan.PLANNING_SD, plan.N_TEST_SETS,
                                    plan.ALPHA / m)["power"],
                places=9)

    def test_registered_margin_power_improved_over_n_five(self):
        """
        Not a transcribed figure: both sides are computed here, and the
        assertion is the INEQUALITY, which is the design claim.
        """
        d = plan.equivalence_margin(0.5144)
        m = len(plan.COMPARISONS)
        at10 = eq.tost_power_exact(d, plan.PLANNING_SD, 10,
                                   plan.ALPHA / m)["power"]
        at5 = eq.tost_power_exact(d, plan.PLANNING_SD, 5,
                                  plan.ALPHA / m)["power"]
        self.assertGreater(at10, at5)
        self.assertGreater(at10, 0.80,
                           "the design change exists to clear the conventional "
                           "0.80 at the conservative Holm step")

    def test_power_at_defaults_to_the_registered_n(self):
        pw = plan.power_at(plan.equivalence_margin(0.5144))
        self.assertEqual(pw["n"], plan.N_TEST_SETS)
        self.assertEqual(pw["df"], plan.N_TEST_SETS - 1)

    def test_the_n_five_figures_are_retained_not_deleted(self):
        sup = plan.POWER["superseded_design"]
        self.assertIn("0.7965", sup)
        self.assertIn("n = 5", sup)
        self.assertIn("retained", sup.lower())

    def test_robustness_to_a_worse_sd_is_recorded_and_true(self):
        note = plan.POWER["robustness_note"]
        self.assertIn("2.9x", note)
        self.assertIn("INDETERMINATE", note)
        d = plan.equivalence_margin(0.5144)
        m = len(plan.COMPARISONS)
        at_1_5x = eq.tost_power_exact(d, plan.PLANNING_SD * 1.5,
                                      plan.N_TEST_SETS, plan.ALPHA / m)["power"]
        self.assertGreater(at_1_5x, 0.90, "the claim in robustness_note")

    def test_power_still_does_not_justify_the_margin(self):
        """Raising n must not have turned power into a margin argument."""
        self.assertIn("does NOT",
                      plan.POWER["power_does_not_justify_a_margin"])


class TestGateEmitsTwoInvocations(unittest.TestCase):
    def _cmds(self, tmp):
        up = portable.fake_upstream(Path(tmp) / "up")["path"]
        return gate.planned_commands(up, Path(tmp) / "out",
                                     argspec=ui.EVAL_NETS_ARGSPEC)

    def test_two_invocations_per_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmds = self._cmds(tmp)
            self.assertEqual(len(cmds),
                             len(plan.TARGET["checkpoints"])
                             * plan.INVOCATIONS_PER_CHECKPOINT)
            per = {}
            for c in cmds:
                per.setdefault(c["checkpoint"], []).append(c)
            for cid, group in per.items():
                self.assertEqual(len(group),
                                 plan.INVOCATIONS_PER_CHECKPOINT, cid)

    def test_each_invocation_carries_a_distinct_registered_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            per = {}
            for c in self._cmds(tmp):
                per.setdefault(c["checkpoint"], []).append(c)
            for cid, group in per.items():
                sufs = sorted(c["invocation_suffix"] for c in group)
                self.assertEqual(sufs, sorted(plan.INVOCATION_SUFFIXES), cid)

    def test_npz_paths_differ_between_invocations(self):
        """
        If they did not, the second invocation would overwrite the
        first's artifact and the pooled sample would be five sets read
        twice rather than ten independent draws.
        """
        with tempfile.TemporaryDirectory() as tmp:
            per = {}
            for c in self._cmds(tmp):
                per.setdefault(c["checkpoint"], []).append(c["npz"])
            for cid, npzs in per.items():
                self.assertEqual(len(set(npzs)), len(npzs),
                                 f"{cid}: NPZ paths collide: {npzs}")

    def test_add_str_is_an_argument_the_target_actually_defines(self):
        self.assertIn("add_str", ui.EVAL_NETS_ARGSPEC)
        with tempfile.TemporaryDirectory() as tmp:
            for c in self._cmds(tmp):
                flags = [a[2:] for a in c["command"] if a.startswith("--")]
                for f in flags:
                    self.assertIn(f, ui.EVAL_NETS_ARGSPEC)
                self.assertIn("add_str", flags)

    def test_invocations_differ_only_in_the_output_suffix(self):
        """The two runs must be the same experiment, not two experiments."""
        with tempfile.TemporaryDirectory() as tmp:
            per = {}
            for c in self._cmds(tmp):
                per.setdefault(c["checkpoint"], []).append(c["command"])
            for cid, cmds in per.items():
                def strip(cmd):
                    out, skip = [], False
                    for tok in cmd:
                        if skip:
                            skip = False
                            continue
                        if tok == "--add_str":
                            skip = True
                            continue
                        out.append(tok)
                    return out
                base = strip(cmds[0])
                for other in cmds[1:]:
                    self.assertEqual(strip(other), base, cid)

    def test_invocation_ids_are_unique_across_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = [c["invocation_id"] for c in self._cmds(tmp)]
            self.assertEqual(len(set(ids)), len(ids), ids)


class TestValidatorConstantsTrackThePlan(unittest.TestCase):
    def test_the_duplicate_constants_agree_with_the_preregistration(self):
        got = evres.assert_matches_preregistration()
        self.assertEqual(got["n_pooled"], plan.N_TEST_SETS)
        self.assertEqual(got["sets_per_invocation"], plan.SETS_PER_INVOCATION)
        self.assertEqual(got["invocations_per_checkpoint"],
                         plan.INVOCATIONS_PER_CHECKPOINT)

    def test_per_invocation_count_is_the_upstream_constant(self):
        self.assertEqual(evres.EXPECTED_SETS_PER_INVOCATION, 5)
        self.assertEqual(evres.EXPECTED_N_SETS,
                         evres.EXPECTED_SETS_PER_INVOCATION,
                         "the old name must keep its old meaning")

    def test_pooled_count_is_what_the_validator_enforces(self):
        self.assertEqual(evres.EXPECTED_N_POOLED, 10)

    def test_a_divergence_would_be_caught(self):
        """
        Simulate the failure this guard exists for: the validator and the
        plan disagreeing after one side is edited alone.
        """
        original = evres.EXPECTED_N_POOLED
        try:
            evres.EXPECTED_N_POOLED = original + 5
            with self.assertRaises(ValueError) as ctx:
                evres.assert_matches_preregistration()
            self.assertIn("never registered", str(ctx.exception))
        finally:
            evres.EXPECTED_N_POOLED = original
        self.assertEqual(evres.assert_matches_preregistration()["n_pooled"],
                         original)


class TestDecisionRequiresTenSets(unittest.TestCase):
    def _accs(self, n, difference=0.0, se=0.0001, ref=0.5144):
        offsets = [i - (n - 1) / 2.0 for i in range(n)]
        scale = math.sqrt(sum(j * j for j in offsets) / (n - 1))
        k = (se * math.sqrt(n)) / scale
        return [ref + difference + j * k for j in offsets]

    def test_ten_sets_are_analysed(self):
        res = plan.decide({"speck3264": self._accs(10),
                           "simon3264": self._accs(10, ref=0.5179)})
        rec = res["per_comparison"]["CMP-SPECK-PAPER"]
        self.assertEqual(rec["n"], 10)
        self.assertEqual(rec["df"], 9)
        self.assertNotIn("missing-data", str(rec.get("reason", "")))

    def test_five_sets_are_now_indeterminate_not_a_smaller_design(self):
        """
        The registered design is ten. A surviving five is NOT silently
        analysed as the old design.
        """
        res = plan.decide({"speck3264": self._accs(5),
                           "simon3264": self._accs(5, ref=0.5179)})
        rec = res["per_comparison"]["CMP-SPECK-PAPER"]
        self.assertEqual(rec["outcome"], eq.INDETERMINATE)
        self.assertIn("10", str(rec["reason"]))

    def test_the_missing_data_rule_forbids_the_fallback_explicitly(self):
        md = plan.DESIGN["missing_data"]
        self.assertIn("ten", md)
        self.assertIn("NOT", md)
        self.assertIn("unregistered fallback", md)


class TestAssemblerPoolsTwoInvocations(unittest.TestCase):
    """
    Exercises the assembler's pooling against synthetic journals and
    NPZs. No real evaluation output is used.
    """

    def _assemble(self, tmp, **kw):
        """
        Assemble against the REAL pinned checkout, because the pooling
        code runs only after the checkpoint digests match what the plan
        recorded, and fake_upstream mismatches by construction. Skips
        when the checkout is absent rather than asserting on a bundle
        that never reached the pooling step.
        """
        from audit.targets.autond import evaluation_assemble as asm
        up = portable.require_real_upstream(self)
        run = portable.staged_run_dir(Path(tmp) / "run", up, plan, **kw)
        with portable.force_unfrozen(plan):
            return asm.assemble(run, upstream=up, repo_root=Path(tmp) / "repo")

    def _problems(self, tmp, **kw):
        return " | ".join(self._assemble(tmp, **kw)["problems"])

    def test_two_invocations_pool_to_ten(self):
        """
        The success path. Needs a FROZEN plan, because assembly refuses to
        emit a record under an unfrozen preregistration - so the freeze is
        simulated here rather than performed in the operator's repository.
        """
        from audit.targets.autond import evaluation_assemble as asm
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            fx = portable.simulate_frozen(plan, Path(tmp) / "repo")
            try:
                run = portable.staged_run_dir(Path(tmp) / "run", up, plan)
                res = asm.assemble(run, upstream=up,
                                   repo_root=Path(tmp) / "repo")
            finally:
                plan.FROZEN_SIDECAR = fx["original_sidecar"]
            self.assertTrue(res["ok"], res["problems"])
            ck = (res.get("record") or {}).get("checkpoints") or {}
            self.assertTrue(ck, res["problems"])
            for cid, rec in ck.items():
                self.assertEqual(len(rec["accuracies"]), plan.N_TEST_SETS, cid)
                self.assertEqual(rec["n_pooled"], plan.N_TEST_SETS, cid)
                self.assertEqual(len(rec["invocations"]),
                                 plan.INVOCATIONS_PER_CHECKPOINT, cid)
                self.assertEqual(
                    rec["accuracies"],
                    [a for i in rec["invocations"] for a in i["accuracies"]],
                    f"{cid}: pooled order must be the invocations in order")

    def test_one_surviving_invocation_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._problems(tmp, invocations=1)
            self.assertIn("usable invocation", p)
            self.assertIn("unregistered fallback", p)

    def test_a_short_npz_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._problems(tmp, n_sets=3)
            self.assertIn("accuracies, not the", p)

    def test_identical_npzs_are_refused_as_not_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._problems(tmp, identical_npzs=True)
            self.assertIn("byte-identical", p)
            self.assertIn("os.urandom", p)

    def test_colliding_suffixes_lose_an_invocation_and_are_refused(self):
        """
        Two invocations sharing --add_str is not merely suspicious: the
        target derives its NPZ name and the gate derives its journal name
        from the suffix, so the second run physically overwrites the
        first. The observable symptom is therefore a MISSING invocation,
        and refusing on that is the correct behaviour. Asserted as the
        shortfall rather than as a suffix message, because the suffix
        message cannot be reached once the file is gone.
        """
        with tempfile.TemporaryDirectory() as tmp:
            p = self._problems(tmp, suffixes=["_a", "_a"])
            self.assertIn("usable invocation", p)
            self.assertIn("not the registered", p)

    def test_the_suffix_check_itself_refuses_when_both_journals_survive(self):
        """
        Defence in depth for the case the shortfall check cannot see: two
        journals that survive on disk under different names but claim the
        same suffix. Reached by rewriting one staged journal, so the
        assembler's own suffix comparison is the thing under test.
        """
        from audit.targets.autond import evaluation_assemble as asm
        up = portable.require_real_upstream(self)
        with tempfile.TemporaryDirectory() as tmp:
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan)
            jdir = run / "journal"
            edited = 0
            for jf in sorted(jdir.glob("exec_*_b.json")):
                rec = json.loads(jf.read_text())
                rec["invocation_suffix"] = "_a"      # collide, keep the filename
                jf.write_text(json.dumps(rec, indent=2, sort_keys=True))
                edited += 1
            self.assertTrue(edited, "fixture produced no '_b' journals")
            with portable.force_unfrozen(plan):
                res = asm.assemble(run, upstream=up,
                                   repo_root=Path(tmp) / "repo")
            p = " | ".join(res["problems"])
            self.assertIn("share the suffix", p)
            self.assertIn("overwrote", p)

    def test_unregistered_suffixes_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._problems(tmp, suffixes=["_q", "_z"])
            self.assertIn("not the registered", p)


class TestJournalNamesArePerInvocation(unittest.TestCase):
    def test_the_gate_writes_one_journal_per_invocation(self):
        """
        Reads the source rather than running the gate: executing it needs
        a real checkout, a frozen plan and TensorFlow. The defect this
        guards - a journal path keyed on the checkpoint - would silently
        discard the first invocation's record, so it is worth pinning
        even by inspection.
        """
        src = Path(gate.__file__).read_text()
        self.assertIn('f"exec_{inv}.json"', src)
        self.assertNotIn('f"exec_{cid}.json"', src)
        self.assertIn('f"{inv}.log"', src)
        self.assertNotIn('f"{cid}.log"', src)

    def test_the_journal_records_which_invocation_it_was(self):
        src = Path(gate.__file__).read_text()
        for field in ("invocation_id", "invocation_index",
                      "invocation_suffix", "sets_this_invocation"):
            self.assertIn(f'"{field}"', src)

    def test_the_checkpoint_is_rehashed_before_every_invocation(self):
        """
        Pooling assumes both invocations measured the same weights, so
        each must verify them rather than trusting the first copy.
        """
        src = Path(gate.__file__).read_text()
        self.assertIn("Re-hashed before EVERY invocation", src)


if __name__ == "__main__":
    unittest.main()
