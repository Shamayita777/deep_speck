"""
Tests for the corrections made in revision 2.

Each class here corresponds to a defect found in review of revision 1.
The tests are written to FAIL against revision 1's behaviour, so they are
regression tests for specific mistakes rather than general smoke tests:

  * a fabricated source digest accepted because a Boolean said "verified"
  * a freeze record accepted with an invented commit, a malformed
    timestamp, or a commit that does not contain the plan
  * a decision rule that concluded equivalence from interval containment
  * `--strict` that reported a failure and wrote its output anyway
  * missing evidence recorded as FAILED, and a stage summary asserting
    "verified" regardless of what its components found
  * an evidence map holding a content digest where the framework expects
    the sha256 of the file at that path
  * a II-1 component hard-wired to NOT_PERFORMED with no ingestion path
  * paper locators that cited the wrong table and a head-width claim the
    publication does not make

No test here runs training, search or an evaluation.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit.implementation.stages import StageOutcome
from audit.targets.autond import evaluation_results as evres
from audit.targets.autond import implementation_stages as st
from audit.targets.autond import paper_source_architecture as psa
from audit.targets.autond.preregistrations import autond_ii_v2 as plan
from audit.targets.autond.tests import portable


# ---------------------------------------------------------------------------
# 5. Preregistration integrity
# ---------------------------------------------------------------------------

class TestSourceHashesAreRecomputed(unittest.TestCase):
    def test_no_fabricated_digest_survives(self):
        """The old plan carried a fake eval_nets.py digest. It must be gone."""
        fake = ("838587127c08834f0dfe56dbbbaa0b4a6fd3b8b73e8a0b7a0e24e8e1e4e8e8e8")
        self.assertNotIn(fake, set(plan.TARGET["source_files"].values()))

    def test_every_recorded_digest_is_well_formed(self):
        for rel, digest in plan.TARGET["source_files"].items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", f"{rel}")
        for cid, rec in plan.TARGET["checkpoints"].items():
            self.assertRegex(rec["sha256"], r"^[0-9a-f]{64}$", f"{cid}")

    def test_no_verified_boolean_governs_the_gate(self):
        """A Boolean asserting a hash was checked must not exist."""
        for rec in plan.TARGET.get("source_files", {}):
            self.assertNotIsInstance(plan.TARGET["source_files"][rec], dict)
        ec = plan.TARGET.get("evaluation_code")
        if isinstance(ec, dict):
            self.assertNotIn("verified", ec,
                             "a mutable 'verified' flag must not gate the plan")

    def test_absent_checkout_fails_closed(self):
        ok, problems = plan.verify_source_hashes(None)
        self.assertFalse(ok)
        self.assertTrue(problems)
        self.assertIn("not verified", " ".join(problems).lower())

    def test_wrong_digests_are_detected(self):
        """A checkout whose files differ must be rejected, not accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = portable.fake_upstream(Path(tmp) / "up")
            ok, problems = plan.verify_source_hashes(fake["path"])
            self.assertFalse(ok, "a synthetic checkout must not verify")
            joined = " ".join(problems)
            self.assertIn("eval_nets.py", joined)
            self.assertTrue(any("!= recorded" in p for p in problems))

    def test_real_checkout_verifies(self):
        up = portable.require_real_upstream(self)
        ok, problems = plan.verify_source_hashes(up)
        self.assertTrue(ok, f"real checkout should verify; got {problems}")

    def test_gate_refuses_without_a_checkout(self):
        permitted, blockers = plan.confirmatory_run_permitted(None)
        self.assertFalse(permitted)
        self.assertTrue(blockers)


class TestFreezeRecordVerification(unittest.TestCase):
    """The sidecar must not be believed on its own say-so."""

    def _sidecar(self, **over):
        rec = {"preregistration": plan.PREREGISTRATION_ID,
               "plan_hash": plan.plan_hash(),
               "frozen_at_utc": "2026-10-15T12:00:00Z",
               "plan_file_commit": "a" * 40,
               "plan_file_path": "audit/targets/autond/preregistrations/"
                                 "autond_ii_v2.py",
               "plan_file_sha256": plan.plan_file_sha256(),
               "approved_by": "tester"}
        rec.update(over)
        return rec

    def test_nonexistent_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, problems = plan.verify_freeze_commit(self._sidecar(), Path(tmp))
            self.assertFalse(ok)
            self.assertTrue(any("does not exist" in p or "not a" in p
                                for p in problems), problems)

    def test_short_or_invented_commit_is_rejected(self):
        for bad in ("abc1234", "", "not-a-sha", "A" * 40, "a" * 39):
            with self.subTest(commit=bad):
                with tempfile.TemporaryDirectory() as tmp:
                    ok, problems = plan.verify_freeze_commit(
                        self._sidecar(plan_file_commit=bad), Path(tmp))
                    self.assertFalse(ok, f"{bad!r} must be rejected")

    def test_malformed_timestamp_is_rejected(self):
        import audit.targets.autond.preregistrations.autond_ii_v2 as mod
        for bad in ("2026-10-15", "15/10/2026", "2026-10-15 12:00:00",
                    "2026-10-15T12:00:00+05:30", "yesterday"):
            with self.subTest(ts=bad):
                self.assertIsNone(mod._ISO_UTC.match(bad),
                                  f"{bad!r} must not be accepted as a UTC instant")
        self.assertIsNotNone(mod._ISO_UTC.match("2026-10-15T12:00:00Z"))

    def test_incomplete_sidecar_is_rejected(self):
        import audit.targets.autond.preregistrations.autond_ii_v2 as mod
        for drop in ("plan_hash", "frozen_at_utc", "plan_file_commit",
                     "plan_file_path", "plan_file_sha256", "approved_by"):
            with self.subTest(missing=drop):
                rec = self._sidecar()
                rec[drop] = ""
                orig = mod.read_freeze_record
                mod.read_freeze_record = lambda r=rec: r
                try:
                    ok, detail = mod.is_frozen()
                    self.assertFalse(ok)
                    self.assertIn("incomplete", detail)
                finally:
                    mod.read_freeze_record = orig

    def test_stale_plan_hash_is_rejected(self):
        import audit.targets.autond.preregistrations.autond_ii_v2 as mod
        rec = self._sidecar(plan_hash="b" * 64)
        orig = mod.read_freeze_record
        mod.read_freeze_record = lambda r=rec: r
        try:
            ok, detail = mod.is_frozen()
            self.assertFalse(ok)
            self.assertIn("changed after it was frozen", detail)
        finally:
            mod.read_freeze_record = orig

    def test_the_plan_is_either_unfrozen_or_legitimately_frozen(self):
        """
        Revision 3a. This previously asserted the sidecar does not exist,
        which was right for the SHIPPED PACKAGE and wrong for the
        operator's working tree: the moment they performed the documented
        freeze, a correct action made the suite fail.

        The packaging rule it was really expressing now lives in
        tests/test_packaging.TestArchiveContract, where it belongs. Here
        the assertion is the stronger one: if no freeze record exists the
        plan must report unfrozen, and if one DOES exist it must be a
        legitimate freeze - the right plan hash, the plan file's own bytes,
        a real commit containing it, and a committed sidecar. A stale or
        hand-written sidecar fails this.
        """
        if not plan.FROZEN_SIDECAR.exists():
            ok, _ = plan.is_frozen()
            self.assertFalse(ok)
            return
        ok, detail = plan.is_frozen()
        self.assertTrue(
            ok,
            f"a freeze record exists at {plan.FROZEN_SIDECAR.name} but does "
            f"not verify: {detail}. Either complete the documented freeze "
            "procedure or remove the record; a sidecar that does not verify "
            "is worse than none.")

    def test_freeze_procedure_is_not_self_referential(self):
        """
        The old instructions asked for `git rev-parse HEAD` AFTER committing
        the sidecar, which the sidecar would then have to contain.
        """
        text = plan.freeze_instructions()
        self.assertIn("plan_file_commit", text)
        self.assertIn("two commits", text.lower())
        self.assertNotIn("after committing", text.lower())


# ---------------------------------------------------------------------------
# 6. The statistical decision rule
# ---------------------------------------------------------------------------

class TestStudentT(unittest.TestCase):
    def test_critical_values_match_published_tables(self):
        known = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 10: 2.228, 30: 2.042}
        for df, want in known.items():
            got = evres.student_t_ppf(0.975, df)
            self.assertAlmostEqual(got, want, places=2, msg=f"df={df}")

    def test_cdf_and_ppf_are_inverses(self):
        for df in (1, 4, 9, 25):
            for p in (0.01, 0.1, 0.5, 0.9, 0.99):
                t = evres.student_t_ppf(p, df)
                self.assertAlmostEqual(evres.student_t_cdf(t, df), p, places=6)

    def test_symmetry(self):
        for df in (2, 4, 7):
            self.assertAlmostEqual(evres.student_t_cdf(0.0, df), 0.5, places=12)
            self.assertAlmostEqual(evres.student_t_cdf(-1.3, df),
                                   1.0 - evres.student_t_cdf(1.3, df), places=10)


class TestEquivalenceMarginIsPrespecified(unittest.TestCase):
    def test_margin_depends_only_on_published_values(self):
        self.assertAlmostEqual(plan.equivalence_margin(0.5144), 0.00144, places=6)
        self.assertAlmostEqual(plan.equivalence_margin(0.5179), 0.00179, places=6)

    def test_margin_exceeds_reporting_granularity(self):
        self.assertGreater(plan.equivalence_margin(0.5144), 5e-5 * 10)

    def test_margin_exceeds_the_published_disagreement(self):
        """
        Stated as a consequence, not hidden: the margin is larger than the
        paper/README gap, so the experiment cannot separate them.
        """
        gap = abs(0.5182 - 0.5179)
        self.assertGreater(plan.equivalence_margin(0.5179), gap)
        self.assertEqual(plan.HYPOTHESES["H-AUTOND-SIMON-REPRO"]["prediction"],
                         plan.INCONCLUSIVE)


class TestDecisionRuleIsNotIntervalContainment(unittest.TestCase):
    def test_rule_text_does_not_conclude_from_containment(self):
        for name, h in plan.HYPOTHESES.items():
            with self.subTest(h=name):
                self.assertNotIn("interval of the five-set mean contains",
                                 h["supported_if"])
                self.assertIn("EQUIVALENT", h["supported_if"])

    def test_wide_data_does_not_conclude_equivalence(self):
        """
        The decisive regression test. These five accuracies are centred on
        the published value but so dispersed that nothing is established.
        Interval containment would have called this SUPPORTED; TOST must
        return INDETERMINATE.
        """
        noisy = [0.5144 + d for d in (-0.02, -0.015, -0.01, -0.005, -0.0025,
                                      0.0025, 0.005, 0.01, 0.015, 0.02)]
        res = plan.decide({"speck3264": noisy, "simon3264": noisy})
        self.assertEqual(res["per_comparison"]["CMP-SPECK-PAPER"]["outcome"],
                         plan.INDETERMINATE)
        self.assertEqual(res["hypotheses"]["H-AUTOND-SPECK-REPRO"]["decision"],
                         plan.INCONCLUSIVE)

    def test_tight_data_on_target_concludes_equivalence(self):
        # Offsets extended to the registered n = 10. Same SD scale as the
        # five-value version, so the fixture still asserts "tight and on
        # target concludes equivalence" rather than passing because a
        # larger n made any fixture significant.
        tight = [0.5144 + d for d in (-2.0e-5, -1.5e-5, -1.0e-5, -0.5e-5,
                                      -0.25e-5, 0.25e-5, 0.5e-5, 1.0e-5,
                                      1.5e-5, 2.0e-5)]
        res = plan.decide({"speck3264": tight, "simon3264": tight})
        self.assertEqual(res["per_comparison"]["CMP-SPECK-PAPER"]["outcome"],
                         plan.EQUIVALENT)
        self.assertEqual(res["hypotheses"]["H-AUTOND-SPECK-REPRO"]["decision"],
                         plan.SUPPORTED)

    def test_clearly_different_data_is_different_not_merely_inconclusive(self):
        low = [0.5050, 0.5048, 0.5052, 0.5049, 0.5051,
               0.5050, 0.5049, 0.5051, 0.5048, 0.5052]
        res = plan.decide({"speck3264": low, "simon3264": low})
        self.assertEqual(res["per_comparison"]["CMP-SPECK-PAPER"]["outcome"],
                         plan.DIFFERENT)
        self.assertEqual(res["hypotheses"]["H-AUTOND-SPECK-REPRO"]["decision"],
                         plan.NOT_SUPPORTED)

    def test_short_sample_is_inconclusive_not_averaged(self):
        res = plan.decide({"speck3264": [0.5144, 0.5143, 0.5145],
                           "simon3264": []})
        for cid, rec in res["per_comparison"].items():
            self.assertEqual(rec["outcome"], plan.INDETERMINATE, cid)
            self.assertIn("missing-data rule", rec.get("reason", ""))
        for name, h in res["hypotheses"].items():
            self.assertEqual(h["decision"], plan.INCONCLUSIVE, name)

    def test_simon_equivalent_to_both_is_inconclusive(self):
        """The predicted outcome: reproduces a value, cannot say which."""
        on_target = [0.51805 + d for d in (-1e-5, 0.0, 1e-5, -5e-6, 5e-6)]
        res = plan.decide({"speck3264": [0.5144] * 5, "simon3264": on_target})
        outs = {c: res["per_comparison"][c]["outcome"]
                for c in ("CMP-SIMON-PAPER", "CMP-SIMON-README")}
        if list(outs.values()).count(plan.EQUIVALENT) == 2:
            self.assertEqual(res["hypotheses"]["H-AUTOND-SIMON-REPRO"]["decision"],
                             plan.INCONCLUSIVE)


class TestHolmProcedure(unittest.TestCase):
    def test_thresholds_are_alpha_over_m_minus_i_plus_one(self):
        res = plan.holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
        self.assertAlmostEqual(res["a"]["threshold"], 0.05 / 3)
        self.assertAlmostEqual(res["b"]["threshold"], 0.05 / 2)
        self.assertAlmostEqual(res["c"]["threshold"], 0.05 / 1)

    def test_step_down_stops_at_first_failure(self):
        """
        Ranks are 0.001 (thr 0.0167), 0.03 (thr 0.025), 0.04 (thr 0.05).
        Rank 2 fails, so rank 3 must NOT be rejected even though 0.04 is
        below its own 0.05 threshold. That is the step-down property, and
        it is the only thing separating Holm from per-test alpha.
        """
        res = plan.holm({"a": 0.001, "b": 0.03, "c": 0.04}, alpha=0.05)
        self.assertEqual(res["a"]["rank"], 1)
        self.assertEqual(res["b"]["rank"], 2)
        self.assertEqual(res["c"]["rank"], 3)
        self.assertTrue(res["a"]["reject"])
        self.assertFalse(res["b"]["reject"], "0.03 is above alpha/2")
        self.assertFalse(res["c"]["reject"],
                         "a later rank must not be rejected after a failure, "
                         "even though 0.04 < 0.05")

    def test_ordering_is_by_pvalue_not_by_key(self):
        res = plan.holm({"z": 0.001, "a": 0.49}, alpha=0.05)
        self.assertEqual(res["z"]["rank"], 1)
        self.assertEqual(res["a"]["rank"], 2)

    def test_family_size_is_three(self):
        self.assertEqual(len(plan.COMPARISONS), 3)
        self.assertEqual(plan.DECISION_RULE["family_size"], 3)


class TestPowerIsComputedNotAsserted(unittest.TestCase):
    def test_power_is_reported_with_its_method(self):
        """
        Revision 3c: the normal approximation with a fixed SE is gone, so
        this no longer looks for the word "approximation" or for
        `approx_power`. It asserts the replacement states that the
        variance is estimated, and that a real power comes back.
        """
        p = plan.power_at(plan.equivalence_margin(0.5144), m=3)
        self.assertIn("estimated from the sample", p["method"])
        self.assertNotIn("approx_power", p)
        self.assertGreater(p["power"], 0.0)
        self.assertLessEqual(p["power"], 1.0)

    def test_conservative_holm_step_lowers_power(self):
        d = plan.equivalence_margin(0.5144)
        self.assertLess(plan.power_at(d, m=3)["power"],
                        plan.power_at(d, m=1)["power"])

    def test_planning_sd_is_declared_design_only(self):
        self.assertIn("planning_use_only", plan.HISTORICAL_UNRETAINED)
        self.assertIn("design", plan.HISTORICAL_UNRETAINED["planning_use_only"])


# ---------------------------------------------------------------------------
# 7. Evaluation gate and result ingestion
# ---------------------------------------------------------------------------

class TestEvaluationGate(unittest.TestCase):
    def test_gate_refuses_while_unfrozen(self):
        from audit.targets.autond import evaluation_gate as gate
        # force_unfrozen so this tests the gate's logic rather than the
        # operator's current repository state.
        with portable.force_unfrozen(plan), tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            pf = gate.preflight(up, Path(tmp) / "out")
            self.assertFalse(pf["permitted"])
            self.assertTrue(any("not frozen" in b for b in pf["blockers"]))

    def test_gate_refuses_output_inside_the_checkout(self):
        from audit.targets.autond import evaluation_gate as gate
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            pf = gate.preflight(up, up / "inside")
            self.assertTrue(any("inside the upstream checkout" in b
                                for b in pf["blockers"]), pf["blockers"])

    def test_launch_returns_nonzero_and_runs_nothing(self):
        from audit.targets.autond import evaluation_gate as gate
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            out = Path(tmp) / "out"
            rc = gate.launch(up, out, execute=True)
            self.assertEqual(rc, 1)
            self.assertFalse(out.exists(),
                             "a refused launch must not create its output dir")

    def test_planned_commands_write_outside_the_checkout(self):
        from audit.targets.autond import evaluation_gate as gate
        with tempfile.TemporaryDirectory() as tmp:
            up = Path(tmp) / "up"
            out = Path(tmp) / "out"
            for c in gate.planned_commands(up, out):
                self.assertTrue(str(c["copy_to"]).startswith(str(out)))
                self.assertFalse(str(c["copy_to"]).startswith(str(up)))


class TestResultIngestion(unittest.TestCase):
    """
    Bundle-level validation moved to test_r3_corrections.

    Revision 2 tested the validator against a hand-written bundle
    fixture. That fixture is gone, because revision 3 can build a real
    one: `tests.portable.staged_run_dir` lays out a run directory with
    real NPZs and journals and `evaluation_assemble.assemble()` derives
    RESULT.json from it. Those tests live in
    test_r3_corrections.TestAssembly and
    test_r3_corrections.TestValidatorHardening, which is also where the
    hardening this review asked for is pinned.

    What remains here are the checks that need no conforming bundle.
    """

    def test_absent_bundle_is_not_a_silent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = evres.validate_bundle(Path(tmp) / "nope")
            self.assertFalse(res["ok"])
            self.assertTrue(res["problems"])

    def test_a_directory_without_result_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "b"
            d.mkdir()
            res = evres.validate_bundle(d)
            self.assertFalse(res["ok"])
            self.assertTrue(any("RESULT.json" in p for p in res["problems"]))

    def test_unparseable_result_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "b"
            d.mkdir()
            (d / "RESULT.json").write_text("{not json")
            res = evres.validate_bundle(d)
            self.assertFalse(res["ok"])
            self.assertTrue(any("not readable JSON" in p
                                for p in res["problems"]))

    def test_component_is_not_hardwired_to_not_performed(self):
        """The ingestion branch must exist and be reachable."""
        src = Path(st.__file__).read_text()
        self.assertIn("_ingest_evaluation", src)
        self.assertIn("StageOutcome.COMPLETED", src)
        self.assertIn("EVALUATION_BUNDLE_DIR", src)

    def test_unfrozen_plan_cannot_yield_a_valid_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            d = repo / st.EVALUATION_BUNDLE_DIR
            d.mkdir(parents=True)
            (d / "RESULT.json").write_text("{}")
            res = st._ingest_evaluation(repo)
            self.assertNotEqual(res["state"], "VALID")

    def test_absent_bundle_directory_is_reported_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = st._ingest_evaluation(Path(tmp) / "repo")
            self.assertEqual(res["state"], "ABSENT")


# ---------------------------------------------------------------------------
# 3. R1 evidence validation
# ---------------------------------------------------------------------------

class TestR1Validation(unittest.TestCase):
    def _validate(self, bundle: Path):
        checks = [st.ArtifactCheck(
            "r1_search_manifest", str(bundle / "MANIFEST.json"), True,
            "0" * 64, None, "OPEN", True, "RECORDED_ONLY", "")]
        return st.validate_r1(bundle, bundle, bundle, {c.label: c for c in checks})

    def test_absent_manifest_yields_no_science(self):
        res = st.validate_r1(Path("/nonexistent"), Path("/nonexistent"),
                             Path("/nonexistent"), {})
        self.assertFalse(res["integrity_ok"])
        self.assertIsNone(res["science"])

    def test_wellformed_bundle_validates_and_reads_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = portable.fake_search_bundle(Path(tmp) / "r1")
            res = self._validate(b)
            if res["harness"] is None:
                self.skipTest("search harness not installed; "
                              f"{res['problems'][:1]}")
            self.assertTrue(res["integrity_ok"], res["problems"])
            self.assertTrue(res["science"]["recovered_expected_difference"])

    def test_torn_log_is_rejected_and_yields_no_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = portable.fake_search_bundle(Path(tmp) / "r1", torn_log=True)
            res = self._validate(b)
            if res["harness"] is None:
                self.skipTest("search harness not installed")
            self.assertFalse(res["integrity_ok"])
            self.assertIsNone(res["science"],
                              "a torn log must not be read as a valid result")

    def test_wrong_source_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = portable.fake_search_bundle(Path(tmp) / "r1", wrong_commit=True)
            res = self._validate(b)
            if res["harness"] is None:
                self.skipTest("search harness not installed")
            self.assertFalse(res["integrity_ok"])
            self.assertIsNone(res["science"])

    def test_completed_status_is_case_insensitive(self):
        for status in ("completed", "COMPLETED", "Completed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                bundle = portable.fake_search_bundle(
                    Path(tmp) / "r1", status=status
                )
                res = self._validate(bundle)
                if res["harness"] is None:
                    self.skipTest("search harness not installed")
                self.assertTrue(
                    res["integrity_ok"],
                    f"status={status!r}; problems={res['problems']}",
                )

    def test_a_non_string_status_is_rejected_not_crashed(self):
        """The companion to the case-insensitivity fix: a null or numeric
        status must be refused, not raise inside the comparison."""
        for status in (None, 0, 1, [], {}):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                bundle = portable.fake_search_bundle(
                    Path(tmp) / "r1", status=status
                )
                res = self._validate(bundle)
                if res["harness"] is None:
                    self.skipTest("search harness not installed")
                self.assertFalse(res["integrity_ok"])
                self.assertIsNone(res["science"])

    def test_incomplete_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = portable.fake_search_bundle(Path(tmp) / "r1", status="crashed")
            res = self._validate(b)
            if res["harness"] is None:
                self.skipTest("search harness not installed")
            self.assertFalse(res["integrity_ok"])
            self.assertTrue(any("not 'completed'" in p for p in res["problems"]),
                            res["problems"])

    def test_behaviour_changing_patch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = portable.fake_search_bundle(Path(tmp) / "r1",
                                            behaviour_changing_patch=True)
            res = self._validate(b)
            if res["harness"] is None:
                self.skipTest("search harness not installed")
            self.assertFalse(res["integrity_ok"])
            self.assertTrue(any("behaviour-changing" in p
                                for p in res["problems"]), res["problems"])

    def test_manifest_not_covering_the_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = portable.fake_search_bundle(Path(tmp) / "r1",
                                            drop_manifest_entry=True)
            res = self._validate(b)
            if res["harness"] is None:
                self.skipTest("search harness not installed")
            self.assertFalse(res["integrity_ok"])

    def test_validation_reuses_the_harness_rather_than_reimplementing(self):
        src = Path(st.__file__).read_text()
        self.assertIn("verify_outputs", src)
        self.assertIn("validation-only", src)


# ---------------------------------------------------------------------------
# 4. Stage records
# ---------------------------------------------------------------------------

class TestMissingEvidenceIsNotFailed(unittest.TestCase):
    def test_absent_required_artifact_maps_to_missing_not_mismatch(self):
        checks = [st.ArtifactCheck("x", "/nope", False, None, "a" * 64,
                                   "RECORDED", True, "MISSING", "")]
        state, problems = st._evidence_state(checks)
        self.assertEqual(state, "MISSING")
        self.assertTrue(problems)

    def test_hash_mismatch_maps_to_mismatch(self):
        checks = [st.ArtifactCheck("x", "/here", True, "b" * 64, "a" * 64,
                                   "RECORDED", True, "HASH_MISMATCH", "")]
        state, _ = st._evidence_state(checks)
        self.assertEqual(state, "MISMATCH")

    def test_missing_evidence_gives_not_performed_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            stage, _ = st.build_ii3(repo, Path(tmp), up)
            ids = {c.component_id: c.status for c in stage.components}
            self.assertEqual(ids["II3-ARCHITECTURE-CENSUS"],
                             StageOutcome.NOT_PERFORMED)
            self.assertEqual(ids["II3-CHECKPOINT-LOAD"],
                             StageOutcome.NOT_PERFORMED)
            for c in stage.components:
                if c.status is StageOutcome.NOT_PERFORMED:
                    self.assertNotIn("verified", c.summary.lower())


class TestStageSummaryReflectsComponents(unittest.TestCase):
    def test_summary_does_not_assert_verification_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            stage, _ = st.build_ii3(repo, Path(tmp), up)
            self.assertNotIn("tensor structure and checkpoint loading verified",
                             stage.summary)
            self.assertIn("NOT_PERFORMED", stage.summary)
            self.assertIn("satisfied", stage.summary)

    def test_summary_counts_match_the_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            for stage, _ in (st.build_ii1(repo, Path(tmp), up),
                             st.build_ii3(repo, Path(tmp), up)):
                n = len(stage.components)
                sat = sum(1 for c in stage.components
                          if c.status in st.SATISFYING)
                self.assertIn(f"{sat}/{n} component(s) satisfied", stage.summary)

    def test_satisfying_set_comes_from_the_framework(self):
        from audit.implementation.stages import SATISFYING_OUTCOMES
        self.assertIs(st.SATISFYING, SATISFYING_OUTCOMES)


class TestEvidenceMapsHoldRealFileHashes(unittest.TestCase):
    def test_every_evidence_value_is_the_sha256_of_the_file_at_its_key(self):
        """
        The old II-3 bound the comparison's CONTENT digest to the path of
        a source file, so the value was not that file's hash.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            for stage, _ in (st.build_ii1(repo, Path(tmp), up),
                             st.build_ii3(repo, Path(tmp), up)):
                for comp in stage.components:
                    for path, digest in comp.evidence.items():
                        p = Path(path)
                        self.assertRegex(digest, r"^[0-9a-f]{64}$",
                                         f"{comp.component_id}:{path}")
                        if p.is_file():
                            self.assertEqual(
                                digest, st.sha256_file(p),
                                f"{comp.component_id}: evidence value for {path} "
                                "is not that file's sha256")

    def test_record_digest_is_not_bound_as_a_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            stage, _ = st.build_ii3(repo, Path(tmp), up)
            rd = psa.record_digest()
            for comp in stage.components:
                self.assertNotIn(rd, set(comp.evidence.values()),
                                 "the comparison's content digest must not "
                                 "appear as a file hash")

    def test_written_record_file_hash_differs_from_record_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = psa.write_record(Path(tmp) / "rec.json")
            self.assertNotEqual(info["sha256"], info["record_digest"])
            self.assertEqual(
                info["sha256"],
                st.sha256_file(Path(tmp) / "rec.json"))

    def test_the_comparison_can_be_serialised_and_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rec.json"
            psa.write_record(p)
            rec = json.loads(p.read_text())
            self.assertEqual(rec["record_digest"], psa.record_digest())
            self.assertEqual(len(rec["claims"]), len(psa.CLAIMS))
            self.assertIn("paper", rec)
            self.assertIn("corrections_from_previous_revision", rec)


class TestStrictFailsClosed(unittest.TestCase):
    def test_strict_marks_problems_and_record_is_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            rec = st.build_record(repo, Path(tmp), up, strict=True)
            self.assertTrue(rec["strict_problems"])
            self.assertFalse(rec["ok"])

    def test_cli_writes_nothing_when_strict_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            out = Path(tmp) / "out.json"
            rc = st.main(["--repo-root", str(repo), "--home", tmp,
                          "--autond-root", str(up), "--strict",
                          "--output", str(out)])
            self.assertEqual(rc, 1)
            self.assertFalse(out.exists(),
                             "a failed strict run must leave no output file")

    def test_hash_mismatch_fails_closed_without_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            cen = repo / "audit/targets/autond/evidence/census"
            cen.mkdir(parents=True)
            for name in ("census_v2_20261015T043855Z.json",
                         "arch_reconciliation_20261015T043855Z.json",
                         "load_verify_r2_20261015T051547Z.json"):
                (cen / name).write_text("{}")
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            rec = st.build_record(repo, Path(tmp), up, strict=False)
            # Revision 3 separates the gates: a hash mismatch blocks
            # regardless of --strict, so it is a blocking problem rather
            # than a strict one.
            self.assertTrue(rec["blocking_problems"])
            self.assertFalse(rec["ok"])


# ---------------------------------------------------------------------------
# 2. The paper-to-source comparison
# ---------------------------------------------------------------------------

class TestPaperComparison(unittest.TestCase):
    def test_dilation_rates_are_derived_not_transcribed(self):
        self.assertEqual(psa.DILATION_RATES, [31, 15, 7, 3])
        self.assertEqual(psa._source_dilation_rates(64), [31, 15, 7, 3])

    def test_filter_schedule_follows_32_plus_16i(self):
        self.assertEqual(psa.FILTER_SCHEDULE, [32, 48, 64, 80])

    def test_width_sequence_is_the_realised_one(self):
        self.assertEqual(psa.WIDTH_SEQUENCE, [64, 33, 18, 11, 8])
        self.assertNotEqual(psa.WIDTH_SEQUENCE, [64, 32, 16, 8, 4])

    def test_flatten_width_matches_the_census(self):
        self.assertEqual(psa.WIDTH_SEQUENCE[-1] * psa.FILTER_SCHEDULE[-1], 640)

    def test_dilation_locator_names_table_3_not_table_4(self):
        c = {x.claim_id: x for x in psa.CLAIMS}["ARCH-03"]
        self.assertIn("Table 3", c.paper_locator)
        self.assertNotIn("Table 4", c.paper_locator)

    def test_parameter_count_locator_names_table_4(self):
        c = {x.claim_id: x for x in psa.CLAIMS}["ARCH-12"]
        self.assertIn("Table 4", c.paper_locator)
        self.assertIn("298", c.paper_says)

    def test_head_widths_are_not_claimed_as_the_papers(self):
        """
        The source head IS 256/256/64, but no paper statement of it was
        located. Claiming agreement would repeat the retracted error.
        """
        c = {x.claim_id: x for x in psa.CLAIMS}["ARCH-09"]
        self.assertIs(c.verdict, psa.Verdict.UNDERSPECIFIED)
        self.assertIn("256", c.source_does)
        self.assertIn("Nothing located", c.paper_says)

    def test_every_claim_has_both_locators(self):
        for c in psa.CLAIMS:
            self.assertTrue(c.paper_locator.strip(), c.claim_id)
            self.assertTrue(c.source_locator.strip(), c.claim_id)

    def test_agreeing_claims_cite_a_source_line(self):
        for c in psa.CLAIMS:
            if c.verdict is psa.Verdict.AGREES:
                self.assertIn("dbitnet.py", c.source_locator.lower() + " "
                              + c.source_locator, c.claim_id)

    def test_paper_version_is_bound_and_hash_is_marked_unverified(self):
        self.assertEqual(psa.PAPER["n_revisions"], 4)
        self.assertEqual(psa.PAPER["last_revision"], "2024-01-24")
        self.assertIsNone(psa.PAPER["pdf_sha256"])
        self.assertIn("UNVERIFIED", psa.PAPER["pdf_sha256_status"])

    def test_corrections_from_revision_1_are_recorded(self):
        ids = {c["claim_id"] for c in psa.CORRECTIONS}
        self.assertEqual(ids, {"ARCH-03", "ARCH-06", "ARCH-09", "ARCH-12"})
        for c in psa.CORRECTIONS:
            self.assertTrue(c["why"].strip())

    def test_contested_claim_is_documented_not_silently_dropped(self):
        arch09 = [c for c in psa.CORRECTIONS if c["claim_id"] == "ARCH-09"][0]
        self.assertIn("contested_claim", arch09)
        self.assertIn("Section 6.2", arch09["contested_claim"])

    def test_summary_is_not_the_stale_3_0_9_1(self):
        s = psa.summarise()
        self.assertNotEqual(
            (s["by_verdict"].get("AGREES"), s["by_verdict"].get("DIFFERS"),
             s["by_verdict"].get("UNDERSPECIFIED"),
             s["by_verdict"].get("NOT_CHECKABLE_HERE")),
            (3, 0, 9, 1))
        self.assertEqual(s["total_claims"], len(psa.CLAIMS))

    def test_verify_paper_pdf_reports_absence_rather_than_guessing(self):
        rec = psa.verify_paper_pdf(Path("/nonexistent/paper.pdf"))
        self.assertFalse(rec["present"])
        self.assertNotIn("sha256", rec)

    def test_pinned_commit_comes_from_the_package(self):
        from audit.targets.autond import PINNED_UPSTREAM_COMMIT
        self.assertEqual(psa.PINNED_COMMIT, PINNED_UPSTREAM_COMMIT)
        self.assertEqual(st.PINNED_COMMIT, PINNED_UPSTREAM_COMMIT)


# ---------------------------------------------------------------------------
# 9. Provenance completeness and portability
# ---------------------------------------------------------------------------

class TestRunManifestBindsGenericHelpers(unittest.TestCase):
    def test_every_generic_module_the_adapter_imports_is_bound(self):
        import ast

        from audit.targets.autond import run_autond_implementation_audit as runner

        bound = set(runner.CODE_PATHS)
        modules = [st, psa, plan, evres, runner]
        missing: list[str] = []
        for mod in modules:
            tree = ast.parse(Path(mod.__file__).read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                elif isinstance(node, ast.Import):
                    names.extend(a.name for a in node.names)
                for name in names:
                    if not name.startswith(("audit.common", "audit.implementation")):
                        continue
                    rel = name.replace(".", "/") + ".py"
                    if rel not in bound:
                        missing.append(f"{mod.__name__} imports {name} "
                                       f"({rel}) which CODE_PATHS does not bind")
        self.assertEqual(missing, [], "\n".join(missing))

    def test_training_provenance_artifacts_are_expected(self):
        labels = {a.label for a in st.EXPECTED}
        for needed in ("training_checkpoint_h5", "training_source_snapshot",
                       "training_environment", "training_results_summary",
                       "training_bundle_manifest"):
            self.assertIn(needed, labels)
        for r in (5, 6, 7, 8, 9):
            self.assertIn(f"training_history_round{r}", labels)

    def test_no_test_module_hardcodes_a_machine_path(self):
        """
        Checks EVERY module in this package, including this one. The
        needles are assembled from fragments so that this test does not
        contain the literals it forbids - otherwise it would either flag
        itself or need an exemption, and an exempted checker is not one.
        """
        needles = ["/" + "home/" + "claude", "/" + "Users" + "/",
                   "C:" + chr(92) + chr(92)]
        pkg = Path(st.__file__).parent
        offenders = []
        for f in sorted(pkg.rglob("*.py")):
            if "search" in f.parts:      # the vendored harness is not ours
                continue
            text = f.read_text()
            for needle in needles:
                if needle in text:
                    offenders.append(f"{f.relative_to(pkg)} contains {needle!r}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_expected_artifact_paths_are_relative(self):
        for a in st.EXPECTED:
            self.assertFalse(Path(a.path).is_absolute(), a.label)
            self.assertIn(a.relative_to, ("repo", "home", "upstream"), a.label)


class TestNothingExecutesAnExperiment(unittest.TestCase):
    MODULES = ("implementation_stages", "paper_source_architecture",
               "evaluation_results")

    def test_no_training_or_search_import(self):
        import importlib

        for name in self.MODULES:
            mod = importlib.import_module(f"audit.targets.autond.{name}")
            src = Path(mod.__file__).read_text()
            for bad in ("import tensorflow", "from tensorflow",
                        "import keras", "from keras"):
                self.assertNotIn(bad, src, f"{name}: {bad}")

    def test_the_only_subprocess_use_is_git(self):
        import ast

        tree = ast.parse(Path(st.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "run":
                first = node.args[0] if node.args else None
                if isinstance(first, ast.List) and first.elts:
                    head = first.elts[0]
                    if isinstance(head, ast.Constant):
                        self.assertEqual(head.value, "git",
                                         "the only subprocess this module may "
                                         "run is git")


if __name__ == "__main__":
    unittest.main(verbosity=2)
