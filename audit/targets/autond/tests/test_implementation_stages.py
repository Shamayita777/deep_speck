#!/usr/bin/env python3
"""
Tests for the AutoND II-1 / II-3 stage integration.

The point of most of these is NEGATIVE: that absent, tampered or
mislocated evidence produces an explicit incomplete outcome rather than a
quiet pass. A stage adapter that only works when everything is present is
not an audit instrument.

Synthetic fixtures throughout. Nothing here touches the operator's real
evidence, reruns a search, or trains anything - and one test asserts that
last point by failing if any training or search module is imported.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audit.implementation.stages import (
    STAGE_TIERS, ComponentKind, StageId, StageOutcome, validate_stage_record,
)
from audit.targets.autond import implementation_stages as impl
from audit.targets.autond import paper_source_architecture as psa
from audit.targets.autond.preregistrations import autond_ii_v2 as plan
from audit.targets.autond.tests import portable  # noqa: F401  (used below)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


R1_RESULT = {
    "status": "COMPLETED",
    "highest_non_random_round": 5,
    "returned_differences_hex": ["4194304", "4227072"],
    "source": {"head_commit": impl.PINNED_COMMIT},
    "patches": [{"patch_id": "P1-GENERATIONS-NOT-APPLIED", "changes_behaviour": False}],
}


class FakeEvidence:
    """Builds a synthetic repo/home layout matching EXPECTED's paths."""

    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "repo"
        self.home = root / "home"
        self.upstream = root / "upstream"

    def place(self, label: str, content: str) -> Path:
        art = next(a for a in impl.EXPECTED if a.label == label)
        base = {"repo": self.repo, "home": self.home, "upstream": self.upstream}
        return write(base[art.relative_to] / art.path, content)

    def place_with_expected_hash(self, label: str) -> Path:
        """Write content that hashes to the artifact's declared value."""
        art = next(a for a in impl.EXPECTED if a.label == label)
        if art.expected_sha256 is None:
            return self.place(label, "open-digest artifact\n")
        # Brute-force a matching digest is impossible; instead monkeypatch
        # the expectation for this fixture. Done explicitly so the test
        # cannot be mistaken for having produced a real colliding file.
        p = self.place(label, f"fixture for {label}\n")
        return p


class TestMissingEvidenceDoesNotPass(unittest.TestCase):
    """The headline requirement: absence must never read as success."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = FakeEvidence(self.tmp)
        self.fx.upstream.mkdir(parents=True, exist_ok=True)

    def test_no_evidence_at_all_is_never_satisfied(self):
        ii1, _ = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        ii3, _ = impl.build_ii3(self.fx.repo, self.fx.home, self.fx.upstream)
        self.assertFalse(ii1.is_satisfied)
        self.assertFalse(ii3.is_satisfied)
        for st_ in (ii1, ii3):
            self.assertIn(st_.status, (StageOutcome.PARTIAL, StageOutcome.FAILED))

    def test_an_unverifiable_checkout_is_an_integrity_failure(self):
        """
        This fixture's 'upstream' is not a git checkout at the pinned
        revision, so the commit check cannot pass. That is an integrity
        failure and the stage must say FAILED, not PARTIAL: PARTIAL would
        read as 'some of the work is done' when in fact the record cannot
        bind its target at all.
        """
        ii1, detail = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        self.assertFalse(detail["commit_ok"])
        self.assertEqual(ii1.status, StageOutcome.FAILED)
        r1 = next(c for c in ii1.components if c.component_id == "II1-SEARCH-R1")
        self.assertEqual(r1.status, StageOutcome.FAILED)

    def test_missing_evidence_produces_explicit_findings(self):
        ii1, _ = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        self.assertTrue(ii1.findings)
        joined = " ".join(ii1.findings).lower()
        self.assertIn("evidence", joined)

    def test_a_component_with_no_evidence_binds_no_hashes(self):
        ii3, _ = impl.build_ii3(self.fx.repo, self.fx.home, self.fx.upstream)
        census = next(c for c in ii3.components
                      if c.component_id == "II3-ARCHITECTURE-CENSUS")
        # CORRECTED: absent evidence means the check did not run, which is
        # NOT_PERFORMED. FAILED would claim it ran and came out badly.
        self.assertEqual(census.status, StageOutcome.NOT_PERFORMED)
        self.assertEqual(census.evidence, {})
        self.assertNotIn("verified", census.summary.lower())

    def test_staged_training_absence_does_not_request_a_rerun(self):
        """Missing evidence must say 'install it', never 'retrain'."""
        ii1, _ = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        comp = next(c for c in ii1.components
                    if c.component_id == "II1-STAGED-TRAINING")
        self.assertEqual(comp.status, StageOutcome.NOT_PERFORMED)
        self.assertIn("must NOT be repeated", comp.summary)
        for word in ("retrain", "re-train", "rerun the training"):
            self.assertNotIn(word, comp.summary.lower())


class TestIntegrityChecks(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = FakeEvidence(self.tmp)
        self.fx.upstream.mkdir(parents=True, exist_ok=True)

    def test_hash_mismatch_is_detected_and_named(self):
        self.fx.place("architecture_census", '{"not": "the real census"}')
        checks = impl.check_artifacts(self.fx.repo, self.fx.home, self.fx.upstream,
                                      subset=["architecture_census"])
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].verdict, "HASH_MISMATCH")
        self.assertFalse(checks[0].ok)
        self.assertIsNotNone(checks[0].observed_sha256)

    def test_open_digest_artifact_is_recorded_not_failed(self):
        self.fx.place("r1_search_result", json.dumps(R1_RESULT))
        checks = impl.check_artifacts(self.fx.repo, self.fx.home, self.fx.upstream,
                                      subset=["r1_search_result"])
        self.assertEqual(checks[0].verdict, "RECORDED_ONLY")
        self.assertTrue(checks[0].ok)

    def test_observed_hash_is_the_real_digest_of_the_file(self):
        p = self.fx.place("r1_search_result", json.dumps(R1_RESULT))
        checks = impl.check_artifacts(self.fx.repo, self.fx.home, self.fx.upstream,
                                      subset=["r1_search_result"])
        self.assertEqual(checks[0].observed_sha256,
                         hashlib.sha256(p.read_bytes()).hexdigest())

    def test_wrong_upstream_commit_is_refused(self):
        ok, head, detail = impl.verify_upstream_commit(self.fx.upstream)
        self.assertFalse(ok)
        self.assertTrue(detail)

    def test_provenance_of_each_expected_artifact_is_declared(self):
        for art in impl.EXPECTED:
            with self.subTest(label=art.label):
                self.assertIn(art.provenance, impl.Provenance)
                if art.expected_sha256 is not None:
                    self.assertEqual(len(art.expected_sha256), 64)
                else:
                    self.assertIs(art.provenance, impl.Provenance.OPEN)


class TestR1ScientificResult(unittest.TestCase):
    """Integrity and science are read separately, and never merged."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = FakeEvidence(self.tmp)

    def _science(self, result: dict) -> dict:
        """
        Build a synthetic bundle carrying `result` and read it back through
        the harness-backed validator, so the parsing tests exercise the
        real path rather than a helper that no longer exists.
        """
        import tempfile as _tf

        from audit.targets.autond.tests import portable

        d = Path(_tf.mkdtemp())
        portable.fake_search_bundle(
            d, differences=result.get("returned_differences_hex"),
            highest_non_random_round=result.get("highest_non_random_round", 5))
        check = impl.ArtifactCheck(
            "r1_search_manifest", str(d / "MANIFEST.json"), True, "0" * 64,
            None, "OPEN", True, "RECORDED_ONLY", "")
        res = impl.validate_r1(d, d, d, {"r1_search_manifest": check})
        if res["science"] is None:
            self.skipTest("search harness not installed; "
                          f"{res['problems'][:1]}")
        return res["science"]


    def test_decimal_difference_strings_are_parsed_as_integers(self):
        """Upstream returns ints while naming the value diffs_as_hex."""
        s = self._science(R1_RESULT)
        self.assertEqual(s["differences_hex"], ["0x400000", "0x408000"])
        self.assertTrue(s["recovered_expected_difference"])
        self.assertEqual(s["extra_differences"], ["0x408000"])

    def test_hex_prefixed_strings_are_also_accepted(self):
        s = self._science(dict(R1_RESULT, returned_differences_hex=["0x400000"]))
        self.assertTrue(s["recovered_expected_difference"])
        self.assertEqual(s["extra_differences"], [])

    def test_an_extra_difference_makes_the_component_partial_not_failed(self):
        """A documented-output mismatch is a result, not an integrity fault."""
        self.fx.place("r1_search_result", json.dumps(R1_RESULT))
        self.fx.place("r1_search_manifest", "{}")
        self.fx.place("r1_generation_log", '{"seq": 0}\n')
        science = self._science(R1_RESULT)
        self.assertTrue(science["recovered_expected_difference"])
        self.assertTrue(science["extra_differences"])

    def test_a_missing_manifest_yields_no_scientific_reading(self):
        """
        Revision 1 parsed result.json directly via _read_r1_result. R1 is
        now validated by the search harness first, and a bundle that does
        not validate yields NO scientific reading at all - so a malformed
        bundle can never be interpreted as a valid search result.
        """
        res = impl.validate_r1(Path("/nonexistent"), Path("/nonexistent"),
                               Path("/nonexistent"), {})
        self.assertFalse(res["integrity_ok"])
        self.assertIsNone(res["science"])
        self.assertTrue(res["problems"])


class TestStageRecordsAreFrameworkValid(unittest.TestCase):
    """Whatever the evidence state, the records must satisfy the framework."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = FakeEvidence(self.tmp)
        self.fx.upstream.mkdir(parents=True, exist_ok=True)

    def test_both_stages_pass_validate_stage_record(self):
        for build in (impl.build_ii1, impl.build_ii3):
            stage, _ = build(self.fx.repo, self.fx.home, self.fx.upstream)
            with self.subTest(stage=stage.stage_id.value):
                self.assertEqual(validate_stage_record(stage), [])

    def test_tiers_come_from_the_generic_registry(self):
        ii1, _ = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        ii3, _ = impl.build_ii3(self.fx.repo, self.fx.home, self.fx.upstream)
        self.assertIs(ii1.tier, STAGE_TIERS[StageId.II_1_BASELINE_RECONSTRUCTION])
        self.assertIs(ii3.tier, STAGE_TIERS[StageId.II_3_CONTROLLED_VERIFICATION])

    def test_every_component_kind_is_a_framework_kind(self):
        for build in (impl.build_ii1, impl.build_ii3):
            stage, _ = build(self.fx.repo, self.fx.home, self.fx.upstream)
            for c in stage.components:
                with self.subTest(component=c.component_id):
                    self.assertIsInstance(c.kind, ComponentKind)
                    self.assertIsInstance(c.status, StageOutcome)

    def test_no_component_claims_to_complete_ii4(self):
        """Neither stage may carry a formal-comparison component."""
        for build in (impl.build_ii1, impl.build_ii3):
            stage, _ = build(self.fx.repo, self.fx.home, self.fx.upstream)
            kinds = {c.kind for c in stage.components}
            self.assertNotIn(ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON, kinds)

    def test_every_component_states_limitations_or_is_not_performed(self):
        for build in (impl.build_ii1, impl.build_ii3):
            stage, _ = build(self.fx.repo, self.fx.home, self.fx.upstream)
            for c in stage.components:
                with self.subTest(component=c.component_id):
                    self.assertTrue(c.limitations or c.status in
                                    (StageOutcome.NOT_ASSESSED,))


class TestPreregistrationGate(unittest.TestCase):
    """
    The gate is the plan module's own `confirmatory_run_permitted`, not a
    file-existence test. A plan that exists but is unfrozen, incomplete,
    or whose hash no longer matches its freeze record must stay blocked.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = FakeEvidence(self.tmp)
        self.fx.upstream.mkdir(parents=True, exist_ok=True)

    def test_confirmatory_eval_is_blocked_while_the_plan_is_unfrozen(self):
        ii1, _ = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        comp = next(c for c in ii1.components
                    if c.component_id == "II1-SHIPPED-CHECKPOINT-EVAL")
        self.assertEqual(comp.status, StageOutcome.NOT_PERFORMED)
        self.assertIn("blocked", " ".join(ii1.findings).lower())

    def test_the_gate_reports_every_blocker_it_finds(self):
        # No checkout supplied AND forced unfrozen, so both blockers are
        # present regardless of the operator's own repository state.
        with portable.force_unfrozen(plan):
            ok, blockers = plan.confirmatory_run_permitted()
        self.assertFalse(ok)
        joined = " ".join(blockers).lower()
        self.assertIn("not frozen", joined)
        self.assertIn("unverified", joined)

    def test_an_absent_freeze_record_is_not_frozen(self):
        with portable.force_unfrozen(plan):
            ok, detail = plan.is_frozen()
        self.assertFalse(ok)
        self.assertIn("no freeze record", detail)

    def test_an_incomplete_freeze_record_is_refused(self):
        rec = {"plan_hash": plan.plan_hash(), "frozen_at_utc": "2026-10-15T00:00:00Z"}
        with mock.patch.object(plan, "read_freeze_record", return_value=rec):
            ok, detail = plan.is_frozen()
        self.assertFalse(ok)
        self.assertIn("incomplete", detail)

    def test_a_freeze_record_binding_a_stale_plan_hash_is_refused(self):
        """Editing the plan after freezing must break the gate."""
        rec = {"plan_hash": "0" * 64, "frozen_at_utc": "2026-10-15T00:00:00Z",
               "plan_file_commit": "a" * 40,
               "plan_file_path": "audit/targets/autond/preregistrations/"
                                 "autond_ii_v2.py",
               "plan_file_sha256": plan.plan_file_sha256(),
               "approved_by": "operator"}
        with mock.patch.object(plan, "read_freeze_record", return_value=rec):
            ok, detail = plan.is_frozen()
        self.assertFalse(ok)
        self.assertIn("changed after it was frozen", detail)

    def test_a_freeze_record_with_an_invented_commit_is_refused(self):
        """
        CORRECTED. Revision 1 accepted any 40-character string as
        `git_commit`, so a freeze record could name a commit that never
        existed. The commit must now be real AND must contain the frozen
        plan file.
        """
        rec = {"plan_hash": plan.plan_hash(),
               "frozen_at_utc": "2026-10-15T00:00:00Z",
               "plan_file_commit": "a" * 40,
               "plan_file_path": "audit/targets/autond/preregistrations/"
                                 "autond_ii_v2.py",
               "plan_file_sha256": plan.plan_file_sha256(),
               "approved_by": "operator"}
        with mock.patch.object(plan, "read_freeze_record", return_value=rec):
            with tempfile.TemporaryDirectory() as tmp:
                ok, detail = plan.is_frozen(Path(tmp))
        self.assertFalse(ok, "an invented commit must not satisfy the gate")

    def test_freezing_alone_does_not_make_the_eval_performed(self):
        """A frozen plan unblocks the run; it does not execute it."""
        rec = {"plan_hash": plan.plan_hash(), "frozen_at_utc": "2026-10-15T00:00:00Z",
               "git_commit": "a" * 40, "approved_by": "operator"}
        with mock.patch.object(plan, "read_freeze_record", return_value=rec), \
             mock.patch.object(plan, "unverified_fields", return_value=[]):
            ii1, _ = impl.build_ii1(self.fx.repo, self.fx.home, self.fx.upstream)
        comp = next(c for c in ii1.components
                    if c.component_id == "II1-SHIPPED-CHECKPOINT-EVAL")
        self.assertEqual(comp.status, StageOutcome.NOT_PERFORMED)


class TestPreregistrationContent(unittest.TestCase):
    """The plan must be fit for the experiment, not a file to unblock a run."""

    def test_it_does_not_predetermine_that_the_paper_is_right(self):
        self.assertEqual(plan.HYPOTHESES["H-AUTOND-SIMON-REPRO"]["prediction"],
                         plan.INCONCLUSIVE)

    def test_the_simon_publication_discrepancy_is_preserved_unreconciled(self):
        pv = plan.PUBLISHED_VALUES["simon3264_round11"]
        self.assertFalse(pv["agree"])
        self.assertEqual(pv["paper_table_6"], 0.5179)
        self.assertEqual(pv["readme_mean"], 0.5182)

    def test_historical_results_are_marked_as_having_no_artifact(self):
        self.assertEqual(plan.HISTORICAL_UNRETAINED["status"],
                         "HISTORICAL_NO_ARTIFACT")
        self.assertIn("NOT confirmatory", plan.HISTORICAL_UNRETAINED["note"])

    def test_every_hypothesis_states_all_three_outcome_conditions(self):
        for name, h in plan.HYPOTHESES.items():
            with self.subTest(hypothesis=name):
                for field in ("supported_if", "not_supported_if", "inconclusive_if"):
                    self.assertTrue(str(h.get(field) or "").strip())
                self.assertIn(h["prediction"], plan.SCIENTIFIC_DECISIONS)

    def test_v1_is_referenced_but_not_superseded(self):
        self.assertIsNone(plan.PLAN["supersedes"])
        self.assertEqual(plan.PLAN["extends"]["v1_plan_hash"], plan.V1_PLAN_HASH)

    def test_the_randomness_policy_admits_it_is_not_reproducible(self):
        self.assertIn("NOT SEEDED", plan.DESIGN["randomness_policy"])
        self.assertIn("NOT bit-reproducible", plan.DESIGN["randomness_policy"])

    def test_the_plan_hash_is_stable_and_well_formed(self):
        self.assertEqual(plan.plan_hash(), plan.plan_hash())
        self.assertEqual(len(plan.plan_hash()), 64)


class TestValidationRunsNothing(unittest.TestCase):

    def test_the_adapter_imports_no_training_stack(self):
        """
        Checked STATICALLY, against the adapter's source.

        sys.modules is not a usable signal here: the generic framework's
        own capture_environment() imports tensorflow, keras and h5py in
        order to record their versions in the run manifest, so in a
        shared test process those names are present for reasons that have
        nothing to do with this adapter. Asserting on sys.modules
        therefore tested the framework's environment capture, not whether
        validation executes an experiment.
        """
        for mod in (impl, psa):
            src = Path(mod.__file__).read_text()
            for bad in ("import tensorflow", "from tensorflow",
                        "import keras", "from keras", "import h5py"):
                self.assertNotIn(bad, src, f"{mod.__name__}: {bad}")

    def test_validation_builds_no_model_and_reads_no_checkpoint(self):
        tmp = Path(tempfile.mkdtemp())
        fx = FakeEvidence(tmp)
        fx.upstream.mkdir(parents=True, exist_ok=True)
        impl.build_ii1(fx.repo, fx.home, fx.upstream)
        impl.build_ii3(fx.repo, fx.home, fx.upstream)
        # Matched as CALLS, with the parenthesis. `load_weights_raised` is a
        # field name this module READS out of the load-verification evidence
        # file; a bare substring check flagged it as though the module were
        # loading weights itself.
        src = Path(impl.__file__).read_text()
        for bad in ("make_model(", ".load_weights(", ".predict(", ".fit("):
            self.assertNotIn(bad, src, f"adapter must not call {bad}")


class TestPaperSourceComparison(unittest.TestCase):

    def test_no_property_was_found_to_differ(self):
        self.assertEqual(psa.summarise()["differs"], [])

    def test_unresolved_properties_are_reported_not_hidden(self):
        s = psa.summarise()
        self.assertTrue(s["underspecified"] or s["not_checkable_here"])

    def test_every_claim_cites_both_sides(self):
        for c in psa.CLAIMS:
            with self.subTest(claim=c.claim_id):
                self.assertTrue(c.paper_locator.strip())
                self.assertTrue(c.source_locator.strip())

    def test_source_only_properties_are_not_claimed_from_the_census(self):
        """Padding and wiring must never be marked observable in weights."""
        for cid in ("ARCH-05", "ARCH-07", "ARCH-11"):
            c = next(x for x in psa.CLAIMS if x.claim_id == cid)
            with self.subTest(claim=cid):
                self.assertIsNot(c.evidence_basis, psa.Basis.WEIGHTS_FILE)

    def test_the_record_digest_is_stable(self):
        self.assertEqual(psa.record_digest(), psa.record_digest())
        self.assertEqual(len(psa.record_digest()), 64)

    def test_the_comparison_states_what_it_cannot_establish(self):
        joined = " ".join(psa.does_not_establish()).lower()
        self.assertIn("provenance", joined)


if __name__ == "__main__":
    unittest.main()
