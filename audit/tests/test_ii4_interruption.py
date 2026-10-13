#!/usr/bin/env python3
"""
Interruption and resume, through the real runner.

An arm is interrupted after a completed epoch, the run is re-entered
exactly as an operator would re-enter it, and the four properties that
matter are checked:

  1. the epoch log holds exactly ONE record per epoch of the schedule;
  2. completed epochs are NOT retrained;
  3. terminal-model verification succeeds on the resumed arm;
  4. the sealed confirmatory test set is NOT consumed twice.

Plus the property that protects the evidence: a smoke run can never
license production evidence, however completely it finishes.

These run on the deterministic fake adapter, which drives the REAL
EpochLog and ArmRecoveryRecord, so they execute in seconds. The same
scenario against real Keras is exercised by
`run_ii4_impl --smoke --interrupt-after-epoch`, whose output is reported
separately; this file is the regression net.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit.common.outcomes import ExecutionMode
from audit.implementation.arm_recovery import ArmRecoveryRecord, EpochLog
from audit.implementation.ii4_comparison_certificate import (
    ComparisonCertificateError,
    build_stage_records,
)
from audit.implementation.ii4_comparison_runner import ComparisonRunner
from audit.tests.test_ii4_comparison import ARM_A, ARM_B, FakeAdapter, design, plan

EPOCHS = FakeAdapter.EPOCHS
INTERRUPT_AFTER = 1          # a completed epoch, mid-schedule


class TestInterruptionAndResume(unittest.TestCase):

    def _interrupt_then_resume(self, tmp, *, arm=ARM_B, block="block0"):
        """Run -> crash in one arm -> re-enter -> finish. Returns both reports."""
        d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
        p = plan(tmp, d=d)

        crashing = FakeAdapter(arm_ids=p.axis.arm_ids,
                               fail_at_epoch={(block, arm): INTERRUPT_AFTER})
        first_error = None
        try:
            ComparisonRunner(plan=p, adapter=crashing, session_id="s1").run(execute=True)
        except Exception as exc:                                 # noqa: BLE001
            first_error = exc

        resuming = FakeAdapter(arm_ids=p.axis.arm_ids)            # no fault this time
        report = ComparisonRunner(plan=plan(tmp, d=d), adapter=resuming,
                                  session_id="s2").run(execute=True)
        return p, first_error, crashing, resuming, report

    # --- 1. one record per epoch -------------------------------------
    def test_epoch_log_has_exactly_one_record_per_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, _, report = self._interrupt_then_resume(tmp)
            log = EpochLog(Path(p.output_dir) / "blocks" / "block0" / ARM_B / "epochs.jsonl")
            epochs = [r["epoch"] for r in log.read()]
            self.assertEqual(epochs, list(range(EPOCHS)),
                             "the epoch log is not a contiguous 0..N-1 history")
            self.assertEqual(len(epochs), len(set(epochs)), "an epoch was logged twice")
            self.assertEqual(log.reconcile(EPOCHS - 1), [])

    def test_the_log_shows_where_it_was_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, _, _ = self._interrupt_then_resume(tmp)
            log = EpochLog(Path(p.output_dir) / "blocks" / "block0" / ARM_B / "epochs.jsonl")
            records = {r["epoch"]: r for r in log.read()}
            self.assertEqual(records[0]["session_id"], "s1")
            self.assertEqual(records[INTERRUPT_AFTER]["session_id"], "s1")
            self.assertEqual(records[INTERRUPT_AFTER + 1]["session_id"], "s2")
            self.assertTrue(records[INTERRUPT_AFTER + 1]["resumed_at_this_epoch"])
            self.assertEqual(log.summary(expected_total=EPOCHS)["sessions"], ["s1", "s2"])

    # --- 2. completed work is not retrained --------------------------
    def test_completed_epochs_are_not_retrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, resuming, _ = self._interrupt_then_resume(tmp)
            resumes = {(a, b): e for kind, a, b, e in
                       (c for c in resuming.calls if c[0] == "resume_from")}
            self.assertEqual(resumes[(ARM_B, "block0")], INTERRUPT_AFTER + 1,
                             "the interrupted arm did not resume at its next completed epoch")
            # The arm that already finished is never entered at all, so its
            # ABSENCE here is the evidence that nothing was retrained.
            self.assertNotIn((ARM_A, "block0"), resumes,
                             "the arm that had already finished was re-entered for training")
            # and the resumed arm skipped exactly the epochs already done
            self.assertEqual(INTERRUPT_AFTER + 1, resumes[(ARM_B, "block0")])

    def test_the_arm_that_already_finished_is_not_retrained(self):
        """
        Session 1 aborts inside block0, so block1 never ran and MUST be
        trained in session 2. What must not be repeated is block0's other
        arm, which had already completed its full schedule.
        """
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, resuming, report = self._interrupt_then_resume(tmp)
            trained = {(a, b) for _, a, b, _ in
                       (c for c in resuming.calls if c[0] == "train")}
            self.assertNotIn((ARM_A, "block0"), trained,
                             "the arm that had already finished was retrained")
            self.assertIn((ARM_B, "block0"), trained)        # the interrupted one
            self.assertIn((ARM_A, "block1"), trained)        # never ran in session 1
            self.assertEqual(report["n_blocks_valid"], 2)

    # --- 3. terminal verification on the resumed arm -----------------
    def test_terminal_model_verification_succeeds_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, _, report = self._interrupt_then_resume(tmp)
            state = json.loads((Path(p.output_dir) / "state.json").read_text())
            rec = state["blocks"]["block0"]["arms"][ARM_B]
            self.assertEqual(rec["status"], "EVALUATED")
            self.assertTrue(rec["terminal_model"]["reload_weight_identical"])
            self.assertTrue(rec["terminal_model"]["architecture_matches_declared"])
            self.assertEqual(rec["terminal_epoch"], EPOCHS)
            self.assertIsNotNone(rec.get("sealed_test_accuracy"))

    def test_the_resumed_arm_trained_the_full_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, _, report = self._interrupt_then_resume(tmp)
            summary = report["epoch_logs"][f"block0/{ARM_B}"]
            self.assertEqual(summary["completed_epochs"], EPOCHS)
            self.assertTrue(summary["contiguous"])
            self.assertEqual(summary["reconciliation_notes"], [])

    # --- 4. the sealed test set is consumed once ---------------------
    def test_the_sealed_test_set_is_not_consumed_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, _, report = self._interrupt_then_resume(tmp)
            consumed = list(report["firewall"]["consumed_evaluation_keys"])
            self.assertEqual(len(consumed), len(set(consumed)),
                             "a (arm, block) pair consumed the sealed test set twice")
            self.assertEqual(len(consumed), 4)          # 2 blocks x 2 arms
            self.assertEqual(len({*report["firewall"]["sealed_dataset_hashes"].values()}), 1)

    # --- the interruption is RECORDED, not glossed -------------------
    def test_the_interruption_reaches_the_report_and_the_arm_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, _, report = self._interrupt_then_resume(tmp)
            self.assertTrue(report["interrupted_arms"], "the interruption was not reported")
            entry = report["interrupted_arms"][0]
            self.assertEqual(entry["block_id"], "block0")
            self.assertEqual(entry["arm"], ARM_B)
            self.assertEqual(entry["resumed_from_epoch"], INTERRUPT_AFTER + 1)
            reasons = {i["reason"] for i in entry["interruptions"]}
            self.assertIn("InjectedInterruption", reasons)
            self.assertIn("resumed", reasons)
            rec = ArmRecoveryRecord.load(
                Path(p.output_dir) / "blocks" / "block0" / ARM_B / "recovery.json")
            self.assertTrue(rec.was_interrupted)
            self.assertEqual(rec.status, "TRAINED")

    def test_the_report_carries_the_exactness_claim_and_its_conditions(self):
        """
        The report must let a reader check the recovery claim rather than
        take it on trust: what was verified, on which stack, and what is
        still not recovered.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, _, report = self._interrupt_then_resume(tmp)
            claim = report["recovery_exactness"]
            self.assertTrue(any("within-epoch" in x for x in claim["not_recovered"]))
            self.assertTrue(any("reseeded" in c for c in claim["conditions"]))
            self.assertIn("RECORDED", claim["consequence"])

            # The CLAIM states that it is re-measured and names no
            # version: it feeds the preregistration hash, which must not
            # depend on the machine.
            self.assertIn("RE-MEASURED", claim["verified_on"]["basis"])
            self.assertNotIn("keras", claim["verified_on"])
            self.assertNotIn("tensorflow", claim["verified_on"])

            # The STACK it was measured on is PROVENANCE, recorded per
            # run - which is what lets a reader check the claim rather
            # than take it on trust. It belongs here, where it may
            # legitimately differ between machines, and not in the
            # hashed claim, where it may not.
            stack = report["software_stack"]
            for component in ("tensorflow", "keras", "python", "numpy"):
                self.assertTrue(stack.get(component),
                                f"the run does not record the {component} it measured "
                                f"on: {sorted(stack)}")
            self.assertRegex(stack["keras"], r"^\d+\.\d+",
                             "the recorded Keras version is not a version")

            # And every session of a resumed run records its own stack,
            # so a resume on a different machine is visible rather than
            # averaged away.
            self.assertTrue(report["session_environments"])
            for session in report["session_environments"]:
                self.assertTrue(session["software_stack"].get("keras"))

    def test_started_is_on_disk_before_the_arm_finishes(self):
        """A crash during epoch 0 must leave a STARTED record, not silence."""
        with tempfile.TemporaryDirectory() as tmp:
            d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
            p = plan(tmp, d=d)
            ad = FakeAdapter(arm_ids=p.axis.arm_ids, fail_at_epoch={("block0", ARM_A): 0})
            with self.assertRaises(Exception):
                ComparisonRunner(plan=p, adapter=ad, session_id="s1").run(execute=True)
            rec = ArmRecoveryRecord.load(
                Path(p.output_dir) / "blocks" / "block0" / ARM_A / "recovery.json")
            self.assertIsNotNone(rec, "no arm record survived the crash")
            self.assertIn(rec.status, ("STARTED", "TRAINING"))
            self.assertTrue(rec.was_interrupted)
            state = json.loads((Path(p.output_dir) / "state.json").read_text())
            self.assertIn(ARM_A, state["blocks"]["block0"]["arms"])

    # --- smoke never licenses evidence -------------------------------
    def test_a_resumed_smoke_run_still_cannot_license_production_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, _, _, _, report = self._interrupt_then_resume(tmp)
            self.assertTrue(report["non_evidentiary"])
            self.assertNotEqual(report["execution_mode"], "PRODUCTION")
            with self.assertRaisesRegex(ComparisonCertificateError, "may not license"):
                build_stage_records(Path(p.output_dir))

    def test_fault_injection_is_refused_on_a_production_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            from audit.implementation.comparison import ArmSeedManifest
            d = design(n_blocks=7, min_valid_blocks=6, max_replacement_blocks=2)
            p = plan(tmp, d=d, mode=ExecutionMode.PRODUCTION, experiment_id="TEST-PROD",
                     seeds=ArmSeedManifest(base_seed=1, n_blocks=d.max_total_blocks,
                                           arm_ids=(ARM_A, ARM_B)))
            runner = ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B)),
                                      interrupt_after_epoch=1)
            with self.assertRaisesRegex(Exception, "never run against evidence"):
                runner.run(execute=True, allow_production=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
