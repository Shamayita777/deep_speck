#!/usr/bin/env python3
"""
The epoch log as a DERIVED record, and the descriptive arm record.

The authoritative half of recovery - the two-slot commit protocol, the
manifest, and what survives a real SIGKILL at each persistence boundary
- is tested in test_durable_state.py. This file tests the parts that sit
beside it:

  * the epoch log carries every metric the evidence trail needs, flushed
    per epoch;
  * it RECONCILES against the manifest instead of competing with it, so
    a lagging log, a duplicate record or a torn trailing line is noted
    and never fatal. An earlier version let the log refuse a resume the
    checkpoint could support, which abandoned arms with 50 good epochs;
  * the descriptive arm record survives a crash, accumulates
    interruptions, and never becomes an authority on resumability.

Runs without TensorFlow.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit.implementation.arm_recovery import (
    ARM_NOT_STARTED,
    ARM_STARTED,
    RECOVERY_EXACTNESS,
    ArmRecoveryRecord,
    EpochLog,
    RecoveryError,
    resume_point,
)
from audit.implementation.durable_state import (
    CommittedCheckpointStore,
    atomic_write_json,
    atomic_write_text,
)


def _epoch(n, **kw):
    base = {"epoch": n, "status": "COMPLETED", "loss": 0.3 - 0.01 * n,
            "accuracy": 0.5 + 0.01 * n, "val_loss": 0.29 - 0.01 * n,
            "val_accuracy": 0.51 + 0.01 * n, "learning_rate": 0.002,
            "epoch_seconds": 1.5, "checkpoint_seconds": 0.2, "session_id": "s0"}
    base.update(kw)
    return base


def _store(tmp, **kw):
    base = dict(arm_dir=Path(tmp), arm_id="reference", block_id="block0", seed=1,
                epochs_total=200, suffix=".json")
    base.update(kw)
    return CommittedCheckpointStore(**base)


class TestEpochLogContent(unittest.TestCase):

    def test_every_field_the_review_asked_for_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpochLog(Path(tmp) / "epochs.jsonl")
            log.append(_epoch(0))
            rec = log.read()[0]
            for field in ("epoch", "loss", "accuracy", "val_loss", "val_accuracy",
                          "learning_rate", "epoch_seconds", "checkpoint_seconds", "status"):
                self.assertIn(field, rec, f"epoch record is missing {field}")

    def test_each_epoch_reaches_disk_as_it_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epochs.jsonl"
            log = EpochLog(path)
            for n in range(3):
                log.append(_epoch(n))
                self.assertEqual(len(path.read_text().strip().splitlines()), n + 1)

    def test_checkpoint_overhead_is_reported_separately_from_training(self):
        """So the run's cost can be attributed rather than guessed."""
        with tempfile.TemporaryDirectory() as tmp:
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n in range(4):
                log.append(_epoch(n, epoch_seconds=10.0, checkpoint_seconds=0.5))
            s = log.summary(expected_total=4, committed_epoch=3)
            self.assertAlmostEqual(s["total_training_seconds"], 40.0)
            self.assertAlmostEqual(s["total_checkpoint_seconds"], 2.0)

    def test_resume_appends_without_duplicating_or_losing_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n in range(4):
                log.append(_epoch(n, session_id="s1"))
            for n in range(log.highest_contiguous(), 10):
                log.append(_epoch(n, session_id="s2"))
            epochs = [r["epoch"] for r in log.read()]
            self.assertEqual(epochs, list(range(10)))
            self.assertEqual(len(epochs), len(set(epochs)))
            self.assertEqual(log.reconcile(9), [])
            self.assertEqual(log.summary(committed_epoch=9)["sessions"], ["s1", "s2"])

    def test_best_value_is_recoverable_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n, vl in enumerate((0.30, 0.22, 0.26, 0.19, 0.25)):
                log.append(_epoch(n, val_loss=vl))
            self.assertAlmostEqual(log.best_value("val_loss", mode="min"), 0.19)
            self.assertAlmostEqual(log.best_value("val_loss", mode="max"), 0.30)
            self.assertIsNone(log.best_value("nonexistent_metric"))


class TestEpochLogIsDerivedNotAuthoritative(unittest.TestCase):
    """
    Every condition below once blocked a resume. Each must now be a
    reconciliation NOTE while the manifest decides.
    """

    def _committed(self, tmp, upto: int):
        store = _store(tmp)
        for e in range(upto + 1):
            store.commit_epoch(epoch=e,
                               save_fn=lambda p, e=e: Path(p).write_text(json.dumps({"e": e})))
        return store

    def test_a_torn_trailing_line_does_not_block_a_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._committed(tmp, 4)
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n in range(5):
                log.append(_epoch(n))
            with (Path(tmp) / "e.jsonl").open("a") as fh:
                fh.write('{"epoch": 5, "sta')
            resume, decision, notes = resume_point(store, log)
            self.assertEqual(resume, 5)
            self.assertEqual(decision.epoch, 4)
            self.assertTrue(any("malformed" in n for n in notes))
            self.assertTrue(any("does not affect recovery" in n for n in notes))

    def test_a_gap_does_not_block_a_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._committed(tmp, 5)
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n in (0, 1, 2, 4, 5):
                log.append(_epoch(n))
            resume, decision, notes = resume_point(store, log)
            self.assertEqual(resume, 6, "a gap in the derived log blocked the resume")
            self.assertTrue(any("missing epoch" in n for n in notes))

    def test_a_log_lagging_the_manifest_is_explained_not_fatal(self):
        """A crash between the manifest commit and the log append."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._committed(tmp, 5)
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n in range(5):                       # log stops at 4, manifest at 5
                log.append(_epoch(n))
            resume, decision, notes = resume_point(store, log)
            self.assertEqual(resume, 6)
            self.assertTrue(any("lags the manifest" in n for n in notes))
            self.assertTrue(any("The epoch IS committed" in n for n in notes))

    def test_uncommitted_log_records_are_ignored(self):
        """A log row for an epoch the manifest never committed."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._committed(tmp, 3)
            log = EpochLog(Path(tmp) / "e.jsonl")
            for n in range(5):                       # log claims 4, manifest commits 3
                log.append(_epoch(n))
            resume, decision, notes = resume_point(store, log)
            self.assertEqual(resume, 4, "an uncommitted log record advanced the resume point")
            self.assertTrue(any("commits only" in n for n in notes))

    def test_duplicates_are_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EpochLog(Path(tmp) / "e.jsonl")
            log.append(_epoch(0)); log.append(_epoch(1)); log.append(_epoch(1, loss=0.1))
            self.assertEqual([r["epoch"] for r in log.read()], [0, 1])
            self.assertEqual(log.read()[1]["loss"], 0.1)
            self.assertTrue(any("duplicate" in n for n in log.reconcile(1)))

    def test_genuine_corruption_still_refuses(self):
        """Fail-closed is preserved where it belongs: the checkpoints."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._committed(tmp, 3)
            for slot in ("a", "b"):
                store.slot_path(slot).write_text("corrupt")
            with self.assertRaisesRegex(RecoveryError, "BOTH checkpoint slots"):
                resume_point(store, EpochLog(Path(tmp) / "e.jsonl"))


class TestArmRecoveryRecord(unittest.TestCase):

    def _record(self, tmp, **kw):
        base = dict(arm_id="reference", block_id="block0", seed=800_000, epochs_total=200)
        base.update(kw)
        return ArmRecoveryRecord(**base), Path(tmp) / "recovery.json"

    def test_started_is_persisted_before_any_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, path = self._record(tmp)
            self.assertEqual(rec.status, ARM_NOT_STARTED)
            rec.status = ARM_STARTED
            rec.save(path)
            self.assertEqual(ArmRecoveryRecord.load(path).status, ARM_STARTED)

    def test_round_trip_preserves_interruptions_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, path = self._record(tmp, epochs_completed=47)
            rec.note_interruption(reason="SIGKILL", detail="quota expiry", at_epoch=47)
            rec.save(path)
            back = ArmRecoveryRecord.load(path)
            self.assertTrue(back.was_interrupted)
            self.assertEqual(back.interruptions[0]["reason"], "SIGKILL")
            self.assertEqual(back.epochs_completed, 47)

    def test_interruptions_accumulate_and_are_never_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, path = self._record(tmp)
            for i in range(3):
                rec.note_interruption(reason=f"r{i}", detail="d", at_epoch=i * 10)
                rec.save(path)
                rec = ArmRecoveryRecord.load(path)
            self.assertEqual(len(rec.interruptions), 3)

    def test_a_torn_descriptive_record_loses_narrative_not_recoverability(self):
        """
        The record is descriptive. A torn one must NOT refuse, because
        the manifest - not this file - governs what is resumable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recovery.json"
            path.write_text('{"arm_id": "refer')
            self.assertIsNone(ArmRecoveryRecord.load(path))

    def test_a_degraded_recovery_is_flagged_on_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            for e in range(3):
                store.commit_epoch(epoch=e,
                                   save_fn=lambda p, e=e: Path(p).write_text(f'{{"e":{e}}}'))
            store.slot_path(store.manifest().committed_slot).write_text("corrupt")
            rec, path = self._record(tmp)
            resume, decision, notes = resume_point(store, EpochLog(Path(tmp) / "e.jsonl"),
                                                   record=rec)
            self.assertTrue(decision.degraded)
            self.assertTrue(rec.was_degraded)
            rec.save(path)
            self.assertTrue(ArmRecoveryRecord.load(path).was_degraded)


class TestAtomicWrites(unittest.TestCase):

    def test_content_is_replaced_not_truncated_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            atomic_write_json(path, {"a": 1})
            atomic_write_json(path, {"a": 2, "b": 3})
            self.assertEqual(json.loads(path.read_text()), {"a": 2, "b": 3})

    def test_no_temporary_files_are_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            for i in range(5):
                atomic_write_json(path, {"i": i})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["state.json"])

    def test_a_failed_serialization_leaves_the_previous_content_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            atomic_write_json(path, {"good": True})
            with self.assertRaises(TypeError):
                atomic_write_text(path, json.dumps({"bad": {1, 2}}))
            self.assertEqual(json.loads(path.read_text()), {"good": True})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["state.json"])


class TestExactnessClaim(unittest.TestCase):

    def test_every_claim_is_measured_and_version_qualified(self):
        for claim in ("optimizer state", "learning-rate schedule", "completed epoch index",
                      "shuffle permutation", "model weights", "best-monitored-value"):
            self.assertTrue(any(claim in x for x in RECOVERY_EXACTNESS["recovered_exactly"]),
                            f"{claim} is not listed among the exactness claims")
        self.assertTrue(any("within-epoch" in x for x in RECOVERY_EXACTNESS["not_recovered"]))
        self.assertEqual(RECOVERY_EXACTNESS["granularity"], "epoch")
        verified = RECOVERY_EXACTNESS["verified_on"]
        self.assertTrue(verified.get("tensorflow") and verified.get("keras"))
        self.assertTrue(any("reseeded" in c for c in RECOVERY_EXACTNESS["conditions"]))
        self.assertIn("RECORDED", RECOVERY_EXACTNESS["consequence"])

    def test_the_claim_names_the_manifest_as_the_authority(self):
        self.assertIn("single source of truth", RECOVERY_EXACTNESS["authority"])
        self.assertIn("no authority over resumability", RECOVERY_EXACTNESS["authority"])

    def test_the_claim_states_the_fail_closed_rule(self):
        self.assertIn("BOTH slots fail", RECOVERY_EXACTNESS["fail_closed"])
        self.assertIn("REFUSES", RECOVERY_EXACTNESS["fail_closed"])

    def test_the_module_docstring_agrees_with_the_measured_claim(self):
        """
        The review found the docstring still asserting the retracted
        "shuffle is not recovered" claim twenty lines above the corrected
        one. A reader hitting the docstring first got the wrong answer.
        """
        import audit.implementation.arm_recovery as mod
        doc = mod.__doc__ or ""

        # The NOT RECOVERED section must not mention the shuffle.
        not_recovered = doc.split("NOT RECOVERED", 1)[1].split("CONSEQUENCE", 1)[0]
        self.assertNotIn("shuffle", not_recovered,
                         "the docstring still lists the shuffle under NOT RECOVERED")
        self.assertIn("within-epoch position", not_recovered)

        # The RECOVERED section must carry the corrected claim.
        recovered = doc.split("RECOVERED EXACTLY", 1)[1].split("NOT RECOVERED", 1)[0]
        self.assertIn("minibatch shuffle", recovered)
        self.assertIn("ABSOLUTE epoch index", recovered)

        # The old reasoning may appear ONLY as a recorded retraction,
        # never as a live claim: keeping the retraction is the honest
        # form, deleting it would hide that the claim once changed.
        if "exposes no shuffle RNG state" in doc:
            self.assertIn("Measurement showed that", doc,
                          "the superseded reasoning appears without its retraction")
            self.assertIn("corrected UPWARDS", doc)
        self.assertIn("SINGLE source of truth", doc)

    def test_the_adapter_docstring_agrees_with_the_measured_claim(self):
        """
        The adapter's own docstring is what a reader of the case study
        hits first. It carried the retracted shuffle claim and said the
        resume point came from the epoch log, both of which are now
        wrong.
        """
        try:
            from audit.implementation.adapters import gohr_impl_comparison as mod
        except Exception as exc:                                 # noqa: BLE001
            self.fail(f"the Gohr adapter is not importable: {type(exc).__name__}: {exc}")
        raw = mod.GohrImplementationComparisonAdapter.train_arm.__doc__ or ""
        doc = " ".join(raw.split())          # whitespace-normalised
        self.assertNotIn("cannot be restored", doc,
                         "the adapter still carries the retracted shuffle claim")
        self.assertIn("COMMIT MANIFEST decides where to resume", doc)
        self.assertIn("not the epoch log", doc,
                      "the adapter still implies the epoch log decides the resume point")
        self.assertIn("corrected upwards", doc)
        self.assertIn("WITHIN-EPOCH position is not recovered", doc)
        self.assertIn("PROMOTED", doc, "the repair path is not described")
        self.assertIn("2.20.0", doc, "the measured stack is not named")

if __name__ == "__main__":
    unittest.main(verbosity=2)
