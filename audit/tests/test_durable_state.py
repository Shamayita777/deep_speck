#!/usr/bin/env python3
"""
Transactional recovery, tested by killing real processes.

Every test here spawns a child that runs the REAL commit protocol and
SIGKILLs itself at a named persistence boundary, then checks what a
subsequent process recovers. SIGKILL is the point: an in-process
exception runs `except`, `finally` and `atexit`, so a test that only
raises exercises a path a real interruption never takes. This suite is
the reason the recovery claim is allowed to be stated at all.

The boundaries covered, each with its required outcome:

    before_inflight_write                    recover epoch k-1
    after_inflight_write                     recover epoch k-1 (orphan)
    after_slot_replace_before_dir_fsync      recover epoch k-1
    after_slot_replace                       recover epoch k-1
    after_hash_verify          recover epoch k-1
    before_manifest_commit     recover epoch k-1   <- the window that
                                                      used to brick the arm
    after_manifest_commit      recover epoch k     <- committed, log lags
    after_log_append           recover epoch k
    partial trailing log line  recover epoch k, log reconciled not fatal

Plus: committed-slot hash mismatch (degrade to previous, one epoch lost,
recorded), both slots corrupt (REFUSE), torn manifest (REFUSE), and the
sealed-test journal killed after measurement, after consumption and
before the result is recorded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from audit.implementation.durable_state import (
    EVAL_CONSUMED,
    EVAL_MEASURED,
    EVAL_RECORDED,
    CheckpointVerificationError,
    CommitManifest,
    CommittedCheckpointStore,
    DurableStateError,
    EvalJournal,
    sha256_file,
)
# The derived epoch log lives with the recovery layer, not the store.
# Imported at module level because `arm_recovery` imports Keras lazily,
# inside `build_callbacks`, so importing it costs nothing here.
from audit.implementation.arm_recovery import EpochLog

REPO_ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 6
KILL_EPOCH = 3                     # mid-schedule, with 0..2 already committed


def run_worker(mode: str, arm_dir: Path, payload: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "TF_CPP_MIN_LOG_LEVEL": "3"}
    return subprocess.run(
        [sys.executable, "-m", "audit.tests.kill_worker", mode, str(arm_dir),
         json.dumps(payload)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120)


def store_for(arm_dir: Path, **kw) -> CommittedCheckpointStore:
    base = dict(arm_dir=arm_dir, arm_id="reference", block_id="block0", seed=1,
                epochs_total=EPOCHS, suffix=".json")
    base.update(kw)
    return CommittedCheckpointStore(**base)


class TestHardKillAtEveryBoundary(unittest.TestCase):
    """A real SIGKILL at each boundary; the recovery point must be exact."""

    def _kill_and_recover(self, boundary: str, tmp: Path, *, epochs=EPOCHS,
                          kill_epoch=KILL_EPOCH, partial_log=None):
        payload = {"epochs": epochs, "boundary": boundary, "kill_epoch": kill_epoch,
                   "session": "s1"}
        if partial_log is not None:
            payload["partial_log_after_epoch"] = partial_log
        first = run_worker("train", tmp, payload)
        self.assertEqual(first.returncode, -9,
                         f"child did not die by SIGKILL (rc={first.returncode}); "
                         f"stderr={first.stderr[-400:]}")
        decision = store_for(tmp).recovery_decision()
        return first, decision

    # --- crashes BEFORE the manifest commit recover epoch k-1 ---------
    def test_before_inflight_write_recovers_the_previous_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("before_inflight_write", Path(t))
            self.assertEqual(d.epoch, KILL_EPOCH - 1)
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH)
            self.assertFalse(d.degraded)

    def test_after_inflight_write_recovers_the_previous_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("after_inflight_write", Path(t))
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH)
            leftovers = list(Path(t).glob("*.inflight.json"))
            self.assertTrue(leftovers, "expected an orphaned inflight file")
            self.assertFalse(d.degraded,
                             "an orphaned inflight file must not degrade recovery")

    def test_after_slot_replace_recovers_the_previous_epoch(self):
        """The new slot is on disk but uncommitted; the manifest rules."""
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("after_slot_replace", Path(t))
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH)
            self.assertFalse(d.degraded)

    def test_after_slot_replace_before_dir_fsync_recovers_the_previous_epoch(self):
        """
        A distinct boundary from the post-fsync one. The rename has
        happened but its directory entry may still be only in the page
        cache. A SIGKILL cannot reproduce the loss of that entry - the
        page cache survives the process - so this test establishes the
        ORDERING (the manifest still names the old epoch, so recovery is
        correct either way) and NOT durability. See
        durable_state.WHAT_SIGKILL_ESTABLISHES.
        """
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("after_slot_replace_before_dir_fsync", Path(t))
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH)
            self.assertEqual(d.epoch, KILL_EPOCH - 1)
            self.assertFalse(d.degraded)

    def test_the_two_slot_replace_boundaries_are_distinct(self):
        """Both must exist and both must recover the same epoch."""
        from audit.implementation.durable_state import BOUNDARIES
        self.assertIn("after_slot_replace_before_dir_fsync", BOUNDARIES)
        self.assertIn("after_slot_replace", BOUNDARIES)
        with tempfile.TemporaryDirectory() as t:
            _, before = self._kill_and_recover(
                "after_slot_replace_before_dir_fsync", Path(t))
        with tempfile.TemporaryDirectory() as t:
            _, after = self._kill_and_recover("after_slot_replace", Path(t))
        self.assertEqual(before.resume_from_epoch, after.resume_from_epoch)
        self.assertEqual(before.epoch, after.epoch)

    def test_after_hash_verify_recovers_the_previous_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("after_hash_verify", Path(t))
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH)

    def test_before_manifest_commit_recovers_the_previous_epoch(self):
        """
        THE WINDOW THAT USED TO BRICK THE ARM. Under the old
        single-checkpoint scheme this left the record holding epoch k-1's
        hash against epoch k's file, and the hash guard then refused the
        resume permanently.
        """
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("before_manifest_commit", Path(t))
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH,
                             "the arm is not resumable at the epoch it had committed")
            self.assertEqual(d.source, "committed")
            self.assertFalse(d.degraded)

    # --- crashes AT OR AFTER the commit recover epoch k ---------------
    def test_after_manifest_commit_recovers_the_new_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("after_manifest_commit", Path(t))
            self.assertEqual(d.epoch, KILL_EPOCH)
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH + 1)
            # the log legitimately lags the manifest by one epoch here
            log = [json.loads(x) for x in
                   (Path(t) / "epochs.jsonl").read_text().splitlines() if x.strip()]
            self.assertEqual([r["epoch"] for r in log], list(range(KILL_EPOCH)))

    def test_after_log_append_recovers_the_new_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            _, d = self._kill_and_recover("after_log_append", Path(t))
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH + 1)

    def test_a_truncated_trailing_log_line_is_not_fatal(self):
        """The likeliest artifact of a hard kill must be a normal condition."""
        with tempfile.TemporaryDirectory() as t:
            first = run_worker("train", Path(t), {
                "epochs": EPOCHS, "partial_log_after_epoch": KILL_EPOCH, "session": "s1"})
            self.assertEqual(first.returncode, -9)
            raw = (Path(t) / "epochs.jsonl").read_text()
            self.assertFalse(raw.endswith("\n"), "expected a truncated trailing line")
            d = store_for(Path(t)).recovery_decision()
            self.assertEqual(d.epoch, KILL_EPOCH,
                             "the manifest committed this epoch; a torn log must not hide it")
            self.assertEqual(d.resume_from_epoch, KILL_EPOCH + 1)

    # --- the arm actually finishes after being resumed ---------------
    def test_a_killed_arm_resumes_and_completes_the_schedule(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            first, d = self._kill_and_recover("before_manifest_commit", tmp)
            second = run_worker("train", tmp, {"epochs": EPOCHS, "session": "s2"})
            self.assertEqual(second.returncode, 0, second.stderr[-500:])
            resumed = json.loads(second.stdout.strip().splitlines()[0])
            self.assertEqual(resumed["resume_from"], KILL_EPOCH,
                             "the resumed process repeated or skipped work")
            final = store_for(tmp).recovery_decision()
            self.assertEqual(final.epoch, EPOCHS - 1)
            self.assertEqual(final.resume_from_epoch, EPOCHS)
            payload = json.loads(store_for(tmp).slot_path(final.slot).read_text())
            self.assertEqual(payload["epoch"], EPOCHS - 1,
                             "the committed checkpoint is not the final epoch's")
            self.assertEqual(payload["optimizer"]["iterations"], EPOCHS * 10,
                             "optimizer state does not correspond to the committed epoch")

    def test_the_committed_checkpoint_always_corresponds_to_its_epoch(self):
        """
        The review's rule: never claim an epoch was recovered unless its
        weights and optimizer state correspond to that epoch. Checked at
        every boundary, by reading the committed slot's contents.
        """
        for boundary, expected in (("before_manifest_commit", KILL_EPOCH - 1),
                                   ("after_manifest_commit", KILL_EPOCH),
                                   ("after_log_append", KILL_EPOCH)):
            with tempfile.TemporaryDirectory() as t:
                _, d = self._kill_and_recover(boundary, Path(t))
                body = json.loads(store_for(Path(t)).slot_path(d.slot).read_text())
                self.assertEqual(body["epoch"], expected, f"{boundary}: wrong epoch in slot")
                self.assertEqual(body["optimizer"]["iterations"], (expected + 1) * 10,
                                 f"{boundary}: optimizer state does not match the epoch")
                self.assertEqual(d.epoch, expected)

    def test_slots_alternate_so_the_committed_one_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            run_worker("train", tmp, {"epochs": 4, "session": "s"})
            manifest = store_for(tmp).manifest()
            self.assertNotEqual(manifest.committed_slot, manifest.previous_slot)
            self.assertEqual(manifest.committed_epoch, 3)
            self.assertEqual(manifest.previous_epoch, 2)
            # both slots exist and hold DIFFERENT, consecutive epochs
            a = json.loads((tmp / "ckpt_a.json").read_text())["epoch"]
            b = json.loads((tmp / "ckpt_b.json").read_text())["epoch"]
            self.assertEqual({a, b}, {2, 3})

    def test_only_two_slots_are_ever_created(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            run_worker("train", tmp, {"epochs": EPOCHS, "session": "s"})
            slots = sorted(p.name for p in tmp.glob("ckpt_*.json")
                           if "inflight" not in p.name)
            self.assertEqual(slots, ["ckpt_a.json", "ckpt_b.json"])


class TestCorruptionIsFailClosed(unittest.TestCase):

    def _complete(self, tmp: Path, epochs=4):
        r = run_worker("train", tmp, {"epochs": epochs, "session": "s"})
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        return store_for(tmp, epochs_total=epochs)

    def test_a_corrupt_committed_slot_degrades_to_the_previous_and_says_so(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._complete(tmp)
            manifest = store.manifest()
            store.slot_path(manifest.committed_slot).write_text('{"corrupt": true}')
            d = store.recovery_decision()
            self.assertTrue(d.degraded)
            self.assertEqual(d.source, "previous")
            self.assertEqual(d.epoch, manifest.previous_epoch)
            self.assertEqual(d.resume_from_epoch, manifest.previous_epoch + 1)
            self.assertTrue(any("fell back to the previous slot" in n for n in d.notes))

    def test_both_slots_corrupt_refuses_rather_than_restarting(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._complete(tmp)
            for slot in ("a", "b"):
                store.slot_path(slot).write_text('{"corrupt": true}')
            with self.assertRaises(CheckpointVerificationError) as cm:
                store.recovery_decision()
            self.assertIn("BOTH checkpoint slots are unusable", str(cm.exception))
            self.assertIn("misreport a fresh run as a resume", str(cm.exception))

    def test_a_missing_committed_slot_degrades_to_the_previous(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._complete(tmp)
            store.slot_path(store.manifest().committed_slot).unlink()
            d = store.recovery_decision()
            self.assertTrue(d.degraded)
            self.assertEqual(d.source, "previous")

    def test_a_torn_manifest_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._complete(tmp)
            store.manifest_path.write_text('{"committed_epoch": 3, "committed_sl')
            with self.assertRaisesRegex(DurableStateError, "not valid JSON"):
                store.recovery_decision()

    def test_an_empty_manifest_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._complete(tmp)
            store.manifest_path.write_text("")
            with self.assertRaisesRegex(DurableStateError, "is empty"):
                store.recovery_decision()

    def test_a_manifest_for_another_architecture_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            run_worker("train", tmp, {"epochs": 3, "session": "s",
                                      "architecture_digest": "a" * 64})
            store = store_for(tmp, epochs_total=3, architecture_digest="b" * 64)
            with self.assertRaisesRegex(DurableStateError,
                                        "architecture_digest .* does not match the declared"):
                store.recovery_decision()

    def test_a_fresh_arm_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as t:
            d = store_for(Path(t)).recovery_decision()
            self.assertEqual(d.resume_from_epoch, 0)
            self.assertEqual(d.source, "fresh")

    def test_the_manifest_can_never_go_backwards(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._complete(tmp)
            with self.assertRaisesRegex(DurableStateError, "must never go backwards"):
                store.commit_epoch(epoch=1, save_fn=lambda p: Path(p).write_text("{}"))


class TestSealedEvaluationJournal(unittest.TestCase):
    """
    A successful sealed-test measurement must survive a crash, and must
    never be repeated. Each kill point is a real SIGKILL.
    """

    PAYLOAD = {"accuracy": 0.9291, "sealed_test_hash": "s" * 64,
               "terminal_model_hash": "m" * 64, "session": "s1"}

    def _kill(self, boundary: str, tmp: Path):
        r = run_worker("eval", tmp, {**self.PAYLOAD, "eval_boundary": boundary})
        self.assertEqual(r.returncode, -9, f"rc={r.returncode} stderr={r.stderr[-300:]}")
        return r

    def test_a_crash_after_measurement_does_not_lose_the_result(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            first = self._kill("after_journal_measured", tmp)
            self.assertIn('"inference": "performed"', first.stdout)
            journal = EvalJournal(tmp / "eval_journal.json").read()
            self.assertEqual(journal["state"], EVAL_MEASURED)
            self.assertEqual(journal["accuracy"], 0.9291)

            second = run_worker("eval", tmp, {**self.PAYLOAD, "session": "s2"})
            self.assertEqual(second.returncode, 0, second.stderr[-400:])
            self.assertIn("SKIPPED_adopted_from_journal", second.stdout,
                          "inference was repeated on a sealed test set")
            self.assertIn('"consumption": "spent_now"', second.stdout)
            self.assertEqual(json.loads((tmp / "arm_state.json").read_text())["accuracy"],
                             0.9291)

    def test_a_crash_after_consumption_does_not_lose_the_block(self):
        """
        The failure the old code had: the permit was spent, the accuracy
        was nowhere, and the block died non-replaceable despite a
        successful measurement.
        """
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._kill("after_firewall_consumption", tmp)
            self.assertEqual(json.loads((tmp / "firewall.json").read_text()),
                             [["reference", "block0"]])
            second = run_worker("eval", tmp, {**self.PAYLOAD, "session": "s2"})
            self.assertEqual(second.returncode, 0, second.stderr[-400:])
            self.assertIn("SKIPPED_adopted_from_journal", second.stdout)
            self.assertIn("already_spent_not_repeated", second.stdout,
                          "the sealed-test permit was spent a second time")
            self.assertEqual(json.loads((tmp / "arm_state.json").read_text())["accuracy"],
                             0.9291)

    def test_a_crash_after_the_journal_is_consumed_still_records_the_result(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._kill("after_journal_consumed", tmp)
            self.assertEqual(EvalJournal(tmp / "eval_journal.json").read()["state"],
                             EVAL_CONSUMED)
            self.assertFalse((tmp / "arm_state.json").exists())
            second = run_worker("eval", tmp, {**self.PAYLOAD, "session": "s2"})
            self.assertEqual(second.returncode, 0)
            self.assertIn("SKIPPED_adopted_from_journal", second.stdout)
            self.assertIn("already_spent_not_repeated", second.stdout)
            self.assertEqual(EvalJournal(tmp / "eval_journal.json").read()["state"],
                             EVAL_RECORDED)

    def test_the_sealed_set_is_consumed_exactly_once_across_three_crashes(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            for boundary in ("after_journal_measured", "after_firewall_consumption",
                             "after_journal_consumed"):
                r = run_worker("eval", tmp, {**self.PAYLOAD, "eval_boundary": boundary})
                if r.returncode == 0:
                    break
            final = run_worker("eval", tmp, {**self.PAYLOAD, "session": "sN"})
            self.assertEqual(final.returncode, 0)
            consumed = json.loads((tmp / "firewall.json").read_text())
            self.assertEqual(consumed, [["reference", "block0"]],
                             "the permit was spent more than once")

    def test_a_journal_for_a_different_sealed_set_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._kill("after_journal_measured", tmp)
            journal = EvalJournal(tmp / "eval_journal.json")
            with self.assertRaisesRegex(DurableStateError, "measurement of different"):
                journal.adopt(sealed_test_hash="x" * 64, terminal_model_hash="m" * 64)

    def test_a_journal_for_a_different_terminal_model_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._kill("after_journal_measured", tmp)
            journal = EvalJournal(tmp / "eval_journal.json")
            with self.assertRaisesRegex(DurableStateError, "different model"):
                journal.adopt(sealed_test_hash="s" * 64, terminal_model_hash="z" * 64)

    def test_a_torn_journal_refuses_rather_than_re_evaluating(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._kill("after_journal_measured", tmp)
            (tmp / "eval_journal.json").write_text('{"state": "MEAS')
            with self.assertRaisesRegex(DurableStateError, "not valid JSON"):
                EvalJournal(tmp / "eval_journal.json").read()

    def test_the_journal_cannot_move_backwards(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            run_worker("eval", tmp, {**self.PAYLOAD, "session": "s"})
            journal = EvalJournal(tmp / "eval_journal.json")
            self.assertEqual(journal.read()["state"], EVAL_RECORDED)
            with self.assertRaisesRegex(DurableStateError, "backwards"):
                journal.advance(EVAL_MEASURED)


# =====================================================================
# ITEM 4 - manifest schema, identity and epoch continuity
# =====================================================================

class TestManifestValidation(unittest.TestCase):
    """
    A manifest is the sole authority on what was committed, so it must
    be shown to be THIS arm's manifest before it is believed. Without
    these checks a manifest copied from another arm, block or seed would
    be read as this one's history: the store would load weights trained
    on different data and report a continuous run.
    """

    def _arm(self, tmp, upto=3, epochs=EPOCHS, **kw):
        store = store_for(tmp, epochs_total=epochs, **kw)
        for e in range(upto + 1):
            store.commit_epoch(epoch=e, best_monitor_value=0.3 - 0.01 * e,
                               best_checkpoint_epoch=e,
                               save_fn=lambda p, e=e: Path(p).write_text(json.dumps(
                                   {"epoch": e, "optimizer": {"iterations": (e + 1) * 10}})))
        return store

    def _tamper(self, store, **fields):
        raw = json.loads(store.manifest_path.read_text())
        raw.update(fields)
        store.manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True))

    def test_a_valid_manifest_passes(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self.assertEqual(store.manifest().validate(
                arm_id="reference", block_id="block0", seed=1, epochs_total=EPOCHS), [])

    def test_an_unknown_schema_version_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, manifest_version="durable-state-v99")
            with self.assertRaisesRegex(DurableStateError, "manifest_version"):
                store.recovery_decision()

    def test_a_manifest_from_another_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, arm_id="independent")
            with self.assertRaisesRegex(DurableStateError,
                                        "does not match this arm"):
                store.recovery_decision()

    def test_a_manifest_from_another_block_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, block_id="block7")
            with self.assertRaisesRegex(DurableStateError, "block_id"):
                store.recovery_decision()

    def test_a_manifest_with_another_seed_is_refused(self):
        """Different seed means different initialization: not this arm."""
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, seed=999999)
            with self.assertRaisesRegex(DurableStateError, "seed .* does not match"):
                store.recovery_decision()

    def test_a_manifest_under_another_schedule_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, epochs_total=999)
            with self.assertRaisesRegex(DurableStateError, "epochs_total"):
                store.recovery_decision()

    def test_a_skipped_epoch_is_refused(self):
        """
        previous_epoch must be exactly one below committed_epoch. The
        two-slot protocol can never produce a gap, so a gap means the
        manifest was edited or corrupted.
        """
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, previous_epoch=0)        # committed is 3
            with self.assertRaisesRegex(DurableStateError, "epoch continuity broken"):
                store.recovery_decision()

    def test_a_committed_epoch_beyond_the_schedule_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, committed_epoch=EPOCHS + 5, previous_epoch=EPOCHS + 4)
            problems = store.manifest().validate(
                arm_id="reference", block_id="block0", seed=1, epochs_total=EPOCHS)
            self.assertTrue(any("outside the schedule" in p for p in problems))

    def test_both_slots_naming_the_same_letter_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            raw = json.loads(store.manifest_path.read_text())
            self._tamper(store, previous_slot=raw["committed_slot"])
            with self.assertRaisesRegex(DurableStateError, "must differ"):
                store.recovery_decision()

    def test_an_invalid_slot_letter_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, committed_slot="z")
            with self.assertRaisesRegex(DurableStateError, "committed_slot"):
                store.recovery_decision()

    def test_a_non_digest_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, committed_sha256="short")
            with self.assertRaisesRegex(DurableStateError, "not a sha256 digest"):
                store.recovery_decision()

    def test_a_previous_epoch_recorded_before_epoch_zero_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t), upto=0)
            self._tamper(store, previous_epoch=-1, previous_slot="b",
                         previous_sha256="f" * 64)
            problems = store.manifest().validate(
                arm_id="reference", block_id="block0", seed=1, epochs_total=EPOCHS)
            self.assertTrue(any("nothing can precede the first epoch" in p
                                for p in problems))

    def test_a_dangling_previous_slot_without_an_epoch_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = self._arm(Path(t))
            self._tamper(store, previous_epoch=None)
            with self.assertRaisesRegex(DurableStateError, "previous_epoch is absent"):
                store.recovery_decision()

    def test_a_foreign_manifest_dropped_into_an_arm_directory_is_refused(self):
        """The realistic accident: a manifest copied between arm folders."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            a = self._arm(tmp / "reference")
            b_store = CommittedCheckpointStore(
                arm_dir=tmp / "independent", arm_id="independent", block_id="block0",
                seed=2, epochs_total=EPOCHS, suffix=".json")
            for e in range(2):
                b_store.commit_epoch(epoch=e, save_fn=lambda p, e=e: Path(p).write_text(
                    json.dumps({"epoch": e, "optimizer": {"iterations": (e + 1) * 10}})))
            # copy reference's manifest over independent's
            b_store.manifest_path.write_text(a.manifest_path.read_text())
            with self.assertRaises(DurableStateError) as cm:
                b_store.recovery_decision()
            msg = str(cm.exception)
            self.assertIn("does not match this arm", msg)
            self.assertIn("seed", msg)


# =====================================================================
# ITEM 5 - best-checkpoint metadata from committed state
# =====================================================================

class TestBestMetadataComesFromCommittedState(unittest.TestCase):

    def test_the_manifest_and_the_log_diverge_after_a_kill(self):
        """
        Establishes that the two sources genuinely differ, so reading the
        best value from the log is not harmless. The worker commits a
        monotonically improving val_loss and is killed at
        after_manifest_commit, leaving the manifest one epoch ahead.
        """
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            first = run_worker("train", tmp, {
                "epochs": EPOCHS, "boundary": "after_manifest_commit",
                "kill_epoch": KILL_EPOCH, "session": "s1"})
            self.assertEqual(first.returncode, -9)

            store = store_for(tmp)
            manifest = store.manifest()
            log = EpochLog(tmp / "epochs.jsonl")

            self.assertEqual(manifest.committed_epoch, KILL_EPOCH)
            self.assertEqual(log.completed_epochs(), list(range(KILL_EPOCH)))

            committed_best = manifest.best_monitor_value
            log_best = log.best_value("val_loss", mode="min")
            self.assertIsNotNone(committed_best)
            self.assertIsNotNone(log_best)
            self.assertLess(committed_best, log_best,
                            "the committed best should already include the epoch the log "
                            "has not recorded; if these are equal the test proves nothing")

    def test_the_callback_restores_the_committed_best_not_the_logged_one(self):
        """
        A stale best would let the first post-resume epoch overwrite a
        better checkpoint - silently, because the file is still called
        "best".
        """
        try:
            import keras                                          # noqa: F401
        except Exception:                                         # noqa: BLE001
            self.fail("Keras is required: this is a required-suite gate, not an optional one")
        from audit.implementation.arm_recovery import (
            ArmRecoveryRecord, build_callbacks,
        )
        import keras as K

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            first = run_worker("train", tmp, {
                "epochs": EPOCHS, "boundary": "after_manifest_commit",
                "kill_epoch": KILL_EPOCH, "session": "s1"})
            self.assertEqual(first.returncode, -9)

            store = store_for(tmp)
            log = EpochLog(tmp / "epochs.jsonl")
            record = ArmRecoveryRecord(arm_id="reference", block_id="block0", seed=1,
                                       epochs_total=EPOCHS)
            callbacks = build_callbacks(
                store=store, log=log, record=record,
                record_path=tmp / "recovery.json", best_path=tmp / "best.keras",
                session_id="s2")
            best_cb = next(c for c in callbacks
                           if isinstance(c, K.callbacks.ModelCheckpoint))
            self.assertAlmostEqual(best_cb.best, store.manifest().best_monitor_value,
                                   places=12)
            self.assertNotAlmostEqual(best_cb.best,
                                      log.best_value("val_loss", mode="min"), places=12)

    def test_the_committed_best_survives_later_commits(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = store_for(tmp)
            store.commit_epoch(epoch=0, best_monitor_value=0.11, best_checkpoint_epoch=0,
                               save_fn=lambda p: Path(p).write_text('{"epoch":0}'))
            store.commit_epoch(epoch=1, save_fn=lambda p: Path(p).write_text('{"epoch":1}'))
            self.assertAlmostEqual(store.manifest().best_monitor_value, 0.11)
            self.assertEqual(store.manifest().best_checkpoint_epoch, 0)



# =====================================================================
# recovery FALLBACK CONTINUATION - the arm must finish, not just read
# =====================================================================

class TestDegradedRecoveryContinues(unittest.TestCase):
    """
    A fallback to the previous slot used to be a DEAD END: the manifest
    still named the corrupt slot as committed, so the next commit was
    refused ("the manifest must never go backwards") and the slot it
    would have targeted was the only surviving good one. The arm could
    be read and never continued.

    `prepare_for_resume` promotes the survivor atomically. These tests
    require the arm to run to the end of its schedule afterwards, with
    the loss of one epoch recorded in two places.
    """

    def _commit_through(self, tmp: Path, upto: int, epochs=EPOCHS):
        store = store_for(tmp, epochs_total=epochs)
        for e in range(upto + 1):
            store.commit_epoch(
                epoch=e, best_monitor_value=0.3 - 0.01 * e, best_checkpoint_epoch=e,
                save_fn=lambda p, e=e: Path(p).write_text(json.dumps(
                    {"epoch": e, "weights": [e] * 4,
                     "optimizer": {"iterations": (e + 1) * 10}})))
        return store

    def test_promotion_makes_the_arm_committable_again(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            before = store.manifest()
            store.slot_path(before.committed_slot).write_text('{"corrupt": true}')

            decision = store.prepare_for_resume()
            self.assertTrue(decision.degraded)
            self.assertEqual(decision.epoch, before.previous_epoch)
            self.assertEqual(decision.resume_from_epoch, before.previous_epoch + 1)

            after = store.manifest()
            self.assertEqual(after.committed_epoch, before.previous_epoch,
                             "the survivor was not promoted in the manifest")
            self.assertEqual(after.committed_slot, before.previous_slot)
            self.assertIsNone(after.previous_epoch,
                              "the promoted manifest must not claim a previous slot")
            self.assertEqual(len(after.repairs), 1)

            # the commit that used to be refused
            store.commit_epoch(epoch=decision.resume_from_epoch,
                               save_fn=lambda p: Path(p).write_text(json.dumps(
                                   {"epoch": 3, "optimizer": {"iterations": 40}})))
            self.assertEqual(store.manifest().committed_epoch, 3)

    def test_the_next_commit_does_not_overwrite_the_only_good_slot(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            before = store.manifest()
            good_slot = before.previous_slot
            store.slot_path(before.committed_slot).write_text('{"corrupt": true}')
            decision = store.prepare_for_resume()
            store.commit_epoch(epoch=decision.resume_from_epoch,
                               save_fn=lambda p: Path(p).write_text(json.dumps(
                                   {"epoch": 3, "optimizer": {"iterations": 40}})))
            m = store.manifest()
            self.assertNotEqual(m.committed_slot, good_slot,
                                "the new epoch overwrote the promoted slot")
            self.assertEqual(m.previous_slot, good_slot)
            promoted = json.loads(store.slot_path(good_slot).read_text())
            self.assertEqual(promoted["epoch"], decision.epoch,
                             "the promoted checkpoint was damaged by the next commit")

    def test_the_corrupt_file_is_quarantined_not_deleted(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            store.slot_path(store.manifest().committed_slot).write_text('{"corrupt": true}')
            store.prepare_for_resume()
            quarantined = list(store.quarantine_dir.glob("*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(json.loads(quarantined[0].read_text()), {"corrupt": True},
                             "the corrupt bytes were not preserved for forensics")

    def test_the_loss_is_recorded_in_the_manifest_and_an_append_only_log(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            lost = store.manifest().committed_epoch
            store.slot_path(store.manifest().committed_slot).write_text('{"corrupt": true}')
            store.prepare_for_resume()

            history = store.corruption_history()
            self.assertEqual(len(history), 1)
            entry = history[0]
            self.assertEqual(entry["lost_epoch"], lost)
            self.assertEqual(entry["epochs_lost"], 1)
            self.assertEqual(entry["promoted_epoch"], lost - 1)
            self.assertIn("NOT continuous", entry["consequence"])
            self.assertIsNotNone(entry["failed_slot_expected_sha256"])
            self.assertIn("quarantined_to", entry)
            self.assertEqual(store.manifest().repairs[0]["lost_epoch"], lost)

    def test_the_corruption_log_outlives_later_manifests(self):
        """
        The manifest is replaced every epoch. A repair recorded only
        there would survive only at the pleasure of later writes, so the
        append-only log is the durable evidence.
        """
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            store.slot_path(store.manifest().committed_slot).write_text('{"corrupt": true}')
            d = store.prepare_for_resume()
            for e in range(d.resume_from_epoch, EPOCHS):
                store.commit_epoch(epoch=e, save_fn=lambda p, e=e: Path(p).write_text(
                    json.dumps({"epoch": e, "optimizer": {"iterations": (e + 1) * 10}})))
            self.assertEqual(len(store.corruption_history()), 1,
                             "the corruption record did not survive later commits")

    def test_end_to_end_the_arm_completes_the_remaining_schedule(self):
        """
        The test the review asked for: corrupt the committed slot, resume
        from the previous epoch, commit the next epoch, and finish - then
        verify checkpoint contents, optimizer iterations, the epoch log
        and manifest consistency.
        """
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            log_path = tmp / "epochs.jsonl"

            store = self._commit_through(tmp, 3)
            for e in range(4):
                with log_path.open("a") as fh:
                    fh.write(json.dumps({"epoch": e, "status": "COMPLETED",
                                         "val_loss": 0.3 - 0.01 * e,
                                         "session_id": "s1"}) + "\n")
            corrupt_epoch = store.manifest().committed_epoch
            store.slot_path(store.manifest().committed_slot).write_text('{"corrupt": true}')

            decision = store.prepare_for_resume()
            self.assertEqual(decision.resume_from_epoch, corrupt_epoch)

            for e in range(decision.resume_from_epoch, EPOCHS):
                store.commit_epoch(
                    epoch=e, best_monitor_value=0.3 - 0.01 * e, best_checkpoint_epoch=e,
                    save_fn=lambda p, e=e: Path(p).write_text(json.dumps(
                        {"epoch": e, "weights": [e] * 4,
                         "optimizer": {"iterations": (e + 1) * 10}})))
                with log_path.open("a") as fh:
                    fh.write(json.dumps({"epoch": e, "status": "COMPLETED",
                                         "val_loss": 0.3 - 0.01 * e,
                                         "session_id": "s2"}) + "\n")

            # 1. the schedule is complete
            final = store.recovery_decision()
            self.assertEqual(final.epoch, EPOCHS - 1)
            self.assertEqual(final.resume_from_epoch, EPOCHS)
            self.assertFalse(final.degraded, "the repaired arm is still reporting degraded")

            # 2. checkpoint CONTENTS and optimizer iterations correspond
            body = json.loads(store.slot_path(final.slot).read_text())
            self.assertEqual(body["epoch"], EPOCHS - 1)
            self.assertEqual(body["weights"], [EPOCHS - 1] * 4)
            self.assertEqual(body["optimizer"]["iterations"], EPOCHS * 10)

            # 3. manifest consistency: both slots valid, consecutive, distinct
            m = store.manifest()
            self.assertEqual(m.committed_epoch, EPOCHS - 1)
            self.assertEqual(m.previous_epoch, EPOCHS - 2)
            self.assertNotEqual(m.committed_slot, m.previous_slot)
            self.assertEqual(sha256_file(store.slot_path(m.committed_slot)),
                             m.committed_sha256)
            self.assertEqual(sha256_file(store.slot_path(m.previous_slot)),
                             m.previous_sha256)
            self.assertEqual(len(m.repairs), 1)

            # 4. the epoch log is contiguous and unique, with the retrained
            #    epoch appearing once
            from audit.implementation.arm_recovery import EpochLog
            log = EpochLog(log_path)
            epochs = [r["epoch"] for r in log.read()]
            self.assertEqual(epochs, list(range(EPOCHS)))
            self.assertEqual(len(epochs), len(set(epochs)))
            raw = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
            retrained = [r for r in raw if r["epoch"] == corrupt_epoch]
            self.assertEqual(len(retrained), 2,
                             "the lost epoch should appear twice in the RAW log (once per "
                             "attempt) and once in the reconciled view")
            self.assertTrue(any("duplicate" in n for n in log.reconcile(EPOCHS - 1)),
                            "the retrained epoch is not disclosed as a duplicate record")

            # 5. the loss remains auditable
            self.assertEqual(store.corruption_history()[0]["lost_epoch"], corrupt_epoch)

    def test_end_to_end_across_real_processes(self):
        """The same, but the resume happens in a separate process."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            # the first session must run the SAME schedule the store is
            # later validated against: epochs_total is part of the
            # manifest's identity, and a mismatch is now refused
            first = run_worker("train", tmp, {"epochs": EPOCHS, "session": "s1",
                                              "boundary": "after_manifest_commit",
                                              "kill_epoch": 3})
            self.assertEqual(first.returncode, -9, first.stderr[-400:])
            store = store_for(tmp, epochs_total=EPOCHS)
            store.slot_path(store.manifest().committed_slot).write_text('{"corrupt": true}')

            second = run_worker("train", tmp, {"epochs": EPOCHS, "session": "s2"})
            self.assertEqual(second.returncode, 0, second.stderr[-500:])
            head = json.loads(second.stdout.strip().splitlines()[0])
            self.assertTrue(head["degraded"])
            self.assertEqual(head["source"], "previous_promoted")
            self.assertEqual(head["resume_from"], 3)
            self.assertEqual(head["repairs"], 1)

            final = store_for(tmp, epochs_total=EPOCHS).recovery_decision()
            self.assertEqual(final.resume_from_epoch, EPOCHS)
            body = json.loads(store.slot_path(final.slot).read_text())
            self.assertEqual(body["epoch"], EPOCHS - 1)
            self.assertEqual(body["optimizer"]["iterations"], EPOCHS * 10)

    def test_a_second_corruption_is_also_repaired_and_recorded(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            for _ in range(2):
                store.slot_path(store.manifest().committed_slot).write_text('{"bad": 1}')
                d = store.prepare_for_resume()
                store.commit_epoch(epoch=d.resume_from_epoch,
                                   save_fn=lambda p, e=d.resume_from_epoch:
                                   Path(p).write_text(json.dumps(
                                       {"epoch": e, "optimizer": {"iterations": (e + 1) * 10}})))
            self.assertEqual(len(store.corruption_history()), 2)
            self.assertEqual(len(store.manifest().repairs), 2)
            self.assertEqual(len(list(store.quarantine_dir.glob("*"))), 2)

    def test_both_slots_corrupt_still_refuses_after_the_repair_path_exists(self):
        """Fail-closed must not have been weakened by adding repair."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            for slot in ("a", "b"):
                store.slot_path(slot).write_text('{"corrupt": true}')
            with self.assertRaises(CheckpointVerificationError):
                store.prepare_for_resume()
            self.assertEqual(store.corruption_history(), [],
                             "a refusal must not record a repair it did not perform")

    def test_a_clean_arm_is_untouched_by_prepare_for_resume(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._commit_through(tmp, 3)
            before = store.manifest().to_dict()
            d = store.prepare_for_resume()
            self.assertFalse(d.degraded)
            self.assertEqual(store.manifest().to_dict(), before,
                             "prepare_for_resume modified a healthy manifest")
            self.assertEqual(store.corruption_history(), [])
            self.assertFalse(store.quarantine_dir.exists())


# =====================================================================
# crash windows INSIDE the repair path itself
# =====================================================================

class TestTheRepairPathIsItselfRecoverable(unittest.TestCase):
    """
    `prepare_for_resume` does three things in order: quarantine the
    corrupt slot, append to the corruption log, write the promoted
    manifest. A kill between any two of them leaves the arm in a state
    the NEXT session has to handle - and the state after the first step
    is the awkward one, because the manifest still names a slot that is
    no longer there.
    """

    def _arm_with_three_epochs(self, tmp):
        store = store_for(tmp)
        for e in range(3):
            store.commit_epoch(
                epoch=e, best_monitor_value=0.3 - 0.01 * e, best_checkpoint_epoch=e,
                save_fn=lambda p, e=e: Path(p).write_text(json.dumps(
                    {"epoch": e, "optimizer": {"iterations": (e + 1) * 10}})))
        return store

    def test_a_kill_after_the_quarantine_and_before_the_promotion_recovers(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._arm_with_three_epochs(tmp)
            manifest = store.manifest()
            committed_slot = manifest.committed_slot

            # the corrupt slot has already been moved aside; the manifest
            # has NOT yet been replaced, so it still commits that slot
            store.slot_path(committed_slot).write_bytes(b"torn")
            store.quarantine_dir.mkdir(parents=True, exist_ok=True)
            os.replace(store.slot_path(committed_slot),
                       store.quarantine_dir / f"interrupted_repair{store.suffix}")
            self.assertFalse(store.slot_path(committed_slot).exists())
            self.assertEqual(store.manifest().committed_slot, committed_slot)

            fresh = store_for(tmp)
            decision = fresh.recovery_decision()
            self.assertEqual(decision.source, "previous")
            self.assertEqual(decision.epoch, 1)
            self.assertTrue(decision.degraded)

            repaired = fresh.prepare_for_resume()
            self.assertEqual(repaired.source, "previous_promoted")
            self.assertEqual(repaired.epoch, 1)
            self.assertEqual(fresh.manifest().committed_epoch, 1)
            self.assertIsNone(fresh.manifest().previous_epoch)

    def test_the_arm_finishes_its_schedule_after_an_interrupted_repair(self):
        """Reading the state is not enough: it must be able to continue."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._arm_with_three_epochs(tmp)
            slot = store.manifest().committed_slot
            store.slot_path(slot).write_bytes(b"torn")
            store.quarantine_dir.mkdir(parents=True, exist_ok=True)
            os.replace(store.slot_path(slot),
                       store.quarantine_dir / f"interrupted_repair{store.suffix}")

            fresh = store_for(tmp)
            fresh.prepare_for_resume()
            for e in range(2, EPOCHS):
                fresh.commit_epoch(epoch=e, best_monitor_value=0.2 - 0.01 * e,
                                   best_checkpoint_epoch=e,
                                   save_fn=lambda p, e=e: Path(p).write_text(
                                       json.dumps({"epoch": e})))
            self.assertEqual(fresh.manifest().committed_epoch, EPOCHS - 1)
            self.assertEqual(fresh.recovery_decision().source, "committed")

    def test_repairing_twice_does_not_lose_the_first_repair_record(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._arm_with_three_epochs(tmp)
            slot = store.manifest().committed_slot
            store.slot_path(slot).write_bytes(b"torn")
            store.prepare_for_resume()
            store.commit_epoch(epoch=2, save_fn=lambda p: Path(p).write_text('{"e":2}'))
            store.commit_epoch(epoch=3, save_fn=lambda p: Path(p).write_text('{"e":3}'))
            second = store.manifest().committed_slot
            store.slot_path(second).write_bytes(b"torn again")
            store.prepare_for_resume()
            self.assertEqual(len(store.manifest().repairs), 2)
            self.assertEqual(len(store.corruption_history()), 2)

    def test_a_torn_line_in_the_corruption_log_does_not_hide_the_rest(self):
        """
        The corruption log is appended with the same flush-and-fsync
        discipline as the epoch log, so a kill mid-append can leave a
        partial final line. The history must still be readable, with the
        damaged line reported rather than dropped or raised on.
        """
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            store = self._arm_with_three_epochs(tmp)
            slot = store.manifest().committed_slot
            store.slot_path(slot).write_bytes(b"torn")
            store.prepare_for_resume()
            self.assertEqual(len(store.corruption_history()), 1)

            with store.corruption_log_path.open("a", encoding="utf-8") as fh:
                fh.write('{"failed_slot": "a", "part')          # killed mid-write

            history = store.corruption_history()
            self.assertEqual(len(history), 2)
            self.assertNotIn("unparseable_record", history[0])
            self.assertIn("unparseable_record", history[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
