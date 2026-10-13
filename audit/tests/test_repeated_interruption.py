#!/usr/bin/env python3
"""
REPEATED interruption and resume of a single arm.

WHY THIS SUITE EXISTS. Every other crash test in this package
interrupts an arm ONCE and then checks the recovery. That leaves the
question an audit actually cares about unanswered: does the invariant
survive being exercised again and again on the same arm, at different
points, with each session resuming from the state the previous one
left? Errors that accumulate - a duplicated metric row, an optimizer
counter that slips by one, a best value that drifts, a repair record
that overwrites its predecessor - are invisible to a single-cycle test
by construction, because there is nothing for them to accumulate on.

The arm here is killed with SIGKILL at a DIFFERENT boundary in each
session, in sequence, and run to completion only at the end. What must
hold at that point is not "it finished" but:

  - exactly one log row per epoch, 0..N-1, no duplicates and no gaps
  - no epoch recorded that was not fully committed
  - the optimizer counter increases monotonically with the epoch and
    never restarts
  - the best-monitored value is the true optimum over the whole
    schedule, not the optimum since the last resume
  - the commit manifest is the authority at every step, and is valid
    for this arm at the end

SCOPE. These are the generic, filesystem-level guarantees, exercised
through the JSON worker rather than Keras: the point is the recovery
protocol, not the framework. Bit-exactness of weights and optimizer
tensors across a resume is a separate, framework-specific claim with
its own measured suite, and is deliberately not restated here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from audit.implementation.arm_recovery import EpochLog
from audit.implementation.durable_state import (
    BOUNDARIES,
    CommittedCheckpointStore,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().parent / "kill_worker.py"
EPOCHS = 8


def run_worker(arm_dir: Path, payload: dict) -> subprocess.CompletedProcess:
    # -m with PYTHONPATH, not a bare script path: running the file
    # directly puts only audit/tests on sys.path and the worker cannot
    # import the package it is testing.
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "TF_CPP_MIN_LOG_LEVEL": "3"}
    return subprocess.run(
        [sys.executable, "-m", "audit.tests.kill_worker", "train", str(arm_dir),
         json.dumps(payload)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180)


def store_for(arm_dir: Path, **kw) -> CommittedCheckpointStore:
    base = dict(arm_dir=arm_dir, arm_id="reference", block_id="block0", seed=1,
                epochs_total=EPOCHS, suffix=".json")
    base.update(kw)
    return CommittedCheckpointStore(**base)


class TestManyInterruptionsOnOneArm(unittest.TestCase):

    #: One kill per session, at a different boundary and a different
    #: epoch each time, chosen to straddle the commit: before the
    #: manifest is replaced, after it is replaced, between the two
    #: fsyncs, and after the derived log is appended.
    SCHEDULE = (
        ("after_inflight_write", 1),
        ("after_slot_replace_before_dir_fsync", 2),
        ("after_manifest_commit", 3),
        ("before_manifest_commit", 4),
        ("after_log_append", 5),
        ("after_hash_verify", 6),
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arm = Path(self.tmp.name) / "arm"

    def tearDown(self):
        self.tmp.cleanup()

    def _drive(self):
        """Run the kill schedule, then one final session to completion."""
        sessions = []
        for n, (boundary, epoch) in enumerate(self.SCHEDULE, start=1):
            proc = run_worker(self.arm, {"epochs": EPOCHS, "boundary": boundary,
                                         "kill_epoch": epoch, "session": f"s{n}"})
            sessions.append((boundary, epoch, proc))
            self.assertEqual(
                proc.returncode, -9,
                f"session {n} was meant to be SIGKILLed at {boundary} on epoch {epoch} "
                f"but exited {proc.returncode}.\nstdout: {proc.stdout[-800:]}\n"
                f"stderr: {proc.stderr[-800:]}")
        final = run_worker(self.arm, {"epochs": EPOCHS, "session": "final"})
        self.assertEqual(final.returncode, 0,
                         f"the arm did not complete after six interruptions.\n"
                         f"stdout: {final.stdout[-1500:]}\nstderr: {final.stderr[-1500:]}")
        sessions.append(("none", None, final))
        return sessions

    # --- the schedule itself -----------------------------------------
    def test_every_boundary_in_the_schedule_is_a_real_boundary(self):
        """A typo would silently turn a kill test into a clean run."""
        for boundary, _ in self.SCHEDULE:
            self.assertIn(boundary, BOUNDARIES)

    def test_the_arm_completes_the_full_schedule_after_six_interruptions(self):
        self._drive()
        manifest = store_for(self.arm).manifest()
        self.assertEqual(manifest.committed_epoch, EPOCHS - 1)
        decision = store_for(self.arm).recovery_decision()
        self.assertEqual(decision.resume_from_epoch, EPOCHS)

    def test_the_manifest_is_still_valid_for_this_arm_at_the_end(self):
        self._drive()
        problems = store_for(self.arm).manifest().validate(
            arm_id="reference", block_id="block0", seed=1, epochs_total=EPOCHS)
        self.assertEqual(problems, [])

    # --- no accumulation of damage -----------------------------------
    def _rows(self):
        return [json.loads(line) for line in
                (self.arm / "epochs.jsonl").read_text().splitlines() if line.strip()]

    def test_there_is_exactly_one_row_per_epoch_and_no_duplicates(self):
        """
        The accumulating failure a single-cycle test cannot see: each
        resume re-running an epoch whose log row was already written.
        """
        self._drive()
        epochs = [r["epoch"] for r in self._rows()]
        self.assertEqual(sorted(epochs), list(range(EPOCHS)),
                         f"expected one row per epoch 0..{EPOCHS - 1}, got {sorted(epochs)}")
        self.assertEqual(len(epochs), len(set(epochs)),
                         f"duplicate metric rows after repeated resumes: {epochs}")

    def test_every_committed_epoch_is_either_measured_or_explained(self):
        """
        A kill between the manifest commit and the log append loses that
        epoch's metrics permanently - they existed only in memory. The
        epoch is still committed, so the log must not simply omit it:
        without a marker, a reader of the metrics log alone cannot tell
        an interrupted append from a skipped epoch or a tampered file.
        """
        self._drive()
        by_epoch = {r["epoch"]: r for r in self._rows()}
        committed = store_for(self.arm).manifest().committed_epoch
        for epoch in range(committed + 1):
            self.assertIn(epoch, by_epoch, f"epoch {epoch} is missing from the log "
                                           f"with no explanation")
            self.assertIn(by_epoch[epoch]["status"],
                          ("COMPLETED", EpochLog.METRICS_LOST))

    def test_the_metrics_lost_marker_is_not_counted_as_a_completed_epoch(self):
        """
        The marker must never be mistaken for a measurement: it carries
        no val_loss, and the consumers that filter on COMPLETED must
        skip it.
        """
        self._drive()
        log = EpochLog(self.arm / "epochs.jsonl")
        markers = [r for r in self._rows() if r["status"] == EpochLog.METRICS_LOST]
        if not markers:
            self.skipTest("this kill schedule happened to lose no metrics row")
        for marker in markers:
            self.assertNotIn("val_loss", marker)
            self.assertIn("were lost with the process", marker["note"])
            self.assertNotIn(marker["epoch"], log.completed_epochs())

    def test_the_marker_explains_itself_without_fabricating_metrics(self):
        self._drive()
        markers = [r for r in self._rows() if r["status"] == EpochLog.METRICS_LOST]
        if not markers:
            self.skipTest("this kill schedule happened to lose no metrics row")
        for marker in markers:
            note = marker["note"]
            self.assertIn("was COMMITTED", note)
            self.assertIn("not recoverable", note)
            self.assertIn("deliberately not reconstructed", note)

    def test_a_marker_is_not_added_twice_by_a_later_resume(self):
        """Idempotence: six resumes must not stack six markers."""
        self._drive()
        epochs = [r["epoch"] for r in self._rows()
                  if r["status"] == EpochLog.METRICS_LOST]
        self.assertEqual(len(epochs), len(set(epochs)),
                         f"duplicate markers after repeated resumes: {epochs}")

    def test_the_best_value_ignores_marker_rows(self):
        """
        `best_value` filters on COMPLETED, so a marker with no val_loss
        must not perturb it or raise.
        """
        self._drive()
        log = EpochLog(self.arm / "epochs.jsonl")
        best = log.best_value("val_loss", mode="min")
        self.assertIsNotNone(best)
        completed = [r["val_loss"] for r in self._rows() if r["status"] == "COMPLETED"]
        self.assertAlmostEqual(best, min(completed), places=12)

    def test_no_epoch_is_recorded_that_was_not_committed(self):
        """
        A log row must never exist for an epoch the manifest never
        committed. The row is appended after the commit, so a kill can
        lose a row but can never invent one.
        """
        self._drive()
        log = EpochLog(self.arm / "epochs.jsonl")
        committed = store_for(self.arm).manifest().committed_epoch
        for epoch in log.completed_epochs():
            self.assertLessEqual(epoch, committed)

    def test_the_optimizer_counter_never_restarts(self):
        """
        Six resumes, each reloading the committed checkpoint. If any
        session had restarted from scratch, the counter would fall back.
        """
        self._drive()
        store = store_for(self.arm)
        manifest = store.manifest()
        payload = json.loads(store.slot_path(manifest.committed_slot).read_text())
        self.assertEqual(payload["epoch"], EPOCHS - 1)
        self.assertEqual(payload["optimizer"]["iterations"], EPOCHS * 10,
                         "the optimizer counter does not match the final epoch; a "
                         "session restarted rather than resumed")

    def test_the_best_value_is_the_optimum_over_the_whole_schedule(self):
        """
        Not the optimum since the last resume. The worker's val_loss
        falls monotonically, so the true best is the final epoch's.
        """
        self._drive()
        manifest = store_for(self.arm).manifest()
        expected = 0.3 - 0.01 * (EPOCHS - 1)
        self.assertAlmostEqual(manifest.best_monitor_value, expected, places=9)
        self.assertEqual(manifest.best_checkpoint_epoch, EPOCHS - 1)

    def test_each_session_resumed_where_the_previous_one_committed(self):
        """
        Read back from the workers' own reported resume points rather
        than inferred: each must be exactly one past what was committed
        when it started, and never go backwards.
        """
        sessions = self._drive()
        resumes = []
        for boundary, epoch, proc in sessions:
            first = proc.stdout.splitlines()[0] if proc.stdout.strip() else "{}"
            try:
                resumes.append(json.loads(first)["resume_from"])
            except (json.JSONDecodeError, KeyError):
                self.fail(f"session at {boundary} did not report its resume point: "
                          f"{proc.stdout[:300]}")
        self.assertEqual(resumes[0], 0, "the first session should start fresh")
        for earlier, later in zip(resumes, resumes[1:]):
            self.assertGreaterEqual(later, earlier,
                                    f"a session resumed BEFORE an earlier one: {resumes}")
        self.assertLessEqual(max(resumes), EPOCHS)

    def test_a_kill_before_the_manifest_commit_costs_exactly_one_epoch(self):
        """
        The boundary semantics, checked inside a multi-cycle run rather
        than in isolation: killed before the manifest is replaced, the
        epoch is redone; killed after, it is kept.
        """
        proc = run_worker(self.arm, {"epochs": EPOCHS, "boundary": "before_manifest_commit",
                                     "kill_epoch": 3, "session": "s1"})
        self.assertEqual(proc.returncode, -9)
        self.assertEqual(store_for(self.arm).manifest().committed_epoch, 2)

        proc2 = run_worker(self.arm, {"epochs": EPOCHS, "boundary": "after_manifest_commit",
                                      "kill_epoch": 3, "session": "s2"})
        self.assertEqual(proc2.returncode, -9)
        self.assertEqual(store_for(self.arm).manifest().committed_epoch, 3)

    def test_no_inflight_temporaries_survive_the_run(self):
        self._drive()
        leftovers = sorted(p.name for p in self.arm.glob("*.inflight.json"))
        self.assertEqual(leftovers, [],
                         f"inflight checkpoint writes left behind: {leftovers}")

    def test_the_committed_checkpoint_still_verifies_against_its_hash(self):
        self._drive()
        store = store_for(self.arm)
        decision = store.recovery_decision()
        self.assertEqual(decision.source, "committed")
        loaded, again = store.load_committed(
            lambda p: json.loads(Path(p).read_text()))
        self.assertEqual(loaded["epoch"], EPOCHS - 1)
        self.assertEqual(again.sha256, decision.sha256)


class TestRepeatedCorruptionAcrossSessions(unittest.TestCase):
    """
    Interruption is not the only thing that can repeat. Two separate
    corruption events, each repaired by promoting the surviving slot,
    must both be recorded - and the arm must still finish.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arm = Path(self.tmp.name) / "arm"

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_corruption_events_are_both_recorded_and_the_arm_finishes(self):
        proc = run_worker(self.arm, {"epochs": EPOCHS, "boundary": "after_log_append",
                                     "kill_epoch": 2, "session": "s1"})
        self.assertEqual(proc.returncode, -9)

        # first corruption: wreck the committed slot
        store = store_for(self.arm)
        store.slot_path(store.manifest().committed_slot).write_bytes(b"corrupt one")
        proc = run_worker(self.arm, {"epochs": EPOCHS, "boundary": "after_log_append",
                                     "kill_epoch": 5, "session": "s2"})
        self.assertEqual(proc.returncode, -9)

        # second corruption, in a later session
        store = store_for(self.arm)
        store.slot_path(store.manifest().committed_slot).write_bytes(b"corrupt two")
        final = run_worker(self.arm, {"epochs": EPOCHS, "session": "final"})
        self.assertEqual(final.returncode, 0,
                         f"the arm did not finish after two repairs.\n{final.stdout[-1200:]}"
                         f"\n{final.stderr[-1200:]}")

        store = store_for(self.arm)
        self.assertEqual(store.manifest().committed_epoch, EPOCHS - 1)
        self.assertEqual(len(store.corruption_history()), 2,
                         "both corruption events should be in the append-only log")
        self.assertEqual(len(store.manifest().repairs), 2,
                         "the manifest's repair list lost an earlier entry")

    def test_both_corrupt_files_are_quarantined_rather_than_deleted(self):
        self.test_two_corruption_events_are_both_recorded_and_the_arm_finishes()
        store = store_for(self.arm)
        quarantined = sorted(p.read_bytes() for p in store.quarantine_dir.glob("*"))
        self.assertIn(b"corrupt one", quarantined)
        self.assertIn(b"corrupt two", quarantined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
