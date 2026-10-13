#!/usr/bin/env python3
"""
An arm that cannot be recovered must be PRESERVED, not reset.

WHY THIS SUITE EXISTS. The both-slots-corrupt refusal used to end with
"Delete this arm's directory to restart it deliberately." In an audit
pipeline that is the wrong instruction: the arm directory holds the
commit manifest, both checkpoint slots, the append-only epoch log, the
corruption log, the evaluation journal and the recovery record. Those
are the only copies of the evidence of what went wrong, and a
corruption event is exactly when that evidence matters - it may be the
only sign that the filesystem or the hardware is faulty, which is a fact
about the machine rather than about one block.

So the requirement is stronger than "do not crash": the refusal must
leave every byte as it was found, say precisely why recovery was
refused, and hand the decision to a person. The automated path stops
here.

Every test below hashes the entire arm directory before the refusal and
re-hashes it afterwards. The only permitted difference is the ADDITION
of a diagnostic report - no existing file may change, be truncated, or
disappear.

WHAT THIS SUITE DOES NOT CLAIM. It does not test recovery from
corruption; corruption of both slots is unrecoverable by construction.
It tests that the unrecoverable case is handled safely.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audit.implementation.durable_state import (
    OPERATOR_PROCEDURE,
    CheckpointVerificationError,
    CommittedCheckpointStore,
    DurableStateError,
    UnrecoverableArmError,
)

EPOCHS = 6


def store_for(arm_dir: Path, **kw) -> CommittedCheckpointStore:
    base = dict(arm_dir=arm_dir, arm_id="reference", block_id="block0", seed=1,
                epochs_total=EPOCHS, suffix=".json")
    base.update(kw)
    return CommittedCheckpointStore(**base)


def fingerprint(directory: Path) -> dict[str, str]:
    """sha256 of every file under `directory`, keyed by relative path."""
    out = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return out


class _PreservationCase(unittest.TestCase):
    """Harness: build a real arm, damage it, refuse, prove nothing moved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arm = Path(self.tmp.name) / "arm"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *, upto: int = 3, **kw) -> CommittedCheckpointStore:
        store = store_for(self.arm, **kw)
        for epoch in range(upto + 1):
            store.commit_epoch(
                epoch=epoch, best_monitor_value=0.3 - 0.01 * epoch,
                best_checkpoint_epoch=epoch,
                save_fn=lambda p, e=epoch: Path(p).write_text(json.dumps(
                    {"epoch": e, "optimizer": {"iterations": (e + 1) * 10}})))
            with (self.arm / "epochs.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"epoch": epoch, "status": "COMPLETED",
                                     "val_loss": 0.3 - 0.01 * epoch}) + "\n")
        return store

    def refuse(self, store, *, expect_code=None):
        """Run recovery, require a refusal, and return the exception."""
        before = fingerprint(self.arm)
        with self.assertRaises(DurableStateError) as caught:
            store.recovery_decision()
        after = fingerprint(self.arm)

        # Nothing may change or vanish. New files may only be added.
        for rel, digest in before.items():
            self.assertIn(rel, after, f"{rel} was DELETED by a refusal")
            self.assertEqual(digest, after[rel], f"{rel} was MODIFIED by a refusal")
        added = set(after) - set(before)
        self.assertTrue(all(a.startswith("unrecoverable-") for a in added),
                        f"a refusal created something other than a diagnostic: {added}")

        exc = caught.exception
        if expect_code is not None:
            self.assertIsInstance(exc, UnrecoverableArmError)
            self.assertEqual(exc.reason_code, expect_code)
        return exc

    def assertDiagnosticIsUsable(self, exc):
        self.assertIsInstance(exc, UnrecoverableArmError)
        self.assertIn("NOTHING HAS BEEN DELETED OR MODIFIED", str(exc))
        self.assertNotIn("Delete this arm", str(exc))
        self.assertIn("reason_code", str(exc))
        self.assertTrue(exc.diagnosis, "no forensic snapshot was attached")
        self.assertTrue(exc.report_path and Path(exc.report_path).is_file(),
                        "no diagnostic report was written")
        report = json.loads(Path(exc.report_path).read_text())
        self.assertTrue(report["nothing_was_deleted_or_modified"])
        self.assertEqual(report["operator_procedure"], OPERATOR_PROCEDURE)
        return report


class TestBothSlotsCorrupt(_PreservationCase):

    def _wreck_both(self):
        store = self.build()
        manifest = store.manifest()
        store.slot_path(manifest.committed_slot).write_bytes(b"corrupt committed")
        store.slot_path(manifest.previous_slot).write_bytes(b"corrupt previous")
        return store, manifest

    def test_recovery_refuses(self):
        store, _ = self._wreck_both()
        exc = self.refuse(store, expect_code="BOTH_SLOTS_UNUSABLE")
        self.assertIsInstance(exc, CheckpointVerificationError)

    def test_both_corrupt_slots_are_preserved_byte_for_byte(self):
        store, manifest = self._wreck_both()
        self.refuse(store)
        self.assertEqual(store.slot_path(manifest.committed_slot).read_bytes(),
                         b"corrupt committed")
        self.assertEqual(store.slot_path(manifest.previous_slot).read_bytes(),
                         b"corrupt previous")

    def test_the_diagnostic_names_both_slots_and_both_hashes(self):
        store, manifest = self._wreck_both()
        exc = self.refuse(store)
        report = self.assertDiagnosticIsUsable(exc)
        slots = report["diagnosis"]["slots"]
        for slot in (manifest.committed_slot, manifest.previous_slot):
            self.assertIn(slot, slots)
            self.assertFalse(slots[slot]["matches"])
            self.assertNotEqual(slots[slot]["actual_sha256"],
                                slots[slot]["expected_sha256"])
            self.assertEqual(len(slots[slot]["expected_sha256"]), 64)
        self.assertIn("no checkpoint slot matches", report["diagnosis"]["verdict"])

    def test_the_refusal_does_not_restart_the_arm(self):
        """
        The failure mode this guards: silently treating a corrupt arm as
        fresh, which would report a rerun as a resume.
        """
        store, manifest = self._wreck_both()
        exc = self.refuse(store)
        self.assertIn("would misreport a fresh run as a resume", str(exc))
        self.assertEqual(store.manifest().committed_epoch, manifest.committed_epoch)

    def test_the_procedure_offers_the_replacement_block_not_deletion(self):
        store, _ = self._wreck_both()
        exc = self.refuse(store)
        text = str(exc)
        self.assertIn("replacement-block mechanism", text)
        self.assertIn("Copy the whole arm directory somewhere safe", text)
        self.assertIn("aborting the run to investigate", text)

    def test_a_second_refusal_does_not_overwrite_the_first_report(self):
        store, _ = self._wreck_both()
        first = self.refuse(store)
        second = self.refuse(store)
        self.assertNotEqual(first.report_path, second.report_path)
        self.assertTrue(Path(first.report_path).is_file())
        self.assertTrue(Path(second.report_path).is_file())


class TestTornOrInvalidManifest(_PreservationCase):

    def test_a_torn_manifest_refuses_and_is_preserved(self):
        store = self.build()
        torn = b'{"arm_id": "reference", "committed_ep'
        store.manifest_path.write_bytes(torn)
        exc = self.refuse(store, expect_code="MANIFEST_UNREADABLE")
        self.assertEqual(store.manifest_path.read_bytes(), torn,
                         "the torn manifest was repaired or replaced")
        self.assertDiagnosticIsUsable(exc)

    def test_the_diagnostic_records_the_raw_bytes_of_an_unparseable_manifest(self):
        """
        The manifest is hashed as BYTES before any attempt to parse it,
        so the snapshot describes what is actually on disk even when it
        is not a manifest at all.
        """
        store = self.build()
        torn = b'{"arm_id": "reference", "committed_ep'
        store.manifest_path.write_bytes(torn)
        exc = self.refuse(store)
        mf = exc.diagnosis["manifest"]
        self.assertEqual(mf["sha256"], hashlib.sha256(torn).hexdigest())
        self.assertEqual(mf["bytes"], len(torn))
        self.assertFalse(mf["parsed"])
        self.assertIn("committed_ep", mf["preview"])

    def test_an_empty_manifest_refuses_rather_than_reading_as_fresh(self):
        store = self.build()
        store.manifest_path.write_bytes(b"")
        exc = self.refuse(store, expect_code="MANIFEST_UNREADABLE")
        self.assertEqual(store.manifest_path.read_bytes(), b"")
        self.assertIn("empty", str(exc).lower())

    def test_a_foreign_manifest_refuses_and_is_preserved(self):
        store = self.build()
        raw = json.loads(store.manifest_path.read_text())
        raw["arm_id"] = "independent"
        foreign = json.dumps(raw, indent=2, sort_keys=True)
        store.manifest_path.write_text(foreign)
        exc = self.refuse(store, expect_code="MANIFEST_INVALID_FOR_THIS_ARM")
        self.assertEqual(store.manifest_path.read_text(), foreign)
        self.assertIn("does not match this arm", str(exc))
        self.assertDiagnosticIsUsable(exc)

    def test_the_diagnostic_lists_every_validation_problem(self):
        store = self.build()
        raw = json.loads(store.manifest_path.read_text())
        raw.update(arm_id="independent", seed=999)
        store.manifest_path.write_text(json.dumps(raw))
        exc = self.refuse(store)
        problems = exc.diagnosis["manifest"]["validation_problems"]
        self.assertTrue(any("arm" in p for p in problems))
        self.assertTrue(any("seed" in p for p in problems))


class TestArchitectureMismatch(_PreservationCase):

    def test_a_manifest_from_a_different_architecture_refuses(self):
        """
        The digest in the manifest is compared with the digest this
        store was configured with. A mismatch means the committed
        weights belong to a different model, so resuming would train a
        different network and report it as this arm.
        """
        store = self.build(architecture_digest="a" * 64)
        reopened = store_for(self.arm, architecture_digest="b" * 64)
        exc = self.refuse(reopened, expect_code="MANIFEST_INVALID_FOR_THIS_ARM")
        self.assertIn("architecture_digest", str(exc))
        self.assertDiagnosticIsUsable(exc)

    def test_the_diagnostic_shows_both_digests(self):
        store = self.build(architecture_digest="a" * 64)
        reopened = store_for(self.arm, architecture_digest="b" * 64)
        exc = self.refuse(reopened)
        self.assertEqual(exc.diagnosis["expected_architecture_digest"], "b" * 64)
        self.assertIn("a" * 64, " ".join(
            exc.diagnosis["manifest"]["validation_problems"]))

    def test_the_committed_checkpoint_is_not_loaded_on_a_mismatch(self):
        """Fail closed BEFORE touching the weights."""
        store = self.build(architecture_digest="a" * 64)
        reopened = store_for(self.arm, architecture_digest="b" * 64)
        loaded = []
        with self.assertRaises(DurableStateError):
            reopened.load_committed(lambda p: loaded.append(p))
        self.assertEqual(loaded, [], "the checkpoint was loaded despite the mismatch")


class TestManifestSlotHashMismatch(_PreservationCase):

    def test_a_single_mismatched_committed_slot_falls_back_only_to_a_VERIFIED_slot(self):
        """
        This case is RECOVERABLE by design, and the thing to prove is
        that the fallback is not blind: the previous slot is accepted
        only because its hash matches what the manifest recorded for it.
        """
        store = self.build()
        manifest = store.manifest()
        store.slot_path(manifest.committed_slot).write_bytes(b"damaged")

        decision = store.recovery_decision()
        self.assertEqual(decision.source, "previous")
        self.assertTrue(decision.degraded)
        self.assertEqual(decision.epoch, manifest.previous_epoch)
        self.assertEqual(decision.sha256, manifest.previous_sha256)
        # the accepted slot really does hash to the recorded value
        import hashlib as _h
        self.assertEqual(
            _h.sha256(store.slot_path(decision.slot).read_bytes()).hexdigest(),
            manifest.previous_sha256)

    def test_a_silently_edited_previous_slot_is_not_accepted(self):
        """
        The fallback must reject a previous slot whose bytes changed,
        even though it is present and parseable. Content, not presence,
        is the test.
        """
        store = self.build()
        manifest = store.manifest()
        store.slot_path(manifest.committed_slot).write_bytes(b"damaged")
        # a VALID json checkpoint for the right epoch, but NOT the bytes
        # that were committed. `{"iterations": 30}` would be the real
        # epoch-2 content and would hash correctly - the point here is
        # content that is plausible and still wrong.
        store.slot_path(manifest.previous_slot).write_text(
            json.dumps({"epoch": manifest.previous_epoch,
                        "optimizer": {"iterations": 999999}}))
        exc = self.refuse(store, expect_code="BOTH_SLOTS_UNUSABLE")
        self.assertIn("does not match the manifest", str(exc))

    def test_a_missing_slot_file_is_reported_as_absent_not_as_a_hash_failure(self):
        store = self.build()
        manifest = store.manifest()
        committed = store.slot_path(manifest.committed_slot)
        previous = store.slot_path(manifest.previous_slot)
        committed.unlink()
        previous.unlink()
        exc = self.refuse(store, expect_code="BOTH_SLOTS_UNUSABLE")
        self.assertIn("is absent", str(exc))
        for slot in exc.diagnosis["slots"].values():
            if slot["role_per_manifest"]:
                self.assertFalse(slot["exists"])


class TestManifestAndDerivedLogDisagree(_PreservationCase):
    """
    The log is DERIVED and is appended after the manifest is replaced, so
    it can legitimately lag by one epoch. A disagreement is therefore
    reported, not treated as corruption - but it must never be allowed
    to decide where to resume, and it must be visible.
    """

    def test_a_lagging_log_does_not_block_recovery(self):
        store = self.build()
        rows = (self.arm / "epochs.jsonl").read_text().splitlines()
        (self.arm / "epochs.jsonl").write_text("\n".join(rows[:-1]) + "\n")
        decision = store.recovery_decision()
        self.assertEqual(decision.source, "committed")
        self.assertEqual(decision.epoch, store.manifest().committed_epoch)

    def test_the_disagreement_is_reported_in_the_diagnostic(self):
        store = self.build()
        rows = (self.arm / "epochs.jsonl").read_text().splitlines()
        (self.arm / "epochs.jsonl").write_text("\n".join(rows[:-1]) + "\n")
        diagnosis = store.diagnose()
        self.assertFalse(diagnosis["epoch_log"]["agrees_with_manifest"])
        self.assertEqual(diagnosis["epoch_log"]["lags_manifest_by"], 1)
        self.assertTrue(diagnosis["verdict"].startswith("RECOVERABLE"))

    def test_a_log_ahead_of_the_manifest_is_reported(self):
        """
        The log cannot legitimately lead the manifest: the append happens
        after the commit. If it does, the snapshot says so.
        """
        store = self.build()
        with (self.arm / "epochs.jsonl").open("a") as fh:
            fh.write(json.dumps({"epoch": 99, "status": "COMPLETED"}) + "\n")
        diagnosis = store.diagnose()
        self.assertFalse(diagnosis["epoch_log"]["agrees_with_manifest"])
        self.assertLess(diagnosis["epoch_log"]["lags_manifest_by"], 0)

    def test_duplicate_log_rows_are_reported(self):
        store = self.build()
        with (self.arm / "epochs.jsonl").open("a") as fh:
            fh.write(json.dumps({"epoch": 2, "status": "COMPLETED"}) + "\n")
        self.assertEqual(store.diagnose()["epoch_log"]["duplicate_epochs"], [2])

    def test_a_truncated_log_line_is_counted_not_raised_on(self):
        store = self.build()
        with (self.arm / "epochs.jsonl").open("a") as fh:
            fh.write('{"epoch": 4, "status": "COMPL')
        diagnosis = store.diagnose()
        self.assertEqual(diagnosis["epoch_log"]["malformed_lines"], 1)


class TestDiagnoseIsReadOnly(_PreservationCase):

    def test_diagnose_changes_nothing_on_a_healthy_arm(self):
        store = self.build()
        before = fingerprint(self.arm)
        store.diagnose()
        self.assertEqual(before, fingerprint(self.arm))

    def test_diagnose_changes_nothing_on_a_wrecked_arm(self):
        store = self.build()
        manifest = store.manifest()
        store.slot_path(manifest.committed_slot).write_bytes(b"x")
        store.slot_path(manifest.previous_slot).write_bytes(b"y")
        store.manifest_path.write_bytes(b"{not json")
        before = fingerprint(self.arm)
        store.diagnose()
        self.assertEqual(before, fingerprint(self.arm),
                         "diagnose() modified the evidence it was describing")

    def test_diagnose_survives_a_completely_empty_directory(self):
        self.arm.mkdir(parents=True)
        diagnosis = store_for(self.arm).diagnose()
        self.assertEqual(diagnosis["verdict"], "FRESH")
        self.assertFalse(diagnosis["manifest"]["exists"])

    def test_diagnose_survives_a_directory_that_does_not_exist(self):
        diagnosis = store_for(self.arm / "never_created").diagnose()
        self.assertEqual(diagnosis["verdict"], "FRESH")
        self.assertEqual(diagnosis["directory_listing"], [])

    def test_the_snapshot_lists_the_whole_directory_for_forensics(self):
        store = self.build()
        listing = store.diagnose()["directory_listing"]
        self.assertIn("commit.json", listing)
        self.assertIn("epochs.jsonl", listing)
        self.assertTrue(any(n.startswith("ckpt_") for n in listing))

    def test_the_snapshot_declares_itself_read_only(self):
        self.assertTrue(self.build().diagnose()["read_only"])


class TestNoCodePathDeletesAnArmDirectory(unittest.TestCase):
    """
    Structural: the guarantee is only as good as the absence of a
    destructive call anywhere in the recovery layer.
    """

    def test_the_recovery_modules_contain_no_recursive_delete(self):
        import audit as pkg
        root = Path(pkg.__file__).resolve().parent / "implementation"
        for name in ("durable_state.py", "arm_recovery.py", "ii4_comparison_runner.py"):
            source = (root / name).read_text(encoding="utf-8")
            with self.subTest(module=name):
                self.assertNotIn("rmtree", source)
                self.assertNotIn("rmdir", source)

    def test_the_only_unlink_is_of_a_temporary_file_we_wrote(self):
        import audit as pkg
        root = Path(pkg.__file__).resolve().parent / "implementation"
        for name in ("durable_state.py", "arm_recovery.py"):
            source = (root / name).read_text(encoding="utf-8")
            for line_no, line in enumerate(source.splitlines(), 1):
                if ".unlink(" in line and not line.strip().startswith("#"):
                    with self.subTest(module=name, line=line_no):
                        # Permitted: the inflight checkpoint write, and the
                        # temp file an atomic write cleans up on its own
                        # failure path. Both are files this code created
                        # moments earlier and neither is ever evidence.
                        self.assertTrue(
                            "inflight" in line or "tmp" in line,
                            f"{name}:{line_no} deletes something that is neither an "
                            f"inflight checkpoint nor an atomic-write temporary: "
                            f"{line.strip()}")

    def test_the_operator_procedure_never_advises_deletion(self):
        self.assertNotIn("Delete", OPERATOR_PROCEDURE)
        self.assertNotIn("delete", OPERATOR_PROCEDURE.replace(
            "NOTHING HAS BEEN DELETED", ""))
        self.assertIn("preserved", OPERATOR_PROCEDURE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
