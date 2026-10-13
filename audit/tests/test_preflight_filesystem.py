#!/usr/bin/env python3
"""
The filesystem backing the output directory must be identified and
classified, and an unverified one must be REFUSED.

WHY THIS SUITE EXISTS. Every durability claim in this experiment rests
on two POSIX guarantees: `os.replace` is atomic within a filesystem, and
`fsync` makes bytes durable. The two-slot checkpoint protocol, the
atomic commit manifest, the sealed-test-set lifecycle and the firewall's
persistence are all built on them, and the SIGKILL tests measure them -
on whichever filesystem those tests happen to run on.

On a FUSE mount they do not hold. Google Drive under Colab is the case
that matters: `os.replace` becomes a copy-then-delete against a remote
API, so a kill during a checkpoint commit can leave the committed slot
deleted and its replacement half-written - the exact state the two-slot
design exists to make impossible. `fsync` can also return success before
anything has left the machine, so the manifest can be lost after it was
reported durable.

Preflight previously measured free space on the output path and never
asked what filesystem it was. A run on a mounted Drive would have passed
every check and then lost the committed state at the first interruption.

Mounts cannot be created in a test container, so the classifier takes an
injected mount table. That is deliberate: it makes every rejected case
testable, including ones that cannot be reproduced here, and it keeps
the policy in one auditable table rather than in a chain of guesses
about the current machine.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from unittest import mock

from audit.implementation.preflight import (
    DURABLE_FILESYSTEMS,
    EPHEMERAL_FILESYSTEMS,
    REJECTED_FILESYSTEMS,
    classify_filesystem,
    durability_probe,
    filesystem_report,
    format_report,
    mount_for,
    preflight,
    read_mounts,
)


def mount(point: str, fstype: str, *, options: str = "rw,relatime",
          device: str = "/dev/sda1") -> dict:
    return {"mount_point": point, "fstype": fstype, "options": options, "device": device}


#: A table resembling a Colab session: local scratch plus a mounted Drive.
COLAB_LIKE = [
    mount("/", "overlay", device="overlay"),
    mount("/content", "overlay", device="overlay"),
    mount("/content/drive", "fuse.drive", device="drive"),
    mount("/tmp", "tmpfs", device="tmpfs"),
]


class TestLocalPathsAreAccepted(unittest.TestCase):

    def test_the_real_filesystem_under_this_test_is_identified(self):
        """
        Not a synthetic case: the actual mount table of THIS machine is
        read and the directory this test writes to is classified. If the
        classifier cannot identify a plain local scratch directory on
        the platform it is running on, it is useless.

        This must hold on Linux (/proc/self/mountinfo) and on macOS and
        the BSDs (the `mount` command), because the suite is run on
        developer machines as well as on the training host.
        """
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(Path(t) / "run")
            self.assertGreater(
                report["mount_table_entries"], 0,
                f"the mount table could not be read on {sys.platform}; the filesystem "
                f"check would refuse every path on this platform")
            self.assertIn("fstype", report)
            self.assertEqual(
                report["verdict"], "ACCEPTED",
                f"a temporary directory on this machine was classified "
                f"{report['verdict']}: {report['problems']}")

    def test_a_local_scratch_path_on_a_journaling_filesystem_is_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(
                Path(t) / "runs" / "ii4",
                mounts=[mount("/", "ext4"), mount(str(Path(t).resolve()), "ext4")])
            self.assertEqual(report["verdict"], "ACCEPTED")
            self.assertEqual(report["problems"], [])

    def test_each_verified_filesystem_is_accepted(self):
        for fstype in sorted(DURABLE_FILESYSTEMS):
            with self.subTest(fstype=fstype):
                verdict, reason = classify_filesystem(fstype)
                self.assertEqual(verdict, "ACCEPTED", f"{fstype}: {reason}")
                self.assertIsNone(reason)

    def test_a_container_overlay_is_accepted_because_that_is_kaggle_and_colab(self):
        """
        /kaggle/working and Colab's /content are overlay mounts whose
        upper layer is ordinary local disk. Rejecting overlay would
        reject every container the run is actually meant for.
        """
        self.assertEqual(classify_filesystem("overlay")[0], "ACCEPTED")

    def test_the_output_directory_need_not_exist_yet(self):
        """The nearest existing ancestor is what gets classified."""
        with tempfile.TemporaryDirectory() as t:
            deep = Path(t) / "a" / "b" / "c" / "run"
            report = filesystem_report(deep, mounts=[mount("/", "ext4")])
            self.assertEqual(report["verdict"], "ACCEPTED")
            self.assertEqual(report["measured_at"], str(Path(t).resolve()))
            self.assertEqual(report["path"], str(deep))


class TestMountedAndRemotePathsAreRejected(unittest.TestCase):

    def _verdict(self, path, mounts):
        return filesystem_report(path, mounts=mounts)

    def test_a_google_drive_mount_under_colab_is_rejected(self):
        report = self._verdict("/content/drive/MyDrive/ii4", COLAB_LIKE)
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertTrue(report["problems"])
        joined = " ".join(report["problems"]).lower()
        self.assertIn("drive", joined)
        self.assertIn("local disk", joined)

    def test_local_scratch_in_the_same_colab_table_is_still_accepted(self):
        """
        The check must reject the Drive mount WITHOUT rejecting the
        session's own working directory, or it would block every run.
        """
        report = self._verdict("/content/ii4_runs", COLAB_LIKE)
        self.assertEqual(report["verdict"], "ACCEPTED")
        self.assertEqual(report["problems"], [])

    def test_every_rejected_filesystem_is_rejected_with_a_reason(self):
        for fstype, expected in REJECTED_FILESYSTEMS.items():
            with self.subTest(fstype=fstype):
                verdict, reason = classify_filesystem(fstype)
                self.assertEqual(verdict, "REJECTED")
                self.assertIn(expected.split(":")[0][:20], reason)

    def test_any_fuse_subtype_is_rejected_even_if_unlisted(self):
        """
        FUSE drivers are open-ended, so the family prefix decides. An
        unknown `fuse.something` must not slip through as unrecognized.
        """
        for fstype in ("fuse.sshfs", "fuse.rclone", "fuse.something-new",
                       "fuse.gcsfuse", "FUSE.Drive"):
            with self.subTest(fstype=fstype):
                verdict, reason = classify_filesystem(fstype)
                self.assertEqual(verdict, "REJECTED", f"{fstype} was not rejected")
                self.assertIn("FUSE", reason.upper())

    def test_network_filesystems_are_rejected(self):
        for fstype in ("nfs", "nfs4", "cifs", "smb3", "9p", "davfs", "ceph", "lustre"):
            with self.subTest(fstype=fstype):
                self.assertEqual(classify_filesystem(fstype)[0], "REJECTED")

    def test_memory_backed_filesystems_are_rejected_for_production_output(self):
        """
        Rename IS atomic on tmpfs, so this is not about atomicity: a host
        restart loses the entire run, which defeats a recovery design
        built for 53-84 GPU-hours of training.
        """
        for fstype in EPHEMERAL_FILESYSTEMS:
            with self.subTest(fstype=fstype):
                verdict, reason = classify_filesystem(fstype)
                self.assertEqual(verdict, "REJECTED")
        # The two that hold run output, as opposed to device nodes, are
        # rejected specifically because a restart loses everything.
        for fstype in ("tmpfs", "ramfs"):
            with self.subTest(fstype=fstype):
                self.assertIn("restart", classify_filesystem(fstype)[1])

    def test_a_drive_path_is_rejected_even_when_the_type_looks_ordinary(self):
        """
        Belt and braces: some Drive mounts report a generic type. The
        path marker catches those.
        """
        report = self._verdict("/content/drive/MyDrive/out",
                               [mount("/", "ext4"),
                                mount("/content/drive", "ext4", device="drivefs")])
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertIn("Google Drive", " ".join(report["problems"]))

    def test_a_read_only_mount_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(
                t, mounts=[mount("/", "ext4"),
                           mount(str(Path(t).resolve()), "ext4", options="ro,relatime")])
            self.assertTrue(any("read-only" in p for p in report["problems"]))


class TestUnverifiedFilesystemsFailClosed(unittest.TestCase):
    """
    The instruction this implements: do not claim crash durability on a
    filesystem that has not been verified. So an unrecognized type is a
    refusal, not a warning.
    """

    def test_an_unrecognized_filesystem_is_unknown_and_blocks(self):
        verdict, reason = classify_filesystem("somethingnobodyhasheardof")
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIn("not claimed", reason)

    def test_an_unknown_verdict_produces_a_blocking_problem(self):
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(
                t, mounts=[mount("/", "somethingnobodyhasheardof")])
            self.assertEqual(report["verdict"], "UNKNOWN")
            self.assertTrue(report["problems"])

    def test_an_empty_mount_table_blocks_rather_than_passes(self):
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(t, mounts=[])
            self.assertEqual(report["verdict"], "UNKNOWN")
            self.assertTrue(any("unidentified" in p for p in report["problems"]))

    def test_an_unreadable_mount_source_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(
                t, mounts_source=str(Path(t) / "no_such_mountinfo"))
            self.assertEqual(report["verdict"], "UNKNOWN")
            self.assertTrue(report["problems"])

    def test_a_blank_filesystem_type_is_unknown(self):
        self.assertEqual(classify_filesystem("")[0], "UNKNOWN")


class TestMountResolution(unittest.TestCase):

    def test_the_longest_matching_mount_point_wins(self):
        chosen = mount_for("/content/drive/MyDrive/x", COLAB_LIKE)
        self.assertEqual(chosen["mount_point"], "/content/drive")

    def test_the_root_mount_is_used_when_nothing_deeper_matches(self):
        chosen = mount_for("/var/lib/whatever", COLAB_LIKE)
        self.assertEqual(chosen["mount_point"], "/")

    def test_a_sibling_prefix_is_not_mistaken_for_a_parent(self):
        """
        `/content/driveway` must not match the `/content/drive` mount:
        string prefixes are not path prefixes.
        """
        chosen = mount_for("/content/driveway/out", COLAB_LIKE)
        self.assertEqual(chosen["mount_point"], "/content")

    def test_the_real_mount_table_parses(self):
        """
        No skip: a skipped check proves nothing, and the launch gate
        treats a skip as a failure. Every supported platform must yield
        a readable mount table by one route or the other.
        """
        mounts = read_mounts()
        self.assertTrue(mounts, f"no mount table could be read on {sys.platform}")
        self.assertTrue(all(m["mount_point"].startswith("/") for m in mounts))
        self.assertTrue(all(m["fstype"] for m in mounts))
        self.assertTrue(any(m["mount_point"] == "/" for m in mounts))

    def test_octal_escapes_in_mount_points_are_decoded(self):
        source = Path(tempfile.mkdtemp()) / "mountinfo"
        source.write_text(
            "25 0 8:1 / /mnt/My\\040Drive rw,relatime - fuse.drive drive rw\n")
        mounts = read_mounts(str(source))
        self.assertEqual(mounts[0]["mount_point"], "/mnt/My Drive")
        self.assertEqual(mounts[0]["fstype"], "fuse.drive")

    def test_malformed_mountinfo_lines_are_skipped_not_raised_on(self):
        source = Path(tempfile.mkdtemp()) / "mountinfo"
        source.write_text("garbage with no separator\n"
                          "25 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
                          "short - ext4\n")
        mounts = read_mounts(str(source))
        self.assertEqual(len(mounts), 1)
        self.assertEqual(mounts[0]["fstype"], "ext4")


class TestPreflightIncludesTheFilesystemVerdict(unittest.TestCase):

    def test_a_rejected_filesystem_makes_the_whole_preflight_not_clear(self):
        """
        End to end: the verdict has to reach `clear_to_launch`, or the
        check is decoration.
        """
        report = preflight(output_path="/content/drive/MyDrive/ii4",
                           mounts=COLAB_LIKE,
                           acknowledged=("idle_gpus",))
        self.assertEqual(report["filesystem"]["verdict"], "REJECTED")
        self.assertFalse(report["clear_to_launch"])
        self.assertTrue(any("Drive" in p for p in report["problems"]))

    def test_the_report_names_the_exact_output_path_it_judged(self):
        report = preflight(output_path="/content/drive/MyDrive/ii4", mounts=COLAB_LIKE)
        self.assertEqual(report["output_path"], "/content/drive/MyDrive/ii4")
        self.assertEqual(report["filesystem"]["path"], "/content/drive/MyDrive/ii4")

    def test_the_formatted_report_shows_the_verdict(self):
        from audit.implementation.preflight import format_report
        text = format_report(preflight(output_path="/content/drive/MyDrive/ii4",
                                       mounts=COLAB_LIKE))
        self.assertIn("filesystem", text)
        self.assertIn("REJECTED", text)


# =====================================================================
# the probe: performing the operations, not just naming the filesystem
# =====================================================================

class TestDurabilityProbe(unittest.TestCase):
    """
    Classifying a mount by its type string says what SHOULD hold. The
    probe performs the three operations the commit protocol depends on -
    same-directory replace, file fsync, directory fsync - and reports
    what DOES hold. Directory fsync is the one that fails or is silently
    ignored outside local filesystems, and it is exactly the operation
    that makes an atomic rename durable.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _fingerprint(self):
        import hashlib
        return {str(p.relative_to(self.dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.dir.rglob("*")) if p.is_file()}

    def test_the_probe_passes_on_a_real_local_directory(self):
        result = durability_probe(self.dir)
        self.assertEqual(result["probe_verdict"], "PASSED",
                         f"problems: {result['problems']}")
        for op in ("file_write_and_fsync", "same_directory_replace",
                   "directory_fsync", "read_back_after_sync"):
            self.assertTrue(result["operations"][op]["supported"], op)

    def test_the_probe_leaves_nothing_behind(self):
        before = sorted(p.name for p in self.dir.iterdir())
        durability_probe(self.dir)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), before,
                         "the probe left a temporary file or directory behind")

    def test_the_probe_does_not_touch_files_that_were_already_there(self):
        """
        It may be run against a directory holding evidence, so it must
        write only inside its own temporary subdirectory.
        """
        (self.dir / "commit.json").write_text('{"committed_epoch": 3}')
        (self.dir / "ckpt_a.keras").write_bytes(b"weights")
        before = self._fingerprint()
        durability_probe(self.dir)
        self.assertEqual(before, self._fingerprint(),
                         "the probe modified pre-existing files")

    def test_the_probe_walks_up_to_an_existing_directory(self):
        target = self.dir / "runs" / "not" / "created" / "yet"
        result = durability_probe(target)
        self.assertEqual(result["probe_verdict"], "PASSED")
        self.assertEqual(result["requested_path"], str(target))
        self.assertEqual(result["probed_at"], str(self.dir.resolve()))

    def test_a_failing_directory_fsync_fails_the_probe(self):
        """
        The failure that matters. Simulated, because a filesystem that
        refuses directory fsync cannot be mounted in this container -
        and an untested failure path is not a failure path.
        """
        real_fsync = os.fsync

        def fsync_refusing_directories(fd):
            if os.path.isdir(f"/proc/self/fd/{fd}"):
                raise OSError(22, "Invalid argument")
            return real_fsync(fd)

        with mock.patch("os.fsync", side_effect=fsync_refusing_directories):
            result = durability_probe(self.dir)
        self.assertEqual(result["probe_verdict"], "FAILED")
        self.assertFalse(result["operations"]["directory_fsync"]["supported"])
        self.assertTrue(any("cannot make a rename durable" in p
                            for p in result["problems"]))
        self.assertTrue(any("did not support" in p for p in result["problems"]))

    def test_a_failed_probe_makes_the_whole_preflight_refuse(self):
        real_fsync = os.fsync

        def fsync_refusing_directories(fd):
            if os.path.isdir(f"/proc/self/fd/{fd}"):
                raise OSError(22, "Invalid argument")
            return real_fsync(fd)

        with mock.patch("os.fsync", side_effect=fsync_refusing_directories):
            report = preflight(output_path=self.dir,
                               mounts=[mount(str(self.dir.resolve()), "ext4")],
                               acknowledged=("idle_gpus",))
        self.assertEqual(report["durability"]["probe_verdict"], "FAILED")
        self.assertFalse(report["clear_to_launch"])

    def test_a_failed_probe_claims_nothing_as_verified(self):
        real_fsync = os.fsync

        def fsync_refusing_directories(fd):
            if os.path.isdir(f"/proc/self/fd/{fd}"):
                raise OSError(22, "Invalid argument")
            return real_fsync(fd)

        with mock.patch("os.fsync", side_effect=fsync_refusing_directories):
            result = durability_probe(self.dir)
        self.assertEqual(result["established"], [])
        self.assertTrue(result["unverified"])

    def test_the_probe_is_not_run_on_a_filesystem_already_refused(self):
        """No point writing to a filesystem we have decided not to use."""
        report = preflight(output_path="/content/drive/MyDrive/ii4", mounts=COLAB_LIKE)
        self.assertEqual(report["durability"]["probe_verdict"], "NOT RUN")
        self.assertIn("already refused", report["durability"]["skipped_because"])

    def test_probing_can_be_disabled_without_claiming_verification(self):
        report = preflight(output_path=self.dir,
                           mounts=[mount(str(self.dir.resolve()), "ext4")],
                           probe=False)
        self.assertEqual(report["durability"]["probe_verdict"], "NOT RUN")
        self.assertEqual(report["verified"], [])
        self.assertTrue(report["unverified"])


class TestThePreflightStatesTheLimitsOfItsEvidence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _report(self):
        return preflight(output_path=self.dir,
                         mounts=[mount(str(self.dir.resolve()), "ext4")],
                         acknowledged=("idle_gpus",))

    def test_power_loss_is_listed_as_unprovable_even_when_everything_passes(self):
        report = self._report()
        self.assertEqual(report["durability"]["probe_verdict"], "PASSED")
        joined = " ".join(report["cannot_be_proven"]).upper()
        self.assertIn("POWER LOSS", joined)
        self.assertIn("KERNEL PANIC", joined)

    def test_it_states_that_a_process_kill_does_not_prove_durability(self):
        joined = " ".join(self._report()["cannot_be_proven"])
        self.assertIn("SIGKILL leaves the page cache intact", joined)

    def test_concurrency_and_enospc_are_listed_as_unprovable(self):
        joined = " ".join(self._report()["cannot_be_proven"]).upper()
        self.assertIn("CONCURRENT", joined)
        self.assertIn("ENOSPC", joined)

    def test_the_formatted_report_separates_verified_from_unprovable(self):
        text = format_report(self._report())
        self.assertIn("VERIFIED ON THIS FILESYSTEM, BY MEASUREMENT", text)
        self.assertIn("CANNOT BE PROVEN BY ANY TEST IN THIS SUITE", text)
        self.assertIn("EXECUTION PERMITTED:", text)
        self.assertLess(text.index("VERIFIED ON THIS FILESYSTEM"),
                        text.index("CANNOT BE PROVEN"))

    def test_the_report_states_whether_execution_is_permitted(self):
        """
        The line must track `clear_to_launch` rather than be asserted to
        a fixed value: on a CPU-only machine the GPU blocker makes the
        honest answer NO whatever the filesystem says, and a test that
        demanded YES would only be passable on a GPU box.
        """
        for report in (self._report(),
                       preflight(output_path="/content/drive/MyDrive/ii4",
                                 mounts=COLAB_LIKE, acknowledged=("idle_gpus",))):
            expected = "YES" if report["clear_to_launch"] else "NO"
            with self.subTest(clear=report["clear_to_launch"]):
                self.assertIn(f"EXECUTION PERMITTED: {expected}",
                              format_report(report))

    def test_a_rejected_filesystem_is_never_reported_as_permitted(self):
        report = preflight(output_path="/content/drive/MyDrive/ii4",
                           mounts=COLAB_LIKE, acknowledged=("idle_gpus",))
        self.assertFalse(report["clear_to_launch"])
        self.assertIn("EXECUTION PERMITTED: NO", format_report(report))

    def test_the_consequence_statement_disclaims_power_loss_in_the_certificate(self):
        from audit.implementation.preflight import WHAT_THE_PROBE_ESTABLISHES
        text = WHAT_THE_PROBE_ESTABLISHES["consequence"]
        self.assertIn("NOT verified against sudden power loss", text)
        self.assertIn("certificate", text)


# =====================================================================
# ITEM 6: device placement, resources, and what preflight must not do
# =====================================================================

class TestDevicePlacementIsObservedNotAssumed(unittest.TestCase):
    """
    The distinction that matters: a visible GPU is not a used GPU. The
    report must record the device a real gradient step actually ran on,
    and acknowledging an idle second GPU must never be mistaken for
    having verified placement.
    """

    def test_the_report_records_the_device_a_real_step_ran_on(self):
        from audit.implementation.preflight import gpu_report
        report = gpu_report()
        self.assertIn("training_device", report)
        self.assertIn("training_is_on_gpu", report)
        if report["training_device"] is not None:
            self.assertIsInstance(report["training_device"], str)
            self.assertTrue(report["training_device"],
                            "the device string is empty; placement was not observed")

    def test_placement_is_derived_from_the_device_string_not_from_visibility(self):
        from audit.implementation.preflight import gpu_report
        report = gpu_report()
        if report["training_device"] is not None:
            self.assertEqual(
                report["training_is_on_gpu"],
                "GPU" in report["training_device"].upper(),
                "training_is_on_gpu disagrees with the observed device string")

    def test_a_cpu_only_machine_is_blocked(self):
        """
        CPU fallback must stay blocked.

        SIMULATED, not observed. An earlier version read the real
        `gpu_report()` and skipped when the machine had a GPU - which
        meant it skipped on every healthy training host, and the launch
        gate counts a skip as a failure. A check that can only run where
        it does not matter is worse than no check: it blocked the gate
        on exactly the machine it was supposed to clear.
        """
        from audit.implementation.preflight import preflight as _preflight

        cpu_only = {
            "tensorflow": "x", "keras": "y", "cuda_built": False,
            "physical_gpus": [], "training_device": "/device:CPU:0",
            "training_is_on_gpu": False,
            "problems": ["no GPU is visible to TensorFlow. The compute estimate "
                         "assumes GPU training."],
            "warnings": [], "requires_acknowledgement": [],
        }
        with mock.patch("audit.implementation.preflight.gpu_report",
                        return_value=cpu_only):
            report = _preflight(output_path=".", mounts=[mount("/", "ext4")],
                                probe=False)
        self.assertFalse(report["clear_to_launch"],
                         "a CPU-only machine was cleared for launch")
        self.assertTrue(any("GPU" in p for p in report["problems"]))

    def test_a_gpu_that_is_visible_but_not_used_is_blocked(self):
        """
        The subtler case: the device exists and training still lands on
        the CPU. Visibility is not placement.
        """
        from audit.implementation.preflight import preflight as _preflight

        visible_but_unused = {
            "tensorflow": "x", "keras": "y", "cuda_built": True,
            "physical_gpus": ["/physical_device:GPU:0"],
            "training_device": "/device:CPU:0", "training_is_on_gpu": False,
            "problems": ["a GPU is visible but the gradient step ran on "
                         "/device:CPU:0."],
            "warnings": [], "requires_acknowledgement": [],
        }
        with mock.patch("audit.implementation.preflight.gpu_report",
                        return_value=visible_but_unused):
            report = _preflight(output_path=".", mounts=[mount("/", "ext4")],
                                probe=False)
        self.assertFalse(report["clear_to_launch"])

    def test_the_idle_gpu_acknowledgement_does_not_clear_a_placement_blocker(self):
        """
        The failure this guards: an operator passing
        --acknowledge-idle-gpus and the gate reading it as "placement
        verified". The acknowledgement list and the blocker list are
        separate, and only the second decides clearance.

        Also simulated, for the same reason as above: on a healthy host
        there are no blockers to acknowledge past, so observing this
        required a broken machine.
        """
        from audit.implementation.preflight import preflight as _preflight

        two_gpus_one_idle_and_on_cpu = {
            "tensorflow": "x", "keras": "y", "cuda_built": True,
            "physical_gpus": ["/physical_device:GPU:0", "/physical_device:GPU:1"],
            "training_device": "/device:CPU:0", "training_is_on_gpu": False,
            "problems": ["a GPU is visible but the gradient step ran on "
                         "/device:CPU:0."],
            "warnings": ["the second GPU is idle"],
            "requires_acknowledgement": ["idle_gpus"],
        }
        with mock.patch("audit.implementation.preflight.gpu_report",
                        return_value=two_gpus_one_idle_and_on_cpu):
            report = _preflight(output_path=".", mounts=[mount("/", "ext4")],
                                acknowledged=("idle_gpus",), probe=False)
        self.assertEqual(report["unacknowledged"], [],
                         "the acknowledgement was not registered")
        self.assertFalse(report["clear_to_launch"],
                         "an acknowledgement cleared a hard placement blocker")

    def test_the_acknowledgement_does_clear_a_warning_when_nothing_else_blocks(self):
        """
        The counterpart: if the flag could never clear anything, the
        test above would pass for the wrong reason.
        """
        from audit.implementation.preflight import preflight as _preflight

        healthy_two_gpus = {
            "tensorflow": "x", "keras": "y", "cuda_built": True,
            "physical_gpus": ["/physical_device:GPU:0", "/physical_device:GPU:1"],
            "training_device": "/device:GPU:0", "training_is_on_gpu": True,
            "problems": [], "warnings": ["the second GPU is idle"],
            "requires_acknowledgement": ["idle_gpus"],
        }
        with mock.patch("audit.implementation.preflight.gpu_report",
                        return_value=healthy_two_gpus), \
             mock.patch("audit.implementation.preflight.memory_report",
                        return_value={"problems": [], "warnings": []}), \
             mock.patch("audit.implementation.preflight.disk_report",
                        return_value={"path": ".", "problems": [], "free_gb": 999.0}):
            without = _preflight(output_path=".", mounts=[mount("/", "ext4")],
                                 probe=False)
            with_ack = _preflight(output_path=".", mounts=[mount("/", "ext4")],
                                  acknowledged=("idle_gpus",), probe=False)
        self.assertFalse(without["clear_to_launch"], "the warning did not block")
        self.assertEqual(without["unacknowledged"], ["idle_gpus"])
        self.assertTrue(with_ack["clear_to_launch"],
                        "the acknowledgement cleared nothing, so the test above is vacuous")

    def test_only_idle_gpus_is_acknowledgeable(self):
        """
        Nothing else may be waved through. If a future edit made a
        placement or memory problem acknowledgeable, this fails.
        """
        from audit.implementation.preflight import gpu_report
        self.assertEqual(set(gpu_report().get("requires_acknowledgement", []))
                         - {"idle_gpus"}, set())

    def test_the_report_states_single_device_sequential_execution(self):
        from audit.implementation.preflight import TRAINING_TOPOLOGY
        self.assertEqual(TRAINING_TOPOLOGY["devices_used_per_fit"], 1)
        self.assertIsNone(TRAINING_TOPOLOGY["distribution_strategy"])
        self.assertIn("SEQUENTIAL", TRAINING_TOPOLOGY["arms"])


class TestTheDiskBudgetIsItemised(unittest.TestCase):

    def _report(self):
        from audit.implementation.run_ii4_impl import run_preflight
        return run_preflight("audit/implementation/runs/ii-4-gohr-impl-v1")

    def test_every_component_the_run_writes_is_named(self):
        budget = self._report()["disk_budget"]
        named = " ".join(item["component"] for item in budget["items"]).lower()
        for component in ("block datasets", "sealed", "rotating checkpoint slots",
                          "terminal", "inflight", "logs", "margin"):
            self.assertIn(component, named, f"the disk budget does not account for "
                                            f"{component!r}")

    def test_the_itemised_subtotal_matches_the_recommended_free_space(self):
        from audit.implementation.run_ii4_impl import _STORAGE
        budget = self._report()["disk_budget"]
        self.assertAlmostEqual(budget["subtotal_gb"],
                               _STORAGE["recommended_free_gb"], places=2)

    def test_the_items_sum_to_the_subtotal(self):
        budget = self._report()["disk_budget"]
        self.assertAlmostEqual(sum(i["gb"] for i in budget["items"]),
                               budget["subtotal_gb"], places=3)

    def test_the_margin_is_a_separate_line_not_hidden_in_the_components(self):
        budget = self._report()["disk_budget"]
        margins = [i for i in budget["items"] if "margin" in i["component"].lower()]
        self.assertEqual(len(margins), 1)
        self.assertGreater(margins[0]["gb"], 0)

    def test_the_headroom_multiple_is_reported(self):
        budget = self._report()["disk_budget"]
        self.assertIn("headroom_multiple", budget)
        self.assertIn("free_gb", budget)


class TestTheComputeEstimateIsNotPresentedAsAGuarantee(unittest.TestCase):

    def _notes(self):
        from audit.implementation.run_ii4_impl import run_preflight
        return run_preflight("audit/implementation/runs/ii-4-gohr-impl-v1")["estimate_notes"]

    def test_gpu_hours_are_distinguished_from_wall_clock(self):
        unit = self._notes()["unit"]
        self.assertIn("GPU-hours", unit)
        self.assertIn("NOT wall-clock", unit)

    def test_both_K_figures_are_given_with_their_assumption(self):
        notes = self._notes()
        keys = " ".join(notes)
        self.assertIn("K=7", keys)
        self.assertIn("K=9", keys)
        self.assertIn("Tesla T4", notes["assumption behind both"])

    def test_the_worst_case_names_the_replacement_blocks(self):
        key = [k for k in self._notes() if "K=9" in k][0]
        self.assertIn("replacement", key.lower())

    def test_the_status_line_disclaims_a_guarantee(self):
        status = self._notes()["status"]
        self.assertIn("AN ESTIMATE", status)
        self.assertIn("not a guarantee", status)

    def test_the_formatted_report_labels_the_estimate_as_such(self):
        from audit.implementation.run_ii4_impl import run_preflight
        text = format_report(run_preflight(
            "audit/implementation/runs/ii-4-gohr-impl-v1"))
        self.assertIn("COMPUTE ESTIMATE (an estimate, not a guarantee)", text)


class TestPreflightDoesNotTrainOrSpendAPermit(unittest.TestCase):
    """
    A preflight that consumed a sealed-evaluation permit, or trained,
    would destroy the thing it is meant to protect.
    """

    def test_the_preflight_module_does_not_import_the_firewall(self):
        """
        Checked on the IMPORT GRAPH and the call names, not on the text:
        the module's docstring legitimately explains why durability
        matters to the firewall's persistence, and a word in prose is
        not a dependency.
        """
        import ast as _ast
        from audit.implementation import preflight as mod

        tree = _ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, _ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertEqual([m for m in imported if "firewall" in m.lower()], [],
                         f"preflight imports a firewall module: {imported}")

        called = {node.func.attr for node in _ast.walk(tree)
                  if isinstance(node, _ast.Call)
                  and isinstance(node.func, _ast.Attribute)}
        for forbidden in ("consume_confirmatory_evaluation", "seal_confirmatory_dataset",
                          "freeze"):
            self.assertNotIn(forbidden, called,
                             f"preflight calls {forbidden!r}")

    def test_the_preflight_branch_of_the_cli_spends_nothing(self):
        import inspect
        from audit.implementation import run_ii4_impl as cli
        source = inspect.getsource(cli.main)
        branch = source[source.index("if args.preflight:"):]
        branch = branch[:branch.index("if args.smoke:")]
        for forbidden in ("consume_confirmatory_evaluation", "ComparisonRunner(",
                          "run(execute=True"):
            self.assertNotIn(forbidden, branch,
                             f"the --preflight branch references {forbidden!r}")

    def test_the_preflight_gradient_step_is_not_training(self):
        """
        `gpu_report` runs one tiny gradient step to observe placement.
        That is deliberate and must stay tiny: it must not touch the
        adapter, the datasets or any checkpoint path.
        """
        import inspect
        from audit.implementation.preflight import gpu_report
        source = inspect.getsource(gpu_report)
        for forbidden in ("adapter", "generate_block_datasets", "CommittedCheckpointStore",
                          "commit_epoch", "fit("):
            self.assertNotIn(forbidden, source,
                             f"the placement probe references {forbidden!r}")

    def test_running_the_preflight_writes_nothing_to_the_output_directory(self):
        """
        The durability probe writes inside its own temporary
        subdirectory and removes it. Nothing else may appear.
        """
        with tempfile.TemporaryDirectory() as t:
            target = Path(t) / "run"
            target.mkdir()
            (target / "pre-existing.json").write_text("{}")
            before = sorted(p.name for p in target.iterdir())
            preflight(output_path=target, mounts=[mount(str(Path(t).resolve()), "ext4")])
            self.assertEqual(sorted(p.name for p in target.iterdir()), before)


class TestTheMacOSFallback(unittest.TestCase):
    """
    macOS and the BSDs have no /proc, so the mount table comes from the
    `mount` command instead. Parsed here from recorded real output, so
    the parser is tested on every platform rather than only where it is
    used.
    """

    SAMPLE = (
        "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n"
        "devfs on /dev (devfs, local, nobrowse)\n"
        "/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse)\n"
        "map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)\n"
        "GoogleDrive:sham@example.com on /Volumes/GoogleDrive (dfsfuse_DFS, "
        "nodev, nosuid, synchronous, mounted by sham)\n"
    )

    def _parsed(self):
        from audit.implementation import preflight as mod
        with mock.patch.object(mod.subprocess, "run") as run:
            run.return_value = type("P", (), {"returncode": 0, "stdout": self.SAMPLE})()
            return mod._mounts_from_mount_command()

    def test_the_mount_output_parses(self):
        mounts = self._parsed()
        self.assertEqual(len(mounts), 5)
        points = {m["mount_point"]: m for m in mounts}
        self.assertEqual(points["/"]["fstype"], "apfs")
        self.assertEqual(points["/System/Volumes/Data"]["fstype"], "apfs")
        self.assertEqual(points["/dev"]["fstype"], "devfs")

    def test_a_mount_point_containing_spaces_parses(self):
        mounts = self._parsed()
        self.assertIn("/System/Volumes/Data/home",
                      {m["mount_point"] for m in mounts})

    def test_the_home_volume_is_accepted_and_the_sealed_root_is_not_used(self):
        """
        A repo under /Users lives on the Data volume, which is rw apfs.
        The sealed system root is read-only and must not be what a path
        under /Users resolves to.
        """
        mounts = self._parsed()
        chosen = mount_for("/Users/sham/Documents/deep_speck", mounts)
        self.assertEqual(chosen["mount_point"], "/")
        self.assertEqual(classify_filesystem("apfs")[0], "ACCEPTED")

    def test_a_read_only_macos_mount_is_detected(self):
        """macOS spells it `read-only`, not `ro`."""
        with tempfile.TemporaryDirectory() as t:
            report = filesystem_report(t, mounts=[
                mount(str(Path(t).resolve()), "apfs",
                      options="sealed,local,read-only,journaled")])
            self.assertTrue(any("read-only" in p for p in report["problems"]),
                            f"a read-only APFS mount was not detected: {report}")

    def test_a_google_drive_volume_on_macos_is_rejected(self):
        mounts = self._parsed()
        report = filesystem_report("/Volumes/GoogleDrive/ii4", mounts=mounts)
        self.assertEqual(report["verdict"], "REJECTED")

    def test_a_failing_mount_command_returns_an_empty_table_not_an_error(self):
        from audit.implementation import preflight as mod
        with mock.patch.object(mod.subprocess, "run", side_effect=OSError("no mount")):
            self.assertEqual(mod._mounts_from_mount_command(), [])

    def test_an_absent_proc_falls_back_to_the_command(self):
        from audit.implementation import preflight as mod
        with mock.patch.object(mod.subprocess, "run") as run:
            run.return_value = type("P", (), {"returncode": 0, "stdout": self.SAMPLE})()
            mounts = mod.read_mounts("/no/such/mountinfo")
        self.assertEqual(len(mounts), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
