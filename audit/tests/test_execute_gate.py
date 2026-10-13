#!/usr/bin/env python3
"""
`--execute` must enforce readiness itself, and must not start training
when any gate fails.

WHY THIS SUITE EXISTS. `--execute` used to require exactly two things: a
frozen record on disk, and `--i-have-reviewed-the-preregistration` on
the command line. Both are statements about the DESIGN - that it was
settled and read. Neither says anything about whether the code still
passes its tests, or whether this machine can finish a 53-84 GPU-hour
run.

The test gate and the resource preflight existed, but only in
`--dry-run`. A gate that lives in a different mode from the one that
launches the run is not a gate: `--execute` could be invoked directly,
with failing or skipped required tests, on a Google Drive mount, with no
GPU, or with too little disk to hold the blocks - and the first sign of
it would be a dead run hours later, or worse, a completed run whose
recovery guarantees were never measured on the stack that produced it.

EVERY TEST HERE ASSERTS THAT NO TRAINING STARTED. `ComparisonRunner` is
replaced by a spy that records construction and raises if `run()` is
called, so a gate that merely prints a warning and proceeds fails the
suite. The gates are driven by patching the module's own functions
rather than by arranging real failures, because arranging a real GPU
absence, a real Drive mount and a real failing test suite in one
container is not possible - and the thing under test is the launcher's
decision, not the checks themselves, which have their own suites.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from audit.implementation import run_ii4_impl as cli

EXECUTE = ["--execute", "--i-have-reviewed-the-preregistration"]


def clear_suites(run: int = 232) -> tuple[bool, dict]:
    return True, {"suites": list(cli.REQUIRED_SUITES), "run": run, "failures": 0,
                  "errors": 0, "skipped": 0, "skipped_detail": [], "failure_detail": [],
                  "clear": True}


def failing_suites() -> tuple[bool, dict]:
    return False, {"suites": list(cli.REQUIRED_SUITES), "run": 232, "failures": 1,
                   "errors": 0, "skipped": 0, "skipped_detail": [],
                   "failure_detail": ["test_a_resume_cannot_demote_the_best_checkpoint"],
                   "clear": False}


def skipping_suites() -> tuple[bool, dict]:
    return False, {"suites": list(cli.REQUIRED_SUITES), "run": 232, "failures": 0,
                   "errors": 0, "skipped": 1,
                   "skipped_detail": ["test_arm_recovery_keras: Keras not installed"],
                   "failure_detail": [], "clear": False}


def clear_preflight() -> dict:
    return {"artifact": "test", "output_path": "/tmp/x",
            "gpu": {}, "memory": {}, "disk": {},
            "filesystem": {"fstype": "ext4", "mount_point": "/", "verdict": "ACCEPTED"},
            "extra": {}, "problems": [], "warnings": [],
            "requires_acknowledgement": [], "acknowledged": [], "unacknowledged": [],
            "clear_to_launch": True}


def preflight_with(problems=(), unacknowledged=(), verdict="ACCEPTED") -> dict:
    report = clear_preflight()
    report["filesystem"]["verdict"] = verdict
    report["problems"] = list(problems)
    report["unacknowledged"] = list(unacknowledged)
    report["requires_acknowledgement"] = list(unacknowledged)
    report["clear_to_launch"] = not problems and not unacknowledged
    return report


class _TrainingStarted(Exception):
    """Raised by the spy the moment the CLI tries to train."""


class _RunnerSpy:
    """
    Stands in for ComparisonRunner and refuses to train.

    `run()` raises a sentinel rather than returning a canned report: the
    question each test asks is whether the CLI reached this point at
    all, and a sentinel records that unambiguously without the test
    having to fabricate an analysis structure the launcher would then
    try to format.
    """

    constructed = 0
    ran = 0

    def __init__(self, *a, **kw):
        type(self).constructed += 1
        self.kwargs = kw

    def run(self, *a, **kw):
        type(self).ran += 1
        raise _TrainingStarted("the CLI proceeded to production training")

    def check_frozen(self, frozen):
        return []

    @staticmethod
    def reset():
        _RunnerSpy.constructed = 0
        _RunnerSpy.ran = 0


class _GateCase(unittest.TestCase):
    """Shared harness: a temp output dir and every gate patched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"
        _RunnerSpy.reset()
        self.frozen = {"config_fingerprint": "c" * 64,
                       "preregistration_sha256": "p" * 64,
                       "declared_architecture_digest": "d" * 64,
                       "source_manifest": {}}

    def tearDown(self):
        self.tmp.cleanup()

    def _execute(self, *, argv=None, frozen=..., suites=None, preflight=None,
                 dir_problems=None):
        """
        Run `main` for --execute with the gates stubbed, capturing stdout
        and asserting throughout that nothing trained.
        """
        argv = list(argv if argv is not None else EXECUTE) + \
            ["--output-dir", str(self.out)]
        buf = io.StringIO()
        with mock.patch.object(cli, "ComparisonRunner", _RunnerSpy), \
             mock.patch.object(cli, "load_frozen_record",
                               return_value=(self.frozen if frozen is ... else frozen)), \
             mock.patch.object(cli, "verify_required_suites",
                               return_value=(suites or clear_suites())), \
             mock.patch.object(cli, "run_preflight",
                               return_value=(preflight or clear_preflight())), \
             mock.patch.object(cli, "check_output_dir",
                               return_value=list(dir_problems or [])), \
             redirect_stdout(buf):
            try:
                code = cli.main(argv)
            except _TrainingStarted:
                code = "TRAINING_STARTED"
        return code, buf.getvalue()

    def assertRefused(self, code, text):
        self.assertEqual(code, 1, f"execute did not refuse. Output:\n{text}")
        self.assertEqual(_RunnerSpy.ran, 0, "training was started despite a refusal")


class TestExecuteRefusesWhenTheTestGateFails(_GateCase):

    def test_a_failing_required_suite_refuses(self):
        code, text = self._execute(suites=failing_suites())
        self.assertRefused(code, text)
        self.assertIn("readiness gate did not clear", text)
        self.assertIn("test_a_resume_cannot_demote_the_best_checkpoint", text)

    def test_a_SKIPPED_required_test_refuses_exactly_like_a_failure(self):
        """
        A skip proves nothing. `test_arm_recovery_keras` skips itself
        when Keras is absent, which is the one stack whose behaviour the
        recovery exactness claim is ABOUT.
        """
        code, text = self._execute(suites=skipping_suites())
        self.assertRefused(code, text)
        self.assertIn("SKIPPED (counts as a failure)", text)
        self.assertIn("Keras not installed", text)

    def test_an_errored_required_suite_refuses(self):
        ok, summary = clear_suites()
        summary.update(errors=1, clear=False, failure_detail=["test_boom"])
        code, text = self._execute(suites=(False, summary))
        self.assertRefused(code, text)

    def test_a_suite_run_that_collected_nothing_refuses(self):
        """
        Zero tests run is not a pass. An import error that empties the
        suite would otherwise look like success.
        """
        ok, summary = clear_suites(run=0)
        summary["clear"] = False
        code, text = self._execute(suites=(False, summary))
        self.assertRefused(code, text)

    def test_the_gate_actually_invokes_the_test_runner(self):
        """
        Guards against the gate being wired to a constant. The real
        `verify_required_suites` must be called with no arguments.
        """
        with mock.patch.object(cli, "verify_required_suites",
                               return_value=clear_suites()) as spy, \
             mock.patch.object(cli, "run_preflight", return_value=clear_preflight()), \
             mock.patch.object(cli, "check_output_dir", return_value=[]):
            ok, gate = cli.readiness_gate(self.out)
        spy.assert_called_once_with()
        self.assertTrue(ok)


class TestExecuteRefusesWhenPreflightIsNotClear(_GateCase):

    def test_a_rejected_filesystem_refuses(self):
        code, text = self._execute(preflight=preflight_with(
            problems=["output directory /content/drive/MyDrive/ii4 is on /content/drive "
                      "(fuse.drive): a FUSE mount. Write the run output to local disk "
                      "instead."],
            verdict="REJECTED"))
        self.assertRefused(code, text)
        self.assertIn("REJECTED", text)
        self.assertIn("fuse.drive", text)

    def test_an_unverified_filesystem_refuses(self):
        code, text = self._execute(preflight=preflight_with(
            problems=["the filesystem backing the output directory could not be "
                      "identified"],
            verdict="UNKNOWN"))
        self.assertRefused(code, text)
        self.assertIn("not claimed for an unverified filesystem", text)

    def test_no_gpu_refuses(self):
        code, text = self._execute(preflight=preflight_with(
            problems=["no GPU is visible to TensorFlow."]))
        self.assertRefused(code, text)
        self.assertIn("no GPU is visible", text)

    def test_insufficient_disk_refuses(self):
        code, text = self._execute(preflight=preflight_with(
            problems=["1.2 GB free against an estimated requirement of 5.0 GB."]))
        self.assertRefused(code, text)

    def test_the_gate_measures_the_exact_output_directory(self):
        """
        Not the repository, not the cwd: the directory the run will
        write to. A preflight on the wrong path is worse than none.
        """
        with mock.patch.object(cli, "verify_required_suites",
                               return_value=clear_suites()), \
             mock.patch.object(cli, "run_preflight",
                               return_value=clear_preflight()) as spy, \
             mock.patch.object(cli, "check_output_dir", return_value=[]):
            cli.readiness_gate(self.out, acknowledged=("idle_gpus",))
        spy.assert_called_once()
        self.assertEqual(Path(spy.call_args.args[0]), self.out)
        self.assertEqual(spy.call_args.kwargs["acknowledged"], ("idle_gpus",))


class TestTheIdleGpuAcknowledgementIsConsistent(_GateCase):

    def test_an_unacknowledged_warning_refuses(self):
        code, text = self._execute(
            preflight=preflight_with(unacknowledged=["idle_gpus"]))
        self.assertRefused(code, text)
        self.assertIn("AWAITING ACKNOWLEDGEMENT", text)
        self.assertIn("--acknowledge-idle-gpus", text)

    def test_the_flag_is_translated_identically_for_every_mode(self):
        """
        One translator, so a flag cannot silence a warning in
        `--preflight` and be ignored in `--execute`.
        """
        class _Args:
            ack_idle = True
        self.assertEqual(cli.acknowledgements_from(_Args()), ("idle_gpus",))
        _Args.ack_idle = False
        self.assertEqual(cli.acknowledgements_from(_Args()), ())
        self.assertEqual(cli.acknowledgements_from(object()), ())

    def test_execute_passes_the_acknowledgement_through_to_the_gate(self):
        with mock.patch.object(cli, "ComparisonRunner", _RunnerSpy), \
             mock.patch.object(cli, "load_frozen_record", return_value=self.frozen), \
             mock.patch.object(cli, "readiness_gate",
                               return_value=(False, {"output_dir": str(self.out),
                                                     "dir_problems": [],
                                                     "suites": clear_suites()[1],
                                                     "preflight": clear_preflight(),
                                                     "clear": False})) as spy, \
             redirect_stdout(io.StringIO()):
            code = cli.main([*EXECUTE, "--output-dir", str(self.out),
                             "--acknowledge-idle-gpus"])
        self.assertEqual(code, 1)
        self.assertEqual(spy.call_args.kwargs["acknowledged"], ("idle_gpus",))

    def test_the_acknowledgement_does_not_excuse_a_blocker(self):
        """
        It acknowledges an idle GPU. It is not a global override.
        """
        code, text = self._execute(
            argv=[*EXECUTE, "--acknowledge-idle-gpus"],
            preflight=preflight_with(problems=["no GPU is visible to TensorFlow."]))
        self.assertRefused(code, text)


class TestExecuteRefusesWhenTheOutputDirectoryIsUnacceptable(_GateCase):

    def test_a_protected_output_directory_refuses(self):
        code, text = self._execute(
            dir_problems=["output_dir lies inside protected subtree "
                          "audit/implementation/runs/ii4-depth"])
        self.assertRefused(code, text)
        self.assertIn("protected subtree", text)


class TestTheDesignGatesStillHold(_GateCase):

    def test_a_missing_frozen_record_refuses_before_anything_else(self):
        code, text = self._execute(frozen=None)
        self.assertRefused(code, text)
        self.assertIn("no frozen record", text)

    def test_omitting_the_reviewed_flag_refuses(self):
        code, text = self._execute(argv=["--execute"])
        self.assertRefused(code, text)
        self.assertIn("i-have-reviewed-the-preregistration", text)

    def test_the_smoke_fault_injector_is_refused_in_production(self):
        code, text = self._execute(
            argv=[*EXECUTE, "--interrupt-after-epoch", "2"])
        self.assertRefused(code, text)
        self.assertIn("smoke-only", text)


class TestACleanGateDoesNotBlockALegitimateRun(_GateCase):
    """
    The counterpart to every refusal above: the gate must not be so
    strict that a correctly prepared run cannot start. Without this,
    a gate that refuses everything would pass this whole suite.
    """

    def test_execute_proceeds_to_the_runner_when_every_gate_clears(self):
        code, text = self._execute()
        self.assertEqual(code, "TRAINING_STARTED",
                         f"the gate blocked a correctly prepared run. Output:\n{text}")
        self.assertEqual(_RunnerSpy.constructed, 1)
        self.assertEqual(_RunnerSpy.ran, 1)
        self.assertIn("GATE                   CLEAR", text)

    def test_the_gate_result_is_printed_before_training_begins(self):
        """
        The operator must be able to see what was verified, in the
        output of the run that acted on it.
        """
        code, text = self._execute()
        self.assertEqual(code, "TRAINING_STARTED")
        self.assertIn("READINESS GATE", text)
        self.assertIn("required suites        232 run", text)
        self.assertIn("filesystem             ext4 on / -> ACCEPTED", text)


class TestTheGateIsNotBypassable(unittest.TestCase):

    def test_execute_calls_the_readiness_gate(self):
        """
        Structural: the execute branch must reference the gate. A future
        edit that drops the call would otherwise only be caught if a
        gate happened to fail.
        """
        import inspect
        source = inspect.getsource(cli.main)
        execute_branch = source[source.index("# --execute"):]
        self.assertIn("readiness_gate(", execute_branch)
        self.assertIn("acknowledgements_from(args)", execute_branch)

    def test_the_dry_run_and_execute_use_the_same_gate_function(self):
        """
        The original defect was two different checks in two modes. Both
        must now call one function, so they cannot drift apart again.
        """
        import inspect
        import re
        source = inspect.getsource(cli.main)
        # A bare call, not `format_readiness_gate(` which contains the
        # same substring.
        calls = re.findall(r"(?<![\w.])readiness_gate\(", source)
        self.assertEqual(len(calls), 2,
                         "exactly two call sites expected: --dry-run and --execute")


if __name__ == "__main__":
    unittest.main(verbosity=2)
