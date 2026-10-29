"""
AutoND D4: epoch-level resume of real fits, and the two-GPU scheduler.

SOFTWARE VALIDATION ON THIS MACHINE (CPU).  What runs for real here: the
pinned AutoND generator and DBitNet on tiny data; real Keras interruption and
restart; the coordinator launching real worker subprocesses (sequential CPU
mode), the single-owner run stage and verify.  What is SYNTHETIC: the
two-GPU launch (subprocess.Popen is replaced to observe concurrency and the
CUDA_VISIBLE_DEVICES assignment).  NOT TESTED HERE: a physical dual-T4 run.

Subprocesses replace only the git/approval anchoring and the repository
source-identity check (a test run directory has no committed preregistration)
through a small shim module; everything else is the real code path.

Run this file in its OWN pytest process (the pinned AutoND binding refuses a
foreign top-level ``train_nets``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_foreign = sys.modules.get("train_nets")
if _foreign is not None and "AutoND" not in str(getattr(_foreign, "__file__", "")):
    pytest.skip("another train_nets is already imported in this process; run this file in its own pytest "
                "process. SKIPPED IS NOT PASSED.", allow_module_level=True)

from audit.dataset import d4_controlled_perturbation as d4  # noqa: E402
from audit.dataset import integrity as I  # noqa: E402
from audit.dataset.adapters import autond_d4 as A  # noqa: E402
import audit.dataset.experiments.d4_autond_label_shuffle as driver  # noqa: E402
import audit.dataset.experiments.d4_autond_two_gpu as sched  # noqa: E402


def _autond_root():
    for candidate in (os.environ.get("AUTOND_ROOT"), "~/Documents/AutoND-upstream"):
        if candidate and (Path(candidate).expanduser() / "main.py").is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


ROOT = _autond_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="pinned AutoND checkout not found; set AUTOND_ROOT. "
                                                     "SKIPPED IS NOT PASSED.")
ANCHOR = "a" * 64
APPROVED = {"decision_rule": "near_chance_baseline_gated_v4", "chance_margin": 0.2, "baseline_adequacy_threshold": 0.505,
            "paired_endpoint_role": "secondary",
            "pipeline_progression_role": "co_primary", "alpha": 0.05, "effect_threshold": 0.01,
            "replicates": 2, "sample_sizes": {"train_per_round": 512, "validation_per_round": 256, "test": 256},
            "starting_round": 1, "max_round": 2, "epochs_per_round": 2,
            "partition_overlap_policy": "refuse_on_any_cross_partition_feature_row",
            "within_partition_duplicates_policy": "record_only",
            "label_conflict_reporting": "report_separately_non_gating",
            "failure_handling": "abort_whole_run_no_exclusions", "calibration": "none_required"}
SOURCE_STUB = {"git_head": "0" * 40, "files": {"x.py": {"sha256": "1" * 64, "clean_at_head": True}}}

SHIM = '''
import json, os, sys
import audit.dataset.experiments.d4_autond_label_shuffle as driver
APPROVED = json.loads(os.environ["AUTOND_D4_TEST_APPROVED"])
driver.load_approved_preregistration = lambda p, a: (
    {}, dict(APPROVED, _commit={"path": "p.json", "sha": "0" * 40}), os.environ["AUTOND_D4_TEST_ANCHOR"])
driver.source_identity = lambda files: json.loads(os.environ["AUTOND_D4_TEST_SOURCE"])
target, argv = sys.argv[1], sys.argv[2:]
if target == "worker":
    from audit.dataset.experiments import d4_autond_two_gpu as two_gpu
    sys.exit(two_gpu.main(["worker", *argv]))
sys.exit(driver.main(argv))
'''


def tree_digest(path: Path) -> dict:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


# ============================================================ real fits: interruption and resume


@pytest.fixture(scope="module")
def tiny():
    rounds, rows, val_rows = [1, 2], 256, 128
    bindings = {r: driver.make_binding(ROOT, r) for r in rounds}
    train = [bindings[r].generate_partition(rows) for r in rounds]
    validation = {r: bindings[r].generate_partition(val_rows) for r in rounds}
    x = np.concatenate([t[0] for t in train])
    y = np.concatenate([t[1] for t in train])
    return SimpleNamespace(rounds=rounds, rows=rows, validation=validation, x=x, y=y, binding=bindings[1])


def make_adapter(tiny, seed=11):
    return A.AutoNDD4Adapter(binding=tiny.binding, rounds=tiny.rounds, round_rows=[tiny.rows] * 2,
                             validation=tiny.validation, evaluation_digest="e" * 64, epochs_per_round=2,
                             batch_size=64, seed=seed, upstream_commit=driver.AUTOND_SOURCE_PIN["upstream_commit"],
                             stop_threshold=0.505, run_binding={"preflight_sha256": "p" * 64})


def train(adapter, tiny, fit_dir, config="c" * 64):
    return adapter.train(tiny.x, tiny.y, checkpoint_path=fit_dir, state_path=fit_dir / "state.json", replicate=1,
                         condition="baseline", run_config_hash=config)


def interrupt_after(monkeypatch, step: int):
    """Raise right after step ``step`` is committed (the crash window after a commit)."""
    real = A._committed_epoch_callback

    def wrapped(**kwargs):
        callback = real(**kwargs)
        commit = callback.on_epoch_end

        def on_epoch_end(epoch, logs=None):
            commit(epoch, logs)
            if kwargs["offset"] + epoch + 1 == step:
                raise RuntimeError(f"simulated crash after committed step {step}")
        callback.on_epoch_end = on_epoch_end
        return callback
    monkeypatch.setattr(A, "_committed_epoch_callback", wrapped)


def committed_records(fit_dir, adapter):
    state = json.loads((fit_dir / "state.json").read_text())
    identity = state["identity"]
    records, _ = A.verify_epoch_chain(fit_dir, plan=adapter.plan, steps=state["completed_steps"], identity=identity)
    return state, records


def test_clean_fit_start_commits_every_step_in_order(tiny, tmp_path):
    adapter = make_adapter(tiny)
    train(adapter, tiny, tmp_path)
    state, records = committed_records(tmp_path, adapter)
    prog = adapter.selection["progression"]
    assert [r["step"] for r in records] == list(range(1, prog["executed_steps"] + 1))
    assert state["completed_steps"] == prog["executed_steps"] and state["retries"] == []
    assert prog["status"] in ("stopped_at_threshold", "cap_reached")
    assert adapter.selection["comparison_round"] == 1


@pytest.mark.parametrize("crash_after", [1, 2])        # inside round 1; at the round-1/round-2 boundary
def test_real_interruption_resumes_at_the_next_committed_step(tiny, tmp_path, monkeypatch, crash_after):
    interrupt_after(monkeypatch, crash_after)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(make_adapter(tiny), tiny, tmp_path)
    monkeypatch.undo()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["completed_steps"] == crash_after
    before = {p: I.sha256_file(p) for p in tmp_path.rglob("checkpoint_epoch_*")}
    head_at_crash = state["record_chain_head"]

    adapter = make_adapter(tiny)
    starts = []
    real_start = adapter._starting_model
    adapter._starting_model = lambda fit_dir, *, completed, identity: (starts.append(completed),
                                                                       real_start(fit_dir, completed=completed,
                                                                                  identity=identity))[1]
    train(adapter, tiny, tmp_path)
    state, records = committed_records(tmp_path, adapter)
    for path, digest in before.items():
        assert I.sha256_file(path) == digest                            # committed steps never retrained
    prog = adapter.selection["progression"]
    if prog["executed_steps"] > crash_after:
        assert starts and starts[0] == crash_after                      # resumed from step N, trains N+1
        assert records[crash_after]["previous_record_sha256"] == head_at_crash
    else:
        assert crash_after == 2 and prog["status"] == "stopped_at_threshold" and starts == []
        assert not (tmp_path / "stage_r02").exists()                    # nothing trained after the stop
    assert [r["step"] for r in records] == list(range(1, prog["executed_steps"] + 1))
    assert state["retries"] == []                                       # committed crash: nothing quarantined


def test_finished_fit_is_verified_and_never_retrained(tiny, tmp_path, monkeypatch):
    import tensorflow as tf

    first = make_adapter(tiny)
    train(first, tiny, tmp_path)
    snapshot = tree_digest(tmp_path)
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *a, **k: pytest.fail("a finished fit was retrained"))
    second = make_adapter(tiny)
    train(second, tiny, tmp_path)
    assert second.selection == first.selection
    assert tree_digest(tmp_path) == snapshot
    assert second.predict(tiny.x[:8]).shape == (8, 1)                   # evaluated from the saved selection


@pytest.mark.parametrize("change, match", [({"seed": 12}, "different fit"), ({"config": "d" * 64}, "different fit")])
def test_resume_refuses_a_different_fit_identity(tiny, tmp_path, monkeypatch, change, match):
    interrupt_after(monkeypatch, 1)
    with pytest.raises(RuntimeError):
        train(make_adapter(tiny), tiny, tmp_path)
    monkeypatch.undo()
    snapshot = tree_digest(tmp_path)
    with pytest.raises(I.IntegrityError, match=match):
        train(make_adapter(tiny, seed=change.get("seed", 11)), tiny, tmp_path, config=change.get("config", "c" * 64))
    assert tree_digest(tmp_path) == snapshot


# ============================================================ scheduler: real CPU execution through subprocesses


def shim_commands(monkeypatch, shim_dir: Path):
    (shim_dir / "autond_d4_test_shim.py").write_text(SHIM)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(shim_dir), str(I.REPOSITORY_ROOT)]))
    monkeypatch.setenv("AUTOND_D4_TEST_APPROVED", json.dumps(APPROVED))
    monkeypatch.setenv("AUTOND_D4_TEST_ANCHOR", ANCHOR)
    monkeypatch.setenv("AUTOND_D4_TEST_SOURCE", json.dumps(SOURCE_STUB))
    real_worker, real_driver = sched._worker_command, sched._driver_command

    def via_shim(cmd, target):
        index = cmd.index("-m")
        return [*cmd[:index], "-m", "autond_d4_test_shim", target, *cmd[index + 3:]] if target == "worker" else \
            [*cmd[:index], "-m", "autond_d4_test_shim", "driver", *cmd[index + 2:]]

    monkeypatch.setattr(sched, "_worker_command", lambda *a: via_shim(real_worker(*a), "worker"))
    monkeypatch.setattr(sched, "_driver_command", lambda *a: via_shim(real_driver(*a), "driver"))


def stub_in_process(monkeypatch):
    monkeypatch.setattr(driver, "load_approved_preregistration", lambda p, a: (
        {}, dict(APPROVED, _commit={"path": "p.json", "sha": "0" * 40}), ANCHOR))
    monkeypatch.setattr(driver, "source_identity", lambda files: SOURCE_STUB)


def exec_args(run_dir, mode="sequential", confirm=None, cpu=True):
    return SimpleNamespace(run_dir=run_dir, preregistration=Path("p.json"), preregistration_sha256=ANCHOR,
                           autond_root=ROOT, mode=mode, confirm=confirm or sched.CONFIRM[mode], cpu=cpu,
                           poll_seconds=0.5)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    stub_in_process(mp)
    run_dir = tmp_path_factory.mktemp("sched") / "d4_run"
    assert driver.main(["preflight", "--preregistration", "p.json", "--preregistration-sha256", ANCHOR,
                        "--run-dir", str(run_dir), "--autond-root", ROOT]) == 0
    yield run_dir
    mp.undo()


@pytest.fixture(scope="module")
def executed(prepared, tmp_path_factory):
    """One complete sequential CPU execution: real workers, real run stage, real verify."""
    mp = pytest.MonkeyPatch()
    stub_in_process(mp)
    shim_commands(mp, tmp_path_factory.mktemp("shim"))
    run_dir = prepared.parent / "executed"
    shutil.copytree(prepared, run_dir)
    code = sched.execute(exec_args(run_dir))
    yield run_dir, code, mp
    mp.undo()


def _execution(run_dir):
    executions = sorted((run_dir / "execution").iterdir())
    return executions[-1], json.loads((executions[-1] / "execution_manifest.json").read_text())


def test_sequential_cpu_execution_trains_both_arms_then_single_owner_run_certifies(executed):
    run_dir, code, _ = executed
    assert code == 0
    directory, manifest = _execution(run_dir)
    assert manifest["mode"] == "sequential" and manifest["cpu"] is True
    assert manifest["physical_two_gpu_confirmed"] is False
    assert manifest["arm_assignment"] == {"baseline": "CPU", "perturbed": "CPU"}
    assert manifest["worker_exit_codes"] == {"baseline": 0, "perturbed": 0}
    assert manifest["final_stage_exit_codes"] == {"run": 0, "verify": 0}
    assert manifest["finalization"] == "COMPLETE"
    assert manifest["scheduler_sha256"] == hashlib.sha256(Path(sched.__file__).read_bytes()).hexdigest()
    for condition in ("baseline", "perturbed"):
        summary = manifest["worker_summaries"][condition]
        assert summary["status"] == "ARM_COMPLETE_AND_VERIFIED" and summary["cuda_visible_devices"] == ""
        assert summary["tensorflow_gpu"]["probe_device"] == "CPU"
        assert [f["replicate"] for f in summary["fits"]] == [1, 2]
        assert all(f["status_before"] == "absent" for f in summary["fits"])
        assert (directory / f"{condition}_worker.log").stat().st_size > 0
    # The run stage evaluated what the workers trained, without retraining.
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    for row in certificate["findings"]["replicates"]:
        for condition in ("baseline", "perturbed"):
            fit = manifest["worker_summaries"][condition]["fits"][row["replicate"] - 1]
            selection = row[f"{condition}_selection"]
            assert selection["record_chain_head"] == fit["record_chain_head"]
            assert selection["checkpoint_sha256"] == fit["selected_checkpoint_sha256"]
    assert "autond_decision" in certificate and certificate["autond_decision"]["pipeline_progression_role"] == "co_primary"
    for lock in (run_dir / driver.SCHEDULER_LOCK, run_dir / driver.RUN_LOCK,
                 driver.arm_lock_path(run_dir, "baseline"), driver.arm_lock_path(run_dir, "perturbed")):
        assert not lock.exists()
    assert driver.verify_run(run_dir)["integrity"] == "VERIFIED"


def test_baseline_and_shuffled_fits_have_independent_directories_and_identities(executed):
    run_dir, _, _ = executed
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    dirs = set()
    for row in certificate["findings"]["replicates"]:
        base, pert = row["baseline_selection"], row["perturbed_selection"]
        assert (base["condition"], pert["condition"]) == ("baseline", "perturbed")
        assert base["train_digest"] != pert["train_digest"] and base["seed"] == pert["seed"]
        assert base["run_binding"] == pert["run_binding"] == driver.run_binding_of(
            json.loads((run_dir / "preflight.json").read_text()), run_dir)
        assert base["adapter"] == A.ADAPTER_IDENTITY
        for condition in ("baseline", "perturbed"):
            dirs.add(d4.checkpoint_paths(run_dir / "fits", row["replicate"], condition)[0])
    assert len(dirs) == 4


def _rerun_copy(executed, tmp_path, name):
    run_dir, _, _ = executed
    copy = tmp_path / name
    shutil.copytree(run_dir, copy)
    shutil.rmtree(copy / "result")
    return copy


def test_coordinator_restart_does_no_duplicate_work(executed, tmp_path):
    copy = _rerun_copy(executed, tmp_path, "restart")
    fits_before = tree_digest(copy / "fits")
    assert sched.execute(exec_args(copy)) == 0
    _, manifest = _execution(copy)
    for condition in ("baseline", "perturbed"):
        assert all(f["status_before"] == "finished" for f in manifest["worker_summaries"][condition]["fits"])
    after = tree_digest(copy / "fits")
    assert {k: v for k, v in after.items() if k in fits_before} == fits_before      # nothing rewritten
    assert set(after) - set(fits_before) <= {k for k in after if k.startswith("predictions/")}
    assert driver.verify_run(copy)["integrity"] == "VERIFIED"


def test_partial_fit_resumes_at_its_next_step_and_other_fits_are_skipped(executed, tmp_path):
    copy = _rerun_copy(executed, tmp_path, "partial")
    shutil.rmtree(copy / "fits" / "predictions")
    fit_dir, state_path = d4.checkpoint_paths(copy / "fits", 2, "perturbed")
    state = json.loads(state_path.read_text())
    executed_steps = state["completed_steps"]
    plan = A.step_plan([1, 2], 2)
    _, head = A.verify_epoch_chain(fit_dir, plan=plan, steps=executed_steps - 1, identity=state["identity"])
    A.write_state(state_path, identity=state["identity"], total_steps=len(plan), completed=executed_steps - 1,
                  chain_head=head, retries=[])
    (fit_dir / "selected_checkpoint.json").unlink()
    assert sched.execute(exec_args(copy)) == 0
    _, manifest = _execution(copy)
    fits = {(c, f["replicate"]): f for c in ("baseline", "perturbed") for f in manifest["worker_summaries"][c]["fits"]}
    assert fits[("perturbed", 2)]["status_before"] == "partial"
    assert fits[("perturbed", 2)]["steps_before"] == executed_steps - 1
    assert fits[("perturbed", 2)]["retries"][0]["reason"] == "complete_uncommitted_orphan"
    assert all(f["status_before"] == "finished" for k, f in fits.items() if k != ("perturbed", 2))


def test_one_worker_failure_preserves_the_other_arms_evidence_and_blocks_the_result(executed, tmp_path):
    copy = _rerun_copy(executed, tmp_path, "failure")
    (copy / "fits" / "perturbations" / "replicate_01_indices.npy").unlink()     # perturbed arm cannot verify
    before = tree_digest(copy / "fits")
    assert sched.execute(exec_args(copy)) == 2
    directory, manifest = _execution(copy)
    assert manifest["finalization"] == "BLOCKED_WORKER_FAILURE_NO_RESULT_ATTEMPTED"
    assert "final_stage_exit_codes" not in manifest                     # run/verify never launched
    assert not (directory / "run.log").exists() and not (directory / "verify.log").exists()
    assert manifest["worker_exit_codes"] == {"baseline": 0, "perturbed": 1}
    assert "missing" in manifest["worker_summaries"]["perturbed"]["error"]
    assert tree_digest(copy / "fits") == before and not (copy / "result").exists()
    assert not (copy / driver.SCHEDULER_LOCK).exists() and not driver.arm_lock_path(copy, "perturbed").exists()


def test_inspect_is_read_only_and_reports_every_fit(executed, tmp_path, capsys):
    copy = _rerun_copy(executed, tmp_path, "inspect")
    before = {p: (I.sha256_file(p), p.stat().st_mtime_ns) for p in copy.rglob("*") if p.is_file()}
    plan = sched.inspect_run(copy, Path("p.json"), ANCHOR)
    assert len(plan["fits"]) == 4 and all(f["status"] == "finished" for f in plan["fits"])
    assert plan["work"] == {"baseline": [], "perturbed": []}
    sched.print_plan(plan, [])
    assert {p: (I.sha256_file(p), p.stat().st_mtime_ns) for p in copy.rglob("*") if p.is_file()} == before
    assert "would train/resume" in capsys.readouterr().out


# ============================================================ locks and isolation


def test_a_second_scheduler_and_concurrent_stages_are_refused(executed, tmp_path, monkeypatch):
    copy = _rerun_copy(executed, tmp_path, "locks")
    lock = copy / driver.SCHEDULER_LOCK
    token = driver.acquire_lock(lock, {"role": "scheduler"})
    content = lock.read_text()
    with pytest.raises(I.IntegrityError, match="Lock .* exists"):
        sched.execute(exec_args(copy))
    assert lock.read_text() == content                                   # never broken automatically
    args = driver.parse_args(["run", "--preregistration", "p.json", "--preregistration-sha256", ANCHOR,
                              "--run-dir", str(copy), "--autond-root", ROOT])
    with pytest.raises(I.IntegrityError, match="held by another process"):
        driver.stage_run(args)
    driver.release_lock(lock, "not-the-owner")
    assert lock.exists()
    driver.release_lock(lock, token)
    arm = driver.arm_lock_path(copy, "baseline")
    arm_token = driver.acquire_lock(arm, {"role": "worker"})
    with pytest.raises(I.IntegrityError, match="Arm worker lock"):
        driver.stage_run(args)
    driver.release_lock(arm, arm_token)
    assert not (copy / "result").exists()


def _worker_subprocess(run_dir, shim_dir, env_extra, condition="baseline", visible="", preimport=False):
    (shim_dir / "autond_d4_test_shim.py").write_text(SHIM)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(shim_dir), str(I.REPOSITORY_ROOT)]),
           "AUTOND_D4_TEST_APPROVED": json.dumps(APPROVED), "AUTOND_D4_TEST_ANCHOR": ANCHOR,
           "AUTOND_D4_TEST_SOURCE": json.dumps(SOURCE_STUB), "CUDA_VISIBLE_DEVICES": visible,
           "TF_CPP_MIN_LOG_LEVEL": "3", **env_extra}
    args = ["worker", "--run-dir", str(run_dir), "--preregistration", "p.json", "--preregistration-sha256", ANCHOR,
            "--autond-root", ROOT, "--condition", condition, "--summary", str(shim_dir / f"s_{os.urandom(4).hex()}.json"),
            "--expected-visible", "0" if preimport else ""]
    if preimport:
        code = ("import tensorflow, sys\nfrom audit.dataset.experiments import d4_autond_two_gpu as t\n"
                f"sys.exit(t.main({args!r}))")
        cmd = [sys.executable, "-B", "-c", code]
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        cmd = [sys.executable, "-B", "-m", "autond_d4_test_shim", *args]
    return subprocess.run(cmd, cwd=str(I.REPOSITORY_ROOT), env=env, capture_output=True, text=True, timeout=600)


def test_worker_refuses_without_the_scheduler_or_with_a_foreign_token_or_a_held_arm(executed, tmp_path):
    copy = _rerun_copy(executed, tmp_path, "worker_refusals")
    before = tree_digest(copy / "fits")
    out = _worker_subprocess(copy, tmp_path, {})
    assert out.returncode != 0 and "only under a scheduler" in out.stderr + out.stdout
    token = driver.acquire_lock(copy / driver.SCHEDULER_LOCK, {"role": "scheduler"})
    out = _worker_subprocess(copy, tmp_path, {driver.TOKEN_ENV: "wrong"})
    assert out.returncode != 0 and "held by another process" in out.stderr + out.stdout
    arm_token = driver.acquire_lock(driver.arm_lock_path(copy, "baseline"), {"role": "worker"})
    out = _worker_subprocess(copy, tmp_path, {driver.TOKEN_ENV: token})
    assert out.returncode != 0 and "exists" in out.stderr + out.stdout       # two workers, same arm: refused
    driver.release_lock(driver.arm_lock_path(copy, "baseline"), arm_token)
    driver.release_lock(copy / driver.SCHEDULER_LOCK, token)
    assert tree_digest(copy / "fits") == before


def test_worker_establishes_its_gpu_assignment_before_tensorflow_is_imported(executed, tmp_path):
    copy = _rerun_copy(executed, tmp_path, "worker_env")
    out = _worker_subprocess(copy, tmp_path, {}, preimport=True)
    assert out.returncode != 0 and "TensorFlow was imported before" in out.stderr + out.stdout
    out = _worker_subprocess(copy, tmp_path, {"CUDA_VISIBLE_DEVICES": "1"})
    assert out.returncode != 0 and "expected CUDA_VISIBLE_DEVICES" in out.stderr + out.stdout
    code = ("import sys\nimport audit.dataset.experiments.d4_autond_two_gpu, audit.dataset.experiments."
            "d4_autond_label_shuffle, audit.dataset.adapters.autond_d4\nprint('tensorflow' in sys.modules)")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(I.REPOSITORY_ROOT), capture_output=True, text=True)
    assert out.stdout.strip() == "False"                                 # coordinator never imports TF


# ============================================================ two-GPU launch (synthetic: no GPUs here)


def test_two_gpu_mode_is_refused_without_two_gpus_and_needs_confirmation(executed, tmp_path, monkeypatch):
    copy = _rerun_copy(executed, tmp_path, "nogpu")
    before = tree_digest(copy)
    monkeypatch.setattr(sched, "gpu_inventory", lambda: [])
    with pytest.raises(SystemExit, match="GPUs 0 and 1 are required"):
        sched.execute(exec_args(copy, mode="two-gpu", cpu=False))
    with pytest.raises(SystemExit, match="requires --confirm"):
        sched.execute(exec_args(copy, mode="two-gpu", confirm="yes", cpu=False))
    with pytest.raises(SystemExit, match="pass --cpu"):
        sched.execute(exec_args(copy, mode="sequential", cpu=False))
    assert tree_digest(copy) == before            # refused before any lock, execution directory or fit write


def test_two_gpu_mode_launches_both_arms_concurrently_on_their_assigned_gpus(executed, tmp_path, monkeypatch):
    copy = _rerun_copy(executed, tmp_path, "twogpu")
    monkeypatch.setattr(sched, "gpu_inventory", lambda: [{"index": 0, "name": "Tesla T4", "uuid": "u0"},
                                                          {"index": 1, "name": "Tesla T4", "uuid": "u1"}])
    events = []

    class FakeProcess:
        def __init__(self, cmd, cwd, env, stdout, stderr):
            condition = cmd[cmd.index("--condition") + 1]
            self.condition, self.summary = condition, Path(cmd[cmd.index("--summary") + 1])
            self.returncode, self.polls = None, 0
            events.append(("start", condition, env["CUDA_VISIBLE_DEVICES"], env.get(driver.TOKEN_ENV)))

        def poll(self):
            self.polls += 1
            if self.polls > 2 and self.returncode is None:
                self.returncode = 0
                self.summary.write_text(json.dumps({"status": "ARM_COMPLETE_AND_VERIFIED", "condition": self.condition}))
                events.append(("finish", self.condition))
            return self.returncode

        def wait(self):
            while self.poll() is None:
                pass
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(sched.subprocess, "Popen", FakeProcess)
    def fake_run(cmd, **kwargs):
        stage = next(token for token in cmd if token in ("run", "verify"))
        events.append(("final", stage, "--rerun-models" in cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sched.subprocess, "run", fake_run)
    assert sched.execute(exec_args(copy, mode="two-gpu", cpu=False)) == 0
    starts = [e for e in events if e[0] == "start"]
    assert [(e[1], e[2]) for e in starts] == [("baseline", "0"), ("perturbed", "1")]
    assert events.index(starts[1]) < min(i for i, e in enumerate(events) if e[0] == "finish")   # concurrent
    finals = [i for i, e in enumerate(events) if e[0] == "final"]
    assert [events[i][1:] for i in finals] == [("run", False), ("verify", True)]
    assert min(finals) > max(i for i, e in enumerate(events) if e[0] == "finish")   # only after both workers
    _, manifest = _execution(copy)
    assert len({e[3] for e in starts}) == 1 and starts[0][3] == manifest["scheduler_token"]
    assert manifest["mode"] == "two-gpu" and manifest["arm_assignment"] == {"baseline": "GPU 0", "perturbed": "GPU 1"}
    assert manifest["physical_two_gpu_confirmed"] is False
    assert not (copy / driver.SCHEDULER_LOCK).exists()
