"""CE1 resume/parallel infrastructure, exercised on synthetic partial-run state."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from audit.cryptography.experiments.ce1 import resume as R
from audit.cryptography.frozen_design import CE1 as CE1_SPEC

EXPECTED = {"config_fingerprint": "fp123", "design_version": "CE-frozen-design-2026-03",
            "assignment_seed": 20260101, "terminal_epoch": 200}


def _state(block, arm, epoch, status="IN_PROGRESS", **over):
    kw = dict(experiment_id="CE1-SIGNAL-DESTRUCTION", block_id=block, arm=arm,
              status=status, last_completed_epoch=epoch, terminal_epoch=200,
              model_seed=1000, assignment_seed=20260101, assignment_flip=0,
              permutation_seed=1000, config_fingerprint="fp123",
              design_version="CE-frozen-design-2026-03",
              train_data_sha256="t" * 64, validation_data_sha256="v" * 64,
              sealed_evaluation_sha256="s" * 64, optimizer_state_included=True)
    kw.update(over)
    return R.ArmState(**kw)


def _write(run, block, arm, epoch, status="IN_PROGRESS", with_model=True, **over):
    st = _state(block, arm, epoch, status, **over)
    if with_model and epoch > 0:
        d = R.arm_dir(run, block, arm); d.mkdir(parents=True, exist_ok=True)
        f = d / f"epoch_{epoch:03d}.keras"; f.write_bytes(b"fake-model")
        st.model_artifact = R.rel_to_run(run, f); st.model_sha256 = R.sha256_file(f)
        if status in ("TRAINING_COMPLETE", "EVALUATED"):
            st.terminal_model_artifact = st.model_artifact; st.terminal_model_sha256 = st.model_sha256
        if status == "EVALUATED":
            st.evaluation = {"accuracy": 0.9, "model_sha256": st.model_sha256,
                             "sealed_evaluation_sha256": "s" * 64, "n_samples": 10}
    R.atomic_write_json(R.arm_state_path(run, block, arm), st.to_dict())
    return st


# ---------------- state semantics ----------------

def test_next_epoch_is_last_completed_plus_one():
    st = _state("block2", "baseline", 180)
    assert st.last_completed_epoch == 180 and st.next_epoch == 181


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = R.atomic_write_json(tmp_path / "s.json", {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_epoch_is_never_counted_complete(tmp_path):
    """A stray artifact without a state update must not advance the counter."""
    _write(tmp_path, "block0", "baseline", 100)
    (R.arm_dir(tmp_path, "block0", "baseline") / "epoch_101.keras").write_bytes(b"partial")
    st = R.load_arm_state(tmp_path, "block0", "baseline")
    assert st.last_completed_epoch == 100 and st.next_epoch == 101


# ---------------- scan + plan ----------------

def test_both_arms_partial_resume_at_exact_epochs(tmp_path):
    _write(tmp_path, "block0", "baseline", 180)
    _write(tmp_path, "block0", "destroyed", 100)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    arms = rep["blocks"]["block0"]["arms"]
    assert arms["baseline"]["next_epoch"] == 181
    assert arms["destroyed"]["next_epoch"] == 101
    plan = R.plan_resume(rep, terminal_epoch=200, n_gpus=2)
    tasks = plan["plan"][0]["tasks"]
    assert tasks[0] == {"arm": "baseline", "action": "TRAIN", "gpu": 0,
                        "from_epoch": 181, "to_epoch": 200}
    assert tasks[1]["gpu"] == 1 and tasks[1]["from_epoch"] == 101
    assert plan["plan"][0]["parallel"] is True


def test_completed_block_is_skipped(tmp_path):
    for arm in R.ARMS:
        _write(tmp_path, "block0", arm, 200, status="EVALUATED")
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert rep["blocks"]["block0"]["block_status"] == "COMPLETE"
    assert R.plan_resume(rep, terminal_epoch=200)["plan"][0]["action"] == "SKIP"


def test_completed_arm_is_not_relaunched(tmp_path):
    _write(tmp_path, "block0", "baseline", 200, status="EVALUATED")
    _write(tmp_path, "block0", "destroyed", 100)
    plan = R.plan_resume(R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED),
                         terminal_epoch=200, n_gpus=2)
    tasks = {t["arm"]: t for t in plan["plan"][0]["tasks"]}
    assert tasks["baseline"]["action"] == "SKIP"
    assert tasks["destroyed"]["action"] == "TRAIN" and tasks["destroyed"]["from_epoch"] == 101
    assert plan["plan"][0]["parallel"] is False       # only one arm left to run


def test_not_started_block_starts_at_epoch_one(tmp_path):
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert rep["blocks"]["block0"]["block_status"] == "NOT_STARTED"
    t = R.plan_resume(rep, terminal_epoch=200)["plan"][0]["tasks"]
    assert all(x["from_epoch"] == 1 for x in t)


def test_mixed_blocks_complete_and_partial(tmp_path):
    for arm in R.ARMS:
        _write(tmp_path, "block0", arm, 200, status="EVALUATED")
    _write(tmp_path, "block1", "baseline", 180)
    _write(tmp_path, "block1", "destroyed", 100)
    plan = R.plan_resume(R.scan_run(tmp_path, n_blocks=2, expected=EXPECTED),
                         terminal_epoch=200, n_gpus=2)
    assert plan["plan"][0]["action"] == "SKIP" and plan["plan"][1]["action"] == "RUN"


def test_failed_arm_preserves_completed_epochs(tmp_path):
    st = _write(tmp_path, "block0", "baseline", 137)
    st = R.record_arm_failure(tmp_path, st, RuntimeError("session died"))
    reloaded = R.load_arm_state(tmp_path, "block0", "baseline")
    assert reloaded.status == "FAILED"
    assert reloaded.last_completed_epoch == 137          # NOT destroyed
    assert reloaded.failure["failed_after_epoch"] == 137
    assert reloaded.next_epoch == 138


# ---------------- fail-closed refusals ----------------

@pytest.mark.parametrize("over,needle", [
    ({"config_fingerprint": "OTHER"}, "config_fingerprint"),
    ({"design_version": "CE-frozen-design-2026-02"}, "design_version"),
    ({"assignment_seed": 1}, "assignment_seed"),
    ({"assignment_flip": 1}, "assignment_flip"),
    ({"sealed_evaluation_sha256": "z" * 64}, "sealed_evaluation_sha256"),
    ({"terminal_epoch": 100}, "terminal_epoch"),
])
def test_incompatible_artifacts_refuse_resume(tmp_path, over, needle):
    _write(tmp_path, "block0", "baseline", 50, **over)
    exp = dict(EXPECTED, assignment_flip=0, sealed_evaluation_sha256="s" * 64)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=exp)
    assert rep["resume_refused"] is True
    assert any(needle in r for r in rep["refusal_reasons"])
    with pytest.raises(R.ResumeRefused):
        R.plan_resume(rep, terminal_epoch=200)


def test_corrupted_checkpoint_refuses_resume(tmp_path):
    _write(tmp_path, "block0", "baseline", 50)
    art = R.resolve_artifact(tmp_path, R.load_arm_state(tmp_path, "block0", "baseline").model_artifact)
    art.write_bytes(b"corrupted-now")
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert any("corrupted" in r for r in rep["refusal_reasons"])


def test_missing_checkpoint_refuses_resume(tmp_path):
    _write(tmp_path, "block0", "baseline", 50)
    R.resolve_artifact(tmp_path, R.load_arm_state(tmp_path, "block0", "baseline").model_artifact).unlink()
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert any("missing" in r for r in rep["refusal_reasons"])


def test_weights_without_optimizer_state_refuse_exact_resume(tmp_path):
    _write(tmp_path, "block0", "baseline", 50, optimizer_state_included=False)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert any("optimizer state" in r for r in rep["refusal_reasons"])


def test_missing_training_data_hash_refuses_exact_resume(tmp_path):
    """os.urandom data are not seed-replayable; without them resume is not exact."""
    _write(tmp_path, "block0", "baseline", 50, train_data_sha256=None)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert any("not seed-replayable" in r for r in rep["refusal_reasons"])


def test_unknown_state_schema_refused(tmp_path):
    d = R.arm_dir(tmp_path, "block0", "baseline"); d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(R.ResumeRefused, match="unknown state schema"):
        R.load_arm_state(tmp_path, "block0", "baseline")


def test_insufficient_valid_blocks_is_visible(tmp_path):
    for i in range(8):
        bid = f"block{i}"
        if i < 5:
            for arm in R.ARMS:
                _write(tmp_path, bid, arm, 200, status="EVALUATED")
    rep = R.scan_run(tmp_path, n_blocks=8, expected=EXPECTED)
    complete = sum(1 for e in rep["blocks"].values() if e["block_status"] == "COMPLETE")
    assert complete == 5 < CE1_SPEC.min_valid_blocks     # caller must refuse to certify


# ---------------- GPU ----------------

def test_pin_gpu_sets_visibility():
    import os
    try:
        assert R.pin_gpu(1) == "1" and os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert R.pin_gpu(0) == "0"
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def test_parallelism_is_reported_honestly():
    info = R.gpu_parallelism_available(2)
    assert set(info) >= {"gpus_visible", "parallel_active", "mode"}
    if info["gpus_visible"] < 2:
        assert info["parallel_active"] is False
        assert info["mode"] == "sequential_fallback"


def test_arms_get_distinct_gpus_when_two_available(tmp_path):
    _write(tmp_path, "block0", "baseline", 0, status="NOT_STARTED", with_model=False)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    tasks = R.plan_resume(rep, terminal_epoch=200, n_gpus=2)["plan"][0]["tasks"]
    gpus = [t["gpu"] for t in tasks if t["action"] == "TRAIN"]
    assert len(set(gpus)) == len(gpus) == 2


def test_single_gpu_plan_does_not_claim_parallelism(tmp_path):
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    plan = R.plan_resume(rep, terminal_epoch=200, n_gpus=1)
    assert plan["plan"][0]["parallel"] is False


def test_format_plan_is_human_readable(tmp_path):
    _write(tmp_path, "block0", "baseline", 180)
    _write(tmp_path, "block0", "destroyed", 100)
    text = R.format_plan(R.plan_resume(R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED),
                                       terminal_epoch=200, n_gpus=2))
    assert "GPU 0 -> epochs 181-200" in text and "GPU 1 -> epochs 101-200" in text


# ---------------- equivalence of interrupted vs uninterrupted ----------------

def test_resumed_run_reaches_the_same_terminal_epoch(tmp_path):
    """Scheduler-level equivalence: interrupted+resumed covers every epoch once."""
    covered = []
    _write(tmp_path, "block0", "baseline", 120)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    t = [x for x in R.plan_resume(rep, terminal_epoch=200)["plan"][0]["tasks"]
         if x["arm"] == "baseline"][0]
    covered += list(range(1, 121)) + list(range(t["from_epoch"], t["to_epoch"] + 1))
    assert covered == list(range(1, 201))            # no gap, no repeat


def test_config_fingerprint_is_stable_and_sensitive():
    a = R.config_fingerprint({"k": 8}, {"rounds": 5}, "abc")
    assert a == R.config_fingerprint({"k": 8}, {"rounds": 5}, "abc")
    assert a != R.config_fingerprint({"k": 6}, {"rounds": 5}, "abc")
    assert a != R.config_fingerprint({"k": 8}, {"rounds": 7}, "abc")
