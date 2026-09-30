"""CE1 resume/parallel test matrix A-K (synthetic artifacts; no production run)."""
import json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from audit.cryptography.experiments.ce1 import ce1_datasets as DS
from audit.cryptography.experiments.ce1 import ce1_ledger as L
from audit.cryptography.experiments.ce1 import legacy_validator as LV
from audit.cryptography.experiments.ce1 import resume as R
from audit.cryptography.frozen_design import CE1 as SPEC

EXPECTED = {"config_fingerprint": "fp", "design_version": "CE-frozen-design-2026-03",
            "assignment_seed": 20260101, "terminal_epoch": 200}


def _arm(run, block, arm, epoch, status="IN_PROGRESS", **over):
    kw = dict(experiment_id="CE1", block_id=block, arm=arm, status=status,
              last_completed_epoch=epoch, terminal_epoch=200, model_seed=1000,
              assignment_seed=20260101, assignment_flip=0, permutation_seed=1000,
              config_fingerprint="fp", design_version="CE-frozen-design-2026-03",
              train_data_sha256="t" * 64, validation_data_sha256="v" * 64,
              sealed_evaluation_sha256="s" * 64, optimizer_state_included=True)
    kw.update(over)
    st = R.ArmState(**kw)
    if epoch > 0 and over.get("model_artifact", True) is not None:
        d = R.arm_dir(run, block, arm); d.mkdir(parents=True, exist_ok=True)
        f = d / f"epoch_{epoch:03d}.keras"; f.write_bytes(b"model")
        st.model_artifact = R.rel_to_run(run, f); st.model_sha256 = R.sha256_file(f)
        if status in ("TRAINING_COMPLETE", "EVALUATED"):
            st.terminal_model_artifact = st.model_artifact; st.terminal_model_sha256 = st.model_sha256
        if status == "EVALUATED":
            st.evaluation = {"accuracy": 0.9, "model_sha256": st.model_sha256,
                             "sealed_evaluation_sha256": "s" * 64, "n_samples": 10}
    R.atomic_write_json(R.arm_state_path(run, block, arm), st.to_dict())
    return st


# ---- A. dataset persistence ----
def test_A_dataset_persist_reload_verify(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (64, 64), dtype=np.uint8); Y = rng.integers(0, 2, 64, dtype=np.uint8)
    m = DS.persist_block_dataset(tmp_path, block_id="block_00", role="train", X=X, Y=Y,
                                 rounds=5, differential=(0x0040, 0x0000))
    X2, Y2, m2 = DS.load_block_dataset(tmp_path, block_id="block_00", role="train",
                                       expected_sha256=m["sha256"], rounds=5,
                                       differential=(0x0040, 0x0000))
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2)
    assert X2.dtype == X.dtype and list(X2.shape) == m["shape_X"]
    assert m2["sha256"] == m["sha256"] and "NOT seed-replayable" in m["provenance"]


def test_A_refuses_to_overwrite_with_different_content(tmp_path):
    a = np.zeros((8, 8), np.uint8); y = np.zeros(8, np.uint8)
    DS.persist_block_dataset(tmp_path, block_id="b", role="train", X=a, Y=y,
                             rounds=5, differential=(0x40, 0))
    with pytest.raises(DS.DatasetStateError, match="Refusing to overwrite"):
        DS.persist_block_dataset(tmp_path, block_id="b", role="train", X=a + 1, Y=y,
                                 rounds=5, differential=(0x40, 0))


def test_A_missing_dataset_fails_closed(tmp_path):
    with pytest.raises(DS.DatasetStateError, match="not seed-replayable"):
        DS.load_block_dataset(tmp_path, block_id="nope", role="train")


# ---- B. epoch checkpoint/state creation ----
def test_B_state_written_per_epoch(tmp_path):
    st = _arm(tmp_path, "block0", "baseline", 0, status="NOT_STARTED")
    for e in (1, 2, 3):
        st = R.record_completed_epoch(tmp_path, st, epoch=e)
        assert R.load_arm_state(tmp_path, "block0", "baseline").last_completed_epoch == e
    assert st.status == "IN_PROGRESS"


def test_B_terminal_epoch_marks_complete(tmp_path):
    st = _arm(tmp_path, "block0", "baseline", 199)
    st = R.record_completed_epoch(tmp_path, st, epoch=200)
    # epoch 200 is TRAINING_COMPLETE, NOT scientific completion
    assert st.status == "TRAINING_COMPLETE"


# ---- C. interruption -> restart -> resume ----
def test_C_interrupted_run_resumes_at_next_epoch(tmp_path):
    _arm(tmp_path, "block0", "baseline", 137)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)          # fresh "process"
    a = rep["blocks"]["block0"]["arms"]["baseline"]
    assert a["last_completed_epoch"] == 137 and a["next_epoch"] == 138
    t = [x for x in R.plan_resume(rep, terminal_epoch=200)["plan"][0]["tasks"]
         if x["arm"] == "baseline"][0]
    assert t["from_epoch"] == 138 and t["to_epoch"] == 200


def test_C_no_completed_epoch_is_retrained(tmp_path):
    _arm(tmp_path, "block0", "baseline", 137)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    t = [x for x in R.plan_resume(rep, terminal_epoch=200)["plan"][0]["tasks"]
         if x["arm"] == "baseline"][0]
    covered = list(range(1, 138)) + list(range(t["from_epoch"], 201))
    assert covered == list(range(1, 201))


def test_C_resume_semantics_are_explicit(tmp_path):
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    sem = rep["resume_semantics"]
    # infrastructure exists; production resume is a SEPARATE claim that only the
    # real interruption integration test can raise
    assert sem["checkpoint_state_resume_infrastructure"] == "IMPLEMENTED"
    assert sem["checkpoint_state_resume_production"] == "SUPPORTED_BY_INTEGRATION_TEST"
    assert "not yet exercised at production scale" in sem["checkpoint_state_resume_evidence"].lower()
    assert sem["bit_exact_trajectory_resumable"] is False


def test_C_resumed_arm_records_broken_shuffle_continuity():
    st = R.mark_resumed(R.ArmState(
        experiment_id="CE1", block_id="b", arm="baseline", status="IN_PROGRESS",
        last_completed_epoch=10, terminal_epoch=200, model_seed=1, assignment_seed=2,
        assignment_flip=0, permutation_seed=3, config_fingerprint="fp",
        design_version="d"))
    assert st.shuffle_stream_continuity is False


def test_C_ledger_records_resume_with_semantics(tmp_path):
    led = L.CE1Ledger(tmp_path / "ledger.jsonl")
    led.record_resume(block_id="block0", arm="baseline", previous_config_hash="fp",
                      resumed_config_hash="fp", from_epoch=138)
    ev = led.events()[-1]
    assert ev["from_epoch"] == 138 and ev["shuffle_stream_continuity"] is False
    assert ev["resume_semantics"]["bit_exact_trajectory_resumable"] is False
    assert "minibatch_shuffle_rng" in ev["resume_semantics"]["not_restored"]


def test_C_ledger_refuses_config_change(tmp_path):
    led = L.CE1Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(L.ResumeViolation, match="configuration hash changed"):
        led.record_resume(block_id="b", arm="baseline", previous_config_hash="A",
                          resumed_config_hash="B", from_epoch=5)


# ---- D. completed work is skipped ----
def test_D_completed_arm_and_block_skipped(tmp_path):
    for arm in R.ARMS:
        _arm(tmp_path, "block0", arm, 200, status="EVALUATED")
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    assert R.plan_resume(rep, terminal_epoch=200)["plan"][0]["action"] == "SKIP"


# ---- E. legacy artifact validation ----
def _legacy(run, block, arm, epoch, *, with_eval=True, with_hash=True, extra=None):
    d = run / "blocks" / block / arm; d.mkdir(parents=True, exist_ok=True)
    mp = d / "model.keras"; mp.write_bytes(b"legacy-model")
    import hashlib
    meta = {"block_id": block, "arm": arm, "model_seed": 1000, "assignment_flip": 0,
            "sealed_evaluation_sha256": "s" * 64,
            "design_version": "CE-frozen-design-2026-03",
            "epochs_completed": epoch, "model_artifact": "model.keras"}
    if with_hash:
        meta["model_sha256"] = hashlib.sha256(mp.read_bytes()).hexdigest()
    if with_eval:
        meta["evaluation_accuracy"] = 0.93
    meta.update(extra or {})
    (d / "metadata.json").write_text(json.dumps(meta))


def test_E_legacy_complete_partial_invalid_unusable(tmp_path):
    _legacy(tmp_path, "block_00", "baseline", 200); _legacy(tmp_path, "block_00", "destroyed", 200)
    _legacy(tmp_path, "block_01", "baseline", 200); _legacy(tmp_path, "block_01", "destroyed", 137)
    _legacy(tmp_path, "block_02", "baseline", 200, with_hash=False)
    rep = LV.validate_legacy_run(tmp_path, n_blocks=4, terminal_epoch=200)
    b = rep["blocks"]
    assert b["block_00"]["block_status"] == LV.COMPLETE
    assert b["block_01"]["block_status"] == LV.PARTIAL
    assert b["block_02"]["block_status"] == LV.INVALID
    assert b["block_03"]["block_status"] == LV.UNUSABLE
    assert rep["complete_blocks"] == ["block_00"]
    assert rep["exact_resume_supported"] is False


def test_E_completion_not_inferred_from_filenames(tmp_path):
    d = tmp_path / "blocks" / "block_00" / "baseline"; d.mkdir(parents=True)
    (d / "epoch_200_FINAL_EPOCH.keras").write_bytes(b"x")     # suggestive name only
    rep = LV.validate_legacy_run(tmp_path, n_blocks=1, terminal_epoch=200)
    assert rep["blocks"]["block_00"]["arms"]["baseline"]["status"] == LV.UNUSABLE


def test_E_partial_legacy_says_rerun_not_resume(tmp_path):
    _legacy(tmp_path, "block_00", "baseline", 200); _legacy(tmp_path, "block_00", "destroyed", 50)
    rep = LV.validate_legacy_run(tmp_path, n_blocks=1, terminal_epoch=200)
    r = rep["blocks"]["block_00"]["arms"]["destroyed"]["reasons"]
    assert any("must be re-run" in x for x in r)
    assert "RERUN" in rep["blocks"]["block_00"]["action"]


# ---- F. tampering ----
def test_F_tampered_dataset_refused(tmp_path):
    X = np.zeros((8, 8), np.uint8); Y = np.zeros(8, np.uint8)
    m = DS.persist_block_dataset(tmp_path, block_id="b", role="train", X=X, Y=Y,
                                 rounds=5, differential=(0x40, 0))
    assert "path" not in m and not Path(m["filename"]).is_absolute()   # portable manifest
    np.savez(tmp_path / m["filename"], X=X + 1, Y=Y)
    with pytest.raises(DS.DatasetStateError, match="has been modified"):
        DS.load_block_dataset(tmp_path, block_id="b", role="train")


def test_F_tampered_checkpoint_refused(tmp_path):
    _arm(tmp_path, "block0", "baseline", 50)
    R.resolve_artifact(tmp_path, R.load_arm_state(tmp_path, "block0", "baseline").model_artifact).write_bytes(b"bad")
    assert any("corrupted" in r for r in
               R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)["refusal_reasons"])


def test_F_tampered_config_refused(tmp_path):
    _arm(tmp_path, "block0", "baseline", 50, config_fingerprint="OTHER")
    assert R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)["resume_refused"]


# ---- G. assignment preservation ----
def test_G_assignment_survives_restart_and_is_not_redrawn():
    from audit.cryptography.experiments.ce1.gohr_signal_destruction import seed_manifest
    a, b = seed_manifest(8), seed_manifest(8)
    assert a == b and a["assignment_seed"] == SPEC.assignment_seed


def test_G_mismatched_assignment_flip_refuses_resume(tmp_path):
    _arm(tmp_path, "block0", "baseline", 50, assignment_flip=1)
    exp = dict(EXPECTED, assignment_flip=0)
    assert any("assignment_flip" in r for r in
               R.scan_run(tmp_path, n_blocks=1, expected=exp)["refusal_reasons"])


# ---- H. serial vs parallel equivalence ----
def test_H_serial_and_parallel_plans_cover_identical_work(tmp_path):
    _arm(tmp_path, "block0", "baseline", 180); _arm(tmp_path, "block0", "destroyed", 100)
    rep = R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED)
    ser = R.plan_resume(rep, terminal_epoch=200, n_gpus=1)["plan"][0]["tasks"]
    par = R.plan_resume(rep, terminal_epoch=200, n_gpus=2)["plan"][0]["tasks"]
    strip = lambda ts: [{k: v for k, v in t.items() if k != "gpu"} for t in ts]
    assert strip(ser) == strip(par)          # identical work, only device differs


# ---- I. GPU isolation ----
def test_I_workers_get_distinct_gpus(tmp_path):
    _arm(tmp_path, "block0", "baseline", 10); _arm(tmp_path, "block0", "destroyed", 10)
    tasks = R.plan_resume(R.scan_run(tmp_path, n_blocks=1, expected=EXPECTED),
                          terminal_epoch=200, n_gpus=2)["plan"][0]["tasks"]
    gpus = [t["gpu"] for t in tasks if t["action"] == "TRAIN"]
    assert sorted(gpus) == [0, 1]


def test_I_parallelism_not_claimed_without_gpus():
    info = R.gpu_parallelism_available(2)
    if info["gpus_visible"] < 2:
        assert info["parallel_active"] is False and info["mode"] == "sequential_fallback"


# ---- J. mixed-state restart ----
def test_J_only_incomplete_arm_resumes(tmp_path):
    _arm(tmp_path, "block4", "baseline", 200, status="EVALUATED")
    _arm(tmp_path, "block4", "destroyed", 137)
    rep = R.scan_run(tmp_path, n_blocks=5, expected=EXPECTED)
    tasks = {t["arm"]: t for t in
             R.plan_resume(rep, terminal_epoch=200)["plan"][4]["tasks"]}
    assert tasks["baseline"]["action"] == "SKIP"
    assert tasks["destroyed"]["from_epoch"] == 138


# ---- K. mixed complete / partial / invalid directory ----
def test_K_controller_actions_for_mixed_run(tmp_path):
    _legacy(tmp_path, "block_00", "baseline", 200); _legacy(tmp_path, "block_00", "destroyed", 200)
    _legacy(tmp_path, "block_01", "baseline", 200); _legacy(tmp_path, "block_01", "destroyed", 90)
    _legacy(tmp_path, "block_02", "baseline", 200, with_eval=False)
    rep = LV.validate_legacy_run(tmp_path, n_blocks=8, terminal_epoch=200)
    assert rep["reusable"] == ["block_00"]
    assert set(rep["must_rerun"]) == {f"block_{i:02d}" for i in range(1, 8)}
    assert rep["n_complete"] < SPEC.min_valid_blocks     # controller must keep running


# ---- F(real Keras): atomic per-epoch save must produce a loadable checkpoint ----

def test_F_real_keras_checkpoint_roundtrip_with_optimizer(tmp_path):
    """Exercises the REAL save path: Keras 3 rejects non-.keras temp names."""
    tf = pytest.importorskip("tensorflow")
    from tensorflow import keras

    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(3),
                              keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="mse")
    X = np.random.default_rng(0).normal(size=(16, 4)); Y = (X[:, 0] > 0).astype("float32")
    model.fit(X, Y, epochs=1, verbose=0)          # give the optimizer real state

    st = _arm(tmp_path, "block0", "baseline", 0, status="NOT_STARTED")
    st = R.record_completed_epoch(tmp_path, st, epoch=1, model=model)

    assert not Path(st.model_artifact).is_absolute()          # run-relative, portable
    art = R.resolve_artifact(tmp_path, st.model_artifact)
    assert art.exists() and art.suffix == ".keras"
    assert not list(art.parent.glob("*.partial.keras"))      # no temp left behind
    assert st.model_sha256 == R.sha256_file(art)
    assert st.optimizer_state_included is True

    reloaded = keras.models.load_model(art)                  # must actually load
    assert reloaded.optimizer is not None
    for a, b in zip(model.get_weights(), reloaded.get_weights()):
        assert np.allclose(a, b)
