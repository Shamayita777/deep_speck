"""
CE1 controller integration tests on the REAL path:

    controller -> dataset persistence -> GohrTrainer -> model.fit(initial_epoch)
    -> per-epoch callback -> checkpoint -> state -> ledger -> restart -> evaluation

Scale is toy (tiny sample counts, few epochs) and runs on CPU. What these
tests establish is the correctness of the control path, not production
throughput. Physical dual-GPU execution is NOT tested here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

tf = pytest.importorskip("tensorflow", reason="real controller path needs TensorFlow")

from audit.cryptography.experiments.ce1 import controller as C          # noqa: E402
from audit.cryptography.experiments.ce1 import resume as R              # noqa: E402
from audit.cryptography.experiments.ce1 import ce1_ledger as L          # noqa: E402

REAL_LEGACY = REPO / "audit/cryptography/evidence_current/ce1/production_20260929"


# --------------------------------------------------------------------- helpers

def _bits(rng, n):
    return rng.integers(0, 2, (n, 64), dtype=np.uint8), rng.integers(0, 2, n, dtype=np.uint8)


def det_generate(seed=7):
    """Deterministic stand-in generator (tests only) so two runs can share data."""
    def gen(n_tr, n_va):
        r = np.random.default_rng(seed)
        return _bits(r, n_tr), _bits(r, n_va)
    return gen


def det_sealed(seed=99):
    return lambda n: _bits(np.random.default_rng(seed), n)


def speck_sealed(cfg):
    from audit.cryptography.gohr import speck as sp
    return lambda n: sp.make_train_data(n, cfg.rounds, diff=cfg.differential)


def cfg_(**kw):
    base = dict(n_blocks=1, min_valid_blocks=1, terminal_epoch=3, train_samples=64,
                validation_samples=64, evaluation_samples=128, batch_size=64, depth=1)
    base.update(kw)
    return C.CE1RunConfig.testing_config(**base)


def events(run, **match):
    return [e for e in C.ledger_for(run).events()
            if all(e.get(k) == v for k, v in match.items())]


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run_script(path: Path, *args, timeout=900):
    return subprocess.run([sys.executable, str(path), *map(str, args)], env=_env(),
                          capture_output=True, text=True, timeout=timeout)


# ===================================================================
# 1. REAL interruption: SIGKILL -> new process -> resume at N+1
# ===================================================================

PROC = textwrap.dedent('''
    import json, os, signal, sys
    from pathlib import Path
    import keras
    from audit.cryptography.experiments.ce1 import controller as C
    from audit.cryptography.gohr import speck as sp
    run, kill_at = Path(sys.argv[1]), int(sys.argv[2])
    cfg = C.CE1RunConfig.from_dict(json.loads((run.parent / "cfg.json").read_text()))
    sealed_gen = lambda n: sp.make_train_data(n, cfg.rounds, diff=cfg.differential)

    class Kill(keras.callbacks.Callback):
        """Placed AFTER the CE1 checkpoint callback: SIGKILL once epoch kill_at is durable."""
        def on_epoch_end(self, epoch, logs=None):
            if kill_at and epoch + 1 == kill_at:
                os.kill(os.getpid(), signal.SIGKILL)

    out = C.run_controller(run, cfg, sealed_generate_fn=sealed_gen,
                           extra_callbacks=[Kill()] if kill_at else None)
    print("RESULT" + json.dumps([b["status"] for b in out["blocks"]]))
''')


def test_real_sigkill_interruption_restart_resume(tmp_path):
    """
    Process 1: persist dataset, train baseline (REAL depth-10 GohrModel) to
    epoch 3, SIGKILL. Process 2: validate, restore model + optimizer, fit with
    initial_epoch=3, finish at 6, train the paired arm, evaluate both.
    """
    cfg = cfg_(depth=10, terminal_epoch=6, train_samples=256, batch_size=64)
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "cfg.json").write_text(json.dumps(cfg.to_dict()))
    script = tmp_path / "proc.py"
    script.write_text(PROC)

    p1 = _run_script(script, run, 3)
    assert p1.returncode == -9, p1.stdout[-2000:] + p1.stderr[-2000:]      # really killed

    st = R.load_arm_state(run, "block0", "baseline")
    assert st.status == R.IN_PROGRESS and st.last_completed_epoch == 3
    assert R.resolve_artifact(run, st.model_artifact).name == "epoch_003.keras"
    assert R.load_arm_state(run, "block0", "destroyed") is None
    commit_before = (run / "blocks/block0/data/dataset_commit.json").read_bytes()
    data_files = {p.name: p.stat().st_mtime_ns for p in (run / "blocks/block0/data").glob("*.npz")}
    assert [e["epoch"] for e in events(run, event="CHECKPOINT", arm="baseline")] == [1, 2, 3]
    plan = C.plan_block(run, cfg, 0, st.sealed_evaluation_sha256)
    assert plan["arms"]["baseline"]["action"] == C.RESUME

    p2 = _run_script(script, run, 0)
    assert p2.returncode == 0, p2.stdout[-3000:] + p2.stderr[-3000:]
    assert 'RESULT["VALID"]' in p2.stdout

    # dataset: loaded, never regenerated
    assert len(events(run, event="DATASET_PREPARED")) == 1
    assert (run / "blocks/block0/data/dataset_commit.json").read_bytes() == commit_before
    assert {p.name: p.stat().st_mtime_ns
            for p in (run / "blocks/block0/data").glob("*.npz")} == data_files
    # resume went through initial_epoch: 4..6 only, each epoch exactly once overall
    res = events(run, event="RESUME", arm="baseline")
    assert len(res) == 1 and res[0]["initial_epoch"] == 3 and res[0]["next_epoch"] == 4
    assert res[0]["optimizer_iterations"] == 3 * cfg.steps_per_epoch()
    assert [e["epoch"] for e in events(run, event="CHECKPOINT", arm="baseline")] == \
        [1, 2, 3, 4, 5, 6]
    assert not events(run, event="TRAIN_START", arm="baseline")[1:]    # one fresh start only
    b = R.load_arm_state(run, "block0", "baseline")
    assert [h["epoch"] for h in b.history] == [1, 2, 3, 4, 5, 6]
    assert b.shuffle_stream_continuity is False and b.resume_count == 1
    assert b.failure_count == 0                              # an interruption is not a failure
    # LR position restored: epoch 4 (0-based index 3) uses the frozen schedule value
    from audit.cryptography.gohr.trainer import GohrTrainer
    sched = GohrTrainer(checkpoint_dir=tmp_path / "x").learning_rate
    for h in b.history:
        assert abs(h["learning_rate"] - sched(h["epoch"] - 1)) <= 1e-6 * sched(h["epoch"] - 1)
    # optimizer state really continued: total steps == 6 epochs of steps
    import keras
    final = keras.models.load_model(R.resolve_artifact(run, b.terminal_model_artifact))
    assert int(final.optimizer.iterations.numpy()) == 6 * cfg.steps_per_epoch()
    # evaluation of both arms, block valid under the evidence gate
    assert {e["arm"] for e in events(run, event="EVALUATED")} == {"baseline", "destroyed"}
    sealed = C.ensure_sealed(run, cfg)
    gate = C.block_validity_gate(run, cfg, 0, sealed, final=True, deep=True)
    assert gate["status"] == "VALID" and all(gate["evidence"].values())
    assert gate["arms"]["baseline"]["resumed"] is True


# ===================================================================
# 2. baseline EVALUATED + destroyed interrupted at 137 -> SKIP / RESUME 138
# ===================================================================

class _Interrupt:
    @staticmethod
    def at(n):
        import keras

        class Stop(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch + 1 == n:
                    raise KeyboardInterrupt(f"simulated session end after epoch {n}")
        return Stop()


def test_controller_skip_evaluated_and_resume_at_138(tmp_path):
    cfg = cfg_(terminal_epoch=200)
    run = tmp_path / "run"
    C.open_run(run, cfg)
    sealed = C.ensure_sealed(run, cfg, generate_fn=speck_sealed(cfg))
    ctx = C.prepare_block(run, cfg, 0, sealed)
    C.train_arm(run, cfg, ctx, "baseline", C.TRAIN)
    C.evaluate_arm(run, cfg, "block0", "baseline", sealed)
    with pytest.raises(KeyboardInterrupt):
        C.train_arm(run, cfg, ctx, "destroyed", C.TRAIN, extra_callbacks=[_Interrupt.at(137)])
    d = R.load_arm_state(run, "block0", "destroyed")
    assert d.status == R.IN_PROGRESS and d.last_completed_epoch == 137
    assert d.failure_count == 0 and events(run, event="INTERRUPTED", arm="destroyed")
    base_hash = R.load_arm_state(run, "block0", "baseline").terminal_model_sha256

    plan = C.plan_block(run, cfg, 0, sealed["sha256"])
    assert plan["arms"]["baseline"]["action"] == C.SKIP
    assert plan["arms"]["destroyed"]["action"] == C.RESUME
    assert plan["arms"]["destroyed"]["reasons"] == ["resume at epoch 138"]

    n_before = len(C.ledger_for(run).events())
    out = C.run_controller(run, cfg)                          # the restart
    new = C.ledger_for(run).events()[n_before:]
    base_ev = [e["event"] for e in new if e.get("arm") == "baseline"]
    assert base_ev == ["SKIP"]                                # never retrained, never re-scored
    dest_ckpt = [e["epoch"] for e in new if e["event"] == "CHECKPOINT"]
    assert dest_ckpt == list(range(138, 201))
    res = [e for e in new if e["event"] == "RESUME"]
    assert res[0]["initial_epoch"] == 137 and res[0]["next_epoch"] == 138
    assert R.load_arm_state(run, "block0", "baseline").terminal_model_sha256 == base_hash
    assert out["blocks"][0]["status"] == "VALID"


# ===================================================================
# 3. TRAINING_COMPLETE without evaluation -> EVALUATE only
# ===================================================================

def test_training_complete_is_evaluated_not_retrained(tmp_path):
    cfg = cfg_()
    run = tmp_path / "run"
    C.open_run(run, cfg)
    sealed = C.ensure_sealed(run, cfg, generate_fn=speck_sealed(cfg))
    ctx = C.prepare_block(run, cfg, 0, sealed)
    for arm in ("baseline", "destroyed"):
        C.train_arm(run, cfg, ctx, arm, C.TRAIN)
    hashes = {a: R.load_arm_state(run, "block0", a).terminal_model_sha256
              for a in ("baseline", "destroyed")}
    plan = C.plan_block(run, cfg, 0, sealed["sha256"])
    assert {a: v["action"] for a, v in plan["arms"].items()} == \
        {"baseline": C.EVALUATE, "destroyed": C.EVALUATE}
    n_before = len(C.ledger_for(run).events())
    C.run_controller(run, cfg)
    new = [e["event"] for e in C.ledger_for(run).events()[n_before:]]
    for forbidden in ("TRAIN_START", "CHECKPOINT", "RESUME", "RERUN", "DATASET_PREPARED"):
        assert forbidden not in new
    assert new.count("EVALUATED") == 2
    for a, h in hashes.items():
        st = R.load_arm_state(run, "block0", a)
        assert st.status == R.EVALUATED and st.terminal_model_sha256 == h
        assert st.evaluation["model_sha256"] == h


def test_evaluation_waits_for_both_arms(tmp_path):
    """No single-arm outcome is produced while the sibling is unfinished."""
    cfg = cfg_()
    run = tmp_path / "run"
    C.open_run(run, cfg)
    sealed = C.ensure_sealed(run, cfg, generate_fn=speck_sealed(cfg))
    ctx = C.prepare_block(run, cfg, 0, sealed)
    C.train_arm(run, cfg, ctx, "baseline", C.TRAIN)
    with pytest.raises(KeyboardInterrupt):
        C.train_arm(run, cfg, ctx, "destroyed", C.TRAIN, extra_callbacks=[_Interrupt.at(1)])
    # a controller pass that is itself interrupted before destroyed finishes
    with pytest.raises(KeyboardInterrupt):
        C.run_controller(run, cfg, extra_callbacks=[_Interrupt.at(2)])
    assert not events(run, event="EVALUATED")
    assert R.load_arm_state(run, "block0", "baseline").status == R.TRAINING_COMPLETE


# ===================================================================
# 4. failure vs interruption accounting (predeclared, outcome-blind)
# ===================================================================

def test_exceptions_count_interruptions_do_not(tmp_path):
    import keras

    class Boom(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            raise RuntimeError("deterministic training fault")

    cfg = cfg_()
    run = tmp_path / "run"
    C.open_run(run, cfg)
    sealed = C.ensure_sealed(run, cfg, generate_fn=speck_sealed(cfg))
    ctx = C.prepare_block(run, cfg, 0, sealed)
    with pytest.raises(RuntimeError):
        C.train_arm(run, cfg, ctx, "baseline", C.TRAIN, extra_callbacks=[Boom()])
    st = R.load_arm_state(run, "block0", "baseline")
    assert st.status == R.FAILED and st.failure_count == 1 and st.last_completed_epoch == 1
    assert C.decide_arm(run, cfg, "block0", "baseline", sealed["sha256"])[0] == C.RESUME
    with pytest.raises(KeyboardInterrupt):
        C.train_arm(run, cfg, ctx, "baseline", C.RESUME, extra_callbacks=[_Interrupt.at(2)])
    assert R.load_arm_state(run, "block0", "baseline").failure_count == 1   # unchanged
    # reaching the OPERATIONAL cap pauses the arm; it is NOT a scientific failure
    st = R.load_arm_state(run, "block0", "baseline")
    st.failure_count = C.MAX_ARM_FAILURES
    R.save_arm_state(run, st)
    d = C.decide_arm_detail(run, cfg, "block0", "baseline", sealed)
    assert d["action"] == C.PAUSE and d["code"] == "E-RETRY-CAP"
    gate = C.block_validity_gate(run, cfg, 0, sealed, final=True)
    assert gate["status"] == "NEEDS_OPERATOR" and gate["code"] == "E-RETRY-CAP"
    assert "NOT_COUNTED" not in json.dumps(gate)          # never consumes a failure slot
    # a recorded operator grant re-arms the cap; the arm then simply resumes
    C.operator_action(run, cfg, 0, "GRANT_RETRY", arm="baseline", reason="OOM on shared host")
    assert C.decide_arm(run, cfg, "block0", "baseline", sealed)[0] == C.RESUME
    assert events(run, event="OPERATOR_ACTION")[-1]["action"] == "GRANT_RETRY"
    with pytest.raises(C.ControllerError, match="not paused by a retry cap"):
        C.operator_action(run, cfg, 0, "GRANT_RETRY", arm="baseline", reason="again")


# ===================================================================
# 5. legacy block0 recovery (synthetic legacy layout, real Keras models)
# ===================================================================

def _make_legacy(tmp_path, cfg):
    """Legacy layout with REAL trained models; bestval files are garbage on purpose."""
    from audit.cryptography.gohr.trainer import GohrTrainer
    from audit.cryptography.sealed_dataset import prepare_sealed_evaluation_set
    legacy = tmp_path / "legacy"
    ck = legacy / "checkpoints"
    ck.mkdir(parents=True)
    sealed = prepare_sealed_evaluation_set(legacy / "sealed", generate_fn=det_sealed(),
                                           rounds=cfg.rounds, differential=cfg.differential,
                                           n_samples=cfg.evaluation_samples)
    seeds = C.seeds_for(cfg)["seeds"]["block0"]
    X, Y = _bits(np.random.default_rng(3), cfg.train_samples)
    models = {}
    for arm in ("baseline", "destroyed"):                    # legacy build order
        GohrTrainer.set_seed(seeds[arm])
        models[arm] = C.build_model(cfg)
    for arm, m in models.items():
        m.fit(X, Y, epochs=cfg.terminal_epoch, batch_size=cfg.batch_size, verbose=0)
        m.save(str(ck / f"block0_{arm}_FINAL_EPOCH.keras"))
        (ck / f"block0_{arm}_bestval_DEBUG_ONLY.keras").write_bytes(b"NOT A MODEL")
    return legacy, sealed["sha256"]


def test_legacy_block0_recovery_is_evaluated_but_not_counted(tmp_path):
    from audit.cryptography.experiments.ce1 import legacy_recovery as LR
    cfg0 = cfg_(n_blocks=2)
    legacy, sealed_sha = _make_legacy(tmp_path, cfg0)
    cfg = cfg_(n_blocks=2, expected_sealed_sha256=sealed_sha)
    originals = {p.name: R.sha256_file(p) for p in (legacy / "checkpoints").iterdir()}

    run = tmp_path / "run"
    C.open_run(run, cfg, legacy_block0_decision="RECOVER")
    with pytest.raises(C.ControllerError, match="has not been seeded"):
        C.run_controller(run, cfg, sealed_source=legacy / "sealed")   # never trains block0
    rep = LR.verify_legacy_block0(legacy, cfg)
    assert rep["verified"] and rep["accuracy_computed"] is False
    LR.recover_legacy_block0(legacy, run, cfg)
    sealed = C.ensure_sealed(run, cfg)
    assert C.block_validity_gate(run, cfg, 0, sealed)["status"] == \
        "LEGACY_RECOVERED_PENDING_EVALUATION"
    with pytest.raises(LR.LegacyRecoveryError):
        LR.recover_legacy_block0(legacy, run, cfg)            # once only

    # corrupting a recovered copy can never lead to retraining a legacy arm
    probe = tmp_path / "probe"
    shutil.copytree(run, probe)
    (probe / "blocks/block0/destroyed/FINAL_EPOCH.keras").write_bytes(b"x")
    d = C.decide_arm_detail(probe, cfg, "block0", "destroyed", sealed)
    assert d["action"] == C.PAUSE and d["code"] == "E-TERMINAL-ARTIFACT"

    n_before = len(C.ledger_for(run).events())
    C.run_controller(run, cfg, blocks=[0])
    new = [e["event"] for e in C.ledger_for(run).events()[n_before:]]
    assert "TRAIN_START" not in new and "DATASET_PREPARED" not in new
    assert new.count("EVALUATED") == 2 and "BLOCK_LEGACY_EVALUATED" in new
    for arm in ("baseline", "destroyed"):
        st = R.load_arm_state(run, "block0", arm)
        assert st.evaluation["model_sha256"] == originals[f"block0_{arm}_FINAL_EPOCH.keras"]
        assert st.evaluation["sealed_evaluation_sha256"] == sealed_sha
        assert C.verify_evaluation(run, cfg, st, sealed) == []
        rec = json.loads((R.arm_dir(run, "block0", arm) / "legacy_recovery.json").read_text())
        assert rec["source_sha256"] == originals[f"block0_{arm}_FINAL_EPOCH.keras"]
        assert rec["sealed_evaluation_sha256"] == sealed_sha
        assert rec["status"].startswith("LEGACY_RECOVERED")
    assert C.block_validity_gate(run, cfg, 0, sealed)["status"] == "LEGACY_RECOVERED_EVALUATED"
    final = C.block_validity_gate(run, cfg, 0, sealed, final=True, deep=True)
    assert final["status"] == "NOT_COUNTED" and final["code"] == "N-INSUFFICIENT-EVIDENCE"
    assert final["counted"] is False
    ev = final["evidence"]
    assert ev["evaluation_bound"] and ev["terminal_epoch"] and ev["assignment"]
    assert ev["shared_training_inputs"] is None and ev["label_construction"] is None
    # original legacy artifacts untouched (bestval garbage never read)
    assert {p.name: R.sha256_file(p) for p in (legacy / "checkpoints").iterdir()} == originals


@pytest.mark.skipif(not (REAL_LEGACY / "checkpoints").exists(), reason="legacy run absent")
def test_real_legacy_block0_verifies_read_only_without_accuracy():
    from audit.cryptography.experiments.ce1 import legacy_recovery as LR
    before = {p.name: R.sha256_file(p) for p in (REAL_LEGACY / "checkpoints").iterdir()}
    rep = LR.verify_legacy_block0(REAL_LEGACY, C.CE1RunConfig.production_config())
    assert rep["verified"], rep["problems"]
    assert rep["accuracy_computed"] is False
    for arm in ("baseline", "destroyed"):
        assert rep["arms"][arm]["optimizer_iterations"] == 200 * 2000
        assert rep["arms"][arm]["n_add"] == 10
    assert rep["checks"][-1]["consistent"] is True               # build-order arm binding
    assert {p.name: R.sha256_file(p) for p in (REAL_LEGACY / "checkpoints").iterdir()} == before


# ===================================================================
# 6. frozen counting rule at finalize (no early stop, all valid blocks)
# ===================================================================

def _fake_keras(path, iterations):
    """Minimal `.keras` zip whose model.weights.h5 carries optimizer/vars/0."""
    import io
    import zipfile

    import h5py
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h:
        h.create_dataset("optimizer/vars/0", data=np.int64(iterations))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("model.weights.h5", buf.getvalue())
        z.writestr("tag.txt", str(path))


def _fake_evaluated_block(run, cfg, sealed, i, acc=(0.9, 0.5), wrong_labels=False,
                          iterations=None):
    """Real committed dataset + REAL evaluation binding path; the model files are
    stand-ins, and predictions are synthesised to hit the requested accuracy."""
    ctx = C.prepare_block(run, cfg, i, sealed, generate_fn=det_generate(100 + i))
    Y = sealed["Y"]
    for arm, a in zip(("baseline", "destroyed"), acc):
        st = C.new_arm_state(cfg, ctx, arm)
        if wrong_labels and arm == "destroyed":
            st.train_labels_sha256 = "0" * 64
        adir = R.arm_dir(run, ctx.block_id, arm)
        adir.mkdir(parents=True, exist_ok=True)
        f = adir / "FINAL_EPOCH.keras"
        _fake_keras(f, cfg.terminal_epoch * cfg.steps_per_epoch() if iterations is None
                    else iterations)
        st.status, st.last_completed_epoch = R.TRAINING_COMPLETE, cfg.terminal_epoch
        st.history = [{"epoch": e} for e in range(1, cfg.terminal_epoch + 1)]
        st.model_artifact = st.terminal_model_artifact = R.rel_to_run(run, f)
        st.model_sha256 = st.terminal_model_sha256 = R.sha256_file(f)
        k = int(round(a * Y.shape[0]))
        preds = np.where(np.arange(Y.shape[0]) < k, Y, 1 - Y).astype(np.float32)
        C.persist_evaluation(run, cfg, st, preds, sealed, evaluator_accuracy=None,
                             ledger=C.ledger_for(run))


def _gate_run(tmp_path, name):
    cfg = cfg_(n_blocks=8, min_valid_blocks=6, train_samples=32, validation_samples=16,
               evaluation_samples=16, batch_size=16)
    run = tmp_path / "audit/cryptography/evidence_current/ce1" / name
    C.open_run(run, cfg)
    sealed = C.ensure_sealed(run, cfg, generate_fn=det_sealed())
    return cfg, run, sealed


def test_finalize_refuses_while_blocks_pending_even_with_six_valid(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "a")
    for i in range(6):
        _fake_evaluated_block(run, cfg, sealed, i)
    with pytest.raises(C.ControllerError, match="no early stop"):
        C.finalize(run, cfg, repo_root=tmp_path)


def test_finalize_analyses_all_valid_blocks_and_reports_not_counted(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "b")
    for i in range(7):
        _fake_evaluated_block(run, cfg, sealed, i, acc=(0.9 - 0.0625 * (i % 3), 0.5))
    _fake_evaluated_block(run, cfg, sealed, 7, wrong_labels=True)   # permanent pairing break
    assert C.block_validity_gate(run, cfg, 7, sealed)["status"] == "PENDING_FINAL_GATE"
    path, cert = C.finalize(run, cfg, repo_root=tmp_path)
    r = cert["results"]
    assert r["valid_blocks_analysed"] == [f"block{i}" for i in range(7)]   # ALL valid, not 6
    assert r["n_blocks"] == 7 and r["failures"][0]["code"] == "N-PAIRING"


def test_retry_cap_never_consumes_a_failure_slot(tmp_path):
    """7 valid + 1 capped block: finalize REFUSES (unresolved), it does not certify on 7."""
    cfg, run, sealed = _gate_run(tmp_path, "cap")
    for i in range(7):
        _fake_evaluated_block(run, cfg, sealed, i, acc=(0.9 - 0.0625 * (i % 3), 0.5))
    ctx = C.prepare_block(run, cfg, 7, sealed, generate_fn=det_generate(200))
    st = C.new_arm_state(cfg, ctx, "baseline")
    st.failure_count = C.MAX_ARM_FAILURES
    R.save_arm_state(run, st)
    with pytest.raises(C.ControllerError, match="NEEDS_OPERATOR"):
        C.finalize(run, cfg, repo_root=tmp_path)
    assert not (run / "certificate.json").exists()


def test_insufficient_valid_blocks_yields_no_certificate(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "c")
    for i in range(5):
        _fake_evaluated_block(run, cfg, sealed, i, acc=(0.9 - 0.0625 * (i % 3), 0.5))
    for i in range(5, 8):
        _fake_evaluated_block(run, cfg, sealed, i, iterations=1)    # not the terminal model
    with pytest.raises(C.ControllerError, match="No inferential result"):
        C.finalize(run, cfg, repo_root=tmp_path)
    assert not (run / "certificate.json").exists()
    rep = json.loads((run / "resolution_report.json").read_text())
    assert rep["n_valid"] == 5 and rep["n_failed"] == 3
    assert {b["code"] for b in rep["blocks"] if b["status"] == "NOT_COUNTED"} == {"N-FINAL-EPOCH"}


def test_gate_is_blind_to_accuracies(tmp_path):
    """Swapping every accuracy value leaves every block resolution unchanged."""
    out = []
    for name, acc in (("hi", (0.99, 0.5)), ("lo", (0.5, 0.99))):
        cfg, run, sealed = _gate_run(tmp_path, name)
        for i in range(8):
            _fake_evaluated_block(run, cfg, sealed, i, acc=acc)
        out.append([C.block_validity_gate(run, cfg, i, sealed, final=True, deep=True)["status"]
                    for i in range(8)])
    assert out[0] == out[1] == ["VALID"] * 8


def test_no_operator_abandonment_exists(tmp_path):
    assert not hasattr(C, "abandon_block") and not hasattr(C, "REFUSE")
    assert "forbidden" in C.BLOCK_RESOLUTION_RULES["operator_actions"]
    assert C.BLOCK_RESOLUTION_RULES["caps_are_not_scientific_failures"] is True
    assert "accuracy" in C.BLOCK_RESOLUTION_RULES["never_used"]
    cfg, run, sealed = _gate_run(tmp_path, "op")
    for bad in ("FAIL_BLOCK", "ABANDON", "EXCLUDE"):
        with pytest.raises(C.ControllerError, match="unknown operator action"):
            C.operator_action(run, cfg, 0, bad, reason="x")
    with pytest.raises(C.ControllerError, match="written reason"):
        C.operator_action(run, cfg, 0, "RESTART_BLOCK", reason=" ")


# ===================================================================
# 7. dataset commit: fail closed, never silently regenerate
# ===================================================================

def test_committed_dataset_tamper_fails_closed(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "d")
    _fake_evaluated_block(run, cfg, sealed, 0)
    ddir = run / "blocks/block0/data"
    f = ddir / json.loads((ddir / "block0_train_manifest.json").read_text())["filename"]
    with np.load(f) as z:
        X, Y = z["X"], z["Y"]
    np.savez(f, X=X ^ 1, Y=Y)
    with pytest.raises(C.DS.DatasetStateError):
        C.load_block_context(run, cfg, 0, sealed)
    g = C.block_validity_gate(run, cfg, 0, sealed, deep=True)
    assert g["status"] == "NEEDS_OPERATOR" and g["code"] == "E-DATASET"     # not a failure
    f.unlink()
    g = C.block_validity_gate(run, cfg, 0, sealed)
    assert g["status"] == "NEEDS_OPERATOR" and g["code"] == "E-DATASET"
    with pytest.raises(C.DS.DatasetStateError):
        C.prepare_block(run, cfg, 0, sealed, generate_fn=det_generate())   # no regeneration


def test_uncommitted_partial_dataset_is_quarantined_not_reused(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "e")
    ddir = run / "blocks/block0/data"
    ddir.mkdir(parents=True)
    (ddir / "block0_train_manifest.json").write_text("{}")         # crash before commit
    ctx = C.prepare_block(run, cfg, 0, sealed, generate_fn=det_generate())
    assert ctx.hashes["train"]
    ev = events(run, event="DATASET_PREPARE_RESTARTED")
    assert len(ev) == 1 and (run / "blocks/block0" / ev[0]["quarantined_to"]).exists()


# ===================================================================
# 8. ledger durability
# ===================================================================

def test_malformed_ledger_fails_closed_and_torn_tail_needs_explicit_repair(tmp_path):
    led = L.CE1Ledger(tmp_path / "ledger.jsonl")
    led.record("RUN_START", block_id=None, arm=None, config_hash="x")
    with open(led.path, "a") as fh:
        fh.write('{"event": "CHECKPOINT", "epo')                    # killed mid-append
    with pytest.raises(L.LedgerCorrupted):
        led.events()
    assert led.repair_torn_tail()["n_bytes"] > 0
    assert [e["event"] for e in led.events()] == ["RUN_START", "LEDGER_REPAIR"]
    led.path.write_text('{"event": "RUN_START"}\nGARBAGE\n{"event": "SKIP"}\n')
    assert led.repair_torn_tail() is None                            # not a torn tail
    with pytest.raises(L.LedgerCorrupted):
        led.verify()
    with pytest.raises(ValueError):
        led.record("MADE_UP", block_id=None, arm=None, config_hash=None)


# ===================================================================
# 9. worker launcher: GPU gating, process isolation, serial/parallel equivalence
# ===================================================================

PARENT = textwrap.dedent('''
    import json, sys
    import numpy as np
    from pathlib import Path
    from audit.cryptography.experiments.ce1 import controller as C
    from audit.cryptography.experiments.ce1.worker import run_auto
    run = Path(sys.argv[1])
    cfg = C.CE1RunConfig.from_dict(json.loads((run.parent / "cfg.json").read_text()))
    def bits(r, n):
        return r.integers(0, 2, (n, 64), dtype=np.uint8), r.integers(0, 2, n, dtype=np.uint8)
    gen = lambda a, b: (lambda r: (bits(r, a), bits(r, b)))(np.random.default_rng(7))
    sealed = lambda n: bits(np.random.default_rng(99), n)
    out = run_auto(run, cfg, gpu_info={"gpus_visible": 0, "devices": []},
                   cpu_process_isolation=True, generate_fn=gen, sealed_generate_fn=sealed)
    print("RESULT" + json.dumps({"statuses": [b["status"] for b in out["blocks"]],
                                 "execution": out["execution"],
                                 "parent_imported_tensorflow": "tensorflow" in sys.modules,
                                 "parent_imported_keras": "keras" in sys.modules}))
''')


def test_gpu_detection_gates_mode_honestly():
    from audit.cryptography.experiments.ce1.worker import detect_gpus
    info = detect_gpus()
    n = info["gpus_visible"]
    assert info["mode"] == ("parallel" if n >= 2 else "sequential")


def test_parallel_workers_isolated_and_equivalent_to_serial(tmp_path):
    cfg = cfg_(terminal_epoch=3, train_samples=128, batch_size=64)
    # --- parallel path: two concurrent CPU worker processes (NOT a GPU test)
    par = tmp_path / "par" / "run"
    par.parent.mkdir()
    (par.parent / "cfg.json").write_text(json.dumps(cfg.to_dict()))
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    p = _run_script(script, par, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = json.loads(p.stdout.split("RESULT")[-1])
    assert res["statuses"] == ["VALID"]
    assert res["parent_imported_tensorflow"] is False and res["parent_imported_keras"] is False
    assert res["execution"]["mode"] == "parallel_workers"
    assert res["execution"]["cpu_process_isolation"] is True
    assert res["execution"]["physical_dual_gpu"] is False
    launches = events(par, event="WORKER_LAUNCH", task="train")
    assert {e["arm"] for e in launches} == {"baseline", "destroyed"}
    pids = {e["arm"]: {c["pid"] for c in events(par, event="CHECKPOINT", arm=e["arm"])}
            for e in launches}
    assert len(pids["baseline"]) == 1 and pids["baseline"].isdisjoint(pids["destroyed"])
    starts = [e["timestamp"] for e in events(par, event="TRAIN_START")]
    ends = [e["timestamp"] for e in events(par, event="TRAINING_COMPLETE")]
    assert max(starts) < min(ends)                          # the two arms really overlapped
    assert all(e["returncode"] == 0 for e in events(par, event="WORKER_EXIT"))

    # --- serial path, same deterministic data
    ser = tmp_path / "ser" / "run"
    C.run_controller(ser, cfg, generate_fn=det_generate(7), sealed_generate_fn=det_sealed(99))
    commit = lambda r: {k: v for k, v in json.loads(
        (r / "blocks/block0/data/dataset_commit.json").read_text()).items()
        if k != "committed_utc"}
    assert commit(ser) == commit(par)
    for arm in ("baseline", "destroyed"):
        a, b = R.load_arm_state(ser, "block0", arm), R.load_arm_state(par, "block0", arm)
        for k in ("model_seed", "assignment_flip", "permutation_seed", "train_data_sha256",
                  "train_labels_sha256", "validation_data_sha256", "sealed_evaluation_sha256",
                  "status", "last_completed_epoch", "config_fingerprint"):
            assert getattr(a, k) == getattr(b, k), k
        assert [h["epoch"] for h in a.history] == [h["epoch"] for h in b.history]
    # identical work; numerical identity of the trained weights is NOT claimed


# ===================================================================
# 10. path containment on LOAD
# ===================================================================

def test_dataset_manifest_paths_are_contained(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "pc")
    C.prepare_block(run, cfg, 0, sealed, generate_fn=det_generate())
    ddir = run / "blocks/block0/data"
    mpath = ddir / "block0_train_manifest.json"
    good = json.loads(mpath.read_text())
    outside = tmp_path / "outside.npz"
    shutil.copy(ddir / good["filename"], outside)
    for bad in (str(outside), "../../../outside.npz", "sub/../x.npz", "/etc/passwd"):
        mpath.write_text(json.dumps({**good, "filename": bad}))
        with pytest.raises(C.DS.PathContainmentError):
            C.DS.load_block_dataset(ddir, block_id="block0", role="train")
    legacy = {k: v for k, v in good.items() if k != "filename"}
    mpath.write_text(json.dumps({**legacy, "path": str(ddir / good["filename"])}))
    with pytest.raises(C.DS.PathContainmentError, match="absolute 'path'"):
        C.DS.load_block_dataset(ddir, block_id="block0", role="train")
    # symlink escape: a link inside the data dir pointing outside it
    link = ddir / "escape.npz"
    link.symlink_to(outside)
    mpath.write_text(json.dumps({**good, "filename": "escape.npz"}))
    with pytest.raises(C.DS.PathContainmentError, match="outside"):
        C.DS.load_block_dataset(ddir, block_id="block0", role="train")
    mpath.write_text(json.dumps(good))
    C.DS.load_block_dataset(ddir, block_id="block0", role="train")          # restored: fine


def test_artifact_paths_are_contained(tmp_path):
    run = tmp_path / "run"
    (run / "a").mkdir(parents=True)
    (run / "a" / "m.keras").write_bytes(b"x")
    (tmp_path / "evil.keras").write_bytes(b"x")
    (run / "a" / "link.keras").symlink_to(tmp_path / "evil.keras")
    assert R.resolve_artifact(run, "a/m.keras") == run / "a" / "m.keras"
    for bad in (str(tmp_path / "evil.keras"), "../evil.keras", "a/../../evil.keras",
                "a/link.keras"):
        with pytest.raises(R.ResumeRefused):
            R.resolve_artifact(run, bad)
    with pytest.raises(R.ResumeRefused):
        R.rel_to_run(run, tmp_path / "evil.keras")


def test_escaped_artifact_pauses_arm_instead_of_loading(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "esc")
    _fake_evaluated_block(run, cfg, sealed, 0)
    st = R.load_arm_state(run, "block0", "baseline")
    st.terminal_model_artifact = "../../../../outside.keras"
    R.save_arm_state(run, st)
    d = C.decide_arm_detail(run, cfg, "block0", "baseline", sealed)
    assert d["action"] == C.PAUSE and d["code"] == "E-TERMINAL-ARTIFACT"


# ===================================================================
# 11. controller protocol version is checked on reopen
# ===================================================================

def test_incompatible_controller_protocol_refused(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "proto")
    m = json.loads((run / C.RUN_MANIFEST).read_text())
    assert m["controller_protocol_version"] == C.CONTROLLER_PROTOCOL_VERSION
    for bad in ("ce1-controller-1", None):
        m2 = dict(m, controller_protocol_version=bad)
        R.atomic_write_json(run / C.RUN_MANIFEST, m2)
        with pytest.raises(R.ResumeRefused, match="incompatible protocol"):
            C.open_run(run, cfg)
    R.atomic_write_json(run / C.RUN_MANIFEST, dict(m, block_resolution_rules={"v": 1}))
    with pytest.raises(R.ResumeRefused, match="block-resolution rules"):
        C.open_run(run, cfg)


# ===================================================================
# 12. evaluation integrity: every tamper is detected and PAUSES (never counted)
# ===================================================================

def _tamper_case(tmp_path, name, mutate):
    cfg, run, sealed = _gate_run(tmp_path, name)
    _fake_evaluated_block(run, cfg, sealed, 0, acc=(0.75, 0.5))
    assert C.block_validity_gate(run, cfg, 0, sealed)["status"] == "VALID"
    mutate(run, R.arm_dir(run, "block0", "baseline"))
    g = C.block_validity_gate(run, cfg, 0, sealed, final=True)
    assert g["status"] == "NEEDS_OPERATOR", g
    return cfg, run, sealed, g


def _edit_eval(adir, **changes):
    p = adir / "evaluation.json"
    rec = json.loads(p.read_text()); rec.update(changes)
    p.write_text(json.dumps(rec))


@pytest.mark.parametrize("name,mutate", [
    ("acc", lambda run, a: _edit_eval(a, accuracy=0.99)),
    ("acc_and_count", lambda run, a: _edit_eval(a, accuracy=15 / 16, n_correct=15)),
    ("model", lambda run, a: _edit_eval(a, model_sha256="f" * 64)),
    ("sealed", lambda run, a: _edit_eval(a, sealed_evaluation_sha256="e" * 64)),
    ("fingerprint", lambda run, a: _edit_eval(a, config_fingerprint="d" * 64)),
    ("missing_eval", lambda run, a: (a / "evaluation.json").unlink()),
    ("missing_preds", lambda run, a: (a / "sealed_predictions.npz").unlink()),
    ("preds_changed", lambda run, a: np.savez(a / "sealed_predictions.npz",
                                              predictions=np.ones(16, np.float32))),
])
def test_evaluation_tamper_detected(tmp_path, name, mutate):
    _, _, _, g = _tamper_case(tmp_path, name, mutate)
    assert g["code"] in ("E-EVALUATION-INTEGRITY", "E-TERMINAL-ARTIFACT")


def test_consistent_forgery_still_caught_by_prediction_recount_and_ledger(tmp_path):
    """Rewrite evaluation.json AND the state copy coherently: recount/ledger still catch it."""
    def forge(run, adir):
        st = R.load_arm_state(run, "block0", "baseline")
        rec = dict(st.evaluation, accuracy=1.0, n_correct=16)
        (adir / "evaluation.json").write_text(json.dumps(rec))
        st.evaluation, st.evaluation_sha256 = rec, C.canonical_sha256(rec)
        R.save_arm_state(run, st)
    _, run, _, g = _tamper_case(tmp_path, "forge", forge)
    joined = " ".join(g["reasons"])
    assert "recomputed" in joined and "ledger" in joined


def test_prediction_count_mismatch_detected(tmp_path):
    def shrink(run, adir):
        np.savez(adir / "sealed_predictions.npz", predictions=np.ones(8, np.float32))
        st = R.load_arm_state(run, "block0", "baseline")
        rec = dict(st.evaluation, predictions_sha256=R.sha256_file(adir / "sealed_predictions.npz"))
        (adir / "evaluation.json").write_text(json.dumps(rec))
        st.evaluation, st.evaluation_sha256 = rec, C.canonical_sha256(rec)
        R.save_arm_state(run, st)
    _, _, _, g = _tamper_case(tmp_path, "count", shrink)
    assert any("sealed count" in r for r in g["reasons"])


def test_tampered_accuracy_never_reaches_certificate(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "cert")
    for i in range(8):
        _fake_evaluated_block(run, cfg, sealed, i, acc=(0.9 - 0.0625 * (i % 3), 0.5))
    _edit_eval(R.arm_dir(run, "block3", "baseline"), accuracy=0.123)
    with pytest.raises(C.ControllerError, match="unresolved"):
        C.finalize(run, cfg, repo_root=tmp_path)
    assert not (run / "certificate.json").exists()


def test_reevaluate_restores_only_the_bound_truth(tmp_path):
    """Operator REEVALUATE on a REAL model: agrees with the ledger -> resolved."""
    cfg = cfg_()
    run = tmp_path / "run"
    C.run_controller(run, cfg, sealed_generate_fn=speck_sealed(cfg))
    sealed = C.ensure_sealed(run, cfg)
    adir = R.arm_dir(run, "block0", "baseline")
    true_rec = json.loads((adir / "evaluation.json").read_text())
    _edit_eval(adir, accuracy=0.999)
    assert C.block_validity_gate(run, cfg, 0, sealed)["code"] == "E-EVALUATION-INTEGRITY"
    with pytest.raises(C.ControllerError, match="not permitted"):
        C.operator_action(run, cfg, 0, "REEVALUATE", arm="destroyed", reason="intact arm")
    out = C.operator_action(run, cfg, 0, "REEVALUATE", arm="baseline",
                            reason="evaluation.json failed verification")
    assert out["remaining_problems"] == []
    rec = json.loads((adir / "evaluation.json").read_text())
    assert rec["n_correct"] == true_rec["n_correct"] and rec["accuracy"] == true_rec["accuracy"]
    assert C.block_validity_gate(run, cfg, 0, sealed)["status"] == "VALID"
    # RESTART_BLOCK is refused once anything in the block was evaluated
    with pytest.raises(C.ControllerError, match="outcome-dependent"):
        C.operator_action(run, cfg, 0, "RESTART_BLOCK", reason="try again")


def test_conflicting_ledger_evaluations_stay_unresolved(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "conf")
    _fake_evaluated_block(run, cfg, sealed, 0, acc=(0.75, 0.5))
    st = R.load_arm_state(run, "block0", "baseline")
    Y = sealed["Y"]
    preds = np.where(np.arange(Y.shape[0]) < 14, Y, 1 - Y).astype(np.float32)
    st.status = R.TRAINING_COMPLETE
    C.persist_evaluation(run, cfg, st, preds, sealed, evaluator_accuracy=None,
                         ledger=C.ledger_for(run))            # a DIFFERENT n_correct, same model
    g = C.block_validity_gate(run, cfg, 0, sealed)
    assert g["status"] == "NEEDS_OPERATOR" and any("conflicting" in r for r in g["reasons"])


def test_restart_block_allowed_only_before_any_evaluation(tmp_path):
    cfg, run, sealed = _gate_run(tmp_path, "rs")
    C.prepare_block(run, cfg, 2, sealed, generate_fn=det_generate())
    out = C.operator_action(run, cfg, 2, "RESTART_BLOCK", reason="committed data lost")
    assert out["action"] == "RESTART_BLOCK" and not (run / "blocks/block2").exists()


# ===================================================================
# 13. CE2/CE3/CE4 reference checkpoint resolves from the source tree
# ===================================================================

def test_ce2_ce3_ce4_checkpoint_independent_of_cwd(tmp_path, monkeypatch):
    from audit.cryptography import audit_config
    from audit.cryptography.integrity import verify_reference_checkpoint
    monkeypatch.chdir(tmp_path)                                 # hostile cwd
    import importlib
    for mod in ("audit.cryptography.experiments.ce2.gohr_theory_consistency",
                "audit.cryptography.experiments.ce3.gohr_representation_interpretation",
                "audit.cryptography.experiments.ce4.gohr_casual_intervention"):
        m = importlib.reload(importlib.import_module(mod))
        assert m.REFERENCE_CHECKPOINT.is_absolute()
        assert m.REFERENCE_CHECKPOINT == ROOT / "Archive" / "best5depth10.h5"
        assert m.REFERENCE_CHECKPOINT.exists()
        src = Path(m.__file__).read_text()
        assert 'Path("Archive' not in src
    assert audit_config.REFERENCE_CHECKPOINT_PATH == ROOT / "Archive" / "best5depth10.h5"
    rep = verify_reference_checkpoint(audit_config.REFERENCE_CHECKPOINT_PATH)
    assert rep                                                     # hash-gated, depth 10


def test_canonical_evidence_root_only():
    assert (ROOT / "evidence_current").is_dir()
    assert not (REPO / "evidence_current").exists(), "root-level evidence_current must not exist"


@pytest.mark.skipif(not (REAL_LEGACY / "checkpoints").exists(), reason="legacy run absent")
def test_legacy_validator_wording_matches_keras3_reality():
    """Legacy terminal artifacts DO hold optimizer state; what is missing is data + per-epoch state."""
    from audit.cryptography.experiments.ce1 import legacy_validator as LV
    rep = LV.validate_legacy_checkpoint_run(REAL_LEGACY, n_blocks=8)
    for arm in rep["blocks"]["block0"]["arms"].values():
        assert arm["optimizer_state"] is True
    text = json.dumps(rep).lower()
    assert "no optimizer state" not in text and "carry no optimizer" not in text
    assert rep["exact_resume_reason"] == LV.LEGACY_NON_RESUMABLE_REASON
    assert "optimizer state may exist inside the saved keras artifact" in text
    assert "not provable" in text


# ===================================================================
# 14. working directory: wrong cwd fails closed, never redirects output
# ===================================================================

def test_wrong_cwd_fails_closed_for_implicit_output_root(tmp_path, monkeypatch):
    from audit.cryptography import output_policy as OP
    OP.assert_audit_output_path("audit/cryptography/evidence_current/x.json")   # repo root: ok
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audit/cryptography/evidence_current").mkdir(parents=True)     # decoy tree
    with pytest.raises(OP.WrongWorkingDirectoryError, match="repository root"):
        OP.assert_audit_output_path("audit/cryptography/evidence_current/x.json")
    # an EXPLICIT repo_root remains an explicit, visible choice (used by tests)
    OP.assert_audit_output_path("audit/cryptography/evidence_current/x.json",
                                repo_root=tmp_path)


def test_ce1_cli_from_wrong_cwd_writes_nothing(tmp_path):
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    p = subprocess.run([sys.executable, "-m",
                        "audit.cryptography.experiments.ce1.gohr_signal_destruction",
                        "--recover-legacy-block0",
                        "--legacy-run-dir", str(REAL_LEGACY),
                        "--sealed-source", str(REAL_LEGACY / "sealed")],
                       cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=300)
    assert p.returncode != 0
    assert "repository root" in (p.stdout + p.stderr)
    assert sorted(p_.relative_to(tmp_path) for p_ in tmp_path.rglob("*")) == before
