"""
CE1 production wiring: the real Gohr Dataset -> Model -> Trainer ->
Evaluator -> CE1 design -> statistics -> certificate chain.

The smoke test uses tiny sample counts and 1-2 epochs. Its purpose is
STRUCTURAL ONLY: prove the real components interoperate and that the
scientific invariants hold. No accuracy produced here means anything.
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from audit.cryptography.audit_config import REFERENCE
from audit.cryptography.experiments.ce1 import gohr_signal_destruction as ce1_driver
from audit.cryptography.preflight import FrozenParameterMissing

tf = pytest.importorskip("tensorflow", reason="real Gohr chain needs TensorFlow")


# --------------------------------------------------------------------
# production configuration
# --------------------------------------------------------------------

def test_production_uses_reference_configuration():
    assert REFERENCE.rounds == 5
    assert REFERENCE.differential == (0x0040, 0x0000)
    assert REFERENCE.depth == 10
    assert REFERENCE.l2_reg == 1e-5


def test_production_overrides_the_dangerous_constructor_defaults():
    """
    GohrDataset defaults to rounds=7 and GohrModel to depth=5. Production
    must pass both explicitly - relying on either default is the exact
    error class this audit exists to detect.
    """
    from audit.cryptography.gohr.dataset import GohrDataset
    from audit.cryptography.gohr.model import GohrModel
    assert inspect.signature(GohrDataset).parameters["rounds"].default == 7
    assert inspect.signature(GohrModel).parameters["depth"].default == 5

    from audit.cryptography.experiments.ce1 import controller as C
    cfg = C.CE1RunConfig.production_config()
    assert (cfg.rounds, tuple(cfg.differential), cfg.depth, cfg.l2_reg) == \
        (REFERENCE.rounds, tuple(REFERENCE.differential), REFERENCE.depth, REFERENCE.l2_reg)
    gen_src = inspect.getsource(C.default_generate_fn)
    assert "rounds=config.rounds" in gen_src and "differential=" in gen_src
    model_src = inspect.getsource(C.build_model)
    assert "depth=config.depth" in model_src and "regularization=config.l2_reg" in model_src
    # the closure path that carried data_fn.last_validation no longer exists
    assert not hasattr(ce1_driver, "build_production_components")


def test_production_model_actually_realizes_depth_10_and_l2():
    from audit.cryptography.gohr.model import GohrModel
    model = GohrModel(depth=REFERENCE.depth, regularization=REFERENCE.l2_reg).build()
    adds = sum(1 for l in model.layers if l.__class__.__name__ == "Add")
    convs = sum(1 for l in model.layers if l.__class__.__name__ == "Conv1D")
    assert adds == 10 and convs == 1 + 2 * 10
    l2 = {round(float(l.kernel_regularizer.l2), 12) for l in model.layers
          if getattr(l, "kernel_regularizer", None) is not None}
    assert l2 == {1e-5}


def test_production_uses_frozen_block_counts_without_cli_flags(tmp_path, capsys):
    """n_blocks/min_valid are FROZEN constants, not operator-supplied values."""
    rc = ce1_driver.main(["--preflight", "--output",
                          str(tmp_path / "audit/cryptography/evidence_current/ce1/c.json"),
                          "--repo-root", str(tmp_path)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frozen_parameters"]["n_blocks"] == 8
    assert report["frozen_parameters"]["min_valid_blocks"] == 6


def test_preflight_mode_runs_no_training(tmp_path, capsys):
    rc = ce1_driver.main(["--preflight",
                          "--output", str(tmp_path / "audit/cryptography/evidence_current/ce1/c.json"),
                          "--repo-root", str(tmp_path)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PREFLIGHT_OK"
    fp = report["frozen_parameters"]
    assert fp["n_blocks"] == 8 and fp["min_valid_blocks"] == 6
    assert fp["checkpoint_rule"] == "FINAL_EPOCH" and fp["terminal_epoch"] == 200
    assert fp["seed_base"] == 1000 and fp["evaluation_samples"] == 10 ** 6
    assert not (tmp_path / "audit/cryptography/evidence_current").exists()      # nothing written


# --------------------------------------------------------------------
# real component chain, tiny
# --------------------------------------------------------------------

def _toy_controller_config(**kw):
    from audit.cryptography.experiments.ce1 import controller as C
    base = dict(n_blocks=2, min_valid_blocks=2, terminal_epoch=2, train_samples=256,
                validation_samples=64, evaluation_samples=64, batch_size=32, depth=1)
    base.update(kw)
    return C.CE1RunConfig.testing_config(**base)


def _sealed_gen(cfg):
    from audit.cryptography.gohr import speck as sp
    return lambda n: sp.make_train_data(n, cfg.rounds, diff=cfg.differential)


def test_real_gohr_chain_smoke(tmp_path):
    """Controller -> persisted BlockContext -> GohrTrainer -> evaluator, on real Speck data."""
    from audit.cryptography.experiments.ce1 import controller as C
    cfg = _toy_controller_config()
    run = tmp_path / "run"
    C.open_run(run, cfg)
    sealed = C.ensure_sealed(run, cfg, generate_fn=_sealed_gen(cfg))
    ctx0 = C.prepare_block(run, cfg, 0, sealed)
    ctx1 = C.prepare_block(run, cfg, 1, sealed)
    # F1: every block receives the IDENTICAL sealed evaluation arrays
    assert np.array_equal(ctx0.X_evaluation, ctx1.X_evaluation)
    assert ctx0.hashes["sealed"] == ctx1.hashes["sealed"] == sealed["sha256"]
    # real Speck data at the reference configuration, persisted before training
    assert ctx0.X_train.shape == (256, 64) and ctx0.Y_train_baseline.shape == (256,)
    assert ctx0.X_validation.shape == (64, 64)
    assert set(np.unique(ctx0.Y_train_baseline)) <= {0, 1}
    # the two arms share inputs and validation; only labels differ
    assert ctx0.training_pair("baseline")[0] is ctx0.training_pair("destroyed")[0]
    assert ctx0.validation_pair("baseline")[1] is ctx0.validation_pair("destroyed")[1]
    assert not np.array_equal(ctx0.Y_train_baseline, ctx0.Y_train_destroyed)
    # a second prepare reloads the SAME committed data (never regenerates)
    again = C.prepare_block(run, cfg, 0, sealed)
    assert again.hashes == ctx0.hashes
    st = C.train_arm(run, cfg, ctx0, "baseline", C.TRAIN)
    assert st.status == "TRAINING_COMPLETE" and st.last_completed_epoch == 2
    st = C.evaluate_arm(run, cfg, "block0", "baseline", sealed)
    assert st.status == "EVALUATED" and 0.0 <= st.evaluation["accuracy"] <= 1.0


def test_real_chain_end_to_end_two_blocks(tmp_path):
    """Full controller pass + finalize on the REAL components, 2 blocks."""
    from audit.cryptography.experiments.ce1 import controller as C
    cfg = _toy_controller_config()
    run = tmp_path / "audit/cryptography/evidence_current/ce1/run"
    out = C.run_controller(run, cfg, sealed_generate_fn=_sealed_gen(cfg))
    assert [b["status"] for b in out["blocks"]] == ["VALID", "VALID"]
    path, cert = C.finalize(run, cfg, repo_root=tmp_path)
    loaded = json.loads(Path(path).read_text())
    r = loaded["results"]
    assert r["n_blocks"] == 2 and r["valid_blocks_analysed"] == ["block0", "block1"]
    for b in r["blocks"]:
        ev = b["evidence"]
        assert ev["shared_training_inputs"] and ev["label_construction"]
        assert ev["shared_validation"] and ev["evaluation_bound"]
    assert loaded["reference_configuration"]["depth"] == 10
    assert loaded["non_evidentiary"] is True
    assert "evidence_current" in str(path)


def test_two_arms_are_independent_models(tmp_path, monkeypatch):
    """Each arm must get its own freshly built model and its own seed."""
    seen = []
    monkeypatch.setattr(ce1_driver, "PRODUCTION_TRAINING",
                        {**ce1_driver.PRODUCTION_TRAINING, "batch_size": 32, "epochs": 2})

    def spy(X_train, Y_train, X_eval, Y_eval, *, seed, arm, block_id):
        seen.append((block_id, arm, seed, id(Y_train)))
        return 0.5 + 0.01 * len(seen)

    def data_fn(i):
        r = np.random.default_rng(i)
        return (r.integers(0, 2, (32, 64), dtype=np.uint8), r.integers(0, 2, 32, dtype=np.uint8),
                r.integers(0, 2, (16, 64), dtype=np.uint8), r.integers(0, 2, 16, dtype=np.uint8))

    ce1_driver.run(n_blocks=2, min_valid_blocks=2, production=False,
                   output_path=tmp_path / "audit/cryptography/evidence_current" / "ce1" / "c.json",
                   train_eval_fn=spy, data_fn=data_fn, repo_root=tmp_path,
                   seed_base=100)
    assert [s[1] for s in seen] == ["baseline", "destroyed"] * 2
    assert len({s[2] for s in seen}) == 4                    # four distinct seeds
    # within a block the two arms received DIFFERENT label arrays
    assert seen[0][3] != seen[1][3]


def test_production_cannot_write_into_historical_evidence(tmp_path):
    from audit.cryptography.output_policy import HistoricalWriteError
    def data_fn(i):
        r = np.random.default_rng(i)
        return (r.integers(0, 2, (16, 8), dtype=np.uint8), r.integers(0, 2, 16, dtype=np.uint8),
                r.integers(0, 2, (8, 8), dtype=np.uint8), r.integers(0, 2, 8, dtype=np.uint8))
    with pytest.raises(HistoricalWriteError):
        ce1_driver.run(n_blocks=2, min_valid_blocks=2,                        output_path=tmp_path / "evidence" / "ce1" / "ce1_certificate.json",
                       train_eval_fn=lambda *a, **k: 0.5, data_fn=data_fn,
                       repo_root=tmp_path, production=False)


# --------------------------------------------------------------------
# Learning-rate schedule: reference conformance
#
# The reference (train_nets.py) uses cyclic_lr(10, 0.002, 0.0001):
#     lr(i) = low + ((C-1) - i % C) / (C-1) * (high - low),  C = 10
# a sawtooth resetting to the high LR every 10 epochs. GohrTrainer
# previously used self.epochs as C, collapsing 20 cycles into one ramp.
# --------------------------------------------------------------------

REFERENCE_LR_CYCLE = 10
REFERENCE_HIGH_LR = 0.002
REFERENCE_LOW_LR = 0.0001


def _reference_cyclic_lr(num_epochs, high_lr, low_lr):
    """Verbatim reference schedule from train_nets.py."""
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (
        high_lr - low_lr)


def _production_trainer(tmp_path, epochs=200):
    from audit.cryptography.gohr.trainer import GohrTrainer
    return GohrTrainer(epochs=epochs, checkpoint_dir=tmp_path / "ckpt",
                       high_learning_rate=REFERENCE_HIGH_LR,
                       low_learning_rate=REFERENCE_LOW_LR)


def test_trainer_uses_reference_cycle_length_of_ten(tmp_path):
    t = _production_trainer(tmp_path)
    assert t.lr_cycle_length == REFERENCE_LR_CYCLE
    # the schedule must RESET every 10 epochs, independently of total epochs
    for start in (0, 10, 20, 190):
        assert t.learning_rate(start) == pytest.approx(REFERENCE_HIGH_LR)


def test_lr_matches_reference_at_every_production_epoch(tmp_path):
    t = _production_trainer(tmp_path, epochs=200)
    ref = _reference_cyclic_lr(REFERENCE_LR_CYCLE, REFERENCE_HIGH_LR, REFERENCE_LOW_LR)
    for i in range(200):
        assert t.learning_rate(i) == pytest.approx(ref(i), rel=0, abs=1e-15)


@pytest.mark.parametrize("epoch,expected", [
    (0, 0.002), (1, 0.001788888888888889), (9, 0.0001),
    (10, 0.002), (19, 0.0001), (20, 0.002),
])
def test_lr_representative_epoch_values(tmp_path, epoch, expected):
    assert _production_trainer(tmp_path).learning_rate(epoch) == pytest.approx(expected)


def test_lr_cycle_is_independent_of_total_epoch_count(tmp_path):
    """
    Regression guard: production must not silently revert to using the
    total epoch count as the cycle length. Trainers configured with very
    different epoch budgets must produce the IDENTICAL schedule.
    """
    a = _production_trainer(tmp_path / "a", epochs=200)
    b = _production_trainer(tmp_path / "b", epochs=50)
    c = _production_trainer(tmp_path / "c", epochs=10)
    for i in range(40):
        assert a.learning_rate(i) == pytest.approx(b.learning_rate(i))
        assert a.learning_rate(i) == pytest.approx(c.learning_rate(i))


def test_trainer_rejects_degenerate_cycle_length(tmp_path):
    from audit.cryptography.gohr.trainer import GohrTrainer
    for bad in (0, 1, -5):
        with pytest.raises(ValueError, match="lr_cycle_length"):
            GohrTrainer(epochs=200, checkpoint_dir=tmp_path / "x", lr_cycle_length=bad)


def test_one_epoch_training_no_longer_divides_by_zero(tmp_path):
    """The old schedule raised ZeroDivisionError at epochs=1."""
    t = _production_trainer(tmp_path, epochs=1)
    assert t.learning_rate(0) == pytest.approx(REFERENCE_HIGH_LR)


def test_ce1_production_declares_the_reference_lr_parameters():
    assert ce1_driver.PRODUCTION_TRAINING["high_learning_rate"] == REFERENCE_HIGH_LR
    assert ce1_driver.PRODUCTION_TRAINING["low_learning_rate"] == REFERENCE_LOW_LR


# --------------------------------------------------------------------
# production CLI (controller-only)
# --------------------------------------------------------------------

_REAL_LEGACY = ROOT / "evidence_current" / "ce1" / "production_20260929"


def test_production_cli_trains_nothing_without_execute(tmp_path, capsys):
    run_dir = tmp_path / "audit/cryptography/evidence_current/ce1/prod"
    rc = ce1_driver.main(["--run-dir", str(run_dir), "--repo-root", str(tmp_path)])
    assert rc == 1 and "requires --execute" in capsys.readouterr().out
    assert not run_dir.exists()


def test_run_production_true_is_retired(tmp_path):
    with pytest.raises(RuntimeError, match="retired"):
        ce1_driver.run(output_path=tmp_path / "audit/cryptography/evidence_current/ce1/c.json",
                       train_eval_fn=lambda *a, **k: 0.5, data_fn=lambda i: None,
                       repo_root=tmp_path, production=True)


@pytest.mark.skipif(not (_REAL_LEGACY / "checkpoints").exists(), reason="legacy run absent")
def test_cli_recovers_real_legacy_block0_without_evaluating(tmp_path, capsys):
    from audit.cryptography.experiments.ce1 import controller as C
    from audit.cryptography.experiments.ce1 import resume as R
    before = {p.name: R.sha256_file(p) for p in (_REAL_LEGACY / "checkpoints").iterdir()}
    run_dir = tmp_path / "audit/cryptography/evidence_current/ce1/prod"
    rc = ce1_driver.main(["--recover-legacy-block0", "--run-dir", str(run_dir),
                          "--repo-root", str(tmp_path),
                          "--legacy-run-dir", str(_REAL_LEGACY),
                          "--sealed-source", str(_REAL_LEGACY / "sealed")])
    assert rc == 0
    cfg = C.load_run_config(run_dir)
    assert cfg.production and cfg.expected_sealed_sha256 == C.PINNED_SEALED_EVALUATION_SHA256
    manifest = json.loads((run_dir / C.RUN_MANIFEST).read_text())
    assert manifest["legacy_block0_decision"] == "RECOVER"
    sealed = C.ensure_sealed(run_dir, cfg)
    gate = C.block_validity_gate(run_dir, cfg, 0, sealed)
    assert gate["status"] == "LEGACY_RECOVERED_PENDING_EVALUATION"
    for arm in ("baseline", "destroyed"):
        st = R.load_arm_state(run_dir, "block0", arm)
        assert st.status == "TRAINING_COMPLETE" and st.evaluation is None   # NOT evaluated
        assert st.terminal_model_sha256 == before[f"block0_{arm}_FINAL_EPOCH.keras"]
    assert not C.ledger_for(run_dir).events() or not [
        e for e in C.ledger_for(run_dir).events() if e["event"] in ("EVALUATION", "EVALUATED")]
    # blocks 1..7 would TRAIN; block0 would only EVALUATE (never train)
    plan0 = C.plan_block(run_dir, cfg, 0, sealed["sha256"])
    assert {v["action"] for v in plan0["arms"].values()} == {"EVALUATE"}
    assert {v["action"] for v in C.plan_block(run_dir, cfg, 1, sealed["sha256"])["arms"]
            .values()} == {"TRAIN"}
    # the decision is immutable
    with pytest.raises(C.ControllerError, match="cannot be changed"):
        C.open_run(run_dir, cfg, repo_root=tmp_path, legacy_block0_decision="RETRAIN")
    assert {p.name: R.sha256_file(p) for p in (_REAL_LEGACY / "checkpoints").iterdir()} == before
