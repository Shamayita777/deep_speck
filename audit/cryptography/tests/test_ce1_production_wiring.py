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

    src = inspect.getsource(ce1_driver.build_production_components)
    assert "rounds=REFERENCE.rounds" in src
    assert "differential=REFERENCE.differential" in src
    assert "depth=REFERENCE.depth" in src
    assert "regularization=REFERENCE.l2_reg" in src


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
                          str(tmp_path / "evidence_current/ce1/c.json"),
                          "--repo-root", str(tmp_path)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frozen_parameters"]["n_blocks"] == 8
    assert report["frozen_parameters"]["min_valid_blocks"] == 6


def test_preflight_mode_runs_no_training(tmp_path, capsys):
    rc = ce1_driver.main(["--preflight",
                          "--output", str(tmp_path / "evidence_current/ce1/c.json"),
                          "--repo-root", str(tmp_path)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PREFLIGHT_OK"
    fp = report["frozen_parameters"]
    assert fp["n_blocks"] == 8 and fp["min_valid_blocks"] == 6
    assert fp["checkpoint_rule"] == "FINAL_EPOCH" and fp["terminal_epoch"] == 200
    assert fp["seed_base"] == 1000 and fp["evaluation_samples"] == 10 ** 6
    assert not (tmp_path / "evidence_current").exists()      # nothing written


# --------------------------------------------------------------------
# real component chain, tiny
# --------------------------------------------------------------------

def test_real_gohr_chain_smoke(tmp_path, monkeypatch):
    """Dataset -> Model -> Trainer -> Evaluator -> CE1 design -> certificate."""
    monkeypatch.setattr(ce1_driver, "PRODUCTION_TRAINING",
                        {**ce1_driver.PRODUCTION_TRAINING, "batch_size": 32, "epochs": 2})
    data_fn, train_eval_fn = ce1_driver.build_production_components(
        train_samples=256, validation_samples=64, evaluation_samples=64,
        checkpoint_dir=tmp_path / "ckpt", sealed_dir=tmp_path / "sealed")

    X_train, Y_train, X_eval, Y_eval = data_fn(0)
    # F1: a second block must receive the IDENTICAL sealed evaluation arrays
    _, _, X_eval2, Y_eval2 = data_fn(1)
    assert np.array_equal(X_eval, X_eval2) and np.array_equal(Y_eval, Y_eval2)
    # real Speck data at the reference configuration
    assert X_train.shape == (256, 64) and Y_train.shape == (256,)
    assert X_eval.shape == (64, 64) and Y_eval.shape == (64,)
    assert set(np.unique(Y_train)) <= {0, 1}
    assert data_fn.last_validation is not None                # training-time split exists

    acc = train_eval_fn(X_train, Y_train, X_eval, Y_eval,
                        seed=0, arm="baseline", block_id="block0")
    assert 0.0 <= acc <= 1.0


def test_real_chain_end_to_end_two_blocks(tmp_path, monkeypatch):
    """Full driver run() on the REAL components, 2 blocks, 1 epoch."""
    monkeypatch.setattr(ce1_driver, "PRODUCTION_TRAINING",
                        {**ce1_driver.PRODUCTION_TRAINING, "batch_size": 32, "epochs": 2})
    data_fn, train_eval_fn = ce1_driver.build_production_components(
        train_samples=256, validation_samples=64, evaluation_samples=64,
        checkpoint_dir=tmp_path / "ckpt", sealed_dir=tmp_path / "sealed")
    path, cert = ce1_driver.run(
        n_blocks=2, min_valid_blocks=2, production=False,
        output_path=tmp_path / "evidence_current" / "ce1" / "certificate.json",
        train_eval_fn=train_eval_fn, data_fn=data_fn, repo_root=tmp_path,
        seed_base=5)

    loaded = json.loads(Path(path).read_text())
    r = loaded["results"]
    # statistical unit and block bookkeeping survive to the certificate
    assert r["n_blocks"] == 2
    assert len(r["blocks"]) == 2
    for b in r["blocks"]:
        assert b["training_labels_permuted"] is True
        assert b["evaluation_labels_intact"] is True
        assert b["evaluation_shared_between_arms"] is True
    assert loaded["reference_configuration"]["depth"] == 10
    assert loaded["reference_configuration"]["rounds"] == 5
    assert loaded["results"]["training_protocol"]["note"]
    assert loaded["provenance"]["seed_policy"]["exact_dataset_replay_available"] is False
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
                   output_path=tmp_path / "evidence_current" / "ce1" / "c.json",
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
