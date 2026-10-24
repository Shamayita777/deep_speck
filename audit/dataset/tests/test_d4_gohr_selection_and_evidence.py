"""
Regression tests for the D4 checkpoint hand-off and evidence gates.

Covers:
    * minimum-val_loss selection over immutable epoch records, and the
      refusals for missing, foreign, tampered or non-finite records;
    * the selected checkpoint actually being the model used downstream
      (real Keras, tiny data);
    * baseline/perturbed selection records staying attributable;
    * persisted perturbations being verified before reuse;
    * the frozen input encoding, content-hashed dataset cache,
      repository-relative output root and no-overwrite certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import audit.dataset.adapters.gohr as gohr
import audit.dataset.d4_controlled_perturbation as d4
import audit.dataset.experiments.d4_gohr_label_shuffle as driver
from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

IDENTITY = {"seed": 7, "condition": "baseline", "replicate": 1, "config_hash": "c" * 64}


def write_epoch(directory: Path, epoch: int, val_loss: float, **overrides) -> Path:
    checkpoint = directory / f"checkpoint_epoch_{epoch:03d}.keras"
    checkpoint.write_bytes(f"weights-{epoch}".encode())
    record = {
        "schema_version": "1.0",
        "epoch": epoch,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": gohr._sha256_file(checkpoint),
        "val_loss": float(val_loss),
        "logs": {"val_loss": float(val_loss)},
        **IDENTITY,
    }
    record.update(overrides)
    gohr.epoch_metrics_path(directory, epoch).write_text(json.dumps(record), encoding="utf-8")
    return checkpoint


def select(directory: Path, total_epochs: int = 3) -> dict:
    return gohr.select_checkpoint(directory, total_epochs=total_epochs, **IDENTITY)


# ============================================================================
# Selection rule
# ============================================================================


def test_selects_minimum_val_loss_not_last_epoch(tmp_path):
    for epoch, loss in [(1, 0.30), (2, 0.10), (3, 0.20)]:
        write_epoch(tmp_path, epoch, loss)
    selection = select(tmp_path)
    assert selection["selected_epoch"] == 2
    assert selection["selected_val_loss"] == pytest.approx(0.10)
    assert selection["rule"] == gohr.MODEL_SELECTION_RULE
    assert selection["condition"] == "baseline"


def test_ties_resolve_to_earliest_epoch(tmp_path):
    for epoch in (1, 2, 3):
        write_epoch(tmp_path, epoch, 0.25)
    assert select(tmp_path)["selected_epoch"] == 1


def test_missing_epoch_record_is_refused(tmp_path):
    write_epoch(tmp_path, 1, 0.3)
    write_epoch(tmp_path, 3, 0.1)
    with pytest.raises(RuntimeError, match="missing"):
        select(tmp_path)


def test_record_from_other_condition_is_refused(tmp_path):
    write_epoch(tmp_path, 1, 0.3)
    write_epoch(tmp_path, 2, 0.1, condition="perturbed")
    write_epoch(tmp_path, 3, 0.2)
    with pytest.raises(RuntimeError, match="condition"):
        select(tmp_path)


def test_record_from_other_config_is_refused(tmp_path):
    write_epoch(tmp_path, 1, 0.3, config_hash="d" * 64)
    with pytest.raises(RuntimeError, match="config_hash"):
        select(tmp_path, total_epochs=1)


def test_tampered_selected_checkpoint_is_refused(tmp_path):
    write_epoch(tmp_path, 1, 0.3)
    best = write_epoch(tmp_path, 2, 0.1)
    write_epoch(tmp_path, 3, 0.2)
    best.write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="hashes to"):
        select(tmp_path)


def test_non_finite_val_loss_is_refused(tmp_path):
    write_epoch(tmp_path, 1, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        select(tmp_path, total_epochs=1)


def test_malformed_record_is_refused(tmp_path):
    write_epoch(tmp_path, 1, 0.3)
    gohr.epoch_metrics_path(tmp_path, 1).write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        select(tmp_path, total_epochs=1)


def test_selection_record_is_never_silently_replaced(tmp_path):
    path = tmp_path / "selected_checkpoint.json"
    gohr.record_selection(path, {"selected_epoch": 2})
    gohr.record_selection(path, {"selected_epoch": 2})  # identical: accepted
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        gohr.record_selection(path, {"selected_epoch": 3})
    assert json.loads(path.read_text())["selected_epoch"] == 2


# ============================================================================
# Real hand-off: the reloaded model is the selected checkpoint
# ============================================================================


def test_adapter_reloads_selected_checkpoint_and_resume_agrees(tmp_path):
    from keras.models import load_model

    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(64, 64), dtype=np.uint8)
    y = rng.integers(0, 2, size=64, dtype=np.uint8)
    vx = rng.integers(0, 2, size=(32, 64), dtype=np.uint8)
    vy = rng.integers(0, 2, size=32, dtype=np.uint8)

    def adapter():
        return gohr.GohrAdapter(
            validation_x=vx, validation_y=vy, test_x=vx, test_y=vy,
            num_rounds=5, depth=1, epochs=3, seed=11,
            model_directory=str(tmp_path / "model"),
        )

    condition_dir = tmp_path / "replicate_01" / "baseline"
    kwargs = dict(
        checkpoint_path=condition_dir, state_path=condition_dir / "state.json",
        replicate=1, condition="baseline", run_config_hash="e" * 64,
    )
    first = adapter()
    model = first.train(x, y, **kwargs)

    records = [
        json.loads(gohr.epoch_metrics_path(condition_dir, e).read_text()) for e in (1, 2, 3)
    ]
    expected_epoch = min(records, key=lambda r: (r["val_loss"], r["epoch"]))["epoch"]
    assert first.selection["selected_epoch"] == expected_epoch
    assert first.selection["condition"] == "baseline"

    reference = load_model(condition_dir / f"checkpoint_epoch_{expected_epoch:03d}.keras")
    for got, want in zip(model.get_weights(), reference.get_weights()):
        np.testing.assert_array_equal(got, want)

    recorded = json.loads((condition_dir / "selected_checkpoint.json").read_text())
    assert recorded == first.selection
    assert not list((tmp_path / "model").glob("best_*"))

    # A completed fit resumed in a new process yields the same selection.
    second = adapter()
    second.train(x, y, **kwargs)
    assert second.selection == first.selection


# ============================================================================
# Engine: attribution and persisted perturbations
# ============================================================================


class SelectingAdapter:
    """Test double that reports a selection record like GohrAdapter."""

    def __init__(self, seed, swap=False):
        self.seed = seed
        self.swap = swap
        self.selection = None
        self.rule = False

    def train(self, x, y, *, checkpoint_path, state_path, replicate, condition, run_config_hash):
        x = np.asarray(x)
        self.rule = bool(np.array_equal(x[:, 0], np.asarray(y).reshape(-1)))
        reported = {"baseline": "perturbed", "perturbed": "baseline"}[condition] if self.swap else condition
        self.selection = {"condition": reported, "replicate": replicate, "selected_epoch": 1}
        return self

    def predict(self, x):
        x = np.asarray(x)
        return x[:, 0].astype(float) if self.rule else np.zeros(len(x))


def fixture(n=400):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=n)
    x = np.stack([y, rng.integers(0, 2, size=n)], axis=1)
    return x, y


def run(tmp_path, factory, replicates=2):
    x, y = fixture()
    return d4.run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=x, train_labels=y, test_features=x, test_labels=y,
        adapter_factory=factory, checkpoint_root=tmp_path / "ck",
        replicates=replicates, audit_seed=0, effect_threshold=0.01,
        bootstrap_replicates=1000, run_config_hash="f" * 64,
    )


def test_selection_provenance_reaches_certificate(tmp_path):
    result = run(tmp_path, lambda seed: SelectingAdapter(seed))
    certificate = d4.generate_certificate(
        result, dataset_id="t", dataset_version="t", generation_procedure="t",
        generation_parameters={}, audit_seed=0, confidence_level=0.95, alpha=0.05,
    )
    rows = certificate["findings"]["replicates"]
    assert [r["baseline_selection"]["condition"] for r in rows] == ["baseline"] * 2
    assert [r["perturbed_selection"]["condition"] for r in rows] == ["perturbed"] * 2
    assert [r["perturbed_selection"]["replicate"] for r in rows] == [1, 2]


def test_crossed_selection_records_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="do not match"):
        run(tmp_path, lambda seed: SelectingAdapter(seed, swap=True))


def test_persisted_perturbation_is_written_once_with_digest(tmp_path):
    path = tmp_path / "labels.npy"
    d4.save_perturbation(path, np.array([1, 0, 1, 0]))
    assert (tmp_path / "labels.npy.sha256").exists()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        d4.save_perturbation(path, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(
        d4.load_verified_perturbation(path, np.array([0, 0, 1, 1])), [1, 0, 1, 0]
    )


def test_tampered_perturbation_is_refused(tmp_path):
    path = tmp_path / "labels.npy"
    d4.save_perturbation(path, np.array([1, 0, 1, 0]))
    np.save(path, np.array([0, 1, 0, 1]))
    with pytest.raises(RuntimeError, match="digest"):
        d4.load_verified_perturbation(path, np.array([1, 0, 1, 0]))


def test_perturbation_without_digest_is_refused(tmp_path):
    path = tmp_path / "labels.npy"
    np.save(path, np.array([1, 0]))
    with pytest.raises(RuntimeError, match="no digest"):
        d4.load_verified_perturbation(path, np.array([1, 0]))


def test_perturbation_of_other_labels_is_refused(tmp_path):
    path = tmp_path / "labels.npy"
    d4.save_perturbation(path, np.array([1, 1, 1, 0]))
    with pytest.raises(RuntimeError, match="not a permutation"):
        d4.load_verified_perturbation(path, np.array([1, 0, 0, 0]))


def test_resumed_run_reuses_verified_perturbation(tmp_path):
    first = run(tmp_path, lambda seed: SelectingAdapter(seed))
    second = run(tmp_path, lambda seed: SelectingAdapter(seed))
    assert [r.perturbed_score for r in first.replicates] == [
        r.perturbed_score for r in second.replicates
    ]


# ============================================================================
# Driver: encoding, dataset cache, paths, certificate
# ============================================================================


def partitions(**overrides):
    rng = np.random.default_rng(5)
    parts = {}
    for name in ("train", "validation", "test"):
        parts[f"{name}_x"] = rng.integers(0, 2, size=(8, 64), dtype=np.uint8)
        parts[f"{name}_y"] = rng.integers(0, 2, size=8, dtype=np.uint8)
    parts.update(overrides)
    return parts


def test_frozen_encoding_accepts_raw_bits():
    driver.validate_input_encoding(partitions())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"train_x": np.zeros((8, 64), dtype=np.float32)}, "dtype"),
        ({"test_x": np.zeros((8, 32), dtype=np.uint8)}, "width"),
        ({"validation_x": np.full((8, 64), 255, dtype=np.uint8)}, "rescaled"),
        ({"train_y": np.full(8, 2, dtype=np.uint8)}, "labels"),
        ({"test_y": np.zeros(7, dtype=np.uint8)}, "length"),
    ],
)
def test_frozen_encoding_rejects_departures(override, message):
    with pytest.raises(RuntimeError, match=message):
        driver.validate_input_encoding(partitions(**override))


def test_dataset_cache_detects_altered_content(tmp_path):
    driver.save_dataset_cache(root=tmp_path, run_config_hash="a" * 64, **partitions())
    driver.validate_dataset_cache(tmp_path, "a" * 64)
    path = driver.dataset_cache_paths(tmp_path)["test_y"]
    altered = np.load(path)
    altered[0] ^= 1
    np.save(path, altered)  # same shape and dtype; only content changes
    with pytest.raises(RuntimeError, match="SHA-256"):
        driver.validate_dataset_cache(tmp_path, "a" * 64)


def test_dataset_cache_without_digests_is_refused(tmp_path):
    driver.save_dataset_cache(root=tmp_path, run_config_hash="a" * 64, **partitions())
    metadata_path = driver.dataset_cache_paths(tmp_path)["metadata"]
    metadata = json.loads(metadata_path.read_text())
    del metadata["sha256"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="SHA-256"):
        driver.validate_dataset_cache(tmp_path, "a" * 64)


def test_output_root_is_repository_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert driver.OUTPUT_DIRECTORY.is_absolute()
    assert (driver.REPOSITORY_ROOT / "audit" / "dataset" / "d4_controlled_perturbation.py").exists()
    assert driver.OUTPUT_DIRECTORY == driver.REPOSITORY_ROOT / "audit" / "dataset" / "evidence" / "d4"


def test_run_config_declares_selection_and_encoding():
    import argparse

    namespace = argparse.Namespace(
        epochs=2, train_samples=10, validation_samples=10, test_samples=10,
        replicates=2, audit_seed=0, bootstrap_replicates=1000,
        effect_threshold=0.01, alpha=0.05,
    )
    config = driver.build_run_config(namespace)
    assert config["training_protocol"]["model_selection"] == gohr.MODEL_SELECTION_RULE
    assert config["training_protocol"]["selection_partition"] == "validation"
    assert config["input_encoding"] == driver.INPUT_ENCODING


def test_certificate_is_never_overwritten(tmp_path):
    path = tmp_path / "d4.json"
    driver.write_certificate_exclusive(path, {"decision": "first"})
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        driver.write_certificate_exclusive(path, {"decision": "second"})
    assert json.loads(path.read_text()) == {"decision": "first"}
    assert not list(tmp_path.glob("*.tmp"))
