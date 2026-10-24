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

IDENTITY = {
    "seed": 7, "condition": "baseline", "replicate": 1,
    "config_hash": "c" * 64, "data_digest": "d" * 64,
}


def write_chain(directory: Path, losses, overrides=None) -> str:
    """Write chained epoch records (as the callback does); return the head."""
    head = gohr.GENESIS_RECORD_SHA256
    for epoch, loss in enumerate(losses, start=1):
        checkpoint = gohr.epoch_checkpoint_path(directory, epoch)
        checkpoint.write_bytes(f"weights-{epoch}".encode())
        record = {
            "schema_version": "2.0",
            "epoch": epoch,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": gohr._sha256_file(checkpoint),
            "val_loss": float(loss),
            "logs": {"val_loss": float(loss)},
            "previous_record_sha256": head,
            **IDENTITY,
        }
        record.update((overrides or {}).get(epoch, {}))
        path = gohr.epoch_metrics_path(directory, epoch)
        path.write_text(json.dumps(record), encoding="utf-8")
        head = gohr._sha256_file(path)
    return head


def select(directory: Path, head: str, total_epochs: int = 3) -> dict:
    return gohr.select_checkpoint(directory, total_epochs=total_epochs, chain_head=head, **IDENTITY)


def edit_record(directory: Path, epoch: int, **changes) -> None:
    path = gohr.epoch_metrics_path(directory, epoch)
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record), encoding="utf-8")


# ============================================================================
# Selection rule and record integrity
# ============================================================================


def test_selects_minimum_val_loss_not_last_epoch(tmp_path):
    head = write_chain(tmp_path, [0.30, 0.10, 0.20])
    selection = select(tmp_path, head)
    assert selection["selected_epoch"] == 2
    assert selection["selected_val_loss"] == pytest.approx(0.10)
    assert selection["rule"] == gohr.MODEL_SELECTION_RULE
    assert selection["record_chain_head"] == head
    assert selection["data_digest"] == "d" * 64


def test_ties_resolve_to_earliest_epoch(tmp_path):
    head = write_chain(tmp_path, [0.25, 0.25, 0.25])
    assert select(tmp_path, head)["selected_epoch"] == 1


def test_missing_epoch_record_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    gohr.epoch_metrics_path(tmp_path, 2).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        select(tmp_path, head)


def test_record_from_other_condition_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2], overrides={2: {"condition": "perturbed"}})
    with pytest.raises(RuntimeError, match="condition"):
        select(tmp_path, head)


def test_record_from_other_data_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3], overrides={1: {"data_digest": "e" * 64}})
    with pytest.raises(RuntimeError, match="data_digest"):
        select(tmp_path, head, total_epochs=1)


def test_tampered_selected_checkpoint_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    gohr.epoch_checkpoint_path(tmp_path, 2).write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="hashes to"):
        select(tmp_path, head)


@pytest.mark.parametrize("epoch", [1, 3])
def test_tampered_non_selected_checkpoint_is_refused(tmp_path, epoch):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    gohr.epoch_checkpoint_path(tmp_path, epoch).write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="hashes to"):
        select(tmp_path, head)


def test_lowered_val_loss_in_earlier_record_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    edit_record(tmp_path, 1, val_loss=0.01, logs={"val_loss": 0.01})
    with pytest.raises(RuntimeError, match="does not chain"):
        select(tmp_path, head)


def test_modified_val_loss_in_last_record_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    edit_record(tmp_path, 3, val_loss=0.01, logs={"val_loss": 0.01})
    with pytest.raises(RuntimeError, match="head committed"):
        select(tmp_path, head)


def test_recomputed_chain_still_disagrees_with_committed_head(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    write_chain(tmp_path, [0.3, 0.1, 0.01])  # consistent forgery of the whole chain
    with pytest.raises(RuntimeError, match="head committed"):
        select(tmp_path, head)


def test_non_finite_val_loss_is_refused(tmp_path):
    head = write_chain(tmp_path, [float("nan")])
    with pytest.raises(RuntimeError, match="non-finite"):
        select(tmp_path, head, total_epochs=1)


def test_malformed_record_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3])
    gohr.epoch_metrics_path(tmp_path, 1).write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        select(tmp_path, head, total_epochs=1)


def test_selection_record_is_never_silently_replaced(tmp_path):
    path = tmp_path / "selected_checkpoint.json"
    gohr.record_selection(path, {"selected_epoch": 2})
    gohr.record_selection(path, {"selected_epoch": 2})  # identical: accepted
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        gohr.record_selection(path, {"selected_epoch": 3})
    assert json.loads(path.read_text())["selected_epoch"] == 2


# ============================================================================
# Crash between the epoch record and the state update
# ============================================================================


def committed_state(directory: Path, completed: int, head: str) -> Path:
    state = directory / "state.json"
    gohr.write_training_state(state, identity=IDENTITY, total_epochs=3, completed=completed, chain_head=head)
    return state


def test_interrupted_state_update_adopts_the_completed_epoch(tmp_path):
    write_chain(tmp_path, [0.3])
    head_1 = gohr._sha256_file(gohr.epoch_metrics_path(tmp_path, 1))
    state = committed_state(tmp_path, 1, head_1)
    head_2 = write_chain(tmp_path, [0.3, 0.2])  # epoch 2 written, then "crash"
    assert json.loads(state.read_text())["completed_epochs"] == 1

    completed, head = gohr.reconcile_uncommitted_epochs(
        tmp_path, state, identity=IDENTITY, total_epochs=3, committed=1, chain_head=head_1
    )
    assert (completed, head) == (2, head_2)
    recorded = json.loads(state.read_text())
    assert recorded["completed_epochs"] == 2 and recorded["record_chain_head"] == head_2
    assert gohr.epoch_metrics_path(tmp_path, 2).exists()
    assert gohr._sha256_file(gohr.epoch_metrics_path(tmp_path, 2)) == head_2


def test_interrupted_first_epoch_without_state_is_adopted(tmp_path):
    head_1 = write_chain(tmp_path, [0.3])
    state = tmp_path / "state.json"
    completed, head = gohr.reconcile_uncommitted_epochs(
        tmp_path, state, identity=IDENTITY, total_epochs=3, committed=0,
        chain_head=gohr.GENESIS_RECORD_SHA256,
    )
    assert (completed, head) == (1, head_1)
    assert json.loads(state.read_text())["completed_epochs"] == 1


def test_unverifiable_uncommitted_epoch_is_set_aside_not_overwritten(tmp_path):
    write_chain(tmp_path, [0.3])
    head_1 = gohr._sha256_file(gohr.epoch_metrics_path(tmp_path, 1))
    state = committed_state(tmp_path, 1, head_1)
    write_chain(tmp_path, [0.3, 0.2])
    gohr.epoch_checkpoint_path(tmp_path, 2).write_bytes(b"torn write")
    completed, head = gohr.reconcile_uncommitted_epochs(
        tmp_path, state, identity=IDENTITY, total_epochs=3, committed=1, chain_head=head_1
    )
    assert (completed, head) == (1, head_1)
    assert not gohr.epoch_checkpoint_path(tmp_path, 2).exists()
    assert not gohr.epoch_metrics_path(tmp_path, 2).exists()
    kept = sorted(p.name for p in tmp_path.glob("*.uncommitted-1"))
    assert kept == ["checkpoint_epoch_002.keras.uncommitted-1",
                    "checkpoint_epoch_002.metrics.json.uncommitted-1"]
    assert (tmp_path / kept[0]).read_bytes() == b"torn write"


def test_checkpoint_without_record_is_set_aside(tmp_path):
    gohr.epoch_checkpoint_path(tmp_path, 1).write_bytes(b"partial")
    completed, _ = gohr.reconcile_uncommitted_epochs(
        tmp_path, tmp_path / "state.json", identity=IDENTITY, total_epochs=3,
        committed=0, chain_head=gohr.GENESIS_RECORD_SHA256,
    )
    assert completed == 0
    assert (tmp_path / "checkpoint_epoch_001.keras.uncommitted-1").read_bytes() == b"partial"


# ============================================================================
# Real hand-off: the reloaded model is the selected checkpoint
# ============================================================================


def tiny_fit(tmp_path, *, epochs=3, train=None):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(64, 64), dtype=np.uint8)
    y = rng.integers(0, 2, size=64, dtype=np.uint8)
    vx = rng.integers(0, 2, size=(32, 64), dtype=np.uint8)
    vy = rng.integers(0, 2, size=32, dtype=np.uint8)
    if train is not None:
        x, y = train
    adapter = gohr.GohrAdapter(
        validation_x=vx, validation_y=vy, test_x=vx, test_y=vy,
        num_rounds=5, depth=1, epochs=epochs, seed=11,
        model_directory=str(tmp_path / "model"),
    )
    condition_dir = tmp_path / "replicate_01" / "baseline"
    adapter.train(
        x, y, checkpoint_path=condition_dir, state_path=condition_dir / "state.json",
        replicate=1, condition="baseline", run_config_hash="e" * 64,
    )
    return adapter, condition_dir, (x, y)


def test_adapter_reloads_selected_checkpoint_and_resume_agrees(tmp_path):
    from keras.models import load_model

    first, condition_dir, _ = tiny_fit(tmp_path)
    records = [
        json.loads(gohr.epoch_metrics_path(condition_dir, e).read_text()) for e in (1, 2, 3)
    ]
    expected_epoch = min(records, key=lambda r: (r["val_loss"], r["epoch"]))["epoch"]
    assert first.selection["selected_epoch"] == expected_epoch
    assert first.selection["condition"] == "baseline"

    reference = load_model(gohr.epoch_checkpoint_path(condition_dir, expected_epoch))
    for got, want in zip(first.model.get_weights(), reference.get_weights()):
        np.testing.assert_array_equal(got, want)

    recorded = json.loads((condition_dir / "selected_checkpoint.json").read_text())
    assert recorded == first.selection
    assert not list((tmp_path / "model").glob("best_*"))

    # A completed fit resumed in a new process yields the same selection.
    second, _, _ = tiny_fit(tmp_path)
    assert second.selection == first.selection


def test_resume_against_substituted_partition_is_refused(tmp_path):
    _, condition_dir, (x, y) = tiny_fit(tmp_path, epochs=1)
    swapped = x.copy()
    swapped[0] ^= 1  # same shape, dtype and configuration; different content
    with pytest.raises(RuntimeError, match="data_digest"):
        tiny_fit(tmp_path, epochs=1, train=(swapped, y))
    assert json.loads((condition_dir / "state.json").read_text())["completed_epochs"] == 1


def test_resume_refuses_tampered_committed_checkpoint(tmp_path):
    _, condition_dir, _ = tiny_fit(tmp_path, epochs=1)
    gohr.epoch_checkpoint_path(condition_dir, 1).write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="hashes to"):
        tiny_fit(tmp_path, epochs=1)


def test_partition_digest_depends_on_content_only_through_bytes():
    a = np.zeros((4, 64), dtype=np.uint8)
    b = a.copy()
    b[3, 63] = 1
    assert gohr.partition_digest({"x": a}) == gohr.partition_digest({"x": a.copy()})
    assert gohr.partition_digest({"x": a}) != gohr.partition_digest({"x": b})
    assert gohr.partition_digest({"x": a}, chunk_rows=1) == gohr.partition_digest({"x": a})


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


# ============================================================================
# Partition overlap audit
# ============================================================================


def bits(rows):
    return np.array([[int(c) for c in f"{r:064b}"] for r in rows], dtype=np.uint8)


def test_overlap_audit_counts_shared_and_duplicate_rows():
    train = bits([1, 2, 2, 3, 2**63 + 5])
    validation = bits([3, 4, 9])
    test = bits([2, 2**63 + 5, 4, 7])
    report = driver.audit_partition_overlap(train, validation, test)
    assert report["rows"] == {"train": 5, "validation": 3, "test": 4}
    assert report["within_duplicate_rows"] == {"train": 1, "validation": 0, "test": 0}
    assert report["cross"]["train_validation"] == {"rows": 1, "distinct": 1}
    assert report["cross"]["train_test"] == {"rows": 2, "distinct": 2}
    assert report["cross"]["validation_test"] == {"rows": 1, "distinct": 1}
    with pytest.raises(RuntimeError, match="separation violated"):
        driver.require_partition_separation(report)


def test_overlap_audit_chunking_matches_unchunked():
    rng = np.random.default_rng(9)
    x = rng.integers(0, 2, size=(1000, 64), dtype=np.uint8)
    np.testing.assert_array_equal(driver._row_keys(x, chunk_rows=7), driver._row_keys(x))


def test_disjoint_partitions_pass_with_duplicates_only_recorded():
    report = driver.audit_partition_overlap(bits([1, 1, 2]), bits([3]), bits([4]))
    assert report["within_duplicate_rows"]["train"] == 1
    driver.require_partition_separation(report)


def test_adapter_recovers_from_crash_before_state_update(tmp_path):
    first, condition_dir, _ = tiny_fit(tmp_path, epochs=2)
    record_2 = gohr.epoch_metrics_path(condition_dir, 2)
    before = (record_2.read_bytes(), gohr.epoch_checkpoint_path(condition_dir, 2).read_bytes())
    # Roll state back to epoch 1, as if the process died after writing the
    # epoch-2 record but before state.json advanced.
    state_path = condition_dir / "state.json"
    state = json.loads(state_path.read_text())
    identity = {k: state[k] for k in ("seed", "condition", "replicate", "config_hash", "data_digest")}
    gohr.write_training_state(
        state_path, identity=identity, total_epochs=2, completed=1,
        chain_head=gohr._sha256_file(gohr.epoch_metrics_path(condition_dir, 1)),
    )
    second, _, _ = tiny_fit(tmp_path, epochs=2)
    assert json.loads(state_path.read_text())["completed_epochs"] == 2
    assert (record_2.read_bytes(), gohr.epoch_checkpoint_path(condition_dir, 2).read_bytes()) == before
    assert second.selection == first.selection
    assert not list(condition_dir.glob("*.uncommitted-*"))
