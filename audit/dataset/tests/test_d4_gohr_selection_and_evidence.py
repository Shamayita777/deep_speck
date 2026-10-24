"""
Gohr D4 adapter lifecycle and stage tests.

Covers the eight checkpoint/recovery situations documented in
``audit.dataset.adapters.gohr_d4``, the orphan-epoch crash window (an
uncommitted record whose val_loss is altered while its checkpoint is not),
data binding, reload of the selected checkpoint, and the driver stages
end to end on tiny data (real speck.make_train_data, real Keras, depth 1).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from audit.dataset import integrity as I
from audit.dataset.adapters import gohr_d4 as G
import audit.dataset.experiments.d4_gohr_label_shuffle as driver

IDENTITY = {"seed": 7, "condition": "baseline", "replicate": 1, "config_hash": "c" * 64,
            "train_digest": "d" * 64, "validation_digest": "e" * 64, "evaluation_digest": "f" * 64,
            "model": {"architecture": "gohr_resnet", "num_rounds": 5, "depth": 1, "epochs": 3,
                      "batch_size": 5000}}


def write_chain(directory: Path, losses, overrides=None) -> str:
    head = G.GENESIS_RECORD_SHA256
    for epoch, loss in enumerate(losses, start=1):
        checkpoint = G.epoch_checkpoint_path(directory, epoch)
        checkpoint.write_bytes(f"weights-{epoch}".encode())
        record = {"schema_version": G.RECORD_SCHEMA, "epoch": epoch, "checkpoint": checkpoint.name,
                  "checkpoint_sha256": I.sha256_file(checkpoint), "val_loss": float(loss),
                  "logs": {"val_loss": float(loss)}, "previous_record_sha256": head, **IDENTITY}
        record.update((overrides or {}).get(epoch, {}))
        path = G.epoch_record_path(directory, epoch)
        path.write_text(json.dumps(record))
        head = I.sha256_file(path)
    return head


def edit_record(directory, epoch, **changes):
    path = G.epoch_record_path(directory, epoch)
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def commit(directory, epoch, retries=()):
    head = I.sha256_file(G.epoch_record_path(directory, epoch)) if epoch else G.GENESIS_RECORD_SHA256
    G.write_state(directory / "state.json", identity=IDENTITY, total_epochs=3, completed=epoch,
                  chain_head=head, retries=list(retries))
    return head


def select(directory, head):
    return G.select_checkpoint(directory, total_epochs=3, identity=IDENTITY, chain_head=head)


# ============================================================ selection


def test_selects_min_val_loss_over_history_and_lists_every_checkpoint(tmp_path):
    head = write_chain(tmp_path, [0.30, 0.10, 0.20])
    selection = select(tmp_path, head)
    assert selection["selected_epoch"] == 2 and selection["rule"] == G.MODEL_SELECTION_RULE
    assert len(selection["epoch_checkpoint_sha256"]) == 3
    tied = write_chain(tmp_path, [0.25, 0.25, 0.25])
    assert select(tmp_path, tied)["selected_epoch"] == 1


@pytest.mark.parametrize("epoch", [1, 2, 3])
def test_any_tampered_checkpoint_is_refused(tmp_path, epoch):            # case 5
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    G.epoch_checkpoint_path(tmp_path, epoch).write_bytes(b"replaced")
    with pytest.raises(I.IntegrityError, match="hashes to"):
        select(tmp_path, head)


@pytest.mark.parametrize("epoch, match", [(1, "does not chain"), (2, "does not chain"), (3, "head committed")])
def test_any_altered_val_loss_is_refused(tmp_path, epoch, match):         # case 6
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    edit_record(tmp_path, epoch, val_loss=0.001, logs={"val_loss": 0.001})
    with pytest.raises(I.IntegrityError, match=match):
        select(tmp_path, head)


def test_rewritten_consistent_chain_still_disagrees_with_committed_head(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    write_chain(tmp_path, [0.3, 0.1, 0.001])
    with pytest.raises(I.IntegrityError, match="head committed"):
        select(tmp_path, head)


def test_val_loss_disagreeing_with_logs_and_unknown_fields_are_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.1, 0.2], overrides={2: {"val_loss": 0.05}})
    with pytest.raises(I.IntegrityError, match="disagrees with its own logs"):
        select(tmp_path, head)
    head = write_chain(tmp_path, [0.3, 0.1, 0.2], overrides={1: {"extra": 1}})
    with pytest.raises(I.IntegrityError, match="schema"):
        select(tmp_path, head)


@pytest.mark.parametrize("field, value", [("condition", "perturbed"), ("replicate", 2), ("seed", 8),
                                          ("config_hash", "x" * 64), ("train_digest", "0" * 64),
                                          ("validation_digest", "0" * 64), ("evaluation_digest", "0" * 64)])
def test_record_from_another_fit_is_refused(tmp_path, field, value):    # case 7
    head = write_chain(tmp_path, [0.3, 0.1, 0.2], overrides={2: {field: value}})
    with pytest.raises(I.IntegrityError, match=field):
        select(tmp_path, head)


def test_non_finite_and_malformed_records_are_refused(tmp_path):          # case 8 (record)
    head = write_chain(tmp_path, [0.3, float("nan"), 0.2])
    with pytest.raises(I.IntegrityError, match="non-finite"):
        select(tmp_path, head)
    head = write_chain(tmp_path, [0.3, 0.1, 0.2])
    G.epoch_record_path(tmp_path, 1).write_text('{"epoch": 1')
    with pytest.raises(I.IntegrityError, match="malformed"):
        select(tmp_path, head)


# ============================================================ committed state and recovery


def test_committed_state_verifies(tmp_path):                             # case 1
    write_chain(tmp_path, [0.3, 0.1])
    commit(tmp_path, 2)
    state = G.load_committed_state(tmp_path / "state.json", tmp_path, identity=IDENTITY, total_epochs=3)
    assert state["completed_epochs"] == 2


def test_truncated_state_is_refused(tmp_path):                           # case 8 (state)
    write_chain(tmp_path, [0.3])
    commit(tmp_path, 1)
    (tmp_path / "state.json").write_text('{"schema_version": ')
    with pytest.raises(I.IntegrityError, match="truncated or malformed"):
        G.load_committed_state(tmp_path / "state.json", tmp_path, identity=IDENTITY, total_epochs=3)


def test_state_of_another_fit_is_refused(tmp_path):                      # case 7 (state)
    write_chain(tmp_path, [0.3])
    commit(tmp_path, 1)
    with pytest.raises(I.IntegrityError, match="different fit"):
        G.load_committed_state(tmp_path / "state.json", tmp_path,
                               identity=dict(IDENTITY, train_digest="0" * 64), total_epochs=3)


def classify(tmp_path, committed, head):
    return G.classify_uncommitted(tmp_path, committed=committed, chain_head=head, identity=IDENTITY)


def test_incomplete_write_is_quarantinable(tmp_path):                    # case 2
    head = write_chain(tmp_path, [0.3])
    commit(tmp_path, 1)
    (tmp_path / "checkpoint_epoch_002.metrics.json.tmp").write_text('{"partial')
    assert classify(tmp_path, 1, head)["reason"] == "incomplete_write"


def test_checkpoint_without_record_is_quarantinable(tmp_path):           # case 3
    head = write_chain(tmp_path, [0.3])
    G.epoch_checkpoint_path(tmp_path, 2).write_bytes(b"torn")
    assert classify(tmp_path, 1, head)["reason"] == "checkpoint_without_record"


def test_valid_orphan_is_quarantined_never_adopted(tmp_path):            # case 4
    head1 = write_chain(tmp_path, [0.3])
    commit(tmp_path, 1)
    write_chain(tmp_path, [0.3, 0.1])
    classification = classify(tmp_path, 1, head1)
    assert classification["reason"] == "complete_uncommitted_orphan"
    entry = G.quarantine(tmp_path, classification)
    kept = tmp_path / entry["quarantine"]
    assert sorted(p.name for p in kept.iterdir()) == ["checkpoint_epoch_002.keras",
                                                     "checkpoint_epoch_002.metrics.json", "note.json"]
    assert not G.epoch_record_path(tmp_path, 2).exists()
    assert json.loads((tmp_path / "state.json").read_text())["completed_epochs"] == 1


def test_orphan_with_altered_val_loss_is_not_trusted(tmp_path):          # the orphan-epoch defect
    head1 = write_chain(tmp_path, [0.3])
    commit(tmp_path, 1)
    write_chain(tmp_path, [0.3, 0.2])
    edit_record(tmp_path, 2, val_loss=0.001, logs={"val_loss": 0.001})   # checkpoint untouched
    classification = classify(tmp_path, 1, head1)
    # Its chain link and checkpoint digest still verify, so nothing can tell
    # this from a genuine orphan: it is quarantined and the epoch retrained.
    assert classification["reason"] == "complete_uncommitted_orphan"
    G.quarantine(tmp_path, classification)
    state = G.load_committed_state(tmp_path / "state.json", tmp_path, identity=IDENTITY, total_epochs=3)
    assert state["completed_epochs"] == 1


@pytest.mark.parametrize("mutate, match", [
    (lambda d: G.epoch_checkpoint_path(d, 2).write_bytes(b"x"), "hashes to"),                 # case 5
    (lambda d: edit_record(d, 2, previous_record_sha256="1" * 64), "does not chain"),          # case 6
    (lambda d: edit_record(d, 2, condition="perturbed"), "condition"),                         # case 7
    (lambda d: G.epoch_record_path(d, 2).write_text("{"), "malformed"),                        # case 8
    (lambda d: G.epoch_checkpoint_path(d, 2).unlink(), "record but no checkpoint"),
    (lambda d: G.epoch_checkpoint_path(d, 3).write_bytes(b"x"), "only epoch 2"),
])
def test_suspicious_uncommitted_artifacts_are_refused_and_left_in_place(tmp_path, mutate, match):
    head1 = write_chain(tmp_path, [0.3])
    commit(tmp_path, 1)
    write_chain(tmp_path, [0.3, 0.2])
    mutate(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(I.IntegrityError, match=match):
        classify(tmp_path, 1, head1)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ============================================================ real Keras fits


def tiny(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, (64, 64), dtype=np.uint8), rng.integers(0, 2, 64, dtype=np.uint8),
            rng.integers(0, 2, (32, 64), dtype=np.uint8), rng.integers(0, 2, 32, dtype=np.uint8))


def fit(tmp_path, *, epochs=2, x=None, y=None, condition="baseline"):
    tx, ty, vx, vy = tiny()
    adapter = G.GohrD4Adapter(validation_x=vx, validation_y=vy, evaluation_digest="f" * 64,
                              num_rounds=5, depth=1, epochs=epochs, seed=11,
                              model_directory=str(tmp_path / "m"))
    directory = tmp_path / "replicate_01" / condition
    adapter.train(tx if x is None else x, ty if y is None else y, checkpoint_path=directory,
                  state_path=directory / "state.json", replicate=1, condition=condition,
                  run_config_hash="e" * 64)
    return adapter, directory


def test_selected_checkpoint_is_reloaded_and_resume_agrees(tmp_path):
    from keras.models import load_model

    first, directory = fit(tmp_path, epochs=3)
    best = first.selection["selected_epoch"]
    reference = load_model(G.epoch_checkpoint_path(directory, best))
    for got, want in zip(first.model.get_weights(), reference.get_weights()):
        np.testing.assert_array_equal(got, want)
    assert json.loads((directory / "selected_checkpoint.json").read_text()) == first.selection
    second, _ = fit(tmp_path, epochs=3)
    assert second.selection == first.selection


def test_crash_window_with_altered_orphan_val_loss_retrains_the_epoch(tmp_path):
    _, directory = fit(tmp_path, epochs=2)
    original = json.loads(G.epoch_record_path(directory, 2).read_text())
    # The process "died" after writing the epoch-2 record, before state.json advanced;
    # afterwards the orphan's val_loss is altered while its checkpoint is not.
    state = json.loads((directory / "state.json").read_text())
    G.write_state(directory / "state.json", identity=state["identity"], total_epochs=2, completed=1,
                  chain_head=I.sha256_file(G.epoch_record_path(directory, 1)), retries=[])
    (directory / "selected_checkpoint.json").unlink()   # a crash precedes selection
    edit_record(directory, 2, val_loss=-1.0, logs=dict(original["logs"], val_loss=-1.0))

    second, _ = fit(tmp_path, epochs=2)
    retries = second.selection["retries"]
    assert [(r["epoch"], r["reason"]) for r in retries] == [(2, "complete_uncommitted_orphan")]
    quarantined = json.loads((directory / retries[0]["quarantine"] / "checkpoint_epoch_002.metrics.json").read_text())
    assert quarantined["val_loss"] == -1.0                       # preserved for investigation
    retrained = json.loads(G.epoch_record_path(directory, 2).read_text())
    assert retrained["val_loss"] != -1.0                         # never adopted
    assert second.selection["selected_val_loss"] != -1.0
    assert json.loads((directory / "state.json").read_text())["retries"] == retries


def test_resume_against_substituted_partition_is_refused(tmp_path):
    _, directory = fit(tmp_path, epochs=1)
    x, y, _, _ = tiny()
    swapped = x.copy()
    swapped[0] ^= 1
    with pytest.raises(I.IntegrityError, match="different fit"):
        fit(tmp_path, epochs=1, x=swapped, y=y)
    assert json.loads((directory / "state.json").read_text())["completed_epochs"] == 1


def test_resume_refuses_tampered_committed_checkpoint(tmp_path):
    _, directory = fit(tmp_path, epochs=2)
    G.epoch_checkpoint_path(directory, 1).write_bytes(b"replaced")
    with pytest.raises(I.IntegrityError, match="hashes to"):
        fit(tmp_path, epochs=2)


def test_adapter_refuses_test_data():
    _, _, vx, vy = tiny()
    with pytest.raises(ValueError, match="test data"):
        G.GohrD4Adapter(validation_x=vx, validation_y=vy, evaluation_digest="f" * 64, test_x=vx, test_y=vy)


# ============================================================ driver stages


def test_protocol_check_reports_pending_draft(capsys):
    assert driver.stage_protocol_check(driver.parse_args(["protocol-check"])) == 0
    assert driver.stage_protocol_check(driver.parse_args(["protocol-check", "--require-approved"])) == 3
    assert "PENDING" in capsys.readouterr().out


def test_preflight_refuses_the_unapproved_draft(tmp_path):
    draft = driver.DEFAULT_PREREGISTRATION
    code = driver.main(["preflight", "--preregistration", str(draft),
                        "--preregistration-sha256", I.sha256_file(draft), "--run-dir", str(tmp_path / "run")])
    assert code == 1 and not (tmp_path / "run").exists()


APPROVED = {"decision_rule": "directional_iut_v2", "alpha": 0.05, "effect_threshold": 0.01,
            "replicates": 2, "sample_sizes": {"train": 512, "validation": 128, "test": 256},
            "epochs": 2, "partition_overlap_policy": "refuse_on_any_cross_partition_feature_row",
            "within_partition_duplicates_policy": "record_only",
            "label_conflict_reporting": "report_separately_non_gating",
            "failure_handling": "abort_whole_run_no_exclusions", "calibration": "none_required"}


def stub_gates(mp, anchor="a" * 64):
    """Stub only the git/approval gate and source identity (tested in test_d4_integrity)."""
    mp.setattr(driver, "DEPTH", 1)
    mp.setattr(driver, "load_approved_preregistration", lambda p, a: (
        {}, dict(APPROVED, _commit={"path": "p.json", "sha": "0" * 40}), anchor))
    mp.setattr(driver, "source_identity", lambda files: {"git_head": "0" * 40, "files": {
        "x.py": {"sha256": "1" * 64, "clean_at_head": True}}})
    return anchor


def stage_args(run_dir, anchor="a" * 64):
    return ["--preregistration", "p.json", "--preregistration-sha256", anchor, "--run-dir", str(run_dir)]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """One complete preflight -> run on tiny real Gohr data, shared by the verify tests."""
    mp = pytest.MonkeyPatch()
    stub_gates(mp)
    run_dir = tmp_path_factory.mktemp("e2e") / "d4_run"
    assert driver.main(["preflight", *stage_args(run_dir)]) == 0
    assert driver.main(["run", *stage_args(run_dir)]) == 0
    yield run_dir, mp
    mp.undo()


def test_end_to_end_verify_passes(completed):
    run_dir, _ = completed
    summary = driver.verify_run(run_dir)
    assert summary["replicates"] == 2
    preflight = json.loads((run_dir / "preflight.json").read_text())
    assert preflight["overlap_audit"]["rows"] == {"train": 512, "validation": 128, "test": 256}
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    assert certificate["provenance"]["partitions"] == preflight["partitions"]


def test_completed_run_and_preflight_are_never_replaced(completed):
    run_dir, _ = completed
    before = I.sha256_file(run_dir / "result" / "d4_certificate.json")
    assert driver.main(["run", *stage_args(run_dir)]) == 1
    assert driver.main(["preflight", *stage_args(run_dir)]) == 1
    assert I.sha256_file(run_dir / "result" / "d4_certificate.json") == before


def _bump_val_loss(path):
    record = json.loads(path.read_text())
    record["val_loss"] = record["logs"]["val_loss"] = record["val_loss"] / 2
    path.write_text(json.dumps(record))


@pytest.mark.parametrize("target, mutate, match", [
    ("fits/replicate_01/perturbed/checkpoint_epoch_001.keras", lambda p: p.write_bytes(b"x"), "hashes to"),
    ("fits/replicate_02/baseline/checkpoint_epoch_001.metrics.json", _bump_val_loss, "does not chain"),
    ("dataset/test_y.npy", lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])),
     "does not match preflight"),
    ("result/d4_certificate.json", lambda p: p.write_text(p.read_text().replace('"seed"', '"seed" ', 1)),
     "does not match the run manifest"),
    ("fits/perturbations/replicate_01_labels.npy",
     lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])), "digest"),
])
def test_verify_detects_tampering(completed, tmp_path, target, mutate, match):
    run_dir, _ = completed
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    mutate(copy / target)
    with pytest.raises(I.IntegrityError, match=match):
        driver.verify_run(copy)


def test_preflight_overlap_refusal_keeps_data_and_writes_no_run_dir(tmp_path, monkeypatch):
    stub_gates(monkeypatch)
    run_dir = tmp_path / "d4_run"
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, (40, 64), dtype=np.uint8)
    y = rng.integers(0, 2, 40, dtype=np.uint8)
    monkeypatch.setattr(driver, "generate_gohr_partitions", lambda sizes: {
        "train": (x, y), "validation": (x[:10].copy(), y[:10].copy()), "test": (x[30:].copy(), y[30:].copy())})
    assert driver.main(["preflight", *stage_args(run_dir)]) == 4
    assert not run_dir.exists()
    refused = json.loads(run_dir.with_name(run_dir.name + ".preflight_refused.json").read_text())
    assert refused["outcome"] == "REFUSED"
    assert refused["overlap_audit"]["cross"]["train__test"]["rows"] == 10
    kept = run_dir.with_name(run_dir.name + ".preflight_refused_dataset")
    assert np.array_equal(np.load(kept / "dataset" / "train_x.npy"), x)
    assert driver.main(["preflight", *stage_args(run_dir)]) == 1   # not silently retried


def test_input_encoding_is_checked_from_arrays():
    good = np.zeros((4, 64), dtype=np.uint8), np.zeros(4, dtype=np.uint8)
    driver.validate_input_encoding({"train": good})
    for bad, match in [((good[0].astype(np.float32), good[1]), "dtype"),
                       ((np.full((4, 64), 255, np.uint8), good[1]), "rescaled"),
                       ((np.zeros((4, 32), np.uint8), good[1]), "width"),
                       ((good[0], np.full(4, 2, np.uint8)), "outside"),
                       ((good[0], np.zeros(3, np.uint8)), "one label per row")]:
        with pytest.raises(I.IntegrityError, match=match):
            driver.validate_input_encoding({"train": bad})
