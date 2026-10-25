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


APPROVED = {"decision_rule": "near_chance_baseline_gated_v4", "chance_margin": 0.2, "paired_endpoint_role": "secondary",
            "baseline_adequacy_threshold": 0.9, "planning_source_attestation": "confirmed_by_protocol_owner",
            "alpha": 0.05, "effect_threshold": 0.01,
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
    ("fits/perturbations/replicate_01_indices.npy",
     lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])), "digest"),
    ("fits/predictions/replicate_02_perturbed.npy",
     lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])), "recorded digest"),
])
def test_verify_detects_tampering(completed, tmp_path, target, mutate, match):
    run_dir, _ = completed
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    mutate(copy / target)
    with pytest.raises(I.IntegrityError, match=match):
        driver.verify_run(copy)


def _rewrite_certificate(run_dir, mutate):
    """Edit the certificate AND re-hash the manifest, so only the design cross-checks can catch it."""
    path = run_dir / "result" / "d4_certificate.json"
    certificate = json.loads(path.read_text())
    mutate(certificate)
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True))
    manifest_path = run_dir / "result" / "d4_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["d4_certificate.json"] = I.sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c["settings"].update(chance_margin=0.3), "chance_margin differs"),
    (lambda c: c["settings"].update(alpha=0.1), "alpha differs"),
    (lambda c: c["settings"].update(paired_endpoint_role="co_primary"), "paired_endpoint_role differs"),
    (lambda c: c["settings"].update(baseline_adequacy_threshold=0.6), "baseline_adequacy_threshold differs"),
    (lambda c: c["methodology"]["baseline_procedure"].update(adequacy_test="two_sided"), "procedure"),
    (lambda c: c["findings"]["replicates"][0]["perturbed_predictions"]["class_counts"]["label_1"].update(rows=0),
     "class counts differ"),
    (lambda c: c["settings"].update(decision_rule="directional_iut_v2"), "decision_rule differs"),
    (lambda c: c["settings"].update(test_examples=255), "test sample size"),
    (lambda c: c["methodology"]["primary_procedure"].update(equivalence_test="one_sided_only"), "procedure"),
    (lambda c: c["findings"]["replicates"][0].update(perturbed_correct=c["findings"]["replicates"][0]
                                                     ["perturbed_correct"] + 1), "persisted predictions give"),
])
def test_verify_refuses_a_certificate_that_departs_from_the_approved_design(completed, tmp_path, mutate, match):
    run_dir, _ = completed
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    _rewrite_certificate(copy, mutate)
    with pytest.raises(I.IntegrityError, match=match):
        driver.verify_run(copy)


def test_verify_reports_the_primary_outcome_and_never_calls_a_non_supported_run_a_pass(completed, capsys):
    run_dir, _ = completed
    summary = driver.verify_run(run_dir)
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    assert summary["integrity"] == "VERIFIED"
    assert summary["outcome"] == certificate["decision"]["outcome"]
    assert summary["overall_supported"] is (summary["outcome"] == "SUPPORTED")
    assert summary["overall_d4_outcome"] == certificate["decision"]["overall_d4_outcome"]
    assert summary["baseline_adequacy_gate"] == certificate["decision"]["baseline_adequacy_gate"]
    assert summary["shuffled_near_chance_endpoint"] == certificate["decision"]["shuffled_near_chance_endpoint"]
    assert summary["paired_degradation_endpoint"]["role"] == "secondary_non_gating"
    rows = certificate["findings"]["primary_endpoint"]["replicates"]
    assert len(rows) == 2 and all("equivalence_interval" in r and "accuracy" in r for r in rows)
    baselines = certificate["findings"]["baseline_adequacy"]["replicates"]
    assert len(baselines) == 2 and all("one_sided_lower_bound" in r for r in baselines)
    assert driver.main(["verify", "--run-dir", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert ("NOT a scientific pass" in out) is (summary["outcome"] != "SUPPORTED")


def test_label_shuffle_end_to_end_permutes_only_the_training_labels(completed):
    """Only training labels change; features, validation, test and every other controlled variable do not."""
    from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

    run_dir, _ = completed
    preflight, partitions = driver.load_preflight(run_dir)
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = (partitions[p] for p in driver.PARTITIONS)
    source = I.labelled_partition_digest(train_x, train_y)
    seeds = np.random.SeedSequence(driver.AUDIT_SEED).spawn(len(certificate["findings"]["replicates"]))
    for row, child in zip(certificate["findings"]["replicates"], seeds):
        r = row["replicate"]
        indices = np.load(run_dir / "fits" / "perturbations" / f"replicate_{r:02d}_indices.npy")
        planned = LabelShufflePerturbation().permutation_indices(len(train_y), np.random.default_rng(child))
        np.testing.assert_array_equal(indices, planned)                           # the audit-seed plan
        assert np.array_equal(np.sort(indices), np.arange(len(train_y)))          # a bijection
        assert not np.array_equal(indices, np.arange(len(train_y)))               # not the identity
        shuffled = np.asarray(train_y)[indices]
        assert np.array_equal(np.bincount(shuffled), np.bincount(train_y))        # label multiset kept
        assert np.any(shuffled != train_y)
        base, pert = row["baseline_selection"], row["perturbed_selection"]
        assert base["train_digest"] == source                                     # baseline: original data
        assert pert["train_digest"] == I.labelled_partition_digest(train_x, shuffled)  # same features, new labels
        assert row["perturbation_record"]["perturbed_train_digest"] == pert["train_digest"]
        assert row["perturbation_record"]["source_train_digest"] == source
        assert {k: row["perturbation_record"][k] for k in ("perturbation", "condition", "replicate", "seed")} == \
            {"perturbation": "label_shuffle", "condition": "perturbed", "replicate": r, "seed": row["seed"]}
        for key in ("seed", "replicate", "config_hash", "validation_digest", "evaluation_digest", "model"):
            assert base[key] == pert[key], key                                    # all else controlled
        assert base["validation_digest"] == I.labelled_partition_digest(val_x, val_y)
        assert base["evaluation_digest"] == preflight["partitions"]["test"]["digest"]
    # Ground truth used for scoring is exactly the preflight test labels.
    assert I.labelled_partition_digest(test_x, test_y) == preflight["partitions"]["test"]["digest"]
    assert certificate["provenance"]["test_label_counts"] == {"zeros": int(np.sum(test_y == 0)),
                                                              "ones": int(np.sum(test_y == 1))}


def test_selected_checkpoint_is_the_val_loss_argmin_and_its_reload_reproduces_the_predictions(completed):
    run_dir, _ = completed
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    for row in certificate["findings"]["replicates"]:
        for condition in ("baseline", "perturbed"):
            selection = row[f"{condition}_selection"]
            directory, _ = driver.d4.checkpoint_paths(run_dir / "fits", row["replicate"], condition)
            losses = [json.loads(G.epoch_record_path(directory, e).read_text())["val_loss"]
                      for e in range(1, APPROVED["epochs"] + 1)]
            assert selection["selected_epoch"] == 1 + int(np.argmin(losses))
            assert selection["checkpoint_sha256"] == I.sha256_file(directory / selection["checkpoint"])
    report = driver.rerun_selected_models(run_dir)
    assert len(report) == 2 * len(certificate["findings"]["replicates"])
    assert all(entry["label_disagreements"] == 0 for entry in report)
    assert driver.main(["verify", "--run-dir", str(run_dir), "--rerun-models"]) == 0


def test_rerun_detects_predictions_not_produced_by_the_selected_checkpoint(completed, monkeypatch):
    run_dir, _ = completed
    real = driver._load_selected_model

    class Inverted:
        def __init__(self, model):
            self.model = model

        def predict(self, x, verbose=0):
            return 1.0 - self.model.predict(x, verbose=verbose)

    monkeypatch.setattr(driver, "_load_selected_model", lambda path: Inverted(real(path)))
    with pytest.raises(I.IntegrityError, match="does not reproduce the persisted predictions"):
        driver.rerun_selected_models(run_dir)


def test_verify_without_a_result_is_not_completed(tmp_path, capsys):
    (tmp_path / "run").mkdir()
    assert driver.main(["verify", "--run-dir", str(tmp_path / "run")]) == 5
    assert "NOT_COMPLETED" in capsys.readouterr().out


def overlapping_partitions(sizes):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, (40, 64), dtype=np.uint8)
    y = rng.integers(0, 2, 40, dtype=np.uint8)
    return {"train": (x, y), "validation": (x[:10].copy(), y[:10].copy()),
            "test": (x[30:].copy(), y[30:].copy())}


def test_preflight_overlap_refusal_keeps_data_and_writes_no_run_dir(tmp_path, monkeypatch):
    stub_gates(monkeypatch)
    run_dir = tmp_path / "d4_run"
    monkeypatch.setattr(driver, "generate_gohr_partitions", overlapping_partitions)
    assert driver.main(["preflight", *stage_args(run_dir)]) == 4
    assert not run_dir.exists()
    refused_dir = driver.refused_directory(run_dir)
    refused = json.loads((refused_dir / "preflight.json").read_text())
    assert refused["outcome"] == "REFUSED"
    assert refused["overlap_audit"]["cross"]["train__test"]["rows"] == 10
    assert np.array_equal(np.load(refused_dir / "dataset" / "train_x.npy"), overlapping_partitions(None)["train"][0])
    assert driver.main(["preflight", *stage_args(run_dir)]) == 1   # not silently retried
    with pytest.raises(I.IntegrityError, match="not a passing"):
        driver.load_preflight(refused_dir)                          # a refusal is never usable as a run


# ============================================================ crash-safe preflight persistence


def small_partitions(sizes):
    rng = np.random.default_rng(2)
    return {r: (rng.integers(0, 2, (n, 64), dtype=np.uint8), rng.integers(0, 2, n, dtype=np.uint8))
            for r, n in (("train", 64), ("validation", 32), ("test", 32))}


def test_failure_during_dataset_write_leaves_nothing_accepted(tmp_path, monkeypatch):
    stub_gates(monkeypatch)
    monkeypatch.setattr(driver, "generate_gohr_partitions", small_partitions)
    run_dir = tmp_path / "d4_run"
    real_open, calls = Path.open, {"n": 0}

    def failing_open(self, mode="r", *a, **k):
        if "w" in mode and ".staging_" in str(self):
            calls["n"] += 1
            if calls["n"] == 4:
                raise OSError("disk full (simulated)")
        return real_open(self, mode, *a, **k)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="simulated"):
        driver.main(["preflight", *stage_args(run_dir)])
    monkeypatch.setattr(Path, "open", real_open)
    assert not run_dir.exists()
    assert I.stale_staging_directories(run_dir) == []              # in-process failure cleans up
    assert driver.main(["preflight", *stage_args(run_dir)]) == 0   # a fresh, complete preflight is possible
    driver.load_preflight(run_dir)


def test_hard_crash_leftover_is_never_accepted_and_blocks_later_stages(tmp_path, monkeypatch):
    stub_gates(monkeypatch)
    monkeypatch.setattr(driver, "generate_gohr_partitions", small_partitions)
    run_dir = tmp_path / "d4_run"
    # Simulate a crash after some files were written but before the rename:
    stale = tmp_path / ".d4_run.staging_crashed"
    (stale / "dataset").mkdir(parents=True)
    (stale / "dataset" / "train_x.npy").write_bytes(b"partial")
    assert driver.main(["preflight", *stage_args(run_dir)]) == 1
    assert not run_dir.exists() and (stale / "dataset" / "train_x.npy").read_bytes() == b"partial"
    assert driver.main(["run", *stage_args(run_dir)]) == 1


def test_crash_during_rename_cannot_expose_a_partial_run_dir(tmp_path, monkeypatch):
    stub_gates(monkeypatch)
    monkeypatch.setattr(driver, "generate_gohr_partitions", small_partitions)
    run_dir = tmp_path / "d4_run"
    monkeypatch.setattr(I.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError("crash (simulated)")))
    with pytest.raises(OSError, match="simulated"):
        driver.main(["preflight", *stage_args(run_dir)])
    assert not run_dir.exists()                                    # claimed empty dir is released


@pytest.mark.parametrize("mutate, match", [
    (lambda d: (d / "preflight.json").unlink(), "missing"),
    (lambda d: (d / "dataset" / "extra.npy").write_bytes(b"x"), "unexpected"),
    (lambda d: (d / "dataset" / "validation_y.npy").unlink(), "missing"),
    (lambda d: (d / "preflight.json").write_text('{"schema_version"'), "truncated or malformed"),
])
def test_incomplete_or_altered_preflight_directory_is_refused(tmp_path, monkeypatch, mutate, match):
    stub_gates(monkeypatch)
    monkeypatch.setattr(driver, "generate_gohr_partitions", small_partitions)
    run_dir = tmp_path / "d4_run"
    assert driver.main(["preflight", *stage_args(run_dir)]) == 0
    mutate(run_dir)
    with pytest.raises(I.IntegrityError, match=match):
        driver.load_preflight(run_dir)
    assert driver.main(["run", *stage_args(run_dir)]) == 1


# ============================================================ wiring and evidence identity

PINNED_GOHR_ADAPTER_SHA256 = "3c0b7ac5327c68233ac792062f3559232cfc965b8a403312e552bcb2b516b488"


def test_experiment_trains_with_the_d4_adapter_not_the_pinned_base(tmp_path):
    _, _, vx, vy = tiny()
    factory = driver.make_adapter_factory(tmp_path, {"epochs": 2}, {"validation": (vx, vy)}, "f" * 64)
    adapter = factory(5)
    assert type(adapter) is G.GohrD4Adapter
    assert adapter.test_x is None and adapter.evaluation_digest == "f" * 64
    import inspect
    source = inspect.getsource(driver)
    # The pinned base adapter is used only to generate data, never to train.
    assert source.count("from audit.dataset.adapters.gohr import GohrAdapter") == 1
    assert "GohrAdapter(num_rounds=NUM_ROUNDS, depth=DEPTH)" in inspect.getsource(driver.generate_gohr_partitions)


def test_completed_run_was_trained_by_the_d4_lifecycle(completed):
    run_dir, _ = completed
    for r in (1, 2):
        for condition in ("baseline", "perturbed"):
            directory = run_dir / "fits" / f"replicate_{r:02d}" / condition
            assert json.loads((directory / "state.json").read_text())["schema_version"] == G.STATE_SCHEMA
            assert json.loads((directory / "selected_checkpoint.json").read_text())["rule"] == G.MODEL_SELECTION_RULE
    assert not list((run_dir / "fits").rglob("best_*"))


def test_pinned_gohr_adapter_is_byte_identical():
    assert I.sha256_file(I.REPOSITORY_ROOT / "audit/dataset/adapters/gohr.py") == PINNED_GOHR_ADAPTER_SHA256


SEALED = ["audit/dataset/evidence/d0_gohr_speck_r5_production_730516c79981/d0_gohr_build_attestation.json",
          "audit/dataset/evidence/d3_gohr_speck3264_r5_confirmatory_730516c79981/d3_gohr_certificate.json"]


@pytest.mark.parametrize("record", SEALED)
def test_every_source_pinned_by_sealed_d0_d3_evidence_still_matches(record):
    path = I.REPOSITORY_ROOT / record
    if not path.is_file():
        pytest.skip(f"sealed evidence not present on this machine: {record}")
    pins = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith(".py") and isinstance(value, dict) and "sha256" in value:
                    pins.append((key, value["sha256"]))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(path.read_text()))
    assert pins, "no source pins found"
    mismatched = [k for k, h in pins if I.sha256_file(I.REPOSITORY_ROOT / k) != h]
    assert mismatched == []


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
