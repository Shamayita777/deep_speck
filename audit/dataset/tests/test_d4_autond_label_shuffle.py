"""
AutoND D4 adapter lifecycle, driver stages and draft protocol.

SOFTWARE VALIDATION, NOT AN EXPERIMENT.  Everything here runs on tiny data
(hundreds of rows, two stages, two epochs): synthetic record chains for the
recovery table, and the real pinned AutoND generator and DBitNet for the
end-to-end stage tests.  No result here says anything about AutoND's
distinguisher.

Run this file in its OWN pytest process: the pinned AutoND binding refuses to
import while another component's top-level ``train_nets`` (e.g. Gohr's, loaded
by the Gohr D4 tests) occupies that module name, and it never evicts it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

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


def _autond_root():
    for candidate in (os.environ.get("AUTOND_ROOT"), "~/Documents/AutoND-upstream"):
        if candidate and (Path(candidate).expanduser() / "main.py").is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


ROOT = _autond_root()
needs_checkout = pytest.mark.skipif(ROOT is None, reason="pinned AutoND checkout not found; set AUTOND_ROOT. "
                                                        "SKIPPED IS NOT PASSED.")

ROUNDS, E, THR = [7, 8], 2, 0.505
PLAN = A.step_plan(ROUNDS, E)
IDENTITY = {"seed": 7, "condition": "baseline", "replicate": 1, "config_hash": "c" * 64, "train_digest": "d" * 64,
            "round_train_digests": ["d1", "d2"], "validation_digests": ["v1", "v2"], "evaluation_digest": "f" * 64,
            "model": {"architecture": "dbitnet", "rounds": ROUNDS}, "run_binding": {"preflight_sha256": "p" * 64},
            "adapter": A.ADAPTER_IDENTITY}
RUN_BINDING = {"preflight_sha256": "p" * 64}
PASS, FAIL = 0.60, 0.50   # round max val_acc above / at-or-below the stop threshold


# ============================================================ synthetic chains (recovery table)


def write_chain(fit_dir: Path, losses, overrides=None, identity=IDENTITY, accs=None) -> str:
    head = A.GENESIS_RECORD_SHA256
    accs = accs or [PASS] * len(losses)
    for step, (loss, acc) in enumerate(zip(losses, accs), start=1):
        stage_round, stage_epoch = PLAN[step - 1]
        checkpoint = A.epoch_checkpoint_path(fit_dir, stage_round, stage_epoch)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"weights-{step}".encode())
        record = {"schema_version": A.RECORD_SCHEMA, "step": step, "stage_round": stage_round,
                  "stage_epoch": stage_epoch, "checkpoint": checkpoint.relative_to(fit_dir).as_posix(),
                  "checkpoint_sha256": I.sha256_file(checkpoint), "val_loss": float(loss), "val_acc": float(acc),
                  "logs": {"val_loss": float(loss), "val_acc": float(acc)}, "previous_record_sha256": head, **identity}
        record.update((overrides or {}).get(step, {}))
        path = A.epoch_record_path(fit_dir, stage_round, stage_epoch)
        path.write_text(json.dumps(record))
        head = I.sha256_file(path)
    return head


def commit(fit_dir: Path, steps: int, retries=()):
    head = A.GENESIS_RECORD_SHA256
    if steps:
        _, head = A.verify_epoch_chain(fit_dir, plan=PLAN, steps=steps, identity=IDENTITY)
    A.write_state(fit_dir / "state.json", identity=IDENTITY, total_steps=len(PLAN), completed=steps,
                  chain_head=head, retries=list(retries))
    return head


def select(fit_dir, head, completed):
    return A.select_checkpoints(fit_dir, plan=PLAN, rounds=ROUNDS, epochs_per_round=E, threshold=THR,
                                identity=IDENTITY, chain_head=head, completed=completed)


def test_full_progression_selects_per_round_and_compares_at_the_first_round(tmp_path):
    head = write_chain(tmp_path, [0.30, 0.20, 0.25, 0.26], accs=[0.55, 0.61, 0.52, 0.51])
    selection = select(tmp_path, head, 4)
    prog = selection["progression"]
    assert prog["status"] == "cap_reached" and prog["rounds_trained"] == [7, 8] and prog["best_round"] == 8
    assert [(s["stage_round"], s["selected_epoch"]) for s in prog["summaries"]] == [(7, 2), (8, 1)]
    assert selection["comparison_round"] == selection["selected_stage_round"] == 7
    assert selection["checkpoint"] == "stage_r07/checkpoint_epoch_002.keras"
    assert all(selection[key] == IDENTITY[key] for key in A.IDENTITY_FIELDS)


def test_stop_uses_the_round_max_val_acc_not_the_selected_checkpoints(tmp_path):
    # Round 7: selected epoch (min val_loss) has val_acc 0.50, but the round's max is 0.51 > 0.505 -> continue.
    head = write_chain(tmp_path, [0.20, 0.30, 0.25, 0.26], accs=[0.50, 0.51, 0.50, 0.50])
    prog = select(tmp_path, head, 4)["progression"]
    assert prog["summaries"][0]["selected_val_acc"] == 0.50 and prog["summaries"][0]["max_val_acc"] == 0.51
    assert prog["rounds_trained"] == [7, 8] and prog["status"] == "stopped_at_threshold" and prog["stop_round"] == 8
    assert prog["best_round"] == 7


@pytest.mark.parametrize("accs, status, stop, best", [
    ([0.505, 0.50], "stopped_at_threshold", 7, None),      # at the threshold stops (<=), shuffled-arm pattern
    ([0.49, 0.50], "stopped_at_threshold", 7, None),
])
def test_first_round_failure_stops_the_pipeline_with_no_best_round(tmp_path, accs, status, stop, best):
    head = write_chain(tmp_path, [0.3, 0.2], accs=accs)
    selection = select(tmp_path, head, 2)
    prog = selection["progression"]
    assert (prog["status"], prog["stop_round"], prog["best_round"]) == (status, stop, best)
    assert prog["rounds_trained"] == [7] and selection["comparison_round"] == 7


def test_training_past_a_stop_or_stopping_early_is_refused(tmp_path):
    head = write_chain(tmp_path, [0.3, 0.2, 0.25, 0.26], accs=[0.50, 0.50, 0.6, 0.6])   # round 7 failed yet 8 trained
    with pytest.raises(I.IntegrityError, match="continued past a stop"):
        select(tmp_path, head, 4)
    other = tmp_path / "early"
    head = write_chain(other, [0.3, 0.2], accs=[0.6, 0.6])                              # round 7 passed, fit ended
    with pytest.raises(I.IntegrityError, match="has not finished"):
        select(other, head, 2)


def test_artifacts_after_a_stop_are_refused(tmp_path):
    write_chain(tmp_path, [0.3, 0.2], accs=[0.50, 0.50])
    head = commit(tmp_path, 2)
    A.epoch_checkpoint_path(tmp_path, 8, 1).parent.mkdir(parents=True, exist_ok=True)
    A.epoch_checkpoint_path(tmp_path, 8, 1).write_bytes(b"x")
    with pytest.raises(I.IntegrityError, match="after the pipeline stopped"):
        A.classify_uncommitted(tmp_path, plan=PLAN, committed=2, chain_head=head, identity=IDENTITY, finished=True)


def test_ties_select_the_earliest_epoch(tmp_path):
    head = write_chain(tmp_path, [0.2, 0.2, 0.3, 0.3])
    prog = select(tmp_path, head, 4)["progression"]
    assert [s["selected_epoch"] for s in prog["summaries"]] == [1, 1]


def test_val_acc_is_required_and_bound_to_its_logs(tmp_path):
    write_chain(tmp_path, [0.3, 0.2, 0.25, 0.26])
    commit(tmp_path, 4)
    _edit(A.epoch_record_path(tmp_path, 7, 1), val_acc=0.9)
    with pytest.raises(I.IntegrityError, match="val_acc disagrees"):
        A.load_committed_state(tmp_path / "state.json", tmp_path, identity=IDENTITY, plan=PLAN)


@pytest.mark.parametrize("mutate, match", [
    (lambda d: A.epoch_checkpoint_path(d, 7, 2).write_bytes(b"x"), "hashes to"),
    (lambda d: _edit(A.epoch_record_path(d, 8, 1), val_loss=0.01, logs={"val_loss": 0.01, "val_acc": PASS}),
     "does not chain"),
    (lambda d: _edit(A.epoch_record_path(d, 7, 1), seed=8), "Refusing to mix fits"),
    (lambda d: _edit(A.epoch_record_path(d, 7, 1), round_train_digests=["x", "y"]), "Refusing to mix fits"),
    (lambda d: A.epoch_record_path(d, 7, 2).write_text("{"), "truncated|malformed|JSON|parse"),
])
def test_tampered_committed_history_is_refused(tmp_path, mutate, match):
    write_chain(tmp_path, [0.3, 0.2, 0.25, 0.26])
    commit(tmp_path, 4)
    mutate(tmp_path)
    with pytest.raises(I.IntegrityError, match=match):
        A.load_committed_state(tmp_path / "state.json", tmp_path, identity=IDENTITY, plan=PLAN)


def _edit(path, **changes):
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def test_rewritten_consistent_chain_disagrees_with_committed_head(tmp_path):
    write_chain(tmp_path, [0.3, 0.2, 0.25, 0.26])
    head = commit(tmp_path, 4)
    write_chain(tmp_path, [0.3, 0.2, 0.25, 0.01])
    with pytest.raises(I.IntegrityError, match="chain head"):
        A.load_committed_state(tmp_path / "state.json", tmp_path, identity=IDENTITY, plan=PLAN)
    with pytest.raises(I.IntegrityError, match="does not end at the head"):
        select(tmp_path, head, 4)


def test_orphan_is_quarantined_never_adopted_and_nothing_is_deleted(tmp_path):
    write_chain(tmp_path, [0.3, 0.2, 0.25])
    head = commit(tmp_path, 2)                       # step 3 (stage 8, epoch 1) is an orphan
    found = A.classify_uncommitted(tmp_path, plan=PLAN, committed=2, chain_head=head, identity=IDENTITY)
    assert found["reason"] == "complete_uncommitted_orphan" and found["step"] == 3
    entry = A.quarantine(tmp_path, found)
    assert entry["stage_round"] == 8 and len(entry["files"]) == 2
    assert not A.epoch_record_path(tmp_path, 8, 1).exists()
    assert sorted(p.name for p in (tmp_path / entry["quarantine"]).iterdir()) == sorted(entry["files"] + ["note.json"])


@pytest.mark.parametrize("make, match", [
    (lambda d: A.epoch_checkpoint_path(d, 8, 2).write_bytes(b"x"), "only step 3"),            # beyond next step
    (lambda d: (A.epoch_checkpoint_path(d, 8, 1).unlink()), "record but no checkpoint"),
    (lambda d: (d / "stage_r09").mkdir() or (d / "stage_r09" / "checkpoint_epoch_001.keras").write_bytes(b"x"),
     "no registered"),
    (lambda d: _edit(A.epoch_record_path(d, 8, 1), condition="perturbed"), "Refusing to mix fits"),
])
def test_suspicious_uncommitted_artifacts_are_refused_and_left_in_place(tmp_path, make, match):
    write_chain(tmp_path, [0.3, 0.2, 0.25])
    head = commit(tmp_path, 2)
    make(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    with pytest.raises(I.IntegrityError, match=match):
        A.classify_uncommitted(tmp_path, plan=PLAN, committed=2, chain_head=head, identity=IDENTITY)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_checkpoint_without_record_and_incomplete_write_are_quarantinable(tmp_path):
    write_chain(tmp_path, [0.3, 0.2])
    head = commit(tmp_path, 2)
    A.epoch_checkpoint_path(tmp_path, 8, 1).parent.mkdir(parents=True, exist_ok=True)
    A.epoch_checkpoint_path(tmp_path, 8, 1).write_bytes(b"partial")
    assert A.classify_uncommitted(tmp_path, plan=PLAN, committed=2, chain_head=head,
                                  identity=IDENTITY)["reason"] == "checkpoint_without_record"
    A.epoch_checkpoint_path(tmp_path, 8, 1).unlink()
    A.epoch_record_path(tmp_path, 8, 1).with_name("checkpoint_epoch_001.metrics.json.tmp").write_text("{")
    assert A.classify_uncommitted(tmp_path, plan=PLAN, committed=2, chain_head=head,
                                  identity=IDENTITY)["reason"] == "incomplete_write"


def test_state_from_another_fit_is_refused(tmp_path):
    write_chain(tmp_path, [0.3, 0.2, 0.25, 0.26])
    commit(tmp_path, 4)
    with pytest.raises(I.IntegrityError, match="different fit"):
        A.load_committed_state(tmp_path / "state.json", tmp_path, identity=dict(IDENTITY, replicate=2), plan=PLAN)


# ============================================================ sampling design (read from the pinned source)


@needs_checkout
def test_pinned_generator_samples_iid_rows_with_unfixed_class_counts():
    source = (Path(ROOT) / "main.py").read_text()
    for line in ("Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;",
                 "keys0 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1)",
                 "C[Y==0] = (np.frombuffer(urandom(num_rand_samples*C0.shape[1]*2),dtype=np.uint8)&1)"
                 ".reshape(num_rand_samples, -1)"):
        assert line in source
    binding = driver.make_binding(ROOT, 8)
    counts = {int(np.sum(binding.generate_partition(64)[1])) for _ in range(20)}
    assert len(counts) > 1


@needs_checkout
def test_pinned_training_constants_match_the_registered_design():
    source = (Path(ROOT) / "train_nets.py").read_text()
    for line in ("ABORT_TRAINING_BELOW_ACC = 0.505", "EPOCHS = 10", "NUM_SAMPLES = 10**7",
                 "NUM_VAL_SAMPLES = 10**6", "BATCHSIZE = 5000", "optimizer = tf.keras.optimizers.Adam(amsgrad=True)",
                 "model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])",
                 "model.load_weights(f'{log_prefix}_{model_name}_round{round_number-1}.h5')",
                 "ModelCheckpoint(f'{log_prefix}_{model_name}_round{round_number}.h5', monitor='val_loss', "
                 "save_best_only = True)"):
        assert line in source, line
    for line in ("val_acc = train_one_round(", "if val_acc <= ABORT_TRAINING_BELOW_ACC:",
                 "return np.max(history.history['val_acc'])", "X, Y = data_generator(num_samples, current_round)"):
        assert line in source, line
    draft = shipped()
    assert draft["fixed_design"]["training_procedure"]["stop_threshold"] == 0.505 == driver.STOP_THRESHOLD
    assert draft["decisions"]["epochs_per_round"]["recommended"] == 10
    assert draft["decisions"]["sample_sizes"]["recommended"]["train_per_round"] == 10 ** 7
    assert draft["decisions"]["sample_sizes"]["recommended"]["validation_per_round"] == 10 ** 6


# ============================================================ draft protocol


def shipped():
    return json.loads(driver.DEFAULT_PREREGISTRATION.read_text())


def approve_all_but(record, *left):
    import copy

    record = copy.deepcopy(record)
    for name, entry in record["decisions"].items():
        if name in left:
            entry.update(status=d4.PENDING, value=None)
        else:
            value = entry["recommended"] if entry["recommended"] is not None else entry["options"][0]
            entry.update(status="APPROVED", value=value)
    record.update(approved=True, approved_by="tester", approved_utc="2026-01-01T00:00:00Z",
                  approval_statement="s", data_examined_before_approval="d")
    return record


def test_shipped_protocol_is_well_formed_and_approved(capsys):
    record = shipped()
    review = d4.review_preregistration(record)
    assert review["problems"] == [] and review["pending"] == []
    assert driver.check_target_design(record) == [] and driver.check_decision_values(review["values"]) == []
    assert record["approved"] is True and record["approved_by"] and record["approved_utc"]
    assert review["values"]["baseline_adequacy_threshold"] == 0.9
    assert review["values"]["pipeline_progression_role"] == "co_primary"
    assert "planning_source_attestation" not in record["decisions"]
    assert "d4_operationalization" not in record["methodology_source"]
    assert driver.stage_protocol_check(driver.parse_args(["protocol-check"])) == 0
    assert driver.stage_protocol_check(driver.parse_args(["protocol-check", "--require-approved"])) == 0


def test_recommended_values_follow_the_full_pipeline_design():
    decisions = shipped()["decisions"]
    assert decisions["chance_margin"]["recommended"] == 0.005
    assert decisions["chance_margin"]["recommended"] == pytest.approx(driver.STOP_THRESHOLD - 0.5)
    assert decisions["baseline_adequacy_threshold"]["recommended"] == 0.9
    assert decisions["starting_round"]["recommended"] == 5 and decisions["max_round"]["recommended"] == 9
    assert decisions["sample_sizes"]["recommended"]["test"] == 10 ** 6
    assert decisions["pipeline_progression_role"]["options"] == ["reported_non_gating", "co_primary"]
    assert decisions["pipeline_progression_role"]["recommended"] == "co_primary"
    assert "stage_rounds" not in decisions and "epochs_per_stage" not in decisions


@pytest.mark.parametrize("left", ["chance_margin", "baseline_adequacy_threshold", "starting_round", "max_round",
                                  "sample_sizes", "epochs_per_round", "pipeline_progression_role",
                                  "alpha"])
def test_require_approved_blocks_while_any_decision_is_pending(left):
    with pytest.raises(I.IntegrityError, match=f"decisions.{left}"):
        d4.require_approved_preregistration(approve_all_but(shipped(), left))


def test_fully_approved_copy_passes_engine_and_runner_checks():
    record = approve_all_but(shipped())
    values = d4.require_approved_preregistration(record)
    assert driver.check_target_design(record) == [] and driver.check_decision_values(values) == []


@pytest.mark.parametrize("key, value, match", [
    ("starting_round", 0, "starting_round"), ("max_round", 5, "max_round"),
    ("pipeline_progression_role", "co_gating", "pipeline_progression_role"),
    ("chance_margin", 0.6, "chance_margin"), ("baseline_adequacy_threshold", 0.5, "baseline_adequacy_threshold"),
    ("sample_sizes", {"train": 1, "validation": 1, "test": 1}, "sample_sizes"),
    ("decision_rule", "directional_iut_v2", "decision_rule"), ("epochs_per_round", 0, "epochs_per_round"),
])
def test_invalid_decision_values_are_refused(key, value, match):
    values = d4.require_approved_preregistration(approve_all_but(shipped()))
    values[key] = value
    assert any(match in p for p in driver.check_decision_values(values))


@pytest.mark.parametrize("section, key, value", [
    ("model", "shuffle", False), ("model", "optimizer", "adam"), ("model", "architecture", "gohr_resnet"),
    ("target", "delta_plain_hex", "0x40"), ("training_procedure", "stop_threshold", 0.5),
    ("training_procedure", "comparison_round", "max_round"),
    ("upstream", "commit", "0" * 40), ("model_selection", "rule", "last_epoch"),
])
def test_runner_refuses_a_changed_fixed_design(section, key, value):
    record = approve_all_but(shipped())
    record["fixed_design"][section][key] = value
    assert any(f"{section}.{key}" in p for p in driver.check_target_design(record))


def test_runner_refuses_an_altered_upstream_digest():
    record = approve_all_but(shipped())
    record["fixed_design"]["upstream"]["source_digests"]["dbitnet.py"] = "0" * 64
    assert any("upstream.source_digests" in p for p in driver.check_target_design(record))


# ============================================================ adapter contract (real pinned DBitNet)


@needs_checkout
def test_adapter_refuses_test_data_and_wrong_lengths():
    binding = driver.make_binding(ROOT, 8)
    x = np.zeros((8, 64), np.uint8)
    y = np.zeros(8, np.uint8)
    kwargs = dict(binding=binding, rounds=[7, 8], round_rows=[4, 4], validation={7: (x, y), 8: (x, y)},
                  evaluation_digest="e" * 64, epochs_per_round=1, batch_size=4, seed=1, upstream_commit="u",
                  stop_threshold=0.505, run_binding=RUN_BINDING)
    with pytest.raises(ValueError, match="must not receive test data"):
        A.AutoNDD4Adapter(test_x=x, test_y=y, **kwargs)
    adapter = A.AutoNDD4Adapter(**kwargs)
    with pytest.raises(I.IntegrityError, match="registered rounds need 8"):
        adapter.split_rounds(x[:7], y[:7])
    with pytest.raises(ValueError, match="one validation partition per registered round"):
        A.AutoNDD4Adapter(**dict(kwargs, validation={8: (x, y)}))
    assert Path(adapter._pinned("dbitnet").__file__).resolve().parent == Path(ROOT)
    adapter.check_stop_threshold()
    with pytest.raises(I.IntegrityError, match="differs from the pinned"):
        A.AutoNDD4Adapter(**dict(kwargs, stop_threshold=0.5)).check_stop_threshold()
    with pytest.raises(RuntimeError, match="before train"):
        adapter.predict(x)


# ============================================================ no Gohr-specific runtime dependency


GOHR_MODULE_FILES = ("speck.py", "train_nets.py")   # Gohr's top-level modules at the repository root


def test_autond_d4_sources_import_no_gohr_module():
    import ast

    for path in (Path(A.__file__), Path(driver.__file__)):
        tree = ast.parse(path.read_text())
        names = [n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names]
        names += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not [n for n in names if "gohr" in n.lower() or n.split(".")[0] in ("speck", "train_nets")], path
    assert not [p for p in driver.SOURCE_FILES if "gohr" in p.name.lower()]


@needs_checkout
def test_running_the_autond_path_loads_no_gohr_module():
    """A fresh process binds AutoND, builds the adapter's model and lists every loaded module."""
    import subprocess

    code = (
        "import sys, json\n"
        "import audit.dataset.experiments.d4_autond_label_shuffle as d\n"
        "from audit.dataset.adapters import autond_d4 as A\n"
        f"b = d.make_binding({ROOT!r}, 1)\n"
        "import numpy as np\n"
        "x = np.zeros((2, 64), np.uint8); y = np.zeros(2, np.uint8)\n"
        "a = A.AutoNDD4Adapter(binding=b, rounds=[1], round_rows=[2], validation={1: (x, y)}, evaluation_digest='e',"
        " epochs_per_round=1, batch_size=2, seed=1, upstream_commit='u', stop_threshold=0.505,"
        " run_binding={'preflight_sha256': 'p'})\n"
        "a.check_stop_threshold(); a.build_model()\n"
        "mods = {n: getattr(m, '__file__', None) for n, m in list(sys.modules.items())}\n"
        "print(json.dumps(mods))\n")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(I.REPOSITORY_ROOT), capture_output=True,
                         text=True, timeout=600, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    assert out.returncode == 0, out.stderr[-2000:]
    modules = json.loads(out.stdout.strip().splitlines()[-1])
    # Upstream AutoND ships its own `gohrnet.py` (imported by its train_nets); that is the pinned
    # target's code, not this repository's Gohr code. Every gohr-named module must resolve there.
    root = Path(ROOT).resolve()
    gohr_named = {n: f for n, f in modules.items() if "gohr" in n.lower()}
    assert all(f and root in Path(f).resolve().parents for f in gohr_named.values()), gohr_named
    assert not [n for n in modules if n.startswith("audit.dataset.adapters.gohr") or "d4_gohr" in n]
    repo = str(I.REPOSITORY_ROOT.resolve())
    leaked = [(n, f) for n, f in modules.items()
              if f and str(Path(f).resolve().parent) == repo and Path(f).name in GOHR_MODULE_FILES]
    assert not leaked, leaked
    assert str(Path(modules["train_nets"]).resolve().parent) == str(Path(ROOT).resolve())


# ============================================================ round transition (deterministic, real DBitNet)


@needs_checkout
def test_round_transition_takes_selected_weights_and_last_epoch_optimizer_state(tmp_path):
    import tensorflow as tf

    binding = driver.make_binding(ROOT, 8)
    x, y = binding.generate_partition(64)
    adapter = A.AutoNDD4Adapter(binding=binding, rounds=ROUNDS, round_rows=[64, 64], validation={7: (x, y), 8: (x, y)},
                                evaluation_digest="e", epochs_per_round=E, batch_size=16, seed=3, upstream_commit="u",
                                stop_threshold=THR, run_binding=RUN_BINDING)
    model = adapter.build_model()
    for epoch in (1, 2):                                  # two genuinely different checkpoints of round 7
        model.fit(x, y, epochs=1, batch_size=16, verbose=0)
        path = A.epoch_checkpoint_path(tmp_path, 7, epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
    head = A.GENESIS_RECORD_SHA256
    for step, (loss, acc) in enumerate([(0.10, 0.60), (0.20, 0.70)], start=1):   # epoch 1 selected; round passes
        checkpoint = A.epoch_checkpoint_path(tmp_path, 7, step)
        record = {"schema_version": A.RECORD_SCHEMA, "step": step, "stage_round": 7, "stage_epoch": step,
                  "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
                  "checkpoint_sha256": I.sha256_file(checkpoint), "val_loss": loss, "val_acc": acc,
                  "logs": {"val_loss": loss, "val_acc": acc}, "previous_record_sha256": head, **IDENTITY}
        A.epoch_record_path(tmp_path, 7, step).write_text(json.dumps(record))
        head = I.sha256_file(A.epoch_record_path(tmp_path, 7, step))
    started = adapter._starting_model(tmp_path, completed=2, identity=IDENTITY)
    best = tf.keras.models.load_model(A.epoch_checkpoint_path(tmp_path, 7, 1), compile=True)
    last = tf.keras.models.load_model(A.epoch_checkpoint_path(tmp_path, 7, 2), compile=True)
    assert all(np.array_equal(u, v) for u, v in zip(started.get_weights(), best.get_weights()))
    assert not all(np.array_equal(u, v) for u, v in zip(started.get_weights(), last.get_weights()))
    assert int(started.optimizer.iterations.numpy()) == int(last.optimizer.iterations.numpy())
    assert int(started.optimizer.iterations.numpy()) != int(best.optimizer.iterations.numpy())


# ============================================================ driver stages end to end (tiny, real)


APPROVED = {"decision_rule": "near_chance_baseline_gated_v4", "chance_margin": 0.2, "baseline_adequacy_threshold": 0.505,
            "paired_endpoint_role": "secondary",
            "pipeline_progression_role": "reported_non_gating", "alpha": 0.05, "effect_threshold": 0.01,
            "replicates": 2, "sample_sizes": {"train_per_round": 1024, "validation_per_round": 256, "test": 256},
            "starting_round": 1, "max_round": 2, "epochs_per_round": 2,
            "partition_overlap_policy": "refuse_on_any_cross_partition_feature_row",
            "within_partition_duplicates_policy": "record_only",
            "label_conflict_reporting": "report_separately_non_gating",
            "failure_handling": "abort_whole_run_no_exclusions", "calibration": "none_required"}
TINY_ROUNDS = [1, 2]


def stub_gates(mp, anchor="a" * 64):
    """Stub only the git/approval gate and repository source identity (tested in test_d4_integrity)."""
    mp.setattr(driver, "load_approved_preregistration", lambda p, a: (
        {}, dict(APPROVED, _commit={"path": "p.json", "sha": "0" * 40}), anchor))
    mp.setattr(driver, "source_identity", lambda files: {"git_head": "0" * 40, "files": {
        "x.py": {"sha256": "1" * 64, "clean_at_head": True}}})
    return anchor


def stage_args(run_dir, anchor="a" * 64):
    return ["--preregistration", "p.json", "--preregistration-sha256", anchor, "--run-dir", str(run_dir),
            "--autond-root", ROOT]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    if ROOT is None:
        pytest.skip("pinned AutoND checkout not found; set AUTOND_ROOT. SKIPPED IS NOT PASSED.")
    mp = pytest.MonkeyPatch()
    stub_gates(mp)
    run_dir = tmp_path_factory.mktemp("autond_e2e") / "d4_run"
    assert driver.main(["preflight", *stage_args(run_dir)]) == 0
    assert driver.main(["run", *stage_args(run_dir)]) == 0
    yield run_dir, mp
    mp.undo()


def _copy(run_dir, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(run_dir, target)
    return target


def _certificate(run_dir):
    return json.loads((run_dir / "result" / "d4_certificate.json").read_text())


def test_end_to_end_verify_and_model_reload_pass(completed, capsys):
    run_dir, _ = completed
    summary = driver.verify_run(run_dir)
    assert summary["integrity"] == "VERIFIED" and summary["replicates"] == 2
    assert summary["overall_supported"] is (summary["outcome"] == "SUPPORTED")
    assert summary["pipeline_progression"] == _certificate(run_dir)["autond_pipeline_progression"]["summary"]
    report = driver.rerun_selected_models(run_dir)
    assert len(report) == 4 and all(e["label_disagreements"] == 0 and e["selected_stage_round"] == 1 for e in report)
    assert driver.main(["verify", "--run-dir", str(run_dir), "--rerun-models"]) == 0
    out = capsys.readouterr().out
    assert ("NOT a scientific pass" in out) is (summary["outcome"] != "SUPPORTED")


def test_preflight_records_every_round_and_the_upstream_identity(completed):
    run_dir, _ = completed
    preflight = json.loads((run_dir / "preflight.json").read_text())
    assert set(preflight["partitions"]) == {"train", "validation_r01", "validation_r02", "test"}
    assert preflight["partitions"]["train"]["rounds"] == TINY_ROUNDS
    assert preflight["partitions"]["train"]["round_rows"] == [1024, 1024]
    assert set(preflight["overlap_audit"]["rows"]) == {"train_r01", "train_r02", "validation_r01",
                                                      "validation_r02", "test"}
    assert preflight["upstream_identity"]["commit"] == driver.AUTOND_SOURCE_PIN["upstream_commit"]
    assert preflight["upstream_identity"]["source_digests"] == driver.AUTOND_SOURCE_PIN["source_digests"]


def test_each_fit_follows_the_upstream_stopping_rule_exactly(completed):
    run_dir, _ = completed
    certificate = _certificate(run_dir)
    for row in certificate["findings"]["replicates"]:
        for condition in ("baseline", "perturbed"):
            selection = row[f"{condition}_selection"]
            prog = selection["progression"]
            fit_dir, _ = d4.checkpoint_paths(run_dir / "fits", row["replicate"], condition)
            for index, stage_round in enumerate(prog["rounds_trained"]):
                accs = [json.loads(A.epoch_record_path(fit_dir, stage_round, e).read_text())["val_acc"] for e in (1, 2)]
                assert prog["rounds_max_val_acc"][index] == max(accs)
                is_last = index == len(prog["rounds_trained"]) - 1
                passed = max(accs) > driver.STOP_THRESHOLD
                assert passed or is_last                                  # a failing round always ends the fit
                if is_last and prog["status"] == "stopped_at_threshold":
                    assert not passed and prog["stop_round"] == stage_round
            trained = sorted(int(p.name[len("stage_r"):]) for p in fit_dir.glob("stage_r*"))
            assert trained == prog["rounds_trained"]                     # nothing trained after a stop
            assert prog["status"] in ("stopped_at_threshold", "cap_reached")
            assert prog["executed_steps"] == selection["completed_steps"] == 2 * len(prog["rounds_trained"])
            assert prog["rounds_trained"][0] == 1 == selection["comparison_round"]


def test_both_conditions_use_the_autond_model_and_differ_only_in_training_labels(completed):
    from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

    run_dir, _ = completed
    preflight, data = driver.load_preflight(run_dir)
    certificate = _certificate(run_dir)
    train_x, train_y = data["train"]
    children = np.random.SeedSequence(driver.AUDIT_SEED).spawn(2)
    for row, child in zip(certificate["findings"]["replicates"], children):
        r = row["replicate"]
        indices = np.load(run_dir / "fits" / "perturbations" / f"replicate_{r:02d}_indices.npy")
        np.testing.assert_array_equal(indices, LabelShufflePerturbation().permutation_indices(
            len(train_y), np.random.default_rng(child)))
        assert not np.array_equal(indices, np.arange(len(train_y)))
        shuffled = np.asarray(train_y)[indices]
        assert np.array_equal(np.bincount(shuffled), np.bincount(train_y)) and np.any(shuffled != train_y)
        base, pert = row["baseline_selection"], row["perturbed_selection"]
        assert base["train_digest"] == I.labelled_partition_digest(train_x, train_y)
        assert pert["train_digest"] == I.labelled_partition_digest(train_x, shuffled)
        for i in range(2):
            rows = slice(1024 * i, 1024 * (i + 1))
            assert pert["round_train_digests"][i] == I.labelled_partition_digest(train_x[rows], shuffled[rows])
            assert base["round_train_digests"][i] == I.labelled_partition_digest(train_x[rows], train_y[rows])
        for key in ("seed", "replicate", "config_hash", "validation_digests", "evaluation_digest", "model"):
            assert base[key] == pert[key], key
        assert base["model"] == driver.expected_model(APPROVED)
        assert base["model"]["architecture"] == "dbitnet" and base["model"]["stop_threshold"] == 0.505
        assert base["model"]["upstream_commit"] == driver.AUTOND_SOURCE_PIN["upstream_commit"]
        assert base["evaluation_digest"] == preflight["partitions"]["test"]["digest"]
        assert base["condition"] == "baseline" and pert["condition"] == "perturbed"


def test_selection_is_attributed_per_condition_and_replicate_and_reload_is_the_argmin(completed):
    run_dir, _ = completed
    for row in _certificate(run_dir)["findings"]["replicates"]:
        for condition in ("baseline", "perturbed"):
            selection = row[f"{condition}_selection"]
            fit_dir, _ = d4.checkpoint_paths(run_dir / "fits", row["replicate"], condition)
            for summary in selection["progression"]["summaries"]:
                losses = [json.loads(A.epoch_record_path(fit_dir, summary["stage_round"], e).read_text())["val_loss"]
                          for e in (1, 2)]
                assert summary["selected_epoch"] == 1 + int(np.argmin(losses))
            assert selection["checkpoint"].startswith("stage_r01/")
            assert selection["checkpoint_sha256"] == I.sha256_file(fit_dir / selection["checkpoint"])
            assert (selection["condition"], selection["replicate"]) == (condition, row["replicate"])


def test_test_partition_is_never_given_to_training(completed):
    run_dir, _ = completed
    preflight, _ = driver.load_preflight(run_dir)
    test_digest = preflight["partitions"]["test"]["digest"]
    for row in _certificate(run_dir)["findings"]["replicates"]:
        for condition in ("baseline", "perturbed"):
            selection = row[f"{condition}_selection"]
            assert test_digest not in selection["round_train_digests"] + selection["validation_digests"]
            assert selection["evaluation_digest"] == test_digest


@pytest.mark.parametrize("target, mutate, match", [
    ("fits/replicate_01/perturbed/stage_r01/checkpoint_epoch_001.keras", lambda p: p.write_bytes(b"x"), "hashes to"),
    ("fits/replicate_02/baseline/stage_r01/checkpoint_epoch_001.metrics.json",
     lambda p: _edit(p, val_acc=0.999, logs={"val_acc": 0.999, "val_loss": json.loads(p.read_text())["val_loss"]}),
     "does not chain"),
    ("dataset/test_y.npy", lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])),
     "does not match preflight"),
    ("dataset/train_y.npy", lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])),
     "does not match preflight"),
    ("fits/predictions/replicate_02_perturbed.npy",
     lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])), "recorded digest"),
    ("fits/perturbations/replicate_01_indices.npy",
     lambda p: p.write_bytes(p.read_bytes()[:-1] + bytes([p.read_bytes()[-1] ^ 1])), "digest"),
    ("result/d4_certificate.json", lambda p: p.write_text(p.read_text().replace('"seed"', '"seed" ', 1)),
     "does not match the run manifest"),
])
def test_verify_detects_tampering(completed, tmp_path, target, mutate, match):
    run_dir, _ = completed
    copy = _copy(run_dir, tmp_path)
    mutate(copy / target)
    with pytest.raises(I.IntegrityError, match=match):
        driver.verify_run(copy)


def _rewrite_certificate(run_dir, mutate):
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
    (lambda c: c["settings"].update(baseline_adequacy_threshold=0.9), "baseline_adequacy_threshold differs"),
    (lambda c: c["decision"].update(outcome="SUPPORTED", overall_d4_outcome="SUPPORTED", overall_supported=True),
     "does not match the recomputed|endpoint"),
    (lambda c: c["findings"]["replicates"][0]["baseline_selection"].update(round_train_digests=["x", "y"]),
     "round_train_digests"),
    (lambda c: c["findings"]["replicates"][0]["perturbed_selection"]["progression"].update(best_round=99),
     "selection does not match"),          # 99: never a real value (the tiny data are os.urandom, unseeded)
    (lambda c: c["autond_pipeline_progression"]["summary"].update(perturbed_stopped_at_first_round=99),
     "pipeline progression does not match"),
    (lambda c: c["provenance"]["upstream_identity"].update(commit="0" * 40), "upstream_identity differs"),
])
def test_verify_refuses_certificates_that_depart_from_the_evidence(completed, tmp_path, mutate, match):
    run_dir, _ = completed
    copy = _copy(run_dir, tmp_path)
    _rewrite_certificate(copy, mutate)
    with pytest.raises(I.IntegrityError, match=match):
        driver.verify_run(copy)


def test_verify_refuses_artifacts_beyond_a_fits_registered_end(completed, tmp_path):
    """Any checkpoint after a fit ended -- past a stop, or past the cap -- makes verify refuse."""
    run_dir, _ = completed
    fits = [(row["replicate"], c, row[f"{c}_selection"]["progression"])
            for row in _certificate(run_dir)["findings"]["replicates"] for c in ("baseline", "perturbed")]
    cases = []
    stopped = [f for f in fits if f[2]["status"] == "stopped_at_threshold" and f[2]["stop_round"] < TINY_ROUNDS[-1]]
    if stopped:                                   # a stopped fit: an epoch of the next round must not exist
        replicate, condition, prog = stopped[0]
        cases.append((replicate, condition, prog["stop_round"] + 1, "after the pipeline stopped"))
    replicate, condition, prog = fits[0]          # always: a round beyond the registered cap
    cases.append((replicate, condition, TINY_ROUNDS[-1] + 1, "no registered"))
    for index, (replicate, condition, stage_round, match) in enumerate(cases):
        copy = _copy(run_dir, tmp_path / f"case{index}")
        fit_dir, _ = d4.checkpoint_paths(copy / "fits", replicate, condition)
        extra = A.epoch_checkpoint_path(fit_dir, stage_round, 1)
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"trained after the fit ended")
        with pytest.raises(I.IntegrityError, match=match):
            driver.verify_run(copy)


def test_completed_run_is_never_replaced_and_integrity_failure_is_not_completed(completed, tmp_path, capsys):
    run_dir, _ = completed
    before = I.sha256_file(run_dir / "result" / "d4_certificate.json")
    assert driver.main(["run", *stage_args(run_dir)]) == 1
    assert driver.main(["preflight", *stage_args(run_dir)]) == 1
    assert I.sha256_file(run_dir / "result" / "d4_certificate.json") == before
    copy = _copy(run_dir, tmp_path)
    shutil.rmtree(copy / "result")
    (copy / "dataset" / "validation_r01_y.npy").write_bytes(b"corrupt")
    assert driver.main(["run", *stage_args(copy)]) == 1
    assert "NOT_COMPLETED" in capsys.readouterr().err
    assert driver.main(["verify", "--run-dir", str(copy)]) == 5


def test_crashed_fit_resumes_through_quarantine_and_records_the_retry(completed, tmp_path):
    """Drop the commit of one fit's last executed step: its artifacts become an orphan, never adopted."""
    run_dir, _ = completed
    copy = _copy(run_dir, tmp_path)
    shutil.rmtree(copy / "result")
    fit_dir, state_path = d4.checkpoint_paths(copy / "fits", 1, "baseline")
    selection = _certificate(run_dir)["findings"]["replicates"][0]["baseline_selection"]
    identity = {k: selection[k] for k in A.IDENTITY_FIELDS}
    executed = selection["progression"]["executed_steps"]
    plan = A.step_plan(TINY_ROUNDS, 2)
    _, head = A.verify_epoch_chain(fit_dir, plan=plan, steps=executed - 1, identity=identity)
    A.write_state(state_path, identity=identity, total_steps=len(plan), completed=executed - 1, chain_head=head,
                  retries=[])
    (fit_dir / "selected_checkpoint.json").unlink()
    preflight, data = driver.load_preflight(copy)
    adapter = driver.make_adapter_factory(driver.design_of(APPROVED, "a" * 64), data, identity["evaluation_digest"],
                                          driver.make_binding(ROOT, 1), driver.run_binding_of(preflight, copy))(identity["seed"])
    adapter.train(data["train"][0], data["train"][1], checkpoint_path=fit_dir, state_path=state_path, replicate=1,
                  condition="baseline", run_config_hash=identity["config_hash"])
    retry = adapter.selection["retries"][0]
    assert retry["reason"] == "complete_uncommitted_orphan" and retry["step"] == executed
    assert len(list((fit_dir / "quarantine" / "attempt_01").iterdir())) == 3
    state = json.loads(state_path.read_text())
    assert state["completed_steps"] >= executed and state["retries"] == adapter.selection["retries"]
    assert adapter.selection["progression"]["status"] != "in_progress"
