"""
Tests for the II-4-GOHR-IMPL-V1-PAR1 block-level two-GPU scheduler.

Workers are real subprocesses (one per "GPU" label) running the real worker
module; training uses the deterministic FakeAdapter, which drives the real
checkpoint store, epoch log and recovery record. Evaluation, firewall, analysis
and report come from the frozen ComparisonRunner.run(). What these tests cannot
establish without GPUs: real device placement, CUDA nondeterminism and real
throughput; those belong to the Kaggle two-GPU smoke.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from audit.implementation.ii4_analysis import analyse_ii4
from audit.implementation.ii4_comparison_runner import ComparisonExecutionError, ComparisonRunner
from audit.implementation.ii4_parallel import verify as V
from audit.implementation.ii4_parallel.scheduler import (
    PARALLEL_SOURCES,
    BlockTrainingRunner,
    ParallelComparisonRunner,
    ParallelSchedulingError,
    claim_block,
    execute_parallel,
    merge_training_records,
)
from audit.implementation.ii4_parallel.tests import ii4_parallel_fixtures as F

FACTORY = "audit.implementation.ii4_parallel.tests.ii4_parallel_fixtures:make"
REPO = Path(__file__).resolve().parents[4]


def _par(tmp, gpus=("0", "1"), worker_args=(), **kw):
    root = Path(tmp) / "run"
    plan, ad = F.make(root, **kw)
    runner = ParallelComparisonRunner(plan=plan, adapter=ad)
    return root, runner, (lambda: execute_parallel(runner, gpus=gpus, factory=FACTORY, factory_kwargs=kw,
                                                   worker_args=worker_args))


def _state(root):
    return json.loads((root / "state.json").read_text())


def _arm(root, bid, arm):
    return _state(root)["blocks"][bid]["arms"][arm]


# --------------------------------------------------------------- normal run

def test_two_workers_train_distinct_blocks_concurrently_and_the_run_verifies(tmp_path):
    root, runner, go = _par(tmp_path)
    rep = go()
    assert rep["n_blocks_valid"] == 4
    v = V.verify(root, repo_root=REPO, require_dual_gpu=True)
    assert v["status"] == "VERIFIED", v["problems"]
    assert v["gpus_used"] == ["0", "1"] and v["cross_gpu_training_overlap_seconds"] > 0
    st = _state(root)
    for i in range(4):
        bid = f"block{i}"
        g = {st["blocks"][bid]["arms"][a]["gpu_assignment"]["gpu"] for a in (F.ARM_A, F.ARM_B)}
        assert len(g) == 1, "both arms of a block run on one worker"
        for a in (F.ARM_A, F.ARM_B):
            ga = st["blocks"][bid]["arms"][a]["gpu_assignment"]
            assert ga["cuda_visible_devices"] == ga["gpu"]
    assert {st["blocks"][f"block{i}"]["arms"][F.ARM_A]["gpu_assignment"]["gpu"] for i in range(4)} == {"0", "1"}


def test_registered_seeds_distinct_arms_and_isolated_outputs(tmp_path):
    root, runner, go = _par(tmp_path)
    go()
    seen_seeds, seen_models, seen_dirs = set(), set(), set()
    for i in range(4):
        for k, a in enumerate((F.ARM_A, F.ARM_B)):
            rec = _arm(root, f"block{i}", a)
            assert rec["seed"] == 900_000 + 2 * i + k == runner.plan.seeds.seed_for(i, a)
            assert Path(rec["arm_dir"]) == root / "blocks" / f"block{i}" / a
            seen_seeds.add(rec["seed"]); seen_models.add(rec["terminal_model"]["hash"]); seen_dirs.add(rec["arm_dir"])
    assert len(seen_seeds) == len(seen_models) == len(seen_dirs) == 8


def test_every_task_runs_exactly_once(tmp_path):
    root, _, go = _par(tmp_path)
    go()
    for i in range(4):
        claims = list((root / "blocks" / f"block{i}" / "claims").glob("*.json"))
        assert len(claims) == 1
        for a in (F.ARM_A, F.ARM_B):
            epochs = [json.loads(l)["epoch"] for l in
                      (root / "blocks" / f"block{i}" / a / "epochs.jsonl").read_text().splitlines()]
            assert epochs == [0, 1, 2]
    fw = json.loads((root / "firewall.json").read_text())
    assert len(fw["consumed_evaluation_keys"]) == 8 == len(set(fw["consumed_evaluation_keys"]))


def test_claim_is_exclusive(tmp_path):
    (tmp_path / "blocks" / "block0").mkdir(parents=True)
    assert claim_block(tmp_path, "block0", "s1", "0") is True
    assert claim_block(tmp_path, "block0", "s1", "1") is False
    assert claim_block(tmp_path, "block0", "s2", "1") is True


# --------------------------------------------------------------- serial equivalence

def test_parallel_and_serial_give_the_same_registered_experiment(tmp_path):
    root, _, go = _par(tmp_path / "p")
    par = go()
    sroot = tmp_path / "s" / "run"
    splan, sad = F.make(sroot)
    ser = ComparisonRunner(plan=splan, adapter=sad).run(execute=True)
    key = lambda rep: [(b["block_id"], b["seeds"], b["values"], b["signed_difference"]) for b in rep["blocks"]]
    assert key(par) == key(ser)
    assert par["analysis"] == ser["analysis"]
    assert par["config_fingerprint"] == ser["config_fingerprint"]
    assert par["preregistration_sha256"] == ser["preregistration_sha256"]
    assert par["experiment_fingerprint"] != ser["experiment_fingerprint"], "amended sources are bound"


def test_tost_inputs_and_results_match_an_independent_recomputation(tmp_path):
    root, runner, go = _par(tmp_path)
    rep = go()
    diffs = [b["signed_difference"] for b in rep["blocks"] if b["is_valid"]]
    ref = analyse_ii4(diffs, n_blocks_requested=4, min_valid_blocks=3, delta=0.01, alpha=0.05)
    r = V.recompute(diffs, delta=0.01, alpha=0.05, min_valid=3, n_requested=4)
    assert r["equivalence"] == ref.equivalence.value == rep["analysis"]["confirmatory"]["equivalence"]["verdict"]
    assert abs(r["tost_p"] - ref.tost_p) < 1e-12 and abs(r["mean"] - ref.mean_difference) < 1e-15


# --------------------------------------------------------------- interruption and resume

def test_resume_after_one_worker_is_interrupted_mid_block(tmp_path):
    root, _, go = _par(tmp_path, fail_at=[["block1", F.ARM_B, 1]])
    with pytest.raises(ParallelSchedulingError, match="exited"):
        go()
    assert not (root / "analysis.json").exists(), "nothing is evaluated after a worker failure"
    done = [i for i in range(4) if (root / "blocks" / f"block{i}" / "training_complete.json").exists()]
    assert 1 not in done and len(done) == 3, "the other worker kept going"
    root2, _, go2 = _par(tmp_path)          # same directory, fault removed: a new session
    assert root2 == root
    rep = go2()
    assert rep["n_blocks_valid"] == 4
    rec = _arm(root, "block1", F.ARM_B)
    assert rec["resumed_from_epoch"] == 2
    assert V.verify(root, repo_root=REPO)["status"] == "VERIFIED"


def test_an_arm_resumed_on_the_other_gpu_still_verifies(tmp_path):
    root, _, go = _par(tmp_path, fail_at=[["block1", F.ARM_B, 1]])
    with pytest.raises(ParallelSchedulingError):
        go()
    ts = json.loads((root / "blocks" / "block1" / "training_state.json").read_text())
    first = ts["blocks"]["block1"]["arms"][F.ARM_A]["gpu_assignment"]["gpu"]   # not merged: session failed
    assert "arms" not in _state(root)["blocks"]["block1"] or not _state(root)["blocks"]["block1"]["arms"]
    rep = _par(tmp_path, gpus=("2", "3"))[2]()          # resume on different device labels
    assert rep["n_blocks_valid"] == 4
    st = _state(root)["blocks"]["block1"]["arms"]
    assert st[F.ARM_A]["gpu_assignment"]["gpu"] == first and st[F.ARM_B]["gpu_assignment"]["gpu"] in ("2", "3")
    v = V.verify(root, repo_root=REPO)
    assert v["status"] == "VERIFIED", v["problems"]


def test_resume_after_both_workers_stop_early(tmp_path):
    root, _, go = _par(tmp_path, worker_args=("--stop-after-blocks", "1"))
    with pytest.raises(ParallelSchedulingError, match="never trained"):
        go()
    assert not (root / "analysis.json").exists()
    rep = _par(tmp_path)[2]()
    assert rep["n_blocks_valid"] == 4
    assert V.verify(root, repo_root=REPO)["status"] == "VERIFIED"


def test_completed_blocks_are_not_retrained_or_rewritten_on_resume(tmp_path):
    root, _, go = _par(tmp_path, fail_at=[["block2", F.ARM_A, 0]])
    with pytest.raises(ParallelSchedulingError):
        go()
    done = sorted(p.parent.name for p in root.glob("blocks/*/training_complete.json"))
    before = {b: hashlib.sha256((root / "blocks" / b / "training_complete.json").read_bytes()).hexdigest() for b in done}
    _par(tmp_path)[2]()
    after = {b: hashlib.sha256((root / "blocks" / b / "training_complete.json").read_bytes()).hexdigest() for b in done}
    assert before == after
    for b in done:
        assert len(list((root / "blocks" / b / "claims").glob("*.json"))) == 1


# --------------------------------------------------------------- identity refusals

def test_resume_under_another_design_is_refused(tmp_path):
    root, _, go = _par(tmp_path, worker_args=("--stop-after-blocks", "1"))
    with pytest.raises(ParallelSchedulingError):
        go()
    plan, ad = F.make(root, n_blocks=5)
    with pytest.raises(ComparisonExecutionError, match="fingerprint"):
        execute_parallel(ParallelComparisonRunner(plan=plan, adapter=ad), gpus=("0", "1"),
                         factory=FACTORY, factory_kwargs={"n_blocks": 5})


def test_frozen_runner_refuses_an_amended_run_and_vice_versa(tmp_path):
    root, _, go = _par(tmp_path / "a")
    go()
    plan, ad = F.make(root)
    with pytest.raises(ComparisonExecutionError, match="fingerprint"):
        ComparisonRunner(plan=plan, adapter=ad).run(execute=True)
    sroot = tmp_path / "b" / "run"
    splan, sad = F.make(sroot)
    ComparisonRunner(plan=splan, adapter=sad).run(execute=True)
    with pytest.raises(ComparisonExecutionError, match="fingerprint"):
        execute_parallel(ParallelComparisonRunner(plan=splan, adapter=sad), gpus=("0", "1"),
                         factory=FACTORY, factory_kwargs={})


def test_worker_refuses_a_run_with_another_fingerprint(tmp_path):
    root, _, go = _par(tmp_path, worker_args=("--stop-after-blocks", "1"))
    with pytest.raises(ParallelSchedulingError):
        go()
    st = _state(root); st["experiment_fingerprint"] = "0" * 64
    (root / "state.json").write_text(json.dumps(st))
    plan, ad = F.make(root)
    w = BlockTrainingRunner(plan=plan, adapter=ad, block_id="block3", gpu_label="0")
    with pytest.raises(ParallelSchedulingError, match="fingerprint"):
        w.train_block(3)


def _trained_not_merged(tmp_path):
    """Train every block, then stop before the merge, by making the merge's first input visible."""
    root, runner, go = _par(tmp_path)
    go()
    return root


def test_merge_refuses_swapped_arm_records(tmp_path):
    root = _trained_not_merged(tmp_path)
    p = root / "blocks" / "block0" / "training_state.json"
    ts = json.loads(p.read_text())
    arms = ts["blocks"]["block0"]["arms"]
    arms[F.ARM_A], arms[F.ARM_B] = arms[F.ARM_B], arms[F.ARM_A]
    p.write_text(json.dumps(ts))
    st = _state(root)
    for a in (F.ARM_A, F.ARM_B):
        st["blocks"]["block0"]["arms"][a]["status"] = "TRAINED"
    (root / "state.json").write_text(json.dumps(st))
    plan, ad = F.make(root)
    r = ParallelComparisonRunner(plan=plan, adapter=ad); r.state = _state(root)
    with pytest.raises(ParallelSchedulingError, match="identity or seed"):
        merge_training_records(r)


def test_merge_refuses_an_altered_terminal_model(tmp_path):
    root = _trained_not_merged(tmp_path)
    tm = json.loads((root / "blocks" / "block2" / "training_state.json").read_text())["blocks"]["block2"]["arms"][F.ARM_A]["terminal_model"]
    Path(tm["path"]).write_text("tampered")
    st = _state(root); st["blocks"]["block2"]["arms"][F.ARM_A]["status"] = "TRAINED"
    (root / "state.json").write_text(json.dumps(st))
    plan, ad = F.make(root)
    r = ParallelComparisonRunner(plan=plan, adapter=ad); r.state = _state(root)
    with pytest.raises(ParallelSchedulingError, match="terminal model"):
        merge_training_records(r)


def test_merge_refuses_a_missing_block(tmp_path):
    root = _trained_not_merged(tmp_path)
    (root / "blocks" / "block3" / "training_state.json").rename(root / "blocks" / "block3" / "moved.json")
    plan, ad = F.make(root)
    r = ParallelComparisonRunner(plan=plan, adapter=ad); r.state = _state(root)
    with pytest.raises(ParallelSchedulingError, match="never trained"):
        merge_training_records(r)


# --------------------------------------------------------------- verifier fails closed

@pytest.mark.parametrize("tamper", ["swap_journal", "edit_accuracy", "drop_consumption", "dup_epoch",
                                    "edit_seed", "edit_analysis", "remove_completion", "terminal_model",
                                    "foreign_worker"])
def test_verifier_refuses_tampered_evidence(tmp_path, tamper):
    root, _, go = _par(tmp_path)
    go()
    assert V.verify(root)["status"] == "VERIFIED"
    b0 = root / "blocks" / "block0"
    if tamper == "swap_journal":
        a, b = b0 / F.ARM_A / "eval_journal.json", b0 / F.ARM_B / "eval_journal.json"
        ta, tb = a.read_text(), b.read_text(); a.write_text(tb); b.write_text(ta)
    elif tamper == "edit_accuracy":
        st = _state(root); st["blocks"]["block0"]["arms"][F.ARM_A]["sealed_test_accuracy"] += 0.01
        (root / "state.json").write_text(json.dumps(st))
    elif tamper == "drop_consumption":
        fw = json.loads((root / "firewall.json").read_text()); fw["consumed_evaluation_keys"].pop()
        (root / "firewall.json").write_text(json.dumps(fw))
    elif tamper == "dup_epoch":
        p = b0 / F.ARM_A / "epochs.jsonl"; lines = p.read_text().splitlines(); p.write_text("\n".join(lines + lines[-1:]) + "\n")
    elif tamper == "edit_seed":
        st = _state(root); st["blocks"]["block1"]["arms"][F.ARM_B]["seed"] += 2
        (root / "state.json").write_text(json.dumps(st))
    elif tamper == "edit_analysis":
        an = json.loads((root / "analysis.json").read_text()); an["analysis"]["confirmatory"]["equivalence"]["verdict"] = "NOT_EQUIVALENT"
        (root / "analysis.json").write_text(json.dumps(an))
    elif tamper == "remove_completion":
        (b0 / "training_complete.json").unlink()
    elif tamper == "terminal_model":
        (b0 / F.ARM_B / "terminal_model.keras").write_text("x")
    elif tamper == "foreign_worker":
        st = _state(root)
        g = st["blocks"]["block0"]["arms"][F.ARM_A]["gpu_assignment"]
        g["gpu"] = "1" if g["gpu"] == "0" else "0"
        (root / "state.json").write_text(json.dumps(st))
    assert V.verify(root)["status"] == "NOT_VERIFIED"


# --------------------------------------------------------------- production guards and preservation

def test_production_guards(tmp_path):
    from audit.common.outcomes import ExecutionMode
    import dataclasses
    root = tmp_path / "run"
    plan, ad = F.make(root)
    prod = dataclasses.replace(plan, execution_mode=ExecutionMode.PRODUCTION)
    r = ParallelComparisonRunner(plan=prod, adapter=ad)
    for kw, msg in (({"gpus": ("0", "1"), "allow_production": False}, "allow_production"),
                    ({"gpus": ("0",), "allow_production": True, "frozen": {}}, "exactly two GPUs"),
                    ({"gpus": ("0", "1"), "allow_production": True}, "frozen record"),
                    ({"gpus": ("0", "1"), "allow_production": True, "frozen": {},
                      "worker_args": ("--stop-after-blocks", "1")}, "fault-injection")):
        with pytest.raises(ComparisonExecutionError, match=msg):
            execute_parallel(r, factory=FACTORY, factory_kwargs={}, **kw)
    assert not (root / "state.json").exists()


def test_original_frozen_record_and_its_files_are_untouched():
    rec = json.loads((REPO / "audit/implementation/preregistrations/ii-4-gohr-impl-v1.frozen.json").read_text())
    for v in rec["source_manifest"].values():
        assert hashlib.sha256((REPO / v["path"]).read_bytes()).hexdigest() == v["sha256"], v["path"]
    assert not set(PARALLEL_SOURCES.values()) & {v["path"] for v in rec["source_manifest"].values()}


# --------------------------------------------------------------- dual-GPU concurrency and epoch provenance

def _rewrite(path, fn):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    path.write_text("".join(json.dumps(r) + "\n" for r in fn(rows)))


def test_single_gpu_execution_is_rejected_as_dual_gpu(tmp_path):
    root, _, go = _par(tmp_path, gpus=("0",))
    go()
    assert V.verify(root)["status"] == "VERIFIED"                    # a valid single-device run
    v = V.verify(root, require_dual_gpu=True)
    assert v["status"] == "NOT_VERIFIED" and any("used GPU(s)" in p for p in v["problems"])


def test_two_gpus_used_one_after_the_other_are_rejected_as_dual_gpu(tmp_path):
    root, _, go = _par(tmp_path, gpus=("0",), worker_args=("--stop-after-blocks", "2"))
    with pytest.raises(ParallelSchedulingError):
        go()
    _par(tmp_path, gpus=("1",))[2]()
    v = V.verify(root, require_dual_gpu=True)
    assert v["gpus_used"] == ["0", "1"]
    assert v["status"] == "NOT_VERIFIED" and any("overlapped" in p for p in v["problems"])


def test_zero_overlap_between_gpus_is_rejected(tmp_path):
    root, _, go = _par(tmp_path)
    go()
    st = _state(root)
    for bid, blk in st["blocks"].items():
        for rec in blk["arms"].values():
            if rec["gpu_assignment"]["gpu"] == "1":            # move GPU 1's work a day later
                for k in ("started_at_utc", "trained_at_utc"):
                    rec[k] = "2099" + rec[k][4:]
    (root / "state.json").write_text(json.dumps(st))
    v = V.verify(root, require_dual_gpu=True)
    assert v["status"] == "NOT_VERIFIED" and any("overlapped" in p for p in v["problems"])


def test_overlap_from_epoch_intervals():
    assert V.overlap_seconds([("0", 0, 10), ("1", 5, 15)]) == 5
    assert V.overlap_seconds([("0", 0, 10), ("1", 10, 20)]) == 0
    assert V.overlap_seconds([("0", 0, 10), ("0", 5, 15)]) == 0


@pytest.mark.parametrize("what", ["epoch_session", "ledger_gpu", "ledger_block", "ledger_arm", "missing_epoch",
                                  "duplicate_epoch", "epoch_out_of_range"])
def test_misattributed_or_missing_epochs_are_rejected(tmp_path, what):
    root, _, go = _par(tmp_path)
    go()
    arm_dir = root / "blocks" / "block0" / F.ARM_A
    log, led = arm_dir / "epochs.jsonl", arm_dir / "worker_sessions.jsonl"
    if what == "epoch_session":
        _rewrite(log, lambda rs: [dict(r, session_id="2026-01-01T000000.000000+0000") if r["epoch"] == 1 else r for r in rs])
    elif what == "ledger_gpu":
        _rewrite(led, lambda rs: [dict(r, gpu="1" if r["gpu"] == "0" else "0") for r in rs])
    elif what == "ledger_block":
        _rewrite(led, lambda rs: [dict(r, block_id="block1") for r in rs])
    elif what == "ledger_arm":
        _rewrite(led, lambda rs: [dict(r, arm=F.ARM_B) for r in rs])
    elif what == "missing_epoch":
        _rewrite(log, lambda rs: [r for r in rs if r["epoch"] != 1])
    elif what == "duplicate_epoch":
        _rewrite(log, lambda rs: rs + [rs[-1]])
    elif what == "epoch_out_of_range":
        _rewrite(led, lambda rs: [dict(r, epochs_after=2) if r["event"] == "arm_session_end" else r for r in rs])
    v = V.verify(root)
    assert v["status"] == "NOT_VERIFIED", what


def test_duplicate_firewall_consumption_is_rejected(tmp_path):
    root, _, go = _par(tmp_path)
    go()
    fw = json.loads((root / "firewall.json").read_text())
    fw["consumed_evaluation_keys"].append(fw["consumed_evaluation_keys"][0])
    (root / "firewall.json").write_text(json.dumps(fw))
    v = V.verify(root)
    assert v["status"] == "NOT_VERIFIED" and any("duplicate firewall" in p for p in v["problems"])


def test_resumed_arm_is_accepted_only_with_provenance_for_every_epoch(tmp_path):
    root, _, go = _par(tmp_path, fail_at=[["block1", F.ARM_B, 1]])
    with pytest.raises(ParallelSchedulingError):
        go()
    _par(tmp_path)[2]()
    led = root / "blocks" / "block1" / F.ARM_B / "worker_sessions.jsonl"
    rows = [json.loads(l) for l in led.read_text().splitlines()]
    sessions = [r["session_id"] for r in rows if r["event"] == "arm_session_start"]
    assert len(sessions) == 2, "one ledger start per session that trained this arm"
    epochs = [json.loads(l) for l in (led.parent / "epochs.jsonl").read_text().splitlines()]
    assert {e["session_id"] for e in epochs if e["epoch"] < 2} == {sessions[0]}
    assert {e["session_id"] for e in epochs if e["epoch"] >= 2} == {sessions[1]}
    assert V.verify(root, require_dual_gpu=True)["status"] == "VERIFIED"
    _rewrite(led, lambda rs: [r for r in rs if not (r["event"] == "arm_session_start" and r["session_id"] == sessions[0])])
    v = V.verify(root)
    assert v["status"] == "NOT_VERIFIED" and any("no worker ledger records" in p for p in v["problems"])
