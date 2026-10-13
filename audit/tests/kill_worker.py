#!/usr/bin/env python3
"""
Child process for the hard-kill recovery tests.

Runs the REAL commit protocol and the REAL evaluation journal, then
SIGKILLs itself at a named persistence boundary. SIGKILL rather than an
exception, because an exception runs `except` clauses, `finally` blocks
and `atexit` handlers, and a test that only raises in-process therefore
exercises a code path a real interruption never takes. SIGKILL cannot be
caught, so what survives on disk is exactly what survives a container
eviction, an OOM kill or a pulled power cable.

Usage (invoked by audit/tests/test_durable_state.py):

    python3 -m audit.tests.kill_worker <mode> <arm_dir> <json-args>

Modes:
    train   commit epochs 0..N-1, killing at --boundary on --kill-epoch
    eval    run the journal protocol, killing at --eval-boundary

The "model" is a small JSON document, so no TensorFlow is involved and
the protocol - not Keras - is what is under test.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

from audit.implementation.arm_recovery import EpochLog, resume_point
from audit.implementation.durable_state import (
    EVAL_CONSUMED,
    EVAL_MEASURED,
    EVAL_RECORDED,
    CommittedCheckpointStore,
    EvalJournal,
    sha256_file,
)


def _hard_kill() -> None:
    """Uncatchable. Flush nothing, run nothing, just stop existing."""
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)
    os._exit(137)                    # unreachable; belt and braces


def run_train(arm_dir: Path, args: dict) -> int:
    epochs = int(args["epochs"])
    kill_boundary = args.get("boundary")
    kill_epoch = args.get("kill_epoch")
    partial_log = args.get("partial_log_after_epoch")

    log_path = arm_dir / "epochs.jsonl"

    def hook(name: str) -> None:
        if kill_boundary == name and store._current_epoch == kill_epoch:
            print(f"KILL {name} epoch {kill_epoch}")
            _hard_kill()

    store = CommittedCheckpointStore(
        arm_dir=arm_dir, arm_id=args.get("arm_id", "reference"),
        block_id=args.get("block_id", "block0"), seed=int(args.get("seed", 1)),
        epochs_total=epochs, suffix=".json",
        architecture_digest=args.get("architecture_digest"),
        boundary_hook=hook)
    store._current_epoch = -1

    # resume_point, which calls prepare_for_resume internally: a
    # degraded recovery must PROMOTE the surviving slot or the next
    # commit is refused, and the log must be reconciled against the
    # manifest. This is the same entry point the real adapter uses, so
    # the worker exercises the production recovery path rather than a
    # simplified one.
    start, decision, notes = resume_point(store, EpochLog(log_path))
    print(json.dumps({"resume_from": start, "source": decision.source,
                      "degraded": decision.degraded, "notes": notes,
                      "repairs": len(store.corruption_history())}))

    for epoch in range(start, epochs):
        store._current_epoch = epoch

        def save_fn(path, epoch=epoch):
            Path(path).write_text(json.dumps({"epoch": epoch, "weights": [epoch] * 4,
                                              "optimizer": {"iterations": (epoch + 1) * 10}}))

        store.commit_epoch(epoch=epoch, save_fn=save_fn,
                           best_monitor_value=0.3 - 0.01 * epoch,
                           best_checkpoint_epoch=epoch, session_id=args.get("session", "s"))

        # the DERIVED log append
        with log_path.open("a", encoding="utf-8") as fh:
            line = json.dumps({"epoch": epoch, "status": "COMPLETED",
                               "val_loss": 0.3 - 0.01 * epoch,
                               "val_accuracy": 0.5 + 0.01 * epoch,
                               "session_id": args.get("session", "s")})
            if partial_log == epoch:
                # Truncate deliberately: the likeliest artifact of a kill
                # between the write and the newline.
                fh.write(line[: len(line) // 2])
                fh.flush()
                os.fsync(fh.fileno())
                print(f"KILL after_partial_log_append epoch {epoch}")
                _hard_kill()
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if kill_boundary == "after_log_append" and epoch == kill_epoch:
            print(f"KILL after_log_append epoch {epoch}")
            _hard_kill()

    print(json.dumps({"completed": epochs}))
    return 0


def run_eval(arm_dir: Path, args: dict) -> int:
    """
    The journal protocol: measure -> journal MEASURED -> consume ->
    journal CONSUMED -> record -> journal RECORDED.
    """
    journal = EvalJournal(arm_dir / "eval_journal.json")
    fw_path = arm_dir / "firewall.json"
    state_path = arm_dir / "arm_state.json"
    boundary = args.get("eval_boundary")
    sealed = args["sealed_test_hash"]
    tm = args["terminal_model_hash"]

    existing = journal.adopt(sealed_test_hash=sealed, terminal_model_hash=tm)
    if existing is None:
        accuracy = float(args["accuracy"])
        print(json.dumps({"inference": "performed", "accuracy": accuracy}))
        journal.write(state=EVAL_MEASURED, accuracy=accuracy, sealed_test_hash=sealed,
                      terminal_model_hash=tm, arm_id="reference", block_id="block0",
                      session_id=args.get("session", "s"))
    else:
        accuracy = float(existing["accuracy"])
        print(json.dumps({"inference": "SKIPPED_adopted_from_journal",
                          "accuracy": accuracy, "journal_state": existing["state"]}))
    if boundary == "after_journal_measured":
        print("KILL after_journal_measured")
        _hard_kill()

    consumed = json.loads(fw_path.read_text()) if fw_path.is_file() else []
    key = ["reference", "block0"]
    if key in consumed:
        print(json.dumps({"consumption": "already_spent_not_repeated"}))
    else:
        consumed.append(key)
        from audit.implementation.durable_state import atomic_write_json
        atomic_write_json(fw_path, consumed)
        print(json.dumps({"consumption": "spent_now"}))
    if boundary == "after_firewall_consumption":
        print("KILL after_firewall_consumption")
        _hard_kill()

    journal.advance(EVAL_CONSUMED)
    if boundary == "after_journal_consumed":
        print("KILL after_journal_consumed")
        _hard_kill()

    from audit.implementation.durable_state import atomic_write_json
    atomic_write_json(state_path, {"status": "EVALUATED", "accuracy": accuracy})
    journal.advance(EVAL_RECORDED)
    print(json.dumps({"recorded": True, "accuracy": accuracy}))
    return 0


def main(argv) -> int:
    mode, arm_dir, payload = argv[1], Path(argv[2]), json.loads(argv[3])
    arm_dir.mkdir(parents=True, exist_ok=True)
    if mode == "train":
        return run_train(arm_dir, payload)
    if mode == "eval":
        return run_eval(arm_dir, payload)
    if mode == "runner":
        return run_runner(arm_dir, payload)
    raise SystemExit(f"unknown mode {mode!r}")




def run_runner(run_dir: Path, args: dict) -> int:
    """
    Drive the REAL ComparisonRunner with the fake adapter, SIGKILLing at
    a named runner-level persistence boundary.

    This is how the firewall and sealed-test-set lifecycles are crash
    tested: the firewall under test is the real `TestSetFirewall`, the
    sealing path is the runner's own, and the kill is a real SIGKILL.
    """
    import audit.implementation.ii4_comparison_runner as runner_mod
    from audit.tests.test_ii4_comparison import FakeAdapter, design, plan

    boundary = args.get("boundary")
    hits = {"n": 0}

    def hook(name: str) -> None:
        if name != boundary:
            return
        hits["n"] += 1
        if hits["n"] >= int(args.get("nth", 1)):
            print(f"KILL {name}")
            _hard_kill()

    d = design(n_blocks=int(args.get("n_blocks", 2)),
               min_valid_blocks=int(args.get("min_valid", 2)),
               max_replacement_blocks=0)
    p = plan(str(run_dir.parent), d=d)
    p.output_dir = run_dir
    adapter = FakeAdapter(arm_ids=p.axis.arm_ids)
    runner = runner_mod.ComparisonRunner(
        plan=p, adapter=adapter, session_id=args.get("session", "s"),
        boundary_hook=hook if boundary else None)
    report = runner.run(execute=True)
    print(json.dumps({"n_blocks_valid": report["n_blocks_valid"],
                      "adopted": len(report["adopted_evaluations"]),
                      "consumed": sorted(
                          report["firewall"]["consumed_evaluation_keys"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
