"""
Block-level two-GPU scheduler for the frozen II-4-GOHR-IMPL-V1 comparison.

WHAT CHANGES AND WHAT DOES NOT
------------------------------
The frozen topology note (audit/implementation/preflight.py, TRAINING_TOPOLOGY)
names one legitimate form of parallelism: "Two SEPARATE blocks could be run as
two independent processes, one pinned to each GPU with CUDA_VISIBLE_DEVICES,
because blocks are independent experimental units." It was not implemented
because "two processes would write to one run directory and one state file".
This module implements exactly that, and nothing else:

  * one worker PROCESS per GPU, started with CUDA_VISIBLE_DEVICES set before the
    interpreter starts, so each sees one device;
  * a worker trains BOTH arms of one block at a time, in the frozen order, with
    the frozen `_build_and_verify_arms` and `_train_arm` and the frozen adapter;
    two workers therefore train two DIFFERENT blocks concurrently;
  * a worker never writes the run's `state.json` or `firewall.json`. Its state
    lives in `blocks/<id>/training_state.json`, and a block is claimed through
    an exclusive-create file `blocks/<id>/claims/<session>.json`, so no block is
    trained by two workers in one session;
  * the parent process is the only writer of `state.json` and `firewall.json`.
    It generates the sealed test set and every block's data before training (in
    the frozen runner these are also generated once and never regenerated), and
    after the workers finish it merges their arm records, with identity checks,
    and calls the frozen `ComparisonRunner.run()` unchanged. That call finds
    every arm TRAINED, so it performs only the frozen evaluation, firewall
    consumption, replacement, analysis and report.

Unchanged: the plan, the seven blocks, the replacement budget and rule, the two
arms and their identities, the (block, arm) seeds, 200 epochs, the data
procedure, the shared sealed test set, terminal-epoch evaluation, pairing, alpha,
the 0.01 margin and every decision rule. The amended source files are added to
the source manifest, so the amended run has its own experiment fingerprint: the
frozen runner refuses to resume an amended run directory and the amended runner
refuses to resume a frozen one. Evidence from the two is never pooled.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit.common.outcomes import ExecutionMode
from audit.common.provenance import sha256_bytes, sha256_file, utc_timestamp
from audit.common.strict_json import dumps_strict
from audit.implementation.durable_state import atomic_write_text
from audit.implementation.ii4_comparison_runner import (
    BlockTechnicalFailure,
    ComparisonExecutionError,
    ComparisonRunner,
)
from audit.implementation.ii4_runner import REPO_ROOT

PARALLEL_VERSION = "ii4-parallel-scheduler-v1"
AMENDMENT_ID = "II-4-GOHR-IMPL-V1-PAR1"

#: Added to the frozen source manifest; their hashes enter the experiment fingerprint.
PARALLEL_SOURCES = {
    "parallel_package": "audit/implementation/ii4_parallel/__init__.py",
    "parallel_scheduler": "audit/implementation/ii4_parallel/scheduler.py",
    "parallel_worker": "audit/implementation/ii4_parallel/worker.py",
    "parallel_launcher": "audit/implementation/ii4_parallel/launcher.py",
    "parallel_verifier": "audit/implementation/ii4_parallel/verify.py",
    "parallel_tests_package": "audit/implementation/ii4_parallel/tests/__init__.py",
    "parallel_tests": "audit/implementation/ii4_parallel/tests/test_ii4_parallel.py",
    "parallel_test_fixtures": "audit/implementation/ii4_parallel/tests/ii4_parallel_fixtures.py",
}

TRAINING_STATE = "training_state.json"
TRAINING_COMPLETE = "training_complete.json"
WORKER_LEDGER = "worker_sessions.jsonl"


class ParallelSchedulingError(ComparisonExecutionError):
    pass


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _write(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, dumps_strict(obj, indent=2, sort_keys=True, default=str))


@dataclass
class ParallelComparisonRunner(ComparisonRunner):
    """The frozen runner with the amendment's sources bound into its provenance."""

    def runtime_provenance(self) -> dict[str, Any]:
        if self._provenance is None or self._provenance.get("scheduler") != PARALLEL_VERSION:
            self._provenance = None
            base = ComparisonRunner.runtime_provenance(self)
            sources = dict(base["source_manifest"])
            missing = []
            for label, rel in sorted(PARALLEL_SOURCES.items()):
                p = REPO_ROOT / rel
                if not p.is_file():
                    missing.append(rel)
                    continue
                sources[label] = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
            if missing:
                raise ParallelSchedulingError(f"amendment source file(s) absent: {missing}")
            manifest_hash = sha256_bytes(dumps_strict(sources, sort_keys=True).encode("utf-8"))
            fingerprint = sha256_bytes(dumps_strict({
                "config_fingerprint": base["config_fingerprint"],
                "source_manifest_sha256": manifest_hash,
                "software_stack": base["software_stack"],
                "scheduler": PARALLEL_VERSION,
            }, sort_keys=True).encode("utf-8"))
            self._provenance = {**base, "source_manifest": sources,
                                "source_manifest_sha256": manifest_hash,
                                "experiment_fingerprint": fingerprint,
                                "scheduler": PARALLEL_VERSION}
        return self._provenance


@dataclass
class BlockTrainingRunner(ParallelComparisonRunner):
    """Trains both arms of ONE block. Never touches state.json or firewall.json."""

    block_id: str = ""
    gpu_label: str = ""

    @property
    def state_path(self) -> Path:
        return self.root / "blocks" / self.block_id / TRAINING_STATE

    def train_block(self, block_index: int) -> dict[str, Any]:
        bid = self.block_id
        if bid != f"block{block_index}":
            raise ParallelSchedulingError(f"block id {bid} does not match index {block_index}")
        fp = self.runtime_provenance()["experiment_fingerprint"]
        main = _json(self.root / "state.json")
        if main.get("experiment_fingerprint") != fp:
            raise ParallelSchedulingError(
                f"{bid}: worker fingerprint {fp} differs from the run's "
                f"{main.get('experiment_fingerprint')}; refusing to train under another identity")
        main_blk = (main.get("blocks") or {}).get(bid)
        if not main_blk or "dataset" not in main_blk or main_blk.get("block_index") != block_index:
            raise ParallelSchedulingError(f"{bid}: the parent has not recorded this block's data")

        if self.state_path.exists():
            self.state = _json(self.state_path)
            if (self.state.get("experiment_fingerprint") != fp or self.state.get("block_id") != bid
                    or self.state["blocks"][bid].get("dataset") != main_blk["dataset"]):
                raise ParallelSchedulingError(
                    f"{bid}: existing training state belongs to another identity or dataset")
        else:
            self.state = {"experiment_fingerprint": fp, "scheduler": PARALLEL_VERSION,
                          "block_id": bid, "sessions": [],
                          "blocks": {bid: {"arms": {}, "block_index": block_index,
                                           "dataset": main_blk["dataset"]}}}
        self.state["sessions"].append({"session_id": self.session_id, "gpu": self.gpu_label,
                                       "host": socket.gethostname(), "pid": os.getpid(),
                                       "started_at_utc": utc_timestamp()})
        self._save_state()
        blk = self.state["blocks"][bid]

        pending = [a for a in self.arm_ids if self._arm_needs_training(bid, a, blk)]
        if pending:
            try:
                built = self._build_and_verify_arms(bid, block_index)
            except BlockTechnicalFailure as exc:
                for arm in pending:
                    blk["arms"][arm] = {"arm": arm, "status": "FAILED",
                                        "seed": self.plan.seeds.seed_for(block_index, arm),
                                        "failure": {"stage": "build", "reason": "construction_failed",
                                                    "detail": str(exc), "technical": True}}
                self._save_state()
                return self._summary(bid)
            data = self.adapter.load_block_datasets(blk["dataset"])
            for arm in pending:
                self._ledger(bid, arm, "arm_session_start", epochs_before=self._completed_epochs(bid, arm))
                rec = self._train_arm(bid, block_index, arm, built["models"][arm],
                                      built["probes"][arm], data)
                self._ledger(bid, arm, "arm_session_end", status=rec.get("status"),
                             epochs_after=self._completed_epochs(bid, arm))
                rec["gpu_assignment"] = {"gpu": self.gpu_label, "session_id": self.session_id,
                                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                         "pid": os.getpid(), "host": socket.gethostname(),
                                         "devices_seen": self.adapter.environment_details().get("gpus")}
                blk["arms"][arm] = rec
                self._save_state()

        summary = self._summary(bid)
        if all(s == "TRAINED" for s in summary["arm_status"].values()):
            done = self.root / "blocks" / bid / TRAINING_COMPLETE
            record = {"block_id": bid, "block_index": block_index, "experiment_fingerprint": fp,
                      "arms": {a: {"seed": blk["arms"][a]["seed"],
                                   "terminal_model": blk["arms"][a]["terminal_model"],
                                   "gpu_assignment": blk["arms"][a].get("gpu_assignment")}
                               for a in self.arm_ids},
                      "dataset": blk["dataset"], "completed_at_utc": utc_timestamp()}
            if done.exists():
                old = _json(done)
                if {k: v for k, v in old.items() if k != "completed_at_utc"} != \
                        {k: v for k, v in record.items() if k != "completed_at_utc"}:
                    raise ParallelSchedulingError(f"{bid}: completion record exists and differs")
            else:
                _write(done, record)
        return summary

    def _ledger(self, bid: str, arm: str, event: str, **fields) -> None:
        """Append-only per-arm record of which worker trained which epochs, in which session.

        Each completed epoch in the frozen epoch log carries its session id; this ledger ties that
        session to the claim, the GPU and the epoch range, so every epoch can be attributed.
        """
        claim = self.root / "blocks" / bid / "claims" / f"{self.session_id}.json"
        line = dumps_strict({"event": event, "block_id": bid, "arm": arm, "session_id": self.session_id,
                             "gpu": self.gpu_label,
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                             "pid": os.getpid(), "host": socket.gethostname(),
                             "claim_sha256": sha256_file(claim) if claim.is_file() else None,
                             "at_utc": utc_timestamp(), **fields}, sort_keys=True)
        path = self.root / "blocks" / bid / arm / WORKER_LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _summary(self, bid: str) -> dict[str, Any]:
        arms = self.state["blocks"][bid]["arms"]
        return {"block_id": bid, "arm_status": {a: (arms.get(a) or {}).get("status", "NOT_RUN")
                                                for a in self.arm_ids}}


def claim_block(root: Path, block_id: str, session_id: str, gpu: str) -> bool:
    """Exclusive-create claim: True for exactly one worker per (block, session)."""
    d = Path(root) / "blocks" / block_id / "claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{session_id}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"block_id": block_id, "session_id": session_id, "gpu": gpu,
                   "pid": os.getpid(), "host": socket.gethostname(),
                   "claimed_at_utc": utc_timestamp()}, f)
        f.flush()
        os.fsync(f.fileno())
    return True


def planned_block_ids(plan) -> list[str]:
    return [f"block{i}" for i in range(plan.design.n_blocks)]


def merge_training_records(runner: ParallelComparisonRunner) -> list[str]:
    """Copy each worker's arm records into state.json, refusing any identity mismatch."""
    fp = runner.runtime_provenance()["experiment_fingerprint"]
    merged = []
    for bid in planned_block_ids(runner.plan):
        i = int(bid[5:])
        path = runner.root / "blocks" / bid / TRAINING_STATE
        if not path.exists():
            raise ParallelSchedulingError(f"{bid}: no training state; the block was never trained")
        ts = _json(path)
        if ts.get("experiment_fingerprint") != fp or ts.get("block_id") != bid:
            raise ParallelSchedulingError(f"{bid}: training state carries another identity")
        tblk = ts["blocks"][bid]
        mblk = runner.state["blocks"][bid]
        if tblk.get("dataset") != mblk.get("dataset") or tblk.get("block_index") != i:
            raise ParallelSchedulingError(f"{bid}: training state names another dataset or index")
        for arm in runner.arm_ids:
            rec = (tblk.get("arms") or {}).get(arm)
            if rec is None:
                raise ParallelSchedulingError(f"{bid}/{arm}: missing from the training state")
            if rec.get("arm") != arm or rec.get("seed") != runner.plan.seeds.seed_for(i, arm):
                raise ParallelSchedulingError(f"{bid}/{arm}: arm identity or seed differs from the plan")
            arm_dir = runner.root / "blocks" / bid / arm
            if rec.get("arm_dir") is not None and Path(rec["arm_dir"]) != arm_dir:
                raise ParallelSchedulingError(f"{bid}/{arm}: record points at {rec['arm_dir']}")
            if rec.get("status") == "TRAINED":
                tm = rec.get("terminal_model") or {}
                if Path(tm.get("path", "")).parent != arm_dir or not Path(tm["path"]).is_file() \
                        or sha256_file(tm["path"]) != tm.get("hash"):
                    raise ParallelSchedulingError(f"{bid}/{arm}: terminal model missing, misplaced or altered")
            existing = mblk["arms"].get(arm)
            if existing and existing.get("status") == "EVALUATED":
                continue                     # already evaluated in an earlier session
            mblk["arms"][arm] = rec
        merged.append(bid)
    runner._save_state()
    return merged


def _sample_gpus(path: Path, stop: list) -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return
    with open(path, "a") as f:
        while not stop:
            r = subprocess.run([exe, "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                "--format=csv,noheader"], capture_output=True, text=True)
            f.write(r.stdout)
            f.flush()
            time.sleep(30)


def execute_parallel(runner: ParallelComparisonRunner, *, gpus: tuple[str, ...], factory: str,
                     factory_kwargs: dict, allow_production: bool = False,
                     frozen: Optional[dict] = None, worker_args: tuple[str, ...] = (),
                     python: str = sys.executable) -> dict[str, Any]:
    """Prepare, train blocks on `gpus` concurrently, merge, then run the frozen evaluation."""
    production = runner.plan.execution_mode is ExecutionMode.PRODUCTION
    if production:
        if not allow_production:
            raise ParallelSchedulingError("production requires allow_production=True")
        if frozen is None:
            raise ParallelSchedulingError("production requires the amendment's frozen record")
        if len(gpus) != 2:
            raise ParallelSchedulingError(f"production requires exactly two GPUs, got {gpus}")
        if worker_args:
            raise ParallelSchedulingError("fault-injection worker arguments are refused in production")
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ParallelSchedulingError(f"GPU list must be non-empty and distinct: {gpus}")
    runner.run(execute=False)                                   # frozen plan and directory checks
    if frozen is not None:
        problems = runner.check_frozen(frozen)
        if problems:
            raise ParallelSchedulingError("not the reviewed amendment:\n  " + "\n  ".join(problems))

    firewall = runner._preregister_or_resume()
    runner._sealed_test_set(firewall)
    for bid in planned_block_ids(runner.plan):
        i = int(bid[5:])
        blk = runner.state["blocks"].setdefault(bid, {"arms": {}, "block_index": i,
                                                      "replaces_block_id": None})
        if bid not in runner.state["block_order"]:
            runner.state["block_order"].append(bid)
        if "dataset" not in blk:
            blk["dataset"] = runner.adapter.generate_block_datasets(bid, runner.root / "blocks" / bid / "data")
            runner._save_state()

    pending = [b for b in planned_block_ids(runner.plan)
               if not (runner.root / "blocks" / b / TRAINING_COMPLETE).exists()]
    session = runner.session_id
    sess = {"session_id": session, "gpus": list(gpus), "pending_blocks": pending,
            "started_at_utc": utc_timestamp(), "scheduler": PARALLEL_VERSION}
    runner.state.setdefault("parallel_sessions", []).append(sess)
    runner._save_state()

    logs = runner.root / "logs"
    logs.mkdir(exist_ok=True)
    codes: dict[str, int] = {}
    if pending:
        stop: list = []
        import threading
        sampler = threading.Thread(target=_sample_gpus, args=(logs / f"gpu_util_{session}.csv", stop),
                                   daemon=True)
        sampler.start()
        procs = {}
        for g in gpus:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            cmd = [python, "-B", "-m", "audit.implementation.ii4_parallel.worker",
                   "--root", str(runner.root), "--factory", factory,
                   "--factory-kwargs", json.dumps(factory_kwargs, sort_keys=True),
                   "--session", session, "--gpu", str(g), "--blocks", ",".join(pending), *worker_args]
            log = open(logs / f"worker_gpu{g}_{session}.log", "a")
            procs[str(g)] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              cwd=str(REPO_ROOT)), log)
        for g, (p, log) in procs.items():
            codes[g] = p.wait()
            log.close()
        stop.append(True)
    sess.update(worker_exit_codes=codes, finished_at_utc=utc_timestamp())
    runner._save_state()
    if any(c != 0 for c in codes.values()):
        raise ParallelSchedulingError(
            f"worker(s) exited {codes}; nothing was evaluated. All epoch logs, checkpoints and "
            "training states are kept. Re-run the same command to resume.")

    merge_training_records(runner)
    report = runner.run(execute=True, allow_production=allow_production, frozen=frozen)
    prov = runner.runtime_provenance()
    _write(runner.root / "parallel_execution.json", {
        "amendment_id": AMENDMENT_ID, "scheduler": PARALLEL_VERSION,
        "experiment_fingerprint": prov["experiment_fingerprint"],
        "source_manifest_sha256": prov["source_manifest_sha256"],
        "sessions": runner.state.get("parallel_sessions", []),
        "analysis_sha256": sha256_file(runner.root / "analysis.json"),
        "written_at_utc": utc_timestamp()})
    return report
