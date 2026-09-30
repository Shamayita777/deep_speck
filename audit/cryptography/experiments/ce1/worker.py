"""
CE1 per-arm worker processes and the 2-GPU launcher.

    Block N
     |-- worker (CUDA_VISIBLE_DEVICES=0) -> one arm
     `-- worker (CUDA_VISIBLE_DEVICES=1) -> paired arm

GPU visibility is fixed in the CHILD'S ENVIRONMENT at exec time, i.e.
before the child interpreter can import TensorFlow. The parent never
imports TensorFlow in parallel mode: GPU detection runs in a throwaway
subprocess, and dataset preparation / planning / certification are
numpy-only.

Mode is decided from GPUs ACTUALLY detected:
    0 GPUs -> sequential (in-process)
    1 GPU  -> sequential (in-process)
    2+     -> parallel workers, one GPU each

Isolation: each worker writes only its own arm directory; the shared
ledger is appended under an exclusive lock. Workers re-validate the
committed dataset and the arm state themselves and refuse if the state
changed since the parent planned (expected action mismatch).

Physical dual-GPU execution is NOT validated by the test-suite unless it
is run on a host with two GPUs; `cpu_process_isolation=True` exercises the
parallel code path with two concurrent CPU processes and is labelled as
such everywhere it is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKER_MODULE = "audit.cryptography.experiments.ce1.worker"


def detect_gpus(timeout: int = 180) -> dict:
    """Ask a SEPARATE interpreter what TensorFlow can see (parent stays TF-free)."""
    code = ("import json, tensorflow as tf; "
            "print('CE1GPU' + json.dumps([d.name for d in "
            "tf.config.list_physical_devices('GPU')]))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=timeout)
        line = [l for l in out.stdout.splitlines() if l.startswith("CE1GPU")]
        devices = json.loads(line[-1][len("CE1GPU"):]) if line else []
        err = None if line else (out.stderr.strip().splitlines() or ["no output"])[-1]
    except Exception as exc:                          # noqa: BLE001
        devices, err = [], f"{type(exc).__name__}: {exc}"
    n = len(devices)
    return {"gpus_visible": n, "devices": devices, "error": err,
            "mode": "parallel" if n >= 2 else "sequential",
            "method": "tensorflow list_physical_devices in a subprocess"}


def _repo_root_of_package() -> Path:
    return Path(__file__).resolve().parents[4]


def _child_env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu                  # set BEFORE the child imports TF
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    root = str(_repo_root_of_package())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def make_executors(run_dir, *, gpu_indices=(0, 1), cpu_process_isolation: bool = False,
                   log_dir=None):
    """
    Build (train_executor, evaluate_executor) for controller.run_block.
    Each launches one worker process per arm, concurrently, and waits.
    """
    from audit.cryptography.experiments.ce1.controller import ledger_for

    run_dir = Path(run_dir)
    log_dir = Path(log_dir) if log_dir else run_dir / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger_for(run_dir)
    config_hash = json.loads((run_dir / "run_manifest.json").read_text())["config_fingerprint"]

    def _launch(block_index: int, tasks: dict, task: str) -> dict:
        procs = {}
        for slot, (arm, action) in enumerate(tasks.items()):
            gpu = "" if cpu_process_isolation else str(gpu_indices[slot % len(gpu_indices)])
            cmd = [sys.executable, "-m", WORKER_MODULE, "--run-dir", str(run_dir),
                   "--block", str(block_index), "--arm", arm, "--task", task,
                   "--expect-action", action]
            log = open(log_dir / f"block{block_index}_{arm}_{task}.log", "ab")
            ledger.record("WORKER_LAUNCH", block_id=f"block{block_index}", arm=arm,
                          config_hash=config_hash, task=task, action=action,
                          cuda_visible_devices=gpu,
                          mode=("cpu_process_isolation" if cpu_process_isolation
                                else "gpu_parallel"))
            procs[arm] = (subprocess.Popen(cmd, env=_child_env(gpu), stdout=log,
                                           stderr=subprocess.STDOUT), log)
        errors = {}
        for arm, (p, log) in procs.items():
            rc = p.wait()
            log.close()
            ledger.record("WORKER_EXIT", block_id=f"block{block_index}", arm=arm,
                          config_hash=config_hash, task=task, returncode=rc)
            if rc != 0:
                errors[arm] = f"worker exited with {rc}"
        return errors

    def train_executor(block_index: int, actions: dict) -> dict:
        return _launch(block_index, actions, "train")

    def evaluate_executor(block_index: int, arms: list) -> dict:
        return _launch(block_index, {a: "EVALUATE" for a in arms}, "evaluate")

    return train_executor, evaluate_executor


def run_auto(run_dir, config, *, gpu_info=None, cpu_process_isolation=False, **kw) -> dict:
    """
    Pick the execution mode from DETECTED GPUs and run one controller pass.
    `cpu_process_isolation=True` forces the parallel path on CPU (tests).
    """
    from audit.cryptography.experiments.ce1 import controller as C

    info = gpu_info if gpu_info is not None else detect_gpus()
    parallel = info["gpus_visible"] >= 2 or cpu_process_isolation
    C.open_run(run_dir, config, repo_root=kw.get("repo_root"))   # manifest before workers
    if parallel:
        tr, ev = make_executors(run_dir, cpu_process_isolation=cpu_process_isolation
                                and info["gpus_visible"] < 2)
        out = C.run_controller(run_dir, config, train_executor=tr, evaluate_executor=ev, **kw)
    else:
        out = C.run_controller(run_dir, config, **kw)
    out["execution"] = {
        "gpu_detection": info,
        "mode": ("parallel_workers" if parallel else "sequential_in_process"),
        "physical_dual_gpu": info["gpus_visible"] >= 2,
        "cpu_process_isolation": bool(cpu_process_isolation and info["gpus_visible"] < 2),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CE1 single-arm worker")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--arm", choices=("baseline", "destroyed"), required=True)
    ap.add_argument("--task", choices=("train", "evaluate"), required=True)
    ap.add_argument("--expect-action", required=True)
    args = ap.parse_args(argv)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    from audit.cryptography.experiments.ce1 import controller as C
    from audit.cryptography.experiments.ce1 import resume as R

    config = C.load_run_config(args.run_dir)
    C.open_run(args.run_dir, config)                  # fingerprint re-verified
    sealed = C.ensure_sealed(args.run_dir, config)    # existing set only; never generates
    bid = R.block_id_for(args.block)
    action, reasons = C.decide_arm(args.run_dir, config, bid, args.arm, sealed)
    if action != args.expect_action:
        print(f"REFUSED: planned {args.expect_action}, state now says {action}: {reasons}")
        return 3
    print(f"worker {bid}/{args.arm} task={args.task} action={action} "
          f"CUDA_VISIBLE_DEVICES={visible!r} pid={os.getpid()}", flush=True)
    if args.task == "train":
        ctx = C.load_block_context(args.run_dir, config, args.block, sealed)
        C.train_arm(args.run_dir, config, ctx, args.arm, action)
    else:
        C.evaluate_arm(args.run_dir, config, bid, args.arm, sealed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
