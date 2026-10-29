"""
AutoND D4 execution scheduler: two concurrent arm workers, one per GPU.

Execution only.  Nothing here changes the scientific design, the data, the
seeds, the model, the epoch budget, the stopping rule, the selection rule or
the decision rule: every fit is trained by ``AutoNDD4Adapter.train`` with the
identity the engine's ``run_d4`` would give it, and the result is assembled by
the unchanged single-owner ``run`` stage of
``audit.dataset.experiments.d4_autond_label_shuffle``.  No Gohr module is
imported.  Design reference (read, never imported): the Gohr two-GPU resume
script supplied by the protocol owner.

ROLES
-----
coordinator  One process.  Holds ``<run>/.d4_autond_scheduler.lock`` for its
             whole life, writes ONLY ``<run>/execution/<id>/`` (plan, logs,
             worker summaries, execution manifest), launches the workers,
             and, only if both succeed, launches the final ``run`` stage and
             ``verify --rerun-models``.  It never imports TensorFlow.
worker       One process per arm.  CUDA_VISIBLE_DEVICES is set by the
             coordinator in the child's environment before Python starts and
             is checked before TensorFlow is imported.  It holds
             ``<run>/fits/.arm_<condition>.lock`` and writes only its arm's
             fit directories (``fits/replicate_NN/<condition>/``) and, for the
             perturbed arm, the perturbation sidecars.  Fits run one at a time.
             Arm assignment: baseline (intact labels) on GPU 0, perturbed
             (shuffled labels) on GPU 1.

MODES (``execute`` requires an explicit mode and confirmation string)
--------------------------------------------------------------------
inspect          Read-only dry run (default): verified status of every fit,
                 locks, GPU inventory, what each worker would do.  Writes nothing.
execute two-gpu  Both workers concurrently; refuses unless nvidia-smi reports
                 GPUs 0 and 1.  --confirm EXECUTE_TWO_GPU_D4.
execute sequential
                 Explicit fallback: baseline worker, then perturbed worker, on
                 GPU 0 (or CPU with --cpu).  --confirm EXECUTE_SEQUENTIAL_D4.
                 Recorded as sequential; never reported as two-GPU execution.

RESTART: rerunning ``execute`` re-inspects every fit; finished fits are
verified and not retrained, partial fits resume at their next committed step
(adapter recovery), absent fits start.  A crash leaves its lock file behind;
locks are never broken automatically -- inspect and remove them by hand.
One worker's failure leaves the other's committed evidence untouched and
blocks finalisation.

Verified here: process isolation, locks, state transitions, CPU execution of
the real pipeline on tiny data.  NOT verified here: a physical dual-T4 run;
that needs the Kaggle runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ARM_GPU = {"baseline": "0", "perturbed": "1"}
CONFIRM = {"two-gpu": "EXECUTE_TWO_GPU_D4", "sequential": "EXECUTE_SEQUENTIAL_D4"}
EXECUTION_SCHEMA = "d4-autond-execution-1"
WORKER_SUMMARY_SCHEMA = "d4-autond-worker-summary-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_MODULE = "audit.dataset.experiments.d4_autond_label_shuffle"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_exclusive_json(path: Path, value: dict) -> None:
    """Create ``path`` (never replace it), fsynced."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def gpu_inventory() -> list[dict]:
    """GPUs reported by nvidia-smi (empty when there is none or no driver)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1], "uuid": parts[2],
                         "memory_total": parts[3] if len(parts) > 3 else None})
    return gpus


# ============================================================
# Inspection (read-only)
# ============================================================


def inspect_run(run_dir: Path, preregistration: Path, anchor: str) -> dict:
    """Verified status of every fit, locks and result; writes nothing."""
    import audit.dataset.experiments.d4_autond_label_shuffle as driver

    run_dir = Path(run_dir)
    record, values, anchor = driver.load_approved_preregistration(preregistration, anchor)
    preflight, data = driver.load_preflight(run_dir)
    design = driver.design_of(values, anchor)
    if preflight["design"] != design:
        raise driver.IntegrityError("The approved preregistration differs from the one preflight was run under.")
    fits = [driver.fit_status(run_dir, preflight, data, design, replicate=r, seed=seed, child=child,
                              condition=condition)
            for condition in ("baseline", "perturbed") for r, seed, child in driver.replicate_plan(design)]
    locks = {name: (json.loads(path.read_text()) if path.exists() else None)
             for name, path in (("scheduler", run_dir / driver.SCHEDULER_LOCK), ("run", run_dir / driver.RUN_LOCK),
                                ("arm_baseline", driver.arm_lock_path(run_dir, "baseline")),
                                ("arm_perturbed", driver.arm_lock_path(run_dir, "perturbed")))}
    return {"run_dir": str(run_dir), "preflight_sha256": _sha256(run_dir / "preflight.json"),
            "run_config_hash": preflight["run_config_hash"], "preregistration_sha256": anchor,
            "replicates": design["replicates"], "fits": fits, "locks": locks,
            "result_exists": (run_dir / "result").exists(),
            "work": {c: [f["replicate"] for f in fits if f["condition"] == c and f["status"] != "finished"]
                     for c in ("baseline", "perturbed")},
            "invalid": [f for f in fits if f["status"] == "invalid"]}


def print_plan(plan: dict, gpus: list) -> None:
    print(f"Run directory: {plan['run_dir']}  (result exists: {plan['result_exists']})")
    print(f"GPUs reported by nvidia-smi: {gpus or 'none'}")
    for fit in plan["fits"]:
        detail = fit.get("error") or f"{fit.get('completed_steps', 0)} committed steps, rounds {fit.get('rounds_trained')}"
        print(f"  {fit['condition']:9s} replicate {fit['replicate']}: {fit['status']:8s} {detail}")
    print(f"Locks: { {k: (v or {}).get('token') for k, v in plan['locks'].items()} }")
    for condition in ("baseline", "perturbed"):
        print(f"Worker {condition} (GPU {ARM_GPU[condition]} in two-gpu mode) would train/resume replicates "
              f"{plan['work'][condition] or 'none (all finished; verified, not retrained)'}")


# ============================================================
# Coordinator
# ============================================================


def _worker_command(args, condition: str, summary: Path, expected_visible: str) -> list[str]:
    return [sys.executable, "-B", "-m", "audit.dataset.experiments.d4_autond_two_gpu", "worker",
            "--run-dir", str(args.run_dir), "--preregistration", str(args.preregistration),
            "--preregistration-sha256", args.preregistration_sha256, "--autond-root", str(args.autond_root),
            "--condition", condition, "--summary", str(summary), "--expected-visible", expected_visible]


def _driver_command(args, stage: str, run_dir: Path) -> list[str]:
    if stage == "run":
        return [sys.executable, "-B", "-m", DRIVER_MODULE, "run", "--preregistration", str(args.preregistration),
                "--preregistration-sha256", args.preregistration_sha256, "--run-dir", str(run_dir),
                "--autond-root", str(args.autond_root)]
    return [sys.executable, "-B", "-m", DRIVER_MODULE, "verify", "--run-dir", str(run_dir), "--rerun-models"]


def execute(args) -> int:
    import audit.dataset.experiments.d4_autond_label_shuffle as driver

    mode = args.mode
    if args.confirm != CONFIRM[mode]:
        raise SystemExit(f"Refusing: execute {mode} requires --confirm {CONFIRM[mode]}.")
    run_dir = Path(args.run_dir).resolve()
    gpus = gpu_inventory()
    if mode == "two-gpu":
        if [g["index"] for g in gpus[:2]] != [0, 1]:
            raise SystemExit(f"Refusing two-gpu execution: nvidia-smi reports {gpus or 'no GPUs'}; GPUs 0 and 1 are "
                             f"required. Use 'execute sequential --confirm {CONFIRM['sequential']}' explicitly.")
    elif not args.cpu and not gpus:
        raise SystemExit("Refusing sequential execution: no GPU is reported; pass --cpu to run on the CPU explicitly.")
    plan = inspect_run(run_dir, args.preregistration, args.preregistration_sha256)
    print_plan(plan, gpus)
    if plan["result_exists"]:
        print("Result already exists; nothing to execute (it is never replaced).")
        return 0
    if plan["invalid"]:
        raise SystemExit(f"Refusing: invalid fit evidence {[(f['condition'], f['replicate']) for f in plan['invalid']]}"
                         " must be investigated first; nothing was changed.")

    script = Path(__file__).resolve()
    lock = run_dir / driver.SCHEDULER_LOCK
    token = driver.acquire_lock(lock, {"role": "scheduler", "mode": mode, "scheduler_sha256": _sha256(script)})
    try:
        execution_dir = run_dir / "execution" / f"{_utc().replace(':', '')}_{uuid.uuid4().hex[:8]}"
        execution_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": EXECUTION_SCHEMA, "purpose": "execution scheduling only; no design change",
            "mode": mode, "cpu": bool(args.cpu), "scheduler_path": str(script), "scheduler_sha256": _sha256(script),
            "scheduler_token": token, "started_utc": _utc(), "gpu_inventory": gpus,
            "arm_assignment": ({c: f"GPU {g}" for c, g in ARM_GPU.items()} if mode == "two-gpu" else
                               {"baseline": "CPU" if args.cpu else "GPU 0", "perturbed": "CPU" if args.cpu else "GPU 0"}),
            "physical_two_gpu_confirmed": False, "preflight_sha256": plan["preflight_sha256"],
            "run_config_hash": plan["run_config_hash"], "preregistration_sha256": plan["preregistration_sha256"],
            "initial_fit_states": plan["fits"],
        }
        _write_exclusive_json(execution_dir / "plan_manifest.json", manifest)
        env_base = {**os.environ, driver.TOKEN_ENV: token, "PYTHONUNBUFFERED": "1"}

        def launch(condition: str, visible: str):
            log = (execution_dir / f"{condition}_worker.log").open("xb")
            env = {**env_base, "CUDA_VISIBLE_DEVICES": visible}
            cmd = _worker_command(args, condition, execution_dir / f"{condition}_summary.json", visible)
            return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log

        exits, procs = {}, {}
        try:
            if mode == "two-gpu":
                procs = {c: launch(c, ARM_GPU[c]) for c in ("baseline", "perturbed")}
                while any(p.poll() is None for p, _ in procs.values()):
                    time.sleep(args.poll_seconds)
            else:
                for condition in ("baseline", "perturbed"):
                    procs[condition] = launch(condition, "" if args.cpu else "0")
                    procs[condition][0].wait()
                    if procs[condition][0].returncode != 0:
                        break
        finally:
            for condition, (proc, log) in procs.items():
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                exits[condition] = proc.returncode
                log.close()
        manifest.update(worker_exit_codes=exits, workers_finished_utc=_utc(), worker_summaries={
            c: json.loads((execution_dir / f"{c}_summary.json").read_text())
            for c in ("baseline", "perturbed") if (execution_dir / f"{c}_summary.json").is_file()})
        if set(exits) != {"baseline", "perturbed"} or any(code != 0 for code in exits.values()):
            manifest["finalization"] = "BLOCKED_WORKER_FAILURE_NO_RESULT_ATTEMPTED"
            _write_exclusive_json(execution_dir / "execution_manifest.json", manifest)
            print(f"A worker failed ({exits}); no result was assembled. All committed evidence is preserved: "
                  f"{execution_dir}", file=sys.stderr)
            return 2

        final_env = {**env_base, "CUDA_VISIBLE_DEVICES": "" if args.cpu else "0"}
        stages = {}
        for name in ("run", "verify"):
            cmd = _driver_command(args, name, run_dir)
            with (execution_dir / f"{name}.log").open("xb") as log:
                stages[name] = subprocess.run(cmd, cwd=str(REPO_ROOT), env=final_env, stdout=log,
                                              stderr=subprocess.STDOUT).returncode
            if stages[name] != 0:
                break
        manifest.update(final_stage_exit_codes=stages, finished_utc=_utc(),
                        finalization="COMPLETE" if stages == {"run": 0, "verify": 0} else f"FAILED_{stages}")
        _write_exclusive_json(execution_dir / "execution_manifest.json", manifest)
        print(f"Execution {manifest['finalization']}; provenance in {execution_dir}")
        return 0 if manifest["finalization"] == "COMPLETE" else 3
    finally:
        driver.release_lock(lock, token)


# ============================================================
# Worker
# ============================================================


def worker(args) -> int:
    """Train (or verify, if finished) every replicate of ONE condition, one fit at a time."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_visible:
        raise SystemExit(f"Worker expected CUDA_VISIBLE_DEVICES={args.expected_visible!r}, found {visible!r}.")
    if "tensorflow" in sys.modules:
        raise SystemExit("TensorFlow was imported before the worker checked its GPU assignment; refusing.")
    import audit.dataset.experiments.d4_autond_label_shuffle as driver

    run_dir, condition = Path(args.run_dir).resolve(), args.condition
    lock = run_dir / driver.SCHEDULER_LOCK
    if not lock.exists():
        raise SystemExit("A worker runs only under a scheduler that holds the scheduler lock; refusing.")
    driver.check_scheduler_permission(run_dir)
    if (run_dir / driver.RUN_LOCK).exists() or (run_dir / "result").exists():
        raise SystemExit("The run stage is active or a result exists; a worker must not train now.")
    arm_lock = driver.arm_lock_path(run_dir, condition)
    arm_token = driver.acquire_lock(arm_lock, {"role": "worker", "condition": condition})
    summary = {"schema_version": WORKER_SUMMARY_SCHEMA, "condition": condition, "cuda_visible_devices": visible,
               "started_utc": _utc(), "fits": [], "status": "FAILED"}
    try:
        import numpy as np
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if visible:
            if len(gpus) != 1:
                raise driver.IntegrityError(f"CUDA_VISIBLE_DEVICES={visible} but TensorFlow sees {len(gpus)} GPUs.")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            with tf.device("/GPU:0"):
                probe = tf.constant(1.0) + 1.0
            if "GPU:0" not in probe.device:
                raise driver.IntegrityError(f"A probe op was placed on {probe.device}, not the assigned GPU.")
            summary["tensorflow_gpu"] = {"devices": [g.name for g in gpus], "probe_device": probe.device,
                                         "details": tf.config.experimental.get_device_details(gpus[0])}
        elif gpus:
            raise driver.IntegrityError(f"CPU mode, but TensorFlow sees GPUs {gpus}.")
        else:
            summary["tensorflow_gpu"] = {"devices": [], "probe_device": "CPU"}

        from audit.dataset import d4_controlled_perturbation as d4
        from audit.dataset.integrity import labelled_partition_digest, partition_digest
        from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

        record, values, anchor = driver.load_approved_preregistration(args.preregistration, args.preregistration_sha256)
        preflight, data = driver.load_preflight(run_dir)
        design = driver.design_of(values, anchor)
        if preflight["design"] != design or preflight["run_config_hash"] != driver.run_config_hash(design):
            raise driver.IntegrityError("The approved design differs from the one preflight was run under.")
        if driver.source_identity(driver.SOURCE_FILES)["files"] != preflight["source_identity"]["files"]:
            raise driver.IntegrityError("AutoND D4 sources differ from those recorded at preflight.")
        binding = driver.make_binding(args.autond_root, design["starting_round"])
        if driver.upstream_identity(binding) != preflight["upstream_identity"]:
            raise driver.IntegrityError("The bound AutoND checkout differs from the one recorded at preflight.")
        factory = driver.make_adapter_factory(design, data, preflight["partitions"]["test"]["digest"], binding,
                                              driver.run_binding_of(preflight, run_dir))
        train_x, train_y = data["train"]
        source_digest = labelled_partition_digest(train_x, train_y)
        label_digest = partition_digest({"labels": train_y})
        shuffle = LabelShufflePerturbation()
        for replicate, seed, child in driver.replicate_plan(design):
            features, labels = train_x, train_y
            if condition == "perturbed":
                # Exactly the engine's run_d4 path, so the sidecar it later verifies is the one written here.
                rng = np.random.default_rng(child)
                indices = shuffle.permutation_indices(len(np.asarray(train_y).reshape(-1)), rng)
                fresh_x, fresh_y = shuffle.apply_indices(train_x, train_y, indices)
                directory = run_dir / "fits" / "perturbations"
                binding_args = dict(replicate=replicate, seed=seed, perturbation=shuffle,
                                    source_train_digest=source_digest, source_label_digest=label_digest,
                                    audit_seed=design["audit_seed"])
                if (directory / f"replicate_{replicate:02d}.perturbation.json").exists():
                    features, labels, _ = d4.load_verified_perturbation(
                        directory, train_features=train_x, train_labels=train_y, expected_indices=indices,
                        **binding_args)
                else:
                    shuffle.verify(train_x, train_y, fresh_x, fresh_y)
                    d4.save_perturbation(directory, features=fresh_x, labels=fresh_y, indices=indices, **binding_args)
                    features, labels = fresh_x, fresh_y
            fit_dir, state_path = d4.checkpoint_paths(run_dir / "fits", replicate, condition)
            before = driver.fit_status(run_dir, preflight, data, design, replicate=replicate, seed=seed, child=child,
                                       condition=condition)
            print(f"[{condition}] replicate {replicate}: {before['status']} "
                  f"({before.get('completed_steps', 0)} committed steps) -> train/resume/verify", flush=True)
            adapter = factory(seed)
            adapter.train(features, labels, checkpoint_path=fit_dir, state_path=state_path, replicate=replicate,
                          condition=condition, run_config_hash=preflight["run_config_hash"])
            selection = adapter.selection
            summary["fits"].append({
                "replicate": replicate, "seed": seed, "status_before": before["status"],
                "steps_before": before.get("completed_steps", 0), "completed_steps": selection["completed_steps"],
                "progression_status": selection["progression"]["status"],
                "rounds_trained": selection["progression"]["rounds_trained"],
                "selected_checkpoint_sha256": selection["checkpoint_sha256"],
                "record_chain_head": selection["record_chain_head"], "retries": selection["retries"]})
        summary["status"] = "ARM_COMPLETE_AND_VERIFIED"
        return 0
    except Exception as exc:  # noqa: BLE001 - recorded, then the worker fails closed
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{condition}] WORKER FAILED: {summary['error']}", file=sys.stderr, flush=True)
        return 1
    finally:
        summary["finished_utc"] = _utc()
        if args.summary:
            _write_exclusive_json(Path(args.summary), summary)
        driver.release_lock(arm_lock, arm_token)


# ============================================================
# CLI
# ============================================================


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoND D4 two-GPU execution scheduler (execution only).")
    sub = parser.add_subparsers(dest="command")
    for name in ("inspect", "execute", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--preregistration", type=Path, required=True)
        p.add_argument("--preregistration-sha256", required=True)
        p.add_argument("--autond-root", default=os.environ.get("AUTOND_ROOT", "~/Documents/AutoND-upstream"))
        if name == "execute":
            p.add_argument("mode", choices=tuple(CONFIRM))
            p.add_argument("--confirm", required=True)
            p.add_argument("--cpu", action="store_true", help="sequential mode only: run on the CPU explicitly")
            p.add_argument("--poll-seconds", type=float, default=10.0)
        if name == "worker":
            p.add_argument("--condition", choices=tuple(ARM_GPU), required=True)
            p.add_argument("--summary", type=Path, required=True)
            p.add_argument("--expected-visible", required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("choose inspect (read-only), execute or worker")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "worker":
        return worker(args)
    if args.command == "execute":
        if args.cpu and args.mode != "sequential":
            raise SystemExit("--cpu is only valid with sequential mode.")
        return execute(args)
    plan = inspect_run(args.run_dir, args.preregistration, args.preregistration_sha256)
    print_plan(plan, gpu_inventory())
    print(json.dumps({"work": plan["work"], "invalid": len(plan["invalid"]), "result_exists": plan["result_exists"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
