"""
CE1 fault-tolerant execution: block- and epoch-level resume, 2-GPU scheduling.

INFRASTRUCTURE ONLY. Nothing here changes the frozen CE1 experiment: the
estimand, K=8, min_valid=6, failure tolerance, sample counts, sealed
evaluation set, randomized arm assignment, assignment_seed, alpha, test
and FINAL_EPOCH=200 are untouched. Resume is a reliability mechanism, and
it must never become a way to select favourable blocks.

WHAT EXACT RESUME REQUIRES
--------------------------
Model weights alone are NOT sufficient. Continuing arm training at epoch
N+1 exactly as an uninterrupted run would have requires:

  * model weights                  (saved)
  * optimizer state                (saved; include_optimizer=True)
  * epoch counter                  (saved, as last_completed_epoch)
  * LR schedule position           (derived from the epoch counter -- the
                                    frozen schedule is a pure function of
                                    the epoch index, cycle length 10)
  * the TRAINING DATA              (must be persisted: generated from
                                    os.urandom and NOT seed-replayable)
  * the sealed evaluation set      (persisted and hash-verified)
  * configuration fingerprint      (to refuse cross-run mixing)

Minibatch shuffle order and dropout RNG are NOT restored: Keras does not
expose that state. Resume is therefore EXACT in weights, optimizer state,
LR position, data and epoch index, but the post-resume shuffle stream
differs from the uninterrupted one. This is documented rather than
papered over, and is recorded in every resumed arm's state as
`shuffle_stream_continuity: false`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATE_SCHEMA = "ce1-arm-state-2"
BLOCK_SCHEMA = "ce1-block-state-1"
#: Arm names are the PRODUCTION names used by the seed manifest, the legacy
#: run and the certificate. Schema 1 used "intact"/"block_00", which no
#: production path ever wrote, so a scan of a real run directory would have
#: found nothing. Schema 1 files are refused by load_arm_state.
ARMS = ("baseline", "destroyed")


def block_id_for(i: int) -> str:
    """Block identifier, identical to seed_manifest() keys and the legacy run."""
    return f"block{i}"


# ------------------------------------------------------------- arm statuses
NOT_STARTED = "NOT_STARTED"
IN_PROGRESS = "IN_PROGRESS"
#: Epoch 200 reached and the terminal model persisted. NOT scientific
#: completion: the arm contributes nothing until it is EVALUATED.
TRAINING_COMPLETE = "TRAINING_COMPLETE"
#: Terminal model scored on the sealed set; result bound to model hash and
#: sealed-set hash. Only EVALUATED arms can form a valid CE1 block.
EVALUATED = "EVALUATED"
FAILED = "FAILED"
ARM_STATUSES = (NOT_STARTED, IN_PROGRESS, TRAINING_COMPLETE, EVALUATED, FAILED)


class ResumeRefused(RuntimeError):
    """Raised whenever continuation would mix incompatible state."""


# ---------------------------------------------------------------- atomic IO

def atomic_write_json(path, payload: dict) -> Path:
    """
    Write JSON atomically: temp file -> flush -> fsync -> rename.

    A crash can then leave the old state or the new state, never a
    half-written one, so a partially written epoch can never be mistaken
    for a completed one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- fingerprint

def config_fingerprint(spec: dict, reference: dict, source_version: str | None = None) -> str:
    """Bind every artifact to the exact frozen design + reference configuration."""
    payload = {"design": spec, "reference": reference, "source_version": source_version}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------- arm state

@dataclass
class ArmState:
    experiment_id: str
    block_id: str
    arm: str                       # "baseline" | "destroyed"
    status: str                    # one of ARM_STATUSES
    last_completed_epoch: int
    terminal_epoch: int
    model_seed: int
    assignment_seed: int
    assignment_flip: int
    permutation_seed: int
    config_fingerprint: str
    design_version: str
    train_data_sha256: str | None = None
    validation_data_sha256: str | None = None
    sealed_evaluation_sha256: str | None = None
    model_artifact: str | None = None
    model_sha256: str | None = None
    optimizer_state_included: bool = False
    shuffle_stream_continuity: bool = True
    destroyed_labels_sha256: str | None = None
    #: The arm's training LABELS hash (baseline labels or permuted labels).
    train_labels_sha256: str | None = None
    history: list = field(default_factory=list)
    failure: dict | None = None
    best_val_loss: float | None = None
    terminal_model_artifact: str | None = None
    terminal_model_sha256: str | None = None
    evaluation: dict | None = None
    resume_count: int = 0
    rerun_count: int = 0
    #: Exceptions raised by training/evaluation code (NOT interruptions such
    #: as SIGKILL / KeyboardInterrupt / SystemExit, which never count).
    failure_count: int = 0
    #: Recorded operator GRANT_RETRY actions (each re-arms the retry caps once).
    operator_retry_grants: int = 0
    #: Optimizer step counter of the terminal artifact, read at TRAINING_COMPLETE.
    terminal_optimizer_iterations: int | None = None
    #: Canonical hash of evaluation.json (bound into the ledger EVALUATED event).
    evaluation_sha256: str | None = None
    provenance: str = "controller"
    schema: str = STATE_SCHEMA
    updated_utc: str = ""

    @property
    def next_epoch(self) -> int:
        return self.last_completed_epoch + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["next_epoch"] = self.next_epoch
        return d


def arm_dir(run_dir, block_id, arm) -> Path:
    return Path(run_dir) / "blocks" / block_id / arm


def arm_state_path(run_dir, block_id, arm) -> Path:
    return arm_dir(run_dir, block_id, arm) / "state.json"


def rel_to_run(run_dir, path) -> str:
    """Store artifact paths RELATIVE to the run directory; refuse anything outside it."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(run_dir).resolve()))
    except ValueError as exc:
        raise ResumeRefused(f"artifact {path} is outside the run directory {run_dir}") from exc


def resolve_artifact(run_dir, recorded: str | None) -> Path | None:
    """
    Resolve a recorded artifact path strictly inside the run directory.
    Absolute paths, '..' traversal and symlink escapes fail closed.
    """
    if not recorded:
        return None
    from audit.cryptography.experiments.ce1.ce1_datasets import (
        PathContainmentError, contained_path)
    try:
        return contained_path(run_dir, recorded)
    except PathContainmentError as exc:
        raise ResumeRefused(f"artifact path refused: {exc}") from exc


def load_arm_state(run_dir, block_id, arm) -> ArmState | None:
    p = arm_state_path(run_dir, block_id, arm)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ResumeRefused(f"{p}: state file unreadable ({exc}); refusing to guess") from exc
    if raw.get("schema") != STATE_SCHEMA:
        raise ResumeRefused(
            f"{p}: unknown state schema {raw.get('schema')!r}; refusing to interpret "
            "artifacts written by a different version.")
    raw.pop("next_epoch", None)
    if raw.get("status") not in ARM_STATUSES:
        raise ResumeRefused(f"{p}: unknown arm status {raw.get('status')!r}")
    try:
        return ArmState(**raw)
    except TypeError as exc:
        raise ResumeRefused(f"{p}: state fields do not match schema: {exc}") from exc


def save_arm_state(run_dir, state: ArmState) -> ArmState:
    state.updated_utc = datetime.now(timezone.utc).isoformat()
    atomic_write_json(arm_state_path(run_dir, state.block_id, state.arm), state.to_dict())
    return state


def mark_resumed(state: ArmState) -> ArmState:
    """
    Flag an arm as continued rather than uninterrupted.

    CE1 supports CHECKPOINT/STATE resume, not bit-exact trajectory resume:
    the shuffle and dropout RNG streams cannot be restored, so this is
    recorded explicitly rather than left to inference.
    """
    state.shuffle_stream_continuity = False
    return state


def record_completed_epoch(run_dir, state: ArmState, *, epoch: int, model=None,
                           keep_last: int = 2, logs: dict | None = None,
                           learning_rate: float | None = None) -> ArmState:
    """
    Mark ONE epoch complete, atomically.

    Order matters: the model artifact is written and hashed FIRST, and only
    then is the state marker replaced. If the process dies between the two,
    the epoch stays incomplete and is simply redone -- never silently
    counted.
    """
    d = arm_dir(run_dir, state.block_id, state.arm)
    d.mkdir(parents=True, exist_ok=True)
    if model is not None:
        target = d / f"epoch_{epoch:03d}.keras"
        # Keras 3 validates the EXTENSION: a name ending in ".keras.tmp" is
        # rejected outright, so the temp file must itself end in ".keras".
        # The placeholder is removed first because Keras writes a fresh zip.
        fd, tmp = tempfile.mkstemp(dir=str(d), suffix=".partial.keras")
        os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        try:
            model.save(tmp, include_optimizer=True)      # weights + optimizer state
            if not Path(tmp).exists():
                raise RuntimeError(f"model.save produced no file at {tmp}")
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        state.model_artifact = rel_to_run(run_dir, target)
        state.model_sha256 = sha256_file(target)
        state.optimizer_state_included = True
        # rolling retention: keep the last `keep_last` epoch artifacts
        kept = sorted(d.glob("epoch_*.keras"))
        for old in kept[:-keep_last] if keep_last else []:
            old.unlink(missing_ok=True)
    if logs is not None or learning_rate is not None:
        entry = {"epoch": epoch, "learning_rate": learning_rate,
                 **{k: float(v) for k, v in (logs or {}).items()}}
        state.history.append(entry)
        vl = entry.get("val_loss")
        if vl is not None and (state.best_val_loss is None or vl < state.best_val_loss):
            state.best_val_loss = vl
    state.last_completed_epoch = epoch
    state.status = TRAINING_COMPLETE if epoch >= state.terminal_epoch else IN_PROGRESS
    state.failure = None
    return save_arm_state(run_dir, state)


def record_arm_failure(run_dir, state: ArmState, exc: BaseException) -> ArmState:
    """Persist failure WITHOUT destroying completed epochs."""
    state.status = FAILED
    state.failure_count += 1
    state.failure = {"reason": type(exc).__name__, "detail": str(exc),
                     "failed_after_epoch": state.last_completed_epoch,
                     "utc": datetime.now(timezone.utc).isoformat()}
    state.updated_utc = state.failure["utc"]
    atomic_write_json(arm_state_path(run_dir, state.block_id, state.arm), state.to_dict())
    return state


# ---------------------------------------------------------------- validation

def validate_arm_for_resume(run_dir, state: ArmState, *, expected: dict) -> list:
    """Return the list of reasons this arm may NOT be resumed (empty = OK)."""
    problems = []
    for key in ("config_fingerprint", "design_version", "assignment_seed",
                "assignment_flip", "model_seed", "permutation_seed",
                "sealed_evaluation_sha256", "terminal_epoch"):
        if key in expected and expected[key] is not None:
            actual = getattr(state, key)
            if actual != expected[key]:
                problems.append(f"{key}: artifact {actual!r} != current {expected[key]!r}")
    if state.status in (TRAINING_COMPLETE, EVALUATED):
        problems += validate_terminal_artifact(run_dir, state)
        if state.status == EVALUATED:
            problems += validate_evaluation_record(state, expected=expected)
        return problems
    if state.last_completed_epoch > 0:
        if not state.model_artifact:
            problems.append("no model artifact recorded for a partially trained arm")
        else:
            try:
                p = resolve_artifact(run_dir, state.model_artifact)
            except ResumeRefused as exc:
                return problems + [str(exc)]
            if not p.exists():
                problems.append(f"model artifact missing: {p}")
            elif state.model_sha256 and sha256_file(p) != state.model_sha256:
                problems.append(f"model artifact corrupted (hash mismatch): {p}")
            elif not state.optimizer_state_included:
                problems.append(
                    "artifact does not include optimizer state; exact epoch continuation "
                    "is not possible from weights alone")
        if not state.train_data_sha256:
            problems.append(
                "training data hash absent: the training set is generated from os.urandom "
                "and is not seed-replayable, so exact continuation requires the persisted "
                "training arrays")
    return problems


def validate_terminal_artifact(run_dir, state: ArmState) -> list:
    """A TRAINING_COMPLETE/EVALUATED arm must point at an intact terminal model."""
    problems = []
    if state.last_completed_epoch != state.terminal_epoch:
        problems.append(f"status {state.status} but last_completed_epoch="
                        f"{state.last_completed_epoch} != terminal {state.terminal_epoch}")
    try:
        p = resolve_artifact(run_dir, state.terminal_model_artifact)
    except ResumeRefused as exc:
        return problems + [str(exc)]
    if p is None:
        problems.append("no terminal model artifact recorded")
    elif not p.exists():
        problems.append(f"terminal model artifact missing: {p}")
    elif not state.terminal_model_sha256:
        problems.append("terminal model hash not recorded; integrity unverifiable")
    elif sha256_file(p) != state.terminal_model_sha256:
        problems.append(f"terminal model artifact corrupted (hash mismatch): {p}")
    return problems


def validate_evaluation_record(state: ArmState, *, expected: dict) -> list:
    """An EVALUATED arm's result must be bound to its model and the sealed set."""
    ev = state.evaluation or {}
    problems = []
    for key in ("accuracy", "model_sha256", "sealed_evaluation_sha256", "n_samples"):
        if ev.get(key) is None:
            problems.append(f"evaluation record missing {key!r}")
    if ev.get("model_sha256") and ev["model_sha256"] != state.terminal_model_sha256:
        problems.append("evaluation was computed on a different model than the recorded "
                        "terminal artifact")
    want = expected.get("sealed_evaluation_sha256")
    if want and ev.get("sealed_evaluation_sha256") and ev["sealed_evaluation_sha256"] != want:
        problems.append("evaluation used a different sealed set than this run")
    return problems


RESUME_SEMANTICS_CLAIM = {
    "checkpoint_state_resume_infrastructure": "IMPLEMENTED",
    "checkpoint_state_resume_production": "SUPPORTED_BY_INTEGRATION_TEST",
    "checkpoint_state_resume_evidence": (
        "tests/test_ce1_integration.py::test_real_sigkill_interruption_restart_resume - real "
        "SIGKILL of the controller process, new process, real GohrTrainer/model.fit with "
        "initial_epoch, optimizer + LR restored; CPU, toy scale. NOT yet exercised at "
        "production scale (10^7 samples) or on GPU."),
    "bit_exact_trajectory_resumable": False,
}


def scan_run(run_dir, *, n_blocks: int, expected: dict) -> dict:
    """
    Read-only audit of an existing run. Never trains, never repairs.
    """
    run_dir = Path(run_dir)
    blocks = {}
    for i in range(n_blocks):
        bid = block_id_for(i)
        entry = {"block_id": bid, "arms": {}}
        for arm in ARMS:
            st = load_arm_state(run_dir, bid, arm)
            if st is None:
                entry["arms"][arm] = {"status": "NOT_STARTED", "last_completed_epoch": 0,
                                      "next_epoch": 1, "resumable": True, "problems": []}
                continue
            problems = validate_arm_for_resume(run_dir, st, expected=expected)
            entry["arms"][arm] = {
                "status": st.status,
                "last_completed_epoch": st.last_completed_epoch,
                "next_epoch": st.next_epoch,
                "resumable": not problems,
                "problems": problems,
                "model_artifact": st.model_artifact,
            }
        statuses = {a["status"] for a in entry["arms"].values()}
        if statuses == {EVALUATED}:
            entry["block_status"] = "COMPLETE"
        elif statuses == {"NOT_STARTED"}:
            entry["block_status"] = "NOT_STARTED"
        else:
            entry["block_status"] = "PARTIAL"
        blocks[bid] = entry
    refusals = [f"{b}/{a}: {p}" for b, e in blocks.items()
                for a, v in e["arms"].items() for p in v["problems"]]
    return {"run_dir": str(run_dir), "n_blocks": n_blocks, "blocks": blocks,
            "resume_refused": bool(refusals), "refusal_reasons": refusals,
            "resume_semantics": RESUME_SEMANTICS_CLAIM,
            "category": "resumable_artifacts"}


# ---------------------------------------------------------------- scheduler

def plan_resume(report: dict, *, terminal_epoch: int, n_gpus: int = 2) -> dict:
    """
    Decide what to run, without running it.

    Within a block the two arms are independent, so they go to different
    GPUs. A COMPLETE arm is never relaunched; a COMPLETE block is skipped
    entirely.
    """
    if report["resume_refused"]:
        raise ResumeRefused("; ".join(report["refusal_reasons"]))
    plan = []
    for bid, entry in report["blocks"].items():
        if entry["block_status"] == "COMPLETE":
            plan.append({"block_id": bid, "action": "SKIP", "reason": "block COMPLETE"})
            continue
        tasks = []
        for idx, arm in enumerate(ARMS):
            a = entry["arms"][arm]
            if a["status"] == EVALUATED:
                tasks.append({"arm": arm, "action": "SKIP", "gpu": None,
                              "reason": "arm EVALUATED"})
            elif a["status"] == TRAINING_COMPLETE:
                tasks.append({"arm": arm, "action": "EVALUATE", "gpu": None,
                              "reason": "terminal model present, not yet evaluated; "
                                        "no retraining"})
            else:
                tasks.append({"arm": arm, "action": "TRAIN",
                              "gpu": idx % max(n_gpus, 1),
                              "from_epoch": a["next_epoch"], "to_epoch": terminal_epoch})
        plan.append({"block_id": bid, "action": "RUN", "tasks": tasks,
                     "parallel": sum(1 for t in tasks if t["action"] == "TRAIN") > 1
                                 and n_gpus >= 2})
    return {"terminal_epoch": terminal_epoch, "n_gpus_requested": n_gpus, "plan": plan}


def format_plan(plan: dict) -> str:
    out = []
    for blk in plan["plan"]:
        if blk["action"] == "SKIP":
            out.append(f"{blk['block_id']}: COMPLETE -> SKIP")
            continue
        out.append(f"{blk['block_id']}:")
        for t in blk["tasks"]:
            if t["action"] == "SKIP":
                out.append(f"    {t['arm']:<10} EVALUATED -> skip")
            elif t["action"] == "EVALUATE":
                out.append(f"    {t['arm']:<10} TRAINING_COMPLETE -> evaluate only")
            else:
                out.append(f"    {t['arm']:<10} GPU {t['gpu']} -> epochs "
                           f"{t['from_epoch']}-{t['to_epoch']}")
    return "\n".join(out)


def pin_gpu(index: int | None) -> str:
    """
    Bind THIS process to exactly one GPU.

    Must be called before TensorFlow initialises; TF reads
    CUDA_VISIBLE_DEVICES once. Returns the value set.
    """
    value = "" if index is None else str(index)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return value


def gpu_parallelism_available(n_required: int = 2) -> dict:
    """Honest report -- never claim parallelism that the host cannot provide."""
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
    except Exception as exc:                      # noqa: BLE001
        return {"gpus_visible": 0, "parallel_active": False, "error": str(exc),
                "mode": "sequential_fallback"}
    n = len(devices)
    return {"gpus_visible": n, "parallel_active": n >= n_required,
            "mode": "parallel" if n >= n_required else "sequential_fallback",
            "devices": [d.name for d in devices]}
