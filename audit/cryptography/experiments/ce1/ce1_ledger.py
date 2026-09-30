"""
Append-only CE1 resume ledger (EV-adapted).

Mirrors EV's ResumeLedger invariant: a resumed run must not change the
scientific experiment, so a resume event whose configuration hash differs
from the one recorded at checkpoint time raises rather than proceeding.

Engineering only; CE1's frozen design is untouched.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class ResumeViolation(RuntimeError):
    pass


class LedgerCorrupted(RuntimeError):
    """The ledger cannot be parsed; the controller refuses to act on it."""


#: Production event vocabulary. Anything else is rejected at write time so a
#: typo cannot create an event class a reader does not know about.
PRODUCTION_EVENTS = frozenset({
    "RUN_START", "DATASET_PREPARED", "DATASET_LOADED", "DATASET_PREPARE_RESTARTED",
    "TRAIN_START", "CHECKPOINT", "RESUME", "RERUN", "SKIP", "TRAINING_COMPLETE",
    "EVALUATION", "EVALUATED", "FAILURE", "REFUSED", "BLOCK_VALID",
    "LEGACY_RECOVERY", "LEDGER_REPAIR", "WORKER_LAUNCH", "WORKER_EXIT", "FINALIZE",
    "INTERRUPTED", "BLOCK_FAILED", "BLOCK_LEGACY_EVALUATED", "PAUSED",
    "OPERATOR_ACTION", "BLOCK_NOT_COUNTED",
})


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


#: Resume semantics CE1 can actually support. Recorded in every event so a
#: reader never has to infer which claim is being made.
RESUME_SEMANTICS = {
    # Claim levels are kept SEPARATE. Infrastructure existing is not the same
    # as production resume being demonstrated; the production level may only
    # be raised on the evidence of a real interruption -> restart -> resume
    # integration test (see tests/test_ce1_integration.py and the release note).
    "checkpoint_state_resume_infrastructure": "IMPLEMENTED",
    "checkpoint_state_resume_production": "SUPPORTED_BY_INTEGRATION_TEST",
    "checkpoint_state_resume_evidence": (
        "tests/test_ce1_integration.py::test_real_sigkill_interruption_restart_resume - real "
        "SIGKILL of the controller process, new process, real GohrTrainer/model.fit with "
        "initial_epoch, optimizer + LR restored; CPU, toy scale. NOT yet exercised at "
        "production scale (10^7 samples) or on GPU."),
    "bit_exact_trajectory_resumable": False,
    "restored": ["model_weights", "optimizer_state", "epoch_counter",
                 "lr_schedule_position", "training_dataset", "validation_dataset",
                 "sealed_evaluation_set"],
    "not_restored": ["minibatch_shuffle_rng", "dropout_rng"],
    "note": ("Keras exposes no shuffle/dropout RNG state, so epochs after a resume do not "
             "follow the minibatch stream an uninterrupted run would have produced. "
             "shuffle_stream_continuity is recorded as false. CE1's training behaviour is "
             "NOT altered (shuffle is not forced off, no deterministic input pipeline) "
             "because that would be a protocol amendment."),
}


@dataclass
class LedgerEvent:
    event: str
    block_id: str
    arm: str
    config_hash: str
    detail: dict


class CE1Ledger:
    """Append-only JSONL record of arm states and resume events."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def _append(self, record: dict) -> None:
        """
        Durable append: one write() of one complete line under an exclusive
        lock, then fsync. Two arm workers may append concurrently; the lock
        prevents interleaved lines and O_APPEND keeps each line whole.
        """
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            except ImportError:                    # non-POSIX: best effort
                fcntl = None
            os.write(fd, line)
            os.fsync(fd)
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def record(self, event: str, *, block_id: str | None, arm: str | None,
               config_hash: str | None, **detail) -> None:
        """Record one production event (see PRODUCTION_EVENTS)."""
        if event not in PRODUCTION_EVENTS:
            raise ValueError(f"unknown ledger event {event!r}")
        self._append({"event": event, "block_id": block_id, "arm": arm,
                      "config_hash": config_hash, "timestamp": utc(),
                      "pid": os.getpid(), **detail})

    def verify(self) -> int:
        """Parse every line; raise LedgerCorrupted on any malformed record."""
        return len(self.events())

    def repair_torn_tail(self) -> dict | None:
        """
        OPERATOR ACTION, never automatic. If (and only if) the LAST line is an
        unterminated, unparseable fragment - the signature of a process killed
        mid-append - move that fragment to a sidecar file and log the repair.
        Any malformed line that is NOT a torn tail is left alone and remains a
        hard error.
        """
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return None
        head, _, tail = raw.rpartition(b"\n")
        try:
            json.loads(tail.decode("utf-8"))
            return None                            # complete record, just unterminated
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        side = self.path.with_suffix(self.path.suffix + f".torn.{int(datetime.now().timestamp())}")
        side.write_bytes(tail)
        tmp = self.path.with_suffix(self.path.suffix + ".repair.tmp")
        tmp.write_bytes(head + b"\n" if head else b"")
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self.record("LEDGER_REPAIR", block_id=None, arm=None, config_hash=None,
                    quarantined_to=side.name, n_bytes=len(tail))
        return {"quarantined_to": str(side), "n_bytes": len(tail)}

    def record_state(self, *, block_id, arm, status, last_completed_epoch,
                     config_hash, checkpoint_hash=None, dataset_hash=None) -> None:
        self._append({"event": "state", "block_id": block_id, "arm": arm,
                      "status": status, "last_completed_epoch": last_completed_epoch,
                      "config_hash": config_hash, "checkpoint_hash": checkpoint_hash,
                      "dataset_hash": dataset_hash, "timestamp": utc()})

    def record_resume(self, *, block_id, arm, previous_config_hash,
                      resumed_config_hash, from_epoch) -> None:
        """Fail closed if the configuration changed since the checkpoint."""
        if previous_config_hash != resumed_config_hash:
            raise ResumeViolation(
                f"Refusing to resume {block_id}/{arm}: configuration hash changed from "
                f"{previous_config_hash} to {resumed_config_hash}. A resumed run must not "
                "change the scientific experiment.")
        self._append({"event": "resume", "block_id": block_id, "arm": arm,
                      "previous_config_hash": previous_config_hash,
                      "resumed_config_hash": resumed_config_hash,
                      "from_epoch": from_epoch,
                      "resume_semantics": RESUME_SEMANTICS,
                      "shuffle_stream_continuity": False,
                      "timestamp": utc()})

    def record_skip(self, *, block_id, arm, reason, config_hash) -> None:
        self._append({"event": "skip", "block_id": block_id, "arm": arm,
                      "reason": reason, "config_hash": config_hash, "timestamp": utc()})

    def events(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerCorrupted(
                    f"{self.path}: line {n} is malformed ({exc}). The controller refuses to "
                    "act on a ledger it cannot read; if this is a torn final line from a "
                    "killed process, run CE1Ledger.repair_torn_tail() explicitly.") from exc
            if not isinstance(rec, dict) or "event" not in rec:
                raise LedgerCorrupted(f"{self.path}: line {n} is not a ledger record")
            out.append(rec)
        return out

    def last_state_for(self, block_id, arm):
        last = None
        for r in self.events():
            if r.get("event") == "state" and r["block_id"] == block_id and r["arm"] == arm:
                last = r
        return last
