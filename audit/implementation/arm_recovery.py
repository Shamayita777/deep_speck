#!/usr/bin/env python3
"""
Per-arm epoch logging and transactional mid-arm recovery - GENERIC.

No Gohr, Speck or cipher knowledge. Keras is imported lazily, inside the
callback factory only, so this module and its tests run without
TensorFlow.

THE PROBLEM. One arm is 200 epochs and 3.7-4.6 GPU hours. An
interruption at epoch 190 must not cost 190 epochs.

AUTHORITY. The commit manifest in `durable_state.CommittedCheckpointStore`
is the SINGLE source of truth for what has been durably committed. This
module's epoch log is a DERIVED record: it carries the per-epoch metrics
the evidence trail needs, and it has no authority over what is
resumable. An earlier design gave the log, the checkpoint and the arm
record independent authority, and two demonstrated crash windows made
them disagree permanently and brick the arm. A log that lags the
manifest by an epoch, or ends in a torn line from a hard kill, is now
reconciled and noted - never fatal.

WHAT IS RECOVERED, AND WHAT IS NOT
----------------------------------
Every claim below was MEASURED against real Keras, not inferred from
documentation, by `audit/tests/test_arm_recovery_keras.py`.

  RECOVERED EXACTLY (measured on TensorFlow 2.20.0 / Keras 3.15.1)
    model weights            element-wise equality across the reload
    optimizer state          Adam's step counter and BOTH moment
                             estimates, compared variable by variable
    completed epoch index    fit() resumes at initial_epoch; the log is
                             a contiguous 0..N-1 history
    learning-rate schedule   the rate in force at every epoch of a
                             resumed run equals the rate at the same
                             epoch of an uninterrupted control
    best-checkpoint record   the monitored best value is restored into
                             the ModelCheckpoint callback, so a resumed
                             arm cannot overwrite a better checkpoint
                             with a worse one (a fresh callback starts at
                             infinity and would)
    minibatch shuffle        Keras derives each epoch's permutation from
                             the ABSOLUTE epoch index, so a resumed arm
                             reseeded to its recorded seed reproduces the
                             uninterrupted run's batch sequence - and its
                             final weights - bit for bit

  This last entry was first recorded as NOT recovered, on the reasoning
  that Keras exposes no shuffle RNG state. Measurement showed that
  reasoning wrong and the claim was corrected UPWARDS. It is an
  empirical property of this Keras version rather than a documented
  guarantee, so the test re-measures it: a stack that behaves otherwise
  fails the suite instead of silently invalidating the claim. A claim
  weaker than the evidence is as wrong as one stronger than it.

  NOT RECOVERED
    within-epoch position    recovery is at EPOCH granularity. An
                             interruption part-way through an epoch
                             redoes that epoch from its start. No
                             partial-epoch record is ever written, so a
                             redone epoch cannot be mistaken for a
                             completed one.

  CONSEQUENCE. On the verified stack an interrupted-and-resumed arm is a
  bit-exact continuation, so resuming does not change the arm's
  scientific status. It is RECORDED anyway: every interruption is
  appended to the arm record and reaches the block, the stage component
  and the certificate, with the TensorFlow and Keras versions the run
  actually used, so a reader can check the claim against the environment
  rather than taking it on trust.

ORDERING. The arm's STARTED state is persisted BEFORE training begins,
so a process killed during the first epoch leaves a record saying so
rather than no trace at all - which would be indistinguishable from an
arm that was never attempted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from audit.implementation.durable_state import (
    CommittedCheckpointStore,
    DurableStateError,
    atomic_write_json,
    atomic_write_text,
    fsync_dir,
    fsync_path,
    sha256_file,
)

RECOVERY_VERSION = "arm-recovery-v2"

ARM_NOT_STARTED = "NOT_STARTED"
ARM_STARTED = "STARTED"
ARM_TRAINING = "TRAINING"
ARM_TRAINED = "TRAINED"
ARM_EVALUATED = "EVALUATED"
ARM_FAILED = "FAILED"

#: Machine-readable form of the exactness claim, hashed into the
#: preregistration so it is frozen with the plan.
RECOVERY_EXACTNESS = {
    "version": RECOVERY_VERSION,
    "granularity": "epoch",
    "authority": (
        "The commit manifest (durable_state.CommittedCheckpointStore) is the single source of "
        "truth for what has been committed. The epoch log is a derived record with no "
        "authority over resumability, so a lagging or torn log is reconciled and noted rather "
        "than fatal."),
    "verified_on": {
        "tensorflow": "2.20.0", "keras": "3.15.1",
        "note": ("The shuffle property is an empirical property of this Keras version, not a "
                 "documented guarantee. test_arm_recovery_keras.py re-measures it, so a Keras "
                 "version that behaves differently fails the suite instead of silently "
                 "invalidating the claim."),
    },
    "recovered_exactly": [
        "model weights (two-slot committed checkpoint, reloaded and compared element-wise)",
        "optimizer state including Adam's step counter and both moment estimates, compared "
        "variable by variable across the reload",
        "completed epoch index (fit resumes at initial_epoch; the log is a contiguous "
        "0..N-1 history with exactly one record per epoch)",
        "learning-rate schedule: the rate in force at every epoch of a resumed run equals the "
        "rate at the same epoch of an uninterrupted control run",
        "best-monitored-value record, restored into the checkpoint callback so a resume cannot "
        "overwrite a better checkpoint with a worse one",
        "per-epoch minibatch shuffle permutation: Keras derives each epoch's permutation from "
        "the absolute epoch index, so a resumed arm reseeded to its recorded seed reproduces "
        "the uninterrupted run's batch sequence - and its final weights - bit-for-bit",
    ],
    "conditions": [
        "the arm is reseeded to its recorded seed immediately before fit() on every session, "
        "including a resume (the adapter does this inside train_arm, so the guarantee does "
        "not depend on the runner's call order)",
        "the committed checkpoint is a FULL model save; a weights-only checkpoint would lose "
        "the optimizer moments and is refused",
        "a crash before the manifest commit recovers the previous committed epoch; a crash at "
        "or after it recovers the new one",
    ],
    "not_recovered": [
        "within-epoch position: recovery is at EPOCH granularity, so an interruption part-way "
        "through an epoch redoes that epoch from its start. No partial-epoch record is ever "
        "written, so a redone epoch cannot be mistaken for a completed one.",
        "the METRICS ROW of an epoch committed in the instant before the log append: the "
        "manifest is replaced before the row is written, so a kill in between leaves the "
        "epoch committed and hash-verified while its val_loss and val_accuracy - which "
        "existed only in memory - are gone. They are not reconstructed. A marker row with "
        "status COMMITTED_METRICS_LOST is appended on resume so the gap is explained in the "
        "log itself, and no consumer counts that row as a completed epoch. The confirmatory "
        "analysis reads the TERMINAL model's sealed-test accuracy, not the per-epoch log, so "
        "a lost row costs a line of the training trace and nothing in the result.",
    ],
    "fail_closed": (
        "Degradation is bounded and explicit. If the committed slot fails hash verification "
        "the previous slot is used and the loss of one epoch is recorded. If BOTH slots fail, "
        "or the manifest is unreadable, recovery REFUSES rather than restarting the arm from "
        "zero while reporting a resume."),
    "consequence": (
        "On the verified stack an interrupted-and-resumed arm is a bit-exact continuation, so "
        "resuming does not alter the arm's scientific status. This is nonetheless RECORDED "
        "rather than assumed: every interruption is appended to the arm record and surfaced on "
        "the block, the stage component and the certificate, together with the TensorFlow and "
        "Keras versions the run actually used, so a reader can check the claim against the "
        "environment instead of taking it on trust. If the exactness test fails on another "
        "stack, the arm is still correct - the design declares "
        "same_training_shuffle_stream=False and exact_replay_available=false - but it must "
        "then be reported as a non-bit-exact continuation."),
}


class RecoveryError(RuntimeError):
    """The arm's state cannot support a safe resume."""


# =====================================================================
# epoch log - a DERIVED record
# =====================================================================

EPOCH_FIELDS = (
    "epoch", "status", "loss", "accuracy", "val_loss", "val_accuracy",
    "learning_rate", "epoch_seconds", "checkpoint_seconds", "cumulative_seconds",
    "timestamp_utc", "session_id", "resumed_at_this_epoch",
)


@dataclass
class EpochLog:
    """
    Append-only JSON-Lines log, one record per COMMITTED epoch.

    JSON Lines rather than one document, because a document must be
    rewritten whole on every append and a kill mid-rewrite loses the
    history. A line is appended, flushed and fsynced after the epoch's
    checkpoint has been committed, so the worst a crash costs is the
    epoch still running.

    DERIVED, NOT AUTHORITATIVE. `reconcile` compares the log against the
    manifest's committed epoch and reports the difference; it never
    decides where to resume. The problems this class reports are
    informational - they belong in the evidence trail, and they must not
    prevent a resume that the manifest says is safe.
    """
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._malformed = 0
        self._raw_epochs: list[int] = []

    def append(self, record: dict[str, Any]) -> None:
        if "epoch" not in record:
            raise RecoveryError("an epoch record must carry its epoch index.")
        record = {**record, "status": record.get("status", "COMPLETED")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read(self) -> list[dict[str, Any]]:
        """
        All records, de-duplicated by epoch (last write wins), ordered.

        A trailing partial line - a process killed between write and
        newline - is skipped and counted. That is the likeliest artifact
        of a hard kill, and it is a NORMAL condition here: the epoch it
        would have described is simply redone, because the manifest never
        committed it.
        """
        self._malformed = 0
        self._raw_epochs = []
        if not self.path.is_file():
            return []
        by_epoch: dict[int, dict[str, Any]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                epoch = int(rec["epoch"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                self._malformed += 1
                continue
            self._raw_epochs.append(epoch)
            by_epoch[epoch] = rec
        return [by_epoch[e] for e in sorted(by_epoch)]

    def completed_epochs(self) -> list[int]:
        return [int(r["epoch"]) for r in self.read() if r.get("status") == "COMPLETED"]

    def highest_contiguous(self) -> int:
        """One past the highest contiguous completed epoch. Informational."""
        done = set(self.completed_epochs())
        n = 0
        while n in done:
            n += 1
        return n

    def best_value(self, key: str, *, mode: str = "min") -> Optional[float]:
        values = [r[key] for r in self.read()
                  if r.get("status") == "COMPLETED" and isinstance(r.get(key), (int, float))]
        if not values:
            return None
        return float(min(values) if mode == "min" else max(values))

    #: Status of a marker row. Deliberately NOT "COMPLETED", so that
    #: `completed_epochs()` and `best_value()` - which filter on
    #: COMPLETED - cannot mistake a marker for a metrics row.
    METRICS_LOST = "COMMITTED_METRICS_LOST"

    def note_committed_without_metrics(self, epoch: int, *, committed_epoch: int,
                                       session_id: Optional[str] = None) -> bool:
        """
        Record that an epoch was committed but its metrics row was lost.

        A kill between the manifest commit and the log append loses that
        epoch's metrics permanently: they existed only in memory. The
        epoch itself IS committed and recovery is correct, but the log
        would otherwise carry an unexplained hole - epoch 3 simply
        absent - and a reader of the metrics log alone could not tell an
        interrupted append from a skipped epoch or a tampered file.

        The metrics are NOT fabricated. A marker row is appended saying
        what happened, with a status that no consumer counts as a
        completed epoch. Idempotent: a second resume does not add a
        second marker.
        """
        for row in self.read():
            if row.get("epoch") == epoch:
                return False                     # already described, either way
        self.append({
            "epoch": epoch,
            "status": self.METRICS_LOST,
            "session_id": session_id,
            "note": (
                f"epoch {epoch} was COMMITTED - its checkpoint is in the manifest and "
                f"hash-verifies - but the process was killed between the manifest commit "
                f"and this log append, so the epoch's metrics were lost with the process. "
                f"They are not recoverable and are deliberately not reconstructed. The "
                f"manifest committed epoch {committed_epoch} at the time of this note."),
        })
        return True

    def reconcile(self, committed_epoch: Optional[int]) -> list[str]:
        """
        Differences between this derived log and the authoritative
        manifest. Informational: never fatal, never consulted for the
        resume point.
        """
        records = self.read()
        notes: list[str] = []
        if self._malformed:
            notes.append(
                f"{self._malformed} malformed line(s) in {self.path.name} were skipped "
                "(most likely a process killed mid-write). The epochs they would have "
                "described were not committed and are redone; this is expected after a hard "
                "kill and does not affect recovery.")
        if len(self._raw_epochs) != len(set(self._raw_epochs)):
            dupes = sorted({e for e in self._raw_epochs if self._raw_epochs.count(e) > 1})
            notes.append(
                f"duplicate epoch record(s) for {dupes}: the last record for each epoch is "
                "used. Reported rather than collapsed silently.")
        done = sorted(set(self.completed_epochs()))
        if done:
            missing = [e for e in range(done[-1] + 1) if e not in set(done)]
            if missing:
                notes.append(
                    f"the log is missing epoch(s) {missing[:8]}"
                    f"{' ...' if len(missing) > 8 else ''} below its highest record "
                    f"{done[-1]}; the manifest, not the log, determines what was committed.")
        if committed_epoch is not None and done:
            if done[-1] > committed_epoch:
                notes.append(
                    f"the log records epoch {done[-1]} but the manifest commits only "
                    f"{committed_epoch}; the uncommitted record(s) are ignored and those "
                    "epochs are redone.")
            elif done[-1] < committed_epoch:
                notes.append(
                    f"the log lags the manifest ({done[-1]} vs committed {committed_epoch}): "
                    "a crash between the manifest commit and the log append. The epoch IS "
                    "committed; its metrics row is missing from the log.")
        return notes

    def summary(self, *, expected_total: Optional[int] = None,
                committed_epoch: Optional[int] = None) -> dict[str, Any]:
        records = self.read()
        done = sorted(set(self.completed_epochs()))
        return {
            "path": str(self.path),
            "records": len(records),
            "completed_epochs": len(done),
            "highest_logged_epoch": done[-1] if done else None,
            "committed_epoch": committed_epoch,
            "contiguous": bool(done) and done == list(range(len(done))),
            "sessions": sorted({r.get("session_id") for r in records if r.get("session_id")}),
            "total_training_seconds": round(
                sum(float(r.get("epoch_seconds") or 0.0) for r in records), 3),
            "total_checkpoint_seconds": round(
                sum(float(r.get("checkpoint_seconds") or 0.0) for r in records), 3),
            "reconciliation_notes": self.reconcile(committed_epoch),
        }


# =====================================================================
# per-arm record - descriptive, never an authority
# =====================================================================

@dataclass
class ArmRecoveryRecord:
    """
    The arm's descriptive state: status, interruptions, provenance.

    NOT an authority on resumability - that is the commit manifest's
    job. This record exists so that a reader (and the certificate) can
    see what happened to the arm, and so that a process killed during
    the first epoch leaves a trace rather than silence.
    """
    arm_id: str
    block_id: str
    seed: int
    epochs_total: int
    status: str = ARM_NOT_STARTED
    epochs_completed: int = 0
    committed_epoch: Optional[int] = None
    committed_slot: Optional[str] = None
    committed_sha256: Optional[str] = None
    best_monitor: str = "val_loss"
    best_monitor_mode: str = "min"
    best_monitor_value: Optional[float] = None
    best_checkpoint: Optional[str] = None
    best_checkpoint_sha256: Optional[str] = None
    best_checkpoint_epoch: Optional[int] = None
    architecture_digest: Optional[str] = None
    interruptions: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    recovery_decisions: list[dict[str, Any]] = field(default_factory=list)
    corruption_repairs: list[dict[str, Any]] = field(default_factory=list)
    reconciliation_notes: list[str] = field(default_factory=list)
    started_at_utc: Optional[str] = None
    updated_at_utc: Optional[str] = None
    recovery_version: str = RECOVERY_VERSION

    @property
    def was_interrupted(self) -> bool:
        return bool(self.interruptions)

    @property
    def was_degraded(self) -> bool:
        return any(d.get("degraded") for d in self.recovery_decisions)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["was_interrupted"] = self.was_interrupted
        d["was_degraded"] = self.was_degraded
        return d

    @classmethod
    def load(cls, path: Path) -> Optional["ArmRecoveryRecord"]:
        path = Path(path)
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return None          # descriptive only: an empty record is not fatal
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Descriptive record, written atomically. A torn one loses
            # narrative, not recoverability: the manifest still governs.
            return None
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        self.updated_at_utc = _utc()
        atomic_write_json(Path(path), self.to_dict())

    def note_interruption(self, *, reason: str, detail: str, at_epoch: int,
                          session_id: Optional[str] = None) -> None:
        self.interruptions.append({
            "reason": reason, "detail": detail[:500],
            "epochs_completed_at_interruption": at_epoch,
            "session_id": session_id, "noted_at_utc": _utc(),
        })

    def note_decision(self, decision) -> None:
        self.recovery_decisions.append({**decision.to_dict(), "noted_at_utc": _utc()})


def _utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =====================================================================
# resume point
# =====================================================================

def resume_point(store: CommittedCheckpointStore, log: EpochLog, *,
                 record: Optional[ArmRecoveryRecord] = None):
    """
    Where to resume, from the MANIFEST, with the log reconciled to it.

    Returns (resume_from_epoch, decision, notes). Raises RecoveryError
    only on genuine corruption - both slots unusable, or an unreadable
    manifest - never because the derived log is untidy.
    """
    try:
        # prepare_for_resume, not recovery_decision: a fallback to the
        # previous slot must PROMOTE it in the manifest, or the next
        # commit is refused and the arm cannot continue.
        decision = store.prepare_for_resume()
    except DurableStateError as exc:
        raise RecoveryError(str(exc)) from exc
    notes = log.reconcile(decision.epoch)
    # Leave an explicit marker in the LOG for any epoch the manifest
    # committed but the log never recorded, so the metrics log explains
    # its own gaps instead of relying on a reader also opening
    # recovery.json. Only epochs at or below the committed epoch are
    # marked: anything above it was never committed and is simply redone.
    if decision.epoch is not None:
        recorded = set(log.completed_epochs())
        for epoch in range(decision.epoch + 1):
            if epoch not in recorded:
                if log.note_committed_without_metrics(
                        epoch, committed_epoch=decision.epoch,
                        session_id=None):
                    notes.append(
                        f"epoch {epoch} was committed but its metrics row was lost to an "
                        f"interruption; a marker row was appended to {log.path.name} so "
                        f"the gap is explained in the log itself.")
    notes.extend(decision.notes)
    if record is not None:
        record.note_decision(decision)
        record.corruption_repairs = store.corruption_history()
        record.reconciliation_notes = list(notes)
        record.committed_epoch = decision.epoch
        record.committed_slot = decision.slot
        record.committed_sha256 = decision.sha256
        record.epochs_completed = decision.resume_from_epoch
    return decision.resume_from_epoch, decision, notes


def reseed_for_session(seeder: Callable[[int], None], seed: int) -> None:
    """
    Reseed immediately before fit(), on EVERY session including a resume.

    This is what makes the shuffle permutation reproducible across a
    process boundary: Keras derives each epoch's permutation from the
    absolute epoch index and the global seed, so an arm resumed in a
    fresh process reproduces the uninterrupted batch sequence provided
    the global seed is the arm's recorded seed. Calling it here rather
    than relying on the runner having built a model first keeps the
    guarantee local to training.
    """
    seeder(seed)


# =====================================================================
# Keras callbacks (lazily imported)
# =====================================================================

class InjectedInterruption(RuntimeError):
    """Raised by the test fault injector. Never raised in production."""


def build_callbacks(*, store: CommittedCheckpointStore, log: EpochLog,
                    record: ArmRecoveryRecord, record_path: Path, best_path: Path,
                    session_id: str,
                    accuracy_key_candidates: tuple[str, ...] = ("acc", "accuracy"),
                    interrupt_after_epoch: Optional[int] = None) -> list:
    """
    The callbacks every arm needs, plus an optional fault injector.

    EpochCommitter   commits the epoch through the two-slot store - which
                     writes the alternate slot, fsyncs, hash-verifies and
                     only then replaces the manifest - and THEN appends
                     the metrics row to the derived log and updates the
                     descriptive record. The ordering is what makes any
                     crash land on a consistent epoch.
    ModelCheckpoint  the separate best-validation-loss artifact, with
                     `best` restored from the manifest so a resume cannot
                     demote it.
    _FaultInjector   raises after a chosen epoch. Present ONLY so the
                     interruption test can interrupt real training at a
                     known point; the production launcher never sets it.
    """
    import keras

    class EpochCommitter(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._epoch_start = None
            self._first_epoch_this_session = None

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start = time.time()
            if self._first_epoch_this_session is None:
                self._first_epoch_this_session = epoch

        def on_epoch_end(self, epoch, logs=None):
            logs = dict(logs or {})
            train_seconds = time.time() - (self._epoch_start or time.time())
            acc = next((logs[k] for k in accuracy_key_candidates if k in logs), None)
            val_acc = next((logs[f"val_{k}"] for k in accuracy_key_candidates
                            if f"val_{k}" in logs), None)
            try:
                lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            except Exception:                                     # noqa: BLE001
                try:
                    lr = float(self.model.optimizer.learning_rate)
                except Exception:                                 # noqa: BLE001
                    lr = None

            val_loss = _num(logs.get(record.best_monitor))
            # From the COMMITTED manifest, not the derived log. The log is
            # appended AFTER the commit, so at this point it is one epoch
            # behind by construction; reading the best value from it would
            # omit the most recently committed epoch and could restore a
            # stale best into the checkpoint callback after a crash
            # between the manifest commit and the log append.
            committed = store.manifest()
            best_so_far = (committed.best_monitor_value if committed is not None
                           else log.best_value(record.best_monitor,
                                               mode=record.best_monitor_mode))
            candidates = [v for v in (best_so_far, val_loss) if v is not None]
            new_best = (min(candidates) if record.best_monitor_mode == "min"
                        else max(candidates)) if candidates else None
            best_epoch = (epoch if (val_loss is not None and new_best == val_loss)
                          else record.best_checkpoint_epoch)

            # 1. COMMIT the epoch: alternate slot -> fsync -> hash-verify
            #    -> atomic manifest replace. The committed checkpoint is
            #    untouched until its replacement is durable and verified.
            ckpt_start = time.time()
            store.commit_epoch(
                epoch=epoch, save_fn=lambda p: self.model.save(p),
                best_monitor_value=new_best, best_checkpoint_epoch=best_epoch,
                session_id=session_id)
            checkpoint_seconds = time.time() - ckpt_start

            # 2. then the DERIVED metrics row.
            log.append({
                "epoch": int(epoch), "status": "COMPLETED",
                "loss": _num(logs.get("loss")), "accuracy": _num(acc),
                "val_loss": _num(logs.get("val_loss")), "val_accuracy": _num(val_acc),
                "learning_rate": _num(lr),
                "epoch_seconds": round(train_seconds, 4),
                "checkpoint_seconds": round(checkpoint_seconds, 4),
                "timestamp_utc": _utc(), "session_id": session_id,
                "resumed_at_this_epoch": (epoch == self._first_epoch_this_session
                                          and epoch != 0),
            })
            store._boundary("after_log_append")

            # 3. then the descriptive record.
            record.epochs_completed = epoch + 1
            record.status = ARM_TRAINING
            manifest = store.manifest()
            if manifest is not None:
                record.committed_epoch = manifest.committed_epoch
                record.committed_slot = manifest.committed_slot
                record.committed_sha256 = manifest.committed_sha256
                record.best_monitor_value = manifest.best_monitor_value
                record.best_checkpoint_epoch = manifest.best_checkpoint_epoch
            if Path(best_path).is_file():
                record.best_checkpoint = str(best_path)
                record.best_checkpoint_sha256 = sha256_file(best_path)
            record.save(record_path)
            store._boundary("after_record_update")

    class _FaultInjector(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == interrupt_after_epoch:
                raise InjectedInterruption(
                    f"injected interruption after committed epoch {epoch}")

    # The best-validation artifact is written through an INFLIGHT path and
    # promoted, for the same reason the committed checkpoint is: Keras
    # writes `ModelCheckpoint`'s target in place, so a kill part-way
    # through leaves a truncated file at a path still called "best". The
    # hash in the record is recomputed from disk at the next epoch end,
    # so the torn bytes would be hashed and reported as the artifact
    # rather than detected. Promotion keeps the previous good best in
    # place until its replacement is fully written and fsynced.
    #
    # The suffix must stay ".keras": the Keras saver refuses any other
    # extension, so ".keras.tmp" is not available as a temp name.
    inflight_best = Path(best_path).with_suffix("")
    inflight_best = inflight_best.with_name(inflight_best.name + ".inflight.keras")

    class _BestCheckpointPromoter(keras.callbacks.Callback):
        """
        Promotes the inflight best-checkpoint write, if one happened.

        Ordered AFTER the ModelCheckpoint and BEFORE the committer, so
        that by the time the record's `best_checkpoint_sha256` is taken
        the promotion is already durable.
        """

        def __init__(self) -> None:
            super().__init__()
            self._seen: Optional[tuple[int, int]] = None

        def on_train_begin(self, logs=None):
            # A leftover inflight file is the residue of a kill during a
            # write. It cannot be distinguished from a complete one
            # without loading it, so it is discarded rather than
            # promoted: the previous promoted best is still in place and
            # the manifest's best value never counted the lost write.
            if inflight_best.exists():
                record.reconciliation_notes = [
                    *record.reconciliation_notes,
                    f"discarded a leftover inflight best-checkpoint write "
                    f"({inflight_best.name}); the promoted best is unaffected"]
                inflight_best.unlink()

        def on_epoch_end(self, epoch, logs=None):
            if not inflight_best.is_file():
                return
            st = inflight_best.stat()
            stamp = (st.st_size, st.st_mtime_ns)
            if stamp == self._seen:
                return                    # no new write: nothing to promote
            self._seen = stamp
            with inflight_best.open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(inflight_best, best_path)
            fsync_dir(Path(best_path).parent)

    best_cb = keras.callbacks.ModelCheckpoint(
        str(inflight_best), monitor=record.best_monitor, mode=record.best_monitor_mode,
        save_best_only=True, save_weights_only=False)
    # Restored from the COMMITTED manifest. The epoch log can lag it by
    # one epoch after a crash, and a stale best would let the first
    # post-resume epoch overwrite a better checkpoint - silently, since
    # the file is still called "best".
    manifest = store.manifest()
    restored = manifest.best_monitor_value if manifest is not None else None
    if restored is not None:
        best_cb.best = restored

    callbacks = [best_cb, _BestCheckpointPromoter(), EpochCommitter()]
    if interrupt_after_epoch is not None:
        callbacks.append(_FaultInjector())
    return callbacks


def _num(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
