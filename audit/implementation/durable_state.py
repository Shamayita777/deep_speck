#!/usr/bin/env python3
"""
Durable, transactional per-arm state - GENERIC.

No Gohr, Speck, Keras or TensorFlow knowledge. Saving and loading a
checkpoint are injected as callables, so the whole commit protocol is
testable in milliseconds with JSON "models" and used in production with
full Keras model saves.

WHY THIS MODULE REPLACES THE EARLIER SINGLE-FILE SCHEME
-------------------------------------------------------
The previous design wrote one `resume_model.keras`, then appended to the
epoch log, then rewrote the arm record. Each step was individually
atomic, and the whole was not. Two windows were demonstrated to brick an
arm permanently:

  * a crash after the checkpoint was replaced but before the record was
    rewritten left the record holding the PREVIOUS epoch's hash against
    the NEW epoch's file. The hash guard that exists to protect the
    resume then refused it, and 200 epochs died in a ~50 ms window.
  * a trailing partial JSONL line - the single likeliest artifact of a
    hard kill - made `integrity_problems` non-empty, which made
    `require_resumable` refuse, which made the runner treat an arm with
    50 good epochs as having none.

Both failures share one cause: THREE INDEPENDENT AUTHORITIES that could
disagree permanently. This module replaces them with one.

THE COMMIT PROTOCOL
-------------------
Two checkpoint slots, `a` and `b`, and one manifest that names which
slot holds which epoch. For epoch k, with the committed state currently
in slot S:

  1. write the new checkpoint into the OTHER slot's inflight path;
  2. flush and fsync it, then fsync its directory;
  3. os.replace it into the other slot T - S is still untouched;
  4. re-read and hash-verify slot T;
  5. ONLY THEN write the manifest naming (k, T, hash) as committed and
     (k-1, S, hash) as previous, atomically via os.replace + fsync;
  6. append to the epoch log, which is a DERIVED record.

The invariants that follow, each asserted by a process-kill test:

  * the committed checkpoint is never overwritten until its replacement
    is durable and hash-verified in the other slot (steps 1-4 touch only
    T; S is overwritten only on the NEXT epoch, by which time T is
    committed);
  * a crash at any point before step 5 recovers epoch k-1, because the
    manifest still names it;
  * a crash at or after step 5 recovers epoch k;
  * a crash during step 5 recovers one or the other, never a blend,
    because os.replace is atomic;
  * a crash during step 6 changes nothing that matters: the manifest is
    the authority and the log is reconciled to it.

THE MANIFEST IS THE SOURCE OF TRUTH. The epoch log is a durable,
append-only RECORD for the evidence trail - it carries the per-epoch
metrics a reader needs - but it has no authority over what is resumable.
A log that lags the manifest, or ends in a torn line, is reconciled and
noted, never fatal.

FAIL-CLOSED IS PRESERVED. Degradation is bounded and explicit: if the
committed slot fails verification the previous slot is used and the loss
of one epoch is recorded. If BOTH slots fail, or the manifest itself is
unreadable, this module REFUSES. That is genuine corruption, and
silently restarting an arm from zero while reporting a resume would be
exactly the dishonesty the audit exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

DURABLE_STATE_VERSION = "durable-state-v1"

SLOTS = ("a", "b")

#: Named persistence boundaries. Tests kill a real process at each one;
#: production code passes through them.
BOUNDARIES = (
    "before_inflight_write",
    "after_inflight_write",
    "after_slot_replace_before_dir_fsync",
    "after_slot_replace",
    "after_hash_verify",
    "before_manifest_commit",
    "after_manifest_commit",
    "after_log_append",
    "after_record_update",
)


class DurableStateError(RuntimeError):
    """Unrecoverable: the arm's durable state cannot be trusted."""


#: What an operator must do when an arm cannot be recovered, written
#: into every unrecoverable refusal so the instruction travels with the
#: error rather than living in a document nobody has open at 3am.
#:
#: An earlier version of this module ended its both-slots-corrupt
#: refusal with "Delete this arm's directory to restart it
#: deliberately." In an audit pipeline that instruction destroys the
#: only copies of the commit manifest, both checkpoint slots, the epoch
#: log, the corruption log and the recovery record - which is to say,
#: the evidence of what went wrong. Nothing in this module deletes,
#: truncates, recreates or resets an arm directory, and the refusal now
#: says how to proceed without doing so.
OPERATOR_PROCEDURE = (
    "NOTHING HAS BEEN DELETED OR MODIFIED. Every artifact in this arm's directory is "
    "preserved exactly as it was found, including both checkpoint slots, the commit "
    "manifest, the epoch log, the corruption log and the evaluation journal.\n"
    "  To proceed:\n"
    "    1. Read the diagnostic report this refusal wrote into the arm directory "
    "(unrecoverable-*.json). It records, for each slot, the expected and actual hash, "
    "and what the manifest and the derived log each claim.\n"
    "    2. Copy the whole arm directory somewhere safe before touching anything. It is "
    "the only record of the corruption and belongs with the run's evidence.\n"
    "    3. Decide deliberately, as the operator, between: (a) treating this block as "
    "FAILED and letting the preregistered replacement-block mechanism run, which is the "
    "sanctioned path and costs one of the pre-declared replacements; or (b) aborting the "
    "run to investigate the filesystem or hardware first.\n"
    "    4. Prefer (b) if more than one arm has been affected. Repeated corruption is a "
    "property of the machine, not of this run, and replacing blocks would hide it.\n"
    "  This code will not choose for you and will not restart the arm from epoch 0: the "
    "manifest records committed work, and silently rerunning it would report a fresh run "
    "as a resume."
)


class CheckpointVerificationError(DurableStateError):
    """A slot's content does not match the hash the manifest records."""


class UnrecoverableArmError(CheckpointVerificationError):
    """
    This arm cannot be resumed and cannot be safely restarted in place.

    Carries the forensic snapshot taken at the moment of refusal, so the
    caller can record WHY recovery was refused without re-reading (and
    possibly re-corrupting) the arm directory. Raising this is the end
    of the automated path: it requires an operator decision.
    """

    def __init__(self, message: str, *, reason_code: str,
                 diagnosis: Optional[dict] = None,
                 report_path: Optional[Path] = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.diagnosis = diagnosis or {}
        self.report_path = report_path


# =====================================================================
# primitives
# =====================================================================

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fsync_path(path) -> None:
    """
    Force a file's bytes to stable storage.

    Writers that go through a library - `model.save()` among them -
    close their handle without fsyncing, so the data may sit in the page
    cache. A power loss or container kill then loses a file that the
    process believed it had written, which is precisely the case this
    whole module exists to survive.
    """
    path = Path(path)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path) -> None:
    """Force a directory entry to stable storage, so a rename survives."""
    fd = os.open(str(Path(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path, text: str) -> None:
    """Durable, atomic replace: temp file in the same directory, fsync, rename, fsync dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def _other(slot: str) -> str:
    if slot not in SLOTS:
        raise DurableStateError(f"unknown slot {slot!r}; expected one of {SLOTS}.")
    return "b" if slot == "a" else "a"


#: What a SIGKILL test does and does not establish. Stated here because
#: "tested against process kill" is routinely over-read.
WHAT_SIGKILL_ESTABLISHES = {
    "establishes": [
        "that the ORDERING of operations is correct: whatever a later process finds on "
        "disk is a state the protocol can interpret, at every named boundary",
        "that no `except`, `finally` or `atexit` path is relied upon, because SIGKILL runs "
        "none of them",
        "that a partially written file, an orphaned inflight file, a lagging log or a torn "
        "trailing line is recoverable rather than fatal",
        "that the recovered epoch's weights and optimizer state correspond to that epoch, "
        "checked by reading the committed slot's contents",
    ],
    "does_not_establish": [
        "DURABILITY against power loss or kernel panic. SIGKILL destroys the process but "
        "leaves the page cache intact, so data written-but-not-fsynced still reaches disk. "
        "Only a crash of the whole machine, or a fault-injecting filesystem, distinguishes "
        "fsynced from merely written. The fsync calls are therefore reasoned, not measured.",
        "atomicity of os.replace on a non-POSIX or network filesystem. On Kaggle's or "
        "Colab's overlay/network-backed working directories, rename may not be atomic and "
        "fsync may not reach stable storage; the whole protocol rests on both.",
        "a kill INSIDE a third-party write. A SIGKILL during Keras's multi-file zip write, "
        "or inside the real firewall's own save, cannot be scheduled deterministically; "
        "those are covered by writing through our own atomic wrapper instead.",
        "anything about concurrent writers. One process per run directory is assumed and "
        "not enforced by a lock.",
    ],
}


def save_firewall_atomically(firewall, path) -> None:
    """
    Persist a `TestSetFirewall` atomically and durably.

    WHY NOT `firewall.save()`. The real framework's save is
    `path.write_text(json.dumps(...))`: it truncates the existing file
    first and never fsyncs. A process killed mid-write therefore leaves a
    TRUNCATED firewall - the record of which confirmatory permits have
    been spent - and the next `TestSetFirewall.load` raises on the torn
    JSON. The firewall is the one artifact whose loss cannot be repaired
    by recomputation, because a spent permit is a fact about the past.

    `framework/firewall.py` is shared with the Experimental Validity
    pillar, is bound by hash into certificate V6, and is NOT modified
    here. Its `to_dict()` is the authoritative serialisation; this
    wrapper only changes HOW those bytes reach the disk - temp file in
    the same directory, fsync, atomic rename, directory fsync.
    """
    atomic_write_json(path, firewall.to_dict())


# =====================================================================
# commit manifest
# =====================================================================

@dataclass
class CommitManifest:
    """
    The single authority on what has been durably committed for one arm.

    `committed_epoch` is the index of the LAST COMPLETED epoch, so a
    fresh arm has no manifest at all and training starts at epoch 0.
    """
    arm_id: str
    block_id: str
    seed: int
    epochs_total: int
    committed_epoch: int
    committed_slot: str
    committed_sha256: str
    previous_epoch: Optional[int] = None
    previous_slot: Optional[str] = None
    previous_sha256: Optional[str] = None
    architecture_digest: Optional[str] = None
    best_monitor: str = "val_loss"
    best_monitor_mode: str = "min"
    best_monitor_value: Optional[float] = None
    best_checkpoint_epoch: Optional[int] = None
    committed_at_utc: Optional[str] = None
    session_id: Optional[str] = None
    #: Append-only record of every corruption repair applied to this arm.
    #: Each entry names the epoch that was LOST, the slot that failed and
    #: why, and the epoch promoted in its place, so a reader can see that
    #: the arm's history is not continuous and exactly where.
    repairs: list = field(default_factory=list)
    manifest_version: str = DURABLE_STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path) -> Optional["CommitManifest"]:
        path = Path(path)
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            raise DurableStateError(
                f"{path} is empty. The manifest is written by atomic replace, so an empty "
                "manifest indicates filesystem corruption rather than an interrupted write. "
                "Refusing to guess which epoch was committed.")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DurableStateError(
                f"{path} is not valid JSON ({exc}). The manifest is written by atomic "
                "replace, so a torn manifest indicates filesystem corruption. Refusing: "
                "restarting the arm silently would misreport a fresh run as a resume.") from exc
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for required in ("arm_id", "block_id", "committed_epoch", "committed_slot",
                         "committed_sha256"):
            if required not in known:
                raise DurableStateError(
                    f"{path} is missing required field {required!r}; it is not a commit "
                    "manifest this version can interpret.")
        return cls(**{**{"seed": -1, "epochs_total": -1}, **known})

    def save(self, path) -> None:
        atomic_write_json(path, self.to_dict())

    def validate(self, *, arm_id: str, block_id: str, seed: int, epochs_total: int,
                 architecture_digest: Optional[str] = None) -> list[str]:
        """
        Schema, identity and epoch-continuity checks.

        A manifest is the sole authority on what was committed, so it
        must be shown to be THIS arm's manifest before it is believed.
        Without these checks a manifest copied from another arm, another
        block or another seed would be read as this one's history: the
        store would load weights trained on different data and report a
        continuous run. The epoch-continuity check catches a manifest
        that claims a committed epoch with a non-adjacent previous
        epoch, which the two-slot protocol can never produce and which
        therefore indicates hand-editing or corruption.
        """
        problems: list[str] = []
        if self.manifest_version != DURABLE_STATE_VERSION:
            problems.append(
                f"manifest_version {self.manifest_version!r} is not {DURABLE_STATE_VERSION!r}; "
                "this manifest was written by a different version of the commit protocol and "
                "its fields cannot be assumed to mean the same thing")
        for field_name, expected, actual in (("arm_id", arm_id, self.arm_id),
                                             ("block_id", block_id, self.block_id)):
            if actual != expected:
                problems.append(
                    f"{field_name} {actual!r} does not match this arm ({expected!r}); this "
                    "manifest belongs to a different arm or block")
        if self.seed != seed:
            problems.append(
                f"seed {self.seed} does not match this arm's predetermined seed {seed}; the "
                "checkpoint was trained from a different initialization")
        if self.epochs_total != epochs_total:
            problems.append(
                f"epochs_total {self.epochs_total} does not match the design's "
                f"{epochs_total}; this manifest was written under a different schedule")
        if architecture_digest and self.architecture_digest != architecture_digest:
            problems.append(
                f"architecture_digest {self.architecture_digest} does not match the declared "
                f"{architecture_digest}")
        if self.committed_slot not in SLOTS:
            problems.append(f"committed_slot {self.committed_slot!r} is not one of {SLOTS}")
        if not (isinstance(self.committed_sha256, str) and len(self.committed_sha256) == 64):
            problems.append("committed_sha256 is not a sha256 digest")
        if not isinstance(self.committed_epoch, int) or self.committed_epoch < 0:
            problems.append(f"committed_epoch {self.committed_epoch!r} is not a valid epoch")
        elif self.committed_epoch >= epochs_total:
            problems.append(
                f"committed_epoch {self.committed_epoch} is outside the schedule "
                f"0..{epochs_total - 1}")
        # epoch continuity
        if self.previous_epoch is None:
            if self.previous_slot is not None or self.previous_sha256 is not None:
                problems.append(
                    "previous_epoch is absent but previous_slot/previous_sha256 are set")
        else:
            if self.previous_epoch != self.committed_epoch - 1:
                problems.append(
                    f"epoch continuity broken: previous_epoch {self.previous_epoch} is not "
                    f"one below committed_epoch {self.committed_epoch}. The two-slot protocol "
                    "can never produce a gap, so this manifest was edited or corrupted")
            if self.previous_slot == self.committed_slot:
                problems.append(
                    f"previous_slot and committed_slot are both {self.committed_slot!r}; the "
                    "two slots must differ")
            if self.previous_slot not in SLOTS:
                problems.append(f"previous_slot {self.previous_slot!r} is not one of {SLOTS}")
            if self.committed_epoch == 0:
                problems.append(
                    "committed_epoch is 0 but a previous epoch is recorded; nothing can "
                    "precede the first epoch")
        return problems


# =====================================================================
# two-slot checkpoint store
# =====================================================================

@dataclass
class RecoveryDecision:
    """What the store decided, and on what basis."""
    resume_from_epoch: int
    #: "fresh"             nothing committed yet
    #: "committed"         the manifest's committed slot verified
    #: "previous"          read-only query: committed bad, previous good
    #: "previous_promoted" the previous slot was promoted so the arm can
    #:                     continue; one epoch was lost (degraded=True)
    source: str
    slot: Optional[str] = None
    epoch: Optional[int] = None
    sha256: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommittedCheckpointStore:
    """
    Two rotating slots plus the commit manifest.

    Disk cost is bounded and small: two full-model checkpoints per arm,
    regardless of how many epochs run. At roughly 1.6 MB per checkpoint
    that is 2 slots x 2 arms x 9 blocks = about 57.6 MB for the whole
    experiment, before the separate best-validation checkpoints, terminal
    models and temporary files.
    """
    arm_dir: Path
    arm_id: str
    block_id: str
    seed: int
    epochs_total: int
    suffix: str = ".keras"
    architecture_digest: Optional[str] = None
    #: Test-only hook: called with each boundary name in BOUNDARIES.
    boundary_hook: Optional[Callable[[str], None]] = None

    def __post_init__(self) -> None:
        self.arm_dir = Path(self.arm_dir)
        self.arm_dir.mkdir(parents=True, exist_ok=True)

    # --- paths --------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.arm_dir / "commit.json"

    def slot_path(self, slot: str) -> Path:
        return self.arm_dir / f"ckpt_{slot}{self.suffix}"

    def inflight_path(self, slot: str) -> Path:
        # The inflight name keeps the real suffix LAST: Keras validates the
        # extension and refuses to save to "ckpt_a.keras.tmp", which broke
        # an earlier version of this code outright.
        return self.arm_dir / f"ckpt_{slot}.inflight{self.suffix}"

    def _boundary(self, name: str) -> None:
        if self.boundary_hook is not None:
            self.boundary_hook(name)

    # --- read ---------------------------------------------------------
    def manifest(self) -> Optional[CommitManifest]:
        return CommitManifest.load(self.manifest_path)

    def _verify(self, slot: Optional[str], digest: Optional[str]) -> tuple[bool, str]:
        if slot is None or digest is None:
            return False, "not recorded"
        path = self.slot_path(slot)
        if not path.is_file():
            return False, f"slot {slot} file {path.name} is absent"
        actual = sha256_file(path)
        if actual != digest:
            return False, (f"slot {slot} hash {actual[:16]}... does not match the manifest's "
                           f"{digest[:16]}...")
        return True, "verified"

    # --- forensics ----------------------------------------------------
    def diagnose(self) -> dict[str, Any]:
        """
        A READ-ONLY forensic snapshot of this arm directory.

        Opens nothing for writing, creates nothing, and tolerates every
        kind of damage it is meant to describe: a torn manifest, an
        absent slot, a truncated log. It is safe to call on a directory
        whose contents are evidence, and it is the first thing an
        operator should look at after a refusal.

        The manifest is read as RAW BYTES and hashed, so the snapshot
        records what is actually on disk even when it cannot be parsed
        as a manifest at all.
        """
        snapshot: dict[str, Any] = {
            "diagnosed_at_utc": _utc(),
            "arm_dir": str(self.arm_dir),
            "arm_id": self.arm_id,
            "block_id": self.block_id,
            "seed": self.seed,
            "epochs_total": self.epochs_total,
            "expected_architecture_digest": self.architecture_digest,
            "read_only": True,
        }

        # --- the manifest, as bytes first and as a manifest second ----
        mf: dict[str, Any] = {"path": str(self.manifest_path),
                              "exists": self.manifest_path.is_file()}
        if mf["exists"]:
            raw = self.manifest_path.read_bytes()
            mf.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                      preview=raw[:400].decode("utf-8", errors="replace"))
            try:
                parsed = CommitManifest.load(self.manifest_path)
            except DurableStateError as exc:
                mf.update(parsed=False, parse_error=str(exc))
                parsed = None
            else:
                mf["parsed"] = True
            if parsed is not None:
                mf["manifest"] = parsed.to_dict() if hasattr(parsed, "to_dict") else None
                mf["validation_problems"] = parsed.validate(
                    arm_id=self.arm_id, block_id=self.block_id, seed=self.seed,
                    epochs_total=self.epochs_total,
                    architecture_digest=self.architecture_digest)
                snapshot["committed_epoch_claimed"] = parsed.committed_epoch
        else:
            parsed = None
        snapshot["manifest"] = mf

        # --- both slots, expected against actual ----------------------
        slots = {}
        for slot in ("a", "b"):
            path = self.slot_path(slot)
            entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
            if entry["exists"]:
                entry["bytes"] = path.stat().st_size
                entry["actual_sha256"] = sha256_file(path)
            expected = None
            role = None
            if parsed is not None:
                if parsed.committed_slot == slot:
                    expected, role = parsed.committed_sha256, "committed"
                elif parsed.previous_slot == slot:
                    expected, role = parsed.previous_sha256, "previous"
            entry["role_per_manifest"] = role
            entry["expected_sha256"] = expected
            if expected is not None and entry["exists"]:
                entry["matches"] = (entry["actual_sha256"] == expected)
            elif expected is not None:
                entry["matches"] = False
            slots[slot] = entry
        snapshot["slots"] = slots
        snapshot["inflight_files"] = sorted(
            p.name for p in self.arm_dir.glob(f"*.inflight{self.suffix}")) \
            if self.arm_dir.is_dir() else []

        # --- the derived log, and whether it agrees with the manifest -
        log_path = self.arm_dir / "epochs.jsonl"
        log: dict[str, Any] = {"path": str(log_path), "exists": log_path.is_file()}
        if log["exists"]:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            epochs, malformed = [], 0
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if "epoch" in row:
                    epochs.append(row["epoch"])
            log.update(lines=len(lines), malformed_lines=malformed,
                       epochs_recorded=epochs,
                       duplicate_epochs=sorted({e for e in epochs
                                                if epochs.count(e) > 1}),
                       highest_recorded=(max(epochs) if epochs else None))
            if parsed is not None and epochs:
                log["agrees_with_manifest"] = (max(epochs) == parsed.committed_epoch)
                log["lags_manifest_by"] = parsed.committed_epoch - max(epochs)
        snapshot["epoch_log"] = log

        # --- the rest of the evidence trail ---------------------------
        try:
            snapshot["corruption_history"] = self.corruption_history()
        except Exception as exc:                                  # noqa: BLE001
            snapshot["corruption_history_error"] = str(exc)
        snapshot["quarantined_files"] = sorted(
            p.name for p in self.quarantine_dir.glob("*")) \
            if self.quarantine_dir.is_dir() else []
        journal_path = self.arm_dir / "eval_journal.json"
        if journal_path.is_file():
            try:
                snapshot["eval_journal"] = json.loads(
                    journal_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                snapshot["eval_journal_error"] = f"unparseable: {exc}"
        snapshot["directory_listing"] = sorted(
            p.name for p in self.arm_dir.iterdir()) if self.arm_dir.is_dir() else []

        # --- the verdict, derived from the snapshot alone -------------
        verdict = "RECOVERABLE"
        if not mf["exists"]:
            verdict = "FRESH"
        elif not mf.get("parsed"):
            verdict = "UNRECOVERABLE: the manifest cannot be parsed"
        elif mf.get("validation_problems"):
            verdict = "UNRECOVERABLE: the manifest is not valid for this arm"
        else:
            usable = [s for s, e in slots.items()
                      if e.get("role_per_manifest") and e.get("matches")]
            if not usable:
                verdict = "UNRECOVERABLE: no checkpoint slot matches the manifest"
            elif slots.get(parsed.committed_slot, {}).get("matches"):
                verdict = "RECOVERABLE: the committed slot verifies"
            else:
                verdict = "RECOVERABLE (DEGRADED): only the previous slot verifies"
        snapshot["verdict"] = verdict
        return snapshot

    def write_diagnostic_report(self, diagnosis: dict[str, Any]) -> Optional[Path]:
        """
        Write the snapshot to a NEW file in the arm directory.

        Adding a file is the only write this module performs on a
        damaged arm. The name carries a timestamp and a counter and is
        never reused, so a second refusal cannot overwrite the record of
        the first - and no existing artifact is touched. A failure to
        write the report is swallowed: the refusal itself must not be
        replaced by a different error, and the report is a convenience
        on top of the exception, not the only copy of it.
        """
        try:
            self.arm_dir.mkdir(parents=True, exist_ok=True)
            stem = f"unrecoverable-{_utc().replace(':', '').replace('-', '')}"
            path = self.arm_dir / f"{stem}.json"
            n = 1
            while path.exists():
                path = self.arm_dir / f"{stem}.{n}.json"
                n += 1
            atomic_write_json(path, {
                "artifact": "unrecoverable-arm-diagnostic-v1",
                "operator_procedure": OPERATOR_PROCEDURE,
                "nothing_was_deleted_or_modified": True,
                "diagnosis": diagnosis,
            })
            return path
        except Exception:                                         # noqa: BLE001
            return None

    def _refuse_unrecoverable(self, reason_code: str, message: str) -> "UnrecoverableArmError":
        """Build the refusal: snapshot first, then report, then raise."""
        try:
            diagnosis = self.diagnose()
        except Exception as exc:                                  # noqa: BLE001
            diagnosis = {"diagnose_failed": str(exc), "arm_dir": str(self.arm_dir)}
        report = self.write_diagnostic_report(diagnosis)
        return UnrecoverableArmError(
            f"{message}\n  reason_code: {reason_code}\n"
            + (f"  diagnostic report: {report}\n" if report else
               "  (the diagnostic report could not be written; the diagnosis is attached "
               "to this exception)\n")
            + OPERATOR_PROCEDURE,
            reason_code=reason_code, diagnosis=diagnosis, report_path=report)

    def recovery_decision(self) -> RecoveryDecision:
        """
        Where to resume, decided from the MANIFEST and verified against
        the slot contents. Never consults the epoch log.
        """
        # A manifest that cannot be PARSED is refused by `CommitManifest.load`,
        # which is a classmethod and has no view of the arm directory. The
        # refusal is re-raised here so that a torn, empty or truncated
        # manifest produces the same forensic snapshot and preserved-evidence
        # guarantee as every other unrecoverable state, instead of a bare
        # parse error the operator has to interpret alone.
        try:
            manifest = self.manifest()
        except UnrecoverableArmError:
            raise
        except DurableStateError as exc:
            raise self._refuse_unrecoverable("MANIFEST_UNREADABLE", str(exc)) from exc
        if manifest is None:
            return RecoveryDecision(resume_from_epoch=0, source="fresh",
                                    notes=["no commit manifest: this arm has not started"])
        problems = manifest.validate(
            arm_id=self.arm_id, block_id=self.block_id, seed=self.seed,
            epochs_total=self.epochs_total,
            architecture_digest=self.architecture_digest)
        if problems:
            raise self._refuse_unrecoverable(
                "MANIFEST_INVALID_FOR_THIS_ARM",
                f"{self.block_id}/{self.arm_id}: the commit manifest at "
                f"{self.manifest_path} is not a valid manifest for this arm:\n  "
                + "\n  ".join(problems)
                + "\nRefusing: reading it would report another arm's or another schedule's "
                  "history as this one's.")
        if manifest.committed_epoch >= self.epochs_total:
            return RecoveryDecision(
                resume_from_epoch=self.epochs_total, source="committed",
                slot=manifest.committed_slot, epoch=manifest.committed_epoch,
                sha256=manifest.committed_sha256,
                notes=["the committed epoch already completes the schedule"])

        ok, why = self._verify(manifest.committed_slot, manifest.committed_sha256)
        if ok:
            return RecoveryDecision(
                resume_from_epoch=manifest.committed_epoch + 1, source="committed",
                slot=manifest.committed_slot, epoch=manifest.committed_epoch,
                sha256=manifest.committed_sha256, notes=[f"committed slot {why}"])

        prev_ok, prev_why = self._verify(manifest.previous_slot, manifest.previous_sha256)
        if prev_ok:
            return RecoveryDecision(
                resume_from_epoch=(manifest.previous_epoch or 0) + 1, source="previous",
                slot=manifest.previous_slot, epoch=manifest.previous_epoch,
                sha256=manifest.previous_sha256, degraded=True,
                notes=[f"committed slot unusable ({why}); fell back to the previous slot, "
                       f"losing epoch {manifest.committed_epoch}", f"previous slot {prev_why}"])

        raise self._refuse_unrecoverable(
            "BOTH_SLOTS_UNUSABLE",
            f"{self.block_id}/{self.arm_id}: BOTH checkpoint slots are unusable "
            f"(committed slot {manifest.committed_slot}: {why}; previous slot "
            f"{manifest.previous_slot}: {prev_why}). This is corruption, not an "
            "interrupted write - the commit protocol never leaves both slots invalid, "
            "because the committed slot is replaced only after its successor has been "
            "written, fsynced and hash-verified. Refusing to continue: restarting the arm "
            f"from epoch 0 while the manifest claims epoch {manifest.committed_epoch} "
            "would misreport a fresh run as a resume.")

    @property
    def corruption_log_path(self) -> Path:
        return self.arm_dir / "corruption.jsonl"

    @property
    def quarantine_dir(self) -> Path:
        return self.arm_dir / "corrupt"

    def _record_corruption(self, entry: dict[str, Any]) -> None:
        """
        Append-only, fsynced corruption record.

        Written IN ADDITION to the manifest's `repairs` list, and
        deliberately in a separate append-only file: the manifest is
        replaced on every subsequent epoch, so a repair recorded only
        there would survive only as long as nobody chose to rewrite it
        differently. The evidence trail for a lost epoch must outlive the
        state that lost it.
        """
        self.corruption_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.corruption_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def corruption_history(self) -> list[dict[str, Any]]:
        """Every repair ever applied to this arm, oldest first."""
        if not self.corruption_log_path.is_file():
            return []
        out = []
        for line in self.corruption_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"unparseable_record": line[:200]})
        return out

    def prepare_for_resume(self) -> RecoveryDecision:
        """
        Decide where to resume AND make that state committable.

        `recovery_decision` is read-only, so after a fallback to the
        previous slot the manifest still named the CORRUPT slot as
        committed. Two consequences, both demonstrated: the next
        `commit_epoch` was refused outright ("the manifest must never go
        backwards"), and the slot it would have targeted was the only
        surviving good one. A degraded recovery was therefore a dead end -
        the arm could be read but never continued.

        This method repairs that atomically. The surviving previous slot
        is PROMOTED to committed, the corrupt file is quarantined so the
        slot it occupied is free for the next epoch, and the loss is
        recorded in two places. Only then is the decision returned.

        The promoted state has NO previous slot until the next epoch
        commits, so the arm runs with reduced redundancy for exactly one
        epoch. That is stated in the decision's notes rather than left
        for a reader to infer.
        """
        decision = self.recovery_decision()
        if not decision.degraded or decision.source != "previous":
            return decision

        manifest = self.manifest()
        assert manifest is not None                      # degraded implies a manifest
        corrupt_slot = manifest.committed_slot
        corrupt_epoch = manifest.committed_epoch
        _, reason = self._verify(corrupt_slot, manifest.committed_sha256)

        entry = {
            "artifact": "arm-corruption-repair-v1",
            "detected_at_utc": _utc(),
            "arm_id": self.arm_id, "block_id": self.block_id,
            "lost_epoch": corrupt_epoch,
            "failed_slot": corrupt_slot,
            "failed_slot_expected_sha256": manifest.committed_sha256,
            "failed_slot_actual_sha256": (
                sha256_file(self.slot_path(corrupt_slot))
                if self.slot_path(corrupt_slot).is_file() else None),
            "failure_reason": reason,
            "promoted_epoch": decision.epoch,
            "promoted_slot": decision.slot,
            "promoted_sha256": decision.sha256,
            "epochs_lost": 1,
            "consequence": (
                f"epoch {corrupt_epoch} is lost and will be retrained from the state "
                f"committed at epoch {decision.epoch}. The arm's training history is NOT "
                "continuous across this point; this record is the evidence of that."),
        }
        # Quarantine FIRST, then record, so the persisted record names the
        # quarantine path. The order is also crash-safe: if the process
        # dies between the two, the slot file is simply absent, which
        # still fails verification, so the next prepare_for_resume redoes
        # the repair and records it then.
        if self.slot_path(corrupt_slot).is_file():
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            # The name must be unique even for two repairs in the same
            # second, or the second would overwrite the first's evidence.
            stem = (f"ckpt_{corrupt_slot}_epoch{corrupt_epoch}_"
                    f"{_utc().replace(':', '')}_"
                    f"{(entry['failed_slot_actual_sha256'] or 'nohash')[:12]}")
            quarantined = self.quarantine_dir / f"{stem}{self.suffix}"
            n = 1
            while quarantined.exists():
                quarantined = self.quarantine_dir / f"{stem}.{n}{self.suffix}"
                n += 1
            os.replace(self.slot_path(corrupt_slot), quarantined)
            fsync_dir(self.quarantine_dir)
            fsync_dir(self.arm_dir)
            entry["quarantined_to"] = str(quarantined)
        self._record_corruption(entry)

        promoted = CommitManifest(
            arm_id=manifest.arm_id, block_id=manifest.block_id, seed=manifest.seed,
            epochs_total=manifest.epochs_total,
            committed_epoch=decision.epoch, committed_slot=decision.slot,
            committed_sha256=decision.sha256,
            previous_epoch=None, previous_slot=None, previous_sha256=None,
            architecture_digest=manifest.architecture_digest,
            best_monitor=manifest.best_monitor,
            best_monitor_mode=manifest.best_monitor_mode,
            best_monitor_value=manifest.best_monitor_value,
            best_checkpoint_epoch=manifest.best_checkpoint_epoch,
            committed_at_utc=_utc(), session_id=manifest.session_id,
            repairs=[*(manifest.repairs or []), entry],
        )
        promoted.save(self.manifest_path)

        repaired = self.recovery_decision()
        repaired.degraded = True
        # NOT "committed": the promoted slot is now what the manifest
        # commits, but the caller must be able to tell that this arm
        # reached that state by falling back, not by running cleanly.
        repaired.source = "previous_promoted"
        repaired.notes = [
            *decision.notes,
            f"REPAIRED: slot {corrupt_slot} (epoch {corrupt_epoch}) failed verification and "
            f"was quarantined; epoch {decision.epoch} in slot {decision.slot} was promoted to "
            "committed so the arm can continue.",
            "the arm has NO previous-slot redundancy until the next epoch commits",
            f"epoch {corrupt_epoch} is lost and will be retrained; see "
            f"{self.corruption_log_path.name} and manifest.repairs",
        ]
        return repaired

    def load_committed(self, load_fn: Callable[[Path], Any]) -> tuple[Any, RecoveryDecision]:
        """Load the state the recovery decision names, hash-verified."""
        decision = self.recovery_decision()
        if decision.source == "fresh":
            raise DurableStateError("there is nothing committed to load for this arm.")
        path = self.slot_path(decision.slot)
        actual = sha256_file(path)
        if actual != decision.sha256:
            raise CheckpointVerificationError(
                f"{path} changed between the decision and the load ({actual} != "
                f"{decision.sha256}).")
        return load_fn(path), decision

    # --- write --------------------------------------------------------
    def commit_epoch(self, *, epoch: int, save_fn: Callable[[Path], None],
                     best_monitor_value: Optional[float] = None,
                     best_checkpoint_epoch: Optional[int] = None,
                     session_id: Optional[str] = None,
                     utc: Optional[str] = None) -> CommitManifest:
        """
        Durably commit one completed epoch. See the module docstring for
        the ordering and the invariants it buys.
        """
        current = self.manifest()
        if current is not None and epoch <= current.committed_epoch:
            raise DurableStateError(
                f"{self.block_id}/{self.arm_id}: refusing to commit epoch {epoch} when epoch "
                f"{current.committed_epoch} is already committed. The manifest must never go "
                "backwards.")
        target = _other(current.committed_slot) if current is not None else "a"
        if current is not None and target == current.committed_slot:   # pragma: no cover
            raise DurableStateError(
                f"{self.block_id}/{self.arm_id}: refusing to write the COMMITTED slot "
                f"{target!r}. The alternate slot is the only legal target.")

        inflight = self.inflight_path(target)
        self._boundary("before_inflight_write")
        save_fn(inflight)
        if not inflight.is_file():
            raise DurableStateError(
                f"the save function did not produce {inflight}; nothing was committed.")
        fsync_path(inflight)
        fsync_dir(self.arm_dir)
        self._boundary("after_inflight_write")

        # The committed slot is NOT touched here: only `target` is, and
        # target is the slot the manifest does not currently name.
        os.replace(inflight, self.slot_path(target))
        # Two DISTINCT boundaries. The rename has happened but its
        # directory entry may still be only in the page cache, so a
        # power loss here can lose the rename even though a running
        # process would see it. Killing a process cannot reproduce that
        # - see WHAT_SIGKILL_ESTABLISHES - but the boundary is named and
        # exercised so the ordering is at least pinned by a test.
        self._boundary("after_slot_replace_before_dir_fsync")
        fsync_dir(self.arm_dir)
        self._boundary("after_slot_replace")

        digest = sha256_file(self.slot_path(target))
        self._boundary("after_hash_verify")

        manifest = CommitManifest(
            arm_id=self.arm_id, block_id=self.block_id, seed=self.seed,
            epochs_total=self.epochs_total,
            committed_epoch=epoch, committed_slot=target, committed_sha256=digest,
            previous_epoch=(current.committed_epoch if current else None),
            previous_slot=(current.committed_slot if current else None),
            previous_sha256=(current.committed_sha256 if current else None),
            architecture_digest=self.architecture_digest,
            best_monitor=(current.best_monitor if current else "val_loss"),
            best_monitor_mode=(current.best_monitor_mode if current else "min"),
            best_monitor_value=(best_monitor_value if best_monitor_value is not None
                                else (current.best_monitor_value if current else None)),
            best_checkpoint_epoch=(best_checkpoint_epoch if best_checkpoint_epoch is not None
                                   else (current.best_checkpoint_epoch if current else None)),
            committed_at_utc=utc or _utc(), session_id=session_id,
            # Carried forward deliberately: a later commit must not erase
            # the record that this arm lost an epoch to corruption. The
            # append-only corruption log is the durable evidence, and the
            # manifest keeps the summary beside the state it describes.
            repairs=list(current.repairs or []) if current is not None else [],
        )
        self._boundary("before_manifest_commit")
        manifest.save(self.manifest_path)
        self._boundary("after_manifest_commit")
        return manifest


def _utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =====================================================================
# sealed-test evaluation journal
# =====================================================================

EVAL_MEASURED = "MEASURED"
EVAL_CONSUMED = "CONSUMED"
EVAL_RECORDED = "RECORDED"
EVAL_STATES = (EVAL_MEASURED, EVAL_CONSUMED, EVAL_RECORDED)


@dataclass
class EvalJournal:
    """
    Three-phase journal around the ONE permitted sealed-test evaluation
    of an (arm, block).

    THE PROBLEM IT SOLVES. The firewall permits exactly one confirmatory
    consumption per (arm, block), which is the right rule. But the
    previous code computed the accuracy, consumed the permit, and only
    then recorded the result in the run state. A crash in between left
    the permit spent and the accuracy nowhere: on resume the arm was not
    EVALUATED, so it was evaluated again, the firewall refused - correctly
    - and the block was recorded failed and non-replaceable. Four GPU
    hours of successful measurement were discarded because the bookkeeping
    outlived the result.

    THE PROTOCOL. The accuracy is journalled BEFORE the permit is spent:

        MEASURED  inference is done and the value is durable, permit not
                  yet spent
        CONSUMED  the permit has been spent and the firewall state saved
        RECORDED  the run state carries the result

    On resume, any existing journal means the measurement already
    happened, so the value is ADOPTED and inference is never repeated.
    Whether to spend the permit is then decided by the firewall's own
    record, so neither a double consumption nor a lost measurement is
    possible.

    The journal binds the sealed-test hash and the terminal-model hash,
    so a journal belonging to different artifacts is refused rather than
    adopted.
    """
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def read(self) -> Optional[dict[str, Any]]:
        if not self.path.is_file():
            return None
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            raise DurableStateError(
                f"{self.path} is empty; the evaluation journal is written by atomic replace, "
                "so an empty journal indicates corruption. Refusing to re-evaluate a sealed "
                "test set whose permit may already be spent.")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DurableStateError(
                f"{self.path} is not valid JSON ({exc}); refusing to re-evaluate a sealed "
                "test set whose permit may already be spent.") from exc
        if record.get("state") not in EVAL_STATES:
            raise DurableStateError(
                f"{self.path} records state {record.get('state')!r}, which is not one of "
                f"{EVAL_STATES}.")
        return record

    def write(self, *, state: str, accuracy: float, sealed_test_hash: str,
              terminal_model_hash: str, arm_id: str, block_id: str,
              session_id: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        if state not in EVAL_STATES:
            raise DurableStateError(f"unknown journal state {state!r}.")
        record = {
            "artifact": "sealed-eval-journal-v1",
            "state": state, "accuracy": float(accuracy),
            "sealed_test_hash": sealed_test_hash,
            "terminal_model_hash": terminal_model_hash,
            "arm_id": arm_id, "block_id": block_id, "session_id": session_id,
            "updated_at_utc": _utc(), **(extra or {}),
        }
        atomic_write_json(self.path, record)
        return record

    def advance_at_least(self, state: str) -> dict:
        """
        Move the journal forward to `state`, or leave it alone if it is
        already at or past it.

        Needed because an evaluation can legitimately be re-entered with
        a journal that is ALREADY fully RECORDED - a crash between the
        firewall consumption and the run-state write leaves exactly that
        shape, and the resumed session walks the same path again. Calling
        `advance` there raised "refusing to move the journal backwards"
        and aborted the run, turning a recoverable situation into a
        crash. The monotonic guard in `advance` is still correct; this is
        the idempotent form the resume path needs.
        """
        record = self.read()
        if record is None:
            raise DurableStateError(
                f"{self.path}: cannot advance a journal that does not exist.")
        if EVAL_STATES.index(record["state"]) >= EVAL_STATES.index(state):
            return record
        return self.advance(state)

    def advance(self, state: str) -> dict:
        record = self.read()
        if record is None:
            raise DurableStateError(
                f"{self.path}: cannot advance a journal that does not exist.")
        if EVAL_STATES.index(state) < EVAL_STATES.index(record["state"]):
            raise DurableStateError(
                f"{self.path}: refusing to move the journal backwards from "
                f"{record['state']} to {state}.")
        record = {**record, "state": state, "updated_at_utc": _utc()}
        atomic_write_json(self.path, record)
        return record

    def adopt(self, *, sealed_test_hash: str, terminal_model_hash: str,
              arm_id: Optional[str] = None,
              block_id: Optional[str] = None) -> Optional[dict]:
        """
        The journalled measurement for these exact artifacts, or None.

        Refuses - rather than adopting - a journal written against a
        different sealed test set or a different terminal model, because
        adopting it would report one model's accuracy as another's.

        The arm and block are checked too when the caller supplies them.
        The artifact hashes alone would almost certainly catch a journal
        that had been copied or left behind by an earlier run, since two
        arms will not produce bit-identical terminal models - but
        "almost certainly" is an argument from improbability, and the
        identity is recorded in the journal already. Checking it turns a
        statistical argument into a verified one, and names the mismatch
        instead of reporting a confusing model-hash difference.
        """
        record = self.read()
        if record is None:
            return None
        for field, expected in (("arm_id", arm_id), ("block_id", block_id)):
            actual = record.get(field)
            if expected is not None and actual != expected:
                raise DurableStateError(
                    f"{self.path}: the journal records {field}={actual!r} but it is being "
                    f"read for {field}={expected!r}. Refusing to adopt a measurement that "
                    "belongs to a different arm or block.")
        if record["sealed_test_hash"] != sealed_test_hash:
            raise DurableStateError(
                f"{self.path}: the journal was written against sealed test set "
                f"{record['sealed_test_hash'][:16]}... but the run's sealed set is "
                f"{sealed_test_hash[:16]}.... Refusing to adopt a measurement of different "
                "data.")
        if record["terminal_model_hash"] != terminal_model_hash:
            raise DurableStateError(
                f"{self.path}: the journal was written against terminal model "
                f"{record['terminal_model_hash'][:16]}... but this arm's terminal model is "
                f"{terminal_model_hash[:16]}.... Refusing to adopt a measurement of a "
                "different model.")
        return record
