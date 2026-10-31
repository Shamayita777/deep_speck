"""
Crash-safe evidence lifecycle: an authoritative hash-chained journal, a declared state machine,
transactional artifact commits, run-record materialisation and recovery with quarantine.

Authority
---------
The journal is the only authoritative record. A run's record file (``record.json``) is a cache
materialised from the journal after each committed transition. On every load the record is
rebuilt from the journal and compared with the cache:

* cache equal to the journal                      -> consistent;
* cache equal to an earlier journal prefix        -> stale cache (crash after the journal commit
                                                     and before the cache write): rewritten from
                                                     the journal and the rewrite is journalled;
* cache absent while the journal has transitions  -> rewritten from the journal and journalled;
* anything else                                   -> ``InconsistentEvidence`` (never repaired).

Commit protocol for an artifact ``name`` of a run (``commit_artifact``)::

    1. journal  artifact_intent {run_id, artifact, token}
    2. write    <run>/.staging/<name>.<token>.part      (complete write, flush, fsync)
    3. hash     SHA-256 of the staged file
    4. validate optional caller check on the staged file (failure -> quarantine, raise)
    5. promote  os.link(part, <run>/<name>)  (atomic, never overwrites), unlink part, fsync dir
    6. commit   the caller's state transition carries {<name>_sha256, <name>_token}

Only step 6 commits. Every interruption before it leaves an artifact that ``recover_run``
classifies deterministically:

====================================================  ==========================================
situation on resume                                   action
====================================================  ==========================================
no artifact                                           nothing; the step is executed
partial or unpromoted file in .staging                quarantined, journalled, step re-executed
complete final file, transition not committed         quarantined, journalled, step re-executed
committed file present and hash equal                 kept
committed file missing or hash different              InconsistentEvidence (run must FAIL)
uncommitted final file next to a committed one        impossible by construction (link refuses)
====================================================  ==========================================

Uncommitted artifacts are never analysed, so re-executing a step after quarantine cannot select
on an outcome. Quarantined files are retained under ``<run>/quarantine/`` with their hashes in
the journal. Nothing here deletes evidence.

Fault injection: ``arm_fault(point)`` makes the named point raise ``InjectedFault``; tests use
it to interrupt every boundary listed in ``FAULT_POINTS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

JOURNAL_SCHEMA = "core-journal-2"
GENESIS = "0" * 64
STAGING = ".staging"
QUARANTINE = "quarantine"
RECORD = "record.json"

FAULT_POINTS = (
    "before_write",             # intent journalled, nothing written
    "during_write",             # a partial staged file exists
    "after_write",              # staged file complete, not hashed
    "after_hash",               # hashed and validated, not promoted
    "after_promote",            # final file exists, transition not committed
    "after_commit",             # transition journalled, record cache not written
    "during_quarantine",        # quarantine intent journalled, file not yet moved
)


class EvidenceError(RuntimeError):
    pass


class InconsistentEvidence(EvidenceError):
    """The evidence cannot be reconciled with its journal. Never repaired automatically."""


class InjectedFault(BaseException):
    """Raised at an armed fault point. BaseException so that ordinary handlers cannot absorb it."""


_ARMED: set[str] = set()


def arm_fault(point: str) -> None:
    if point not in FAULT_POINTS:
        raise ValueError(f"unknown fault point {point!r}")
    _ARMED.add(point)


def disarm_faults() -> None:
    _ARMED.clear()


def _fault(point: str) -> None:
    if point in _ARMED:
        _ARMED.discard(point)
        raise InjectedFault(point)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False, default=str).encode("ascii")


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # some filesystems do not support directory fsync
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes, *, replace: bool = False) -> str:
    """Write via a fsynced temporary file and rename. Refuses to overwrite unless ``replace``."""
    path = Path(path)
    if path.exists() and not replace:
        raise EvidenceError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
    return sha256_bytes(data)


def write_json(path: Path, obj, *, replace: bool = False) -> str:
    return atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False,
                                                default=str) + "\n").encode(), replace=replace)


# --------------------------------------------------------------------------- state machine


@dataclass(frozen=True)
class StateMachine:
    initial: str
    transitions: dict            # state -> tuple of permitted next states
    terminal: tuple

    @property
    def states(self) -> tuple:
        s = {self.initial, *self.transitions, *self.terminal}
        for v in self.transitions.values():
            s.update(v)
        return tuple(sorted(s))

    def permitted(self, frm: str, to: str) -> bool:
        return frm not in self.terminal and to in self.transitions.get(frm, ())

    def check(self, frm: str, to: str) -> None:
        if frm in self.terminal:
            raise EvidenceError(f"state {frm} is terminal; it is never left or rewritten")
        if not self.permitted(frm, to):
            raise EvidenceError(f"transition {frm} -> {to} is not permitted")


# --------------------------------------------------------------------------------- journal


class Journal:
    """Append-only JSON-lines journal. Each line carries a continuous ``seq``, the experiment id,
    and the SHA-256 of the previous line's bytes. A line is committed once it is complete
    (newline-terminated) and fsynced."""

    def __init__(self, path: Path, experiment_id: str):
        self.path = Path(path)
        self.experiment_id = experiment_id
        self._tail: tuple[int, str] | None = None     # (last seq, sha of last line)

    # -- reading
    def raw_lines(self) -> list[bytes]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise EvidenceError(f"{self.path} has a torn final line; call recover_torn_tail()")
        return data.splitlines()

    def entries(self) -> list[dict]:
        return [json.loads(ln) for ln in self.raw_lines()]

    def _load_tail(self) -> tuple[int, str]:
        if self._tail is None:
            lines = self.raw_lines()
            if not lines:
                self._tail = (-1, GENESIS)
            else:
                self._tail = (json.loads(lines[-1])["seq"], sha256_bytes(lines[-1]))
        return self._tail

    # -- recovery of an interrupted append
    def recover_torn_tail(self) -> dict | None:
        """Move an incomplete final line aside (it was never committed) and journal the move."""
        if not self.path.exists():
            return None
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return None
        cut = data.rfind(b"\n") + 1
        tail = data[cut:]
        aside = self.path.with_name(f"{self.path.name}.torn-{sha256_bytes(tail)[:16]}")
        if not aside.exists():
            atomic_write_bytes(aside, tail)
        with open(self.path, "r+b") as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
        self._tail = None
        info = {"file": aside.name, "bytes": len(tail), "sha256": sha256_bytes(tail)}
        self.append("journal_torn_tail_recovered", **info)
        return info

    # -- writing
    def append(self, event: str, **payload) -> dict:
        seq, prev = self._load_tail()
        entry = {"schema": JOURNAL_SCHEMA, "seq": seq + 1, "utc": utc(),
                 "experiment_id": self.experiment_id, "event": event, "prev_sha256": prev, **payload}
        line = canonical(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write(line + b"\n")
            f.flush()
            os.fsync(f.fileno())
        self._tail = (seq + 1, sha256_bytes(line))
        return entry


def chain_problems(lines: list[bytes], experiment_id: str) -> list[str]:
    """Structural problems of a journal: sequence, previous-hash chain, schema, experiment id."""
    out, prev = [], GENESIS
    for i, ln in enumerate(lines):
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            out.append(f"line {i}: not JSON")
            continue
        if e.get("seq") != i:
            out.append(f"line {i}: seq {e.get('seq')} (expected {i})")
        if e.get("prev_sha256") != prev:
            out.append(f"line {i}: prev_sha256 does not match line {i - 1}")
        if e.get("experiment_id") != experiment_id:
            out.append(f"line {i}: experiment_id {e.get('experiment_id')!r}")
        if e.get("schema") != JOURNAL_SCHEMA:
            out.append(f"line {i}: schema {e.get('schema')!r}")
        prev = sha256_bytes(ln)
    return out


# ------------------------------------------------------------------------- run records


TRANSITION = "run_transition"


def materialise(entries: list[dict], run_id: str, machine: StateMachine, *,
                upto_seq: int | None = None, record_schema: str = "core-run-record-2") -> dict:
    """Replay a run's committed transitions. Raises InconsistentEvidence on an illegal history."""
    rec = {"schema": record_schema, "run_id": run_id, "state": machine.initial, "history": [],
           "journal_seq": None}
    for e in entries:
        if upto_seq is not None and e["seq"] > upto_seq:
            break
        if e.get("event") != TRANSITION or e.get("run_id") != run_id:
            continue
        if e.get("from") != rec["state"]:
            raise InconsistentEvidence(f"{run_id}: journal transition at seq {e['seq']} starts from "
                                       f"{e.get('from')} but the replayed state is {rec['state']}")
        if not machine.permitted(e["from"], e["to"]):
            raise InconsistentEvidence(f"{run_id}: journal transition {e['from']} -> {e['to']} "
                                       f"at seq {e['seq']} is not permitted")
        rec.update(e.get("fields", {}))
        rec["state"] = e["to"]
        rec["history"] = rec["history"] + [{"state": e["to"], "seq": e["seq"], "utc": e["utc"]}]
        rec["journal_seq"] = e["seq"]
    return rec


@dataclass
class EvidenceStore:
    """One experiment output directory: journal, state machine, run records and artifacts."""
    root: Path
    experiment_id: str
    machine: StateMachine
    record_schema: str = "core-run-record-2"
    runs_dir: str = "runs"
    journal_name: str = "journal.jsonl"
    _journal: Journal | None = field(default=None, repr=False)

    def __post_init__(self):
        self.root = Path(self.root)

    @property
    def journal(self) -> Journal:
        if self._journal is None:
            self._journal = Journal(self.root / self.journal_name, self.experiment_id)
        return self._journal

    def run_dir(self, run_id: str) -> Path:
        return self.root / self.runs_dir / run_id

    # ---------------------------------------------------------------- open / reconcile
    def open(self) -> list[str]:
        """Recover a torn journal tail and check the chain. Returns notes of what was done."""
        notes = []
        torn = self.journal.recover_torn_tail()
        if torn:
            notes.append(f"recovered torn journal tail ({torn['bytes']} bytes) into {torn['file']}")
        problems = chain_problems(self.journal.raw_lines(), self.experiment_id)
        if problems:
            raise InconsistentEvidence(f"journal chain is invalid: {problems[:5]}")
        return notes

    def record(self, run_id: str) -> dict:
        """The authoritative record (from the journal), after reconciling the record cache."""
        entries = self.journal.entries()
        rec = materialise(entries, run_id, self.machine, record_schema=self.record_schema)
        path = self.run_dir(run_id) / RECORD
        if not path.exists():
            if rec["journal_seq"] is not None:
                self._write_cache(run_id, rec, reason="record cache absent")
            return rec
        cached = json.loads(path.read_text())
        if cached == rec:
            return rec
        seq = cached.get("journal_seq")
        if isinstance(seq, int) and (rec["journal_seq"] or -1) > seq:
            earlier = materialise(entries, run_id, self.machine, upto_seq=seq,
                                  record_schema=self.record_schema)
            if earlier == cached:
                self._write_cache(run_id, rec, reason=f"stale record cache at seq {seq}")
                return rec
        raise InconsistentEvidence(f"{run_id}: record.json does not match the journal "
                                   f"(cache state {cached.get('state')!r}, journal state {rec['state']!r})")

    def _write_cache(self, run_id: str, rec: dict, *, reason: str | None = None) -> None:
        write_json(self.run_dir(run_id) / RECORD, rec, replace=True)
        if reason:
            self.journal.append("record_cache_rematerialised", run_id=run_id, reason=reason,
                                state=rec["state"], journal_seq=rec["journal_seq"])

    # ------------------------------------------------------------------- transitions
    def transition(self, run_id: str, to: str, **fields) -> dict:
        rec = self.record(run_id)
        self.machine.check(rec["state"], to)
        self.journal.append(TRANSITION, run_id=run_id, **{"from": rec["state"], "to": to},
                            fields=json.loads(canonical(fields)))
        _fault("after_commit")
        new = self.record_from_journal(run_id)
        self._write_cache(run_id, new)
        return new

    def record_from_journal(self, run_id: str) -> dict:
        return materialise(self.journal.entries(), run_id, self.machine, record_schema=self.record_schema)

    # ----------------------------------------------------------------------- artifacts
    def commit_artifact(self, run_id: str, name: str, writer: Callable[[Path], None], *,
                        validate: Callable[[Path], None] | None = None) -> dict:
        """Steps 1-5 of the commit protocol. Returns {sha256, token}; the caller commits them
        in its next transition. Refuses if an uncommitted file is in the way (run recover_run)."""
        rdir = self.run_dir(run_id)
        final = rdir / name
        if final.exists():
            raise EvidenceError(f"{final} exists; recover_run must classify it first")
        token = uuid.uuid4().hex
        self.journal.append("artifact_intent", run_id=run_id, artifact=name, token=token)
        _fault("before_write")
        staging = rdir / STAGING
        staging.mkdir(parents=True, exist_ok=True)
        part = staging / f"{name}.{token}.part"
        writer(part)
        if not part.exists():
            raise EvidenceError(f"writer did not create {part}")
        with open(part, "rb") as f:
            os.fsync(f.fileno())
        _fault("after_write")
        sha = sha256_file(part)
        if validate is not None:
            validate(part)
        _fault("after_hash")
        os.link(part, final)            # atomic; raises FileExistsError instead of overwriting
        os.unlink(part)
        fsync_dir(rdir)
        fsync_dir(staging)
        _fault("after_promote")
        return {f"{_key(name)}_sha256": sha, f"{_key(name)}_token": token}

    def writer_with_fault(self, write: Callable[[Path], None]) -> Callable[[Path], None]:
        """Wrap a writer so the 'during_write' fault leaves a genuinely partial staged file."""
        def w(path: Path) -> None:
            if "during_write" in _ARMED:
                tmp = path.with_name(path.name + ".full")
                write(tmp)
                data = tmp.read_bytes()
                tmp.unlink()
                path.write_bytes(data[: max(1, len(data) // 2)])
                _fault("during_write")
            write(path)
        return w

    # ------------------------------------------------------------------------ recovery
    def recover_run(self, run_id: str, artifact_names: tuple) -> list[dict]:
        """Classify every artifact of a run against its committed record (see module table).
        Quarantines uncommitted files; raises InconsistentEvidence for committed-artifact damage."""
        rec = self.record(run_id)
        rdir = self.run_dir(run_id)
        actions = []
        for name in artifact_names:
            final = rdir / name
            committed = rec.get(f"{_key(name)}_sha256")
            if committed:
                if not final.exists():
                    raise InconsistentEvidence(f"{run_id}: committed artifact {name} is missing")
                got = sha256_file(final)
                if got != committed:
                    raise InconsistentEvidence(f"{run_id}: committed artifact {name} has SHA-256 {got}, "
                                               f"journal records {committed}")
            elif final.exists():
                actions.append(self._quarantine(run_id, final, "complete artifact whose transition "
                                                               "was never committed"))
        staging = rdir / STAGING
        if staging.exists():
            for p in sorted(staging.iterdir()):
                if p.is_file():
                    actions.append(self._quarantine(run_id, p, "partial or unpromoted staged write"))
        return actions

    def _quarantine(self, run_id: str, path: Path, reason: str) -> dict:
        qdir = self.run_dir(run_id) / QUARANTINE
        qdir.mkdir(parents=True, exist_ok=True)
        sha = sha256_file(path)
        dst = qdir / f"{uuid.uuid4().hex[:12]}__{path.name}"
        info = {"run_id": run_id, "source": path.relative_to(self.root).as_posix(),
                "destination": dst.relative_to(self.root).as_posix(), "sha256": sha,
                "bytes": path.stat().st_size, "reason": reason}
        self.journal.append("artifact_quarantined", **info)
        _fault("during_quarantine")
        os.rename(path, dst)
        fsync_dir(qdir)
        fsync_dir(path.parent)
        return info


def _key(name: str) -> str:
    """Record field prefix for an artifact file name: 'data.npz' -> 'data'."""
    return name.split(".", 1)[0]
