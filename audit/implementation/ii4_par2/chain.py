"""
Hash-chained gate record for PAR2: one JSON object per line, beside the run directory.

Each entry carries `seq`, `prev_sha256` (the previous entry's `entry_sha256`,
64 zeros for the first) and `entry_sha256`, the SHA-256 of the entry's
canonical JSON without `entry_sha256`. Entries are appended, never rewritten.
The verifier (verify_par2.py) recomputes the chain with its own code.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def chain_path(output_dir) -> Path:
    """`<run dir>.par2_gate.jsonl`: outside the run directory, so the frozen directory checks are unaffected."""
    out = Path(output_dir)
    return out.parent / f"{out.name}.par2_gate.jsonl"


def read_chain(path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def check_chain(entries: list[dict[str, Any]]) -> list[str]:
    problems, prev = [], GENESIS
    for i, e in enumerate(entries):
        body = {k: v for k, v in e.items() if k != "entry_sha256"}
        if e.get("seq") != i:
            problems.append(f"entry {i}: seq {e.get('seq')}")
        if e.get("prev_sha256") != prev:
            problems.append(f"entry {i}: prev_sha256 does not match entry {i - 1}")
        if hashlib.sha256(canonical(body)).hexdigest() != e.get("entry_sha256"):
            problems.append(f"entry {i}: entry_sha256 does not match its content")
        prev = e.get("entry_sha256")
    return problems


def append(path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append one entry; refuses if the existing chain does not verify."""
    p = Path(path)
    existing = read_chain(p)
    problems = check_chain(existing)
    if problems:
        raise RuntimeError(f"refusing to extend a broken gate record {p}: {problems[:3]}")
    body = json.loads(json.dumps(entry, default=str, allow_nan=False))      # normalise before hashing
    body.update({"seq": len(existing), "prev_sha256": existing[-1]["entry_sha256"] if existing else GENESIS,
                 "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    body["entry_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, canonical(body) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return body
