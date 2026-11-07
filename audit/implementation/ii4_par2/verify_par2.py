#!/usr/bin/env python3
"""
Independent verifier for a PAR2 production run. Standard library only; imports nothing from the project.

  python -I audit/implementation/ii4_par2/verify_par2.py --run-dir RUN --repo-root . \
      --frozen audit/implementation/preregistrations/ii-4-gohr-impl-v1-par1.frozen.json --require-dual-gpu

Checks, with its own code:
  1. the gate record `<RUN>.par2_gate.jsonl`: sequence, prev links and every entry hash;
  2. every entry names the PAR2 record on disk, and (with --repo-root) every PAR2 source on
     disk equals that record;
  3. every execute_start entry is preceded, in the same boot, by a CLEAR gate entry (the one it
     names) and by a PASS memory probe;
  4. every parallel session in the run's state.json started after its own execute_start entry,
     one entry per session;
  5. the frozen PAR1 verifier (audit/implementation/ii4_parallel/verify.py) reports VERIFIED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

GENESIS = "0" * 64
PAR2_RECORD = "audit/implementation/preregistrations/ii-4-gohr-impl-v1-par2.frozen.json"
PAR1_VERIFIER = "audit/implementation/ii4_parallel/verify.py"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_entries(entries: list, record_sha: str) -> list[str]:
    P, prev = [], GENESIS
    for i, e in enumerate(entries):
        body = {k: v for k, v in e.items() if k != "entry_sha256"}
        if e.get("seq") != i or e.get("prev_sha256") != prev:
            P.append(f"gate record entry {i}: broken sequence or link")
        if hashlib.sha256(_canon(body)).hexdigest() != e.get("entry_sha256"):
            P.append(f"gate record entry {i}: content does not match its hash")
        if e.get("par2_record_sha256") != record_sha:
            P.append(f"gate record entry {i}: names PAR2 record {e.get('par2_record_sha256')}, not {record_sha}")
        prev = e.get("entry_sha256")
    by_hash = {e.get("entry_sha256"): e for e in entries}
    for i, e in enumerate(entries):
        if e.get("kind") != "execute_start":
            continue
        g = by_hash.get(e.get("gate_entry_sha256"))
        if not g or g.get("kind") != "gate" or g.get("seq", 1 << 30) > i:
            P.append(f"execute_start {i}: does not name an earlier gate entry")
            continue
        if g.get("clear") is not True:
            P.append(f"execute_start {i}: its gate entry {g['seq']} was not CLEAR")
        if (g.get("corrected_memory") or {}).get("clear") is not True:
            P.append(f"execute_start {i}: corrected memory in gate entry {g['seq']} was not clear")
        boot = e.get("boot_id")
        if not boot or (g.get("corrected_memory") or {}).get("boot_id") != boot:
            P.append(f"execute_start {i}: gate entry from another boot")
        probes = [p for p in entries[:i] if p.get("kind") == "memprobe" and p.get("boot_id") == boot]
        if not probes or probes[-1].get("status") != "PASS":
            P.append(f"execute_start {i}: no PASS memory probe earlier in the same boot")
        elif not probes[-1].get("full_size"):
            P.append(f"execute_start {i}: the probe was not full size")
    return P


def check_sessions(entries: list, state: dict) -> list[str]:
    P = []
    starts = [e for e in entries if e.get("kind") == "execute_start"]
    sessions = state.get("parallel_sessions") or []
    if len(sessions) > len(starts):
        P.append(f"{len(sessions)} parallel sessions but {len(starts)} execute_start entries")
    used = set()
    for s in sessions:
        t = _t(s["started_at_utc"])
        before = [e for e in starts if _t(e["recorded_at_utc"]) <= t and e["entry_sha256"] not in used]
        if not before:
            P.append(f"session {s.get('session_id')} started {s['started_at_utc']} with no unused "
                     "execute_start entry before it")
            continue
        used.add(before[-1]["entry_sha256"])
    return P


def run_frozen_verifier(repo: Path, run_dir: Path, frozen, dual: bool) -> dict:
    cmd = [sys.executable, "-I", str(repo / PAR1_VERIFIER), "--run-dir", str(run_dir), "--repo-root", str(repo)]
    if frozen:
        cmd += ["--frozen", str(frozen)]
    if dual:
        cmd += ["--require-dual-gpu"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"status": "NOT_VERIFIED", "problems": [f"frozen verifier output unreadable: {r.stderr[-500:]}"]}


def verify(run_dir, *, repo_root, frozen=None, require_dual_gpu=False, frozen_verifier=run_frozen_verifier) -> dict:
    run_dir, repo = Path(run_dir), Path(repo_root)
    P: list[str] = []
    gate_path = run_dir.parent / f"{run_dir.name}.par2_gate.jsonl"
    rec_path = repo / PAR2_RECORD
    if not gate_path.is_file() or not rec_path.is_file():
        return {"status": "NOT_VERIFIED", "problems": [f"missing {gate_path if not gate_path.is_file() else rec_path}"]}
    rec_sha = _sha(rec_path)
    rec = json.loads(rec_path.read_text())
    for label, s in (rec.get("par2_sources") or {}).items():
        p = repo / s["path"]
        if not p.is_file() or _sha(p) != s["sha256"]:
            P.append(f"PAR2 source on disk differs from the record: {s['path']}")
    for a in rec.get("amends", []):
        if not (repo / a["record"]).is_file() or _sha(repo / a["record"]) != a["sha256"]:
            P.append(f"amended record differs: {a['record']}")
    try:
        entries = [json.loads(line) for line in gate_path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        return {"status": "NOT_VERIFIED", "problems": [f"gate record unreadable: {exc}"]}
    P += check_entries(entries, rec_sha)
    state_path = run_dir / "state.json"
    if state_path.is_file():
        P += check_sessions(entries, json.loads(state_path.read_text()))
    else:
        P.append("state.json missing")
    fv = frozen_verifier(repo, run_dir, frozen, require_dual_gpu)
    if fv.get("status") != "VERIFIED":
        P.append(f"frozen PAR1 verifier: {fv.get('status')} {fv.get('problems', [])[:5]}")
    return {"status": "VERIFIED" if not P else "NOT_VERIFIED", "problems": P,
            "par2_record_sha256": rec_sha, "gate_record": str(gate_path),
            "gate_record_sha256": _sha(gate_path), "n_entries": len(entries),
            "execute_starts": sum(e.get("kind") == "execute_start" for e in entries),
            "frozen_verifier": fv}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--frozen", type=Path, default=None)
    ap.add_argument("--require-dual-gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = verify(a.run_dir, repo_root=a.repo_root, frozen=a.frozen, require_dual_gpu=a.require_dual_gpu)
    text = json.dumps(rep, indent=1, sort_keys=True, default=str)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if rep["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
