"""
The PAR2 readiness gate: the frozen gate, unchanged, with only its memory measurement superseded.

`readiness_gate` (audit/implementation/run_ii4_impl.py, frozen) is called
exactly as the PAR1 launcher calls it. Its preflight problems are the
concatenation of the GPU, memory, disk, filesystem and durability problems;
PAR2 removes precisely the entries that the frozen memory report produced
(as a multiset) and keeps every other one. The corrected measurement is then
required in their place, compared with the same requirement the frozen gate
used. The frozen memory report is kept verbatim in the gate entry.

PAR2 can therefore only change a refusal that was caused by the memory
measurement; every other refusal stands, and a frozen memory PASS does not
excuse a corrected memory failure.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any, Callable, Optional

from audit.implementation.ii4_par2 import memory

PAR1_SUITE = "audit/implementation/ii4_parallel/tests"
PAR2_SUITE = "audit/implementation/ii4_par2/tests"


def frozen_readiness(out) -> tuple[bool, dict]:
    from audit.implementation.run_ii4_impl import readiness_gate
    return readiness_gate(out, acknowledged=("idle_gpus",))         # exactly as the PAR1 launcher's _gate


def run_suite(path: str, python: str = sys.executable) -> dict[str, Any]:
    r = subprocess.run([python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", path],
                       capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    return {"suite": path, "returncode": r.returncode, "tail": tail,
            "clear": r.returncode == 0 and "skipped" not in tail}


def combine(frozen_ok: bool, frozen_gate: dict, corrected: dict, suites: list[dict]) -> dict[str, Any]:
    """Pure: the PAR2 decision from the frozen gate's output, the corrected measurement and the suites."""
    pre = frozen_gate["preflight"]
    fmem = pre["memory"]
    remaining, inconsistent = list(pre["problems"]), []
    for p in fmem.get("problems", []):
        if p in remaining:
            remaining.remove(p)
        else:
            inconsistent.append(p)
    reasons = []
    if frozen_gate.get("dir_problems"):
        reasons.append(f"output directory: {frozen_gate['dir_problems']}")
    if not frozen_gate["suites"].get("clear"):
        reasons.append("frozen required suites not clear")
    if remaining:
        reasons.append(f"frozen preflight problems other than memory: {remaining}")
    if inconsistent:
        reasons.append(f"memory problems absent from the preflight list (inconsistent report): {inconsistent}")
    if pre.get("unacknowledged"):
        reasons.append(f"unacknowledged warnings: {pre['unacknowledged']}")
    if fmem.get("required_gb") is None:
        reasons.append("the frozen gate stated no memory requirement")
    elif abs(fmem["required_gb"] * 1e9 - corrected["required_bytes"]) > 0.5:
        reasons.append("corrected measurement compared with a different requirement than the frozen gate")
    if not corrected["clear"]:
        reasons.append(f"corrected memory: {corrected['blocks']}")
    for s in suites:
        if not s["clear"]:
            reasons.append(f"suite {s['suite']}: {s['tail']}")
    return {
        "kind": "gate",
        "clear": not reasons,
        "refusal_reasons": reasons,
        "frozen_gate_clear": bool(frozen_ok),
        "frozen_memory_report": fmem,
        "frozen_memory_superseded": {"problems_removed": [p for p in fmem.get("problems", [])
                                                          if p not in inconsistent],
                                     "basis": "SC_AVPHYS_PAGES is MemFree on Linux glibc"},
        "frozen_other_preflight_problems": remaining,
        "frozen_dir_problems": frozen_gate.get("dir_problems"),
        "frozen_suites": frozen_gate["suites"],
        "frozen_preflight_unacknowledged": pre.get("unacknowledged"),
        "frozen_preflight_warnings": pre.get("warnings"),
        "corrected_memory": corrected,
        "suites": suites,
    }


def par2_gate(out, *, frozen: Callable = frozen_readiness, measure: Optional[Callable] = None,
              suites: Optional[Callable] = None) -> dict[str, Any]:
    ok, g = frozen(out)
    req = g["preflight"]["memory"].get("required_gb")
    corrected = (measure or memory.measure)(required_bytes=int(round((req or 0) * 1e9)))
    s = suites() if suites else [run_suite(PAR1_SUITE), run_suite(PAR2_SUITE)]
    entry = combine(ok, g, corrected, s)
    entry["frozen_gate"] = g
    return entry
