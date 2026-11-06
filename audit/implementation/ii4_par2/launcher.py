#!/usr/bin/env python3
"""
Launcher for II-4-GOHR-IMPL-V1-PAR2, the gate-only amendment of PAR1. Run from the repository root.

  --freeze-par2  write the PAR2 record (refuses to overwrite).
  --preflight    PAR2 readiness gate: the frozen gate, with its memory measurement superseded
                 by the cgroup-aware one, plus the PAR1 and PAR2 suites and two GPUs.
  --memprobe     full-size concurrent memory probe on two GPUs (about ten minutes). Not evidence.
  --dry-run      PAR2 record, PAR1 record and plan checks, the PAR2 gate, and a PASS probe
                 from this boot; writes nothing in the run directory.
  --execute      production through the frozen PAR1 scheduler: everything --dry-run requires,
                 all three review flags and exactly two GPUs.
  --verify       PAR2 verifier (which also runs the frozen PAR1 verifier).

Every gate, probe and execute decision is appended to `<run dir>.par2_gate.jsonl`,
a hash chain kept beside the run directory. Smoke, interruption and resume tests
use the PAR1 launcher unchanged: they do not pass through the readiness gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from audit.implementation.ii4_par2 import amendment, chain
from audit.implementation.ii4_par2.gate import par2_gate
from audit.implementation.ii4_parallel.launcher import (
    AMENDMENT_FROZEN, DEFAULT_OUTPUT, FACTORY_PRODUCTION, production_factory, visible_gpus)


def _stamp(entry: dict, rec: dict, gpus_seen: list) -> dict:
    entry.update({"amendment_id": amendment.PAR2_ID, "gate_version": amendment.PAR2_VERSION,
                  "par2_record_sha256": rec.get("record_sha256"), "gpus_visible": gpus_seen})
    return entry


def _record_or_refuse() -> tuple[bool, dict]:
    problems, rec = amendment.check_record()
    if problems:
        print("PAR2 record:", "\n  ".join(["NOT VALID"] + problems))
        return False, rec
    print(f"PAR2 record MATCHES ({rec['record_sha256']})")
    return True, rec


def _gate(out: Path, rec: dict, seen: list) -> dict:
    entry = _stamp(par2_gate(out), rec, seen)
    if len(seen) < 2:
        entry["clear"] = False
        entry["refusal_reasons"].append(f"{len(seen)} GPU(s) visible; two are required")
    e = chain.append(chain.chain_path(out), entry)
    m = e["corrected_memory"]
    f = e["frozen_memory_report"]
    print("PAR2 READINESS GATE", "CLEAR" if e["clear"] else "NOT CLEAR")
    print(f"  frozen memory (superseded)  {f.get('available_gb')} GB (SC_AVPHYS_PAGES = MemFree)")
    print(f"  MemAvailable                {_gb(m.get('host_mem_available'))}")
    for lv in m.get("cgroup_levels", []):
        print(f"  cgroup {lv['path']:<20} limit {_gb(lv.get('limit'))} current {_gb(lv.get('current'))} "
              f"free {_gb(lv.get('free'))} reclaimable {_gb(lv.get('reclaimable'))} "
              f"credit {'yes' if lv.get('credit_allowed') else 'no ' + str(lv.get('no_credit_because'))}")
    print("  usable U(phi)               " + ", ".join(f"phi={k}: {_gb(v)}" for k, v in m["usable_bytes"].items()))
    print(f"  requirement                 {m['required_gb']:.3f} GB (frozen); gating phi = {m['phi']}")
    for r in e["refusal_reasons"]:
        print("  REFUSED:", r)
    print(f"  gate record entry {e['seq']} {e['entry_sha256']} -> {chain.chain_path(out)}")
    return e


def _gb(v) -> str:
    return "n/a" if v is None else f"{v / 1e9:.3f} GB"


def _probe_ok(out: Path, boot_id) -> tuple[bool, str]:
    probes = [e for e in chain.read_chain(chain.chain_path(out)) if e.get("kind") == "memprobe"]
    same = [e for e in probes if boot_id and e.get("boot_id") == boot_id]
    if not same:
        return False, "no memory probe recorded in this boot (run --memprobe)"
    last = same[-1]
    return last["status"] == "PASS", f"latest probe this boot: entry {last['seq']} {last['status']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--freeze-par2", "--preflight", "--memprobe", "--dry-run", "--execute", "--verify"):
        m.add_argument(flag, action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--probe-dir", type=Path, default=None)
    ap.add_argument("--i-have-reviewed-the-preregistration", action="store_true", dest="reviewed")
    ap.add_argument("--i-have-reviewed-the-amendment", action="store_true", dest="reviewed_amendment")
    ap.add_argument("--i-have-reviewed-the-par2-amendment", action="store_true", dest="reviewed_par2")
    a = ap.parse_args(argv)
    gpus = tuple(g for g in a.gpus.split(",") if g)
    out = a.output_dir or DEFAULT_OUTPUT

    if a.freeze_par2:
        path = amendment.write_record()
        print(f"wrote {path} {amendment.sha256_file(path)}")
        return 0
    if a.verify:
        from audit.implementation.ii4_par2 import verify_par2
        return verify_par2.main(["--run-dir", str(out), "--repo-root", ".", "--require-dual-gpu",
                                 "--frozen", str(AMENDMENT_FROZEN)])

    ok_rec, rec = _record_or_refuse()
    seen = visible_gpus()
    print(f"GPUs visible: {len(seen)}")
    if not ok_rec:
        return 1

    if a.memprobe:
        from audit.implementation.ii4_par2.memprobe import run_probe
        if len(gpus) != 2 or len(seen) < 2:
            print("REFUSED: the probe needs two GPUs, as production does.")
            return 1
        req = int(round(amendment.frozen_threshold_gb() * 1e9))
        pdir = a.probe_dir or Path(f"{out}.memprobe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        e = chain.append(chain.chain_path(out),
                         _stamp(run_probe(workdir=pdir, gpus=gpus, required_bytes=req), rec, seen))
        print(f"MEMORY PROBE {e['status']} (entry {e['seq']} {e['entry_sha256']})")
        for g, c in sorted(e["children"].items()):
            print(f"  GPU {g}: rc {c.get('returncode')} arms {c.get('arms_completed')} "
                  f"VmHWM {_gb(c.get('final_VmHWM'))} sampled RSS peak {_gb(e['sampled_peak_rss_bytes'].get(g))} "
                  f"{c.get('seconds', 0):.0f} s")
        print(f"  sum of child peaks {_gb(e['sum_child_vmhwm_bytes'])}; sampled peak memory.current "
              f"{_gb(e['sampled_peak_cgroup_current_bytes'])}")
        print(f"  memory.peak before {e['memory_peak_before']} after {e['memory_peak_after']}")
        for r in e["fail_reasons"]:
            print("  FAIL:", r)
        return 0 if e["status"] == "PASS" else 1

    g = _gate(out, rec, seen)
    if a.preflight:
        return 0 if g["clear"] else 1

    from audit.implementation.ii4_parallel.scheduler import ParallelComparisonRunner, ParallelSchedulingError
    from audit.implementation.ii4_parallel.scheduler import execute_parallel
    plan, adapter = production_factory(out)
    runner = ParallelComparisonRunner(plan=plan, adapter=adapter)
    frozen = json.loads(AMENDMENT_FROZEN.read_text()) if AMENDMENT_FROZEN.exists() else None
    if frozen is None:
        print(f"REFUSED: no PAR1 amendment record at {AMENDMENT_FROZEN}.")
        return 1
    problems = runner.check_frozen(frozen)
    print("PAR1 amendment record", "MATCHES" if not problems else "DIFFERS:\n  " + "\n  ".join(problems))
    probe_ok, probe_msg = _probe_ok(out, g["corrected_memory"].get("boot_id"))
    print(f"memory probe: {probe_msg}")
    ready = g["clear"] and probe_ok and not problems and len(seen) >= 2
    if a.dry_run:
        runner.run(execute=False)
        return 0 if ready else 1
    if not (a.reviewed and a.reviewed_amendment and a.reviewed_par2):
        print("REFUSED: --execute needs --i-have-reviewed-the-preregistration, "
              "--i-have-reviewed-the-amendment and --i-have-reviewed-the-par2-amendment.")
        return 1
    if not ready or len(gpus) != 2:
        print("REFUSED: the PAR1 record, the PAR2 gate, the memory probe or the GPU check did not clear.")
        return 1
    start = chain.append(chain.chain_path(out), _stamp(
        {"kind": "execute_start", "gate_entry_sha256": g["entry_sha256"], "gpus": list(gpus),
         "boot_id": g["corrected_memory"].get("boot_id"),
         "experiment_fingerprint": runner.runtime_provenance()["experiment_fingerprint"]}, rec, seen))
    print(f"execute_start entry {start['seq']} {start['entry_sha256']}")
    try:
        rep = execute_parallel(runner, gpus=gpus, factory=FACTORY_PRODUCTION, factory_kwargs={},
                               allow_production=True, frozen=frozen)
    except ParallelSchedulingError as exc:
        chain.append(chain.chain_path(out), _stamp({"kind": "execute_end", "status": "STOPPED",
                                                    "reason": str(exc)}, rec, seen))
        print(f"STOPPED (resumable): {exc}")
        return 1
    chain.append(chain.chain_path(out), _stamp({"kind": "execute_end", "status": "COMPLETE",
                                                "n_blocks_valid": rep["n_blocks_valid"]}, rec, seen))
    print(json.dumps({"n_blocks_valid": rep["n_blocks_valid"], "analysis": rep["analysis"],
                      "interpretation": rep["interpretation"]}, indent=1, default=str))
    print(f"\nNext: python -m audit.implementation.ii4_par2.launcher --verify --output-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
