"""
Full-size concurrent host-memory probe for PAR2. Not evidence; trains nothing that is kept.

WHY. The frozen requirement (8.0 GB free, from a 4.0 GB single-process peak
estimate) was set for a runner that trained one arm at a time. PAR1 runs two
workers at once, and its amendment never revisited memory. This probe
measures the concurrent peak instead of assuming it.

WHAT. One child per GPU (CUDA_VISIBLE_DEVICES set as the PAR1 scheduler sets
it), started together. Each child runs the FROZEN adapter's own code path at
production scale: generate_block_datasets (10^7 training and 10^6 validation
rows), load_block_datasets, then build_arm_model and train_arm for both arms
with one epoch, as a worker does for one block. Probe seeds are fixed here,
recorded, and unrelated to the experiment's seed manifest. The probe's data
directory is deleted afterwards.

RECORDED. Each child's VmHWM (kernel high-water mark) and the parent's 1 s
samples of each child's VmRSS; cgroup memory.peak before and after; the
sampled peak of memory.current; memory.events and memory.pressure before and
after; the corrected measurement before and after.

PASS iff both children exit 0 having trained both arms at scale 1.0, and no
cgroup level's oom or oom_kill count increased. The peaks are reported against
the frozen estimates; they do not change the requirement.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from audit.implementation.ii4_par2 import memory

PROBE_SEED_BASE = 20261106                      # non-evidentiary; not in the experiment's seed manifest
FROZEN_PEAK_ESTIMATE_GB = 4.0                   # DESIGN_RATIONALE memory_measured.peak_host_gb_estimate


def _status(pid="self") -> dict[str, Optional[int]]:
    out: dict[str, Optional[int]] = {"VmHWM": None, "VmRSS": None}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in out:
                out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def child(gpu: str, workdir: Path, seed: int, result: Path, scale: float) -> int:
    from audit.implementation.adapters.gohr_impl_comparison import (
        ARM_INDEPENDENT, ARM_REFERENCE, GohrImplementationComparisonAdapter)
    rec: dict[str, Any] = {"gpu": gpu, "scale": scale, "seed": seed, "arms_completed": [],
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "phases": []}
    t0 = time.time()
    try:
        adapter = GohrImplementationComparisonAdapter(scale=scale, epochs=1)
        rec["n_train"], rec["n_val"] = adapter.n_train, adapter.n_val
        ds = adapter.generate_block_datasets(f"memprobe_gpu{gpu}", workdir / "data")
        rec["phases"].append({"phase": "generated", "t": time.time() - t0, **_status()})
        data = adapter.load_block_datasets(ds)
        rec["phases"].append({"phase": "loaded", "t": time.time() - t0, **_status()})
        for i, arm in enumerate((ARM_REFERENCE, ARM_INDEPENDENT)):
            model = adapter.build_arm_model(arm, seed=seed + i)
            (workdir / arm).mkdir(parents=True, exist_ok=True)
            adapter.train_arm(arm, model, data, arm_dir=workdir / arm, seed=seed + i, session_id="memprobe")
            rec["arms_completed"].append(arm)
            rec["phases"].append({"phase": f"trained_{arm}", "t": time.time() - t0, **_status()})
            del model
            gc.collect()
        rec["ok"] = True
    except Exception as exc:                                              # noqa: BLE001
        rec["ok"], rec["error"] = False, f"{type(exc).__name__}: {exc}"
    rec["seconds"] = time.time() - t0
    rec.update({f"final_{k}": v for k, v in _status().items()})
    result.write_text(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 1


def _oom_counts(m: dict) -> dict[str, tuple[int, int]]:
    return {lv["path"]: ((lv.get("events") or {}).get("oom", 0), (lv.get("events") or {}).get("oom_kill", 0))
            for lv in m.get("cgroup_levels", [])}


def summarise(before: dict, after: dict, children: dict[str, dict], samples: dict[str, Any],
              required_bytes: int) -> dict[str, Any]:
    """Pure: the probe entry and its PASS/FAIL from the recorded observations."""
    reasons = []
    for g, c in sorted(children.items()):
        if c.get("returncode") != 0 or not c.get("ok"):
            reasons.append(f"GPU {g} child failed: rc={c.get('returncode')} {c.get('error', '')}".strip())
        if len(c.get("arms_completed") or []) != 2:
            reasons.append(f"GPU {g} child trained {c.get('arms_completed')}, not both arms")
        if c.get("scale") != 1.0:
            reasons.append(f"GPU {g} child ran at scale {c.get('scale')}, not full size")
    if len(children) != 2:
        reasons.append(f"{len(children)} children, not one per GPU for two GPUs")
    ob, oa = _oom_counts(before), _oom_counts(after)
    for path, (o, k) in oa.items():
        o0, k0 = ob.get(path, (0, 0))
        if o > o0 or k > k0:
            reasons.append(f"cgroup {path}: oom {o0}->{o}, oom_kill {k0}->{k} during the probe")
    hwm = {g: c.get("final_VmHWM") for g, c in children.items()}
    known = [v for v in hwm.values() if v is not None]
    total = sum(known) if len(known) == len(hwm) and hwm else None

    def peak(m):
        return {lv["path"]: lv.get("memory_peak") for lv in m.get("cgroup_levels", [])}

    return {
        "kind": "memprobe",
        "status": "PASS" if not reasons else "FAIL",
        "fail_reasons": reasons,
        "full_size": all(c.get("scale") == 1.0 for c in children.values()) and bool(children),
        "children": children,
        "child_vmhwm_bytes": hwm,
        "sum_child_vmhwm_bytes": total,
        "sum_child_vmhwm_gb": total / 1e9 if total is not None else None,
        "sampled_peak_rss_bytes": samples.get("peak_rss"),
        "sampled_peak_cgroup_current_bytes": samples.get("peak_current"),
        "n_samples": samples.get("n"),
        "memory_peak_before": peak(before),
        "memory_peak_after": peak(after),
        "events_before": {lv["path"]: lv.get("events") for lv in before.get("cgroup_levels", [])},
        "events_after": {lv["path"]: lv.get("events") for lv in after.get("cgroup_levels", [])},
        "pressure_before": {lv["path"]: lv.get("pressure") for lv in before.get("cgroup_levels", [])},
        "pressure_after": {lv["path"]: lv.get("pressure") for lv in after.get("cgroup_levels", [])},
        "corrected_memory_before": before,
        "corrected_memory_after": after,
        "against_frozen_estimates": {
            "frozen_peak_estimate_per_process_gb": FROZEN_PEAK_ESTIMATE_GB,
            "two_processes_estimate_gb": 2 * FROZEN_PEAK_ESTIMATE_GB,
            "requirement_gb": required_bytes / 1e9,
            "measured_sum_child_peak_gb": total / 1e9 if total is not None else None,
            "requirement_changed": False,
        },
        "boot_id": before.get("boot_id"),
    }


def run_probe(*, workdir: Path, gpus: tuple[str, ...], required_bytes: int, scale: float = 1.0,
              python: str = sys.executable, interval: float = 1.0) -> dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=False)
    before = memory.measure(required_bytes=required_bytes)
    procs = {}
    for i, g in enumerate(gpus):
        d = workdir / f"gpu{g}"
        d.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [python, "-B", "-m", "audit.implementation.ii4_par2.memprobe", "--child", "--gpu", str(g),
               "--workdir", str(d), "--seed", str(PROBE_SEED_BASE + 10 * i), "--scale", str(scale),
               "--result", str(d / "result.json")]
        log = open(workdir / f"child_gpu{g}.log", "w")
        procs[str(g)] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, d)
    peak_rss = {g: 0 for g in procs}
    peak_current, n = 0, 0
    own = None
    while any(p.poll() is None for p, _, _ in procs.values()):
        for g, (p, _, _) in procs.items():
            r = _status(p.pid)["VmRSS"]
            if r:
                peak_rss[g] = max(peak_rss[g], r)
        raw = memory.read_sources()
        own = next((lv for lv in raw["levels"] if lv.get("visible")), None)
        if own and own["files"].get("memory.current"):
            try:
                peak_current = max(peak_current, memory.parse_int(own["files"]["memory.current"]))
            except memory.Malformed:
                pass
        n += 1
        time.sleep(interval)
    children = {}
    for g, (p, log, d) in procs.items():
        log.close()
        rpath = d / "result.json"
        c = json.loads(rpath.read_text()) if rpath.exists() else {"ok": False, "error": "no result written"}
        c["returncode"] = p.returncode
        c["log_tail"] = (workdir / f"child_gpu{g}.log").read_text()[-2000:]
        children[g] = c
    after = memory.measure(required_bytes=required_bytes)
    entry = summarise(before, after, children,
                      {"peak_rss": peak_rss, "peak_current": peak_current or None, "n": n}, required_bytes)
    entry["probe_seeds"] = {g: PROBE_SEED_BASE + 10 * i for i, g in enumerate(gpus)}
    shutil.rmtree(workdir)
    entry["workdir_removed"] = not workdir.exists()
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--gpu")
    ap.add_argument("--workdir", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--result", type=Path)
    a = ap.parse_args(argv)
    if not a.child:
        ap.error("run the probe through audit.implementation.ii4_par2.launcher --memprobe")
    return child(a.gpu, a.workdir, a.seed, a.result, a.scale)


if __name__ == "__main__":
    sys.exit(main())
