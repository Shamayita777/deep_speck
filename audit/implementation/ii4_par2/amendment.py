"""
The PAR2 amendment's identity: its sources, its frozen record, and what it leaves unchanged.

PAR2 is gate-only. It replaces the frozen preflight's host-memory measurement
inside the readiness gate and adds a full-size concurrent memory probe. It
does not touch the PAR1 scheduler, worker, runner, adapter, plan, seeds,
design or decision rules, so the PAR1 experiment fingerprint is unchanged and
PAR2 is bound by its own record rather than by that fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit.implementation.ii4_par2.memory import MEASUREMENT_VERSION, PHI, SENSITIVITY_PHIS

PAR2_ID = "II-4-GOHR-IMPL-V1-PAR2"
PAR2_VERSION = "ii4-par2-gate-1"
REPO_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_FROZEN = Path("audit/implementation/preregistrations/ii-4-gohr-impl-v1.frozen.json")
PAR1_FROZEN = Path("audit/implementation/preregistrations/ii-4-gohr-impl-v1-par1.frozen.json")
PAR2_FROZEN = Path("audit/implementation/preregistrations/ii-4-gohr-impl-v1-par2.frozen.json")

PAR2_SOURCES = {
    "par2_package": "audit/implementation/ii4_par2/__init__.py",
    "par2_amendment": "audit/implementation/ii4_par2/amendment.py",
    "par2_memory": "audit/implementation/ii4_par2/memory.py",
    "par2_chain": "audit/implementation/ii4_par2/chain.py",
    "par2_gate": "audit/implementation/ii4_par2/gate.py",
    "par2_memprobe": "audit/implementation/ii4_par2/memprobe.py",
    "par2_launcher": "audit/implementation/ii4_par2/launcher.py",
    "par2_verifier": "audit/implementation/ii4_par2/verify_par2.py",
    "par2_document": "audit/implementation/ii4_par2/AMENDMENT.md",
    "par2_tests_package": "audit/implementation/ii4_par2/tests/__init__.py",
    "par2_test_memory": "audit/implementation/ii4_par2/tests/test_memory.py",
    "par2_test_gate": "audit/implementation/ii4_par2/tests/test_gate.py",
    "par2_test_chain_verify": "audit/implementation/ii4_par2/tests/test_chain_verify.py",
    "par2_test_memprobe": "audit/implementation/ii4_par2/tests/test_memprobe.py",
}

#: The frozen requirement, cited rather than restated: DESIGN_RATIONALE in the frozen preregistration.
THRESHOLD_SOURCE = ("audit/implementation/preregistrations/ii4_impl_gohr_v1.py: "
                    "DESIGN_RATIONALE['compute_estimate']['memory_measured']['recommended_free_host_gb']")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    out = {}
    for label, rel in sorted(PAR2_SOURCES.items()):
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"PAR2 source absent: {rel}")
        out[label] = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def frozen_threshold_gb() -> float:
    from audit.implementation.preregistrations.ii4_impl_gohr_v1 import DESIGN_RATIONALE
    return float(DESIGN_RATIONALE["compute_estimate"]["memory_measured"]["recommended_free_host_gb"])


def build_record(root: Path = REPO_ROOT) -> dict[str, Any]:
    original = json.loads((root / ORIGINAL_FROZEN).read_text())
    par1 = json.loads((root / PAR1_FROZEN).read_text())
    from audit.implementation.preregistrations.ii4_impl_gohr_v1 import DESIGN_RATIONALE
    mm = DESIGN_RATIONALE["compute_estimate"]["memory_measured"]
    sources = source_manifest(root)
    return {
        "artifact": "ii4-par2-amendment-record-1",
        "amendment_id": PAR2_ID,
        "gate_version": PAR2_VERSION,
        "experiment_id": par1["experiment_id"],
        "amends": [
            {"record": str(ORIGINAL_FROZEN), "sha256": sha256_file(root / ORIGINAL_FROZEN)},
            {"record": str(PAR1_FROZEN), "sha256": sha256_file(root / PAR1_FROZEN),
             "amendment_id": par1["amendment_id"]},
        ],
        "scope": ("Pre-execution readiness only: the host-memory measurement inside the readiness gate, "
                  "plus a full-size concurrent memory probe that must pass in the same boot before "
                  "execution. No training, evaluation, seed, design, plan, scheduler, worker, runner or "
                  "decision-rule code is changed or replaced."),
        "defect": {
            "location": "audit/implementation/preflight.py memory_report (frozen, unchanged)",
            "measurement": "os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')",
            "meaning_on_linux_glibc": "sysinfo().freeram, i.e. MemFree: excludes reclaimable page cache "
                                      "and ignores the cgroup limit",
            "observed_on_kaggle_as_reported": {"preflight_available_gb": 0.24, "MemFree_MiB": 237,
                                               "MemAvailable_GiB": 29, "psutil_available_GiB": 29.85,
                                               "memory.max_GiB": 30, "memory.current_GiB": 25.88,
                                               "inactive_file_GiB": 19.08, "memory.pressure": "zero",
                                               "memory.events oom/max": 0},
        },
        "measurement": {
            "version": MEASUREMENT_VERSION,
            "formula": ("U(phi) = min(MemAvailable, min over visible cgroup levels with a finite limit of "
                        "max(0, min(memory.max, memory.high) - memory.current) + phi * max(0, inactive_file "
                        "- file_dirty - file_writeback - file_mapped))"),
            "phi": PHI,
            "phi_status": "conservative policy chosen before use, not estimated from data",
            "sensitivity_phis": list(SENSITIVITY_PHIS),
            "block_rules": ["MemAvailable absent or unparseable", "cgroup v1 memory controller",
                            "v2 cgroup named but no v2 hierarchy mounted",
                            "finite limit with memory.current unreadable", "any interface file unparseable",
                            "memory.events oom > 0 or oom_kill > 0", "memory.pressure full avg10/60/300 > 0",
                            "U(phi) below the requirement"],
            "no_credit_rules": ["memory.events max > 0 or high > 0", "memory.pressure some avg60 > 0",
                                "memory.stat, memory.events or memory.pressure unreadable"],
            "never_credited": ["active_file", "slab_reclaimable", "swap"],
        },
        "threshold": {"source": THRESHOLD_SOURCE, "recommended_free_host_gb": mm["recommended_free_host_gb"],
                      "peak_host_gb_estimate": mm["peak_host_gb_estimate"], "unit": "GB = 1e9 bytes",
                      "changed": False},
        "concurrent_memory_probe": {
            "what": ("one child per GPU, concurrently, each running the frozen adapter's "
                     "generate_block_datasets, load_block_datasets, build_arm_model and train_arm for both "
                     "arms at scale 1.0 with one epoch; non-evidentiary probe seeds; data deleted afterwards"),
            "records": ["each child's VmHWM and sampled VmRSS peak", "cgroup memory.peak before and after",
                        "sampled peak memory.current", "memory.events and memory.pressure before and after",
                        "the corrected measurement before and after"],
            "pass_rule": ("both children exit 0 having trained both arms at scale 1.0, and no cgroup level's "
                          "oom or oom_kill count increased"),
            "required_before_execute": "a PASS probe entry with the same boot_id as the execute entry",
        },
        "unchanged": {
            "config_fingerprint": par1["config_fingerprint"],
            "par1_source_manifest_sha256": par1["source_manifest_sha256"],
            "preregistration_sha256": par1["preregistration_sha256"],
            "experiment_fingerprint_rule": "audit/implementation/ii4_parallel/scheduler.py "
                                           "ParallelComparisonRunner.runtime_provenance (PAR1)",
            "frozen_readiness_gate": "audit/implementation/run_ii4_impl.py readiness_gate, called unchanged; "
                                     "only its memory problems are superseded",
        },
        "par2_sources": sources,
        "par2_sources_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_record(root: Path = REPO_ROOT) -> Path:
    path = root / PAR2_FROZEN
    rec = build_record(root)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)     # never overwrites
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return path


def check_record(root: Path = REPO_ROOT) -> tuple[list[str], dict[str, Any]]:
    """Problems with the PAR2 record against the files on disk; empty when it binds them exactly."""
    path = root / PAR2_FROZEN
    if not path.is_file():
        return [f"no PAR2 record at {PAR2_FROZEN}"], {}
    rec = json.loads(path.read_text())
    problems = []
    for a in rec.get("amends", []):
        if sha256_file(root / a["record"]) != a["sha256"]:
            problems.append(f"amended record changed: {a['record']}")
    try:
        now = source_manifest(root)
    except FileNotFoundError as exc:
        return [str(exc)], rec
    if {k: v["sha256"] for k, v in now.items()} != {k: v["sha256"] for k, v in rec.get("par2_sources", {}).items()}:
        problems.append("PAR2 sources on disk differ from the PAR2 record")
    if rec.get("threshold", {}).get("recommended_free_host_gb") != frozen_threshold_gb():
        problems.append("PAR2 record threshold differs from the frozen preregistration")
    return problems, {"record_sha256": sha256_file(path), **rec}
