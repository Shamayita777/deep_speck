#!/usr/bin/env python3
"""
Launcher for II-4-GOHR-IMPL-V1-PAR1, the block-level two-GPU amendment of the
frozen II-4-GOHR-IMPL-V1 comparison. Run from the repository root.

  --freeze      write the amendment's frozen record (refuses to overwrite). It
                checks that the plan's configuration fingerprint and
                preregistration hash equal the ORIGINAL frozen record's, i.e. that
                only scheduling changed, and binds the amended source manifest.
  --preflight   frozen readiness preflight plus a two-GPU visibility check.
  --dry-run     validation, frozen-record check and readiness gate; writes nothing.
  --smoke       two-GPU smoke at reduced scale into a temporary directory. Not evidence.
  --execute     production: needs the amendment record, both review flags, a clear
                readiness gate and exactly two visible GPUs.
  --verify      run the independent verifier on a finished run directory.

The original frozen record and runner are never modified or used to write here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from audit.common.outcomes import ExecutionMode
from audit.implementation.arm_recovery import atomic_write_json
from audit.implementation.ii4_parallel.scheduler import (
    AMENDMENT_ID,
    PARALLEL_SOURCES,
    PARALLEL_VERSION,
    ParallelComparisonRunner,
    ParallelSchedulingError,
    execute_parallel,
)
from audit.implementation.preregistrations.ii4_impl_gohr_v1 import EXPERIMENT_ID, build_plan
from audit.implementation.preflight import TRAINING_TOPOLOGY

ORIGINAL_FROZEN = Path("audit/implementation/preregistrations/ii-4-gohr-impl-v1.frozen.json")
AMENDMENT_FROZEN = Path("audit/implementation/preregistrations/ii-4-gohr-impl-v1-par1.frozen.json")
DEFAULT_OUTPUT = Path("audit/implementation/runs") / AMENDMENT_ID.lower()
FACTORY_PRODUCTION = "audit.implementation.ii4_parallel.launcher:production_factory"
FACTORY_SMOKE = "audit.implementation.ii4_parallel.launcher:smoke_factory"


def production_factory(root):
    from audit.implementation.adapters.gohr_impl_comparison import GohrImplementationComparisonAdapter
    return build_plan(Path(root), mode=ExecutionMode.PRODUCTION), GohrImplementationComparisonAdapter()


def smoke_factory(root, scale=0.0005):
    from audit.implementation.adapters.gohr_impl_comparison import GohrImplementationComparisonAdapter
    from audit.implementation.run_ii4_impl import _smoke_design
    plan = build_plan(Path(root), mode=ExecutionMode.SMOKE, experiment_id=f"{EXPERIMENT_ID}-SMOKE",
                      design=_smoke_design())
    return plan, GohrImplementationComparisonAdapter(scale=scale, epochs=plan.design.epochs)


def visible_gpus() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    r = subprocess.run([exe, "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze() -> int:
    if AMENDMENT_FROZEN.exists():
        print(f"REFUSED: {AMENDMENT_FROZEN} exists; an amendment record is never overwritten.")
        return 1
    original = json.loads(ORIGINAL_FROZEN.read_text())
    plan, adapter = production_factory(DEFAULT_OUTPUT)
    runner = ParallelComparisonRunner(plan=plan, adapter=adapter)
    prov = runner.runtime_provenance()
    problems = []
    if prov["config_fingerprint"] != original["config_fingerprint"]:
        problems.append("configuration fingerprint differs from the original frozen record")
    if plan.preregistration_hash() != original["preregistration_sha256"]:
        problems.append("preregistration hash differs from the original frozen record")
    orig_sources = {r["path"]: r["sha256"] for r in original["source_manifest"].values()}
    now_sources = {r["path"]: r["sha256"] for r in prov["source_manifest"].values()}
    changed = sorted(p for p in orig_sources if now_sources.get(p) != orig_sources[p])
    if changed:
        problems.append(f"original frozen source files changed: {changed}")
    added = sorted(set(now_sources) - set(orig_sources))
    if sorted(added) != sorted(PARALLEL_SOURCES.values()):
        problems.append(f"added sources {added} are not exactly the amendment's files")
    declared = adapter.expected_arm_structure(plan.axis.arm_ids[0])
    if declared != original["declared_architecture_digest"]:
        problems.append("declared architecture differs from the original frozen record")
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems))
        return 1
    record = {
        "artifact": "ii4-frozen-record-v1-amendment",
        "amendment_id": AMENDMENT_ID, "experiment_id": EXPERIMENT_ID, "scheduler": PARALLEL_VERSION,
        "amends": {"path": str(ORIGINAL_FROZEN), "sha256": _sha(ORIGINAL_FROZEN),
                   "source_manifest_sha256": original["source_manifest_sha256"]},
        "scope": ("scheduling only: block-level parallelism over two GPUs as named in the frozen "
                  "TRAINING_TOPOLOGY['legitimate_parallelism']; plan, seeds, data procedure, epochs, "
                  "evaluation and every decision rule unchanged (configuration fingerprint and "
                  "preregistration hash equal to the original record)"),
        "frozen_topology_note": TRAINING_TOPOLOGY,
        "config_fingerprint": prov["config_fingerprint"],
        "preregistration_sha256": plan.preregistration_hash(),
        "declared_architecture_digest": declared,
        "source_manifest": prov["source_manifest"],
        "source_manifest_sha256": prov["source_manifest_sha256"],
        "added_sources": added,
        "frozen_environment": prov["software_stack"],
        "binding_note": original["binding_note"],
    }
    atomic_write_json(AMENDMENT_FROZEN, record)
    print(f"wrote {AMENDMENT_FROZEN}\n  source files {len(prov['source_manifest'])} "
          f"({len(added)} added)\n  source manifest {prov['source_manifest_sha256']}\n"
          f"  config fingerprint {prov['config_fingerprint']} (unchanged)\n"
          f"  record sha256 {_sha(AMENDMENT_FROZEN)}")
    return 0


def _gate(out: Path):
    from audit.implementation.run_ii4_impl import format_readiness_gate, readiness_gate
    ok, gate = readiness_gate(out, acknowledged=("idle_gpus",))
    print(format_readiness_gate(gate))
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                        "audit/implementation/ii4_parallel/tests"], capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    amended_ok = r.returncode == 0 and "skipped" not in tail
    print(f"  amendment suite        {tail} -> {'CLEAR' if amended_ok else 'NOT CLEAR'}")
    return ok and amended_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--freeze", "--preflight", "--dry-run", "--smoke", "--execute", "--verify"):
        m.add_argument(flag, action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--smoke-scale", type=float, default=0.0005)
    ap.add_argument("--interrupt-after-epoch", type=int, default=None, help="smoke only")
    ap.add_argument("--i-have-reviewed-the-preregistration", action="store_true", dest="reviewed")
    ap.add_argument("--i-have-reviewed-the-amendment", action="store_true", dest="reviewed_amendment")
    a = ap.parse_args(argv)
    gpus = tuple(g for g in a.gpus.split(",") if g)
    out = a.output_dir or DEFAULT_OUTPUT

    if a.freeze:
        return freeze()
    if a.verify:
        from audit.implementation.ii4_parallel import verify
        args = ["--run-dir", str(out), "--require-dual-gpu"]
        if AMENDMENT_FROZEN.exists():
            args += ["--frozen", str(AMENDMENT_FROZEN)]
        return verify.main(args)

    seen = visible_gpus()
    print(f"GPUs visible: {len(seen)}")
    for line in seen:
        print(f"  {line}")

    if a.preflight:
        ok = _gate(out)
        if len(seen) < 2:
            print("NOT CLEAR: two GPUs are required for the amended scheduler.")
            ok = False
        return 0 if ok else 1

    if a.smoke:
        sdir = a.output_dir or Path(tempfile.mkdtemp(prefix="ii4_par1_smoke_"))
        plan, adapter = smoke_factory(sdir, a.smoke_scale)
        runner = ParallelComparisonRunner(plan=plan, adapter=adapter)
        extra = (("--interrupt-after-epoch", str(a.interrupt_after_epoch))
                 if a.interrupt_after_epoch is not None else ())
        print(f"SMOKE (non-evidentiary) -> {sdir} on GPUs {gpus}")
        try:
            rep = execute_parallel(runner, gpus=gpus, factory=FACTORY_SMOKE,
                                   factory_kwargs={"scale": a.smoke_scale}, worker_args=extra)
        except ParallelSchedulingError as exc:
            print(f"SMOKE STOPPED: {exc}")
            return 2
        print(json.dumps({"n_blocks_valid": rep["n_blocks_valid"],
                          "equivalence": (rep["analysis"].get("confirmatory") or {}).get("equivalence"),
                          "non_evidentiary": rep["non_evidentiary"]}, indent=1, default=str))
        print("SMOKE COMPLETE. Not evidence. Run --verify --output-dir", sdir)
        return 0

    plan, adapter = production_factory(out)
    runner = ParallelComparisonRunner(plan=plan, adapter=adapter)
    frozen = json.loads(AMENDMENT_FROZEN.read_text()) if AMENDMENT_FROZEN.exists() else None
    if frozen is None:
        print(f"REFUSED: no amendment record at {AMENDMENT_FROZEN}; run --freeze and review it.")
        return 1
    problems = runner.check_frozen(frozen)
    print("amendment record", "MATCHES" if not problems else "DIFFERS:\n  " + "\n  ".join(problems))
    gate_ok = _gate(out)
    if len(seen) < 2:
        print("NOT CLEAR: two GPUs are required.")
        gate_ok = False
    if a.dry_run:
        runner.run(execute=False)
        return 0 if gate_ok and not problems else 1
    if not (a.reviewed and a.reviewed_amendment):
        print("REFUSED: --execute needs --i-have-reviewed-the-preregistration and "
              "--i-have-reviewed-the-amendment.")
        return 1
    if problems or not gate_ok or len(gpus) != 2:
        print("REFUSED: the amendment record, readiness gate or GPU check did not clear.")
        return 1
    try:
        rep = execute_parallel(runner, gpus=gpus, factory=FACTORY_PRODUCTION, factory_kwargs={},
                               allow_production=True, frozen=frozen)
    except ParallelSchedulingError as exc:
        print(f"STOPPED (resumable): {exc}")
        return 1
    print(json.dumps({"n_blocks_valid": rep["n_blocks_valid"], "analysis": rep["analysis"],
                      "interpretation": rep["interpretation"]}, indent=1, default=str))
    print(f"\nNext: python -m audit.implementation.ii4_parallel.launcher --verify --output-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
