#!/usr/bin/env python3
"""
Launcher for II-4-GOHR-IMPL-V1, the formal reference-versus-independent
implementation comparison.

Modes, in the order they must be used:

  --capture-architecture
        Build both arms, verify their ORDERED architecture signatures are
        identical, and commit the declared signature beside the adapter.
        Must be run before --freeze. If a declared signature already
        exists and differs, prints the field-level diff and refuses:
        changing the declared architecture is a scientific change and
        requires a new freeze, not an overwrite.

  --preflight
        Which device actually trains (observed, not inferred), how many
        devices the runner uses, and whether disk and memory clear the
        measured requirements. Reports honestly that the second GPU is
        unused and claims no speedup.

  --freeze
        Validate the plan and write the FROZEN RECORD: the configuration
        fingerprint, the preregistration hash, the declared architecture
        digest, and the sha256 of every source file that can influence a
        result. Production is bound to this record, so a source change
        after review requires a new freeze.

  --dry-run
        Full validation plus the experiment fingerprint. Writes nothing,
        trains nothing.

  --smoke [--interrupt-after-epoch N]
        A NON-EVIDENTIARY end-to-end run at reduced scale. With
        --interrupt-after-epoch it deliberately kills one arm after a
        completed epoch so the resume path can be exercised against real
        Keras; re-run without the flag to resume and finish. A smoke
        artifact can never license production evidence.

  --execute
        The real run. Requires the frozen record, and refuses if the
        configuration, the preregistration, the declared architecture or
        ANY source file has changed since the freeze. Resumable: re-run
        the same command after an interruption and completed arms and
        epochs are reused rather than repeated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit.common.outcomes import ExecutionMode
from audit.implementation.adapters.gohr_impl_comparison import (
    ARM_INDEPENDENT,
    ARM_REFERENCE,
    DECLARED_ARCHITECTURE_PATH,
    EXPECTED_COUNTS,
    GohrImplementationComparisonAdapter,
    load_declared_architecture,
)
from audit.implementation.architecture_signature import (
    SIGNATURE_VERSION,
    architecture_signature,
    diff_signatures,
)
from audit.implementation.arm_recovery import atomic_write_json
from audit.implementation.ii4_comparison_design import ComparisonDesign
from audit.implementation.ii4_comparison_runner import (
    ComparisonExecutionError,
    ComparisonRunner,
    check_output_dir,
)
from audit.implementation.preflight import format_report, preflight
from audit.implementation.preregistrations.ii4_impl_gohr_v1 import (
    DESIGN,
    DESIGN_RATIONALE,
    EXPERIMENT_ID,
    build_plan,
    freeze as print_freeze,
)

DEFAULT_OUTPUT = Path("audit/implementation/runs") / EXPERIMENT_ID.lower()
FROZEN_RECORD_PATH = (Path("audit/implementation/preregistrations")
                      / f"{EXPERIMENT_ID.lower()}.frozen.json")

_STORAGE = DESIGN_RATIONALE["compute_estimate"]["storage_measured"]
_MEMORY = DESIGN_RATIONALE["compute_estimate"]["memory_measured"]
_CKPT = DESIGN_RATIONALE["compute_estimate"]["checkpoint_overhead_measured"]
_COMPUTE = DESIGN_RATIONALE["compute_estimate"]


REQUIRED_SUITES = (
    "audit.tests.test_durable_state",
    "audit.tests.test_firewall_real_integration",
    # Kills the process inside the firewall's own persistence windows.
    # Required because the exactly-once sealed-test guarantee rests on
    # that file surviving a crash: if it can be left unreadable, a
    # resumed run would either lose the record of a consumed evaluation
    # or refuse to continue, and both are launch-blocking.
    "audit.tests.test_firewall_crash_recovery",
    "audit.tests.test_arm_recovery",
    "audit.tests.test_arm_recovery_keras",
    "audit.tests.test_freeze_manifest_coverage",
    "audit.tests.test_ii4_interruption",
    "audit.tests.test_ii4_comparison",
    # The launcher's own gates. The filesystem suite evidences that an
    # output location whose rename and fsync semantics are unverified is
    # refused; the execute suite evidences that --execute cannot start
    # production with a failing or skipped test gate, an unclear
    # preflight or an unacceptable output directory. Both are gates on
    # this gate, so they belong inside it.
    "audit.tests.test_preflight_filesystem",
    "audit.tests.test_execute_gate",
    # An unrecoverable arm must be preserved for forensics, never reset:
    # the refusal path is evidence handling, so it is gated.
    "audit.tests.test_unrecoverable_state",
    # The exactly-once invariant on the sealed-test permit, audited
    # across the journal/firewall conflict matrix and as an ordering
    # property of the source.
    "audit.tests.test_journal_exactly_once",
    # Repeated interruption of ONE arm. Every other crash suite
    # interrupts once, which cannot see damage that accumulates.
    "audit.tests.test_repeated_interruption",
    # Meta: the suite must not strand tests after its entry point
    # and no test module may sit outside this gate unexplained.
    "audit.tests.test_suite_integrity",
)


def verify_required_suites() -> tuple[bool, dict]:
    """
    Run the suites that EVIDENCE the recovery and structural claims, and
    treat a SKIP as a failure.

    A skipped test proves nothing. `test_arm_recovery_keras` skips
    itself when Keras is absent, which is right for a developer machine
    and wrong for a production-readiness gate: the recovery exactness
    claim is a claim about the stack the run will actually use, so if it
    was not measured HERE it has not been measured. A skip therefore
    blocks launch exactly as a failure does.
    """
    import io
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromNames(REQUIRED_SUITES))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    summary = {
        "suites": list(REQUIRED_SUITES),
        "run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "skipped_detail": [f"{t}: {why}" for t, why in result.skipped],
        "failure_detail": [f"{t}" for t, _ in result.failures + result.errors],
    }
    ok = (result.testsRun > 0 and not result.failures and not result.errors
          and not result.skipped)
    summary["clear"] = ok
    return ok, summary


def run_preflight(output_dir, *, acknowledged: tuple[str, ...] = (),
                  mounts=None) -> dict:
    """
    Preflight for the EXACT directory the run will write to.

    Separated out so `--preflight`, `--dry-run` and `--execute` cannot
    diverge: all three measure the same path with the same thresholds,
    and the only difference between them is what they do with the
    answer.
    """
    out = Path(output_dir)
    return preflight(
        output_path=out.parent if not out.exists() else out,
        disk_required_gb=_STORAGE["recommended_free_gb"],
        memory_required_gb=_MEMORY["recommended_free_host_gb"],
        acknowledged=acknowledged,
        mounts=mounts,
        # Itemised so a reader can check the arithmetic rather than
        # trust a single number. Every component the run writes is
        # named, including the inflight temporaries and the logs, and
        # the margin is its own line.
        disk_items=[
            ("block datasets, 9 blocks compressed",
             _STORAGE["nine_blocks_gb"]),
            ("sealed global test set", _STORAGE["sealed_test_gb"]),
            ("rotating checkpoint slots (2 slots x 2 arms x 9 blocks)",
             round(_CKPT["disk_two_slots_two_arms_nine_blocks_mb"] / 1024, 3)),
            ("best-validation and terminal checkpoints",
             round((_CKPT["disk_all_checkpoints_mb"]
                    - _CKPT["disk_two_slots_two_arms_nine_blocks_mb"]) / 1024, 3)),
            ("inflight temporaries (one slot per arm in flight)",
             round(_CKPT["slot_size_mb"] * 2 / 1024, 4)),
            ("epoch logs, corruption logs, manifests, state and analysis JSON",
             0.01),
            ("safety margin to the recommended free space",
             round(_STORAGE["recommended_free_gb"]
                   - _STORAGE["nine_blocks_gb"] - _STORAGE["sealed_test_gb"]
                   - _CKPT["disk_all_checkpoints_mb"] / 1024
                   - _CKPT["slot_size_mb"] * 2 / 1024 - 0.01, 3)),
        ],
        # GPU-hours are not wall-clock, and both figures are estimates
        # extrapolated from a previous experiment on specific hardware.
        # Stated here so the preflight cannot be read as a schedule.
        estimate_notes={
            "unit": "GPU-hours of compute, NOT wall-clock time. Data generation, "
                    "evaluation, checkpointing and any queueing or preemption on the "
                    "host are additional.",
            "K=7 (planned, 7 blocks)":
                f"{_COMPUTE['planned_K7']['gpu_hours_low']}-"
                f"{_COMPUTE['planned_K7']['gpu_hours_high']} GPU-h "
                f"(midpoint {_COMPUTE['planned_K7']['midpoint']})",
            "K=9 (worst case, both pre-declared replacement blocks used)":
                f"{_COMPUTE['worst_case_K9_both_replacements_used']['gpu_hours_low']}-"
                f"{_COMPUTE['worst_case_K9_both_replacements_used']['gpu_hours_high']} "
                f"GPU-h (midpoint "
                f"{_COMPUTE['worst_case_K9_both_replacements_used']['midpoint']})",
            "assumption behind both": _COMPUTE["basis"],
            "per block": _COMPUTE["per_block_hours"],
            "checkpoint overhead": _CKPT["share_of_total_runtime"],
            "status": "AN ESTIMATE. It is an extrapolation from a different experiment "
                      "on 2x Tesla T4 and is not a guarantee. A slower GPU, a different "
                      "driver, host contention or a single failed block all change it.",
        },
        extra={
            "block data (measured)": f"{_STORAGE['per_block_compressed_gb']} GB/block, "
                                     f"{_STORAGE['nine_blocks_gb']} GB for 9 blocks",
            "checkpoints": f"{_STORAGE['checkpoints_gb']} GB",
            "declared architecture": (
                (load_declared_architecture() or {}).get("signature", {}).get("digest")
                or "NOT CAPTURED - run --capture-architecture"),
            "frozen record": ("present" if FROZEN_RECORD_PATH.is_file()
                              else "ABSENT - run --freeze"),
        })


def acknowledgements_from(args) -> tuple[str, ...]:
    """
    The one place the idle-GPU acknowledgement is translated from the
    CLI, so `--preflight` and `--execute` cannot interpret the same flag
    differently. A flag that silences a warning in one mode and not in
    another is worse than no flag.
    """
    return ("idle_gpus",) if getattr(args, "ack_idle", False) else ()


def readiness_gate(output_dir, *, acknowledged: tuple[str, ...] = (),
                   mounts=None) -> tuple[bool, dict]:
    """
    Everything that must hold before production training starts, in one
    place, returning a refusal rather than a warning.

    WHY THIS EXISTS. Until now `--execute` checked only that a frozen
    record existed and that the operator had passed
    `--i-have-reviewed-the-preregistration`. The test gate and the
    resource preflight ran in `--dry-run` only, so a run could be
    launched with failing or skipped required tests, on a filesystem
    that cannot support the recovery design, with no GPU, or with the
    disk too small to hold the blocks - and the first sign of it would
    be a dead run hours later. A gate that lives in a different mode
    from the one that launches the run is not a gate.

    The three parts are:
      output directory  nothing of a previous run is overwritten
      required suites   every suite passes with no failures and NO SKIPS
      preflight         GPU, memory, disk and FILESYSTEM for this exact
                        output path, on this machine, now
    """
    out = Path(output_dir)
    gate: dict = {"output_dir": str(out)}

    gate["dir_problems"] = check_output_dir(out)
    suites_ok, suites = verify_required_suites()
    gate["suites"] = suites
    report = run_preflight(out, acknowledged=acknowledged, mounts=mounts)
    gate["preflight"] = report

    gate["clear"] = bool(
        not gate["dir_problems"] and suites_ok and report["clear_to_launch"])
    return gate["clear"], gate


def format_readiness_gate(gate: dict) -> str:
    """The gate's result, with the reason for every refusal."""
    suites = gate["suites"]
    report = gate["preflight"]
    lines = [
        "READINESS GATE",
        f"  output dir             {gate['output_dir']}",
        f"  output dir acceptable  {not gate['dir_problems']}"
        + (f" ({gate['dir_problems']})" if gate["dir_problems"] else ""),
        f"  required suites        {suites['run']} run, {suites['failures']} failed, "
        f"{suites['errors']} errors, {suites['skipped']} skipped -> "
        f"{'CLEAR' if suites['clear'] else 'NOT CLEAR'}",
    ]
    for d in suites["skipped_detail"]:
        lines.append(f"    SKIPPED (counts as a failure): {d}")
    for d in suites["failure_detail"]:
        lines.append(f"    FAILED: {d}")
    if not suites["clear"]:
        lines.append("    A skipped required test proves nothing. The recovery exactness")
        lines.append("    claim is about the stack THIS run will use, so it must be")
        lines.append("    measured here.")
    fs = report.get("filesystem") or {}
    lines.append(f"  filesystem             {fs.get('fstype', 'UNIDENTIFIED')} on "
                 f"{fs.get('mount_point', '?')} -> {fs.get('verdict', 'UNKNOWN')}")
    lines.append(f"  preflight              "
                 f"{'CLEAR' if report['clear_to_launch'] else 'NOT CLEAR'}")
    for p in report["problems"]:
        lines.append(f"    BLOCKER: {p}")
    for n in report.get("unacknowledged", []):
        lines.append(f"    AWAITING ACKNOWLEDGEMENT: {n} "
                     f"(pass --acknowledge-idle-gpus)")
    lines.append(f"  GATE                   {'CLEAR' if gate['clear'] else 'REFUSED'}")
    return "\n".join(lines)


def _flat(report):
    from audit.implementation.ii4_comparison_certificate import read_analysis
    return read_analysis(report["analysis"])


def _smoke_design() -> ComparisonDesign:
    """Tiny, fast, and deliberately NOT the frozen design."""
    import dataclasses
    return dataclasses.replace(DESIGN, n_blocks=2, min_valid_blocks=2,
                               max_replacement_blocks=1, epochs=4)


# =====================================================================
# --capture-architecture
# =====================================================================

def capture_architecture(adapter) -> int:
    from audit.implementation.adapters.gohr_impl_comparison import (
        load_independent, load_reference,
    )
    print("Building both arms to capture the declared architecture...\n")
    _, train_nets = load_reference()
    ref = train_nets.make_resnet(depth=10, reg_param=1e-5)
    build_gohr_model, _ = load_independent()
    ind = build_gohr_model(depth=10, filters=32, dense_1=64, dense_2=64, kernel_size=3,
                           word_size=16, input_words=4, l2_strength=1e-5)

    sig_ref = architecture_signature(ref)
    sig_ind = architecture_signature(ind)
    print(f"  reference    {sig_ref['layer_count']} layers, "
          f"{sig_ref['total_parameters']:,} params  digest {sig_ref['digest']}")
    print(f"  independent  {sig_ind['layer_count']} layers, "
          f"{sig_ind['total_parameters']:,} params  digest {sig_ind['digest']}")

    if sig_ref["digest"] != sig_ind["digest"]:
        print("\nREFUSED: the two implementations realize DIFFERENT architectures. An "
              "implementation comparison presupposes a common architecture; otherwise the "
              "measured difference confounds implementation with architecture.\n")
        for line in diff_signatures(sig_ref, sig_ind, label_a="reference",
                                    label_b="independent"):
            print(f"    {line}")
        return 1
    print("\n  ordered signatures IDENTICAL -- the comparison premise holds")

    env = adapter.environment_details()
    payload = {
        "artifact": "gohr-declared-architecture-v1",
        "signature_version": SIGNATURE_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "signature": sig_ref,
        "arms_agree": True,
        "coarse_counts_cross_check": EXPECTED_COUNTS,
        "captured_with": {"tensorflow": env.get("tensorflow"), "keras": env.get("keras")},
        "builders": {
            ARM_REFERENCE: "reference/train_nets.py make_resnet(depth=10, reg_param=1e-5)",
            ARM_INDEPENDENT: ("ii2_independent/model_independent.py build_gohr_model("
                              "depth=10, l2_strength=1e-5)"),
        },
    }

    existing = load_declared_architecture()
    if existing is not None:
        if existing["signature"]["digest"] == sig_ref["digest"]:
            print(f"\n  declared architecture unchanged ({DECLARED_ARCHITECTURE_PATH})")
            return 0
        print("\nREFUSED: a declared architecture already exists and DIFFERS from what the "
              "sources now build. This is a change of declared architecture, not a "
              "re-capture.\n")
        for line in diff_signatures(existing["signature"], sig_ref,
                                    label_a="committed", label_b="current"):
            print(f"    {line}")
        print(f"\n    committed with tensorflow={existing.get('captured_with', {}).get('tensorflow')} "
              f"keras={existing.get('captured_with', {}).get('keras')}")
        print(f"    current   tensorflow={env.get('tensorflow')} keras={env.get('keras')}")
        print("\n  If every difference above is a serialization detail of a different Keras "
              "version, delete the committed file, re-capture, and RE-FREEZE. If any "
              "difference is a real configuration change, the architecture has changed and "
              "the comparison premise must be re-established before freezing.")
        return 1

    atomic_write_json(DECLARED_ARCHITECTURE_PATH, payload)
    print(f"\n  wrote {DECLARED_ARCHITECTURE_PATH}")
    print(f"  declared digest {sig_ref['digest']}")
    print("\nCommit this file, then run --freeze.")
    return 0


# =====================================================================
# --freeze
# =====================================================================

def write_frozen_record(plan, adapter) -> int:
    rc = print_freeze()
    if rc != 0:
        return rc
    declared = load_declared_architecture()
    if declared is None:
        print("\nREFUSED: no declared architecture has been captured. Run "
              "--capture-architecture first; a production run may not train against an "
              "undeclared architecture.")
        return 1

    runner = ComparisonRunner(plan=plan, adapter=adapter)
    prov = runner.runtime_provenance()
    record = {
        "artifact": "ii4-frozen-record-v1",
        "experiment_id": EXPERIMENT_ID,
        "config_fingerprint": prov["config_fingerprint"],
        "preregistration_sha256": plan.preregistration_hash(),
        "declared_architecture_digest": declared["signature"]["digest"],
        "source_manifest": prov["source_manifest"],
        "source_manifest_sha256": prov["source_manifest_sha256"],
        "frozen_environment": prov["software_stack"],
        "binding_note": (
            "Production is bound to the configuration AND the source code recorded here. The "
            "runner refuses to execute if the configuration, the preregistration, the "
            "declared architecture, or ANY source file below has changed since this record "
            "was written. The software stack is RECORDED but not frozen: it legitimately "
            "varies between machines and is captured in each run's provenance."),
    }
    atomic_write_json(FROZEN_RECORD_PATH, record)
    print(f"\n  source files frozen   {len(prov['source_manifest'])}")
    print(f"  source manifest hash  {prov['source_manifest_sha256']}")
    print(f"  declared architecture {declared['signature']['digest']}")
    print(f"\n  wrote {FROZEN_RECORD_PATH}")
    print("\nCommit the frozen record. --execute reads it and refuses on any drift.")
    return 0


def load_frozen_record() -> dict | None:
    if not FROZEN_RECORD_PATH.is_file():
        return None
    return json.loads(FROZEN_RECORD_PATH.read_text())


# =====================================================================
# main
# =====================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-architecture", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--i-have-reviewed-the-preregistration", action="store_true",
                    dest="reviewed")
    ap.add_argument("--smoke-scale", type=float, default=0.0005)
    ap.add_argument("--interrupt-after-epoch", type=int, default=None,
                    help="SMOKE ONLY: kill one arm after this completed epoch")
    ap.add_argument("--acknowledge-idle-gpus", action="store_true", dest="ack_idle",
                    help="acknowledge that extra GPUs idle; required on a multi-GPU host")
    ap.add_argument("--verbose-training", type=int, default=0)
    ap.add_argument("--session-id", type=str, default=None)
    args = ap.parse_args(argv)

    out = args.output_dir or DEFAULT_OUTPUT

    if args.capture_architecture:
        return capture_architecture(GohrImplementationComparisonAdapter())

    if args.preflight:
        report = run_preflight(out, acknowledged=acknowledgements_from(args))
        print(format_report(report))
        return 0 if report["clear_to_launch"] else 1

    if args.smoke:
        sdir = args.output_dir or (Path("/tmp") / f"{EXPERIMENT_ID.lower()}-smoke")
        plan = build_plan(sdir, mode=ExecutionMode.SMOKE,
                          experiment_id=f"{EXPERIMENT_ID}-SMOKE", design=_smoke_design())
        adapter = GohrImplementationComparisonAdapter(
            scale=args.smoke_scale, epochs=plan.design.epochs,
            verbose=args.verbose_training)
        print(f"SMOKE (non-evidentiary): scale={args.smoke_scale} "
              f"epochs={plan.design.epochs} blocks={plan.design.n_blocks} -> {sdir}")
        if args.interrupt_after_epoch is not None:
            print(f"  FAULT INJECTION: one arm will be killed after epoch "
                  f"{args.interrupt_after_epoch}; re-run without the flag to resume.")
        print()
        runner = ComparisonRunner(
            plan=plan, adapter=adapter,
            interrupt_after_epoch=args.interrupt_after_epoch,
            **({"session_id": args.session_id} if args.session_id else {}))
        try:
            report = runner.run(execute=True)
        except Exception as exc:                                 # noqa: BLE001
            print(f"SMOKE INTERRUPTED: {type(exc).__name__}: {exc}")
            print("\nPer-arm epoch logs are on disk; re-run without --interrupt-after-epoch "
                  "to resume.")
            return 2
        print(json.dumps({
            "experiment_id": report["experiment_id"],
            "non_evidentiary": report["non_evidentiary"],
            "n_blocks_valid": report["n_blocks_valid"],
            "blocks": [{"block_id": b["block_id"], "is_valid": b["is_valid"],
                        "values": b["values"], "signed_difference": b["signed_difference"]}
                       for b in report["blocks"]],
            "epoch_logs": report["epoch_logs"],
            "interrupted_arms": report["interrupted_arms"],
            "materiality": _flat(report)["materiality"],
            "equivalence": _flat(report)["equivalence"],
            "interpretation": (report["interpretation"] or {}).get("conclusion"),
        }, indent=2, default=str))
        print("\nSMOKE COMPLETE. This artifact is NOT evidence.")
        return 0

    plan = build_plan(out, mode=ExecutionMode.PRODUCTION)
    adapter = GohrImplementationComparisonAdapter(verbose=args.verbose_training)

    if args.freeze:
        return write_frozen_record(plan, adapter)

    if args.dry_run:
        gate_ok, gate = readiness_gate(out, acknowledged=acknowledgements_from(args))
        result = ComparisonRunner(plan=plan, adapter=adapter).run(execute=False)
        frozen = load_frozen_record()
        print(f"DRY RUN - {EXPERIMENT_ID}\n")
        print(f"  plan validation        {result['validation']}")
        print(f"  production scale       {adapter.is_production_scale}")
        print(f"  declared architecture  {adapter.expected_arm_structure(ARM_REFERENCE)}")
        print(f"  preregistration hash   {result['preregistration_sha256']}")
        print(f"  config fingerprint     {result['config_fingerprint']}")
        print(f"  source manifest hash   {result['source_manifest_sha256']}")
        print(f"  experiment fingerprint {result['experiment_fingerprint']}")
        env = adapter.environment_details()
        print(f"\n  tensorflow / keras     {env.get('tensorflow')} / {env.get('keras')}")
        print(f"  GPUs visible           {env.get('gpus')}")
        print()
        print(format_readiness_gate(gate))
        if not gate_ok:
            print("\n  --execute would REFUSE on this gate. Nothing was written.")
            return 1
        if frozen is None:
            print("\n  FROZEN RECORD ABSENT - run --freeze before --execute.")
            return 1
        problems = ComparisonRunner(plan=plan, adapter=adapter).check_frozen(frozen)
        if problems:
            print("\n  DRIFT SINCE THE FREEZE:")
            for p in problems:
                print(f"    - {p}")
            return 1
        print("\n  frozen record           MATCHES (configuration, preregistration, declared "
              "architecture and every source file)")
        print("\nNothing was written. To execute:\n"
              "  python3 -m audit.implementation.run_ii4_impl --execute \\\n"
              "      --i-have-reviewed-the-preregistration")
        return 0

    # --execute
    #
    # Every gate below must be satisfied HERE, in the mode that actually
    # starts training. A frozen record and a reviewed-preregistration
    # flag say that the DESIGN was settled; they say nothing about
    # whether the code still passes its tests or whether this machine
    # can finish the run. Both have to be checked where the run begins.
    frozen = load_frozen_record()
    if frozen is None:
        print(f"REFUSED: no frozen record at {FROZEN_RECORD_PATH}. Run --capture-architecture, "
              "then --freeze, and review the result first.")
        return 1
    if not args.reviewed:
        print("REFUSED: --execute requires --i-have-reviewed-the-preregistration. This run "
              "costs 53-84 GPU-hours and produces confirmatory evidence.")
        return 1
    if args.interrupt_after_epoch is not None:
        print("REFUSED: --interrupt-after-epoch is a smoke-only fault injector.")
        return 1

    gate_ok, gate = readiness_gate(out, acknowledged=acknowledgements_from(args))
    print(format_readiness_gate(gate))
    print()
    if not gate_ok:
        print("REFUSED: the readiness gate did not clear, so production training was not "
              "started. Nothing was written.")
        print("  Crash durability is not claimed for an unverified filesystem, and a "
              "skipped required test is treated as a failed one.")
        return 1

    try:
        report = ComparisonRunner(
            plan=plan, adapter=adapter,
            **({"session_id": args.session_id} if args.session_id else {}),
        ).run(execute=True, allow_production=True, frozen=frozen)
    except ComparisonExecutionError as exc:
        print(f"REFUSED / ABORTED: {exc}")
        return 1

    a = _flat(report)
    print(f"\n{EXPERIMENT_ID} COMPLETE\n")
    print(f"  valid blocks           {report['n_blocks_valid']} / {plan.design.n_blocks} "
          f"(min {plan.design.min_valid_blocks})")
    print(f"  replacements used      {len(report['replacements'])}")
    print(f"  interrupted arms       {len(report['interrupted_arms'])}")
    print(f"  mean signed difference {a['mean_difference']}  ({plan.axis.sign_convention})")
    print(f"  95% CI                 [{a['ci95_low']}, {a['ci95_high']}]")
    print(f"  materiality            {a['materiality']}")
    print(f"  equivalence            {a['equivalence']}")
    print(f"  sign-flip p            {a['sign_flip_p']}")
    print(f"  CONCLUSION             {(report['interpretation'] or {}).get('conclusion')}")
    print(f"\n  artifact               {out / 'analysis.json'}")
    print("\nNext: assemble the certificate with "
          "audit.implementation.ii4_comparison_certificate.build_stage_records,")
    print("then re-run the Implementation Integrity runner to issue V7.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
