#!/usr/bin/env python3
"""
AutoND Implementation Integrity runner (case study 2).

Assembles II-1..II-5 for the AutoND case study and emits an
Implementation Integrity certificate through the GENERIC framework
entry point, audit.implementation.certificate.build_implementation_evidence.

WHAT THIS MODULE IS
    A target adapter plus a thin entry point. It contains no audit
    logic of its own: stage semantics, tiering, validation, the
    outcome decision and the evidence tag all come from
    audit.implementation.* and audit.common.*. Nothing in the generic
    framework was modified or extended to accommodate AutoND; the
    AutoND-specific material lives entirely under audit/targets/autond/.

WHAT IT DOES NOT DO
    It trains nothing, searches nothing, evaluates no checkpoint and
    recomputes no statistic. Every number it reports is read from an
    artifact that already exists on disk and is hash-verified at read
    time. Missing evidence is recorded as missing. A result that
    disagrees with the publication is reported as observed and is NOT
    treated as an integrity failure.

STAGE COVERAGE, STATED PLAINLY
    II-1 and II-3 are assembled from real artifacts by
    audit.targets.autond.implementation_stages.
    II-2, II-4 and II-5 have not been performed for AutoND. All five
    II stages are CORE in this framework (audit.implementation.stages.
    STAGE_TIERS), so these three are recorded explicitly as
    NOT_PERFORMED with their reason. They are never silently omitted
    and never described as having been performed at reduced rigor.
    Consequently the pillar decision is expected to be INCONCLUSIVE,
    and that is the correct reading of the present evidence, not a
    defect of this runner.

EVIDENCE TAG
    The successor preregistration audit.targets.autond.preregistrations.
    autond_ii_v2 is not yet frozen (no sidecar, and one recorded hash
    is still RECORDED_PENDING_VERIFICATION). No pre-execution
    preregistration artifact is therefore passed, so the generic
    builder derives the EXPLORATORY tag. Confirmatory status is not
    manufactured after the fact.

Usage:
    python3 -m audit.targets.autond.run_autond_implementation_audit --dry-run
    python3 -m audit.targets.autond.run_autond_implementation_audit \
        --output-dir audit/targets/autond/evidence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from audit.common.evidence import Evidence, validate_evidence_provenance
from audit.common.outcomes import EnhancedAssessmentStatus, ExecutionMode
from audit.common.provenance import sha256_bytes, sha256_file, utc_timestamp
from audit.common.run_manifest import RunManifest, capture_environment
from audit.common.strict_json import dumps_strict
from audit.common.versioning import VersionRegistry
from audit.implementation.certificate import build_implementation_evidence
from audit.implementation.stages import (
    AssessmentTier,
    STAGE_TIERS,
    StageId,
    StageOutcome,
    StageStatus,
    validate_stage_record,
)
from audit.targets.autond import implementation_stages as autond_stages
from audit.targets.autond import paper_source_architecture as arch
from audit.targets.autond.preregistrations import autond_ii_v2 as plan

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOME = Path.home()

EVIDENCE_ID = "EV-II-AUTOND-001"
CLAIM_ID = "C1-AUTOND"
EXPERIMENT_ID = "II-AUTOND-FIVE-STAGE-V1"
RUN_ID = "ii-autond-assembly-001"

# Code whose content defines this assembly. Hashed into the run manifest
# so the certificate is bound to the exact code that produced it.
#
# EVERY generic helper this adapter actually imports is bound, not just
# the two headline framework modules. If a helper changes, the manifest
# changes, and a certificate produced against a different framework
# revision is distinguishable from this one. The list is checked against
# the modules actually imported by a test, so it cannot silently fall
# behind the code.
CODE_PATHS = (
    # the AutoND target adapter
    "audit/targets/autond/__init__.py",
    "audit/targets/autond/run_autond_implementation_audit.py",
    "audit/targets/autond/implementation_stages.py",
    "audit/targets/autond/paper_source_architecture.py",
    "audit/targets/autond/evaluation_results.py",
    "audit/targets/autond/evaluation_gate.py",
    "audit/targets/autond/preregistrations/autond_ii_v2.py",
    # the generic pillar
    "audit/implementation/stages.py",
    "audit/implementation/certificate.py",
    "audit/implementation/findings.py",
    # the generic helpers this adapter depends on
    "audit/common/equivalence.py",
    "audit/common/evidence.py",
    "audit/common/ids.py",
    "audit/common/outcomes.py",
    "audit/common/provenance.py",
    "audit/common/run_manifest.py",
    "audit/common/strict_json.py",
    "audit/common/versioning.py",
)

# Stages not performed for AutoND, with the reason recorded verbatim in
# the certificate. Each is CORE; none is downgraded or omitted.
NOT_PERFORMED: dict[StageId, str] = {
    StageId.II_2_INDEPENDENT_REIMPLEMENTATION: (
        "No independent reimplementation of AutoND exists in this project. The published "
        "Algorithm 1 reimplementation (R3) is specified in the successor preregistration but "
        "has not been written or run. Nothing here reimplements the differential-search "
        "procedure independently of the released repository."
    ),
    StageId.II_4_COMPARATIVE_EVALUATION: (
        "No comparative evaluation of a reference and an independent AutoND implementation "
        "under identical protocols has been performed, because no independent implementation "
        "exists (II-2). The R1 repository reproduction compares this execution against the "
        "repository's own released code and its README, which is a reproduction check, not a "
        "formal implementation comparison in the sense of II-4."
    ),
    StageId.II_5_STATISTICAL_ASSESSMENT: (
        "No statistical assessment exists for AutoND. The single R1 search execution yields one "
        "observation per cipher with no replicates, no variance estimate and no preregistered "
        "test. The shipped-checkpoint evaluation that would supply replicate accuracies is "
        "gated by the successor preregistration and has not been launched."
    ),
}


def _not_performed_stage(stage_id: StageId, reason: str) -> StageStatus:
    """Record a CORE stage that was not performed, with its reason."""
    return StageStatus(
        stage_id=stage_id,
        tier=STAGE_TIERS[stage_id],
        status=StageOutcome.NOT_PERFORMED,
        summary=reason,
        findings=[f"{stage_id.value}: NOT_PERFORMED. {reason}"],
        components=(),
    )


def assemble(repo_root: Path, home: Path, upstream: Path) -> tuple[RunManifest, dict[str, Any]]:
    """
    Read every AutoND artifact, build the five stage records and the run
    manifest. Reads only; writes nothing.
    """
    started = utc_timestamp()

    # --- artifact verification (hash-checked at read time) -------------
    checks = autond_stages.check_artifacts(repo_root, home, upstream)
    artifact_report = [c.to_dict() for c in checks]

    # --- stages built from real artifacts ------------------------------
    ii1, ii1_detail = autond_stages.build_ii1(repo_root, home, upstream)
    ii3, ii3_detail = autond_stages.build_ii3(repo_root, home, upstream)

    # --- stages that were not performed, recorded explicitly ----------
    ii2 = _not_performed_stage(
        StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
        NOT_PERFORMED[StageId.II_2_INDEPENDENT_REIMPLEMENTATION],
    )
    ii4 = _not_performed_stage(
        StageId.II_4_COMPARATIVE_EVALUATION,
        NOT_PERFORMED[StageId.II_4_COMPARATIVE_EVALUATION],
    )
    ii5 = _not_performed_stage(
        StageId.II_5_STATISTICAL_ASSESSMENT,
        NOT_PERFORMED[StageId.II_5_STATISTICAL_ASSESSMENT],
    )

    stages = [ii1, ii2, ii3, ii4, ii5]

    # --- the framework validates its own stage records -----------------
    stage_problems: list[str] = []
    for stage in stages:
        stage_problems.extend(
            f"{stage.stage_id.value}: {p}" for p in validate_stage_record(stage)
        )
    if stage_problems:
        raise RuntimeError(
            "stage records rejected by the generic validator:\n  - "
            + "\n  - ".join(stage_problems)
        )

    # --- II-3 paper-to-source architecture comparison ------------------
    arch_record = arch.to_record()
    arch_digest = arch.record_digest()

    # --- preregistration gate state, read not asserted -----------------
    frozen, freeze_detail = plan.is_frozen(repo_root)
    freeze_problems = [freeze_detail] if freeze_detail else []
    permitted, gate_problems = plan.confirmatory_run_permitted(upstream, repo_root)

    # --- artifact hashes actually verified on this machine -------------
    artifact_hashes: dict[str, str] = {}
    for rec in artifact_report:
        if rec["verdict"] == "VERIFIED" and rec["observed_sha256"]:
            artifact_hashes[rec["label"]] = rec["observed_sha256"]

    code_hashes: dict[str, str] = {}
    missing_code: list[str] = []
    for rel in CODE_PATHS:
        path = repo_root / rel
        if path.is_file():
            code_hashes[rel] = sha256_file(path)
        else:
            missing_code.append(rel)
    if missing_code:
        raise RuntimeError(
            "code paths this certificate must bind are missing: " + ", ".join(missing_code)
        )

    manifest = RunManifest(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        condition_id="autond_five_stage_assembly",
        replicate_id=None,
        claim_ids=[CLAIM_ID],
        hypothesis_ids=sorted(plan.HYPOTHESES),
        execution_mode=ExecutionMode.PRODUCTION,
        config_hash=sha256_bytes(
            dumps_strict(plan.TARGET, sort_keys=True, default=str).encode("utf-8")
        ),
        # The successor preregistration is NOT frozen, so no preregistration
        # hash is claimed here. See the EXPLORATORY tag.
        preregistration_hash=None,
        dataset_snapshot_id=None,
        dataset_hashes={},
        code_hashes=code_hashes,
        model_source_hash=None,
        environment=capture_environment(),
        seeds={},
        command="python3 -m audit.targets.autond.run_autond_implementation_audit",
        start_time_utc=started,
    )
    manifest.mark_completed(
        exit_code=0,
        metrics={
            "artifacts_expected": len(artifact_report),
            "artifacts_verified": sum(1 for r in artifact_report if r["verdict"] == "VERIFIED"),
            "artifacts_missing": sum(1 for r in artifact_report if r["verdict"] == "MISSING"),
            "artifacts_hash_mismatch": sum(
                1 for r in artifact_report if r["verdict"] == "HASH_MISMATCH"
            ),
            "architecture_claims_checked": arch_record["summary"]["total_claims"],
            "architecture_claims_agree": arch_record["summary"]["by_verdict"].get("AGREES", 0),
            "architecture_claims_differ": arch_record["summary"]["by_verdict"].get("DIFFERS", 0),
            "core_stages_satisfied": sum(1 for s in stages if s.is_satisfied),
            "core_stages_total": len(stages),
        },
    )

    context = {
        "manifest_payload": dumps_strict(manifest.to_dict(), indent=2, sort_keys=True),
        "stages": stages,
        "artifact_report": artifact_report,
        "artifact_hashes": artifact_hashes,
        "arch_record": arch_record,
        "arch_digest": arch_digest,
        "ii1_detail": ii1_detail,
        "ii3_detail": ii3_detail,
        "upstream_root": str(upstream),
        "prereg_frozen": frozen,
        "prereg_freeze_problems": freeze_problems,
        "prereg_permitted": permitted,
        "prereg_gate_problems": gate_problems,
    }
    return manifest, context


def build_evidence(manifest_ref: dict, ctx: dict) -> Evidence:
    """Call the GENERIC builder. No AutoND-specific framework change."""
    stages = ctx["stages"]
    arch_record = ctx["arch_record"]
    report = ctx["artifact_report"]

    unavailable: list[str] = []
    for rec in report:
        if rec["verdict"] == "MISSING":
            unavailable.append(f"{rec['label']}: {rec['detail']}")
        elif rec["verdict"] == "HASH_MISMATCH":
            unavailable.append(
                f"{rec['label']}: HASH_MISMATCH - observed {rec['observed_sha256']} "
                f"!= expected {rec['expected_sha256']}"
            )
        elif rec["verdict"] == "RECORDED_ONLY":
            unavailable.append(
                f"{rec['label']}: recorded by the author but not hash-verified on this "
                "machine; not inspected here"
            )
    if not ctx["prereg_frozen"]:
        unavailable.append(
            "successor preregistration autond_ii_v2 is not frozen: "
            + "; ".join(ctx["prereg_freeze_problems"])
        )

    return build_implementation_evidence(
        evidence_id=EVIDENCE_ID,
        claim_id=CLAIM_ID,
        experiment_id=EXPERIMENT_ID,
        run_ids=[RUN_ID],
        stages=stages,
        findings=[],
        # No independent reimplementation of AutoND exists.
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={
            "runner": "audit.targets.autond.run_autond_implementation_audit",
            "target": "AutoND",
            "case_study": 2,
            "framework_entry_point": (
                "audit.implementation.certificate.build_implementation_evidence"
            ),
            "framework_modified_for_this_target": False,
            "framework_modules_bound": len([
                p for p in CODE_PATHS
                if p.startswith(("audit/common/", "audit/implementation/"))]),
            "upstream_pinned_commit": plan.TARGET["pinned_commit"],
        },
        run_manifest_references=[manifest_ref],
        additional_artifact_hashes=ctx["artifact_hashes"],
        # Not frozen -> no artifact -> builder derives EXPLORATORY.
        preregistration_reference=None,
        unavailable_essential_information=unavailable,
        statistics_note=(
            "II-5 is NOT_PERFORMED. No statistical assessment of AutoND exists: the R1 search "
            "reproduction is a single execution per cipher with no replicates and no variance "
            "estimate, and the shipped-checkpoint evaluation that would supply replicate "
            "accuracies has not been launched (gated by the successor preregistration). No "
            "p-value, confidence interval or effect size is reported for this target."
        ),
        additional_observation={
            "methodology": {
                "section": "3.5.4",
                "stages": [s.value for s in StageId],
                "note": (
                    "All five II stages are CORE in audit.implementation.stages.STAGE_TIERS. "
                    "II-2, II-4 and II-5 are recorded NOT_PERFORMED for AutoND with their "
                    "reasons, not omitted and not downgraded."
                ),
            },
            "stage_coverage": {
                s.stage_id.value: {
                    "tier": s.tier.value,
                    "status": s.status.value,
                    "satisfied": s.is_satisfied,
                    "n_components": len(s.components),
                }
                for s in stages
            },
            "not_performed_reasons": {k.value: v for k, v in NOT_PERFORMED.items()},
            "artifact_verification": [
                {
                    "label": rec["label"],
                    "verdict": rec["verdict"],
                    "provenance": rec["provenance"],
                    "required": rec["required"],
                    "observed_sha256": rec["observed_sha256"],
                    "detail": rec["detail"],
                }
                for rec in report
            ],
            "paper_source_architecture": {
                "record_digest": ctx["arch_digest"],
                "summary": arch_record["summary"],
                "establishes": arch_record["establishes"],
                "does_not_establish": arch_record["does_not_establish"],
                "note": (
                    "II-3 paper-to-source comparison over "
                    f"{arch_record['summary']['total_claims']} architecture claims. "
                    "UNDERSPECIFIED means the publication does not state the value precisely "
                    "enough to compare, which is an observation about the paper's reporting, "
                    "not a defect of the code."
                ),
            },
            "published_value_discrepancy": {
                "description": (
                    "The AutoND paper and the repository README report different SIMON "
                    "accuracies. Preserved unreconciled; this audit does not choose between them."
                ),
                "values": plan.PUBLISHED_VALUES,
            },
            "preregistration_gate": {
                "plan_module": "audit.targets.autond.preregistrations.autond_ii_v2",
                "plan_hash": plan.plan_hash(),
                "unverified_fields": plan.unverified_fields(),
                "frozen": ctx["prereg_frozen"],
                "freeze_problems": ctx["prereg_freeze_problems"],
                "confirmatory_run_permitted": ctx["prereg_permitted"],
                "gate_problems": ctx["prereg_gate_problems"],
                "note": (
                    "The shipped-checkpoint evaluation is withheld until this plan is frozen "
                    "with a sidecar binding plan_hash, timestamp and git commit."
                ),
            },
            "outstanding_empirical_work": {
                "ii2_independent_algorithm1": (
                    "Implement published Algorithm 1 from the paper text alone (R3), without "
                    "reading optimizer.evo(), and run it on SPECK32/64 and SIMON32/64."
                ),
                "ii4_formal_comparison": (
                    "Run the released repository and the independent Algorithm 1 "
                    "implementation under identical protocols and compare recovered "
                    "differences and highest_non_random_round."
                ),
                "ii5_statistical_assessment": (
                    "Replicate the search with independent seeds to obtain a variance estimate, "
                    "and evaluate the shipped checkpoints once the preregistration is frozen."
                ),
            },
        },
    )


def run(repo_root: Path, home: Path, upstream: Path, output_dir: Path, *, write: bool) -> int:
    manifest, ctx = assemble(repo_root, home, upstream)
    manifest_name = f"run_manifest_{RUN_ID}.json"
    manifest_path = output_dir / manifest_name
    if write and manifest_path.exists():
        print(f"refusing to overwrite existing {manifest_path}")
        return 1

    manifest_ref = {
        "run_id": RUN_ID,
        "artifact_path": manifest_name,
        "content_hash": sha256_bytes(ctx["manifest_payload"].encode("utf-8")),
    }
    evidence = build_evidence(manifest_ref, ctx)

    problems = validate_evidence_provenance(
        evidence, known_run_manifests={RUN_ID: manifest}
    )
    if problems:
        print(
            "EVIDENCE PROVENANCE VALIDATION FAILED - refusing to write:",
            *problems,
            sep="\n  - ",
        )
        return 1

    payload = dumps_strict(evidence.to_dict(), indent=2, sort_keys=True)

    # --- report -------------------------------------------------------
    print("AutoND Implementation Integrity (case study 2)")
    print(f"upstream pinned commit: {plan.TARGET['pinned_commit']}")
    print(f"upstream root:          {ctx['upstream_root']}")
    print(f"upstream HEAD ok:       {ctx['ii1_detail']['commit_ok']}  "
          f"({ctx['ii1_detail']['head'] or 'unreadable'})")
    print()
    print(f"{'stage':<38} {'tier':<5} status")
    for stage in evidence.observation["stages"]:
        print(f"{stage['stage_id']:<38} {stage['tier']:<5} {stage['status']}")
    print()
    for rec in ctx["artifact_report"]:
        flag = "  " if rec["verdict"] in ("VERIFIED", "RECORDED_ONLY") else "!!"
        req = "required" if rec["required"] else "optional"
        print(f"  {flag} {rec['verdict']:<14} {rec['label']:<34} [{req}, {rec['provenance']}]")
    print()
    counts = ctx["arch_record"]["summary"]["by_verdict"]
    print(
        "II-3 paper-to-source: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + f"   digest={ctx['arch_digest'][:16]}..."
    )
    print(
        f"preregistration frozen: {ctx['prereg_frozen']}   "
        f"confirmatory run permitted: {ctx['prereg_permitted']}"
    )
    print(f"Decision: {evidence.decision}   tag: {evidence.tag.value}")
    if not write:
        print("Dry run: nothing written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(ctx["manifest_payload"])
    registry = VersionRegistry(output_dir / "version_registry.json")
    record = registry.register(
        "autond-implementation-certificate",
        sha256_bytes(payload.encode("utf-8")),
        note=(
            "AutoND Implementation Integrity certificate (case study 2). II-1 and II-3 "
            "assembled from verified artifacts; II-2, II-4, II-5 NOT_PERFORMED."
        ),
    )
    out_path = output_dir / f"{record.identifier}.json"
    if out_path.exists():
        print(f"refusing to overwrite existing {out_path}")
        return 1
    out_path.write_text(payload)
    print(f"Run manifest: {manifest_path}")
    print(f"Versioned evidence written: {out_path} ({record.identifier})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the AutoND Implementation Integrity certificate."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME,
                        help="root for author-recorded local evidence")
    parser.add_argument("--autond-root", type=Path, required=True,
                        help="pinned upstream AutoND checkout (read-only)")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "audit/targets/autond/evidence"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="assemble and verify; write nothing"
    )
    args = parser.parse_args()
    return run(
        args.repo_root.resolve(),
        args.home.expanduser(),
        args.autond_root.expanduser().resolve(),
        args.output_dir,
        write=not args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
