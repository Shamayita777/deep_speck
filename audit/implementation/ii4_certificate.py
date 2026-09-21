"""
II-4 run -> Implementation Integrity certificate (next version, e.g. V4).

Converts a completed II-4 run directory into a versioned Implementation
Integrity certificate. It runs no experiment and computes no new
statistic: every verdict is read from the persisted analysis, and the
interpretation is taken from the PREREGISTRATION FILE written before any
training - never from the code constant, so a post-hoc edit to
`ii4_design.INTERPRETATION_RULE` cannot re-interpret an existing run.

Historical certificates are never modified: the version registry is
append-only, and a new certificate receives the next version number.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

from audit.common.evidence import validate_evidence_provenance
from audit.common.outcomes import (
    AssessmentTier,
    EnhancedAssessmentStatus,
    ExecutionMode,
)
from audit.common.provenance import sha256_bytes, sha256_file
from audit.common.run_manifest import RunManifest
from audit.common.strict_json import dumps_strict
from audit.common.versioning import VersionRegistry
from audit.implementation.certificate import build_implementation_evidence
from audit.implementation.findings import (
    DEPTH_FINDING_V1,
    ScientificImpactStatus,
)
from audit.implementation.ii4_design import apply_interpretation_rule
from audit.implementation.stages import (
    STAGE_TIERS,
    StageId,
    StageOutcome,
    StageStatus,
    not_assessed_stage,
)

CONVERTER_VERSION = "ii4-certificate-v1"
REAL_EVIDENCE_DIRNAME = "evidence"


class II4CertificateError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise II4CertificateError(f"required II-4 artifact missing: {path}")
    return json.loads(path.read_text())


def convert_ii4_run(
    run_dir: Path, certificate_dir: Path, *, allow_non_evidentiary: bool = False,
) -> dict[str, Any]:
    """
    Build the next Implementation Integrity certificate from an II-4 run.

    `allow_non_evidentiary` exists only so the conversion path can be
    tested on smoke runs; a smoke run may never be written into a
    directory named 'evidence'.
    """
    run_dir, certificate_dir = Path(run_dir), Path(certificate_dir)
    prereg_path = run_dir / "ii4_preregistration.json"
    analysis_path = run_dir / "ii4_analysis.json"
    state_path = run_dir / "ii4_state.json"
    prereg, report, state = _load(prereg_path), _load(analysis_path), _load(state_path)

    # --- integrity of the run ------------------------------------------------
    if report.get("config_fingerprint") != prereg.get("experiment_fingerprint"):
        raise II4CertificateError(
            "analysis fingerprint does not match the preregistration; the run directory is "
            "inconsistent and cannot be certified.")
    if not state.get("completed_at_utc"):
        raise II4CertificateError("II-4 run has not completed; refusing to certify a partial run.")
    non_evidentiary = bool(report.get("non_evidentiary"))
    if non_evidentiary and not allow_non_evidentiary:
        raise II4CertificateError(
            "refusing to certify a NON-EVIDENTIARY (smoke) II-4 run as evidence.")
    if non_evidentiary and REAL_EVIDENCE_DIRNAME in certificate_dir.resolve().parts:
        raise II4CertificateError(
            "a non-evidentiary run may never be written into an 'evidence' directory.")
    if "interpretation_rule" not in prereg:
        raise II4CertificateError("preregistration carries no interpretation rule.")

    # --- apply the PREREGISTERED interpretation ------------------------------
    conf = report["analysis"]["confirmatory"]
    materiality = conf["materiality"]["verdict"]
    equivalence = conf["equivalence"]["verdict"]
    clause = apply_interpretation_rule(prereg["interpretation_rule"], materiality, equivalence)
    impact = ScientificImpactStatus(clause["scientific_impact"])
    sufficient = report["analysis"]["replication"]["sufficient"]

    # --- bind run artifacts next to the certificate --------------------------
    fp = prereg["experiment_fingerprint"]
    bound_rel = Path("ii4_bound") / fp[:16]
    bound = certificate_dir / bound_rel
    bound.mkdir(parents=True, exist_ok=True)
    artifact_hashes: dict[str, str] = {}
    for src in (prereg_path, analysis_path, state_path, run_dir / "ii4_firewall.json"):
        if src.is_file():
            dst = bound / src.name
            shutil.copyfile(src, dst)
            artifact_hashes[str(bound_rel / src.name)] = sha256_file(dst)
    prereg_rel = str(bound_rel / prereg_path.name)
    prereg_hash = artifact_hashes[prereg_rel]
    for label, entry in prereg.get("source_manifest", {}).items():
        artifact_hashes[f"source:{entry['path']}"] = entry["sha256"]

    # --- finding (new version; V1 untouched) ---------------------------------
    finding = dataclasses.replace(
        DEPTH_FINDING_V1,
        finding_id="II-FINDING-DEPTH-V2",
        scientific_impact=impact,
        actual={
            **DEPTH_FINDING_V1.actual,
            "ii4_experiment_id": report["experiment_id"],
            "ii4_mean_signed_difference_depth10_minus_depth5": conf["mean_difference"],
            "ii4_ci95": conf["ci95"],
            "ii4_materiality": materiality,
            "ii4_equivalence": equivalence,
            "ii4_valid_blocks": report["analysis"]["replication"]["n_blocks_valid"],
            "interpretation_clause": clause["meaning"],
        },
        supporting_artifacts={**DEPTH_FINDING_V1.supporting_artifacts, **artifact_hashes},
        open_questions=(
            ["EQUIVALENT_WITHIN_MARGIN establishes ACCURACY conformance only; it does NOT "
             "validate transferability of Cryptographic Evidence obtained on the depth-5 model."]
            if impact is ScientificImpactStatus.REFUTED else list(DEPTH_FINDING_V1.open_questions)
        ),
    )

    # --- stages -------------------------------------------------------------
    stages = [
        StageStatus(StageId.II_1_BASELINE_RECONSTRUCTION, STAGE_TIERS[StageId.II_1_BASELINE_RECONSTRUCTION],
                    StageOutcome.CONFORMANCE_VERIFIED, "Reference protocol hash-verified (unchanged from V3)."),
        not_assessed_stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
                           reason="ENHANCED stage; not performed. Contributes no convergence source; imposes no cap."),
        StageStatus(StageId.II_3_CONTROLLED_VERIFICATION, STAGE_TIERS[StageId.II_3_CONTROLLED_VERIFICATION],
                    StageOutcome.CONFORMANCE_VERIFIED, "Realized depth 5 vs declared 10 (unchanged from V3)."),
        StageStatus(StageId.II_4_COMPARATIVE_EVALUATION, STAGE_TIERS[StageId.II_4_COMPARATIVE_EVALUATION],
                    StageOutcome.COMPLETED if sufficient else StageOutcome.PARTIAL,
                    f"Dataset-blocked paired comparison, {report['analysis']['replication']['n_blocks_valid']}"
                    f"/{report['analysis']['replication']['n_blocks_requested']} valid blocks."),
        StageStatus(StageId.II_5_STATISTICAL_ASSESSMENT, STAGE_TIERS[StageId.II_5_STATISTICAL_ASSESSMENT],
                    StageOutcome.COMPLETED,
                    f"Signed 95% CI materiality = {materiality}; TOST equivalence = {equivalence}."),
    ]

    # --- run manifest ---------------------------------------------------------
    run_id = f"{report['experiment_id']}-run"
    manifest = RunManifest(
        run_id=run_id, experiment_id=report["experiment_id"], condition_id=None,
        replicate_id=None, claim_ids=["C1"], hypothesis_ids=["H-II4-CONFORMANCE-IMPACT"],
        execution_mode=ExecutionMode(report["execution_mode"]),
        config_hash=prereg["config_fingerprint"], preregistration_hash=prereg_hash,
        dataset_snapshot_id=None, dataset_hashes={"sealed_test_set": report["sealed_test_set"]["hash"]},
        code_hashes={e["path"]: e["sha256"] for e in prereg.get("source_manifest", {}).values()},
        model_source_hash=None,
        environment=(report.get("session_environments") or [{}])[0],
        seeds={f"{b['block_id']}_{arm}": b[f"{arm}_seed"]
               for b in report["blocks"] for arm in ("declared", "realized")},
        command="python -m audit.implementation.run_ii4", start_time_utc=state["created_at_utc"],
    )
    manifest.mark_completed(exit_code=0, metrics={"materiality": materiality, "equivalence": equivalence})
    manifest_path = bound / f"run_manifest_{run_id}.json"
    manifest_path.write_text(dumps_strict(manifest.to_dict(), indent=2, sort_keys=True))
    manifest_ref = {"run_id": run_id, "artifact_path": str(bound_rel / manifest_path.name),
                    "content_hash": sha256_file(manifest_path)}

    # --- certificate ------------------------------------------------------------
    # The preregistration file was verified above (copied, hashed, and its
    # fingerprint checked against the analysis), so the certificate builder is
    # told it has been verified through that channel.
    evidence = build_implementation_evidence(
        evidence_id="EV-II-DEPTH-002", claim_id="C1", experiment_id=report["experiment_id"],
        run_ids=[run_id], stages=stages, findings=[finding],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"ii4_experiment_fingerprint": fp, "converter_version": CONVERTER_VERSION,
                    "bound_artifacts": sorted(artifact_hashes)},
        run_manifest_references=[manifest_ref],
        preregistration_reference={"artifact_path": prereg_rel, "content_hash": prereg_hash},
        verify_preregistration_artifact=False,
    )
    if evidence.decision != clause["ii_outcome"]:
        raise II4CertificateError(
            f"certificate decision {evidence.decision} disagrees with the preregistered "
            f"interpretation {clause['ii_outcome']}; refusing to issue an inconsistent certificate.")
    problems = validate_evidence_provenance(evidence, known_run_manifests={run_id: manifest})
    if problems:
        raise II4CertificateError(f"provenance validation failed: {problems}")

    payload = dumps_strict({**evidence.to_dict(), "non_evidentiary": non_evidentiary},
                           indent=2, sort_keys=True)
    registry = VersionRegistry(certificate_dir / "version_registry.json")
    record = registry.register("implementation-certificate", sha256_bytes(payload.encode("utf-8")),
                               note="Implementation Integrity certificate incorporating II-4.")
    out = certificate_dir / f"{record.identifier}.json"
    if out.exists() and sha256_file(out) != record.content_hash:
        raise II4CertificateError(f"refusing to overwrite existing certificate {out.name}.")
    out.write_text(payload)
    return {"identifier": record.identifier, "path": str(out), "decision": evidence.decision,
            "tag": evidence.tag.value, "clause": clause, "non_evidentiary": non_evidentiary}
