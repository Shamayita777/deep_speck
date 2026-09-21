#!/usr/bin/env python3
"""
Implementation Integrity runner.

Executes the CORE Implementation Integrity stages against the Gohr
case study, records the ENHANCED stage as NOT_ASSESSED, and writes a
versioned evidence artifact. Never modifies any historical D/EV/CE
evidence - the CE checkpoints it probes are opened READ-ONLY.

Usage:
    python3 -m audit.implementation.run_implementation_audit \
        --ce-evidence-dir /path/to/cryptography/evidence/ce1 \
        --output-dir audit/implementation/evidence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audit.common.ids import Dimension
from audit.common.outcomes import AssessmentTier, EnhancedAssessmentStatus
from audit.common.provenance import (
    Activity,
    Agent,
    Entity,
    ProvenanceGraph,
    sha256_file,
    utc_timestamp,
)
from audit.common.evidence import validate_evidence_provenance
from audit.common.provenance import sha256_bytes
from audit.common.strict_json import dumps_strict
from audit.common.versioning import VersionRegistry
from audit.implementation.certificate import build_implementation_evidence
from audit.implementation.findings import (
    ALL_FINDINGS,
    DEPTH_FINDING_V1,
    FindingObservationMismatch,
    verify_finding_against_observations,
)
from audit.implementation.gohr_adapter import (
    DECLARED_BASELINE,
    REFERENCE_FILE_HASHES,
    probe_checkpoint_architecture,
)
from audit.implementation.stages import (
    STAGE_TIERS,
    StageId,
    StageOutcome,
    StageStatus,
    not_assessed_stage,
)
from audit.common.run_manifest import RunManifest, capture_environment
from audit.common.outcomes import ExecutionMode, RunStatus


def verify_reference_files(reference_dir: Path) -> tuple[bool, list[str]]:
    """II-1: confirm the reference protocol files are the expected, unmodified ones."""
    problems: list[str] = []
    for name, expected in REFERENCE_FILE_HASHES.items():
        path = reference_dir / name
        if not path.exists():
            problems.append(f"missing reference file: {name}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{name}: hash {actual} != expected {expected}")
    return (len(problems) == 0, problems)


def run(ce_evidence_dir: Path, output_dir: Path, reference_dir: Path) -> int:
    graph = ProvenanceGraph()
    graph.add_agent(Agent("audit.implementation.runner", "software_agent"))

    stages: list[StageStatus] = []
    unavailable: list[str] = []

    # --- II-1 Baseline Reconstruction ---
    activity = Activity("ii1_baseline_reconstruction", "verification", started_at_utc=utc_timestamp())
    graph.add_activity(activity)
    graph.associated_with(activity.activity_id, "audit.implementation.runner")

    refs_ok, ref_problems = verify_reference_files(reference_dir)
    for name, digest in REFERENCE_FILE_HASHES.items():
        entity = Entity(f"reference/{name}", "reference_source", content_hash=digest)
        graph.add_entity(entity)
        graph.used(activity.activity_id, entity.entity_id)

    stages.append(StageStatus(
        stage_id=StageId.II_1_BASELINE_RECONSTRUCTION,
        tier=STAGE_TIERS[StageId.II_1_BASELINE_RECONSTRUCTION],
        status=StageOutcome.CONFORMANCE_VERIFIED if refs_ok else StageOutcome.FAILED,
        summary=(
            "Declared reference protocol transcribed from the original Gohr source "
            f"(hash-verified): rounds={DECLARED_BASELINE['rounds']}, "
            f"depth={DECLARED_BASELINE['depth']}, epochs={DECLARED_BASELINE['epochs']}, "
            f"batch_size={DECLARED_BASELINE['batch_size']}."
            if refs_ok else f"Reference file verification failed: {ref_problems}"
        ),
        findings=ref_problems,
    ))
    if not refs_ok:
        unavailable.extend(ref_problems)

    # --- II-2 Independent Reimplementation (ENHANCED) ---
    stages.append(not_assessed_stage(
        StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
        reason=(
            "ENHANCED stage. Not performed for this audit. No independent reimplementation "
            "campaign was conducted, therefore Implementation Integrity contributes no "
            "independent-implementation convergence source to the integration engine. This "
            "imposes no cap on the attainable evidence level: convergence may arise from other "
            "qualifying sources (independent datasets or architectures) and the evidence level "
            "is determined by the integration engine from the convergence evidence that "
            "actually exists. This audit must NOT be described as an enhanced or full "
            "independent methodological reproduction."
        ),
    ))

    # --- II-3 Controlled Verification ---
    activity3 = Activity("ii3_controlled_verification", "verification", started_at_utc=utc_timestamp())
    graph.add_activity(activity3)
    graph.associated_with(activity3.activity_id, "audit.implementation.runner")

    probes = []
    probe_findings: list[str] = []
    for checkpoint_name in ("best5depth10 (10).h5", "signal_destroyed.h5"):
        path = ce_evidence_dir / checkpoint_name
        if not path.exists():
            probe_findings.append(f"checkpoint not available for probing: {checkpoint_name}")
            unavailable.append(str(path))
            continue
        probe = probe_checkpoint_architecture(path)
        probes.append(probe)
        entity = Entity(checkpoint_name, "model_checkpoint", content_hash=probe.sha256,
                        attributes=probe.to_dict())
        graph.add_entity(entity)
        graph.used(activity3.activity_id, entity.entity_id)
        probe_findings.append(
            f"{checkpoint_name}: {probe.n_conv1d_layers} Conv1D layers -> "
            f"derived depth={probe.derived_depth} (declared depth={DECLARED_BASELINE['depth']})"
        )

    stages.append(StageStatus(
        stage_id=StageId.II_3_CONTROLLED_VERIFICATION,
        tier=STAGE_TIERS[StageId.II_3_CONTROLLED_VERIFICATION],
        status=StageOutcome.CONFORMANCE_VERIFIED if probes else StageOutcome.FAILED,
        summary=(
            "Deterministic conformance verification: realized architecture read directly from "
            "the checkpoint artifacts via h5py layer census (not inferred from filenames). This "
            "establishes WHAT the artifact is, not what effect any divergence had."
        ),
        findings=probe_findings,
    ))

    # --- II-4 Comparative Evaluation ---
    mismatch = any(p.derived_depth != DECLARED_BASELINE["depth"] for p in probes) if probes else False
    stages.append(StageStatus(
        stage_id=StageId.II_4_COMPARATIVE_EVALUATION,
        tier=STAGE_TIERS[StageId.II_4_COMPARATIVE_EVALUATION],
        status=StageOutcome.NOT_PERFORMED,
        summary=(
            "NOT PERFORMED. A declared-vs-realized CONFIGURATION comparison was completed under "
            "II-3 and confirms the discrepancy"
            + (" (depth 5 realized vs 10 declared)" if mismatch else " (none detected)")
            + ", but that is a conformance check, not a comparative EXPERIMENTAL evaluation. "
            "II-4 requires executing the reference-conformant configuration and comparing "
            "experimental outcomes; no such re-execution exists, so no comparative evaluation "
            "has been performed."
        ),
        findings=probe_findings,
    ))

    # --- II-5 Statistical Assessment ---
    stages.append(StageStatus(
        stage_id=StageId.II_5_STATISTICAL_ASSESSMENT,
        tier=STAGE_TIERS[StageId.II_5_STATISTICAL_ASSESSMENT],
        status=StageOutcome.NOT_APPLICABLE,
        summary=(
            "NOT APPLICABLE to the evidence currently available: the only completed verification "
            "is a deterministic, exactly-verified architectural mismatch with no sampling "
            "variability, for which inferential statistics are not the appropriate instrument. "
            "II-5 becomes APPLICABLE (and then required) once II-4 supplies comparable "
            "populations of training runs from a controlled re-execution."
        ),
        findings=[
            "Quantitative attribution of the 0.9291 -> 0.6108 accuracy gap to depth is NOT "
            "established and is recorded as an open question requiring a controlled re-execution.",
        ],
    ))

    # Close out provenance activities: an activity that generated
    # evidence must record its end time.
    graph.complete_activity(activity.activity_id)
    graph.complete_activity(activity3.activity_id)

    findings = list(ALL_FINDINGS) if mismatch else [
        f for f in ALL_FINDINGS if f.finding_id != DEPTH_FINDING_V1.finding_id
    ]

    # CORRECTION 4: bind every stored finding to what was ACTUALLY
    # observed this execution. The certificate must never cite a hash or
    # a measured value that differs from the artifact really inspected.
    observed_artifact_hashes = {Path(pr.path).name: pr.sha256 for pr in probes}
    # Aggregate observations only where the inspected artifacts AGREE.
    # Where they disagree the key is deliberately omitted, and the
    # fail-closed check in verify_finding_against_observations then
    # refuses to publish the corresponding stored factual claim.
    observed_values = {}
    derived_depths = {pr.derived_depth for pr in probes}
    if len(derived_depths) == 1:
        observed_values["depth"] = derived_depths.pop()
    observed_conv_counts = {pr.n_conv1d_layers for pr in probes}
    if len(observed_conv_counts) == 1:
        observed_values["n_conv1d_layers"] = observed_conv_counts.pop()

    binding_problems: list[str] = []
    for finding in findings:
        binding_problems.extend(verify_finding_against_observations(
            finding, observed_artifact_hashes=observed_artifact_hashes,
            observed_values=observed_values,
            verifiable_keys=("depth", "n_conv1d_layers"),
        ))
    if binding_problems:
        print("FINDING/OBSERVATION BINDING FAILED - refusing to write evidence:")
        for problem in binding_problems:
            print("  -", problem)
        return 1
    print(f"Finding/observation binding verified against {len(observed_artifact_hashes)} "
          f"inspected artifact(s); observed values: {observed_values}")

    # A real RunManifest so the evidence's run_ids are traceable rather
    # than free-floating strings.
    run_manifest = RunManifest(
        run_id="ii-core-run-001",
        experiment_id="II-CORE-GOHR-5R",
        condition_id="declared_vs_realized_conformance",
        replicate_id=None,
        claim_ids=["C1"],
        hypothesis_ids=["H-II-CONFORMANCE"],
        execution_mode=ExecutionMode.PRODUCTION,
        config_hash=sha256_bytes(
            dumps_strict(DECLARED_BASELINE, sort_keys=True, default=str).encode("utf-8")
        ),
        # CORRECTED: no pre-execution preregistration artifact exists for this
        # historical conformance execution. Hashing a runtime-constructed string
        # would manufacture the appearance of preregistration without the
        # substance, so preregistration_hash is explicitly None and the evidence
        # is tagged EXPLORATORY (see build_implementation_evidence).
        preregistration_hash=None,
        dataset_snapshot_id=None,
        dataset_hashes={},
        code_hashes=dict(REFERENCE_FILE_HASHES),
        model_source_hash=None,
        environment=capture_environment(),
        seeds={},
        command="python3 -m audit.implementation.run_implementation_audit",
        start_time_utc=activity.started_at_utc,
    )
    run_manifest.mark_completed(exit_code=0, metrics={
        "checkpoints_probed": len(probes),
        "architectural_discrepancy_confirmed": bool(mismatch),
    })

    # Persist the manifest FIRST so its real on-disk content hash can be
    # embedded in the evidence, making Evidence -> RunManifest traceable
    # from the artifact alone (correction 3).
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"run_manifest_{run_manifest.run_id}.json"
    manifest_path.write_text(dumps_strict(run_manifest.to_dict(), indent=2, sort_keys=True))
    manifest_reference = {
        "run_id": run_manifest.run_id,
        "artifact_path": manifest_path.name,
        "content_hash": sha256_file(manifest_path),
    }

    # Register the RunManifest as a first-class provenance entity and
    # derive it from the activities that produced it, so the embedded
    # graph is a complete trace rather than a partial one.
    manifest_entity_id = manifest_reference["artifact_path"]
    graph.add_entity(Entity(
        manifest_entity_id, "run_manifest",
        content_hash=manifest_reference["content_hash"],
        attributes={"run_id": run_manifest.run_id},
    ))
    graph.generated(manifest_entity_id, activity3.activity_id)

    # The evidence entity and its generating relation must exist in the
    # graph BEFORE the graph is serialized into the evidence, otherwise
    # the embedded copy is a stale snapshot with no evidence node and no
    # generated edge (which is exactly what V2 contained).
    evidence_id = "EV-II-DEPTH-001"
    graph.add_entity(Entity(evidence_id, "evidence"))
    graph.generated(evidence_id, activity3.activity_id)
    graph.derived_from(evidence_id, manifest_entity_id)
    for probe in probes:
        graph.derived_from(evidence_id, Path(probe.path).name)

    prov_problems = graph.validate()
    if prov_problems:
        print("PROVENANCE VALIDATION FAILED - refusing to write evidence:")
        for problem in prov_problems:
            print("  -", problem)
        return 1

    evidence = build_implementation_evidence(
        evidence_id=evidence_id,
        claim_id="C1",
        experiment_id="II-CORE-GOHR-5R",
        run_ids=["ii-core-run-001"],
        stages=stages,
        findings=findings,
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance=graph.to_dict(),
        run_manifest_references=[manifest_reference],
        # Every artifact represented in provenance must also appear in
        # the evidence's artifact record, so the two never disagree.
        additional_artifact_hashes={
            f"reference/{name}": digest for name, digest in REFERENCE_FILE_HASHES.items()
        },
        # No pre-execution preregistration artifact exists for this historical
        # conformance execution, so none is supplied and the evidence is
        # tagged EXPLORATORY by build_implementation_evidence.
        preregistration_reference=None,
        unavailable_essential_information=unavailable or None,
    )

    # The evidence's run_ids must resolve to real manifests before writing.
    ev_problems = validate_evidence_provenance(
        evidence, known_run_manifests={run_manifest.run_id: run_manifest},
        provenance_graph=graph,
    )
    if ev_problems:
        print("EVIDENCE PROVENANCE VALIDATION FAILED - refusing to write evidence:")
        for problem in ev_problems:
            print("  -", problem)
        return 1

    payload = dumps_strict(evidence.to_dict(), indent=2, sort_keys=True)

    registry = VersionRegistry(output_dir / "version_registry.json")
    record = registry.register(
        "implementation-certificate", sha256_bytes(payload.encode("utf-8")),
        note="Core Implementation Integrity audit of the Gohr 5-round distinguisher.",
    )
    out_path = output_dir / f"{record.identifier}.json"
    out_path.write_text(payload)

    print(f"Implementation Integrity outcome: {evidence.decision}")
    print(f"  discrepancy_confirmed: {evidence.observation['discrepancy_confirmed']}")
    print(f"  scientific_impact_demonstrated: "
          f"{evidence.observation['scientific_impact_demonstrated']}")
    print(f"Independent reimplementation (ENHANCED): NOT_ASSESSED")
    print(f"Convergence contribution from this pillar: "
          f"{evidence.observation['convergence_contribution']['contributes']}")
    print(f"Downstream requiring re-evaluation: "
          f"{evidence.observation['downstream_requires_reevaluation']}")
    print(f"Evidence tag: {evidence.tag.value}")
    print(f"Run manifest: {manifest_path} (hash {manifest_reference['content_hash'][:16]}...)")
    print(f"Versioned evidence written: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path(__file__).resolve().parent / "reference",
    )
    args = parser.parse_args()
    return run(args.ce_evidence_dir, args.output_dir, args.reference_dir)


if __name__ == "__main__":
    sys.exit(main())
