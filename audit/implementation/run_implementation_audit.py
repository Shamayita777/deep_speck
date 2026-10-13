#!/usr/bin/env python3
"""
Implementation Integrity runner (five-stage methodology, Section 3.5.4).

Assembles II-1..II-5 for the Gohr case study from the evidence that
actually exists, verifies every artifact it cites (hash, and realized
architecture where applicable), and writes the next versioned
Implementation Integrity certificate. It trains nothing and recomputes
no statistic: II-4/II-5 results are read from the frozen, bound
II-4-GOHR-DEPTH-V1 artifacts.

Nothing historical is modified: CE checkpoints and certificates are
opened read-only, earlier certificates (V1-V5) and the depth
experiment's preregistration/state/analysis are never rewritten, and the
version registry is append-only.

Certificates V1-V3 were produced by an earlier revision of this runner
(git history); V4-V5 by audit.implementation.ii4_certificate.

Usage:
    python3 -m audit.implementation.run_implementation_audit --dry-run
    python3 -m audit.implementation.run_implementation_audit \
        --output-dir audit/implementation/evidence
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from audit.common.evidence import Evidence, validate_evidence_provenance
from audit.common.outcomes import EnhancedAssessmentStatus, ExecutionMode
from audit.common.provenance import sha256_bytes, sha256_file, utc_timestamp
from audit.common.run_manifest import RunManifest, capture_environment
from audit.common.strict_json import dumps_strict
from audit.common.versioning import VersionRegistry
from audit.implementation.certificate import build_implementation_evidence
from audit.implementation.findings import (
    reconcile_downstream,
    verify_finding_against_observations,
)
from audit.implementation.gohr_adapter import (
    CURRENT_CE_CERTIFICATES,
    DECLARED_BASELINE,
    DEPTH_EXPERIMENT_BOUND_DIR,
    DEPTH_EXPERIMENT_ID,
    DEPTH_EXPERIMENT_ORIGINAL_DIR,
    FIXED_CHECKPOINT_COMPARISON_PROPOSAL,
    FORMAL_COMPARISON_PROPOSAL,
    HISTORICAL_CE_CERTIFICATES,
    HISTORICAL_DEPTH5_CHECKPOINTS,
    II2_EVIDENCE,
    II2_SOURCES,
    REFERENCE_DEPTH10_CHECKPOINT,
    REFERENCE_FILE_HASHES,
    probe_checkpoint_architecture,
    verify_ce_certificate,
)
from audit.implementation.ii4_certificate import conformance_impact_result
from audit.implementation.stages import (
    STAGE_TIERS,
    ComponentKind,
    StageComponent,
    StageId,
    StageOutcome,
    StageStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ID = "EV-II-FIVE-STAGE-003"
EXPERIMENT_ID = "II-GOHR-5R-FIVE-STAGE-V1"
RUN_ID = "ii-five-stage-run-001"
PREDECESSOR = "implementation-certificate-V5"
DEPTH_FINDING_ID = "II-FINDING-DEPTH-V3"
OUTSTANDING = ("The formal comparison of the reference and independent implementations under "
               "identical experimental protocols has NOT been performed.")


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


def _hash_all(repo_root: Path, rel_paths: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    hashes, missing = {}, []
    for rel in rel_paths.values():
        path = repo_root / rel
        if path.is_file():
            hashes[rel] = sha256_file(path)
        else:
            missing.append(rel)
    return hashes, missing


# --- II-2 ------------------------------------------------------------------

def assess_ii2(repo_root: Path) -> tuple[StageStatus, dict[str, Any]]:
    """
    II-2 Independent Reimplementation, from the evidence that exists.
    COMPLETED means the reimplementation exists, its own evidence is
    internally consistent with its source, and it was exercised
    (deterministic checks + one training run). It does NOT mean it
    agrees with the reference at the level of experimental outcomes.
    """
    ev_hashes, ev_missing = _hash_all(repo_root, II2_EVIDENCE)
    src_hashes, src_missing = _hash_all(repo_root, II2_SOURCES)
    problems = [f"missing: {p}" for p in ev_missing + src_missing]
    if problems:
        stage = StageStatus(StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
                            STAGE_TIERS[StageId.II_2_INDEPENDENT_REIMPLEMENTATION],
                            StageOutcome.FAILED, "II-2 evidence incomplete.", findings=problems)
        return stage, {"problems": problems}

    load = lambda key: json.loads((repo_root / II2_EVIDENCE[key]).read_text())  # noqa: E731
    m1, training, m7 = (load("m1_deterministic_verification"),
                        load("m1_independent_training_report"), load("m7_theory_verification"))
    src = {k: src_hashes[v] for k, v in II2_SOURCES.items()}
    ref_speck = REFERENCE_FILE_HASHES["speck.py"]

    # M1 deterministic verification: PASS, and bound to the sources present now.
    for key in ("independent_dataset", "independent_model", "independent_speck"):
        if m1["provenance"]["sha256"].get(key) != src[key]:
            problems.append(f"M1 records {key} sha256 that differs from the current source.")
    if m1["provenance"]["sha256"].get("reference_speck") != ref_speck:
        problems.append("M1 reference_speck hash differs from the verified reference speck.py.")
    if m1.get("status") != "PASS" or any(c.get("status") != "PASS" for c in m1["checks"].values()):
        problems.append("M1 deterministic verification is not PASS on every check.")

    # Independent training report: complete, bound to the sources, declared protocol.
    prov = training["provenance"]
    for key, rec in (("independent_dataset", "independent_dataset_sha256"),
                     ("independent_model", "independent_model_sha256"),
                     ("independent_speck", "independent_speck_sha256"),
                     ("independent_train", "independent_train_sha256")):
        if prov.get(rec) != src[key]:
            problems.append(f"training report {rec} differs from the current source.")
    if training.get("status") != "COMPLETE" or \
            training.get("scientific_status") != "TRAINED_INDEPENDENT_IMPLEMENTATION":
        problems.append("training report is not COMPLETE / TRAINED_INDEPENDENT_IMPLEMENTATION.")
    proto = training["protocol"]
    for key, declared in (("depth", "depth"), ("epochs", "epochs"), ("batch_size", "batch_size"),
                          ("loss", "loss"), ("optimizer", "optimizer"),
                          ("train_samples", "train_samples"),
                          ("validation_samples", "validation_samples")):
        if proto.get(key) != DECLARED_BASELINE[declared]:
            problems.append(f"training protocol {key}={proto.get(key)!r} != declared "
                            f"{DECLARED_BASELINE[declared]!r}.")

    # Independent model artifact: readable container, realized depth 10.
    model_path = repo_root / II2_EVIDENCE["m1_independent_model"]
    try:
        with zipfile.ZipFile(model_path) as archive:
            if archive.testzip() is not None:
                problems.append("independent model archive has a corrupt member.")
        probe = probe_checkpoint_architecture(model_path)
        if probe.derived_depth != DECLARED_BASELINE["depth"] or not probe.is_consistent:
            problems.append(f"independent model realized depth {probe.derived_depth} "
                            f"(Add={probe.n_add_layers}) != declared {DECLARED_BASELINE['depth']}.")
    except zipfile.BadZipFile:
        problems.append("independent model is not a valid .keras ZIP container.")
        probe = None

    # M7 independent theory implementation.
    for rel, digest in m7.get("source_hashes", {}).items():
        path = repo_root / rel
        if not path.is_file() or sha256_file(path) != digest:
            problems.append(f"M7 source {rel} is missing or differs from its recorded hash.")
    if m7.get("verdict") != "VERIFIED" or m7.get("failed_checks"):
        problems.append("M7 verdict is not VERIFIED with no failed checks.")

    rel = II2_EVIDENCE
    discrepancy = training["source_discrepancy"]["resolution"]
    components = (
        StageComponent(
            component_id="II2-INDEPENDENT-SOURCE", kind=ComponentKind.INDEPENDENT_IMPLEMENTATION,
            status=StageOutcome.COMPLETED,
            question="Does an independently written implementation exist, bound by hash?",
            summary="Independent Speck, dataset, model, training and theory modules, with their "
                    "verification drivers.",
            evidence={f"source:{p}": h for p, h in src_hashes.items()},
        ),
        StageComponent(
            component_id=m1["experiment_id"], kind=ComponentKind.DETERMINISTIC_EQUIVALENCE,
            status=StageOutcome.COMPLETED if m1.get("status") == "PASS" else StageOutcome.FAILED,
            question="Does the independent implementation agree with the reference on every "
                     "deterministic component?",
            summary=("PASS: published test vector; key schedule (1000 keys, 5000 round keys); "
                     "round traces (10000 word states); binary encoding; dataset construction; "
                     f"architecture ({m1['checks']['model_architecture']['conv1d_layers']} Conv1D, "
                     f"{m1['checks']['model_architecture']['residual_add_layers']} Add, "
                     f"{m1['checks']['model_architecture']['parameter_count']} parameters)."),
            evidence={rel["m1_deterministic_verification"]: ev_hashes[rel["m1_deterministic_verification"]]},
            limitations=["No training; establishes deterministic agreement only.",
                         m1["checks"]["model_architecture"]["paper_code_discrepancy"]],
        ),
        StageComponent(
            component_id=training["experiment_id"], kind=ComponentKind.INDEPENDENT_IMPLEMENTATION,
            status=StageOutcome.COMPLETED,
            question="Can the independent implementation be trained under the declared protocol?",
            summary=(f"One independent 200-epoch training run: fresh-test accuracy "
                     f"{training['results']['fresh_test_accuracy']:.4f} (reference-reported "
                     f"{DECLARED_BASELINE['reported_best_validation_accuracy']}); best validation "
                     f"epoch {training['results']['best_validation_epoch_zero_based']} (0-based)."),
            evidence={rel["m1_independent_training_report"]: ev_hashes[rel["m1_independent_training_report"]],
                      rel["m1_independent_model"]: ev_hashes[rel["m1_independent_model"]]},
            limitations=[
                "ONE training run with no recorded seed (seed = null): training-seed "
                "reproducibility is NOT established, and no run-to-run variance is available.",
                f"Paper-prose vs public-code BatchNormalization discrepancy: {discrepancy}",
                "The training report does not record the model artifact's hash; the model is "
                "bound to the report by location only.",
                "Agreement of this run's accuracy with the reference-reported figure is a single "
                "observation, not a comparison under identical protocols (that is II-4).",
            ],
        ),
        StageComponent(
            component_id="M7-" + m7["independent_implementation_id"],
            kind=ComponentKind.DETERMINISTIC_EQUIVALENCE,
            status=StageOutcome.COMPLETED if m7.get("verdict") == "VERIFIED" else StageOutcome.FAILED,
            question="Does an independent implementation of the differential theory agree with "
                     "the audited one?",
            summary=f"{m7['verdict']}: checks T1-T6 exact; independent algorithm: "
                    f"{m7['independent_algorithm']}.",
            evidence={rel["m7_theory_verification"]: ev_hashes[rel["m7_theory_verification"]]},
            limitations=list(m7.get("limitations", [])) + (
                ["M7 record has no git commit (git_commit = UNAVAILABLE)."]
                if m7.get("git_commit") == "UNAVAILABLE" else []),
        ),
    )
    status = StageOutcome.COMPLETED if not problems else StageOutcome.PARTIAL
    stage = StageStatus(
        StageId.II_2_INDEPENDENT_REIMPLEMENTATION, STAGE_TIERS[StageId.II_2_INDEPENDENT_REIMPLEMENTATION],
        status,
        ("Independent reimplementation performed and its evidence verified against its source "
         "hashes. Completion records that the reimplementation exists and was exercised; it is "
         "NOT evidence that it agrees with the reference implementation in experimental outcomes."
         if not problems else "II-2 evidence is inconsistent; see findings."),
        findings=problems, components=components,
    )
    return stage, {"evidence_hashes": {**ev_hashes, **{f"source:{p}": h for p, h in src_hashes.items()}},
                   "independent_probe": probe.to_dict() if probe else None,
                   "training": training, "problems": problems}


# --- CE reconciliation ------------------------------------------------------

def reconcile_ce(repo_root: Path, historical_probes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    For each downstream warning in the depth finding, verify (a) the
    historical certificate it refers to and (b) the current certificate
    that re-executed it, including that certificate's actual checkpoints.
    """
    out = {}
    for ref, current_rel in CURRENT_CE_CERTIFICATES.items():
        current = verify_ce_certificate(repo_root, current_rel)
        historical, missing = {}, []
        for rel in HISTORICAL_CE_CERTIFICATES[ref]:
            path = repo_root / rel
            if path.is_file():
                historical[rel] = sha256_file(path)
            else:
                missing.append(rel)
        out[ref] = {
            **current,
            "historical_certificates": historical,
            "historical_certificates_missing": missing,
            "historical_checkpoints": historical_probes,
            "historical_scope": (
                "Historical certificate preserved unchanged; it remains scoped to the realized "
                "depth-5 artifacts it was produced with. It records no checkpoint hash itself; "
                "the depth-5 binding rests on II-FINDING-DEPTH-V1's source inspection and the "
                "probe of those artifacts."),
            "verified": current["verified"] and not missing,
        }
    return out


# --- assembly -----------------------------------------------------------------

def assemble(repo_root: Path) -> tuple[RunManifest, dict[str, Any]]:
    """Verify all evidence and build the stage records; the Evidence is built in
    build_evidence once the run manifest's content hash is known."""
    started = utc_timestamp()
    artifact_hashes: dict[str, str] = {}

    # II-1
    refs_ok, ref_problems = verify_reference_files(repo_root / "audit/implementation/reference")
    artifact_hashes.update({f"reference/{n}": h for n, h in REFERENCE_FILE_HASHES.items()})
    ii1 = StageStatus(
        StageId.II_1_BASELINE_RECONSTRUCTION, STAGE_TIERS[StageId.II_1_BASELINE_RECONSTRUCTION],
        StageOutcome.CONFORMANCE_VERIFIED if refs_ok else StageOutcome.FAILED,
        ("Declared reference protocol transcribed from the original Gohr source (hash-verified): "
         f"rounds={DECLARED_BASELINE['rounds']}, depth={DECLARED_BASELINE['depth']}, "
         f"epochs={DECLARED_BASELINE['epochs']}, batch_size={DECLARED_BASELINE['batch_size']}."
         if refs_ok else f"Reference file verification failed: {ref_problems}"),
        findings=ref_problems,
    )

    # II-2
    ii2, ii2_info = assess_ii2(repo_root)
    artifact_hashes.update(ii2_info.get("evidence_hashes", {}))

    # II-3: architecture census + deterministic conformance evidence
    census: dict[str, Any] = {}
    historical_probes: dict[str, Any] = {}
    for name, rel in HISTORICAL_DEPTH5_CHECKPOINTS.items():
        probe = probe_checkpoint_architecture(repo_root / rel)
        historical_probes[name] = {"path": rel, "sha256": probe.sha256,
                                   "realized_depth": probe.derived_depth,
                                   "n_conv1d_layers": probe.n_conv1d_layers,
                                   "n_add_layers": probe.n_add_layers}
        census[rel] = probe
    ref_probe = probe_checkpoint_architecture(repo_root / REFERENCE_DEPTH10_CHECKPOINT)
    census[REFERENCE_DEPTH10_CHECKPOINT] = ref_probe
    census[II2_EVIDENCE["m1_independent_model"]] = probe_checkpoint_architecture(
        repo_root / II2_EVIDENCE["m1_independent_model"])
    ce = reconcile_ce(repo_root, historical_probes)
    for record in ce.values():
        artifact_hashes[record["certificate"]] = record["certificate_sha256"]
        artifact_hashes.update(record["historical_certificates"])
        for chk in record["checkpoints"]:
            artifact_hashes[chk["path"]] = chk["sha256"]
    for rel, probe in census.items():
        artifact_hashes[rel] = probe.sha256
    inconsistent = [rel for rel, p in census.items() if not p.is_consistent]
    census_lines = [f"{rel}: {p.n_conv1d_layers} Conv1D, {p.n_add_layers} Add -> depth {p.derived_depth}"
                    for rel, p in census.items()]
    n_ce_ckpts = len({c["sha256"] for r in ce.values() for c in r["checkpoints"]})
    ii3_components = (
        StageComponent(
            component_id="II3-DETERMINISTIC-CONFORMANCE", kind=ComponentKind.DETERMINISTIC_EQUIVALENCE,
            status=StageOutcome.COMPLETED if not ii2_info["problems"] else StageOutcome.PARTIAL,
            question="Do the deterministic components of the reference and independent "
                     "implementations agree exactly?",
            summary="Bound II-2 deterministic evidence (M1 cipher/dataset/architecture checks, M7 "
                    "theory checks), all PASS/VERIFIED.",
            evidence={k: v for k, v in ii2_info["evidence_hashes"].items()
                      if k in (II2_EVIDENCE["m1_deterministic_verification"],
                               II2_EVIDENCE["m7_theory_verification"])},
            limitations=["Deterministic agreement does not imply equal trained-model behaviour."],
        ),
        StageComponent(
            component_id="II3-ARCHITECTURE-CENSUS", kind=ComponentKind.ARCHITECTURE_CENSUS,
            status=StageOutcome.COMPLETED if not inconsistent else StageOutcome.FAILED,
            question="What architecture does each artifact actually realize?",
            summary=(f"Read from the artifacts (not filenames): the 2 historical CE1 artifacts "
                     f"realize depth 5; the reference depth-10 checkpoint, the independent model, "
                     f"and all {n_ce_ckpts} distinct checkpoints cited by the current CE1-CE4 "
                     "certificates realize depth 10."),
            evidence={rel: p.sha256 for rel, p in census.items()},
            limitations=["A census establishes structure only. It does NOT establish that two "
                         "artifacts of equal structure behave equally in experiments."],
        ),
    )
    ii3 = StageStatus(
        StageId.II_3_CONTROLLED_VERIFICATION, STAGE_TIERS[StageId.II_3_CONTROLLED_VERIFICATION],
        StageOutcome.CONFORMANCE_VERIFIED if refs_ok and not inconsistent else StageOutcome.FAILED,
        ("Conformance evidence: reference sources hash-verified (II-1); deterministic agreement of "
         "the independent implementation with the reference (M1, M7); architecture census of every "
         "cited checkpoint. Confirms the historical depth-5 discrepancy and that the current CE "
         "evidence uses depth-10 artifacts. Structural and deterministic conformance only - not "
         "evidence of equal experimental behaviour, which is the subject of II-4."),
        findings=census_lines + [f"inconsistent census: {rel}" for rel in inconsistent],
        components=ii3_components,
    )

    # II-4 / II-5 from the bound, frozen depth experiment
    bound = repo_root / DEPTH_EXPERIMENT_BOUND_DIR
    original = repo_root / DEPTH_EXPERIMENT_ORIGINAL_DIR
    depth_hashes: dict[str, str] = {}
    for name in ("ii4_preregistration.json", "ii4_analysis.json", "ii4_state.json", "ii4_firewall.json"):
        digest = sha256_file(bound / name)
        if (original / name).is_file() and sha256_file(original / name) != digest:
            raise RuntimeError(f"bound copy of {name} differs from {DEPTH_EXPERIMENT_ORIGINAL_DIR}.")
        depth_hashes[f"{DEPTH_EXPERIMENT_BOUND_DIR}/{name}"] = digest
    prereg = json.loads((bound / "ii4_preregistration.json").read_text())
    report = json.loads((bound / "ii4_analysis.json").read_text())
    if report["experiment_id"] != DEPTH_EXPERIMENT_ID or \
            report["config_fingerprint"] != prereg["experiment_fingerprint"]:
        raise RuntimeError("bound depth-experiment artifacts are inconsistent.")
    depth_manifest_path = bound / f"run_manifest_{DEPTH_EXPERIMENT_ID}-run.json"
    depth_manifest = json.loads(depth_manifest_path.read_text())
    depth_manifest_ref = {"run_id": depth_manifest["run_id"],
                          "artifact_path": f"{DEPTH_EXPERIMENT_BOUND_DIR}/{depth_manifest_path.name}",
                          "content_hash": sha256_file(depth_manifest_path)}
    artifact_hashes.update(depth_hashes)
    for entry in prereg.get("source_manifest", {}).values():
        depth_hashes[f"source:{entry['path']}"] = entry["sha256"]

    finding, clause, depth_ii4, depth_ii5 = conformance_impact_result(
        prereg, report, depth_hashes, finding_id=DEPTH_FINDING_ID)
    finding = dataclasses.replace(finding, downstream=reconcile_downstream(finding.downstream, ce))

    fixed = FIXED_CHECKPOINT_COMPARISON_PROPOSAL
    fixed_component = StageComponent(
        component_id=fixed["proposed_experiment_id"],
        kind=ComponentKind.FIXED_CHECKPOINT_IMPLEMENTATION_COMPARISON,
        status=StageOutcome.NOT_PERFORMED, question=fixed["question"],
        summary=f"NOT PERFORMED (feasibility assessed). {fixed['feasibility']}",
        evidence={REFERENCE_DEPTH10_CHECKPOINT: ref_probe.sha256,
                  II2_EVIDENCE["m1_independent_model"]:
                      census[II2_EVIDENCE["m1_independent_model"]].sha256},
        limitations=list(fixed["limitations"]),
    )
    formal = FORMAL_COMPARISON_PROPOSAL
    formal_component = StageComponent(
        component_id=formal["proposed_experiment_id"],
        kind=ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON,
        status=StageOutcome.NOT_PERFORMED, question=formal["question"],
        summary="NOT PERFORMED. Requires replicated training of both implementations under one "
                "frozen protocol; a design is proposed (not preregistered) in "
                "observation.outstanding_empirical_work.",
        limitations=list(formal["limitations"]),
    )
    ii4 = StageStatus(
        StageId.II_4_COMPARATIVE_EVALUATION, STAGE_TIERS[StageId.II_4_COMPARATIVE_EVALUATION],
        StageOutcome.PARTIAL,
        (f"PARTIAL. Completed: {DEPTH_EXPERIMENT_ID}, a conformance-impact experiment (declared "
         "depth 10 vs realized depth 5, both arms trained with the same implementation) - "
         f"{depth_ii4.summary} It answers whether the depth discrepancy matters, not whether the "
         f"reference and independent implementations agree. {OUTSTANDING} A fixed-checkpoint "
         "comparison of the existing depth-10 artifacts is feasible but not performed."),
        components=(depth_ii4, fixed_component, formal_component),
    )
    formal_stats = StageComponent(
        component_id=f"{formal['proposed_experiment_id']}-statistics",
        kind=ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON,
        status=StageOutcome.NOT_PERFORMED,
        question="Statistical assessment of the formal reference-vs-independent comparison.",
        summary=f"NOT PERFORMED: no data exist. Proposed method: {formal['statistical_method']}.",
    )
    ii5 = StageStatus(
        StageId.II_5_STATISTICAL_ASSESSMENT, STAGE_TIERS[StageId.II_5_STATISTICAL_ASSESSMENT],
        StageOutcome.PARTIAL,
        (f"PARTIAL. Completed for {DEPTH_EXPERIMENT_ID} only: {depth_ii5.summary} "
         f"No statistical assessment of the reference-vs-independent comparison exists. {OUTSTANDING}"),
        components=(depth_ii5, formal_stats),
    )

    # Every factual claim in the finding must match what was inspected now.
    observed = {name: rec["sha256"] for name, rec in historical_probes.items()}
    observed.update(depth_hashes)
    depths = {rec["realized_depth"] for rec in historical_probes.values()}
    convs = {rec["n_conv1d_layers"] for rec in historical_probes.values()}
    values = {}
    if len(depths) == 1:
        values["depth"] = depths.pop()
    if len(convs) == 1:
        values["n_conv1d_layers"] = convs.pop()
    binding = verify_finding_against_observations(
        finding, observed_artifact_hashes=observed, observed_values=values,
        verifiable_keys=("depth", "n_conv1d_layers"))
    if binding:
        raise RuntimeError(f"finding/observation binding failed: {binding}")

    # Predecessor, preserved as historical evidence.
    predecessor_path = repo_root / "audit/implementation/evidence" / f"{PREDECESSOR}.json"
    predecessor = json.loads(predecessor_path.read_text())
    artifact_hashes[f"audit/implementation/evidence/{PREDECESSOR}.json"] = sha256_file(predecessor_path)

    code_paths = ["audit/implementation/run_implementation_audit.py", "audit/implementation/stages.py",
                  "audit/implementation/certificate.py", "audit/implementation/findings.py",
                  "audit/implementation/gohr_adapter.py", "audit/implementation/ii4_certificate.py"]
    manifest = RunManifest(
        run_id=RUN_ID, experiment_id=EXPERIMENT_ID, condition_id="five_stage_assembly",
        replicate_id=None, claim_ids=["C1"], hypothesis_ids=["H-II-CONFORMANCE"],
        execution_mode=ExecutionMode.PRODUCTION,
        config_hash=sha256_bytes(dumps_strict(DECLARED_BASELINE, sort_keys=True, default=str).encode()),
        # No pre-execution preregistration exists for this assembly; see EXPLORATORY tag.
        preregistration_hash=None, dataset_snapshot_id=None, dataset_hashes={},
        code_hashes={p: sha256_file(repo_root / p) for p in code_paths},
        model_source_hash=None, environment=capture_environment(), seeds={},
        command="python3 -m audit.implementation.run_implementation_audit", start_time_utc=started,
    )
    manifest.mark_completed(exit_code=0, metrics={
        "ce_certificates_verified": sum(r["verified"] for r in ce.values()),
        "checkpoints_probed": len(census) + sum(r["n_checkpoints"] for r in ce.values()),
    })

    context = {
        "manifest_payload": dumps_strict(manifest.to_dict(), indent=2, sort_keys=True),
        "depth_manifest_ref": depth_manifest_ref, "depth_manifest": depth_manifest,
        "ce": ce, "artifact_hashes": artifact_hashes, "finding": finding, "clause": clause,
        "stages": [ii1, ii2, ii3, ii4, ii5], "ii2_problems": ii2_info["problems"],
        "predecessor": predecessor, "depth_fingerprint": prereg["experiment_fingerprint"],
    }
    return manifest, context


def build_evidence(manifest_ref: dict, manifest: RunManifest, ctx: dict) -> Evidence:
    ce, finding = ctx["ce"], ctx["finding"]
    ii2_completed = ctx["stages"][1].status is StageOutcome.COMPLETED
    predecessor = ctx["predecessor"]
    return build_implementation_evidence(
        evidence_id=EVIDENCE_ID, claim_id="C1", experiment_id=EXPERIMENT_ID,
        run_ids=[RUN_ID, ctx["depth_manifest_ref"]["run_id"]],
        stages=ctx["stages"], findings=[finding],
        independent_reimplementation_status=(
            EnhancedAssessmentStatus.ASSESSED if ii2_completed else EnhancedAssessmentStatus.NOT_ASSESSED),
        provenance={"runner": "audit.implementation.run_implementation_audit",
                    "supersedes": PREDECESSOR,
                    "depth_experiment_fingerprint": ctx["depth_fingerprint"]},
        run_manifest_references=[manifest_ref, ctx["depth_manifest_ref"]],
        additional_artifact_hashes=ctx["artifact_hashes"],
        # The pillar assembly was not preregistered; only its depth-experiment component was.
        preregistration_reference=None,
        statistics_note=(
            f"II-5 is PARTIAL. Statistics exist for {DEPTH_EXPERIMENT_ID} only (depth 10 vs 5 within "
            "one implementation). No statistical assessment of the reference-vs-independent "
            "implementation comparison exists, because that comparison has not been run."),
        additional_observation={
            "methodology": {
                "section": "3.5.4",
                "stages": [s.value for s in StageId],
                "note": "All five stages are CORE. No methodology revision making II-2 optional "
                        "exists in the repository; V1-V5 recorded II-2 as ENHANCED.",
            },
            "supersedes": {
                "certificate": f"{PREDECESSOR}.json",
                "sha256": ctx["artifact_hashes"][f"audit/implementation/evidence/{PREDECESSOR}.json"],
                "predecessor_decision": predecessor["decision"],
                "preserved_unchanged": True,
                "corrections": [
                    "II-2 is CORE (was ENHANCED) and is COMPLETED with bound evidence.",
                    "II-4 and II-5 are PARTIAL (were COMPLETED): the depth experiment does not "
                    "compare the reference and independent implementations.",
                    "Downstream CE re-evaluation warnings are cleared one by one, only where the "
                    "current CE certificate and its checkpoints were verified at depth 10.",
                    "The pillar decision is INCONCLUSIVE (was CONDITIONAL_PASS) while the formal "
                    "comparison is outstanding; the depth experiment's own preregistered "
                    "interpretation is reported separately as conformance_impact_outcome.",
                ],
            },
            "conformance_impact_outcome": ctx["clause"]["ii_outcome"],
            "implementation_comparison": {
                "conformance_impact_experiment": {"experiment_id": DEPTH_EXPERIMENT_ID,
                                                  "status": "COMPLETED",
                                                  "answers_formal_ii4": False},
                "fixed_checkpoint_comparison": {"status": "NOT_PERFORMED", "feasible": True},
                "formal_comparison": {"status": "NOT_PERFORMED"},
            },
            "ce_reconciliation": ce,
            "outstanding_empirical_work": {
                "fixed_checkpoint_comparison": FIXED_CHECKPOINT_COMPARISON_PROPOSAL,
                "formal_comparison": FORMAL_COMPARISON_PROPOSAL,
            },
        },
    )


def run(repo_root: Path, output_dir: Path, *, write: bool) -> int:
    manifest, ctx = assemble(repo_root)
    manifest_name = f"run_manifest_{RUN_ID}.json"
    manifest_path = output_dir / manifest_name
    if write and manifest_path.exists():
        print(f"refusing to overwrite existing {manifest_path}")
        return 1
    manifest_ref = {"run_id": RUN_ID, "artifact_path": manifest_name,
                    "content_hash": sha256_bytes(ctx["manifest_payload"].encode("utf-8"))}
    evidence = build_evidence(manifest_ref, manifest, ctx)
    problems = validate_evidence_provenance(evidence, known_run_manifests={
        RUN_ID: manifest, ctx["depth_manifest"]["run_id"]: ctx["depth_manifest"]})
    if problems:
        print("EVIDENCE PROVENANCE VALIDATION FAILED - refusing to write:", *problems, sep="\n  - ")
        return 1
    payload = dumps_strict(evidence.to_dict(), indent=2, sort_keys=True)

    for stage in evidence.observation["stages"]:
        print(f"{stage['stage_id']:<36} {stage['tier']:<5} {stage['status']}")
    for ref, rec in ctx["ce"].items():
        print(f"CE {ref}: verified={rec['verified']} depths={rec['realized_depths']} "
              f"checkpoints={rec['n_checkpoints']}")
    print(f"Decision: {evidence.decision}   tag: {evidence.tag.value}   "
          f"conformance_impact_outcome: {ctx['clause']['ii_outcome']}")
    print(f"Downstream still requiring re-evaluation: "
          f"{evidence.observation['downstream_requires_reevaluation']}")
    if not write:
        print("Dry run: nothing written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(ctx["manifest_payload"])
    registry = VersionRegistry(output_dir / "version_registry.json")
    record = registry.register(
        "implementation-certificate", sha256_bytes(payload.encode("utf-8")),
        note=f"Five-stage Implementation Integrity certificate; supersedes {PREDECESSOR} "
             "(preserved).")
    out_path = output_dir / f"{record.identifier}.json"
    if out_path.exists():
        print(f"refusing to overwrite existing {out_path}")
        return 1
    out_path.write_text(payload)
    print(f"Run manifest: {manifest_path}")
    print(f"Versioned evidence written: {out_path} ({record.identifier})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "audit/implementation/evidence")
    parser.add_argument("--dry-run", action="store_true", help="assemble and verify; write nothing")
    args = parser.parse_args()
    return run(args.repo_root.resolve(), args.output_dir, write=not args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
