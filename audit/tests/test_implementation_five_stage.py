"""
Five-stage Implementation Integrity: stage classification, evidence
binding, the distinction between a conformance-impact (depth) experiment
and the formal reference-vs-independent comparison, CE reconciliation,
and the assembled successor certificate.
"""

import json
import zipfile
from pathlib import Path

import pytest

from audit.common.ids import Dimension
from audit.common.outcomes import AssessmentTier, DimensionOutcome, EnhancedAssessmentStatus
from audit.common.provenance import sha256_file
from audit.implementation.certificate import build_implementation_evidence
from audit.implementation.findings import (
    DEPTH_FINDING_V1,
    DownstreamDependency,
    DownstreamStatus,
    reconcile_downstream,
)
from audit.implementation.gohr_adapter import (
    probe_checkpoint_architecture,
    verify_ce_certificate,
)
from audit.implementation.stages import (
    CORE_STAGES,
    STAGE_TIERS,
    ComponentKind,
    StageComponent,
    StageId,
    StageOutcome,
    StageStatus,
    convergence_contribution,
    validate_stage_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
H = "a" * 64


def _stage(sid, status, *components):
    return StageStatus(sid, STAGE_TIERS[sid], status, "s", components=tuple(components))


def _component(kind, status=StageOutcome.COMPLETED, evidence=None):
    return StageComponent(f"c-{kind.value}", kind, status, "q", "s", evidence=evidence or {})


DEPTH = _component(ComponentKind.CONFORMANCE_IMPACT_EXPERIMENT, evidence={"ii4_analysis.json": H})
FIXED = _component(ComponentKind.FIXED_CHECKPOINT_IMPLEMENTATION_COMPARISON)
FORMAL = _component(ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON)


# --- stage classification -------------------------------------------------

@pytest.mark.parametrize("sid", [StageId.II_4_COMPARATIVE_EVALUATION, StageId.II_5_STATISTICAL_ASSESSMENT])
@pytest.mark.parametrize("components", [(), (DEPTH,), (FIXED,), (DEPTH, FIXED)])
def test_ii4_ii5_cannot_be_completed_without_formal_comparison(sid, components):
    assert validate_stage_record(_stage(sid, StageOutcome.COMPLETED, *components))


@pytest.mark.parametrize("sid", [StageId.II_4_COMPARATIVE_EVALUATION, StageId.II_5_STATISTICAL_ASSESSMENT])
def test_ii4_ii5_completed_by_formal_comparison(sid):
    assert validate_stage_record(_stage(sid, StageOutcome.COMPLETED, DEPTH, FORMAL)) == []


def test_formal_component_not_completed_does_not_count():
    pending = _component(ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON, StageOutcome.NOT_PERFORMED)
    assert validate_stage_record(
        _stage(StageId.II_4_COMPARATIVE_EVALUATION, StageOutcome.COMPLETED, DEPTH, pending))


def test_depth_experiment_records_partial_ii4_validly():
    stage = _stage(StageId.II_4_COMPARATIVE_EVALUATION, StageOutcome.PARTIAL, DEPTH, FIXED)
    assert validate_stage_record(stage) == []
    assert stage.is_satisfied is False


def test_ii2_completed_requires_bound_evidence():
    bare = _component(ComponentKind.INDEPENDENT_IMPLEMENTATION)
    assert validate_stage_record(_stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
                                        StageOutcome.COMPLETED, bare))
    bound = _component(ComponentKind.INDEPENDENT_IMPLEMENTATION, evidence={"m1.json": H})
    assert validate_stage_record(_stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
                                        StageOutcome.COMPLETED, bound)) == []


def test_ii2_recorded_with_enhanced_tier_is_rejected():
    stage = StageStatus(StageId.II_2_INDEPENDENT_REIMPLEMENTATION, AssessmentTier.ENHANCED,
                        StageOutcome.NOT_ASSESSED, "legacy")
    assert any("tier" in p for p in validate_stage_record(stage))


def test_evidence_without_sha256_is_rejected():
    bad = _component(ComponentKind.INDEPENDENT_IMPLEMENTATION, evidence={"m1.json": "abc"})
    assert validate_stage_record(_stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION,
                                        StageOutcome.COMPLETED, bad))


# --- certificate generation -------------------------------------------------

def _stages(ii4_status, *ii4_components):
    ii2 = _component(ComponentKind.INDEPENDENT_IMPLEMENTATION, evidence={"m1.json": H})
    return [
        _stage(StageId.II_1_BASELINE_RECONSTRUCTION, StageOutcome.CONFORMANCE_VERIFIED),
        _stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION, StageOutcome.COMPLETED, ii2),
        _stage(StageId.II_3_CONTROLLED_VERIFICATION, StageOutcome.CONFORMANCE_VERIFIED),
        _stage(StageId.II_4_COMPARATIVE_EVALUATION, ii4_status, *ii4_components),
        _stage(StageId.II_5_STATISTICAL_ASSESSMENT, ii4_status, *ii4_components),
    ]


def _build(stages):
    return build_implementation_evidence(
        evidence_id="EV-T", claim_id="C1", experiment_id="X", run_ids=["r"], stages=stages,
        findings=[], independent_reimplementation_status=EnhancedAssessmentStatus.ASSESSED,
        provenance={})


def test_builder_refuses_ii4_completed_by_depth_experiment():
    with pytest.raises(ValueError, match="FORMAL_IMPLEMENTATION_COMPARISON"):
        _build(_stages(StageOutcome.COMPLETED, DEPTH))


def test_depth_only_certificate_is_inconclusive_and_offers_no_convergence():
    ev = _build(_stages(StageOutcome.PARTIAL, DEPTH))
    assert ev.decision == DimensionOutcome.INCONCLUSIVE.value
    assert set(ev.observation["core_stages_unsatisfied"]) == {
        "II-4_COMPARATIVE_EVALUATION", "II-5_STATISTICAL_ASSESSMENT"}
    assert ev.observation["convergence_contribution"]["contributes"] is False


def test_formal_comparison_completed_allows_verdict_and_convergence():
    ev = _build(_stages(StageOutcome.COMPLETED, DEPTH, FORMAL))
    assert ev.decision == DimensionOutcome.PASS.value
    assert ev.observation["convergence_contribution"]["contributes"] is True


def test_component_limitations_surface_in_certificate():
    limited = StageComponent("c-ii2", ComponentKind.INDEPENDENT_IMPLEMENTATION, StageOutcome.COMPLETED,
                             "q", "s", evidence={"m1.json": H}, limitations=["seed not recorded"])
    stages = _stages(StageOutcome.PARTIAL, DEPTH)
    stages[1] = _stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION, StageOutcome.COMPLETED, limited)
    ev = _build(stages)
    assert "II-2_INDEPENDENT_REIMPLEMENTATION/c-ii2: seed not recorded" in ev.limitations


def test_additional_observation_cannot_overwrite_computed_keys():
    with pytest.raises(ValueError, match="overwrite"):
        build_implementation_evidence(
            evidence_id="EV-T", claim_id="C1", experiment_id="X", run_ids=["r"],
            stages=_stages(StageOutcome.PARTIAL, DEPTH), findings=[],
            independent_reimplementation_status=EnhancedAssessmentStatus.ASSESSED,
            provenance={}, additional_observation={"core_stages_satisfied": True})


def test_convergence_explicit_false_comparison_withholds_contribution():
    c = convergence_contribution(independent_reimplementation_status=EnhancedAssessmentStatus.ASSESSED,
                                 implementation_comparison_completed=False)
    assert c.contributes is False and "not established" in c.rationale


# --- downstream reconciliation ---------------------------------------------------

def test_reconcile_clears_only_verified_warnings():
    successors = {
        "evidence/ce1/ce1_certificate.json": {"certificate": "new_ce1.json", "verified": True},
        "evidence/ce3/ce3_certificate.json": {"certificate": "new_ce3.json", "verified": False},
    }
    out = {d.evidence_reference: d for d in reconcile_downstream(DEPTH_FINDING_V1.downstream, successors)}
    assert out["evidence/ce1/ce1_certificate.json"].status is DownstreamStatus.REEVALUATED
    assert out["evidence/ce1/ce1_certificate.json"].successor["certificate"] == "new_ce1.json"
    assert out["evidence/ce3/ce3_certificate.json"].status is DownstreamStatus.REQUIRES_REEVALUATION
    assert out["evidence/ce4/ce4_certificate.json"].status is DownstreamStatus.REQUIRES_REEVALUATION
    assert out["dataset/evidence/d1..d5"].status is DownstreamStatus.UNAFFECTED
    # the historical finding object is untouched
    assert all(d.status is not DownstreamStatus.REEVALUATED for d in DEPTH_FINDING_V1.downstream)


def test_reevaluated_requires_successor_record():
    with pytest.raises(ValueError):
        DownstreamDependency("x", Dimension.CRYPTOGRAPHIC, DownstreamStatus.REEVALUATED, "r")


# --- CE certificate verification against real artifact bytes -----------------------

def _fake_keras(path, depth):
    layers = [{"class_name": "Conv1D"} for _ in range(1 + 2 * depth)]
    layers += [{"class_name": "Add"} for _ in range(depth)] + [{"class_name": "Dense"}] * 3
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps({"config": {"layers": layers}}))
    return path


def _ce_cert(root, model, sha, depth=10):
    cert = {"experiment_id": "CE2", "run_id": "r", "results": {"decision": "SUPPORTED"},
            "reference_model": {"path": str(model.relative_to(root)), "sha256": sha,
                                "realized_depth": depth,
                                "properties": {"conv1d_layers": 1 + 2 * depth, "residual_merges": depth}}}
    (root / "cert.json").write_text(json.dumps(cert))
    return "cert.json"


def test_probe_reads_keras_zip_architecture(tmp_path):
    probe = probe_checkpoint_architecture(_fake_keras(tmp_path / "m.keras", 10))
    assert (probe.n_conv1d_layers, probe.n_add_layers, probe.derived_depth) == (21, 10, 10)
    assert probe.is_consistent


def test_ce_certificate_verified_when_hash_and_architecture_match(tmp_path):
    model = _fake_keras(tmp_path / "m.keras", 10)
    rec = verify_ce_certificate(tmp_path, _ce_cert(tmp_path, model, sha256_file(model)))
    assert rec["verified"] is True and rec["realized_depths"] == [10]


def test_ce_certificate_not_verified_on_hash_mismatch(tmp_path):
    model = _fake_keras(tmp_path / "m.keras", 10)
    rec = verify_ce_certificate(tmp_path, _ce_cert(tmp_path, model, "0" * 64))
    assert rec["verified"] is False and any("sha256" in p for p in rec["problems"])


def test_ce_certificate_not_verified_when_artifact_is_depth5(tmp_path):
    """The certificate claims depth 10, but the file says otherwise: the file wins."""
    model = _fake_keras(tmp_path / "m.keras", 5)
    rec = verify_ce_certificate(tmp_path, _ce_cert(tmp_path, model, sha256_file(model)))
    assert rec["verified"] is False and rec["realized_depths"] == [5]


# --- the assembled certificate on the real repository evidence ---------------------

_NEEDED = ["audit/implementation/evidence/ii2/m1_independent_best.zip",
           "audit/cryptography/Archive/best5depth10.h5",
           "audit/implementation/evidence/ii4_bound/7893611dffa86656/ii4_analysis.json",
           "audit/cryptography/evidence_current/ce2/production_20261003/certificatece2.json"]


@pytest.fixture(scope="module")
def assembled():
    if not all((REPO_ROOT / p).is_file() for p in _NEEDED):
        pytest.skip("repository evidence artifacts not present")
    pytest.importorskip("h5py")
    import audit.implementation.run_implementation_audit as runner
    manifest, ctx = runner.assemble(REPO_ROOT)
    ref = {"run_id": runner.RUN_ID, "artifact_path": "m.json", "content_hash": H}
    return ctx, runner.build_evidence(ref, manifest, ctx)


def test_assembled_stage_statuses(assembled):
    _, ev = assembled
    stages = {s["stage_id"]: s for s in ev.observation["stages"]}
    assert [s["tier"] for s in ev.observation["stages"]] == ["CORE"] * 5
    assert {k: v["status"] for k, v in stages.items()} == {
        "II-1_BASELINE_RECONSTRUCTION": "CONFORMANCE_VERIFIED",
        "II-2_INDEPENDENT_REIMPLEMENTATION": "COMPLETED",
        "II-3_CONTROLLED_VERIFICATION": "CONFORMANCE_VERIFIED",
        "II-4_COMPARATIVE_EVALUATION": "PARTIAL",
        "II-5_STATISTICAL_ASSESSMENT": "PARTIAL",
    }
    assert ev.decision == "INCONCLUSIVE"


def test_assembled_ii2_binds_actual_evidence_and_keeps_limitations(assembled):
    _, ev = assembled
    for rel in ("audit/implementation/evidence/ii2/m1_deterministic_verification.json",
                "audit/implementation/evidence/ii2/m1_independent_training_report.json",
                "audit/implementation/evidence/ii2/m1_independent_best.zip",
                "audit/implementation/evidence/m7/m7_theory_verification.json"):
        assert ev.artifact_hashes[rel] == sha256_file(REPO_ROOT / rel)
    joined = " ".join(ev.limitations)
    assert "no recorded seed" in joined and "NOT established" in joined
    assert "BatchNormalization" in joined


def test_assembled_ii4_separates_depth_experiment_from_formal_comparison(assembled):
    _, ev = assembled
    ii4 = next(s for s in ev.observation["stages"] if s["stage_id"] == "II-4_COMPARATIVE_EVALUATION")
    by_kind = {c["kind"]: c for c in ii4["components"]}
    assert by_kind["CONFORMANCE_IMPACT_EXPERIMENT"]["component_id"] == "II-4-GOHR-DEPTH-V1"
    assert by_kind["CONFORMANCE_IMPACT_EXPERIMENT"]["status"] == "COMPLETED"
    assert by_kind["FIXED_CHECKPOINT_IMPLEMENTATION_COMPARISON"]["status"] == "NOT_PERFORMED"
    assert by_kind["FORMAL_IMPLEMENTATION_COMPARISON"]["status"] == "NOT_PERFORMED"
    assert ev.observation["implementation_comparison"]["conformance_impact_experiment"][
        "answers_formal_ii4"] is False


def test_assembled_ce_reconciliation_verifies_depth10_checkpoints(assembled):
    _, ev = assembled
    ce = ev.observation["ce_reconciliation"]
    assert all(r["verified"] and r["realized_depths"] == [10] for r in ce.values())
    assert ce["evidence/ce2/ce2_certificate_{1..5}.json"]["distinct_checkpoint_hashes"] == [
        "256eb4a5ba93414f0a46ffd498d6121a03705698ac2a81764c2375207c5d6737"]
    assert ev.observation["downstream_requires_reevaluation"] == []


def test_historical_evidence_is_untouched(assembled):
    ctx, ev = assembled
    v5 = REPO_ROOT / "audit/implementation/evidence/implementation-certificate-V5.json"
    assert sha256_file(v5) == "0ce25b3b8850032ef7376d557e5b8b0551af84e0ea4d0117f0ebf171c28bcc26"
    assert ev.observation["supersedes"]["preserved_unchanged"] is True
    prereg = REPO_ROOT / "audit/implementation/evidence/ii4/ii4_preregistration.json"
    assert sha256_file(prereg) == "77b757809f0d525b3ac40a273bd729cd53fbb13c70037d1c69ed90310ab1c5a4"
    # historical depth-5 artifacts are still what V1 says they are
    for rec in ctx["ce"]["evidence/ce1/ce1_certificate.json"]["historical_checkpoints"].values():
        assert rec["realized_depth"] == 5
