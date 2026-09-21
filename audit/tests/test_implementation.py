import numpy as np
import pytest

from audit.common.ids import Dimension
from audit.common.outcomes import (
    AssessmentTier,
    DimensionOutcome,
    EnhancedAssessmentStatus,
)
from audit.implementation.certificate import (
    build_implementation_evidence,
    decide_implementation_outcome,
)
from audit.implementation.equivalence import (
    check_exact_equivalence,
    check_numeric_equivalence,
    compare_declared_vs_actual,
)
import dataclasses

from audit.implementation.findings import (
    ALL_FINDINGS,
    CE3_CLARITY_FINDING_V1,
    DEPTH_FINDING_V1,
    DownstreamStatus,
    FindingSeverity,
    ScientificImpactStatus,
)
from audit.implementation.gohr_adapter import (
    DECLARED_BASELINE,
    depth_from_conv1d_count,
    implied_conv1d_count,
)
from audit.implementation.stages import (
    CORE_STAGES,
    ENHANCED_STAGES,
    STAGE_TIERS,
    StageId,
    StageOutcome,
    StageStatus,
    convergence_contribution,
    not_assessed_stage,
)


# --- CORE vs ENHANCED tiering (methodology revision) ---

def test_independent_reimplementation_is_enhanced_not_core():
    assert STAGE_TIERS[StageId.II_2_INDEPENDENT_REIMPLEMENTATION] is AssessmentTier.ENHANCED
    assert StageId.II_2_INDEPENDENT_REIMPLEMENTATION not in CORE_STAGES
    assert StageId.II_2_INDEPENDENT_REIMPLEMENTATION in ENHANCED_STAGES


def test_four_core_stages_are_exactly_the_revised_set():
    assert set(CORE_STAGES) == {
        StageId.II_1_BASELINE_RECONSTRUCTION,
        StageId.II_3_CONTROLLED_VERIFICATION,
        StageId.II_4_COMPARATIVE_EVALUATION,
        StageId.II_5_STATISTICAL_ASSESSMENT,
    }


def test_not_assessed_ii2_contributes_no_convergence_but_imposes_no_cap():
    """
    CORRECTED: II-2 NOT_ASSESSED must NOT act as a universal cap on
    LEVEL_4. It only means this pillar supplies no convergence source;
    the integration engine decides LEVEL_4 from actual convergence
    evidence across all pillars.
    """
    contribution = convergence_contribution(
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
    )
    assert contribution.contributes is False
    assert contribution.source_kind is None
    # It must explicitly disclaim being a cap.
    assert "LEVEL_4 remains determinable" in contribution.rationale


def test_assessed_ii2_contributes_independent_implementation_convergence():
    contribution = convergence_contribution(
        independent_reimplementation_status=EnhancedAssessmentStatus.ASSESSED,
    )
    assert contribution.contributes is True
    assert contribution.source_kind == "independent_implementation"


def test_no_evidence_level_cap_api_remains():
    """The universal-cap API must be gone, not merely unused."""
    import audit.implementation.stages as stages_module
    assert not hasattr(stages_module, "max_supportable_evidence_level")


def test_not_assessed_stage_is_explicit_never_silent():
    stage = not_assessed_stage(StageId.II_2_INDEPENDENT_REIMPLEMENTATION, reason="not performed")
    assert stage.status is StageOutcome.NOT_ASSESSED
    assert stage.tier is AssessmentTier.ENHANCED


# --- equivalence primitives ---

def test_exact_equivalence_detects_bit_differences():
    a = np.array([0, 1, 1, 0], dtype=np.uint8)
    b = np.array([0, 1, 0, 0], dtype=np.uint8)
    result = check_exact_equivalence(a, b, label="bits")
    assert result.equivalent is False
    assert result.n_mismatched == 1
    assert result.first_mismatch_index == 2


def test_exact_equivalence_passes_on_identical_arrays():
    a = np.array([1, 2, 3])
    assert check_exact_equivalence(a, a.copy()).equivalent is True


def test_exact_equivalence_handles_shape_mismatch_without_crashing():
    result = check_exact_equivalence(np.zeros(4), np.zeros(5))
    assert result.equivalent is False
    assert "shape mismatch" in result.detail


def test_numeric_equivalence_respects_preregistered_tolerance():
    a = np.array([0.1, 0.2, 0.3])
    b = a + 1e-7
    tight = check_numeric_equivalence(a, b, absolute_tolerance=1e-9)
    loose = check_numeric_equivalence(a, b, absolute_tolerance=1e-6)
    assert tight.within_tolerance is False
    assert loose.within_tolerance is True


def test_numeric_equivalence_reports_decision_agreement():
    a = np.array([0.4, 0.6, 0.45])
    b = np.array([0.45, 0.65, 0.55])  # last one crosses the 0.5 boundary
    result = check_numeric_equivalence(a, b, absolute_tolerance=0.2, decision_threshold=0.5)
    assert result.agreement_rate == pytest.approx(2 / 3)


def test_compare_declared_vs_actual_treats_missing_key_as_mismatch():
    result = compare_declared_vs_actual({"depth": 10, "rounds": 5}, {"rounds": 5})
    assert result.equivalent is False
    assert "depth" in result.mismatched_keys


# --- Gohr architecture arithmetic ---

def test_implied_conv1d_count_matches_reference_architecture():
    assert implied_conv1d_count(10) == 21
    assert implied_conv1d_count(5) == 11


def test_depth_derivation_from_observed_layer_count():
    """The actual observed value from the real checkpoints was 11 -> depth 5."""
    assert depth_from_conv1d_count(11) == 5
    assert depth_from_conv1d_count(21) == 10
    assert depth_from_conv1d_count(0) is None
    assert depth_from_conv1d_count(10) is None  # even count is not of form 1+2d


def test_declared_baseline_matches_reference_source():
    assert DECLARED_BASELINE["rounds"] == 5
    assert DECLARED_BASELINE["depth"] == 10
    assert DECLARED_BASELINE["batch_size"] == 5000
    assert DECLARED_BASELINE["epochs"] == 200


# --- decision rules (methodology 3.5.6) ---

def test_material_but_undemonstrated_impact_yields_inconclusive_not_fail():
    """
    CORRECTED (Rule III): a CONFIRMED discrepancy whose scientific
    impact is UNDEMONSTRATED must NOT be a FAIL. Rule III requires
    demonstrated practically meaningful change to a conclusion.
    """
    outcome, rationale = decide_implementation_outcome(
        findings=[DEPTH_FINDING_V1], core_stages_satisfied=True,
    )
    assert outcome is DimensionOutcome.INCONCLUSIVE
    assert "UNDEMONSTRATED" in rationale
    assert "CONFIRMED" in rationale


def test_material_with_demonstrated_impact_yields_fail():
    demonstrated = dataclasses.replace(
        DEPTH_FINDING_V1, finding_id="II-FINDING-DEPTH-V2",
        scientific_impact=ScientificImpactStatus.DEMONSTRATED,
    )
    outcome, rationale = decide_implementation_outcome(
        findings=[demonstrated], core_stages_satisfied=True,
    )
    assert outcome is DimensionOutcome.FAIL
    assert "Rule III" in rationale


def test_material_but_refuted_impact_yields_conditional_pass():
    refuted = dataclasses.replace(
        DEPTH_FINDING_V1, finding_id="II-FINDING-DEPTH-V3",
        scientific_impact=ScientificImpactStatus.REFUTED,
    )
    outcome, rationale = decide_implementation_outcome(
        findings=[refuted], core_stages_satisfied=True,
    )
    assert outcome is DimensionOutcome.CONDITIONAL_PASS
    assert "Rule II" in rationale


def test_clarity_finding_alone_does_not_cause_fail():
    outcome, _ = decide_implementation_outcome(
        findings=[CE3_CLARITY_FINDING_V1], core_stages_satisfied=True,
    )
    assert outcome is DimensionOutcome.PASS


def test_unsatisfied_core_stages_yield_inconclusive_not_fail():
    """Rule IV: unperformed work must never be read as evidence against the study."""
    outcome, rationale = decide_implementation_outcome(
        findings=[DEPTH_FINDING_V1], core_stages_satisfied=False,
    )
    assert outcome is DimensionOutcome.INCONCLUSIVE
    assert "Rule IV" in rationale


def test_unavailable_information_yields_inconclusive_not_conditional_pass():
    """
    CORRECTED (Rule IV): missing essential evidence must produce
    INCONCLUSIVE. CONDITIONAL_PASS would assert that differences are
    practically negligible - a positive claim missing evidence cannot
    support.
    """
    outcome, rationale = decide_implementation_outcome(
        findings=[], core_stages_satisfied=True,
        unavailable_essential_information=["training logs"],
    )
    assert outcome is DimensionOutcome.INCONCLUSIVE
    assert outcome is not DimensionOutcome.CONDITIONAL_PASS
    assert "Rule IV" in rationale


# --- the depth finding and dependency propagation ---

def test_depth_finding_is_material_and_versioned():
    assert DEPTH_FINDING_V1.severity is FindingSeverity.MATERIAL
    assert DEPTH_FINDING_V1.finding_id.endswith("-V1")
    assert DEPTH_FINDING_V1.declared["depth"] == 10
    assert DEPTH_FINDING_V1.actual["depth"] == 5


def test_depth_finding_marks_all_four_ce_for_reevaluation():
    refs = DEPTH_FINDING_V1.requires_reevaluation()
    assert any("ce1" in r for r in refs)
    assert any("ce2" in r for r in refs)
    assert any("ce3" in r for r in refs)
    assert any("ce4" in r for r in refs)


def test_depth_finding_leaves_dataset_pillar_unaffected():
    """The finding is downstream of the dataset; D1-D5 must NOT be reopened."""
    dataset_deps = [d for d in DEPTH_FINDING_V1.downstream if d.dimension is Dimension.DATASET]
    assert dataset_deps
    assert all(d.status is DownstreamStatus.UNAFFECTED for d in dataset_deps)


def test_depth_finding_leaves_experimental_pillar_unaffected():
    ev_deps = [d for d in DEPTH_FINDING_V1.downstream if d.dimension is Dimension.EXPERIMENTAL]
    assert ev_deps
    assert all(d.status is DownstreamStatus.UNAFFECTED for d in ev_deps)


def test_all_downstream_dependencies_preserve_historical_artifacts():
    for finding in ALL_FINDINGS:
        for dep in finding.downstream:
            assert dep.historical_artifact_preserved is True


def test_depth_finding_records_open_questions_rather_than_asserting_causation():
    """Attribution of the accuracy gap to depth must be OPEN, not asserted."""
    joined = " ".join(DEPTH_FINDING_V1.open_questions).lower()
    assert "not established" in joined
    assert "0.6108" in joined or "controlled depth-10" in joined


# --- evidence assembly ---

def _satisfied_core_stages():
    return [
        StageStatus(stage_id=s, tier=AssessmentTier.CORE,
                    status=StageOutcome.COMPLETED, summary="ok")
        for s in CORE_STAGES
    ]


def _as_run_in_production():
    """Mirrors the real runner: conformance verified, II-4 not performed, II-5 N/A."""
    mapping = {
        StageId.II_1_BASELINE_RECONSTRUCTION: StageOutcome.CONFORMANCE_VERIFIED,
        StageId.II_3_CONTROLLED_VERIFICATION: StageOutcome.CONFORMANCE_VERIFIED,
        StageId.II_4_COMPARATIVE_EVALUATION: StageOutcome.NOT_PERFORMED,
        StageId.II_5_STATISTICAL_ASSESSMENT: StageOutcome.NOT_APPLICABLE,
    }
    return [
        StageStatus(stage_id=s, tier=AssessmentTier.CORE, status=mapping[s], summary="")
        for s in CORE_STAGES
    ]


def test_not_performed_core_stage_is_not_satisfied():
    """II-4 NOT_PERFORMED must not count as satisfying its CORE obligation."""
    stage = StageStatus(
        stage_id=StageId.II_4_COMPARATIVE_EVALUATION, tier=AssessmentTier.CORE,
        status=StageOutcome.NOT_PERFORMED, summary="",
    )
    assert stage.is_satisfied is False


def test_not_applicable_core_stage_is_satisfied():
    """II-5 NOT_APPLICABLE is a legitimate terminal state, unlike NOT_PERFORMED."""
    stage = StageStatus(
        stage_id=StageId.II_5_STATISTICAL_ASSESSMENT, tier=AssessmentTier.CORE,
        status=StageOutcome.NOT_APPLICABLE, summary="",
    )
    assert stage.is_satisfied is True


def test_production_stage_configuration_yields_inconclusive():
    """
    End-to-end semantic check: the stage configuration the real runner
    produces must yield INCONCLUSIVE, because II-4 was never performed.
    """
    evidence = build_implementation_evidence(
        evidence_id="EV-II-PROD-SIM", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_as_run_in_production(), findings=[DEPTH_FINDING_V1],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"x": 1},
    )
    assert evidence.decision == DimensionOutcome.INCONCLUSIVE.value
    assert evidence.observation["core_stages_satisfied"] is False
    assert "II-4_COMPARATIVE_EVALUATION" in evidence.observation["core_stages_unsatisfied"]


def test_build_evidence_records_impact_status_not_a_level_cap():
    evidence = build_implementation_evidence(
        evidence_id="EV-II-TEST-001", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[DEPTH_FINDING_V1],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={},
    )
    assert evidence.dimension is Dimension.IMPLEMENTATION
    # Confirmed discrepancy, undemonstrated impact -> INCONCLUSIVE, not FAIL.
    assert evidence.decision == DimensionOutcome.INCONCLUSIVE.value
    assert evidence.observation["discrepancy_confirmed"] is True
    assert evidence.observation["scientific_impact_demonstrated"] is False
    assert evidence.observation["convergence_contribution"]["contributes"] is False
    assert any("UNDEMONSTRATED" in lim for lim in evidence.limitations)


def test_build_evidence_propagates_downstream_reevaluation_list():
    evidence = build_implementation_evidence(
        evidence_id="EV-II-TEST-002", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[DEPTH_FINDING_V1],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={},
    )
    assert len(evidence.observation["downstream_requires_reevaluation"]) == 4


def test_build_evidence_carries_artifact_hashes_for_traceability():
    evidence = build_implementation_evidence(
        evidence_id="EV-II-TEST-003", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[DEPTH_FINDING_V1],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={},
    )
    assert "best5depth10 (10).h5" in evidence.artifact_hashes
    assert len(evidence.artifact_hashes["best5depth10 (10).h5"]) == 64



# --- finding impact semantics ---

def test_depth_finding_impact_is_undemonstrated_not_demonstrated():
    """The audit confirmed the discrepancy; it did NOT demonstrate its effect."""
    assert DEPTH_FINDING_V1.scientific_impact is ScientificImpactStatus.UNDEMONSTRATED


def test_clarity_finding_impact_is_not_applicable():
    assert CE3_CLARITY_FINDING_V1.scientific_impact is ScientificImpactStatus.NOT_APPLICABLE


# --- correction 1: no stale Level-4 cap language anywhere ---

def test_no_stale_level4_cap_language_in_source():
    import inspect
    import audit.implementation.run_implementation_audit as runner
    import audit.implementation.certificate as cert
    import audit.implementation.stages as stages_mod
    for module in (runner, cert, stages_mod):
        src = inspect.getsource(module)
        assert "capped below" not in src, f"stale cap language in {module.__name__}"
        assert "is capped" not in src, f"stale cap language in {module.__name__}"


# --- correction 2: confirmatory status must not be manufactured ---

def test_evidence_without_preregistration_is_exploratory():
    from audit.common.outcomes import EvidenceTag
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T1", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1}, preregistration_reference=None,
    )
    assert evidence.tag is EvidenceTag.EXPLORATORY
    assert any("EXPLORATORY" in lim for lim in evidence.limitations)


def test_evidence_with_genuine_preregistration_is_confirmatory(tmp_path):
    """A preregistration reference confers CONFIRMATORY only when the artifact
    genuinely exists and its hash verifies (see issue-4 regression tests)."""
    from audit.common.outcomes import EvidenceTag
    from audit.common.provenance import sha256_file
    artifact = tmp_path / "II-CORE-v1.md"
    artifact.write_text("pre-execution preregistration")
    digest = sha256_file(artifact)
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T2", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1},
        preregistration_reference={"artifact_path": str(artifact), "content_hash": digest},
    )
    assert evidence.tag is EvidenceTag.CONFIRMATORY
    assert evidence.artifact_hashes[str(artifact)] == digest


def test_incomplete_preregistration_reference_does_not_confer_confirmatory():
    from audit.common.outcomes import EvidenceTag
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T3", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1},
        preregistration_reference={"artifact_path": "prereg/x.md"},  # no hash
    )
    assert evidence.tag is EvidenceTag.EXPLORATORY


# --- correction 3: Evidence -> RunManifest traceable from the artifact alone ---

def test_manifest_reference_is_embedded_in_evidence():
    ref = {"run_id": "r1", "artifact_path": "run_manifest_r1.json", "content_hash": "b" * 64}
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T4", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1}, run_manifest_references=[ref],
    )
    assert evidence.observation["run_manifest_references"] == [ref]
    # And the hash is in artifact_hashes, so the chain is self-contained.
    assert evidence.artifact_hashes["run_manifest_r1.json"] == "b" * 64


# --- correction 4: findings bound to actual observations ---

def test_verify_finding_detects_hash_mismatch():
    from audit.implementation.findings import verify_finding_against_observations
    problems = verify_finding_against_observations(
        DEPTH_FINDING_V1,
        observed_artifact_hashes={"best5depth10 (10).h5": "deadbeef",
                                  "signal_destroyed.h5": "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308"},
        observed_values={},
    )
    assert any("hash mismatch" in p for p in problems)


def test_verify_finding_detects_uninspected_artifact():
    from audit.implementation.findings import verify_finding_against_observations
    problems = verify_finding_against_observations(
        DEPTH_FINDING_V1, observed_artifact_hashes={}, observed_values={},
    )
    assert any("was not inspected" in p for p in problems)


def test_verify_finding_detects_observed_value_mismatch():
    from audit.implementation.findings import verify_finding_against_observations
    problems = verify_finding_against_observations(
        DEPTH_FINDING_V1,
        observed_artifact_hashes={
            "best5depth10 (10).h5": "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea",
            "signal_destroyed.h5": "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308",
        },
        observed_values={"depth": 10},  # finding records 5
    )
    assert any("does not match observed value" in p for p in problems)


def test_verify_finding_passes_when_observations_match():
    from audit.implementation.findings import verify_finding_against_observations
    problems = verify_finding_against_observations(
        DEPTH_FINDING_V1,
        observed_artifact_hashes={
            "best5depth10 (10).h5": "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea",
            "signal_destroyed.h5": "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308",
        },
        observed_values={"depth": 5, "n_conv1d_layers": 11},
    )
    assert problems == []


# --- regression tests for independently verified issues 1, 2, 4, 5 ---

def test_certificate_docstring_matches_implemented_decision_logic():
    """REGRESSION (issue 1): docstring claimed MATERIAL -> FAIL unconditionally."""
    import audit.implementation.certificate as cert_mod
    doc = cert_mod.__doc__
    assert "A MATERIAL finding does NOT by itself map to FAIL" in doc
    assert "MATERIAL + UNDEMONSTRATED -> INCONCLUSIVE" in doc


def test_preregistration_reference_to_nonexistent_artifact_is_rejected():
    """REGRESSION (issue 4): a fabricated reference previously conferred CONFIRMATORY."""
    from audit.common.outcomes import EvidenceTag
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T9", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1},
        preregistration_reference={"artifact_path": "/nonexistent/fabricated.md",
                                   "content_hash": "f" * 64},
    )
    assert evidence.tag is EvidenceTag.EXPLORATORY
    assert any("does not exist" in lim for lim in evidence.limitations)


def test_preregistration_reference_with_wrong_hash_is_rejected(tmp_path):
    from audit.common.outcomes import EvidenceTag
    artifact = tmp_path / "prereg.md"
    artifact.write_text("preregistered plan")
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T10", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1},
        preregistration_reference={"artifact_path": str(artifact), "content_hash": "0" * 64},
    )
    assert evidence.tag is EvidenceTag.EXPLORATORY
    assert any("hash mismatch" in lim for lim in evidence.limitations)


def test_genuine_verified_preregistration_confers_confirmatory(tmp_path):
    from audit.common.outcomes import EvidenceTag
    from audit.common.provenance import sha256_file
    artifact = tmp_path / "prereg.md"
    artifact.write_text("genuinely pre-existing preregistration")
    evidence = build_implementation_evidence(
        evidence_id="EV-II-T11", claim_id="C1", experiment_id="II-CORE-TEST",
        run_ids=["r1"], stages=_satisfied_core_stages(), findings=[],
        independent_reimplementation_status=EnhancedAssessmentStatus.NOT_ASSESSED,
        provenance={"g": 1},
        preregistration_reference={"artifact_path": str(artifact),
                                   "content_hash": sha256_file(artifact)},
    )
    assert evidence.tag is EvidenceTag.CONFIRMATORY


def test_binding_fails_closed_when_asserted_value_unverified():
    """REGRESSION (issue 5): conflicting checkpoints previously left actual['depth'] unchecked."""
    from audit.implementation.findings import verify_finding_against_observations
    problems = verify_finding_against_observations(
        DEPTH_FINDING_V1,
        observed_artifact_hashes={
            "best5depth10 (10).h5": "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea",
            "signal_destroyed.h5": "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308",
        },
        observed_values={},  # conflict -> runner omits the aggregate
        verifiable_keys=("depth", "n_conv1d_layers"),
    )
    assert any("was NOT verified" in p for p in problems)


def test_binding_passes_when_all_verifiable_keys_supported():
    from audit.implementation.findings import verify_finding_against_observations
    problems = verify_finding_against_observations(
        DEPTH_FINDING_V1,
        observed_artifact_hashes={
            "best5depth10 (10).h5": "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea",
            "signal_destroyed.h5": "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308",
        },
        observed_values={"depth": 5, "n_conv1d_layers": 11},
        verifiable_keys=("depth", "n_conv1d_layers"),
    )
    assert problems == []
