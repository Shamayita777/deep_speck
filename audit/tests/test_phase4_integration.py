import pytest

from audit.common.outcomes import (
    CryptographicOutcome,
    DimensionOutcome,
    EvidenceLevel,
    FinalAuditOutcome,
)
from audit.integration.ce_mapping import (
    ALL_CE_MAPPINGS,
    CE1,
    CE2,
    CE3,
    CE4,
    KEY_RECOVERY_SCOPE_NOTE,
    SHARED_TARGET_QUANTITY,
)
from audit.integration.conflict_resolution import (
    ALL_CONFLICTS,
    CE2_VS_CE3_CE4_CONFLICT,
    Conflict,
    ConflictStatus,
    ConflictingObservation,
    blocking_conflicts,
)
from audit.integration.decision_engine import (
    ClaimDecision,
    ConvergenceSource,
    assign_evidence_level,
    decide_claim,
)


# --- CE mapping fidelity to verified source facts ---

def test_all_ce_experiments_are_five_round_primary_scope():
    for mapping in ALL_CE_MAPPINGS:
        assert mapping.rounds == 5, f"{mapping.ce_id} is not 5-round"
        assert mapping.differential == (0x0040, 0x0000)


def test_ce1_trains_its_own_model_unlike_ce2_ce3_ce4():
    assert "TRAINS its own" in CE1.model_provenance
    for mapping in (CE2, CE3, CE4):
        assert "best5depth10" in mapping.model_provenance


def test_ce2_ce3_ce4_share_the_same_target_quantity():
    for mapping in (CE2, CE3, CE4):
        assert mapping.cryptographic_mechanism == SHARED_TARGET_QUANTITY


def test_target_quantity_is_not_overstated():
    """Must not be described as the cipher's exact differential probability."""
    q = SHARED_TARGET_QUANTITY
    assert "NOT the cipher's exact differential probability" in q
    assert "NOT the full trail probability" in q
    assert "single-trail" in q


def test_ce3_statistical_unit_is_replicate_not_fold():
    assert "Independent evaluation replicate" in CE3.statistical_unit
    assert "NOT treated as independent" in CE3.statistical_unit


def test_ce4_does_not_overclaim_causal_necessity():
    joined = " ".join(CE4.limitations)
    assert "intervention-sensitive dependence" in joined
    assert "stronger than what is established" in joined


def test_key_recovery_is_explicitly_out_of_scope():
    assert "NOT part of this claim chain" in KEY_RECOVERY_SCOPE_NOTE


def test_recorded_decisions_match_historical_evidence():
    assert CE1.recorded_decision == "INCONCLUSIVE"
    assert CE2.recorded_decision == "NOT_SUPPORTED"
    assert CE3.recorded_decision == "SUPPORTED"
    assert CE4.recorded_decision == "SUPPORTED"


# --- conflict resolution ---

def test_conflict_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        Conflict(conflict_id="X", shared_subject="s",
                 observations=[ConflictingObservation("e", "s", "o", "d")],
                 status=ConflictStatus.UNRESOLVED)


def test_ce_conflict_retains_all_three_observations():
    """3.8.8: conflicting evidence is never discarded."""
    assert len(CE2_VS_CE3_CE4_CONFLICT.observations) == 3
    sources = {o.source_id for o in CE2_VS_CE3_CE4_CONFLICT.observations}
    assert sources == {"CE2", "CE3", "CE4"}


def test_ce_conflict_is_not_resolved_by_majority_vote():
    rationale = CE2_VS_CE3_CE4_CONFLICT.rationale
    assert "NOT resolved by majority vote" in rationale
    assert CE2_VS_CE3_CE4_CONFLICT.status is not ConflictStatus.RESOLVED_BY_METHODOLOGICAL_QUALITY


def test_ce_conflict_blocks_the_claim():
    assert CE2_VS_CE3_CE4_CONFLICT.blocks_claim is True
    assert blocking_conflicts(ALL_CONFLICTS) == [CE2_VS_CE3_CE4_CONFLICT]


def test_resolved_conflict_does_not_block():
    resolved = Conflict(
        conflict_id="R", shared_subject="s",
        observations=[ConflictingObservation("e1", "A", "o1", "d1"),
                      ConflictingObservation("e2", "B", "o2", "d2")],
        status=ConflictStatus.EXPLAINED_BY_KNOWN_CONFOUNDER,
    )
    assert resolved.blocks_claim is False


def test_depth_finding_is_recorded_as_not_resolving_the_conflict():
    joined = " ".join(CE2_VS_CE3_CE4_CONFLICT.candidate_confounders)
    assert "does not resolve this conflict" in joined


# --- decision engine determinism and rules ---

def test_decision_engine_is_deterministic():
    kwargs = dict(
        claim_id="C1", implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
    )
    assert decide_claim(**kwargs).to_dict() == decide_claim(**kwargs).to_dict()


def test_any_dimension_fail_yields_not_supported():
    d = decide_claim(
        claim_id="C1", implementation=DimensionOutcome.FAIL, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED)
    assert d.final_outcome is FinalAuditOutcome.NOT_SUPPORTED


def test_blocking_conflict_forces_inconclusive():
    d = decide_claim(
        claim_id="C1", implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        conflicts=ALL_CONFLICTS)
    assert d.final_outcome is FinalAuditOutcome.INCONCLUSIVE
    assert "CONFLICT-CE2-CE34-V1" in d.blocking_conflicts


def test_conditional_pass_yields_supported_with_limitations():
    d = decide_claim(
        claim_id="C1", implementation=DimensionOutcome.CONDITIONAL_PASS,
        dataset=DimensionOutcome.PASS, experimental=DimensionOutcome.PASS,
        cryptographic=CryptographicOutcome.SUPPORTED)
    assert d.final_outcome is FinalAuditOutcome.SUPPORTED_WITH_LIMITATIONS


def test_all_pass_yields_supported():
    d = decide_claim(
        claim_id="C1", implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED)
    assert d.final_outcome is FinalAuditOutcome.SUPPORTED


# --- evidence-level assignment (3.8.4) ---

def test_robustness_alone_does_not_reach_level_3():
    """Rule I: predictive performance surviving scrutiny is Level 2, not cryptographic evidence."""
    level, _ = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.NOT_SUPPORTED,
        convergence_sources=[], has_blocking_conflict=False)
    assert level is EvidenceLevel.LEVEL_2_ROBUST_EXPERIMENTAL


def test_level_4_requires_two_distinct_convergence_sources():
    one = [ConvergenceSource("independent_implementation", "e1", "c")]
    level, _ = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        convergence_sources=one, has_blocking_conflict=False)
    assert level is EvidenceLevel.LEVEL_3_CRYPTOGRAPHIC


def test_level_4_reachable_without_independent_reimplementation():
    """
    CRITICAL: LEVEL_4 must NOT be universally capped merely because
    independent reimplementation was not assessed. Convergence from
    other qualifying sources suffices.
    """
    sources = [ConvergenceSource("independent_dataset", "e1", "c"),
               ConvergenceSource("independent_architecture", "e2", "c")]
    level, rationale = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        convergence_sources=sources, has_blocking_conflict=False)
    assert level is EvidenceLevel.LEVEL_4_STRONG_CRYPTOGRAPHIC
    assert "independent_dataset" in rationale


def test_inconclusive_dimension_caps_at_level_1():
    level, _ = assign_evidence_level(
        implementation=DimensionOutcome.INCONCLUSIVE, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        convergence_sources=[], has_blocking_conflict=False)
    assert level is EvidenceLevel.LEVEL_1_PREDICTIVE


def test_current_actual_pillar_state_yields_inconclusive_level_1():
    """
    The real state of this audit, end to end. Pillar state is now
    DERIVED from the persisted ledger (Phase-5 remediation), not read
    from a hand-written constant.
    """
    import pathlib

    from audit.integration.evidence_ledger import EvidenceLedger
    from audit.integration.run_integration import derive_pillar_state

    ledger_path = pathlib.Path("audit/integration/evidence/evidence_ledger.json")
    if not ledger_path.exists():
        pytest.skip("Phase 5 ledger not generated in this environment")
    state, _ = derive_pillar_state(EvidenceLedger.load(ledger_path))
    d = decide_claim(claim_id="C1", conflicts=ALL_CONFLICTS, **state)
    assert d.final_outcome is FinalAuditOutcome.INCONCLUSIVE
    assert d.evidence_level is EvidenceLevel.LEVEL_1_PREDICTIVE
