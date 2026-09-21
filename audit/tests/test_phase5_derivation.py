"""
Tests that the evidence ledger is CAUSALLY UPSTREAM of the claim
decision, not decorative.

The Phase-5 integrity gate previously failed because
current_pillar_state() returned hand-written constants. These tests
exist so that regression cannot recur silently: they mutate a
decision-bearing ledger entry and assert the derived state and/or the
claim decision changes accordingly.
"""

from __future__ import annotations

import inspect
import json

import pytest

from audit.common.ids import Dimension
from audit.common.outcomes import CryptographicOutcome, DimensionOutcome, EvidenceTag
from audit.integration.conflict_resolution import ALL_CONFLICTS
from audit.integration.decision_engine import decide_claim
from audit.integration.evidence_ledger import EvidenceLedger, IngestedEvidenceReference
from audit.integration.pillar_interpretation import (
    AGGREGATION_RULE_ID,
    Mappability,
    RuleProvenance,
    aggregate_dimension,
    interpret_native_status,
)
from audit.integration.run_integration import derive_pillar_state


def _ingested(evidence_id, pillar, dimension, decision, path="/tmp/x.json"):
    return IngestedEvidenceReference(
        evidence_id=evidence_id, claim_id="C1", dimension=dimension,
        source_pillar_id=pillar, artifact_path=path, artifact_hash="a" * 64,
        recorded_decision=decision,
    )


def _full_ledger(**overrides) -> EvidenceLedger:
    """A ledger covering every decision-bearing component, all PASS/SUPPORTED by default."""
    defaults = {
        "D1": "PASS", "D2": "PASS", "D3": "PASS", "D4": "PASS", "D5": "PASS",
        # Neutral synthetic ids: CE1..CE4 are parties to the real unresolved
        # conflict, so using them here would exercise conflict precedence
        # rather than the aggregation rule these fixtures target.
        "CX1": "SUPPORTED", "CX2": "SUPPORTED",
        "II": "PASS", "EV": "PASS",
    }
    defaults.update(overrides)
    ledger = EvidenceLedger()
    for pillar, decision in defaults.items():
        dimension = (
            Dimension.DATASET if pillar.startswith("D") and pillar != "EV"
            else Dimension.CRYPTOGRAPHIC if pillar.startswith("C")
            else Dimension.IMPLEMENTATION if pillar == "II"
            else Dimension.EXPERIMENTAL
        )
        ledger.add_ingested(_ingested(f"EV-{pillar}-T", pillar, dimension, decision))
    return ledger


# --- the production path must contain no hard-coded outcomes ---

def test_derive_pillar_state_contains_no_hardcoded_outcomes():
    src = inspect.getsource(derive_pillar_state)
    for forbidden in ("DimensionOutcome.PASS", "DimensionOutcome.FAIL",
                      "DimensionOutcome.INCONCLUSIVE", "DimensionOutcome.CONDITIONAL_PASS",
                      "CryptographicOutcome.SUPPORTED", "CryptographicOutcome.NOT_SUPPORTED"):
        assert forbidden not in src, f"production derivation hard-codes {forbidden}"


def test_run_integration_module_has_no_hardcoded_pillar_constants():
    import audit.integration.run_integration as mod
    src = inspect.getsource(mod)
    assert "DimensionOutcome." not in src


# --- causality: changing the ledger changes the derived state ---

def test_all_pass_ledger_yields_all_pass_state():
    state, _ = derive_pillar_state(_full_ledger())
    assert state["dataset"] is DimensionOutcome.PASS
    assert state["implementation"] is DimensionOutcome.PASS
    assert state["experimental"] is DimensionOutcome.PASS
    assert state["cryptographic"] is CryptographicOutcome.SUPPORTED


def test_changing_one_dataset_entry_to_fail_changes_dimension_to_fail():
    """Causal test: mutate D1 and the dataset dimension must follow."""
    baseline, _ = derive_pillar_state(_full_ledger())
    mutated, _ = derive_pillar_state(_full_ledger(D1="FAIL"))
    assert baseline["dataset"] is DimensionOutcome.PASS
    assert mutated["dataset"] is DimensionOutcome.FAIL


def test_changing_one_dataset_entry_to_inconclusive_changes_dimension():
    mutated, _ = derive_pillar_state(_full_ledger(D2="INCONCLUSIVE"))
    assert mutated["dataset"] is DimensionOutcome.INCONCLUSIVE


def test_unmappable_native_status_drives_dimension_inconclusive():
    """D3's real DESCRIPTIVE_ONLY must not be coerced, and must not yield PASS."""
    mutated, interps = derive_pillar_state(_full_ledger(D3="DESCRIPTIVE_ONLY"))
    assert mutated["dataset"] is DimensionOutcome.INCONCLUSIVE
    records = interps[Dimension.DATASET.value]["records"]
    d3 = next(r for r in records if r["source_pillar_id"] == "D3")
    assert d3["mappability"] == Mappability.UNMAPPABLE.value
    assert d3["normalized_outcome"] is None


def test_changing_ledger_changes_the_final_claim_decision():
    """End-to-end causality: ledger -> state -> decision."""
    all_pass_state, _ = derive_pillar_state(_full_ledger())
    good = decide_claim(claim_id="C1", conflicts=[], **all_pass_state)

    degraded_state, _ = derive_pillar_state(_full_ledger(D1="FAIL"))
    bad = decide_claim(claim_id="C1", conflicts=[], **degraded_state)

    assert good.final_outcome.value == "SUPPORTED"
    assert bad.final_outcome.value == "NOT_SUPPORTED"
    assert good.final_outcome is not bad.final_outcome


def test_removing_a_decision_bearing_entry_yields_inconclusive_not_pass():
    """
    Absence must not read as success: dropping D5 entirely must still
    produce INCONCLUSIVE via the explicit NOT_PRODUCED representation.
    """
    ledger = _full_ledger()
    ledger.entries = [e for e in ledger.entries if e["source_pillar_id"] != "D5"]
    ledger._ids.discard("EV-D5-T")
    state, interps = derive_pillar_state(ledger)
    assert state["dataset"] is DimensionOutcome.INCONCLUSIVE
    pillars = {r["source_pillar_id"] for r in interps[Dimension.DATASET.value]["records"]}
    assert "D5" in pillars  # represented as absent, not silently dropped


def test_hardcoded_expectation_cannot_override_disagreeing_ledger():
    """
    A hand-written expected value must not be able to make the decision
    pass when the ledger disagrees. The derived state wins.
    """
    hand_written_expectation = DimensionOutcome.PASS
    derived, _ = derive_pillar_state(_full_ledger(D1="FAIL"))
    assert derived["dataset"] is DimensionOutcome.FAIL
    assert derived["dataset"] is not hand_written_expectation
    decision = decide_claim(claim_id="C1", conflicts=[], **derived)
    assert decision.final_outcome.value == "NOT_SUPPORTED"


# --- rule provenance must be explicit and honest ---

def test_direct_vocabulary_mapping_is_source_defined():
    rec = interpret_native_status(
        source_evidence_id="E", source_pillar_id="D1", dimension=Dimension.DATASET,
        native_status="PASS")
    assert rec.rule_provenance is RuleProvenance.SOURCE_DEFINED
    assert rec.mappability is Mappability.DIRECTLY_MAPPABLE


def test_non_coercion_is_labelled_integration_operationalization():
    for native in ("DESCRIPTIVE_ONLY", "EFFECT_DETECTED", "NOT_PERFORMED",
                   "NOT_ASSESSED", "NOT_LOCATED"):
        rec = interpret_native_status(
            source_evidence_id="E", source_pillar_id="X", dimension=Dimension.DATASET,
            native_status=native)
        assert rec.rule_provenance is RuleProvenance.INTEGRATION_OPERATIONALIZATION
        assert rec.normalized_outcome is None, f"{native} was coerced"


def test_aggregation_rule_is_declared_as_operationalization_not_methodology():
    interp = aggregate_dimension(Dimension.DATASET, [
        interpret_native_status(source_evidence_id="E", source_pillar_id="D1",
                                dimension=Dimension.DATASET, native_status="PASS")])
    assert interp.rule_id == AGGREGATION_RULE_ID
    assert interp.rule_provenance is RuleProvenance.INTEGRATION_OPERATIONALIZATION
    assert "NOT defined by the methodology" in interp.rationale


def test_every_interpretation_record_references_its_evidence():
    _, interps = derive_pillar_state(_full_ledger())
    for dim_interp in interps.values():
        for record in dim_interp["records"]:
            assert record["source_evidence_id"]
            assert record["rationale"]
            assert record["version"] == "pillar-interpretation-v1"


# --- the current, real derived state ---

def test_current_real_derived_state_matches_expected_fixture():
    """
    Test fixture (permitted per the spec) asserting today's expected
    result. This is an independent CHECK on the derivation, not the
    mechanism that produces it.
    """
    import pathlib
    base = pathlib.Path("audit/integration/evidence")
    if not (base / "audit_snapshot.json").exists():
        pytest.skip("Phase 5 documents not generated in this environment")
    snapshot = json.loads((base / "audit_snapshot.json").read_text())
    assert snapshot["pillar_outcomes"] == {
        "implementation_integrity": "INCONCLUSIVE",
        "dataset_integrity": "INCONCLUSIVE",
        "experimental_validity": "INCONCLUSIVE",
        "cryptographic_evidence": "INCONCLUSIVE",
    }
    assert snapshot["decision"]["final_outcome"] == "INCONCLUSIVE"
    assert snapshot["decision"]["evidence_level"] == "LEVEL_1_PREDICTIVE"


# --- regression: CE2 ingestion and conflict precedence (final gate findings) ---

def test_ce2_evidence_is_ingested_not_reported_as_not_produced():
    """
    REGRESSION: CE2 has five real NOT_SUPPORTED run certificates. An
    earlier ledger builder omitted CE2 entirely, and the absence-filler
    then misreported genuinely-existing evidence as NOT_PRODUCED.
    """
    import pathlib
    ledger_path = pathlib.Path("audit/integration/evidence/evidence_ledger.json")
    if not ledger_path.exists():
        pytest.skip("Phase 5 ledger not generated")
    ledger = json.loads(ledger_path.read_text())
    ce2 = [e for e in ledger["entries"] if e.get("source_pillar_id") == "CE2"]
    assert len(ce2) == 5, "all five CE2 run certificates must be ingested"
    assert all(e["recorded_decision"] == "NOT_SUPPORTED" for e in ce2)
    assert all(e["entry_kind"] == "INGESTED_REFERENCE" for e in ce2)


def test_source_defined_conflict_rule_overrides_integration_aggregation():
    """
    REGRESSION: CE2's NOT_SUPPORTED alone would drive the cryptographic
    dimension to NOT_SUPPORTED under RULE-AGG-CONSERVATIVE-V1 (an
    integration operationalization). Methodology 3.8.8 (SOURCE_DEFINED)
    requires INCONCLUSIVE while the conflict is unresolved, and a
    source-defined rule must win.
    """
    from audit.integration.pillar_interpretation import CONFLICT_PRECEDENCE_RULE_ID

    recs = [interpret_native_status(
        source_evidence_id=f"EV-{p}", source_pillar_id=p,
        dimension=Dimension.CRYPTOGRAPHIC, native_status=s)
        for p, s in [("CE1", "INCONCLUSIVE"), ("CE2", "NOT_SUPPORTED"),
                     ("CE3", "SUPPORTED"), ("CE4", "SUPPORTED")]]

    # Without conflict awareness the aggregation alone resolves to NOT_SUPPORTED.
    unaware = aggregate_dimension(Dimension.CRYPTOGRAPHIC, recs)
    assert unaware.outcome == "NOT_SUPPORTED"

    # With the conflict declared, the source-defined rule takes precedence.
    aware = aggregate_dimension(Dimension.CRYPTOGRAPHIC, recs,
                                conflicted_evidence_ids={"CE2", "CE3", "CE4"})
    assert aware.outcome == "INCONCLUSIVE"
    assert aware.rule_id == CONFLICT_PRECEDENCE_RULE_ID
    assert aware.rule_provenance is RuleProvenance.SOURCE_DEFINED


def test_ce_dimension_outcome_comes_from_conflict_rule_in_current_state():
    import pathlib
    path = pathlib.Path("audit/integration/evidence/pillar_interpretation.json")
    if not path.exists():
        pytest.skip("Phase 5 interpretation not generated")
    interp = json.loads(path.read_text())["interpretations"]
    ce = interp[Dimension.CRYPTOGRAPHIC.value]
    assert ce["outcome"] == "INCONCLUSIVE"
    assert ce["rule_id"] == "RULE-CONFLICT-PRECEDENCE"
    assert ce["rule_provenance"] == "SOURCE_DEFINED"


def test_ce_vocabulary_is_never_coerced_into_dimension_vocabulary():
    """SUPPORTED must not become PASS, nor NOT_SUPPORTED become FAIL."""
    supported = interpret_native_status(
        source_evidence_id="E", source_pillar_id="CE3",
        dimension=Dimension.CRYPTOGRAPHIC, native_status="SUPPORTED")
    assert supported.normalized_outcome == "SUPPORTED"
    assert supported.normalized_outcome != "PASS"

    # A dimension-vocabulary value is not valid in the CE dimension.
    cross = interpret_native_status(
        source_evidence_id="E", source_pillar_id="CEx",
        dimension=Dimension.CRYPTOGRAPHIC, native_status="PASS")
    assert cross.mappability is Mappability.UNMAPPABLE
    assert cross.normalized_outcome is None
