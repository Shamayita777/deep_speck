"""
Phase 6 - adversarial tests.

Each test encodes a way the framework could silently upgrade weak
evidence into a stronger scientific conclusion, and asserts that it
does not. These are attack cases, not happy-path coverage: every one
of them should FAIL LOUDLY or degrade conservatively.

Operates entirely on the FROZEN Phase-5 state; modifies no artifact.
"""

from __future__ import annotations

import json
import math

import pytest

from audit.common.evidence import Evidence, EvidenceValidationError
from audit.common.ids import Dimension, InvalidIdentifierError, validate_identifier
from audit.common.outcomes import (
    CryptographicOutcome,
    DimensionOutcome,
    EvidenceLevel,
    EvidenceTag,
    FinalAuditOutcome,
)
from audit.common.provenance import (
    Activity,
    Entity,
    ProvenanceCollisionError,
    ProvenanceGraph,
)
from audit.common.strict_json import StrictJSONError, dumps_strict
from audit.integration.conflict_resolution import ALL_CONFLICTS
from audit.integration.decision_engine import (
    ConvergenceSource,
    assign_evidence_level,
    decide_claim,
)
from audit.integration.evidence_ledger import EvidenceLedger, LedgerIntegrityError
from audit.integration.pillar_interpretation import (
    aggregate_dimension,
    interpret_native_status,
)


# =====================================================================
# A. Non-finite / degenerate numerics must never reach a certificate
# =====================================================================

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_statistic_cannot_enter_evidence(bad):
    with pytest.raises(EvidenceValidationError):
        Evidence(
            evidence_id="EV-ADV", claim_id="C1", dimension=Dimension.DATASET,
            experiment_id="X", run_ids=["r1"], input_artifacts=[], artifact_hashes={},
            observation={}, statistics={"effect": bad}, decision="PASS",
            limitations=[], provenance={}, tag=EvidenceTag.EXPLORATORY,
        )


def test_non_finite_cannot_be_smuggled_via_nested_structure():
    with pytest.raises(StrictJSONError):
        dumps_strict({"a": {"b": [{"c": float("nan")}]}})


def test_zero_denominator_ratio_is_rejected_not_serialized_as_inf():
    """A zero-denominator ratio must be caught, not written as Infinity."""
    with pytest.raises(StrictJSONError):
        dumps_strict({"ratio": 1.0 / 1e-323 * 1e308})  # overflows to inf


# =====================================================================
# B. Weak evidence must not be upgraded to a strong conclusion
# =====================================================================

def test_statistical_significance_alone_cannot_produce_cryptographic_level():
    """
    Rule I (3.8.6): predictive performance is never cryptographic
    evidence, however significant. All dimensions passing with the
    cryptographic interpretation unsupported caps at LEVEL_2.
    """
    level, rationale = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.NOT_SUPPORTED,
        convergence_sources=[], has_blocking_conflict=False,
    )
    assert level is EvidenceLevel.LEVEL_2_ROBUST_EXPERIMENTAL
    assert "does not constitute cryptographic evidence" in rationale


def test_single_convergence_source_cannot_reach_level_4():
    level, _ = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        convergence_sources=[ConvergenceSource("independent_dataset", "e", "c")],
        has_blocking_conflict=False,
    )
    assert level is EvidenceLevel.LEVEL_3_CRYPTOGRAPHIC


def test_duplicate_convergence_sources_of_same_kind_do_not_reach_level_4():
    """Two sources of the SAME kind are not independent convergence."""
    level, _ = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        convergence_sources=[ConvergenceSource("independent_dataset", "e1", "c"),
                             ConvergenceSource("independent_dataset", "e2", "c")],
        has_blocking_conflict=False,
    )
    assert level is EvidenceLevel.LEVEL_3_CRYPTOGRAPHIC


def test_blocking_conflict_prevents_any_cryptographic_level():
    level, _ = assign_evidence_level(
        implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED,
        convergence_sources=[ConvergenceSource("independent_dataset", "e1", "c"),
                             ConvergenceSource("independent_architecture", "e2", "c")],
        has_blocking_conflict=True,
    )
    assert level is EvidenceLevel.LEVEL_1_PREDICTIVE


def test_unmappable_status_cannot_be_laundered_into_pass():
    """Feeding a non-standard status must never yield PASS."""
    for native in ("DESCRIPTIVE_ONLY", "EFFECT_DETECTED", "NOT_PERFORMED",
                   "NOT_ASSESSED", "NOT_LOCATED", "NOT_PRODUCED", "TOTALLY_MADE_UP"):
        rec = interpret_native_status(
            source_evidence_id="E", source_pillar_id="X",
            dimension=Dimension.DATASET, native_status=native)
        assert rec.normalized_outcome is None
        agg = aggregate_dimension(Dimension.DATASET, [rec])
        assert agg.outcome == "INCONCLUSIVE", f"{native} laundered into {agg.outcome}"


def test_one_pass_cannot_outvote_an_unmappable_sibling():
    """A single PASS must not carry a dimension containing unresolved evidence."""
    recs = [
        interpret_native_status(source_evidence_id="E1", source_pillar_id="D1",
                                dimension=Dimension.DATASET, native_status="PASS"),
        interpret_native_status(source_evidence_id="E2", source_pillar_id="D3",
                                dimension=Dimension.DATASET, native_status="DESCRIPTIVE_ONLY"),
    ]
    assert aggregate_dimension(Dimension.DATASET, recs).outcome == "INCONCLUSIVE"


# =====================================================================
# C. Integrity attacks on the ledger / provenance
# =====================================================================

def test_corrupted_artifact_hash_is_detected(tmp_path):
    from audit.common.provenance import sha256_file
    from audit.integration.evidence_ledger import IngestedEvidenceReference

    artifact = tmp_path / "e.json"
    artifact.write_text('{"decision": {"outcome": "PASS"}}')
    ledger = EvidenceLedger()
    ledger.add_ingested(IngestedEvidenceReference(
        evidence_id="EV-X", claim_id="C1", dimension=Dimension.DATASET,
        source_pillar_id="D1", artifact_path=str(artifact),
        artifact_hash=sha256_file(artifact), recorded_decision="PASS"))
    artifact.write_text('{"decision": {"outcome": "FAIL"}}')
    assert any("hash changed" in p for p in ledger.verify_artifact_hashes())


def test_duplicate_evidence_id_cannot_shadow_an_existing_entry():
    from audit.integration.evidence_ledger import IngestedEvidenceReference
    ref = IngestedEvidenceReference(
        evidence_id="EV-DUP", claim_id="C1", dimension=Dimension.DATASET,
        source_pillar_id="D1", artifact_path="/tmp/a", artifact_hash="a" * 64,
        recorded_decision="PASS")
    ledger = EvidenceLedger()
    ledger.add_ingested(ref)
    with pytest.raises(LedgerIntegrityError):
        ledger.add_ingested(ref)


def test_provenance_entity_cannot_be_silently_replaced():
    g = ProvenanceGraph()
    g.add_entity(Entity("e", "dataset", content_hash="a" * 64))
    with pytest.raises(ProvenanceCollisionError):
        g.add_entity(Entity("e", "dataset", content_hash="b" * 64))


def test_provenance_rejects_dangling_reference():
    g = ProvenanceGraph()
    g.add_activity(Activity("a", "t", started_at_utc="t0"))
    g.used("a", "ghost")
    assert any("unknown node" in p for p in g.validate())


def test_malformed_identifier_is_rejected():
    for bad in ("has space", "1leading", "", "tab\tchar"):
        with pytest.raises(InvalidIdentifierError):
            validate_identifier(bad)


# =====================================================================
# D. Post-hoc / policy manipulation
# =====================================================================

def test_exploratory_evidence_cannot_masquerade_as_confirmatory():
    from audit.common.evidence import validate_evidence_provenance
    ev = Evidence(
        evidence_id="EV-PH", claim_id="C1", dimension=Dimension.DATASET,
        experiment_id="X", run_ids=["r1"], input_artifacts=[], artifact_hashes={},
        observation={}, statistics={}, decision="PASS", limitations=[],
        provenance={"g": 1}, tag=EvidenceTag.CONFIRMATORY)
    problems = validate_evidence_provenance(
        ev, known_run_manifests={"r1": {"preregistration_hash": None}})
    assert any("no preregistration_hash" in p for p in problems)


def test_decision_policy_version_is_recorded_so_policy_change_is_visible():
    d = decide_claim(
        claim_id="C1", implementation=DimensionOutcome.PASS, dataset=DimensionOutcome.PASS,
        experimental=DimensionOutcome.PASS, cryptographic=CryptographicOutcome.SUPPORTED)
    assert d.policy_version
    assert d.to_dict()["policy_version"] == d.policy_version


def test_decision_engine_introduces_no_new_evidence():
    """The engine must be a pure function of its declared inputs."""
    import inspect

    from audit.integration import decision_engine
    src = inspect.getsource(decision_engine)
    for forbidden in ("open(", "requests.", "np.random", "random.", "subprocess"):
        assert forbidden not in src, f"decision engine performs {forbidden}"


def test_conflict_cannot_be_dropped_to_unblock_a_claim():
    """Removing the conflict changes the outcome - so it cannot be ignored silently."""
    kwargs = dict(claim_id="C1", implementation=DimensionOutcome.PASS,
                  dataset=DimensionOutcome.PASS, experimental=DimensionOutcome.PASS,
                  cryptographic=CryptographicOutcome.SUPPORTED)
    with_conflict = decide_claim(conflicts=ALL_CONFLICTS, **kwargs)
    without = decide_claim(conflicts=[], **kwargs)
    assert with_conflict.final_outcome is FinalAuditOutcome.INCONCLUSIVE
    assert without.final_outcome is FinalAuditOutcome.SUPPORTED
    assert with_conflict.blocking_conflicts and not without.blocking_conflicts


# =====================================================================
# E. End-to-end invariants on the FROZEN Phase-5 state
# =====================================================================

def _frozen():
    import pathlib
    path = pathlib.Path("audit/integration/frozen/PHASE5_FREEZE_MANIFEST.json")
    if not path.exists():
        pytest.skip("Phase 5 not frozen in this environment")
    return json.loads(path.read_text())


def test_frozen_phase5_component_hashes_still_match():
    import hashlib
    import pathlib
    manifest = _frozen()
    base = pathlib.Path("audit/integration/evidence")
    for name, expected in manifest["components"].items():
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        assert actual == expected, f"frozen component {name} changed after freeze"


def test_frozen_decision_is_inconclusive_level_1():
    import pathlib
    manifest = _frozen()
    snap = json.loads(
        (pathlib.Path("audit/integration/evidence")
         / manifest["authoritative_snapshot"]).read_text())
    assert snap["decision"]["final_outcome"] == "INCONCLUSIVE"
    assert snap["decision"]["evidence_level"] == "LEVEL_1_PREDICTIVE"


def test_no_dimension_claims_more_than_its_evidence_supports():
    """Every dimension outcome must be traceable to named driving evidence."""
    import pathlib
    path = pathlib.Path("audit/integration/evidence/pillar_interpretation.json")
    if not path.exists():
        pytest.skip("interpretation not generated")
    interps = json.loads(path.read_text())["interpretations"]
    for dim, interp in interps.items():
        if interp["outcome"] in ("PASS", "SUPPORTED"):
            assert interp["driving_evidence_ids"], (
                f"{dim} claims {interp['outcome']} with no driving evidence")


def test_frozen_archive_is_independently_verifiable():
    """
    REGRESSION (packaging): the frozen archive must carry every artifact
    its ledger cites, with archive-relative paths, so the integrity gate
    runs from a clean extraction with no external dependency.
    """
    import hashlib
    import pathlib

    manifest = _frozen()
    assert manifest.get("independently_verifiable") is True

    bundle_dir = pathlib.Path("audit/evidence_bundle")
    bundle = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    for rel, meta in bundle["files"].items():
        f = bundle_dir / rel
        assert f.exists(), f"bundle file missing: {rel}"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == meta["sha256"]

    # No ledger entry may cite an absolute/machine-specific path.
    ledger = json.loads(
        pathlib.Path("audit/integration/evidence/evidence_ledger.json").read_text())
    for entry in ledger["entries"]:
        path = entry.get("artifact_path")
        if path:
            assert not path.startswith("/"), f"absolute path in ledger: {path}"
            assert pathlib.Path(path).exists(), f"ledger cites missing artifact: {path}"


def test_vendored_bundle_is_byte_identical_to_original_sources():
    """Vendoring must copy, never transform, historical evidence."""
    bundle_dir = pathlib.Path("audit/evidence_bundle") if False else None
    import pathlib as _p
    bundle = json.loads((_p.Path("audit/evidence_bundle") / "BUNDLE_MANIFEST.json").read_text())
    checked = [m for m in bundle["files"].values()
               if m["byte_identical_to_original"] is not None]
    assert checked, "no byte-identity record present"
    assert all(m["byte_identical_to_original"] for m in checked)


# =====================================================================
# F. Phase 7 - report must assert nothing beyond the evidence
# =====================================================================

def test_report_generation_fails_closed_on_missing_document(tmp_path):
    from audit.integration.generate_report import ReportGenerationError, generate
    with pytest.raises(ReportGenerationError):
        generate(tmp_path, tmp_path, tmp_path)


def test_generated_report_matches_persisted_decision():
    """The report must not state an outcome the snapshot does not record."""
    import pathlib
    report_path = pathlib.Path("audit/integration/AUDIT_REPORT.md")
    if not report_path.exists():
        pytest.skip("report not generated")
    report = report_path.read_text()
    manifest = _frozen()
    snap = json.loads((pathlib.Path("audit/integration/evidence")
                       / manifest["authoritative_snapshot"]).read_text())
    assert f"**{snap['decision']['final_outcome']}**" in report
    assert f"**{snap['decision']['evidence_level']}**" in report
    for conflict_id in snap["decision"]["blocking_conflicts"]:
        assert conflict_id in report


def test_report_declares_rule_provenance_honestly():
    import pathlib
    report_path = pathlib.Path("audit/integration/AUDIT_REPORT.md")
    if not report_path.exists():
        pytest.skip("report not generated")
    report = report_path.read_text()
    assert "INTEGRATION_OPERATIONALIZATION" in report
    assert "NOT presented as methodology findings" in report
    assert "SOURCE_DEFINED" in report


def test_report_states_inconclusive_is_not_falsity():
    import pathlib
    report_path = pathlib.Path("audit/integration/AUDIT_REPORT.md")
    if not report_path.exists():
        pytest.skip("report not generated")
    report = report_path.read_text()
    assert "not that the claim is false" in report
