import json

import pytest

from audit.common.evidence import Evidence
from audit.common.ids import Dimension
from audit.common.outcomes import EvidenceTag
from audit.integration.artifact_index import ArtifactIndex, build_from_ledger
from audit.integration.audit_snapshot import AuditSnapshot
from audit.integration.evidence_ledger import (
    EntryKind,
    EvidenceLedger,
    IngestedEvidenceReference,
    LedgerIntegrityError,
)


def _native(evidence_id="EV-N1", artifacts=None):
    return Evidence(
        evidence_id=evidence_id, claim_id="C1", dimension=Dimension.IMPLEMENTATION,
        experiment_id="EXP1", run_ids=["r1"], input_artifacts=list((artifacts or {}).keys()),
        artifact_hashes=dict(artifacts or {}), observation={}, statistics={},
        decision="INCONCLUSIVE", limitations=[], provenance={"g": 1},
        tag=EvidenceTag.EXPLORATORY,
    )


def _ingested(evidence_id="EV-I1", path="/tmp/x.json", digest="a" * 64, decision="PASS"):
    return IngestedEvidenceReference(
        evidence_id=evidence_id, claim_id="C1", dimension=Dimension.DATASET,
        source_pillar_id="D1", artifact_path=path, artifact_hash=digest,
        recorded_decision=decision, provenance_gaps=["no run manifest"],
    )


# --- ledger append-only semantics ---

def test_ledger_distinguishes_native_from_ingested():
    ledger = EvidenceLedger()
    ledger.add_native(_native())
    ledger.add_ingested(_ingested())
    kinds = [e["entry_kind"] for e in ledger.entries]
    assert kinds == [EntryKind.NATIVE.value, EntryKind.INGESTED_REFERENCE.value]


def test_ledger_rejects_duplicate_evidence_id():
    ledger = EvidenceLedger()
    ledger.add_native(_native("EV-DUP"))
    with pytest.raises(LedgerIntegrityError):
        ledger.add_ingested(_ingested("EV-DUP"))


def test_ingested_entry_records_provenance_gaps_rather_than_faking_them():
    """Historical artifacts must never be dressed up as fully-traceable native evidence."""
    entry = _ingested().to_dict()
    assert entry["provenance_gaps"]
    assert "run_ids" not in entry
    assert "NOT translated" in entry["recorded_decision_note"]


def test_ingested_decision_is_transcribed_not_translated():
    """D3's DESCRIPTIVE_ONLY and D4's EFFECT_DETECTED must survive verbatim."""
    for verbatim in ("DESCRIPTIVE_ONLY", "EFFECT_DETECTED"):
        entry = _ingested(decision=verbatim).to_dict()
        assert entry["recorded_decision"] == verbatim


def test_ledger_round_trips_through_disk(tmp_path):
    ledger = EvidenceLedger()
    ledger.add_native(_native())
    ledger.add_ingested(_ingested())
    path = tmp_path / "ledger.json"
    ledger.save(path)
    reloaded = EvidenceLedger.load(path)
    assert reloaded.evidence_ids() == ledger.evidence_ids()


def test_ledger_hash_reverification_detects_tampering(tmp_path):
    artifact = tmp_path / "hist.json"
    artifact.write_text('{"decision": {"outcome": "PASS"}}')
    from audit.common.provenance import sha256_file
    ledger = EvidenceLedger()
    ledger.add_ingested(_ingested(path=str(artifact), digest=sha256_file(artifact)))
    assert ledger.verify_artifact_hashes() == []
    artifact.write_text('{"decision": {"outcome": "FAIL"}}')  # tamper
    problems = ledger.verify_artifact_hashes()
    assert any("hash changed since ingestion" in p for p in problems)


def test_ledger_reports_missing_artifact_rather_than_passing():
    ledger = EvidenceLedger()
    ledger.add_ingested(_ingested(path="/nonexistent/missing.json"))
    problems = ledger.verify_artifact_hashes()
    assert any("not found" in p for p in problems)


def test_ledger_query_helpers():
    ledger = EvidenceLedger()
    ledger.add_native(_native())
    ledger.add_ingested(_ingested())
    assert len(ledger.by_dimension(Dimension.IMPLEMENTATION)) == 1
    assert len(ledger.by_dimension(Dimension.DATASET)) == 1
    assert len(ledger.by_claim("C1")) == 2


# --- artifact index ---

def test_index_records_referrers_for_each_artifact():
    index = ArtifactIndex()
    index.register(artifact_path="a.h5", recorded_hash="h1", artifact_role="ckpt",
                   referenced_by="EV-1")
    index.register(artifact_path="a.h5", recorded_hash="h1", artifact_role="ckpt",
                   referenced_by="EV-2")
    assert index.dependents_of("a.h5") == ["EV-1", "EV-2"]


def test_index_flags_conflicting_hashes_rather_than_overwriting():
    index = ArtifactIndex()
    index.register(artifact_path="a.h5", recorded_hash="h1", artifact_role="ckpt",
                   referenced_by="EV-1")
    index.register(artifact_path="a.h5", recorded_hash="h2", artifact_role="ckpt",
                   referenced_by="EV-2")
    assert index.conflicts
    assert index.records["a.h5"].recorded_hash == "h1"  # original not overwritten


def test_index_marks_unlocatable_artifacts_not_verified(tmp_path):
    index = ArtifactIndex()
    index.register(artifact_path="ghost.h5", recorded_hash="h1", artifact_role="ckpt",
                   referenced_by="EV-1")
    index.verify(search_paths=[tmp_path])
    assert index.records["ghost.h5"].verification_status == "NOT_LOCATED"


def test_index_detects_hash_mismatch_on_disk(tmp_path):
    artifact = tmp_path / "real.json"
    artifact.write_text("content")
    index = ArtifactIndex()
    index.register(artifact_path="real.json", recorded_hash="0" * 64,
                   artifact_role="x", referenced_by="EV-1")
    index.verify(search_paths=[tmp_path])
    assert index.records["real.json"].verification_status == "HASH_MISMATCH"


def test_index_builds_from_ledger_covering_both_entry_kinds():
    ledger = EvidenceLedger()
    ledger.add_native(_native(artifacts={"reference/speck.py": "s" * 64,
                                         "run_manifest_r1.json": "m" * 64}))
    ledger.add_ingested(_ingested())
    index = build_from_ledger(ledger.to_dict())
    assert "reference/speck.py" in index.records
    assert index.records["reference/speck.py"].artifact_role == "reference_source"
    assert index.records["run_manifest_r1.json"].artifact_role == "run_manifest"
    assert index.records["/tmp/x.json"].artifact_role == "historical_evidence_certificate"


# --- snapshot ---

def test_snapshot_binds_components_by_content_hash(tmp_path):
    snap = AuditSnapshot(
        claim_id="C1", claim_text="t", decision={"final_outcome": "INCONCLUSIVE"},
        pillar_outcomes={}, conflicts=[], ledger_summary={}, artifact_index_summary={},
        integrity_checks={}, scope_notes={},
    )
    snap.bind("evidence_ledger", "evidence_ledger.json", '{"a": 1}')
    d = snap.to_dict()
    assert d["component_bindings"][0]["component"] == "evidence_ledger"
    assert len(d["component_bindings"][0]["content_hash"]) == 64


def test_snapshot_binding_changes_when_component_changes():
    snap1 = AuditSnapshot("C1", "t", {}, {}, [], {}, {}, {}, {})
    snap1.bind("x", "x.json", "payload-A")
    snap2 = AuditSnapshot("C1", "t", {}, {}, [], {}, {}, {}, {})
    snap2.bind("x", "x.json", "payload-B")
    assert (snap1.to_dict()["component_bindings"][0]["content_hash"]
            != snap2.to_dict()["component_bindings"][0]["content_hash"])


def test_snapshot_records_open_blockers(tmp_path):
    snap = AuditSnapshot("C1", "t", {}, {}, [], {}, {}, {}, {},
                         open_blockers=["conflict unresolved"])
    payload = snap.save(tmp_path / "snap.json")
    assert "conflict unresolved" in payload


# --- end-to-end on the real generated artifacts ---

def test_generated_phase5_documents_are_internally_consistent():
    import pathlib
    base = pathlib.Path("audit/integration/evidence")
    if not (base / "audit_snapshot.json").exists():
        pytest.skip("Phase 5 documents not generated in this environment")

    snapshot = json.loads((base / "audit_snapshot.json").read_text())
    ledger = json.loads((base / "evidence_ledger.json").read_text())
    index = json.loads((base / "artifact_index.json").read_text())

    # Snapshot's ledger summary must match the ledger it bound.
    assert snapshot["evidence_ledger_summary"]["entry_counts"] == ledger["entry_counts"]
    assert snapshot["artifact_index_summary"]["artifact_count"] == index["artifact_count"]

    # Every evidence id the snapshot lists must exist in the ledger.
    assert set(snapshot["evidence_ledger_summary"]["evidence_ids"]) == {
        e["evidence_id"] for e in ledger["entries"]}

    # No hash conflicts, and the recorded integrity check agrees.
    assert index["hash_conflicts"] == []
    assert snapshot["integrity_checks"]["ledger_artifact_hash_reverification"] == "ALL_VERIFIED"

    # The blocking conflict must be visible in the decision.
    assert "CONFLICT-CE2-CE34-V1" in snapshot["decision"]["blocking_conflicts"]
    assert snapshot["decision"]["final_outcome"] == "INCONCLUSIVE"


def test_generated_ledger_preserves_verbatim_historical_decisions():
    import pathlib
    base = pathlib.Path("audit/integration/evidence")
    if not (base / "evidence_ledger.json").exists():
        pytest.skip("Phase 5 documents not generated in this environment")
    ledger = json.loads((base / "evidence_ledger.json").read_text())
    decisions = {e["source_pillar_id"]: e["recorded_decision"]
                 for e in ledger["entries"]
                 if e["entry_kind"] == "INGESTED_REFERENCE"}
    # These are the actual recorded values in the historical artifacts.
    assert decisions.get("D1") == "PASS"
    assert decisions.get("D2") == "INCONCLUSIVE"
    assert decisions.get("D3") == "DESCRIPTIVE_ONLY"
    assert decisions.get("D4") == "EFFECT_DETECTED"
    assert decisions.get("CE1") == "INCONCLUSIVE"
    assert decisions.get("CE3") == "SUPPORTED"
    assert decisions.get("CE4") == "SUPPORTED"
