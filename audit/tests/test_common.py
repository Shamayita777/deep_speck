import pytest

from audit.common.evidence import Evidence, EvidenceValidationError
from audit.common.ids import Dimension, InvalidIdentifierError, validate_identifier, validate_versioned_identifier
from audit.common.outcomes import CryptographicOutcome, DimensionOutcome, EvidenceTag
from audit.common.provenance import Activity, Entity, ProvenanceGraph, sha256_bytes
from audit.common.run_manifest import RunManifest, capture_environment
from audit.common.outcomes import ExecutionMode, RunStatus
from audit.common.strict_json import StrictJSONError, dumps_strict, find_non_finite_paths
from audit.common.versioning import VersionRegistry


def test_validate_identifier_rejects_whitespace_and_leading_digit():
    with pytest.raises(InvalidIdentifierError):
        validate_identifier("has space")
    with pytest.raises(InvalidIdentifierError):
        validate_identifier("1LeadingDigit")
    assert validate_identifier("C1") == "C1"
    assert validate_identifier("H-EV-SHUFFLE") == "H-EV-SHUFFLE"


def test_validate_versioned_identifier():
    base, version = validate_versioned_identifier("DATASET-AUDIT-V1")
    assert base == "DATASET-AUDIT" and version == 1
    with pytest.raises(InvalidIdentifierError):
        validate_versioned_identifier("DATASET-AUDIT")  # no version suffix
    with pytest.raises(InvalidIdentifierError):
        validate_versioned_identifier("DATASET-AUDIT-V0")  # version must be >= 1


def test_strict_json_rejects_nan_inf_and_reports_path():
    with pytest.raises(StrictJSONError):
        dumps_strict({"a": {"b": [1.0, float("nan")]}})
    paths = find_non_finite_paths({"a": {"b": [1.0, float("inf"), float("-inf")]}})
    assert paths == ["$.a.b[1]", "$.a.b[2]"]


def test_strict_json_accepts_finite_values():
    s = dumps_strict({"a": 1, "b": [1.0, 2.5], "c": None})
    assert "1" in s


def test_provenance_graph_used_generated_derived_from():
    g = ProvenanceGraph()
    g.add_entity(Entity("dataset", "dataset_snapshot"))
    g.add_activity(Activity("train", "training", started_at_utc="t0"))
    g.used("train", "dataset")
    g.add_entity(Entity("checkpoint", "model_checkpoint"))
    g.generated("checkpoint", "train")
    g.add_entity(Entity("evidence1", "evidence"))
    g.derived_from("evidence1", "checkpoint")

    trace = g.trace_backward("evidence1")
    assert "checkpoint" in trace["entities"]


def test_provenance_graph_merge_is_non_destructive():
    g1 = ProvenanceGraph()
    g1.add_entity(Entity("e1", "x"))
    g2 = ProvenanceGraph()
    g2.add_entity(Entity("e2", "y"))
    merged = g1.merge(g2)
    assert "e1" in merged.entities and "e2" in merged.entities
    assert "e2" not in g1.entities  # original untouched


def test_evidence_requires_run_ids():
    with pytest.raises(EvidenceValidationError):
        Evidence(
            evidence_id="EV-1", claim_id="C1", dimension=Dimension.DATASET, experiment_id="D1",
            run_ids=[], input_artifacts=[], artifact_hashes={}, observation={}, statistics={},
            decision=DimensionOutcome.PASS.value, limitations=[], provenance={}, tag=EvidenceTag.CONFIRMATORY,
        )


def test_evidence_rejects_non_finite_statistics():
    with pytest.raises(EvidenceValidationError):
        Evidence(
            evidence_id="EV-1", claim_id="C1", dimension=Dimension.DATASET, experiment_id="D1",
            run_ids=["r1"], input_artifacts=[], artifact_hashes={}, observation={},
            statistics={"p_value": float("nan")}, decision=DimensionOutcome.PASS.value,
            limitations=[], provenance={}, tag=EvidenceTag.CONFIRMATORY,
        )


def test_evidence_valid_construction_round_trips_to_dict():
    ev = Evidence(
        evidence_id="EV-1", claim_id="C1", dimension=Dimension.CRYPTOGRAPHIC, experiment_id="CE1",
        run_ids=["r1"], input_artifacts=["a.json"], artifact_hashes={"a.json": sha256_bytes(b"x")},
        observation={"note": "ok"}, statistics={"n": 5}, decision=CryptographicOutcome.INCONCLUSIVE.value,
        limitations=["small n"], provenance={}, tag=EvidenceTag.CONFIRMATORY,
    )
    d = ev.to_dict()
    assert d["dimension"] == "CRYPTOGRAPHIC_EVIDENCE"
    assert d["tag"] == "CONFIRMATORY"


def test_run_manifest_lifecycle():
    env = capture_environment()
    manifest = RunManifest(
        run_id="run1", experiment_id="EXP1", condition_id="baseline", replicate_id="r0",
        claim_ids=["C1"], hypothesis_ids=["H1"], execution_mode=ExecutionMode.SMOKE,
        config_hash="abc123", preregistration_hash=None, dataset_snapshot_id=None,
        dataset_hashes={}, code_hashes={}, model_source_hash=None, environment=env,
        seeds={"model_seed": 42, "dataset_seed": None}, command="pytest",
        start_time_utc="2026-01-01T00:00:00Z",
    )
    assert manifest.status == RunStatus.SKIPPED
    manifest.mark_completed(exit_code=0, metrics={"accuracy": 0.9})
    assert manifest.status == RunStatus.COMPLETED
    assert manifest.to_dict()["metrics"]["accuracy"] == 0.9


def test_run_manifest_failure_and_abort_remain_visible():
    manifest = RunManifest(
        run_id="run2", experiment_id="EXP1", condition_id=None, replicate_id=None,
        claim_ids=[], hypothesis_ids=[], execution_mode=ExecutionMode.PRODUCTION,
        config_hash="x", preregistration_hash=None, dataset_snapshot_id=None,
        dataset_hashes={}, code_hashes={}, model_source_hash=None, environment={},
        seeds={}, command="run.py", start_time_utc="t0",
    )
    manifest.mark_aborted(reason="OOM")
    assert manifest.status == RunStatus.ABORTED
    assert "OOM" in manifest.limitations[0]


def test_version_registry_never_overwrites_different_content(tmp_path):
    reg = VersionRegistry(tmp_path / "reg.json")
    v1 = reg.register("X", sha256_bytes(b"content-a"))
    v2 = reg.register("X", sha256_bytes(b"content-b"))
    assert v1.identifier == "X-V1"
    assert v2.identifier == "X-V2"
    # v1 must remain inspectable after v2 is created.
    assert reg.get("X", 1).content_hash == sha256_bytes(b"content-a")
    assert reg.latest("X").identifier == "X-V2"
    assert len(reg.all_versions("X")) == 2


def test_version_registry_idempotent_for_identical_content(tmp_path):
    reg = VersionRegistry(tmp_path / "reg.json")
    v1 = reg.register("X", sha256_bytes(b"same"))
    v1b = reg.register("X", sha256_bytes(b"same"))
    assert v1.identifier == v1b.identifier
    assert len(reg.all_versions("X")) == 1


# --- provenance integrity corrections ---

def test_entity_id_collision_with_different_content_raises():
    from audit.common.provenance import ProvenanceCollisionError
    g = ProvenanceGraph()
    g.add_entity(Entity("e1", "dataset", content_hash="aaa"))
    with pytest.raises(ProvenanceCollisionError):
        g.add_entity(Entity("e1", "dataset", content_hash="bbb"))


def test_entity_reregistration_with_identical_content_is_idempotent():
    g = ProvenanceGraph()
    e = Entity("e1", "dataset", content_hash="aaa")
    g.add_entity(e)
    g.add_entity(Entity("e1", "dataset", content_hash="aaa"))
    assert len(g.entities) == 1


def test_merge_detects_conflicting_ids():
    from audit.common.provenance import ProvenanceCollisionError
    g1 = ProvenanceGraph(); g1.add_entity(Entity("shared", "dataset", content_hash="aaa"))
    g2 = ProvenanceGraph(); g2.add_entity(Entity("shared", "dataset", content_hash="bbb"))
    with pytest.raises(ProvenanceCollisionError):
        g1.merge(g2)


def test_merge_succeeds_and_deduplicates_relations():
    g1 = ProvenanceGraph()
    g1.add_entity(Entity("e1", "x")); g1.add_activity(Activity("a1", "t", started_at_utc="t0"))
    g1.used("a1", "e1")
    g2 = ProvenanceGraph()
    g2.add_entity(Entity("e1", "x")); g2.add_activity(Activity("a1", "t", started_at_utc="t0"))
    g2.used("a1", "e1")
    merged = g1.merge(g2)
    assert len(merged.relations) == 1


def test_validate_flags_generating_activity_without_end_timestamp():
    g = ProvenanceGraph()
    g.add_activity(Activity("a1", "training", started_at_utc="t0"))
    g.add_entity(Entity("e1", "checkpoint"))
    g.generated("e1", "a1")
    problems = g.validate()
    assert any("ended_at_utc" in p for p in problems)


def test_validate_passes_after_complete_activity():
    g = ProvenanceGraph()
    g.add_activity(Activity("a1", "training", started_at_utc="t0"))
    g.add_entity(Entity("e1", "checkpoint"))
    g.generated("e1", "a1")
    g.complete_activity("a1")
    assert g.validate() == []


def test_validate_flags_relation_to_unknown_node():
    g = ProvenanceGraph()
    g.add_activity(Activity("a1", "t", started_at_utc="t0"))
    g.used("a1", "nonexistent-entity")
    assert any("unknown node" in p for p in g.validate())


# --- evidence provenance validation ---

def _evidence(tag=EvidenceTag.CONFIRMATORY, run_ids=("r1",), provenance=None):
    return Evidence(
        evidence_id="EV-1", claim_id="C1", dimension=Dimension.IMPLEMENTATION,
        experiment_id="EXP1", run_ids=list(run_ids), input_artifacts=[], artifact_hashes={},
        observation={}, statistics={}, decision="PASS", limitations=[],
        provenance=provenance if provenance is not None else {"graph": "x"}, tag=tag,
    )


def test_evidence_citing_unknown_run_id_is_flagged():
    from audit.common.evidence import validate_evidence_provenance
    problems = validate_evidence_provenance(_evidence(), known_run_manifests={})
    assert any("no corresponding RunManifest" in p for p in problems)


def test_confirmatory_evidence_without_provenance_is_flagged():
    from audit.common.evidence import validate_evidence_provenance
    manifests = {"r1": {"preregistration_hash": "abc"}}
    problems = validate_evidence_provenance(
        _evidence(provenance={}), known_run_manifests=manifests)
    assert any("carries no provenance" in p for p in problems)


def test_confirmatory_evidence_without_preregistration_is_flagged():
    from audit.common.evidence import validate_evidence_provenance
    manifests = {"r1": {"preregistration_hash": None}}
    problems = validate_evidence_provenance(_evidence(), known_run_manifests=manifests)
    assert any("no preregistration_hash" in p for p in problems)


def test_exploratory_evidence_does_not_require_preregistration():
    from audit.common.evidence import validate_evidence_provenance
    manifests = {"r1": {"preregistration_hash": None}}
    problems = validate_evidence_provenance(
        _evidence(tag=EvidenceTag.EXPLORATORY), known_run_manifests=manifests)
    assert problems == []


def test_fully_traceable_confirmatory_evidence_validates_clean():
    from audit.common.evidence import validate_evidence_provenance
    manifests = {"r1": {"preregistration_hash": "abc"}}
    assert validate_evidence_provenance(_evidence(), known_run_manifests=manifests) == []
