#!/usr/bin/env python3
"""
Cross-pillar integration runner.

Consumes ALREADY-PRODUCED pillar evidence and emits a versioned
claim-level decision. Runs no experiments and modifies no pillar
evidence (D1-D5, EV, CE and the II certificates are all read-only
inputs here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from audit.common.outcomes import CryptographicOutcome, DimensionOutcome
from audit.common.provenance import sha256_bytes
from audit.common.strict_json import dumps_strict
from audit.common.versioning import VersionRegistry
from audit.common.evidence import Evidence
from audit.common.ids import Dimension
from audit.common.provenance import sha256_file
from audit.integration.ce_mapping import ALL_CE_MAPPINGS, KEY_RECOVERY_SCOPE_NOTE
from audit.integration.artifact_index import build_from_ledger
from audit.integration.audit_snapshot import AuditSnapshot
from audit.integration.conflict_resolution import ALL_CONFLICTS
from audit.integration.decision_engine import decide_claim
from audit.integration.evidence_ledger import EvidenceLedger, IngestedEvidenceReference
from audit.integration.pillar_interpretation import (
    aggregate_dimension,
    interpret_native_status,
    to_cryptographic_outcome,
    to_dimension_outcome,
)

import json

# Historical artifacts ingested BY REFERENCE. Each is a real file whose
# hash is computed at ingestion time; `recorded_decision` is transcribed
# verbatim from the artifact, never translated. `provenance_gaps` names
# what these pre-framework artifacts genuinely lack.
HISTORICAL_SOURCES = [
    ("EV-D1-INGESTED", "D1", Dimension.DATASET,
     "dataset/d1/d1_gohr_10000000_1000000_1000000_5r.json"),
    ("EV-D2-INGESTED", "D2", Dimension.DATASET,
     "dataset/d2/d2_full_v64.json"),
    ("EV-D3-INGESTED", "D3", Dimension.DATASET,
     "dataset/d3/d3_gohr_10000000_1000000_1000000_5r.json"),
    ("EV-D4-INGESTED", "D4", Dimension.DATASET,
     "dataset/d4/d4_gohr_label_shuffle.json"),
    ("EV-CE1-INGESTED", "CE1", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce1/ce1_certificate.json"),
    # CE2 has FIVE independent run certificates, all recording
    # NOT_SUPPORTED. Each is ingested separately: collapsing five runs
    # into one entry would discard the replication, and omitting them
    # entirely (as an earlier build did) caused genuinely-existing
    # evidence to be misreported as NOT_PRODUCED.
    ("EV-CE2-RUN1-INGESTED", "CE2", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce2/ce2_certificate_1.json"),
    ("EV-CE2-RUN2-INGESTED", "CE2", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce2/ce2_certificate_2.json"),
    ("EV-CE2-RUN3-INGESTED", "CE2", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce2/ce2_certificate_3.json"),
    ("EV-CE2-RUN4-INGESTED", "CE2", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce2/ce2_certificate_4.json"),
    ("EV-CE2-RUN5-INGESTED", "CE2", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce2/ce2_certificate_5.json"),
    ("EV-CE3-INGESTED", "CE3", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce3/ce3_certificate.json"),
    ("EV-CE4-INGESTED", "CE4", Dimension.CRYPTOGRAPHIC,
     "cryptography/ce4/ce4_certificate.json"),
]

COMMON_PROVENANCE_GAPS = [
    "No RunManifest: produced before the framework's run-manifest schema existed.",
    "No preregistration artifact, so the entry cannot be certified CONFIRMATORY.",
    "No structured provenance graph binding it to its inputs.",
]


def _transcribe_decision(payload: dict) -> str:
    """Read the artifact's own recorded outcome verbatim; never translate it."""
    for keys in (("decision", "outcome"), ("decision", "status")):
        node = payload
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str):
            return node
    if isinstance(payload.get("decision"), str):
        return payload["decision"]
    return "NOT_RECORDED_IN_ARTIFACT"


def build_ledger(source_root: Path, implementation_certificate: Optional[Path]) -> EvidenceLedger:
    ledger = EvidenceLedger()

    if implementation_certificate and implementation_certificate.exists():
        data = json.loads(implementation_certificate.read_text())
        ledger.add_native(Evidence(
            evidence_id=data["evidence_id"], claim_id=data["claim_id"],
            dimension=Dimension(data["dimension"]), experiment_id=data["experiment_id"],
            run_ids=data["run_ids"], input_artifacts=data["input_artifacts"],
            artifact_hashes=data["artifact_hashes"], observation=data["observation"],
            statistics=data["statistics"], decision=data["decision"],
            limitations=data["limitations"], provenance=data["provenance"],
            tag=data["tag"],
        ))

    for evidence_id, pillar, dimension, rel_path in HISTORICAL_SOURCES:
        path = source_root / rel_path
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        ledger.add_ingested(IngestedEvidenceReference(
            evidence_id=evidence_id, claim_id=CLAIM_ID, dimension=dimension,
            source_pillar_id=pillar, artifact_path=str(path),
            artifact_hash=sha256_file(path),
            recorded_decision=_transcribe_decision(payload),
            provenance_gaps=list(COMMON_PROVENANCE_GAPS),
            limitations=(
                ["Produced against the depth-5 checkpoint that diverges from the declared "
                 "reference protocol (II-FINDING-DEPTH-V1)."]
                if dimension is Dimension.CRYPTOGRAPHIC else []
            ),
        ))
    return ledger

BUNDLE_DIR_NAME = "audit/evidence_bundle"
CLAIM_ID = "C1"
CLAIM_TEXT = (
    "For the 5-round Gohr/Speck32/64 neural distinguisher under differential "
    "(0x0040, 0x0000), the observed above-chance distinguishing performance is supported by "
    "cryptographically meaningful evidence after dataset, implementation and experimental "
    "alternative explanations have been evaluated."
)


# Which pillars/sub-audits are DECISION-BEARING for claim C1, and why.
# Declared explicitly so a reviewer can see what is allowed to move the
# decision and what is merely recorded.
DECISION_BEARING: dict[str, bool] = {
    # Dataset: every D sub-audit is decision-bearing for dataset integrity.
    # D1..D5 are evidence COMPONENTS, not independently decisive subtests -
    # established from their own text (D3 "does not establish equality of the
    # full underlying distributions"; D5 self-describes as
    # "descriptive/estimative"). A component that disclaims establishing its
    # own conclusion still bears on whether the DIMENSION is established.
    "D1": True, "D2": True, "D3": True, "D4": True, "D5": True,
    # Cryptographic: CE1..CE4 each test a distinct cryptographic hypothesis.
    "CE1": True, "CE2": True, "CE3": True, "CE4": True,
    # Implementation: the native II certificate.
    "II": True,
    # Experimental: no production evidence exists; represented explicitly below.
    "EV": True,
}


def derive_pillar_state(ledger: EvidenceLedger) -> tuple[dict, dict]:
    """
    Derive the four dimension outcomes FROM THE LEDGER via the
    interpretation layer.

    This function contains NO hard-coded DimensionOutcome/
    CryptographicOutcome production values. Every outcome it returns is
    computed from evidence entries actually present in the ledger. If a
    ledger entry's recorded decision changes, the derived state changes
    with it - that causal link is exercised by
    tests/test_phase5_derivation.py.

    Returns (state, interpretations) where `interpretations` carries the
    full audit trail for the snapshot.
    """
    by_dimension: dict[Dimension, list] = {d: [] for d in Dimension}

    for entry in ledger.entries:
        dimension = Dimension(entry["dimension"])
        if entry["entry_kind"] == "INGESTED_REFERENCE":
            pillar = entry["source_pillar_id"]
            native = entry["recorded_decision"]
            refs = [entry["artifact_path"]]
            lims = entry.get("limitations", [])
        else:
            pillar = "II"
            native = entry["decision"]
            refs = sorted((entry.get("artifact_hashes") or {}).keys())
            lims = entry.get("limitations", [])
        by_dimension[dimension].append(interpret_native_status(
            source_evidence_id=entry["evidence_id"], source_pillar_id=pillar,
            dimension=dimension, native_status=native,
            supporting_references=refs, limitations=lims,
            decision_bearing=DECISION_BEARING.get(pillar, True),
        ))

    # Components with NO evidence in the ledger must still be represented,
    # otherwise a dimension could appear established purely because nothing
    # was recorded for it. Absence is evidence of absence here, and is
    # interpreted as NOT_PRODUCED rather than ignored.
    present_pillars = {r.source_pillar_id for recs in by_dimension.values() for r in recs}
    for pillar, dimension, note in (
        ("D5", Dimension.DATASET,
         "D5 has no production certificate: its artifacts carry findings and interpretation "
         "but no decision/outcome field (verified by inspection)."),
        ("EV", Dimension.EXPERIMENTAL,
         "Experimental Validity has no production evidence: the framework exists but production "
         "runs remain blocked on epsilon, replicate count and runtime feasibility."),
    ):
        if pillar not in present_pillars:
            by_dimension[dimension].append(interpret_native_status(
                source_evidence_id=f"EV-{pillar}-ABSENT", source_pillar_id=pillar,
                dimension=dimension, native_status="NOT_PRODUCED",
                supporting_references=[], limitations=[note],
                decision_bearing=DECISION_BEARING.get(pillar, True),
            ))

    # Evidence party to an UNRESOLVED conflict, derived from the
    # conflict records themselves - not hard-coded here.
    conflicted_ids: set = set()
    for conflict in ALL_CONFLICTS:
        if not conflict.blocks_claim:
            continue
        for observation in conflict.observations:
            conflicted_ids.add(observation.source_id)

    interpretations = {
        d: aggregate_dimension(d, by_dimension[d], conflicted_evidence_ids=conflicted_ids)
        for d in Dimension
    }
    state = {
        "implementation": to_dimension_outcome(interpretations[Dimension.IMPLEMENTATION]),
        "dataset": to_dimension_outcome(interpretations[Dimension.DATASET]),
        "experimental": to_dimension_outcome(interpretations[Dimension.EXPERIMENTAL]),
        "cryptographic": to_cryptographic_outcome(interpretations[Dimension.CRYPTOGRAPHIC]),
    }
    return state, {d.value: i.to_dict() for d, i in interpretations.items()}


def run(output_dir: Path, source_root: Path, implementation_certificate: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # The ledger is built FIRST and the pillar state is derived FROM it,
    # so the ledger is causally upstream of the decision.
    ledger = build_ledger(source_root, implementation_certificate)
    state, interpretations = derive_pillar_state(ledger)

    decision = decide_claim(
        claim_id=CLAIM_ID,
        implementation=state["implementation"],
        dataset=state["dataset"],
        experimental=state["experimental"],
        cryptographic=state["cryptographic"],
        conflicts=ALL_CONFLICTS,
        convergence_sources=[],  # none demonstrated
        supporting_evidence_ids=[
            "implementation-certificate-V3", "evidence/ce1/ce1_certificate.json",
            "evidence/ce2/ce2_certificate_{1..5}.json", "evidence/ce3/ce3_certificate.json",
            "evidence/ce4/ce4_certificate.json", "dataset/evidence/d1..d5",
        ],
        limitations=[
            "No Experimental Validity production evidence exists yet.",
            "Dataset Integrity is not fully established: D2 is INCONCLUSIVE and explicitly "
            "records that it is not proof of independence; D5 has smoke evidence only.",
            "All CE1-CE4 evidence was produced against a depth-5 checkpoint that diverges "
            "from the declared reference protocol's depth=10 (II-FINDING-DEPTH-V1).",
            "No independent convergence source exists, so LEVEL_4 is unreachable on the "
            "current evidence - this reflects absent convergence evidence, not a policy cap.",
            KEY_RECOVERY_SCOPE_NOTE,
        ],
    )

    # --- Evidence ledger (already built above; persisted here) ---
    ledger_payload = ledger.save(output_dir / "evidence_ledger.json")
    hash_problems = ledger.verify_artifact_hashes()

    # --- Artifact index ---
    index = build_from_ledger(ledger.to_dict())
    index.verify(search_paths=[
        Path("."), output_dir,
        Path("audit/implementation/evidence"), Path("audit/implementation"),
        # Inspected checkpoints are vendored into the archive, so they
        # re-verify with no external path. Anything still unlocatable stays
        # NOT_LOCATED and is never assumed valid.
        Path(BUNDLE_DIR_NAME) / "cryptography" / "ce1",
        source_root / "cryptography" / "ce1",
    ])
    index_payload = index.save(output_dir / "artifact_index.json")

    # --- Decision document ---
    decision_payload = dumps_strict({
        "claim_id": CLAIM_ID, "claim_text": CLAIM_TEXT,
        "decision": decision.to_dict(),
        "ce_mappings": [m.to_dict() for m in ALL_CE_MAPPINGS],
        "conflicts": [c.to_dict() for c in ALL_CONFLICTS],
        "scope_notes": {"key_recovery": KEY_RECOVERY_SCOPE_NOTE},
    }, indent=2, sort_keys=True)
    (output_dir / "decision.json").write_text(decision_payload)

    # --- Snapshot, self-bound to the documents above ---
    ledger_dict = ledger.to_dict()
    index_dict = index.to_dict()
    snapshot = AuditSnapshot(
        claim_id=CLAIM_ID, claim_text=CLAIM_TEXT, decision=decision.to_dict(),
        pillar_outcomes={
            "implementation_integrity": state["implementation"].value,
            "dataset_integrity": state["dataset"].value,
            "experimental_validity": state["experimental"].value,
            "cryptographic_evidence": state["cryptographic"].value,
        },
        conflicts=[c.to_dict() for c in ALL_CONFLICTS],
        ledger_summary={
            "schema_version": ledger_dict["schema_version"],
            "entry_counts": ledger_dict["entry_counts"],
            "evidence_ids": ledger.evidence_ids(),
        },
        artifact_index_summary={
            "artifact_count": index_dict["artifact_count"],
            "verification_summary": index_dict["verification_summary"],
            "hash_conflicts": index_dict["hash_conflicts"],
        },
        integrity_checks={
            "ledger_artifact_hash_reverification": (
                "ALL_VERIFIED" if not hash_problems else "PROBLEMS_FOUND"
            ),
            "ledger_artifact_hash_problems": hash_problems,
            "artifact_index_hash_conflicts": index_dict["hash_conflicts"],
        },
        scope_notes={"key_recovery": KEY_RECOVERY_SCOPE_NOTE},
        limitations=list(decision.limitations),
        open_blockers=[
            "CONFLICT-CE2-CE34-V1 unresolved; blocks claim C1.",
            "CE1-CE4 evidence bound to a non-conformant depth-5 checkpoint.",
            "No Experimental Validity production evidence (epsilon, replicate count, runtime "
            "benchmark all unresolved).",
            "Dataset Integrity incomplete: D2 INCONCLUSIVE, D5 smoke-only.",
            "No independent convergence source, so LEVEL_4 is unreachable on current evidence.",
        ],
    )
    interpretation_payload = dumps_strict(
        {"version": "pillar-interpretation-v1", "interpretations": interpretations},
        indent=2, sort_keys=True)
    (output_dir / "pillar_interpretation.json").write_text(interpretation_payload)
    snapshot.bind("pillar_interpretation", "pillar_interpretation.json", interpretation_payload)
    snapshot.bind("evidence_ledger", "evidence_ledger.json", ledger_payload)
    snapshot.bind("artifact_index", "artifact_index.json", index_payload)
    snapshot.bind("decision", "decision.json", decision_payload)
    snapshot_payload = snapshot.save(output_dir / "audit_snapshot.json")

    registry = VersionRegistry(output_dir / "version_registry.json")
    record = registry.register(
        "audit-snapshot", sha256_bytes(snapshot_payload.encode("utf-8")),
        note="Cross-pillar audit snapshot for the 5-round Gohr distinguisher case study.",
    )
    (output_dir / f"{record.identifier}.json").write_text(snapshot_payload)

    print(f"Claim {CLAIM_ID} final outcome : {decision.final_outcome.value}")
    print(f"Evidence level               : {decision.evidence_level.value}")
    print(f"  implementation : {decision.implementation.value}")
    print(f"  dataset        : {decision.dataset.value}")
    print(f"  experimental   : {decision.experimental.value}")
    print(f"  cryptographic  : {decision.cryptographic.value}")
    print(f"Blocking conflicts           : {decision.blocking_conflicts}")
    print(f"Ledger entries               : {ledger_dict['entry_counts']}")
    print("Pillar state DERIVED from ledger:")
    for dim, interp in interpretations.items():
        print(f"    {dim:24} {interp['outcome']:18} "
              f"driven by {interp['driving_evidence_ids']}")
    print(f"Artifact index               : {index_dict['artifact_count']} artifacts, "
          f"{index_dict['verification_summary']}")
    print(f"Ledger hash re-verification  : "
          f"{'ALL_VERIFIED' if not hash_problems else hash_problems}")
    print(f"Snapshot written             : {output_dir / (record.identifier + '.json')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("audit/integration/evidence"))
    parser.add_argument("--source-root", type=Path, default=Path("audit/evidence_bundle"),
                        help=("Root of the hash-bound evidence bundle. Defaults to the "
                              "vendored audit/evidence_bundle, which makes the archive "
                              "independently verifiable with no external dependency."))
    parser.add_argument("--implementation-certificate", type=Path,
                        default=Path("audit/implementation/evidence/"
                                     "implementation-certificate-V3.json"))
    args = parser.parse_args()
    return run(args.output_dir, args.source_root, args.implementation_certificate)


if __name__ == "__main__":
    sys.exit(main())
