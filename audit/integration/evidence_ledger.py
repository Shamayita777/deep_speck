"""
Append-only evidence ledger (methodology Sections 3.4.3, 3.13.4).

The ledger is the single place a reviewer starts from a decision and
walks backwards to the artifacts that support it. It runs no
experiments, computes no statistics, and never edits an entry once
written.

TWO ENTRY KINDS, deliberately distinguished
-------------------------------------------
NATIVE
    A full audit.common.evidence.Evidence object produced by this
    framework, carrying run_ids that resolve to real RunManifests,
    provenance, and a confirmatory/exploratory tag justified by an
    actual preregistration check.

INGESTED_REFERENCE
    A pointer to a HISTORICAL artifact produced before this framework
    existed (the D1-D5 and CE1-CE4 certificates). These artifacts are
    real evidence and are cited by content hash, but they do NOT carry
    the run-manifest/provenance structure a NATIVE entry requires.

Forcing a historical artifact into the NATIVE schema would mean
inventing run_ids and provenance it never had - manufacturing exactly
the traceability the schema exists to guarantee. So the ledger records
them honestly as references, with an explicit `provenance_gaps` list
naming what is missing. A reviewer can therefore see, per entry,
whether the traceability chain is complete or merely asserted.

The ledger never rewrites, reconciles, or upgrades an ingested entry
into a native one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from audit.common.evidence import Evidence
from audit.common.ids import Dimension, validate_identifier
from audit.common.provenance import sha256_file, utc_timestamp
from audit.common.strict_json import dumps_strict

LEDGER_SCHEMA_VERSION = "evidence-ledger-v1"


class EntryKind(str, Enum):
    NATIVE = "NATIVE"
    INGESTED_REFERENCE = "INGESTED_REFERENCE"


class LedgerIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestedEvidenceReference:
    """
    A historical evidence artifact cited by content hash.

    `recorded_decision` is transcribed verbatim from the artifact - it
    is NOT translated into this framework's vocabulary here. D3's
    "DESCRIPTIVE_ONLY" and D4's "EFFECT_DETECTED" are precise,
    deliberately-chosen statements by their own pillar; coercing them
    into PASS/FAIL would discard information. Any interpretation is the
    decision engine's job and is recorded separately as an explicit
    step.
    """
    evidence_id: str
    claim_id: str
    dimension: Dimension
    source_pillar_id: str          # e.g. "D1", "CE2"
    artifact_path: str
    artifact_hash: str
    recorded_decision: str
    provenance_gaps: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_identifier(self.evidence_id, context="evidence_id")
        validate_identifier(self.claim_id, context="claim_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_kind": EntryKind.INGESTED_REFERENCE.value,
            "evidence_id": self.evidence_id, "claim_id": self.claim_id,
            "dimension": self.dimension.value, "source_pillar_id": self.source_pillar_id,
            "artifact_path": self.artifact_path, "artifact_hash": self.artifact_hash,
            "recorded_decision": self.recorded_decision,
            "recorded_decision_note": (
                "Transcribed verbatim from the source artifact; NOT translated into this "
                "framework's outcome vocabulary."
            ),
            "provenance_gaps": list(self.provenance_gaps),
            "limitations": list(self.limitations),
        }


@dataclass
class EvidenceLedger:
    """Append-only ledger. Entries are immutable once added."""
    entries: list[dict[str, Any]] = field(default_factory=list)
    _ids: set = field(default_factory=set)

    def _guard_duplicate(self, evidence_id: str) -> None:
        if evidence_id in self._ids:
            raise LedgerIntegrityError(
                f"evidence_id {evidence_id!r} is already in the ledger. The ledger is "
                "append-only; re-adding an id would either duplicate or silently shadow an "
                "existing entry. Version the evidence instead."
            )

    def add_native(self, evidence: Evidence) -> None:
        self._guard_duplicate(evidence.evidence_id)
        entry = {"entry_kind": EntryKind.NATIVE.value, **evidence.to_dict()}
        self.entries.append(entry)
        self._ids.add(evidence.evidence_id)

    def add_ingested(self, reference: IngestedEvidenceReference) -> None:
        self._guard_duplicate(reference.evidence_id)
        self.entries.append(reference.to_dict())
        self._ids.add(reference.evidence_id)

    def by_dimension(self, dimension: Dimension) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("dimension") == dimension.value]

    def by_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("claim_id") == claim_id]

    def evidence_ids(self) -> list[str]:
        return [e["evidence_id"] for e in self.entries]

    def verify_artifact_hashes(self, *, base_dirs: Optional[dict[str, Path]] = None) -> list[str]:
        """
        Re-verify every INGESTED entry's artifact hash against the file
        on disk. Returns a list of problems; empty means every cited
        artifact still matches the hash recorded when it was ingested.

        This is what makes the ledger tamper-evident: if a historical
        artifact were modified after ingestion, this reports it rather
        than silently continuing to cite a stale hash.
        """
        problems: list[str] = []
        base_dirs = base_dirs or {}
        for entry in self.entries:
            if entry.get("entry_kind") != EntryKind.INGESTED_REFERENCE.value:
                continue
            path = Path(entry["artifact_path"])
            if not path.is_absolute():
                base = base_dirs.get(entry["source_pillar_id"])
                if base is not None:
                    path = base / path
            if not path.exists():
                problems.append(
                    f"{entry['evidence_id']}: artifact {entry['artifact_path']!r} not found "
                    "for hash re-verification."
                )
                continue
            actual = sha256_file(path)
            if actual != entry["artifact_hash"]:
                problems.append(
                    f"{entry['evidence_id']}: artifact {entry['artifact_path']!r} hash changed "
                    f"since ingestion (recorded {entry['artifact_hash']}, actual {actual})."
                )
        return problems

    def to_dict(self) -> dict[str, Any]:
        native = sum(1 for e in self.entries if e["entry_kind"] == EntryKind.NATIVE.value)
        ingested = len(self.entries) - native
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "generated_at_utc": utc_timestamp(),
            "entry_counts": {"native": native, "ingested_reference": ingested,
                             "total": len(self.entries)},
            "entries": list(self.entries),
        }

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dumps_strict(self.to_dict(), indent=2, sort_keys=True)
        path.write_text(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceLedger":
        data = json.loads(Path(path).read_text())
        ledger = cls()
        for entry in data["entries"]:
            ledger.entries.append(entry)
            ledger._ids.add(entry["evidence_id"])
        return ledger
