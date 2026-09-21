"""
Audit snapshot (methodology Sections 3.10.7, 3.13).

An immutable, versioned, point-in-time capture of the complete audit
state: every pillar's outcome, the evidence ledger, the artifact index,
the retained conflicts, the claim-level decision, and the integrity
checks that were actually run when the snapshot was taken.

The snapshot is SELF-BINDING: it records the content hash of each
component document it summarizes, so a reviewer can confirm that the
snapshot describes exactly those documents and not later-edited
versions of them. It creates no evidence and re-decides nothing; every
field is copied from an already-produced artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit.common.provenance import sha256_bytes, utc_timestamp
from audit.common.strict_json import dumps_strict

SNAPSHOT_SCHEMA_VERSION = "audit-snapshot-v1"


@dataclass(frozen=True)
class ComponentBinding:
    """Binds the snapshot to one component document by content hash."""
    component: str
    artifact_path: str
    content_hash: str

    def to_dict(self) -> dict[str, str]:
        return {"component": self.component, "artifact_path": self.artifact_path,
                "content_hash": self.content_hash}


@dataclass
class AuditSnapshot:
    claim_id: str
    claim_text: str
    decision: dict[str, Any]
    pillar_outcomes: dict[str, str]
    conflicts: list[dict[str, Any]]
    ledger_summary: dict[str, Any]
    artifact_index_summary: dict[str, Any]
    integrity_checks: dict[str, Any]
    scope_notes: dict[str, str]
    bindings: list[ComponentBinding] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    open_blockers: list[str] = field(default_factory=list)

    def bind(self, component: str, artifact_path: str, payload: str) -> None:
        self.bindings.append(ComponentBinding(
            component=component, artifact_path=artifact_path,
            content_hash=sha256_bytes(payload.encode("utf-8")),
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "generated_at_utc": utc_timestamp(),
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "pillar_outcomes": dict(self.pillar_outcomes),
            "decision": self.decision,
            "conflicts": list(self.conflicts),
            "evidence_ledger_summary": self.ledger_summary,
            "artifact_index_summary": self.artifact_index_summary,
            "integrity_checks": self.integrity_checks,
            "component_bindings": [b.to_dict() for b in self.bindings],
            "scope_notes": dict(self.scope_notes),
            "limitations": list(self.limitations),
            "open_blockers": list(self.open_blockers),
        }

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dumps_strict(self.to_dict(), indent=2, sort_keys=True)
        path.write_text(payload)
        return payload
