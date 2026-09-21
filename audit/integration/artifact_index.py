"""
Artifact index (methodology Section 3.13, "Artifact Index" reporting
requirement).

Answers, for any artifact cited anywhere in the audit: what is it, what
is its content hash, does that hash still verify, and which evidence
entries depend on it. This is the reverse direction of the evidence
ledger - the ledger goes decision -> artifacts, the index goes
artifact -> dependents - and it is what makes the downstream impact of
a changed or invalidated artifact visible.

Generic: knows nothing about ciphers, models, or pillars beyond the
Dimension enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit.common.provenance import sha256_file, utc_timestamp
from audit.common.strict_json import dumps_strict

ARTIFACT_INDEX_SCHEMA_VERSION = "artifact-index-v1"


@dataclass
class ArtifactRecord:
    artifact_path: str
    recorded_hash: str
    artifact_role: str
    referenced_by: list[str] = field(default_factory=list)
    verification_status: str = "NOT_VERIFIED"
    verification_detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path, "recorded_hash": self.recorded_hash,
            "artifact_role": self.artifact_role,
            "referenced_by": sorted(set(self.referenced_by)),
            "verification_status": self.verification_status,
            "verification_detail": self.verification_detail,
        }


@dataclass
class ArtifactIndex:
    records: dict[str, ArtifactRecord] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def register(
        self, *, artifact_path: str, recorded_hash: str, artifact_role: str, referenced_by: str,
    ) -> None:
        """
        Register (or add a referrer to) an artifact.

        If the same path is registered with a DIFFERENT hash, that is
        recorded as a conflict rather than silently overwritten: two
        evidence entries citing the same path with different hashes
        means at least one of them is describing an artifact that no
        longer exists in that form, which a reviewer must see.
        """
        existing = self.records.get(artifact_path)
        if existing is None:
            self.records[artifact_path] = ArtifactRecord(
                artifact_path=artifact_path, recorded_hash=recorded_hash,
                artifact_role=artifact_role, referenced_by=[referenced_by],
            )
            return
        if existing.recorded_hash != recorded_hash:
            self.conflicts.append(
                f"Artifact {artifact_path!r} is cited with conflicting hashes: "
                f"{existing.recorded_hash} (by {existing.referenced_by}) vs {recorded_hash} "
                f"(by {referenced_by})."
            )
        existing.referenced_by.append(referenced_by)

    def verify(self, *, search_paths: Optional[list[Path]] = None) -> None:
        """
        Re-verify each artifact's hash against disk where the file can
        be located. Artifacts that cannot be located are marked
        NOT_LOCATED rather than assumed valid.
        """
        search_paths = search_paths or [Path(".")]
        for record in self.records.values():
            resolved: Optional[Path] = None
            candidate = Path(record.artifact_path)
            if candidate.is_absolute() and candidate.exists():
                resolved = candidate
            else:
                for base in search_paths:
                    trial = base / record.artifact_path
                    if trial.exists():
                        resolved = trial
                        break
                    trial = base / Path(record.artifact_path).name
                    if trial.exists():
                        resolved = trial
                        break
            if resolved is None:
                record.verification_status = "NOT_LOCATED"
                record.verification_detail = (
                    "File not found in the supplied search paths; hash could not be "
                    "re-verified. Not treated as verified."
                )
                continue
            actual = sha256_file(resolved)
            if actual == record.recorded_hash:
                record.verification_status = "VERIFIED"
                record.verification_detail = f"Hash matches file at {resolved}."
            else:
                record.verification_status = "HASH_MISMATCH"
                record.verification_detail = (
                    f"File at {resolved} hashes to {actual}, not the recorded "
                    f"{record.recorded_hash}."
                )

    def dependents_of(self, artifact_path: str) -> list[str]:
        record = self.records.get(artifact_path)
        return sorted(set(record.referenced_by)) if record else []

    def to_dict(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for record in self.records.values():
            statuses[record.verification_status] = statuses.get(record.verification_status, 0) + 1
        return {
            "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
            "generated_at_utc": utc_timestamp(),
            "artifact_count": len(self.records),
            "verification_summary": statuses,
            "hash_conflicts": list(self.conflicts),
            "artifacts": {k: v.to_dict() for k, v in sorted(self.records.items())},
        }

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dumps_strict(self.to_dict(), indent=2, sort_keys=True)
        path.write_text(payload)
        return payload


def build_from_ledger(ledger_dict: dict[str, Any]) -> ArtifactIndex:
    """
    Construct an artifact index from an evidence ledger's serialized
    form. Handles both entry kinds: NATIVE entries expose
    `artifact_hashes`, INGESTED_REFERENCE entries expose a single
    artifact_path/artifact_hash pair.
    """
    index = ArtifactIndex()
    for entry in ledger_dict["entries"]:
        evidence_id = entry["evidence_id"]
        if entry["entry_kind"] == "INGESTED_REFERENCE":
            index.register(
                artifact_path=entry["artifact_path"], recorded_hash=entry["artifact_hash"],
                artifact_role="historical_evidence_certificate", referenced_by=evidence_id,
            )
            continue
        for path, digest in (entry.get("artifact_hashes") or {}).items():
            role = "reference_source" if path.startswith("reference/") else (
                "run_manifest" if "run_manifest" in path else "inspected_artifact"
            )
            index.register(artifact_path=path, recorded_hash=digest,
                           artifact_role=role, referenced_by=evidence_id)
    return index
