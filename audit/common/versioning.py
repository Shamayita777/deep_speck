"""
Immutable versioning for audit artifacts.

Implements the methodology's repeated requirement that scientific
evidence is never silently overwritten: a "DATASET-AUDIT-V1"-style
identifier, once registered against a specific content hash, is
permanent. Registering the same base name with DIFFERENT content
creates "-V2", "-V3", etc.; the original remains readable forever.
This module knows nothing about what an "artifact" actually is
(a dataset snapshot, a certificate, a decision) - it only manages the
name -> version -> content_hash mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from audit.common.ids import validate_identifier
from audit.common.provenance import utc_timestamp
from audit.common.strict_json import dumps_strict


@dataclass(frozen=True)
class VersionRecord:
    base_name: str
    version: int
    identifier: str
    content_hash: str
    created_at_utc: str
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "base_name": self.base_name,
            "version": self.version,
            "identifier": self.identifier,
            "content_hash": self.content_hash,
            "created_at_utc": self.created_at_utc,
            "note": self.note,
        }


class VersioningConflictError(RuntimeError):
    pass


class VersionRegistry:
    """
    On-disk, append-only registry of versioned artifacts. One JSON file
    per registry (e.g. one for dataset snapshots, one for certificates
    of a given kind) - callers choose the scope.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(dumps_strict({}, indent=2, sort_keys=True))

    def _load(self) -> dict:
        return json.loads(self.path.read_text())

    def _save(self, data: dict) -> None:
        self.path.write_text(dumps_strict(data, indent=2, sort_keys=True))

    def register(self, base_name: str, content_hash: str, *, note: Optional[str] = None) -> VersionRecord:
        """
        Register a new content hash under base_name. If a version with
        the IDENTICAL content_hash already exists, that same version is
        returned (idempotent - re-registering the same evidence twice
        does not create a spurious new version). If base_name exists
        but with a DIFFERENT content_hash, a new, incremented version is
        created and the previous one is left untouched.
        """
        validate_identifier(base_name, context="base_name")
        data = self._load()
        versions = data.setdefault(base_name, [])

        for existing in versions:
            if existing["content_hash"] == content_hash:
                return VersionRecord(**existing)

        next_version = max((v["version"] for v in versions), default=0) + 1
        identifier = f"{base_name}-V{next_version}"
        record = VersionRecord(
            base_name=base_name, version=next_version, identifier=identifier,
            content_hash=content_hash, created_at_utc=utc_timestamp(), note=note,
        )
        versions.append(record.to_dict())
        self._save(data)
        return record

    def get(self, base_name: str, version: int) -> Optional[VersionRecord]:
        data = self._load()
        for existing in data.get(base_name, []):
            if existing["version"] == version:
                return VersionRecord(**existing)
        return None

    def latest(self, base_name: str) -> Optional[VersionRecord]:
        data = self._load()
        versions = data.get(base_name, [])
        if not versions:
            return None
        return VersionRecord(**max(versions, key=lambda v: v["version"]))

    def all_versions(self, base_name: str) -> list[VersionRecord]:
        data = self._load()
        return [VersionRecord(**v) for v in sorted(data.get(base_name, []), key=lambda v: v["version"])]
