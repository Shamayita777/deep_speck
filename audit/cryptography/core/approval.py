"""
Production approval records and the governance lifecycle.

Lifecycle (each status is a distinct, recorded state; none implies the next)::

    DRAFT -> TECHNICAL_REVIEW -> APPROVAL_PENDING -> APPROVED_FOR_PRODUCTION
                                                         |
                                                         +-> INVALIDATED (terminal)

Production execution, verification and certification are recorded by the experiment itself,
not in the approval record.

An approval record is a JSON document::

    {
      "schema": "core-approval-record-1",
      "experiment_id": "...", "target": {...}, "design_version": "...", "design_sha256": "<64 hex>",
      "status": "APPROVED_FOR_PRODUCTION",
      "approved_by": "<name or identifier of the accountable person>",
      "approved_role": "<role>",
      "approved_utc": "<ISO 8601 with timezone>",
      "decisions_resolved": {"<decision key>": <value>, ...},
      "source_sha256": {"<repo-relative path>": "<64 hex>", ...},
      "signature": {"type": "UNSIGNED_LOCAL_RECORD"} | {"type": "<scheme>", "value": "..."},
      "invalidated": null | {"utc": "...", "by": "...", "reason": "..."}
    }

Assurance is reported, never inflated. An ``UNSIGNED_LOCAL_RECORD`` attests only that the named
person's approval was recorded in this repository; it is not a cryptographic signature and does
not establish independent review. This module cannot verify signatures; a record that carries one
is reported as ``SIGNATURE_PRESENT_NOT_VERIFIED``.

Nothing in this module creates or fills in an approval. ``template`` returns a record whose
status is APPROVAL_PENDING and whose identity fields are empty.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "core-approval-record-1"
STATUSES = ("DRAFT", "TECHNICAL_REVIEW", "APPROVAL_PENDING", "APPROVED_FOR_PRODUCTION", "INVALIDATED")
UNSIGNED = "UNSIGNED_LOCAL_RECORD"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ApprovalCheck:
    authorised: bool
    problems: tuple
    assurance: str
    record_status: str | None

    def to_dict(self) -> dict:
        return {"authorised": self.authorised, "problems": list(self.problems),
                "assurance": self.assurance, "record_status": self.record_status}


def _parse_utc(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def check(record: dict | None, *, experiment_id: str, target: dict, design_sha256: str,
          required_decisions: dict, source_sha256: dict | None = None,
          now: datetime | None = None) -> ApprovalCheck:
    """Validate an approval record against the exact experiment about to run.

    ``required_decisions`` maps every decision that must be resolved to the value the frozen
    design uses; the record must resolve each one to exactly that value.
    ``source_sha256`` maps source files to their current hashes; every one must be bound by the
    record with the same hash, so any later source change invalidates the approval.
    """
    if record is None:
        return ApprovalCheck(False, ("no approval record",), "NONE", None)
    p = []
    status = record.get("status")
    if record.get("schema") != SCHEMA:
        p.append(f"schema {record.get('schema')!r} is not {SCHEMA!r}")
    if status not in STATUSES:
        p.append(f"unknown status {status!r}")
    elif status != "APPROVED_FOR_PRODUCTION":
        p.append(f"status is {status}, not APPROVED_FOR_PRODUCTION")
    if record.get("invalidated"):
        p.append(f"approval was invalidated: {record['invalidated']}")
    if record.get("experiment_id") != experiment_id:
        p.append(f"experiment_id {record.get('experiment_id')!r} != {experiment_id!r}")
    if record.get("target") != target:
        p.append("target does not match the experiment about to run")
    dh = record.get("design_sha256")
    if not (isinstance(dh, str) and HEX64.match(dh)):
        p.append("design_sha256 is not a 64-hex digest")
    elif dh != design_sha256:
        p.append(f"design_sha256 {dh[:16]}... does not match the current design {design_sha256[:16]}...")
    who = record.get("approved_by")
    if not (isinstance(who, str) and who.strip()):
        p.append("approved_by is empty")
    if not (isinstance(record.get("approved_role"), str) and record["approved_role"].strip()):
        p.append("approved_role is empty")
    t = _parse_utc(record.get("approved_utc"))
    if t is None:
        p.append("approved_utc is missing or not an ISO 8601 timestamp with a timezone")
    elif t > (now or datetime.now(timezone.utc)):
        p.append("approved_utc is in the future")
    resolved = record.get("decisions_resolved") or {}
    for key, value in required_decisions.items():
        if key not in resolved:
            p.append(f"decision {key} is unresolved")
        elif json.dumps(resolved[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            p.append(f"decision {key} resolved to {resolved[key]!r}, the frozen design uses {value!r}")
    if source_sha256 is not None:
        bound = record.get("source_sha256") or {}
        for path, h in source_sha256.items():
            if bound.get(path) != h:
                p.append(f"source {path} differs from the hash bound by the approval")
    sig = record.get("signature") or {}
    if sig.get("type") == UNSIGNED:
        assurance = ("UNSIGNED_LOCAL_RECORD: the named person's approval is recorded in this repository; "
                     "not a cryptographic signature; not evidence of independent review")
    elif sig.get("type"):
        assurance = f"SIGNATURE_PRESENT_NOT_VERIFIED ({sig.get('type')}); this module verifies no signatures"
    else:
        p.append("signature block missing (use type UNSIGNED_LOCAL_RECORD if unsigned)")
        assurance = "NONE"
    return ApprovalCheck(not p, tuple(p), assurance, status)


def load(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def template(*, experiment_id: str, target: dict, design_version: str, design_sha256: str,
             required_decisions: dict, source_sha256: dict) -> dict:
    """A record for a person to complete. Status APPROVAL_PENDING; identity and time empty."""
    return {"schema": SCHEMA, "experiment_id": experiment_id, "target": target,
            "design_version": design_version, "design_sha256": design_sha256,
            "status": "APPROVAL_PENDING", "approved_by": "", "approved_role": "", "approved_utc": "",
            "decisions_resolved": dict(required_decisions), "source_sha256": dict(source_sha256),
            "signature": {"type": UNSIGNED}, "invalidated": None,
            "instructions": ("Review the design and the bound sources. To approve, set status to "
                             "APPROVED_FOR_PRODUCTION and fill approved_by, approved_role and "
                             "approved_utc yourself. Any change to the design or a bound source "
                             "invalidates this record automatically.")}
