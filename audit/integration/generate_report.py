#!/usr/bin/env python3
"""
Phase 7 - human-readable audit report generation.

Generates the final report from the MACHINE-READABLE documents only.
It introduces no facts of its own: every statement it emits is read
from the frozen snapshot, ledger, artifact index, interpretation
record, or conflict record, and every conclusion carries the evidence
ID that supports it (methodology Section 3.10.7 - "the reasoning
leading from empirical observations to the final decision shall be
fully traceable within the audit report").

If a required document is missing the generator FAILS rather than
emitting a report with gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from audit.common.provenance import utc_timestamp

REPORT_VERSION = "audit-report-v1"


class ReportGenerationError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ReportGenerationError(
            f"Required machine-readable document missing: {path}. Refusing to emit a report "
            "with unsupported gaps."
        )
    return json.loads(path.read_text())


def _bullet(lines: list[str]) -> str:
    return "\n".join(f"- {l}" for l in lines) if lines else "- (none recorded)"


def generate(evidence_dir: Path, frozen_dir: Path, bundle_dir: Path) -> str:
    freeze = _load(frozen_dir / "PHASE5_FREEZE_MANIFEST.json")
    snapshot = _load(evidence_dir / freeze["authoritative_snapshot"])
    ledger = _load(evidence_dir / "evidence_ledger.json")
    index = _load(evidence_dir / "artifact_index.json")
    interpretation = _load(evidence_dir / "pillar_interpretation.json")
    decision_doc = _load(evidence_dir / "decision.json")
    bundle = _load(bundle_dir / "BUNDLE_MANIFEST.json")

    decision = snapshot["decision"]
    out: list[str] = []
    w = out.append

    w(f"# CipherMind Audit Report ({REPORT_VERSION})")
    w("")
    w(f"Generated: {utc_timestamp()}")
    w(f"Authoritative snapshot: `{freeze['authoritative_snapshot']}` "
      f"(freeze version {freeze['freeze_version']})")
    w("")
    w("> This report is generated mechanically from the machine-readable evidence "
      "documents. It asserts nothing that is not recorded in them, and every "
      "conclusion below cites the evidence IDs that produced it.")
    w("")

    # 1. Claim and scope
    w("## 1. Claim Identification and Scope")
    w("")
    w(f"**Claim {snapshot['claim_id']}**: {snapshot['claim_text']}")
    w("")
    for name, note in snapshot.get("scope_notes", {}).items():
        w(f"**Scope exclusion - {name}**: {note}")
    w("")

    # 2. Final decision
    w("## 2. Final Claim-Level Decision")
    w("")
    w(f"| Field | Value |")
    w(f"|---|---|")
    w(f"| Final outcome | **{decision['final_outcome']}** |")
    w(f"| Evidence level | **{decision['evidence_level']}** |")
    w(f"| Decision policy | `{decision['policy_version']}` |")
    w(f"| Blocking conflicts | {decision['blocking_conflicts'] or 'none'} |")
    w("")
    w(f"**Rationale (from the decision record):** {decision['rationale']}")
    w("")

    # 3. Dimension findings with traceability
    w("## 3. Dimension Findings")
    w("")
    w("| Dimension | Outcome | Rule | Rule provenance | Driving evidence |")
    w("|---|---|---|---|---|")
    for dim, interp in sorted(interpretation["interpretations"].items()):
        drivers = ", ".join(f"`{d}`" for d in interp["driving_evidence_ids"]) or "-"
        w(f"| {dim} | **{interp['outcome']}** | `{interp['rule_id']}` | "
          f"{interp['rule_provenance']} | {drivers} |")
    w("")
    w("### Rule provenance")
    w("")
    w("- **SOURCE_DEFINED** - stated by the governing methodology.")
    w("- **INTEGRATION_OPERATIONALIZATION** - introduced by the integration layer because "
      "the methodology does not specify within-dimension aggregation. These are choices of "
      "this layer and are NOT presented as methodology findings.")
    w("")

    # 4. Per-evidence interpretation trail
    w("## 4. Evidence Interpretation Trail")
    w("")
    w("| Evidence ID | Pillar | Native status | Mappability | Normalized |")
    w("|---|---|---|---|---|")
    for dim, interp in sorted(interpretation["interpretations"].items()):
        for rec in interp["records"]:
            w(f"| `{rec['source_evidence_id']}` | {rec['source_pillar_id']} | "
              f"`{rec['source_native_status']}` | {rec['mappability']} | "
              f"{rec['normalized_outcome'] or '-'} |")
    w("")
    w("Native statuses shown as `-` under *Normalized* were deliberately NOT coerced into "
      "the standardized vocabulary; see each record's rationale in "
      "`pillar_interpretation.json`.")
    w("")

    # 5. Conflicts
    w("## 5. Conflict Resolution")
    w("")
    conflicts = snapshot.get("conflicts", [])
    if not conflicts:
        w("No conflicts recorded.")
    for conflict in conflicts:
        w(f"### `{conflict['conflict_id']}` - status: **{conflict['status']}** "
          f"(blocks claim: {conflict['blocks_claim']})")
        w("")
        w(f"**Shared subject:** {conflict['shared_subject']}")
        w("")
        w("| Source | Decision | Observation |")
        w("|---|---|---|")
        for obs in conflict["observations"]:
            w(f"| {obs['source_id']} | {obs['decision']} | {obs['observation']} |")
        w("")
        w(f"**Resolution rationale:** {conflict['rationale']}")
        w("")
        if conflict.get("required_experimentation"):
            w("**Additional experimentation required:**")
            w(_bullet(conflict["required_experimentation"]))
            w("")

    # 6. Evidence ledger summary
    w("## 6. Evidence Ledger")
    w("")
    counts = ledger["entry_counts"]
    w(f"- Native entries: **{counts['native']}** (full provenance, resolvable run manifests)")
    w(f"- Ingested references: **{counts['ingested_reference']}** (historical artifacts cited "
      f"by content hash)")
    w(f"- Total: **{counts['total']}**")
    w("")
    w("| Evidence ID | Kind | Pillar/Dim | Recorded decision |")
    w("|---|---|---|---|")
    for e in ledger["entries"]:
        pillar = e.get("source_pillar_id", e.get("experiment_id", "-"))
        dec = e.get("recorded_decision", e.get("decision", "-"))
        w(f"| `{e['evidence_id']}` | {e['entry_kind']} | {pillar} | {dec} |")
    w("")

    # 7. Artifact index
    w("## 7. Artifact Index")
    w("")
    w(f"- Artifacts tracked: **{index['artifact_count']}**")
    w(f"- Verification: {index['verification_summary']}")
    w(f"- Hash conflicts: {index['hash_conflicts'] or 'none'}")
    w(f"- Evidence bundle: `{bundle['bundle_id']}` ({bundle['file_count']} vendored files, "
      f"byte-identical to their original sources)")
    w("")

    # 8. Reproducibility
    w("## 8. Reproducibility and Integrity")
    w("")
    for check, value in snapshot.get("integrity_checks", {}).items():
        w(f"- **{check}**: {value}")
    w("")
    w("Snapshot component bindings (each document bound to the snapshot by content hash):")
    w("")
    for b in snapshot.get("component_bindings", []):
        w(f"- `{b['component']}` -> `{b['artifact_path']}` ({b['content_hash'][:16]}...)")
    w("")

    # 9. Limitations and blockers
    w("## 9. Limitations")
    w("")
    w(_bullet(snapshot.get("limitations", [])))
    w("")
    w("## 10. Open Blockers")
    w("")
    w(_bullet(snapshot.get("open_blockers", [])))
    w("")

    # 11. Scope statement
    w("## 11. Evidential Scope Statement")
    w("")
    w("Per methodology Section 3.12, this audit evaluates the **evidential support** for the "
      "claim above. It does not establish mathematical security properties, does not prove "
      "the underlying cryptographic hypothesis true or false, and does not certify software "
      "correctness. An INCONCLUSIVE outcome means the available evidence is insufficient to "
      "decide - not that the claim is false.")
    w("")
    w(f"CE mappings, per-experiment hypotheses, controls and limitations are recorded in "
      f"`decision.json` ({len(decision_doc.get('ce_mappings', []))} CE experiments mapped).")
    w("")

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=Path("audit/integration/evidence"))
    parser.add_argument("--frozen-dir", type=Path, default=Path("audit/integration/frozen"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("audit/evidence_bundle"))
    parser.add_argument("--output", type=Path, default=Path("audit/integration/AUDIT_REPORT.md"))
    args = parser.parse_args()

    try:
        report = generate(args.evidence_dir, args.frozen_dir, args.bundle_dir)
    except ReportGenerationError as exc:
        print(f"REPORT GENERATION FAILED: {exc}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written: {args.output} ({len(report.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
