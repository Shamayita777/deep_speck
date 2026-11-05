"""Re-evaluate the D2.1 records in existing D2 certificates (schema 7.0).

READ-ONLY.  This tool never writes to, regenerates or replaces a certificate,
and never reruns an audit.  It reads recorded counts and recomputes the
reference quantity in closed form.

WHY A RE-EVALUATION IS POSSIBLE WITHOUT RERUNNING.  The superseded D2.1
method's defect was in its REFERENCE quantity, not in its observed count:
hit counts came from exact Hamming verification of retrieved candidates, so a
recorded hit is a real verified neighbour.  The corrected denominator is a
closed-form function of (rows, bits, radius) alone.

WHY THE RE-EVALUATION IS ONE-SIDED.  Retrieval was not exhaustive, so a
recorded count is a LOWER bound on the exhaustive count.  The recomputed
ratio is therefore a lower bound:

  * both gates crossing on the lower bound  ->  EXCESS_DETECTED is sound
  * otherwise                               ->  INCONCLUSIVE_LOWER_BOUND

Absence can never be concluded from a legacy record.  Establishing absence
requires rerunning that partition with ``near_neighbor_exact_screen``, and
this tool prints the query budget that rerun would need.

Usage:
    python -m audit.dataset.tools.reevaluate_d21 CERT.json [CERT.json ...]
    python -m audit.dataset.tools.reevaluate_d21 --json out.json CERT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from audit.dataset.d2_sample_dependence import (
    D21_SUPERSEDED_METHOD_DEFECTS,
    declared_pair_probability,
    minimum_detectable_near_duplicate_ratio,
    near_duplicate_regime,
    reevaluate_legacy_d21_record,
    required_queries_for_power,
)

LEGACY_HITS = "rows_with_verified_neighbor_within_radius"
EXACT_HITS = "rows_with_neighbor_within_radius"


def _find_partition_sizes(blob: Any) -> dict[str, dict[str, int]]:
    """Locate the provenance partition block wherever it sits.

    Certificate layout has changed across schema versions, so the block is
    found by shape rather than by a hardcoded path.  Returning an empty dict
    is a legitimate outcome and is reported as UNVERIFIED rather than guessed.
    """
    found: dict[str, dict[str, int]] = {}

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            if key == "partitions" and node and all(
                isinstance(v, dict) and "sample_count" in v
                for v in node.values()
            ):
                for name, v in node.items():
                    found[name] = {
                        "rows": int(v["sample_count"]),
                        "bits": int(v.get("feature_count", 0)),
                        "sha256": v.get("sha256"),
                    }
                return
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk(blob)
    return found


def _find_d21(blob: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "d2_1_near_duplicate_structure" in node:
                out.update(node["d2_1_near_duplicate_structure"])
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blob)
    return out


def _find_config(blob: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "familywise_test_count" in node and "schema_version" in node:
                out.update(node)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blob)
    return out


def _partition_of(component: str) -> str:
    # components are keyed "within:<partition>"
    return component.split(":", 1)[1] if ":" in component else component


def reevaluate_certificate(path: Path) -> dict[str, Any]:
    blob = json.loads(path.read_text())
    sizes = _find_partition_sizes(blob)
    d21 = _find_d21(blob)
    config = _find_config(blob)
    schema = str(config.get("schema_version", "UNKNOWN"))
    alpha = float(config.get("familywise_alpha", 0.01))

    report: dict[str, Any] = {
        "certificate": str(path),
        "schema_version": schema,
        "familywise_alpha": alpha,
        "partition_sizes_found": bool(sizes),
        "components": {},
        "notes": [],
    }
    if not d21:
        report["notes"].append(
            "no d2_1_near_duplicate_structure block found; nothing to "
            "re-evaluate (marked UNVERIFIED)")
        return report
    if not sizes:
        report["notes"].append(
            "partition row counts not found in this certificate; the "
            "corrected reference quantity needs (rows, bits) and cannot be "
            "computed, so every component below is UNVERIFIED")

    for component, record in d21.items():
        part = _partition_of(component)
        size = sizes.get(part, {})
        rows, bits = size.get("rows"), size.get("bits")

        # Schema 7.0 records are already exact; nothing to re-evaluate.
        if isinstance(record, dict) and "by_radius" in record:
            report["components"][component] = {
                "record_method": "D2.1-exact (schema >= 7.0)",
                "action": "NO_REEVALUATION_NEEDED",
                "decisions": {r: v.get("decision")
                              for r, v in record["by_radius"].items()},
            }
            continue

        radii = sorted(int(k) for k in record if str(k).lstrip("-").isdigit())
        entry: dict[str, Any] = {
            "record_method": "superseded (schema <= 6.5)",
            "defects": list(D21_SUPERSEDED_METHOD_DEFECTS),
            "rows": rows,
            "bits": bits,
            "by_radius": {},
        }
        if rows is None or not bits:
            entry["action"] = "UNVERIFIED"
            entry["reason"] = (
                f"row/bit counts for partition '{part}' not found in the "
                "certificate")
            report["components"][component] = entry
            continue

        for r in radii:
            legacy = record[str(r)] if str(r) in record else record[r]
            # Normalise the count field name across method versions.
            norm = dict(legacy)
            if LEGACY_HITS not in norm and EXACT_HITS in norm:
                norm[LEGACY_HITS] = norm[EXACT_HITS]
            out = reevaluate_legacy_d21_record(
                norm, rows=rows, bits=bits, radius=r,
                alpha=alpha, radius_family_size=max(1, len(radii)))
            # What a rerun would cost, so absence can actually be established.
            p = declared_pair_probability(bits, r)
            q = -math.expm1((rows - 1) * math.log1p(-p))
            if 0.0 < q < 1.0:
                out["rerun_guidance"] = {
                    "regime": near_duplicate_regime(
                        queries=int(norm.get("queried_rows") or 1),
                        null_query_probability=q),
                    "budget_for_80pc_power_at_1.25x": required_queries_for_power(
                        1.25, null_query_probability=q,
                        alpha=alpha / max(1, len(radii))).get("required_queries"),
                }
            entry["by_radius"][str(r)] = out
        entry["action"] = "REEVALUATED_ONE_SIDED"
        report["components"][component] = entry

    return report


def _print(report: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"certificate     {report['certificate']}")
    print(f"schema recorded {report['schema_version']}")
    print(f"familywise a    {report['familywise_alpha']}")
    for note in report["notes"]:
        print(f"NOTE: {note}")
    for component, entry in sorted(report["components"].items()):
        print(f"\n--- {component}  [{entry.get('action')}] ---")
        if entry.get("action") == "NO_REEVALUATION_NEEDED":
            print("    already produced by the exact method; decisions:")
            for r, d in sorted(entry["decisions"].items(), key=lambda kv: int(kv[0])):
                print(f"      radius {r:>3}: {d}")
            continue
        if entry.get("action") == "UNVERIFIED":
            print(f"    UNVERIFIED: {entry.get('reason')}")
            continue
        print(f"    rows={entry['rows']:,} bits={entry['bits']}")
        print(f"    {'r':>3} {'hits':>7} {'queries':>9} {'E[hits]':>10} "
              f"{'ratio LB':>12} {'status':>26}")
        unstable = False
        for r, out in sorted(entry["by_radius"].items(), key=lambda kv: int(kv[0])):
            rc = out.get("recomputed")
            if rc is None:
                print(f"    {r:>3} {'-':>7} {'-':>9} {'-':>10} {'-':>12} "
                      f"{out.get('status'):>26}")
                continue
            detection = (out.get("rerun_guidance") or {}).get("regime") == "DETECTION"
            unstable = unstable or detection
            ratio = f"{rc['excess_ratio_lower_bound']:.4g}" + ("*" if detection else "")
            print(f"    {r:>3} {rc['observed_hits_recorded']:>7,} "
                  f"{rc['queried_rows']:>9,} {rc['expected_query_hits']:>10.3f} "
                  f"{ratio:>12} {out['status']:>26}")
        if unstable:
            print("    * DETECTION regime: expected null hits < 1, so the ratio")
            print("      is dominated by whether a single occurrence was seen and")
            print("      is NOT a stable proportional excess. Read hits against")
            print("      E[hits]. A large ratio here means 'one was found where")
            print("      almost none was expected', not a manyfold excess.")
        for r, out in sorted(entry["by_radius"].items(), key=lambda kv: int(kv[0])):
            g = out.get("rerun_guidance")
            if not g:
                continue
            budget = g["budget_for_80pc_power_at_1.25x"]
            shown = (f"{budget:,}" if isinstance(budget, int)
                     else "not estimable at this scale -- read as a DETECTION test")
            print(f"      radius {r}: regime={g['regime']}, budget for 80% "
                  f"power at 1.25x = {shown}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("certificates", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full report as JSON")
    a = ap.parse_args(argv)

    reports = []
    for path in a.certificates:
        if not path.is_file():
            print(f"UNVERIFIED: {path} is not a readable file", file=sys.stderr)
            reports.append({"certificate": str(path), "schema_version": "UNKNOWN",
                            "familywise_alpha": 0.01, "partition_sizes_found": False,
                            "components": {}, "notes": ["file not readable"]})
            continue
        rep = reevaluate_certificate(path)
        _print(rep)
        reports.append(rep)

    print("=" * 78)
    print("SUMMARY")
    any_excess = False
    any_unverified = False
    for rep in reports:
        for component, entry in sorted(rep["components"].items()):
            if entry.get("action") == "UNVERIFIED" or rep["notes"]:
                any_unverified = True
            for r, out in (entry.get("by_radius") or {}).items():
                if out.get("status") == "EXCESS_DETECTED":
                    any_excess = True
                    print(f"  EXCESS CONFIRMED (one-sided): "
                          f"{Path(rep['certificate']).name} {component} radius {r}")
    if not any_excess:
        print("  no legacy D2.1 record supports a confirmed excess.")
    print("  No legacy record can establish ABSENCE of near-duplicate")
    print("  structure: recorded counts are lower bounds. Rerun with")
    print("  near_neighbor_exact_screen to establish absence.")
    if any_unverified:
        print("  Some components are UNVERIFIED; see the notes above.")

    if a.json:
        a.json.write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
