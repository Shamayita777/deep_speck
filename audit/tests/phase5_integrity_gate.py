#!/usr/bin/env python3
"""
Phase-5 integrity gate.

Independent verification of the generated Phase-5 documents. This
script deliberately RE-DERIVES what it checks from the artifacts on
disk rather than importing the runner's own conclusions, so that a
failure in the runner cannot hide itself here.

Exits non-zero if any check fails or is inconclusive.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE = Path("audit/integration/evidence")
BUNDLE = Path("audit/evidence_bundle")
IMPL = Path("audit/implementation/evidence")

results: list[tuple[str, str, str]] = []   # (check, status, detail)


def record(check: str, ok: bool | None, detail: str) -> None:
    status = "PASS" if ok is True else ("FAIL" if ok is False else "INCONCLUSIVE")
    results.append((check, status, detail))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ledger = json.loads((BASE / "evidence_ledger.json").read_text())
index = json.loads((BASE / "artifact_index.json").read_text())
snapshot = json.loads((BASE / "audit_snapshot.json").read_text())
decision_doc = json.loads((BASE / "decision.json").read_text())


# --- CHECK 2: ledger entry <-> artifact <-> run reference ---
problems: list[str] = []
for entry in ledger["entries"]:
    eid = entry["evidence_id"]
    if entry["entry_kind"] == "NATIVE":
        if not entry.get("run_ids"):
            problems.append(f"{eid}: NATIVE entry has no run_ids")
        for run_id in entry.get("run_ids", []):
            candidates = list(IMPL.glob(f"run_manifest_{run_id}.json"))
            if not candidates:
                problems.append(f"{eid}: run_id {run_id!r} has no RunManifest file")
                continue
            manifest = json.loads(candidates[0].read_text())
            if manifest.get("run_id") != run_id:
                problems.append(f"{eid}: manifest run_id mismatch for {run_id!r}")
        refs = entry.get("observation", {}).get("run_manifest_references", [])
        for ref in refs:
            path = IMPL / ref["artifact_path"]
            if not path.exists():
                problems.append(f"{eid}: manifest reference {ref['artifact_path']} missing")
            elif sha256_file(path) != ref["content_hash"]:
                problems.append(f"{eid}: manifest reference hash mismatch")
    else:
        path = Path(entry["artifact_path"])
        if not path.exists():
            problems.append(f"{eid}: ingested artifact {path} missing")
record("2. ledger<->artifact<->run references", not problems,
       "all entries resolve" if not problems else "; ".join(problems))


# --- CHECK 3: recompute every artifact hash ---
hash_problems, verified = [], 0
for entry in ledger["entries"]:
    if entry["entry_kind"] == "INGESTED_REFERENCE":
        p = Path(entry["artifact_path"])
        if p.exists():
            if sha256_file(p) == entry["artifact_hash"]:
                verified += 1
            else:
                hash_problems.append(f"{entry['evidence_id']}: ingested hash mismatch")
        else:
            hash_problems.append(f"{entry['evidence_id']}: artifact missing")
    else:
        for name, digest in (entry.get("artifact_hashes") or {}).items():
            found = None
            # Archive-relative only: no machine-specific prefix, so the gate
            # runs identically from any extraction of the frozen archive.
            for base in (Path("."), IMPL, Path("audit/implementation"),
                         BUNDLE / "cryptography" / "ce1"):
                cand = base / name
                if cand.exists():
                    found = cand
                    break
            if found is None:
                hash_problems.append(f"{entry['evidence_id']}: artifact {name} not locatable")
                continue
            if sha256_file(found) == digest:
                verified += 1
            else:
                hash_problems.append(f"{entry['evidence_id']}: {name} hash mismatch")
# Also verify the vendored evidence bundle against its own manifest.
bundle_manifest = json.loads((BUNDLE / "BUNDLE_MANIFEST.json").read_text())
for rel, meta in bundle_manifest["files"].items():
    f = BUNDLE / rel
    if not f.exists():
        hash_problems.append(f"bundle file missing: {rel}")
    elif sha256_file(f) != meta["sha256"]:
        hash_problems.append(f"bundle file hash mismatch: {rel}")
    else:
        verified += 1

record("3. recompute every artifact hash", not hash_problems,
       f"{verified} artifact hashes recomputed and matched"
       if not hash_problems else "; ".join(hash_problems))


# --- CHECK 4: snapshot self-binding ---
binding_problems = []
component_files = {"evidence_ledger": "evidence_ledger.json",
                   "artifact_index": "artifact_index.json",
                   "decision": "decision.json",
                   "pillar_interpretation": "pillar_interpretation.json"}
for binding in snapshot["component_bindings"]:
    fname = component_files.get(binding["component"])
    if fname is None:
        binding_problems.append(f"unknown component {binding['component']}")
        continue
    actual = hashlib.sha256((BASE / fname).read_bytes()).hexdigest()
    if actual != binding["content_hash"]:
        binding_problems.append(
            f"{binding['component']}: bound {binding['content_hash'][:16]} "
            f"but file hashes {actual[:16]}")
bound = {b["component"] for b in snapshot["component_bindings"]}
missing_bindings = set(component_files) - bound
if missing_bindings:
    binding_problems.append(f"components not bound: {sorted(missing_bindings)}")
record("4. snapshot self-binding", not binding_problems,
       f"{len(bound)} components bound, all hashes match"
       if not binding_problems else "; ".join(binding_problems))


# --- CHECK 5: bidirectional traceability ---
trace_problems = []
ledger_ids = {e["evidence_id"] for e in ledger["entries"]}
snap_ids = set(snapshot["evidence_ledger_summary"]["evidence_ids"])
if ledger_ids != snap_ids:
    trace_problems.append(f"snapshot/ledger id mismatch: {ledger_ids ^ snap_ids}")

# forward: every indexed artifact's referrers must exist in the ledger
for path, rec in index["artifacts"].items():
    for ref in rec["referenced_by"]:
        if ref not in ledger_ids:
            trace_problems.append(f"index cites unknown evidence {ref}")
# backward: every ledger artifact must appear in the index
for entry in ledger["entries"]:
    names = ([entry["artifact_path"]] if entry["entry_kind"] == "INGESTED_REFERENCE"
             else list((entry.get("artifact_hashes") or {}).keys()))
    for name in names:
        if name not in index["artifacts"]:
            trace_problems.append(f"ledger artifact {name} absent from index")
# provenance backward trace on the native entry
for entry in ledger["entries"]:
    if entry["entry_kind"] != "NATIVE":
        continue
    prov = entry.get("provenance", {})
    if entry["evidence_id"] not in prov.get("entities", {}):
        trace_problems.append("native evidence entity absent from its own provenance graph")
    gen = [r for r in prov.get("relations", [])
           if r["relation_type"] == "generated" and r["subject_id"] == entry["evidence_id"]]
    if not gen:
        trace_problems.append("no generated relation for native evidence")
record("5. bidirectional traceability", not trace_problems,
       "ledger<->index<->snapshot<->provenance all consistent"
       if not trace_problems else "; ".join(trace_problems))


# --- CHECK 6: is the decision DERIVED from the ledger, or hard-coded? ---
sys.path.insert(0, ".")
from audit.integration.conflict_resolution import ALL_CONFLICTS          # noqa: E402
from audit.integration.decision_engine import decide_claim               # noqa: E402
from audit.integration.evidence_ledger import EvidenceLedger             # noqa: E402
from audit.integration.run_integration import derive_pillar_state        # noqa: E402
import inspect                                                           # noqa: E402

# 6a: re-derive the pillar state FROM THE PERSISTED LEDGER and reproduce
# the recorded decision.
reloaded = EvidenceLedger.load(BASE / "evidence_ledger.json")
derived_state, derived_interp = derive_pillar_state(reloaded)
rederived = decide_claim(claim_id="C1", conflicts=ALL_CONFLICTS, **derived_state)
recorded = decision_doc["decision"]
derivation_ok = (
    rederived.final_outcome.value == recorded["final_outcome"]
    and rederived.evidence_level.value == recorded["evidence_level"]
    and rederived.blocking_conflicts == recorded["blocking_conflicts"]
)
record("6a. decision re-derivable from persisted ledger", derivation_ok,
       "re-loading the ledger and re-deriving reproduces the recorded decision exactly"
       if derivation_ok else "re-derived decision differs from recorded decision")

# 6b: the production derivation must contain no hard-coded outcomes.
src = inspect.getsource(derive_pillar_state)
import audit.integration.run_integration as _runner
module_src = inspect.getsource(_runner)
forbidden = [t for t in ("DimensionOutcome.PASS", "DimensionOutcome.FAIL",
                         "DimensionOutcome.INCONCLUSIVE", "DimensionOutcome.CONDITIONAL_PASS",
                         "CryptographicOutcome.SUPPORTED", "CryptographicOutcome.NOT_SUPPORTED")
             if t in module_src]
reads_ledger = "ledger.entries" in src
record("6b. pillar state derived from ledger", (reads_ledger and not forbidden),
       "derive_pillar_state() reads ledger.entries and the module contains no hard-coded "
       "pillar outcome constants"
       if (reads_ledger and not forbidden)
       else f"reads_ledger={reads_ledger}, hard-coded constants present: {forbidden}")

# 6c: CAUSALITY - mutating a decision-bearing ledger entry must change the outcome.
import copy                                                              # noqa: E402
mutated = EvidenceLedger.load(BASE / "evidence_ledger.json")
changed = False
for entry in mutated.entries:
    if entry.get("source_pillar_id") == "D1":
        entry["recorded_decision"] = "FAIL"
        changed = True
        break
if not changed:
    record("6c. ledger is causally upstream of decision", None,
           "no D1 entry found to mutate; causality could not be exercised")
else:
    mutated_state, _ = derive_pillar_state(mutated)
    mutated_decision = decide_claim(claim_id="C1", conflicts=ALL_CONFLICTS, **mutated_state)
    causal = (mutated_state["dataset"].value == "FAIL"
              and mutated_decision.final_outcome.value == "NOT_SUPPORTED"
              and mutated_decision.final_outcome.value != recorded["final_outcome"])
    record("6c. ledger is causally upstream of decision", causal,
           "mutating D1 to FAIL changes dataset dimension to FAIL and the claim to "
           "NOT_SUPPORTED, proving the ledger drives the decision"
           if causal else
           f"mutation did not propagate: dataset={mutated_state['dataset'].value}, "
           f"claim={mutated_decision.final_outcome.value}")


# --- CHECK 7: historical evidence unmodified ---
HISTORICAL = {
    "implementation-certificate-V1.json":
        "76ce4b59327599473746d9420aa2e95d73b397042a349929c50f54a2b5baa6f7",
    "implementation-certificate-V2.json":
        "cf55bd4ffc77f4509792c31daee7846518d8f125d8dada98d5c97dd6f5e72bc6",
}
hist_problems = []
for name, expected in HISTORICAL.items():
    actual = sha256_file(IMPL / name)
    if actual != expected:
        hist_problems.append(f"{name} MODIFIED ({actual[:16]} != {expected[:16]})")
for entry in ledger["entries"]:
    if entry["entry_kind"] != "INGESTED_REFERENCE":
        continue
    p = Path(entry["artifact_path"])
    if p.exists() and sha256_file(p) != entry["artifact_hash"]:
        hist_problems.append(f"historical source {p.name} MODIFIED")
record("7. historical evidence unmodified", not hist_problems,
       f"{len(HISTORICAL)} II certificates + 7 D/CE source artifacts unchanged"
       if not hist_problems else "; ".join(hist_problems))


print("=" * 78)
print("PHASE-5 INTEGRITY GATE")
print("=" * 78)
failed = 0
for check, status, detail in results:
    print(f"[{status:12}] {check}")
    print(f"               {detail}")
    if status != "PASS":
        failed += 1
print("=" * 78)
print(f"RESULT: {len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
