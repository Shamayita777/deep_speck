"""Mechanically derive the final test and tamper matrices from the actual run.

    python -m audit.cryptography.ce234.build_matrices <junit.xml>

Counts are computed from the artifacts, never typed by hand.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX_DIR = HERE / "matrices"
CE_OF = {"ce2": "CE2", "ce3": "CE3", "ce4": "CE4"}


def classify(name: str) -> str:
    n = name.lower()
    for key, cls in (("verifier_rejects", "tamper"), ("tamper", "tamper"),
                     ("smoke", "smoke/verifier"), ("verif", "smoke/verifier"),
                     ("preflight", "fail-closed"), ("refus", "fail-closed"),
                     ("fails_closed", "fail-closed"), ("binding", "model-binding"),
                     ("checkpoint", "model-binding"), ("control_validity", "intervention-invariant"),
                     ("intervention", "intervention-invariant"), ("eligibility", "intervention-invariant"),
                     ("magnitude", "intervention-invariant"), ("holm", "statistical"),
                     ("sign_test", "statistical"), ("effect", "statistical"),
                     ("unit", "statistical"), ("plan", "provenance"),
                     ("hash", "provenance"), ("unspecified", "provenance"),
                     ("frozen", "provenance"), ("generator", "determinism"),
                     ("deterministic", "determinism"), ("nondeterministic", "determinism")):
        if key in n:
            return cls
    return "unit"


def ce_of(name: str) -> str:
    n = name.lower()
    for k, v in CE_OF.items():
        if k in n:
            return v
    return "shared"


def build(junit_path) -> dict:
    root = ET.parse(junit_path).getroot()
    rows = []
    for tc in root.iter("testcase"):
        name = tc.get("name")
        status = "PASSED"
        detail = ""
        for child in tc:
            if child.tag == "failure" or child.tag == "error":
                status, detail = "FAILED", (child.get("message") or "")[:200]
            elif child.tag == "skipped":
                status, detail = "SKIPPED", (child.get("message") or "")[:200]
        rows.append({
            "test_id": name, "description": name.replace("_", " "),
            "ce": ce_of(name), "test_class": classify(name),
            "expected": "PASS", "actual": status, "status": status,
            "deterministic": True, "replayable": True,
            "evidence_path": "audit/cryptography/ce234/matrices/junit.xml",
            "detail": detail})
    counts = Counter(r["status"] for r in rows)
    return {"schema": "ce234-test-matrix-1", "n_tests": len(rows),
            "counts": dict(counts), "tests": rows,
            "determinism_note": ("every validation test draws from an explicit RNG or a "
                                 "fixed seed corpus; production keeps the frozen "
                                 "os.urandom convention")}


def tamper_matrix() -> dict:
    log = MATRIX_DIR / "tamper_log.jsonl"
    rows = []
    seen = set()
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["mutation_id"] in seen:
                    continue
                seen.add(r["mutation_id"])
                rows.append(r)
    rows.sort(key=lambda r: r["mutation_id"])
    counts = Counter(r["status"] for r in rows)
    return {"schema": "ce234-tamper-matrix-1", "n_mutations": len(rows),
            "n_rejected": counts.get("REJECTED", 0),
            "n_not_rejected": counts.get("NOT_REJECTED", 0),
            "counts": dict(counts), "mutations": rows}


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    tm = tamper_matrix()
    (MATRIX_DIR / "TAMPER_MATRIX.json").write_text(json.dumps(tm, indent=2))
    out = {"tamper": {"n": tm["n_mutations"], "rejected": tm["n_rejected"]}}
    if argv:
        test = build(argv[0])
        (MATRIX_DIR / "TEST_MATRIX.json").write_text(json.dumps(test, indent=2))
        out["tests"] = {"n": test["n_tests"], **test["counts"]}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
