"""
Execute the REGISTERED SDB-V2 benchmark. Refuses unless:
  * registration_sdb_v2.json and manifest_sdb_v2.json exist and the registration names the manifest hash;
  * every file the manifest binds (sources, evidence, fixtures) has its recorded hash on disk;
  * the registration and manifest are byte-identical to the copies in the anchor commit, and that
    commit is contained in origin/audit-framework (registration anchored before execution).

  python -B audit/framework_validation/sdb_v2/run_v2.py --anchor <commit>

Writes results/<run id>/result.json and raw/ (every observation the verifier needs). Never overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REG = HERE / "registration_sdb_v2.json"
MAN = HERE / "manifest_sdb_v2.json"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def preconditions(anchor: str) -> tuple[list[str], dict]:
    P = []
    if not REG.is_file() or not MAN.is_file():
        return ["registration or manifest absent"], {}
    reg, man = json.loads(REG.read_text()), json.loads(MAN.read_text())
    if reg.get("manifest_sha256") != sha(MAN):
        P.append("registration does not name this manifest")
    for section in ("sources", "evidence", "fixtures"):
        for path, h in man[section].items():
            p = REPO / path
            if not p.is_file() or sha(p) != h:
                P.append(f"{section}: {path} differs from the manifest")
    anchor_info = {"commit": anchor}
    try:
        subprocess.run(["git", "-C", str(REPO), "fetch", "origin", "audit-framework"], capture_output=True, check=True)
        contains = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", anchor,
                                   "origin/audit-framework"]).returncode == 0
        anchor_info["in_origin_audit_framework"] = contains
        anchor_info["committed_at"] = git("show", "-s", "--format=%cI", anchor)
        for f in (REG, MAN):
            rel = str(f.relative_to(REPO))
            blob = subprocess.run(["git", "-C", str(REPO), "show", f"{anchor}:{rel}"], capture_output=True).stdout
            same = hashlib.sha256(blob).hexdigest() == sha(f)
            anchor_info[f"{f.name}_matches_anchor"] = same
            if not same:
                P.append(f"{rel} differs from the anchored copy")
        if not contains:
            P.append("anchor commit is not in origin/audit-framework")
    except subprocess.CalledProcessError as exc:
        P.append(f"anchor check failed: {exc}")
    return P, anchor_info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    a = ap.parse_args(argv)
    problems, anchor = preconditions(a.anchor)
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems[:20]))
        return 1
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / run_id
    (out / "raw").mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(REPO))
    from audit.framework_validation.sdb_v2 import baselines_v2 as B
    from audit.framework_validation.sdb_v2 import cases_v2 as C
    reg = json.loads(REG.read_text())
    expected = reg["expected_outcomes"]
    started = datetime.now(timezone.utc).isoformat()
    cases = {}
    for case in C.ORDER:
        t0 = time.time()
        try:
            obs, raw = C.ADAPTERS[case](case)
            err = None
        except Exception as exc:                                        # noqa: BLE001
            obs, raw, err = {}, {}, f"{type(exc).__name__}: {exc}"
        match = {lvl: C.matches(expected[case].get(lvl), obs.get(lvl)) for lvl in expected[case]}
        cases[case] = {"group": C.GROUPS[case], "levels": C.LEVELS[case], "expected": expected[case],
                       "observed": obs, "match": match, "all_levels_match": bool(match) and all(match.values()),
                       "error": err, "seconds": time.time() - t0}
        (out / "raw" / f"{case}.json").write_text(json.dumps(raw, indent=1, sort_keys=True, default=str))
        print(f"{case:3} {'MATCH' if cases[case]['all_levels_match'] else 'MISMATCH'} {match} {err or ''}")
    b1, b1_status = {}, {}
    for case in C.ORDER:
        b1_status[case] = B.b1_status(case)
    for case in B.B1_CASES:
        res, raw = B.b1(case)
        b1[case] = res
        np.savez(out / "raw" / f"B1_{case}.npz", correct_packbits=raw.pop("correct_packbits"))
        (out / "raw" / f"B1_{case}.json").write_text(json.dumps(raw, indent=1, sort_keys=True))
    b1s, b1s_raw = B.b1s()
    b1s_match = C.matches(expected["_H2_B1S"], b1s)
    (out / "raw" / "B1S.json").write_text(json.dumps(b1s_raw, indent=1, sort_keys=True))
    b2 = {case: B.b2(case) for case in C.ORDER}
    man = json.loads(MAN.read_text())
    bound = set(man["sources"]) | set(man["evidence"]) | set(man["fixtures"])
    loaded = sorted({str(Path(m.__file__).resolve().relative_to(REPO)) for m in list(sys.modules.values())
                     if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(REPO)})
    result = {
        "artifact": "sdb-v2-result-1", "run_id": run_id, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha(REG), "manifest_sha256": sha(MAN), "anchor": anchor,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__},
        "modules_loaded": loaded, "modules_loaded_not_in_manifest": [m for m in loaded if m not in bound],
        "cases": cases,
        "baselines": {"B1": b1, "B1_status": b1_status, "B1S": {"observed": b1s, "matches_certificate": b1s_match},
                      "B2": b2,
                      "B1_naive": "illustrative policy only; not executed or scored"},
        "metrics": metrics(cases, b2, b1_status, b1s_match),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps(result["metrics"], indent=1))
    print(f"result {out / 'result.json'} {sha(out / 'result.json')}")
    return 0


def metrics(cases: dict, b2: dict, b1_status: dict, b1s_match: bool) -> dict:
    """Separate counts per method; nothing is pooled across groups."""
    grp = lambda g: [c for c, v in cases.items() if v["group"] in g]
    det = grp(("historical_defect", "synthetic_defect"))
    clean = grp(("clean_control",))
    abst = grp(("abstention_control",))
    c_match = lambda c: cases[c]["match"].get("C", None)
    return {
        "framework": {
            "detection": {"cases": det, "detected": [c for c in det if c_match(c)], "n": len(det)},
            "false_alarms": {"cases": clean, "false_alarms": [c for c in clean if not c_match(c)], "n": len(clean)},
            "positive_control": {"P1": cases["P1"]["all_levels_match"]},
            "abstention": {"cases": abst, "correct": [c for c in abst if cases[c]["all_levels_match"]], "n": len(abst)},
            "d_and_r_level_matches": {c: {k: v for k, v in cases[c]["match"].items() if k in ("D", "R")}
                                      for c in cases if any(k in ("D", "R") for k in cases[c]["match"])},
            "h2_replay": {k: cases["H2"]["match"].get(k) for k in ("replay", "D", "R")},
            "errors": [c for c, v in cases.items() if v["error"]],
        },
        "B2": {
            "detection": {"flagged": [c for c in det if b2[c]["output"] == "FLAGGED"], "n": len(det)},
            "false_alarms": {"flagged": [c for c in clean if b2[c]["output"] == "FLAGGED"], "n": len(clean)},
            "abstention_cases_output": {c: b2[c]["output"] for c in abst},
            "coverage_items": {c: {s: sum(1 for i in b2[c]["items"].values() if i["result"] == s)
                                   for s in ("PASS", "FAIL", "NOT_ASSESSED", "NOT_APPLICABLE")} for c in cases},
            "outputs": {c: b2[c]["output"] for c in cases},
        },
        "B1": {"status_counts": {s: sum(1 for v in b1_status.values() if v == s)
                                 for s in ("EXECUTED", "B1-S", "NOT_ASSESSED", "NOT_APPLICABLE")},
               "B1S_matches_certificate": b1s_match},
    }


if __name__ == "__main__":
    sys.exit(main())
