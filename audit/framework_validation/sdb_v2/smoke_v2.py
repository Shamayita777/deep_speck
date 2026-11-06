"""
Pre-registration plumbing smoke. NOT the benchmark and never evidence.

Builds a separate fixture root with different seeds (outside the repository) and a synthetic H2
bundle with the real bundle's layout, then runs every adapter and B2 there, reporting only whether
each ran and returned every registered key. Values are not printed or compared, so no registered
result is observed before registration. The checkpoint cases (H1, S2, K3) and B1 are excluded: their
inputs are the registered checkpoints themselves.

  python -B audit/framework_validation/sdb_v2/smoke_v2.py <scratch-dir>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SKIP = {"H1", "S2", "K3"}


def build(root: Path) -> None:
    sys.path.insert(0, str(REPO))
    import numpy as np
    from audit.framework_validation.sdb_v2 import fixtures_v2 as F
    root.mkdir(parents=True)
    F.build(root / "fixtures", seed_offset=1000)
    h2 = root / "h2"
    cert = json.loads((REPO / F.AD4_CERT).read_text())
    rng = np.random.default_rng(1)
    (h2 / "dataset").mkdir(parents=True)
    np.save(h2 / "dataset" / "test_y.npy", rng.integers(0, 2, 1_000_000, dtype=np.uint8))
    (h2 / "fits" / "predictions").mkdir(parents=True)
    step = 0
    for row in cert["findings"]["replicates"]:
        for cond, rounds in (("baseline", [0.9, 0.8, 0.7, 0.6, 0.501]), ("perturbed", [0.503])):
            np.save(h2 / "fits" / "predictions" / row[f"{cond}_predictions"]["file"],
                    rng.random(1_000_000, dtype=np.float32))
            for i, rec in enumerate(F._fit(cond, row["replicate"], rounds)):
                rec["step"] = i + 1
                d = h2 / "fits" / f"replicate_{row['replicate']:02d}" / cond / f"stage_r{rec['stage_round']:02d}"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"checkpoint_epoch_{rec['stage_epoch']:03d}.metrics.json").write_text(json.dumps(rec))
                step += 1


CHILD = r"""
import json, sys, traceback
from audit.framework_validation.sdb_v2 import cases_v2 as C, baselines_v2 as B
skip = set(sys.argv[1].split(","))
exp = C.expected_outcomes()
report = {}
def keys(e, o, path=""):
    if isinstance(e, dict):
        if not isinstance(o, dict):
            return [path or "<root>"]
        return [m for k, v in e.items() for m in ([f"{path}/{k}"] if k not in o else keys(v, o[k], f"{path}/{k}"))]
    return []
for case in C.ORDER:
    if case in skip:
        continue
    try:
        obs, raw = C.ADAPTERS[case](case)
        missing = keys(exp[case], obs)
        b2 = B.b2(case)
        report[case] = {"ran": True, "missing_keys": missing, "b2_items": sorted(b2["items"]), "b1": B.b1_status(case)}
    except Exception:
        report[case] = {"ran": False, "error": traceback.format_exc()[-1500:]}
try:
    out, _ = B.b1s()
    report["B1S"] = {"ran": True, "arrays": len(out), "missing_files": sorted(set(exp["_H2_B1S"]) - set(out))}
except Exception:
    report["B1S"] = {"ran": False, "error": traceback.format_exc()[-1500:]}
print("SMOKE_JSON" + json.dumps(report))
"""


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    if root.is_relative_to(REPO):
        print("REFUSED: the smoke root must lie outside the repository")
        return 1
    build(root)
    env = dict(os.environ, SDB_V2_FIXTURES_ROOT=str(root / "fixtures"), SDB_V2_H2_BUNDLE=str(root / "h2"))
    r = subprocess.run([sys.executable, "-B", "-c", CHILD, ",".join(sorted(SKIP))], cwd=REPO, env=env,
                       capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("SMOKE_JSON")), None)
    if line is None:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return 1
    rep = json.loads(line[len("SMOKE_JSON"):])
    ok = True
    for case, v in rep.items():
        good = v["ran"] and not v.get("missing_keys") and not v.get("missing_files")
        ok &= good
        print(f"{case:4} {'ok ' if good else 'BAD'} " + ("" if good else json.dumps(v)[:1500]))
    print("SMOKE", "PASSED" if ok else "FAILED", f"({len(rep)} entries; values not shown by design)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
