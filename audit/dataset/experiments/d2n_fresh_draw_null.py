"""D2-N: near-duplicate structure against a fresh-draw null.

D2.1 asks whether rows of a dataset have other rows within a small Hamming
distance more often than independent draws from the generator would. The
schema 6.5 certificates could not answer it (D21_SUPERSEDED_METHOD_DEFECTS),
and a declared closed-form null cannot answer it for structured data: at five
rounds a real ciphertext pair is not a uniform 64-bit row, so the uniform
null over-predicts neither the cipher's own structure nor anything else.

The generator is its own null. Every canonical instance was built by
``canonical._resolve_generator(spec)`` from the specification recorded in its
manifest, with rows drawn i.i.d. from ``os.urandom``. Calling the same
generator again with the same specification and partition sizes yields
datasets that share every property of the generator and none of the
within-dataset dependence D2.1 exists to detect. The audited count is ranked
among M such counts:

    T   = rows, among the first Q queries, with another row within radius r
          (exact scan over all n-1 rows, ``near_neighbor_exact_screen``)
    p   = (1 + #{j : T_j >= T_obs}) / (M + 1)          Monte Carlo rank test
    R   = T_obs / mean(T_j)                            practical ratio

The design is fixed in the preregistration file and hashed; production
refuses to run unless that hash carries an approval.

    python -m audit.dataset.experiments.d2n_fresh_draw_null plan   --prereg P
    python -m audit.dataset.experiments.d2n_fresh_draw_null smoke  --prereg P --pipeline gohr --instance-dir D --out-dir O
    python -m audit.dataset.experiments.d2n_fresh_draw_null measure --prereg P --pipeline autond --instance-dir D --autond-root U
    python -m audit.dataset.experiments.d2n_fresh_draw_null run    --prereg P --pipeline autond --instance-dir D --autond-root U --out-dir O --workers 4 --confirm RUN_D2N_PRODUCTION
    python -m audit.dataset.experiments.d2n_fresh_draw_null analyse --prereg P --run-dir O
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PARTITIONS = ("train", "validation", "test")
CONFIRM_TOKEN = "RUN_D2N_PRODUCTION"
RECORD_SCHEMA = "d2n-record-1"
CERT_SCHEMA = "d2n-certificate-1"

#: Files whose content decides the result. Production refuses a dirty tree for
#: any of them, and every record carries their hashes.
BOUND_SOURCES = (
    "audit/dataset/experiments/d2n_fresh_draw_null.py",
    "audit/dataset/d2_sample_dependence.py",
    "audit/dataset/canonical.py",
    "audit/dataset/adapters/gohr.py",
    "audit/dataset/adapters/autond.py",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def source_hashes() -> dict[str, str]:
    return {rel: sha256_file(REPO_ROOT / rel) for rel in BOUND_SOURCES}


def git_state() -> dict[str, Any]:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    try:
        head = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--", *BOUND_SOURCES)
    except FileNotFoundError:
        return {"head": None, "bound_sources_dirty": None}
    return {"head": head or None, "bound_sources_dirty": bool(dirty),
            "dirty_entries": dirty.splitlines() if dirty else []}


# ---------------------------------------------------------------------------
# preregistration
# ---------------------------------------------------------------------------

def load_prereg(path: Path) -> tuple[dict, str]:
    doc = json.loads(Path(path).read_text())
    if "design" not in doc or "approval" not in doc:
        raise SystemExit("REFUSED: preregistration needs 'design' and 'approval'.")
    return doc, canonical_sha256(doc["design"])


def require_approval(doc: dict, design_sha: str) -> None:
    a = doc["approval"]
    if a.get("status") != "APPROVED":
        raise SystemExit(f"REFUSED: preregistration status is {a.get('status')!r}, "
                         "not APPROVED. Production never runs on an unapproved design.")
    if a.get("design_sha256") != design_sha:
        raise SystemExit(f"REFUSED: approval names design {a.get('design_sha256')}, "
                         f"but the design hashes to {design_sha}.")
    if not a.get("approved_by") or not a.get("approved_utc"):
        raise SystemExit("REFUSED: approval lacks approved_by / approved_utc.")


def pipeline_design(design: dict, pipeline: str) -> dict:
    if pipeline not in design["pipelines"]:
        raise SystemExit(f"REFUSED: pipeline {pipeline!r} is not registered.")
    return design["pipelines"][pipeline]


# ---------------------------------------------------------------------------
# data: the audited instance and fresh draws from the same generator
# ---------------------------------------------------------------------------

def load_instance(instance_dir: Path, pdesign: dict) -> tuple[dict, dict]:
    from audit.dataset import canonical as C

    man = C.load_manifest(instance_dir)
    if man["canonical_instance_id"] != pdesign["canonical_instance_id"]:
        raise SystemExit(f"REFUSED: instance {man['canonical_instance_id']} is not the "
                         f"registered {pdesign['canonical_instance_id']}.")
    ver = C.verify(instance_dir, log=lambda *a, **k: None)
    if not ver.get("verified") or ver.get("canonical_instance_id") != pdesign["canonical_instance_id"]:
        raise SystemExit(f"REFUSED: canonical instance did not verify: "
                         f"{ver.get('partitions')}")
    spec = man["specification"]
    X = {}
    for name in PARTITIONS:
        x, _ = C.load_partition(instance_dir, name, spec=spec, verify_file_hash=True)
        if man["partitions"][name]["samples"] != x.shape[0]:
            raise SystemExit(f"REFUSED: {name} row count differs from the manifest.")
        X[name] = np.ascontiguousarray(x)
    return man, X


def make_generator(man: dict, *, autond_root: str | None):
    from audit.dataset import canonical as C

    spec = dict(man["specification"])
    if spec.get("target", "gohr-speck").startswith("autond"):
        if not autond_root:
            raise SystemExit("REFUSED: --autond-root is required for the autond pipeline.")
        spec["autond_root"] = str(Path(autond_root).expanduser().resolve())
        check_upstream_pin(spec)
    generate, path = C._resolve_generator(spec)
    return generate, path, spec


def check_upstream_pin(spec: dict) -> None:
    """The fresh-draw generator must be the upstream code the instance was built from."""
    pins = (spec.get("adapter_provenance") or {}).get("source_digests") or {}
    if not pins:
        raise SystemExit("REFUSED: the instance manifest records no upstream source digests.")
    root = Path(spec["autond_root"])
    bad = {f: {"expected": h, "found": sha256_file(root / f) if (root / f).exists() else None}
           for f, h in pins.items()
           if not (root / f).exists() or sha256_file(root / f) != h}
    if bad:
        raise SystemExit(f"REFUSED: AutoND checkout differs from the files the instance was "
                         f"built from: {bad}")


def check_environment(design: dict) -> None:
    """Production runs only under the numpy and Python the instances were built with."""
    if np.__version__ != "1.26.4" or platform.python_version_tuple()[:2] != ("3", "11"):
        raise SystemExit(f"REFUSED: production needs Python 3.11 and numpy 1.26.4 "
                         f"({design['environment_requirement']}); found Python "
                         f"{platform.python_version()}, numpy {np.__version__}.")


def recorded_generator_path(man: dict) -> str | None:
    return man.get("generator_path")


# ---------------------------------------------------------------------------
# one dataset: generate (null) or load (observed), scan every partition
# ---------------------------------------------------------------------------

def scan(x: np.ndarray, *, radii, queries: int, seed: int) -> dict:
    from audit.dataset.d2_sample_dependence import near_neighbor_exact_screen

    res = near_neighbor_exact_screen(
        x, radii, rng=np.random.default_rng(seed), null_model={"kind": "uniform"},
        query_count=queries)
    per = {}
    for r in radii:
        rec = res["by_radius"][str(r)]
        per[str(r)] = {"hits": int(rec["rows_with_neighbor_within_radius"]),
                       "queried_rows": int(rec["queried_rows"])}
    return {"per_radius": per, "rows": int(res["rows"]),
            "exact_duplicates": res.get("exact_duplicates")}


def query_seed(base: int, index: int, part: int) -> int:
    return int(base) + 1000 * int(index) + int(part)


def dataset_task(args: dict) -> dict:
    """Worker body. Index 0 is the audited instance; 1..M are fresh draws."""
    t0 = time.time()
    design, pdesign = args["design"], args["pdesign"]
    index, sizes = args["index"], args["sizes"]
    rec: dict[str, Any] = {"schema": RECORD_SCHEMA, "pipeline": args["pipeline"],
                           "index": index, "role": "observed" if index == 0 else "null",
                           "design_sha256": args["design_sha256"],
                           "source_hashes": args["source_hashes"], "started_utc": utc(),
                           "partitions": {}}
    if index == 0:
        man, X = load_instance(Path(args["instance_dir"]), pdesign)
        data = X
        rec["instance_id"] = man["canonical_instance_id"]
    else:
        from audit.dataset import canonical as C
        man = json.loads((Path(args["instance_dir"]) / "d0_canonical_manifest.json").read_text())
        generate, gpath, spec = make_generator(man, autond_root=args.get("autond_root"))
        rec["generator_path"] = gpath
        data = {}
        for name in PARTITIONS:
            x, y = generate(int(sizes[name]))
            x = np.ascontiguousarray(x)
            probs = C.validate_generated_partition(name, x, y, requested=int(sizes[name]),
                                                   spec=spec)
            if probs:
                raise SystemExit(f"REFUSED: fresh {name} partition is non-conforming: {probs}")
            data[name] = x
            rec["partitions"].setdefault(name, {})["features_sha256"] = array_sha256(x)
            rec["partitions"][name]["label_positive"] = int(np.asarray(y).sum())
    for k, name in enumerate(PARTITIONS):
        s = scan(data[name], radii=tuple(design["radii"]),
                 queries=int(design["queries_per_partition"]),
                 seed=query_seed(design["query_seed_base"], index, k))
        rec["partitions"].setdefault(name, {}).update(s)
        rec["partitions"][name]["query_seed"] = query_seed(design["query_seed_base"], index, k)
        del data[name]
    rec["finished_utc"] = utc()
    rec["seconds"] = round(time.time() - t0, 1)
    rec["environment"] = environment()
    rec["record_sha256"] = canonical_sha256({k: v for k, v in rec.items()
                                             if k != "record_sha256"})
    out = Path(args["out_dir"]) / "records" / f"{index:03d}.json"
    write_json_atomic(out, rec)
    return {"index": index, "seconds": rec["seconds"],
            "hits_r8": {p: rec["partitions"][p]["per_radius"].get("8", {}).get("hits")
                        for p in PARTITIONS}}


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "cpus": os.cpu_count()}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def rank_p(observed: int, null: list[int]) -> float:
    return (1 + sum(1 for v in null if v >= observed)) / (len(null) + 1)


def classify(observed: int, null: list[int], *, alpha_per: float, ratio_gate: float) -> dict:
    p = rank_p(observed, null)
    mean = float(np.mean(null)) if null else float("nan")
    ratio = (float("inf") if mean == 0 and observed > 0 else
             (1.0 if mean == 0 else observed / mean))
    stat = p <= alpha_per + 1e-12          # 1/60 == 0.05/3 exactly; never decide on rounding
    prac = ratio >= ratio_gate
    if stat and prac:
        outcome = "EXCESS"
    elif stat:
        outcome = "INCONCLUSIVE"
    else:
        outcome = "NO_EXCESS"
    return {"observed": int(observed), "null_mean": mean,
            "null_sd": float(np.std(null, ddof=1)) if len(null) > 1 else None,
            "null_min": int(min(null)) if null else None,
            "null_max": int(max(null)) if null else None,
            "rank_p": p, "ratio": ratio, "statistical_gate": stat,
            "practical_gate": prac, "outcome": outcome}


def analyse(run_dir: Path, design: dict, design_sha: str, pipeline: str) -> dict:
    recs = {}
    for f in sorted((run_dir / "records").glob("*.json")):
        r = json.loads(f.read_text())
        body = {k: v for k, v in r.items() if k != "record_sha256"}
        if canonical_sha256(body) != r.get("record_sha256"):
            raise SystemExit(f"REFUSED: record {f.name} fails its own hash.")
        if r["design_sha256"] != design_sha:
            raise SystemExit(f"REFUSED: record {f.name} was produced under another design.")
        recs[r["index"]] = r
    M = int(design["null_datasets"])
    missing = [i for i in range(M + 1) if i not in recs]
    pdesign = pipeline_design(design, pipeline)
    if missing:
        return {"outcome": "INCOMPLETE", "missing_indices": missing}
    primary = str(design["primary_radius"])
    alpha_per = float(design["alpha"]) / len(PARTITIONS)
    results, flags = {}, []
    for name in PARTITIONS:
        results[name] = {}
        for r in design["radii"]:
            obs = recs[0]["partitions"][name]["per_radius"][str(r)]["hits"]
            null = [recs[j]["partitions"][name]["per_radius"][str(r)]["hits"]
                    for j in range(1, M + 1)]
            c = classify(obs, null, alpha_per=alpha_per,
                         ratio_gate=float(design["practical_ratio"]))
            c["role"] = "primary" if str(r) == primary else "detection"
            results[name][str(r)] = c
            if str(r) != primary and obs > 0 and max(null) == 0:
                flags.append(f"{name}: radius {r} has {obs} observed hit(s) and none in "
                             f"any of {M} fresh datasets")
    prim = [results[p][primary]["outcome"] for p in PARTITIONS]
    if "EXCESS" in prim:
        outcome = "FAIL"
    elif "INCONCLUSIVE" in prim or flags:
        outcome = "INCONCLUSIVE"
    else:
        outcome = "PASS"
    sources = {json.dumps(recs[i]["source_hashes"], sort_keys=True) for i in recs}
    return {"schema": CERT_SCHEMA, "pipeline": pipeline, "design_sha256": design_sha,
            "canonical_instance_id": pdesign["canonical_instance_id"],
            "null_datasets": M, "alpha_per_partition": alpha_per,
            "results": results, "detection_flags": flags,
            "d2n_outcome": outcome,
            "d2_mapping": pdesign["d2_mapping"],
            "single_source_identity": len(sources) == 1,
            "records": {str(i): recs[i]["record_sha256"] for i in sorted(recs)},
            "generated_utc": utc()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def sizes_for(pdesign: dict, man: dict) -> dict:
    return {p: int(man["partitions"][p]["samples"]) for p in PARTITIONS}


def _run_tasks(tasks: list[dict], workers: int) -> None:
    if workers <= 1:
        for t in tasks:
            print(json.dumps(dataset_task(t)), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(dataset_task, t): t["index"] for t in tasks}
        for fut in as_completed(futs):
            print(json.dumps(fut.result()), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("plan", "smoke", "measure", "run", "analyse"))
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--pipeline", choices=("gohr", "autond"))
    ap.add_argument("--instance-dir", type=Path)
    ap.add_argument("--autond-root")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--confirm")
    a = ap.parse_args(argv)

    doc, dsha = load_prereg(a.prereg)
    design = doc["design"]
    if a.command == "plan":
        print(json.dumps({"design_sha256": dsha, "approval": doc["approval"],
                          "source_hashes": source_hashes(), "git": git_state()}, indent=2))
        return 0

    if a.command == "analyse":
        run_dir = a.run_dir or a.out_dir
        meta = json.loads((run_dir / "run.json").read_text())
        cert = analyse(run_dir, design, dsha, meta["pipeline"])
        cert["approval"] = doc["approval"]
        write_json_atomic(run_dir / "d2n_certificate.json", cert)
        print(json.dumps({"outcome": cert.get("d2n_outcome", cert.get("outcome")),
                          "primary": {p: cert["results"][p][str(design["primary_radius"])]
                                      for p in PARTITIONS} if "results" in cert else None},
                         indent=2))
        return 0

    if not a.pipeline or not a.instance_dir:
        ap.error("--pipeline and --instance-dir are required")
    pdesign = pipeline_design(design, a.pipeline)
    from audit.dataset import canonical as C
    man = C.load_manifest(a.instance_dir)
    sizes = sizes_for(pdesign, man)
    common = {"design": design, "pdesign": pdesign, "design_sha256": dsha,
              "pipeline": a.pipeline, "instance_dir": str(a.instance_dir),
              "autond_root": a.autond_root, "source_hashes": source_hashes()}

    if a.command == "smoke":
        # Tiny partitions, two null draws: exercises every code path. Not evidence.
        smoke_design = dict(design, queries_per_partition=256, null_datasets=2)
        out = a.out_dir or Path("d2n_smoke")
        man_s = {"partitions": {p: {"samples": 20_000} for p in PARTITIONS}}
        tasks = [dict(common, design=smoke_design, index=j, out_dir=str(out),
                      sizes=sizes_for(pdesign, man_s)) for j in (1, 2)]
        for t in tasks:
            print(json.dumps(dataset_task(t)))
        print("SMOKE ONLY: fresh-draw path exercised at 20,000 rows; not evidence.")
        return 0

    if a.command == "measure":
        t = dict(common, index=1, out_dir=str(a.out_dir or Path("d2n_measure")), sizes=sizes)
        t0 = time.time()
        print(json.dumps(dataset_task(t)))
        print(f"one full-size fresh dataset (generate + 3 exact scans): "
              f"{time.time() - t0:.0f} s. Not evidence; output is discarded.")
        return 0

    # run
    if a.confirm != CONFIRM_TOKEN:
        raise SystemExit(f"REFUSED: production needs --confirm {CONFIRM_TOKEN}.")
    require_approval(doc, dsha)
    check_environment(design)
    g = git_state()
    if g["bound_sources_dirty"]:
        raise SystemExit(f"REFUSED: bound sources are not committed: {g['dirty_entries']}")
    if not a.out_dir:
        ap.error("--out-dir is required")
    out = a.out_dir
    meta_path = out / "run.json"
    meta = {"pipeline": a.pipeline, "design_sha256": dsha, "git": g,
            "source_hashes": common["source_hashes"], "sizes": sizes,
            "workers": a.workers}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for k in ("pipeline", "design_sha256", "source_hashes", "sizes"):
            if old.get(k) != meta[k]:
                raise SystemExit(f"REFUSED: resume with a different {k}.")
        meta["started_utc"] = old["started_utc"]
    else:
        meta["started_utc"] = utc()
    write_json_atomic(meta_path, meta)
    M = int(design["null_datasets"])
    done = {int(f.stem) for f in (out / "records").glob("*.json")} if (out / "records").exists() else set()
    tasks = [dict(common, index=j, out_dir=str(out), sizes=sizes)
             for j in range(M + 1) if j not in done]
    print(f"D2-N {a.pipeline}: {len(done)} of {M + 1} datasets already recorded; "
          f"{len(tasks)} to run with {a.workers} worker(s).", flush=True)
    _run_tasks(tasks, a.workers)
    print("All datasets recorded. Next: analyse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
