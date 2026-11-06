"""
G-CE5 runner. Stages, in registered order:

    tables       build the 4-round aKR table and the 5-round LR table (Algorithm 1) and run
                 the independent table check; the hashes are then recorded in the design
    tablecheck   re-verify the registered tables with a fresh independent sample (non-confirmatory)
    benchmark    scoring cost, memory and scalar agreement                (non-confirmatory)
    variability  5 independent tables, each scored on its own fresh pairs, giving the
                 gate tolerance                                          (non-confirmatory)
    calibrate    10 blinded replicates giving the spread of Delta        (non-confirmatory)
    production   authorisation -> model binding -> table hashes -> gate -> replicates -> analysis

Each pre-approval stage refuses while design.stage_problems(stage) is non-empty
and writes only to paths that do not yet exist. Calibration and production
refuse unless at least 2 GiB of memory is available. Production refuses unless
production_authorisation() passes and runs no replicate unless the gate is PASS.

Usage, from the repository root:
    python -m audit.cryptography.experiments.ce5.runner tables      OUT
    python -m audit.cryptography.experiments.ce5.runner tablecheck  OUT --tables T
    python -m audit.cryptography.experiments.ce5.runner benchmark   OUT --tables T
    python -m audit.cryptography.experiments.ce5.runner variability OUT
    python -m audit.cryptography.experiments.ce5.runner calibrate   OUT --tables T
    python -m audit.cryptography.experiments.ce5.runner production  OUT --tables T --confirm TOKEN \
        --prereg PATH --prereg-sha256 HEX
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.cryptography.experiments.ce5 import akr as A
from audit.cryptography.experiments.ce5 import analysis as AN
from audit.cryptography.experiments.ce5 import design as DS
from audit.cryptography.experiments.ce5 import reference as REF

CE5_ROOT = Path(__file__).resolve().parent
CRYPTO_ROOT = CE5_ROOT.parents[1]
SOURCE_ROOT = Path(__file__).resolve().parent
FROZEN_DEPENDENCIES = ("ce234/production.py", "ce234/frozen_plan.py", "gohr/speck.py")
ROUNDS = 5
DIFF = (0x0040, 0x0000)
TABLE_FILES = {"akr": "addt_r4_counts.npz", "lr": "addt_r5_counts.npz"}
TABLE_ROUNDS = {"akr": ROUNDS - 1, "lr": ROUNDS}
MIN_AVAILABLE_BYTES = 2 << 30


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return sha256_file(path)


def write_json_new(path: Path, obj) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    return write_json(path, obj)


def require_fresh(*paths: Path) -> None:
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"refused: output already exists: {taken}")


def write_npz(path: Path, **arrays) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)
    return sha256_file(path)


def peak_rss_mib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


def available_memory_bytes() -> int | None:
    """free + inactive + speculative pages from vm_stat (macOS), else MemAvailable."""
    import re
    import subprocess
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1))
                        for k in ("Pages free", "Pages inactive", "Pages speculative"))
            return pages * page
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def require_memory(minimum: int = MIN_AVAILABLE_BYTES, probe=available_memory_bytes) -> int:
    avail = probe()
    if avail is None or avail < minimum:
        raise SystemExit(f"refused: available memory {avail} bytes < required {minimum}; close other applications")
    return avail


def require_stage(stage: str) -> None:
    problems = DS.stage_problems(stage)
    if problems:
        raise SystemExit(f"{stage} refused: design has unset items {problems}")


def environment() -> dict:
    import os

    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}


# ---------------------------------------------------------------- tables

def table_checks(counts, n: int, rounds: int) -> dict:
    mask = A.MASK
    res = {r: REF.table_check(counts, n, rounds=r, diff=DIFF, mask=mask) for r in (rounds - 1, rounds, rounds + 1)}
    ok = res[rounds]["p_value"] >= 1e-3 and all(res[r]["p_value"] < 1e-12 for r in (rounds - 1, rounds + 1))
    return {"by_rounds": {str(r): v for r, v in res.items()}, "decision": "PASS" if ok else "FAIL"}


def build_tables(out: Path, n_pairs: int, *, rng=None) -> dict:
    require_stage("tables")
    require_fresh(out / "tables_manifest.json", *(out / f for f in TABLE_FILES.values()))
    rec = {"created_utc": utc(), "n_pairs": n_pairs, "mask": [hex(m) for m in A.MASK],
           "input_difference": [hex(x) for x in DIFF], "algorithm": A.ALGORITHM,
           "count_dtype": "uint32", "smoothing": "none", "environment": environment(), "files": {}}
    for name, rounds in TABLE_ROUNDS.items():
        t0, c0 = time.time(), time.process_time()
        counts = A.build_table(n_pairs, rounds, DIFF, rng=rng)
        if int(counts.sum(dtype=np.uint64)) != n_pairs:
            raise RuntimeError(f"{name} table holds {int(counts.sum())} counts, expected {n_pairs}")
        built = time.time() - t0
        check = table_checks(counts, n_pairs, rounds)
        if check["decision"] != "PASS":
            rec["files"][name] = {"rounds": rounds, "table_check": check, "written": False}
            write_json_new(out / f"tables_manifest_FAILED_{utc().replace(':', '')}.json", rec)
            raise SystemExit(f"tables refused: the independent check of the {name} table failed; no table written")
        digest = write_npz(out / TABLE_FILES[name], counts=counts, n_pairs=np.int64(n_pairs),
                           rounds=np.int64(rounds), mask=np.array(A.MASK, dtype=np.uint16))
        rec["files"][name] = {"file": TABLE_FILES[name], "sha256": digest, "rounds": rounds,
                              "nonzero_cells": int((counts > 0).sum()), "max_count": int(counts.max()),
                              "build_seconds": round(built, 1),
                              "cpu_seconds": round(time.process_time() - c0, 1),
                              "table_check": check}
    rec["peak_rss_mib"] = round(peak_rss_mib(), 1)
    write_json_new(out / "tables_manifest.json", rec)
    return rec


def load_tables(tdir: Path, expected: dict | None = None, expected_pairs: int | None = None) -> dict:
    out = {}
    for name, fn in TABLE_FILES.items():
        p = Path(tdir) / fn
        digest = sha256_file(p)
        if expected is not None and digest != expected[name]:
            raise RuntimeError(f"{fn}: sha256 {digest} is not the registered {expected[name]}")
        with np.load(p) as z:
            if int(z["rounds"]) != TABLE_ROUNDS[name] or tuple(int(m) for m in z["mask"]) != A.MASK:
                raise RuntimeError(f"{fn}: rounds or mask differ from the design")
            counts, n = z["counts"], int(z["n_pairs"])
        if counts.dtype != np.uint32 or counts.size != A.MaskedIndex().size:
            raise RuntimeError(f"{fn}: counts are not {A.MaskedIndex().size} uint32 cells")
        if int(counts.sum(dtype=np.uint64)) != n:
            raise RuntimeError(f"{fn}: counts do not sum to n_pairs")
        if expected_pairs is not None and n != expected_pairs:
            raise RuntimeError(f"{fn}: n_pairs {n} is not the registered {expected_pairs}")
        out[name] = (counts, n, digest)
    return out


def registered_tables(tdir: Path, d: dict) -> dict:
    return load_tables(tdir, {"akr": d["baseline_D"]["table_sha256"], "lr": d["secondary_comparator_LR"]["table_sha256"]},
                       int(d["baseline_D"]["table_pairs"]))


def require_table_validation(d: dict, root: Path | None = None) -> dict:
    """The registered table-validation record (a path relative to the CE5
    directory) must exist, match its registered hash, report overall PASS and
    name the registered tables."""
    tv = d["baseline_D"]["table_validation"]
    if Path(tv["record"]).is_absolute():
        raise SystemExit("refused: the table-validation record must be registered as a path relative to the CE5 directory")
    p = (CE5_ROOT if root is None else root) / tv["record"]
    if not p.exists():
        raise SystemExit(f"refused: table-validation record {tv['record']} is missing")
    if sha256_file(p) != tv["sha256"]:
        raise SystemExit("refused: table-validation record does not match its registered hash")
    rec = json.loads(p.read_text())
    if rec.get("decision") != "PASS":
        raise SystemExit(f"refused: table validation decision is {rec.get('decision')!r}")
    names = {k: v["sha256"] for k, v in rec.get("tables", {}).items()}
    if names != {"akr": d["baseline_D"]["table_sha256"], "lr": d["secondary_comparator_LR"]["table_sha256"]}:
        raise SystemExit("refused: table-validation record does not name the registered tables")
    return {"record": tv["record"], "sha256": tv["sha256"], "decision": "PASS"}


# ---------------------------------------------------------------- scoring

def bind_model():
    from audit.cryptography.ce234 import production as PR

    binding = PR.verify_reference_model(behavioural=True)
    if binding["sha256"] != DS.REFERENCE_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint is not the registered one")
    return binding, PR.load_reference_model()


def one_replicate(model, tables, n: int, index) -> dict:
    from audit.cryptography.ce234 import production as PR
    from audit.cryptography.gohr import speck as S

    X, T, F = PR.generate_theory_data(n, with_factors=True)
    c = A.words_from_binary(X)
    N = PR.predict(model, X)
    D = A.akr_scores(*c, tables["akr"][0], tables["akr"][1], index=index)
    LR = A.difference_scores(*c, tables["lr"][0], tables["lr"][1], index=index)
    cr = A.ciphertext_pairs(n, ROUNDS, DIFF, real=False)
    Nr = PR.predict(model, S.convert_to_binary(list(cr)))
    Dr = A.akr_scores(*cr, tables["akr"][0], tables["akr"][1], index=index)
    return {"real": {"c0l": c[0], "c0r": c[1], "c1l": c[2], "c1r": c[3],
                     "N": N, "T": T, "F": F, "D": D, "LR": LR},
            "random": {"c0l": cr[0], "c0r": cr[1], "c1l": cr[2], "c1r": cr[3], "N": Nr, "D": Dr}}


def save_replicate(path: Path, data: dict) -> str:
    return write_npz(path, **{f"{cls}_{k}": v for cls, d in data.items() for k, v in d.items()})


def run_gate(out: Path, tables, index, d: dict) -> dict:
    half = int(d["gate"]["pairs_per_class"])
    cr = A.ciphertext_pairs(half, ROUNDS, DIFF, real=True)
    cn = A.ciphertext_pairs(half, ROUNDS, DIFF, real=False)
    sr = A.akr_scores(*cr, tables["akr"][0], tables["akr"][1], index=index)
    sn = A.akr_scores(*cn, tables["akr"][0], tables["akr"][1], index=index)
    tp = int(A.akr_decide(sr).sum())
    tn = int((~A.akr_decide(sn)).sum())
    g = AN.gate(tp, half - tp, tn, half - tn, DS.gate_spec(d))
    g["raw_sha256"] = write_npz(out / "gate_raw.npz", real_score=sr, random_score=sn,
                                real_c=np.stack(cr), random_c=np.stack(cn))
    return g


def recover_gate(out: Path, tables, index, d: dict) -> dict:
    """Adopt gate_raw.npz left by an interrupted run only if its stored
    ciphertexts re-score exactly to its stored scores at the registered size;
    otherwise refuse and leave the file in place."""
    p = out / "gate_raw.npz"
    half = int(d["gate"]["pairs_per_class"])
    with np.load(p) as z:
        sr, sn, cr, cn = z["real_score"], z["random_score"], z["real_c"], z["random_c"]
    if sr.shape != (half,) or sn.shape != (half,) or cr.shape != (4, half) or cn.shape != (4, half):
        raise SystemExit(f"production refused: {p} does not have the registered gate size; it is kept. "
                         "Start a new output directory.")
    if not (np.array_equal(A.akr_scores(*cr, tables["akr"][0], tables["akr"][1], index=index), sr)
            and np.array_equal(A.akr_scores(*cn, tables["akr"][0], tables["akr"][1], index=index), sn)):
        raise SystemExit(f"production refused: {p} scores do not re-derive from its ciphertexts; it is kept. "
                         "Start a new output directory.")
    tp, tn = int(A.akr_decide(sr).sum()), int((~A.akr_decide(sn)).sum())
    g = AN.gate(tp, half - tp, tn, half - tn, DS.gate_spec(d))
    g["raw_sha256"] = sha256_file(p)
    g["recovered"] = "gate_raw.npz from an interrupted run; all scores re-derived exactly from its ciphertexts"
    return g


def record_ots_verification(prereg_path: Path, *, ots_bin: str = "ots", out: Path | None = None,
                            timeout: int = 30) -> dict:
    """Run `ots -v verify` on prereg_path.ots and record the command, exit code,
    stdout, stderr and the file and proof hashes before and after. Reads the
    proof only; writes only the new record."""
    import os
    import subprocess
    prereg_path = Path(prereg_path)
    proof = Path(str(prereg_path) + ".ots")
    out = Path(out) if out else Path(str(proof) + ".verification.json")
    require_fresh(out)
    before = {"file_sha256": sha256_file(prereg_path), "proof_sha256": sha256_file(proof)}
    env = dict(os.environ)
    try:
        import certifi
        env.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    cmd = [ots_bin, "-v", "verify", "--timeout", str(timeout), str(proof)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    after = {"file_sha256": sha256_file(prereg_path), "proof_sha256": sha256_file(proof)}
    if after != before:
        raise SystemExit("ots verify changed the file or proof; record not written")
    rec = {"created_utc": utc(), "command": cmd, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
           **before}
    write_json_new(out, rec)
    return rec


# ---------------------------------------------------------------- non-confirmatory stages

def validate_tables(out: Path, tdir: Path, *, seed: int = 20261106, n_sample: int = 100_000) -> dict:
    """Re-verify the registered tables with a fresh independent sample, and show
    that the same check rejects a (dl, dr)-keyed table and a wrong input difference."""
    require_stage("benchmark")
    require_fresh(out / "table_validation.json")
    d = DS.design()
    tables = registered_tables(tdir, d)
    index = A.MaskedIndex()
    lut_ok = bool(np.array_equal(np.sort(index.lut_l[np.unique(np.arange(1 << 16, dtype=np.uint16)
                                                                 & np.uint16(A.MASK[0]))]), np.arange(1 << 13)))
    rec = {"created_utc": utc(), "excluded_from_evidence": True, "seed": seed, "n_sample": n_sample,
           "compaction_bijective": lut_ok, "tables": {}}
    for name, (counts, n, digest) in tables.items():
        rounds = TABLE_ROUNDS[name]
        checks = {f"rounds_{r}": REF.table_check(counts, n, rounds=r, diff=DIFF, mask=A.MASK, n_sample=n_sample,
                                                 seed=seed) for r in (rounds - 1, rounds, rounds + 1)}
        checks["swapped_difference"] = REF.table_check(counts, n, rounds=rounds, diff=(0x0000, 0x0040),
                                                       mask=A.MASK, n_sample=n_sample, seed=seed)
        rec["tables"][name] = {"file": TABLE_FILES[name], "sha256": digest, "rounds": rounds, "n_pairs": n,
                               "dtype": str(counts.dtype), "cells": int(counts.size),
                               "sum_equals_n": int(counts.sum(dtype=np.uint64)) == n,
                               "max_count": int(counts.max()), "nonzero_cells": int((counts > 0).sum()),
                               "checks": checks,
                               "decision": "PASS" if (checks[f"rounds_{rounds}"]["p_value"] >= 1e-3 and all(
                                   checks[k]["p_value"] < 1e-12 for k in checks if k != f"rounds_{rounds}"))
                               else "FAIL"}
    c0l, c0r, c1l, c1r = A.ciphertext_pairs(int(d["baseline_D"]["table_pairs"]), 4, DIFF, real=True)
    wrong = np.zeros(index.size, dtype=np.uint32)
    np.add.at(wrong, index(c0l ^ c1l, c0r ^ c1r), 1)
    rec["dl_dr_table_rejected"] = REF.table_check(wrong, int(d["baseline_D"]["table_pairs"]), rounds=4, diff=DIFF,
                                                  mask=A.MASK, n_sample=n_sample, seed=seed)
    del wrong, c0l, c0r, c1l, c1r
    parts = {"compaction_bijective": lut_ok,
             "dl_dr_control_rejected": rec["dl_dr_table_rejected"]["p_value"] < 1e-12,
             **{f"{k}_table": v["decision"] == "PASS" and v["sum_equals_n"] and v["n_pairs"] == int(d["baseline_D"]["table_pairs"])
                for k, v in rec["tables"].items()}}
    rec["components"] = parts
    rec["decision"] = "PASS" if all(parts.values()) else "FAIL"
    rec["peak_rss_mib"] = round(peak_rss_mib(), 1)
    write_json_new(out / "table_validation.json", rec)
    return rec


def benchmark(out: Path, tdir: Path, n_pairs: int = 2048, n_scalar: int = 4, n_full: int = 50_000) -> dict:
    require_stage("benchmark")
    require_fresh(out / "benchmark.json")
    d = DS.design()
    tables = registered_tables(tdir, d)
    counts, n4, _ = tables["akr"]
    index = A.MaskedIndex()
    mem_start = available_memory_bytes()
    c = A.ciphertext_pairs(n_pairs, ROUNDS, DIFF, real=True)
    timings = {}
    for label, n, fn in (("published_2^12", n_pairs, A.akr_scores), ("all_2^16_test_only", n_pairs // 16, A.full_average)):
        w0, c0 = time.perf_counter(), time.process_time()
        s = fn(*(a[:n] for a in c), counts, n4, index=index)
        wall = time.perf_counter() - w0
        timings[label] = {"pairs": n, "wall_seconds": wall, "seconds_per_pair": wall / n,
                          "cpu_utilisation": (time.process_time() - c0) / wall, "scores": s}
    m = timings["all_2^16_test_only"]["pairs"]
    exact = bool(np.array_equal(timings["published_2^12"]["scores"][:m], timings["all_2^16_test_only"]["scores"]))
    w0 = time.perf_counter()
    ref = [REF.score((int(c[0][j]), int(c[1][j])), (int(c[2][j]), int(c[3][j])), counts, n4, A.MASK)
           for j in range(n_scalar)]
    ref_sec = (time.perf_counter() - w0) / n_scalar
    agree = max(abs(r - timings["published_2^12"]["scores"][j]) for j, r in enumerate(ref))
    from audit.cryptography.ce234 import production as PR

    model = PR.load_reference_model()
    w0, c0 = time.perf_counter(), time.process_time()
    data = one_replicate(model, tables, n_full, index)
    full_wall = time.perf_counter() - w0
    full_cpu = time.process_time() - c0
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"{cls}_{k}": v for cls, dd in data.items() for k, v in dd.items()})
    del data
    spp = full_wall / n_full
    bpp = buf.getbuffer().nbytes / n_full
    rec = {"created_utc": utc(), "excluded_from_evidence": True, "environment": environment(),
           "note": "timing only; the replicate built here is discarded and no statistic is computed from it",
           "timings": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in timings.items()},
           "available_memory_bytes_at_start": mem_start,
           "published_loop_equals_2^16_average_exactly": exact, "scalar_reference_seconds_per_pair": ref_sec,
           "scalar_reference_max_abs_difference": agree,
           "full_replicate": {"real_pairs": n_full, "wall_seconds": full_wall,
                              "cpu_utilisation": full_cpu / full_wall,
                              "seconds_per_real_pair": spp, "raw_bytes_per_real_pair": bpp},
           "peak_rss_mib": round(peak_rss_mib(), 1),
           "pairs_rule_at_R60": AN.choose_pairs(spp, 60, bytes_per_real_pair=bpp)}
    write_json_new(out / "benchmark.json", rec)
    return rec


def table_variability(out: Path, n_tables: int = 5, n_eval: int = 100_000) -> dict:
    """Non-confirmatory instrument calibration for the gate tolerance: no network,
    no trail proxy and no replicate data. Each table is built from fresh pairs
    and scored on its own fresh real and random pairs; tables and scores are kept."""
    require_stage("table_variability")
    require_fresh(out / "table_variability.json", out / "variability")
    d = DS.design()
    index = A.MaskedIndex()
    n_table, rounds = int(d["baseline_D"]["table_pairs"]), int(d["baseline_D"]["table_rounds"])
    rows = []
    for i in range(n_tables):
        t0 = time.time()
        counts = A.build_table(n_table, rounds, DIFF, index=index)
        if int(counts.sum(dtype=np.uint64)) != n_table:
            raise RuntimeError("table does not hold n_table counts")
        cr = A.ciphertext_pairs(n_eval, ROUNDS, DIFF, real=True)
        cn = A.ciphertext_pairs(n_eval, ROUNDS, DIFF, real=False)
        sr = A.akr_scores(*cr, counts, n_table, index=index)
        sn = A.akr_scores(*cn, counts, n_table, index=index)
        tp, tn = int(A.akr_decide(sr).sum()), int((~A.akr_decide(sn)).sum())
        table_sha = write_npz(out / "variability" / f"table_{i}.npz", counts=counts, n_pairs=np.int64(n_table),
                              rounds=np.int64(rounds), mask=np.array(A.MASK, dtype=np.uint16))
        score_sha = write_npz(out / "variability" / f"scores_{i}.npz", real_score=sr, random_score=sn,
                              real_c=np.stack(cr), random_c=np.stack(cn))
        rows.append({"table": i, "table_sha256": table_sha, "scores_sha256": score_sha,
                     "confusion": {"tp": tp, "fn": n_eval - tp, "tn": tn, "fp": n_eval - tn},
                     "accuracy": (tp + tn) / (2 * n_eval), "tpr": tp / n_eval, "tnr": tn / n_eval,
                     "seconds": round(time.time() - t0, 1)})
        del counts, sr, sn, cr, cn
        print(f"  variability table {i + 1}/{n_tables} done", flush=True)
    sd = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in ("accuracy", "tpr", "tnr")}
    proposed = {"accuracy": max(0.005, 3 * sd["accuracy"]), "rates": max(0.010, 3 * max(sd["tpr"], sd["tnr"]))}
    rec = {"created_utc": utc(), "excluded_from_evidence": True, "randomness": "os.urandom, fresh per table and per evaluation set",
           "n_tables": n_tables, "table_pairs": n_table, "table_rounds": rounds, "n_eval_per_class": n_eval,
           "rows": rows, "sd": sd, "three_sd": {k: 3 * v for k, v in sd.items()},
           "proposed_tolerance": proposed, "rule": d["gate"]["tolerance_rule"],
           "status": "proposal; floors 0.005 and 0.010 are not approved values",
           "environment": environment(), "peak_rss_mib": round(peak_rss_mib(), 1)}
    write_json_new(out / "table_variability.json", rec)
    return rec


def calibrate(out: Path, tdir: Path, n: int, replicates: int = 10, *, binder=None, replicate=None,
              memory_probe=available_memory_bytes) -> dict:
    """Blinded: only the spread of Delta is recorded or shown. Its location and
    every correlation are discarded, so calibration cannot reveal the likely
    outcome before the margin and replicate count are fixed."""
    require_stage("calibrate")
    require_fresh(out / "calibration.json")
    d = DS.design()
    if n != int(d["replicate"]["real_pairs_per_replicate"]):
        raise SystemExit(f"calibrate refused: {n} pairs is not the registered "
                         f"{d['replicate']['real_pairs_per_replicate']}")
    require_memory(probe=memory_probe)
    validation = require_table_validation(d)
    binding, model = (binder or bind_model)()
    tables = registered_tables(tdir, d)
    index = A.MaskedIndex()
    deltas, seconds = [], []
    for i in range(replicates):
        t0 = time.time()
        data = (replicate or one_replicate)(model, tables, n, index)
        deltas.append(AN.replicate_statistics(data["real"], data["random"])["delta"])
        del data
        seconds.append(round(time.time() - t0, 1))
        print(f"  calibration {i + 1}/{replicates} done ({seconds[-1]}s)", flush=True)
    centred = [float(x - np.mean(deltas)) for x in deltas]
    plan = AN.planning_sd(deltas)
    sd_hi = plan["sigma_plan"]
    rec = {"created_utc": utc(), "excluded_from_evidence": True, "blinded": True,
           "design_sha256": DS.design_sha256(d), "environment": environment(),
           "planning": {"sims": AN.PLANNING_SIMS, "seed": 0, "draws": "numpy default_rng([0, R]).standard_normal",
                        "cell_alpha": AN.CELL_ALPHA, "simultaneous_alpha": AN.SIMULTANEOUS_ALPHA,
                        "target": 0.80, "grid": list(range(20, 61, 5))},
           "data_source": "os.urandom, fresh per replicate; arrays discarded after the blinded statistic",
           "note": ("non-confirmatory; sets the replicate count only and is never evidence; the location of "
                    "Delta and all correlations were discarded unrecorded"),
           "model_binding": binding, "table_validation": validation, "pairs_per_replicate": n,
           "seconds_per_replicate": seconds,
           "centred_deltas": centred, "planning_spread": plan,
           "replicate_rule": AN.choose_replicates(sd_hi, d["primary"]["margin"]),
           "peak_rss_mib": round(peak_rss_mib(), 1)}
    del deltas
    write_json_new(out / "calibration.json", rec)
    return rec


# ---------------------------------------------------------------- production

def source_manifest(source_root: Path = SOURCE_ROOT, crypto_root: Path = CRYPTO_ROOT) -> dict:
    """SHA-256 of every CE5 Python source file (historical/ and caches
    excluded) and of the frozen modules CE5 imports."""
    files = sorted(p for p in Path(source_root).rglob("*.py")
                   if "historical" not in p.relative_to(source_root).parts and "__pycache__" not in p.parts)
    return {"code_sha256": {str(p.relative_to(source_root)): sha256_file(p) for p in files},
            "frozen_dependencies_sha256": {f: sha256_file(Path(crypto_root) / f) for f in FROZEN_DEPENDENCIES
                                           if (Path(crypto_root) / f).exists()}}


def source_integrity(prereg: dict, *, source_root: Path = SOURCE_ROOT, crypto_root: Path = CRYPTO_ROOT) -> dict:
    """The CE5 sources and frozen dependencies on disk must equal the manifest
    registered in the preregistration: same file set, same hashes."""
    reg = prereg.get("registered_identities", {})
    want_code, want_dep = reg.get("code_sha256"), reg.get("frozen_dependencies_sha256")
    if not want_code or not want_dep:
        raise SystemExit("production refused: the preregistration registers no source manifest")
    have = source_manifest(source_root, crypto_root)
    missing = sorted(set(want_code) - set(have["code_sha256"]))
    unexpected = sorted(set(have["code_sha256"]) - set(want_code))
    changed = sorted(f for f in set(want_code) & set(have["code_sha256"]) if want_code[f] != have["code_sha256"][f])
    dep_missing = sorted(set(want_dep) - set(have["frozen_dependencies_sha256"]))
    dep_changed = sorted(f for f in want_dep if f in have["frozen_dependencies_sha256"]
                         and want_dep[f] != have["frozen_dependencies_sha256"][f])
    dep_unexpected = sorted(set(FROZEN_DEPENDENCIES) - set(want_dep))
    problems = {k: v for k, v in (("missing", missing), ("unexpected", unexpected), ("changed", changed),
                                  ("dependency_missing", dep_missing), ("dependency_changed", dep_changed),
                                  ("dependency_unregistered", dep_unexpected)) if v}
    if problems:
        raise SystemExit(f"production refused: source files differ from the registered manifest: {problems}")
    return {"decision": "PASS", "files": len(want_code), "dependencies": len(want_dep)}


def load_approved_preregistration(prereg_path: Path, prereg_sha256: str) -> dict:
    """Load exactly the approved preregistration file, identified by path and
    SHA-256; there is no default path."""
    if not prereg_path or not prereg_sha256:
        raise SystemExit("production refused: an explicit preregistration path and its approved SHA-256 are required")
    prereg_path = Path(prereg_path)
    if not prereg_path.is_file():
        raise SystemExit(f"production refused: preregistration {prereg_path} does not exist")
    actual = sha256_file(prereg_path)
    if actual != prereg_sha256.lower():
        raise SystemExit(f"production refused: preregistration SHA-256 {actual} is not the approved {prereg_sha256}")
    return DS.load_preregistration(prereg_path)


def production(out: Path, tdir: Path, token: str | None, *, prereg_path: Path | None = None,
               prereg_sha256: str | None = None, binder=bind_model, replicate=one_replicate,
               memory_probe=available_memory_bytes, source_root: Path = SOURCE_ROOT) -> dict:
    prereg = load_approved_preregistration(prereg_path, prereg_sha256)
    ok, why = DS.production_authorisation(prereg, token, prereg_path=prereg_path)
    if not ok:
        raise SystemExit(f"production refused: {why}")
    d = prereg["design"]
    if d != DS.design():
        raise SystemExit("production refused: preregistration design differs from design.py")
    integrity = source_integrity(prereg, source_root=source_root)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else None
    if man and man["design_sha256"] != DS.design_sha256(d):
        raise SystemExit("production refused: output directory belongs to a different design")
    if man and man.get("preregistration_sha256") != prereg_sha256.lower():
        raise SystemExit("production refused: output directory belongs to a different preregistration")
    avail = require_memory(probe=memory_probe)

    validation = require_table_validation(d)
    binding, model = binder()
    tables = registered_tables(tdir, d)
    index = A.MaskedIndex()
    if man is None:
        man = {"design_sha256": DS.design_sha256(d), "preregistration_sha256": sha256_file(prereg_path),
               "started_utc": utc(), "environment": environment(), "model_binding": binding,
               "available_memory_bytes": avail, "algorithm": A.ALGORITHM,
               "preregistration_path": str(prereg_path), "source_integrity": integrity,
               "source_manifest": prereg["registered_identities"],
               "ots_attestations": DS.ots_attestations(Path(str(prereg_path) + ".ots")),
               "tables": {k: v[2] for k, v in tables.items()}, "table_validation": validation, "replicates": {}}
        write_json(man_path, man)
    if "gate" not in man:
        if (out / "gate_raw.npz").exists():
            man["gate"] = recover_gate(out, tables, index, d)
        else:
            man["gate"] = run_gate(out, tables, index, d)
        write_json(man_path, man)
        print(f"  gate: accuracy {man['gate']['accuracy']:.5f} CP95 {man['gate']['cp95']} "
              f"-> {man['gate']['decision']}", flush=True)
    if man["gate"]["decision"] != "PASS":
        man["decision"] = "UNADJUDICATED"
        man["decision_reason"] = "baseline reproduction gate FAIL; no replicate run (registered rule)"
        write_json(man_path, man)
        return man

    R = int(d["replicate"]["n_replicates"])
    n = int(d["replicate"]["real_pairs_per_replicate"])
    for i in range(R):
        rid = f"rep{i:02d}"
        if man["replicates"].get(rid, {}).get("status") == "OK":
            continue
        t0 = time.time()
        try:
            data = replicate(model, tables, n, index)
        except (MemoryError, OSError) as e:
            man.setdefault("infrastructure_failures", []).append({"replicate": rid, "error": repr(e), "utc": utc()})
            write_json(man_path, man)
            raise
        raw = out / "raw" / f"{rid}.npz"
        if raw.exists():
            man.setdefault("infrastructure_failures", []).append(
                {"replicate": rid, "error": "orphan raw file from an interrupted run, kept", "utc": utc()})
            raw.rename(raw.with_suffix(f".orphan{int(time.time())}.npz"))
        digest = save_replicate(raw, data)
        st = AN.replicate_statistics(data["real"], data["random"])
        man["replicates"][rid] = {"status": "OK", "raw_file": f"raw/{rid}.npz", "raw_sha256": digest,
                                  "n_real": n, "n_random": n, "seconds": round(time.time() - t0, 1), **st}
        write_json(man_path, man)
        print(f"  {rid}: delta={st['delta']:+.4f} ({man['replicates'][rid]['seconds']}s)", flush=True)

    reps = [man["replicates"][f"rep{i:02d}"] for i in range(R)]
    man["analysis"] = AN.analyse(reps, margin=float(d["primary"]["margin"]), minimum_replicates=R,
                                 sensitivity_margins=d["primary"]["margin_sensitivity"])
    man["decision"] = man["analysis"]["decision"]
    man["finished_utc"] = utc()
    write_json(man_path, man)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ce5")
    ap.add_argument("stage", choices=["tables", "tablecheck", "benchmark", "variability", "calibrate", "production"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--table-pairs", type=int, default=10_000_000)
    ap.add_argument("--confirm")
    ap.add_argument("--prereg", type=Path, help="production: path of the approved preregistration (required)")
    ap.add_argument("--prereg-sha256", help="production: approved SHA-256 of that file (required)")
    a = ap.parse_args(argv)
    if a.stage == "tables":
        r = build_tables(a.out, a.table_pairs)
    elif a.stage == "tablecheck":
        r = validate_tables(a.out, a.tables)
        print(json.dumps({"decision": r["decision"], "components": r["components"]}, indent=2))
        return 0 if r["decision"] == "PASS" else 1
    elif a.stage == "benchmark":
        r = benchmark(a.out, a.tables)
    elif a.stage == "variability":
        r = table_variability(a.out)
    elif a.stage == "calibrate":
        r = calibrate(a.out, a.tables, int(DS.design()["replicate"]["real_pairs_per_replicate"]))
    else:
        r = production(a.out, a.tables, a.confirm, prereg_path=a.prereg, prereg_sha256=a.prereg_sha256)
    print(json.dumps({k: v for k, v in r.items() if k != "replicates"}, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
