#!/usr/bin/env python3
"""
Independent verifier for II-4-GOHR-IMPL-V1-PAR1 run directories.

Imports nothing from the audit package: standard library plus scipy.stats for the
t distribution. From the raw run files it reconstructs the expected (block, arm)
task matrix and seeds, checks every identity, checkpoint, journal, epoch log,
claim and firewall record, recomputes the paired differences, the t-based
intervals, the materiality rule, TOST and the exact sign-flip test, and compares
them with analysis.json. It fails closed: VERIFIED only if every check passes.

    python -I audit/implementation/ii4_parallel/verify.py --run-dir <dir> \
        [--frozen audit/implementation/preregistrations/ii-4-gohr-impl-v1-par1.frozen.json] \
        [--repo-root .] [--out report.json]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

#: The amendment's own files, listed here independently of the scheduler module.
AMENDMENT_FILES = (
    "audit/implementation/ii4_parallel/__init__.py",
    "audit/implementation/ii4_parallel/scheduler.py",
    "audit/implementation/ii4_parallel/worker.py",
    "audit/implementation/ii4_parallel/launcher.py",
    "audit/implementation/ii4_parallel/verify.py",
    "audit/implementation/ii4_parallel/tests/__init__.py",
    "audit/implementation/ii4_parallel/tests/test_ii4_parallel.py",
    "audit/implementation/ii4_parallel/tests/ii4_parallel_fixtures.py",
)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isinf(a):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def recompute(diffs, *, delta, alpha, min_valid, n_requested):
    from scipy import stats
    n = len(diffs)
    out = {"n": n, "sufficient": n >= min_valid}
    if not out["sufficient"]:
        out.update(materiality="INSUFFICIENT_BLOCKS", equivalence="NOT_ASSESSED")
        return out
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    out.update(mean=mean, sd=sd, se=se, df=df)
    if se == 0.0:
        out.update(materiality="NO_DIRECTIONAL_CLAIM", equivalence="INCONCLUSIVE")
        return out
    t = mean / se
    t95, t90 = stats.t.ppf(1 - alpha / 2, df), stats.t.ppf(1 - alpha, df)
    lo95, hi95 = mean - t95 * se, mean + t95 * se
    out.update(t=t, p=float(2 * (1 - stats.t.cdf(abs(t), df))), ci95=(lo95, hi95),
               ci90=(mean - t90 * se, mean + t90 * se))
    out["materiality"] = ("MATERIALLY_HIGHER" if lo95 > delta else "MATERIALLY_LOWER" if hi95 < -delta
                          else "NO_DIRECTIONAL_CLAIM")
    p_lo = float(1 - stats.t.cdf((mean + delta) / se, df))
    p_hi = float(stats.t.cdf((mean - delta) / se, df))
    out.update(tost_p_lower=p_lo, tost_p_upper=p_hi, tost_p=max(p_lo, p_hi))
    out["equivalence"] = ("EQUIVALENT_WITHIN_MARGIN" if max(p_lo, p_hi) < alpha
                          else "NOT_EQUIVALENT" if out["materiality"] != "NO_DIRECTIONAL_CLAIM"
                          else "INCONCLUSIVE")
    if 2 / 2 ** n < alpha:
        obs = abs(mean)
        hits = sum(1 for s in itertools.product((1, -1), repeat=n)
                   if abs(sum(d * x for d, x in zip(diffs, s)) / n) >= obs - 1e-15)
        out["sign_flip_p"] = hits / 2 ** n
    else:
        out["sign_flip_p"] = None
    return out


def _ts(s):
    from datetime import datetime
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def epoch_provenance(arm_dir: Path, bid: str, arm: str, epochs: int, claims: dict, sessions: dict, need):
    """Attribute every epoch row to a session ledger entry, the block's claim and a GPU.

    Legitimate hard-kill artefacts are accepted and nothing else: a torn line only from a session
    that never wrote its end entry, and a COMMITTED_METRICS_LOST marker in place of a lost row.
    Returns per-epoch (gpu, start, end) intervals when the rows carry timestamps.
    """
    ledger = [json.loads(l) for l in (arm_dir / "worker_sessions.jsonl").read_text().splitlines() if l.strip()] \
        if (arm_dir / "worker_sessions.jsonl").is_file() else []
    need(ledger, f"{bid}/{arm}: no worker ledger")
    starts = {e["session_id"]: e for e in ledger if e.get("event") == "arm_session_start"}
    ends = {e["session_id"]: e for e in ledger if e.get("event") == "arm_session_end"}
    need(len(starts) == sum(e.get("event") == "arm_session_start" for e in ledger), f"{bid}/{arm}: two ledger starts in one session")
    for s, e in starts.items():
        c = claims.get(s)
        need(e.get("block_id") == bid and e.get("arm") == arm, f"{bid}/{arm}: ledger entry names another block or arm")
        need(c is not None and c.get("gpu") == e.get("gpu") and s in sessions and e.get("gpu") in sessions[s]["gpus"]
             and str(e.get("cuda_visible_devices")) == str(e.get("gpu")),
             f"{bid}/{arm}: session {s} has no matching claim, recorded session or device")
        if c is not None:
            need(e.get("claim_sha256") == c["_sha256"], f"{bid}/{arm}: ledger names another claim file in session {s}")
    rows, torn = [], 0
    for line in (arm_dir / "epochs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            torn += 1
    need(torn <= len(set(starts) - set(ends)), f"{bid}/{arm}: {torn} unreadable epoch line(s) not explained by a killed session")
    seen, intervals, timed = {}, [], True
    for r in rows:
        e, s = r.get("epoch"), r.get("session_id")
        if not isinstance(e, int) or not 0 <= e < epochs:
            need(False, f"{bid}/{arm}: epoch row outside 0..{epochs - 1}: {e}")
            continue
        if r.get("status") not in ("COMPLETED", "COMMITTED_METRICS_LOST"):
            need(False, f"{bid}/{arm}: epoch {e} has status {r.get('status')}")
            continue
        need(e not in seen, f"{bid}/{arm}: duplicate record for epoch {e}")
        seen[e] = r
        st = starts.get(s)
        need(st is not None, f"{bid}/{arm}: epoch {e} attributed to session {s}, which no worker ledger records")
        if st is None:
            continue
        lo, hi = st.get("epochs_before", 0), (ends.get(s) or {}).get("epochs_after", epochs)
        need(lo <= e < hi, f"{bid}/{arm}: epoch {e} lies outside session {s}'s range [{lo}, {hi})")
        if r.get("status") == "COMPLETED" and r.get("timestamp_utc") and isinstance(r.get("epoch_seconds"), (int, float)):
            end = _ts(r["timestamp_utc"])
            intervals.append((st["gpu"], end - float(r["epoch_seconds"]), end))
        elif r.get("status") == "COMPLETED":
            timed = False
    need(sorted(seen) == list(range(epochs)), f"{bid}/{arm}: epochs {sorted(set(range(epochs)) - set(seen))[:6]} missing")
    return intervals, timed, sorted({e["gpu"] for e in starts.values()})


def overlap_seconds(intervals) -> float:
    """Total time during which work on two different GPUs ran at once."""
    total = 0.0
    for (g1, s1, e1), (g2, s2, e2) in itertools.combinations(intervals, 2):
        if g1 != g2:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def verify(run_dir: Path, frozen_path: Path | None = None, repo_root: Path | None = None,
           require_dual_gpu: bool | None = None) -> dict:
    run_dir = Path(run_dir)
    P: list[str] = []
    need = lambda ok, msg: ok or P.append(msg)
    load = lambda name: json.loads((run_dir / name).read_text())
    for name in ("preregistration.json", "state.json", "analysis.json", "firewall.json",
                 "parallel_execution.json"):
        need((run_dir / name).is_file(), f"{name} missing")
    if P:
        return {"status": "NOT_VERIFIED", "problems": P}
    pre, st, an, fw, par = (load(n) for n in ("preregistration.json", "state.json", "analysis.json",
                                              "firewall.json", "parallel_execution.json"))
    fp = st.get("experiment_fingerprint")
    need(fp and fp == an.get("experiment_fingerprint") == pre.get("experiment_fingerprint")
         == par.get("experiment_fingerprint"), "experiment fingerprints disagree")
    need(an.get("source_manifest_sha256") == pre.get("source_manifest_sha256")
         == par.get("source_manifest_sha256"), "source manifest hashes disagree")
    need(par.get("analysis_sha256") == sha(run_dir / "analysis.json"), "analysis.json changed after the run")
    srcs = {r["path"]: r["sha256"] for r in (an.get("source_manifest") or {}).values()}
    need(all(f in srcs for f in AMENDMENT_FILES), "amendment files missing from the source manifest")
    if frozen_path:
        fr = json.loads(Path(frozen_path).read_text())
        need({r["path"]: r["sha256"] for r in fr["source_manifest"].values()} == srcs,
             "source manifest differs from the amendment's frozen record")
        need(fr.get("config_fingerprint") == an.get("config_fingerprint"), "config fingerprint differs from frozen")
        need(fr.get("preregistration_sha256") == an.get("preregistration_sha256"), "preregistration differs from frozen")
    if repo_root:
        for path, h in srcs.items():
            p = Path(repo_root) / path
            need(p.is_file() and sha(p) == h, f"source on disk differs: {path}")

    d = pre["design"]
    n_blocks, epochs = int(d["n_blocks"]), int(d["epochs"])
    delta, alpha = float(d["delta_materiality"]), float(d["alpha"])
    sm = pre["seed_manifest"]
    arms = list(sm["arm_ids"])
    a_arm, b_arm = arms
    seed = lambda i, arm: int(sm["base_seed"]) + len(arms) * i + arms.index(arm)
    need(all(sm["seeds"][f"block{i}"][arm] == seed(i, arm) for i in range(int(sm["n_blocks"])) for arm in arms),
         "registered seed manifest does not follow base + 2*block + arm")

    reps = st.get("replacements") or []
    expected = [f"block{i}" for i in range(n_blocks)] + [r["replacement"] for r in reps]
    need(len(reps) <= int(d["max_replacement_blocks"]), "replacement budget exceeded")
    need(st.get("block_order") == expected, f"block order {st.get('block_order')} != expected {expected}")
    need([b["block_id"] for b in an["blocks"]] == expected, "analysis blocks differ from the task matrix")

    test_hash = (st.get("test_set") or {}).get("hash")
    need(test_hash and set(fw.get("sealed_dataset_hashes", {}).values()) == {test_hash}, "sealed set hash mismatch")
    need(fw.get("frozen") and (fw.get("freeze_record") or {}).get("frozen_config_hash") == fp,
         "firewall not frozen for this run")
    consumed_list = list(fw.get("consumed_evaluation_keys") or [])
    need(len(consumed_list) == len(set(consumed_list)),
         f"duplicate firewall consumption(s): {sorted({k for k in consumed_list if consumed_list.count(k) > 1})}")
    consumed, terminal_hashes, diffs = set(consumed_list), {}, []
    expected_consumed = set()
    sessions = {s["session_id"]: s for s in st.get("parallel_sessions") or []}
    planned = [f"block{k}" for k in range(n_blocks)]
    epoch_iv, arm_iv, timed_all, gpus_seen = [], [], True, set()
    for blk_out in an["blocks"]:
        bid = blk_out["block_id"]
        i = int(bid[5:])
        blk = st["blocks"][bid]
        need(blk.get("block_index") == i, f"{bid}: index")
        vals = {}
        cdir = run_dir / "blocks" / bid / "claims"
        claims = {}
        for cf in (sorted(cdir.glob("*.json")) if cdir.is_dir() else []):
            c = json.loads(cf.read_text())
            c["_sha256"] = sha(cf)
            need(c.get("block_id") == bid and cf.stem == c.get("session_id") and c.get("session_id") in sessions,
                 f"{bid}: claim {cf.name} names another block or an unrecorded session")
            need(c.get("session_id") not in claims, f"{bid}: two claims in one session")
            claims[c.get("session_id")] = c
        for arm in arms:
            rec = blk["arms"].get(arm) or {}
            arm_dir = run_dir / "blocks" / bid / arm
            need(rec.get("seed") == seed(i, arm), f"{bid}/{arm}: seed {rec.get('seed')} != {seed(i, arm)}")
            if rec.get("status") != "EVALUATED":
                continue
            need(Path(rec["arm_dir"]).resolve() == arm_dir.resolve(), f"{bid}/{arm}: arm_dir points elsewhere")
            tm = rec["terminal_model"]
            need(Path(tm["path"]).resolve().parent == arm_dir.resolve() and sha(tm["path"]) == tm["hash"],
                 f"{bid}/{arm}: terminal model missing, misplaced or altered")
            need(tm["hash"] not in terminal_hashes, f"{bid}/{arm}: terminal model shared with {terminal_hashes.get(tm['hash'])}")
            terminal_hashes[tm["hash"]] = f"{bid}/{arm}"
            j = json.loads((arm_dir / "eval_journal.json").read_text())
            need(j["arm_id"] == arm and j["block_id"] == bid and j["sealed_test_hash"] == test_hash
                 and j["terminal_model_hash"] == tm["hash"] and j["state"] in ("RECORDED", "CONSUMED")
                 and close(float(j["accuracy"]), float(rec["sealed_test_accuracy"]), 0),
                 f"{bid}/{arm}: evaluation journal disagrees with the state")
            c = json.loads((arm_dir / "commit.json").read_text())
            need(c["arm_id"] == arm and c["block_id"] == bid and c["committed_epoch"] == epochs - 1
                 and c["epochs_total"] == epochs, f"{bid}/{arm}: commit manifest not at the final epoch")
            g = rec.get("gpu_assignment")
            if bid in planned:
                need(g is not None and g.get("session_id") in sessions and g.get("gpu") in sessions[g["session_id"]]["gpus"],
                     f"{bid}/{arm}: no valid GPU assignment from a recorded parallel session")
                iv, timed, gs = epoch_provenance(arm_dir, bid, arm, epochs, claims, sessions, need)
                epoch_iv += iv
                timed_all = timed_all and timed
                gpus_seen.update(gs)
                if g and rec.get("started_at_utc") and rec.get("trained_at_utc"):
                    arm_iv.append((g.get("gpu"), _ts(rec["started_at_utc"]), _ts(rec["trained_at_utc"])))
            else:
                completed = [json.loads(l).get("epoch") for l in (arm_dir / "epochs.jsonl").read_text().splitlines()
                             if l.strip() and l.strip().startswith("{")]
                need(sorted(completed) == list(range(epochs)), f"{bid}/{arm}: replacement-block epochs incomplete")
            vals[arm] = float(rec["sealed_test_accuracy"])
            expected_consumed.add(f"{arm}:{bid}")
        if bid in [f"block{k}" for k in range(n_blocks)]:
            done = run_dir / "blocks" / bid / "training_complete.json"
            need(done.is_file(), f"{bid}: training completion record missing")
            if done.is_file():
                tc = json.loads(done.read_text())
                need(tc.get("experiment_fingerprint") == fp and tc.get("block_id") == bid, f"{bid}: completion identity")
                for arm in arms:
                    rec = blk["arms"].get(arm) or {}
                    need((tc["arms"][arm]["terminal_model"] or {}).get("hash") == (rec.get("terminal_model") or {}).get("hash"),
                         f"{bid}/{arm}: completion record names another terminal model")
                    need(tc["arms"][arm].get("gpu_assignment") == rec.get("gpu_assignment"),
                         f"{bid}/{arm}: completion record names another GPU assignment")
            # One worker per block per session. An arm resumed in a later session may
            # continue on the other GPU; every epoch is attributed through its own session.
            for arm in arms:
                g = (blk["arms"].get(arm) or {}).get("gpu_assignment") or {}
                owner = claims.get(g.get("session_id"))
                need(owner is not None and owner.get("gpu") == g.get("gpu"),
                     f"{bid}/{arm}: trained by a worker that did not hold the block's claim in that session")
        valid = set(vals) == set(arms)
        need(blk_out.get("is_valid") == valid, f"{bid}: validity differs from the analysis")
        if valid:
            dlt = vals[a_arm] - vals[b_arm]
            need(close(blk_out.get("signed_difference"), dlt, 0), f"{bid}: signed difference")
            diffs.append(dlt)
    need(consumed == expected_consumed, f"firewall consumptions {sorted(consumed ^ expected_consumed)} differ")

    r = recompute(diffs, delta=delta, alpha=alpha, min_valid=int(d["min_valid_blocks"]), n_requested=n_blocks)
    A = an["analysis"]
    conf = A.get("confirmatory") or {}
    need(A["replication"]["n_blocks_valid"] == r["n"] and A["replication"]["sufficient"] == r["sufficient"],
         "replication summary differs")
    if r["sufficient"] and "t" in r:
        need(close(conf["mean_difference"], r["mean"]) and close(conf["sd_difference"], r["sd"]), "mean/sd differ")
        need(close(conf["ci95"]["low"], r["ci95"][0]) and close(conf["ci95"]["high"], r["ci95"][1]), "95% CI differs")
        need(close(conf["ci90"]["low"], r["ci90"][0]) and close(conf["ci90"]["high"], r["ci90"][1]), "90% CI differs")
        need(close(conf["paired_t"]["p_value"], r["p"], 1e-7), "t-test p differs")
        eq = conf["equivalence"]
        need(close(eq["p_lower"], r["tost_p_lower"], 1e-7) and close(eq["p_upper"], r["tost_p_upper"], 1e-7),
             "TOST p-values differ")
        need(eq["verdict"] == r["equivalence"], f"equivalence verdict {eq['verdict']} != {r['equivalence']}")
        need(conf["materiality"]["verdict"] == r["materiality"], "materiality verdict differs")
        need(close(eq["margin"], delta, 0), "margin differs from the design")
        sf = (A.get("secondary_robustness") or {}).get("sign_flip_p")
        need(close(sf, r["sign_flip_p"]), "sign-flip p differs")

    production = pre.get("execution_mode") == "production"
    if require_dual_gpu is None:
        require_dual_gpu = production
    basis = "epoch timestamps" if (timed_all and epoch_iv) else "arm start and end times"
    overlap = overlap_seconds(epoch_iv if basis == "epoch timestamps" else arm_iv)
    if require_dual_gpu:
        need(len(gpus_seen) >= 2, f"dual-GPU execution required but training used GPU(s) {sorted(gpus_seen)}")
        need(overlap > 0, "dual-GPU execution required but no training on two GPUs overlapped in time")
        if production:
            need(basis == "epoch timestamps", "production overlap must be established from per-epoch timestamps")
    return {"status": "VERIFIED" if not P else "NOT_VERIFIED", "run_dir": str(run_dir),
            "experiment_fingerprint": fp, "blocks": expected, "valid_blocks": len(diffs),
            "gpus_used": sorted(gpus_seen), "dual_gpu_required": bool(require_dual_gpu),
            "overlap_basis": basis, "cross_gpu_training_overlap_seconds": round(overlap, 3),
            "recomputed": {k: v for k, v in r.items()}, "problems": P}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent verifier for II-4-GOHR-IMPL-V1-PAR1 (fails closed).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--frozen", type=Path)
    ap.add_argument("--repo-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--require-dual-gpu", action="store_true",
                    help="require two GPUs with overlapping training (always required for production runs)")
    a = ap.parse_args(argv)
    rep = verify(a.run_dir, a.frozen, a.repo_root, True if a.require_dual_gpu else None)
    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        with open(a.out, "x") as f:
            f.write(text + "\n")
    print(text)
    return 0 if rep["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
