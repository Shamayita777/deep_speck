#!/usr/bin/env python3
"""
Independent verifier for an SDB-V2 result. Imports the standard library, numpy, scipy and h5py only:
nothing from seeded_defects.py, claim_engine.py, any sdb_v2 module, or any framework component.

  python -I audit/framework_validation/sdb_v2/verify_sdb_v2.py --result <results/RUN/result.json> --repo-root .

It fails closed (NOT_VERIFIED, with every discrepancy listed) unless:
  1. the result names the registration and manifest on disk; every manifest-bound file has its hash;
     the per-case expected outcomes in the result equal the registration's;
  2. the registration and manifest equal the anchor commit's copies, the anchor is in
     origin/audit-framework, and the anchor commit precedes the run's start;
  3. every case's registered observations, rebuilt here from fixtures and raw records with this file's
     own code, equal the recorded observations, and every MATCH flag follows;
  4. B1, B1-S and B2 recompute; and every aggregate recomputes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

FROZEN_REFERENCE_SHA256 = "256eb4a5ba93414f0a46ffd498d6121a03705698ac2a81764c2375207c5d6737"
SPECK_VECTOR = ((0x1918, 0x1110, 0x0908, 0x0100), (0x6574, 0x694C), (0xA868, 0x42F2))


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --- own SPECK32/64 ---------------------------------------------------------------------------------

def _speck(alpha: int, beta: int = 2):
    M = 0xFFFF
    ror = lambda x, k: ((x >> k) | (x << (16 - k))) & M
    rol = lambda x, k: ((x << k) | (x >> (16 - k))) & M

    def rnd(x, y, k):
        x = (ror(x, alpha) + y) & M
        x ^= k
        y = rol(y, beta) ^ x
        return x, y

    def expand(key, t):
        ks, l = [key[-1]], list(reversed(key[:-1]))
        for i in range(t - 1):
            l_i, k_i = rnd(l[i], ks[i], i)
            l.append(l_i)
            ks.append(k_i)
        return ks

    def enc(p, ks):
        x, y = p
        for k in ks:
            x, y = rnd(x, y, k)
        return x, y
    return expand, enc


def _alpha_of(path: Path) -> int:
    m = re.search(r"def ALPHA\(\):\s*\n\s*return\s*\(?\s*(\d+)\s*\)?\s*;?", path.read_text())
    if not m:
        raise ValueError(f"no ALPHA definition in {path}")
    return int(m.group(1))


def m1_outcomes(path: Path) -> dict:
    a = _alpha_of(path)
    exp_r, enc_r = _speck(7)
    exp_m, enc_m = _speck(a)
    key, pt, ct = SPECK_VECTOR
    pv = "PASS" if enc_m(pt, exp_m(key, 22)) == ct else "FAIL"
    rng = np.random.default_rng(1)
    keys = [tuple(int(v) for v in rng.integers(0, 1 << 16, 4)) for _ in range(50)]
    ksch = "PASS" if all(exp_m(k, 5) == exp_r(k, 5) for k in keys) else "FAIL"
    trace = "PASS" if all(enc_m((k[0], k[1]), exp_m(k, 5)) == enc_r((k[0], k[1]), exp_r(k, 5)) for k in keys) else "FAIL"
    return {"published_vector": pv, "key_schedule": ksch, "round_traces": trace, "alpha": a}


# --- own architecture reading ------------------------------------------------------------------------

def h5_arch(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        cfg = f.attrs["model_config"]
        cfg = json.loads(cfg.decode() if isinstance(cfg, bytes) else cfg)
        names = re.findall(r'"class_name":\s*"(\w+)"', json.dumps(cfg["config"]["layers"]))
        n = [0]
        f["model_weights"].visititems(lambda _, o: n.__setitem__(0, n[0] + o.size) if isinstance(o, h5py.Dataset) else None)
    conv = names.count("Conv1D")
    return {"conv1d": conv, "add": names.count("Add"), "params": n[0], "depth": (conv - 1) // 2}


# --- own progression rule (upstream: stop at the first round whose max val_acc <= 0.505) -------------

def progression(recs: list, rounds=(5, 6, 7, 8, 9), epochs=10, threshold=0.505) -> dict:
    trained, maxes, best, stop, first = [], [], None, None, None
    for i, r in enumerate(rounds):
        block = recs[i * epochs:(i + 1) * epochs]
        if len(block) < epochs:
            raise ValueError("incomplete round")
        m = max(x["val_acc"] for x in block)
        sel = sorted(block, key=lambda x: (x["val_loss"], x["stage_epoch"]))[0]
        first = first or sel
        trained.append(r)
        maxes.append(m)
        if m > threshold:
            best = r
        else:
            stop = r
            break
    if len(recs) != len(trained) * epochs:
        raise ValueError("records beyond the stop")
    return {"best_round": best, "stop_round": stop, "rounds_trained": trained, "rounds_max_val_acc": maxes,
            "status": "stopped_at_threshold" if stop is not None else "cap_reached",
            "selected_stage_round": first["stage_round"], "selected_epoch": first["stage_epoch"]}


def criterion(fits: list, start=5) -> dict:
    pert = all(f["best_round"] is None and f["stop_round"] == start for f in fits if f["condition"] == "perturbed")
    base = all(f["best_round"] is not None for f in fits if f["condition"] == "baseline")
    return {"met": pert and base, "perturbed": pert, "baseline": base}


# --- own claim precedence (COMPLETE reproduction, levels L0-L3; inputs outside this scope are refused) ---

def decide(required, dims, L, ce, predictive=True, reproduction="COMPLETE"):
    if reproduction != "COMPLETE":
        raise ValueError("outside the verifier's decision scope")
    fails = sorted(d for d in required if dims.get(d) == "FAIL")
    ce_fail = [t["id"] for t in ce if t["required"] and t["outcome"] == "NOT_SUPPORTED"]
    if fails or ce_fail:
        return {"verdict": "NOT_SUPPORTED", "rule": "1", "blocking": fails + ce_fail}
    open_ = sorted(d for d in required if dims.get(d) in (None, "INCONCLUSIVE"))
    if open_:
        return {"verdict": "INCONCLUSIVE", "rule": "2",
                "blocking": [d if d in dims else f"{d}: missing_prerequisite" for d in open_]}
    if any(t["required"] and t["outcome"] == "INCONCLUSIVE" for t in ce):
        raise ValueError("outside the verifier's decision scope (X4)")
    level = 0 if not predictive else 2 if all(dims.get(d) == "PASS" for d in required) else 1
    if level == 2 and any(t["outcome"] == "SUPPORTED" and t["prereg"] and t["powered"] and not t["control"] for t in ce):
        level = 3
    if level < L:
        return {"verdict": "INCONCLUSIVE", "rule": "3", "level": level}
    return {"verdict": "SUPPORTED", "rule": "4", "level": level}


def fixture_claim(dim, outcome, tpl):
    dims = {d: tpl["others"] for d in ("II", "DI", "EV") if d != dim} | {dim: outcome}
    r = decide([dim], dims, tpl["L"], [])
    return {"verdict": r["verdict"], "rule": r["rule"]}


# --- own EV and D4 statistics ----------------------------------------------------------------------------

def ev_decision(f: dict) -> dict:
    valid = [u for u in f["units"] if u["status"] == "valid"]
    if len(valid) < max(f["minimum_valid"], 2):
        return {"ev_decision": "INCONCLUSIVE", "sufficient": False}
    d = np.array([u["b"] - u["a"] for u in valid])
    n, eps, a = len(d), f["epsilon"], f["alpha"]
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if se > 0 else 1.0
    tq = stats.t.ppf(1 - a / 2, n - 1)
    lo, hi = m - tq * se, m + tq * se
    p_lo = stats.t.sf((m + eps) / se, n - 1)
    p_hi = stats.t.cdf((m - eps) / se, n - 1)
    equivalent = max(p_lo, p_hi) < a
    if p < a:
        dec = "EFFECT" if (lo > eps or hi < -eps) else "INCONCLUSIVE"
    else:
        dec = "EQUIVALENT" if equivalent else "INCONCLUSIVE"
    return {"ev_decision": dec, "sufficient": True}


def d4_decision(f: dict) -> dict:
    k, a, c, mg = f["planned_replicates"], f["alpha"], f["chance"], f["margin"]
    lo_b, hi_b = c - mg, c + mg
    up = lambda x, n, p: 1.0 if x <= 0 else float(stats.binom.sf(x - 1, n, p))
    dn = lambda x, n, p: 1.0 if x >= n else float(stats.binom.cdf(x, n, p))
    states, bstates = [], []
    for r in f["rows"]:
        x, n = r["perturbed_correct"], r["test_examples"]
        eq = up(x, n, lo_b) < a and dn(x, n, hi_b) < a
        above, below = up(x, n, hi_b) < a / (2 * k), dn(x, n, lo_b) < a / (2 * k)
        states.append("EQUIVALENT" if eq else "ABOVE_BAND" if above else "BELOW_BAND" if below else "INDETERMINATE")
        xb, t = r["baseline_correct"], f["baseline_threshold"]
        bstates.append("ADEQUATE" if up(xb, n, t) < a else "INADEQUATE" if dn(xb, n, t) < a / k else "INDETERMINATE")
    primary = ("SUPPORTED" if all(s == "EQUIVALENT" for s in states) else
               "NOT_SUPPORTED" if any(s in ("ABOVE_BAND", "BELOW_BAND") for s in states) else "INCONCLUSIVE")
    base = "PASSED" if all(s == "ADEQUATE" for s in bstates) else "FAILED" if "INADEQUATE" in bstates else "INDETERMINATE"
    overall = ("NOT_SUPPORTED" if primary == "NOT_SUPPORTED" else "BASELINE_INADEQUATE" if base == "FAILED" else
               "SUPPORTED" if primary == "SUPPORTED" and base == "PASSED" else "INCONCLUSIVE")
    return {"d4_outcome": overall, "primary": primary, "baseline": base, "fit_states": states}


def d4_to_di(o):
    return {"SUPPORTED": "PASS", "NOT_SUPPORTED": "FAIL", "BASELINE_INADEQUATE": "FAIL"}.get(o, "INCONCLUSIVE")


def d2n(observed: int, null: list, alpha_per: float, ratio_gate: float) -> str:
    p = (1 + sum(1 for v in null if v >= observed)) / (len(null) + 1)
    mean = float(np.mean(null))
    ratio = math.inf if mean == 0 and observed > 0 else (1.0 if mean == 0 else observed / mean)
    stat = p <= alpha_per + 1e-12
    return "EXCESS" if stat and ratio >= ratio_gate else "INCONCLUSIVE" if stat else "NO_EXCESS"


def rows_with_neighbour(x: np.ndarray, radius: int) -> int:
    packed = np.packbits(x, axis=1).view(">u8").ravel()
    hit = np.zeros(len(packed), bool)
    for i in range(0, len(packed), 2000):
        dist = np.bitwise_count(packed[i:i + 2000, None] ^ packed[None, :])
        rows = np.arange(len(dist))
        dist[rows, i + rows] = 99                                   # a row is not its own neighbour
        hit[i:i + 2000] = (dist <= radius).any(axis=1)
    return int(hit.sum())


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sub(expected, observed):
    """Restrict `observed` to the registered keys (same shape as `expected`)."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        return {k: sub(v, observed.get(k)) for k, v in expected.items()}
    return observed


# --- the verification -------------------------------------------------------------------------------------

def verify(result_path: Path, repo: Path) -> dict:
    P: list[str] = []
    need = lambda ok, msg: ok or P.append(msg)
    res = json.loads(result_path.read_text())
    out = result_path.parent
    here = repo / "audit/framework_validation/sdb_v2"
    reg_p, man_p = here / "registration_sdb_v2.json", here / "manifest_sdb_v2.json"
    reg, man = json.loads(reg_p.read_text()), json.loads(man_p.read_text())
    need(res["registration_sha256"] == sha(reg_p), "result names another registration")
    need(res["manifest_sha256"] == sha(man_p) == reg["manifest_sha256"], "manifest hash chain broken")
    for section in ("sources", "evidence", "fixtures"):
        for path, h in man[section].items():
            p = repo / path
            need(p.is_file() and sha(p) == h, f"{section} differs: {path}")
    bound = set(man["sources"]) | set(man["evidence"]) | set(man["fixtures"])
    need(not [m for m in res["modules_loaded"] if m not in bound] and not res["modules_loaded_not_in_manifest"],
         f"modules loaded during the run but not bound: {[m for m in res['modules_loaded'] if m not in bound][:5]}")
    # 2. anchor
    a = res["anchor"]["commit"]
    git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True)
    need(git("merge-base", "--is-ancestor", a, "origin/audit-framework").returncode == 0,
         "anchor not in origin/audit-framework")
    for f in (reg_p, man_p):
        blob = git("show", f"{a}:{f.relative_to(repo)}").stdout
        need(hashlib.sha256(blob).hexdigest() == sha(f), f"{f.name} differs from the anchor commit")
    t_anchor = git("show", "-s", "--format=%cI", a).stdout.decode().strip()
    need(bool(t_anchor) and datetime.fromisoformat(t_anchor) < datetime.fromisoformat(res["started_at_utc"]),
         "anchor commit does not precede the run")

    exp = reg["expected_outcomes"]
    tpl = reg["fixture_claim_template"]
    fix = here / "fixtures"
    raw = lambda c: json.loads((out / "raw" / f"{c}.json").read_text())
    bundle = lambda c: json.loads((fix / c / "bundle.json").read_text())
    rebuilt: dict = {}

    for c in ("H1", "S2", "K3"):
        b = bundle(c)
        ar = h5_arch(repo / b["model"])
        o = {"probe_n_conv1d": ar["conv1d"], "probe_n_add": ar["add"], "probe_derived_depth": ar["depth"]}
        if sha(repo / b["model"]) != FROZEN_REFERENCE_SHA256:
            o.update({"binding": "REFUSED", "binding_refusal_kind": "sha256"})
        else:
            acc = raw(c)["binding_report"]["behavioural_binding"]["accuracy_by_rounds"]
            ok = ar["conv1d"] == 21 and ar["add"] == 10 and acc["5"] >= 0.85 and acc["6"] <= 0.6 and acc["7"] <= 0.6
            o.update({"binding": "ACCEPTED" if ok else "REFUSED", "behavioural_verdict": "PASS" if ok else "FAIL"})
        rebuilt[c] = {"C": o}
    for c in ("S1", "K4"):
        m = m1_outcomes(repo / bundle(c)["cipher_module"])
        r = {"C": {k: m[k] for k in ("published_vector", "key_schedule", "round_traces")}}
        if c == "S1":
            ii = "FAIL" if "FAIL" in r["C"].values() else "NOT_ROLLED_UP"
            r["D"] = {"II": ii}
            r["R"] = fixture_claim("II", ii, tpl)
        rebuilt[c] = r
    for c in ("S3", "K5"):
        p = np.load(repo / bundle(c)["partitions"], allow_pickle=False)
        parts = [p[k] for k in ("train", "validation", "test")]
        sets = [{r.tobytes() for r in x} for x in parts]
        dup = sum(len(x) - len(s) for x, s in zip(parts, sets))
        cross = len(sets[0] & sets[1]) + len(sets[0] & sets[2]) + len(sets[1] & sets[2])
        o = "FAIL" if dup or cross else "PASS"
        r = {"C": {"d1_outcome": o}}
        if c == "S3":
            r["D"] = {"DI": "FAIL" if o == "FAIL" else "NOT_ROLLED_UP"}
            r["R"] = fixture_claim("DI", r["D"]["DI"], tpl)
        rebuilt[c] = r
    for c in ("S4", "K6"):
        b, rw = bundle(c), raw(c)
        cfg = b["d2n"]
        rebuilt[c] = {"C": {"d2n_outcome": d2n(rw["observed_hits"], rw["null_hits"], cfg["alpha_per"], cfg["ratio_gate"])}}
        test = np.load(repo / b["partitions"], allow_pickle=False)["test"]
        exact = rows_with_neighbour(test, cfg["radius"])
        need(len(rw["null_hits"]) == cfg["null_draws"], f"{c}: null list length")
        need((exact == 0) == (rw["observed_hits"] == 0) and rw["observed_hits"] <= cfg["queries"],
             f"{c}: recorded hits inconsistent with the exact count ({exact} rows with a neighbour)")
        if c == "S4":
            need(exact >= 2 * b["planted_one_bit_near_duplicates"], "S4: planted near-duplicates not present")
    for c in ("S5", "K8"):
        f = json.loads((repo / bundle(c)["records"]).read_text())
        fits = [dict(progression(r), condition=r[0]["condition"]) for r in f["fits"]]
        cr = criterion(fits)
        pick = lambda cond, k: [x[k] for x in fits if x["condition"] == cond]
        rebuilt[c] = {"C": {"perturbed_best_rounds": pick("perturbed", "best_round"),
                            "perturbed_stop_rounds": pick("perturbed", "stop_round"),
                            "baseline_best_rounds": pick("baseline", "best_round"),
                            "baseline_stop_rounds": pick("baseline", "stop_round"), "criterion_met": cr["met"],
                            "perturbed_component": cr["perturbed"], "baseline_component": cr["baseline"]}}
    for c in ("S6", "K2", "U6"):
        r = {"C": ev_decision(json.loads((repo / bundle(c)["records"]).read_text()))}
        if c in ("S6", "U6"):
            ev = {"EFFECT": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE"}.get(r["C"]["ev_decision"], "NOT_ROLLED_UP")
            r["D"] = {"EV": ev}
            r["R"] = fixture_claim("EV", ev, tpl)
        rebuilt[c] = r
    f = json.loads((repo / bundle("U1")["records"]).read_text())
    d4 = d4_decision(f)
    rebuilt["U1"] = {"C": d4, "D": {"DI": d4_to_di(d4["d4_outcome"])},
                     "R": fixture_claim("DI", d4_to_di(d4["d4_outcome"]), tpl)}
    for c in ("K7", "U4"):
        run = repo / bundle(c)["run_dir"]
        par = json.loads((run / "parallel_execution.json").read_text())
        tampered = sha(run / "analysis.json") != par["analysis_sha256"]
        rep = raw(c)["verifier_report"]
        o = {"par1_verifier": "NOT_VERIFIED" if tampered else rep["status"]}
        if c == "U4":
            o["problem_mentions"] = "analysis.json changed after the run" if tampered else "other"
        need(tampered == (c == "U4"), f"{c}: analysis hash binding is not as registered")
        rebuilt[c] = {"C": o}
    for c in ("P1", "U2", "U3", "U5", "U7"):
        entries = json.loads((fix / c / "declared.json").read_text())["asserted_inputs"]
        outs, fields = {}, {}
        for e in entries:
            ce = []
            for t in e["ce"]:
                t = dict(t)
                for key, dest in (("outcome_from", "outcome"), ("control_from", "control")):
                    if key in t:
                        obj = json.loads((repo / t[key]["certificate"]).read_text())
                        for k in t[key]["field"]:
                            obj = obj[k]
                        fields[f"{t['id']}.{dest}"] = obj
                        if dest == "outcome":
                            t["outcome"] = obj
                        else:
                            cf = t[key]
                            t["control"] = obj == cf["control_satisfied_if"] if "control_satisfied_if" in cf else \
                                obj != cf["control_satisfied_unless"]
                ce.append(t)
            outs[e["id"]] = decide(e["required"], e["dims"], e["L"], ce, e["predictive"], e["reproduction"])
        if c == "P1":
            r = outs["P1"] | {"input_e6_ce2_admission": fields["CE2.control"]}
        elif c == "U2":
            r = outs["U2"] | {"input_a3_ce3_scientific_decision": fields["CE3.outcome"],
                              "input_a3_ce3_adversarial_outcome": fields["CE3.control"]}
        elif c == "U7":
            r = outs
        else:
            r = outs[c]
        rebuilt[c] = {"R": r}
    # H2 from the bundle
    bund = here / "evidence" / "h2_ad4"
    recs: dict = {}
    for p in sorted((bund / "fits").glob("replicate_*/*/stage_r*/*.metrics.json")):
        r = json.loads(p.read_text())
        recs.setdefault(f"{r['condition']}#{r['replicate']}", []).append(r)
    fits = {k: progression(sorted(v, key=lambda r: r["step"])) for k, v in recs.items()}
    cr = criterion([dict(v, condition=k.split("#")[0]) for k, v in fits.items()])
    cert = json.loads((repo / "audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/result/d4_certificate.json").read_text())
    eng = cert["decision"]["overall_d4_outcome"]
    autond = "PROGRESSION_NOT_CONFIRMED" if eng == "SUPPORTED" and not cr["met"] else eng
    n36 = sum(1 for k, v in fits.items() if k.startswith("perturbed") and v["best_round"] == 5 and v["stop_round"] == 6)
    claims = {proto: {k: v for k, v in decide(["II", "DI", "EV"], dims, 2, []).items() if k in ("verdict", "blocking")}
              for proto, dims in reg["a1_recorded_dimensions"].items()}
    rebuilt["H2"] = {"replay": {"fits": fits, "criterion": cr, "engine_overall_d4_outcome": eng,
                                "autond_overall_outcome": autond,
                                "baseline_adequacy_gate": cert["decision"]["baseline_adequacy_gate"],
                                "perturbed_passed_round5_and_stopped_round6": n36},
                     "D": {"DI_for_A1": d4_to_di(autond)}, "R": claims}

    # 3. compare rebuilt with recorded, and recompute MATCH flags
    cases = res["cases"]
    need(set(cases) == set(reg["case_order"]) and len(cases) == 23, "case set differs")
    match_flags = {}
    for c in reg["case_order"]:
        need(cases[c]["expected"] == exp[c], f"{c}: expected outcome in the result differs from the registration")
        flags = {}
        for lvl, e in exp[c].items():
            mine = sub(e, rebuilt[c].get(lvl))
            need(sub(e, cases[c]["observed"].get(lvl)) == mine, f"{c}/{lvl}: recorded observation differs from the rebuild")
            flags[lvl] = mine == e
            need(cases[c]["match"].get(lvl) == flags[lvl], f"{c}/{lvl}: MATCH flag differs")
        match_flags[c] = flags

    # 4. baselines
    for c in ("H1", "S2", "K3"):
        bits = np.load(out / "raw" / f"B1_{c}.npz")["correct_packbits"]
        rec = res["baselines"]["B1"][c]
        corr = np.unpackbits(bits)[:rec["n"]]
        k, n = int(corr.sum()), rec["n"]
        lo, hi = wilson(k, n)
        need(k == rec["correct"] and abs(k / n - rec["accuracy"]) < 1e-12 and abs(lo - rec["wilson95"][0]) < 1e-12
             and abs(hi - rec["wilson95"][1]) < 1e-12 and abs(rec["delta"] - (k / n - rec["reported_accuracy"])) < 1e-12,
             f"B1 {c}: does not recompute")
        need(json.loads((out / "raw" / f"B1_{c}.json").read_text())["model_sha256"] == sha(repo / bundle(c)["model"]),
             f"B1 {c}: model hash")
    y = np.load(bund / "dataset" / "test_y.npy", allow_pickle=False)
    for name, e in exp["_H2_B1S"].items():
        pr = np.load(bund / "fits" / "predictions" / name, allow_pickle=False)
        cc = (pr >= 0.5) == y
        mine = {"correct": int(cc.sum()), "test_examples": int(len(y)), "label_0_correct": int(cc[y == 0].sum()),
                "label_1_correct": int(cc[y == 1].sum())}
        need(res["baselines"]["B1S"]["observed"][name] == mine, f"B1-S {name}: does not recompute")
    b1s_match = all(res["baselines"]["B1S"]["observed"][n] == e for n, e in exp["_H2_B1S"].items())
    need(res["baselines"]["B1S"]["matches_certificate"] == b1s_match, "B1-S match flag differs")
    for c in reg["case_order"]:
        b = bundle(c)
        rec = res["baselines"]["B2"][c]["items"]
        for item, spec in b["b2"].items():
            r = rec[item]["result"]
            if spec["status"] != "E":
                need(r == ("NOT_ASSESSED" if spec["status"] == "NS" else "NOT_APPLICABLE"), f"B2 {c}/{item}: status")
                continue
            mine = None
            if item == "C1":
                mine = "PASS" if all((repo / f).is_file() for f in b["files"]) else "FAIL"
            elif item == "C3":
                mine = "PASS" if (b.get("environment") or {}).get("python") and b["environment"].get("numpy") else "FAIL"
            elif item == "C5":
                mine = m1_outcomes(repo / b["cipher_module"])["published_vector"]
            elif item == "C6":
                p = np.load(repo / b["partitions"], allow_pickle=False)
                parts = [{x.tobytes() for x in p[k]} for k in p.files if not k.endswith("_y")]
                mine = "FAIL" if any(parts[i] & parts[j] for i in range(len(parts)) for j in range(i + 1, len(parts))) else "PASS"
            elif item == "C7":
                bad = [k for k, h in b["published_checksums"].items() if sha(repo / k) != h]
                mine = "FAIL" if bad else "PASS"
            elif item == "C8":
                ar, pub = h5_arch(repo / b["model"]), b["published_architecture"]
                mine = "PASS" if all(ar[k] == pub[k] for k in ("conv1d", "add", "params")) else "FAIL"
            if mine is not None:
                need(r == mine, f"B2 {c}/{item}: recorded {r}, rebuilt {mine}")
        rs = [v["result"] for v in rec.values()]
        o = ("FLAGGED" if "FAIL" in rs else "NO_ISSUE_FOUND" if "PASS" in rs and all(x in ("PASS", "NOT_APPLICABLE") for x in rs)
             else "NOT_ASSESSED")
        need(res["baselines"]["B2"][c]["output"] == o, f"B2 {c}: output")

    # 5. aggregates
    groups = reg["case_groups"]
    det = [c for c in reg["case_order"] if groups[c] in ("historical_defect", "synthetic_defect")]
    clean = [c for c in reg["case_order"] if groups[c] == "clean_control"]
    abst = [c for c in reg["case_order"] if groups[c] == "abstention_control"]
    allm = lambda c: bool(match_flags[c]) and all(match_flags[c].values())
    m = res["metrics"]
    need(m["framework"]["detection"]["detected"] == [c for c in det if match_flags[c].get("C")], "detection aggregate")
    need(m["framework"]["false_alarms"]["false_alarms"] == [c for c in clean if not match_flags[c].get("C")], "false-alarm aggregate")
    need(m["framework"]["abstention"]["correct"] == [c for c in abst if allm(c)], "abstention aggregate")
    need(m["framework"]["positive_control"]["P1"] == allm("P1"), "positive-control aggregate")
    need(m["framework"]["h2_replay"] == {k: match_flags["H2"].get(k) for k in ("replay", "D", "R")}, "H2 aggregate")
    b2o = {c: res["baselines"]["B2"][c]["output"] for c in reg["case_order"]}
    need(m["B2"]["detection"]["flagged"] == [c for c in det if b2o[c] == "FLAGGED"], "B2 detection aggregate")
    need(m["B2"]["false_alarms"]["flagged"] == [c for c in clean if b2o[c] == "FLAGGED"], "B2 false-alarm aggregate")
    st = res["baselines"]["B1_status"]
    need(m["B1"]["status_counts"] == {s: sum(1 for v in st.values() if v == s)
                                      for s in ("EXECUTED", "B1-S", "NOT_ASSESSED", "NOT_APPLICABLE")}, "B1 counts")
    need(st == reg["b1_status"], "B1 status differs from the registration")
    return {"status": "VERIFIED" if not P else "NOT_VERIFIED", "problems": P,
            "result_sha256": sha(result_path), "registration_sha256": sha(reg_p), "manifest_sha256": sha(man_p),
            "anchor": a, "anchor_committed_at": t_anchor, "match_flags": match_flags,
            "aggregates_recomputed": {"detected": [c for c in det if match_flags[c].get("C")],
                                      "false_alarms": [c for c in clean if not match_flags[c].get("C")],
                                      "abstention_correct": [c for c in abst if allm(c)], "P1": allm("P1"),
                                      "H2": match_flags["H2"], "B2_flagged_defects": [c for c in det if b2o[c] == "FLAGGED"],
                                      "B2_flagged_clean": [c for c in clean if b2o[c] == "FLAGGED"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rep = verify(a.result.resolve(), a.repo_root.resolve())
    text = json.dumps(rep, indent=1, sort_keys=True, default=str)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if rep["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
