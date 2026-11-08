"""
Independent recomputation of a G-CE5 production directory.

Imports the registered design (the specification) and reference.py (the
independent scalar implementation) only; never akr.py, analysis.py, runner.py
or the gohr package. It reads raw files and tables, not the analysis summary,
and compares what it recomputes with what the manifest records.

    python -m audit.cryptography.experiments.ce5.verify OUT --tables T --prereg PATH [--model]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.stats import beta as beta_dist
from scipy.stats import norm, rankdata

from audit.cryptography.experiments.ce5 import design as DS
from audit.cryptography.experiments.ce5 import reference as REF

SUBSAMPLE = 32
N_RECOMPUTE = 256
MASK = (0xFF8F, 0xFF8F)
TABLES = (("addt_r4_counts.npz", "baseline_D", 4), ("addt_r5_counts.npz", "secondary_comparator_LR", 5))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rho(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def _matched(D, T) -> np.ndarray:
    tie_ranks = sorted(rankdata(T))
    out = np.empty(len(D))
    for pos, i in enumerate(sorted(range(len(D)), key=lambda j: (D[j], j))):
        out[i] = tie_ranks[pos]
    return out


def _nscore(x) -> np.ndarray:
    return norm.ppf(rankdata(x) / (len(x) + 1))


def _poly(*cols) -> np.ndarray:
    return np.column_stack([np.ones(len(cols[0]))] + [c ** p for c in cols for p in (1, 2, 3)])


def _incremental(N, T, D) -> tuple[float, float]:
    h = len(N) // 2
    fit = {k: _nscore(v[:h]) for k, v in (("N", N), ("T", T), ("D", D))}
    ev = {k: _nscore(v[h:2 * h]) for k, v in (("N", N), ("T", T), ("D", D))}

    def X(q, m):
        if m == "T":
            return _poly(q["T"])
        if m == "D":
            return _poly(q["D"])
        return np.column_stack([_poly(q["T"], q["D"]), q["T"] * q["D"]])

    loss = {}
    for m in ("T", "D", "TD"):
        b = linalg.lstsq(X(fit, m), fit["N"])[0]
        loss[m] = float(np.mean((ev["N"] - X(ev, m) @ b) ** 2))
    return loss["T"] - loss["TD"], loss["D"] - loss["TD"]


def _interval(values):
    v = sorted(values)
    R = len(v)
    k = 0
    while k + 1 <= R and sum(comb(R, j) for j in range(k + 1)) / 2 ** R <= 0.025:
        k += 1
    return (v[k - 1], v[R - k]) if k else (None, None)


def _decide(lo, hi, m):
    if lo is None:
        return "INCONCLUSIVE"
    if -m < lo and hi < m:
        return "EQUIVALENT"
    if lo >= m:
        return "DISTRIBUTION_GREATER"
    if hi <= -m:
        return "TRAIL_GREATER"
    return "INCONCLUSIVE"


def _qualifier(lo, hi, m):
    if _decide(lo, hi, m) != "INCONCLUSIVE" or lo is None:
        return None
    return "DIRECTION_D" if lo > 0 else "DIRECTION_T" if hi < 0 else "UNDETERMINED"


def _close(a, b, tol=1e-12) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def _cp(k: int, n: int) -> tuple[float, float]:
    lo = float(beta_dist.ppf(0.025, k, n - k + 1)) if k else 0.0
    hi = float(beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def _sign_p(values) -> float | None:
    pos = sum(1 for x in values if x > 0)
    n = pos + sum(1 for x in values if x < 0)
    if n == 0:
        return None
    tail = sum(comb(n, j) for j in range(min(pos, n - pos) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _holm(p: dict) -> dict:
    items = sorted(((k, v) for k, v in p.items() if v is not None), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def _bits(words) -> np.ndarray:
    return np.stack([(np.asarray(w, dtype=np.uint32) >> s) & 1 for w in words for s in range(15, -1, -1)],
                    axis=1).astype(np.uint8)


def verify(out: Path, tdir: Path, *, seed: int = 0, model_check: bool = False,
           table_sample: int = 100_000, subsample: int = SUBSAMPLE, prereg_path: Path | None = None,
           checkpoint: Path | None = None, source_root: Path | None = None,
           amendment_path: Path | None = None) -> dict:
    man = json.loads((out / "manifest.json").read_text())
    d = DS.design()
    problems, checks = [], {}

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    need(man.get("design_sha256") == DS.design_sha256(d), "manifest design hash differs from design.py")
    if prereg_path is None:
        raise ValueError("an explicit preregistration path is required")
    prereg = Path(prereg_path)
    need(prereg.exists() and _sha(prereg) == man.get("preregistration_sha256"), "preregistration hash mismatch")
    need(man.get("model_binding", {}).get("sha256") == DS.REFERENCE_CHECKPOINT_SHA256, "model is not the registered checkpoint")
    if prereg.exists():
        pre = json.loads(prereg.read_text())
        need(pre.get("design") == d, "preregistered design differs from design.py")
        need(pre.get("approval", {}).get("status") == "APPROVED", "preregistration is not APPROVED")
        need(pre.get("approval", {}).get("design_sha256") == DS.design_sha256(d), "approval hash is not the design hash")
        proof = Path(str(prereg) + ".ots")
        need(DS.ots_digest(proof) == _sha(prereg), "timestamp proof does not commit to the preregistration")
        need(DS.ots_attestations(proof)["bitcoin"] >= 1, "timestamp proof has no Bitcoin attestation")
        amendment = None
        if amendment_path is None:
            need(man.get("amendment_sha256") is None, "manifest names an amendment but none was supplied")
            ok_ots, why_ots = DS.ots_verification_ok(Path(str(proof) + ".verification.json"), prereg)
        else:   # amendment 01: the Java OTS record replaces the ots verify record
            asha = _sha(amendment_path)
            amendment = json.loads(Path(amendment_path).read_text())
            need(man.get("amendment_sha256") == asha, "manifest was not made under this amendment")
            ok_ots, why_ots = DS.amendment_ok(amendment, prereg)
            if ok_ots:
                ok_ots, why_ots = DS.java_ots_verification_ok(Path(str(proof) + DS.JAVA_EVIDENCE_SUFFIX), prereg, asha)
        need(ok_ots, f"ots verify record: {why_ots}")
    need(man.get("tables") == {"akr": d["baseline_D"]["table_sha256"], "lr": d["secondary_comparator_LR"]["table_sha256"]},
         "manifest tables are not the registered tables")
    if prereg.exists():
        reg = (amendment or json.loads(prereg.read_text())).get("registered_identities", {})
        src = Path(source_root or Path(__file__).resolve().parent)
        on_disk = {str(p.relative_to(src)): _sha(p) for p in sorted(src.rglob("*.py"))
                   if "historical" not in p.relative_to(src).parts and "__pycache__" not in p.parts}
        need(bool(reg.get("code_sha256")) and on_disk == reg.get("code_sha256"),
             "CE5 sources on disk differ from the registered manifest")
        crypto = src.parents[1]
        need(bool(reg.get("frozen_dependencies_sha256")) and all(
            (crypto / f).exists() and _sha(crypto / f) == h for f, h in reg.get("frozen_dependencies_sha256", {}).items()),
            "frozen dependencies differ from the registered manifest")
        need(man.get("source_integrity", {}).get("decision") == "PASS", "manifest has no source-integrity PASS")

    tables = {}
    for fn, key, rounds in TABLES:
        digest = _sha(tdir / fn)
        need(digest == d[key]["table_sha256"], f"{fn} hash {digest} is not registered")
        with np.load(tdir / fn) as z:
            t, n = z["counts"], int(z["n_pairs"])
            need(int(z["rounds"]) == rounds, f"{fn} rounds field is not {rounds}")
        need(int(t.sum(dtype=np.uint64)) == n, f"{fn} counts do not sum to n_pairs")
        tc = REF.table_check(t, n, rounds=rounds, diff=(0x0040, 0), mask=MASK, n_sample=table_sample)
        checks[f"table_check_{fn}"] = tc
        need(tc["p_value"] >= 1e-3, f"{fn} fails the independent table check (p = {tc['p_value']:.3g})")
        tables[key] = (t, n)
    t4, n4 = tables["baseline_D"]

    rng = np.random.default_rng(seed)
    ref_checks = 0

    def check_scores(c, scores, label):
        nonlocal ref_checks
        for j in rng.choice(len(scores), size=min(subsample, len(scores)), replace=False):
            ref = REF.score((int(c[0][j]), int(c[1][j])), (int(c[2][j]), int(c[3][j])), t4, n4, MASK)
            ref_checks += 1
            need(abs(ref - float(scores[j])) <= 1e-12, f"{label} pair {j}: D {scores[j]} != reference {ref}")

    g = man.get("gate", {})
    gp = out / "gate_raw.npz"
    need(gp.exists() and _sha(gp) == g.get("raw_sha256"), "gate raw file missing or hash mismatch")
    gate_decision = None
    if gp.exists():
        with np.load(gp) as z:
            sr, sn, crr, cnn = z["real_score"], z["random_score"], z["real_c"], z["random_c"]
        half = int(d["gate"]["pairs_per_class"])
        need(sr.shape == (half,) and sn.shape == (half,) and crr.shape == (4, half) and cnn.shape == (4, half),
             f"gate arrays do not have the registered {half} pairs per class")
        need(crr.dtype == np.uint16 and cnn.dtype == np.uint16 and sr.dtype == np.float64 and sn.dtype == np.float64,
             "gate array dtypes differ from the registered layout")
        need(bool(np.isfinite(sr).all() and np.isfinite(sn).all()), "gate scores are not all finite")
        check_scores(crr, sr, "gate real")
        check_scores(cnn, sn, "gate random")
        tp = sum(REF.decide(float(s), MASK) for s in sr)
        tn = sum(not REF.decide(float(s), MASK) for s in sn)
        gd = d["gate"]
        a = float(gd["published_akr"]["accuracy"])
        tau, tau_r = float(gd["tolerance"]["accuracy"]), float(gd["tolerance"]["rates"])
        acc = _cp(tp + tn, len(sr) + len(sn))
        tpr, tnr = _cp(tp, len(sr)), _cp(tn, len(sn))

        def within(ci, centre, t):
            return centre - t <= ci[0] and ci[1] <= centre + t
        reproduced = (within(acc, a, tau) and within(tpr, float(gd["published_akr"]["tpr"]), tau_r)
                      and within(tnr, float(gd["published_akr"]["tnr"]), tau_r))
        gate_decision = "PASS" if reproduced else "FAIL"
        need(g.get("confusion") == {"tp": tp, "fn": len(sr) - tp, "tn": tn, "fp": len(sn) - tn},
             "gate confusion matrix does not recompute")
        need(g.get("separated") == (acc[0] > (a + float(gd["published_ddt_accuracy"])) / 2),
             "gate separation does not recompute")
        need(g.get("decision") == gate_decision, f"gate decision recomputes to {gate_decision}")
        n_g = len(sr) + len(sn)
        for key, val in (("accuracy", (tp + tn) / n_g), ("tpr", tp / len(sr)), ("tnr", tn / len(sn))):
            need(_close(g.get(key), val), f"gate {key} does not recompute")
        need(g.get("n") == n_g, "gate sample count does not recompute")
        need(g.get("reproduced") == reproduced, "gate reproduction component does not recompute")
        cp = g.get("cp95", {})
        for key, ci in (("accuracy", acc), ("tpr", tpr), ("tnr", tnr)):
            need(len(cp.get(key, [])) == 2 and all(_close(x, y) for x, y in zip(cp.get(key, []), ci)),
                 f"gate {key} interval does not recompute")

    model = None
    if model_check:
        import keras
        ck = Path(checkpoint or Path(__file__).resolve().parents[2] / d["model"]["checkpoint"])
        need(_sha(ck) == DS.REFERENCE_CHECKPOINT_SHA256, "checkpoint file hash mismatch")
        model = keras.models.load_model(ck, compile=False)

    R = int(d["replicate"]["n_replicates"])
    vals = {k: [] for k in ("delta", "delta_matched", "delta_LR", "delta_D", "delta_T")}
    if gate_decision == "PASS":
        for i in range(R):
            rid = f"rep{i:02d}"
            rec = man.get("replicates", {}).get(rid)
            if not rec or rec.get("status") != "OK":
                problems.append(f"{rid} missing")
                continue
            p = out / rec["raw_file"]
            if not p.exists() or _sha(p) != rec["raw_sha256"]:
                problems.append(f"{rid} raw file missing or hash mismatch")
                continue
            with np.load(p) as z:
                r = {k: z[k] for k in z.files}
            n_reg = int(d["replicate"]["real_pairs_per_replicate"])
            layout = {**{f"{c}_{w}": ((n_reg,), np.uint16) for c in ("real", "random") for w in ("c0l", "c0r", "c1l", "c1r")},
                      "real_N": ((n_reg,), np.float32), "random_N": ((n_reg,), np.float32),
                      "real_T": ((n_reg,), np.float64), "real_F": ((5, n_reg), np.float64),
                      "real_D": ((n_reg,), np.float64), "random_D": ((n_reg,), np.float64),
                      "real_LR": ((n_reg,), np.float64)}
            if set(r) != set(layout):
                problems.append(f"{rid} raw keys {sorted(set(r) ^ set(layout))} differ from the registered layout")
                continue
            bad = [k for k, (shape, dt) in layout.items() if r[k].shape != shape or r[k].dtype != dt
                   or (r[k].dtype.kind == "f" and not np.isfinite(r[k]).all())]
            if bad:
                problems.append(f"{rid} arrays with wrong shape, dtype or non-finite values: {bad}")
                continue
            cr = [r[f"real_{w}"] for w in ("c0l", "c0r", "c1l", "c1r")]
            cn = [r[f"random_{w}"] for w in ("c0l", "c0r", "c1l", "c1r")]
            check_scores(cr, r["real_D"], f"{rid} real")
            check_scores(cn, r["random_D"], f"{rid} random")
            real_rate = float(np.mean([REF.decide(float(x), MASK) for x in r["real_D"]]))
            rand_rate = float(np.mean([REF.decide(float(x), MASK) for x in r["random_D"]]))
            need(real_rate - rand_rate >= 0.5,
                 f"{rid} class labels implausible: aKR real-decision rate {real_rate:.3f} (real) vs {rand_rate:.3f} (random)")
            need(len(r["real_T"]) == len(r["real_N"]) == len(r["real_D"]) and "random_T" not in r,
                 f"{rid} array lengths or class layout differ from the registered layout")
            F = r["real_F"]
            feas = (F > 0).all(axis=0)
            need(np.allclose(r["real_T"][feas], np.prod(F[:, feas], axis=0), rtol=1e-9, atol=0),
                 f"{rid} T is not the product of its trail factors")
            need(bool(np.all(r["real_T"][~feas] == 0)), f"{rid} T is not 0 on infeasible trails")
            need(rec.get("distinct_T") == int(np.unique(r["real_T"]).size), f"{rid} distinct_T does not recompute")
            if model is not None:
                for cls, c in (("real", cr), ("random", cn)):
                    j = rng.choice(len(c[0]), size=min(N_RECOMPUTE, len(c[0])), replace=False)
                    pred = model.predict(_bits([w[j] for w in c]), verbose=0).ravel()
                    need(np.allclose(pred, r[f"{cls}_N"][j], atol=1e-5), f"{rid} {cls} N does not recompute")
            N, T, D, LR = r["real_N"], r["real_T"], r["real_D"], r["real_LR"]
            rhos = {"rho_N_D_real": _rho(N, D), "rho_N_T_real": _rho(N, T), "rho_N_LR_real": _rho(N, LR),
                    "rho_N_Dmatched_real": _rho(N, _matched(D, T))}
            for k, v in rhos.items():
                need(_close(v, rec.get(k), 1e-9), f"{rid} {k} {rec.get(k)} recomputes to {v}")
                vals.setdefault(k, []).append(v)
            need(_close(rec.get("z_D"), np.arctanh(rhos["rho_N_D_real"]), 1e-9)
                 and _close(rec.get("z_T"), np.arctanh(rhos["rho_N_T_real"]), 1e-9), f"{rid} Fisher z does not recompute")
            need(rec.get("n_real") == n_reg and rec.get("n_random") == n_reg, f"{rid} recorded sizes differ")
            zt = np.arctanh(_rho(N, T))
            got = {"delta": np.arctanh(_rho(N, D)) - zt,
                   "delta_matched": np.arctanh(_rho(N, _matched(D, T))) - zt,
                   "delta_LR": np.arctanh(_rho(N, LR)) - zt}
            got["delta_D"], got["delta_T"] = _incremental(N, T, D)
            for k, v in got.items():
                tol = 1e-8 if k in ("delta_D", "delta_T") else 1e-9
                need(abs(float(v) - rec[k]) <= tol, f"{rid} {k} {rec[k]} recomputes to {float(v)}")
                vals[k].append(float(v))
            rnd = _rho(r["random_N"], r["random_D"])
            need(_close(rnd, rec.get("rho_N_D_random"), 1e-9), f"{rid} rho_N_D_random does not recompute")
            vals.setdefault("rho_N_D_random", []).append(rnd)

    complete = gate_decision == "PASS" and len(vals["delta"]) == R
    if gate_decision == "FAIL":
        decision = "UNADJUDICATED"
        lo = hi = None
    elif complete:
        lo, hi = _interval(vals["delta"])
        m = float(d["primary"]["margin"])
        decision = _decide(lo, hi, m)
        an = man.get("analysis", {})
        iv = an.get("primary", {}).get("interval", {})
        need(_close(iv.get("lower"), lo, 1e-9) and _close(iv.get("upper"), hi, 1e-9),
             "primary interval does not recompute")
        need(an.get("qualifier") == _qualifier(lo, hi, m), "qualifier does not recompute")
        pr = an.get("primary", {})
        need(len(pr.get("deltas", [])) == R and all(_close(a, b, 1e-9) for a, b in zip(pr.get("deltas", []), vals["delta"])),
             "stored primary deltas do not recompute")
        kk = 0
        while kk + 1 <= R and sum(comb(R, j) for j in range(kk + 1)) / 2 ** R <= 0.025:
            kk += 1
        cov = 1 - 2 * sum(comb(R, j) for j in range(kk)) / 2 ** R
        need(iv.get("k") == kk and _close(iv.get("coverage"), cov, 1e-12)
             and _close(iv.get("median"), float(np.median(vals["delta"])), 1e-9), "primary interval metadata does not recompute")
        pos, neg = sum(x > 0 for x in vals["delta"]), sum(x < 0 for x in vals["delta"])
        need(pr.get("n_positive") == pos and pr.get("n_negative") == neg
             and _close(pr.get("p_value"), _sign_p(vals["delta"]), 1e-12), "primary sign test does not recompute")
        need(an.get("n_valid") == R, "n_valid does not recompute")
        sec_all = an.get("secondary", {})
        for key in ("rho_N_D_real", "rho_N_T_real", "rho_N_LR_real", "rho_N_Dmatched_real", "rho_N_D_random"):
            v_ = vals.get(key, [])
            agg = sec_all.get(key, {})
            need(bool(v_) and _close(agg.get("median"), float(np.median(v_)), 1e-9) and _close(agg.get("min"), min(v_), 1e-9)
                 and _close(agg.get("max"), max(v_), 1e-9), f"secondary aggregate {key} does not recompute")
        for key in ("delta_D", "delta_T"):
            agg = sec_all.get(key, {})
            need(_close(agg.get("median"), float(np.median(vals[key])), 1e-8)
                 and _close(agg.get("p_value"), _sign_p(vals[key]), 1e-12), f"secondary {key} does not recompute")
        for key in ("delta_matched", "delta_LR"):
            slo, shi = _interval(vals[key])
            sec = an.get("secondary", {}).get(key, {})
            need(_close(sec.get("interval", {}).get("lower"), slo, 1e-9)
                 and _close(sec.get("interval", {}).get("upper"), shi, 1e-9)
                 and sec.get("outcome") == _decide(slo, shi, m), f"secondary {key} does not recompute")
        sens = {str(m): _decide(lo, hi, m) for m in d["primary"]["margin_sensitivity"]}
        need(man["analysis"]["secondary"]["margin_sensitivity"] == sens, "margin sensitivity does not recompute")
        holm = _holm({k: _sign_p(vals[k]) for k in ("delta_D", "delta_T")})
        for k, v in holm.items():
            need(abs(man["analysis"]["secondary"]["holm_adjusted_p"][k] - v) <= 1e-12, f"Holm {k} does not recompute")
    else:
        decision = "INCONCLUSIVE"
        lo = hi = None
    need(decision == man.get("decision"), f"decision {man.get('decision')} recomputes to {decision}")
    rec = {"verified": not problems, "status": "VERIFIED" if not problems else "NOT_VERIFIED",
           "recomputed_decision": decision, "interval": [lo, hi], "reference_score_checks": ref_checks,
           "model_recomputed": model is not None, "checks": checks, "problems": problems}
    target = out / "verification.json"
    k = 1
    while target.exists():
        target = out / f"verification_{k}.json"
        k += 1
    target.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    rec["written_to"] = target.name
    return rec


def verify_variability(vdir: Path, *, seed: int = 1, per_table: int = 8) -> dict:
    """Independent recomputation of a table-variability record: file hashes,
    table sums, strict decisions and confusion matrices from the stored scores,
    the SDs and proposed tolerances, and a literal scalar Algorithm-2 check of
    `per_table` stored scores per class against each stored table."""
    rec = json.loads((vdir / "table_variability.json").read_text())
    problems = []
    rng = np.random.default_rng(seed)
    n_eval = int(rec["n_eval_per_class"])
    accs, tprs, tnrs = [], [], []
    rescored = 0
    for row in rec["rows"]:
        i = row["table"]
        tp_, sp_ = vdir / "variability" / f"table_{i}.npz", vdir / "variability" / f"scores_{i}.npz"
        if _sha(tp_) != row["table_sha256"] or _sha(sp_) != row["scores_sha256"]:
            problems.append(f"table {i}: file hash mismatch")
            continue
        with np.load(tp_) as z:
            t, n = z["counts"], int(z["n_pairs"])
        with np.load(sp_) as z:
            sr, sn = z["real_score"], z["random_score"]
            cr = z["real_c"] if "real_c" in z.files else None
            cn = z["random_c"] if "random_c" in z.files else None
        if int(t.sum(dtype=np.uint64)) != n or n != int(rec["table_pairs"]):
            problems.append(f"table {i}: counts do not sum to the registered table size")
        tp = int(np.sum(sr > 2.0 ** -26))
        tn = int(np.sum(sn <= 2.0 ** -26))
        if row["confusion"] != {"tp": tp, "fn": n_eval - tp, "tn": tn, "fp": n_eval - tn}:
            problems.append(f"table {i}: confusion matrix does not recompute")
        if cr is None or cn is None:
            problems.append(f"table {i}: ciphertexts not stored; scores cannot be recomputed")
        else:
            for c, scores, label in ((cr, sr, "real"), (cn, sn, "random")):
                for j in rng.choice(len(scores), size=per_table, replace=False):
                    ref = REF.score((int(c[0][j]), int(c[1][j])), (int(c[2][j]), int(c[3][j])), t, n, MASK)
                    rescored += 1
                    if abs(ref - float(scores[j])) > 1e-12:
                        problems.append(f"table {i} {label} pair {j}: score {scores[j]} != reference {ref}")
        accs.append((tp + tn) / (2 * n_eval))
        tprs.append(tp / n_eval)
        tnrs.append(tn / n_eval)
        for key, val in (("accuracy", (tp + tn) / (2 * n_eval)), ("tpr", tp / n_eval), ("tnr", tn / n_eval)):
            if not _close(row.get(key), val):
                problems.append(f"table {i}: {key} does not recompute")
    d = DS.design()
    if rec.get("n_tables") != len(rec["rows"]) or [r["table"] for r in rec["rows"]] != list(range(len(rec["rows"]))):
        problems.append("row count or numbering differs from n_tables")
    if rec.get("table_pairs") != int(d["baseline_D"]["table_pairs"]) or rec.get("table_rounds") != int(d["baseline_D"]["table_rounds"]):
        problems.append("variability table size or rounds differ from the design")
    if rec.get("rule") != d["gate"]["tolerance_rule"]:
        problems.append("recorded tolerance rule differs from the design")
    sd = {"accuracy": float(np.std(accs, ddof=1)), "tpr": float(np.std(tprs, ddof=1)), "tnr": float(np.std(tnrs, ddof=1))}
    if any(not _close(rec.get("three_sd", {}).get(k), 3 * v) for k, v in sd.items()):
        problems.append("three_sd does not recompute")
    for k, v in sd.items():
        if abs(v - rec["sd"][k]) > 1e-12:
            problems.append(f"sd {k} does not recompute")
    tol = {"accuracy": max(0.005, 3 * sd["accuracy"]), "rates": max(0.010, 3 * max(sd["tpr"], sd["tnr"]))}
    if any(abs(tol[k] - rec["proposed_tolerance"][k]) > 1e-12 for k in tol):
        problems.append("proposed tolerance does not recompute")
    return {"verified": not problems, "status": "VERIFIED" if not problems else "NOT_VERIFIED",
            "sd": sd, "proposed_tolerance": tol, "scalar_rescored_pairs": rescored, "problems": problems}


def verify_calibration(cdir: Path) -> dict:
    """Independent check of a calibration record: blinding, registered
    configuration, planning spread and the replicate choice, recomputed with
    this module's own Monte Carlo and Clopper-Pearson code."""
    from scipy.stats import chi2
    rec = json.loads((cdir / "calibration.json").read_text())
    d = DS.design()
    problems = []

    def need(c, m):
        if not c:
            problems.append(m)
    text = json.dumps(rec)
    need(rec.get("excluded_from_evidence") is True and rec.get("blinded") is True, "not marked blinded and excluded")
    need('"rho' not in text and '"delta"' not in text and "mean" not in text, "record contains location or correlation fields")
    need(rec.get("pairs_per_replicate") == int(d["replicate"]["real_pairs_per_replicate"]), "pair count is not registered")
    need(rec.get("model_binding", {}).get("sha256") == DS.REFERENCE_CHECKPOINT_SHA256, "model binding is not the registered checkpoint")
    tv = d["baseline_D"]["table_validation"]
    need(rec.get("table_validation", {}).get("sha256") == tv["sha256"] and rec["table_validation"].get("decision") == "PASS",
         "table validation is not the registered PASS record")
    c = np.asarray(rec.get("centred_deltas", []), dtype=float)
    need(c.size == 10, f"{c.size} calibration replicates, registered 10")
    need(abs(float(c.sum())) < 1e-9, "centred values do not sum to zero")
    df = c.size - 1
    s = float(np.std(c, ddof=1))
    mad = 1.4826 * float(np.median(np.abs(c - np.median(c))))
    factor = float(np.sqrt(df / chi2.ppf(0.20, df)))
    sigma = max(s, mad) * factor
    ps = rec.get("planning_spread", {})
    need(_close(ps.get("sample_sd"), s, 1e-12) and _close(ps.get("robust_sd"), mad, 1e-12)
         and _close(ps.get("sigma_plan"), sigma, 1e-12), "planning spread does not recompute")
    m = float(d["primary"]["margin"])
    rr = rec.get("replicate_rule", {})
    need(_close(rr.get("margin"), m) and _close(rr.get("sd_used"), sigma, 1e-12), "replicate rule inputs differ")
    alpha = 0.01 / 3
    chosen = None
    for R in range(20, 61, 5):
        z = np.random.default_rng([0, R]).standard_normal(size=(50_000, R))
        k = 0
        while k + 1 <= R and sum(comb(R, j) for j in range(k + 1)) / 2 ** R <= 0.025:
            k += 1
        ok = True
        for cond, delta, outcome in (("delta=0", 0.0, "EQUIVALENT"), ("delta=+2m", 2 * m, "DISTRIBUTION_GREATER"),
                                     ("delta=-2m", -2 * m, "TRAIL_GREATER")):
            x = np.sort(delta + sigma * z, axis=1)
            lo, hi = x[:, k - 1], x[:, R - k]
            eq = (lo > -m) & (hi < m)
            hit = {"EQUIVALENT": eq, "DISTRIBUTION_GREATER": ~eq & (lo >= m), "TRAIL_GREATER": ~eq & (hi <= -m)}[outcome]
            cnt = int(hit.sum())
            lower = 0.0 if cnt == 0 else float(beta_dist.ppf(alpha, cnt, 50_000 - cnt + 1))
            stored = rr.get("outcome_probabilities", {}).get(str(R), rr.get("outcome_probabilities", {}).get(R, {}))
            need(stored.get(cond, {}).get(outcome, {}).get("count") == cnt, f"R={R} {cond} count does not recompute")
            ok = ok and lower >= 0.80
        if ok and chosen is None:
            chosen = R
    expected = chosen if chosen is not None else 60
    need(rr.get("chosen") == expected and rr.get("met_target") == (chosen is not None),
         f"chosen R {rr.get('chosen')} recomputes to {expected}")
    return {"verified": not problems, "status": "VERIFIED" if not problems else "NOT_VERIFIED",
            "sigma_plan": sigma, "chosen": expected, "met_target": chosen is not None, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ce5-verify")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--prereg", type=Path, required=True, help="path of the approved preregistration")
    ap.add_argument("--model", action="store_true", help="recompute N for a subsample from the checkpoint")
    ap.add_argument("--amendment", type=Path, help="approved CE5 amendment the production ran under")
    a = ap.parse_args(argv)
    r = verify(a.out, a.tables, model_check=a.model, prereg_path=a.prereg, amendment_path=a.amendment)
    print(json.dumps({k: v for k, v in r.items() if k != "checks"}, indent=2)[:4000])
    return 0 if r["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
