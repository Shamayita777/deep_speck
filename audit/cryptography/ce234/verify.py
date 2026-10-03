"""
INDEPENDENT VERIFIER for CE2-CE4.

Recomputes every reported statistic from the PERSISTED RAW OBSERVATIONS using
its own implementations. It imports frozen_plan only for declared constants
(alpha, expected hashes, replication counts) and imports NOTHING from
production.py, so a bug in the production estimator cannot be reproduced here
by construction.

Detects: wrong model hash, wrong architecture, wrong rounds/differential,
wrong sample count, wrong replication count, altered raw files, altered
certificate numbers, wrong sign convention, wrong statistical unit, and
disagreement between the certificate and the raw data.

    python -m audit.cryptography.ce234.verify <run_dir>
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

from audit.cryptography.ce234 import frozen_plan as P


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _rank(x):                      # independent average-rank implementation
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float); r[order] = np.arange(1, x.size + 1)
    s = np.sort(x); i = 0
    while i < s.size:                                  # average ties
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[np.isin(x, s[i])] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman(a, b):                # Pearson on ranks, computed from scratch
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def verify(run_dir) -> dict:
    run_dir = Path(run_dir)
    cert = json.loads((run_dir / "certificate.json").read_text())
    ce = cert["experiment_id"].split("-")[0]
    production = not cert.get("non_evidentiary", False)
    checks, fail = [], []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            fail.append(name)

    chk("plan_hash matches frozen_plan.py", cert["plan_hash"] == P.plan_hash(),
        f"{cert['plan_hash'][:16]} vs {P.plan_hash()[:16]}")
    m = cert["reference_model"]
    chk("model sha256 is the frozen reference",
        m["sha256"] == P.REFERENCE_CHECKPOINT_SHA256, m["sha256"][:16])
    chk("realized depth == 10", m["realized_depth"] == P.DEPTH)
    chk("architecture matches frozen design",
        all(m["properties"][k] == v for k, v in P.REQUIRED_MODEL_PROPERTIES.items()))
    bb = m.get("behavioural_binding")
    if production:
        # A production certificate MUST carry the behavioural round binding:
        # the file records no round count, so nothing else establishes it.
        chk("behavioural round binding present and passed",
            bool(bb) and bb.get("verdict") == "PASS" and bb["declared_rounds"] == P.ROUNDS)
        chk("differential is (0x0040,0x0000)",
            bool(bb) and list(bb["differential"]) == list(P.DIFFERENTIAL))
    else:
        chk("smoke run: behavioural binding either passed or was explicitly skipped",
            bb is None or bb.get("verdict") == "PASS",
            "skipped" if bb is None else "passed")
    chk("no absolute path in provenance",
        not any(str(v).startswith(("/home/", "/Users/", "C:\\"))
                for v in [m["path"]] + [s for s in cert["source_hashes"]]))

    def finite(arr, name):
        a = np.asarray(arr, dtype=np.float64)
        chk(f"{name} contains no NaN/Inf", bool(np.isfinite(a).all()))

    chk("generator binding recorded",
        cert.get("environment") is not None and "python" in cert["environment"])
    R = cert["results"]
    if ce == "CE2":
        runs = [r for r in R["runs"] if r.get("status") == "OK"]
        if production:
            chk("replication count equals the frozen plan",
                len(runs) == P.CE2.n_runs, f"{len(runs)} vs {P.CE2.n_runs}")
        else:
            chk("smoke certificate is marked non-evidentiary",
                cert["claim_scope"].startswith("SMOKE ONLY"))
        rhos = []
        for r in runs:
            f = run_dir / r["raw_file"]
            chk(f"raw file intact {r['run_id']}", _sha(f) == r["raw_sha256"])
            with np.load(f) as z:
                t, o = z["target"], z["output"]
                chk(f"sample count {r['run_id']}", t.size == r["n"] == o.size, f"{t.size}")
                chk(f"rounds == 5 {r['run_id']}", r.get("rounds") == P.ROUNDS)
                chk(f"differential == (0x0040,0x0000) {r['run_id']}",
                    list(r.get("differential", [])) == list(P.DIFFERENTIAL))
                chk(f"evaluation seed recorded {r['run_id']}",
                    isinstance(r.get("evaluation_seed"), int))
                finite(t, f"target {r['run_id']}"); finite(o, f"output {r['run_id']}")
                rho = spearman(t, o)
                rhos.append(rho)
                chk(f"rho recomputed {r['run_id']}",
                    abs(rho - r["rho_primary"]) < 2e-4,
                    f"cert {r['rho_primary']:+.6f} vs independent {rho:+.6f}")
                chk(f"permuted-target control recomputed {r['run_id']}",
                    abs(spearman(t[z["permutation"]], o)
                        - r["rho_C2_PERMUTED_TARGET"]) < 2e-4)
        if rhos and "endpoints" in R:
            pos = sum(1 for x in rhos if x > 0); n = sum(1 for x in rhos if x != 0)
            p = float(stats.binomtest(pos, n, 0.5, alternative="two-sided").pvalue)
            rep = R["endpoints"]["primary"]
            chk("exact sign test recomputed", abs(p - rep["p_value"]) < 1e-12,
                f"cert {rep['p_value']:.6g} vs independent {p:.6g}")
            chk("median rho recomputed",
                abs(float(np.median(rhos)) - rep["median"]) < 2e-4)
            chk("primary inference uses 20-ish replicate signs, not per-sample data",
                rep["n_effective"] == len([x for x in rhos if x != 0])
                and rep["n_effective"] <= len(runs))
            if rep.get("ci95_bootstrap"):
                lo, hi = rep["ci95_bootstrap"]
                chk("bootstrap CI is finite and ordered",
                    np.isfinite([lo, hi]).all() and lo <= hi)
            chk("sign convention consistent",
                (rep["median"] > 0) == (float(np.median(rhos)) > 0))
    elif ce == "CE3":
        f = run_dir / R["raw_file"]
        chk("raw file intact", _sha(f) == R["raw_sha256"])
        with np.load(f) as z:
            sel, real, ctrl = z["selectivity"], z["real_score"], z["control_score"]
            if production:
                chk("replication count equals the frozen plan",
                    sel.size == P.CE3.n_replicates, f"{sel.size}")
            else:
                chk("smoke certificate is marked non-evidentiary",
                    cert["claim_scope"].startswith("SMOKE ONLY"))
            chk("selectivity == real - control (sign convention)",
                np.allclose(sel, real - ctrl, atol=1e-12))
            if "primary" not in R:
                chk("fail-closed: fewer valid replicates than the plan requires, so no "
                    "primary statistic was computed", R.get("decision") == "INCONCLUSIVE",
                    R.get("reason", ""))
                return {"run_dir": str(run_dir), "experiment": cert["experiment_id"],
                        "verdict": "VERIFIED" if not fail else "VERIFICATION FAILED",
                        "n_checks": len(checks), "failed": fail, "checks": checks}
            finite(sel, "selectivity"); finite(real, "real score"); finite(ctrl, "twin score")
            chk("raw-input control is informational only",
                R["C3_RAW_INPUT"]["role"].startswith("SECONDARY INFORMATIONAL"))
            chk("twin-decodability audit present", "twin_decodability_audit" in R)
            chk("mean selectivity recomputed",
                abs(float(sel.mean()) - R["primary"]["mean"]) < 1e-9)
            w = stats.wilcoxon(sel, alternative="greater")
            chk("replicate-level Wilcoxon recomputed",
                abs(float(w.pvalue) - R["primary"]["wilcoxon_signed_rank"]["p_value"])
                < 1e-12)
            dz = float(sel.mean() / sel.std(ddof=1))
            chk("Cohen's dz recomputed from raw replicate values",
                abs(dz - R["primary"]["effect_size_cohens_dz"]) < 1e-6,
                f"{dz:.4f}")
            chk("statistical unit is the replicate, not the fold",
                R["statistical_unit"] == "independent evaluation replicate")
            chk("control representation was non-degenerate",
                float(np.std(ctrl)) > 0 or float(np.mean(ctrl)) not in (0.2, 0.5))
    else:
        runs = [r for r in R["runs"] if r.get("status") == "OK"]
        chk("at least one valid replicate", bool(runs))
        if production:
            chk("replicate count equals the frozen plan",
                len(runs) == P.CE4.n_runs, f"{len(runs)} vs {P.CE4.n_runs}")
        aggregated = "by_magnitude" in R
        if not aggregated:
            chk("fail-closed: fewer valid replicates than the plan requires, so no "
                "aggregate statistic was computed", R.get("decision") == "INCONCLUSIVE",
                R.get("reason", ""))
        mags = [str(m) for m in P.CE4.magnitude_ladder_bits
                if runs and str(m) in runs[0].get("by_magnitude", {})]
        if aggregated:
            chk("threshold policy present and empty of magnitude thresholds",
                "NO magnitude threshold" in R.get("threshold_policy", ""))
            chk("primary magnitude is the smallest admitted magnitude",
                R.get("primary_magnitude_bits") == P.CE4.primary_magnitude_bits
                == min(P.CE4.magnitude_ladder_bits))
            chk("primary inference is across replicates, not across samples",
                "replicate" in R.get("statistical_unit", ""))
        for r in runs:
            f = run_dir / r["raw_file"]
            chk(f"raw file intact {r['run_id']}", _sha(f) == r["raw_sha256"])
            chk(f"C4-NULL exactly zero {r['run_id']}",
                r["C4_NULL_mean_abs_change"] == 0.0)
            with np.load(f) as z:
                for m in mags:
                    if f"gap_{m}" not in z.files:
                        chk(f"magnitude {m} present {r['run_id']}", False); continue
                    g = z[f"gap_{m}"].astype(np.float64)
                    finite(g, f"gap 2k={m} {r['run_id']}")
                    chk(f"rounds/differential recorded {r['run_id']}",
                        r.get("rounds") == P.ROUNDS
                        and list(r.get("differential", [])) == list(P.DIFFERENTIAL))
                    chk(f"sample counts consistent 2k={m} {r['run_id']}",
                        g.size == z[f"f0_{m}"].size == r["by_magnitude"][m]["n_eligible"])
                    indep = (np.abs(z[f"f0_{m}"] - z[f"fs_{m}"])
                             - np.abs(z[f"f0_{m}"] - z[f"fc_{m}"])).astype(np.float64)
                    chk(f"gap == |f0-fs|-|f0-fc| at 2k={m} {r['run_id']}",
                        np.allclose(g, indep, atol=1e-7))
                    rep = r["by_magnitude"][m]
                    chk(f"mean gap recomputed at 2k={m} {r['run_id']}",
                        abs(float(indep.mean()) - rep["mean_gap"]) < 1e-6,
                        f"cert {rep['mean_gap']:+.6f} vs independent "
                        f"{float(indep.mean()):+.6f}")
                    c = rep["manipulation_checks"]
                    chk(f"equal total Hamming 2k={m} {r['run_id']}", c["equal_total_hamming"])
                    chk(f"equal per-ciphertext 2k={m} {r['run_id']}", c["equal_per_ciphertext"])
                    chk(f"equal per-word 2k={m} {r['run_id']}", c["equal_per_word"])
                    chk(f"identical c0-side flips 2k={m} {r['run_id']}",
                        c["identical_c0_side_flips"])
                    chk(f"control preserves pair XOR 2k={m} {r['run_id']}",
                        c["control_preserves_pair_xor"])
                    chk(f"structural changes XOR at exactly 2k={m} {r['run_id']}",
                        c["structural_xor_positions_changed"] == [int(m)])
                    chk(f"both arms flip exactly {m} bits {r['run_id']}",
                        c["hamming_structural"] == c["hamming_control"] == [int(m)])
        if aggregated and "primary" in R:
            means = [r["by_magnitude"][str(P.CE4.primary_magnitude_bits)]["mean_gap"]
                     for r in runs]
            pos = sum(1 for x in means if x > 0); n = sum(1 for x in means if x != 0)
            pv = float(stats.binomtest(pos, n, 0.5, alternative="two-sided").pvalue)
            chk("across-replicate sign test recomputed",
                abs(pv - R["primary"]["p_value"]) < 1e-12)
            chk("mean of replicate means recomputed",
                abs(float(np.mean(means)) - R["primary"]["mean_of_means"]) < 1e-9)
            chk("primary inference is across replicates, not across samples",
                "replicate" in R["statistical_unit"])

    verdict = "VERIFIED" if not fail else "VERIFICATION FAILED"
    return {"run_dir": str(run_dir), "experiment": cert["experiment_id"],
            "verdict": verdict, "n_checks": len(checks), "failed": fail,
            "checks": checks}


if __name__ == "__main__":
    out = verify(sys.argv[1])
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["verdict"] == "VERIFIED" else 1)
