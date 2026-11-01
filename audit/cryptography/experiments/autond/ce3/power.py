from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from . import design as D
from . import probes as PR
from . import statistics as ST

SESOI = 0.005
N_CANDIDATES = (50_000, 100_000, 200_000)
B_CANDIDATES = (5, 11, 21)
B_EMPIRICAL = (5, 11)
POWER_TARGET = 0.90
LOWER_QUANTILE = 0.05
D_FEATURES = 64
N_BOOT = 400
N_INNER = 100_000
N_DIRECT = 20_000
SYNTHETIC_INDEX_BASE = 1_000_000
KAPPAS = (0.0, 0.25, 0.5, 0.75, 0.9)
EFFECT_SCALES = (0.5, 0.75, 1.0, 1.25)
TAUS = (0.0, 0.001, 0.002, 0.003)
TIE_RATES = (0.0, 0.05, 0.1, 0.2)
FAILURE_RATES = (0.0, 0.005, 0.01)
CAL_CORRELATIONS = (None, 0.0, 0.9)


def critical_count(n: int, alpha: float = D.ALPHA) -> int:
    a = Fraction(str(alpha))
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) * a.denominator < a.numerator * 2 ** n:
            return k
    return n + 1


def cp_interval(k: int, n: int, confidence: float = 0.95) -> list[float]:
    t = (1.0 - confidence) / 2.0
    return [0.0 if k == 0 else float(beta.ppf(t, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1.0 - t, k + 1, n - k))]


def _relu_features(rng, n: int, mix: np.ndarray) -> np.ndarray:
    return np.maximum(rng.standard_normal((n, D_FEATURES)) @ mix, 0.0)


def _ordinal_classes(rng, score: np.ndarray, lam: float, fractions) -> np.ndarray:
    latent = lam * score + rng.standard_normal(score.size)
    cuts = np.quantile(latent, np.cumsum(fractions)[:-1])
    return np.searchsorted(cuts, latent, side="left").astype(np.int8)


def _standard_score(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = h @ w
    return (s - s.mean()) / s.std()


def calibration_lambda(rng, mix: np.ndarray, w: np.ndarray, accuracy: float, n: int = 400_000) -> float:
    s = _standard_score(_relu_features(rng, n, mix), w)
    u = rng.random(n)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        y = u < 1.0 / (1.0 + np.exp(-mid * s))
        acc = float(np.mean((s > 0) == y))
        lo, hi = (mid, hi) if acc < accuracy else (lo, mid)
    return (lo + hi) / 2


def _balanced_binary(rng, n_per_class: int, mix: np.ndarray, w: np.ndarray, lam: float):
    pool = 4 * n_per_class
    h = _relu_features(rng, pool, mix)
    s = _standard_score(h, w)
    y = (rng.random(pool) < 1.0 / (1.0 + np.exp(-lam * s))).astype(np.int8)
    ones, zeros = np.flatnonzero(y == 1)[:n_per_class], np.flatnonzero(y == 0)[:n_per_class]
    if ones.size < n_per_class or zeros.size < n_per_class:
        raise RuntimeError("synthetic calibration pool too small")
    idx = rng.permutation(np.concatenate([ones, zeros]))
    return h[idx], y[idx], s[idx]


def target_lambda(rng, fractions, oracle_effect: float, n: int = 400_000) -> float:
    s = rng.standard_normal(n)
    lo, hi = 0.0, 2.0
    k = len(fractions)
    for _ in range(40):
        mid = (lo + hi) / 2
        latent = mid * s + rng.standard_normal(n)
        cuts = np.quantile(latent, np.cumsum(fractions)[:-1])
        y = np.searchsorted(cuts, latent, side="left")
        pred = np.searchsorted(np.quantile(mid * s, np.cumsum(fractions)[:-1]), mid * s, side="left")
        ba = np.mean([np.mean(pred[y == c] == c) for c in range(k)])
        lo, hi = (mid, hi) if ba - 1.0 / k < oracle_effect else (lo, mid)
    return (lo + hi) / 2


def _single_thread() -> None:
    threadpool_limits(limits=1)


def pipeline_replicate(args) -> dict:
    with threadpool_limits(limits=1):
        return _pipeline_replicate(args)


def _arm(X, y, folds, k):
    r = PR.fit_arm(X, y, folds, k, max_iter=D.MAX_ITER)
    pred = r["pred"].astype(np.int64)
    ex = PR.balanced_accuracy_exact(y.astype(np.int64), pred, k)
    return r["ba"], ex


def _pipeline_replicate(args) -> dict:
    n, n_cal_per_class, fractions, lam_target, index, seed = args
    rng = np.random.Generator(np.random.PCG64(seed))
    k = len(fractions)
    mixes = [rng.standard_normal((D_FEATURES, D_FEATURES)) / math.sqrt(D_FEATURES) for _ in range(4)]
    w = [rng.standard_normal(D_FEATURES) for _ in range(4)]
    h_a = _relu_features(rng, n, mixes[0])
    y = _ordinal_classes(rng, _standard_score(h_a, w[0]), lam_target, fractions).astype(np.int64)
    h_t = _relu_features(rng, n, mixes[1])
    folds = PR.fold_ids(n, int(rng.integers(1 << 30)))
    t0 = time.time()
    ba_a, ex_a = _arm(h_a, y, folds, k)
    perm, s1 = {}, {}
    for b in B_EMPIRICAL:
        idx = PR.permutations(n, index, b)
        res = [_arm(h_a, y[idx[j]], folds, k) for j in range(b)]
        tb = rng.permutation(b + 1)
        rank = PR.s1_rank(ex_a, [r[1] for r in res], tb)
        perm[str(b)] = [r[0] for r in res]
        s1[str(b)] = {"rank": rank["rank"], "upper_half": rank["upper_half"],
                      "ties_with_observed": rank["ties_with_observed"],
                      "ties_among_permuted": b - len({r[1] for r in res})}
    ba_t, _ = _arm(h_t, y, folds, k)
    lam_c = calibration_lambda(rng, mixes[2], w[2], D.PUBLISHED_ACCURACY)
    hc, yc, sc = _balanced_binary(rng, n_cal_per_class, mixes[2], w[2], lam_c)
    hct = _relu_features(rng, 2 * n_cal_per_class, mixes[3])
    cfolds = PR.fold_ids(2 * n_cal_per_class, int(rng.integers(1 << 30)))
    yc = yc.astype(np.int64)
    cal_probe, _ = _arm(hc, yc, cfolds, 2)
    cal_twin, _ = _arm(hct, yc, cfolds, 2)
    cal_model = PR.balanced_accuracy(yc, (sc > 0).astype(np.int64), 2)
    return {"n": n, "index": index, "seed": seed, "ba_a": ba_a, "ba_perm": perm, "s1": s1, "ba_twin": ba_t,
            "ba_cal_probe": cal_probe, "ba_cal_twin": cal_twin, "ba_cal_model": cal_model,
            "lambda_calibration": lam_c, "seconds": time.time() - t0}


def null_replicate(args) -> dict:
    with threadpool_limits(limits=1):
        n, b, fractions, index, seed = args
        rng = np.random.Generator(np.random.PCG64(seed))
        k = len(fractions)
        h = _relu_features(rng, n, rng.standard_normal((D_FEATURES, D_FEATURES)) / math.sqrt(D_FEATURES))
        y = _ordinal_classes(rng, np.zeros(n), 0.0, fractions).astype(np.int64)
        folds = PR.fold_ids(n, int(rng.integers(1 << 30)))
        _, ex_a = _arm(h, y, folds, k)
        idx = PR.permutations(n, index, b)
        ex_p = [_arm(h, y[idx[j]], folds, k)[1] for j in range(b)]
        r = PR.s1_rank(ex_a, ex_p, rng.permutation(b + 1))
        return {"rank": r["rank"], "upper_half": r["upper_half"], "ties_with_observed": r["ties_with_observed"]}


def null_check(fractions, *, n: int = 5_000, b: int = 5, reps: int = 400, workers: int = 1,
               seed: int = 20261107) -> dict:
    jobs = [(n, b, list(fractions), SYNTHETIC_INDEX_BASE + 500_000 + j, int(seed * 1000 + j)) for j in range(reps)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_single_thread) as ex:
            res = list(ex.map(null_replicate, jobs))
    else:
        res = [null_replicate(j) for j in jobs]
    up = sum(r["upper_half"] for r in res)
    counts = [sum(1 for r in res if r["rank"] == k) for k in range(1, b + 2)]
    return {"n": n, "B": b, "replications": reps, "upper_half": up, "fraction_upper_half": up / reps,
            "ci95": cp_interval(up, reps), "rank_counts": counts,
            "replications_with_ties": sum(1 for r in res if r["ties_with_observed"]),
            "method": ("pipeline-faithful null: the class labels carry no signal (independent of the representation), "
                       "the CE3 probe, permutation group and tie-break are run unchanged; under the null the upper-half "
                       "fraction is exactly 1/2 and every rank has probability 1/(B + 1)")}


def pipeline_faithful(fractions, *, reps_per_n: dict, seed: int = 20261103, workers: int = 1,
                      oracle_effect: float = SESOI) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    lam_t = target_lambda(rng, fractions, oracle_effect)
    jobs, j = [], 0
    for n, r in reps_per_n.items():
        for _ in range(r):
            jobs.append((n, D.N_CALIBRATION_PER_CLASS, list(fractions), lam_t, SYNTHETIC_INDEX_BASE + j,
                         int(seed * 1000 + j)))
            j += 1
    t = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_single_thread) as ex:
            results = list(ex.map(pipeline_replicate, jobs))
    else:
        results = [pipeline_replicate(x) for x in jobs]
    return {"lambda_target": lam_t, "oracle_effect": oracle_effect, "calibration_per_class": D.N_CALIBRATION_PER_CLASS,
            "calibration_accuracy_anchor": D.PUBLISHED_ACCURACY, "independent_pipeline_replications": len(results),
            "replications_per_n": {str(n): r for n, r in reps_per_n.items()}, "replicates": results,
            "seconds": time.time() - t}


def estimate(reps: list[dict]) -> dict:
    a = np.array([r["ba_a"] for r in reps])
    t = np.array([r["ba_twin"] for r in reps])
    P = np.array([sum((r["ba_perm"][str(b)] for b in B_EMPIRICAL), []) for r in reps])
    m_a, m_t, m_p = a.mean(), t.mean(), P.mean()
    da, dt, dP = a - m_a, t - m_t, P - m_p
    R, J = P.shape
    off = (dP.sum(1) ** 2 - (dP ** 2).sum(1)) / (J * (J - 1))
    cal = np.array([[r["ba_cal_probe"], r["ba_cal_twin"], r["ba_cal_model"]] for r in reps])
    return {"n_replications": int(R), "m_a": float(m_a), "m_t": float(m_t), "m_p": float(m_p),
            "v_a": float(np.sum(da ** 2) / (R - 1)), "v_t": float(np.sum(dt ** 2) / (R - 1)),
            "v_p": float(np.sum(dP ** 2) / (R * J - 1)), "c_at": float(np.sum(da * dt) / (R - 1)),
            "c_ap": float(np.sum(da[:, None] * dP) / (J * (R - 1))),
            "c_tp": float(np.sum(dt[:, None] * dP) / (J * (R - 1))), "c_pp": float(np.sum(off) / (R - 1)),
            "cal_mean": cal.mean(0).tolist(), "cal_cov": np.cov(cal, rowvar=False, ddof=1).tolist()}


def _psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.clip(w, 0.0, None)) @ V.T


def _main_cov(th: dict, b: int, tau: float) -> tuple[np.ndarray, np.ndarray]:
    m = b + 2
    C = np.empty((m, m))
    C[0, 0], C[1, 1], C[0, 1] = th["v_a"] + tau ** 2, th["v_t"], th["c_at"]
    C[1, 0] = C[0, 1]
    C[0, 2:], C[1, 2:] = th["c_ap"], th["c_tp"]
    C[2:, 0], C[2:, 1] = th["c_ap"], th["c_tp"]
    C[2:, 2:] = th["c_pp"]
    np.fill_diagonal(C[2:, 2:], th["v_p"])
    return np.array([th["m_a"], th["m_t"]] + [th["m_p"]] * b), _psd(C)


def main_cells(th: dict, b: int, *, scale: float = 1.0, tau: float = 0.0, tie_rate: float = 0.0,
               n_inner: int = N_INNER, seed: int = 1) -> np.ndarray:
    th = dict(th)
    th["m_p"] = th["m_a"] - scale * (th["m_a"] - th["m_p"])
    th["m_t"] = th["m_a"] - scale * (th["m_a"] - th["m_t"])
    mu, C = _main_cov(th, b, tau)
    z = np.random.Generator(np.random.PCG64(seed)).multivariate_normal(mu, C, size=n_inner, method="eigh")
    s1 = np.count_nonzero(z[:, 2:] < z[:, :1], axis=1) >= (b + 1) // 2
    s2 = z[:, 0] - z[:, 1] > 0
    cells = np.array([np.mean(s1 & s2), np.mean(s1 & ~s2), np.mean(~s1 & s2), np.mean(~s1 & ~s2)])
    return (1 - tie_rate) * cells + tie_rate * np.array([0.0, 0.5, 0.0, 0.5])


def cal_cells(th: dict, *, kappa: float = D.KAPPA, tau: float = 0.0, corr: float | None = None,
              n_inner: int = N_INNER, seed: int = 2) -> np.ndarray:
    C = np.array(th["cal_cov"], dtype=np.float64)
    C[0, 0] += tau ** 2
    z = np.random.Generator(np.random.PCG64(seed)).multivariate_normal(np.array(th["cal_mean"]), _psd(C),
                                                                       size=n_inner, method="eigh")
    ct = z[:, 0] - 0.5 - kappa * np.maximum(z[:, 2] - 0.5, 0.0) > 0
    cw = z[:, 0] - z[:, 1] > 0
    if corr is not None:
        pt, pw = ct.mean(), cw.mean()
        return _cells_with_correlation(pt, pw, corr)
    return np.array([np.mean(ct & cw), np.mean(ct & ~cw), np.mean(~ct & cw), np.mean(~ct & ~cw)])


def _cells_with_correlation(p1: float, p2: float, corr: float) -> np.ndarray:
    p11 = p1 * p2 + corr * math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    p11 = min(max(p11, max(0.0, p1 + p2 - 1)), min(p1, p2))
    return np.array([p11, p1 - p11, p2 - p11, 1 - p1 - p2 + p11])


def joint_count_power(cells, n: int = D.N_REPLICATES, crit: int | None = None) -> float:
    crit = critical_count(n) if crit is None else crit
    p11, p10, p01, p00 = (max(float(c), 0.0) for c in cells)
    total = 0.0
    lf = [math.lgamma(i + 1) for i in range(n + 1)]
    for a in range(n + 1):
        for b in range(n - a + 1):
            if a + b < crit:
                continue
            for c in range(max(0, crit - a), n - a - b + 1):
                d = n - a - b - c
                lp = lf[n] - lf[a] - lf[b] - lf[c] - lf[d]
                for k, p in ((a, p11), (b, p10), (c, p01), (d, p00)):
                    if k:
                        if p <= 0.0:
                            lp = -math.inf
                            break
                        lp += k * math.log(p)
                if lp > -math.inf:
                    total += math.exp(lp)
    return min(total, 1.0)


def marginal_power(p: float, n: int = D.N_REPLICATES) -> float:
    crit = critical_count(n)
    return float(sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(crit, n + 1)))


def power_at(th: dict, b: int, *, kappa: float = D.KAPPA, scale: float = 1.0, tau: float = 0.0,
             tie_rate: float = 0.0, failure_rate: float = 0.0, cal_corr: float | None = None,
             n_inner: int = N_INNER, seed: int = 1) -> dict:
    mc = main_cells(th, b, scale=scale, tau=tau, tie_rate=tie_rate, n_inner=n_inner, seed=seed)
    cc = cal_cells(th, kappa=kappa, tau=tau, corr=cal_corr, n_inner=n_inner, seed=seed + 1)
    p_valid = (1.0 - failure_rate) ** D.N_REPLICATES
    pc, ps = joint_count_power(cc), joint_count_power(mc)
    q = {"C_task": float(cc[0] + cc[1]), "C_twin": float(cc[0] + cc[2]), "S1": float(mc[0] + mc[1]),
         "S2": float(mc[0] + mc[2])}
    marg = {k: marginal_power(v) for k, v in q.items()}
    return {"p_supported": p_valid * pc * ps, "p_C": pc, "p_S1_and_S2": ps, "p_all_valid": p_valid,
            "q_positive": q, "marginal_rejection": marg,
            "p_supported_if_independent": p_valid * float(np.prod(list(marg.values()))),
            "p_supported_frechet_lower": p_valid * max(0.0, sum(marg.values()) - 3.0),
            "main_cells": mc.tolist(), "cal_cells": cc.tolist()}


def bootstrap(reps: list[dict], b_values=B_CANDIDATES, *, n_boot: int = N_BOOT, n_inner: int = N_INNER // 2,
              seed: int = 20261104) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    R = len(reps)
    vals = {str(b): [] for b in b_values}
    for i in range(n_boot):
        th = estimate([reps[j] for j in rng.integers(0, R, R)])
        for b in b_values:
            vals[str(b)].append(power_at(th, b, n_inner=n_inner, seed=1000 + i)["p_supported"])
    return {str(b): {"median": float(np.median(v)), "lower": float(np.quantile(v, LOWER_QUANTILE)),
                     "mean": float(np.mean(v)), "quantiles": {f"{q:g}": float(np.quantile(v, q))
                                                              for q in (0.01, 0.05, 0.1, 0.25, 0.5)}}
            for b, v in vals.items()} | {"n_boot": n_boot, "n_inner": n_inner,
                                         "method": "nonparametric bootstrap over the independent synthetic pipeline "
                                                   "replications; each resample re-estimates every mean, variance "
                                                   "and covariance and recomputes the exact decision power"}


def empirical_check(reps: list[dict], *, n_draws: int = 2000, seed: int = 20261105) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {}
    cal = [(r["ba_cal_probe"] - 0.5 - D.KAPPA * max(r["ba_cal_model"] - 0.5, 0.0) > 0,
            r["ba_cal_probe"] - r["ba_cal_twin"] > 0) for r in reps]
    cc = np.array([sum(1 for t, w in cal if (t, w) == key) for key in ((1, 1), (1, 0), (0, 1), (0, 0))])
    for b in B_EMPIRICAL:
        s = [(r["s1"][str(b)]["upper_half"], r["ba_a"] - r["ba_twin"] > 0) for r in reps]
        mc = np.array([sum(1 for x, y in s if (x, y) == key) for key in ((1, 1), (1, 0), (0, 1), (0, 0))])
        draws = []
        for _ in range(n_draws):
            draws.append(joint_count_power(rng.dirichlet(cc + 0.5)) * joint_count_power(rng.dirichlet(mc + 0.5)))
        out[str(b)] = {"main_cell_counts": mc.tolist(), "cal_cell_counts": cc.tolist(),
                       "plug_in": joint_count_power(cc / cc.sum()) * joint_count_power(mc / mc.sum()),
                       "posterior_median": float(np.median(draws)),
                       "posterior_lower": float(np.quantile(draws, LOWER_QUANTILE)),
                       "s1_upper_half_fraction": float(np.mean([x for x, _ in s])),
                       "replications_with_s1_ties": int(sum(1 for r in reps if r["s1"][str(b)]["ties_with_observed"]))}
    out["method"] = ("model-free check: per-replication outcome cells observed with the actual cyclic group and "
                     "tie-break, Jeffreys Dirichlet(counts + 1/2) posterior, exact decision power per draw")
    return out


def direct_simulation(th: dict, b: int, *, n_sim: int = N_DIRECT, seed: int = 20261106) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    mu, C = _main_cov(th, b, 0.0)
    m = rng.multivariate_normal(mu, C, size=(n_sim, D.N_REPLICATES), method="eigh")
    c = rng.multivariate_normal(np.array(th["cal_mean"]), _psd(np.array(th["cal_cov"])), size=(n_sim, D.N_REPLICATES),
                                method="eigh")
    supported = 0
    states = {f"r{i:02d}": "VALID" for i in range(D.N_REPLICATES)}
    for s in range(n_sim):
        eps = {}
        for i in range(D.N_REPLICATES):
            a, t, p = m[s, i, 0], m[s, i, 1], m[s, i, 2:]
            tb = rng.permutation(b + 1)
            rank = 1 + sum(1 for j in range(b) if (p[j], tb[j + 1]) < (a, tb[0]))
            pr, tw, mo = c[s, i]
            eps[f"r{i:02d}"] = {"C_task": pr - 0.5 - D.KAPPA * max(mo - 0.5, 0.0), "C_twin": pr - tw,
                                "S1": rank - (b + 2) / 2, "S2": a - t, "raw_gap": 0.0, "S1_margin": 0.0}
        dec = ST.analyse(eps, states, alpha=D.ALPHA, n_required=D.N_REPLICATES, precheck_ok=True)
        supported += dec["decision"] == "SUPPORTED"
    return {"n_sim": n_sim, "B": b, "p_supported": supported / n_sim, "ci95": cp_interval(supported, n_sim),
            "method": ("outer Monte Carlo of the complete decision procedure (statistics.analyse) on 20 replicates "
                       "drawn from the fitted model, with a fresh uniform tie-break per replicate")}


def sensitivity(th: dict, n: int) -> list[dict]:
    rows = []
    base = {"kappa": D.KAPPA, "scale": 1.0, "tau": 0.0, "tie_rate": 0.0, "failure_rate": 0.0, "cal_corr": None}
    grid = ([("kappa", v) for v in KAPPAS] + [("scale", v) for v in EFFECT_SCALES] + [("tau", v) for v in TAUS]
            + [("tie_rate", v) for v in TIE_RATES] + [("failure_rate", v) for v in FAILURE_RATES]
            + [("cal_corr", v) for v in CAL_CORRELATIONS])
    for b in B_CANDIDATES:
        for name, v in grid:
            cfg = {**base, name: v}
            r = power_at(th, b, **cfg)
            rows.append({"n": n, "B": b, "varied": name, **cfg, **{k: r[k] for k in (
                "p_supported", "p_C", "p_S1_and_S2", "p_supported_if_independent", "p_supported_frechet_lower",
                "q_positive")}})
        pess = {"kappa": D.KAPPA, "scale": 0.75, "tau": 0.002, "tie_rate": 0.05, "failure_rate": 0.005,
                "cal_corr": None}
        r = power_at(th, b, **pess)
        rows.append({"n": n, "B": b, "varied": "combined_pessimistic", **pess,
                     **{k: r[k] for k in ("p_supported", "p_C", "p_S1_and_S2", "p_supported_if_independent",
                                          "p_supported_frechet_lower", "q_positive")}})
    return rows


def power_analysis(pf: dict, *, n_boot: int = N_BOOT, direct: bool = True) -> dict:
    by_n = {}
    for r in pf["replicates"]:
        by_n.setdefault(r["n"], []).append(r)
    per_n = {}
    for n in sorted(by_n):
        reps = by_n[n]
        th = estimate(reps)
        plug = {str(b): power_at(th, b) for b in B_CANDIDATES}
        mc_repeat = {str(b): power_at(th, b, seed=77)["p_supported"] for b in B_CANDIDATES}
        boot = bootstrap(reps, n_boot=n_boot)
        per_n[str(n)] = {"replications": len(reps), "estimates": th, "plug_in": plug,
                         "plug_in_inner_seed_repeat": mc_repeat, "bootstrap": boot,
                         "empirical": empirical_check(reps), "sensitivity": sensitivity(th, n),
                         "ties": {"replications_with_s1_tie_with_observed": {
                             str(b): int(sum(1 for r in reps if r["s1"][str(b)]["ties_with_observed"]))
                             for b in B_EMPIRICAL},
                             "replications_with_ties_among_permuted": {
                             str(b): int(sum(1 for r in reps if r["s1"][str(b)]["ties_among_permuted"]))
                             for b in B_EMPIRICAL}},
                         "probe_recovery_fraction": float(np.mean([(r["ba_cal_probe"] - 0.5) / (r["ba_cal_model"] - 0.5)
                                                                   for r in reps if r["ba_cal_model"] > 0.5])),
                         "synthetic_model_accuracy": {"mean": th["cal_mean"][2],
                                                      "sd": math.sqrt(th["cal_cov"][2][2])}}
    chosen_n = next((n for n in sorted(by_n) if per_n[str(n)]["bootstrap"][str(max(B_CANDIDATES))]["lower"]
                     >= POWER_TARGET), None)
    chosen_b = None
    if chosen_n is not None:
        p = per_n[str(chosen_n)]
        top = p["plug_in"][str(max(B_CANDIDATES))]["p_supported"]
        chosen_b = next((b for b in B_CANDIDATES if top - p["plug_in"][str(b)]["p_supported"] < 0.01
                         and p["bootstrap"][str(b)]["lower"] >= POWER_TARGET), None)
        if direct:
            p["direct_simulation"] = direct_simulation(p["estimates"], chosen_b or max(B_CANDIDATES))
    selection = {"main_pairs": chosen_n, "permutations": chosen_b,
                 "n_rule": D.DESIGN["sampling"]["main_pairs_rule"], "b_rule": D.DESIGN["sampling"]["permutations_rule"],
                 "lower_bound": f"{LOWER_QUANTILE:g} quantile of the bootstrap distribution of P(SUPPORTED)",
                 "status": "SELECTED" if chosen_n and chosen_b else "NO_CANDIDATE_QUALIFIES"}
    return {"reference_scenario": {
                "oracle_effect": SESOI, "kappa": D.KAPPA, "tau": 0.0, "tie_rate": 0.0, "failure_rate": 0.0,
                "S1_S2": ("the synthetic target carries a population oracle balanced-accuracy effect of SESOI; the twin "
                          "and the permuted arms carry none; the probe-level means, variances and covariances of "
                          "BA(A), BA(T) and BA(P_b) are estimated from the independent synthetic replications at "
                          "each n, so the n-dependent estimation loss is included"),
                "C": ("calibration means and covariance of (probe BA, twin BA, model BA) estimated from the same "
                      f"replications, with the synthetic model's sign-rule accuracy set to {D.PUBLISHED_ACCURACY} "
                      "in each replication's own geometry and the calibration size fixed at the design's 50000 per "
                      "class for every n")},
            "levels": {"independent_synthetic_pipeline_replications": pf["independent_pipeline_replications"],
                       "inner_draws_per_power_evaluation": N_INNER,
                       "bootstrap_resamples_per_n": n_boot,
                       "outer_decision_simulations_direct_check": N_DIRECT if direct else 0,
                       "decision_power": ("exact: the 20-replicate counts of the four endpoint signs are multinomial "
                                          "in the per-replicate outcome cells, and P(C) and P(S1 and S2) are summed "
                                          "exactly over all count vectors (C is computed on calibration data that "
                                          "are independent of the main data)")},
            "per_n": per_n, "selection": selection}


def run(fractions, *, reps_per_n: dict, workers: int, n_boot: int = N_BOOT) -> dict:
    pf = pipeline_faithful(fractions, reps_per_n=reps_per_n, workers=workers)
    null = null_check(fractions, workers=workers)
    return {"report": "AutoND CE3 power after the S1 rank correction: pipeline-faithful replications, parameter "
                      "uncertainty and sensitivity",
            "evidentiary": False, "real_representations_used": False, "class_fractions": list(fractions),
            "class_fraction_source": D.P1_RECORD_RELPATH, "sesoi": SESOI,
            "published_accuracy": D.PUBLISHED_ACCURACY, "published_accuracy_source": D.PUBLISHED_ACCURACY_SOURCE,
            "n_candidates": list(N_CANDIDATES), "b_candidates": list(B_CANDIDATES), "power_target": POWER_TARGET,
            "probe": D.PROBE, "kappa": D.KAPPA, "alpha": D.ALPHA, "n_replicates": D.N_REPLICATES,
            "critical_count": critical_count(D.N_REPLICATES),
            "s1_procedure": D.DESIGN["endpoints"]["S1"], "zero_policy": D.DESIGN["statistics"]["zero_policy"],
            "permutation_construction": D.DESIGN["sampling"]["permutation_construction"],
            "synthetic_model": ("64 ReLU features relu(G M) with Gaussian G and a random mixing matrix M drawn per "
                                "replication; the target class is an ordinal coarsening of a noisy linear score of "
                                "the features with the P1 class fractions; the twin is an independent signal-free "
                                "ReLU representation; the calibration labels follow a logistic model in a linear "
                                "score whose sign rule reaches the published accuracy; the CE3 fold, permutation "
                                "group, tie-break, probe and exact balanced-accuracy code is run unchanged"),
            "s1_null_check": null, "pipeline_faithful": pf, **power_analysis(pf, n_boot=n_boot)}
