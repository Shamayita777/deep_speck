"""
AutoND CE2 power: scenario-based sensitivity analysis of the complete decision rule.

What this is. For assumed run-level effects, the probability that the preregistered procedure
(exact two-sided sign test, then Holm over two one-sided control-separation sign tests) returns
SUPPORTED. The true effect, and the dependence between the published-model and twin-model
coefficients, are unknown, so every number is conditional on a stated scenario. It is not a
guarantee of power, and no exploratory coefficient from the published model is used.

Run-level model (``simulate_joint_power``), in units of se = 1/sqrt(N - 1):

    rho_primary,i = delta + e_p,i
    rho_perm,i    = e_q,i,                     corr(e_p, e_q) = perm_error_corr
    rho_twin,i    = twin_ratio * delta + e_t,i, corr(e_p, e_t) = twin_error_corr

with (e_p, e_q, e_t) jointly standard normal. Each estimate carries its Monte Carlo standard
error and a Clopper-Pearson 95% interval.

Checking the model (``pipeline_faithful_check``). The Gaussian model assumes that the Spearman
coefficient of N pairs with the real, heavily tied, discrete target has variance (N - 1)^-1 and is
close to normal, under the null and under small alternatives. The check simulates the actual
estimator (average ranks, no jitter) on the real model-free target vector, with synthetic scores
``score = lam * standardised(target) + noise``, and reports N * var(rho_hat), the skewness, and the
agreement between the simulated sign probability and Phi(mean / sd). The scores are synthetic and
model-free; nothing from the published model enters.
"""

from __future__ import annotations

import math

import numpy as np

from .analysis import average_ranks, binom_cdf


def _rejection_counts_two_sided(n: int, alpha: float) -> int:
    for k in range((n + 1) // 2, n + 1):
        if min(1.0, 2.0 * binom_cdf(n - k, n)) <= alpha:
            return k
    return n + 1


def primary_power_exact(n: int, p_positive: float, alpha: float) -> float:
    k = _rejection_counts_two_sided(n, alpha)
    pmf = [math.comb(n, i) * p_positive ** i * (1 - p_positive) ** (n - i) for i in range(n + 1)]
    return float(sum(pmf[i] for i in range(n + 1) if i >= k or i <= n - k))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval by bisection on the binomial tail (no SciPy)."""
    a = (1 - level) / 2

    def tail_ge(p, k):     # Pr[X >= k]
        if k <= 0:
            return 1.0
        lo_terms = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k)) if n <= 2000 else None
        if lo_terms is not None:
            return 1.0 - lo_terms
        # normal approximation with continuity correction for large n (Monte Carlo counts)
        mu, sd = n * p, math.sqrt(max(n * p * (1 - p), 1e-300))
        return 0.5 * math.erfc((k - 0.5 - mu) / (sd * math.sqrt(2)))

    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: tail_ge(p, k), a)
    # upper bound: Pr[X <= k; p] = a, i.e. the increasing Pr[X >= k + 1; p] = 1 - a
    upper = 1.0 if k == n else solve(lambda p: tail_ge(p, k + 1), 1 - a)
    return lower, upper


def _estimate(hits: np.ndarray) -> dict:
    n, k = int(hits.size), int(hits.sum())
    p = k / n
    lo, hi = clopper_pearson(k, n)
    return {"estimate": p, "mc_se": math.sqrt(p * (1 - p) / n), "ci95": [lo, hi], "n_sim": n,
            "ci95_method": ("Clopper-Pearson" if n <= 2000 else
                            "binomial tail inversion with a continuity-corrected normal tail (n_sim > 2000)")}


def simulate_joint_power(*, n_runs: int, delta: float, twin_ratio: float, twin_error_corr: float,
                         alpha: float, n_sim: int, seed: int, perm_error_corr: float = 0.0) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    cov = np.array([[1.0, perm_error_corr, twin_error_corr],
                    [perm_error_corr, 1.0, 0.0],
                    [twin_error_corr, 0.0, 1.0]])
    L = np.linalg.cholesky(cov)
    e = rng.standard_normal((n_sim, n_runs, 3)) @ L.T
    rp, rq, rt = delta + e[..., 0], e[..., 1], twin_ratio * delta + e[..., 2]
    k_hi = _rejection_counts_two_sided(n_runs, alpha)
    pos = (rp > 0).sum(axis=1)
    stage1 = (pos >= k_hi) | (pos <= n_runs - k_hi)
    tail = np.array([1.0 - binom_cdf(k - 1, n_runs) for k in range(n_runs + 1)])
    p1 = tail[(np.abs(rp) > np.abs(rq)).sum(axis=1)]
    p2 = tail[(np.abs(rp) > np.abs(rt)).sum(axis=1)]
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    holm_both = (lo <= alpha / 2) & (hi <= alpha)
    return {"n_runs": n_runs, "delta_se_units": delta, "twin_ratio": twin_ratio,
            "twin_error_corr": twin_error_corr, "perm_error_corr": perm_error_corr,
            "sign_probability": float(0.5 * math.erfc(-delta / math.sqrt(2))),
            "power_primary": _estimate(stage1), "power_supported": _estimate(stage1 & holm_both),
            "seed": seed}


def power_analysis(*, alpha: float = 0.05, n_sim: int = 50_000, seed: int = 20261030) -> dict:
    sign_probs = (0.65, 0.70, 0.75, 0.80, 0.85)
    primary = {str(n): {str(p): primary_power_exact(n, p, alpha) for p in sign_probs}
               for n in (10, 15, 20, 25, 30, 40)}
    joint, i = [], 0
    for n_runs in (10, 20, 30):
        for twin_ratio in (0.0, 0.25, 0.5, 0.75):
            for corr in (-0.5, 0.0, 0.5, 0.9):
                for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                    joint.append(simulate_joint_power(n_runs=n_runs, delta=d, twin_ratio=twin_ratio,
                                                      twin_error_corr=corr, alpha=alpha, n_sim=n_sim,
                                                      seed=seed + i))
                    i += 1
    perm_dep = [simulate_joint_power(n_runs=20, delta=d, twin_ratio=0.25, twin_error_corr=0.5,
                                     perm_error_corr=c, alpha=alpha, n_sim=n_sim, seed=seed + 10_000 + j)
                for j, (d, c) in enumerate((d, c) for d in (2.0, 3.0) for c in (-0.3, 0.0, 0.3))]
    return {"nature": "scenario-based sensitivity analysis; conditional on the stated run-level model",
            "alpha": alpha, "primary_exact_power": primary, "joint_power_simulation": joint,
            "permutation_dependence_sensitivity": perm_dep,
            "smallest_attainable_two_sided_p": {str(n): 2.0 / 2 ** n for n in (10, 15, 20, 25, 30, 40)},
            "rejection_threshold_positive_runs": {str(n): _rejection_counts_two_sided(n, alpha)
                                                   for n in (10, 15, 20, 25, 30, 40)}}


def required_delta(results: list, *, n_runs: int, twin_ratio: float, twin_error_corr: float,
                   target: float = 0.90) -> float | None:
    """Smallest simulated delta (se units) whose power_supported lower 95% bound reaches target."""
    rows = sorted((r for r in results if r["n_runs"] == n_runs and r["twin_ratio"] == twin_ratio
                   and r["twin_error_corr"] == twin_error_corr and r["perm_error_corr"] == 0.0),
                  key=lambda r: r["delta_se_units"])
    for r in rows:
        if r["power_supported"]["ci95"][0] >= target:
            return r["delta_se_units"]
    return None


# ----------------------------------------------------------- pipeline-faithful check


def _spearman_fast(score: np.ndarray, rt_centred: np.ndarray, rt_norm: float) -> float:
    rs = average_ranks(score)
    rs -= rs.mean()
    return float(rs @ rt_centred) / math.sqrt(float(rs @ rs)) / rt_norm


def pipeline_faithful_check(target: np.ndarray, *, n: int, lambdas=(0.0, 0.002, 0.005), n_sim: int = 400,
                            seed: int = 20261031) -> dict:
    """Simulate the actual Spearman estimator on the real discrete target (subsampled to n)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    t = np.asarray(target, dtype=np.float64)
    out = {"n": n, "n_sim": n_sim, "seed": seed, "target_source": "model-free genuine-pair trajectories",
           "score_model": "lam * standardised(target) + N(0, 1) noise (synthetic, model-free)", "rows": []}
    for lam in lambdas:
        rhos = np.empty(n_sim)
        for s in range(n_sim):
            tt = t[rng.integers(0, t.size, n)]
            z = (tt - tt.mean()) / tt.std()
            score = lam * z + rng.standard_normal(n)
            rt = average_ranks(tt)
            rt -= rt.mean()
            rhos[s] = _spearman_fast(score, rt, math.sqrt(float(rt @ rt)))
        m, sd = float(rhos.mean()), float(rhos.std(ddof=1))
        skew = float(np.mean(((rhos - m) / sd) ** 3))
        out["rows"].append({
            "lambda": lam, "mean_rho": m, "sd_rho": sd, "n_minus_1_times_var": (n - 1) * sd * sd,
            "n_minus_1_times_var_mc_se": (n - 1) * sd * sd * math.sqrt(2.0 / (n_sim - 1)),
            "skewness": skew, "simulated_sign_probability": float(np.mean(rhos > 0)),
            "gaussian_sign_probability": float(0.5 * math.erfc(-(m / sd) / math.sqrt(2))),
            "delta_se_units": m * math.sqrt(n - 1)})
    return out
