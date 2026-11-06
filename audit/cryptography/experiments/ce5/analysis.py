"""
G-CE5 statistics. Per-replicate quantities are computed from persisted raw
arrays; inference uses replicate-level values only (docs/STATISTICS.md).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from audit.cryptography.ce234 import frozen_plan as P

OUTCOMES = ("DISTRIBUTION_GREATER", "TRAIL_GREATER", "EQUIVALENT", "INCONCLUSIVE")


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def fisher_z(r: float) -> float:
    return float(np.arctanh(r))


def normal_scores(x) -> np.ndarray:
    """van der Waerden scores; average ranks for ties."""
    x = np.asarray(x, dtype=np.float64)
    return stats.norm.ppf(stats.rankdata(x) / (x.size + 1))


def resolution_matched(D, T) -> np.ndarray:
    """D's order with T's tie groups: the k-th smallest D receives the k-th
    smallest average rank of T."""
    out = np.empty(len(D), dtype=np.float64)
    out[np.argsort(np.asarray(D), kind="stable")] = np.sort(stats.rankdata(T))
    return out


def _basis(cols) -> np.ndarray:
    parts = [np.ones(cols[0].size)]
    for c in cols:
        parts += [c, c ** 2, c ** 3]
    return np.column_stack(parts)


def incremental_value(N, T, D) -> dict:
    """Out-of-sample MSE for predicting normal scores of N from T, D and both.
    First half of the pairs fits, second half evaluates; scores are computed
    within each half so the evaluation half never informs the fit."""
    N, T, D = (np.asarray(a, dtype=np.float64) for a in (N, T, D))
    h = N.size // 2
    halves = [slice(0, h), slice(h, 2 * h)]
    z = [{k: normal_scores(v[s]) for k, v in (("N", N), ("T", T), ("D", D))} for s in halves]

    def design_matrix(q, model):
        if model == "T":
            return _basis([q["T"]])
        if model == "D":
            return _basis([q["D"]])
        return np.column_stack([_basis([q["T"], q["D"]]), q["T"] * q["D"]])

    loss = {}
    for model in ("T", "D", "TD"):
        beta, *_ = np.linalg.lstsq(design_matrix(z[0], model), z[0]["N"], rcond=None)
        resid = z[1]["N"] - design_matrix(z[1], model) @ beta
        loss[model] = float(np.mean(resid ** 2))
    return {"loss": loss, "delta_D": loss["T"] - loss["TD"], "delta_T": loss["D"] - loss["TD"]}


def replicate_statistics(real: dict, random: dict) -> dict:
    """real: N, T, D, LR on real pairs; random: N, D on random pairs."""
    r_nd = spearman(real["N"], real["D"])
    r_nt = spearman(real["N"], real["T"])
    r_nl = spearman(real["N"], real["LR"])
    r_nm = spearman(real["N"], resolution_matched(real["D"], real["T"]))
    out = {
        "rho_N_D_real": r_nd, "rho_N_T_real": r_nt, "rho_N_LR_real": r_nl, "rho_N_Dmatched_real": r_nm,
        "rho_N_D_random": spearman(random["N"], random["D"]),
        "z_D": fisher_z(r_nd), "z_T": fisher_z(r_nt),
        "delta": fisher_z(r_nd) - fisher_z(r_nt),
        "delta_matched": fisher_z(r_nm) - fisher_z(r_nt),
        "delta_LR": fisher_z(r_nl) - fisher_z(r_nt),
        "distinct_T": int(np.unique(real["T"]).size),
    }
    iv = incremental_value(real["N"], real["T"], real["D"])
    out.update(incremental_loss=iv["loss"], delta_D=iv["delta_D"], delta_T=iv["delta_T"])
    return out


def median_interval(values, conf: float = 0.95) -> dict:
    """Order-statistic interval for the median: (x_(k), x_(R-k+1)) with the
    largest k such that P(Bin(R, 1/2) < k) <= (1 - conf) / 2."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    R = v.size
    tail = (1 - conf) / 2
    k = 0
    while k + 1 <= R and stats.binom.cdf(k, R, 0.5) <= tail:
        k += 1
    if k == 0:
        return {"lower": None, "upper": None, "coverage": None, "k": 0, "R": R}
    coverage = 1 - 2 * stats.binom.cdf(k - 1, R, 0.5)
    return {"lower": float(v[k - 1]), "upper": float(v[R - k]), "coverage": float(coverage),
            "k": k, "R": R, "median": float(np.median(v))}


def classify(lower, upper, margin: float) -> str:
    if lower is None or upper is None:
        return "INCONCLUSIVE"
    if -margin < lower and upper < margin:
        return "EQUIVALENT"
    if lower >= margin:
        return "DISTRIBUTION_GREATER"
    if upper <= -margin:
        return "TRAIL_GREATER"
    return "INCONCLUSIVE"


def qualifier(lower, upper, margin: float) -> str | None:
    """Descriptive only, for INCONCLUSIVE: a direction without a demonstrated
    practically meaningful size."""
    if classify(lower, upper, margin) != "INCONCLUSIVE" or lower is None:
        return None
    if lower > 0:
        return "DIRECTION_D"
    if upper < 0:
        return "DIRECTION_T"
    return "UNDETERMINED"


def analyse(reps: list[dict], *, margin: float, minimum_replicates: int,
            sensitivity_margins=()) -> dict:
    valid = [r for r in reps if r.get("status") == "OK"]
    if len(valid) < minimum_replicates:
        return {"decision": "INCONCLUSIVE", "n_valid": len(valid),
                "reason": f"{len(valid)} valid replicates < registered {minimum_replicates}"}
    deltas = [r["delta"] for r in valid]
    ci = median_interval(deltas)
    decision = classify(ci["lower"], ci["upper"], margin)
    qual = qualifier(ci["lower"], ci["upper"], margin)
    sec = {}
    for key in ("delta_matched", "delta_LR"):
        iv = median_interval([r[key] for r in valid])
        sec[key] = {"interval": iv, "outcome": classify(iv["lower"], iv["upper"], margin),
                    "role": "secondary, descriptive"}
    sec["margin_sensitivity"] = {str(m): classify(ci["lower"], ci["upper"], m) for m in sensitivity_margins}
    fam = {}
    for key in ("delta_D", "delta_T"):
        vals = [r[key] for r in valid]
        sec[key] = {"median": float(np.median(vals)), **P.exact_sign_test(vals)}
        fam[key] = sec[key]["p_value"]
    sec["holm_adjusted_p"] = P.holm(fam)
    for key in ("rho_N_D_real", "rho_N_T_real", "rho_N_LR_real", "rho_N_Dmatched_real", "rho_N_D_random"):
        vals = [r[key] for r in valid]
        sec[key] = {"median": float(np.median(vals)), "min": float(min(vals)), "max": float(max(vals))}
    return {"decision": decision, "qualifier": qual, "n_valid": len(valid), "primary": {
        "deltas": deltas, "interval": ci, "margin": margin, **P.exact_sign_test(deltas)},
        "secondary": sec}


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def gate(tp: int, fn: int, tn: int, fp: int, spec: dict) -> dict:
    """spec from design.gate_spec: acceptance intervals for accuracy, TPR and
    TNR, and the separation threshold."""
    n = tp + fn + tn + fp
    acc = clopper_pearson(tp + tn, n)
    tpr = clopper_pearson(tp, tp + fn)
    tnr = clopper_pearson(tn, tn + fp)

    def inside(ci, iv):
        return iv[0] <= ci[0] and ci[1] <= iv[1]
    reproduced = inside(acc, spec["accuracy"]) and inside(tpr, spec["tpr"]) and inside(tnr, spec["tnr"])
    separated = acc[0] > spec["separation_lower"]
    return {"accuracy": (tp + tn) / n, "tpr": tp / (tp + fn), "tnr": tn / (tn + fp), "n": n,
            "cp95": {"accuracy": list(acc), "tpr": list(tpr), "tnr": list(tnr)},
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
            "acceptance": {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()},
            "reproduced": reproduced, "separated": separated,
            "decision": "PASS" if reproduced else "FAIL"}


PLANNING_SIMS = 50_000
SIMULTANEOUS_ALPHA = 0.01
REQUIREMENTS = (("delta=0", "EQUIVALENT"), ("delta=+2m", "DISTRIBUTION_GREATER"), ("delta=-2m", "TRAIL_GREATER"))
CELL_ALPHA = SIMULTANEOUS_ALPHA / len(REQUIREMENTS)


def planning_draws(R: int, sims: int = PLANNING_SIMS, seed: int = 0) -> np.ndarray:
    """Standard normal draws shared by every true Delta at a given R."""
    return np.random.default_rng([seed, R]).standard_normal(size=(sims, R))


def outcome_probabilities(sd: float, R: int, margin: float, true_delta: float, *,
                          sims: int = PLANNING_SIMS, seed: int = 0, draws: np.ndarray | None = None) -> dict:
    """Monte Carlo frequency of each registered outcome under
    Delta_r ~ N(true_delta, sd^2), computed as true_delta + sd * Z on the
    supplied draws Z (common random numbers), with its standard error and a
    one-sided CELL_ALPHA Clopper-Pearson lower bound."""
    z = planning_draws(R, sims, seed) if draws is None else draws
    n = z.shape[0]
    x = np.sort(true_delta + sd * z, axis=1)
    k = median_interval(np.zeros(R))["k"]
    lo, hi = x[:, k - 1], x[:, R - k]
    eq = (lo > -margin) & (hi < margin)
    dg = ~eq & (lo >= margin)
    tg = ~eq & (hi <= -margin)
    out = {}
    for name, hit in (("EQUIVALENT", eq), ("DISTRIBUTION_GREATER", dg), ("TRAIL_GREATER", tg),
                      ("INCONCLUSIVE", ~(eq | dg | tg))):
        c = int(hit.sum())
        lower = 0.0 if c == 0 else float(stats.beta.ppf(CELL_ALPHA, c, n - c + 1))
        out[name] = {"count": c, "p": c / n, "mc_se": float(np.sqrt(c / n * (1 - c / n) / n)), "cp_lower": lower}
    out["sims"] = n
    return out


def robust_sd(values) -> float:
    """1.4826 * MAD, a consistent estimator of sigma for normal data."""
    v = np.asarray(values, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def planning_sd(values, level: float = 0.80) -> dict:
    """Conservative planning spread.

    u_sd = s * sqrt(df / chi2_{1-level, df}) is the exact one-sided `level`
    upper confidence bound for sigma when the values are i.i.d. normal.
    u_mad applies the same factor to 1.4826 * MAD; it has no exact coverage
    and only guards against a sample SD deflated by the data.
    sigma_plan = max(u_sd, u_mad) >= u_sd, so P(sigma <= sigma_plan) >= level
    under normality; no coverage is claimed beyond that."""
    v = np.asarray(values, dtype=np.float64)
    df = v.size - 1
    factor = float(np.sqrt(df / stats.chi2.ppf(1 - level, df)))
    s, r = float(np.std(v, ddof=1)), robust_sd(v)
    return {"n": int(v.size), "sample_sd": s, "robust_sd": r, "factor": factor, "u_sd": s * factor,
            "u_mad": r * factor, "sigma_plan": max(s, r) * factor, "level": level,
            "coverage_claim": f"P(sigma <= sigma_plan) >= {level} for i.i.d. normal values; none otherwise"}


def sd_upper_bound(values, level: float = 0.80) -> float:
    return planning_sd(values, level)["sigma_plan"]


def choose_replicates(sd: float, margin: float, grid=range(20, 61, 5), target: float = 0.80,
                      sims: int = PLANNING_SIMS, seed: int = 0) -> dict:
    """Smallest R whose three registered requirements all have a one-sided
    Clopper-Pearson lower bound >= target at level CELL_ALPHA = 0.01 / 3
    (Bonferroni). For a given R, the probability that any of the three true
    probabilities lies below its bound is at most SIMULTANEOUS_ALPHA. No
    guarantee is claimed jointly across the R grid. All conditions at one R
    use the same draws (common random numbers)."""
    table = {}
    for R in grid:
        z = planning_draws(R, sims, seed)
        table[R] = {"delta=0": outcome_probabilities(sd, R, margin, 0.0, draws=z),
                    "delta=+2m": outcome_probabilities(sd, R, margin, 2 * margin, draws=z),
                    "delta=-2m": outcome_probabilities(sd, R, margin, -2 * margin, draws=z),
                    "delta=+m": outcome_probabilities(sd, R, margin, margin, draws=z),
                    "delta=+m/2": outcome_probabilities(sd, R, margin, margin / 2, draws=z)}

    def ok(R):
        return all(table[R][cond][out]["cp_lower"] >= target for cond, out in REQUIREMENTS)

    passing = [R for R in grid if ok(R)]
    return {"chosen": passing[0] if passing else max(grid), "met_target": bool(passing),
            "rule": (f"all of {[o for _, o in REQUIREMENTS]} with one-sided Clopper-Pearson lower bound >= {target} "
                     f"at level {CELL_ALPHA:.5f} each (Bonferroni, simultaneous {1 - SIMULTANEOUS_ALPHA:.0%} per R)"),
            "outcome_probabilities": table, "sd_used": sd, "margin": margin, "target": target, "sims": sims}


def choose_pairs(seconds_per_real_pair: float, n_replicates: int, *, bytes_per_real_pair: float = 0.0,
                 budget_hours: float = 12.0, budget_bytes: float = 10e9,
                 grid=(10_000, 30_000, 100_000, 300_000, 1_000_000)) -> dict:
    """seconds_per_real_pair covers one real and one random pair: generation,
    inference and scoring."""
    hours = {n: n * n_replicates * seconds_per_real_pair / 3600 for n in grid}
    disk = {n: n * n_replicates * bytes_per_real_pair for n in grid}
    ok = [n for n in grid if hours[n] <= budget_hours and disk[n] <= budget_bytes]
    return {"chosen": max(ok) if ok else None, "projected_hours": hours, "projected_bytes": disk,
            "budget_hours": budget_hours, "budget_bytes": budget_bytes}
