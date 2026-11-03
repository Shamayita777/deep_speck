"""
Non-evidentiary sensitivity analysis of the AutoND EV paired design (planning only, never a decision input).

For each number of VALID pairs n, each multiple of the calibrated sigma upper bounds and each state of the two
primary hypotheses, it simulates the complete registered rule exactly as framework.power does (paired t ->
difference Holm -> TOST -> equivalence Holm -> decide_final semantics) and reports, per cell, the marginal
probability of each hypothesis's correct final decision, the joint probability of both, the phi correlation
of the two correctness indicators, a Wilson 95 % interval and the 3 x 3 counts of final decisions. It repeats
the grid for non-normal paired differences with the same mean and SD:

    normal        N(0, 1)
    t5, t3        Student t / sqrt(v / (v - 2))
    contaminated  0.9 N(0, 1) + 0.1 N(0, 3^2), divided by sqrt(1.8)
    lognormal     (LogN(0, 0.8) - exp(0.32)) / sqrt((exp(0.64) - 1) exp(0.64))

Every cell has its own PCG64 seed (SEED_BASE + 100000 n + 10 round(1000 m) + scenario + 7 distribution).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats

from autond.baseline import PRIMARY_FAMILY
from framework.power import _holm_two
from framework.provenance import utc_timestamp

ARTIFACT = "autond-ev-power-sensitivity-1"
SEED_BASE = 20261103
SCENARIOS = {"both_effect": (1, 1), "first_effect_second_null": (1, 0), "first_null_second_effect": (0, 1),
             "both_null": (0, 0)}
DISTRIBUTIONS = ("normal", "t5", "t3", "contaminated", "lognormal")


def standardised(rng, dist: str, shape) -> np.ndarray:
    if dist == "normal":
        return rng.standard_normal(shape)
    if dist in ("t3", "t5"):
        v = int(dist[1:])
        return rng.standard_t(v, shape) / math.sqrt(v / (v - 2))
    if dist == "contaminated":
        c = rng.random(shape) < 0.10
        return np.where(c, 3.0 * rng.standard_normal(shape), rng.standard_normal(shape)) / math.sqrt(1.8)
    if dist == "lognormal":
        s = 0.8
        return (rng.lognormal(0.0, s, shape) - math.exp(s * s / 2)) / math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
    raise ValueError(f"unknown distribution {dist!r}")


def _decisions(n: int, effects: dict, sigmas: dict, dist: str, seed: int, n_sim: int, alpha: float, epsilon: float):
    rng = np.random.Generator(np.random.PCG64(seed))
    df, per = n - 1, {}
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    for h in PRIMARY_FAMILY:
        d = effects[h] + sigmas[h] * standardised(rng, dist, (n_sim, n))
        mean = d.mean(1)
        se = np.where(d.std(1, ddof=1) == 0, np.finfo(float).tiny, d.std(1, ddof=1) / math.sqrt(n))
        per[h] = {"p": 2 * stats.t.sf(np.abs(mean / se), df),
                  "pt": np.maximum(stats.t.sf((mean + epsilon) / se, df), stats.t.cdf((mean - epsilon) / se, df)),
                  "lo": mean - tcrit * se, "hi": mean + tcrit * se}
    a, b = PRIMARY_FAMILY
    adj_d = dict(zip(PRIMARY_FAMILY, _holm_two(per[a]["p"], per[b]["p"], alpha)))
    adj_t = dict(zip(PRIMARY_FAMILY, _holm_two(per[a]["pt"], per[b]["pt"], alpha)))
    out = {}
    for h in PRIMARY_FAMILY:
        sig, eq = adj_d[h] < alpha, adj_t[h] < alpha
        outside = (per[h]["lo"] > epsilon) | (per[h]["hi"] < -epsilon)
        out[h] = np.where(sig & outside, "S", np.where(~sig & eq, "N", "I"))
    return out


def _wilson(k: int, n: int) -> list[float]:
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def cell(n: int, multiplier: float, scenario: str, dist: str, *, sigmas: dict, target: float, epsilon: float,
         alpha: float, n_sim: int) -> dict:
    ea, eb = SCENARIOS[scenario]
    a, b = PRIMARY_FAMILY
    effects, sig = {a: ea * target, b: eb * target}, {h: sigmas[h] * multiplier for h in PRIMARY_FAMILY}
    seed = SEED_BASE + 100000 * n + 10 * round(1000 * multiplier) + list(SCENARIOS).index(scenario) \
        + 7 * DISTRIBUTIONS.index(dist)
    dec = _decisions(n, effects, sig, dist, seed, n_sim, alpha, epsilon)
    want = {a: "S" if ea else "N", b: "S" if eb else "N"}
    ca, cb = dec[a] == want[a], dec[b] == want[b]
    pa, pb, pj = float(ca.mean()), float(cb.mean()), float((ca & cb).mean())
    phi: Optional[float] = (pj - pa * pb) / math.sqrt(pa * (1 - pa) * pb * (1 - pb)) \
        if 0 < pa < 1 and 0 < pb < 1 else None
    return {"n_valid_pairs": n, "sigma_multiplier": multiplier, "scenario": scenario, "distribution": dist,
            "seed": seed, "n_simulations": n_sim, "true_means": effects, "sigmas": sig,
            "correct": {a: pa, b: pb, "joint": pj},
            "wilson_95": {a: _wilson(int(ca.sum()), n_sim), b: _wilson(int(cb.sum()), n_sim),
                          "joint": _wilson(int((ca & cb).sum()), n_sim)},
            "phi_correlation": phi,
            "decision_counts": {f"{x}{y}": int(((dec[a] == x) & (dec[b] == y)).sum()) for x in "SNI" for y in "SNI"}}


def analyse(variance: dict, config: dict, *, n_values=(3, 4, 5, 6), multipliers=(1.0, 1.25, 1.5, 2.0),
            n_sim: int = 100_000) -> dict:
    d = config["decisions"]
    kw = {"sigmas": variance["sigma_Delta_upper_95"], "target": d["target_effect"], "epsilon": d["practical_threshold"],
          "alpha": d["alpha"], "n_sim": n_sim}
    grid = [cell(n, m, s, "normal", **kw) for m in multipliers for n in n_values for s in SCENARIOS]
    robust = [cell(n, 1.0, s, dist, **kw) for dist in DISTRIBUTIONS for n in (3, 5, 6) for s in SCENARIOS]
    worst = {}
    for c in grid:
        key = f"n={c['n_valid_pairs']},sigma_x{c['sigma_multiplier']}"
        worst[key] = min(worst.get(key, 1.0), c["correct"]["joint"])
    return {"artifact": ARTIFACT, "non_evidentiary": True, "generated_at_utc": utc_timestamp(),
            "variance_execution_order_protocol": variance.get("execution_order_protocol") or "fixed-ab-legacy",
            "inputs": {"sigma_Delta_upper_95": kw["sigmas"], "target_effect": kw["target"], "epsilon": kw["epsilon"],
                       "alpha": kw["alpha"], "n_simulations_per_cell": n_sim, "seed_base": SEED_BASE},
            "worst_joint_correct_by_cell": worst, "grid": grid, "non_normal": robust,
            "statement": ("planning sensitivity only; it neither changes the registered power plan nor validates "
                          "sigma_Delta under an execution order other than the one its calibration used")}
