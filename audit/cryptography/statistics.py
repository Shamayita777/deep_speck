"""
Underflow-safe p-value reporting and replicate-level aggregation.

A p-value is never reported as literal 0.0: with large samples the true
value underflows double precision, and printing "p = 0" asserts an exact
zero probability that was not computed. We report the floor actually
supported by the arithmetic instead.
"""

from __future__ import annotations

import math

import numpy as np

#: Smallest positive double; any computed p at or below this underflowed.
P_UNDERFLOW_FLOOR = 1e-300
P_CONVENTION = (
    "p-values are computed in double precision. A computed value at or below 1e-300 has "
    "underflowed; it is reported as p_value=null with p_underflow=true, "
    "p_upper_bound=1e-300 and p_representation='p < 1e-300'. It is NEVER reported as the "
    "number 0.0, which would assert an exact zero probability that was not computed. "
    "log10_p is reported where finite."
)


def p_value_report(p: float) -> dict:
    """Structured, honest representation of a p-value."""
    p = float(p)
    if not math.isfinite(p) or p < 0.0:
        return {"p_value": None, "p_underflow": False, "p_representation": "undefined",
                "p_upper_bound": None, "log10_p": None, "convention": P_CONVENTION}
    if p <= P_UNDERFLOW_FLOOR:
        # p_value is None, never 0.0: a numeric zero would be read by any
        # downstream consumer as an exact zero probability, which was not
        # computed. The bound that WAS established is reported instead.
        return {"p_value": None, "p_underflow": True,
                "p_upper_bound": P_UNDERFLOW_FLOOR,
                "p_representation": f"p < {P_UNDERFLOW_FLOOR:g}",
                "log10_p": None, "convention": P_CONVENTION}
    return {"p_value": p, "p_underflow": False, "p_upper_bound": None,
            "p_representation": f"p = {p:.6g}", "log10_p": math.log10(p),
            "convention": P_CONVENTION}


def replicate_level_summary(values, *, alpha: float = 0.05, rng=None,
                            n_bootstrap: int = 10000) -> dict:
    """
    Inference over INDEPENDENT REPLICATE values - never over CV folds.

    The statistical unit is one stabilized value per independent
    replicate. Folds nested inside a replicate are a variance-reduction
    device, not additional experimental units; treating 20x5 as 100
    observations would be pseudo-replication.
    """
    from scipy import stats

    v = np.asarray(list(values), dtype=float)
    n = v.size
    if n < 2:
        raise ValueError("replicate-level inference needs at least 2 independent replicates.")
    if not np.all(np.isfinite(v)):
        raise ValueError("replicate values contain non-finite entries.")
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    # Wilcoxon signed-rank against 0 (no distributional assumption on the
    # replicate values); t-CI reported alongside for interpretability.
    try:
        w = stats.wilcoxon(v, alternative="greater")
        w_p = float(w.pvalue)
    except ValueError:
        w_p = None
    rng = rng or np.random.default_rng(0)
    boot = rng.choice(v, size=(n_bootstrap, n), replace=True).mean(axis=1)
    return {
        "statistical_unit": "independent evaluation replicate",
        "n_replicates": int(n),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_t": [mean - tcrit * se, mean + tcrit * se],
        "ci95_bootstrap": [float(np.percentile(boot, 100 * alpha / 2)),
                           float(np.percentile(boot, 100 * (1 - alpha / 2)))],
        "effect_size_cohens_dz": (mean / sd) if sd > 0 else None,
        "wilcoxon_signed_rank": p_value_report(w_p) if w_p is not None else None,
        "raw_replicate_values": [float(x) for x in v],
    }
