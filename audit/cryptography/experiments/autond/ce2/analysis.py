"""
AutoND CE2 run-level measurement, inference and decision.

Statistical unit: one independent evaluation run (a fresh set of genuine difference-bearing pairs).
The pairs inside a run estimate that run's Spearman coefficient; they are never treated as
replicates in the confirmatory inference.

Procedure (hierarchical, familywise alpha):
  Stage 1  exact two-sided sign test over the nonzero run-level primary coefficients.
  Stage 2  only if Stage 1 rejects: for each control c, D_i = |rho_primary,i| - |rho_c,i|, exact
           one-sided sign test (H1: Pr[D > 0] > 1/2), Holm across the two control tests.
Serial gatekeeping (Stage 2 is tested only after Stage 1 rejects at alpha, and Holm controls the
Stage 2 family at alpha) bounds the familywise error rate of the whole procedure by alpha.

Exact arithmetic is used for every confirmatory p-value; SciPy is not used.

A sign test with no nonzero value (n_effective = 0) has no defined parameter: p_+ (or the control's
sign probability) is undefined. It is reported with status NO_EFFECTIVE_OBSERVATIONS and
p_value None, never as p = 1.0, and it never rejects. The runs stay VALID.

Undefined coefficients raise ``UndefinedCorrelation`` (a ``core.theory_consistency.
UndefinedAssociation``, deliberately neither a ValueError nor a RuntimeError), carrying an
``AssociationStatus``. Within-run p-values are ``core.numerics.PValue`` records; an underflow is
reported as UNDERFLOW with its log10, and |rho| = 1 as DEGENERATE, never as 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ....core.hypothesis import AssociationStatus
from ....core.numerics import correlation_t_approximation
from ....core.theory_consistency import UndefinedAssociation

SUPPORTED = "SUPPORTED"
ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED = "ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED"
INCONCLUSIVE = "INCONCLUSIVE"
PRECHECK_FAILED = "PRECHECK_FAILED"
DECISIONS = (SUPPORTED, ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED, INCONCLUSIVE, PRECHECK_FAILED)

TESTED = "TESTED"
NO_EFFECTIVE_OBSERVATIONS = "NO_EFFECTIVE_OBSERVATIONS"

CONDITIONS = ("primary", "permuted_target", "untrained_twin")
CONTROLS = ("permuted_target", "untrained_twin")


class UndefinedCorrelation(UndefinedAssociation):
    """Spearman's coefficient is undefined. Not a ValueError, so a caller that maps ValueError to a
    numeric score (as the legacy framework test does) cannot turn it into 0.0."""

    def __init__(self, reason: str, status: AssociationStatus = AssociationStatus.UNDEFINED):
        super().__init__(status, reason)


_STATUS = {"shape mismatch": AssociationStatus.INVALID_INPUT,
           "fewer than 3 pairs": AssociationStatus.INSUFFICIENT_OBSERVATIONS,
           "non-finite model output": AssociationStatus.INVALID_INPUT,
           "non-finite target": AssociationStatus.INVALID_INPUT,
           "constant model output": AssociationStatus.UNDEFINED,
           "constant target": AssociationStatus.UNDEFINED}


# --------------------------------------------------------------------- run level


def average_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties given their average rank."""
    v = np.asarray(v)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    boundary = np.concatenate(([True], sv[1:] != sv[:-1]))
    group = np.cumsum(boundary) - 1
    starts = np.flatnonzero(boundary)
    ends = np.concatenate((starts[1:], [v.size]))
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = avg[group]
    return ranks


def tie_profile(v: np.ndarray) -> dict:
    _, counts = np.unique(np.asarray(v), return_counts=True)
    return {"n": int(np.asarray(v).size), "n_unique": int(counts.size),
            "largest_tie_group": int(counts.max()) if counts.size else 0,
            "n_in_tie_groups": int(counts[counts > 1].sum())}


def definedness(score: np.ndarray, target: np.ndarray) -> str | None:
    """Reason the coefficient is undefined, or None if it is defined."""
    score, target = np.asarray(score), np.asarray(target)
    if score.shape != target.shape or score.ndim != 1:
        return "shape mismatch"
    if score.size < 3:
        return "fewer than 3 pairs"
    if not np.all(np.isfinite(score)):
        return "non-finite model output"
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.all(score == score[0]):
        return "constant model output"
    if np.all(target == target[0]):
        return "constant target"
    return None


def spearman(score: np.ndarray, target: np.ndarray) -> float:
    """Spearman's rho with average ranks (no jitter). Raises UndefinedCorrelation."""
    reason = definedness(score, target)
    if reason:
        raise UndefinedCorrelation(reason, _STATUS[reason])
    rx = average_ranks(np.asarray(score, dtype=np.float64))
    ry = average_ranks(np.asarray(target, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    rho = float(rx @ ry) / den if den > 0 else float("nan")
    if not math.isfinite(rho):
        raise UndefinedCorrelation("non-finite coefficient")
    return rho


def within_run_p_value(rho: float | None, n: int) -> dict:
    """Descriptive within-run two-sided p-value (t statistic, normal reference) as a PValue record.
    It describes association inside one run only and never enters the confirmatory inference."""
    return correlation_t_approximation(rho, n).to_dict()


# --------------------------------------------------------------- exact inference


def binom_cdf(k: int, n: int) -> float:
    """Pr[X <= k] for X ~ Binomial(n, 1/2), exact rational arithmetic."""
    if k < 0:
        return 0.0
    k = min(k, n)
    return sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n


def _sign_counts(values, test: str, p_of) -> dict:
    v = np.asarray(values, dtype=np.float64)
    pos, neg = int(np.sum(v > 0)), int(np.sum(v < 0))
    n = pos + neg
    out = {"test": test, "n_positive": pos, "n_negative": neg, "n_zero_ties_removed": int(v.size - n),
           "n_effective": n}
    if n == 0:
        return {**out, "status": NO_EFFECTIVE_OBSERVATIONS, "p_value": None,
                "reason": "no nonzero value: the sign probability is undefined and the test is not performed"}
    return {**out, "status": TESTED, "p_value": p_of(pos, neg, n)}


def sign_test_two_sided(values) -> dict:
    """Exact two-sided sign test; zeros are ties, removed and counted. p = min(1, 2 * smaller tail).
    With no nonzero value: status NO_EFFECTIVE_OBSERVATIONS, p_value None."""
    return _sign_counts(values, "exact two-sided binomial sign test",
                        lambda pos, neg, n: min(1.0, 2.0 * binom_cdf(min(pos, neg), n)))


def sign_test_one_sided_greater(values) -> dict:
    """Exact one-sided sign test of H1: Pr[value > 0] > 1/2; zeros removed and counted.
    With no nonzero value: status NO_EFFECTIVE_OBSERVATIONS, p_value None."""
    return _sign_counts(values, "exact one-sided binomial sign test (H1: Pr[D > 0] > 1/2)",
                        lambda pos, neg, n: 1.0 - binom_cdf(pos - 1, n))


def holm(p_values: dict, alpha: float) -> dict:
    """Holm step-down; returns adjusted p-values and rejections keyed like the input."""
    items = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out, running, stop = {}, 0.0, False
    for i, (key, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        reject = (not stop) and p <= alpha / (m - i)
        stop = stop or not reject
        out[key] = {"p_value": p, "holm_adjusted_p": adj, "threshold": alpha / (m - i), "rejected": reject}
    return out


def median_order_statistic_ci(values, level: float = 0.95) -> dict:
    """Distribution-free CI for the median: [x_(k), x_(n-k+1)] with the largest k such that
    Pr[Binomial(n, 1/2) <= k - 1] <= (1 - level) / 2. Reports the exact coverage achieved."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    tail = (1.0 - level) / 2.0
    k = 0
    while k + 1 <= n and binom_cdf(k, n) <= tail:
        k += 1
    if k == 0:
        return {"method": "order statistic", "n": int(n), "available": False,
                "reason": f"n = {n} is too small for a {level:.0%} order-statistic interval"}
    return {"method": "order statistic", "n": int(n), "available": True, "lower_rank": k,
            "upper_rank": n - k + 1, "lower": float(x[k - 1]), "upper": float(x[n - k]),
            "exact_coverage": 1.0 - 2.0 * binom_cdf(k - 1, n)}


def run_bootstrap_median_ci(values, *, n_resamples: int, seed: int, level: float = 0.95) -> dict:
    """Sensitivity only: percentile bootstrap of the median, resampling whole runs."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    med = np.median(x[idx], axis=1)
    lo, hi = np.quantile(med, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"method": "percentile bootstrap over whole runs (sensitivity only)", "n_resamples": n_resamples,
            "seed": seed, "rng": "numpy PCG64", "lower": float(lo), "upper": float(hi)}


def describe(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return {"n": int(x.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


# ------------------------------------------------------------------- decision


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: str                 # VALID | EXCLUDED | FAILED
    rho: dict                   # condition -> float, for VALID runs
    reason: str | None = None


def analyse(outcomes: list[RunOutcome], *, alpha: float, min_valid_runs: int, n_planned: int,
            bootstrap_resamples: int, bootstrap_seed: int, precheck_ok: bool = True,
            precheck_reason: str | None = None) -> dict:
    """Primary inference, control separation and the decision category, from run outcomes only."""
    valid = [o for o in outcomes if o.status == "VALID"]
    not_valid = [{"run_id": o.run_id, "status": o.status, "reason": o.reason}
                 for o in outcomes if o.status != "VALID"]
    out = {"n_planned": n_planned, "n_recorded": len(outcomes), "n_valid": len(valid),
           "not_valid": not_valid, "alpha": alpha, "min_valid_runs": min_valid_runs}
    if not precheck_ok:
        out.update(decision=PRECHECK_FAILED, decision_reason=precheck_reason)
        return out
    if len(outcomes) != n_planned:
        out.update(decision=INCONCLUSIVE,
                   decision_reason=f"{len(outcomes)} of {n_planned} planned runs have a terminal record")
        return out
    rho = {c: [o.rho[c] for o in valid] for c in CONDITIONS}
    out["run_level_rho"] = {c: dict(zip([o.run_id for o in valid], rho[c])) for c in CONDITIONS}
    if len(valid) < min_valid_runs:
        out.update(decision=INCONCLUSIVE,
                   decision_reason=f"{len(valid)} valid runs; the frozen minimum is {min_valid_runs}")
        return out
    primary = sign_test_two_sided(rho["primary"])
    out["primary"] = {**primary, "summary": describe(rho["primary"]),
                      "median_ci": median_order_statistic_ci(rho["primary"]),
                      "bootstrap_sensitivity": run_bootstrap_median_ci(
                          rho["primary"], n_resamples=bootstrap_resamples, seed=bootstrap_seed),
                      "rejected": primary["status"] == TESTED and primary["p_value"] <= alpha}
    for c in CONTROLS:
        out[f"control_{c}"] = {"summary": describe(rho[c])}
    if primary["status"] == NO_EFFECTIVE_OBSERVATIONS:
        out.update(decision=INCONCLUSIVE, decision_reason=(
            f"primary sign test not testable: all {len(valid)} valid coefficients are exactly zero, so p_+ "
            "is undefined; this is not evidence that the association is zero"))
        return out
    if not out["primary"]["rejected"]:
        out.update(decision=INCONCLUSIVE, decision_reason=(
            "primary sign test not significant; this is not evidence that the association is zero"))
        return out
    diffs = {c: np.abs(rho["primary"]) - np.abs(rho[c]) for c in CONTROLS}
    tests = {c: sign_test_one_sided_greater(diffs[c]) for c in CONTROLS}
    # Holm over the whole family of two; an untestable control never rejects (placeholder p = 1 for
    # ordering only) and keeps the family size at two, so the other control is not favoured
    adj = holm({c: 1.0 if tests[c]["p_value"] is None else tests[c]["p_value"] for c in CONTROLS}, alpha)
    for c in CONTROLS:
        if tests[c]["status"] == NO_EFFECTIVE_OBSERVATIONS:
            adj[c] = {"p_value": None, "holm_adjusted_p": None, "threshold": None, "rejected": False,
                      "status": NO_EFFECTIVE_OBSERVATIONS}
    for c in CONTROLS:
        out[f"control_{c}"].update(paired_difference=describe(diffs[c]), separation_test=tests[c],
                                   holm=adj[c])
    if all(adj[c]["rejected"] for c in CONTROLS):
        out.update(decision=SUPPORTED, decision_reason=(
            "primary association significant and both Holm-adjusted control-separation tests rejected"))
    else:
        failed = [c for c in CONTROLS if not adj[c]["rejected"]]
        untestable = [c for c in CONTROLS if tests[c]["status"] == NO_EFFECTIVE_OBSERVATIONS]
        out.update(decision=ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED, decision_reason=(
            f"primary association significant; separation not established for {failed}"
            + (f" (not testable, every paired difference exactly zero: {untestable})" if untestable else "")))
    return out


# ------------------------------------------------------------------ secondary


def secondary_descriptives(weights: np.ndarray, score: np.ndarray) -> dict:
    """Descriptive only. Factors are latent trajectory-derived quantities."""
    w = np.asarray(weights, dtype=np.int64)
    out = {"role": "descriptive secondary analysis; never confirmatory",
           "per_round_weight_distribution": {}}
    for r in range(w.shape[1]):
        vals, counts = np.unique(w[:, r], return_counts=True)
        out["per_round_weight_distribution"][f"round_{r + 1}"] = {
            "weights": vals.tolist(), "counts": counts.tolist(), "mean_log2_contribution": float(-w[:, r].mean())}
    for name, t in (("last_round_factor_log2", -w[:, -1]), ("first_seven_rounds_log2", -w[:, :-1].sum(axis=1))):
        try:
            out[name] = {"rho": spearman(score, t)}
        except UndefinedCorrelation as exc:
            out[name] = {"rho": None, "undefined": str(exc)}
    return out
