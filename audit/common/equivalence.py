#!/usr/bin/env python3
"""
Equivalence testing, multiplicity control and power — generic.

WHY THIS MODULE IS IN audit/common/
    Nothing here is specific to any audited target, and the module names
    none - a test asserts that. Deciding whether a reproduced measurement
    agrees with a published one, to within a tolerance fixed in advance,
    is a question every target in this framework faces. Keeping the
    machinery here means a target adapter supplies only its own reference
    values, margin and hypotheses, and no target can quietly grow its own
    private copy of the statistics.

    This is a pure ADDITION. No existing file under audit/common/ or
    audit/implementation/ is modified by it.

WHAT IT PROVIDES
    * Student-t CDF/SF/quantile, exact to double precision, no scipy.
    * `tost` — two one-sided tests for EQUIVALENCE within +-delta.
    * `outside_margin` — one-sided tests for a discrepancy OUTSIDE
      +-delta, in either direction.
    * `holm` — Holm-Bonferroni step-down over a family.
    * `classify` — the three-way outcome mapping.
    * `tost_power_exact` and `tost_power_mc` — power of the TOST
      procedure with the variance ESTIMATED from the sample, not assumed
      known.

THE ASYMMETRY THAT MATTERS
    Failing to establish equivalence is NOT evidence of a meaningful
    discrepancy. These are different hypotheses and they need different
    tests:

        equivalence   H0: |d| >= delta     rejected  =>  EQUIVALENT
        outside       H0: |d| <= delta     rejected  =>  DIFFERENT
        neither rejected                            =>  INDETERMINATE

    A point-null test of d = 0 answers neither question. A tiny
    discrepancy can be highly significant against zero while sitting
    well inside the margin; calling that DIFFERENT would report a
    meaningful disagreement that the data do not support. `classify`
    therefore never consults a point-null test, and accepts the
    point-null p-value only as a reported diagnostic.

WHY min() IS SOUND FOR THE OUTSIDE TEST
    `outside_margin` tests H0: d <= +delta and H0: d >= -delta and takes
    the smaller p-value. For delta > 0 the two rejection regions are
    DISJOINT - rejecting both would require d > delta and d < -delta at
    once - so the minimum controls the type-I error at the nominal level
    without further correction. The family-wise correction across
    comparisons is applied separately, by `holm`.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

MODULE_VERSION = "audit-common-equivalence-v1"

#: Outcome vocabulary. A target adapter must reuse these rather than
#: inventing parallel statuses.
EQUIVALENT = "EQUIVALENT"
DIFFERENT = "DIFFERENT"
INDETERMINATE = "INDETERMINATE"
OUTCOMES = (EQUIVALENT, DIFFERENT, INDETERMINATE)


# --------------------------------------------------------------------------
# Student-t, computed rather than tabulated
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float, *, itmax: int = 400,
            eps: float = 3.0e-16, fpmin: float = 1.0e-300) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(
        f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must lie in [0, 1], got {x}")
    if x in (0.0, 1.0):
        return x
    lead = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lead) * _betacf(a, b, x) / a
    other = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + b * math.log1p(-x) + a * math.log(x))
    return 1.0 - math.exp(other) * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """P(T <= t)."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * regularised_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_sf(t: float, df: int) -> float:
    """P(T > t)."""
    return 1.0 - student_t_cdf(t, df)


def student_t_ppf(p: float, df: int, *, tol: float = 1e-12) -> float:
    """The t with P(T <= t) = p, by bisection on the CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    lo, hi = -1.0e4, 1.0e4
    for _ in range(500):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# The two hypotheses
# --------------------------------------------------------------------------

def _check(d: float, se: float, df: int, delta: float) -> None:
    for name, v in (("d", d), ("se", se), ("delta", delta)):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name} must be a finite number, got {v!r}")
    if se <= 0.0:
        raise ValueError(f"se must be positive, got {se}")
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")


def tost(d: float, se: float, df: int, delta: float) -> dict:
    """
    Two one-sided tests for EQUIVALENCE of d to zero within +-delta.

        H0_lower: d <= -delta      t = (d + delta)/se, upper tail
        H0_upper: d >= +delta      t = (d - delta)/se, lower tail

    p_tost is the larger of the two: equivalence requires BOTH to reject,
    which is the intersection-union principle and needs no multiplicity
    adjustment between the pair.
    """
    _check(d, se, df, delta)
    t_lower = (d + delta) / se
    t_upper = (d - delta) / se
    p_lower = student_t_sf(t_lower, df)
    p_upper = student_t_cdf(t_upper, df)
    return {"t_lower": t_lower, "t_upper": t_upper,
            "p_lower": p_lower, "p_upper": p_upper,
            "p_tost": max(p_lower, p_upper), "delta": delta, "df": df}


def outside_margin(d: float, se: float, df: int, delta: float) -> dict:
    """
    One-sided tests for a discrepancy OUTSIDE +-delta.

        H0_above: d <= +delta      t = (d - delta)/se, upper tail
        H0_below: d >= -delta      t = (d + delta)/se, lower tail

    p_outside is the SMALLER of the two. For delta > 0 the rejection
    regions are disjoint, so the minimum controls type-I error at the
    nominal level; see the module docstring.

    This is the test that licenses DIFFERENT. A point-null test of d = 0
    does not, and is not used here.
    """
    _check(d, se, df, delta)
    t_above = (d - delta) / se
    t_below = (d + delta) / se
    p_above = student_t_sf(t_above, df)
    p_below = student_t_cdf(t_below, df)
    return {"t_above": t_above, "t_below": t_below,
            "p_above": p_above, "p_below": p_below,
            "p_outside": min(p_above, p_below),
            "direction": ("above" if p_above <= p_below else "below"),
            "delta": delta, "df": df}


def point_null(d: float, se: float, df: int) -> dict:
    """
    Two-sided test of d = 0. REPORTED ONLY, never used to classify.

    Retained because it is informative - it says whether the data can
    distinguish the reproduction from the published value at all - but a
    significant result here says nothing about whether the discrepancy
    exceeds the margin, which is the question the audit asks.
    """
    if se <= 0.0:
        raise ValueError(f"se must be positive, got {se}")
    t = d / se
    return {"t_diff": t, "p_diff": 2.0 * student_t_sf(abs(t), df),
            "role": "diagnostic only; does not enter the outcome decision"}


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """
    Holm-Bonferroni step-down over a family.

    Sorts ascending, compares p_(i) against alpha/(m - i + 1), and stops
    at the first failure so that no later rank is rejected. Controls the
    family-wise error rate at alpha without assuming independence.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    for k, v in pvalues.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v) \
                or not 0.0 <= v <= 1.0:
            raise ValueError(f"p-value for {k!r} is not in [0, 1]: {v!r}")
    m = len(pvalues)
    out: dict[str, dict] = {}
    still = True
    for i, (key, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1]),
                                 start=1):
        threshold = alpha / (m - i + 1)
        reject = bool(still and p < threshold)
        if not reject:
            still = False
        out[key] = {"p": p, "rank": i, "m": m, "threshold": threshold,
                    "reject": reject}
    return out


def classify(*, equivalence_rejected: bool, outside_rejected: bool) -> str:
    """
    The three-way outcome mapping.

    Both cannot be true: EQUIVALENT needs the confidence interval for d
    inside (-delta, +delta) and DIFFERENT needs it wholly outside, so the
    combination is unreachable from any single dataset. It is checked
    rather than assumed, because a future change to either test that
    broke the property should fail loudly.
    """
    if equivalence_rejected and outside_rejected:
        raise ArithmeticError(
            "equivalence and outside-margin were both concluded, which is "
            "impossible for a single dataset with delta > 0. One of the two "
            "tests is mis-specified.")
    if equivalence_rejected:
        return EQUIVALENT
    if outside_rejected:
        return DIFFERENT
    return INDETERMINATE


# --------------------------------------------------------------------------
# Power, with the variance ESTIMATED rather than assumed known
# --------------------------------------------------------------------------

def tost_power_exact(delta: float, sd: float, n: int, alpha: float = 0.05,
                     true_difference: float = 0.0, *,
                     nodes: int = 2001) -> dict:
    """
    Exact power of the TOST procedure when s is estimated from the sample.

    THE ERROR THIS REPLACES. The previous implementation computed
    `2*Phi(delta/se - t_crit) - 1` with se treated as a known constant.
    That is wrong twice over: it ignores the sampling variability of s
    (with n = 5 there are four degrees of freedom, so s is a poor
    estimate), and the expression goes NEGATIVE whenever
    delta/se < t_crit, which was then clamped to zero. Clamped zeros were
    reported as though a power of zero had been computed.

    THE CALCULATION. TOST rejects iff

        xbar > -delta + c*s/sqrt(n)   and   xbar < +delta - c*s/sqrt(n)

    with c = t_{1-alpha, n-1}. Conditional on s, xbar is normal with mean
    `true_difference` and standard deviation sd/sqrt(n), and xbar is
    independent of s. So

        P(reject | s) = Phi((delta - c*s/sqrt(n) - mu)/(sd/sqrt(n)))
                      - Phi((-delta + c*s/sqrt(n) - mu)/(sd/sqrt(n)))

    floored at zero, which is positive only while s < delta*sqrt(n)/c.
    Marginalising over (n-1) s^2/sd^2 ~ chi^2_{n-1} gives the power as a
    one-dimensional integral, evaluated here by Simpson's rule over the
    finite interval where the integrand is non-zero.

    No scipy, and no normal approximation to the t distribution.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for a variance estimate, got {n}")
    if sd <= 0.0:
        raise ValueError(f"sd must be positive, got {sd}")
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must lie in (0, 0.5), got {alpha}")

    df = n - 1
    c = student_t_ppf(1.0 - alpha, df)
    root_n = math.sqrt(n)
    se_true = sd / root_n

    # s < delta*sqrt(n)/c  <=>  w < df * (delta*sqrt(n)/(c*sd))^2
    s_max = delta * root_n / c
    w_max = df * (s_max / sd) ** 2
    if w_max <= 0.0:
        return {"power": 0.0, "delta": delta, "sd": sd, "n": n, "df": df,
                "alpha": alpha, "t_critical": c,
                "true_difference": true_difference,
                "method": "exact integration; rejection region is empty"}

    log_norm = (df / 2.0) * math.log(2.0) + math.lgamma(df / 2.0)

    def integrand(w: float) -> float:
        if w <= 0.0:
            return 0.0
        s = sd * math.sqrt(w / df)
        hi = (delta - c * s / root_n - true_difference) / se_true
        lo = (-delta + c * s / root_n - true_difference) / se_true
        inner = _phi(hi) - _phi(lo)
        if inner <= 0.0:
            return 0.0
        log_pdf = (df / 2.0 - 1.0) * math.log(w) - w / 2.0 - log_norm
        return inner * math.exp(log_pdf)

    # Simpson's rule needs an even number of intervals.
    m = nodes if nodes % 2 == 1 else nodes + 1
    h = w_max / (m - 1)
    total = integrand(0.0) + integrand(w_max)
    for i in range(1, m - 1):
        total += (4.0 if i % 2 else 2.0) * integrand(i * h)
    power = total * h / 3.0
    power = min(max(power, 0.0), 1.0)

    return {"power": power, "delta": delta, "sd": sd, "n": n, "df": df,
            "alpha": alpha, "t_critical": c, "true_difference": true_difference,
            "delta_over_se_if_sd_known": delta / se_true,
            "s_max_for_rejection": s_max,
            "method": ("exact: P(reject | s) integrated over the chi "
                       "distribution of s by Simpson's rule; the variance is "
                       "estimated from the sample, not assumed known"),
            "nodes": m}


def tost_power_mc(delta: float, sd: float, n: int, alpha: float = 0.05,
                  true_difference: float = 0.0, *,
                  n_sim: int = 200_000, seed: int = 20261015,
                  confidence: float = 0.99) -> dict:
    """
    Monte Carlo power, simulating the ACTUAL decision rule.

    Draws n observations per replication, computes the sample mean and
    sample SD exactly as the analysis will, applies `tost`, and counts
    rejections. An independent check on `tost_power_exact`: the two agree
    to Monte Carlo error, and a test asserts it.

    The seed is fixed and recorded so the figure is reproducible.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    rng = random.Random(seed)
    df = n - 1
    c = student_t_ppf(1.0 - alpha, df)
    root_n = math.sqrt(n)
    hits = 0
    for _ in range(n_sim):
        xs = [rng.gauss(true_difference, sd) for _ in range(n)]
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / df
        s = math.sqrt(var)
        if s <= 0.0:
            continue
        se = s / root_n
        if (mean + delta) / se > c and (mean - delta) / se < -c:
            hits += 1
    p = hits / n_sim
    z = {0.95: 1.959964, 0.99: 2.575829}.get(confidence, 2.575829)
    half = z * math.sqrt(max(p * (1.0 - p), 1e-18) / n_sim)
    return {"power": p, "n_sim": n_sim, "seed": seed,
            "mc_interval": (max(0.0, p - half), min(1.0, p + half)),
            "mc_half_width": half, "confidence": confidence,
            "delta": delta, "sd": sd, "n": n, "alpha": alpha,
            "true_difference": true_difference,
            "method": ("Monte Carlo over the actual decision rule: n draws "
                       "per replication, sample SD recomputed each time")}


def decide_one(*, d: float, se: float, df: int, delta: float) -> dict:
    """
    All three statistics for one comparison, without multiplicity.

    Returns the TOST and outside-margin results plus the point-null
    diagnostic. The caller applies `holm` across its family and then
    `classify`, because the correction is a property of the family rather
    than of any single comparison.
    """
    t = tost(d, se, df, delta)
    o = outside_margin(d, se, df, delta)
    pn = point_null(d, se, df)
    return {**t, **o, **pn, "difference": d, "se": se,
            "mean_inside_margin": abs(d) < delta}
