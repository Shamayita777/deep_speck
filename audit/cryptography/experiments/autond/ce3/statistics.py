from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.stats import beta

NO_EFFECTIVE = "NO_VALID_OBSERVATIONS"
CP_METHOD = "two-sided exact 95% Clopper-Pearson interval for q = P(D > 0) over all valid replicates"
ZERO_POLICY = "zero (including -0.0) is a non-positive outcome; nothing is removed from the denominator"


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> list[float]:
    if n <= 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n and n > 0")
    a = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return [lo, hi]


def sign_test(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("sign-test values must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite endpoint values must be excluded before testing")
    n = int(x.size)
    pos, neg = int(np.count_nonzero(x > 0)), int(np.count_nonzero(x < 0))
    base = {"n_valid": n, "n_positive": pos, "n_nonpositive": n - pos, "n_negative": neg, "n_zero": n - pos - neg,
            "zero_policy": ZERO_POLICY, "ci95_method": CP_METHOD}
    if n == 0:
        return {**base, "status": NO_EFFECTIVE, "p_value": None, "p_numerator": None, "p_denominator": None,
                "ci95": None}
    num = sum(math.comb(n, j) for j in range(pos, n + 1))
    den = 1 << n
    return {**base, "status": "TESTED", "p_value": num / den, "p_numerator": num, "p_denominator": den,
            "ci95": clopper_pearson(pos, n)}


def rejects(test: dict, alpha: float) -> bool:
    if test["status"] != "TESTED":
        return False
    a = Fraction(str(alpha))
    return test["p_numerator"] * a.denominator < a.numerator * test["p_denominator"]


def median_ci(values, confidence: float = 0.95) -> dict:
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = int(x.size)
    best = None
    for k in range(1, (n + 1) // 2 + 1):
        cov = 1.0 - 2.0 * sum(math.comb(n, j) for j in range(k)) / 2 ** n
        if cov >= confidence:
            best = (k, cov)
        else:
            break
    if best is None:
        return {"lower": None, "upper": None, "coverage": None, "method": "too few values for the interval"}
    k, cov = best
    return {"lower": float(x[k - 1]), "upper": float(x[n - k]), "lower_order": k, "upper_order": n - k + 1,
            "coverage": cov, "method": "distribution-free exact order-statistic interval for the median"}


def summary(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x)), "median_ci95": median_ci(x),
            "q1": float(np.quantile(x, 0.25)), "q3": float(np.quantile(x, 0.75)), "minimum": float(np.min(x)),
            "maximum": float(np.max(x)), "values": [float(v) for v in x]}


def decide(tests: dict, *, alpha: float, n_valid: int, n_required: int, precheck_ok: bool,
           precheck_reason: str | None = None) -> dict:
    if not precheck_ok:
        return {"decision": "PRECHECK_FAILED", "reason": precheck_reason or "precheck failed", "sequence": []}
    if n_valid < n_required:
        return {"decision": "INCONCLUSIVE", "reason": f"{n_valid} VALID replicates, {n_required} required",
                "sequence": []}
    seq = []
    for name, parts in (("C", ("C_task", "C_twin")), ("S1", ("S1",)), ("S2", ("S2",))):
        untestable = [p for p in parts if tests[p]["status"] == NO_EFFECTIVE]
        rej = {p: rejects(tests[p], alpha) for p in parts}
        seq.append({"step": name, "components": list(parts), "rejected_components": rej,
                    "untestable_components": untestable, "rejected": all(rej.values()) and not untestable})
        if untestable:
            return {"decision": "INCONCLUSIVE", "reason": f"{name}: {untestable} {NO_EFFECTIVE}", "sequence": seq}
        if not seq[-1]["rejected"]:
            status = {"C": "POSITIVE_CONTROL_FAILED", "S1": "INCONCLUSIVE",
                      "S2": "TARGET_DECODABLE_TWIN_UNRESOLVED"}[name]
            return {"decision": status, "reason": f"{name} not rejected at alpha {alpha}", "sequence": seq}
    return {"decision": "SUPPORTED", "reason": "C, S1 and S2 rejected in the frozen order", "sequence": seq}


def analyse(endpoints_by_replicate: dict, states: dict, *, alpha: float, n_required: int, precheck_ok: bool,
            precheck_reason: str | None = None) -> dict:
    valid = sorted(r for r, s in states.items() if s == "VALID")
    values = {e: [endpoints_by_replicate[r][e] for r in valid]
              for e in ("C_task", "C_twin", "S1", "S2", "raw_gap", "S1_margin")}
    tests = {e: sign_test(values[e]) for e in ("C_task", "C_twin", "S1", "S2")}
    dec = decide(tests, alpha=alpha, n_valid=len(valid), n_required=n_required, precheck_ok=precheck_ok,
                 precheck_reason=precheck_reason)
    tested = {s["step"] for s in dec["sequence"]}
    roles = {"C_task": "C", "C_twin": "C", "S1": "S1", "S2": "S2"}
    return {"alpha": alpha, "n_planned": len(states), "n_valid": len(valid), "n_required": n_required,
            "valid_replicates": valid,
            "not_valid": {r: s for r, s in sorted(states.items()) if s != "VALID"},
            "endpoints": {e: {"test": tests[e], "summary": summary(values[e]),
                              "tested_in_sequence": roles[e] in tested} for e in tests},
            "raw_gap": {"summary": summary(values["raw_gap"]),
                        "role": "descriptive linear-accessibility contrast; never tested, never an information claim"},
            "S1_margin": {"summary": summary(values["S1_margin"]),
                          "role": "descriptive BA(A) - median BA(P_b); the S1 test uses the exact tie-broken rank"},
            **dec}
