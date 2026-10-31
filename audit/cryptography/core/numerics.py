"""
Numerically honest tail probabilities.

A p-value that underflows double precision is not zero. ``PValue`` carries the value when it is
representable, its base-10 logarithm in every case, and a status that says which:

    EXACT        computed in exact rational arithmetic (sign tests)
    APPROXIMATE  a finite, normal (non-subnormal) double from an asymptotic approximation
    UNDERFLOW    the approximation is below the smallest normal double; ``value`` is None and
                 only ``log10`` is reported (from a convergent continued-fraction tail bound)
    DEGENERATE   the approximation does not apply (e.g. |rho| = 1 makes the t statistic infinite);
                 ``value`` and ``log10`` are None
    UNDEFINED    the statistic itself is undefined; no p-value exists

Nothing here returns 0.0 for a probability.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

SMALLEST_NORMAL = sys.float_info.min          # 2.2250738585072014e-308
LOG10_E = math.log10(math.e)


@dataclass(frozen=True)
class PValue:
    status: str
    value: float | None
    log10: float | None
    method: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "value": self.value, "log10": self.log10, "method": self.method,
                "note": self.note}

    def rejects(self, alpha: float) -> bool:
        """Decision at level alpha. UNDERFLOW rejects (its log10 is far below log10(alpha));
        DEGENERATE and UNDEFINED never reject."""
        if self.status in ("EXACT", "APPROXIMATE"):
            return self.value <= alpha
        if self.status == "UNDERFLOW":
            return self.log10 <= math.log10(alpha)
        return False


def log_normal_upper_tail(z: float) -> float:
    """Natural log of Pr[Z > z] for z >= 0, accurate for any finite z (no underflow).
    Uses erfc where it is representable and the Laplace continued fraction otherwise."""
    if z < 0:
        raise ValueError("z must be non-negative")
    q = 0.5 * math.erfc(z / math.sqrt(2.0))
    if q >= SMALLEST_NORMAL:
        return math.log(q)
    # Laplace continued fraction for Mills' ratio R(z) = Q(z)/phi(z), evaluated bottom-up.
    cf = z
    for k in range(200, 0, -1):
        cf = z + k / cf
    return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - math.log(cf)


def normal_two_sided(z: float, *, method: str) -> PValue:
    if not math.isfinite(z):
        return PValue("DEGENERATE", None, None, method, "test statistic is not finite")
    ln_p = math.log(2.0) + log_normal_upper_tail(abs(z))
    ln_p = min(ln_p, 0.0)
    log10 = ln_p * LOG10_E
    value = math.exp(ln_p)
    if value < SMALLEST_NORMAL:
        return PValue("UNDERFLOW", None, log10, method,
                      "below the smallest normal double; log10 from a continued-fraction tail")
    return PValue("APPROXIMATE", value, log10, method)


def correlation_t_approximation(rho: float | None, n: int) -> PValue:
    """Two-sided p-value of a correlation coefficient by the t statistic with a normal reference.
    Descriptive within-sample use only."""
    method = "t = rho*sqrt((n-2)/(1-rho^2)), standard normal reference"
    if rho is None or not math.isfinite(rho):
        return PValue("UNDEFINED", None, None, method, "correlation undefined")
    if n < 3:
        return PValue("UNDEFINED", None, None, method, "fewer than 3 observations")
    if abs(rho) >= 1.0:
        return PValue("DEGENERATE", None, None, method,
                      "|rho| = 1: the t approximation is infinite and has no finite p-value")
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    return normal_two_sided(t, method=method)


def exact_binomial_half_cdf(k: int, n: int) -> Fraction:
    """Pr[X <= k] for X ~ Binomial(n, 1/2), as an exact fraction."""
    if k < 0:
        return Fraction(0)
    k = min(k, n)
    return Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)


def exact_pvalue(frac: Fraction, method: str) -> PValue:
    frac = min(frac, Fraction(1))
    if frac <= 0:
        raise ArithmeticError("an exact sign-test p-value is never zero")
    log10 = (math.log10(frac.numerator) - math.log10(frac.denominator))
    note = f"exact fraction {frac.numerator}/{frac.denominator}" if frac.denominator < 10 ** 60 else \
        f"exact fraction with a {frac.denominator.bit_length()}-bit denominator"
    value = float(frac)
    if value < SMALLEST_NORMAL:
        return PValue("UNDERFLOW", None, log10, method, "exact rational below the smallest normal double; " + note)
    return PValue("EXACT", value, log10, method, note)
