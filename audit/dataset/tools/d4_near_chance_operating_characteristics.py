"""
Analytical operating characteristics of D4 rule near_chance_baseline_gated_v4.

Design inputs only (test size, alpha, number of fits); no D4 outcome data is
read.  For each candidate chance margin it prints exact binomial
probabilities of the per-fit states, computed from the engine's own
acceptance regions:

    P(fit EQUIVALENT | p = 0.5), and the union lower bound for all k fits
    P(EQUIVALENT | p = 0.5 + delta/2)
    P(EQUIVALENT | p = 0.5 + delta)      (the size; must be <= alpha)
    P(ABOVE_BAND | p = 0.51)             (illustrative one-point advantage)
    P(ABOVE_BAND | p = 0.5 + 2 delta)

and, for the intact-label baseline-adequacy gate at floor t,
    P(fit ADEQUATE | q), with the union lower bound for all k fits, and
    P(fit INADEQUATE | q), for several true baseline accuracies q.

The union bound P(all k) >= 1 - k * P(not EQUIVALENT) needs no independence
between fits, which share one test sample.

USAGE (from the repository root)
    $PY -B -m audit.dataset.tools.d4_near_chance_operating_characteristics
"""

from __future__ import annotations

import argparse

from scipy.stats import binom

from audit.dataset import d4_controlled_perturbation as d4


def _state(x: int, n: int, delta: float, alpha: float, k: int) -> str:
    return d4.near_chance_replicate(x, n, chance=0.5, margin=delta, alpha=alpha, replicates=k)["state"]


def _first(lo: int, hi: int, test) -> int:
    """Smallest x in [lo, hi] with test(x), for test monotone False->True."""
    while lo < hi:
        mid = (lo + hi) // 2
        if test(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def acceptance(n: int, delta: float, alpha: float, k: int) -> tuple[int, int, int]:
    """(first EQUIVALENT x, last EQUIVALENT x, first ABOVE_BAND x)."""
    half = n // 2
    first_eq = _first(0, half, lambda x: _state(x, n, delta, alpha, k) == "EQUIVALENT")
    last_eq = _first(half, n, lambda x: _state(x, n, delta, alpha, k) != "EQUIVALENT") - 1
    first_above = _first(half, n, lambda x: _state(x, n, delta, alpha, k) == "ABOVE_BAND")
    return first_eq, last_eq, first_above


def baseline_region(n: int, floor: float, alpha: float, k: int) -> tuple[int, int]:
    """(first ADEQUATE x, last INADEQUATE x) at the given floor."""
    def st(x):
        return d4.baseline_adequacy_replicate(x, n, threshold=floor, alpha=alpha, replicates=k)["state"]
    first_adequate = _first(0, n, lambda x: st(x) == "ADEQUATE")
    last_inadequate = _first(0, n, lambda x: st(x) != "INADEQUATE") - 1
    return first_adequate, last_inadequate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.90)
    parser.add_argument("--qs", type=float, nargs="+",
                        default=[0.929, 0.91, 0.905, 0.902, 0.901, 0.9005, 0.90, 0.899, 0.895])
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01])
    args = parser.parse_args(argv)
    n, alpha, k = args.n, args.alpha, args.k
    print(f"n_test={n} alpha={alpha} k={k}")
    print("delta    accept_x          P(eq|0.5)  all-k>=   P(eq|0.5+d/2)  P(eq|U)  P(above|0.51)   P(above|0.5+2d)")
    for delta in args.deltas:
        a, b, s = acceptance(n, delta, alpha, k)
        p_eq = lambda p: float(binom.cdf(b, n, p) - binom.cdf(a - 1, n, p))
        p_above = lambda p: float(binom.sf(s - 1, n, p))
        at_chance = p_eq(0.5)
        print(f"{delta:<8} [{a}, {b}]  {at_chance:.6f}   {max(0.0, 1 - k * (1 - at_chance)):.6f}  "
              f"{p_eq(0.5 + delta / 2):.4f}         {p_eq(0.5 + delta):.4f}   "
              f"{p_above(0.51):.4f}          {p_above(0.5 + 2 * delta):.4f}")
    first_adequate, last_inadequate = baseline_region(n, args.floor, alpha, k)
    print(f"\nbaseline floor t={args.floor}: ADEQUATE iff x >= {first_adequate}; INADEQUATE iff x <= {last_inadequate}")
    print("q        P(ADEQUATE|q)  all-k>=    P(INADEQUATE|q)")
    for q in args.qs:
        p_adequate = float(binom.sf(first_adequate - 1, n, q))
        p_inadequate = float(binom.cdf(last_inadequate, n, q))
        print(f"{q:<8} {p_adequate:.6f}       {max(0.0, 1 - k * (1 - p_adequate)):.6f}   {p_inadequate:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
