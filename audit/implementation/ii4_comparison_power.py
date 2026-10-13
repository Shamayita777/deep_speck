#!/usr/bin/env python3
"""
Design adequacy for a paired-block II-4 comparison - GENERIC.

Contains no knowledge of Gohr, Speck, neural networks, residual depth,
or any particular comparison axis. It answers three questions that every
paired-block II-4 design must answer before it is frozen, and it answers
them by SIMULATING THE ACTUAL DECISION RULE rather than by quoting a
closed-form power formula for one of its components.

WHY SIMULATE THE JOINT RULE. The conclusion that two arms are
equivalent is not "TOST was significant". Under the frozen
interpretation rule it is a CONJUNCTION:

    TOST rejects both one-sided nulls at alpha
      AND the signed 95% CI supports no directional claim

Powering TOST alone overstates the design, because it ignores the second
condition. This project has already been bitten by exactly that error
once, in the Experimental Validity power planner, which powered TOST
alone instead of simulating the complete joint decision procedure. The
same mistake is not repeated here: `joint_rule_power` draws block
differences, runs the real frozen analysis function, and applies the real
frozen interpretation rule to whatever it returns.

THE THREE QUESTIONS
-------------------
1. SENSITIVITY (`joint_rule_power`)
   If the two arms really are equivalent, how often does the design say
   so? Reported against a RANGE of between-block standard deviations,
   because the SD of a NEW comparison is unknown; the SD measured on a
   previous experiment is a starting point, not a prediction. The useful
   output is the SD multiplier at which the design stops working.

2. SPECIFICITY (`boundary_false_equivalence_rate`)
   If the arms differ by exactly the margin, how often does the design
   wrongly declare equivalence? For TOST this is bounded by alpha at the
   boundary; the conjunction can only lower it. Simulated rather than
   assumed, so that a mis-specified rule is caught.

3. FAILURE TOLERANCE (`valid_block_probability`)
   A design with K blocks and a minimum of M valid blocks tolerates K-M
   block losses. A block is lost if EITHER arm fails. This is the
   question most often skipped, and it is the one that actually sinks
   runs: a design can be statistically ideal and still return
   INCONCLUSIVE because too few blocks survived. The per-arm failure
   probability is not observable in advance, so the output is a curve,
   plus a confidence bound derived from however many arm-trainings have
   already been observed to succeed (rule of three, exact binomial).

NOTHING HERE CHOOSES A DESIGN. It reports adequacy so that a human
freezes a design with the trade-off in front of them.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy import stats

POWER_VERSION = "ii4-comparison-power-v1"

#: Default simulation count. Matches the Experimental Validity power
#: planner's frozen value so that the two tiers report power computed to
#: the same Monte Carlo resolution.
DEFAULT_SIMULATIONS = 50_000


# =====================================================================
# 1 + 2. decision-rule simulation
# =====================================================================

@dataclass
class PowerPoint:
    """Power of the joint decision rule at one (K, sd, true_delta)."""
    n_blocks: int
    sd: float
    true_delta: float
    delta_margin: float
    alpha: float
    simulations: int
    p_equivalent: float
    p_differ: float
    p_inconclusive: float
    mc_standard_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_blocks": self.n_blocks, "sd": self.sd, "true_delta": self.true_delta,
            "delta_margin": self.delta_margin, "alpha": self.alpha,
            "simulations": self.simulations,
            "p_equivalent": self.p_equivalent, "p_differ": self.p_differ,
            "p_inconclusive": self.p_inconclusive,
            "mc_standard_error": self.mc_standard_error,
        }


def _default_rule(result) -> str:
    """
    The frozen conjunction, expressed over an II4AnalysisResult.

    Kept as a module-level default so the caller can pass the project's
    real `apply_interpretation_rule` instead and have the simulation
    exercise that code path rather than a restatement of it.
    """
    materiality = getattr(result.materiality, "value", str(result.materiality))
    equivalence = getattr(result.equivalence, "value", str(result.equivalence))
    if equivalence == "EQUIVALENT_WITHIN_MARGIN" and materiality == "NO_DIRECTIONAL_CLAIM":
        return "EQUIVALENT"
    if materiality in ("MATERIALLY_HIGHER", "MATERIALLY_LOWER"):
        return "DIFFER"
    return "INCONCLUSIVE"


def joint_rule_power(
    *,
    n_blocks: int,
    sd: float,
    delta_margin: float,
    true_delta: float = 0.0,
    alpha: float = 0.05,
    simulations: int = DEFAULT_SIMULATIONS,
    analyse: Optional[Callable] = None,
    classify: Callable = _default_rule,
    seed: int = 20261012,
) -> PowerPoint:
    """
    Simulate the complete decision procedure.

    `analyse` must be the project's frozen analysis function with the
    signature of `analyse_ii4(block_differences, *, n_blocks_requested,
    min_valid_blocks, delta, alpha)`. It is injected rather than imported
    so that this module stays free of package-layout assumptions and so
    that the caller cannot accidentally power a DIFFERENT analysis than
    the one the experiment will run.

    Block differences are drawn as Normal(true_delta, sd). That is the
    design's own stated inference model: a one-sample t procedure on K
    block differences assumes approximate normality of those
    differences. Simulating under the model the analysis assumes
    measures the design, not the model's realism - which is why the
    frozen analysis also reports the assumption-free sign-flip
    cross-check, and why `analyse_ii4` emits a normality-assumption
    warning below K=10.
    """
    if analyse is None:
        raise ValueError(
            "joint_rule_power requires the project's frozen `analyse` function; refusing to "
            "power a reimplementation of the analysis. Pass analyse=analyse_ii4.")
    rng = np.random.default_rng(seed)
    counts = {"EQUIVALENT": 0, "DIFFER": 0, "INCONCLUSIVE": 0}
    for _ in range(simulations):
        diffs = rng.normal(true_delta, sd, size=n_blocks)
        result = analyse(
            list(map(float, diffs)), n_blocks_requested=n_blocks,
            min_valid_blocks=n_blocks, delta=delta_margin, alpha=alpha)
        counts[classify(result)] += 1

    p_eq = counts["EQUIVALENT"] / simulations
    return PowerPoint(
        n_blocks=n_blocks, sd=sd, true_delta=true_delta, delta_margin=delta_margin,
        alpha=alpha, simulations=simulations,
        p_equivalent=p_eq,
        p_differ=counts["DIFFER"] / simulations,
        p_inconclusive=counts["INCONCLUSIVE"] / simulations,
        mc_standard_error=math.sqrt(max(p_eq * (1 - p_eq), 1e-12) / simulations),
    )


def boundary_false_equivalence_rate(**kwargs) -> PowerPoint:
    """
    Specificity check: true difference set to EXACTLY the margin.

    At the boundary the TOST null is true, so a correct procedure
    declares equivalence at most alpha of the time. A simulated rate
    materially above alpha means the rule as implemented is not the rule
    as described.
    """
    delta_margin = kwargs["delta_margin"]
    kwargs["true_delta"] = delta_margin
    return joint_rule_power(**kwargs)


def sd_breaking_point(
    *, target_power: float = 0.80, sd_multipliers: Sequence[float], reference_sd: float, **kwargs
) -> dict[str, Any]:
    """
    Smallest simulated SD multiplier at which the design drops below
    `target_power`. Reported as a multiplier of a reference SD so that
    the result is legible as "this design tolerates N times the spread
    we have previously measured" rather than as an absolute number whose
    scale a reader has no feel for.
    """
    rows = []
    breaking = None
    for m in sorted(sd_multipliers):
        point = joint_rule_power(sd=reference_sd * m, **kwargs)
        rows.append({"sd_multiplier": m, **point.to_dict()})
        if breaking is None and point.p_equivalent < target_power:
            breaking = m
    return {
        "reference_sd": reference_sd, "target_power": target_power,
        "curve": rows,
        "breaking_multiplier": breaking,
        "interpretation": (
            f"design holds target power {target_power} up to at least "
            f"{max(sd_multipliers) if breaking is None else breaking}x the reference SD"
            + (" (no breaking point within the simulated range)" if breaking is None else "")
        ),
    }


# =====================================================================
# 3. block-failure tolerance
# =====================================================================

def valid_block_probability(
    *, n_blocks: int, min_valid_blocks: int, per_arm_failure_probability: float, n_arms: int = 2,
) -> float:
    """
    P(at least min_valid_blocks blocks survive), exact binomial.

    A block survives only if ALL of its arms succeed, because the frozen
    protocol invalidates the whole block when any arm fails - retaining
    a surviving arm would convert a paired observation into an unpaired
    one. Arm failures are treated as independent across arms and blocks,
    which is optimistic: a systematic cause (an OOM at a given batch
    size, a driver fault, a quota expiry) fails arms together. The number
    below is therefore an UPPER bound on survival under correlated
    failure, and is labelled as such wherever it is reported.
    """
    if not 0.0 <= per_arm_failure_probability <= 1.0:
        raise ValueError("per_arm_failure_probability must lie in [0, 1].")
    if min_valid_blocks > n_blocks:
        raise ValueError("min_valid_blocks cannot exceed n_blocks.")
    p_block_ok = (1.0 - per_arm_failure_probability) ** n_arms
    return float(sum(
        math.comb(n_blocks, j) * p_block_ok ** j * (1.0 - p_block_ok) ** (n_blocks - j)
        for j in range(min_valid_blocks, n_blocks + 1)
    ))


def failure_rate_upper_bound(*, observed_successes: int, observed_failures: int = 0,
                             confidence: float = 0.95) -> float:
    """
    One-sided upper confidence bound on the per-arm failure probability
    given however many arm-trainings have already been observed.

    With zero observed failures this is the exact Clopper-Pearson bound
    1 - (1 - confidence)^(1/n), the quantity the "rule of three"
    approximates. Zero failures is NOT evidence of a zero failure rate,
    and planning as if it were is how a design ends up with no failure
    tolerance at all.
    """
    n = observed_successes + observed_failures
    if n <= 0:
        return 1.0
    if observed_failures == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / n))
    return float(stats.beta.ppf(confidence, observed_failures + 1, n - observed_failures))


def failure_tolerance_report(
    *, n_blocks: int, min_valid_blocks: int, observed_arm_successes: int,
    observed_arm_failures: int = 0, n_arms: int = 2,
    rates: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20),
    survival_target: float = 0.90,
) -> dict[str, Any]:
    """Failure tolerance of one design, with the evidence-based bound applied."""
    ucl = failure_rate_upper_bound(observed_successes=observed_arm_successes,
                                   observed_failures=observed_arm_failures)
    curve = [
        {"per_arm_failure_probability": r,
         "p_block_lost": 1.0 - (1.0 - r) ** n_arms,
         "p_enough_valid_blocks": valid_block_probability(
             n_blocks=n_blocks, min_valid_blocks=min_valid_blocks,
             per_arm_failure_probability=r, n_arms=n_arms)}
        for r in rates
    ]
    tolerable = [row["per_arm_failure_probability"] for row in curve
                 if row["p_enough_valid_blocks"] >= survival_target]
    return {
        "n_blocks": n_blocks,
        "min_valid_blocks": min_valid_blocks,
        "block_losses_tolerated": n_blocks - min_valid_blocks,
        "observed_arm_trainings": observed_arm_successes + observed_arm_failures,
        "observed_arm_failures": observed_arm_failures,
        "per_arm_failure_rate_point_estimate": (
            observed_arm_failures / max(observed_arm_successes + observed_arm_failures, 1)),
        "per_arm_failure_rate_upper95": ucl,
        "p_enough_valid_blocks_at_upper95": valid_block_probability(
            n_blocks=n_blocks, min_valid_blocks=min_valid_blocks,
            per_arm_failure_probability=ucl, n_arms=n_arms),
        "survival_target": survival_target,
        "max_tolerable_per_arm_failure_rate": max(tolerable) if tolerable else None,
        "curve": curve,
        "independence_caveat": (
            "Arm failures are modelled as independent. A systematic cause (out-of-memory at the "
            "frozen batch size, driver fault, session or quota expiry) fails arms together, so "
            "every survival probability here is an UPPER bound."
        ),
    }


def compute_cost_estimate(
    *, n_blocks: int, n_arms: int, hours_per_arm_low: float, hours_per_arm_high: float,
    overhead_hours_per_block: float = 0.0, basis: str = "",
) -> dict[str, Any]:
    """Expected compute, reported as a range because the basis is a range."""
    low = n_blocks * (n_arms * hours_per_arm_low + overhead_hours_per_block)
    high = n_blocks * (n_arms * hours_per_arm_high + overhead_hours_per_block)
    return {
        "n_blocks": n_blocks, "n_arms": n_arms,
        "hours_per_arm_range": [hours_per_arm_low, hours_per_arm_high],
        "overhead_hours_per_block": overhead_hours_per_block,
        "gpu_hours_low": round(low, 1), "gpu_hours_high": round(high, 1),
        "gpu_hours_midpoint": round((low + high) / 2, 1),
        "basis": basis,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired-block II-4 design adequacy (generic).")
    ap.add_argument("--reference-sd", type=float, required=True,
                    help="between-block SD of differences from a previous comparable experiment")
    ap.add_argument("--delta", type=float, required=True, help="equivalence margin")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--blocks", type=int, nargs="+", default=[6, 7, 8, 10])
    ap.add_argument("--min-valid", type=int, nargs="+", default=None,
                    help="min valid blocks per design, aligned with --blocks")
    ap.add_argument("--sd-multipliers", type=float, nargs="+",
                    default=[1, 3, 10, 20, 30, 40, 60])
    ap.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    ap.add_argument("--observed-arm-successes", type=int, default=0)
    ap.add_argument("--observed-arm-failures", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    from audit.implementation.ii4_analysis import analyse_ii4   # frozen analysis, injected

    mins = args.min_valid or [max(k - 1, 2) for k in args.blocks]
    if len(mins) != len(args.blocks):
        print("REFUSED: --min-valid must have one value per --blocks value.")
        return 1

    report: dict[str, Any] = {
        "artifact": POWER_VERSION,
        "inputs": {k: v for k, v in vars(args).items() if k != "output"},
        "designs": [],
    }
    for k, m in zip(args.blocks, mins):
        sens = sd_breaking_point(
            sd_multipliers=args.sd_multipliers, reference_sd=args.reference_sd,
            n_blocks=k, delta_margin=args.delta, alpha=args.alpha,
            simulations=args.simulations, analyse=analyse_ii4)
        spec = boundary_false_equivalence_rate(
            n_blocks=k, sd=args.reference_sd, delta_margin=args.delta, alpha=args.alpha,
            simulations=args.simulations, analyse=analyse_ii4)
        tol = failure_tolerance_report(
            n_blocks=k, min_valid_blocks=m,
            observed_arm_successes=args.observed_arm_successes,
            observed_arm_failures=args.observed_arm_failures)
        report["designs"].append({
            "n_blocks": k, "min_valid_blocks": m,
            "sign_flip_min_attainable_p": 2 / (2 ** m),
            "sign_flip_feasible_at_min_valid": (2 / (2 ** m)) < args.alpha,
            "sensitivity": sens, "specificity_at_boundary": spec.to_dict(),
            "failure_tolerance": tol,
        })

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}\n")

    for d in report["designs"]:
        print(f"K={d['n_blocks']} (min valid {d['min_valid_blocks']}):")
        print(f"  sign-flip at min valid : min p={d['sign_flip_min_attainable_p']:.4f} "
              f"feasible={d['sign_flip_feasible_at_min_valid']}")
        print(f"  equivalence power      : ", end="")
        print("  ".join(f"{r['sd_multiplier']:g}x={r['p_equivalent']:.3f}"
                        for r in d["sensitivity"]["curve"]))
        print(f"  breaking multiplier    : {d['sensitivity']['breaking_multiplier']}")
        print(f"  false equivalence at boundary: "
              f"{d['specificity_at_boundary']['p_equivalent']:.4f} (alpha={args.alpha})")
        t = d["failure_tolerance"]
        print(f"  P(enough valid blocks) at per-arm failure upper95="
              f"{t['per_arm_failure_rate_upper95']:.3f}: "
              f"{t['p_enough_valid_blocks_at_upper95']:.3f}")
        print(f"  max tolerable per-arm failure rate (>={t['survival_target']}): "
              f"{t['max_tolerable_per_arm_failure_rate']}")
        print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
