"""
Simulation-based power analysis for Experimental Validity.

REVISION NOTE: previously simulated power for scipy.stats.wilcoxon,
which no longer matches the final analysis procedure (see
framework/statistics.py's revision note - the primary test is now the
paired t-test). This module now simulates power for the ACTUAL planned
procedures:

    - "difference_ttest": power to detect a true mean paired difference
      via the paired t-test used in production (framework.statistics.
      paired_analysis's primary test).
    - "equivalence_tost": power to correctly conclude equivalence via
      TOST when the true effect is within (-epsilon, +epsilon)
      (typically simulated at the true effect = 0, the standard
      "power to detect equivalence when truly equivalent" question).

Anti-misuse guards (do not remove or bypass these) - UNCHANGED from the
prior version:

    - `target_effect` must be supplied explicitly by the caller and
      tagged with a `source`. `source="observed_production"` is
      REFUSED (raises ValueError).
    - If no defensible target effect exists, use `power_unavailable()`.
    - `noise_sd` may legitimately come from pilot/calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import stats as scipy_stats


ALLOWED_TARGET_EFFECT_SOURCES = {"predeclared", "literature", "pilot_variance_only"}
FORBIDDEN_TARGET_EFFECT_SOURCES = {"observed_production"}


@dataclass(frozen=True)
class PowerResult:
    procedure: str
    target_effect: float
    target_effect_source: str
    noise_sd: float
    alpha: float
    epsilon: Optional[float]
    n_replicates: int
    n_simulations: int
    empirical_power: float
    prospective: bool

    def to_dict(self) -> dict:
        return {
            "procedure": self.procedure,
            "target_effect": self.target_effect,
            "target_effect_source": self.target_effect_source,
            "noise_sd": self.noise_sd,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "n_replicates": self.n_replicates,
            "n_simulations": self.n_simulations,
            "empirical_power": self.empirical_power,
            "prospective": self.prospective,
        }


def power_unavailable(reason: str) -> dict:
    """Explicit, honest 'power cannot be determined' record."""
    return {"prospective_power": "UNAVAILABLE", "reason": reason}


def _validate_target_effect_source(source: str) -> None:
    if source in FORBIDDEN_TARGET_EFFECT_SOURCES:
        raise ValueError(
            f"target_effect_source={source!r} is forbidden: a target effect size "
            "derived from the observed production result cannot be presented as "
            "prospective power. Supply a predeclared, literature-based, or "
            "pilot-variance-only target instead, or call power_unavailable()."
        )
    if source not in ALLOWED_TARGET_EFFECT_SOURCES:
        raise ValueError(
            f"Unrecognized target_effect_source={source!r}. "
            f"Allowed: {sorted(ALLOWED_TARGET_EFFECT_SOURCES)}"
        )


def simulate_power(
    *,
    procedure: Literal["difference_ttest", "equivalence_tost"] = "difference_ttest",
    target_effect: float,
    target_effect_source: str,
    noise_sd: float,
    n_replicates: int,
    alpha: float = 0.05,
    epsilon: Optional[float] = None,
    n_simulations: int = 2000,
    rng: np.random.Generator,
) -> PowerResult:
    """
    Monte Carlo power for the paired difference-detection t-test
    (procedure="difference_ttest") or for TOST equivalence
    (procedure="equivalence_tost"), under an assumed true paired-
    difference distribution Normal(target_effect, noise_sd).

    For procedure="equivalence_tost", `epsilon` is required and power is
    the empirical fraction of simulated datasets for which TOST
    concludes equivalence (max(p_lower, p_upper) < alpha). To assess
    "power to correctly conclude equivalence when truly equivalent",
    callers should pass target_effect=0.0 with
    target_effect_source="pilot_variance_only" (the standard framing for
    this question - see docs/statistical_plan.md).
    """
    _validate_target_effect_source(target_effect_source)
    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive.")
    if n_replicates < 2:
        raise ValueError("n_replicates must be >= 2 for a paired analysis.")
    if procedure == "equivalence_tost" and epsilon is None:
        raise ValueError("epsilon is required when procedure='equivalence_tost'.")

    rejections = 0
    valid_sims = 0
    for _ in range(n_simulations):
        diffs = rng.normal(loc=target_effect, scale=noise_sd, size=n_replicates)
        sd = np.std(diffs, ddof=1)
        if sd == 0:
            continue
        valid_sims += 1
        se = sd / np.sqrt(n_replicates)
        mean_diff = np.mean(diffs)
        df = n_replicates - 1

        if procedure == "difference_ttest":
            t_stat = mean_diff / se
            p = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=df))
            if p < alpha:
                rejections += 1
        elif procedure == "equivalence_tost":
            t_lower = (mean_diff - (-epsilon)) / se
            t_upper = (mean_diff - epsilon) / se
            p_lower = 1 - scipy_stats.t.cdf(t_lower, df=df)
            p_upper = scipy_stats.t.cdf(t_upper, df=df)
            if max(p_lower, p_upper) < alpha:
                rejections += 1
        else:
            raise ValueError(f"Unknown procedure: {procedure!r}")

    empirical_power = rejections / valid_sims if valid_sims > 0 else float("nan")

    return PowerResult(
        procedure=procedure,
        target_effect=target_effect,
        target_effect_source=target_effect_source,
        noise_sd=noise_sd,
        alpha=alpha,
        epsilon=epsilon,
        n_replicates=n_replicates,
        n_simulations=n_simulations,
        empirical_power=empirical_power,
        prospective=(target_effect_source != "pilot_variance_only"),
    )


def required_replicates_for_power(
    *,
    procedure: Literal["difference_ttest", "equivalence_tost"] = "difference_ttest",
    target_effect: float,
    target_effect_source: str,
    noise_sd: float,
    alpha: float = 0.05,
    epsilon: Optional[float] = None,
    target_power: float = 0.80,
    n_simulations: int = 1000,
    max_replicates: int = 200,
    rng: np.random.Generator,
) -> Optional[int]:
    """
    Search for the smallest n_replicates (up to max_replicates) whose
    simulated power meets or exceeds target_power. Returns None if
    max_replicates is insufficient - callers must treat this as
    "infeasible under this budget", never silently substitute a lower
    replicate count.
    """
    for n in range(2, max_replicates + 1):
        result = simulate_power(
            procedure=procedure, target_effect=target_effect, target_effect_source=target_effect_source,
            noise_sd=noise_sd, n_replicates=n, alpha=alpha, epsilon=epsilon,
            n_simulations=n_simulations, rng=rng,
        )
        if result.empirical_power >= target_power:
            return n
    return None


class UnderpoweredReplicatePlanError(ValueError):
    pass


def validate_replicate_plan_against_power(
    *,
    minimum_valid_replicates: int,
    power_analysis_required_n: Optional[int],
    underpowered_justification: Optional[str] = None,
) -> None:
    """
    Guard against an evidentiary experiment being configured with fewer
    minimum-valid-replicates than its own declared prospective power
    analysis says are required (e.g. power analysis says n=24 are
    needed for 80% power, but minimum_valid_replicates=20 - the
    certificate could then declare a confirmatory decision at a
    replicate count known in advance to be underpowered for the
    predeclared effect/power target).

    `requested_replicates` is intentionally NOT constrained here beyond
    what callers already enforce elsewhere (requested >= minimum) -
    requesting MORE than the power-sized n is the normal, expected way
    to predeclare tolerance for replicate attrition (failed/invalid
    runs), and is never restricted by this function.

    power_analysis_required_n=None means no power analysis has been
    declared/performed yet - this function does not raise in that case
    (the separate placeholder-based fail-closed check in
    scripts/run_ev.py already blocks production runs with an
    unresolved replicate count; this function only adds a check for
    when a concrete required_n IS declared).

    If minimum_valid_replicates is below the declared required_n, an
    explicit, non-empty underpowered_justification string must be
    supplied, or this raises UnderpoweredReplicatePlanError. The
    justification is recorded (by the caller) in provenance, never
    invented here.
    """
    if power_analysis_required_n is None:
        return
    if minimum_valid_replicates >= power_analysis_required_n:
        return
    if not underpowered_justification or not underpowered_justification.strip():
        raise UnderpoweredReplicatePlanError(
            f"minimum_valid_replicates={minimum_valid_replicates} is below the "
            f"power-analysis-required n={power_analysis_required_n}, and no "
            "underpowered_justification was supplied. Refusing to configure an "
            "evidentiary experiment that is known in advance to be underpowered "
            "for its own predeclared target effect/power without an explicit, "
            "recorded justification for accepting that."
        )
