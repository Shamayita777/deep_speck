"""
II-4 confirmatory statistical analysis.

Block-level paired analysis with the dataset block as experimental unit.
Generic: operates on a list of signed per-block differences and knows
nothing about depth, Speck, or accuracy semantics beyond "higher is
better is not assumed" - the analysis is two-sided and direction-aware.

THREE LOGICALLY SEPARATE QUESTIONS, never conflated:

  1. DIFFERENCE DETECTION - is Delta distinguishable from 0?
     Paired t-test. A non-significant result is NOT evidence of no
     effect and NOT evidence of equivalence.

  2. PRACTICAL MATERIALITY - is Delta at least delta in some direction?
     Decided on the SIGNED 95% CI [L, U]:
         materially higher iff L > +delta
         materially lower  iff U < -delta
     NOT on the point estimate: a point-estimate rule carries no Type-I
     control and its false-"material" rate is governed entirely by the
     unknown sigma_Delta.

  3. EQUIVALENCE - is Delta within (-delta, +delta)?
     TOST (Schuirmann 1987; Lakens 2017), computed independently of (1).
     Equivalence is NEVER inferred from a non-significant (1).

Directional claims from the two-sided 95% CI are each one-sided at
alpha/2 = 2.5%. TOST at alpha=0.05 corresponds to the 90% CI lying
inside the margins. Both conventions are recorded explicitly so a
reader can see which error rate attaches to which claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Optional

import numpy as np
from scipy import stats

ANALYSIS_VERSION = "ii4-analysis-v1"


class MaterialityVerdict(str, Enum):
    MATERIALLY_HIGHER = "MATERIALLY_HIGHER"      # L > +delta
    MATERIALLY_LOWER = "MATERIALLY_LOWER"        # U < -delta
    NO_DIRECTIONAL_CLAIM = "NO_DIRECTIONAL_CLAIM"
    INSUFFICIENT_BLOCKS = "INSUFFICIENT_BLOCKS"


class EquivalenceVerdict(str, Enum):
    EQUIVALENT_WITHIN_MARGIN = "EQUIVALENT_WITHIN_MARGIN"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass
class II4AnalysisResult:
    n_blocks_valid: int
    n_blocks_requested: int
    min_valid_required: int
    sufficient: bool

    mean_difference: Optional[float] = None
    sd_difference: Optional[float] = None
    se_difference: Optional[float] = None
    ci95_low: Optional[float] = None
    ci95_high: Optional[float] = None
    ci90_low: Optional[float] = None
    ci90_high: Optional[float] = None

    t_statistic: Optional[float] = None
    df: Optional[int] = None
    p_value: Optional[float] = None
    cohens_dz: Optional[float] = None

    sign_flip_p: Optional[float] = None
    sign_flip_feasible: Optional[bool] = None
    sign_flip_min_attainable_p: Optional[float] = None

    materiality: MaterialityVerdict = MaterialityVerdict.INSUFFICIENT_BLOCKS
    equivalence: EquivalenceVerdict = EquivalenceVerdict.NOT_ASSESSED
    tost_p_lower: Optional[float] = None
    tost_p_upper: Optional[float] = None
    tost_p: Optional[float] = None

    delta: Optional[float] = None
    alpha: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    analysis_version: str = ANALYSIS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_version": self.analysis_version,
            "replication": {
                "n_blocks_requested": self.n_blocks_requested,
                "n_blocks_valid": self.n_blocks_valid,
                "min_valid_required": self.min_valid_required,
                "sufficient": self.sufficient,
            },
            "confirmatory": {
                "estimand": "E[Delta] ; Delta_k = Y_k(declared) - Y_k(realized)",
                "experimental_unit": "dataset block",
                "mean_difference": self.mean_difference,
                "sd_difference": self.sd_difference,
                "se_difference": self.se_difference,
                "ci95": {"low": self.ci95_low, "high": self.ci95_high},
                "ci90": {"low": self.ci90_low, "high": self.ci90_high},
                "paired_t": {"statistic": self.t_statistic, "df": self.df,
                             "p_value": self.p_value},
                "cohens_dz": self.cohens_dz,
                "materiality": {
                    "verdict": self.materiality.value,
                    "delta": self.delta,
                    "rule": ("signed 95% CI: MATERIALLY_HIGHER iff L > +delta; "
                             "MATERIALLY_LOWER iff U < -delta"),
                    "directional_claim_alpha": (self.alpha / 2) if self.alpha else None,
                },
                "equivalence": {
                    "verdict": self.equivalence.value,
                    "margin": self.delta,
                    "method": "TOST (Schuirmann 1987; Lakens 2017), alpha-level two one-sided t-tests",
                    "p_lower": self.tost_p_lower,
                    "p_upper": self.tost_p_upper,
                    "p_tost": self.tost_p,
                    "note": ("Computed independently of the difference test. Equivalence is "
                             "never inferred from a non-significant difference test."),
                },
            },
            "secondary_robustness": {
                "sign_flip_p": self.sign_flip_p,
                "sign_flip_feasible": self.sign_flip_feasible,
                "sign_flip_min_attainable_p": self.sign_flip_min_attainable_p,
            },
            "alpha": self.alpha,
            "warnings": list(self.warnings),
        }


def exact_sign_flip_p(diffs: np.ndarray) -> float:
    """
    Exact paired sign-flip randomization p-value (two-sided).

    Permuted object: the SIGN of each block difference. Exchangeability
    assumption: under H0 of no conformance-factor effect, the labelling
    'declared minus realized' within a block is arbitrary, because arm
    assignment within a block is a design choice rather than a property
    of the data.

    Enumerated exactly (2^K); with K<=20 this is cheap and avoids Monte
    Carlo error.
    """
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    observed = abs(float(np.mean(diffs)))
    count = 0
    for signs in product((1.0, -1.0), repeat=k):
        if abs(float(np.mean(diffs * np.array(signs)))) >= observed - 1e-15:
            count += 1
    return count / (2 ** k)


def analyse_ii4(
    block_differences: list[float],
    *,
    n_blocks_requested: int,
    min_valid_blocks: int,
    delta: float,
    alpha: float = 0.05,
) -> II4AnalysisResult:
    """
    Run the frozen II-4 confirmatory analysis on signed per-block
    differences. Returns a result object; NEVER returns a bare PASS and
    never derives a decision from a p-value alone.
    """
    diffs = np.asarray([d for d in block_differences], dtype=float)
    n = len(diffs)
    result = II4AnalysisResult(
        n_blocks_valid=n,
        n_blocks_requested=n_blocks_requested,
        min_valid_required=min_valid_blocks,
        sufficient=(n >= min_valid_blocks),
        delta=delta,
        alpha=alpha,
    )

    if not result.sufficient:
        result.warnings.append(
            f"Only {n} valid blocks (minimum {min_valid_blocks}). No confirmatory verdict is "
            "issued; the result is INCONCLUSIVE by the preregistered rule, not by inspection "
            "of the data."
        )
        result.materiality = MaterialityVerdict.INSUFFICIENT_BLOCKS
        result.equivalence = EquivalenceVerdict.NOT_ASSESSED
        return result

    if n < 2:
        result.warnings.append("Fewer than 2 blocks; no variance estimate possible.")
        return result

    # Sign-flip resolution depends only on the number of blocks, so it is
    # recorded before any early return on degenerate variance.
    min_p = 2 / (2 ** n)
    result.sign_flip_min_attainable_p = min_p
    result.sign_flip_feasible = min_p < alpha

    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1

    result.mean_difference = mean
    result.sd_difference = sd
    result.se_difference = se
    result.df = df

    if se == 0.0:
        result.warnings.append(
            "Zero variance across block differences: t-based inference, Cohen's d_z and TOST "
            "are undefined. Reporting the point estimate only."
        )
        result.materiality = MaterialityVerdict.NO_DIRECTIONAL_CLAIM
        result.equivalence = EquivalenceVerdict.INCONCLUSIVE
        return result

    # --- (1) difference detection -------------------------------------
    t_stat = mean / se
    result.t_statistic = t_stat
    result.p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    result.cohens_dz = mean / sd

    # --- confidence intervals -----------------------------------------
    t95 = stats.t.ppf(1 - alpha / 2, df)          # two-sided 95%
    t90 = stats.t.ppf(1 - alpha, df)              # 90% == TOST-equivalent interval at alpha
    result.ci95_low, result.ci95_high = mean - t95 * se, mean + t95 * se
    result.ci90_low, result.ci90_high = mean - t90 * se, mean + t90 * se

    # --- (2) practical materiality: SIGNED 95% CI ----------------------
    if result.ci95_low > delta:
        result.materiality = MaterialityVerdict.MATERIALLY_HIGHER
    elif result.ci95_high < -delta:
        result.materiality = MaterialityVerdict.MATERIALLY_LOWER
    else:
        result.materiality = MaterialityVerdict.NO_DIRECTIONAL_CLAIM

    # --- (3) equivalence: TOST, computed independently -----------------
    t_lower = (mean - (-delta)) / se
    t_upper = (mean - delta) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df))   # H0: Delta <= -delta
    p_upper = float(stats.t.cdf(t_upper, df))       # H0: Delta >= +delta
    result.tost_p_lower, result.tost_p_upper = p_lower, p_upper
    result.tost_p = max(p_lower, p_upper)
    if result.tost_p < alpha:
        result.equivalence = EquivalenceVerdict.EQUIVALENT_WITHIN_MARGIN
    elif result.materiality in (MaterialityVerdict.MATERIALLY_HIGHER,
                                MaterialityVerdict.MATERIALLY_LOWER):
        result.equivalence = EquivalenceVerdict.NOT_EQUIVALENT
    else:
        result.equivalence = EquivalenceVerdict.INCONCLUSIVE

    # --- secondary robustness -----------------------------------------
    if result.sign_flip_feasible:
        result.sign_flip_p = exact_sign_flip_p(diffs)
    else:
        result.warnings.append(
            f"Exact sign-flip test cannot reach alpha={alpha} with {n} blocks "
            f"(minimum attainable two-sided p = {min_p:.4f}); reported as infeasible rather "
            "than as a non-significant result."
        )

    if n < 10:
        result.warnings.append(
            f"n_blocks={n}: t-based inference relies on approximate normality of the block "
            "differences, which is assumed rather than testable at this sample size. The "
            "sign-flip analysis is the assumption-free cross-check."
        )
    return result
