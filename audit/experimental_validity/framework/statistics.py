"""
Generic statistical procedures for Experimental Validity.

The statistical unit throughout this module is the independent
training-run / model replicate. Every function here operates on a
1-D array of one scalar metric per replicate.

REVISION NOTE (this file was corrected for an estimand/test mismatch):
the primary analysis previously reported mean_diff/Cohen's d_z (which
target the MEAN of paired differences) but drew its p-value from
scipy.stats.wilcoxon, whose actual null hypothesis - per SciPy's own
documentation - is that the distribution of differences is SYMMETRIC
ABOUT ZERO, not that the mean equals zero. That is a different claim
unless symmetry is separately assumed.

Since the frozen primary estimand for H-EV-SHUFFLE and
H-EV-REPRESENTATION is mu_D = E[D] (the mean paired accuracy
difference), the primary inferential test is now the PAIRED T-TEST
(H0: mu_D = 0), which targets the mean directly and requires no
symmetry assumption (only that the sampling distribution of the mean
is approximately normal, which the paired t-test relies on via the
CLT for the replicate counts used here). Cohen's d_z is retained
unchanged - it is literally the standardized statistic underlying the
paired t-test (t = d_z * sqrt(n)), so effect size and test are now
fully coherent.

The percentile bootstrap CI is RETAINED as a secondary, explicitly
labeled sensitivity check (robust to non-normality, does not rely on
the same assumption as the primary t-based CI) - it no longer drives
any decision.

TOST equivalence (Schuirmann, 1987; Lakens, 2017, Soc. Psychol.
Personal. Sci. 8(4):355-362) is implemented here as the single
authoritative equivalence procedure for a mean paired difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from framework.confidence_intervals import percentile_bootstrap_ci
from framework.effect_sizes import cohens_dz


@dataclass(frozen=True)
class DescriptiveStats:
    n: int
    mean: float
    sd: float
    minimum: float
    maximum: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "min": self.minimum,
            "max": self.maximum,
            "confidence_interval": {"low": self.ci_low, "high": self.ci_high},
        }


def bootstrap_ci_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Thin wrapper over the authoritative bootstrap CI implementation."""
    return percentile_bootstrap_ci(values, rng=rng, statistic=np.mean, n_boot=n_boot, alpha=alpha)


def descriptive_statistics(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> DescriptiveStats:
    """
    Descriptive statistics for a set of replicate-level metric values
    (EV-BASELINE, EV-NOISE). No hypothesis test is performed here -
    these experiments are foundational/characterizing, not confirmatory.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        ci_low, ci_high = float("nan"), float("nan")
    else:
        ci_low, ci_high = bootstrap_ci_mean(values, rng=rng, n_boot=n_boot, alpha=alpha)
    return DescriptiveStats(
        n=len(values),
        mean=float(np.mean(values)) if len(values) else float("nan"),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        minimum=float(np.min(values)) if len(values) else float("nan"),
        maximum=float(np.max(values)) if len(values) else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
    )


@dataclass(frozen=True)
class PairedAnalysisResult:
    n_pairs: int
    mean_diff: float
    sd_diff: float
    se_diff: float
    # Primary CI: t-distribution based, matches the primary test exactly.
    ci_low: float
    ci_high: float
    effect_size_dz: float
    # Primary difference-detection test.
    test_statistic: Optional[float]
    degrees_of_freedom: Optional[int]
    p_value: Optional[float]
    test_name: str
    # Secondary sensitivity CI (bootstrap), reported but not decision-driving.
    secondary_bootstrap_ci_low: float
    secondary_bootstrap_ci_high: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "mean_diff": self.mean_diff,
            "sd_diff": self.sd_diff,
            "se_diff": self.se_diff,
            "confidence_interval": {"low": self.ci_low, "high": self.ci_high, "method": "t_distribution"},
            "effect_size_cohens_dz": self.effect_size_dz,
            "test_name": self.test_name,
            "statistic": self.test_statistic,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": self.p_value,
            "secondary_bootstrap_ci": {
                "low": self.secondary_bootstrap_ci_low,
                "high": self.secondary_bootstrap_ci_high,
                "note": "Sensitivity check only; does not drive any decision.",
            },
            "warnings": self.warnings,
        }


def paired_analysis(
    condition_a: np.ndarray,
    condition_b: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> PairedAnalysisResult:
    """
    Paired analysis of D = condition_b - condition_a, where each index i
    is one matched replicate pair (same dataset instance, same model
    seed, differing only in the manipulated factor).

    PRIMARY ESTIMAND: mu_D = E[D]. PRIMARY TEST: paired t-test
    (H0: mu_D = 0, H1: mu_D != 0), via scipy.stats.ttest_1samp on the
    paired differences (mathematically identical to a paired t-test).
    PRIMARY CI: t-distribution based CI for mu_D.

    This function computes and reports a DIFFERENCE-DETECTION test. It
    does NOT and must never compute or imply a practical-equivalence /
    robustness verdict - that requires a predeclared epsilon and is
    handled exclusively via tost_equivalence() /
    framework.certificate.assess_practical_equivalence.
    """
    condition_a = np.asarray(condition_a, dtype=float)
    condition_b = np.asarray(condition_b, dtype=float)
    if len(condition_a) != len(condition_b):
        raise ValueError("Paired analysis requires equal-length matched arrays.")
    n = len(condition_a)
    warnings: list[str] = []

    diffs = condition_b - condition_a

    if n < 2:
        raise ValueError("Paired analysis requires at least 2 matched pairs.")

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se_diff = sd_diff / np.sqrt(n) if sd_diff > 0 else 0.0

    effect_size_dz, dz_warning = cohens_dz(diffs)
    if dz_warning:
        warnings.append(dz_warning)

    secondary_ci_low, secondary_ci_high = bootstrap_ci_mean(diffs, rng=rng, n_boot=n_boot, alpha=alpha)

    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    df: Optional[int] = None
    test_name = "paired_t_test"

    if np.allclose(diffs, 0.0):
        warnings.append("All paired differences are exactly zero; the paired t-test is degenerate (t undefined).")
        test_name = "paired_t_test_degenerate"
        ci_low, ci_high = 0.0, 0.0
    else:
        result = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        test_statistic, p_value, df = float(result.statistic), float(result.pvalue), n - 1
        if n < 10:
            warnings.append(
                f"n_pairs={n} is small; the paired t-test relies on approximate normality of "
                "the sampling distribution of the mean (CLT) or normality of the differences "
                "themselves. Interpret with appropriate caution and consult the secondary "
                "bootstrap CI below."
            )
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_low = mean_diff - t_crit * se_diff
        ci_high = mean_diff + t_crit * se_diff

    return PairedAnalysisResult(
        n_pairs=n,
        mean_diff=mean_diff,
        sd_diff=sd_diff,
        se_diff=se_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size_dz=effect_size_dz,
        test_statistic=test_statistic,
        degrees_of_freedom=df,
        p_value=p_value,
        test_name=test_name,
        secondary_bootstrap_ci_low=secondary_ci_low,
        secondary_bootstrap_ci_high=secondary_ci_high,
        warnings=warnings,
    )


@dataclass(frozen=True)
class TOSTResult:
    epsilon: float
    mean_diff: float
    se_diff: float
    n_pairs: int
    df: int
    p_lower: float
    p_upper: float
    p_tost: float
    alpha: float
    equivalent: bool

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "mean_diff": self.mean_diff,
            "se_diff": self.se_diff,
            "n_pairs": self.n_pairs,
            "degrees_of_freedom": self.df,
            "p_lower": self.p_lower,
            "p_upper": self.p_upper,
            "p_tost": self.p_tost,
            "alpha": self.alpha,
            "equivalent": self.equivalent,
            "method": "TOST (Schuirmann 1987; Lakens 2017)",
        }


def tost_equivalence(paired_result: PairedAnalysisResult, *, epsilon: float, alpha: float = 0.05) -> TOSTResult:
    """
    Two One-Sided Tests (TOST) procedure for the equivalence bounds
    (-epsilon, +epsilon) around the paired mean difference mu_D.

        H0_lower: mu_D <= -epsilon   vs.  H1_lower: mu_D > -epsilon
        H0_upper: mu_D >= +epsilon   vs.  H1_upper: mu_D < +epsilon

    Equivalence (reject the non-equivalence null) is concluded iff
    BOTH one-sided nulls are rejected, i.e. max(p_lower, p_upper) < alpha
    (Schuirmann, 1987; Lakens, 2017). No additional multiplicity
    correction is applied within TOST itself - the intersection-union
    principle already controls the Type I error rate at alpha without it.

    Requires n_pairs >= 2 and non-degenerate variance in the paired
    differences (se_diff > 0); raises ValueError otherwise, since TOST
    is not meaningfully defined in that case.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if paired_result.se_diff <= 0:
        raise ValueError(
            "TOST requires a non-degenerate standard error of the paired differences "
            f"(got se_diff={paired_result.se_diff}); cannot compute an equivalence test."
        )
    df = paired_result.n_pairs - 1
    t_lower = (paired_result.mean_diff - (-epsilon)) / paired_result.se_diff
    t_upper = (paired_result.mean_diff - epsilon) / paired_result.se_diff

    p_lower = 1 - scipy_stats.t.cdf(t_lower, df=df)   # H0_lower: mu_D <= -eps
    p_upper = scipy_stats.t.cdf(t_upper, df=df)        # H0_upper: mu_D >= +eps
    p_tost = max(p_lower, p_upper)

    return TOSTResult(
        epsilon=epsilon,
        mean_diff=paired_result.mean_diff,
        se_diff=paired_result.se_diff,
        n_pairs=paired_result.n_pairs,
        df=df,
        p_lower=float(p_lower),
        p_upper=float(p_upper),
        p_tost=float(p_tost),
        alpha=alpha,
        equivalent=bool(p_tost < alpha),
    )
