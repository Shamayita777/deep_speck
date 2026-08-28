"""
D2 — Pairwise Hamming-Distance Goodness-of-Fit Audit.

Generic Dataset Integrity audit.

Scientific scope
----------------
D2 evaluates whether sampled pairwise Hamming distances between
binary feature vectors are compatible with an explicitly supplied
reference distribution.

For the Gohr/Speck case study, the adapter supplies:

    n_bits = 64
    reference = Binomial(64, 0.5)

Those are CASE-STUDY parameters. They are not hard-coded into the
generic D2 statistical machinery.

D2 is a distributional audit. It is not a direct near-duplicate
detector and it does not establish full statistical independence.

The audit examines six comparison types when the caller supplies
three partitions:

    - train within-partition
    - validation within-partition
    - test within-partition
    - train-validation cross-partition
    - train-test cross-partition
    - validation-test cross-partition

Pair observations are sampled uniformly with replacement. Within a
partition, self-pairs are excluded.

The implementation retains the original D2 decision semantics:

    PASS
        Every comparison has every replicate with an upper bootstrap
        TVD bound <= the pre-specified TVD threshold, and no
        replicate rejects the reference with Pearson chi-square.

    CONDITIONAL_PASS
        The practical TVD criterion passes everywhere, but at least
        one chi-square replicate rejects the reference.

    FAIL
        At least one comparison/replicate has an upper bootstrap TVD
        bound above the threshold.

Important scientific limitation
-------------------------------
The reference distribution is a model assumption. A PASS means only
that the audited pairwise-distance distribution is compatible with
that supplied reference under the stated sampling design and
tolerance. It does not prove independence, absence of near
duplicates, absence of metadata leakage, or absence of higher-order
dependence.

The bootstrap output is retained as a percentile-bootstrap uncertainty
interval, following the original implementation. It must not be
described as a formally calibrated equivalence-test confidence bound
without additional methodological justification.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from audit.dataset.common import (
    array_sha256,
    build_provenance,
    make_certificate,
    write_certificate,
)


# ============================================================
# Audit configuration
# ============================================================

AUDIT_ID = "D2"
AUDIT_NAME = "Pairwise Hamming-Distance Goodness-of-Fit Audit"

CHI_SQUARE_ALPHA = 0.01
CHI_SQUARE_MIN_EXPECTED = 5.0

TVD_THRESHOLD = 0.05
TVD_CONFIDENCE_LEVEL = 0.95

DEFAULT_PAIRS_PER_COMPARISON = 100_000
DEFAULT_AUDIT_REPLICATES = 5
DEFAULT_TVD_BOOTSTRAP_REPLICATES = 1_000


# ============================================================
# Validation helpers
# ============================================================

def _validate_binary_features(
    features: np.ndarray,
    *,
    n_bits: int,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Validate and return a 2-D binary feature matrix."""

    array = np.asarray(features)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D feature array; "
            f"got ndim={array.ndim}."
        )

    if array.shape[1] != n_bits:
        raise ValueError(
            f"{name} must have {n_bits} features; "
            f"got shape={array.shape}."
        )

    if not allow_empty and array.shape[0] < 1:
        raise ValueError(
            f"{name} must contain at least one sample."
        )

    if not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"{name} contains values other than 0 and 1."
        )

    return array


def _validate_reference_pmf(
    reference_pmf: np.ndarray,
    *,
    n_bits: int,
) -> np.ndarray:
    """Validate a discrete PMF indexed by Hamming distance."""

    pmf = np.asarray(reference_pmf, dtype=np.float64)

    if pmf.ndim != 1:
        raise ValueError("reference_pmf must be one-dimensional.")

    if len(pmf) != n_bits + 1:
        raise ValueError(
            f"reference_pmf must have length {n_bits + 1}; "
            f"got {len(pmf)}."
        )

    if not np.all(np.isfinite(pmf)):
        raise ValueError("reference_pmf must contain finite values.")

    if np.any(pmf < 0.0):
        raise ValueError("reference_pmf cannot contain negative values.")

    total = float(pmf.sum())

    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(
            f"reference_pmf must sum to 1; got {total}."
        )

    return pmf


# ============================================================
# Reference distribution
# ============================================================

def binomial_reference_distribution(
    *,
    n_bits: int,
    p: float = 0.5,
) -> np.ndarray:
    """
    Return the exact Binomial(n_bits, p) probability mass function.

    The returned array has length n_bits + 1 and index d represents:

        P(Hamming distance == d)
    """

    if n_bits < 1:
        raise ValueError("n_bits must be >= 1.")

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1].")

    probabilities = np.empty(n_bits + 1, dtype=np.float64)

    for distance in range(n_bits + 1):
        probabilities[distance] = (
            math.comb(n_bits, distance)
            * (p ** distance)
            * ((1.0 - p) ** (n_bits - distance))
        )

    total = float(probabilities.sum())

    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Failed to construct a valid binomial PMF.")

    probabilities /= total
    return probabilities


# ============================================================
# Pair sampling
# ============================================================

def sample_within_partition_distances(
    features: np.ndarray,
    *,
    num_pairs: int,
    rng: np.random.Generator,
    n_bits: Optional[int] = None,
) -> np.ndarray:
    """
    Sample Hamming distances between distinct samples in one partition.

    Ordered sample-index pairs are sampled uniformly with replacement.
    Self-pairs are excluded.
    """

    features = np.asarray(features)

    if n_bits is None:
        n_bits = int(features.shape[1]) if features.ndim == 2 else 0

    features = _validate_binary_features(
        features,
        n_bits=n_bits,
        name="features",
    )

    if num_pairs < 1:
        raise ValueError("num_pairs must be >= 1.")

    n_samples = features.shape[0]

    if n_samples < 2:
        raise ValueError(
            "At least two samples are required for within-partition "
            "pair sampling."
        )

    left = rng.integers(
        0,
        n_samples,
        size=num_pairs,
        dtype=np.int64,
    )

    right = rng.integers(
        0,
        n_samples - 1,
        size=num_pairs,
        dtype=np.int64,
    )

    # Uniformly maps right onto all indices except left.
    right = np.where(right >= left, right + 1, right)

    distances = np.count_nonzero(
        features[left] != features[right],
        axis=1,
    )

    return distances.astype(np.int16, copy=False)


def sample_cross_partition_distances(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    num_pairs: int,
    rng: np.random.Generator,
    n_bits: Optional[int] = None,
) -> np.ndarray:
    """
    Sample Hamming distances between one sample from each partition.
    """

    features_a = np.asarray(features_a)
    features_b = np.asarray(features_b)

    if n_bits is None:
        if features_a.ndim == 2:
            n_bits = int(features_a.shape[1])
        elif features_b.ndim == 2:
            n_bits = int(features_b.shape[1])
        else:
            n_bits = 0

    features_a = _validate_binary_features(
        features_a,
        n_bits=n_bits,
        name="features_a",
    )
    features_b = _validate_binary_features(
        features_b,
        n_bits=n_bits,
        name="features_b",
    )

    if num_pairs < 1:
        raise ValueError("num_pairs must be >= 1.")

    left = rng.integers(
        0,
        features_a.shape[0],
        size=num_pairs,
        dtype=np.int64,
    )

    right = rng.integers(
        0,
        features_b.shape[0],
        size=num_pairs,
        dtype=np.int64,
    )

    distances = np.count_nonzero(
        features_a[left] != features_b[right],
        axis=1,
    )

    return distances.astype(np.int16, copy=False)


# ============================================================
# Statistical utilities
# ============================================================

def _merge_sparse_bins(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_expected: float = CHI_SQUARE_MIN_EXPECTED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge adjacent categories until expected counts reach the
    configured minimum.

    The reference probabilities themselves are not changed; only
    categories used by the Pearson statistic are grouped.
    """

    if len(observed) != len(expected):
        raise ValueError(
            "Observed and expected arrays must have equal length."
        )

    if minimum_expected <= 0.0:
        raise ValueError("minimum_expected must be > 0.")

    merged_observed: list[float] = []
    merged_expected: list[float] = []

    running_observed = 0.0
    running_expected = 0.0

    for obs, exp in zip(observed, expected):
        running_observed += float(obs)
        running_expected += float(exp)

        if running_expected >= minimum_expected:
            merged_observed.append(running_observed)
            merged_expected.append(running_expected)
            running_observed = 0.0
            running_expected = 0.0

    if running_expected > 0.0:
        if merged_expected:
            merged_observed[-1] += running_observed
            merged_expected[-1] += running_expected
        else:
            merged_observed.append(running_observed)
            merged_expected.append(running_expected)

    return (
        np.asarray(merged_observed, dtype=np.float64),
        np.asarray(merged_expected, dtype=np.float64),
    )


def chi_square_gof(
    distances: np.ndarray,
    *,
    reference_pmf: np.ndarray,
    n_bits: Optional[int] = None,
    alpha: float = CHI_SQUARE_ALPHA,
) -> Dict[str, Any]:
    """Compute Pearson chi-square goodness-of-fit."""

    distances = np.asarray(distances)

    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("distances must be a non-empty 1-D array.")

    if n_bits is None:
        n_bits = len(reference_pmf) - 1

    reference_pmf = _validate_reference_pmf(
        reference_pmf,
        n_bits=n_bits,
    )

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    if np.any(distances < 0) or np.any(distances > n_bits):
        raise ValueError("distances contain values outside [0, n_bits].")

    from scipy.stats import chi2

    observed = np.bincount(
        distances.astype(np.int64, copy=False),
        minlength=len(reference_pmf),
    ).astype(np.float64)

    expected = reference_pmf * float(len(distances))

    observed_merged, expected_merged = _merge_sparse_bins(
        observed,
        expected,
    )

    statistic = float(
        np.sum(
            (observed_merged - expected_merged) ** 2
            / expected_merged
        )
    )

    degrees_of_freedom = len(expected_merged) - 1

    if degrees_of_freedom < 1:
        raise ValueError(
            "Chi-square calculation requires at least two grouped bins."
        )

    p_value = float(
        chi2.sf(
            statistic,
            degrees_of_freedom,
        )
    )

    return {
        "chi_square_statistic": statistic,
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": p_value,
        "alpha": alpha,
        "reject_reference": bool(p_value < alpha),
        "bins_after_expected_count_merging": int(len(expected_merged)),
        "minimum_expected_count": CHI_SQUARE_MIN_EXPECTED,
    }


def total_variation_distance(
    distances: np.ndarray,
    *,
    reference_pmf: np.ndarray,
    n_bits: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute empirical TVD from the supplied reference PMF."""

    distances = np.asarray(distances)

    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("distances must be a non-empty 1-D array.")

    if n_bits is None:
        n_bits = len(reference_pmf) - 1

    reference_pmf = _validate_reference_pmf(
        reference_pmf,
        n_bits=n_bits,
    )

    if np.any(distances < 0) or np.any(distances > n_bits):
        raise ValueError("distances contain values outside [0, n_bits].")

    observed_counts = np.bincount(
        distances.astype(np.int64, copy=False),
        minlength=len(reference_pmf),
    ).astype(np.float64)

    observed_pmf = observed_counts / float(len(distances))

    tvd = 0.5 * float(
        np.sum(np.abs(observed_pmf - reference_pmf))
    )

    return {
        "tvd": tvd,
        "threshold": TVD_THRESHOLD,
        "within_threshold": bool(tvd <= TVD_THRESHOLD),
    }


def tvd_bootstrap_confidence_interval(
    distances: np.ndarray,
    *,
    reference_pmf: np.ndarray,
    n_bits: Optional[int] = None,
    confidence_level: float = TVD_CONFIDENCE_LEVEL,
    bootstrap_replicates: int = DEFAULT_TVD_BOOTSTRAP_REPLICATES,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Estimate a percentile-bootstrap interval for empirical TVD.

    This is deliberately named as an uncertainty interval in the
    scientific documentation. The original audit called it a CI;
    the numerical procedure is preserved.
    """

    distances = np.asarray(distances)

    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("distances must be a non-empty 1-D array.")

    if n_bits is None:
        n_bits = len(reference_pmf) - 1

    reference_pmf = _validate_reference_pmf(
        reference_pmf,
        n_bits=n_bits,
    )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be >= 1.")

    if rng is None:
        rng = np.random.default_rng()

    observed_counts = np.bincount(
        distances.astype(np.int64, copy=False),
        minlength=len(reference_pmf),
    ).astype(np.int64)

    observed_pmf = observed_counts / float(len(distances))

    bootstrap_counts = rng.multinomial(
        len(distances),
        observed_pmf,
        size=bootstrap_replicates,
    )

    bootstrap_pmfs = bootstrap_counts / float(len(distances))

    bootstrap_tvds = 0.5 * np.sum(
        np.abs(bootstrap_pmfs - reference_pmf),
        axis=1,
    )

    alpha = 1.0 - confidence_level

    lower = float(np.quantile(bootstrap_tvds, alpha / 2.0))
    upper = float(np.quantile(bootstrap_tvds, 1.0 - alpha / 2.0))

    return {
        "confidence_level": confidence_level,
        "bootstrap_replicates": int(bootstrap_replicates),
        "tvd_ci_lower": lower,
        "tvd_ci_upper": upper,
        "upper_ci_within_threshold": bool(
            upper <= TVD_THRESHOLD
        ),
    }


def summarize_distance_sample(
    distances: np.ndarray,
    *,
    reference_pmf: np.ndarray,
    bootstrap_rng: np.random.Generator,
    n_bits: Optional[int] = None,
    tvd_bootstrap_replicates: int = DEFAULT_TVD_BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Produce the complete statistical summary for one replicate."""

    distances = np.asarray(distances)

    if len(distances) == 0:
        raise ValueError("Cannot analyze an empty distance sample.")

    if n_bits is None:
        n_bits = len(reference_pmf) - 1

    chi_square = chi_square_gof(
        distances,
        reference_pmf=reference_pmf,
        n_bits=n_bits,
    )

    tvd = total_variation_distance(
        distances,
        reference_pmf=reference_pmf,
        n_bits=n_bits,
    )

    tvd_uncertainty = tvd_bootstrap_confidence_interval(
        distances,
        reference_pmf=reference_pmf,
        n_bits=n_bits,
        confidence_level=TVD_CONFIDENCE_LEVEL,
        bootstrap_replicates=tvd_bootstrap_replicates,
        rng=bootstrap_rng,
    )

    histogram = np.bincount(
        distances.astype(np.int64, copy=False),
        minlength=len(reference_pmf),
    )

    return {
        "sampled_pairs": int(len(distances)),
        "mean_hamming_distance": float(np.mean(distances)),
        "std_hamming_distance": float(np.std(distances)),
        "minimum_hamming_distance": int(np.min(distances)),
        "maximum_hamming_distance": int(np.max(distances)),
        "distance_histogram": [int(x) for x in histogram],
        "chi_square": chi_square,
        "total_variation": tvd,
        "tvd_uncertainty": tvd_uncertainty,
    }


# ============================================================
# Replicate aggregation
# ============================================================

def aggregate_replicate_results(
    replicate_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate replicate results without averaging away unfavorable
    replicates.

    Replicates are independent pair-sampling procedures over the same
    dataset instance. They are not independent datasets.
    """

    if not replicate_results:
        raise ValueError("At least one replicate is required.")

    replicate_count = len(replicate_results)

    chi_square_passes = sum(
        not result["chi_square"]["reject_reference"]
        for result in replicate_results
    )

    tvd_passes = sum(
        result["total_variation"]["within_threshold"]
        for result in replicate_results
    )

    p_values = [
        float(result["chi_square"]["p_value"])
        for result in replicate_results
    ]

    tvds = [
        float(result["total_variation"]["tvd"])
        for result in replicate_results
    ]

    tvd_upper_cis = [
        float(result["tvd_uncertainty"]["tvd_ci_upper"])
        for result in replicate_results
    ]

    return {
        "replicate_count": int(replicate_count),
        "replicates": [dict(result) for result in replicate_results],
        "chi_square_pass_count": int(chi_square_passes),
        "chi_square_fail_count": int(
            replicate_count - chi_square_passes
        ),
        "tvd_pass_count": int(tvd_passes),
        "tvd_fail_count": int(replicate_count - tvd_passes),
        "minimum_chi_square_p_value": min(p_values),
        "maximum_chi_square_p_value": max(p_values),
        "minimum_tvd": min(tvds),
        "maximum_tvd": max(tvds),
        "all_chi_square_pass": bool(
            chi_square_passes == replicate_count
        ),
        "all_tvd_pass": bool(
            tvd_passes == replicate_count
        ),
        "minimum_tvd_ci_upper": min(tvd_upper_cis),
        "maximum_tvd_ci_upper": max(tvd_upper_cis),
        "all_tvd_ci_pass": bool(
            max(tvd_upper_cis) <= TVD_THRESHOLD
        ),
    }


# ============================================================
# Audit decision
# ============================================================

def evaluate_d2(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Apply the original D2 decision rule.

    The six comparison-level analyses are treated as a fixed audit
    battery. No averaging across comparisons is used.
    """

    if not comparisons:
        raise ValueError("At least one comparison is required.")

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for name, result in comparisons.items():
        all_tvd_ci_pass = bool(result["all_tvd_ci_pass"])
        all_chi_square_pass = bool(result["all_chi_square_pass"])

        if not all_tvd_ci_pass:
            failures.append(
                {
                    "comparison": name,
                    "reason": (
                        "At least one pair-sampling replicate has an "
                        "upper TVD uncertainty bound above the "
                        f"pre-specified threshold of {TVD_THRESHOLD}."
                    ),
                    "replicate_count": result["replicate_count"],
                    "minimum_tvd": result["minimum_tvd"],
                    "maximum_tvd": result["maximum_tvd"],
                    "minimum_tvd_ci_upper": result[
                        "minimum_tvd_ci_upper"
                    ],
                    "maximum_tvd_ci_upper": result[
                        "maximum_tvd_ci_upper"
                    ],
                }
            )

        elif not all_chi_square_pass:
            warnings.append(
                {
                    "comparison": name,
                    "reason": (
                        "At least one pair-sampling replicate rejects "
                        "the reference at the pre-specified chi-square "
                        "alpha while all upper TVD uncertainty bounds "
                        "remain within the practical tolerance."
                    ),
                    "chi_square_pass_count": result[
                        "chi_square_pass_count"
                    ],
                    "chi_square_fail_count": result[
                        "chi_square_fail_count"
                    ],
                    "replicate_count": result["replicate_count"],
                    "minimum_tvd_ci_upper": result[
                        "minimum_tvd_ci_upper"
                    ],
                    "maximum_tvd_ci_upper": result[
                        "maximum_tvd_ci_upper"
                    ],
                    "minimum_chi_square_p_value": result[
                        "minimum_chi_square_p_value"
                    ],
                }
            )

    if failures:
        outcome = "FAIL"
    elif warnings:
        outcome = "CONDITIONAL_PASS"
    else:
        outcome = "PASS"

    return {
        "outcome": outcome,
        "failures": failures,
        "warnings": warnings,
        "rule": (
            "FAIL iff any audited comparison has at least one "
            "replicate whose upper bootstrap TVD uncertainty bound "
            f"exceeds {TVD_THRESHOLD}. CONDITIONAL_PASS iff all "
            "upper TVD uncertainty bounds remain within the threshold "
            "but at least one chi-square replicate rejects the supplied "
            f"reference at alpha={CHI_SQUARE_ALPHA}. PASS requires "
            "all replicates of all comparisons to satisfy the practical "
            "TVD criterion and all chi-square replicates to remain "
            "compatible with the supplied reference."
        ),
    }


# ============================================================
# Certificate
# ============================================================

def array_summary(array: np.ndarray) -> Dict[str, Any]:
    """Return compact provenance for an audited feature array."""

    array = np.asarray(array)

    return {
        "sample_count": int(array.shape[0]),
        "feature_count": int(array.shape[1]),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": array_sha256(array),
    }


def build_d2_certificate(
    *,
    comparisons: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    partitions: Mapping[str, np.ndarray],
    dataset_id: str,
    dataset_version: Optional[str],
    generation_procedure: Optional[str],
    generation_parameters: Optional[Mapping[str, Any]],
    generation_random_seed: Optional[int],
    feature_bits: int,
    reference_description: str,
    reference_model_description: str,
    pairs_per_comparison: int,
    audit_replicates: int,
    tvd_bootstrap_replicates: int,
    audit_seed: int,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Build the generic D2 certificate.

    No Gohr/Speck-specific provenance is inserted here. The caller
    supplies dataset-generation metadata.
    """

    provenance = build_provenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        generation_procedure=generation_procedure,
        generation_parameters=generation_parameters,
        random_seed=generation_random_seed,
        partitions={
            name: {
                "sample_count": int(np.asarray(array).shape[0]),
                "feature_count": int(np.asarray(array).shape[1]),
                "dtype": str(np.asarray(array).dtype),
                "shape": list(np.asarray(array).shape),
                "sha256": array_sha256(np.asarray(array)),
            }
            for name, array in partitions.items()
        },
        audit_configuration={
            "audit_id": AUDIT_ID,
            "reference_distribution": reference_description,
            "reference_model": reference_model_description,
            "feature_bits": int(feature_bits),
            "statistical_unit": "sampled pair of feature vectors",
            "pair_sampling": "uniform index sampling with replacement",
            "within_partition_self_pairs": "excluded",
            "cross_partition_self_pairs": "not applicable",
            "pairs_per_comparison": int(pairs_per_comparison),
            "audit_replicates": int(audit_replicates),
            "audit_sampling_seed": int(audit_seed),
            "chi_square_alpha": CHI_SQUARE_ALPHA,
            "chi_square_minimum_expected": CHI_SQUARE_MIN_EXPECTED,
            "tvd_threshold": TVD_THRESHOLD,
            "tvd_confidence_level": TVD_CONFIDENCE_LEVEL,
            "tvd_bootstrap_replicates": int(tvd_bootstrap_replicates),
        },
    )

    certificate = make_certificate(
        audit_id=AUDIT_ID,
        audit_name=AUDIT_NAME,
        claim=(
            "Sampled pairwise Hamming-distance distributions within "
            "and across the supplied dataset partitions are evaluated "
            "for compatibility with an explicitly supplied reference "
            "distribution using pre-specified Pearson chi-square and "
            "total-variation criteria."
        ),
        outcome=str(decision["outcome"]),
        findings={
            "comparisons": dict(comparisons),
            "failed_comparisons": list(decision["failures"]),
            "warning_comparisons": list(decision["warnings"]),
        },
        methodology={
            "reference_distribution": reference_description,
            "reference_model": reference_model_description,
            "feature_bits": int(feature_bits),
            "statistical_unit": "sampled pair of feature vectors",
            "sampling_design": (
                "uniform index sampling with replacement; self-pairs "
                "excluded for within-partition comparisons"
            ),
            "chi_square_alpha": CHI_SQUARE_ALPHA,
            "chi_square_minimum_expected": CHI_SQUARE_MIN_EXPECTED,
            "tvd_threshold": TVD_THRESHOLD,
            "tvd_confidence_level": TVD_CONFIDENCE_LEVEL,
            "tvd_bootstrap_replicates": int(tvd_bootstrap_replicates),
            "tvd_decision_criterion": (
                "The primary practical criterion is that the upper "
                "percentile-bootstrap uncertainty bound for empirical "
                "TVD remains at or below the pre-specified TVD threshold."
            ),
            "audit_replicates": int(audit_replicates),
            "pairs_per_comparison": int(pairs_per_comparison),
            "decision_rule": decision["rule"],
            "multiple_comparisons": (
                "The comparison battery is fixed in the experiment "
                "driver. Replicate-level evidence is retained rather "
                "than averaged away."
            ),
            "replicate_aggregation": (
                "Each comparison satisfies the practical criterion "
                "only when every replicate has an upper TVD uncertainty "
                "bound at or below the threshold."
            ),
            "inferential_scope": "pairwise Hamming-distance structure only",
        },
        provenance=provenance,
        limitations=[
            (
                "D2 tests pairwise Hamming-distance structure only. "
                "It does not establish full statistical independence."
            ),
            (
                "The supplied reference distribution is a model "
                "assumption for the supplied representation; it is "
                "not a universal null for arbitrary binary data."
            ),
            (
                "Pairwise distances are estimated from a finite Monte "
                "Carlo sample rather than exhaustively enumerated."
            ),
            (
                "Sampling with replacement allows dataset elements to "
                "participate in multiple sampled pairs."
            ),
            (
                "Audit replicates are independent pair-sampling "
                "procedures over the same dataset instance, not "
                "independent datasets."
            ),
            (
                "The percentile-bootstrap interval quantifies "
                "finite-sample uncertainty in empirical TVD. It is not "
                "by itself a formally calibrated equivalence test."
            ),
            (
                "A D2 PASS does not prove absence of near duplicates, "
                "metadata leakage, higher-order dependence, or other "
                "forms of dataset leakage."
            ),
        ],
        evidence_level="DATASET_INTEGRITY_PAIRWISE_GOODNESS_OF_FIT",
    )

    write_certificate(certificate, str(output_path))
    return certificate


# ============================================================
# Reporting
# ============================================================

def print_report(
    comparisons: Mapping[str, Mapping[str, Any]],
    certificate: Mapping[str, Any],
) -> None:
    """Print a human-readable D2 report."""

    methodology = certificate["methodology"]

    print("=" * 72)
    print("Dataset Integrity Audit")
    print("D2 — Pairwise Hamming-Distance Goodness-of-Fit Audit")
    print("=" * 72)
    print()
    print(
        f"Reference distribution : "
        f"{methodology['reference_distribution']}"
    )
    print(
        f"Feature bits           : "
        f"{methodology['feature_bits']}"
    )
    print(
        f"Chi-square alpha       : "
        f"{methodology['chi_square_alpha']}"
    )
    print(
        f"TVD threshold          : "
        f"{methodology['tvd_threshold']}"
    )
    print(
        f"Bootstrap level        : "
        f"{methodology['tvd_confidence_level']}"
    )
    print()

    for name, result in comparisons.items():
        print(f"[{name}]")
        print(
            f"  Replicates              : "
            f"{result['replicate_count']}"
        )
        print(
            f"  Pairs/replicate         : "
            f"{result['replicates'][0]['sampled_pairs']}"
        )
        print(
            f"  Chi-square passes       : "
            f"{result['chi_square_pass_count']}/"
            f"{result['replicate_count']}"
        )
        print(
            f"  TVD point-estimate pass : "
            f"{result['tvd_pass_count']}/"
            f"{result['replicate_count']}"
        )
        print(
            f"  Minimum chi-square p    : "
            f"{result['minimum_chi_square_p_value']:.6g}"
        )
        print(
            f"  Maximum chi-square p    : "
            f"{result['maximum_chi_square_p_value']:.6g}"
        )
        print(
            f"  Minimum TVD             : "
            f"{result['minimum_tvd']:.6f}"
        )
        print(
            f"  Maximum TVD             : "
            f"{result['maximum_tvd']:.6f}"
        )
        print(
            f"  Minimum upper TVD bound : "
            f"{result['minimum_tvd_ci_upper']:.6f}"
        )
        print(
            f"  Maximum upper TVD bound : "
            f"{result['maximum_tvd_ci_upper']:.6f}"
        )
        print()

    print(
        f"Decision                 : "
        f"{certificate['decision']['outcome']}"
    )


# ============================================================
# Generic execution API
# ============================================================

def run_d2(
    *,
    partitions: Mapping[str, np.ndarray],
    reference_pmf: np.ndarray,
    feature_bits: int,
    pairs_per_comparison: int = DEFAULT_PAIRS_PER_COMPARISON,
    audit_replicates: int = DEFAULT_AUDIT_REPLICATES,
    tvd_bootstrap_replicates: int = DEFAULT_TVD_BOOTSTRAP_REPLICATES,
    audit_seed: int = 0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute D2 on supplied partitions.

    Required partition names are:

        train
        validation
        test

    Returns
    -------
    comparisons, decision
    """

    required = ("train", "validation", "test")

    missing = [name for name in required if name not in partitions]

    if missing:
        raise ValueError(
            f"Missing required partitions: {missing}"
        )

    if feature_bits < 1:
        raise ValueError("feature_bits must be >= 1.")

    if pairs_per_comparison < 1:
        raise ValueError("pairs_per_comparison must be >= 1.")

    if audit_replicates < 1:
        raise ValueError("audit_replicates must be >= 1.")

    if tvd_bootstrap_replicates < 100:
        raise ValueError(
            "tvd_bootstrap_replicates must be >= 100."
        )

    validated = {
        name: _validate_binary_features(
            np.asarray(partitions[name]),
            n_bits=feature_bits,
            name=name,
        )
        for name in required
    }

    reference_pmf = _validate_reference_pmf(
        reference_pmf,
        n_bits=feature_bits,
    )

    seed_sequence = np.random.SeedSequence(audit_seed)

    comparison_names = (
        "train_within",
        "validation_within",
        "test_within",
        "train_validation",
        "train_test",
        "validation_test",
    )

    child_sequences = seed_sequence.spawn(
        len(comparison_names) * audit_replicates
    )

    within = {
        "train_within": validated["train"],
        "validation_within": validated["validation"],
        "test_within": validated["test"],
    }

    cross = {
        "train_validation": (
            validated["train"],
            validated["validation"],
        ),
        "train_test": (
            validated["train"],
            validated["test"],
        ),
        "validation_test": (
            validated["validation"],
            validated["test"],
        ),
    }

    comparisons: Dict[str, Dict[str, Any]] = {}
    child_index = 0

    for name in comparison_names:
        replicate_results = []

        for _replicate in range(audit_replicates):
            rng = np.random.default_rng(
                child_sequences[child_index]
            )
            child_index += 1

            if name in within:
                distances = sample_within_partition_distances(
                    within[name],
                    num_pairs=pairs_per_comparison,
                    rng=rng,
                    n_bits=feature_bits,
                )
            else:
                features_a, features_b = cross[name]

                distances = sample_cross_partition_distances(
                    features_a,
                    features_b,
                    num_pairs=pairs_per_comparison,
                    rng=rng,
                    n_bits=feature_bits,
                )

            replicate_results.append(
                summarize_distance_sample(
                    distances,
                    reference_pmf=reference_pmf,
                    bootstrap_rng=rng,
                    n_bits=feature_bits,
                    tvd_bootstrap_replicates=tvd_bootstrap_replicates,
                )
            )

        comparisons[name] = aggregate_replicate_results(
            replicate_results
        )

    decision = evaluate_d2(comparisons)

    return comparisons, decision


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """
    Generic CLI.

    The generic module intentionally does not generate a dataset.
    Dataset-specific experiment drivers should construct arrays
    through an adapter and call run_d2().
    """

    parser = argparse.ArgumentParser(
        description=(
            "D2 generic pairwise Hamming-distance audit. "
            "Use a dataset-specific experiment driver to supply "
            "the actual partitions."
        )
    )

    parser.add_argument(
        "--help-architecture",
        action="store_true",
        help=(
            "Print the intended separation between generic D2 "
            "logic and dataset-specific adapters."
        ),
    )

    args = parser.parse_args()

    if args.help_architecture:
        print(
            "Generic D2 owns pair sampling, statistics, decisions, "
            "and certificate construction. Dataset-specific experiment "
            "drivers own dataset generation and adapter configuration."
        )
        return

    parser.error(
        "d2_near_duplicate_gof.py is a generic audit module. "
        "Run the dataset-specific experiment driver instead."
    )


if __name__ == "__main__":
    main()
