"""
D3 — Marginal and Low-Order Distributional Consistency Audit.

Generic Dataset Integrity component.

Scientific scope
----------------
D3 evaluates whether independently generated dataset partitions are
consistent with an explicitly specified intended distribution and with
each other at marginal and low-order (second-order) levels.

D3 covers:
    * class distribution;
    * first-order binary feature marginals;
    * global mean/variance and marginal entropy as descriptive diagnostics;
    * second-order feature-pair joint distributions across partitions;
    * partition-to-partition distribution consistency;
    * uncertainty, hypothesis tests, effect sizes, and multiple-comparison
      correction for the confirmatory low-order checks.

D3 deliberately does NOT claim equality of the full 64-dimensional joint
law. D1 owns exact duplicate/overlap checks and D2 owns sample-order /
pairwise Hamming dependence. A second-order feature check therefore adds
coverage without duplicating D1/D2.

The generic module contains no Gohr-specific generation logic.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, chi2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dataset(dataset: Any) -> np.ndarray:
    x = np.asarray(dataset)
    if x.ndim < 1:
        raise ValueError("Dataset must have at least one dimension.")
    if x.shape[0] == 0:
        raise ValueError("Dataset must contain at least one sample.")
    return x


def _validate_labels(labels: Any, n_samples: int) -> np.ndarray:
    y = np.asarray(labels).reshape(-1)
    if len(y) != n_samples:
        raise ValueError("Number of labels must equal number of samples.")
    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError("D3 requires binary labels encoded as 0 and 1.")
    return y.astype(np.uint8, copy=False)


def _validate_binary_2d(dataset: Any) -> np.ndarray:
    x = _validate_dataset(dataset)
    if x.ndim != 2:
        raise ValueError("Binary low-order statistics require a 2-D array.")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("Binary statistics require entries exactly equal to 0 or 1.")
    return x.astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")

    # Avoid a dependency on a separate normal-quantile implementation.
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + confidence / 2.0))
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * sqrt((phat * (1.0 - phat) / n) + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact binomial interval used for the class-ratio diagnostic."""
    from scipy.stats import beta
    alpha = 1.0 - confidence
    if successes == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2.0, successes, n - successes + 1))
    if successes == n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    return lo, hi


# ---------------------------------------------------------------------------
# First-order diagnostics
# ---------------------------------------------------------------------------

def class_distribution(labels: Any, *, confidence: float = 0.95) -> Dict[str, Any]:
    y = _validate_labels(labels, len(np.asarray(labels).reshape(-1)))
    n = len(y)
    positive = int(np.sum(y == 1))
    negative = n - positive
    lo, hi = _clopper_pearson(positive, n, confidence)
    return {
        "samples": n,
        "positive_samples": positive,
        "negative_samples": negative,
        "positive_ratio": positive / n,
        "negative_ratio": negative / n,
        "positive_ratio_ci": {"confidence_level": confidence, "lower": lo, "upper": hi},
    }


def class_balance_test(
    observed: Mapping[str, Any],
    expected_positive_ratio: float,
    *,
    practical_tolerance: float,
    alpha: float,
) -> Dict[str, Any]:
    n = int(observed["samples"])
    k = int(observed["positive_samples"])
    p = float(observed["positive_ratio"])
    expected = float(expected_positive_ratio)
    exact = binomtest(k, n, p=expected, alternative="two-sided")
    difference = abs(p - expected)
    return {
        "expected_positive_ratio": expected,
        "observed_positive_ratio": p,
        "absolute_difference": difference,
        "practical_tolerance": float(practical_tolerance),
        "practically_consistent": bool(difference <= practical_tolerance),
        "p_value": float(exact.pvalue),
        "alpha": float(alpha),
        "statistically_inconsistent": bool(exact.pvalue < alpha),
    }


def binary_marginal_distribution(dataset: Any) -> Dict[str, Any]:
    x = _validate_binary_2d(dataset)
    probs = np.mean(x, axis=0, dtype=np.float64)
    return {
        "feature_count": int(x.shape[1]),
        "per_feature_one_probability": probs.tolist(),
        "mean_one_probability": float(np.mean(probs)),
        "minimum_one_probability": float(np.min(probs)),
        "maximum_one_probability": float(np.max(probs)),
    }


def compare_binary_marginals(
    observed: Mapping[str, Any],
    expected_probability: float,
    *,
    practical_tolerance: float,
) -> Dict[str, Any]:
    probs = np.asarray(observed["per_feature_one_probability"], dtype=np.float64)
    diff = np.abs(probs - float(expected_probability))
    return {
        "expected_probability": float(expected_probability),
        "mean_absolute_difference": float(np.mean(diff)),
        "rms_difference": float(np.sqrt(np.mean(diff ** 2))),
        "maximum_absolute_difference": float(np.max(diff)),
        "q95_absolute_difference": float(np.quantile(diff, 0.95)),
        "q99_absolute_difference": float(np.quantile(diff, 0.99)),
        "practical_tolerance": float(practical_tolerance),
        "practically_consistent": bool(np.max(diff) <= practical_tolerance),
        "per_feature_absolute_difference": diff.tolist(),
    }


def binary_entropy(dataset: Any) -> Dict[str, Any]:
    x = _validate_binary_2d(dataset)
    p = np.mean(x, axis=0, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)
    h = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return {
        "mean_entropy": float(np.mean(h)),
        "minimum_entropy": float(np.min(h)),
        "maximum_entropy": float(np.max(h)),
        "per_feature_entropy": h.tolist(),
    }


def feature_statistics(dataset: Any) -> Dict[str, float]:
    x = _validate_dataset(dataset)
    return {
        "feature_mean": float(np.mean(x, dtype=np.float64)),
        "feature_variance": float(np.var(x, dtype=np.float64)),
    }


# ---------------------------------------------------------------------------
# Second-order feature-pair structure
# ---------------------------------------------------------------------------

def _pair_counts(x: np.ndarray) -> np.ndarray:
    """Return a feature-by-feature matrix of n11 counts.

    Computed in row blocks to keep peak memory bounded. Only 64x64 state is
    retained; this avoids materialising an N x 64 floating-point matrix.
    """
    x = _validate_binary_2d(x)
    d = x.shape[1]
    counts11 = np.zeros((d, d), dtype=np.float64)
    block = 100_000
    for start in range(0, len(x), block):
        xb = x[start:start + block].astype(np.float64, copy=False)
        counts11 += xb.T @ xb
    return counts11


def pairwise_joint_tables(dataset: Any) -> Dict[str, Any]:
    """Compute all unordered 2-feature joint tables.

    Each table is ordered as [(0,0),(0,1),(1,0),(1,1)].
    """
    x = _validate_binary_2d(dataset)
    n, d = x.shape
    ones = np.sum(x, axis=0, dtype=np.int64)
    n11 = _pair_counts(x)

    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            a11 = int(n11[i, j])
            a10 = int(ones[i] - a11)
            a01 = int(ones[j] - a11)
            a00 = int(n - a11 - a10 - a01)
            pairs.append((i, j, np.array([a00, a01, a10, a11], dtype=np.int64)))
    return {"sample_count": int(n), "feature_count": int(d), "pairs": pairs}


def _tvd(observed: np.ndarray, reference: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(observed - reference)))


def _phi_from_table(table: np.ndarray) -> float:
    a, b, c, d = [float(v) for v in table]
    denom = sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0.0:
        return 0.0
    return float((a * d - b * c) / denom)


def pairwise_joint_summary(dataset: Any) -> Dict[str, Any]:
    """Summarise second-order structure without assuming pairwise independence.

    This is a descriptive representation used for partition consistency.
    It deliberately does not declare the Gohr feature-pair distribution to be
    uniform, because doing so could incorrectly classify legitimate cipher
    structure as a dataset defect.
    """
    data = pairwise_joint_tables(dataset)
    rows = []
    for i, j, table in data["pairs"]:
        probs = table.astype(np.float64) / data["sample_count"]
        rows.append({
            "feature_i": i,
            "feature_j": j,
            "joint_probabilities": probs.tolist(),
            "tvd_from_uniform_4cell": _tvd(probs, np.full(4, 0.25)),
            "phi": _phi_from_table(table),
        })
    tvd = np.asarray([r["tvd_from_uniform_4cell"] for r in rows], dtype=np.float64)
    phi = np.asarray([abs(r["phi"]) for r in rows], dtype=np.float64)
    return {
        "feature_count": data["feature_count"],
        "pair_count": len(rows),
        "pair_summaries": rows,
        "aggregate": {
            "mean_tvd_from_uniform_4cell": float(np.mean(tvd)),
            "maximum_tvd_from_uniform_4cell": float(np.max(tvd)),
            "q95_tvd_from_uniform_4cell": float(np.quantile(tvd, 0.95)),
            "mean_absolute_phi": float(np.mean(phi)),
            "maximum_absolute_phi": float(np.max(phi)),
            "q95_absolute_phi": float(np.quantile(phi, 0.95)),
        },
    }


def compare_pairwise_joint_distributions(
    dataset_a: Any,
    dataset_b: Any,
    *,
    practical_tvd_tolerance: float,
    familywise_alpha: float,
) -> Dict[str, Any]:
    """Compare every feature-pair joint distribution between two partitions.

    For each feature pair, a 2x4 chi-square homogeneity test is performed.
    The familywise alpha is Bonferroni-adjusted across all pairs. TVD is the
    practical effect size. No assumption of pairwise feature independence is
    made.
    """
    a = _validate_binary_2d(dataset_a)
    b = _validate_binary_2d(dataset_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Partitions must have the same feature dimension.")

    ta = pairwise_joint_tables(a)
    tb = pairwise_joint_tables(b)
    n_a, n_b = ta["sample_count"], tb["sample_count"]
    pair_count = len(ta["pairs"])
    alpha_pair = float(familywise_alpha / max(1, pair_count))

    records = []
    p_values = []
    tvds = []
    phis = []
    for (i, j, ca), (_, _, cb) in zip(ta["pairs"], tb["pairs"]):
        observed = np.vstack([ca, cb]).astype(np.float64)
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        total = float(observed.sum())
        expected = row_totals @ col_totals / total
        valid = expected > 0
        stat = float(np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid]))
        df = int((2 - 1) * (4 - 1))
        p = float(chi2.sf(stat, df))

        pa = ca.astype(np.float64) / n_a
        pb = cb.astype(np.float64) / n_b
        tvd = _tvd(pa, pb)
        phi_a = _phi_from_table(ca)
        phi_b = _phi_from_table(cb)
        delta_phi = abs(phi_a - phi_b)
        p_values.append(p)
        tvds.append(tvd)
        phis.append(delta_phi)
        records.append({
            "feature_i": i,
            "feature_j": j,
            "tvd": tvd,
            "delta_phi": delta_phi,
            "chi_square": stat,
            "degrees_of_freedom": df,
            "p_value": p,
            "alpha_bonferroni": alpha_pair,
            "statistically_inconsistent": bool(p < alpha_pair),
            "practically_inconsistent": bool(tvd > practical_tvd_tolerance),
            "joint_a": pa.tolist(),
            "joint_b": pb.tolist(),
        })

    pvals = np.asarray(p_values, dtype=np.float64)
    tvd_arr = np.asarray(tvds, dtype=np.float64)
    phi_arr = np.asarray(phis, dtype=np.float64)
    statistical_count = int(np.sum(pvals < alpha_pair))
    practical_count = int(np.sum(tvd_arr > practical_tvd_tolerance))
    return {
        "sample_sizes": {"partition_a": n_a, "partition_b": n_b},
        "pair_count": pair_count,
        "multiple_comparison": {
            "method": "Bonferroni",
            "familywise_alpha": float(familywise_alpha),
            "per_pair_alpha": alpha_pair,
        },
        "practical_effect": {
            "measure": "total_variation_distance",
            "tolerance": float(practical_tvd_tolerance),
            "mean_tvd": float(np.mean(tvd_arr)),
            "maximum_tvd": float(np.max(tvd_arr)),
            "q95_tvd": float(np.quantile(tvd_arr, 0.95)),
            "practically_inconsistent_pair_count": practical_count,
        },
        "statistical_effect": {
            "test": "2x4 chi-square homogeneity",
            "statistically_inconsistent_pair_count": statistical_count,
            "minimum_p_value": float(np.min(pvals)),
        },
        "joint_structure_effect": {
            "measure": "absolute change in phi",
            "mean_absolute_delta_phi": float(np.mean(phi_arr)),
            "maximum_absolute_delta_phi": float(np.max(phi_arr)),
        },
        "pair_summaries": records,
        "decision": {
            "family_statistical_pass": statistical_count == 0,
            "family_practical_pass": practical_count == 0,
            "family_pass": statistical_count == 0 and practical_count == 0,
        },
    }


# ---------------------------------------------------------------------------
# Partition summaries and intended distribution
# ---------------------------------------------------------------------------

def summarize_partition(dataset: Any, labels: Any, *, confidence: float = 0.95) -> Dict[str, Any]:
    x = _validate_dataset(dataset)
    y = _validate_labels(labels, len(x))
    summary = {
        "samples": int(len(x)),
        "feature_shape": list(x.shape[1:]),
        "dtype": str(x.dtype),
        "class_distribution": class_distribution(y, confidence=confidence),
        "feature_statistics": feature_statistics(x),
    }
    if x.ndim == 2 and np.all((x == 0) | (x == 1)):
        summary["binary_marginal_distribution"] = binary_marginal_distribution(x)
        summary["binary_entropy"] = binary_entropy(x)
        # Second-order summary is intentionally descriptive; partition-level
        # inferential comparison is performed below.
        summary["second_order_structure"] = pairwise_joint_summary(x)
    return summary


def compare_partitions(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    familywise_alpha: float,
    pairwise_tvd_tolerance: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "positive_ratio_difference": abs(a["class_distribution"]["positive_ratio"] - b["class_distribution"]["positive_ratio"]),
        "feature_mean_difference": abs(a["feature_statistics"]["feature_mean"] - b["feature_statistics"]["feature_mean"]),
        "feature_variance_difference": abs(a["feature_statistics"]["feature_variance"] - b["feature_statistics"]["feature_variance"]),
    }
    if "binary_marginal_distribution" in a and "binary_marginal_distribution" in b:
        pa = np.asarray(a["binary_marginal_distribution"]["per_feature_one_probability"], dtype=np.float64)
        pb = np.asarray(b["binary_marginal_distribution"]["per_feature_one_probability"], dtype=np.float64)
        diff = np.abs(pa - pb)
        result["binary_marginal_difference"] = {
            "mean_absolute_difference": float(np.mean(diff)),
            "rms_difference": float(np.sqrt(np.mean(diff ** 2))),
            "maximum_absolute_difference": float(np.max(diff)),
            "q95_absolute_difference": float(np.quantile(diff, 0.95)),
            "q99_absolute_difference": float(np.quantile(diff, 0.99)),
        }
    return result


def compare_with_intended_distribution(
    partition: Mapping[str, Any],
    *,
    expected_class_ratio: float,
    expected_bit_probability: float,
    class_ratio_tolerance: float,
    bit_probability_tolerance: float,
    familywise_alpha: float,
) -> Dict[str, Any]:
    # There are two confirmatory first-order families here: class balance and
    # 64 feature marginals. Bonferroni controls the declared familywise alpha.
    n_features = int(partition["binary_marginal_distribution"]["feature_count"])
    alpha_class = familywise_alpha / (n_features + 1)
    class_result = class_balance_test(
        partition["class_distribution"], expected_class_ratio,
        practical_tolerance=class_ratio_tolerance,
        alpha=alpha_class,
    )

    probs = np.asarray(partition["binary_marginal_distribution"]["per_feature_one_probability"], dtype=np.float64)
    n = int(partition["samples"])
    bit_records = []
    bit_p = []
    for idx, p in enumerate(probs):
        k = int(round(float(p) * n))
        exact = binomtest(k, n, p=float(expected_bit_probability), alternative="two-sided")
        lo, hi = wilson_interval(k, n, confidence=0.95)
        bit_p.append(float(exact.pvalue))
        bit_records.append({
            "feature": idx,
            "observed_probability": float(p),
            "absolute_difference": abs(float(p) - expected_bit_probability),
            "p_value": float(exact.pvalue),
            "alpha_bonferroni": float(alpha_class),
            "statistically_inconsistent": bool(exact.pvalue < alpha_class),
            "ci95": {"lower": lo, "upper": hi},
        })
    bit_diff = np.abs(probs - expected_bit_probability)
    return {
        "class_distribution": class_result,
        "binary_marginals": {
            "expected_bit_probability": float(expected_bit_probability),
            "mean_absolute_difference": float(np.mean(bit_diff)),
            "rms_difference": float(np.sqrt(np.mean(bit_diff ** 2))),
            "maximum_absolute_difference": float(np.max(bit_diff)),
            "practical_tolerance": float(bit_probability_tolerance),
            "practically_consistent": bool(np.max(bit_diff) <= bit_probability_tolerance),
            "statistically_inconsistent_feature_count": int(np.sum(np.asarray(bit_p) < alpha_class)),
            "per_feature": bit_records,
        },
        "multiple_comparison": {
            "method": "Bonferroni",
            "familywise_alpha": float(familywise_alpha),
            "tests_in_first_order_family": n_features + 1,
            "per_test_alpha": float(alpha_class),
        },
    }


def audit_distribution(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    test: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    *,
    intended_distribution: Mapping[str, Any],
    familywise_alpha: float = 0.01,
) -> Dict[str, Any]:
    partitions = {
        "train": summarize_partition(train, train_labels),
        "validation": summarize_partition(validation, validation_labels),
    }
    if test is not None:
        if test_labels is None:
            raise ValueError("test_labels must be supplied when test is supplied.")
        partitions["test"] = summarize_partition(test, test_labels)

    pair_comparisons: Dict[str, Any] = {}
    names = list(partitions)
    raw_arrays = {"train": train, "validation": validation}
    raw_arrays["test"] = test
    for a, b in combinations(names, 2):
        pair_comparisons[f"{a}_{b}"] = compare_partitions(
            partitions[a], partitions[b],
            familywise_alpha=familywise_alpha,
            pairwise_tvd_tolerance=float(intended_distribution["pairwise_tvd_tolerance"]),
        )
        pair_comparisons[f"{a}_{b}"]["second_order_feature_pairs"] = compare_pairwise_joint_distributions(
            raw_arrays[a], raw_arrays[b],
            practical_tvd_tolerance=float(intended_distribution["pairwise_tvd_tolerance"]),
            familywise_alpha=familywise_alpha,
        )

    intended = {
        name: compare_with_intended_distribution(
            partition,
            expected_class_ratio=float(intended_distribution["expected_class_ratio"]),
            expected_bit_probability=float(intended_distribution["expected_bit_probability"]),
            class_ratio_tolerance=float(intended_distribution["class_ratio_tolerance"]),
            bit_probability_tolerance=float(intended_distribution["bit_probability_tolerance"]),
            familywise_alpha=familywise_alpha,
        )
        for name, partition in partitions.items()
    }

    return {
        "audit_id": "D3",
        "audit_name": "Marginal and Low-Order Distributional Consistency Audit",
        "scientific_scope": (
            "Marginal and low-order (second-order feature-pair) distributional "
            "consistency; not equality of the complete joint distribution."
        ),
        "partitions": partitions,
        "intended_distribution": intended,
        "partition_comparisons": pair_comparisons,
        "scope_exclusions": {
            "exact_duplicates": "D1",
            "exact_cross_partition_overlap": "D1",
            "sample_order_dependence": "D2",
            "pairwise_sample_hamming_structure": "D2",
        },
    }


def evaluate_decision(results: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply D3-specific statistical/practical decision semantics.

    The decision is deliberately kept separate from the report schema.  D3
    has its own first-order and second-order findings, so the returned
    decision object exposes D3-native categories rather than assuming a
    generic ``failed_checks`` field used by other audits.

    PASS
        No statistically significant and no practically meaningful
        inconsistency was identified within the declared D3 scope.

    CONDITIONAL_PASS
        At least one statistical sensitivity was detected, but none of the
        detected effects were practically meaningful.

    INCONCLUSIVE
        At least one practically meaningful deviation was observed without
        statistical support sufficient to classify it as a confirmed
        inconsistency.

    FAIL
        At least one deviation was both statistically supported and
        practically meaningful.

    Limits of this legacy rule (retained unchanged so the historical Gohr
    certificate stays reproducible; see test_d3_method_validity.py):

    * PASS requires no evidence of precision. A design too small to resolve
      the tolerance returns PASS whenever the point estimate happens to lie
      inside it, so PASS here is "no deviation detected", not equivalence.
    * familywise_alpha is applied per partition (first order) and per
      partition comparison (second order). Across three partitions and three
      comparisons the union bound on false statistical findings is 6x the
      declared value, not the declared value.
    * The pair test uses a fixed df = 3 and the asymptotic chi-square for
      every table, which is conservative for tables with an empty column and
      can be anti-conservative for sparse, unbalanced tables.
    * Partition-level first-order differences are descriptive and not read.

    New confirmatory work uses evaluate_confirmatory_decision.
    """
    statistical_only = []
    practical_only = []
    confirmed = []

    # First-order comparisons against the explicitly declared intended
    # distribution.
    for name, finding in results["intended_distribution"].items():
        c = finding["class_distribution"]
        b = finding["binary_marginals"]

        c_stat = bool(c["statistically_inconsistent"])
        c_prac = not bool(c["practically_consistent"])
        if c_stat and c_prac:
            confirmed.append(f"{name}:class_distribution")
        elif c_stat:
            statistical_only.append(f"{name}:class_distribution")
        elif c_prac:
            practical_only.append(f"{name}:class_distribution")

        b_stat = int(b["statistically_inconsistent_feature_count"]) > 0
        b_prac = not bool(b["practically_consistent"])
        if b_stat and b_prac:
            confirmed.append(f"{name}:binary_marginals")
        elif b_stat:
            statistical_only.append(f"{name}:binary_marginals")
        elif b_prac:
            practical_only.append(f"{name}:binary_marginals")

    # Second-order partition-consistency comparisons.
    for name, comparison in results["partition_comparisons"].items():
        so = comparison.get("second_order_feature_pairs")
        if so is None:
            continue

        stat = (
            int(
                so["statistical_effect"][
                    "statistically_inconsistent_pair_count"
                ]
            )
            > 0
        )
        prac = (
            int(
                so["practical_effect"][
                    "practically_inconsistent_pair_count"
                ]
            )
            > 0
        )

        if stat and prac:
            confirmed.append(f"{name}:second_order_feature_pairs")
        elif stat:
            statistical_only.append(f"{name}:second_order_feature_pairs")
        elif prac:
            practical_only.append(f"{name}:second_order_feature_pairs")

    if confirmed:
        outcome = "FAIL"
    elif practical_only:
        outcome = "INCONCLUSIVE"
    elif statistical_only:
        outcome = "CONDITIONAL_PASS"
    else:
        outcome = "PASS"

    return {
        "outcome": outcome,
        "confirmed_statistical_and_practical_failures": confirmed,
        "statistical_only_sensitivities": statistical_only,
        "practical_only_deviations": practical_only,
        "counts": {
            "confirmed_failures": len(confirmed),
            "statistical_only": len(statistical_only),
            "practical_only": len(practical_only),
        },
        "interpretation": {
            "PASS": (
                "No statistically or practically meaningful inconsistency "
                "was identified within the declared D3 scope."
            ),
            "CONDITIONAL_PASS": (
                "Statistical sensitivity was observed, but the corresponding "
                "deviations were not practically meaningful."
            ),
            "INCONCLUSIVE": (
                "A practically meaningful deviation was observed without "
                "adequate statistical support; additional evidence is required."
            ),
            "FAIL": (
                "A declared D3 deviation was both statistically supported "
                "and practically meaningful."
            ),
        }[outcome],
    }


def print_report(
    results: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    """Print the D3-specific human-readable report.

    This intentionally does not reuse a generic ``failed_checks`` field.
    D3 reports its own marginal, descriptive, and second-order metrics.
    """
    print("=" * 78)
    print("Dataset Integrity — D3 Marginal and Low-Order Distributional Consistency")
    print("=" * 78)

    for name, finding in results["intended_distribution"].items():
        c = finding["class_distribution"]
        b = finding["binary_marginals"]
        print(
            f"{name:12s} class | obs={c['observed_positive_ratio']:.9f} "
            f"diff={c['absolute_difference']:.6g} "
            f"practical={c['practically_consistent']} "
            f"stat_inconsistent={c['statistically_inconsistent']} "
            f"p={c['p_value']:.3g}"
        )
        # The intended-distribution result is a hypothesis-test record and
        # intentionally does not duplicate the partition-level CI.  Pull the
        # confidence interval from the underlying partition summary.
        ci = results["partitions"][name]["class_distribution"]["positive_ratio_ci"]
        print(
            f"{'':12s} class CI | "
            f"[{ci['lower']:.6f}, {ci['upper']:.6f}] "
            f"({ci['confidence_level']:.2f} level)"
        )
        print(
            f"{'':12s} bits  | max_diff={b['maximum_absolute_difference']:.6g} "
            f"MAD={b['mean_absolute_difference']:.6g} "
            f"practical={b['practically_consistent']} "
            f"stat_sig_features={b['statistically_inconsistent_feature_count']}"
        )

        part = results["partitions"][name]
        stats = part["feature_statistics"]
        entropy = part["binary_entropy"]
        print(
            f"{'':12s} moments | mean={stats['feature_mean']:.9f} "
            f"variance={stats['feature_variance']:.9f}"
        )
        print(
            f"{'':12s} entropy | mean={entropy['mean_entropy']:.9f} "
            f"min={entropy['minimum_entropy']:.9f} "
            f"max={entropy['maximum_entropy']:.9f}"
        )

    print("-" * 78)
    print("Second-order feature-pair partition consistency:")
    for name, comparison in results["partition_comparisons"].items():
        so = comparison.get("second_order_feature_pairs")
        if so is None:
            continue

        pe = so["practical_effect"]
        se = so["statistical_effect"]
        mc = so["multiple_comparison"]

        print(
            f"{name:24s} pairs={so['pair_count']} "
            f"max_TVD={pe['maximum_tvd']:.6g} "
            f"q95_TVD={pe['q95_tvd']:.6g} "
            f"practical_inconsistent={pe['practically_inconsistent_pair_count']} "
            f"statistically_inconsistent={se['statistically_inconsistent_pair_count']}"
        )
        print(
            f"{'':24s} min_p={se['minimum_p_value']:.6g} "
            f"Bonferroni_alpha={mc['per_pair_alpha']:.6g} "
            f"max_|delta_phi|={so['joint_structure_effect']['maximum_absolute_delta_phi']:.6g}"
        )

    print("=" * 78)
    print(f"Outcome: {decision['outcome']}")
    print(
        "Confirmed statistical+practical failures: "
        f"{decision['counts']['confirmed_failures']}"
    )
    print(
        "Statistical-only sensitivities: "
        f"{decision['counts']['statistical_only']}"
    )
    print(
        "Practical-only deviations: "
        f"{decision['counts']['practical_only']}"
    )
    print(f"Interpretation: {decision['interpretation']}")

def save_json(results: Mapping[str, Any], filename: str) -> None:
    import json
    from pathlib import Path
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ===========================================================================
# Confirmatory methodology (D3 method version 4)
# ===========================================================================
#
# Everything ABOVE this banner is the legacy D3 path, byte-for-byte the
# implementation at commit c603522 apart from one docstring. The historical
# Gohr driver reaches only that path, and tests/test_d3_legacy_golden.py pins
# its complete output to the c603522 engine.
#
# Everything BELOW is the confirmatory method. It shares no decision logic and
# no statistical helper with the legacy path (it reuses only the unchanged
# legacy label validator).
#
# GUARANTEES (finite-sample, exact; rows i.i.d. within each sample, samples
# independent). Every quantity that can change a decision comes from an exact
# method, so neither guarantee depends on an asymptotic approximation:
#
#   (G1) Bounds. Every reported effect bound comes from Clopper-Pearson
#        intervals, Bonferroni-corrected over all cells of all samples, plus
#        interval arithmetic. With probability >= 1 - alpha_total all bounds
#        of all confirmatory hypotheses hold simultaneously.
#   (G2) Detection. Every "significant" flag comes from an exact test (exact
#        binomial; Fisher's exact conditional test per cell, Bonferroni over
#        live cells). P(any false detection across all confirmatory
#        hypotheses) <= alpha_total under arbitrary dependence.
#
#   alpha_total is the sum of the confirmatory family alphas. G1 and G2 each
#   run at alpha_total, so the global alpha is JOINT over the two guarantees:
#   P(any G1 or G2 error) <= GUARANTEE_COUNT * alpha_total <= global alpha
#   (union bound). joint_alpha_problem() is the single rule every verifier uses.
#   Tolerance statements in an outcome (FAIL: effect > tol; PASS/CONDITIONAL_PASS:
#   effect <= tol) rest on G1 only. "A difference was detected" rests on G2.
#   A family may be declared exploratory: it is computed and reported at
#   exploratory_alpha within the family, outside alpha_total, and never gates.
#
# The asymptotic chi-square and the conditional Monte Carlo test are
# DIAGNOSTICS: recorded, never gating. Their size at production alpha is
# supported only by "no detectable inflation" simulations, which is not a
# guarantee, so no decision depends on them.
#
# DECISION GATES. evaluate_confirmatory_decision() accepts only sealed objects
# produced by this module: an AuditResult (class_stratified_audit), a
# CalibrationBinding (verify_calibration_evidence) and a PreregistrationBinding
# (verify_preregistration). Each is re-verified at decision time; family
# outcomes are re-derived from the sealed hypothesis records, never read from
# a caller's dictionary. These checks detect accidental substitution,
# editing and staleness; they cannot stop a determined in-process forger, and
# do not replace committing evidence to version control.

import datetime as _datetime
import hashlib as _hashlib
import json as _json
import math as _math
import subprocess as _subprocess
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path

from scipy.stats import beta as _beta_dist, hypergeom as _hypergeom

D3_METHOD_VERSION = "d3-confirmatory-5"
DECISION_RULE_VERSION = (
    "d3-decision-4: FAIL = lower bound > tol; PASS = upper bound <= tol and no exact detection; "
    "CONDITIONAL_PASS = upper bound <= tol with exact detection; else INCONCLUSIVE")
BOUND_CONSTRUCTION = (
    "simultaneous exact Clopper-Pearson intervals for every cell probability of every "
    "sample (Bonferroni over all of them), followed by worst-case interval arithmetic "
    "for |p_a - p_b| and TVD = 0.5 * sum_c |p_ac - p_bc|")
DETECTION_TEST = (
    "exact two-sided binomial (scipy minlike) for one-sample hypotheses; for two-sample "
    "hypotheses, Fisher's exact conditional test per cell (cell vs rest; doubled one-sided "
    "hypergeometric tail), Bonferroni over live cells")
CALIBRATION_SCHEMA_VERSION = "d3-method-validity-3"
AUDIT_SCHEMA_VERSION = "d3-audit-result-2"
PREREGISTRATION_SCHEMA_VERSION = "d3-preregistration-3"

#: The exact statement a human approver must sign in the preregistration.
APPROVAL_STATEMENT = (
    "I approve these D3 tolerances as operational FAIL thresholds, not validated equivalence "
    "margins, and the joint global alpha, the family alpha allocation and the family roles, for "
    "this audit. I fixed them before examining any D3 statistic of a confirmatory family on the "
    "audited data, and data_examined_before_approval records what was examined before approval.")

#: Marks an attestation in a draft that still awaits the approver's answer. A
#: record containing it anywhere is refused, so a draft cannot be approved with
#: an open attestation.
PENDING_MARKER = "PENDING APPROVER ANSWER"

#: Guarantees that share the global alpha: G1 (bounds) and G2 (detections)
#: each run at the summed confirmatory allocation.
GUARANTEE_COUNT = 2


def joint_alpha_problem(global_alpha: float, allocation: Mapping[str, float]) -> Optional[str]:
    """None if the allocation fits the JOINT global alpha, else the reason.

    The single joint-alpha rule used by the preregistration validator, the
    calibration verifier, the decision gate and the target runners:
    GUARANTEE_COUNT * sum(allocation) <= global_alpha.
    """
    total = float(sum(allocation.values()))
    if GUARANTEE_COUNT * total > float(global_alpha) * (1 + 1e-12):
        return (f"joint alpha violated: {GUARANTEE_COUNT} x sum(allocation) = {GUARANTEE_COUNT * total!r} "
                f"exceeds the global alpha {float(global_alpha)!r}")
    return None

#: Diagnostic Monte Carlo budget (non-gating).
MINIMUM_EXPECTED_COUNT = 5.0
MC_MIN_DRAWS = 9_999
MC_MAX_DRAWS = 2_000_000
MC_RESOLUTION_FACTOR = 20.0
MC_CHUNK = 250_000

#: Outward rounding applied to every Clopper-Pearson limit.
_OUTWARD = 1e-14

CATEGORY_OUTCOME = {
    "CONFIRMED_DEVIATION": "FAIL",
    "STATISTICAL_ONLY": "CONDITIONAL_PASS",
    "EQUIVALENT_WITHIN_TOLERANCE": "PASS",
    "SIGNIFICANT_PRACTICALITY_UNRESOLVED": "INCONCLUSIVE",
    "PRACTICAL_UNSUPPORTED": "INCONCLUSIVE",
    "INSUFFICIENT_PRECISION": "INCONCLUSIVE",
    "SIGNIFICANT_NO_TOLERANCE": "INCONCLUSIVE",
    "NO_TOLERANCE_DECLARED": "INCONCLUSIVE",
}
_OUTCOME_ORDER = ("FAIL", "INCONCLUSIVE", "CONDITIONAL_PASS", "PASS")


class AuditIntegrityError(ValueError):
    """A sealed audit result is missing, altered, inconsistent or substituted."""


class CalibrationRejected(ValueError):
    """Calibration evidence is missing, malformed, incomplete, stale or mismatched."""


class PreregistrationRejected(ValueError):
    """A preregistration is missing, malformed, unapproved, unanchored or modified."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _is_real(v: Any) -> bool:
    return not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, float, np.integer, np.floating))


def _check_alpha(alpha: Any, name: str = "alpha") -> float:
    if not _is_real(alpha):
        raise TypeError(f"{name} must be a real number, got {type(alpha).__name__}.")
    a = float(alpha)
    if not _math.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1), got {alpha!r}.")
    return a


def _check_probability(p: Any, name: str) -> float:
    if not _is_real(p):
        raise TypeError(f"{name} must be a real number.")
    v = float(p)
    if not _math.isfinite(v) or not 0.0 < v < 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1), got {p!r}.")
    return v


def _check_tolerance(tol: Any, name: str = "tolerance") -> Optional[float]:
    """None (undeclared) or a finite number in (0, 1]."""
    if tol is None:
        return None
    if not _is_real(tol):
        raise TypeError(f"{name} must be None or a real number.")
    v = float(tol)
    if not _math.isfinite(v) or not 0.0 < v <= 1.0:
        raise ValueError(f"{name} must be None or finite in (0, 1], got {tol!r}.")
    return v


def _check_count(v: Any, name: str) -> int:
    """An exact non-negative integer count. Floats are refused even when integral."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{name} must be an integer count, got {type(v).__name__} {v!r}.")
    if int(v) < 0:
        raise ValueError(f"{name} must be non-negative, got {v!r}.")
    return int(v)


def _check_count_array(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be an integer count array, got dtype {arr.dtype}.")
    arr = arr.astype(np.int64)
    if np.any(arr < 0):
        raise ValueError(f"{name} contains negative counts.")
    return arr


def _check_seed(seed: Any, name: str = "mc_seed") -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return int(seed)


def _binary_matrix(dataset: Any) -> np.ndarray:
    """2-D 0/1 matrix, without full-size temporaries. Signed and float arrays are refused."""
    x = np.asarray(dataset)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("a non-empty 2-D binary matrix is required.")
    if x.dtype == np.bool_:
        return x.view(np.uint8)
    if not np.issubdtype(x.dtype, np.unsignedinteger):
        raise TypeError(f"binary features must be unsigned integers or bool, got {x.dtype}.")
    if int(x.max()) > 1:
        raise ValueError("binary features must be exactly 0 or 1.")
    return x if x.dtype == np.uint8 else x.astype(np.uint8)


def _array_sha256(array: np.ndarray) -> str:
    """dtype + shape + C-order bytes; the D0 convention (common.provenance.array_sha256)."""
    from audit.dataset.common.provenance import array_sha256
    return array_sha256(array)


def _canonical(obj: Any) -> bytes:
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(b: bytes) -> str:
    return _hashlib.sha256(b).hexdigest()


def _sha256_file(path: Any) -> str:
    return _hashlib.sha256(_Path(path).read_bytes()).hexdigest()


def engine_sha256() -> str:
    """SHA-256 of this engine file as it exists on disk now."""
    return _sha256_file(__file__)


def _parse_utc(s: Any, name: str) -> _datetime.datetime:
    if not isinstance(s, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp string.")
    try:
        t = _datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} {s!r} is not ISO-8601.") from None
    if t.tzinfo is None:
        raise ValueError(f"{name} must carry a timezone.")
    return t.astimezone(_datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Exact detection tests (gating)
# ---------------------------------------------------------------------------

def exact_cell_pvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fisher's exact conditional test for every cell of m two-row tables.

    For cell c, under H0 (p_ac = p_bc) with independent samples, a_c given the
    column total t_c = a_c + b_c is Hypergeometric(N, t_c, n_a). The two-sided
    p-value is min(1, 2 * min(P(X <= a_c), P(X >= a_c))), which satisfies
    P(p <= u) <= u for every u (doubling a valid one-sided p-value). A cell
    with t_c = 0 carries no information and gets p = 1.
    """
    a = _check_count_array(a, "row a")
    b = _check_count_array(b, "row b")
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] < 2:
        raise ValueError("rows must be (m, K) integer arrays of equal shape with K >= 2.")
    na = a.sum(axis=1, keepdims=True)
    nb = b.sum(axis=1, keepdims=True)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError("each row needs at least one observation.")
    total = np.broadcast_to(na + nb, a.shape)
    col = a + b
    nab = np.broadcast_to(na, a.shape)
    lower = _hypergeom.cdf(a, total, col, nab)
    upper = _hypergeom.sf(a - 1, total, col, nab)
    p = np.minimum(1.0, 2.0 * np.minimum(lower, upper))
    p = np.where(col == 0, 1.0, p)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("non-finite exact p-value.")
    return p


def exact_homogeneity_test_batch(a: np.ndarray, b: np.ndarray, *, per_hypothesis_alpha: float) -> Dict[str, Any]:
    """Exact test of homogeneity for m two-row tables.

    p_table = min(1, L * min_c p_c), L = number of live cells (column total
    > 0): a Bonferroni combination of exact per-cell tests, so
    P(p_table <= u) <= u under H0 for every u. For K = 2 the two cells are
    the same test, so p_table = p_cell (L is taken as 1).
    """
    alpha = _check_alpha(per_hypothesis_alpha, "per_hypothesis_alpha")
    pc = exact_cell_pvalues(a, b)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    live = (a + b) > 0
    if a.shape[1] == 2:
        p = pc[:, 1]
        mult = np.ones(len(p), dtype=np.int64)
    else:
        mult = np.maximum(live.sum(axis=1), 1)
        p = np.minimum(1.0, mult * pc.min(axis=1))
    return {"p_value": p, "live_cells": mult, "significant": p < alpha}


def exact_binomial_pvalue(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p-value (scipy minlike)."""
    return float(binomtest(_check_count(k, "k"), _check_count(n, "n"),
                           p=_check_probability(p0, "p0"), alternative="two-sided").pvalue)


# ---------------------------------------------------------------------------
# Diagnostic (non-gating) homogeneity tests
# ---------------------------------------------------------------------------

def _homogeneity_batch(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """Pearson homogeneity statistics (df = live columns - 1). DIAGNOSTIC."""
    a = _check_count_array(a, "row a")
    b = _check_count_array(b, "row b")
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] < 2:
        raise ValueError("rows must be (m, K) integer arrays of equal shape with K >= 2.")
    obs = np.stack([a, b], axis=1).astype(np.float64)
    row = obs.sum(axis=2)
    if np.any(row == 0):
        raise ValueError("each row of a homogeneity table needs at least one observation.")
    col = obs.sum(axis=1)
    total = row.sum(axis=1)
    expected = row[:, :, None] * col[:, None, :] / total[:, None, None]
    live = col > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        cells = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0)
    stat = cells.sum(axis=(1, 2))
    df = live.sum(axis=1) - 1
    min_expected = np.where(live[:, None, :], expected, np.inf).min(axis=(1, 2))
    min_expected = np.where(np.isfinite(min_expected), min_expected, 0.0)
    p = np.ones_like(stat)
    has_df = df > 0
    p[has_df] = chi2.sf(stat[has_df], df[has_df])
    return {"chi_square": stat, "degrees_of_freedom": df,
            "minimum_expected_count": min_expected, "asymptotic_p_value": p}


def homogeneity_chi_square(table: Any) -> Dict[str, Any]:
    """Pearson chi-square homogeneity for one 2 x K table. DIAGNOSTIC."""
    t = _check_count_array(table, "table")
    if t.ndim != 2 or t.shape[0] != 2:
        raise ValueError("homogeneity_chi_square expects a 2 x K table.")
    r = _homogeneity_batch(t[0][None, :], t[1][None, :])
    min_e = float(r["minimum_expected_count"][0])
    return {"chi_square": float(r["chi_square"][0]),
            "degrees_of_freedom": int(r["degrees_of_freedom"][0]),
            "minimum_expected_count": min_e,
            "asymptotic_approximation_adequate": bool(min_e >= MINIMUM_EXPECTED_COUNT),
            "asymptotic_p_value": float(r["asymptotic_p_value"][0])}


def monte_carlo_draws_for_alpha(alpha: float) -> int:
    a = _check_alpha(alpha)
    return int(max(MC_MIN_DRAWS, np.ceil(MC_RESOLUTION_FACTOR / a) - 1))


def monte_carlo_homogeneity_p_value(row_a, row_b, *, draws: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Monte Carlo approximation to the exact conditional (Pearson-ordered) test. DIAGNOSTIC.

    Conditional on both margins, row A is multivariate hypergeometric under
    H0; p = (1 + E) / (B + 1) is valid for every B. The 99% Clopper-Pearson
    interval of E / B is the Monte Carlo uncertainty of the exact tail.
    """
    a = _check_count_array(row_a, "row_a").reshape(-1)
    b = _check_count_array(row_b, "row_b").reshape(-1)
    draws = _check_count(draws, "draws")
    if a.shape != b.shape or a.sum() == 0 or b.sum() == 0:
        raise ValueError("rows must have equal length and non-zero totals.")
    col = a + b
    live = col > 0
    a, col = a[live], col[live]
    n_a, n = int(a.sum()), int(col.sum())
    if live.sum() <= 1 or draws == 0:
        return {"p_value": 1.0, "draws": 0, "exceedances": 0, "tail_probability_ci99": [0.0, 1.0]}
    exp_a, exp_b = n_a * col / n, (n - n_a) * col / n

    def stat(x):
        y = col - x
        return (((x - exp_a) ** 2) / exp_a).sum(axis=-1) + (((y - exp_b) ** 2) / exp_b).sum(axis=-1)

    t_obs = float(stat(a.astype(np.float64)))
    threshold = t_obs - 1e-9 * max(1.0, t_obs)
    exceed, remaining = 0, draws
    while remaining > 0:
        size = min(remaining, MC_CHUNK)
        sims = rng.multivariate_hypergeometric(col, n_a, size=size).astype(np.float64)
        exceed += int(np.sum(stat(sims) >= threshold))
        remaining -= size
    lo, hi = clopper_pearson_bounds(np.int64(exceed), np.int64(draws), alpha=0.01)
    return {"p_value": (1 + exceed) / (draws + 1), "draws": draws, "exceedances": exceed,
            "tail_probability_ci99": [float(lo), float(hi)]}


def homogeneity_test_batch(a, b, *, per_hypothesis_alpha: float, mc_seed: int,
                           mc_stream: Sequence[int] = (0,), mc_max_draws: int = MC_MAX_DRAWS) -> Dict[str, Any]:
    """Asymptotic chi-square (observed df), Monte Carlo where sparse. DIAGNOSTIC, never gating."""
    alpha = _check_alpha(per_hypothesis_alpha, "per_hypothesis_alpha")
    seed = _check_seed(mc_seed)
    stream = [_check_seed(s, "mc_stream entry") for s in mc_stream]
    cap = _check_count(mc_max_draws, "mc_max_draws")
    h = _homogeneity_batch(a, b)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    m = len(h["chi_square"])
    p = h["asymptotic_p_value"].copy()
    method = np.full(m, "asymptotic", dtype=object)
    attainable = np.ones(m, dtype=bool)
    mc_draws = np.zeros(m, dtype=np.int64)
    mc_exceed = np.zeros(m, dtype=np.int64)
    degenerate = h["degrees_of_freedom"] == 0
    method[degenerate] = "degenerate"
    sparse = (~degenerate) & (h["minimum_expected_count"] < MINIMUM_EXPECTED_COUNT)
    draws = min(cap, monte_carlo_draws_for_alpha(alpha))
    for i in np.flatnonzero(sparse):
        mc = monte_carlo_homogeneity_p_value(a[i], b[i], draws=draws,
                                             rng=np.random.default_rng([seed, *stream, int(i)]))
        p[i], method[i] = mc["p_value"], "monte_carlo"
        mc_draws[i], mc_exceed[i] = mc["draws"], mc["exceedances"]
        attainable[i] = 1.0 / (draws + 1) < alpha
    return {"chi_square": h["chi_square"], "degrees_of_freedom": h["degrees_of_freedom"],
            "minimum_expected_count": h["minimum_expected_count"], "p_value": p, "method": method,
            "monte_carlo_draws": mc_draws, "monte_carlo_exceedances": mc_exceed,
            "rejection_attainable": attainable, "significant": (p < alpha) & attainable}


# ---------------------------------------------------------------------------
# Finite-sample confidence bounds (BOUND_CONSTRUCTION)
# ---------------------------------------------------------------------------

def clopper_pearson_bounds(k: Any, n: Any, *, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact two-sided binomial interval at confidence >= 1 - alpha, vectorised.

    For every p in [0, 1], P_p(p not in [L(K), U(K)]) <= alpha with
    K ~ Binomial(n, p), for every n. Limits are rounded outward and clipped
    to [0, 1], which can only widen the interval.
    """
    a = _check_alpha(alpha)
    k = _check_count_array(k, "k")
    n = _check_count_array(n, "n")
    k, n = np.broadcast_arrays(k, n)
    if np.any(n < 1) or np.any(k > n):
        raise ValueError("require n >= 1 and 0 <= k <= n.")
    kf, nf = k.astype(np.float64), n.astype(np.float64)
    with np.errstate(invalid="ignore"):
        lo = np.where(k > 0, _beta_dist.ppf(a / 2.0, kf, nf - kf + 1.0), 0.0)
        hi = np.where(k < n, _beta_dist.ppf(1.0 - a / 2.0, kf + 1.0, nf - kf), 1.0)
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise FloatingPointError("Clopper-Pearson limit is not finite.")
    return np.clip(lo - _OUTWARD, 0.0, 1.0), np.clip(hi + _OUTWARD, 0.0, 1.0)


def _difference_bounds(la, ua, lb, ub):
    upper = np.maximum(ua - lb, ub - la)
    lower = np.maximum(0.0, np.maximum(la - ub, lb - ua))
    return lower, upper


def proportion_difference_bounds(k1, n1, k2, n2, *, alpha: float) -> Dict[str, np.ndarray]:
    """Simultaneous bounds on |p1 - p2|: Clopper-Pearson at alpha / 2 per proportion."""
    a = _check_alpha(alpha)
    l1, u1 = clopper_pearson_bounds(k1, n1, alpha=a / 2.0)
    l2, u2 = clopper_pearson_bounds(k2, n2, alpha=a / 2.0)
    lower, upper = _difference_bounds(l1, u1, l2, u2)
    return {"abs_lower": lower, "abs_upper": np.minimum(upper, 1.0),
            "difference_lower": l1 - u2, "difference_upper": u1 - l2}


def tvd_confidence_bounds(counts_a, counts_b, *, alpha: float) -> Dict[str, np.ndarray]:
    """Simultaneous finite-sample bounds on TVD between two K-cell laws.

    All 2K cell probabilities get Clopper-Pearson intervals at alpha / (2K);
    by Bonferroni all cover with probability >= 1 - alpha, and on that event
    0.5 * sum d_c <= TVD <= min(1, 0.5 * sum D_c) (interval arithmetic).
    """
    a = _check_alpha(alpha)
    ca = _check_count_array(counts_a, "counts_a")
    cb = _check_count_array(counts_b, "counts_b")
    if ca.shape != cb.shape or ca.shape[-1] < 2:
        raise ValueError("counts must share a shape with at least two cells.")
    k_cells = ca.shape[-1]
    na = ca.sum(axis=-1, keepdims=True)
    nb = cb.sum(axis=-1, keepdims=True)
    if np.any(na < 1) or np.any(nb < 1):
        raise ValueError("each sample needs at least one observation.")
    per = a / (2.0 * k_cells)
    la, ua = clopper_pearson_bounds(ca, na, alpha=per)
    lb, ub = clopper_pearson_bounds(cb, nb, alpha=per)
    d, big_d = _difference_bounds(la, ua, lb, ub)
    return {"lower": 0.5 * d.sum(axis=-1), "upper": np.minimum(1.0, 0.5 * big_d.sum(axis=-1))}


# ---------------------------------------------------------------------------
# Hypothesis classification
# ---------------------------------------------------------------------------

def classify_hypothesis(*, significant: bool, effect: float, lower_bound: float,
                        upper_bound: float, tolerance: Optional[float]) -> str:
    """Category of one hypothesis (DECISION_RULE_VERSION).

        lower_bound > tol                      CONFIRMED_DEVIATION          FAIL   (G1)
        upper_bound <= tol, exact detection    STATISTICAL_ONLY             CONDITIONAL_PASS (G1, G2)
        upper_bound <= tol, no detection       EQUIVALENT_WITHIN_TOLERANCE  PASS   (G1)
        bounds straddle tol:
            exact detection                    SIGNIFICANT_PRACTICALITY_UNRESOLVED
            point effect > tol                 PRACTICAL_UNSUPPORTED
            otherwise                          INSUFFICIENT_PRECISION

    A point estimate above tol never establishes a practical deviation (the
    TVD estimator is biased upward); non-detection never establishes equivalence.
    """
    tol = _check_tolerance(tolerance)
    vals = (float(effect), float(lower_bound), float(upper_bound))
    if not all(_math.isfinite(v) for v in vals):
        raise ValueError("effect and bounds must be finite.")
    eff, lo, hi = vals
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"bounds must satisfy 0 <= lower <= upper <= 1, got {lo}, {hi}.")
    sig = bool(significant)
    if tol is None:
        return "SIGNIFICANT_NO_TOLERANCE" if sig else "NO_TOLERANCE_DECLARED"
    if lo > tol:
        return "CONFIRMED_DEVIATION"
    if hi <= tol:
        return "STATISTICAL_ONLY" if sig else "EQUIVALENT_WITHIN_TOLERANCE"
    if sig:
        return "SIGNIFICANT_PRACTICALITY_UNRESOLVED"
    if eff > tol:
        return "PRACTICAL_UNSUPPORTED"
    return "INSUFFICIENT_PRECISION"


def _combine_outcomes(outcomes: Sequence[str]) -> str:
    for o in _OUTCOME_ORDER:
        if o in outcomes:
            return o
    return "INCONCLUSIVE"


def allocate_familywise_alpha(global_alpha: float, family_ids: Sequence[str]) -> Dict[str, float]:
    """Equal split of one global alpha over families (a design choice needing preregistration)."""
    g = _check_alpha(global_alpha, "global_alpha")
    ids = list(family_ids)
    if not ids or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("family_ids must be non-empty, distinct strings.")
    return {f: g / len(ids) for f in ids}


def _family(*, family_id, role, description, familywise_alpha, records, categories,
            tolerance, tolerance_key, effect_measure, hypothesis_kind) -> Dict[str, Any]:
    if role not in ("confirmatory", "exploratory"):
        raise ValueError("role must be 'confirmatory' or 'exploratory'.")
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("family_id must be a non-empty string.")
    fa = _check_alpha(familywise_alpha, "familywise_alpha")
    m = len(categories)
    counts: Dict[str, int] = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    outcome = _combine_outcomes([CATEGORY_OUTCOME[c] for c in categories]) if role == "confirmatory" else "NOT_GATED"
    return {"family_id": family_id, "role": role, "description": description,
            "hypothesis_kind": hypothesis_kind, "hypothesis_count": m, "correction": "Bonferroni",
            "familywise_alpha": fa, "per_hypothesis_alpha": fa / max(1, m),
            "detection_test": DETECTION_TEST, "bound_construction": BOUND_CONSTRUCTION,
            "effect_measure": effect_measure, "tolerance": tolerance, "tolerance_key": tolerance_key,
            "category_counts": dict(sorted(counts.items())), "outcome": outcome, "records": records}


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def binomial_reference_family(items, *, family_id, description, familywise_alpha, tolerance,
                              tolerance_key=None, role="confirmatory") -> Dict[str, Any]:
    """One-sample conformity: exact binomial detection, Clopper-Pearson bounds.

    Each item: {"label", "successes" (int), "trials" (int), "expected_probability"}.
    """
    m = len(items)
    if m == 0:
        raise ValueError("a family needs at least one hypothesis.")
    tol = _check_tolerance(tolerance)
    alpha_h = _check_alpha(familywise_alpha, "familywise_alpha") / m
    keys = ("label", "successes", "trials", "expected_probability", "observed_probability",
            "absolute_difference", "p_value", "ci_lower", "ci_upper",
            "absolute_difference_lower_bound", "absolute_difference_upper_bound", "significant", "category")
    rec: Dict[str, list] = {k: [] for k in keys}
    cats = []
    for it in items:
        unknown = set(it) - {"label", "successes", "trials", "expected_probability"}
        if unknown:
            raise ValueError(f"unknown item fields {sorted(unknown)}; the tolerance is family-level.")
        k = _check_count(it["successes"], "successes")
        n = _check_count(it["trials"], "trials")
        if n < 1 or k > n:
            raise ValueError(f"invalid binomial counts k={k}, n={n}.")
        p0 = _check_probability(it["expected_probability"], "expected_probability")
        pv = exact_binomial_pvalue(k, n, p0)
        lo, hi = clopper_pearson_bounds(np.int64(k), np.int64(n), alpha=alpha_h)
        lo, hi = float(lo), float(hi)
        diff = abs(k / n - p0)
        ub = min(1.0, max(abs(lo - p0), abs(hi - p0)))
        lb = max(0.0, lo - p0, p0 - hi)
        sig = bool(pv < alpha_h)
        cat = classify_hypothesis(significant=sig, effect=diff, lower_bound=lb, upper_bound=ub, tolerance=tol)
        cats.append(cat)
        for key, val in zip(keys, (str(it["label"]), k, n, p0, k / n, diff, pv, lo, hi, lb, ub, sig, cat)):
            rec[key].append(val)
    return _family(family_id=family_id, role=role, description=description, familywise_alpha=familywise_alpha,
                   records=rec, categories=cats, tolerance=tol, tolerance_key=tolerance_key,
                   effect_measure="absolute difference from the reference probability",
                   hypothesis_kind="binomial_reference")


def two_sample_proportion_family(items, *, family_id, description, familywise_alpha, tolerance, mc_seed,
                                 tolerance_key=None, role="confirmatory") -> Dict[str, Any]:
    """Two independent binomial samples: H0 p1 == p2. Fisher exact detection; CP bounds.

    Each item: {"label", "k1", "n1", "k2", "n2"} (exact integers).
    """
    m = len(items)
    if m == 0:
        raise ValueError("a family needs at least one hypothesis.")
    for it in items:
        unknown = set(it) - {"label", "k1", "n1", "k2", "n2"}
        if unknown:
            raise ValueError(f"unknown item fields {sorted(unknown)}; the tolerance is family-level.")
    tol = _check_tolerance(tolerance)
    alpha_h = _check_alpha(familywise_alpha, "familywise_alpha") / m
    k1 = np.array([_check_count(i["k1"], "k1") for i in items], dtype=np.int64)
    n1 = np.array([_check_count(i["n1"], "n1") for i in items], dtype=np.int64)
    k2 = np.array([_check_count(i["k2"], "k2") for i in items], dtype=np.int64)
    n2 = np.array([_check_count(i["n2"], "n2") for i in items], dtype=np.int64)
    if np.any(n1 < 1) or np.any(n2 < 1) or np.any(k1 > n1) or np.any(k2 > n2):
        raise ValueError("invalid two-sample binomial counts.")
    ta, tb = np.stack([n1 - k1, k1], 1), np.stack([n2 - k2, k2], 1)
    ex = exact_homogeneity_test_batch(ta, tb, per_hypothesis_alpha=alpha_h)
    dg = homogeneity_test_batch(ta, tb, per_hypothesis_alpha=alpha_h, mc_seed=mc_seed, mc_stream=(0,))
    bd = proportion_difference_bounds(k1, n1, k2, n2, alpha=alpha_h)
    diff = np.abs(k1 / n1 - k2 / n2)
    cats = [classify_hypothesis(significant=bool(ex["significant"][i]), effect=float(diff[i]),
                                lower_bound=float(bd["abs_lower"][i]), upper_bound=float(bd["abs_upper"][i]),
                                tolerance=tol) for i in range(m)]
    rec = {"label": [str(i["label"]) for i in items], "k1": k1.tolist(), "n1": n1.tolist(),
           "k2": k2.tolist(), "n2": n2.tolist(), "p1": (k1 / n1).tolist(), "p2": (k2 / n2).tolist(),
           "absolute_difference": diff.tolist(), "difference_lower": bd["difference_lower"].tolist(),
           "difference_upper": bd["difference_upper"].tolist(),
           "absolute_difference_lower_bound": bd["abs_lower"].tolist(),
           "absolute_difference_upper_bound": bd["abs_upper"].tolist(),
           "p_value": ex["p_value"].tolist(), "significant": ex["significant"].tolist(),
           "diagnostic_chi_square_p_value": dg["p_value"].tolist(),
           "diagnostic_method": dg["method"].tolist(), "category": cats}
    return _family(family_id=family_id, role=role, description=description, familywise_alpha=familywise_alpha,
                   records=rec, categories=cats, tolerance=tol, tolerance_key=tolerance_key,
                   effect_measure="absolute difference in proportions", hypothesis_kind="two_sample_proportion")


def _pair_cell_counts(x: np.ndarray) -> np.ndarray:
    """(pairs, 4) exact integer counts [(0,0),(0,1),(1,0),(1,1)] per i<j pair."""
    x = _binary_matrix(x)
    n, d = x.shape
    n11 = np.zeros((d, d), dtype=np.int64)
    ones = np.zeros(d, dtype=np.int64)
    for start in range(0, n, 100_000):
        xb = x[start:start + 100_000].astype(np.float64)
        n11 += np.rint(xb.T @ xb).astype(np.int64)
        ones += np.rint(xb.sum(axis=0)).astype(np.int64)
    iu, ju = np.triu_indices(d, k=1)
    a11 = n11[iu, ju]
    a10 = ones[iu] - a11
    a01 = ones[ju] - a11
    return np.stack([n - a11 - a10 - a01, a01, a10, a11], axis=1)


def _phi_batch(t: np.ndarray) -> np.ndarray:
    a, b, c, d = (t[:, k].astype(np.float64) for k in range(4))
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (a * d - b * c) / denom, 0.0)


def pair_group_labels(feature_count: int, feature_groups) -> list:
    """Label each i<j pair 'within:<g>' or 'cross:<g1>|<g2>' (or 'all')."""
    iu, ju = np.triu_indices(feature_count, k=1)
    if not feature_groups:
        return ["all"] * len(iu)
    owner = {}
    for g, cols in feature_groups.items():
        for c in cols:
            c = _check_count(c, "feature index")
            if c in owner:
                raise ValueError(f"feature {c} is in more than one group.")
            owner[c] = g
    if set(owner) != set(range(feature_count)):
        raise ValueError("feature_groups must partition every feature exactly once.")
    order = list(feature_groups)
    out = []
    for i, j in zip(iu, ju):
        gi, gj = owner[int(i)], owner[int(j)]
        out.append(f"within:{gi}" if gi == gj else "cross:{}|{}".format(*sorted((gi, gj), key=order.index)))
    return out


def pairwise_homogeneity_family(comparisons, *, family_id, description, familywise_alpha, tvd_tolerance,
                                mc_seed, tolerance_key=None, role="confirmatory", feature_groups=None):
    """Second-order homogeneity of every feature pair across comparisons.

    Family = all pairs of all comparisons (Bonferroni divisor pairs x
    comparisons). Detection: exact per-cell Fisher tests; effect: TVD with
    simultaneous Clopper-Pearson bounds. Chi-square is recorded as a diagnostic.
    """
    if not comparisons:
        raise ValueError("at least one comparison is required.")
    tol = _check_tolerance(tvd_tolerance, "tvd_tolerance")
    seed = _check_seed(mc_seed)
    tables, d = {}, None
    for name, (xa, xb) in comparisons.items():
        xa, xb = _binary_matrix(xa), _binary_matrix(xb)
        if xa.shape[1] != xb.shape[1] or (d is not None and xa.shape[1] != d) or xa.shape[1] < 2:
            raise ValueError("all compared partitions must share one feature dimension >= 2.")
        d = xa.shape[1]
        tables[str(name)] = (_pair_cell_counts(xa), len(xa), _pair_cell_counts(xb), len(xb))
    pairs_per = d * (d - 1) // 2
    alpha_h = _check_alpha(familywise_alpha, "familywise_alpha") / (pairs_per * len(tables))
    iu, ju = np.triu_indices(d, k=1)
    groups = pair_group_labels(d, feature_groups)
    g_arr = np.asarray(groups)
    per_comparison, cats_all = {}, []
    for c_idx, (name, (ta, na, tb, nb)) in enumerate(tables.items()):
        ex = exact_homogeneity_test_batch(ta, tb, per_hypothesis_alpha=alpha_h)
        dg = homogeneity_test_batch(ta, tb, per_hypothesis_alpha=alpha_h, mc_seed=seed, mc_stream=(c_idx + 1,))
        tvd = 0.5 * np.abs(ta / na - tb / nb).sum(axis=1)
        bd = tvd_confidence_bounds(ta, tb, alpha=alpha_h)
        dphi = np.abs(_phi_batch(ta) - _phi_batch(tb))
        sig = ex["significant"]
        cats = [classify_hypothesis(significant=bool(sig[k]), effect=float(tvd[k]), lower_bound=float(bd["lower"][k]),
                                    upper_bound=float(bd["upper"][k]), tolerance=tol) for k in range(pairs_per)]
        cats_all.extend(cats)
        by_group = {}
        for g in dict.fromkeys(groups):
            sel = g_arr == g
            by_group[g] = {"pairs": int(sel.sum()), "maximum_tvd": float(tvd[sel].max()),
                           "mean_tvd": float(tvd[sel].mean()),
                           "maximum_tvd_lower_bound": float(bd["lower"][sel].max()),
                           "maximum_tvd_upper_bound": float(bd["upper"][sel].max()),
                           "maximum_absolute_delta_phi": float(dphi[sel].max()),
                           "significant_pairs": int(sig[sel].sum()),
                           "minimum_p_value": float(ex["p_value"][sel].min())}
        per_comparison[name] = {
            "sample_sizes": {"partition_a": int(na), "partition_b": int(nb)},
            "summary": {"maximum_tvd": float(tvd.max()), "q95_tvd": float(np.quantile(tvd, 0.95)),
                        "mean_tvd": float(tvd.mean()), "maximum_tvd_lower_bound": float(bd["lower"].max()),
                        "maximum_tvd_upper_bound": float(bd["upper"].max()),
                        "minimum_tvd_upper_bound": float(bd["upper"].min()),
                        "maximum_absolute_delta_phi": float(dphi.max()), "significant_pairs": int(sig.sum()),
                        "minimum_p_value": float(ex["p_value"].min()),
                        "diagnostic_chi_square_significant_pairs": int(dg["significant"].sum()),
                        "diagnostic_methods": {k: int(v) for k, v in
                                               zip(*np.unique(dg["method"].astype(str), return_counts=True))}},
            "by_pair_group": by_group,
            "pairs": {"feature_i": iu.tolist(), "feature_j": ju.tolist(), "group": groups,
                      "cell_counts_a": ta.tolist(), "cell_counts_b": tb.tolist(),
                      "tvd": tvd.tolist(), "tvd_lower_bound": bd["lower"].tolist(),
                      "tvd_upper_bound": bd["upper"].tolist(), "delta_phi": dphi.tolist(),
                      "p_value": ex["p_value"].tolist(), "significant": sig.tolist(),
                      "diagnostic_chi_square_p_value": dg["p_value"].tolist(),
                      "diagnostic_degrees_of_freedom": dg["degrees_of_freedom"].tolist(),
                      "diagnostic_method": dg["method"].tolist(), "category": cats}}
    fam = _family(family_id=family_id, role=role, description=description, familywise_alpha=familywise_alpha,
                  records={"comparisons": per_comparison}, categories=cats_all, tolerance=tol,
                  tolerance_key=tolerance_key,
                  effect_measure="total variation distance between 4-cell feature-pair laws",
                  hypothesis_kind="pairwise_homogeneity")
    fam["comparisons"] = list(tables)
    fam["pairs_per_comparison"] = pairs_per
    return fam


# ---------------------------------------------------------------------------
# Sealed audit result
# ---------------------------------------------------------------------------

_SEAL_TOKEN = object()
_BINDING_TOKEN = object()
_PREREG_TOKEN = object()


@_dataclass(frozen=True)
class AuditResult:
    """Sealed output of class_stratified_audit().

    ``sealed_json`` is the canonical JSON of the complete result (families,
    design, input provenance, configuration); ``seal_sha256`` is its SHA-256.
    The decision is computed from a fresh parse of ``sealed_json`` after the
    seal is re-verified, so editing a returned dictionary cannot change it.
    """
    sealed_json: bytes = _field(repr=False)
    seal_sha256: str = ""
    _token: object = _field(default=None, repr=False, compare=False)

    def payload(self) -> Dict[str, Any]:
        """A fresh, independent copy of the sealed result."""
        return _json.loads(self.sealed_json)


#: The six confirmatory families and their tolerance keys.
CLASS_STRATIFIED_CONFIRMATORY_FAMILIES = ("R1", "R2", "A", "B", "C", "D")
TOLERANCE_KEYS = ("class_ratio", "class_conditional_bit", "pairwise_tvd", "cross_class_bit")
FAMILY_TOLERANCE_KEY = {"R1": "class_ratio", "R2": "class_conditional_bit", "A": "pairwise_tvd",
                        "B": "pairwise_tvd", "C": "pairwise_tvd", "D": "cross_class_bit"}
FAMILY_KIND = {"R1": "binomial_reference", "R2": "binomial_reference", "A": "pairwise_homogeneity",
               "B": "pairwise_homogeneity", "C": "pairwise_homogeneity", "D": "two_sample_proportion",
               "E": "pairwise_homogeneity"}


def expected_hypothesis_counts(partition_count: int, feature_dimension: int) -> Dict[str, int]:
    """Hypothesis count of every family, from the design alone."""
    p, d = _check_count(partition_count, "partition_count"), _check_count(feature_dimension, "feature_dimension")
    pairs, comps = d * (d - 1) // 2, p * (p - 1) // 2
    return {"R1": p, "R2": 2 * p * d, "A": comps * pairs, "B": comps * pairs, "C": comps * pairs,
            "D": p * d, "E": p * pairs}


def class_stratified_audit(partitions, *, expected_class_ratio, expected_bit_probability, tolerances,
                           family_alpha_allocation, mc_seed, feature_groups=None,
                           exploratory_alpha=None, family_roles=None) -> AuditResult:
    """Every D3 family for binary features with binary labels, sealed with its inputs' provenance.

        R1  class balance per partition vs expected_class_ratio
        R2  per-class, per-partition, per-bit marginal vs expected_bit_probability[label]
        A   pooled rows: every feature pair, every partition comparison
        B   Y == 1 rows only: as A
        C   Y == 0 rows only: as A
        D   Y == 1 vs Y == 0 per-bit marginals, per partition
        E   EXPLORATORY (only if exploratory_alpha is given); never gated

    family_roles, if given, maps each of R1..D to "confirmatory" or
    "exploratory", plus optionally "E": "exploratory" (E is built only if
    listed). Exploratory families are computed at exploratory_alpha within the
    family, outside the confirmatory allocation, and never gate. Without it,
    R1..D are confirmatory and E is built iff exploratory_alpha is given.
    family_alpha_allocation covers exactly the confirmatory families.

    The returned AuditResult records the partition order, sizes, feature
    dimension, class counts and the array SHA-256 of every input X and Y,
    computed here from the arrays actually analysed.
    """
    missing = [k for k in TOLERANCE_KEYS if k not in tolerances]
    if missing:
        raise ValueError(f"tolerances must declare {missing} (None means undeclared).")
    if set(tolerances) - set(TOLERANCE_KEYS):
        raise ValueError(f"unknown tolerance keys {sorted(set(tolerances) - set(TOLERANCE_KEYS))}.")
    tols = {k: _check_tolerance(tolerances[k], f"tolerance {k}") for k in TOLERANCE_KEYS}
    e_alpha = None if exploratory_alpha is None else _check_alpha(exploratory_alpha, "exploratory_alpha")
    if family_roles is None:
        roles = {f: "confirmatory" for f in CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
        if e_alpha is not None:
            roles["E"] = "exploratory"
    else:
        if not isinstance(family_roles, Mapping):
            raise TypeError("family_roles must be a mapping.")
        roles = {str(k): v for k, v in family_roles.items()}
        if (set(roles) - {"E"} != set(CLASS_STRATIFIED_CONFIRMATORY_FAMILIES)
                or any(v not in ("confirmatory", "exploratory") for v in roles.values())
                or roles.get("E", "exploratory") != "exploratory"):
            raise ValueError("family_roles must give each of R1..D a role, and E (optional) must be exploratory.")
    confirm = [f for f in CLASS_STRATIFIED_CONFIRMATORY_FAMILIES if roles[f] == "confirmatory"]
    if not confirm:
        raise ValueError("at least one family must be confirmatory.")
    if set(family_alpha_allocation) != set(confirm):
        raise ValueError(f"family_alpha_allocation must cover exactly the confirmatory families {confirm}.")
    if e_alpha is None and "exploratory" in roles.values():
        raise ValueError("exploratory families need exploratory_alpha.")
    alloc = {f: _check_alpha(family_alpha_allocation[f], f"alpha for {f}") for f in confirm}
    fam_alpha = {f: alloc.get(f, e_alpha) for f in CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
    ratio = _check_probability(expected_class_ratio, "expected_class_ratio")
    if set(expected_bit_probability) != {0, 1}:
        raise ValueError("expected_bit_probability must give labels 0 and 1.")
    ref = {lab: _check_probability(expected_bit_probability[lab], f"bit probability Y={lab}") for lab in (0, 1)}
    seed = _check_seed(mc_seed)
    names = [n for n in partitions]
    if (len(names) < 2 or len(set(names)) != len(names)
            or not all(isinstance(n, str) and n and "_" not in n and ":" not in n for n in names)):
        raise ValueError("at least two distinct partition names (non-empty strings without '_' or ':') are required.")
    xs, ys = {}, {}
    for name, (x, y) in zip(names, partitions.values()):
        xs[name] = _binary_matrix(x)
        ys[name] = _validate_labels(y, len(xs[name]))
        if ys[name].min() == ys[name].max():
            raise ValueError(f"partition {name!r} contains a single class.")
    dims = {x.shape[1] for x in xs.values()}
    if len(dims) != 1:
        raise ValueError("all partitions must share one feature dimension.")
    d = dims.pop()
    pairs_of = list(combinations(names, 2))
    pos = {n: ys[n] == 1 for n in names}
    subset = {n: {1: xs[n][pos[n]], 0: xs[n][~pos[n]]} for n in names}
    ones = {n: {lab: subset[n][lab].sum(axis=0, dtype=np.int64) for lab in (0, 1)} for n in names}
    count = {n: {lab: int(subset[n][lab].shape[0]) for lab in (0, 1)} for n in names}

    fams = [binomial_reference_family(
        [{"label": n, "successes": count[n][1], "trials": len(ys[n]), "expected_probability": ratio} for n in names],
        family_id="R1", description="class balance per partition against the declared ratio",
        familywise_alpha=fam_alpha["R1"], tolerance=tols["class_ratio"], tolerance_key="class_ratio",
        role=roles["R1"])]
    fams.append(binomial_reference_family(
        [{"label": f"{n}:Y={lab}:bit{j}", "successes": ones[n][lab][j], "trials": count[n][lab],
          "expected_probability": ref[lab]} for n in names for lab in (1, 0) for j in range(d)],
        family_id="R2", description="class-conditional bit marginals against the declared reference",
        familywise_alpha=fam_alpha["R2"], tolerance=tols["class_conditional_bit"],
        tolerance_key="class_conditional_bit", role=roles["R2"]))
    for fid, sel, desc, off in (("A", None, "pooled rows", 1), ("B", 1, "Y == 1 rows", 2), ("C", 0, "Y == 0 rows", 3)):
        def rows(n, sel=sel):
            return xs[n] if sel is None else subset[n][sel]
        fams.append(pairwise_homogeneity_family(
            {f"{a}_{b}": (rows(a), rows(b)) for a, b in pairs_of}, family_id=fid,
            description=f"partition consistency of feature-pair laws, {desc}", familywise_alpha=fam_alpha[fid],
            tvd_tolerance=tols["pairwise_tvd"], tolerance_key="pairwise_tvd", mc_seed=seed + off,
            feature_groups=feature_groups, role=roles[fid]))
    fams.append(two_sample_proportion_family(
        [{"label": f"{n}:bit{j}", "k1": ones[n][1][j], "n1": count[n][1], "k2": ones[n][0][j], "n2": count[n][0]}
         for n in names for j in range(d)],
        family_id="D", description="cross-class first-order marginals, Y == 1 vs Y == 0, per partition",
        familywise_alpha=fam_alpha["D"], tolerance=tols["cross_class_bit"], tolerance_key="cross_class_bit",
        mc_seed=seed + 4, role=roles["D"]))
    if "E" in roles:
        fams.append(pairwise_homogeneity_family(
            {f"{n}:Y1_vs_Y0": (subset[n][1], subset[n][0]) for n in names}, family_id="E", role="exploratory",
            description="EXPLORATORY cross-class feature-pair structure; characterised, never gated",
            familywise_alpha=e_alpha, tvd_tolerance=None, mc_seed=seed + 5, feature_groups=feature_groups))
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "method_version": D3_METHOD_VERSION, "decision_rule_version": DECISION_RULE_VERSION,
        "engine_sha256": engine_sha256(),
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "design": {"partition_order": names, "partition_sizes": {n: int(len(ys[n])) for n in names},
                   "feature_dimension": int(d),
                   "class_counts": {n: {"positive": count[n][1], "negative": count[n][0]} for n in names},
                   "comparisons": [f"{a}_{b}" for a, b in pairs_of]},
        "input_provenance": {n: {"features_sha256": _array_sha256(xs[n]), "labels_sha256": _array_sha256(ys[n])}
                             for n in names},
        "configuration": {"expected_class_ratio": ratio, "expected_bit_probability": {str(k): v for k, v in ref.items()},
                          "tolerances": tols, "family_alpha_allocation": alloc, "exploratory_alpha": e_alpha,
                          "family_roles": roles, "mc_seed": seed, "feature_groups": ({str(g): [int(c) for c in cols]
                                                               for g, cols in feature_groups.items()}
                                                              if feature_groups else None)},
        "families": fams,
    }
    blob = _canonical(payload)
    return AuditResult(sealed_json=blob, seal_sha256=_sha256_bytes(blob), _token=_SEAL_TOKEN)


def _reclassify(fam: Mapping[str, Any]) -> list:
    """Re-derive every category from the family's own hypothesis records."""
    tol = fam["tolerance"]
    if fam["hypothesis_kind"] == "pairwise_homogeneity":
        out = []
        for c in fam["records"]["comparisons"].values():
            p = c["pairs"]
            out.extend(classify_hypothesis(significant=s, effect=e, lower_bound=lo, upper_bound=hi, tolerance=tol)
                       for s, e, lo, hi in zip(p["significant"], p["tvd"], p["tvd_lower_bound"], p["tvd_upper_bound"]))
        return out
    r = fam["records"]
    return [classify_hypothesis(significant=s, effect=e, lower_bound=lo, upper_bound=hi, tolerance=tol)
            for s, e, lo, hi in zip(r["significant"], r["absolute_difference"],
                                    r["absolute_difference_lower_bound"], r["absolute_difference_upper_bound"])]


def _recompute_mismatches(fam: Mapping[str, Any]) -> list:
    """Recompute every gating quantity from the family's own sufficient statistics.

    Exact p-values, significance flags, effects and bounds are recomputed from
    the stored integer counts with the same exact methods and compared with
    the records; any difference is returned. Diagnostics are not recomputed
    (they never gate).
    """
    out = []
    ah = fam["per_hypothesis_alpha"]

    def cmp(name, got, want, exact=False):
        got, want = np.asarray(got), np.asarray(want)
        same = (got.shape == want.shape) and (np.array_equal(got, want) if exact
                                              else np.allclose(got, want, rtol=1e-10, atol=1e-15))
        if not same:
            out.append(f"{fam['family_id']} {name} does not follow from its counts")

    kind = fam["hypothesis_kind"]
    if kind == "binomial_reference":
        r = fam["records"]
        k = np.asarray(r["successes"], dtype=np.int64)
        n = np.asarray(r["trials"], dtype=np.int64)
        p0 = np.asarray(r["expected_probability"], dtype=np.float64)
        pv = np.array([exact_binomial_pvalue(int(a), int(b), float(c)) for a, b, c in zip(k, n, p0)])
        lo, hi = clopper_pearson_bounds(k, n, alpha=ah)
        cmp("p_value", r["p_value"], pv)
        cmp("significant", r["significant"], pv < ah, exact=True)
        cmp("absolute_difference", r["absolute_difference"], np.abs(k / n - p0))
        cmp("upper bound", r["absolute_difference_upper_bound"], np.minimum(1.0, np.maximum(np.abs(lo - p0), np.abs(hi - p0))))
        cmp("lower bound", r["absolute_difference_lower_bound"], np.maximum(0.0, np.maximum(lo - p0, p0 - hi)))
    elif kind == "two_sample_proportion":
        r = fam["records"]
        k1, n1, k2, n2 = (np.asarray(r[x], dtype=np.int64) for x in ("k1", "n1", "k2", "n2"))
        ex = exact_homogeneity_test_batch(np.stack([n1 - k1, k1], 1), np.stack([n2 - k2, k2], 1), per_hypothesis_alpha=ah)
        bd = proportion_difference_bounds(k1, n1, k2, n2, alpha=ah)
        cmp("p_value", r["p_value"], ex["p_value"])
        cmp("significant", r["significant"], ex["significant"], exact=True)
        cmp("absolute_difference", r["absolute_difference"], np.abs(k1 / n1 - k2 / n2))
        cmp("upper bound", r["absolute_difference_upper_bound"], bd["abs_upper"])
        cmp("lower bound", r["absolute_difference_lower_bound"], bd["abs_lower"])
    else:
        for name, c in fam["records"]["comparisons"].items():
            pr = c["pairs"]
            ta = np.asarray(pr["cell_counts_a"], dtype=np.int64)
            tb = np.asarray(pr["cell_counts_b"], dtype=np.int64)
            ex = exact_homogeneity_test_batch(ta, tb, per_hypothesis_alpha=ah)
            bd = tvd_confidence_bounds(ta, tb, alpha=ah)
            na, nb = ta.sum(axis=1, keepdims=True), tb.sum(axis=1, keepdims=True)
            cmp(f"{name} p_value", pr["p_value"], ex["p_value"])
            cmp(f"{name} significant", pr["significant"], ex["significant"], exact=True)
            cmp(f"{name} tvd", pr["tvd"], 0.5 * np.abs(ta / na - tb / nb).sum(axis=1))
            cmp(f"{name} upper bound", pr["tvd_upper_bound"], bd["upper"])
            cmp(f"{name} lower bound", pr["tvd_lower_bound"], bd["lower"])
    return out


def _recorded_categories(fam: Mapping[str, Any]) -> list:
    if fam["hypothesis_kind"] == "pairwise_homogeneity":
        return [c for comp in fam["records"]["comparisons"].values() for c in comp["pairs"]["category"]]
    return list(fam["records"]["category"])


def verify_audit_result(audit: Any) -> Dict[str, Any]:
    """Re-verify a sealed AuditResult and return its payload.

    Raises AuditIntegrityError unless: the object was produced by
    class_stratified_audit and its seal matches; the schema, method and
    decision-rule versions are current; the configuration records a valid role
    for each of R1..D (and E iff E is present), the allocation covers exactly
    the confirmatory families, and exactly those families are present, each
    with its recorded role, kind, tolerance key, configured tolerance and
    allocated (confirmatory) or exploratory alpha; every
    family's hypothesis count equals the count implied by the design and its
    per-hypothesis alpha equals family alpha / count; every recorded sample
    size equals the design's partition or class-stratum size; every exact
    p-value, detection flag, effect and bound equals the value recomputed from
    the stored integer counts; and every category and family outcome equals
    the one re-derived from those values.
    """
    if not isinstance(audit, AuditResult) or audit._token is not _SEAL_TOKEN:
        raise AuditIntegrityError("not an AuditResult produced by class_stratified_audit.")
    if _sha256_bytes(audit.sealed_json) != audit.seal_sha256:
        raise AuditIntegrityError("audit seal does not match its content.")
    p = audit.payload()
    probs = []
    for key, want in (("schema_version", AUDIT_SCHEMA_VERSION), ("method_version", D3_METHOD_VERSION),
                      ("decision_rule_version", DECISION_RULE_VERSION)):
        if p.get(key) != want:
            probs.append(f"{key} is {p.get(key)!r}, expected {want!r}")
    des, cfg = p["design"], p["configuration"]
    names = des["partition_order"]
    sizes, cc = des["partition_sizes"], des["class_counts"]
    if set(sizes) != set(names) or set(cc) != set(names) or set(p["input_provenance"]) != set(names):
        probs.append("design and provenance do not cover exactly the partitions")
    for n in names:
        if cc[n]["positive"] + cc[n]["negative"] != sizes[n] or min(cc[n].values()) < 1:
            probs.append(f"class counts of {n} are inconsistent with its size")
    want_counts = expected_hypothesis_counts(len(names), des["feature_dimension"])
    fams = {}
    for f in p["families"]:
        if f["family_id"] in fams:
            probs.append(f"family {f['family_id']} appears twice")
        fams[f["family_id"]] = f
    roles = cfg.get("family_roles")
    if (not isinstance(roles, Mapping) or set(roles) - {"E"} != set(CLASS_STRATIFIED_CONFIRMATORY_FAMILIES)
            or any(v not in ("confirmatory", "exploratory") for v in roles.values())
            or roles.get("E", "exploratory") != "exploratory"):
        probs.append("configuration does not record a valid role for every family")
        roles = {}
    if set(cfg["family_alpha_allocation"]) != {f for f, r in roles.items() if r == "confirmatory"}:
        probs.append("allocation does not cover exactly the confirmatory families")
    extra = set(fams) - set(roles)
    missing = set(roles) - set(fams)
    if extra or missing:
        probs.append(f"family set wrong: missing {sorted(missing)}, unexpected {sorted(extra)}")
    for fid, f in fams.items():
        if fid not in FAMILY_KIND or fid not in roles:
            continue
        role = roles[fid]
        if f["role"] != role or f["hypothesis_kind"] != FAMILY_KIND[fid]:
            probs.append(f"family {fid} role/kind is {f['role']}/{f['hypothesis_kind']}")
        if f["hypothesis_count"] != want_counts[fid]:
            probs.append(f"family {fid} has {f['hypothesis_count']} hypotheses, design implies {want_counts[fid]}")
        alpha = cfg["family_alpha_allocation"].get(fid) if role == "confirmatory" else cfg["exploratory_alpha"]
        if f["familywise_alpha"] != alpha or not _math.isclose(
                f["per_hypothesis_alpha"], f["familywise_alpha"] / max(1, f["hypothesis_count"]), rel_tol=1e-12):
            probs.append(f"family {fid} alpha is not the allocated alpha")
        if fid != "E":
            if f["tolerance_key"] != FAMILY_TOLERANCE_KEY[fid] or f["tolerance"] != cfg["tolerances"][FAMILY_TOLERANCE_KEY[fid]]:
                probs.append(f"family {fid} tolerance differs from the configuration")
        try:
            probs.extend(_recompute_mismatches(f))
            recat = _reclassify(f)
        except (KeyError, TypeError, ValueError) as exc:
            probs.append(f"family {fid} records are malformed: {exc}")
            continue
        if len(recat) != f["hypothesis_count"] or recat != _recorded_categories(f):
            probs.append(f"family {fid} categories do not follow from its records")
        derived = _combine_outcomes([CATEGORY_OUTCOME[c] for c in recat]) if role == "confirmatory" else "NOT_GATED"
        if f["outcome"] != derived:
            probs.append(f"family {fid} outcome {f['outcome']} != re-derived {derived}")
        # recorded sample sizes must be the design's
        try:
            if fid == "R1" and f["records"]["trials"] != [sizes[n] for n in names]:
                probs.append("R1 trials differ from partition sizes")
            if fid == "R2" and f["records"]["trials"] != [cc[n]["positive" if lab else "negative"]
                                                           for n in names for lab in (1, 0)
                                                           for _ in range(des["feature_dimension"])]:
                probs.append("R2 trials differ from class counts")
            if fid == "D" and (f["records"]["n1"] != [cc[n]["positive"] for n in names for _ in range(des["feature_dimension"])]
                               or f["records"]["n2"] != [cc[n]["negative"] for n in names for _ in range(des["feature_dimension"])]):
                probs.append("D sample sizes differ from class counts")
            if fid in ("A", "B", "C"):
                key = {"A": None, "B": "positive", "C": "negative"}[fid]
                for comp, rec in f["records"]["comparisons"].items():
                    a, b = comp.split("_", 1)[0], comp.split("_", 1)[1]
                    want = ((sizes[a], sizes[b]) if key is None else (cc[a][key], cc[b][key]))
                    got = (rec["sample_sizes"]["partition_a"], rec["sample_sizes"]["partition_b"])
                    if got != want or [sum(r) for r in rec["pairs"]["cell_counts_a"]] != [want[0]] * len(rec["pairs"]["cell_counts_a"]):
                        probs.append(f"{fid} {comp} sample sizes differ from the design")
                # canonical JSON sorts mapping keys, so compare as sets
                if sorted(f["records"]["comparisons"]) != sorted(des["comparisons"]) \
                        or len(des["comparisons"]) != len(set(des["comparisons"])):
                    probs.append(f"{fid} comparisons differ from the design")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            probs.append(f"family {fid} sizes unverifiable: {exc}")
    if probs:
        raise AuditIntegrityError("audit result rejected: " + "; ".join(probs))
    return p


# ---------------------------------------------------------------------------
# Calibration binding
# ---------------------------------------------------------------------------

#: Exactly these consistency checks must be present, each exactly True.
REQUIRED_CALIBRATION_CHECKS = frozenset({
    "S0_unit_suite", "S1_binomial_exact", "S2_complete_path", "S6_exact_clopper_pearson",
    "S7_bound_coverage", "S8_exact_fisher"})
#: These evidence sections must all be present ...
REQUIRED_CALIBRATION_SECTIONS = frozenset({
    "schema_version", "status", "run_id", "checks", "identity", "design", "guarantee",
    "S0_unit_suite", "S1_single_test_size", "S2_complete_path_null", "S6_exact_clopper_pearson",
    "S7_bound_coverage", "S8_exact_fisher"})
#: ... and no section outside REQUIRED | OPTIONAL may appear.
OPTIONAL_CALIBRATION_SECTIONS = frozenset({
    "diagnostics", "S3_complete_path_power", "S4_tvd_noise_floor_production", "S5_design_resolution",
    "validation_criteria", "claims", "limitations", "budget", "rng", "environment",
    "started_utc", "finished_utc", "elapsed_seconds", "operating_conditions"})
CALIBRATION_STATUS = "IMPLEMENTATION_CONSISTENT"
CALIBRATION_EVIDENCE_NAME = "d3_method_validity.json"
CALIBRATION_MANIFEST_NAME = "d3_method_validity_run_manifest.json"


@_dataclass(frozen=True)
class CalibrationBinding:
    """Token from verify_calibration_evidence(): a complete, internally consistent
    calibration run produced for this engine file, tool and test suite.

    It is evidence that the IMPLEMENTATION matches the exact methods it claims
    (numerical verification at the design's alphas, unit suite, coverage and
    consistency simulations). The statistical guarantees themselves are the
    theorems G1/G2; the binding does not authenticate the files.
    """
    run_id: str
    evidence_sha256: str
    manifest_sha256: str
    engine_sha256: str
    tool_sha256: str
    test_suite_sha256: str
    design_json: str
    _token: object = _field(default=None, repr=False, compare=False)

    def design(self) -> Dict[str, Any]:
        return _json.loads(self.design_json)

    def as_record(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "evidence_sha256": self.evidence_sha256,
                "manifest_sha256": self.manifest_sha256, "engine_sha256": self.engine_sha256,
                "tool_sha256": self.tool_sha256, "test_suite_sha256": self.test_suite_sha256,
                "design": self.design()}


def _alphas_used(design_alloc: Mapping[str, float], counts: Mapping[str, int]) -> Dict[str, list]:
    """Per-test alphas the exact components run at, by component."""
    out = {"binomial": [], "clopper_pearson": [], "fisher_cell": []}
    for f in design_alloc:
        ah = design_alloc[f] / counts[f]
        if f in ("R1", "R2"):
            out["binomial"].append(ah)
            out["clopper_pearson"].append(ah)
        elif f == "D":
            out["fisher_cell"].append(ah)
            out["clopper_pearson"].append(ah / 2)
        else:
            out["fisher_cell"].append(ah / 4)
            out["clopper_pearson"].append(ah / 8)
    return {k: sorted(set(v)) for k, v in out.items()}


def verify_calibration_evidence(evidence_dir: Any, *, tool_path: Any, test_suite_path: Any) -> CalibrationBinding:
    """Verify one calibration run's completeness and identity.

    Refuses (CalibrationRejected) unless: the manifest is COMPLETE and names the
    evidence file with its correct SHA-256 and the same run_id; the evidence
    has exactly REQUIRED_CALIBRATION_SECTIONS, schema CALIBRATION_SCHEMA_VERSION
    and status CALIBRATION_STATUS; its checks are exactly
    REQUIRED_CALIBRATION_CHECKS, each the boolean True; the S0 record shows
    exit code 0 for the bound test suite; every S6/S8 row is valid and the
    rows cover every per-test alpha the recorded design uses; every S7 row is
    valid; and the recorded engine, tool and test-suite hashes and the method,
    decision-rule, bound and detection versions equal the current ones.
    Compatibility with a particular audit is checked at decision time.
    """
    d = _Path(evidence_dir)
    man_p, ev_p = d / CALIBRATION_MANIFEST_NAME, d / CALIBRATION_EVIDENCE_NAME
    if not man_p.is_file() or not ev_p.is_file():
        raise CalibrationRejected(f"calibration evidence or manifest missing in {d}.")
    try:
        man = _json.loads(man_p.read_text())
        ev = _json.loads(ev_p.read_text())
    except (OSError, ValueError) as exc:
        raise CalibrationRejected(f"calibration evidence is not readable JSON: {exc}") from None
    if not isinstance(man, dict) or not isinstance(ev, dict):
        raise CalibrationRejected("calibration manifest and evidence must be JSON objects.")
    ev_sha = _sha256_file(ev_p)
    probs = []
    if man.get("run_status") != "COMPLETE":
        probs.append("manifest run_status is not COMPLETE")
    if (man.get("files") or {}).get(CALIBRATION_EVIDENCE_NAME) != ev_sha:
        probs.append("manifest evidence hash does not match the evidence file")
    if not ev.get("run_id") or man.get("run_id") != ev.get("run_id"):
        probs.append("manifest and evidence run_id differ or are missing")
    keys = set(ev)
    allowed = REQUIRED_CALIBRATION_SECTIONS | OPTIONAL_CALIBRATION_SECTIONS
    if not REQUIRED_CALIBRATION_SECTIONS <= keys or keys - allowed:
        probs.append(f"sections: missing {sorted(REQUIRED_CALIBRATION_SECTIONS - keys)}, "
                     f"unexpected {sorted(keys - allowed)}")
    if ev.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        probs.append(f"schema_version {ev.get('schema_version')!r} != {CALIBRATION_SCHEMA_VERSION!r}")
    if ev.get("status") != CALIBRATION_STATUS:
        probs.append(f"status is {ev.get('status')!r}, not {CALIBRATION_STATUS}")
    checks = ev.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CALIBRATION_CHECKS:
        got = set(checks) if isinstance(checks, dict) else set()
        probs.append(f"checks: missing {sorted(REQUIRED_CALIBRATION_CHECKS - got)}, "
                     f"unexpected {sorted(got - REQUIRED_CALIBRATION_CHECKS)}")
    elif not all(v is True for v in checks.values()):
        probs.append(f"checks not exactly True: {sorted(k for k, v in checks.items() if v is not True)}")
    ident = ev.get("identity") if isinstance(ev.get("identity"), dict) else {}
    expect = {"engine_sha256": engine_sha256(), "tool_sha256": _sha256_file(tool_path),
              "test_suite_sha256": _sha256_file(test_suite_path), "method_version": D3_METHOD_VERSION,
              "decision_rule_version": DECISION_RULE_VERSION, "bound_construction": BOUND_CONSTRUCTION,
              "detection_test": DETECTION_TEST}
    for k, v in expect.items():
        if ident.get(k) != v:
            probs.append(f"{k} recorded {ident.get(k)!r}, current {v!r}")
    s0 = ev.get("S0_unit_suite") if isinstance(ev.get("S0_unit_suite"), dict) else {}
    if s0.get("exit_code") != 0 or s0.get("suite_sha256") != expect["test_suite_sha256"]:
        probs.append("S0 does not record a passing run of the bound test suite")
    design = ev.get("design") if isinstance(ev.get("design"), dict) else {}
    try:
        g = _check_alpha(design["global_alpha"], "global_alpha")
        alloc = {str(k): _check_alpha(v, f"allocation {k}") for k, v in design["family_alpha_allocation"].items()}
        if not alloc or set(alloc) - set(CLASS_STRATIFIED_CONFIRMATORY_FAMILIES):
            probs.append("design allocation is not a non-empty set of R1..D families")
        joint = joint_alpha_problem(g, alloc)
        if joint:
            probs.append(f"design {joint}")
        sizes = {str(k): _check_count(v, "partition size") for k, v in design["partition_sizes"].items()}
        dim = _check_count(design["feature_dimension"], "feature_dimension")
        lo_f, hi_f = design["supported_class_fraction"]
        if not (0 < float(lo_f) <= 0.5 <= float(hi_f) < 1):
            probs.append("supported_class_fraction invalid")
        counts = expected_hypothesis_counts(len(sizes), dim)
        used = _alphas_used(alloc, counts)
        for sec, comp in (("S6_exact_clopper_pearson", "clopper_pearson"), ("S8_exact_fisher", "fisher_cell")):
            rows = (ev.get(sec) or {}).get("rows") or []
            if not rows or not all(r.get("valid") is True for r in rows):
                probs.append(f"{sec} has no rows or an invalid row")
            covered = {r["alpha"] for r in rows}
            if any(not any(_math.isclose(a, c, rel_tol=1e-12) for c in covered) for a in used[comp]):
                probs.append(f"{sec} does not cover every {comp} alpha the design uses")
        rows7 = (ev.get("S7_bound_coverage") or {}).get("results") or []
        if not rows7 or not all(r.get("valid") is True for r in rows7):
            probs.append("S7 has no rows or an invalid row")
    except (KeyError, TypeError, ValueError) as exc:
        probs.append(f"design malformed: {exc}")
    if probs:
        raise CalibrationRejected("calibration evidence rejected: " + "; ".join(probs))
    return CalibrationBinding(run_id=str(ev["run_id"]), evidence_sha256=ev_sha, manifest_sha256=_sha256_file(man_p),
                              engine_sha256=expect["engine_sha256"], tool_sha256=expect["tool_sha256"],
                              test_suite_sha256=expect["test_suite_sha256"],
                              design_json=_canonical(design).decode(), _token=_BINDING_TOKEN)


# ---------------------------------------------------------------------------
# Preregistration
# ---------------------------------------------------------------------------

@_dataclass(frozen=True)
class PreregistrationBinding:
    """Token from verify_preregistration(): an approved record, anchored to an
    externally supplied SHA-256 and to a git commit, whose bytes are held here.

    The engine verifies integrity, approval form, the anchor and the commit;
    it cannot verify that the approver is a person or that the tolerances are
    scientifically justified. Those are attested by the approver's signed
    statement and commit, and must be reviewed by people.
    """
    path: str
    record_bytes: bytes = _field(repr=False)
    record_sha256: str = ""
    commit: str = ""
    commit_author: str = ""
    commit_utc: str = ""
    _token: object = _field(default=None, repr=False, compare=False)

    def record(self) -> Dict[str, Any]:
        return _json.loads(self.record_bytes)


def validate_preregistration_content(r: Any) -> Dict[str, Any]:
    """Validate the content of a preregistration record (raises PreregistrationRejected)."""
    if not isinstance(r, Mapping):
        raise PreregistrationRejected("preregistration must be a JSON object.")
    required = {"schema_version", "approved", "approved_by", "approved_utc", "approval_statement",
                "data_examined_before_approval", "global_familywise_alpha", "family_alpha_allocation",
                "family_roles", "tolerances", "tolerance_justification"}
    keys = set(r)
    if keys != required:
        raise PreregistrationRejected(f"fields: missing {sorted(required - keys)}, unexpected {sorted(keys - required)}")
    if r["schema_version"] != PREREGISTRATION_SCHEMA_VERSION:
        raise PreregistrationRejected(f"schema_version must be {PREREGISTRATION_SCHEMA_VERSION!r}.")
    if PENDING_MARKER in _json.dumps(r):
        raise PreregistrationRejected(f"the record still contains an open attestation ({PENDING_MARKER!r}).")
    if r["approved"] is not True:
        raise PreregistrationRejected("preregistration is not approved (approved must be the boolean true).")
    if r["approval_statement"] != APPROVAL_STATEMENT:
        raise PreregistrationRejected("approval_statement is not the required statement verbatim.")
    for k in ("approved_by", "data_examined_before_approval"):
        if not isinstance(r[k], str) or len(r[k].strip()) < 3:
            raise PreregistrationRejected(f"{k} must be a substantive string.")
    try:
        t = _parse_utc(r["approved_utc"], "approved_utc")
        g = _check_alpha(r["global_familywise_alpha"], "global_familywise_alpha")
        alloc = {str(k): _check_alpha(v, f"allocation {k}") for k, v in r["family_alpha_allocation"].items()}
        tols = {str(k): _check_tolerance(v, f"tolerance {k}") for k, v in r["tolerances"].items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise PreregistrationRejected(str(exc)) from None
    if t > _datetime.datetime.now(_datetime.timezone.utc) + _datetime.timedelta(minutes=5):
        raise PreregistrationRejected("approved_utc is in the future.")
    roles = r["family_roles"]
    if not isinstance(roles, Mapping) or any(v not in ("confirmatory", "exploratory") for v in roles.values()):
        raise PreregistrationRejected("family_roles must map to 'confirmatory' or 'exploratory'.")
    if {k for k, v in roles.items() if v == "confirmatory"} != set(alloc):
        raise PreregistrationRejected("allocation must cover exactly the confirmatory families.")
    joint = joint_alpha_problem(g, alloc)
    if joint:
        raise PreregistrationRejected(f"family alphas exceed the joint global alpha ({joint}).")
    just = r["tolerance_justification"]
    if not isinstance(just, Mapping) or set(just) != set(tols):
        raise PreregistrationRejected("tolerance_justification must cover exactly the tolerance keys.")
    for k, v in tols.items():
        if v is not None and (not isinstance(just[k], str) or len(just[k].strip()) < 20):
            raise PreregistrationRejected(f"tolerance {k} is declared without a substantive justification.")
    return {"global_familywise_alpha": g, "family_alpha_allocation": alloc, "family_roles": dict(roles),
            "tolerances": tols, "approved_utc": t.isoformat()}


def _git(args, cwd) -> str:
    out = _subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise PreregistrationRejected(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def verify_preregistration(path: Any, *, expected_sha256: str) -> PreregistrationBinding:
    """Verify an approved, committed preregistration and bind its exact bytes.

    Refuses unless: the file's SHA-256 equals ``expected_sha256`` (an anchor
    the approver records outside the file); the content passes
    validate_preregistration_content; the file is tracked by git, has no
    staged or unstaged modification, and its committed blob at HEAD is
    byte-identical; and the approval time is not after the commit time.
    """
    p = _Path(path).resolve()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise PreregistrationRejected("expected_sha256 must be a 64-hex anchor supplied by the approver.")
    if not p.is_file():
        raise PreregistrationRejected(f"preregistration {p} is missing.")
    data = p.read_bytes()
    if _sha256_bytes(data) != expected_sha256:
        raise PreregistrationRejected("preregistration SHA-256 differs from the approver's anchor.")
    try:
        rec = _json.loads(data)
    except ValueError as exc:
        raise PreregistrationRejected(f"preregistration is not JSON: {exc}") from None
    content = validate_preregistration_content(rec)
    root = _Path(_git(["rev-parse", "--show-toplevel"], p.parent).strip())
    rel = p.relative_to(root.resolve()).as_posix()
    _git(["ls-files", "--error-unmatch", rel], root)
    if _git(["status", "--porcelain", "--", rel], root).strip():
        raise PreregistrationRejected("preregistration has uncommitted modifications.")
    blob = _subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(root), capture_output=True, timeout=30)
    if blob.returncode != 0 or blob.stdout != data:
        raise PreregistrationRejected("committed preregistration differs from the file.")
    log = _git(["log", "-1", "--format=%H%x00%an <%ae>%x00%cI", "--", rel], root).strip().split("\x00")
    if len(log) != 3:
        raise PreregistrationRejected("no commit records the preregistration.")
    commit_t = _parse_utc(log[2], "commit time")
    if _parse_utc(content["approved_utc"], "approved_utc") > commit_t + _datetime.timedelta(minutes=5):
        raise PreregistrationRejected("approval is dated after the commit that records it.")
    return PreregistrationBinding(path=str(p), record_bytes=data, record_sha256=expected_sha256,
                                  commit=log[0], commit_author=log[1], commit_utc=commit_t.isoformat(),
                                  _token=_PREREG_TOKEN)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def evaluate_confirmatory_decision(
    audit: AuditResult,
    *,
    calibration: Optional[CalibrationBinding] = None,
    preregistration: Optional[PreregistrationBinding] = None,
    expected_input_provenance: Optional[Mapping[str, Mapping[str, str]]] = None,
    invalid_inputs: Sequence[str] = (),
) -> Dict[str, Any]:
    """Four-way decision over a verified, sealed audit.

    The audit is re-verified (verify_audit_result raises AuditIntegrityError
    on any alteration or substitution). ``provisional_outcome`` combines the
    re-derived confirmatory family outcomes. The reported ``outcome`` equals
    it only when every gate passes; otherwise it is INCONCLUSIVE and every
    failing gate is listed:

      * expected_input_provenance (e.g. the D0 manifest's array hashes) is
        given and equals the audit's own input provenance;
      * calibration is a verified CalibrationBinding, still current for this
        engine file, whose design equals the audit's partition sizes and
        feature dimension, whose supported class fraction covers every
        class stratum, and whose allocation equals the audit's;
      * preregistration is a verified PreregistrationBinding, re-checked
        against its file now, approved and committed before the audit was
        created, whose alpha, allocation, roles and tolerances equal the
        audit's configuration.
    """
    p = verify_audit_result(audit)
    des, cfg = p["design"], p["configuration"]
    fams = p["families"]
    confirm = [f for f in fams if f["role"] == "confirmatory"]
    provisional = _combine_outcomes([f["outcome"] for f in confirm])
    alloc = cfg["family_alpha_allocation"]
    alpha_total = float(sum(alloc.values()))
    reasons = [f"invalid input: {r}" for r in invalid_inputs]

    if expected_input_provenance is None:
        reasons.append("no expected input provenance to bind the audit to its source data")
    elif {str(k): dict(v) for k, v in expected_input_provenance.items()} != p["input_provenance"]:
        reasons.append("audit input hashes differ from the expected source data")

    cal = calibration
    if not isinstance(cal, CalibrationBinding) or cal._token is not _BINDING_TOKEN:
        reasons.append("no verified CalibrationBinding")
    else:
        cd = cal.design()
        if cal.engine_sha256 != engine_sha256() or p["engine_sha256"] != cal.engine_sha256:
            reasons.append("calibration, audit and current engine file are not the same engine")
        if cd["partition_sizes"] != des["partition_sizes"]:
            reasons.append(f"calibrated partition sizes {cd['partition_sizes']} != audited {des['partition_sizes']}")
        if cd["feature_dimension"] != des["feature_dimension"]:
            reasons.append("calibrated feature dimension differs from the audit's")
        lo_f, hi_f = cd["supported_class_fraction"]
        for n, c in des["class_counts"].items():
            frac = c["positive"] / des["partition_sizes"][n]
            if not lo_f <= frac <= hi_f:
                reasons.append(f"class fraction {frac:.6f} of {n} is outside the calibrated range")
        if cd["family_alpha_allocation"] != alloc:
            reasons.append("calibrated allocation differs from the audit's")

    pr = preregistration
    if not isinstance(pr, PreregistrationBinding) or pr._token is not _PREREG_TOKEN:
        reasons.append("no verified PreregistrationBinding")
    else:
        try:
            if _sha256_bytes(pr.record_bytes) != pr.record_sha256:
                raise PreregistrationRejected("binding bytes were altered")
            again = verify_preregistration(pr.path, expected_sha256=pr.record_sha256)
            content = validate_preregistration_content(again.record())
        except PreregistrationRejected as exc:
            reasons.append(f"preregistration no longer verifies: {exc}")
        else:
            joint = joint_alpha_problem(content["global_familywise_alpha"], alloc)
            if joint:
                reasons.append(f"preregistered global alpha does not cover the audit jointly: {joint}")
            if content["family_alpha_allocation"] != alloc:
                reasons.append("preregistered allocation differs from the audit's")
            roles = {f["family_id"]: f["role"] for f in fams}
            if content["family_roles"] != roles:
                reasons.append(f"preregistered roles {content['family_roles']} != audit roles {roles}")
            if content["tolerances"] != cfg["tolerances"]:
                reasons.append("preregistered tolerances differ from the audit's")
            created = _parse_utc(p["created_utc"], "audit created_utc")
            if (_parse_utc(content["approved_utc"], "approved_utc") >= created
                    or _parse_utc(again.commit_utc, "commit time") >= created):
                reasons.append("preregistration was not approved and committed before the audit")
    outcome = provisional if not reasons else "INCONCLUSIVE"
    return {
        "method_version": D3_METHOD_VERSION, "decision_rule_version": DECISION_RULE_VERSION,
        "audit_seal_sha256": audit.seal_sha256, "outcome": outcome, "provisional_outcome": provisional,
        "outcome_overridden_because": reasons,
        "family_outcomes": {f["family_id"]: f["outcome"] for f in fams},
        "gating_families": [f["family_id"] for f in confirm],
        "non_gating_families": [f["family_id"] for f in fams if f["role"] != "confirmatory"],
        "guarantees": {
            "alpha_total": alpha_total,
            "G1_bounds": ("with probability >= 1 - alpha_total every reported effect bound of every confirmatory "
                          "hypothesis holds simultaneously (exact, finite-sample; rows i.i.d. within samples)"),
            "G2_detection": ("P(any false detection among confirmatory hypotheses) <= alpha_total (exact tests, "
                             "Bonferroni, arbitrary dependence; samples independent)"),
            "joint_error_bound": GUARANTEE_COUNT * alpha_total,
            "joint": ("P(any G1 or G2 error) <= joint_error_bound (union bound); the preregistered global "
                      "alpha must cover it"),
            "exploratory_families": ("computed at exploratory_alpha within each family, outside alpha_total; "
                                     "never gate and carry no confirmatory guarantee"),
            "outcome_basis": ("FAIL / PASS / CONDITIONAL_PASS tolerance statements rest on G1; the detection "
                              "statement of CONDITIONAL_PASS rests on G2"),
            "diagnostics": "chi-square and Monte Carlo p-values are recorded and never gate",
        },
        "calibration": cal.as_record() if isinstance(cal, CalibrationBinding) else None,
        "preregistration": ({"path": pr.path, "sha256": pr.record_sha256, "commit": pr.commit,
                             "commit_author": pr.commit_author} if isinstance(pr, PreregistrationBinding) else None),
        "input_provenance": p["input_provenance"], "design": des,
        "scope_boundary": ("a PASS bounds only the declared marginal and feature-pair effects, at the declared "
                           "tolerances, sample sizes and families, under independent rows; it is not a claim that "
                           "the complete joint distributions are equal"),
    }


def design_resolution(n_a: int, n_b: Optional[int] = None, *, per_hypothesis_alpha: float,
                      cell_probabilities: Sequence[float] = (0.25, 0.25, 0.25, 0.25)) -> Dict[str, float]:
    """Pre-data precision: the bound obtained when observed tables equal their expectation."""
    n_a = _check_count(n_a, "n_a")
    p = np.asarray(cell_probabilities, dtype=np.float64)
    if n_b is None:
        k = int(round(p[0] * n_a))
        lo, hi = clopper_pearson_bounds(np.int64(k), np.int64(n_a), alpha=per_hypothesis_alpha)
        return {"minimum_certifiable_tolerance": float(max(p[0] - lo, hi - p[0]))}
    n_b = _check_count(n_b, "n_b")
    ta, tb = np.rint(p * n_a).astype(np.int64), np.rint(p * n_b).astype(np.int64)
    return {"minimum_certifiable_tolerance": float(tvd_confidence_bounds(ta, tb, alpha=per_hypothesis_alpha)["upper"])}
