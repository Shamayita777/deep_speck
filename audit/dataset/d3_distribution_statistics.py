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
