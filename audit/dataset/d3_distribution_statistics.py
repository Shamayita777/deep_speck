"""
Dataset Integrity Audit D3
Distribution Statistics and Partition Consistency.

D3 characterizes the empirical distribution of each supplied
dataset partition and reports prespecified partition-consistency
statistics.

Scientific scope
----------------
D3 answers descriptive questions such as:

    - What is the empirical class balance?
    - What are the global feature mean and variance?
    - For binary representations, what are the marginal bit
      densities and binary entropies?
    - For byte representations, what are the byte frequencies
      and Shannon entropy?
    - How different are these quantities between partitions?

D3 does NOT establish:

    - equality of the full train/validation/test distributions;
    - statistical independence;
    - absence of near duplicates;
    - absence of metadata leakage;
    - cryptographic security;
    - correctness of the underlying generator;
    - absence of higher-order dependencies.

Those questions require other audits or analyses.

Design principles
-----------------
1. This module is dataset-agnostic.
2. It does not import a dataset generator.
3. It operates only on arrays explicitly supplied by the caller.
4. All audited partitions are summarized.
5. All supplied partition pairs are compared.
6. Binary per-feature marginal differences are reported explicitly.
7. Aggregate statistics are not treated as proof of distributional
   equality.
8. Representation detection is validated before representation-
   specific statistics are computed.
9. Integer-valued floating-point arrays are handled safely.
10. No arbitrary PASS/FAIL threshold is imposed on descriptive
    statistics.

For binary data, the expected representation is an array whose
entries are exactly 0 and 1.

For byte data, the expected representation is an integer-valued
array with entries in [0, 255].
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .utils import detect_representation


# ============================================================
# Validation helpers
# ============================================================

def _validate_dataset(dataset: np.ndarray) -> np.ndarray:
    """
    Validate and normalize a dataset array.

    A dataset must have at least one dimension and at least one
    sample.
    """

    dataset = np.asarray(dataset)

    if dataset.ndim < 1:
        raise ValueError(
            "Dataset must contain at least one dimension."
        )

    if dataset.shape[0] == 0:
        raise ValueError(
            "Dataset must contain at least one sample."
        )

    return dataset


def _validate_labels(
    labels: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Validate labels against the dataset sample count.
    """

    labels = np.asarray(labels)

    if labels.ndim == 0:
        raise ValueError(
            "Labels must contain a sample dimension."
        )

    if len(labels) != n_samples:
        raise ValueError(
            "Number of labels must equal number of samples."
        )

    return labels


def _validate_binary_dataset(
    dataset: np.ndarray,
) -> np.ndarray:
    """
    Validate that a dataset is genuinely binary.

    Returns
    -------
    np.ndarray
        Dataset converted to float64 for numerical statistics.

    Raises
    ------
    ValueError
        If entries are not exactly 0 or 1.
    """

    dataset = _validate_dataset(dataset)

    if not np.all(
        (dataset == 0) | (dataset == 1)
    ):
        raise ValueError(
            "Binary statistics require every dataset entry "
            "to be exactly 0 or 1."
        )

    return dataset.astype(np.float64, copy=False)


def _validate_byte_dataset(
    dataset: np.ndarray,
) -> np.ndarray:
    """
    Validate that a dataset contains integer-valued bytes.

    Integer-valued floating-point arrays are accepted after
    explicit validation and conversion to uint8. This avoids
    passing floating-point arrays directly to np.bincount().
    """

    dataset = _validate_dataset(dataset)

    if not np.issubdtype(
        dataset.dtype,
        np.integer,
    ):
        if not np.issubdtype(
            dataset.dtype,
            np.floating,
        ):
            raise ValueError(
                "Byte representation must use integer-valued "
                "or floating-point data."
            )

        if not np.all(
            np.isfinite(dataset)
        ):
            raise ValueError(
                "Byte-valued floating-point data must be finite."
            )

        if not np.all(
            dataset == np.floor(dataset)
        ):
            raise ValueError(
                "Byte-valued floating-point data must contain "
                "only integer-valued entries."
            )

    minimum = np.min(dataset)
    maximum = np.max(dataset)

    if minimum < 0 or maximum > 255:
        raise ValueError(
            "Byte-valued data must lie in the inclusive range "
            "[0, 255]."
        )

    return dataset.astype(
        np.uint8,
        copy=False,
    )


# ============================================================
# Class balance
# ============================================================

def class_balance(
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary class counts and proportions.

    The function explicitly requires labels to be binary.
    """

    labels = np.asarray(labels)

    if labels.ndim == 0:
        raise ValueError(
            "Labels must contain a sample dimension."
        )

    total = len(labels)

    if total == 0:
        raise ValueError(
            "Labels must contain at least one sample."
        )

    unique_labels = np.unique(labels)

    if not np.all(
        np.isin(unique_labels, [0, 1])
    ):
        raise ValueError(
            "D3 class-balance statistics require binary "
            "labels encoded as 0 and 1."
        )

    positive = int(
        np.sum(labels == 1)
    )

    negative = int(
        np.sum(labels == 0)
    )

    return {
        "positive_samples": positive,
        "negative_samples": negative,
        "positive_ratio": positive / total,
        "negative_ratio": negative / total,
    }


# ============================================================
# Global feature statistics
# ============================================================

def feature_statistics(
    dataset: np.ndarray,
) -> Dict[str, float]:
    """
    Compute global mean and variance over all feature entries.
    """

    dataset = _validate_dataset(dataset)

    return {
        "feature_mean": float(
            np.mean(dataset, dtype=np.float64)
        ),
        "feature_variance": float(
            np.var(dataset, dtype=np.float64)
        ),
    }


# ============================================================
# Binary marginal statistics
# ============================================================

def bit_frequency(
    dataset: np.ndarray,
) -> Dict[str, object]:
    """
    Compute marginal bit densities.

    For binary data, the j-th entry of
    per_feature_bit_density is the empirical probability that
    feature j equals one.

    The function therefore characterizes first-order marginal
    structure only; it does not characterize dependencies among
    bits.
    """

    dataset = _validate_binary_dataset(dataset)

    per_feature = np.mean(
        dataset,
        axis=0,
        dtype=np.float64,
    )

    overall = float(
        np.mean(per_feature)
    )

    return {
        "average_bit_density": overall,
        "minimum_bit_density": float(
            np.min(per_feature)
        ),
        "maximum_bit_density": float(
            np.max(per_feature)
        ),
        "per_feature_bit_density": (
            per_feature.tolist()
        ),
    }


# ============================================================
# Binary entropy
# ============================================================

def binary_entropy(
    dataset: np.ndarray,
) -> Dict[str, object]:
    """
    Compute marginal binary entropy for every feature.

    Entropy is measured in bits.

    This is marginal entropy of individual features, not joint
    entropy of the complete representation.
    """

    dataset = _validate_binary_dataset(dataset)

    probabilities = np.mean(
        dataset,
        axis=0,
        dtype=np.float64,
    )

    epsilon = np.finfo(
        np.float64
    ).eps

    probabilities = np.clip(
        probabilities,
        epsilon,
        1.0 - epsilon,
    )

    entropy = (
        -probabilities * np.log2(probabilities)
        -(
            1.0 - probabilities
        ) * np.log2(
            1.0 - probabilities
        )
    )

    return {
        "average_binary_entropy": float(
            np.mean(entropy)
        ),
        "minimum_binary_entropy": float(
            np.min(entropy)
        ),
        "maximum_binary_entropy": float(
            np.max(entropy)
        ),
        "per_feature_entropy": (
            entropy.tolist()
        ),
    }


# ============================================================
# Byte histogram
# ============================================================

def byte_histogram(
    dataset: np.ndarray,
) -> Dict[str, object]:
    """
    Compute a normalized histogram over byte values 0..255.
    """

    dataset = _validate_byte_dataset(dataset)

    values = dataset.reshape(-1)

    histogram = np.bincount(
        values,
        minlength=256,
    ).astype(
        np.float64
    )

    total = histogram.sum()

    if total == 0:
        raise ValueError(
            "Cannot compute byte histogram for empty data."
        )

    probabilities = histogram / total

    return {
        "byte_histogram": (
            probabilities.tolist()
        ),
    }


# ============================================================
# Byte entropy
# ============================================================

def byte_entropy(
    dataset: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Shannon entropy of the empirical byte distribution.

    Entropy is measured in bits per byte.
    """

    histogram_result = byte_histogram(
        dataset
    )

    probabilities = np.asarray(
        histogram_result["byte_histogram"],
        dtype=np.float64,
    )

    nonzero = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        nonzero * np.log2(nonzero)
    )

    return {
        "byte_entropy": float(entropy),
    }


# ============================================================
# Partition summarization
# ============================================================

def summarize_partition(
    dataset: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    """
    Produce the complete D3 summary for one partition.
    """

    dataset = _validate_dataset(dataset)

    labels = _validate_labels(
        labels,
        len(dataset),
    )

    representation = detect_representation(
        dataset
    )

    if representation not in {
        "binary",
        "byte",
    }:
        raise ValueError(
            "D3 currently supports binary and byte-valued "
            "representations. Detected representation: "
            f"{representation!r}"
        )

    summary: Dict[str, object] = {
        "samples": int(len(dataset)),
        "feature_shape": list(
            dataset.shape[1:]
        ),
        "dtype": str(dataset.dtype),
        "representation": representation,
        "class_balance": class_balance(
            labels
        ),
        "feature_statistics": feature_statistics(
            dataset
        ),
    }

    if representation == "binary":

        summary.update(
            bit_frequency(dataset)
        )

        summary.update(
            binary_entropy(dataset)
        )

    elif representation == "byte":

        summary.update(
            byte_histogram(dataset)
        )

        summary.update(
            byte_entropy(dataset)
        )

    return summary


# ============================================================
# Binary partition comparison
# ============================================================

def _compare_binary_partitions(
    partition_a: Mapping[str, object],
    partition_b: Mapping[str, object],
) -> Dict[str, Any]:
    """
    Compare binary marginal statistics.

    The comparison explicitly reports the entire per-feature
    density difference vector, as well as its mean, maximum,
    RMS, and quantiles.

    These are effect-size diagnostics, not hypothesis tests.
    """

    density_a = np.asarray(
        partition_a["per_feature_bit_density"],
        dtype=np.float64,
    )

    density_b = np.asarray(
        partition_b["per_feature_bit_density"],
        dtype=np.float64,
    )

    if density_a.shape != density_b.shape:
        raise ValueError(
            "Binary partitions have incompatible feature shapes."
        )

    absolute_difference = np.abs(
        density_a - density_b
    )

    return {
        "mean_absolute_bit_density_difference": float(
            np.mean(absolute_difference)
        ),
        "rms_bit_density_difference": float(
            np.sqrt(
                np.mean(
                    absolute_difference ** 2
                )
            )
        ),
        "maximum_absolute_bit_density_difference": float(
            np.max(absolute_difference)
        ),
        "bit_density_difference_quantiles": {
            "q50": float(
                np.quantile(
                    absolute_difference,
                    0.50,
                )
            ),
            "q90": float(
                np.quantile(
                    absolute_difference,
                    0.90,
                )
            ),
            "q95": float(
                np.quantile(
                    absolute_difference,
                    0.95,
                )
            ),
            "q99": float(
                np.quantile(
                    absolute_difference,
                    0.99,
                )
            ),
        },
        "per_feature_absolute_bit_density_difference": (
            absolute_difference.tolist()
        ),
    }


# ============================================================
# Byte partition comparison
# ============================================================

def _compare_byte_partitions(
    partition_a: Mapping[str, object],
    partition_b: Mapping[str, object],
) -> Dict[str, Any]:
    """
    Compare empirical byte distributions.

    Reports total variation distance in addition to aggregate
    entropy difference.
    """

    histogram_a = np.asarray(
        partition_a["byte_histogram"],
        dtype=np.float64,
    )

    histogram_b = np.asarray(
        partition_b["byte_histogram"],
        dtype=np.float64,
    )

    if histogram_a.shape != histogram_b.shape:
        raise ValueError(
            "Byte histograms have incompatible shapes."
        )

    total_variation = 0.5 * np.sum(
        np.abs(
            histogram_a - histogram_b
        )
    )

    return {
        "byte_total_variation_distance": float(
            total_variation
        ),
        "maximum_absolute_byte_probability_difference": float(
            np.max(
                np.abs(
                    histogram_a - histogram_b
                )
            )
        ),
        "per_byte_absolute_probability_difference": (
            np.abs(
                histogram_a - histogram_b
            ).tolist()
        ),
    }


# ============================================================
# General partition comparison
# ============================================================

def compare_partitions(
    partition_a: Mapping[str, object],
    partition_b: Mapping[str, object],
) -> Dict[str, Any]:
    """
    Compare two summarized partitions.

    Aggregate differences are always reported.

    Representation-specific first-order distributional
    diagnostics are also reported when both partitions use the
    same supported representation.
    """

    if (
        partition_a["representation"]
        != partition_b["representation"]
    ):
        raise ValueError(
            "Cannot compare partitions with different "
            "representations."
        )

    class_a = partition_a[
        "class_balance"
    ]

    class_b = partition_b[
        "class_balance"
    ]

    stats_a = partition_a[
        "feature_statistics"
    ]

    stats_b = partition_b[
        "feature_statistics"
    ]

    comparison: Dict[str, Any] = {
        "positive_ratio_difference": abs(
            class_a["positive_ratio"]
            - class_b["positive_ratio"]
        ),
        "negative_ratio_difference": abs(
            class_a["negative_ratio"]
            - class_b["negative_ratio"]
        ),
        "feature_mean_difference": abs(
            stats_a["feature_mean"]
            - stats_b["feature_mean"]
        ),
        "feature_variance_difference": abs(
            stats_a["feature_variance"]
            - stats_b["feature_variance"]
        ),
    }

    if (
        partition_a["representation"]
        == "binary"
    ):
        comparison.update(
            _compare_binary_partitions(
                partition_a,
                partition_b,
            )
        )

        comparison[
            "average_binary_entropy_difference"
        ] = abs(
            partition_a[
                "average_binary_entropy"
            ]
            - partition_b[
                "average_binary_entropy"
            ]
        )

    elif (
        partition_a["representation"]
        == "byte"
    ):
        comparison.update(
            _compare_byte_partitions(
                partition_a,
                partition_b,
            )
        )

        comparison[
            "byte_entropy_difference"
        ] = abs(
            partition_a["byte_entropy"]
            - partition_b["byte_entropy"]
        )

    return comparison


# ============================================================
# Full D3 audit
# ============================================================

def audit_distribution(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    test: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Perform Dataset Integrity Audit D3.

    Every supplied partition is summarized.

    Every pair among the supplied partitions is compared.

    No PASS/FAIL decision is imposed because D3 is a descriptive
    and consistency audit rather than a formal proof of
    distributional equality.
    """

    train = _validate_dataset(train)
    validation = _validate_dataset(validation)

    train_labels = _validate_labels(
        train_labels,
        len(train),
    )

    validation_labels = _validate_labels(
        validation_labels,
        len(validation),
    )

    results: Dict[str, object] = {
        "audit_id": "D3",
        "audit_name": (
            "Distribution Statistics and "
            "Partition Consistency"
        ),
        "scientific_scope": (
            "Descriptive characterization and effect-size "
            "comparison of supplied dataset partitions."
        ),
        "partitions": {},
        "comparisons": {},
        "interpretation": {
            "decision_status": (
                "DESCRIPTIVE_ONLY"
            ),
            "warning": (
                "Aggregate agreement does not establish equality "
                "of the full underlying distributions."
            ),
        },
    }

    results["partitions"]["train"] = (
        summarize_partition(
            train,
            train_labels,
        )
    )

    results["partitions"]["validation"] = (
        summarize_partition(
            validation,
            validation_labels,
        )
    )

    if test is not None:
        if test_labels is None:
            raise ValueError(
                "test_labels must be supplied when test is supplied."
            )

        test = _validate_dataset(test)

        test_labels = _validate_labels(
            test_labels,
            len(test),
        )

        results["partitions"]["test"] = (
            summarize_partition(
                test,
                test_labels,
            )
        )

    partition_names = list(
        results["partitions"].keys()
    )

    for name_a, name_b in combinations(
        partition_names,
        2,
    ):
        results["comparisons"][
            f"{name_a}_{name_b}"
        ] = compare_partitions(
            results["partitions"][name_a],
            results["partitions"][name_b],
        )

    return results


# ============================================================
# Human-readable report
# ============================================================

def _print_partition(
    name: str,
    partition: Mapping[str, object],
) -> None:
    """
    Print one partition summary.
    """

    print(name.capitalize())
    print("-" * 56)

    print(
        f"Samples             : "
        f"{partition['samples']}"
    )

    print(
        f"Feature shape       : "
        f"{partition['feature_shape']}"
    )

    print(
        f"Dtype               : "
        f"{partition['dtype']}"
    )

    print(
        f"Representation      : "
        f"{partition['representation']}"
    )

    balance = partition[
        "class_balance"
    ]

    print(
        f"Positive samples    : "
        f"{balance['positive_samples']}"
    )

    print(
        f"Negative samples    : "
        f"{balance['negative_samples']}"
    )

    print(
        f"Positive ratio      : "
        f"{balance['positive_ratio']:.9f}"
    )

    print(
        f"Negative ratio      : "
        f"{balance['negative_ratio']:.9f}"
    )

    statistics = partition[
        "feature_statistics"
    ]

    print(
        f"Feature mean        : "
        f"{statistics['feature_mean']:.9f}"
    )

    print(
        f"Feature variance    : "
        f"{statistics['feature_variance']:.9f}"
    )

    if partition["representation"] == "binary":

        print(
            f"Average bit density: "
            f"{partition['average_bit_density']:.9f}"
        )

        print(
            f"Minimum bit density: "
            f"{partition['minimum_bit_density']:.9f}"
        )

        print(
            f"Maximum bit density: "
            f"{partition['maximum_bit_density']:.9f}"
        )

        print(
            f"Average bit entropy : "
            f"{partition['average_binary_entropy']:.9f}"
        )

        print(
            f"Minimum bit entropy : "
            f"{partition['minimum_binary_entropy']:.9f}"
        )

        print(
            f"Maximum bit entropy : "
            f"{partition['maximum_binary_entropy']:.9f}"
        )

    elif partition["representation"] == "byte":

        print(
            f"Byte entropy        : "
            f"{partition['byte_entropy']:.9f}"
        )


def _print_comparison(
    name: str,
    comparison: Mapping[str, Any],
) -> None:
    """
    Print one pairwise comparison.
    """

    print(name)
    print("-" * 56)

    print(
        f"Positive ratio diff : "
        f"{comparison['positive_ratio_difference']:.9g}"
    )

    print(
        f"Negative ratio diff : "
        f"{comparison['negative_ratio_difference']:.9g}"
    )

    print(
        f"Feature mean diff   : "
        f"{comparison['feature_mean_difference']:.9g}"
    )

    print(
        f"Feature variance diff: "
        f"{comparison['feature_variance_difference']:.9g}"
    )

    if (
        "mean_absolute_bit_density_difference"
        in comparison
    ):
        print(
            f"Mean |bit density diff| : "
            f"{comparison['mean_absolute_bit_density_difference']:.9g}"
        )

        print(
            f"RMS bit density diff    : "
            f"{comparison['rms_bit_density_difference']:.9g}"
        )

        print(
            f"Max |bit density diff|  : "
            f"{comparison['maximum_absolute_bit_density_difference']:.9g}"
        )

        print(
            f"Average entropy diff    : "
            f"{comparison['average_binary_entropy_difference']:.9g}"
        )

    if (
        "byte_total_variation_distance"
        in comparison
    ):
        print(
            f"Byte TVD             : "
            f"{comparison['byte_total_variation_distance']:.9g}"
        )

        print(
            f"Max byte probability diff: "
            f"{comparison['maximum_absolute_byte_probability_difference']:.9g}"
        )

        print(
            f"Byte entropy diff    : "
            f"{comparison['byte_entropy_difference']:.9g}"
        )


def print_report(
    results: Mapping[str, object],
) -> None:
    """
    Print the D3 audit report.
    """

    print("=" * 72)
    print(
        "Dataset Integrity Audit"
    )
    print(
        "D3 – Distribution Statistics and "
        "Partition Consistency"
    )
    print("=" * 72)
    print()

    for name, partition in (
        results["partitions"].items()
    ):
        _print_partition(
            name,
            partition,
        )
        print()

    print("=" * 72)
    print("Pairwise Partition Comparisons")
    print("=" * 72)
    print()

    for name, comparison in (
        results["comparisons"].items()
    ):
        _print_comparison(
            name,
            comparison,
        )
        print()

    print(
        "Interpretation       : DESCRIPTIVE_ONLY"
    )

    print(
        "Scientific warning   : "
        "These statistics do not establish equality "
        "of the full underlying distributions."
    )


# ============================================================
# JSON persistence
# ============================================================

def save_json(
    results: Mapping[str, object],
    filename: str,
) -> None:
    """
    Save D3 results as JSON.
    """

    import json

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        f"Statistics saved to: {filename}"
    )