import json
from typing import Dict, Optional

import numpy as np
import speck as sp

# ==========================================================
# Generic Statistics Functions
# ==========================================================

def class_balance(
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute class distribution statistics.

    Parameters
    ----------
    labels : np.ndarray
        Binary class labels.

    Returns
    -------
    dict
        Class counts and ratios.
    """

    total = len(labels)

    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))

    return {

        "positive_samples": positive,

        "negative_samples": negative,

        "positive_ratio": positive / total,

        "negative_ratio": negative / total,
    }


def feature_statistics(
    dataset: np.ndarray,
) -> Dict[str, float]:
    """
    Compute global feature statistics.

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    dict
        Mean and variance of all feature values.
    """

    return {

        "feature_mean":
            float(np.mean(dataset)),

        "feature_variance":
            float(np.var(dataset)),
    }


def bit_frequency(
    dataset: np.ndarray,
) -> Dict[str, object]:
    """
    Compute bit frequency statistics for binary datasets.

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    dict
        Overall and per-feature bit densities.
    """

    per_feature = np.mean(dataset, axis=0)

    overall = float(np.mean(per_feature))

    return {

        "average_bit_density": overall,

        "per_feature_bit_density":
            per_feature.tolist(),
    }


def byte_histogram(
    dataset: np.ndarray,
) -> Dict[str, object]:
    """
    Compute normalized byte histogram.

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    dict
        Byte histogram.
    """

    histogram = np.bincount(
        dataset.flatten(),
        minlength=256,
    )

    histogram = histogram / histogram.sum()

    return {

        "byte_histogram":
            histogram.tolist(),
    }


def binary_entropy(
    dataset: np.ndarray,
) -> Dict[str, object]:
    """
    Compute average binary entropy.

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    dict
        Average and per-feature entropy.
    """

    probabilities = np.mean(
        dataset,
        axis=0,
    )

    epsilon = 1e-12

    probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon,
    )

    entropy = (

        -probabilities * np.log2(probabilities)

        -(1 - probabilities)
        * np.log2(1 - probabilities)

    )

    return {

        "average_binary_entropy":
            float(np.mean(entropy)),

        "per_feature_entropy":
            entropy.tolist(),
    }


def byte_entropy(
    dataset: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Shannon entropy for byte-valued datasets.

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    dict
        Byte entropy.
    """

    histogram = np.bincount(
        dataset.flatten(),
        minlength=256,
    )

    probabilities = histogram / histogram.sum()

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return {

        "byte_entropy":
            float(entropy),
    }


def detect_representation(
    dataset: np.ndarray,
) -> str:
    """
    Detect the feature representation of a dataset.

    Parameters
    ----------
    dataset : np.ndarray
        Input feature matrix.

    Returns
    -------
    str
        One of:
            "binary"
            "byte"
            "continuous"
    """

    unique_values = np.unique(dataset)

    # Binary representation (0/1 values)
    if np.all(np.isin(unique_values, [0, 1])):
        return "binary"

    # Byte representation (integer values in the range 0–255),
    # irrespective of whether they are stored as integers or floats.
    if (
        np.all(dataset >= 0)
        and np.all(dataset <= 255)
        and np.all(dataset == np.floor(dataset))
    ):
        return "byte"

    # Continuous representation
    return "continuous"
def summarize_partition(
    dataset: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    """
    Summarize a dataset partition.

    Parameters
    ----------
    dataset : np.ndarray

    labels : np.ndarray

    Returns
    -------
    dict
        Partition statistics.
    """

    representation = detect_representation(
        dataset
    )

    summary = {

        "samples":
            len(dataset),

        "representation":
            representation,

        "class_balance":
            class_balance(labels),

        "feature_statistics":
            feature_statistics(dataset),
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

    else:

        summary["minimum"] = float(
            np.min(dataset)
        )

        summary["maximum"] = float(
            np.max(dataset)
        )

    return summary


def compare_partitions(
    partition_a: Dict[str, object],
    partition_b: Dict[str, object],
) -> Dict[str, float]:
    """
    Compare two dataset partitions.

    Parameters
    ----------
    partition_a : dict

    partition_b : dict

    Returns
    -------
    dict
        Absolute statistical differences.
    """

    comparison = {

        "positive_ratio_difference":

            abs(

                partition_a["class_balance"]["positive_ratio"]

                -

                partition_b["class_balance"]["positive_ratio"]

            ),

        "feature_mean_difference":

            abs(

                partition_a["feature_statistics"]["feature_mean"]

                -

                partition_b["feature_statistics"]["feature_mean"]

            ),

        "feature_variance_difference":

            abs(

                partition_a["feature_statistics"]["feature_variance"]

                -

                partition_b["feature_statistics"]["feature_variance"]

            ),
    }

    if (

        "average_binary_entropy"

        in partition_a

        and

        "average_binary_entropy"

        in partition_b

    ):

        comparison["entropy_difference"] = abs(

            partition_a["average_binary_entropy"]

            -

            partition_b["average_binary_entropy"]

        )

    elif (

        "byte_entropy"

        in partition_a

        and

        "byte_entropy"

        in partition_b

    ):

        comparison["entropy_difference"] = abs(

            partition_a["byte_entropy"]

            -

            partition_b["byte_entropy"]

        )

    return comparison


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
    """

    results = {

        "train":

            summarize_partition(
                train,
                train_labels,
            ),

        "validation":

            summarize_partition(
                validation,
                validation_labels,
            ),
    }

    results["comparison"] = compare_partitions(

        results["train"],

        results["validation"],

    )

    if (

        test is not None

        and

        test_labels is not None

    ):

        results["test"] = summarize_partition(

            test,

            test_labels,

        )

    return results

# ==========================================================
# Reporting Functions
# ==========================================================

def print_report(
    results: Dict[str, object],
) -> None:
    """
    Print a formatted Dataset Integrity D3 report.

    Parameters
    ----------
    results : dict
        Results returned by audit_distribution().
    """

    train = results["train"]
    validation = results["validation"]

    print("=" * 50)
    print("Dataset Integrity Audit")
    print("D3 – Distribution Statistics")
    print("=" * 50)
    print()

    print(
        f"Representation      : "
        f"{train['representation'].capitalize()}"
    )

    print()

    # ------------------------------------------------------

    print("Training")
    print("-" * 40)

    print(
        f"Samples             : "
        f"{train['samples']}"
    )

    print(
        f"Positive Samples    : "
        f"{train['class_balance']['positive_samples']}"
    )

    print(
        f"Negative Samples    : "
        f"{train['class_balance']['negative_samples']}"
    )

    print(
        f"Positive Ratio      : "
        f"{train['class_balance']['positive_ratio']:.6f}"
    )

    print(
        f"Negative Ratio      : "
        f"{train['class_balance']['negative_ratio']:.6f}"
    )

    print(
        f"Feature Mean        : "
        f"{train['feature_statistics']['feature_mean']:.6f}"
    )

    print(
        f"Feature Variance    : "
        f"{train['feature_statistics']['feature_variance']:.6f}"
    )

    if train["representation"] == "binary":

        print(
            f"Bit Density         : "
            f"{train['average_bit_density']:.6f}"
        )

        print(
            f"Average Entropy     : "
            f"{train['average_binary_entropy']:.6f}"
        )

    elif train["representation"] == "byte":

        print(
            f"Byte Entropy        : "
            f"{train['byte_entropy']:.6f}"
        )

    print()

    # ------------------------------------------------------

    print("Validation")
    print("-" * 40)

    print(
        f"Samples             : "
        f"{validation['samples']}"
    )

    print(
        f"Positive Samples    : "
        f"{validation['class_balance']['positive_samples']}"
    )

    print(
        f"Negative Samples    : "
        f"{validation['class_balance']['negative_samples']}"
    )

    print(
        f"Positive Ratio      : "
        f"{validation['class_balance']['positive_ratio']:.6f}"
    )

    print(
        f"Negative Ratio      : "
        f"{validation['class_balance']['negative_ratio']:.6f}"
    )

    print(
        f"Feature Mean        : "
        f"{validation['feature_statistics']['feature_mean']:.6f}"
    )

    print(
        f"Feature Variance    : "
        f"{validation['feature_statistics']['feature_variance']:.6f}"
    )

    if validation["representation"] == "binary":

        print(
            f"Bit Density         : "
            f"{validation['average_bit_density']:.6f}"
        )

        print(
            f"Average Entropy     : "
            f"{validation['average_binary_entropy']:.6f}"
        )

    elif validation["representation"] == "byte":

        print(
            f"Byte Entropy        : "
            f"{validation['byte_entropy']:.6f}"
        )

    print()

    # ------------------------------------------------------

    print("Consistency")
    print("-" * 40)

    print(
        f"Class Ratio Diff    : "
        f"{results['comparison']['positive_ratio_difference']:.6f}"
    )

    print(
        f"Feature Mean Diff   : "
        f"{results['comparison']['feature_mean_difference']:.6f}"
    )

    print(
        f"Variance Diff       : "
        f"{results['comparison']['feature_variance_difference']:.6f}"
    )

    if "entropy_difference" in results["comparison"]:

        print(
            f"Entropy Diff        : "
            f"{results['comparison']['entropy_difference']:.6f}"
        )


def save_json(
    results: Dict[str, object],
    filename: str = "d3_distribution_statistics.json",
) -> None:
    """
    Save audit statistics as a JSON file.

    Parameters
    ----------
    results : dict
        Results returned by audit_distribution().

    filename : str
        Output JSON filename.
    """

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print()

    print(
        f"Statistics saved to: {filename}"
    )

# ==========================================================
# Gohr Dataset Reconstruction
# ==========================================================

def generate_gohr_training_data(
    num_rounds: int = 5,
):
    """
    Reconstruct the datasets used by Gohr's training pipeline.

    This wrapper faithfully reproduces the dataset generation
    performed inside train_speck_distinguisher().

    Parameters
    ----------
    num_rounds : int
        Number of Speck rounds.

    Returns
    -------
    tuple
        (
            X_train,
            Y_train,
            X_validation,
            Y_validation,
        )
    """

    X_train, Y_train = sp.make_train_data(
        10**7,
        num_rounds,
    )

    X_validation, Y_validation = sp.make_train_data(
        10**6,
        num_rounds,
    )

    return (
        X_train,
        Y_train,
        X_validation,
        Y_validation,
    )

def main():

    X_train, Y_train, X_validation, Y_validation = (
        generate_gohr_training_data()
    )

    results = audit_distribution(
        train=X_train,
        train_labels=Y_train,
        validation=X_validation,
        validation_labels=Y_validation,
    )

    print_report(results)

    save_json(results)


if __name__ == "__main__":
    main()