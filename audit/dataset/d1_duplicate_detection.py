import hashlib
from typing import Dict, Optional, Set

import numpy as np

import speck as sp

# ==========================================================
# Generic Audit Functions
# ==========================================================

def sample_hash(sample: np.ndarray) -> str:
    """
    Compute a SHA-256 fingerprint for a single dataset sample.

    Parameters
    ----------
    sample : np.ndarray
        One sample represented as a NumPy array.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(
        np.ascontiguousarray(sample).tobytes()
    ).hexdigest()


def partition_duplicates(dataset: np.ndarray) -> int:
    """
    Count duplicate samples within a single dataset partition.

    Parameters
    ----------
    dataset : np.ndarray

    Returns
    -------
    int
        Number of duplicate samples.
    """

    seen: Set[str] = set()
    duplicates = 0

    for sample in dataset:

        fingerprint = sample_hash(sample)

        if fingerprint in seen:
            duplicates += 1
        else:
            seen.add(fingerprint)

    return duplicates


def partition_overlap(
    dataset_a: np.ndarray,
    dataset_b: np.ndarray,
) -> int:
    """
    Count overlapping samples between two dataset partitions.

    Parameters
    ----------
    dataset_a : np.ndarray

    dataset_b : np.ndarray

    Returns
    -------
    int
        Number of shared samples.
    """

    hashes_a = {
        sample_hash(sample)
        for sample in dataset_a
    }

    hashes_b = {
        sample_hash(sample)
        for sample in dataset_b
    }

    return len(hashes_a.intersection(hashes_b))


def audit_duplicates(
    train: np.ndarray,
    validation: np.ndarray,
    test: Optional[np.ndarray] = None,
) -> Dict[str, Optional[int]]:
    """
    Perform Dataset Integrity Audit D1.

    Parameters
    ----------
    train : np.ndarray
        Training partition.

    validation : np.ndarray
        Validation partition.

    test : np.ndarray, optional
        Test partition. If omitted, only train/validation
        statistics are computed.

    Returns
    -------
    dict
        Dictionary containing duplicate and overlap statistics.
    """

    results = {

        "train_duplicates":
            partition_duplicates(train),

        "validation_duplicates":
            partition_duplicates(validation),

        "train_validation_overlap":
            partition_overlap(train, validation),

        "test_duplicates": None,

        "train_test_overlap": None,

        "validation_test_overlap": None,
    }

    if test is not None:

        results["test_duplicates"] = (
            partition_duplicates(test)
        )

        results["train_test_overlap"] = (
            partition_overlap(train, test)
        )

        results["validation_test_overlap"] = (
            partition_overlap(validation, test)
        )

    return results


def print_report(
    results: Dict[str, Optional[int]]
) -> None:
    """
    Print a formatted Dataset Integrity D1 report.
    """

    print("=" * 50)
    print("Dataset Integrity Audit")
    print("D1 – Duplicate Sample Detection")
    print("=" * 50)
    print()
    print(
        f"Training duplicates          : "
        f"{results['train_duplicates']}"
    )

    print(
        f"Validation duplicates        : "
        f"{results['validation_duplicates']}"
    )

    if results["test_duplicates"] is not None:

        print(
            f"Test duplicates              : "
            f"{results['test_duplicates']}"
        )

    print()

    print(
        f"Train ↔ Validation overlap   : "
        f"{results['train_validation_overlap']}"
    )

    if results["train_test_overlap"] is not None:

        print(
            f"Train ↔ Test overlap         : "
            f"{results['train_test_overlap']}"
        )

        print(
            f"Validation ↔ Test overlap    : "
            f"{results['validation_test_overlap']}"
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
    #X_train, Y_train = sp.make_train_data(
    #    10000,
    #    num_rounds,
    #)

    #X_validation, Y_validation = sp.make_train_data(
    #    1000,
    #    num_rounds,
    #)
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

# ==========================================================
# Main
# ==========================================================

def main():
    """
    Execute Dataset Integrity Audit D1 on Gohr's reconstructed
    training pipeline.
    """

    # Reconstruct Gohr's datasets
    X_train, Y_train, X_val, Y_val = generate_gohr_training_data()
    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    # Perform duplicate and overlap audit
    results = audit_duplicates(
        train=X_train,
        validation=X_val,
    )

    # Display audit report
    print_report(results)


if __name__ == "__main__":
    main()