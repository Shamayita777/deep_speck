"""
Dataset Integrity Audit D2
Phase 1: Data Models

This module defines the core data structures used by the
Partition Integrity & Leakage Detection audit framework.

These classes are intentionally generic and contain no
cryptography-specific logic. They can represent datasets from
neural cryptanalysis, side-channel analysis, or other machine
learning domains.

Author: Shamayita Moitra
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from collections import Counter
import hashlib
import numpy as np
from .utils import detect_representation
import speck as sp
import json
# ==========================================================
# Metadata
# ==========================================================

@dataclass
class MetadataField:
    """
    Represents a single metadata field associated with a dataset
    partition.

    Examples
    --------
    Gohr:
        name = "plaintext_pairs"

    Side-channel:
        name = "power_traces"

    AES:
        name = "ivs"

    Parameters
    ----------
    name : str
        Human-readable metadata name.

    values : np.ndarray
        Metadata values. Shape depends on the dataset.

    description : str, optional
        Additional information about the metadata.
    """

    name: str
    values: np.ndarray
    description: Optional[str] = None

    @property
    def num_samples(self) -> int:
        """Return the number of metadata samples."""
        return len(self.values)

# ==========================================================
# Partition
# ==========================================================

@dataclass
class Partition:
    """
    Represents a dataset partition.

    A partition is typically one of

        - Training
        - Validation
        - Test

    but may represent any subset of a dataset.

    Parameters
    ----------
    name : str
        Partition name.

    features : np.ndarray
        Feature matrix.

    labels : np.ndarray
        Ground-truth labels.

    metadata : Dict[str, MetadataField]
        Optional metadata associated with the samples.
    """

    name: str

    features: np.ndarray
    labels: np.ndarray

    metadata: Dict[str, MetadataField] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        """Number of features."""
        return self.features.shape[1]

    def add_metadata(
        self,
        name: str,
        values: np.ndarray,
        description: Optional[str] = None,
    ) -> None:
        """
        Add a metadata field.

        Parameters
        ----------
        name
            Metadata field name.

        values
            Metadata values.

        description
            Optional description.
        """
        self.metadata[name] = MetadataField(
            name=name,
            values=values,
            description=description,
        )

    def get_metadata(self, name: str) -> MetadataField:
        """
        Retrieve a metadata field.

        Raises
        ------
        KeyError
            If the metadata field does not exist.
        """
        return self.metadata[name]

    def has_metadata(self, name: str) -> bool:
        """Return True if a metadata field exists."""
        return name in self.metadata

    def metadata_names(self) -> list[str]:
        """Return all metadata field names."""
        return list(self.metadata.keys())

    def summary(self) -> Dict[str, Any]:
        """
        Return a lightweight summary of the partition.
        """

        return {
            "partition": self.name,
            "samples": self.num_samples,
            "features": self.num_features,
            "metadata_fields": self.metadata_names(),
        }

# ==========================================================
# Dataset
# ==========================================================

@dataclass
class Dataset:
    """
    Collection of dataset partitions.

    A dataset may contain any number of partitions, such as
    training, validation, and test sets.
    """

    partitions: Dict[str, Partition] = field(default_factory=dict)

    def add_partition(self, partition: Partition) -> None:
        """Add a partition to the dataset."""
        self.partitions[partition.name] = partition

    def get_partition(self, name: str) -> Partition:
        """Retrieve a partition by name."""
        return self.partitions[name]

    def has_partition(self, name: str) -> bool:
        """Return True if the partition exists."""
        return name in self.partitions

    def partition_names(self) -> list[str]:
        """Return the names of all partitions."""
        return list(self.partitions.keys())

    @property
    def num_partitions(self) -> int:
        """Return the number of partitions."""
        return len(self.partitions)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the dataset."""
        return {
            "partitions": self.partition_names(),
            "num_partitions": self.num_partitions,
        }


#==========================================================
#Generic Utility Functions
#==========================================================

def sample_hash(sample: np.ndarray) -> str:
    """
    Compute a stable hash for a single sample.

    Parameters
    ----------
    sample : np.ndarray
        One feature vector.

    Returns
    -------
    str
        SHA-256 hash.
    """
    return hashlib.sha256(sample.tobytes()).hexdigest()

def partition_hashes(partition: Partition) -> list[str]:
    """
    Compute hashes for every feature vector in a partition.
    """

    return [
        sample_hash(sample)
        for sample in partition.features
    ]

def duplicate_statistics(hashes: list[str]) -> Dict[str, Any]:
    """
    Compute duplicate statistics for a collection of hashes.
    """

    counts = Counter(hashes)

    duplicate_groups = sum(
        1
        for count in counts.values()
        if count > 1
    )

    duplicate_samples = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    maximum_multiplicity = max(counts.values())

    total = len(hashes)

    unique = len(counts)

    collision_rate = duplicate_samples / total if total else 0.0

    return {

        "total_samples": total,

        "unique_samples": unique,

        "duplicate_samples": duplicate_samples,

        "duplicate_groups": duplicate_groups,

        "maximum_multiplicity": maximum_multiplicity,

        "collision_rate": collision_rate,
    }

def overlap_statistics(
    hashes_a: list[str],
    hashes_b: list[str],
) -> Dict[str, Any]:
    """
    Compute overlap statistics between two partitions.
    """

    set_a = set(hashes_a)

    set_b = set(hashes_b)

    overlap = set_a.intersection(set_b)

    overlap_percentage = (
        len(overlap) / min(len(set_a), len(set_b))
        if min(len(set_a), len(set_b))
        else 0.0
    )

    return {

        "overlap_samples": len(overlap),

        "overlap_percentage": overlap_percentage,
    }

def hamming_distance(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> int:
    """
    Compute the Hamming distance between two binary vectors.
    """

    return int(np.count_nonzero(sample_a != sample_b))

def label_consistency(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Detect identical feature vectors assigned different labels.
    """

    mapping = {}

    conflicts = 0

    for feature, label in zip(features, labels):

        h = sample_hash(feature)

        if h not in mapping:

            mapping[h] = label

        elif mapping[h] != label:

            conflicts += 1

    return {

        "label_conflicts": conflicts,

        "consistent": conflicts == 0,
    }

# ==========================================================
# Generic Dataset Audit
# ==========================================================

def audit_feature_vectors(
    partition: Partition,
) -> Dict[str, Any]:
    """
    Audit duplicate feature vectors within a partition.
    """

    hashes = partition_hashes(partition)

    return duplicate_statistics(hashes)


def audit_partition_overlap(
    partition_a: Partition,
    partition_b: Partition,
) -> Dict[str, Any]:
    """
    Audit overlap between two dataset partitions.
    """

    hashes_a = partition_hashes(partition_a)
    hashes_b = partition_hashes(partition_b)

    return overlap_statistics(
        hashes_a,
        hashes_b,
    )

def audit_binary_near_duplicates(
    partition: Partition,
    threshold: int = 1,
) -> Dict[str, Any]:
    """
    Detect near-duplicate binary feature vectors using
    exhaustive pairwise Hamming-distance comparison.

    Notes
    -----
    The current implementation performs an exact exhaustive
    search with O(n²) time complexity. This guarantees that
    all near-duplicate pairs within the specified Hamming
    distance threshold are detected.

    Future versions may replace this implementation with an
    indexed search algorithm (e.g., Hamming-ball indexing)
    while preserving the public API.
    """

    features = partition.features

    n = len(features)

    near_duplicate_pairs = 0

    distance_distribution = {}

    for i in range(n):

        for j in range(i + 1, n):

            distance = hamming_distance(
                features[i],
                features[j],
            )

            distance_distribution[distance] = (
                distance_distribution.get(distance, 0) + 1
            )

            if 0 < distance <= threshold:

                near_duplicate_pairs += 1

    return {

        "representation": "binary",

        "metric": "hamming",

        "algorithm": "pairwise_exact",

        "threshold": threshold,

        "near_duplicate_pairs": near_duplicate_pairs,

        "distance_distribution": distance_distribution,
    }

def audit_byte_near_duplicates(
    partition: Partition,
    threshold: int = 1,
) -> Dict[str, Any]:
    """
    Detect near-duplicate byte-valued feature vectors.

    Currently not implemented.
    """

    return {

        "representation": "byte",

        "metric": None,

        "implemented": False,

        "message":
            "Byte-valued near-duplicate detection "
            "is not implemented in the current version.",
    }


def audit_continuous_near_duplicates(
    partition: Partition,
    threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Detect near-duplicate continuous feature vectors.

    Currently not implemented.
    """

    return {

        "representation": "continuous",

        "metric": None,

        "implemented": False,

        "message":
            "Continuous-valued near-duplicate detection "
            "is not implemented in the current version.",
    }


def audit_near_duplicates(
    partition: Partition,
    threshold: int = 1,
) -> Dict[str, Any]:
    """
    Detect near-duplicate feature vectors using a
    representation-appropriate distance metric.
    """

    representation = detect_representation(
        partition.features
    )

    if representation == "binary":

        return audit_binary_near_duplicates(
            partition,
            threshold=threshold,
        )

    elif representation == "byte":

        return audit_byte_near_duplicates(
            partition,
            threshold=threshold,
        )

    elif representation == "continuous":

        return audit_continuous_near_duplicates(
            partition,
            threshold=float(threshold),
        )

    raise ValueError(
        f"Unsupported feature representation: "
        f"{representation}"
    )

def audit_labels(
    partition: Partition,
) -> Dict[str, Any]:
    """
    Audit label consistency within a partition.
    """

    return label_consistency(
        partition.features,
        partition.labels,
    )


def audit_partition(
    partition: Partition,
    threshold: int = 1,
) -> Dict[str, Any]:
    """
    Run all generic audits on a dataset partition.
    """

    return {

        "duplicates":
            audit_feature_vectors(partition),

        "near_duplicates":
            audit_near_duplicates(
                partition,
                threshold=threshold,
            ),

        "labels":
            audit_labels(partition),

        "metadata":
            audit_metadata_dictionary(
                partition
            )
    }


def audit_partition_pair(
    partition_a: Partition,
    partition_b: Partition,
) -> Dict[str, Any]:
    """
    Run pairwise audits between two partitions.
    """

    return {

        "overlap":
            audit_partition_overlap(
                partition_a,
                partition_b,
            )
    }


def audit_dataset(
    dataset: Dataset,
    threshold: int = 1,
) -> Dict[str, Any]:
    """
    Audit every partition contained in a dataset.

    Performs

    - duplicate detection
    - near duplicate detection
    - label consistency
    - pairwise partition overlap
    """

    results = {

        "partitions": {},

        "partition_overlap": {},
    }

    names = dataset.partition_names()

    for name in names:

        results["partitions"][name] = audit_partition(

            dataset.get_partition(name),

            threshold=threshold,
        )

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            first = names[i]
            second = names[j]

            key = f"{first}_vs_{second}"

            results["partition_overlap"][key] = (

                audit_partition_pair(

                    dataset.get_partition(first),

                    dataset.get_partition(second),
                )
            )

    return results

# ==========================================================
# Metadata Audit Engine
# ==========================================================

def count_missing(
    values: np.ndarray,
) -> int:
    """
    Count missing values in a metadata array.

    Parameters
    ----------
    values : np.ndarray
        Metadata values.

    Returns
    -------
    int
        Number of missing values.
    """

    values = np.asarray(values)

    if np.issubdtype(values.dtype, np.floating):

        return int(np.isnan(values).sum())

    return int(np.sum(values == None))


def audit_metadata_field(
    field: MetadataField,
    expected_samples: int,
) -> Dict[str, Any]:
    """
    Audit a single metadata field.

    Parameters
    ----------
    field : MetadataField
        Metadata field to audit.

    expected_samples : int
        Expected number of samples.

    Returns
    -------
    dict
        Metadata integrity statistics.
    """

    return {

        "name":
            field.name,

        "description":
            field.description,

        "samples":
            field.num_samples,

        "expected_samples":
            expected_samples,

        "complete":
            field.num_samples == expected_samples,

        "missing_values":
            count_missing(field.values),
    }


def audit_metadata_dictionary(
    partition: Partition,
) -> Dict[str, Any]:
    """
    Audit all metadata fields associated with a dataset partition.

    Parameters
    ----------
    partition : Partition
        Dataset partition.

    Returns
    -------
    dict
        Audit results for every metadata field.
    """

    results = {}

    for name, field in partition.metadata.items():

        results[name] = audit_metadata_field(

            field,

            partition.num_samples,

        )

    return results

# ==========================================================
# Generator Adapters
# ==========================================================

def generate_gohr_training_data(
    num_rounds: int = 5,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Reconstruct the datasets generated by Gohr's training
    pipeline.

    Parameters
    ----------
    num_rounds : int, optional
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
        #10**7,
        10000,
        num_rounds,
    )

    X_validation, Y_validation = sp.make_train_data(
        #10**6,
        1000,
        num_rounds,
    )

    return (

        X_train,
        Y_train,

        X_validation,
        Y_validation,
    )


def build_gohr_dataset(
    num_rounds: int = 5,
) -> Dataset:
    """
    Construct a generic Dataset object from Gohr's dataset
    generator.

    Parameters
    ----------
    num_rounds : int, optional
        Number of Speck rounds.

    Returns
    -------
    Dataset
        Dataset containing training and validation partitions.
    """

    (
        X_train,
        Y_train,
        X_validation,
        Y_validation,
    ) = generate_gohr_training_data(
        num_rounds=num_rounds,
    )

    dataset = Dataset()

    train_partition = Partition(

        name="train",

        features=X_train,

        labels=Y_train,
    )

    validation_partition = Partition(

        name="validation",

        features=X_validation,

        labels=Y_validation,
    )

    dataset.add_partition(
        train_partition,
    )

    dataset.add_partition(
        validation_partition,
    )

    return dataset

# ==========================================================
# Certificate
# ==========================================================

def evaluate_certificate(
    results: Dict[str, Any],
    duplicate_threshold: int = 0,
    overlap_threshold: int = 0,
    near_duplicate_threshold: int = 0,
    label_conflict_threshold: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate the overall Dataset Integrity Audit certificate.

    Parameters
    ----------
    results : dict
        Results returned by audit_dataset().

    duplicate_threshold : int, optional
        Maximum allowable duplicate samples.

    overlap_threshold : int, optional
        Maximum allowable partition overlap.

    near_duplicate_threshold : int, optional
        Maximum allowable near-duplicate pairs.

    label_conflict_threshold : int, optional
        Maximum allowable label conflicts.

    Returns
    -------
    dict
        Dataset Integrity certificate.
    """

    duplicate_samples = 0
    near_duplicates = 0
    label_conflicts = 0
    partition_overlap = 0

    metadata_complete = True
    metadata_fields = 0

    for partition in results["partitions"].values():

        duplicate_samples += (
            partition["duplicates"]["duplicate_samples"]
        )

        near_duplicates += (
            partition["near_duplicates"]["near_duplicate_pairs"]
        )

        label_conflicts += (
            partition["labels"]["label_conflicts"]
        )

        if "metadata" in partition:

            metadata_fields += len(partition["metadata"])

            for field in partition["metadata"].values():

                metadata_complete &= field["complete"]

    for overlap in results["partition_overlap"].values():

        partition_overlap += (
            overlap["overlap"]["overlap_samples"]
        )

    certificate = {

        "Feature Duplicates":
            "PASS"
            if duplicate_samples <= duplicate_threshold
            else "FAIL",

        "Feature Leakage":
            "PASS"
            if partition_overlap <= overlap_threshold
            else "FAIL",

        "Metadata Integrity":
            (
                "NOT APPLICABLE"
                if metadata_fields == 0
                else (
                    "PASS"
                    if metadata_complete
                    else "FAIL"
                )
            ),

        "Near Duplicates":
            "PASS"
            if near_duplicates <= near_duplicate_threshold
            else "WARNING",

        "Label Consistency":
            "PASS"
            if label_conflicts <= label_conflict_threshold
            else "FAIL",
    }
    statuses = [
        status
        for status in certificate.values()
            if status != "NOT APPLICABLE"
    ]
    if all(
        status == "PASS"
        for status in statuses
    ):

        overall = "PASS"

    elif any(
        status == "FAIL"
        for status in statuses
    ):

        overall = "FAIL"

    else:

        overall = "WARNING"

    certificate["Overall"] = overall

    return certificate

# ==========================================================
# Reporting Functions
# ==========================================================

def print_report(
    results: Dict[str, Any],

) -> None:
    """
    Print a formatted Dataset Integrity D2 report.
    """

    print("=" * 60)
    print("Dataset Integrity Audit")
    print("D2 – Partition Integrity & Leakage Detection")
    print("=" * 60)
    print()

    print("Partition Audits")
    print("-" * 60)

    for name, partition in results["partitions"].items():

        print(f"{name.capitalize()}")

        print(
            f"  Duplicate Samples : "
            f"{partition['duplicates']['duplicate_samples']}"
        )

        print(
            f"  Duplicate Groups  : "
            f"{partition['duplicates']['duplicate_groups']}"
        )

        print(
            f"  Near Duplicates   : "
            f"{partition['near_duplicates']['near_duplicate_pairs']}"
        )

        print(
            f"  Label Conflicts   : "
            f"{partition['labels']['label_conflicts']}"
        )

        print()

    print("Partition Overlap")
    print("-" * 60)

    for name, overlap in results["partition_overlap"].items():

        print(

            f"{name}: "

            f"{overlap['overlap']['overlap_samples']} "

            f"overlapping samples"

        )

    print()

def print_certificate(
    certificate: Dict[str, str],
) -> None:
    """
    Print the Dataset Integrity Audit certificate.
    """

    print()
    print("=" * 60)
    print("Dataset Integrity Audit D2")
    print("=" * 60)
    print()

    for name, status in certificate.items():

        if name == "Overall":
            continue

        print(f"{name:<30}{status}")

    print("-" * 60)

    print(
        f"{'Overall':<30}"
        f"{certificate['Overall']}"
    )

# ==========================================================
# JSON
# ==========================================================

def save_json(
    results: Dict[str, Any],
    certificate: Dict[str, Any],
    filename: str = (
        "d2_partition_integrity.json"
    ),
) -> None:
    """
    Save audit results to a JSON file.
    """

    output = {

        "results": results,

        "certificate": certificate,
    }

    with open(

        filename,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            output,

            file,

            indent=4,
        )

    print()

    print(
        f"Results saved to: {filename}"
    )


# ==========================================================
# main()
# ==========================================================

def main():

    dataset = build_gohr_dataset()

    results = audit_dataset(
        dataset,
    )

    certificate = evaluate_certificate(
        results,
    )

    print_report(
        results,
    )

    print_certificate(
        certificate,
    )

    save_json(
        results,
        certificate,
    )


if __name__ == "__main__":
    main()