"""
Dataset Integrity Audit D1
Exact Duplicate and Partition Overlap Census.

D1 audits a concrete dataset instance for:

1. exact duplicate samples within each partition;
2. exact feature overlap between partitions;
3. optional exact-feature label conflicts.

Scientific scope
----------------
D1 answers the narrow question:

    "Does the audited dataset instance contain exact duplicate
     feature representations within partitions or exact feature
     overlap across audited partitions?"

D1 does NOT establish:

- absence of near duplicates;
- statistical independence;
- absence of generation-order leakage;
- absence of metadata leakage;
- distributional equivalence;
- general dataset validity.

Those questions belong to other Dataset Integrity audits.

Provenance
----------
D1 operates on the arrays explicitly supplied to it.

If those arrays were freshly generated from a dataset-generation
procedure, this audit certifies the generated dataset instance
that was actually supplied to D1. It does not retroactively
certify an unavailable historical dataset instance.

D0 infrastructure
-----------------
Certificate construction and provenance recording are provided
by audit.dataset.common.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .common import (
    array_sha256,
    build_provenance,
    make_certificate,
    write_certificate,
)


# ============================================================
# Exact sample fingerprint
# ============================================================

def sample_hash(sample: np.ndarray) -> str:
    """
    Compute a SHA-256 fingerprint for one dataset sample.

    The fingerprint commits to the sample's:

    - dtype
    - shape
    - contiguous raw byte representation

    This defines exactly what D1 considers an "exactly equal"
    feature representation.
    """

    return array_sha256(np.asarray(sample))


# ============================================================
# Within-partition duplicate census
# ============================================================

def partition_duplicate_statistics(
    dataset: np.ndarray,
) -> Dict[str, int]:
    """
    Perform an exhaustive exact-duplicate census within one
    dataset partition.

    Parameters
    ----------
    dataset:
        Dataset partition whose first dimension indexes samples.

    Returns
    -------
    dict
        Exact duplicate statistics.

    Notes
    -----
    If a sample occurs k times, it contributes k-1 duplicate
    samples.

    Example
    -------
    Samples:

        A, B, B, B, C

    produce:

        duplicate_groups = 1
        duplicate_samples = 2
    """

    dataset = np.asarray(dataset)

    counts: Counter[str] = Counter()

    for sample in dataset:
        counts[sample_hash(sample)] += 1

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

    maximum_multiplicity = (
        max(counts.values())
        if counts
        else 0
    )

    return {
        "total_samples": int(len(dataset)),
        "unique_samples": int(len(counts)),
        "duplicate_groups": int(duplicate_groups),
        "duplicate_samples": int(duplicate_samples),
        "maximum_multiplicity": int(maximum_multiplicity),
    }


# ============================================================
# Partition hash sets
# ============================================================

def partition_hash_set(
    dataset: np.ndarray,
) -> set[str]:
    """
    Compute the set of exact sample fingerprints for a partition.
    """

    return {
        sample_hash(sample)
        for sample in np.asarray(dataset)
    }


# ============================================================
# Cross-partition overlap
# ============================================================

def partition_overlap(
    dataset_a: np.ndarray,
    dataset_b: np.ndarray,
) -> int:
    """
    Count exact feature representations shared by two partitions.

    Each shared representation is counted once, regardless of
    how many times it occurs in either partition.

    This is intentionally a set-based exact-overlap census.
    """

    hashes_a = partition_hash_set(dataset_a)
    hashes_b = partition_hash_set(dataset_b)

    return int(
        len(hashes_a.intersection(hashes_b))
    )


# ============================================================
# Optional label-conflict diagnostic
# ============================================================

def label_conflict_statistics(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Detect exact feature representations associated with more
    than one label.

    This is an accompanying diagnostic. It does not establish
    semantic correctness of labels or absence of ambiguity.
    """

    features = np.asarray(features)
    labels = np.asarray(labels)

    if len(features) != len(labels):
        raise ValueError(
            "features and labels must contain the same number "
            "of samples."
        )

    labels_by_hash: Dict[str, set[Any]] = {}

    for feature, label in zip(features, labels):

        fingerprint = sample_hash(feature)

        try:
            label_value = np.asarray(label).item()
        except ValueError:
            label_value = str(np.asarray(label))

        if fingerprint not in labels_by_hash:
            labels_by_hash[fingerprint] = set()

        try:
            labels_by_hash[fingerprint].add(label_value)
        except TypeError:
            # Fallback for non-hashable label representations.
            labels_by_hash[fingerprint].add(
                repr(label_value)
            )

    conflicting_hashes = {
        fingerprint
        for fingerprint, label_values
        in labels_by_hash.items()
        if len(label_values) > 1
    }

    return {
        "label_conflict_groups": int(
            len(conflicting_hashes)
        ),
        "consistent": (
            len(conflicting_hashes) == 0
        ),
    }


# ============================================================
# Full D1 census
# ============================================================

def audit_duplicates(
    train: np.ndarray,
    validation: np.ndarray,
    test: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    validation_labels: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Execute Dataset Integrity Audit D1.

    Parameters
    ----------
    train:
        Training partition.

    validation:
        Validation partition.

    test:
        Optional test partition.

    train_labels:
        Optional training labels for exact-feature label-conflict
        diagnostics.

    validation_labels:
        Optional validation labels.

    test_labels:
        Optional test labels.

    Returns
    -------
    dict
        Machine-readable D1 findings.
    """

    train = np.asarray(train)
    validation = np.asarray(validation)

    if train.ndim < 1:
        raise ValueError(
            "Training dataset must contain a sample dimension."
        )

    if validation.ndim < 1:
        raise ValueError(
            "Validation dataset must contain a sample dimension."
        )

    results: Dict[str, Any] = {
        "audit_id": "D1",
        "audit_name": (
            "Exact Duplicate and Partition Overlap Census"
        ),
        "partitions": {
            "train": partition_duplicate_statistics(
                train
            ),
            "validation": partition_duplicate_statistics(
                validation
            ),
        },
        "partition_overlap": {
            "train_validation": partition_overlap(
                train,
                validation,
            ),
        },
    }

    # --------------------------------------------------------
    # Optional test partition
    # --------------------------------------------------------

    if test is not None:

        test = np.asarray(test)

        if test.ndim < 1:
            raise ValueError(
                "Test dataset must contain a sample dimension."
            )

        results["partitions"]["test"] = (
            partition_duplicate_statistics(test)
        )

        results["partition_overlap"].update({
            "train_test": partition_overlap(
                train,
                test,
            ),
            "validation_test": partition_overlap(
                validation,
                test,
            ),
        })

    # --------------------------------------------------------
    # Optional label diagnostics
    # --------------------------------------------------------

    if train_labels is not None:

        results["partitions"]["train"][
            "label_consistency"
        ] = label_conflict_statistics(
            train,
            train_labels,
        )

    if validation_labels is not None:

        results["partitions"]["validation"][
            "label_consistency"
        ] = label_conflict_statistics(
            validation,
            validation_labels,
        )

    if test is not None and test_labels is not None:

        results["partitions"]["test"][
            "label_consistency"
        ] = label_conflict_statistics(
            test,
            test_labels,
        )

    return results


# ============================================================
# D1 decision rule
# ============================================================

def evaluate_d1(
    results: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Apply the explicit D1 decision rule.

    PASS
        No exact duplicate samples within any audited partition
        and no exact feature overlap between audited partitions.

    FAIL
        One or more exact duplicate samples or exact
        cross-partition overlaps are observed.

    INCONCLUSIVE
        Reserved for cases where the audit cannot validly be
        executed or the required evidence is unavailable.
        Input/execution failures should normally be raised before
        certificate construction rather than silently converted
        into PASS.
    """

    duplicate_samples = sum(
        int(partition["duplicate_samples"])
        for partition in results["partitions"].values()
    )

    exact_partition_overlap = sum(
        int(value)
        for value in results["partition_overlap"].values()
    )

    label_conflict_groups = sum(
        int(
            partition.get(
                "label_consistency",
                {}
            ).get(
                "label_conflict_groups",
                0,
            )
        )
        for partition in results["partitions"].values()
    )

    if (
        duplicate_samples == 0
        and exact_partition_overlap == 0
    ):
        outcome = "PASS"
    else:
        outcome = "FAIL"

    return {
        "outcome": outcome,
        "duplicate_samples": duplicate_samples,
        "exact_partition_overlap": (
            exact_partition_overlap
        ),
        "label_conflict_groups": label_conflict_groups,
        "rule": (
            "PASS iff every audited partition contains zero "
            "exact duplicate samples and every audited partition "
            "pair contains zero exact feature overlap."
        ),
    }


# ============================================================
# Certificate construction
# ============================================================

def build_d1_certificate(
    results: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_version: Optional[str] = None,
    generation_procedure: Optional[str] = None,
    generation_parameters: Optional[
        Mapping[str, Any]
    ] = None,
    random_seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct and optionally persist a D1 evidence certificate.

    The certificate records:

    - the D1 claim;
    - empirical findings;
    - explicit decision rule;
    - provenance;
    - methodology;
    - limitations.
    """

    decision = evaluate_d1(results)

    partition_provenance: Dict[str, Any] = {}

    for (
        partition_name,
        partition_results,
    ) in results["partitions"].items():

        partition_provenance[
            partition_name
        ] = {
            "sample_count": partition_results[
                "total_samples"
            ],
            "unique_sample_count": partition_results[
                "unique_samples"
            ],
        }

    provenance = build_provenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        generation_procedure=generation_procedure,
        generation_parameters=generation_parameters,
        random_seed=random_seed,
        partitions=partition_provenance,
        audit_configuration={
            "audit_id": "D1",
            "hash_algorithm": "SHA-256",
            "hash_scope": (
                "sample dtype + sample shape + contiguous "
                "raw sample bytes"
            ),
            "exact_duplicate_detection": True,
            "exact_partition_overlap_detection": True,
            "label_conflict_diagnostic": (
                any(
                    "label_consistency" in partition
                    for partition
                    in results["partitions"].values()
                )
            ),
        },
    )

    certificate = make_certificate(
        audit_id="D1",
        audit_name=(
            "Exact Duplicate and Partition Overlap Census"
        ),
        claim=(
            "The audited dataset instance contains no exact "
            "duplicate feature samples within partitions and "
            "no exact feature overlap across audited partitions."
        ),
        outcome=decision["outcome"],
        findings={
            "duplicate_samples": decision[
                "duplicate_samples"
            ],
            "exact_partition_overlap": decision[
                "exact_partition_overlap"
            ],
            "label_conflict_groups": decision[
                "label_conflict_groups"
            ],
            "partition_results": results[
                "partitions"
            ],
            "partition_overlap": results[
                "partition_overlap"
            ],
        },
        methodology={
            "statistical_unit": (
                "individual dataset sample"
            ),
            "method": (
                "exhaustive SHA-256 exact-equality census"
            ),
            "decision_rule": decision["rule"],
            "inferential_scope": (
                "exact duplicates and exact feature overlap only"
            ),
        },
        provenance=provenance,
        limitations=[
            (
                "D1 addresses exact equality only. It does not "
                "establish the absence of near duplicates or "
                "statistical dependence."
            ),
            (
                "If the audited data were regenerated from a "
                "generation procedure, the certificate describes "
                "the generated dataset instance actually supplied "
                "to D1; it does not retroactively certify an "
                "unavailable historical dataset instance."
            ),
            (
                "A zero-overlap result is evidence that no exact "
                "overlap was observed in the audited data under "
                "the stated representation. It is not evidence "
                "that all forms of dataset correlation are absent."
            ),
        ],
        evidence_level=(
            "DATASET_INTEGRITY_EXACT_CENSUS"
        ),
    )

    if output_path is not None:

        write_certificate(
            certificate,
            output_path,
        )

    return certificate


# ============================================================
# Human-readable report
# ============================================================

def print_report(
    results: Mapping[str, Any],
    certificate: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Print a human-readable D1 audit report.
    """

    print("=" * 64)
    print("Dataset Integrity Audit")
    print(
        "D1 – Exact Duplicate and "
        "Partition Overlap Census"
    )
    print("=" * 64)
    print()

    for (
        partition_name,
        partition,
    ) in results["partitions"].items():

        print(f"[{partition_name}]")

        print(
            f"  Samples              : "
            f"{partition['total_samples']}"
        )

        print(
            f"  Unique samples       : "
            f"{partition['unique_samples']}"
        )

        print(
            f"  Duplicate groups     : "
            f"{partition['duplicate_groups']}"
        )

        print(
            f"  Duplicate samples    : "
            f"{partition['duplicate_samples']}"
        )

        print(
            f"  Maximum multiplicity : "
            f"{partition['maximum_multiplicity']}"
        )

        consistency = partition.get(
            "label_consistency"
        )

        if consistency is not None:

            print(
                f"  Label conflict groups: "
                f"{consistency['label_conflict_groups']}"
            )

        print()

    print("[Exact partition overlap]")

    for (
        partition_pair,
        count,
    ) in results["partition_overlap"].items():

        print(
            f"  {partition_pair:24s}: {count}"
        )

    if certificate is not None:

        print()

        print(
            "Decision               : "
            f"{certificate['decision']['outcome']}"
        )
