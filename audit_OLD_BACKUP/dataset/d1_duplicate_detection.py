"""
Dataset Integrity Audit D1
Exact Duplicate and Partition Overlap Census.

D1 audits a concrete dataset instance for:

1. dataset schema consistency across audited partitions;
2. exact duplicate samples within each partition;
3. exact feature overlap between compatible partitions;
4. optional exact-feature label conflicts.

Scientific scope
----------------
D1 answers the narrow question:

    "Does the audited dataset instance have a consistent feature/label
     schema, and does it contain exact duplicate feature representations
     within partitions or exact feature overlap across compatible audited
     partitions?"

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
# Dataset schema validation
# ============================================================

def _sample_schema(array: np.ndarray) -> Dict[str, Any]:
    """Describe the computational schema of one dataset partition."""

    array = np.asarray(array)
    if array.ndim < 1:
        raise ValueError("Dataset must contain a sample dimension.")
    if array.shape[0] < 1:
        raise ValueError("Dataset must contain at least one sample.")

    return {
        "dtype": str(array.dtype),
        "ndim": int(array.ndim),
        "sample_shape": list(array.shape[1:]),
    }


def validate_dataset_schema(
    partitions: Mapping[str, np.ndarray],
    labels: Optional[Mapping[str, np.ndarray]] = None,
    *,
    expected_feature_schema: Optional[Mapping[str, Any]] = None,
    expected_label_schema: Optional[Mapping[str, Any]] = None,
    representation_convention: Optional[str] = None,
    feature_encoding: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate dataset schema before exact-equality census.

    The validation has two levels:

    1. All supplied feature partitions must use the same computational
       schema (dtype, dimensionality, and per-sample shape).
    2. If an expected schema is supplied, every partition must match it.

    Representation convention and feature encoding are semantic metadata and
    cannot safely be inferred from NumPy arrays alone. They are therefore
    recorded only when explicitly supplied by the dataset-specific driver.

    Label schema is checked analogously when labels are supplied.
    """

    if not partitions:
        raise ValueError("At least one feature partition is required.")

    feature_schemas = {
        name: _sample_schema(array)
        for name, array in partitions.items()
    }

    reference_name = next(iter(feature_schemas))
    reference = feature_schemas[reference_name]

    cross_partition_mismatches: Dict[str, Dict[str, Any]] = {}
    for name, schema in feature_schemas.items():
        mismatches = {
            key: {"reference": reference[key], "observed": schema[key]}
            for key in ("dtype", "ndim", "sample_shape")
            if schema[key] != reference[key]
        }
        if mismatches:
            cross_partition_mismatches[name] = mismatches

    expected_feature_mismatches: Dict[str, Dict[str, Any]] = {}
    if expected_feature_schema is not None:
        for name, schema in feature_schemas.items():
            mismatches = {
                key: {
                    "expected": expected_feature_schema[key],
                    "observed": schema[key],
                }
                for key in ("dtype", "ndim", "sample_shape")
                if key in expected_feature_schema
                and schema[key] != expected_feature_schema[key]
            }
            if mismatches:
                expected_feature_mismatches[name] = mismatches

    label_schemas: Dict[str, Dict[str, Any]] = {}
    label_mismatches: Dict[str, Dict[str, Any]] = {}
    if labels is not None:
        for name, feature_array in partitions.items():
            if name not in labels:
                raise ValueError(f"Missing labels for partition {name!r}.")
            label_array = np.asarray(labels[name])
            if label_array.ndim < 1:
                raise ValueError(f"Labels for partition {name!r} must have a sample dimension.")
            if label_array.shape[0] != np.asarray(feature_array).shape[0]:
                raise ValueError(
                    f"Label/sample count mismatch for partition {name!r}: "
                    f"{label_array.shape[0]} labels for {np.asarray(feature_array).shape[0]} samples."
                )
            label_schemas[name] = {
                "dtype": str(label_array.dtype),
                "ndim": int(label_array.ndim),
                "sample_shape": list(label_array.shape[1:]),
            }

        label_reference = label_schemas[reference_name]
        for name, schema in label_schemas.items():
            mismatches = {
                key: {"reference": label_reference[key], "observed": schema[key]}
                for key in ("dtype", "ndim", "sample_shape")
                if schema[key] != label_reference[key]
            }
            if mismatches:
                label_mismatches[name] = mismatches

        if expected_label_schema is not None:
            for name, schema in label_schemas.items():
                mismatches = {
                    key: {
                        "expected": expected_label_schema[key],
                        "observed": schema[key],
                    }
                    for key in ("dtype", "ndim", "sample_shape")
                    if key in expected_label_schema
                    and schema[key] != expected_label_schema[key]
                }
                if mismatches:
                    label_mismatches[name] = mismatches

    valid = not cross_partition_mismatches and not expected_feature_mismatches and not label_mismatches

    return {
        "valid": bool(valid),
        "features": feature_schemas,
        "labels": label_schemas,
        "cross_partition_feature_mismatches": cross_partition_mismatches,
        "expected_feature_schema": dict(expected_feature_schema) if expected_feature_schema is not None else None,
        "expected_feature_mismatches": expected_feature_mismatches,
        "expected_label_schema": dict(expected_label_schema) if expected_label_schema is not None else None,
        "label_mismatches": label_mismatches,
        "representation_convention": representation_convention,
        "feature_encoding": feature_encoding,
    }


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
    *,
    expected_feature_schema: Optional[Mapping[str, Any]] = None,
    expected_label_schema: Optional[Mapping[str, Any]] = None,
    representation_convention: Optional[str] = None,
    feature_encoding: Optional[str] = None,
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

    partitions: Dict[str, np.ndarray] = {
        "train": train,
        "validation": validation,
    }
    if test is not None:
        partitions["test"] = np.asarray(test)

    label_partitions: Optional[Dict[str, np.ndarray]] = None
    supplied_labels = {
        "train": train_labels,
        "validation": validation_labels,
        "test": test_labels,
    }
    if any(value is not None for value in supplied_labels.values()):
        label_partitions = {}
        for name in partitions:
            value = supplied_labels[name]
            if value is None:
                raise ValueError(f"Labels must be supplied for every audited partition; missing {name!r}.")
            label_partitions[name] = np.asarray(value)

    schema = validate_dataset_schema(
        partitions,
        label_partitions,
        expected_feature_schema=expected_feature_schema,
        expected_label_schema=expected_label_schema,
        representation_convention=representation_convention,
        feature_encoding=feature_encoding,
    )

    results: Dict[str, Any] = {
        "schema": schema,
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
        "partition_overlap": {},
    }

    # Exact cross-partition comparison is meaningful only when the
    # computational schemas of the two partitions agree. If they do not,
    # record the comparison as unavailable rather than manufacturing a
    # misleading zero-overlap result from incompatible representations.
    feature_schema_pairs = [("train", "validation")]
    if test is not None:
        feature_schema_pairs.extend([("train", "test"), ("validation", "test")])

    for left, right in feature_schema_pairs:
        if schema["features"][left] == schema["features"][right]:
            results["partition_overlap"][f"{left}_{right}"] = partition_overlap(
                partitions[left], partitions[right]
            )
        else:
            results["partition_overlap"][f"{left}_{right}"] = None

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
        if value is not None
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

    schema_valid = bool(results.get("schema", {}).get("valid", True))

    if (
        schema_valid
        and duplicate_samples == 0
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
        "schema_valid": schema_valid,
        "rule": (
            "PASS iff all audited feature/label schemas are consistent, "
            "every audited partition contains zero exact duplicate samples, "
            "and every compatible audited partition pair contains zero "
            "exact feature overlap."
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
            "schema_consistency_check": True,
            "representation_convention": results["schema"].get("representation_convention"),
            "feature_encoding": results["schema"].get("feature_encoding"),
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
            "schema": results["schema"],
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
                "D1 requires consistent computational schema across audited partitions "
                "before exact duplicate/overlap results are considered valid."
            ),
            (
                "Representation convention and feature encoding are semantic properties "
                "and are enforced only when explicitly supplied by the dataset-specific driver."
            ),
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

    print("[Dataset schema]")

    schema = results.get("schema", {})
    print(f"  Valid                  : {schema.get('valid')}")
    print(f"  Representation        : {schema.get('representation_convention')}")
    print(f"  Feature encoding      : {schema.get('feature_encoding')}")
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
