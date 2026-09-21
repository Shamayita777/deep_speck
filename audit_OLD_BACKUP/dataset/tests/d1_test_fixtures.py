"""
Synthetic validation fixtures for Dataset Integrity Audit D1.

Scope
-----
These fixtures exist to validate D1's narrow, explicitly-scoped contract
(schema consistency across audited partitions, exact within-partition
duplicate samples, exact cross-partition feature overlap, and the optional
exact-feature label-conflict diagnostic) as implemented in
``audit.dataset.d1_duplicate_detection``.

They are NOT evidence about the Gohr/Speck dataset, and they say nothing
about near duplicates, statistical independence, generation-order effects,
metadata leakage, distributional equivalence, or any other property outside
D1's defined scope. See the module docstring of
``audit.dataset.d1_duplicate_detection`` for D1's authoritative scientific
scope statement.

Determinism
-----------
Every fixture below is a small, explicit, hand-constructed NumPy array.
None of this module uses ``os.urandom``, ``numpy.random``, or any other
nondeterministic source, and none of it depends on Gohr/Speck generation,
TensorFlow, or a GPU. Each fixture's expected duplicate/overlap/schema
outcome is intended to be verifiable by reading the fixture construction
itself -- not by running the production implementation and trusting its
output as ground truth.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------
# Elementary named samples.
#
# Each sample is a 1-D feature vector of 4 uint8 elements. Values are
# chosen so every named sample is visually distinguishable from every
# other, and so no elementary sample's bytes coincide with any other's.
# ---------------------------------------------------------------------
SAMPLE_A = np.array([0, 1, 2, 3], dtype=np.uint8)
SAMPLE_B = np.array([4, 5, 6, 7], dtype=np.uint8)
SAMPLE_C = np.array([8, 9, 10, 11], dtype=np.uint8)
SAMPLE_D = np.array([12, 13, 14, 15], dtype=np.uint8)
SAMPLE_E = np.array([16, 17, 18, 19], dtype=np.uint8)
SAMPLE_F = np.array([20, 21, 22, 23], dtype=np.uint8)
SAMPLE_G = np.array([24, 25, 26, 27], dtype=np.uint8)
SAMPLE_H = np.array([28, 29, 30, 31], dtype=np.uint8)
SAMPLE_I = np.array([32, 33, 34, 35], dtype=np.uint8)

# A single 0-dimensional (scalar) array: no sample dimension at all.
SCALAR_INPUT = np.array(5, dtype=np.uint8)

# An explicitly empty partition: a sample dimension of length zero.
EMPTY_PARTITION = np.empty((0, 4), dtype=np.uint8)


def stack(*samples: np.ndarray) -> np.ndarray:
    """Stack elementary samples into one partition array, first axis = sample index."""
    return np.stack(samples, axis=0)


# ============================================================
# Category A / D: known-clean, known-no-overlap null.
# ============================================================

def clean_partitions() -> dict[str, np.ndarray]:
    """
    Three mutually disjoint partitions with no internal duplicates and no
    cross-partition exact overlap.

        train      : A, B, C
        validation : D, E
        test       : F, G, H

    Every one of the 8 elementary samples used here is distinct, and no
    sample appears more than once anywhere. This is the known-clean /
    known-no-overlap null fixture.
    """
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_D, SAMPLE_E),
        "test": stack(SAMPLE_F, SAMPLE_G, SAMPLE_H),
    }


# ============================================================
# Category B: internal exact duplicates.
# ============================================================

def train_with_internal_duplicate() -> dict[str, np.ndarray]:
    """train contains exactly one duplicated sample (A, twice); validation
    and test are unchanged from clean_partitions() and share nothing with
    train or each other."""
    return {
        "train": stack(SAMPLE_A, SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_D, SAMPLE_E),
        "test": stack(SAMPLE_F, SAMPLE_G, SAMPLE_H),
    }


def validation_with_internal_duplicate() -> dict[str, np.ndarray]:
    """validation contains exactly one duplicated sample (D, twice); train
    and test are unchanged from clean_partitions() and share nothing with
    validation or each other."""
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_D, SAMPLE_D, SAMPLE_E),
        "test": stack(SAMPLE_F, SAMPLE_G, SAMPLE_H),
    }


def test_with_internal_duplicate() -> dict[str, np.ndarray]:
    """test contains exactly one duplicated sample (F, twice); train and
    validation are unchanged from clean_partitions() and share nothing with
    test or each other."""
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_D, SAMPLE_E),
        "test": stack(SAMPLE_F, SAMPLE_F, SAMPLE_G, SAMPLE_H),
    }


def multiplicity_three_fixture() -> np.ndarray:
    """
    One partition array: A, A, A, B.

    By the documented k-1 rule (a sample occurring k times contributes k-1
    duplicate samples), this single array is independently known to have:

        total_samples        = 4
        unique_samples        = 2   (A, B)
        duplicate_groups      = 1   (only A recurs)
        duplicate_samples     = 2   (A occurs 3 times: 3 - 1 = 2)
        maximum_multiplicity  = 3   (A's occurrence count)

    This also doubles as a "smallest-partition-with-repetition" edge case.
    """
    return stack(SAMPLE_A, SAMPLE_A, SAMPLE_A, SAMPLE_B)


# ============================================================
# Category C: cross-partition exact overlap.
# ============================================================

def train_validation_overlap_fixture() -> dict[str, np.ndarray]:
    """train and validation share exactly one exact representation (B).
    No partition has an internal duplicate. test shares nothing with
    either."""
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_B, SAMPLE_D),
        "test": stack(SAMPLE_E, SAMPLE_F, SAMPLE_G),
    }


def train_test_overlap_fixture() -> dict[str, np.ndarray]:
    """train and test share exactly one exact representation (C). No
    partition has an internal duplicate. validation shares nothing with
    either."""
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_D, SAMPLE_E),
        "test": stack(SAMPLE_C, SAMPLE_F, SAMPLE_G),
    }


def validation_test_overlap_fixture() -> dict[str, np.ndarray]:
    """validation and test share exactly one exact representation (E). No
    partition has an internal duplicate. train shares nothing with
    either."""
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_D, SAMPLE_E),
        "test": stack(SAMPLE_E, SAMPLE_F, SAMPLE_G),
    }


def repeated_shared_representation_overlap_fixture() -> dict[str, np.ndarray]:
    """
    B is repeated within BOTH train and validation (so both partitions also
    carry their own internal-duplicate signal), and B is the only
    representation shared between train and validation.

        train      : A, B, B, C   (B occurs twice -> internal duplicate)
        validation : B, B, D      (B occurs twice -> internal duplicate)
        test       : E, F         (disjoint from everything)

    This isolates the documented overlap semantics: partition_overlap is
    the number of UNIQUE shared exact representations (here {B}, so
    train_validation overlap == 1), never the number of pairwise
    occurrence-matches (which would be 2*2=4) and never the raw repeated-
    occurrence count. It also demonstrates that the internal-duplicate
    count and the cross-partition overlap count are independent numbers
    computed correctly even when they involve the same underlying value.
    """
    return {
        "train": stack(SAMPLE_A, SAMPLE_B, SAMPLE_B, SAMPLE_C),
        "validation": stack(SAMPLE_B, SAMPLE_B, SAMPLE_D),
        "test": stack(SAMPLE_E, SAMPLE_F),
    }


# ============================================================
# Category E: partition-size edge cases.
# ============================================================

def singleton_partitions() -> dict[str, np.ndarray]:
    """Each partition contains exactly one sample; all three are distinct."""
    return {
        "train": stack(SAMPLE_A),
        "validation": stack(SAMPLE_B),
        "test": stack(SAMPLE_C),
    }


# ============================================================
# Category F: schema integrity.
# ============================================================

def incompatible_schema_fixture() -> dict[str, np.ndarray]:
    """
    train and test use one computational schema (samples are 1-D vectors of
    shape (4,)); validation uses a different one (samples are 2x2 matrices
    of shape (2, 2)) -- even though both "shapes" hold 4 elements. Per D1's
    schema definition, per-sample shape is part of the computational
    schema, so validation is NOT compatible with train or test.

    train and test remain mutually schema-COMPATIBLE with each other (both
    (4,) uint8) and share no samples, so their pairwise overlap is a real
    computed 0, not None -- only pairs touching validation become None.
    """
    return {
        "train": stack(SAMPLE_A, SAMPLE_B),
        "validation": np.stack(
            [
                np.array([[0, 1], [2, 3]], dtype=np.uint8),
                np.array([[8, 9], [10, 11]], dtype=np.uint8),
            ],
            axis=0,
        ),
        "test": stack(SAMPLE_C, SAMPLE_D),
    }


# ============================================================
# Category G: label-conflict diagnostic.
# ============================================================

def label_conflict_features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """
    Feature A occurs twice within one partition: once labeled 0, once
    labeled 1. Feature B and C each occur once, with labels that do not
    conflict with anything.

        features: A, A, B, C
        labels  : 0, 1, 0, 1

    Because a label conflict requires the SAME exact feature representation
    to recur (necessarily within one partition, since the label-conflict
    diagnostic is computed per-partition), this fixture also has an
    unavoidable feature-level duplicate signal for A (duplicate_samples=1)
    alongside the label conflict (label_conflict_groups=1). The test suite
    verifies both numbers independently, and separately verifies -- via a
    hand-built minimal results object exercising evaluate_d1 directly --
    that label_conflict_groups is not what drives the FAIL outcome here.
    """
    features = stack(SAMPLE_A, SAMPLE_A, SAMPLE_B, SAMPLE_C)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    return features, labels
