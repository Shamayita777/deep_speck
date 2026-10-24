"""
D4 Label-Shuffle Perturbation.

Preserves the training feature matrix and empirical label multiset while
randomly destroying the original feature/label correspondence.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from audit.dataset.d4_controlled_perturbation import Perturbation


class LabelShufflePerturbation(Perturbation):
    """Randomly permute training labels using an explicit RNG."""

    label_only = True
    index_based = True

    def permutation_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """A uniformly random bijection of range(n) as little-endian int64."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")
        return rng.permutation(int(n)).astype("<i8")

    def apply_indices(self, features, labels, indices):
        """Perturbed labels[i] = labels[indices[i]]; features unchanged."""
        return features, np.asarray(labels)[np.asarray(indices)]

    def __init__(self) -> None:
        super().__init__(
            name="label_shuffle",
            description=(
                "Randomly permute training labels while preserving the "
                "feature matrix and empirical label multiset."
            ),
        )

    def apply(
        self,
        features: Any,
        labels: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features)
        labels = np.asarray(labels)

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")
        if features.ndim < 2:
            raise ValueError("features must have sample and feature dimensions.")
        if labels.ndim < 1:
            raise ValueError("labels must have a sample dimension.")
        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of samples.")

        _, shuffled = self.apply_indices(features, labels, self.permutation_indices(len(labels), rng))

        if not np.array_equal(
            np.sort(labels.reshape(-1)),
            np.sort(shuffled.reshape(-1)),
        ):
            raise RuntimeError("Label permutation failed to preserve the label multiset.")

        # A full random permutation can rarely leave some positions unchanged;
        # the perturbation is still valid because the correspondence is randomized.
        return features, shuffled

    def verify(self, features, labels, perturbed_features, perturbed_labels) -> None:
        """Definition check: features untouched, labels a permutation of the originals.

        Features must be the same array object or byte-identical; labels
        must have the same length, dtype and label multiset.
        """
        if perturbed_features is not features:
            a, b = np.asarray(features), np.asarray(perturbed_features)
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                raise ValueError("label_shuffle must leave the feature matrix unchanged.")
        original = np.asarray(labels).reshape(-1)
        shuffled = np.asarray(perturbed_labels).reshape(-1)
        if original.shape != shuffled.shape or original.dtype != shuffled.dtype:
            raise ValueError("label_shuffle output labels differ in length or dtype from the input.")
        u0, c0 = np.unique(original, return_counts=True)
        u1, c1 = np.unique(shuffled, return_counts=True)
        if not (np.array_equal(u0, u1) and np.array_equal(c0, c1)):
            raise ValueError("label_shuffle output is not a permutation of the input labels.")
