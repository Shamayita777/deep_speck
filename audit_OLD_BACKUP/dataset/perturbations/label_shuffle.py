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

        shuffled = rng.permutation(labels)

        if not np.array_equal(
            np.sort(labels.reshape(-1)),
            np.sort(shuffled.reshape(-1)),
        ):
            raise RuntimeError("Label permutation failed to preserve the label multiset.")

        # A full random permutation can rarely leave some positions unchanged;
        # the perturbation is still valid because the correspondence is randomized.
        return features, shuffled
