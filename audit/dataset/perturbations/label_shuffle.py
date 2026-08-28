"""
D4 Label-Shuffle Perturbation.

This perturbation preserves the feature matrix and randomly
permutes the labels.

Scientific interpretation
-------------------------
A label permutation preserves the empirical label counts but
destroys the original correspondence between individual feature
vectors and their labels.

Therefore, a performance change under this perturbation tests
the dependence of the trained predictor on the original
feature/label correspondence.

It does NOT, by itself, establish that the model learned
cryptographic structure. The observed effect may also reflect
dataset-construction artifacts, implementation artifacts, or
other structure correlated with the original labels.

Randomness
----------
The perturbation requires an explicit NumPy Generator. This keeps
the perturbation randomness separate from dataset-generation
randomness and makes the permutation reproducible when the audit
seed is controlled.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from audit.dataset.d4_controlled_perturbation import (
    Perturbation,
)


class LabelShufflePerturbation(Perturbation):
    """
    Randomly permute labels while leaving features unchanged.
    """

    def __init__(self) -> None:
        super().__init__(
            name="label_shuffle",
            description=(
                "Randomly permute training labels while preserving "
                "the original feature vectors and empirical class "
                "counts."
            ),
        )

    def apply(
        self,
        features: Any,
        labels: Any,
        *,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a random permutation to the labels.

        Parameters
        ----------
        features:
            Feature matrix.

        labels:
            Corresponding training labels.

        rng:
            Dedicated NumPy random generator supplied by D4.

        Returns
        -------
        (features, shuffled_labels)
            The original feature array and a permutation of the
            original labels.

        Raises
        ------
        ValueError
            If the number of features and labels differs.
        TypeError
            If rng is not a NumPy Generator.
        """

        features = np.asarray(features)
        labels = np.asarray(labels)

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be an instance of "
                "numpy.random.Generator."
            )

        if features.ndim < 1:
            raise ValueError(
                "features must contain a sample dimension."
            )

        if labels.ndim < 1:
            raise ValueError(
                "labels must contain a sample dimension."
            )

        if len(features) != len(labels):
            raise ValueError(
                "features and labels must contain the same "
                "number of samples."
            )

        shuffled_labels = rng.permutation(labels)

        # Explicit invariant: permutation preserves the empirical
        # label multiset exactly.
        if not np.array_equal(
            np.sort(labels.reshape(-1)),
            np.sort(shuffled_labels.reshape(-1)),
        ):
            raise RuntimeError(
                "Label permutation failed to preserve the "
                "empirical label multiset."
            )

        return features, shuffled_labels
