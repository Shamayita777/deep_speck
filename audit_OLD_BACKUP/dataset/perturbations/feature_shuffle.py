"""
D4 Feature-Shuffle Perturbation.

Randomly permutes training feature rows while preserving the original
label array. This destroys row-wise feature/label correspondence while
preserving the empirical feature distribution and label multiset.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from audit.dataset.d4_controlled_perturbation import Perturbation


class FeatureShufflePerturbation(Perturbation):
    """Permute training feature rows using an explicit RNG."""

    def __init__(self) -> None:
        super().__init__(
            name="feature_shuffle",
            description=(
                "Randomly permute training feature rows while preserving "
                "the original labels."
            ),
        )

    def apply(
        self,
        features: Any,
        labels: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features)
        y = np.asarray(labels)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")
        if x.ndim < 2 or y.ndim < 1:
            raise ValueError("Invalid feature/label dimensions.")
        if len(x) != len(y):
            raise ValueError("features and labels must contain the same number of samples.")
        return rng.permutation(x, axis=0), y
