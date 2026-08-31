"""
D4 Metadata-Removal Perturbation.

Removes explicitly identified metadata feature columns. This perturbation
is only appropriate when the dataset representation documents those
columns as metadata rather than cryptographic/input features.
"""

from __future__ import annotations
from typing import Any, Iterable
import numpy as np
from audit.dataset.d4_controlled_perturbation import Perturbation


class MetadataRemovalPerturbation(Perturbation):
    """Remove validated metadata columns from the feature matrix."""

    def __init__(self, metadata_indices: Iterable[int]) -> None:
        indices = np.asarray(list(metadata_indices), dtype=int)
        if indices.ndim != 1:
            raise ValueError("metadata_indices must be one-dimensional.")
        if len(indices) and np.any(indices < 0):
            raise ValueError("metadata_indices cannot contain negative indices.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("metadata_indices must be unique.")
        super().__init__(
            name="metadata_removal",
            description="Remove explicitly designated metadata feature columns.",
        )
        self.metadata_indices = indices

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
        if len(self.metadata_indices) == 0:
            return x, y
        if np.max(self.metadata_indices) >= x.shape[1]:
            raise ValueError("Metadata index exceeds the number of feature columns.")
        return np.delete(x, self.metadata_indices, axis=1), y
