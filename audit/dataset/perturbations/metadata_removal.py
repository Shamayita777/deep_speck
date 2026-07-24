"""
Metadata Removal Perturbation

Removes user-specified metadata features while
preserving the remaining feature representation.
"""

import numpy as np

from audit.dataset.d4_controlled_perturbation import Perturbation

class MetadataRemovalPerturbation(Perturbation):
    """
    Remove selected metadata columns from the feature
    representation.

    This perturbation is intended for datasets where
    metadata is represented explicitly as feature
    columns.
    """

    def __init__(
        self,
        metadata_indices,
    ) -> None:

        super().__init__(
            name="metadata_removal",
            description=(
                "Remove selected metadata features "
                "from the dataset."
            ),
        )

        self.metadata_indices = np.asarray(
            metadata_indices,
            dtype=int,
        )

    def apply(
        self,
        features,
        labels,
    ):
        """
        Remove metadata columns.

        Parameters
        ----------
        features
            Original feature matrix.

        labels
            Original labels.

        Returns
        -------
        tuple
            Features with metadata removed and the
            original labels.
        """

        # No metadata specified.
        if len(self.metadata_indices) == 0:
            return (
                features,
                labels,
            )

        # Ensure all requested metadata columns exist.
        if np.max(self.metadata_indices) >= features.shape[1]:
            raise ValueError(
                "Metadata index exceeds the number of feature columns."
            )

        remaining = np.delete(
            features,
            self.metadata_indices,
            axis=1,
        )

        return (
            remaining,
            labels,
        )