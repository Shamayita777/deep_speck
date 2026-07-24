"""
Feature Shuffle Perturbation

Randomly permutes feature vectors while preserving
the original labels.
"""

import numpy as np

from audit.dataset.d4_controlled_perturbation import Perturbation


class FeatureShufflePerturbation(Perturbation):
    """
    Randomly permute the order of feature vectors
    (samples) while leaving the labels unchanged.

    This perturbation destroys the correspondence
    between individual feature vectors and their
    associated labels.
    """

    def __init__(self) -> None:

        super().__init__(
            name="feature_shuffle",
            description=(
                "Randomly permute the order of feature "
                "vectors (samples) while preserving labels."
            ),
        )

    def apply(
        self,
        features,
        labels,
    ):
        """
        Apply the sample-order shuffle perturbation.

        Parameters
        ----------
        features
            Original feature matrix.

        labels
            Original labels.

        Returns
        -------
        tuple
            Shuffled features and original labels.
        """

        shuffled_features = np.random.permutation(features)

        return (
            shuffled_features,
            labels,
        )