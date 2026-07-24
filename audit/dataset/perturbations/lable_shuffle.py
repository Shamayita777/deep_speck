"""
Label Shuffle Perturbation

Randomly permutes dataset labels while preserving
the original feature vectors.
"""

import numpy as np

from audit.dataset.d4_controlled_perturbation import Perturbation


class LabelShufflePerturbation(Perturbation):
    """
    Randomly shuffle labels while leaving the
    feature vectors unchanged.

    This perturbation destroys the correspondence
    between samples and labels.
    """

    def __init__(self) -> None:

        super().__init__(
            name="label_shuffle",
            description=(
                "Randomly permute labels while "
                "preserving feature vectors."
            ),
        )

    def apply(
        self,
        features,
        labels,
    ):
        """
        Apply the label shuffle perturbation.

        Parameters
        ----------
        features
            Original feature matrix.

        labels
            Original labels.

        Returns
        -------
        tuple
            Original features and shuffled labels.
        """

        shuffled_labels = np.random.permutation(labels)

        return (
            features,
            shuffled_labels,
        )