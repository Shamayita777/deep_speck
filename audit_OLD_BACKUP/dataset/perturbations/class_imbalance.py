"""
Class Imbalance Perturbation

Artificially introduces class imbalance by removing
a fraction of samples from one target class.
"""
import numpy as np

from audit.dataset.d4_controlled_perturbation import Perturbation


class ClassImbalancePerturbation(Perturbation):
    """
    Introduce class imbalance by subsampling one class.
    """

    def __init__(
        self,
        target_class=1,
        retain_fraction=0.5,
        random_state=42,
    ) -> None:

        super().__init__(
            name="class_imbalance",
            description=(
                "Artificially introduce class imbalance."
            ),
        )

        self.target_class = target_class
        self.retain_fraction = retain_fraction
        self.random_state = random_state

    def apply(
        self,
        features,
        labels,
    ):
        """
        Apply class imbalance.

        Parameters
        ----------
        features
            Original feature matrix.

        labels
            Original labels.

        Returns
        -------
        tuple
            Imbalanced feature matrix and labels.
        """

        rng = np.random.default_rng(
            self.random_state,
        )

        target = np.where(
            labels == self.target_class
        )[0]

        other = np.where(
            labels != self.target_class
        )[0]
        # If the requested class is not present,
        # return the dataset unchanged.
        if len(target) == 0:
            return (
                features,
                labels,
            )
        retained = rng.choice(
            target,
            size=max(
                1,
                int(
                    len(target)
                    * self.retain_fraction
                ),
            ),
            replace=False,
        )

        keep = np.concatenate(
            [
                retained,
                other,
            ]
        )

        rng.shuffle(keep)

        return (
            features[keep],
            labels[keep],
        )