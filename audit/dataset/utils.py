import numpy as np


def detect_representation(
    dataset: np.ndarray,
) -> str:
    """
    Detect the feature representation of a dataset.

    Parameters
    ----------
    dataset : np.ndarray
        Input feature matrix.

    Returns
    -------
    str
        One of:
            "binary"
            "byte"
            "continuous"
    """

    unique_values = np.unique(dataset)

    # Binary representation (0/1 values)
    if np.all(np.isin(unique_values, [0, 1])):
        return "binary"

    # Byte representation (integer values in the range 0–255),
    # irrespective of whether they are stored as integers or floats.
    if (
        np.all(dataset >= 0)
        and np.all(dataset <= 255)
        and np.all(dataset == np.floor(dataset))
    ):
        return "byte"

    # Continuous representation
    return "continuous"