"""
Independent real-vs-random dataset construction for CipherMind II-2.

This follows Gohr's published training-data procedure:

    - draw uniformly random keys;
    - draw uniformly random plaintexts;
    - use Delta = 0x0040/0000;
    - draw binary real/random labels;
    - for real examples, retain the fixed input difference;
    - for random examples, replace the second plaintext with a fresh
      random plaintext;
    - encrypt both branches for the requested number of rounds;
    - encode the resulting four 16-bit words into 64 binary features.

The public paper states that /dev/urandom was used for keys, plaintext
pairs and binary labels, and that data was generated separately for
training and validation. The supplementary implementation uses Python's
os.urandom, which on Linux maps to the operating-system random source.

No reference CipherMind dataset generator is imported here.
"""

from __future__ import annotations

from os import urandom

import numpy as np

from .speck_independent import (
    GOHR_INPUT_DIFFERENCE,
    convert_to_binary,
    encrypt,
    expand_key,
)


GOHR_DIFFERENTIAL = GOHR_INPUT_DIFFERENCE


def _fresh_uint16(count: int) -> np.ndarray:
    """Draw uniformly random 16-bit words from the OS source."""
    if count <= 0:
        return np.empty(0, dtype=np.uint16)

    return np.frombuffer(
        urandom(2 * count),
        dtype=np.uint16,
    )


def construct_from_material(
    *,
    labels: np.ndarray,
    master_keys: np.ndarray,
    plaintext_left: np.ndarray,
    plaintext_right: np.ndarray,
    rounds: int,
    differential: tuple[int, int] = GOHR_DIFFERENTIAL,
    random_plaintext_left: np.ndarray | None = None,
    random_plaintext_right: np.ndarray | None = None,
):
    """
    Construct a Gohr-style dataset from explicitly supplied material.

    This function is the deterministic testable core of dataset
    construction. Supplying all random inputs externally allows the audit
    comparator to feed exactly the same material to independent and
    reference implementations.

    Parameters
    ----------
    labels:
        Binary labels of shape (N,).

    master_keys:
        Four 16-bit words per sample, shape (4, N).

    plaintext_left/right:
        First plaintext words, shape (N,).

    rounds:
        Number of reduced Speck rounds.

    differential:
        Fixed input XOR difference.

    random_plaintext_left/right:
        Fresh independently generated second-plaintext words used for
        label-0 examples. These must be supplied whenever zero-label
        samples exist.
    """
    labels = np.asarray(
        labels,
        dtype=np.uint8,
    )

    master_keys = np.asarray(
        master_keys,
        dtype=np.uint16,
    )

    plaintext_left = np.asarray(
        plaintext_left,
        dtype=np.uint16,
    )

    plaintext_right = np.asarray(
        plaintext_right,
        dtype=np.uint16,
    )

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    count = labels.size

    if master_keys.shape != (4, count):
        raise ValueError(
            f"master_keys must have shape (4, {count})"
        )

    if plaintext_left.shape != (count,):
        raise ValueError(
            "plaintext_left has the wrong shape"
        )

    if plaintext_right.shape != (count,):
        raise ValueError(
            "plaintext_right has the wrong shape"
        )

    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(
            "labels must contain only 0 and 1"
        )

    second_left = (
        plaintext_left ^ np.uint16(differential[0])
    )
    second_right = (
        plaintext_right ^ np.uint16(differential[1])
    )

    random_mask = labels == 0
    random_count = int(np.count_nonzero(random_mask))

    if random_count:
        if (
            random_plaintext_left is None
            or random_plaintext_right is None
        ):
            raise ValueError(
                "random plaintext material is required for label-0 samples"
            )

        random_plaintext_left = np.asarray(
            random_plaintext_left,
            dtype=np.uint16,
        )

        random_plaintext_right = np.asarray(
            random_plaintext_right,
            dtype=np.uint16,
        )

        if random_plaintext_left.shape != (random_count,):
            raise ValueError(
                "random_plaintext_left has the wrong shape"
            )

        if random_plaintext_right.shape != (random_count,):
            raise ValueError(
                "random_plaintext_right has the wrong shape"
            )

        second_left[random_mask] = random_plaintext_left
        second_right[random_mask] = random_plaintext_right

    round_keys = expand_key(
        master_keys,
        rounds,
    )

    cipher0_left, cipher0_right = encrypt(
        (plaintext_left, plaintext_right),
        round_keys,
    )

    cipher1_left, cipher1_right = encrypt(
        (second_left, second_right),
        round_keys,
    )

    X = convert_to_binary(
        [
            cipher0_left,
            cipher0_right,
            cipher1_left,
            cipher1_right,
        ]
    )

    return X, labels.copy()


def generate_dataset(
    num_samples: int,
    rounds: int,
    differential: tuple[int, int] = GOHR_DIFFERENTIAL,
):
    """
    Generate one complete real-vs-random dataset.

    This uses the operating-system random source exactly as described
    by Gohr's paper/public code.

    The caller should invoke this independently for training, validation,
    and test data.
    """
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive"
        )

    if rounds <= 0:
        raise ValueError(
            "rounds must be positive"
        )

    labels = (
        np.frombuffer(
            urandom(num_samples),
            dtype=np.uint8,
        )
        & 1
    )

    master_keys = np.frombuffer(
        urandom(8 * num_samples),
        dtype=np.uint16,
    ).reshape(4, -1)

    plaintext_left = np.frombuffer(
        urandom(2 * num_samples),
        dtype=np.uint16,
    )

    plaintext_right = np.frombuffer(
        urandom(2 * num_samples),
        dtype=np.uint16,
    )

    real_second_left = (
        plaintext_left
        ^ np.uint16(differential[0])
    )

    real_second_right = (
        plaintext_right
        ^ np.uint16(differential[1])
    )

    random_mask = labels == 0
    random_count = int(
        np.count_nonzero(random_mask)
    )

    random_left = _fresh_uint16(random_count)
    random_right = _fresh_uint16(random_count)

    real_second_left[random_mask] = random_left
    real_second_right[random_mask] = random_right

    round_keys = expand_key(
        master_keys,
        rounds,
    )

    cipher0_left, cipher0_right = encrypt(
        (plaintext_left, plaintext_right),
        round_keys,
    )

    cipher1_left, cipher1_right = encrypt(
        (real_second_left, real_second_right),
        round_keys,
    )

    X = convert_to_binary(
        [
            cipher0_left,
            cipher0_right,
            cipher1_left,
            cipher1_right,
        ]
    )

    return X, labels