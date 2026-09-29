"""
Independent Lipmaa-Moriai reference implementation
===================================================

This module independently implements the closed-form XOR differential
probability of modular addition described by Lipmaa and Moriai:

    H. Lipmaa and S. Moriai,
    "Efficient Algorithms for Computing Differential Properties of Addition",
    FSE 2001.

PURPOSE
-------
This implementation exists solely as an independent verification source
for CipherMind's production theoretical reference.

It intentionally does NOT import:

    audit.cryptography.gohr.speck.xdp_plus

or any other CipherMind implementation of the theory.

The implementation is written in a deliberately different style from the
production vectorized implementation:

    - scalar bit extraction;
    - explicit validity checking;
    - explicit counting of non-equal bit positions;
    - Python floating-point power-of-two construction.

This is an audit reference, not the production implementation.

SCIENTIFIC SCOPE
----------------
The returned probability is the probability of one specific modular-
addition XOR differential transition:

    (alpha, beta) -> gamma

For a chain of transitions, the probabilities are multiplied under the
same Markov/independence assumption used by the CipherMind target quantity.

This is NOT:

    - the exact full-cipher differential probability;
    - the probability of all trails;
    - a claim about what a neural network learned.

The default Speck32/64 word size is 16 bits.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


DEFAULT_WORD_SIZE = 16


def _get_bit(value: int, position: int) -> int:
    """Return one bit of an integer, least-significant bit first."""
    return (int(value) >> position) & 1


def _all_equal_at_bit(alpha: int, beta: int, gamma: int, position: int) -> bool:
    """Return True exactly when alpha_i = beta_i = gamma_i."""
    a = _get_bit(alpha, position)
    b = _get_bit(beta, position)
    g = _get_bit(gamma, position)
    return a == b == g


def _is_valid_transition(
    alpha: int,
    beta: int,
    gamma: int,
    word_size: int,
) -> bool:
    """
    Check the Lipmaa-Moriai validity conditions for one transition.

    The LSB condition is:

        alpha_0 XOR beta_0 XOR gamma_0 = 0

    For every i in [0, word_size-2], whenever the three bits at i
    are equal, the next gamma bit must satisfy:

        gamma_{i+1}
            = alpha_{i+1} XOR beta_{i+1} XOR beta_i
    """
    if word_size < 2:
        raise ValueError("word_size must be at least 2")

    # Least-significant-bit condition.
    if (
        _get_bit(alpha, 0)
        ^ _get_bit(beta, 0)
        ^ _get_bit(gamma, 0)
    ):
        return False

    # Carry-consistency conditions.
    for i in range(word_size - 1):
        if not _all_equal_at_bit(alpha, beta, gamma, i):
            continue

        expected_next_gamma = (
            _get_bit(alpha, i + 1)
            ^ _get_bit(beta, i + 1)
            ^ _get_bit(beta, i)
        )

        actual_next_gamma = _get_bit(gamma, i + 1)

        if actual_next_gamma != expected_next_gamma:
            return False

    return True


def independent_xdp_plus(
    alpha: int,
    beta: int,
    gamma: int,
    word_size: int = DEFAULT_WORD_SIZE,
) -> float:
    """
    Independently compute Lipmaa-Moriai xdp+ for one transition.

    Parameters
    ----------
    alpha, beta, gamma:
        XOR differences defining the modular-addition transition

            (alpha, beta) -> gamma

    word_size:
        Width n of the modular addition.

    Returns
    -------
    float
        The exact dyadic probability represented as a Python float.

        Returns 0.0 if the transition is invalid.

    Notes
    -----
    The probability is:

        2^(-w)

    where w is the number of bit positions i = 0,...,n-2 for which
    alpha_i, beta_i, gamma_i are not all equal.

    This function intentionally uses scalar bit operations rather than
    the production implementation's vectorized mask/popcount approach.
    """
    if word_size < 2:
        raise ValueError("word_size must be at least 2")

    if word_size > 63:
        raise ValueError(
            "word_size > 63 is not supported by this independent scalar reference"
        )

    mask = (1 << word_size) - 1

    alpha = int(alpha) & mask
    beta = int(beta) & mask
    gamma = int(gamma) & mask

    if not _is_valid_transition(
        alpha,
        beta,
        gamma,
        word_size,
    ):
        return 0.0

    unequal_count = 0

    for i in range(word_size - 1):
        if not _all_equal_at_bit(alpha, beta, gamma, i):
            unequal_count += 1

    return math.ldexp(1.0, -unequal_count)


def independent_xdp_plus_array(
    alpha: Iterable[int] | np.ndarray,
    beta: Iterable[int] | np.ndarray,
    gamma: Iterable[int] | np.ndarray,
    word_size: int = DEFAULT_WORD_SIZE,
) -> np.ndarray:
    """
    Scalar-reference implementation applied elementwise to arrays.

    This wrapper deliberately performs a Python-level scalar calculation
    for each element rather than reproducing the production vectorized
    bit-mask implementation.
    """
    alpha_arr = np.asarray(alpha)
    beta_arr = np.asarray(beta)
    gamma_arr = np.asarray(gamma)

    if not (
        alpha_arr.shape == beta_arr.shape == gamma_arr.shape
    ):
        raise ValueError(
            "alpha, beta, and gamma must have identical shapes"
        )

    flat_alpha = alpha_arr.reshape(-1)
    flat_beta = beta_arr.reshape(-1)
    flat_gamma = gamma_arr.reshape(-1)

    values = [
        independent_xdp_plus(
            int(a),
            int(b),
            int(g),
            word_size=word_size,
        )
        for a, b, g in zip(
            flat_alpha,
            flat_beta,
            flat_gamma,
        )
    ]

    return np.asarray(values, dtype=np.float64).reshape(alpha_arr.shape)


def chain_transition_probabilities(
    transitions: np.ndarray,
    word_size: int = DEFAULT_WORD_SIZE,
) -> np.ndarray:
    """
    Independently chain per-round transition probabilities.

    Parameters
    ----------
    transitions:
        Array of shape:

            (num_samples, num_rounds, 3)

        where the final dimension contains:

            [alpha, beta, gamma]

        for one modular-addition differential transition.

    Returns
    -------
    np.ndarray
        One chained probability per sample.

    A single invalid transition makes the entire chained probability zero.
    Otherwise:

        P_chain = product_r P_r

    under the Markov/independence assumption.
    """
    transitions = np.asarray(transitions)

    if transitions.ndim != 3:
        raise ValueError(
            "transitions must have shape (num_samples, num_rounds, 3)"
        )

    if transitions.shape[-1] != 3:
        raise ValueError(
            "the final transition dimension must contain alpha, beta, gamma"
        )

    num_samples = transitions.shape[0]
    num_rounds = transitions.shape[1]

    result = np.ones(num_samples, dtype=np.float64)

    for sample_index in range(num_samples):
        probability = 1.0

        for round_index in range(num_rounds):
            alpha = int(transitions[sample_index, round_index, 0])
            beta = int(transitions[sample_index, round_index, 1])
            gamma = int(transitions[sample_index, round_index, 2])

            probability *= independent_xdp_plus(
                alpha,
                beta,
                gamma,
                word_size=word_size,
            )

            if probability == 0.0:
                break

        result[sample_index] = probability

    return result