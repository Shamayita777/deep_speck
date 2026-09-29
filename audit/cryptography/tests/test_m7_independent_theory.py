"""
Tests for M7: independent Lipmaa-Moriai theory implementation.

These tests compare CipherMind's independent theory implementation against:

1. Known mathematical examples;
2. Exhaustive exact modular-addition enumeration for 4-bit words;
3. Exact enumeration for randomized 8-bit parameter pairs;
4. The current production xdp_plus() implementation for randomized
   16-bit transitions;
5. Real 5-round Speck32/64 trajectories generated from the current
   cipher implementation, comparing chained theory results.

No neural-network training is performed.
"""

from __future__ import annotations

import numpy as np

from audit.cryptography.gohr import speck as production_speck
from audit.cryptography.independent_reference.lipmaa_moriai_independent import (
    chain_transition_probabilities,
    independent_xdp_plus,
    independent_xdp_plus_array,
)


def _exact_distribution(
    alpha: int,
    beta: int,
    word_size: int,
) -> np.ndarray:
    """
    Compute the exact xdp+ distribution for fixed alpha/beta.

    This is independent of both the production theory implementation
    and the independent closed-form implementation.

    It enumerates all x,y pairs and counts the resulting gamma:

        gamma =
            (x + y)
            XOR
            ((x XOR alpha) + (y XOR beta))
    """
    modulus = 1 << word_size
    mask = modulus - 1

    values = np.arange(modulus, dtype=np.uint64)

    x = values[:, None]
    y = values[None, :]

    left = (x + y) & mask
    right = (
        ((x ^ np.uint64(alpha)) + (y ^ np.uint64(beta)))
        & mask
    )

    gamma = left ^ right

    counts = np.bincount(
        gamma.reshape(-1).astype(np.int64),
        minlength=modulus,
    )

    return counts.astype(np.float64) / float(modulus * modulus)


def test_known_lipmaa_moriai_example() -> None:
    """
    Verify the worked five-bit example recorded in the project notes.

        alpha = 11100
        beta  = 00110
        gamma = 10110

    Expected probability = 2^-2 = 1/4.
    """
    alpha = int("11100", 2)
    beta = int("00110", 2)
    gamma = int("10110", 2)

    result = independent_xdp_plus(
        alpha,
        beta,
        gamma,
        word_size=5,
    )

    assert result == 0.25


def test_invalid_transition_returns_zero() -> None:
    """
    Construct a transition violating the LSB condition.
    """
    alpha = 0b0001
    beta = 0b0000
    gamma = 0b0000

    result = independent_xdp_plus(
        alpha,
        beta,
        gamma,
        word_size=4,
    )

    assert result == 0.0


def test_exhaustive_four_bit_against_exact_enumeration() -> None:
    """
    Exhaustively test every alpha/beta pair at n=4.

    For each fixed alpha,beta, every possible gamma is compared against
    exact enumeration over all 2^(2n) = 256 input pairs.
    """
    word_size = 4
    modulus = 1 << word_size

    for alpha in range(modulus):
        for beta in range(modulus):
            exact = _exact_distribution(
                alpha,
                beta,
                word_size,
            )

            for gamma in range(modulus):
                expected = float(exact[gamma])

                actual = independent_xdp_plus(
                    alpha,
                    beta,
                    gamma,
                    word_size=word_size,
                )

                assert actual == expected, (
                    f"Mismatch for n={word_size}, "
                    f"alpha={alpha:#x}, "
                    f"beta={beta:#x}, "
                    f"gamma={gamma:#x}: "
                    f"independent={actual}, exact={expected}"
                )


def test_randomized_eight_bit_against_exact_enumeration() -> None:
    """
    Use exact enumeration for 20 deterministic random alpha/beta pairs
    at n=8.

    Each distribution is evaluated over all 65,536 possible (x,y) pairs.
    """
    rng = np.random.default_rng(20260929)

    word_size = 8
    modulus = 1 << word_size

    for _ in range(20):
        alpha = int(rng.integers(0, modulus))
        beta = int(rng.integers(0, modulus))

        exact = _exact_distribution(
            alpha,
            beta,
            word_size,
        )

        for gamma in range(modulus):
            expected = float(exact[gamma])

            actual = independent_xdp_plus(
                alpha,
                beta,
                gamma,
                word_size=word_size,
            )

            assert actual == expected, (
                f"Mismatch for n={word_size}, "
                f"alpha={alpha:#x}, "
                f"beta={beta:#x}, "
                f"gamma={gamma:#x}: "
                f"independent={actual}, exact={expected}"
            )


def test_randomized_sixteen_bit_agrees_with_production() -> None:
    """
    Compare 20,000 randomized 16-bit transitions against the current
    production xdp_plus() implementation.

    Exact equality is expected because all nonzero values are powers of
    two and therefore exactly representable as binary floating-point.
    """
    rng = np.random.default_rng(20260929)

    sample_count = 20_000

    alpha = rng.integers(
        0,
        1 << 16,
        size=sample_count,
        dtype=np.uint32,
    )
    beta = rng.integers(
        0,
        1 << 16,
        size=sample_count,
        dtype=np.uint32,
    )
    gamma = rng.integers(
        0,
        1 << 16,
        size=sample_count,
        dtype=np.uint32,
    )

    independent = independent_xdp_plus_array(
        alpha,
        beta,
        gamma,
        word_size=16,
    )

    production = production_speck.xdp_plus(
        alpha,
        beta,
        gamma,
    )

    assert independent.shape == production.shape
    assert np.array_equal(independent, production)


def test_real_speck_trajectory_chain_agrees_with_production() -> None:
    """
    Generate deterministic real 5-round Speck32/64 differential
    trajectories and compare:

        independent chained theory
                    vs
        production xdp_plus chained over exactly the same transitions.

    This tests the theory implementation on realistic round-by-round
    cryptographic transitions rather than arbitrary triples alone.
    """
    rng = np.random.default_rng(20260929)

    sample_count = 1_000
    rounds = 5
    differential = (0x0040, 0x0000)

    transitions = np.empty(
        (sample_count, rounds, 3),
        dtype=np.uint32,
    )

    for sample_index in range(sample_count):
        key_words = tuple(
            int(x)
            for x in rng.integers(
                0,
                1 << 16,
                size=4,
                dtype=np.uint16,
            )
        )

        round_keys = production_speck.expand_key(
            key_words,
            rounds,
        )

        x0 = int(rng.integers(0, 1 << 16))
        y0 = int(rng.integers(0, 1 << 16))

        x1 = x0 ^ differential[0]
        y1 = y0 ^ differential[1]

        for round_index, round_key in enumerate(round_keys):
            before0 = (
                production_speck.ror(
                    x0,
                    production_speck.ALPHA(),
                ),
                y0,
            )

            before1 = (
                production_speck.ror(
                    x1,
                    production_speck.ALPHA(),
                ),
                y1,
            )

            alpha = before0[0] ^ before1[0]
            beta = before0[1] ^ before1[1]

            after_add0 = (
                before0[0] + before0[1]
            ) & production_speck.MASK_VAL

            after_add1 = (
                before1[0] + before1[1]
            ) & production_speck.MASK_VAL

            gamma = after_add0 ^ after_add1

            transitions[
                sample_index,
                round_index,
                0,
            ] = alpha

            transitions[
                sample_index,
                round_index,
                1,
            ] = beta

            transitions[
                sample_index,
                round_index,
                2,
            ] = gamma

            x0, y0 = production_speck.enc_one_round(
                (x0, y0),
                round_key,
            )

            x1, y1 = production_speck.enc_one_round(
                (x1, y1),
                round_key,
            )

    independent_chain = chain_transition_probabilities(
        transitions,
        word_size=16,
    )

    flat = transitions.reshape(-1, 3)

    production_round_probs = production_speck.xdp_plus(
        flat[:, 0],
        flat[:, 1],
        flat[:, 2],
    ).reshape(
        sample_count,
        rounds,
    )

    production_chain = np.prod(
        production_round_probs,
        axis=1,
    )

    assert independent_chain.shape == production_chain.shape
    assert np.array_equal(
        independent_chain,
        production_chain,
    )


def test_chain_function_shape_and_bounds() -> None:
    """
    Basic contract test for the independent chain function.
    """
    transitions = np.array(
        [
            [
                [0, 0, 0],
                [1, 1, 0],
            ],
            [
                [2, 0, 2],
                [3, 1, 2],
            ],
        ],
        dtype=np.uint32,
    )

    values = chain_transition_probabilities(
        transitions,
        word_size=4,
    )

    assert values.shape == (2,)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)