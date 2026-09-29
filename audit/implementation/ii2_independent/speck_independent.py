"""
Independent Speck32/64 implementation for CipherMind II-2.

This file is a clean reconstruction from the Gohr publication and the
public supplementary implementation. It does not import the project's
reference Speck implementation.

Speck32/64:
    block size = 32 bits
    key size   = 64 bits
    word size  = 16 bits
    alpha      = 7
    beta       = 2
    full rounds = 22

The public Gohr implementation expresses its state as two 16-bit words
and applies:

    x <- ROR(x, 7)
    x <- (x + y) mod 2^16
    x <- x XOR k
    y <- ROL(y, 2)
    y <- y XOR x

The key schedule uses the same round primitive, with the loop counter
used as the round constant.

Source:
    Gohr, CRYPTO 2019, Section 2.2.
    Public supplementary file: deep_speck/speck.py
"""

from __future__ import annotations

from os import urandom
from typing import Sequence

import numpy as np


WORD_SIZE = 16
ALPHA = 7
BETA = 2
MASK = (1 << WORD_SIZE) - 1
FULL_ROUNDS = 22

GOHR_INPUT_DIFFERENCE = (0x0040, 0x0000)

CHECK_VECTOR = {
    "key": (0x1918, 0x1110, 0x0908, 0x0100),
    "plaintext": (0x6574, 0x694C),
    "rounds": 22,
    "ciphertext": (0xA868, 0x42F2),
}


def _as_uint16_array(value) -> np.ndarray:
    """Convert input to an unsigned 16-bit NumPy array."""
    return np.asarray(value, dtype=np.uint16)


def rotate_left(value, amount: int):
    """
    Rotate a 16-bit word/array left.

    The implementation supports both scalar integers and NumPy arrays.
    """
    amount %= WORD_SIZE

    value32 = np.asarray(value, dtype=np.uint32) & MASK

    result = (
        ((value32 << amount) & MASK)
        | (value32 >> (WORD_SIZE - amount))
    )

    if np.ndim(value) == 0:
        return int(result)
    return result.astype(np.uint16)


def rotate_right(value, amount: int):
    """
    Rotate a 16-bit word/array right.
    """
    amount %= WORD_SIZE

    value32 = np.asarray(value, dtype=np.uint32) & MASK

    result = (
        (value32 >> amount)
        | ((value32 << (WORD_SIZE - amount)) & MASK)
    )

    if np.ndim(value) == 0:
        return int(result)
    return result.astype(np.uint16)


def round_encrypt(
    state: tuple,
    round_key,
    *,
    round_constant=0,
):
    """
    Encrypt one Speck round.

    When round_constant is nonzero this same primitive is also used
    by the Speck32/64 key schedule.
    """
    left, right = state

    left = rotate_right(left, ALPHA)
    left = (np.asarray(left, dtype=np.uint32)
            + np.asarray(right, dtype=np.uint32)) & MASK
    left = left ^ np.asarray(round_key, dtype=np.uint32)

    right = rotate_left(right, BETA)
    right = right ^ left

    return (
        np.asarray(left, dtype=np.uint16),
        np.asarray(right, dtype=np.uint16),
    )


def _key_schedule_batch(
    master_keys: np.ndarray,
    num_rounds: int,
) -> np.ndarray:
    """
    Expand a batch of Speck32/64 keys.

    Parameters
    ----------
    master_keys:
        Shape (4, N). The word ordering follows the original Gohr
        representation.

    num_rounds:
        Number of round keys requested.

    Returns
    -------
    np.ndarray
        Shape (num_rounds, N).
    """
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive")

    master_keys = np.asarray(master_keys, dtype=np.uint16)

    if master_keys.ndim != 2 or master_keys.shape[0] != 4:
        raise ValueError(
            "master_keys must have shape (4, batch_size)"
        )

    batch_size = master_keys.shape[1]

    round_keys = np.empty(
        (num_rounds, batch_size),
        dtype=np.uint16,
    )

    # Gohr's ordering:
    #   k[0] = highest-index master-key word
    #   l = reversed(k[:-1])
    current_key = master_keys[3].copy()

    l_words = [
        master_keys[2].copy(),
        master_keys[1].copy(),
        master_keys[0].copy(),
    ]

    round_keys[0] = current_key

    for i in range(num_rounds - 1):
        slot = i % 3

        updated_l, updated_key = round_encrypt(
            (
                l_words[slot],
                current_key,
            ),
            i,
        )

        l_words[slot] = updated_l
        current_key = updated_key
        round_keys[i + 1] = current_key

    return round_keys


def expand_key(
    master_key,
    num_rounds: int = FULL_ROUNDS,
):
    """
    Expand a Speck32/64 master key.

    Supports either:

        (4,)       for a single key

    or:

        (4, N)     for a batch of keys.

    The single-key output is a Python list of Python integers, matching
    the natural scalar interpretation of the specification.
    """
    key_array = np.asarray(master_key, dtype=np.uint16)

    if key_array.ndim == 1:
        if key_array.shape[0] != 4:
            raise ValueError(
                "A Speck32/64 key must contain four 16-bit words"
            )

        batch = key_array.reshape(4, 1)

        generated = _key_schedule_batch(
            batch,
            num_rounds,
        )

        return [
            int(generated[index, 0])
            for index in range(num_rounds)
        ]

    if key_array.ndim == 2:
        return _key_schedule_batch(
            key_array,
            num_rounds,
        )

    raise ValueError(
        "master_key must have shape (4,) or (4, N)"
    )


def encrypt(
    plaintext,
    round_keys,
):
    """
    Encrypt a Speck32/64 block or batch.

    plaintext may be:

        (x, y)

    where x and y are scalars or NumPy arrays.

    round_keys may be either the scalar list returned by expand_key()
    or the batched array returned by expand_key() for batched keys.
    """
    x, y = plaintext

    for key in round_keys:
        x, y = round_encrypt(
            (x, y),
            key,
        )

    return x, y


def decrypt_round(
    state: tuple,
    round_key,
):
    """Invert one Speck encryption round."""
    x, y = state

    y = np.asarray(y, dtype=np.uint32) ^ np.asarray(
        x,
        dtype=np.uint32,
    )

    y = rotate_right(y, BETA)

    x = np.asarray(x, dtype=np.uint32) ^ np.asarray(
        round_key,
        dtype=np.uint32,
    )

    x = (
        np.asarray(x, dtype=np.uint32)
        - np.asarray(y, dtype=np.uint32)
    ) & MASK

    x = rotate_left(x, ALPHA)

    return (
        np.asarray(x, dtype=np.uint16),
        np.asarray(y, dtype=np.uint16),
    )


def decrypt(
    ciphertext,
    round_keys,
):
    """Decrypt a Speck32/64 block or batch."""
    x, y = ciphertext

    for key in reversed(list(round_keys)):
        x, y = decrypt_round(
            (x, y),
            key,
        )

    return x, y


def convert_to_binary(words: Sequence[np.ndarray]) -> np.ndarray:
    """
    Convert four 16-bit words to Gohr's 64-bit neural input format.

    The input order is:

        [cipher0_left,
         cipher0_right,
         cipher1_left,
         cipher1_right]

    Bits within each word are emitted most-significant bit first.
    """
    if len(words) != 4:
        raise ValueError(
            "Exactly four 16-bit words are required"
        )

    arrays = [
        np.asarray(word, dtype=np.uint16)
        for word in words
    ]

    batch_size = arrays[0].size

    for array in arrays:
        if array.size != batch_size:
            raise ValueError(
                "All words must contain the same number of samples"
            )

    output = np.empty(
        (batch_size, 4 * WORD_SIZE),
        dtype=np.uint8,
    )

    column = 0

    for word in arrays:
        value = word.astype(np.uint32)

        for bit in range(WORD_SIZE - 1, -1, -1):
            output[:, column] = (
                (value >> bit) & 1
            ).astype(np.uint8)
            column += 1

    return output


def check_test_vector() -> bool:
    """
    Verify the Speck32/64 test vector used by the public implementation.
    """
    round_keys = expand_key(
        CHECK_VECTOR["key"],
        CHECK_VECTOR["rounds"],
    )

    observed = encrypt(
        CHECK_VECTOR["plaintext"],
        round_keys,
    )

    observed_tuple = (
        int(observed[0]),
        int(observed[1]),
    )

    return observed_tuple == CHECK_VECTOR["ciphertext"]


def generate_random_differential_material(
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate raw cryptographic material using the operating-system
    random source.

    This mirrors the public Gohr methodology, which obtains keys,
    plaintexts and labels from /dev/urandom.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    labels = (
        np.frombuffer(
            urandom(num_samples),
            dtype=np.uint8,
        )
        & 1
    )

    keys = np.frombuffer(
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

    return (
        labels,
        keys,
        plaintext_left,
        plaintext_right,
    )