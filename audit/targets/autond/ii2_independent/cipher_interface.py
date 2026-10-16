"""What a ``cipher`` argument is for the II-2 search (SPECIFICATION.md §6).

From the paper, the search needs:
  * the width n of an input difference (Alg. 1 L1, L13 [given PDF]);
  * E^r_K(X) for uniformly random X and K at a chosen round count r
    (Definition 2; §5.2);
  * the widths of the output and of the key (Definition 2; A-020).

Bit-array convention (NOT IN PAPER: chosen as below, see AMBIGUITY_LOG.md#A-026):
  plaintexts, keys and outputs are numpy uint8 arrays holding 0/1, one row per
  sample; column 0 holds the most significant bit of the integer view.
Differences are Python ints. delta is XORed into the plaintext only (A-021).
What one "round" means is the cipher's own definition (A-022).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

__all__ = ["Cipher", "FunctionCipher", "validate_cipher", "int_to_bits", "bits_to_int"]

_BITS_PER_BYTE = 8  # representation constant, not an algorithm parameter


class Cipher(Protocol):
    """Structural type of the ``cipher`` argument to ``search``.

    Attributes
    ----------
    name : str
    input_bits : int      n, the difference / plaintext width (>= 1)
    output_bits : int     width of E's output (>= 1); equals n for block ciphers
    key_bits : int        k (>= 0); 0 for keyless permutations
    nominal_rounds : int | None
        Used only as the default round cap (A-011).
    """

    name: str
    input_bits: int
    output_bits: int
    key_bits: int
    nominal_rounds: Optional[int]

    def encrypt(self, plaintexts: np.ndarray, keys: np.ndarray, rounds: int) -> np.ndarray:
        """Encrypt each row under the matching key row for ``rounds`` rounds.

        plaintexts: uint8, shape (t, input_bits), values 0/1, read-only
        keys:       uint8, shape (t, key_bits),   values 0/1, read-only
        returns:    integer or bool array, shape (t, output_bits), values 0/1
        Must be vectorized over rows and deterministic in its inputs.
        """
        ...


@dataclass(frozen=True)
class FunctionCipher:
    """Convenience wrapper that turns a plain function into a ``Cipher``."""

    name: str
    input_bits: int
    output_bits: int
    key_bits: int
    encrypt_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
    nominal_rounds: Optional[int] = None

    def encrypt(self, plaintexts: np.ndarray, keys: np.ndarray, rounds: int) -> np.ndarray:
        return self.encrypt_fn(plaintexts, keys, rounds)


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def validate_cipher(cipher) -> None:
    """Raise TypeError / ValueError if ``cipher`` does not satisfy ``Cipher``."""
    for attr in ("name", "input_bits", "output_bits", "key_bits", "nominal_rounds", "encrypt"):
        if not hasattr(cipher, attr):
            raise TypeError(f"cipher lacks required attribute {attr!r}")
    if not isinstance(cipher.name, str):
        raise TypeError("cipher.name must be a str")
    for attr, minimum in (("input_bits", 1), ("output_bits", 1), ("key_bits", 0)):
        value = getattr(cipher, attr)
        if not _is_int(value):
            raise TypeError(f"cipher.{attr} must be an int, got {type(value).__name__}")
        if value < minimum:
            raise ValueError(f"cipher.{attr} must be >= {minimum}, got {value}")
    nominal = cipher.nominal_rounds
    if nominal is not None:
        if not _is_int(nominal):
            raise TypeError("cipher.nominal_rounds must be an int or None")
        if nominal < 1:
            raise ValueError(f"cipher.nominal_rounds must be >= 1, got {nominal}")
    if not callable(cipher.encrypt):
        raise TypeError("cipher.encrypt must be callable")


def int_to_bits(value: int, width: int) -> np.ndarray:
    """Integer -> uint8 bit array of length ``width``, most significant bit first."""
    if not _is_int(value) or not _is_int(width):
        raise TypeError("value and width must be ints")
    value, width = int(value), int(width)
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if value < 0 or (value >> width) != 0:
        raise ValueError(f"{value} does not fit in {width} unsigned bits")
    nbytes = (width + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
    raw = np.frombuffer(value.to_bytes(nbytes, "big"), dtype=np.uint8)
    bits = np.unpackbits(raw)
    return bits[nbytes * _BITS_PER_BYTE - width:].copy()


def bits_to_int(bits) -> int:
    """Inverse of ``int_to_bits``: 1-D array of 0/1, most significant bit first."""
    arr = np.asarray(bits)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("bits must be a non-empty 1-D array")
    if ((arr != 0) & (arr != 1)).any():
        raise ValueError("bits must contain only 0 and 1")
    arr = arr.astype(np.uint8)
    pad = (-arr.size) % _BITS_PER_BYTE
    padded = np.concatenate([np.zeros(pad, dtype=np.uint8), arr])
    return int.from_bytes(np.packbits(padded).tobytes(), "big")
