#!/usr/bin/env python3
"""
A toy cipher for exercising the search harness in unit tests.

WHY A TOY. The tests must exercise the harness - patches, logging,
output, refusals - without running the research experiment on the real
target. Nothing in the test suite searches Speck32/64 or Simon32/64 at
any setting, so no test artifact can be mistaken for research evidence.

WHY IT MUST DECORRELATE, AND WHY A LINEAR TOY DOES NOT WORK.
`optimizer.optimize` runs `while True`, advancing one round at a time,
and breaks only when the best bias score across the surviving population
falls below its threshold `T = 0.05`. A toy built from XOR alone is
linear over GF(2), so every input difference propagates to a fixed output
difference at every round, every score stays at 0.5, and the loop never
terminates. (Observed: a first version of this fixture hung the suite.)
This version therefore uses a non-linear integer mixer so that scores
collapse to sampling noise from round 2 onwards.

BEHAVIOUR BY ROUND.
  r <= 1 : the plaintext passes straight through, so every candidate
           difference scores 0.5 and the search advances.
  r >= 2 : output is a deterministic but strongly mixing function of
           plaintext and key, so output differences are unbiased and the
           best score falls to sampling noise - with 32 bits and 1000
           samples that is of order 0.01, comfortably under the 0.05
           threshold. `optimize` then stops with
           highest_non_random_round = 1.

A complete toy search therefore takes well under a second.

INTERFACE. Matches what `optimizer.optimize` requires of a cipher module:
`plain_bits`, `key_bits`, and `encrypt(p, k, r)` returning a uint8 array
of shape (n, plain_bits).
"""

from __future__ import annotations

import numpy as np

plain_bits = 32
key_bits = 32
word_size = 32

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_M32 = np.uint64(0xFFFFFFFF)


def _pack(bits: np.ndarray) -> np.ndarray:
    """(n, 32) uint8 bit matrix, most significant bit first -> (n,) uint64."""
    weights = (np.uint64(1) << np.arange(31, -1, -1, dtype=np.uint64))
    return (bits.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def _unpack(values: np.ndarray) -> np.ndarray:
    """(n,) uint64 -> (n, 32) uint8 bit matrix, most significant bit first."""
    shifts = np.arange(31, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)


def _mix(v: np.ndarray) -> np.ndarray:
    """A splitmix64-style avalanche step, masked back to 64 bits."""
    with np.errstate(over="ignore"):
        v = (v + np.uint64(0x9E3779B97F4A7C15)) & _M64
        v = ((v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        v = ((v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        v = v ^ (v >> np.uint64(31))
    return v & _M64


def encrypt(p, k, r):
    p = np.asarray(p, dtype=np.uint8)
    k = np.asarray(k, dtype=np.uint8)
    r = int(r)
    if r <= 1:
        return p.copy()                       # pure leak: differences propagate exactly
    v = _pack(p) ^ (_pack(k) << np.uint64(17))
    for _ in range(r - 1):
        v = _mix(v)
    return _unpack(v & _M32)
