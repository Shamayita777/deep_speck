"""
Independent scalar reference for G-CE5. Imports nothing from akr.py, runner.py,
analysis.py or audit.cryptography.gohr; SPECK32/64 is written here from the
specification (Beaulieu et al., "The SIMON and SPECK Families of Lightweight
Block Ciphers", ePrint 2013/404) and checked against its published test vector.

Independence is of implementation only: this module and akr.py share the
reading of the aKR algorithm recorded in docs/SOURCE_VERIFICATION.md. The table
check below is what guards that shared reading against round-count, word-order
and input-difference errors in the table itself.

    rotations      S^a x = x <<< a, n = 16, alpha = 7, beta = 2
    round          x <- (S^-alpha x + y) ^ k ;  y <- S^beta y ^ x
    key schedule   (l2, l1, l0, k0) ; l_{i+3} = (k_i + S^-alpha l_i) ^ i ;
                   k_{i+1} = S^beta k_i ^ l_{i+3}
    test vector    key 1918 1110 0908 0100, pt 6574 694c -> ct a868 42f2
    table index    bits of (dl & M_L) at the set positions of M_L, lowest first,
                   then the same for dr, concatenated as (left << HW(M_R)) | right
    score          mean over all 2^16 subkeys (no reduction) of
                   table[index(dec(C0,k) ^ dec(C1,k))] / n_table
"""

from __future__ import annotations

import random

import numpy as np
from scipy.stats import chi2

W = 0xFFFF
ALPHA, BETA = 7, 2
TEST_VECTOR = {"key": (0x1918, 0x1110, 0x0908, 0x0100), "pt": (0x6574, 0x694C), "ct": (0xA868, 0x42F2)}


def rotl(x: int, a: int) -> int:
    return ((x << a) | (x >> (16 - a))) & W


def rotr(x: int, a: int) -> int:
    return ((x >> a) | (x << (16 - a))) & W


def round_enc(x: int, y: int, k: int) -> tuple[int, int]:
    x = ((rotr(x, ALPHA) + y) & W) ^ k
    y = rotl(y, BETA) ^ x
    return x, y


def round_dec(x: int, y: int, k: int) -> tuple[int, int]:
    y = rotr(x ^ y, BETA)
    x = rotl(((x ^ k) - y) & W, ALPHA)
    return x, y


def key_schedule(key: tuple[int, int, int, int], rounds: int) -> list[int]:
    """key = (l2, l1, l0, k0), the specification's word order."""
    l = [key[2], key[1], key[0]]
    k = [key[3]]
    for i in range(rounds - 1):
        l.append(((k[i] + rotr(l[i], ALPHA)) & W) ^ i)
        k.append(rotl(k[i], BETA) ^ l[i + 3])
    return k[:rounds]


def encrypt(pt: tuple[int, int], key, rounds: int) -> tuple[int, int]:
    x, y = pt
    for k in key_schedule(key, rounds):
        x, y = round_enc(x, y, k)
    return x, y


def self_test() -> bool:
    return encrypt(TEST_VECTOR["pt"], TEST_VECTOR["key"], 22) == TEST_VECTOR["ct"]


def _positions(m: int) -> list[int]:
    return [i for i in range(16) if (m >> i) & 1]


def table_index(dl: int, dr: int, mask: tuple[int, int], _pos=None) -> int:
    pl, pr = _pos or (_positions(mask[0]), _positions(mask[1]))
    left = sum(((dl >> p) & 1) << j for j, p in enumerate(pl))
    right = sum(((dr >> p) & 1) << j for j, p in enumerate(pr))
    return (left << len(pr)) | right


def score(c0: tuple[int, int], c1: tuple[int, int], table, n_table: int,
          mask: tuple[int, int]) -> float:
    """Literal average over all 2^16 last-round subkeys."""
    pos = (_positions(mask[0]), _positions(mask[1]))
    total = 0
    for k in range(1 << 16):
        a = round_dec(c0[0], c0[1], k)
        b = round_dec(c1[0], c1[1], k)
        total += int(table[table_index(a[0] ^ b[0], a[1] ^ b[1], mask, pos)])
    return total / ((1 << 16) * float(n_table))


def decide(s: float, mask: tuple[int, int]) -> bool:
    return s >= 2.0 ** -(bin(mask[0]).count("1") + bin(mask[1]).count("1"))


def sample_differences(n: int, rounds: int, diff: tuple[int, int], mask, seed: int) -> dict[int, int]:
    """Masked output-difference counts of n real pairs from this module's own
    SPECK and random number stream, for checking a recorded table."""
    rnd = random.Random(seed)
    out: dict[int, int] = {}
    for _ in range(n):
        key = tuple(rnd.getrandbits(16) for _ in range(4))
        ks = key_schedule(key, rounds)
        x0, y0 = rnd.getrandbits(16), rnd.getrandbits(16)
        x1, y1 = x0 ^ diff[0], y0 ^ diff[1]
        for k in ks:
            x0, y0 = round_enc(x0, y0, k)
            x1, y1 = round_enc(x1, y1, k)
        i = table_index(x0 ^ x1, y0 ^ y1, mask)
        out[i] = out.get(i, 0) + 1
    return out


def table_check(table, n_table: int, *, rounds: int, diff, mask, n_sample: int = 100_000,
                seed: int = 20261105, top: int = 200) -> dict:
    """Pearson chi-square of an independent sample against the recorded table
    over its `top` most frequent cells with expected count >= 5, plus one pooled
    remainder cell. With no admissible cell the check cannot pass."""
    sample = sample_differences(n_sample, rounds, diff, mask, seed)
    t = np.asarray(table)
    nz = np.flatnonzero(t)
    ranked = [int(i) for i in nz[np.argsort(t[nz], kind="stable")[::-1][:top]]]
    cells = [i for i in ranked if n_sample * int(table[i]) / n_table >= 5]
    if not cells:
        return {"rounds": rounds, "n_sample": n_sample, "cells": 0, "chi2": None, "df": 0,
                "p_value": 0.0, "seed": seed, "note": "no cell has expected count >= 5"}
    exp = [n_sample * int(table[i]) / n_table for i in cells]
    obs = [sample.get(i, 0) for i in cells]
    rest_e, rest_o = n_sample - sum(exp), n_sample - sum(obs)
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)
    if rest_e > 0:
        stat += (rest_o - rest_e) ** 2 / rest_e
    df = len(cells)
    return {"rounds": rounds, "n_sample": n_sample, "cells": len(cells), "chi2": stat, "df": df,
            "p_value": float(chi2.sf(stat, df)), "seed": seed}
