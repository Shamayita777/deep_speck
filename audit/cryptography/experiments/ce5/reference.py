"""
Independent scalar reference for G-CE5: a line-by-line transcription of P1
Algorithms 1 and 2 (ePrint 2021/287, p. 32). Imports nothing from akr.py,
runner.py, analysis.py or audit.cryptography.gohr; SPECK32/64 is written here
from the specification (Beaulieu et al., ePrint 2013/404) and checked against
its published test vector.

Independence is of implementation: this module and akr.py share the reading
of Algorithms 1 and 2 recorded in docs/SOURCE_VERIFICATION.md, including the
masked lookup in Algorithm 2 L9. The table check guards the recorded table
against round-count, word-order and input-difference errors.

    rotations      x <<< a, n = 16, alpha = 7, beta = 2
    round          x <- ((x >>> 7) + y) ^ k ;  y <- (y <<< 2) ^ x
    key schedule   (l2, l1, l0, k0); l_{i+3} = (k_i + (l_i >>> 7)) ^ i ; k_{i+1} = (k_i <<< 2) ^ l_{i+3}
    test vector    key 1918 1110 0908 0100, pt 6574 694c -> ct a868 42f2
    cell           32-bit value (dl || dv) & m, stored at
                   pext(dl, m_l) << HW(m_v) | pext(dv, m_v)
"""

from __future__ import annotations

import random

import numpy as np
from scipy.stats import chi2

W = 0xFFFF
ALPHA, BETA = 7, 2
TEST_VECTOR = {"key": (0x1918, 0x1110, 0x0908, 0x0100), "pt": (0x6574, 0x694C), "ct": (0xA868, 0x42F2)}
N_CANDIDATES = 1 << 12


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


def cell(dl: int, dv: int, mask: tuple[int, int], _pos=None) -> int:
    pl, pv = _pos or (_positions(mask[0]), _positions(mask[1]))
    left = sum(((dl >> p) & 1) << j for j, p in enumerate(pl))
    right = sum(((dv >> p) & 1) << j for j, p in enumerate(pv))
    return (left << len(pv)) | right


def deltas(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Algorithm 1 L9-10 / Algorithm 2 L7-8."""
    return a[0] ^ b[0], a[0] ^ a[1] ^ b[0] ^ b[1]


def build_approx_ddt(nr: int, n_pairs: int, mask, seed: int, diff=(0x0040, 0x0000)) -> dict[int, int]:
    """Algorithm 1 with integer counts (L11's 1/n applied at lookup)."""
    rnd = random.Random(seed)
    pos = (_positions(mask[0]), _positions(mask[1]))
    out: dict[int, int] = {}
    for _ in range(n_pairs):
        p = (rnd.getrandbits(16), rnd.getrandbits(16))
        q = (p[0] ^ diff[0], p[1] ^ diff[1])
        key = tuple(rnd.getrandbits(16) for _ in range(4))
        c = cell(*deltas(encrypt(p, key, nr), encrypt(q, key, nr)), mask, pos)
        out[c] = out.get(c, 0) + 1
    return out


def score(C: tuple[int, int], Cp: tuple[int, int], table, n_table: int, mask: tuple[int, int],
          candidates=range(N_CANDIDATES)) -> float:
    """Algorithm 2 L3-10 and the division of L11."""
    pos = (_positions(mask[0]), _positions(mask[1]))
    total, nk = 0, 0
    for key in candidates:
        a = round_dec(C[0], C[1], key)
        b = round_dec(Cp[0], Cp[1], key)
        total += int(table[cell(*deltas(a, b), mask, pos)])
        nk += 1
    return total / (nk * float(n_table))


def decide(s: float, mask: tuple[int, int]) -> bool:
    """Algorithm 2 L11, strict."""
    return s > 2.0 ** -(bin(mask[0]).count("1") + bin(mask[1]).count("1"))


def table_check(table, n_table: int, *, rounds: int, diff, mask, n_sample: int = 100_000,
                seed: int = 20261105, top: int = 200) -> dict:
    """Pearson chi-square of an independent Algorithm-1 sample against the
    recorded table over its `top` most frequent cells with expected count >= 5,
    plus one pooled remainder cell. With no admissible cell the check cannot pass."""
    sample = build_approx_ddt(rounds, n_sample, mask, seed, diff)
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
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    if rest_e > 0:
        stat += (rest_o - rest_e) ** 2 / rest_e
    df = len(cells)
    return {"rounds": rounds, "n_sample": n_sample, "cells": len(cells), "chi2": stat, "df": df,
            "p_value": float(chi2.sf(stat, df)), "seed": seed}
