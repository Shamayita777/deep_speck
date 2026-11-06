"""
G-CE5 baseline: the average-key-rank (aKR) distinguisher of Benamira, Gerault,
Peyrin and Tan (EUROCRYPT 2021, Sect. 4.5; long version ePrint 2021/287) for
5-round SPECK32/64, and a masked ciphertext-difference score used as a
secondary comparator. docs/SOURCE_VERIFICATION.md records which details are
verified against which source; docs/CRYPTO_CORRECTNESS.md gives the derivations.

Notation (Gohr's conventions, audit.cryptography.gohr.speck): a state is
(l, r), 16-bit words; one round is l <- ((l >>> 7) + r) ^ k, r <- (r <<< 2) ^ l.
A ciphertext pair is (C0, C1) = ((c0l, c0r), (c1l, c1r)).

    table    T[d] = #{real (Nr-1)-round pairs with (X0 ^ X1) & M = d}, from
             n_table pairs with input difference (0x0040, 0x0000)
    decrypt  for a last-round subkey k: X = dec_one_round(C, k) for C0 and C1
    score    s(C0, C1) = 2^-16 * sum_k T[(X0(k) ^ X1(k)) & M] / n_table
    decide   real iff s >= 2^-|M|                       (|M| = HW(M) = 26)

Exact subkey reduction. After one-round decryption the right word
r' = (l ^ r) >>> 2 does not depend on k, and l' = (((l ^ k) - r') mod 2^16) <<< 7.
Output bit j of l' is bit (j - 7) mod 16 of the difference s = (l ^ k) - r'. A
subtraction's bit b depends only on bits 0..b of its operands, so if every
masked output bit maps to s-bits <= h, key bits above h never change the masked
difference. For M_L = 0xff8f, h = 12: the 2^16 subkeys fall into 2^13 classes
of 8 with identical masked decryptions, and the average over 2^16 equals the
average over one representative per class exactly (integer sums, the same real
quotient). relevant_key_bits() derives h from the mask, so no constant is
assumed; tests check equality with the full 2^16 enumeration.

Counts are uint32 (n_table = 10^7 < 2^32); sums over subkeys are uint64
(<= 2^16 * 10^7 < 2^40). The table holds 2^26 uint32 = 256 MiB.
"""

from __future__ import annotations

from os import urandom

import numpy as np

from audit.cryptography.gohr import speck as S

MASK = (0xFF8F, 0xFF8F)
N_SUBKEYS = 1 << 16


def mask_weight(mask=MASK) -> int:
    return sum(bin(m).count("1") for m in mask)


def relevant_key_bits(mask_l: int = MASK[0]) -> int:
    """Number of low subkey bits that can change (X0 ^ X1) & M after one-round
    decryption; the remaining high bits are provably irrelevant."""
    needed = [(j - S.ALPHA()) % 16 for j in range(16) if (mask_l >> j) & 1]
    return max(needed) + 1


def subkey_representatives(mask=MASK) -> np.ndarray:
    return np.arange(1 << relevant_key_bits(mask[0]), dtype=np.uint16)


def _compaction_lut(m: int) -> np.ndarray:
    """lut[v] packs the bits of v at the set positions of m, lowest first."""
    pos = [i for i in range(16) if (m >> i) & 1]
    v = np.arange(1 << 16, dtype=np.uint32)
    out = np.zeros(1 << 16, dtype=np.uint32)
    for j, p in enumerate(pos):
        out |= ((v >> p) & 1) << j
    return out


class MaskedIndex:
    """Table layout: index = pext(dl, M_L) << HW(M_R) | pext(dr, M_R)."""

    def __init__(self, mask=MASK):
        self.mask = tuple(int(m) for m in mask)
        self.lut_l = _compaction_lut(self.mask[0])
        self.lut_r = _compaction_lut(self.mask[1])
        self.shift = bin(self.mask[1]).count("1")
        self.size = 1 << mask_weight(self.mask)

    def __call__(self, dl, dr) -> np.ndarray:
        return (self.lut_l[dl] << self.shift) | self.lut_r[dr]


def _pairs(n: int, rounds: int, diff, rng, real: bool):
    if rng is None:
        keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
        p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        q = np.frombuffer(urandom(4 * n), dtype=np.uint16).reshape(2, -1)
    else:
        keys = rng.integers(0, 1 << 16, size=(4, n), dtype=np.uint16)
        p0l = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)
        p0r = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)
        q = rng.integers(0, 1 << 16, size=(2, n), dtype=np.uint16)
    if real:
        p1l, p1r = p0l ^ np.uint16(diff[0]), p0r ^ np.uint16(diff[1])
    else:
        p1l, p1r = q[0], q[1]
    ks = S.expand_key(keys, rounds)
    c0l, c0r = S.encrypt((p0l, p0r), ks)
    c1l, c1r = S.encrypt((p1l, p1r), ks)
    return c0l, c0r, c1l, c1r


def ciphertext_pairs(n: int, rounds: int, diff, *, real: bool, rng=None):
    """Gohr's construction: one random key per pair; real pairs carry `diff`,
    random pairs encrypt an independent random second plaintext.
    rng=None draws from os.urandom (production); rng is for tests."""
    return _pairs(n, rounds, diff, rng, real)


def build_table(n_pairs: int, rounds: int, diff, *, index: MaskedIndex | None = None,
                rng=None, chunk: int = 1 << 20) -> np.ndarray:
    """Counts of masked output differences of `rounds`-round real pairs."""
    index = index or MaskedIndex()
    if n_pairs >= 1 << 32:
        raise ValueError("uint32 counts would overflow")
    counts = np.zeros(index.size, dtype=np.uint32)
    done = 0
    while done < n_pairs:
        m = min(chunk, n_pairs - done)
        c0l, c0r, c1l, c1r = _pairs(m, rounds, diff, rng, True)
        np.add.at(counts, index(c0l ^ c1l, c0r ^ c1r), 1)
        done += m
    return counts


def akr_scores(c0l, c0r, c1l, c1r, counts: np.ndarray, n_table: int, *,
               index: MaskedIndex | None = None, subkeys=None,
               pair_chunk: int = 256) -> np.ndarray:
    """s(C0, C1) as a probability. subkeys=None uses the exact class
    representatives; pass np.arange(2**16) for the literal enumeration."""
    index = index or MaskedIndex()
    k = (subkey_representatives(index.mask) if subkeys is None
         else np.asarray(subkeys, dtype=np.uint16))[None, :]
    c0l, c0r, c1l, c1r = (np.asarray(a, dtype=np.uint16) for a in (c0l, c0r, c1l, c1r))
    out = np.empty(c0l.size, dtype=np.float64)
    for s in range(0, c0l.size, pair_chunk):
        e = slice(s, s + pair_chunk)
        r0 = S.ror(c0l[e] ^ c0r[e], S.BETA())
        r1 = S.ror(c1l[e] ^ c1r[e], S.BETA())
        l0 = S.rol((c0l[e, None] ^ k) - r0[:, None], S.ALPHA())
        l1 = S.rol((c1l[e, None] ^ k) - r1[:, None], S.ALPHA())
        idx = (index.lut_l[l0 ^ l1] << index.shift) | index.lut_r[r0 ^ r1][:, None]
        out[e] = counts[idx].sum(axis=1, dtype=np.uint64) / (k.shape[1] * float(n_table))
    return out


def akr_decide(scores: np.ndarray, mask=MASK) -> np.ndarray:
    """Published decision: real iff the average score is at least 2^-|M|."""
    return np.asarray(scores) >= 2.0 ** -mask_weight(mask)


def difference_scores(c0l, c0r, c1l, c1r, counts: np.ndarray, n_table: int, *,
                      index: MaskedIndex | None = None) -> np.ndarray:
    """Secondary comparator: probability of the masked ciphertext difference
    under the Nr-round table, with no key guessing. Not a published distinguisher."""
    index = index or MaskedIndex()
    return counts[index(np.asarray(c0l ^ c1l, dtype=np.uint16),
                        np.asarray(c0r ^ c1r, dtype=np.uint16))] / float(n_table)


def words_from_binary(X: np.ndarray):
    """Inverse of gohr.speck.convert_to_binary for 64-bit pair encodings."""
    X = np.asarray(X, dtype=np.uint32).reshape(-1, 4, 16)
    w = (X << np.arange(15, -1, -1, dtype=np.uint32)).sum(axis=2)
    return tuple(w[:, i].astype(np.uint16) for i in range(4))
