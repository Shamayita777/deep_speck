"""
G-CE5 baseline: the average-key-rank differential distinguisher of Benamira,
Gerault, Peyrin and Tan, "A Deeper Look at Machine Learning-Based
Cryptanalysis" (ePrint 2021/287; P1). Section 4.5 (pp. 17-18), Supplementary
Algorithms 1 and 2 (p. 32), Table 7 (p. 19). docs/SOURCE_VERIFICATION.md holds
the verbatim transcription; each step below cites its line.

Algorithm 1, BuildApproxDDT(nr, np, m)                      -> build_table
    L2   aDDT <- ZEROS(2^32)
    L4-6 p <- rand(0, 2^32-1); p' <- p xor 0x0040/0000; key <- rand(0, 2^64-1)
    L7-8 (l, r) <- ENC_key(p, nr); (l', r') <- ENC_key(p', nr)
    L9   dl <- l xor l'
    L10  dv <- l xor r xor l' xor r'
    L11  aDDT[(dl || dv) & m] += 1/n

Algorithm 2, AKRDD(nr, C, C', m)                            -> akr_scores, akr_decide
    L2   aDDT <- BuildApproxDDT(nr - 1, 10^7, m)
    L4   for key = 0 to 2^12 - 1
    L5-6 (l, r) <- DEC_key(C, 1); (l', r') <- DEC_key(C', 1)
    L7-8 dl, dv as in Algorithm 1
    L9   score += aDDT[dl || dv]
    L11  return score / 2^12 > 2^-HW(m)

Representation. Only indices of the form x & m are ever written by L11, so
the 2^32 table is stored losslessly as the 2^26 masked cells (pext of the 13
masked bits of dl, then of dv) holding integer counts; the probability is
count / n, applied after summation, which equals L11's sum of 1/n terms
exactly (no floating accumulation).

Interpretation (recorded, not silent): L9 writes aDDT[dl || dv] without "& m".
The table holds entries only at masked indices, and the text (p. 18) says the
score is "the average of the probabilities of all partial decryptions ... read
from aDDT (nr-1)", so the lookup is taken at (dl || dv) & m, as written to in
Algorithm 1 L11. The gate tests this reading against Table 7.

Candidates. key runs over 0 .. 2^12 - 1 literally (L4) and the sum is divided
by 2^12 (L11). P1 p. 18: "we enumerate the 2^12 keys affecting the rightmost
13 bits of dl_{nr-1} covered by our mask". relevant_key_bits() derives 12 from
the mask (docs/CRYPTO_CORRECTNESS.md section 3); full_average() is the
2^16-subkey average used only by tests to show the equality for this mask and
the inequality for masks where it does not hold.

Conventions: SPECK32/64 from audit.cryptography.gohr.speck (Gohr's code); a
state is (l, r); DEC_key(C, 1) is speck.dec_one_round(C, key); p is the 32-bit
plaintext (l || r). Counts are uint32 (n = 10^7 < 2^32); sums are uint64.
"""

from __future__ import annotations

from os import urandom

import numpy as np

from audit.cryptography.gohr import speck as S

MASK = (0xFF8F, 0xFF8F)
N_CANDIDATES = 1 << 12
CANDIDATES = np.arange(N_CANDIDATES, dtype=np.uint16)
ALGORITHM = "P1 Algorithms 1 and 2 (p. 32), literal 2^12 candidate loop, strict threshold"


def mask_weight(mask=MASK) -> int:
    return sum(bin(m).count("1") for m in mask)


def relevant_key_bits(mask=MASK) -> int:
    """Number of low subkey bits that can change (dl || dv) & m after one-round
    decryption. Bit j of l' is bit (j - 7) mod 16 of s = (l ^ k) - r'; in the
    XOR of two such s the key bit at a position cancels, so difference bit b
    depends only on key bits below b. dv = dl ^ dr' with dr' key-free, so the
    masked dv bits need the same s-bits as masked dl bits at mask[1]'s positions."""
    needed = [(j - S.ALPHA()) % 16 for m in mask for j in range(16) if (m >> j) & 1]
    return max(needed)


def _compaction_lut(m: int) -> np.ndarray:
    """lut[v] packs the bits of v at the set positions of m, lowest first."""
    pos = [i for i in range(16) if (m >> i) & 1]
    v = np.arange(1 << 16, dtype=np.uint32)
    out = np.zeros(1 << 16, dtype=np.uint32)
    for j, p in enumerate(pos):
        out |= ((v >> p) & 1) << j
    return out


class MaskedIndex:
    """Cell of (dl || dv) & m: pext(dl, m_l) << HW(m_v) | pext(dv, m_v)."""

    def __init__(self, mask=MASK):
        self.mask = tuple(int(m) for m in mask)
        self.lut_l = _compaction_lut(self.mask[0])
        self.lut_v = _compaction_lut(self.mask[1])
        self.shift = bin(self.mask[1]).count("1")
        self.size = 1 << mask_weight(self.mask)

    def __call__(self, dl, dv) -> np.ndarray:
        return (self.lut_l[dl] << self.shift) | self.lut_v[dv]


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
    """Gohr's construction (P1 p. 7): real pairs carry `diff`; random pairs
    encrypt an independent random second plaintext under the same key.
    rng=None draws from os.urandom (production); rng is for tests."""
    return _pairs(n, rounds, diff, rng, real)


def differences(c0l, c0r, c1l, c1r):
    """Algorithm 1 L9-10 / Algorithm 2 L7-8."""
    return c0l ^ c1l, (c0l ^ c0r) ^ (c1l ^ c1r)


def build_table(n_pairs: int, rounds: int, diff, *, index: MaskedIndex | None = None,
                rng=None, chunk: int = 1 << 20) -> np.ndarray:
    """Algorithm 1 as integer counts over the masked cells."""
    index = index or MaskedIndex()
    if n_pairs >= 1 << 32:
        raise ValueError("uint32 counts would overflow")
    counts = np.zeros(index.size, dtype=np.uint32)
    done = 0
    while done < n_pairs:
        m = min(chunk, n_pairs - done)
        dl, dv = differences(*_pairs(m, rounds, diff, rng, True))
        np.add.at(counts, index(dl, dv), 1)
        done += m
    return counts


def akr_scores(c0l, c0r, c1l, c1r, counts: np.ndarray, n_table: int, *,
               index: MaskedIndex | None = None, candidates=CANDIDATES,
               pair_chunk: int = 256) -> np.ndarray:
    """Algorithm 2 L3-10 and the division of L11: score / 2^12, continuous.
    `candidates` other than CANDIDATES is for tests only."""
    index = index or MaskedIndex()
    k = np.asarray(candidates, dtype=np.uint16)[None, :]
    c0l, c0r, c1l, c1r = (np.asarray(a, dtype=np.uint16) for a in (c0l, c0r, c1l, c1r))
    out = np.empty(c0l.size, dtype=np.float64)
    for s in range(0, c0l.size, pair_chunk):
        e = slice(s, s + pair_chunk)
        r0 = S.ror(c0l[e] ^ c0r[e], S.BETA())
        r1 = S.ror(c1l[e] ^ c1r[e], S.BETA())
        l0 = S.rol((c0l[e, None] ^ k) - r0[:, None], S.ALPHA())
        l1 = S.rol((c1l[e, None] ^ k) - r1[:, None], S.ALPHA())
        dl = l0 ^ l1
        dv = dl ^ (r0 ^ r1)[:, None]
        idx = (index.lut_l[dl] << index.shift) | index.lut_v[dv]
        out[e] = counts[idx].sum(axis=1, dtype=np.uint64) / (k.shape[1] * float(n_table))
    return out


def akr_decide(scores: np.ndarray, mask=MASK) -> np.ndarray:
    """Algorithm 2 L11: real iff score / 2^12 > 2^-HW(m), strictly."""
    return np.asarray(scores) > 2.0 ** -mask_weight(mask)


def full_average(c0l, c0r, c1l, c1r, counts, n_table, *, index: MaskedIndex | None = None):
    """Average over all 2^16 last-round subkeys. Tests only: not the published loop."""
    return akr_scores(c0l, c0r, c1l, c1r, counts, n_table, index=index,
                      candidates=np.arange(1 << 16, dtype=np.uint16))


def difference_scores(c0l, c0r, c1l, c1r, counts: np.ndarray, n_table: int, *,
                      index: MaskedIndex | None = None) -> np.ndarray:
    """Secondary comparator: probability of the masked (dl || dv) of the
    ciphertexts themselves under an Algorithm-1 table at nr rounds, no key
    guessing. Not a published distinguisher."""
    index = index or MaskedIndex()
    dl, dv = differences(*(np.asarray(a, dtype=np.uint16) for a in (c0l, c0r, c1l, c1r)))
    return counts[index(dl, dv)] / float(n_table)


def words_from_binary(X: np.ndarray):
    """Inverse of gohr.speck.convert_to_binary for 64-bit pair encodings."""
    X = np.asarray(X, dtype=np.uint32).reshape(-1, 4, 16)
    w = (X << np.arange(15, -1, -1, dtype=np.uint32)).sum(axis=2)
    return tuple(w[:, i].astype(np.uint16) for i in range(4))
