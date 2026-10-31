"""
Model-free differential theory for AutoND CE2.

XOR-differential probability of addition modulo 2^n,

    xdp+(alpha, beta -> gamma) = Pr_{x,y} [ (x + y) ^ ((x ^ alpha) + (y ^ beta)) = gamma ],

over uniformly distributed n-bit operands x, y. Three independent code paths are provided:

1. `lm_weight`       Lipmaa-Moriai closed form (FSE 2001, Theorem 1). For n-bit words with
                     eq(a, b, c) = (~a ^ b) & (~a ^ c) (bit set iff a_i = b_i = c_i) and `<<`
                     a shift (not a rotation) truncated to n bits:
                         xdp+ = 0                               if eq(a<<1, b<<1, g<<1) & (a ^ b ^ g ^ (b<<1)) != 0
                         xdp+ = 2^-wt(~eq(a, b, g) & (2^(n-1) - 1))   otherwise.
                     The most significant bit is excluded from the weight because its carry-out is
                     discarded by the modular reduction.
2. `carry_count`     Carry-pair automaton. Both additions x + y and x' + y' (x' = x ^ alpha,
                     y' = y ^ beta) are evaluated bit by bit from the least significant bit, tracking
                     the carry pair (c, c') in {0,1}^2. At bit i every operand-bit choice (x_i, y_i)
                     is enumerated; the sum bits are s = x_i ^ y_i ^ c and s' = x'_i ^ y'_i ^ c', and
                     a path survives iff s ^ s' = gamma_i. Carries update by majority. The carry out
                     of bit n-1 is discarded. The number of surviving paths is the exact number of
                     operand pairs (x, y) producing gamma; xdp+ = count / 4^n. Nothing from path 1
                     is used.
3. `literal_histogram` / `literal_count`
                     Direct enumeration of every operand pair.

Every probability is dyadic. Results are exchanged as an integer WEIGHT k with
xdp+ = 2^-k (k = -1 marks an impossible transition), so all comparisons are exact.

This module depends only on NumPy. It never touches a model, a dataset or the runner.
"""

from __future__ import annotations

import numpy as np

IMPOSSIBLE = -1

_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)


def _popcount(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.uint64)
    out = np.zeros(v.shape, dtype=np.int64)
    while np.any(v):
        out += _POPCOUNT16[(v & np.uint64(0xFFFF)).astype(np.int64)]
        v = v >> np.uint64(16)
    return out


def _as_words(*arrays):
    return [np.asarray(a, dtype=np.uint64) for a in arrays]


def lm_weight(alpha, beta, gamma, n: int) -> np.ndarray:
    """Lipmaa-Moriai weight k (xdp+ = 2^-k), or IMPOSSIBLE. Vectorised over equal-shape inputs."""
    if not 1 <= n <= 32:
        raise ValueError(f"word size {n} outside 1..32")
    a, b, g = _as_words(alpha, beta, gamma)
    mask = np.uint64((1 << n) - 1)
    one = np.uint64(1)
    if np.any((a | b | g) & ~mask):
        raise ValueError("difference wider than the word size")

    def eq(x, y, z):
        return (~x ^ y) & (~x ^ z) & mask

    a1, b1, g1 = (a << one) & mask, (b << one) & mask, (g << one) & mask
    impossible = (eq(a1, b1, g1) & (a ^ b ^ g ^ b1)) != 0
    weight = _popcount(~eq(a, b, g) & np.uint64((1 << (n - 1)) - 1))
    return np.where(impossible, IMPOSSIBLE, weight).astype(np.int64)


def carry_count(alpha, beta, gamma, n: int) -> np.ndarray:
    """Exact number of operand pairs (x, y) in [0, 2^n)^2 with output difference gamma (automaton)."""
    if not 1 <= n <= 31:
        raise ValueError(f"word size {n} outside 1..31 (counts must fit in int64)")
    a, b, g = (np.asarray(v, dtype=np.int64) for v in (alpha, beta, gamma))
    # state index 2*c + c'; counts of surviving operand-bit paths
    state = np.zeros(a.shape + (4,), dtype=np.int64)
    state[..., 0] = 1
    for i in range(n):
        ai, bi, gi = (a >> i) & 1, (b >> i) & 1, (g >> i) & 1
        new = np.zeros_like(state)
        for c in (0, 1):
            for cp in (0, 1):
                current = state[..., 2 * c + cp]
                for xi in (0, 1):
                    for yi in (0, 1):
                        xpi, ypi = xi ^ ai, yi ^ bi
                        s = xi ^ yi ^ c
                        sp = xpi ^ ypi ^ cp
                        keep = (s ^ sp) == gi
                        nc = (xi & yi) | (xi & c) | (yi & c)
                        ncp = (xpi & ypi) | (xpi & cp) | (ypi & cp)
                        idx = 2 * nc + ncp
                        contrib = np.where(keep, current, 0)
                        for k in range(4):
                            new[..., k] += np.where(idx == k, contrib, 0)
        state = new
    return state.sum(axis=-1)


def weight_from_count(count, n: int) -> np.ndarray:
    """Exact weight from an operand-pair count; IMPOSSIBLE for 0. Raises if a count is not dyadic."""
    count = np.asarray(count, dtype=np.int64)
    out = np.full(count.shape, IMPOSSIBLE, dtype=np.int64)
    pos = count > 0
    if np.any(pos & ((count & (count - 1)) != 0)):
        raise ArithmeticError("operand-pair count that is not a power of two")
    exp = np.zeros_like(count)
    exp[pos] = np.round(np.log2(count[pos].astype(np.float64))).astype(np.int64)
    if np.any(pos & (np.left_shift(np.ones_like(count), exp) != count)):
        raise ArithmeticError("log2 of a dyadic count was not exact")
    out[pos] = 2 * n - exp[pos]
    return out


def literal_histogram(alpha: int, beta: int, n: int) -> np.ndarray:
    """Counts of every output difference for one (alpha, beta), over all 4^n operand pairs (n <= 12)."""
    if n > 12:
        raise ValueError("use literal_count for word sizes above 12")
    mask = (1 << n) - 1
    x = np.arange(1 << n, dtype=np.int64)[:, None]
    y = np.arange(1 << n, dtype=np.int64)[None, :]
    d = ((x + y) & mask) ^ (((x ^ alpha) + (y ^ beta)) & mask)
    return np.bincount(d.ravel(), minlength=1 << n).astype(np.int64)


def literal_count(alpha: int, beta: int, gamma: int, n: int, chunk_bits: int = 8) -> int:
    """Count of operand pairs giving gamma, by direct enumeration of all 4^n pairs (chunked)."""
    mask = (1 << n) - 1
    y = np.arange(1 << n, dtype=np.int64)[None, :]
    y_beta = y ^ beta
    total = 0
    step = 1 << min(chunk_bits, n)
    for x0 in range(0, 1 << n, step):
        x = np.arange(x0, x0 + step, dtype=np.int64)[:, None]
        d = ((x + y) & mask) ^ (((x ^ alpha) + y_beta) & mask)
        total += int(np.count_nonzero(d == gamma))
    return total


def log2_target(weights: np.ndarray) -> np.ndarray:
    """T = sum_r log2 xdp+ = -sum_r k_r (exact integers). Refuses any impossible factor."""
    w = np.asarray(weights, dtype=np.int64)
    if np.any(w < 0):
        raise ArithmeticError("an observed transition has xdp+ = 0; trajectory or convention error")
    return -w.sum(axis=-1)


# ----------------------------------------------------------------- correctness gates


def gate_c_exhaustive(n: int) -> dict:
    """Closed form vs automaton over every (alpha, beta, gamma) for word size n."""
    size = 1 << n
    g = np.arange(size, dtype=np.int64)
    checked = mismatches = possible = 0
    first = None
    for alpha in range(size):
        a = np.full(size * size, alpha, dtype=np.int64)
        b = np.repeat(np.arange(size, dtype=np.int64), size)
        gg = np.tile(g, size)
        lm = lm_weight(a, b, gg, n)
        au = weight_from_count(carry_count(a, b, gg, n), n)
        bad = lm != au
        checked += a.size
        possible += int(np.count_nonzero(lm >= 0))
        if np.any(bad):
            mismatches += int(np.count_nonzero(bad))
            if first is None:
                i = int(np.argmax(bad))
                first = {"alpha": alpha, "beta": int(b[i]), "gamma": int(gg[i]),
                         "lm": int(lm[i]), "automaton": int(au[i])}
    return {"gate": "C", "word_size": n, "coverage": "exhaustive over all (alpha, beta, gamma)",
            "transitions_checked": checked, "possible_transitions": possible,
            "mismatches": mismatches, "first_mismatch": first, "passed": mismatches == 0}


def _selected_pairs(n: int, n_random: int, seed: int) -> list[tuple[int, int]]:
    mask = (1 << n) - 1
    msb, alt = 1 << (n - 1), int("10" * n, 2) & mask
    pats = {0, 1, mask, msb, msb | 1, alt, alt ^ mask, (1 << (n // 2)) - 1, mask ^ 1, 3, 3 << (n - 2)}
    pats |= {1 << i for i in range(n)}
    pairs = {(p, q) for p in pats for q in pats}
    rng = np.random.Generator(np.random.PCG64(seed))
    pairs |= {(int(a), int(b)) for a, b in rng.integers(0, mask + 1, size=(n_random, 2))}
    return sorted(pairs)


def gate_d_literal(n: int, *, exhaustive: bool, n_random: int = 64, seed: int = 0) -> dict:
    """Literal operand enumeration vs closed form vs automaton, comparing full gamma histograms."""
    size = 1 << n
    pairs = ([(a, b) for a in range(size) for b in range(size)] if exhaustive
             else _selected_pairs(n, n_random, seed))
    g = np.arange(size, dtype=np.int64)
    mismatches = 0
    first = None
    for a, b in pairs:
        hist = literal_histogram(a, b, n)
        lit = weight_from_count(hist, n)
        aa, bb = np.full(size, a, dtype=np.int64), np.full(size, b, dtype=np.int64)
        lm = lm_weight(aa, bb, g, n)
        au = weight_from_count(carry_count(aa, bb, g, n), n)
        bad = (lit != lm) | (lit != au)
        if np.any(bad):
            mismatches += int(np.count_nonzero(bad))
            if first is None:
                i = int(np.argmax(bad))
                first = {"alpha": a, "beta": b, "gamma": i, "literal": int(lit[i]),
                         "lm": int(lm[i]), "automaton": int(au[i])}
    coverage = ("exhaustive: every (alpha, beta), full output-difference histogram" if exhaustive else
                f"selected: {len(pairs)} deterministic (alpha, beta) pairs (structured patterns plus "
                f"{n_random} PCG64(seed={seed}) draws), each with its full output-difference histogram; "
                "NOT exhaustive over the transition space")
    return {"gate": "D", "word_size": n, "coverage": coverage, "alpha_beta_pairs": len(pairs),
            "transitions_checked": len(pairs) * size, "mismatches": mismatches,
            "first_mismatch": first, "passed": mismatches == 0}


def gate_d_word16(transitions) -> dict:
    """Literal 2^32-pair enumeration for selected 16-bit (alpha, beta, gamma) transitions."""
    rows = []
    for a, b, g in transitions:
        a, b, g = int(a), int(b), int(g)
        lit = int(weight_from_count(np.int64(literal_count(a, b, g, 16)), 16))
        lm = int(lm_weight(np.int64(a), np.int64(b), np.int64(g), 16))
        au = int(weight_from_count(carry_count(np.int64(a), np.int64(b), np.int64(g), 16), 16))
        rows.append({"alpha": a, "beta": b, "gamma": g, "literal": lit, "lm": lm, "automaton": au,
                     "agree": lit == lm == au})
    return {"gate": "D", "word_size": 16,
            "coverage": f"{len(rows)} selected trajectory transitions, each by literal enumeration of all 2^32 operand pairs",
            "transitions": rows, "passed": all(r["agree"] for r in rows)}
