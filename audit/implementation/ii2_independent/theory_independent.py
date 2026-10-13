"""
M7 - INDEPENDENT IMPLEMENTATION OF THE DIFFERENTIAL THEORY.

The cryptographic-evidence pillar (CE2, and CE3 whose target is CE2's
quantity discretised) rests entirely on one computation: the XOR-differential
probability of modular addition, xdp+(alpha, beta -> gamma), chained along a
realised trail. In the audited tree that computation exists exactly once, in
audit/cryptography/gohr/speck.py, as the Lipmaa-Moriai closed form. A single
implementation of the quantity a claim depends on is a single point of
failure.

This module implements the same quantity by a DIFFERENT ALGORITHM. It is not
a re-typing of the closed form.

    Lipmaa-Moriai (FSE 2001) gives a closed form: a feasibility mask test,
    then 2^-popcount(...). One formula, evaluated in O(1) word operations.

    This module instead evaluates a CARRY-DIFFERENCE AUTOMATON. Addition is
    a sequential circuit whose only state is the carry. Writing c_i for the
    carry into bit i of x + y and c'_i for the carry of (x^alpha)+(y^beta),
    the output difference at bit i is

        gamma_i = alpha_i ^ beta_i ^ delta_i,     delta_i = c_i ^ c'_i,

    so the carry-difference trajectory delta_0..delta_{n-1} is FORCED by
    (alpha, beta, gamma), with delta_0 = 0 because neither addition has a
    carry into bit 0. The probability is then obtained by propagating a
    distribution over the carry state c_i across bit positions, summing over
    the four equally likely (x_i, y_i) at each step and discarding any
    transition whose realised carry difference contradicts the forced
    delta_{i+1}. The carry out of the top bit is discarded, so no constraint
    is applied there.

The two routes share no code, no intermediate quantity and no reasoning step
beyond the definition of xdp+ itself, which is the point.

Also provided:
  * `xdp_plus_bruteforce` - exact enumeration over all (x, y) for small word
    sizes. This is ground truth, not a third model: it IS the definition.
  * `independent_trail_targets` - regenerates the CE2 target end to end using
    the independent Speck implementation (ii2_independent.speck_independent)
    and the automaton above, so the whole CE2 input can be reproduced without
    executing any audited cryptography code.

Nothing here imports audit.cryptography. The comparison against the audited
implementation lives in verify_m7.py, exactly as M1 keeps its comparator
separate from the independent implementation.
"""

from __future__ import annotations

import numpy as np

THEORY_IMPLEMENTATION_ID = "m7-carry-automaton-1"


# ---------------------------------------------------------------------------
# core: carry-difference automaton
# ---------------------------------------------------------------------------

def _maj(u, v, w):
    """Majority of three 0/1 values, elementwise."""
    return (u & v) | (u & w) | (v & w)


def xdp_plus_dp(alpha, beta, gamma, word_size: int = 16) -> np.ndarray:
    """
    xdp+(alpha, beta -> gamma) by carry-difference automaton.

    Returns, for each input triple, the probability over uniformly random
    (x, y) of

        ((x + y) mod 2^n) ^ (((x ^ alpha) + (y ^ beta)) mod 2^n) == gamma.

    Vectorised over numpy arrays of integer word values. Exact up to
    floating-point accumulation: every transition weight is a power of two,
    so for word sizes used here the arithmetic is exact in float64.
    """
    alpha = np.asarray(alpha, dtype=np.int64)
    beta = np.asarray(beta, dtype=np.int64)
    gamma = np.asarray(gamma, dtype=np.int64)
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    n = alpha.size
    shape = alpha.shape
    a = alpha.reshape(-1); b = beta.reshape(-1); g = gamma.reshape(-1)

    bit = lambda arr, i: ((arr >> i) & 1).astype(np.uint8)        # noqa: E731
    delta = [bit(a, i) ^ bit(b, i) ^ bit(g, i) for i in range(word_size)]

    # dp[:, c] = P(carry into current bit position == c, trajectory so far
    # consistent with the forced carry differences). Both additions start
    # with carry 0, so the only admissible initial state is c_0 = 0.
    dp = np.zeros((n, 2), dtype=np.float64)
    dp[:, 0] = 1.0

    for i in range(word_size):
        a_i, b_i, d_i = bit(a, i), bit(b, i), delta[i]
        d_next = delta[i + 1] if i + 1 < word_size else None
        nxt = np.zeros_like(dp)
        for c in (0, 1):
            p = dp[:, c]
            if not p.any():
                continue
            c_arr = np.full(n, c, dtype=np.uint8)
            cp = c_arr ^ d_i                      # carry of the second addition
            for x in (0, 1):
                for y in (0, 1):
                    xa = np.uint8(x) ^ a_i
                    yb = np.uint8(y) ^ b_i
                    c_out = _maj(np.uint8(x), np.uint8(y), c_arr)
                    cp_out = _maj(xa, yb, cp)
                    contrib = p * 0.25
                    if d_next is not None:
                        ok = (c_out ^ cp_out) == d_next
                        contrib = np.where(ok, contrib, 0.0)
                    # scatter by realised carry-out state
                    for state in (0, 1):
                        sel = c_out == state
                        if sel.any():
                            nxt[sel, state] += contrib[sel]
        dp = nxt

    prob = dp.sum(axis=1)
    # delta_0 must be zero: neither addition has a carry into the lowest bit,
    # so gamma_0 is forced to alpha_0 ^ beta_0.
    prob = np.where(delta[0] == 0, prob, 0.0)
    return prob.reshape(shape)


def xdp_plus_bruteforce(alpha: int, beta: int, gamma: int,
                        word_size: int) -> float:
    """
    Exact xdp+ by enumerating every (x, y). This is the DEFINITION, used as
    ground truth for small word sizes; 4^word_size operations, so keep
    word_size <= 8.
    """
    m = (1 << word_size) - 1
    x = np.arange(1 << word_size, dtype=np.int64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    out = ((X + Y) & m) ^ (((X ^ alpha) + (Y ^ beta)) & m)
    return float((out == gamma).mean())


# ---------------------------------------------------------------------------
# independent end-to-end regeneration of the CE2 target
# ---------------------------------------------------------------------------

def independent_trail_targets(keys, p0l, p0r, *, rounds: int, differential,
                              word_size: int = 16):
    """
    Regenerate CE2's input end to end from supplied cryptographic material,
    using the INDEPENDENT Speck implementation and the automaton above.

    Returns (X, target, factors):
        X        the 64-bit neural input encoding of the ciphertext pair
        target   the single-trail probability of each sample's realised trail
        factors  the per-round xdp+ factors, shape (rounds, n)

    Supplying the material explicitly (rather than drawing it internally)
    is what makes this comparable with the audited path: both implementations
    can be driven with identical keys and plaintexts.
    """
    from audit.implementation.ii2_independent import speck_independent as SI

    mask = (1 << word_size) - 1
    alpha_rot, beta_rot = SI.ALPHA, SI.BETA

    keys = np.asarray(keys, dtype=np.uint16)
    p0l = np.asarray(p0l, dtype=np.uint16)
    p0r = np.asarray(p0r, dtype=np.uint16)
    p1l = (p0l ^ np.uint16(differential[0])).astype(np.uint16)
    p1r = (p0r ^ np.uint16(differential[1])).astype(np.uint16)

    round_keys = SI.expand_key(keys, rounds)
    x0, y0 = p0l.copy(), p0r.copy()
    x1, y1 = p1l.copy(), p1r.copy()

    factors = []
    for k in round_keys:
        # Operands of the modular addition in this round, for both members.
        a0 = SI.rotate_right(x0, alpha_rot).astype(np.int64)
        a1 = SI.rotate_right(x1, alpha_rot).astype(np.int64)
        b0 = y0.astype(np.int64)
        b1 = y1.astype(np.int64)
        gamma = (((a0 + b0) & mask) ^ ((a1 + b1) & mask)).astype(np.int64)
        factors.append(xdp_plus_dp(a0 ^ a1, b0 ^ b1, gamma, word_size=word_size))
        x0, y0 = SI.round_encrypt((x0, y0), k)
        x1, y1 = SI.round_encrypt((x1, y1), k)

    F = np.asarray(factors)
    feasible = (F > 0).all(axis=0)
    log2p = np.log2(np.where(F > 0, F, 1.0)).sum(axis=0)
    target = np.where(feasible, np.exp2(log2p), 0.0)
    X = SI.convert_to_binary([x0, y0, x1, y1])
    return X, target, F
