"""
Speck32/64 cipher primitives.

Ported verbatim (identical logic) from archive/speck.py, which is this
project's faithful reproduction of Gohr's original DeepSpeck
implementation (CRYPTO 2019). This is the task-defining cryptographic
substrate: EV's independence from CE's and D's orchestration code
(see gohr/__init__.py) does not extend to the cipher/data-generation
math itself, which is reused because it defines the task being audited,
not a procedural choice under investigation.

Correctness of this module is verified independently of any EV
experiment via check_testvector(), reproduced from the original source
- see tests/test_gohr_speck_testvector.py.

IMPORTANT: make_train_data() uses os.urandom(), which is NOT seedable
from Python's standard random-number-generator layer. Per this
project's frozen randomness rules, no function in this module accepts
or fabricates a "dataset seed" - callers must record
exact_replay_available=False with an explicit reason (see
gohr.dataset.generate_dataset).
"""

from __future__ import annotations

from os import urandom

import numpy as np


def WORD_SIZE() -> int:
    return 16


def ALPHA() -> int:
    return 7


def BETA() -> int:
    return 2


MASK_VAL = 2 ** WORD_SIZE() - 1


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))


def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return c0, c1


def dec_one_round(c, k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return c0, c1


def expand_key(k, t):
    ks = [0 for _ in range(t)]
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[: len(k) - 1]))
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i)
    return ks


def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return x, y


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k)
    return x, y


def check_testvector() -> bool:
    """Gohr's published Speck32/64 test vector, reproduced verbatim."""
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694C)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    return ct == (0xA868, 0x42F2)


def convert_to_binary(arr) -> np.ndarray:
    """
    Convert an array of 4 ciphertext-word arrays [ct0a, ct1a, ct0b, ct1b]
    into a (n, 64) bit-vector array. Word-major, MSB-first within each
    word - identical to archive/speck.py. This is the ORIGINAL Gohr
    representation (H-EV-REPRESENTATION Condition A / baseline). The
    perturbed representation (Condition B) is implemented separately in
    gohr.representation and is NEVER substituted here.
    """
    X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    return X.transpose()


def make_train_data(n: int, nr: int, diff=(0x0040, 0)):
    """
    Baseline training-data generator, ported verbatim from
    archive/speck.py. Uses os.urandom() for every random draw (labels,
    keys, plaintexts) - see module docstring re: exact_replay_available.
    """
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(2 * int(num_rand_samples)), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * int(num_rand_samples)), dtype=np.uint16)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X, Y
