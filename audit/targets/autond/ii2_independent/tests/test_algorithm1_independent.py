"""Tests of paper-stated and definition-derived properties of II-2.

No test asserts a difference, a round count or a score that the paper
reports for a real cipher. The toy ciphers below are test fixtures with
answers that follow from their construction.
"""
import math
import random

import numpy as np
import pytest

from ..algorithm1_independent import (
    PAPER_DEFAULTS,
    PAPER_REDUCED,
    bias_score,
    generate_offspring,
    search,
)
from ..cipher_interface import FunctionCipher, bits_to_int, int_to_bits, validate_cipher

# ---------------------------------------------------------------------------
# Test-only ciphers (not from the paper)
# ---------------------------------------------------------------------------
# SplitMix64 finalizer constants (Steele, Lea & Flood, OOPSLA 2014), used only
# as a pseudo-random function for the test cipher.
_SM_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_M2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(z):
    with np.errstate(over="ignore"):
        z = z + _SM_GAMMA
        z = (z ^ (z >> np.uint64(30))) * _SM_M1
        z = (z ^ (z >> np.uint64(27))) * _SM_M2
        return z ^ (z >> np.uint64(31))


def _rows_to_u64(bits):
    width = bits.shape[1]
    weights = np.uint64(1) << np.arange(width - 1, -1, -1, dtype=np.uint64)
    return (bits.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def _u64_to_rows(values, width):
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)


def random_function_cipher(n=16, out=None, key_bits=32, nominal_rounds=8):
    out = n if out is None else out
    assert n <= 32 and key_bits <= 32 and 1 <= out <= 64

    def enc(pt, keys, rounds):
        v = (_rows_to_u64(keys) << np.uint64(32)) ^ _rows_to_u64(pt)
        h = _splitmix64(_splitmix64(v) ^ np.uint64(rounds))
        return _u64_to_rows(h >> np.uint64(64 - out), out)

    return FunctionCipher("random-fn", n, out, key_bits, enc, nominal_rounds)


def identity_cipher(n=16, nominal_rounds=None):
    return FunctionCipher("identity", n, n, 0, lambda pt, keys, r: pt.copy(), nominal_rounds)


def staged_cipher(n=16, identity_rounds=2, nominal_rounds=6):
    rnd = random_function_cipher(n, key_bits=32)

    def enc(pt, keys, rounds):
        return pt.copy() if rounds <= identity_rounds else rnd.encrypt(pt, keys, rounds)

    return FunctionCipher("staged", n, n, 32, enc, nominal_rounds)


def _never_called(pt, keys, rounds):
    raise AssertionError("encrypt must not be called when a scorer is injected")


STUB64 = FunctionCipher("stub64", 64, 64, 0, _never_called, nominal_rounds=4)


def _constant_scorer(value):
    def scorer(cipher, delta, rounds, samples, rng):
        return value
    return scorer


SMALL = dict(initial_population=64, population_size=8, generations=3, samples=2000, threshold=0.05)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1, 7, 16, 65, 384])
def test_bit_roundtrip_and_msb_first(width):
    rng = np.random.Generator(np.random.PCG64(width))
    for _ in range(20):
        value = int.from_bytes(rng.bytes((width + 7) // 8), "big") & ((1 << width) - 1)
        bits = int_to_bits(value, width)
        assert bits.shape == (width,) and bits.dtype == np.uint8
        assert bits_to_int(bits) == value
    top = int_to_bits(1 << (width - 1), width)
    assert top[0] == 1 and top[1:].sum() == 0


def test_int_to_bits_rejects_out_of_range():
    with pytest.raises(ValueError):
        int_to_bits(16, 4)
    with pytest.raises(ValueError):
        int_to_bits(-1, 4)


def test_validate_cipher_rejects_bad_widths():
    with pytest.raises(ValueError):
        validate_cipher(FunctionCipher("bad", 0, 8, 0, _never_called))
    with pytest.raises(TypeError):
        validate_cipher(FunctionCipher("bad", True, 8, 0, _never_called))


# ---------------------------------------------------------------------------
# Definition 2
# ---------------------------------------------------------------------------
def test_bias_score_of_zero_difference_is_one_half():
    rng = np.random.Generator(np.random.PCG64(1))
    assert bias_score(random_function_cipher(), 0, 3, 500, rng) == 0.5


def test_bias_score_identity_cipher_is_one_half_for_nonzero_delta():
    rng = np.random.Generator(np.random.PCG64(2))
    for delta in (1, 0x8000, 0x1234, 0xFFFF):
        assert bias_score(identity_cipher(16), delta, 1, 300, rng) == 0.5


def test_bias_score_in_range():
    rng = np.random.Generator(np.random.PCG64(3))
    cipher = random_function_cipher()
    for delta in range(1, 30):
        s = bias_score(cipher, delta, 2, 200, rng)
        assert 0.0 <= s <= 0.5


def test_random_baseline_matches_section_5_4():
    """§5.4: for random data at t = 10^4 the mean is about 0.004, and the SD for
    32 bits is 0.00052. Normal approximation: mean 1/sqrt(2*pi*t) = 0.00399,
    SD (1/(2 sqrt t)) sqrt(1 - 2/pi) / sqrt(32) = 0.00053."""
    cipher = random_function_cipher(n=32, out=32)
    rng = np.random.Generator(np.random.PCG64(4))
    deltas = np.random.Generator(np.random.PCG64(5)).integers(1, 2**32, size=200)
    values = np.array([bias_score(cipher, int(d), 1, 10_000, rng) for d in deltas])
    expected_mean = 1.0 / math.sqrt(2 * math.pi * 10_000)
    assert abs(values.mean() - expected_mean) < 3e-4
    assert 4.2e-4 < values.std(ddof=1) < 6.5e-4


def test_cipher_output_is_validated():
    rng = np.random.Generator(np.random.PCG64(6))
    short = FunctionCipher("short", 8, 8, 0, lambda pt, k, r: pt[:, :-1])
    with pytest.raises(ValueError):
        bias_score(short, 1, 1, 10, rng)
    twos = FunctionCipher("twos", 8, 8, 0, lambda pt, k, r: pt * 2)
    with pytest.raises(ValueError):
        bias_score(twos, 1, 1, 10, rng)


# ---------------------------------------------------------------------------
# Algorithm 1, L5-L15
# ---------------------------------------------------------------------------
def _population(size, n, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    values = set()
    while len(values) < size:
        values.add(int.from_bytes(rng.bytes((n + 7) // 8), "big") & ((1 << n) - 1) or 1)
    return sorted(values)


def test_offspring_pairs_count_order_and_mutation_m1():
    pop = _population(32, 64, 7)
    triples = generate_offspring(pop, 64, 1.0, np.random.Generator(np.random.PCG64(8)))
    assert len(triples) == 496                                   # C(32, 2); §8
    assert [(i, j) for i, j, _ in triples] == [(i, j) for i in range(32) for j in range(i + 1, 32)]
    for i, j, child in triples:
        assert (child ^ pop[i] ^ pop[j]).bit_count() == 1         # M = 1: exactly one bit flipped


def test_offspring_without_mutation_is_pure_xor():
    pop = _population(10, 64, 9)
    for i, j, child in generate_offspring(pop, 64, 0.0, np.random.Generator(np.random.PCG64(10))):
        assert child == pop[i] ^ pop[j]


def test_mutation_position_within_width():
    pop = _population(10, 5, 11)
    rng = np.random.Generator(np.random.PCG64(12))
    for _ in range(50):
        for i, j, child in generate_offspring(pop, 5, 1.0, rng):
            assert 0 <= child < 2**5


# ---------------------------------------------------------------------------
# Search: determinism, invariants, termination
# ---------------------------------------------------------------------------
def test_deterministic_under_fixed_seed():
    a = search(staged_cipher(), seed=42, **SMALL)
    b = search(staged_cipher(), seed=42, **SMALL)
    assert a == b


def test_different_seeds_change_the_search():
    a = search(staged_cipher(), seed=1, **SMALL)
    b = search(staged_cipher(), seed=2, **SMALL)
    assert a["per_round"][0]["differences"] != b["per_round"][0]["differences"]


def test_single_round_mode_reproduces_round_of_full_run():
    full = search(staged_cipher(), seed=5, **SMALL)
    for r in (1, 2):
        single = search(staged_cipher(), seed=5, rounds=r, **SMALL)
        assert single["input_differences"] == full["per_round"][r - 1]["differences"]
        assert single["mode"] == "single_round"
        assert "rescored_matrix" not in single


def test_outer_procedure_on_staged_cipher():
    """Identity for rounds <= 2, random afterwards. By construction R = 2."""
    res = search(staged_cipher(), seed=9, **SMALL)
    assert res["terminated_by"] == "threshold"
    assert res["rounds_reached"] == 2
    assert len(res["per_round"]) == 3
    assert [p["passed_threshold"] for p in res["per_round"]] == [True, True, False]
    expected_union = list(dict.fromkeys(res["per_round"][0]["differences"] + res["per_round"][1]["differences"]))
    assert res["input_differences"] == expected_union          # all S equal, so stable order
    assert all(row == [0.5, 0.5] for row in res["rescored_matrix"])
    assert all(s == 1.5 for s in res["score_round_weighted"])   # 0.5*1 + 0.5*2
    assert all(s == 1.0 for s in res["score_uniform"])
    assert res["score_per_difference"] == res["score_round_weighted"]
    uni = search(staged_cipher(), seed=9, round_weighting="uniform", **SMALL)
    assert uni["score_per_difference"] == uni["score_uniform"]


def test_population_invariants():
    res = search(staged_cipher(), seed=13, **SMALL)
    p = SMALL["population_size"]
    for rnd in res["per_round"]:
        diffs = rnd["differences"]
        assert len(diffs) <= p and len(set(diffs)) == len(diffs)
        assert all(1 <= d < 2**16 for d in diffs)
        assert all(size <= p for size in rnd["population_sizes"])
        assert len(rnd["final_candidates"]) <= p * (p - 1) // 2 or rnd["generations_completed"] == 0


def test_tiny_width_never_emits_zero_or_duplicates():
    res = search(identity_cipher(4, nominal_rounds=1), seed=0, generations=2, samples=50)
    rnd = res["per_round"][0]
    assert rnd["zeros_discarded"] > 0 and rnd["duplicates_discarded"] > 0
    assert all(1 <= d <= 15 for d in rnd["differences"])
    assert len(set(rnd["differences"])) == len(rnd["differences"]) <= 15
    assert all(1 <= d <= 15 for d in res["input_differences"])


def test_threshold_comparison_is_strict():
    at = search(STUB64, seed=0, scorer=_constant_scorer(0.05), **SMALL)
    assert at["terminated_by"] == "threshold" and at["rounds_reached"] == 0
    assert at["input_differences"] == [] and at["generations_completed"] == SMALL["generations"]
    above = search(STUB64, seed=0, scorer=_constant_scorer(0.06), max_rounds=2, **SMALL)
    assert above["terminated_by"] == "round_cap" and above["rounds_reached"] == 2
    assert all(v == 0.06 for row in above["rescored_matrix"] for v in row)


def test_round_cap_on_identity_cipher():
    res = search(identity_cipher(16, nominal_rounds=3), seed=0, **SMALL)
    assert res["terminated_by"] == "round_cap" and res["rounds_reached"] == 3


def test_missing_round_cap_raises():
    with pytest.raises(ValueError):
        search(identity_cipher(16, nominal_rounds=None), seed=0, **SMALL)


def test_paper_defaults_and_evaluation_accounting():
    res = search(STUB64, seed=0, scorer=_constant_scorer(0.0))
    rnd = res["per_round"][0]
    assert res["generations_completed"] == 50
    assert all(size == 32 for size in rnd["population_sizes"])
    assert rnd["candidates_generated"] == 1024 + 50 * 496
    assert rnd["evaluations"] == rnd["candidates_generated"] - rnd["zeros_discarded"] - rnd["duplicates_discarded"]
    assert res["evaluations"] == rnd["evaluations"]             # R = 0: no rescoring
    for name, value in PAPER_DEFAULTS.items():
        assert res["params_used"][name] == value
        assert res["paper_parameters"][name] == value
    assert set(res["paper_parameters"]) == set(res["paper_parameter_sources"])
    assert "deduplicate" not in res["paper_parameters"]         # not from the paper
    assert "round_weighting" not in res["paper_parameters"]     # II-2's choice (A-017)
    assert "encryptions" not in res                             # injected scorer


def test_reduced_preset_is_attributed_to_section_5_4():
    res = search(STUB64, seed=0, scorer=_constant_scorer(0.0), **PAPER_REDUCED)
    assert res["generations_completed"] == 10
    for name in PAPER_REDUCED:
        assert "§5.4" in res["paper_parameter_sources"][name]


def test_encryptions_reported_with_default_scorer():
    res = search(staged_cipher(), seed=0, **SMALL)
    assert res["encryptions"] == 2 * SMALL["samples"] * res["evaluations"]


def test_global_rngs_untouched():
    np_before = np.random.get_state()
    py_before = random.getstate()
    search(staged_cipher(), seed=3, **SMALL)
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert random.getstate() == py_before


def test_elitism_keeps_best_score_non_decreasing():
    res = search(random_function_cipher(), seed=4, rounds=1, elitism=True,
                 **{**SMALL, "generations": 4})
    best = res["per_round"][0]["best_score_per_generation"]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))


@pytest.mark.parametrize("kwargs", [
    {"population_size": 1},
    {"population_size": 65},
    {"generations": -1},
    {"mutation_probability": 1.5},
    {"mutation_probability": True},
    {"samples": 0},
    {"threshold": float("nan")},
    {"round_weighting": "quadratic"},
    {"deduplicate": 1},
    {"rounds": 0},
    {"max_rounds": 0},
])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        search(STUB64, seed=0, scorer=_constant_scorer(0.0), **{**SMALL, **kwargs})


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_rejected(seed):
    with pytest.raises((ValueError, TypeError)):
        search(STUB64, seed=seed, scorer=_constant_scorer(0.0), **SMALL)
