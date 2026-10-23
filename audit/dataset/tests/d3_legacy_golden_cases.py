"""Deterministic inputs for the legacy-D3 golden regression (no randomness at import).

Shared by tests/test_d3_legacy_golden.py and by the one-off generator that
ran the c603522 engine (loaded from git) to produce
fixtures/d3_legacy_golden_c603522.json. Inputs come from fixed PCG64 seeds.
"""
from __future__ import annotations

import numpy as np

LEGACY_COMMIT = "c603522601b5d05c251d8cad20517ac1fcdd20f0"
LEGACY_PATH = "audit/dataset/d3_distribution_statistics.py"

INTENDED = {"expected_class_ratio": 0.5, "expected_bit_probability": 0.5,
            "class_ratio_tolerance": 0.005, "bit_probability_tolerance": 0.01,
            "pairwise_tvd_tolerance": 0.01}


def _part(seed: int, n: int, d: int, p: float = 0.5):
    rng = np.random.default_rng(seed)
    return (rng.random((n, d)) < p).astype(np.uint8), rng.integers(0, 2, n).astype(np.uint8)


def audit_cases():
    """name -> (list of (X, Y) partitions, intended, familywise_alpha)."""
    cases = {}
    cases["ordinary_d6"] = ([_part(1, 400, 6), _part(2, 200, 6), _part(3, 200, 6)], INTENDED, 0.01)
    const = [_part(4, 300, 5), _part(5, 150, 5), _part(6, 150, 5)]
    for x, _ in const:
        x[:, 2] = 0                       # constant feature: empty columns in its pair tables
    cases["constant_feature_d5"] = (const, INTENDED, 0.01)
    cases["sparse_d4"] = ([_part(7, 12, 4), _part(8, 9, 4), _part(9, 7, 4)], INTENDED, 0.05)
    cases["unbalanced_rare_d4"] = ([_part(10, 2000, 4, 0.03), _part(11, 15, 4, 0.03),
                                    _part(12, 15, 4, 0.03)], INTENDED, 0.01)
    shifted = [_part(13, 3000, 4), _part(14, 3000, 4)]
    shifted[1][0][:1500, 3] = shifted[1][0][:1500, 1]   # strong pair dependence in one partition
    cases["shifted_two_partitions_d4"] = (shifted, INTENDED, 0.01)
    return cases


def pair_cases():
    """name -> (A, B) for compare_pairwise_joint_distributions."""
    a = np.array([[0, 1], [1, 1], [0, 0], [1, 0], [1, 1]], dtype=np.uint8)
    b = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.uint8)
    z = np.zeros((10, 3), dtype=np.uint8)
    z[::2, 1] = 1
    w = np.zeros((7, 3), dtype=np.uint8)
    w[::3, 1] = 1
    return {"tiny": (a, b), "zero_columns": (z, w), "identical": (a, a.copy())}
