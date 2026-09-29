"""
Pytest coverage for CipherMind II-2 M1.

These tests cover the cheap deterministic portion of M1.
They deliberately do not launch the 200-epoch neural training run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from audit.implementation.ii2_independent import (
    CHECK_VECTOR,
    build_gohr_model,
    convert_to_binary,
    encrypt,
    expand_key,
    round_encrypt,
)
from audit.implementation.ii2_independent import (
    dataset_independent,
)


ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    ROOT
    / "audit"
    / "implementation"
    / "reference"
    / "speck.py"
)


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "reference_speck_for_m1_tests",
        REFERENCE_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_published_speck32_64_vector():
    keys = expand_key(
        CHECK_VECTOR["key"],
        CHECK_VECTOR["rounds"],
    )

    ciphertext = encrypt(
        CHECK_VECTOR["plaintext"],
        keys,
    )

    assert (
        int(ciphertext[0]),
        int(ciphertext[1]),
    ) == CHECK_VECTOR["ciphertext"]


def test_independent_key_schedule_matches_reference():
    reference = load_reference()

    rng = np.random.default_rng(20260929)

    keys = rng.integers(
        0,
        1 << 16,
        size=(4, 100),
        dtype=np.uint16,
    )

    independent = expand_key(
        keys,
        5,
    )

    reference_keys = np.asarray(
        reference.expand_key(keys, 5),
        dtype=np.uint16,
    )

    assert np.array_equal(
        independent,
        reference_keys,
    )


def test_round_by_round_matches_reference():
    reference = load_reference()

    rng = np.random.default_rng(20260929)

    count = 100

    keys = rng.integers(
        0,
        1 << 16,
        size=(4, count),
        dtype=np.uint16,
    )

    left = rng.integers(
        0,
        1 << 16,
        size=count,
        dtype=np.uint16,
    )

    right = rng.integers(
        0,
        1 << 16,
        size=count,
        dtype=np.uint16,
    )

    independent_keys = expand_key(
        keys,
        5,
    )

    reference_keys = reference.expand_key(
        keys,
        5,
    )

    independent_left = left.copy()
    independent_right = right.copy()

    reference_left = left.copy()
    reference_right = right.copy()

    for round_index in range(5):
        independent_left, independent_right = (
            round_encrypt(
                (
                    independent_left,
                    independent_right,
                ),
                independent_keys[round_index],
            )
        )

        reference_left, reference_right = (
            reference.enc_one_round(
                (
                    reference_left,
                    reference_right,
                ),
                reference_keys[round_index],
            )
        )

        assert np.array_equal(
            independent_left,
            reference_left,
        )

        assert np.array_equal(
            independent_right,
            reference_right,
        )


def test_dataset_construction_matches_reference_with_fixed_material():
    reference = load_reference()

    rng = np.random.default_rng(20260929)

    count = 1000

    labels = rng.integers(
        0,
        2,
        size=count,
        dtype=np.uint8,
    )

    keys = rng.integers(
        0,
        1 << 16,
        size=(4, count),
        dtype=np.uint16,
    )

    p0_left = rng.integers(
        0,
        1 << 16,
        size=count,
        dtype=np.uint16,
    )

    p0_right = rng.integers(
        0,
        1 << 16,
        size=count,
        dtype=np.uint16,
    )

    zero_count = int(
        np.count_nonzero(labels == 0)
    )

    random_left = rng.integers(
        0,
        1 << 16,
        size=zero_count,
        dtype=np.uint16,
    )

    random_right = rng.integers(
        0,
        1 << 16,
        size=zero_count,
        dtype=np.uint16,
    )

    independent_X, independent_Y = (
        dataset_independent.construct_from_material(
            labels=labels,
            master_keys=keys,
            plaintext_left=p0_left,
            plaintext_right=p0_right,
            rounds=5,
            random_plaintext_left=random_left,
            random_plaintext_right=random_right,
        )
    )

    p1_left = (
        p0_left
        ^ np.uint16(0x0040)
    )

    p1_right = p0_right.copy()

    mask = labels == 0

    p1_left = p1_left.copy()
    p1_right = p1_right.copy()

    p1_left[mask] = random_left
    p1_right[mask] = random_right

    reference_keys = reference.expand_key(
        keys,
        5,
    )

    ref0_left, ref0_right = reference.encrypt(
        (p0_left, p0_right),
        reference_keys,
    )

    ref1_left, ref1_right = reference.encrypt(
        (p1_left, p1_right),
        reference_keys,
    )

    reference_X = reference.convert_to_binary(
        [
            ref0_left,
            ref0_right,
            ref1_left,
            ref1_right,
        ]
    )

    assert np.array_equal(
        independent_Y,
        labels,
    )

    assert np.array_equal(
        independent_X,
        reference_X,
    )


def test_binary_encoding():
    observed = convert_to_binary(
        [
            np.asarray([0x8000], dtype=np.uint16),
            np.asarray([0x0001], dtype=np.uint16),
            np.asarray([0xFFFF], dtype=np.uint16),
            np.asarray([0x0000], dtype=np.uint16),
        ]
    )

    expected = np.asarray(
        [[
            *([1] + [0] * 15),
            *([0] * 15 + [1]),
            *([1] * 16),
            *([0] * 16),
        ]],
        dtype=np.uint8,
    )

    assert np.array_equal(
        observed,
        expected,
    )


def test_depth10_model_structure():
    model = build_gohr_model()

    conv_layers = sum(
        layer.__class__.__name__ == "Conv1D"
        for layer in model.layers
    )

    residual_adds = sum(
        layer.__class__.__name__ == "Add"
        for layer in model.layers
    )

    dense_layers = sum(
        layer.__class__.__name__ == "Dense"
        for layer in model.layers
    )

    assert model.input_shape == (None, 64)
    assert model.output_shape == (None, 1)
    assert conv_layers == 21
    assert residual_adds == 10
    assert dense_layers == 3
    assert model.count_params() == 102497