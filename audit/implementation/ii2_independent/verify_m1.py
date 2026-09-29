"""
Deterministic M1 verification for CipherMind II-2.

This stage does NOT train a neural network.

It verifies that the independently written implementation agrees with
the frozen Gohr reference on:

    1. published Speck32/64 test vector;
    2. complete key schedule for deterministic cases;
    3. round-by-round encryption state;
    4. five-round ciphertext output;
    5. real-vs-random dataset construction when supplied identical
       cryptographic random material;
    6. 64-bit binary input encoding;
    7. independent neural architecture structure.

The reference implementation is imported only inside the comparator.
The independent implementation itself never imports it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import dataset_independent
from . import model_independent
from . import speck_independent


ROOT = Path(__file__).resolve().parents[3]

REFERENCE_SPECK = (
    ROOT
    / "audit"
    / "implementation"
    / "reference"
    / "speck.py"
)

INDEPENDENT_SPECK = (
    ROOT
    / "audit"
    / "implementation"
    / "ii2_independent"
    / "speck_independent.py"
)

INDEPENDENT_DATASET = (
    ROOT
    / "audit"
    / "implementation"
    / "ii2_independent"
    / "dataset_independent.py"
)

INDEPENDENT_MODEL = (
    ROOT
    / "audit"
    / "implementation"
    / "ii2_independent"
    / "model_independent.py"
)

OUTPUT_DIR = (
    ROOT
    / "audit"
    / "implementation"
    / "evidence"
    / "m1"
)

REPORT = OUTPUT_DIR / "m1_deterministic_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return completed.stdout.strip()


def load_reference_speck():
    """
    Load the frozen reference Speck source without making it a dependency
    of the independent implementation package.
    """
    spec = importlib.util.spec_from_file_location(
        "gohr_frozen_reference_speck",
        REFERENCE_SPECK,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load reference module: {REFERENCE_SPECK}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_published_vector(reference) -> dict:
    """
    Compare the independent implementation against Gohr's published
    Speck32/64 vector.
    """
    expected = speck_independent.CHECK_VECTOR

    independent_keys = speck_independent.expand_key(
        expected["key"],
        expected["rounds"],
    )

    reference_keys = reference.expand_key(
        expected["key"],
        expected["rounds"],
    )

    if tuple(independent_keys) != tuple(
        int(value) for value in reference_keys
    ):
        raise AssertionError(
            "Key schedule mismatch on published test vector."
        )

    independent_ct = speck_independent.encrypt(
        expected["plaintext"],
        independent_keys,
    )

    reference_ct = reference.encrypt(
        expected["plaintext"],
        reference_keys,
    )

    independent_ct = (
        int(independent_ct[0]),
        int(independent_ct[1]),
    )

    reference_ct = (
        int(reference_ct[0]),
        int(reference_ct[1]),
    )

    expected_ct = expected["ciphertext"]

    if independent_ct != expected_ct:
        raise AssertionError(
            f"Independent published-vector output mismatch: "
            f"{independent_ct!r} != {expected_ct!r}"
        )

    if reference_ct != expected_ct:
        raise AssertionError(
            f"Reference published-vector output mismatch: "
            f"{reference_ct!r} != {expected_ct!r}"
        )

    return {
        "status": "PASS",
        "key_schedule_rounds_checked": expected["rounds"],
        "ciphertext": [
            hex(independent_ct[0]),
            hex(independent_ct[1]),
        ],
    }


def compare_key_schedules(reference) -> dict:
    """
    Compare all 5-round round keys for deterministic random keys.
    """
    rng = np.random.default_rng(20260929)

    count = 1000

    master_keys = rng.integers(
        0,
        1 << 16,
        size=(4, count),
        dtype=np.uint16,
    )

    independent = speck_independent.expand_key(
        master_keys,
        5,
    )

    reference_keys = reference.expand_key(
        master_keys,
        5,
    )

    reference_array = np.asarray(
        reference_keys,
        dtype=np.uint16,
    )

    if independent.shape != reference_array.shape:
        raise AssertionError(
            "Key schedule shapes differ."
        )

    if not np.array_equal(
        independent,
        reference_array,
    ):
        mismatch = np.argwhere(
            independent != reference_array
        )[0]

        raise AssertionError(
            "Key schedule mismatch at "
            f"round={int(mismatch[0])}, "
            f"sample={int(mismatch[1])}"
        )

    return {
        "status": "PASS",
        "master_keys_checked": count,
        "rounds": 5,
        "round_keys_compared": count * 5,
    }


def compare_round_traces(reference) -> dict:
    """
    Compare every intermediate state for 5-round encryption.
    """
    rng = np.random.default_rng(20260929)

    count = 1000

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

    independent_keys = speck_independent.expand_key(
        keys,
        5,
    )

    reference_keys = reference.expand_key(
        keys,
        5,
    )

    ind_left = left.copy()
    ind_right = right.copy()

    ref_left = left.copy()
    ref_right = right.copy()

    states_compared = 0

    for round_index in range(5):
        ind_left, ind_right = (
            speck_independent.round_encrypt(
                (ind_left, ind_right),
                independent_keys[round_index],
            )
        )

        ref_left, ref_right = (
            reference.enc_one_round(
                (ref_left, ref_right),
                reference_keys[round_index],
            )
        )

        if not np.array_equal(
            ind_left,
            np.asarray(ref_left, dtype=np.uint16),
        ):
            raise AssertionError(
                f"Left-state mismatch after round {round_index + 1}"
            )

        if not np.array_equal(
            ind_right,
            np.asarray(ref_right, dtype=np.uint16),
        ):
            raise AssertionError(
                f"Right-state mismatch after round {round_index + 1}"
            )

        states_compared += 2 * count

    return {
        "status": "PASS",
        "samples": count,
        "rounds": 5,
        "word_states_compared": states_compared,
    }


def compare_differential_dataset(reference) -> dict:
    """
    Feed exactly identical deterministic material to both implementations.
    """
    rng = np.random.default_rng(20260929)

    count = 10000

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
            differential=speck_independent.GOHR_INPUT_DIFFERENCE,
            random_plaintext_left=random_left,
            random_plaintext_right=random_right,
        )
    )

    # Reconstruct the same semantics using the frozen reference cipher.
    p1_left = (
        p0_left
        ^ np.uint16(speck_independent.GOHR_INPUT_DIFFERENCE[0])
    )

    p1_right = (
        p0_right
        ^ np.uint16(speck_independent.GOHR_INPUT_DIFFERENCE[1])
    )

    zero_mask = labels == 0

    p1_left = p1_left.copy()
    p1_right = p1_right.copy()

    p1_left[zero_mask] = random_left
    p1_right[zero_mask] = random_right

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

    reference_X = np.asarray(
        reference_X,
        dtype=np.uint8,
    )

    if not np.array_equal(
        independent_Y,
        labels,
    ):
        raise AssertionError(
            "Independent labels changed during construction."
        )

    if not np.array_equal(
        independent_X,
        reference_X,
    ):
        mismatch_count = int(
            np.count_nonzero(
                independent_X != reference_X
            )
        )

        raise AssertionError(
            "Dataset representation mismatch: "
            f"{mismatch_count} differing bits."
        )

    return {
        "status": "PASS",
        "samples": count,
        "rounds": 5,
        "differential": ["0x0040", "0x0000"],
        "representation_shape": list(
            independent_X.shape
        ),
        "zero_label_samples": zero_count,
        "one_label_samples": int(
            np.count_nonzero(labels == 1)
        ),
        "differing_bits": 0,
    }


def verify_binary_encoding() -> dict:
    """
    Verify the independent 64-bit representation for known words.
    """
    words = [
        np.asarray([0x8000], dtype=np.uint16),
        np.asarray([0x0001], dtype=np.uint16),
        np.asarray([0xFFFF], dtype=np.uint16),
        np.asarray([0x0000], dtype=np.uint16),
    ]

    observed = speck_independent.convert_to_binary(
        words
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

    if not np.array_equal(observed, expected):
        raise AssertionError(
            "64-bit binary representation mismatch."
        )

    return {
        "status": "PASS",
        "shape": list(observed.shape),
    }


def verify_model_architecture() -> dict:
    """
    Verify the independent model's structural properties against the
    architecture specified by Gohr's supplementary implementation.
    """
    model = model_independent.build_gohr_model(
        depth=10,
        filters=32,
        dense_1=64,
        dense_2=64,
        kernel_size=3,
        word_size=16,
        l2_reg=1e-5,
    )

    layers = model.layers

    class_counts: dict[str, int] = {}

    for layer in layers:
        name = layer.__class__.__name__
        class_counts[name] = (
            class_counts.get(name, 0)
            + 1
        )

    if class_counts.get("Conv1D", 0) != 21:
        raise AssertionError(
            "Expected 21 Conv1D layers, got "
            f"{class_counts.get('Conv1D', 0)}"
        )

    if class_counts.get("Add", 0) != 10:
        raise AssertionError(
            "Expected 10 residual Add layers, got "
            f"{class_counts.get('Add', 0)}"
        )

    if class_counts.get("Dense", 0) != 3:
        raise AssertionError(
            "Expected 3 Dense layers, got "
            f"{class_counts.get('Dense', 0)}"
        )

    if model.input_shape != (None, 64):
        raise AssertionError(
            f"Unexpected input shape: {model.input_shape}"
        )

    if model.output_shape != (None, 1):
        raise AssertionError(
            f"Unexpected output shape: {model.output_shape}"
        )

    # The actual public code path yields 102,497 parameters at depth 10.
    if model.count_params() != 102497:
        raise AssertionError(
            "Unexpected depth-10 parameter count: "
            f"{model.count_params()} != 102497"
        )

    return {
        "status": "PASS",
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameter_count": model.count_params(),
        "conv1d_layers": class_counts["Conv1D"],
        "residual_add_layers": class_counts["Add"],
        "dense_layers": class_counts["Dense"],
        "paper_code_discrepancy": (
            "Paper prose omits BN after dense layer 2; "
            "public supplementary code includes it. "
            "M1 follows the public code path."
        ),
    }


def main() -> None:
    if not REFERENCE_SPECK.exists():
        raise FileNotFoundError(
            f"Missing frozen reference: {REFERENCE_SPECK}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_reference_speck()

    report = {
        "experiment_id": "II2-DETERMINISTIC-GOHR-5R",
        "status": "RUNNING",
        "training_performed": False,
        "source_basis": {
            "paper": (
                "Gohr, A. (2019), Improving Attacks on "
                "Round-Reduced Speck32/64 Using Deep Learning"
            ),
            "supplementary_repository": (
                "https://github.com/agohr/deep_speck"
            ),
        },
        "checks": {},
        "provenance": {
            "git_commit": git_commit(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    report["checks"]["published_test_vector"] = (
        test_published_vector(reference)
    )

    report["checks"]["key_schedule"] = (
        compare_key_schedules(reference)
    )

    report["checks"]["round_traces"] = (
        compare_round_traces(reference)
    )

    report["checks"]["dataset_construction"] = (
        compare_differential_dataset(reference)
    )

    report["checks"]["binary_encoding"] = (
        verify_binary_encoding()
    )

    report["checks"]["model_architecture"] = (
        verify_model_architecture()
    )

    report["status"] = "PASS"

    report["provenance"]["sha256"] = {
        "independent_speck": sha256_file(
            INDEPENDENT_SPECK
        ),
        "independent_dataset": sha256_file(
            INDEPENDENT_DATASET
        ),
        "independent_model": sha256_file(
            INDEPENDENT_MODEL
        ),
        "reference_speck": sha256_file(
            REFERENCE_SPECK
        ),
    }

    with REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("=" * 72)
    print("M1 DETERMINISTIC VERIFICATION")
    print("=" * 72)
    print("Published test vector:      PASS")
    print("Key schedule:               PASS")
    print("5-round round traces:       PASS")
    print("Dataset construction:       PASS")
    print("64-bit representation:      PASS")
    print("Depth-10 architecture:      PASS")
    print()
    print("OVERALL: PASS")
    print()
    print(f"Evidence: {REPORT}")
    print("=" * 72)


if __name__ == "__main__":
    main()