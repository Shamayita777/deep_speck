"""
Gohr-specific D2 adapter.

This module contains ONLY the assumptions needed to expose the Gohr
representation to the generic D2 engine. The generic D2 implementation
never imports this module.

The existing GohrAdapter remains responsible for dataset generation and
model/training behavior. This adapter observes the original generator's
random streams; it does not reimplement make_train_data().
"""
from __future__ import annotations

from typing import Any

import numpy as np
import speck as sp

from audit.dataset.adapters.gohr import GohrAdapter


class GohrD2Adapter:
    """Expose Gohr's observable D2 representations."""

    DATASET_ID = "gohr-speck"
    DATASET_VERSION = "original-make-train-data"
    NUM_ROUNDS = 5
    FEATURE_BITS = 64

    def __init__(self, *, num_rounds: int = NUM_ROUNDS) -> None:
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1.")
        self.num_rounds = int(num_rounds)

    @staticmethod
    def _combine_u16(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left.astype(np.uint32) << 16) | right.astype(np.uint32)

    @staticmethod
    def _feature_words(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.uint8)
        if x.ndim != 2 or x.shape[1] != 64:
            raise ValueError("Expected Gohr feature matrix with shape (n, 64).")
        weights = 2 ** np.arange(15, -1, -1, dtype=np.uint16)
        words = np.empty((len(x), 4), dtype=np.uint16)
        for index in range(4):
            start = index * 16
            words[:, index] = np.sum(
                x[:, start:start + 16].astype(np.uint16) * weights,
                axis=1,
                dtype=np.uint32,
            ).astype(np.uint16)
        return words[:, 0], words[:, 1], words[:, 2], words[:, 3]

    def generate_partition(self, samples: int) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
        """Generate through the original GohrAdapter and expose semantic views."""
        if samples < 2:
            raise ValueError("samples must be >= 2.")

        original_urandom = sp.urandom
        captured: list[bytes] = []

        def capture_urandom(num_bytes: int) -> bytes:
            value = original_urandom(num_bytes)
            captured.append(bytes(value))
            return value

        sp.urandom = capture_urandom
        try:
            adapter = GohrAdapter(
                validation_x=None,
                validation_y=None,
                test_x=None,
                test_y=None,
                num_rounds=self.num_rounds,
            )
            features, labels = adapter.generate_partition(samples)
        finally:
            sp.urandom = original_urandom

        expected_prefix = [samples, 8 * samples, 2 * samples, 2 * samples]
        actual_prefix = [len(v) for v in captured[:4]]
        if len(captured) < 4 or actual_prefix != expected_prefix:
            raise RuntimeError(
                "Gohr generator urandom structure changed. "
                f"Expected prefix {expected_prefix}, got {actual_prefix}."
            )

        labels = np.asarray(labels).reshape(-1)
        captured_labels = np.frombuffer(captured[0], dtype=np.uint8) & 1
        if not np.array_equal(captured_labels, labels):
            raise RuntimeError("Captured Gohr label stream does not match returned labels.")

        if len(captured) != 6:
            raise RuntimeError(
                "Gohr make_train_data() changed its urandom call count; "
                f"expected 6 calls, got {len(captured)}."
            )

        plain0l = np.frombuffer(captured[2], dtype=np.uint16).copy()
        plain0r = np.frombuffer(captured[3], dtype=np.uint16).copy()
        plain1l = plain0l ^ np.uint16(0x0040)
        plain1r = plain0r

        random_count = int(np.sum(captured_labels == 0))
        if len(captured[4]) != 2 * random_count or len(captured[5]) != 2 * random_count:
            raise RuntimeError("Gohr conditional plaintext random-stream length changed.")

        random_plain1l = np.frombuffer(captured[4], dtype=np.uint16)
        random_plain1r = np.frombuffer(captured[5], dtype=np.uint16)
        plain1l[captured_labels == 0] = random_plain1l
        plain1r[captured_labels == 0] = random_plain1r

        keys = np.frombuffer(captured[1], dtype=np.uint16).reshape(4, -1).copy()
        key64 = (
            keys[0].astype(np.uint64)
            | (keys[1].astype(np.uint64) << 16)
            | (keys[2].astype(np.uint64) << 32)
            | (keys[3].astype(np.uint64) << 48)
        )

        ct0l, ct0r, ct1l, ct1r = self._feature_words(features)
        views = {
            "key64": {
                "values": key64,
                "domain_size": 2 ** 64,
                "description": "64-bit Gohr key reconstructed from generator random words.",
            },
            "plaintext0_block32": {
                "values": self._combine_u16(plain0l, plain0r),
                "domain_size": 2 ** 32,
                "description": "32-bit first plaintext block.",
            },
            "plaintext1_block32": {
                "values": self._combine_u16(plain1l, plain1r),
                "domain_size": 2 ** 32,
                "description": "32-bit second plaintext block after Gohr's label-conditioned construction.",
            },
            "ciphertext0_block32": {
                "values": self._combine_u16(ct0l, ct0r),
                "domain_size": 2 ** 32,
                "description": "32-bit first ciphertext block recovered from the Gohr feature representation.",
            },
            "ciphertext1_block32": {
                "values": self._combine_u16(ct1l, ct1r),
                "domain_size": 2 ** 32,
                "description": "32-bit second ciphertext block recovered from the Gohr feature representation.",
            },
        }
        return np.asarray(features), labels, views

    def reference_specification(self) -> dict[str, Any]:
        """Return Gohr case-study assumptions; these do not enter generic D2."""
        return {
            "feature_bits": self.FEATURE_BITS,
            "pairwise_hamming_reference": "Binomial(64, 0.5) nominal Gohr diagnostic reference; confirmatory D2 Hamming tests condition on observed per-bit marginals",
            "structured_collision_reference": "Independent uniform finite-domain collision model",
            "generator": "speck.make_train_data",
            "num_rounds": self.num_rounds,
            "randomness_source": "os.urandom",
            "null_model_status": "case-study assumptions; these are explicit statistical null models, not universal independence theorems",
            "audit_seed_semantics": "audit_seed controls D2 sampling and statistical resampling only; dataset generation remains OS-randomized",
        }
