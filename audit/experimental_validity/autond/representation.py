"""
The frozen within-word bit permutation for H-EV-AUTOND-REPRESENTATION.

Input layout, from the pinned source (commit 8d66e0a4):
    ciphers/speck3264.py convert_to_binary  column i holds bit (15 - i % 16) of word i // 16 (MSB first)
    ciphers/speck3264.py encrypt            returns convert_to_binary([x, y]): left word, then right word
    main.py make_train_data                 returns np.hstack([C0, C1])
so the 64 columns are four 16-bit words, MSB first: C0 left, C0 right, C1 left, C1 right
(tests/autond_tf_cases.py checks this against the generator itself).

Condition B permutes the 16 bit positions within each word, independently per word, keeping the word
order: destination column j of word w receives source column mapping[w, j] of that word. The same
frozen mapping is applied to the training, validation and test features; labels are never touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from framework.provenance import array_sha256

WORD_BITS, WORDS = 16, 4
WORD_NAMES = ("C0_left", "C0_right", "C1_left", "C1_right")


@dataclass(frozen=True)
class WordPermutation:
    mapping: np.ndarray        # (4, 16) int: mapping[w, j] = source column (within word w) of destination j

    def __post_init__(self) -> None:
        problems = bijection_problems(self.mapping)
        if problems:
            raise ValueError(f"not a within-word bit permutation: {problems}")

    @property
    def hash(self) -> str:
        return array_sha256(np.asarray(self.mapping, dtype=np.int64))

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = _features(X)
        out = np.empty_like(X)
        for w in range(WORDS):
            block = slice(w * WORD_BITS, (w + 1) * WORD_BITS)
            out[:, block] = X[:, block][:, self.mapping[w]]
        return out

    def invert(self, X: np.ndarray) -> np.ndarray:
        X = _features(X)
        out = np.empty_like(X)
        for w in range(WORDS):
            block = slice(w * WORD_BITS, (w + 1) * WORD_BITS)
            out[:, block] = X[:, block][:, np.argsort(self.mapping[w])]
        return out

    def to_dict(self) -> dict:
        return {"mapping": np.asarray(self.mapping).tolist(), "sha256": self.hash, "words": list(WORD_NAMES),
                "convention": "destination column j of word w receives source column mapping[w][j]"}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "WordPermutation":
        record = json.loads(Path(path).read_text())
        permutation = cls(np.asarray(record["mapping"], dtype=np.int64))
        if permutation.hash != record["sha256"]:
            raise ValueError(f"{path}: mapping does not match its recorded SHA-256")
        return permutation


def _features(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != WORDS * WORD_BITS:
        raise ValueError(f"expected features of shape (n, {WORDS * WORD_BITS}), got {X.shape}")
    return X


def bijection_problems(mapping) -> list[str]:
    mapping = np.asarray(mapping)
    if mapping.shape != (WORDS, WORD_BITS) or not np.issubdtype(mapping.dtype, np.integer):
        return [f"mapping must be an integer array of shape ({WORDS}, {WORD_BITS}), "
                f"got {mapping.dtype} {mapping.shape}"]
    return [f"{WORD_NAMES[w]} is not a bijection on range(16)" for w in range(WORDS)
            if sorted(mapping[w].tolist()) != list(range(WORD_BITS))]


def generate(seed: int) -> WordPermutation:
    """Four independent permutations of range(16), drawn in word order from numpy's default_rng(seed)."""
    rng = np.random.default_rng(int(seed))
    return WordPermutation(np.stack([rng.permutation(WORD_BITS) for _ in range(WORDS)]).astype(np.int64))


def frozen(config: dict) -> WordPermutation:
    """The configured permutation, refused unless it realises exactly the declared SHA-256."""
    rep = config["representation"]
    permutation = generate(rep["generation_seed"])
    if permutation.hash != rep["expected_sha256"]:
        raise ValueError(f"permutation from seed {rep['generation_seed']} has SHA-256 {permutation.hash}, "
                         f"not the declared {rep['expected_sha256']}")
    return permutation


def validate(X: np.ndarray, Y: np.ndarray, permutation: WordPermutation, seed: int) -> dict:
    """Every invariant the intervention relies on, checked on representative data."""
    X, Y = _features(X), np.asarray(Y)
    labels_before = Y.copy()
    Xp = permutation.apply(X)
    words = [slice(w * WORD_BITS, (w + 1) * WORD_BITS) for w in range(WORDS)]
    report = {
        "bijection": not bijection_problems(permutation.mapping),
        "round_trip": bool(np.array_equal(permutation.invert(Xp), X)),
        # each row keeps exactly its bits per word; with 0/1 data the per-word bit count is that multiset
        "content_preserved": all(np.array_equal(X[:, s].sum(axis=1), Xp[:, s].sum(axis=1)) for s in words),
        "labels_unchanged": bool(np.array_equal(Y, labels_before)),
        "deterministic": bool(np.array_equal(generate(seed).apply(X), Xp) and np.array_equal(permutation.apply(X), Xp)),
        "dtype_and_shape_preserved": Xp.dtype == X.dtype and Xp.shape == X.shape,
        "sha256": permutation.hash,
    }
    report["all_passed"] = all(v for k, v in report.items() if k != "sha256")
    return report
