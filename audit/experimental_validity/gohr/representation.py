"""
Candidate-1 representation perturbation for H-EV-REPRESENTATION.

Per the frozen Round-4 design: for each of the four 16-bit ciphertext
words (ct0a, ct1a, ct0b, ct1b), generate one fixed, independent
permutation of that word's 16 bit positions, and apply it to the
convert_to_binary() output before the (unmodified) Reshape/Permute/
Conv1D architecture consumes it.

This is NOT "architecture-neutral" - it is an information-preserving
representation perturbation applied under a fixed architecture, and it
intentionally tests the architecture's dependence on Gohr's specific
cross-word bit-alignment. See the Round-4 design document for the full
rationale and the two rejected alternative candidates.

The permutation must be generated once, fixed, persisted, hashed, and
applied identically to train/validation/confirmatory-test partitions.
This module never re-randomizes the permutation per call - callers
must explicitly generate it once and thread the same
Candidate1Permutation object (or its persisted/reloaded form) through
every partition and every replicate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audit.experimental_validity.framework.provenance import array_sha256

WORD_SIZE = 16
NUM_WORDS = 4  # ct0a, ct1a, ct0b, ct1b


@dataclass(frozen=True)
class Candidate1Permutation:
    """
    One fixed permutation per word. `permutations` has shape (4, 16);
    row w gives, for word w, the source column (within that word's
    16-bit block) that ends up at each destination position.
    """
    permutations: np.ndarray  # shape (4, 16), dtype=int

    def __post_init__(self) -> None:
        if self.permutations.shape != (NUM_WORDS, WORD_SIZE):
            raise ValueError(f"Expected shape (4, 16), got {self.permutations.shape}")

    @property
    def hash(self) -> str:
        return array_sha256(self.permutations.astype(np.int64))

    def to_dict(self) -> dict:
        return {
            "permutations": self.permutations.tolist(),
            "hash": self.hash,
            "num_words": NUM_WORDS,
            "word_size": WORD_SIZE,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Candidate1Permutation":
        data = json.loads(Path(path).read_text())
        perm = cls(permutations=np.array(data["permutations"], dtype=int))
        if perm.hash != data["hash"]:
            raise ValueError(
                f"Loaded permutation file {path} is corrupted or was tampered with: "
                f"recorded hash {data['hash']} does not match recomputed hash {perm.hash}."
            )
        return perm


def generate_candidate1_permutation(rng: np.random.Generator) -> Candidate1Permutation:
    """
    Generate one fixed permutation per word (4 independent permutations
    of range(16)). Must be called exactly once for a given experiment
    and then persisted/reused - see gohr.experiments for the freeze
    discipline that enforces this.
    """
    perms = np.stack([rng.permutation(WORD_SIZE) for _ in range(NUM_WORDS)], axis=0)
    return Candidate1Permutation(permutations=perms)


def identity_permutation() -> Candidate1Permutation:
    """The baseline (Gohr original) 'permutation' - the identity map."""
    perms = np.stack([np.arange(WORD_SIZE) for _ in range(NUM_WORDS)], axis=0)
    return Candidate1Permutation(permutations=perms)


def apply_candidate1(X: np.ndarray, permutation: Candidate1Permutation) -> np.ndarray:
    """
    Apply the fixed per-word bit-position permutation to a
    convert_to_binary()-style (n, 64) array. Word w occupies columns
    [w*16 : (w+1)*16]; within that block, destination column j receives
    the bit currently at source column permutation.permutations[w, j].
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != NUM_WORDS * WORD_SIZE:
        raise ValueError(f"Expected shape (n, {NUM_WORDS * WORD_SIZE}), got {X.shape}")

    out = np.empty_like(X)
    for w in range(NUM_WORDS):
        start = w * WORD_SIZE
        end = start + WORD_SIZE
        out[:, start:end] = X[:, start:end][:, permutation.permutations[w]]
    return out


def invert_candidate1(X_scrambled: np.ndarray, permutation: Candidate1Permutation) -> np.ndarray:
    """Exact inverse of apply_candidate1, used for round-trip validation."""
    X_scrambled = np.asarray(X_scrambled)
    out = np.empty_like(X_scrambled)
    for w in range(NUM_WORDS):
        start = w * WORD_SIZE
        end = start + WORD_SIZE
        inverse = np.argsort(permutation.permutations[w])
        out[:, start:end] = X_scrambled[:, start:end][:, inverse]
    return out


# ---------------------------------------------------------------------
# Validation (required before any production run - see
# tests/test_gohr_representation_transform.py, executed automatically
# by scripts/validate_ev.py)
# ---------------------------------------------------------------------


def verify_bijection(permutation: Candidate1Permutation) -> tuple[bool, list[str]]:
    problems = []
    for w in range(NUM_WORDS):
        row = permutation.permutations[w]
        if sorted(row.tolist()) != list(range(WORD_SIZE)):
            problems.append(f"Word {w}'s permutation is not a bijection on range(16): {row.tolist()}")
    return (len(problems) == 0), problems


def verify_round_trip(X: np.ndarray, permutation: Candidate1Permutation) -> bool:
    """decode(encode(x)) == x for representative data."""
    encoded = apply_candidate1(X, permutation)
    decoded = invert_candidate1(encoded, permutation)
    return bool(np.array_equal(decoded, X))


def verify_content_preserved(X: np.ndarray, X_scrambled: np.ndarray) -> bool:
    """
    Per-word-block bit count (popcount) must be preserved for every
    row, since the transform only reorders bits within each word's
    16-column block - it never moves bits across blocks or changes
    their values. This is a stronger check than a whole-row popcount
    match, since it would also catch an accidental cross-block leak.
    """
    for w in range(NUM_WORDS):
        start = w * WORD_SIZE
        end = start + WORD_SIZE
        if not np.array_equal(
            X[:, start:end].sum(axis=1), X_scrambled[:, start:end].sum(axis=1)
        ):
            return False
    return True


def verify_labels_unchanged(Y: np.ndarray, Y_after: np.ndarray) -> bool:
    return bool(np.array_equal(Y, Y_after))


def verify_deterministic(rng_seed: int, X: np.ndarray) -> bool:
    """Applying a permutation generated with the same seed twice yields identical output."""
    perm_a = generate_candidate1_permutation(np.random.default_rng(rng_seed))
    perm_b = generate_candidate1_permutation(np.random.default_rng(rng_seed))
    return bool(np.array_equal(apply_candidate1(X, perm_a), apply_candidate1(X, perm_b)))


def run_full_validation(X: np.ndarray, Y: np.ndarray, permutation: Candidate1Permutation) -> dict:
    """
    Run every required Candidate-1 validation check and return a
    machine-readable result. Called automatically before any production
    run (scripts/validate_ev.py) - a failure here must block production.

    ISSUE 6 FIX (Round 6): the labels_unchanged check previously called
    verify_labels_unchanged(Y, Y) - comparing the same object/array to
    itself, which is tautologically True regardless of anything and
    therefore tested nothing. Candidate-1's public functions
    (apply_candidate1, invert_candidate1) never accept Y at all - this
    is independently verified by test_labels_are_never_touched via
    signature inspection, which is the real protection. What THIS
    function's check can meaningfully add is a regression guard against
    the validation pipeline itself accidentally mutating the caller's Y
    array in place (e.g. via a future refactor that passes Y through a
    shared buffer). That requires an INDEPENDENT copy of Y taken before
    any validation work runs, compared against the original afterward -
    not the same reference compared to itself.
    """
    Y_reference_copy = np.array(Y, copy=True)

    bijection_ok, bijection_problems = verify_bijection(permutation)
    X_scrambled = apply_candidate1(X, permutation)
    results = {
        "bijection": bijection_ok,
        "bijection_problems": bijection_problems,
        "round_trip": verify_round_trip(X, permutation),
        "content_preserved": verify_content_preserved(X, X_scrambled),
        # Compares an independent pre-validation copy against the
        # (possibly, in principle) mutated post-validation array -
        # non-tautological, unlike the previous verify_labels_unchanged(Y, Y).
        "labels_unchanged": verify_labels_unchanged(Y_reference_copy, Y),
        "deterministic": verify_deterministic(12345, X),
        "permutation_hash_recorded": permutation.hash is not None and len(permutation.hash) == 64,
    }
    results["all_passed"] = all(
        v for k, v in results.items() if isinstance(v, bool)
    ) and len(bijection_problems) == 0
    return results
