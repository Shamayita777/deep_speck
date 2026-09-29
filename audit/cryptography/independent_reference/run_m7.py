"""
M7 independent-theory verification runner.

Produces a persistent evidence artifact showing that the independent
Lipmaa-Moriai implementation agrees with:

1. exact exhaustive modular-addition enumeration for small word sizes;
2. the production CipherMind xdp_plus() implementation on randomized
   16-bit transitions;
3. production-theory calculations over actual 5-round Speck32/64
   trajectories.

No model training is performed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from audit.cryptography.gohr import speck as production_speck
from audit.cryptography.independent_reference.lipmaa_moriai_independent import (
    chain_transition_probabilities,
    independent_xdp_plus,
    independent_xdp_plus_array,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

INDEPENDENT_SOURCE = (
    REPO_ROOT
    / "audit"
    / "cryptography"
    / "independent_reference"
    / "lipmaa_moriai_independent.py"
)

PRODUCTION_SOURCE = (
    REPO_ROOT
    / "audit"
    / "cryptography"
    / "gohr"
    / "speck.py"
)

TEST_SOURCE = (
    REPO_ROOT
    / "audit"
    / "cryptography"
    / "tests"
    / "test_m7_independent_theory.py"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "audit"
    / "cryptography"
    / "evidence_current"
    / "m7"
)

OUTPUT_FILE = OUTPUT_DIR / "m7_independent_theory_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip()


def exact_distribution(
    alpha: int,
    beta: int,
    word_size: int,
) -> np.ndarray:
    """
    Exact distribution for one alpha/beta pair by exhaustive enumeration.
    """
    modulus = 1 << word_size
    mask = modulus - 1

    values = np.arange(modulus, dtype=np.uint64)

    x = values[:, None]
    y = values[None, :]

    left = (x + y) & mask
    right = (
        ((x ^ np.uint64(alpha)) + (y ^ np.uint64(beta)))
        & mask
    )

    gamma = left ^ right

    counts = np.bincount(
        gamma.reshape(-1).astype(np.int64),
        minlength=modulus,
    )

    return counts.astype(np.float64) / float(modulus * modulus)


def run_small_word_exhaustive() -> dict:
    """
    Exhaustively verify the independent formula at n=4.

    Every alpha, beta, gamma triple is covered.
    """
    word_size = 4
    modulus = 1 << word_size

    total_cases = 0

    for alpha in range(modulus):
        for beta in range(modulus):
            exact = exact_distribution(
                alpha,
                beta,
                word_size,
            )

            for gamma in range(modulus):
                total_cases += 1

                independent = independent_xdp_plus(
                    alpha,
                    beta,
                    gamma,
                    word_size=word_size,
                )

                expected = float(exact[gamma])

                if independent != expected:
                    raise AssertionError(
                        "Exhaustive n=4 mismatch: "
                        f"alpha={alpha:#x}, "
                        f"beta={beta:#x}, "
                        f"gamma={gamma:#x}, "
                        f"independent={independent}, "
                        f"exact={expected}"
                    )

    return {
        "word_size": word_size,
        "alpha_values": modulus,
        "beta_values": modulus,
        "gamma_values": modulus,
        "triples_checked": total_cases,
        "status": "PASS",
    }


def run_randomized_eight_bit_exact() -> dict:
    """
    Randomized exact checks at n=8.

    Each selected alpha/beta pair is evaluated against all 2^8 gamma
    values using exhaustive enumeration over all 2^16 input pairs.
    """
    rng = np.random.default_rng(20260929)

    word_size = 8
    modulus = 1 << word_size
    pairs_checked = 20
    triples_checked = 0

    for _ in range(pairs_checked):
        alpha = int(rng.integers(0, modulus))
        beta = int(rng.integers(0, modulus))

        exact = exact_distribution(
            alpha,
            beta,
            word_size,
        )

        for gamma in range(modulus):
            triples_checked += 1

            independent = independent_xdp_plus(
                alpha,
                beta,
                gamma,
                word_size=word_size,
            )

            expected = float(exact[gamma])

            if independent != expected:
                raise AssertionError(
                    "Randomized n=8 mismatch: "
                    f"alpha={alpha:#x}, "
                    f"beta={beta:#x}, "
                    f"gamma={gamma:#x}, "
                    f"independent={independent}, "
                    f"exact={expected}"
                )

    return {
        "word_size": word_size,
        "alpha_beta_pairs_checked": pairs_checked,
        "triples_checked": triples_checked,
        "input_pairs_per_distribution": modulus * modulus,
        "status": "PASS",
    }


def run_randomized_sixteen_bit_production() -> dict:
    """
    Compare 20,000 randomized 16-bit transitions against the current
    production implementation.
    """
    rng = np.random.default_rng(20260929)

    sample_count = 20_000

    alpha = rng.integers(
        0,
        1 << 16,
        size=sample_count,
        dtype=np.uint32,
    )

    beta = rng.integers(
        0,
        1 << 16,
        size=sample_count,
        dtype=np.uint32,
    )

    gamma = rng.integers(
        0,
        1 << 16,
        size=sample_count,
        dtype=np.uint32,
    )

    independent = independent_xdp_plus_array(
        alpha,
        beta,
        gamma,
        word_size=16,
    )

    production = production_speck.xdp_plus(
        alpha,
        beta,
        gamma,
    )

    equal = np.array_equal(
        independent,
        production,
    )

    if not equal:
        differences = np.abs(independent - production)

        mismatch_count = int(np.count_nonzero(independent != production))
        max_abs_difference = float(np.max(differences))

        raise AssertionError(
            "Production comparison failed: "
            f"mismatch_count={mismatch_count}, "
            f"max_abs_difference={max_abs_difference}"
        )

    return {
        "word_size": 16,
        "random_seed": 20260929,
        "transitions_checked": sample_count,
        "exact_matches": sample_count,
        "mismatches": 0,
        "max_absolute_difference": 0.0,
        "status": "PASS",
    }


def generate_real_speck_transitions(
    *,
    sample_count: int,
    rounds: int,
    differential: tuple[int, int],
) -> np.ndarray:
    """
    Generate actual 5-round Speck transition triples using the current
    production cipher implementation.

    Only the theory calculation is independently implemented.

    The cipher transition extraction is deliberately shared with the
    production cipher path because M7 tests the mathematical reference,
    not reimplementation of Speck itself.
    """
    rng = np.random.default_rng(20260929)

    transitions = np.empty(
        (sample_count, rounds, 3),
        dtype=np.uint32,
    )

    for sample_index in range(sample_count):
        key_words = tuple(
            int(x)
            for x in rng.integers(
                0,
                1 << 16,
                size=4,
                dtype=np.uint16,
            )
        )

        round_keys = production_speck.expand_key(
            key_words,
            rounds,
        )

        x0 = int(rng.integers(0, 1 << 16))
        y0 = int(rng.integers(0, 1 << 16))

        x1 = x0 ^ differential[0]
        y1 = y0 ^ differential[1]

        for round_index, round_key in enumerate(round_keys):
            add_input_0_x = production_speck.ror(
                x0,
                production_speck.ALPHA(),
            )
            add_input_0_y = y0

            add_input_1_x = production_speck.ror(
                x1,
                production_speck.ALPHA(),
            )
            add_input_1_y = y1

            alpha = add_input_0_x ^ add_input_1_x
            beta = add_input_0_y ^ add_input_1_y

            add_output_0 = (
                add_input_0_x + add_input_0_y
            ) & production_speck.MASK_VAL

            add_output_1 = (
                add_input_1_x + add_input_1_y
            ) & production_speck.MASK_VAL

            gamma = add_output_0 ^ add_output_1

            transitions[
                sample_index,
                round_index,
                0,
            ] = alpha

            transitions[
                sample_index,
                round_index,
                1,
            ] = beta

            transitions[
                sample_index,
                round_index,
                2,
            ] = gamma

            x0, y0 = production_speck.enc_one_round(
                (x0, y0),
                round_key,
            )

            x1, y1 = production_speck.enc_one_round(
                (x1, y1),
                round_key,
            )

    return transitions


def run_real_trajectory_comparison() -> dict:
    """
    Compare chained independent probabilities with the current production
    xdp_plus implementation over actual 5-round Speck32/64 trajectories.
    """
    sample_count = 1_000
    rounds = 5
    differential = (0x0040, 0x0000)

    transitions = generate_real_speck_transitions(
        sample_count=sample_count,
        rounds=rounds,
        differential=differential,
    )

    independent_chain = chain_transition_probabilities(
        transitions,
        word_size=16,
    )

    flat = transitions.reshape(-1, 3)

    production_round_probs = production_speck.xdp_plus(
        flat[:, 0],
        flat[:, 1],
        flat[:, 2],
    ).reshape(
        sample_count,
        rounds,
    )

    production_chain = np.prod(
        production_round_probs,
        axis=1,
    )

    exact_equal = np.array_equal(
        independent_chain,
        production_chain,
    )

    if not exact_equal:
        differences = np.abs(
            independent_chain - production_chain
        )

        raise AssertionError(
            "Real-trajectory chain comparison failed: "
            f"mismatches={np.count_nonzero(independent_chain != production_chain)}, "
            f"max_abs_difference={np.max(differences)}"
        )

    return {
        "sample_count": sample_count,
        "rounds": rounds,
        "differential": [
            hex(differential[0]),
            hex(differential[1]),
        ],
        "exact_chain_matches": sample_count,
        "mismatches": 0,
        "max_absolute_difference": 0.0,
        "status": "PASS",
    }


def run_known_example() -> dict:
    """
    Check the project-recorded worked example.
    """
    alpha = int("11100", 2)
    beta = int("00110", 2)
    gamma = int("10110", 2)

    observed = independent_xdp_plus(
        alpha,
        beta,
        gamma,
        word_size=5,
    )

    expected = 0.25

    if observed != expected:
        raise AssertionError(
            f"Known example failed: observed={observed}, expected={expected}"
        )

    return {
        "alpha": bin(alpha),
        "beta": bin(beta),
        "gamma": bin(gamma),
        "expected_probability": expected,
        "observed_probability": observed,
        "status": "PASS",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INDEPENDENT_SOURCE.exists():
        raise FileNotFoundError(INDEPENDENT_SOURCE)

    if not PRODUCTION_SOURCE.exists():
        raise FileNotFoundError(PRODUCTION_SOURCE)

    if not TEST_SOURCE.exists():
        raise FileNotFoundError(TEST_SOURCE)

    report = {
        "m7": {
            "name": "Independent Lipmaa-Moriai Theory Verification",
            "status": "RUNNING",
            "scope": (
                "Independent implementation of the modular-addition "
                "XOR differential probability used by CipherMind's "
                "single-trail theoretical target."
            ),
            "neural_training_performed": False,
        },
        "checks": {},
        "provenance": {
            "git_commit": get_git_commit(),
            "repository_root": str(REPO_ROOT),
            "independent_source": str(INDEPENDENT_SOURCE.relative_to(REPO_ROOT)),
            "production_source": str(PRODUCTION_SOURCE.relative_to(REPO_ROOT)),
            "test_source": str(TEST_SOURCE.relative_to(REPO_ROOT)),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    # Run every check.
    report["checks"]["known_example"] = run_known_example()
    report["checks"]["exhaustive_n4"] = run_small_word_exhaustive()
    report["checks"]["randomized_exact_n8"] = run_randomized_eight_bit_exact()
    report["checks"]["randomized_production_n16"] = (
        run_randomized_sixteen_bit_production()
    )
    report["checks"]["real_speck_trajectories"] = (
        run_real_trajectory_comparison()
    )

    report["m7"]["status"] = "PASS"

    report["provenance"]["sha256"] = {
        "independent_theory": sha256_file(
            INDEPENDENT_SOURCE
        ),
        "production_theory": sha256_file(
            PRODUCTION_SOURCE
        ),
        "m7_test": sha256_file(
            TEST_SOURCE
        ),
    }

    with OUTPUT_FILE.open(
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
    print("M7 INDEPENDENT THEORY VERIFICATION")
    print("=" * 72)
    print("Known example:                PASS")
    print("Exhaustive n=4:               PASS")
    print("Randomized exact n=8:         PASS")
    print("Randomized production n=16:  PASS")
    print("Real 5-round trajectories:    PASS")
    print()
    print("OVERALL: PASS")
    print()
    print(f"Evidence: {OUTPUT_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()