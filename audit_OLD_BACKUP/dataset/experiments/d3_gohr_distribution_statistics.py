"""
Gohr D3 — Marginal and Low-Order Distributional Consistency Audit.

D3 is deliberately scoped as marginal and low-order distributional
consistency. It is not a claim of full 64-dimensional distributional
identity.

The experiment uses the canonical GohrAdapter for generation. D3 adds
second-order feature-pair *partition-consistency* diagnostics. These do not
assume that all ciphertext-feature pairs are independent; instead they test
whether the same low-order structure is reproduced across independently
generated train/validation/test partitions.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from audit.dataset.adapters.gohr import GohrAdapter
from ..d3_distribution_statistics import audit_distribution, evaluate_decision, print_report


NUM_ROUNDS = 5
FEATURE_BITS = 64
DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"

DEFAULT_TRAIN_SAMPLES = 10_000_000
DEFAULT_VALIDATION_SAMPLES = 1_000_000
DEFAULT_TEST_SAMPLES = 1_000_000

# These are practical-equivalence criteria, not universal statistical laws.
# They must remain fixed before the confirmatory run.
EXPECTED_CLASS_RATIO = 0.5
EXPECTED_BIT_PROBABILITY = 0.5
CLASS_RATIO_TOLERANCE = 0.005
BIT_PROBABILITY_TOLERANCE = 0.01
PAIRWISE_TVD_TOLERANCE = 0.01
FAMILYWISE_ALPHA = 0.01
CONFIDENCE_LEVEL = 0.95


def generate_gohr_partition(samples: int, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    adapter = GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=rounds,
        depth=10,
        epochs=1,
    )
    return adapter.generate_partition(samples, num_rounds=rounds)


def generate_gohr_dataset(train_samples: int, validation_samples: int, test_samples: int, rounds: int):
    print(f"Generating training partition through GohrAdapter...")
    train_x, train_y = generate_gohr_partition(train_samples, rounds)
    print(f"  shape={train_x.shape}, dtype={train_x.dtype}")
    print("Generating validation partition through GohrAdapter...")
    validation_x, validation_y = generate_gohr_partition(validation_samples, rounds)
    print(f"  shape={validation_x.shape}, dtype={validation_x.dtype}")
    print("Generating test partition through GohrAdapter...")
    test_x, test_y = generate_gohr_partition(test_samples, rounds)
    print(f"  shape={test_x.shape}, dtype={test_x.dtype}")
    return train_x, train_y, validation_x, validation_y, test_x, test_y


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Gohr D3 marginal and low-order distributional consistency audit.")
    p.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    p.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    p.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    p.add_argument("--class-ratio-tolerance", type=float, default=CLASS_RATIO_TOLERANCE)
    p.add_argument("--bit-probability-tolerance", type=float, default=BIT_PROBABILITY_TOLERANCE)
    p.add_argument("--pairwise-tvd-tolerance", type=float, default=PAIRWISE_TVD_TOLERANCE)
    p.add_argument("--familywise-alpha", type=float, default=FAMILYWISE_ALPHA)
    p.add_argument("--output", type=str, default=None)
    return p


def validate_args(args: argparse.Namespace) -> None:
    for name in ("train_samples", "validation_samples", "test_samples"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_','-')} must be >= 1")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    for name in ("class_ratio_tolerance", "bit_probability_tolerance", "pairwise_tvd_tolerance"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_','-')} must be in [0,1]")
    if not 0.0 < args.familywise_alpha < 1.0:
        raise ValueError("--familywise-alpha must be in (0,1)")


def default_output(args: argparse.Namespace) -> Path:
    return Path(
        "audit/dataset/evidence/d3/"
        f"d3_gohr_distribution_consistency_{args.train_samples}_"
        f"{args.validation_samples}_{args.test_samples}_{args.rounds}r.json"
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output = Path(args.output) if args.output else default_output(args)

    print("=" * 78)
    print("Gohr D3 — Marginal and Low-Order Distributional Consistency Audit")
    print("=" * 78)
    print(f"Dataset                    : {DATASET_ID}")
    print(f"Dataset version            : {DATASET_VERSION}")
    print(f"Speck rounds               : {args.rounds}")
    print(f"Feature bits               : {FEATURE_BITS}")
    print(f"Training samples           : {args.train_samples}")
    print(f"Validation samples         : {args.validation_samples}")
    print(f"Test samples               : {args.test_samples}")
    print("Generator                  : GohrAdapter -> speck.make_train_data")
    print("D3 scope                   : marginal + second-order partition consistency")
    print("=" * 78)

    train_x, train_y, val_x, val_y, test_x, test_y = generate_gohr_dataset(
        args.train_samples, args.validation_samples, args.test_samples, args.rounds
    )

    intended = {
        "expected_class_ratio": EXPECTED_CLASS_RATIO,
        "expected_bit_probability": EXPECTED_BIT_PROBABILITY,
        "class_ratio_tolerance": args.class_ratio_tolerance,
        "bit_probability_tolerance": args.bit_probability_tolerance,
        "pairwise_tvd_tolerance": args.pairwise_tvd_tolerance,
    }

    results = audit_distribution(
        train_x, train_y,
        val_x, val_y,
        test_x, test_y,
        intended_distribution=intended,
        familywise_alpha=args.familywise_alpha,
    )
    decision = evaluate_decision(results)
    print_report(results, decision)

    certificate = {
        "certificate_schema_version": "1.0",
        "audit": {
            "id": "D3",
            "name": "Marginal and Low-Order Distributional Consistency Audit",
            "claim": (
                "The observed Gohr partitions are consistent with the explicitly "
                "specified intended first-order distribution and reproduce their "
                "second-order feature-pair structure across independently generated partitions "
                "within pre-specified practical limits."
            ),
        },
        "decision": decision,
        "hypothesis": {
            "null": (
                "Within the declared D3 scope, observed class balance, binary feature "
                "marginals, and second-order feature-pair distributions are consistent "
                "with the intended distribution and across independent partitions."
            ),
            "alternative": (
                "One or more declared first- or second-order distributional properties "
                "show a statistically and practically meaningful inconsistency."
            ),
        },
        "findings": results,
        "methodology": {
            "scope": "marginal and low-order distributional consistency",
            "first_order": [
                "binary class balance",
                "per-feature binary marginals",
                "global feature mean and variance as descriptive diagnostics",
                "per-feature binary entropy as a descriptive diagnostic",
            ],
            "second_order": {
                "object": "all unordered feature pairs",
                "pair_count": FEATURE_BITS * (FEATURE_BITS - 1) // 2,
                "comparison": "2x4 chi-square homogeneity between independently generated partitions",
                "effect_size": "total variation distance between 4-cell joint distributions",
                "practical_tolerance": args.pairwise_tvd_tolerance,
                "interpretation": (
                    "Second-order testing is partition-consistency testing; it does not assume "
                    "that Gohr ciphertext features are mutually independent or uniformly paired."
                ),
            },
            "confidence_level": CONFIDENCE_LEVEL,
            "multiple_comparison": {
                "method": "Bonferroni",
                "familywise_alpha": args.familywise_alpha,
                "first_order_tests_per_partition": FEATURE_BITS + 1,
                "second_order_tests_per_partition_pair": FEATURE_BITS * (FEATURE_BITS - 1) // 2,
            },
            "practical_significance": {
                "class_ratio_tolerance": args.class_ratio_tolerance,
                "bit_probability_tolerance": args.bit_probability_tolerance,
                "pairwise_tvd_tolerance": args.pairwise_tvd_tolerance,
                "rationale": (
                    "Pre-specified practical-equivalence thresholds are reported separately from "
                    "statistical significance; they are not claims of universal equality."
                ),
            },
            "scope_separation": {
                "D1": "exact duplicates and exact cross-partition overlap",
                "D2": "sample-order dependence, pairwise sample Hamming structure, and near-duplicate structure",
                "D3": "marginal and low-order distributional consistency",
            },
        },
        "provenance": {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "generation_procedure": "GohrAdapter.generate_partition() -> speck.make_train_data(n, nr)",
            "generator": "speck.make_train_data",
            "randomness_source": "os.urandom",
            "num_rounds": args.rounds,
            "feature_bits": FEATURE_BITS,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "partitions_generated_independently": True,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
        "limitations": [
            "D3 does not prove equality of the complete high-dimensional joint distribution.",
            "Second-order tests detect changes in feature-pair structure across partitions but cannot detect a higher-order difference that leaves all tested lower-order marginals unchanged.",
            "D3 does not perform exact duplicate or exact cross-partition overlap testing; D1 owns those threats.",
            "D3 does not perform sample-order or pairwise Hamming dependence testing; D2 owns those threats.",
            "Gohr's public generator interface does not expose internal key/plaintext records as separate audit fields; D3 therefore does not fabricate unavailable histograms.",
            "Statistical significance and practical significance are interpreted separately; statistical rejection alone is not a confirmed dataset confounder under the framework.",
            "The final audit conclusion is limited to the declared sample sizes, tests, thresholds, and Gohr configuration.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Certificate                : {output}")


if __name__ == "__main__":
    main()
