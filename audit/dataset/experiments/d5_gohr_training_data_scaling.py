"""
Gohr D5 — strengthened training-data scaling experiment.

Design:
    - default training sizes are 2.5M, 5M, 7.5M, and 10M;
    - optional --training-sizes may override the declared size conditions for
      controlled smoke/testing runs without changing the production default;
    - default 5 independent replicates;
    - one independently generated maximum-size dataset per replicate;
    - nested prefixes form the declared size conditions within each replicate;
    - the same model seed is reused across sizes within a replicate;
    - independent replicate seeds quantify dataset/model stochasticity;
    - one fixed 1M held-out test partition is shared by every condition;
    - 200 epochs and batch size 5000 are the defaults (overridable via
      --epochs / --batch-size, which now propagate consistently to the
      adapter, the training loop, the manifest, and the certificate);
    - resumable per-condition checkpoints and immutable run manifest;
    - the smallest declared training size is the explicit baseline condition
      for every paired contrast.

Dataset generation and seed semantics:
    speck.make_train_data() (via GohrAdapter.generate_partition) uses
    os.urandom() internally. The integer replicate/audit seed therefore does
    NOT control dataset content and does NOT make dataset generation
    deterministically replayable. It controls model initialization and this
    audit's own RNG stream only. Exact historical dataset replay is not
    available from that seed; the persisted, hash-verified dataset files
    produced by audit.dataset.d5_training_data_scaling are the
    reproducibility/resumability object instead. See
    DATASET_REPLAY_REASON / REPLICATE_SEED_ROLE below, which are passed
    through into every persisted metadata/state file and into the
    certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit.dataset.adapters.gohr import GohrAdapter
from audit.dataset.d5_training_data_scaling import (
    generate_certificate,
    print_report,
    run_d5,
    write_learning_curve_plot,
)

NUM_ROUNDS = 5
DEPTH = 10
EPOCHS = 200
BATCH_SIZE = 5000
TEST_SAMPLES = 1_000_000
TRAINING_SIZES = (2_500_000, 5_000_000, 7_500_000, 10_000_000)
REPLICATES = 5
BOOTSTRAP_REPLICATES = 5000
DEFAULT_AUDIT_SEED = 0

DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"
OUTPUT_DIRECTORY = Path("audit/dataset/evidence/d5")

# Gohr's generator (speck.make_train_data, via GohrAdapter.generate_partition)
# draws from os.urandom() internally. These two strings are the accurate,
# case-study-specific provenance text passed into the generic D5 framework;
# they are what ends up in every persisted dataset/test metadata file, every
# per-condition checkpoint state, the run manifest, and the certificate. Do
# NOT change these to imply determinism the generator does not have.
DATASET_REPLAY_REASON = (
    "Gohr dataset generation delegates to speck.make_train_data() (via "
    "GohrAdapter.generate_partition), which draws from os.urandom() "
    "internally. The integer audit/replicate seed does not control dataset "
    "content and cannot be used to deterministically regenerate a "
    "bit-identical dataset. The persisted dataset files, together with their "
    "file and logical content SHA-256 hashes (including a hash for every "
    "declared nested-prefix size), are the reproducibility/resumability "
    "object instead."
)
REPLICATE_SEED_ROLE = (
    "model_initialization_and_audit_RNG_stream; NOT dataset generation. "
    "Gohr's generator (speck.make_train_data via os.urandom) is independent "
    "of this seed."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the strengthened Gohr D5 scaling audit.")
    p.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    p.add_argument("--replicates", type=int, default=REPLICATES)
    p.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    p.add_argument("--test-samples", type=int, default=TEST_SAMPLES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument(
        "--training-sizes",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Override the declared training-set sizes. Supply at least two "
            "positive integers. If omitted, the production TRAINING_SIZES "
            "defaults are used unchanged."
        ),
    )
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def resolve_training_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    """Resolve the declared D5 size conditions without changing production defaults.

    The omitted-argument path returns TRAINING_SIZES exactly. Supplied values
    are validated, sorted, and deduplicated so the CLI follows the framework's
    canonical size-condition ordering semantics.
    """
    if args.training_sizes is None:
        return TRAINING_SIZES

    if len(args.training_sizes) < 2:
        raise ValueError("--training-sizes must contain at least two sizes.")
    if any(size <= 0 for size in args.training_sizes):
        raise ValueError("every --training-sizes value must be a positive integer.")

    resolved = tuple(sorted(set(args.training_sizes)))
    if len(resolved) < 2:
        raise ValueError(
            "--training-sizes must contain at least two distinct positive sizes."
        )
    return resolved


def validate_args(args: argparse.Namespace) -> None:
    if args.audit_seed < 0:
        raise ValueError("--audit-seed must be non-negative.")
    if args.replicates < 2:
        raise ValueError("--replicates must be >= 2.")
    if args.bootstrap_replicates < 1000:
        raise ValueError("--bootstrap-replicates must be >= 1000.")
    if args.test_samples < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("test-samples, epochs and batch-size must be positive.")


def make_adapter(seed: int, *, epochs: int, batch_size: int) -> GohrAdapter:
    """Construct a GohrAdapter whose epochs/batch_size genuinely reflect the
    values actually in effect for this run (CLI overrides included), rather
    than the module-level EPOCHS/BATCH_SIZE defaults. Every caller below
    (dataset generation, model construction, training callbacks) goes
    through this single function so they cannot silently disagree.
    """
    return GohrAdapter(
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds=NUM_ROUNDS,
        depth=DEPTH,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )


def generate_gohr_partition(samples: int, seed: int, *, epochs: int, batch_size: int):
    # Gohr's original generator uses os.urandom(); seed is retained only as
    # provenance for the independent replicate/model RNG stream, and epochs/
    # batch_size are irrelevant to dataset generation but threaded through so
    # every adapter instance is constructed the same, consistent way.
    del seed
    return make_adapter(0, epochs=epochs, batch_size=batch_size).generate_partition(samples)


def evaluate_gohr_model(model, test_x, test_y):
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    return float(loss), float(accuracy)


def main() -> None:
    args = parse_args()
    validate_args(args)
    training_sizes = resolve_training_sizes(args)

    root = OUTPUT_DIRECTORY / "training_data_scaling_nested_replicates"

    # These factories close over the CLI-resolved args.epochs /
    # args.batch_size (Change 15 fix: previously make_adapter used the
    # module constants EPOCHS/BATCH_SIZE regardless of --epochs/--batch-size,
    # which could silently desynchronize the adapter's own configuration
    # from the declared run configuration).
    def train_dataset_factory(samples: int, seed: int):
        return generate_gohr_partition(samples, seed, epochs=args.epochs, batch_size=args.batch_size)

    def test_dataset_factory(samples: int, seed: int):
        return generate_gohr_partition(samples, seed, epochs=args.epochs, batch_size=args.batch_size)

    def model_factory(seed: int):
        return make_adapter(seed, epochs=args.epochs, batch_size=args.batch_size).build_model()

    def training_callbacks_factory(seed: int):
        return make_adapter(seed, epochs=args.epochs, batch_size=args.batch_size).training_callbacks()

    print("=" * 72)
    print("Gohr D5 — Strengthened Training-Data Scaling Audit")
    print("=" * 72)
    print(f"Dataset                    : {DATASET_ID}")
    print(f"Dataset version            : {DATASET_VERSION}")
    print(f"Speck rounds               : {NUM_ROUNDS}")
    print(f"ResNet depth               : {DEPTH}")
    print(f"Epochs                     : {args.epochs}")
    print(f"Batch size                 : {args.batch_size}")
    print(f"Fixed test samples         : {args.test_samples:,}")
    print(f"Training sizes             : {', '.join(f'{n:,}' for n in training_sizes)}")
    print(f"Baseline training size     : {min(training_sizes):,}")
    print(f"Independent replicates     : {args.replicates}")
    print(f"Bootstrap replicates       : {args.bootstrap_replicates}")
    print(
        f"Dataset design             : independent {max(training_sizes):,}-sample "
        "replicate + nested prefixes"
    )
    print("Model-seed design          : same seed across sizes within replicate")
    print("Generator                  : speck.make_train_data")
    print("Generator randomness       : os.urandom (dataset replay NOT available from audit seed)")
    print("Smoke runs                 : non-evidentiary")
    print()

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "training_sizes": list(training_sizes),
        "baseline_training_size": min(training_sizes),
        "model": {"architecture": "Gohr tn.make_resnet", "depth": DEPTH, "reg_param": 1e-5},
        "optimizer": "adam",
        "loss": "mse",
        "training_procedure": "Gohr original cyclic learning-rate schedule",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "generation": {
            "procedure": "speck.make_train_data(n, nr)",
            "num_rounds": NUM_ROUNDS,
            "randomness_source": "os.urandom",
            "deterministic_seed": None,
            "dataset_replay_available": False,
            "dataset_replay_reason": DATASET_REPLAY_REASON,
            "input_difference": {"left_word": "0x0040", "right_word": "0x0000"},
        },
    }

    result = run_d5(
        root=root,
        training_sizes=training_sizes,
        test_samples=args.test_samples,
        replicates=args.replicates,
        audit_seed=args.audit_seed,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        train_dataset_factory=train_dataset_factory,
        test_dataset_factory=test_dataset_factory,
        model_factory=model_factory,
        training_callbacks_factory=training_callbacks_factory,
        evaluate_model=evaluate_gohr_model,
        experiment_name="Gohr Speck training-data scaling with nested independent replicates",
        bootstrap_replicates=args.bootstrap_replicates,
        manifest=manifest,
        dataset_replay_reason=DATASET_REPLAY_REASON,
        replicate_seed_role=REPLICATE_SEED_ROLE,
    )

    print_report(result)

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "generation_procedure": "speck.make_train_data(n, nr)",
        "generation_parameters": {
            "generator": "speck.make_train_data",
            "num_rounds": NUM_ROUNDS,
            "training_sizes": list(training_sizes),
            "baseline_training_size": min(training_sizes),
            "replicates": args.replicates,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "randomness_source": "os.urandom",
            "deterministic_seed": None,
            "dataset_replay_available": False,
            "dataset_replay_reason": DATASET_REPLAY_REASON,
            "input_difference": {"left_word": "0x0040", "right_word": "0x0000"},
            "smoke_runs_non_evidentiary": True,
        },
    }
    certificate = generate_certificate(
        result,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        generation_procedure=provenance["generation_procedure"],
        generation_parameters=provenance["generation_parameters"],
        audit_seed=args.audit_seed,
    )

    output_path = (
        Path(args.output) if args.output else
        OUTPUT_DIRECTORY / "d5_gohr_training_data_scaling_nested_replicates.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_path = output_path.with_suffix(".png")
    write_learning_curve_plot(result, plot_path)

    print()
    print("=" * 72)
    print("D5 COMPLETE")
    print("=" * 72)
    print(f"Certificate                : {output_path}")
    print(f"Learning-curve plot        : {plot_path}")
    print("Interpretation             : sample-size scaling under the declared Gohr protocol")


if __name__ == "__main__":
    main()
