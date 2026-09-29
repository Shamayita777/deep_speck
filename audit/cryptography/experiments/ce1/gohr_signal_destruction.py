"""
CE1 - Distinguishing-Signal Destruction (current audit implementation)

QUESTION
    Does destroying the differential training signal cause loss of
    distinguishing performance?

DESIGN (see experiments/ce1/design.py)
    Independent paired blocks. Within a block ONE dataset is generated and
    shared by both arms; ONLY the TRAINING labels are permuted; the
    evaluation set and its labels are identical for both arms. The
    statistical unit is the per-block paired difference.

    Historical CE1 generated different datasets per arm and shuffled the
    validation labels too, so the two models were scored against different
    targets. That artifact is preserved under evidence/ce1/ as historical
    evidence; this file is the authoritative production path.

FROZEN PARAMETERS REQUIRED FOR PRODUCTION
    n_blocks           - predeclared from a power rationale
    equivalence_margin - predeclared; without it the destroyed arm's
                         comparison to chance can only be INCONCLUSIVE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from audit.cryptography.audit_config import REFERENCE
from audit.cryptography.certificate import CERTIFICATE_SCHEMA_VERSION, write_certificate
from audit.cryptography.experiments.ce1.design import build_ce1_block, paired_block_analysis
from audit.cryptography.frozen_design import (
    CE1 as CE1_SPEC,
    CE1_REPORTING_RULE,
    DESIGN_SPECIFICATION_VERSION,
)
from audit.cryptography.preflight import preflight, require_frozen
from audit.cryptography.integrity import sha256_file
from audit.cryptography.sealed_dataset import verify_shared_invariant
from audit.cryptography.provenance import EXPERIMENT_DESIGN_VERSION, build_provenance

EXPERIMENT_ID = "CE1-SIGNAL-DESTRUCTION"

#: Deterministic seed manifest. Every seed is derived from one declared
#: base by a fixed rule, so the full assignment is reproducible and
#: auditable in advance. Arms within a block MUST receive different seeds:
#: they are independent models, not one fit evaluated twice.
FROZEN_SEED_BASE = 1000


def seed_manifest(n_blocks: int, seed_base: int = FROZEN_SEED_BASE,
                  assignment_seed: int = None) -> dict:
    """
    Block i -> its two model seeds, its permutation seed, and the RANDOMIZED
    assignment of seeds to arms.

    The assignment coin flip is what makes the exact sign-flip test valid.
    It is drawn from a declared `assignment_seed` that is SEPARATE from the
    model seeds, so the assignment is reproducible and auditable in advance
    while still being a genuine 1/2-1/2 randomization independent across
    blocks. Without it, intact always received the even seed and destroyed
    the odd one, and there was no randomization distribution to enumerate.
    """
    import numpy as _np

    assignment_seed = (CE1_SPEC.assignment_seed if assignment_seed is None
                       else assignment_seed)
    rng = _np.random.default_rng(assignment_seed)
    manifest = {}
    for i in range(n_blocks):
        s_a, s_b = seed_base + 2 * i, seed_base + 2 * i + 1
        flip = int(rng.integers(0, 2))          # 0 -> s_A to intact, 1 -> swapped
        manifest[f"block{i}"] = {
            "permutation": seed_base + i,
            "seed_A": s_a,
            "seed_B": s_b,
            "assignment_flip": flip,
            "baseline": s_b if flip else s_a,   # intact arm
            "destroyed": s_a if flip else s_b,  # signal-destroyed arm
        }
    model_seeds = [b[a] for b in manifest.values() for a in ("baseline", "destroyed")]
    return {
        "seed_base": seed_base,
        "rule": ("permutation = base + i; block seeds {s_A, s_B} = {base+2i, base+2i+1}; "
                 "assignment of {s_A, s_B} to {intact, destroyed} is RANDOMIZED by a fair "
                 "coin flip drawn from assignment_seed"),
        "assignment_seed": assignment_seed,
        "randomization": (
            "Per block, a prospectively recorded fair coin flip assigns which of the two "
            "model seeds goes to the intact arm. This is the randomization the exact "
            "sign-flip test enumerates; it is drawn before production from a declared seed "
            "and is fixed thereafter."),
        "n_blocks": n_blocks,
        "seeds": manifest,
        # F13: state precisely what is and is not distinct. MODEL-TRAINING seeds
        # must be pairwise distinct (the two arms are independent models). The
        # permutation seeds intentionally share the numeric range with them;
        # they index a different RNG use (label permutation) and no claim of
        # global distinctness across the two roles is made.
        "model_training_seeds_pairwise_distinct": len(set(model_seeds)) == len(model_seeds),
        "n_model_training_seeds": len(model_seeds),
        "distinctness_claim": ("model-training seeds are pairwise distinct; permutation "
                               "seeds are a separate role and are not claimed distinct "
                               "from them"),
        "dataset_generation_randomness": "os.urandom (NOT seed-replayable)",
        "reproducibility_scope": (
            "The declared seed governs model initialisation and training-time "
            "randomness. It does NOT make the generated Speck datasets replayable: "
            "they are drawn from os.urandom. Exact dataset replay requires the "
            "persisted artifacts, which is why the evaluation set is sealed."),
    }


#: Parameters that DEFINE the experiment. An operator may not change these
#: and still obtain a production certificate. Operational parameters
#: (--output, --repo-root, --preflight, --dry-run) remain free.
FROZEN_SCIENTIFIC_PARAMETERS = (
    "n_blocks", "min_valid_blocks", "seed_base",
    "train_samples", "validation_samples", "evaluation_samples",
)


def validate_frozen_parameters(**supplied) -> dict:
    """
    Reject any production run whose scientific parameters differ from the
    frozen specification. Defaults alone are insufficient: a default can be
    overridden on the command line, and a certificate would then record a
    design nobody preregistered.
    """
    expected = {
        "n_blocks": CE1_SPEC.n_blocks,
        "min_valid_blocks": CE1_SPEC.min_valid_blocks,
        "seed_base": CE1_SPEC.seed_base,
        "train_samples": CE1_SPEC.train_samples,
        "validation_samples": CE1_SPEC.validation_samples,
        "evaluation_samples": CE1_SPEC.evaluation_samples,
    }
    mismatches = {k: {"supplied": supplied[k], "frozen": v}
                  for k, v in expected.items()
                  if k in supplied and supplied[k] is not None and supplied[k] != v}
    if mismatches:
        raise ValueError(
            "REFUSING PRODUCTION: frozen scientific parameters were overridden: "
            f"{mismatches}. These define the preregistered experiment and may not be "
            "changed on the command line.")
    return expected


def _call_arm(fn, X_train, Y_train, X_eval, Y_eval, *, seed, arm, block_id):
    """Call the arm trainer, passing identity only if it accepts it (toy stubs do not)."""
    import inspect

    if "arm" in inspect.signature(fn).parameters:
        return fn(X_train, Y_train, X_eval, Y_eval, seed=seed, arm=arm, block_id=block_id)
    return fn(X_train, Y_train, X_eval, Y_eval)


def run(*, n_blocks=None, output_path, train_eval_fn, data_fn, repo_root=None,
        production=True, seed_base=FROZEN_SEED_BASE, min_valid_blocks=None):
    """
    Execute CE1.

    `data_fn(block_index)` returns (X_train, Y_train, X_eval, Y_eval);
    `train_eval_fn(X_train, Y_train, X_eval, Y_eval)` trains one model and
    returns its accuracy on the untouched evaluation set. Injecting both
    keeps the scientific design testable on CPU without stubbing science.
    """
    n_blocks = CE1_SPEC.n_blocks if n_blocks is None else n_blocks
    min_valid_blocks = (CE1_SPEC.min_valid_blocks if min_valid_blocks is None
                        else min_valid_blocks)
    if production:
        validate_frozen_parameters(n_blocks=n_blocks, min_valid_blocks=min_valid_blocks,
                                   seed_base=seed_base)
        if n_blocks != CE1_SPEC.n_blocks or min_valid_blocks != CE1_SPEC.min_valid_blocks:
            raise ValueError(
                f"frozen design fixes n_blocks={CE1_SPEC.n_blocks} and "
                f"min_valid_blocks={CE1_SPEC.min_valid_blocks}; got {n_blocks}/"
                f"{min_valid_blocks}. Changing the replication count invalidates the "
                "preregistered design.")
        if min_valid_blocks < 6:
            raise ValueError(
                "min_valid_blocks must be >= 6: the exact sign-flip test cannot reach "
                "alpha=0.05 with fewer blocks (min attainable p = 2/2^K).")
    pre = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                    differential=REFERENCE.differential, depth=REFERENCE.depth,
                    l2_reg=REFERENCE.l2_reg, checkpoint=None, output_path=output_path,
                    repo_root=repo_root,
                    frozen={"n_blocks": n_blocks, "min_valid_blocks": min_valid_blocks,
                            "checkpoint_rule": CE1_SPEC.checkpoint_rule},
                    production=production)

    seeds = seed_manifest(n_blocks, seed_base)
    baseline_acc, destroyed_acc, blocks, failures = [], [], [], []
    # F3 FIX: a failed block is RECORDED and execution continues. Failing fast
    # would discard every later block after an early crash, which on a
    # multi-session GPU run is the difference between losing one block and
    # losing the experiment. Completed blocks are unaffected by later failures.
    for i in range(n_blocks):
        bid = f"block{i}"
        try:
            Xtr, Ytr, Xev, Yev = data_fn(i)
            block = build_ce1_block(Xtr, Ytr, Xev, Yev, block_id=bid,
                                    permutation_seed=seeds["seeds"][bid]["permutation"])
            record = block.verify_invariants()
            record["sealed_evaluation_sha256"] = getattr(data_fn, "sealed_sha256", None)
        # Two INDEPENDENT models from the same block; only the training
        # labels differ. Distinct seeds so the arms are not the same fit.
            acc_b = _call_arm(train_eval_fn, block.X_train, block.Y_train_baseline,
                              block.X_eval, block.Y_eval,
                              seed=seeds["seeds"][bid]["baseline"], arm="baseline",
                              block_id=bid)
            art_b = getattr(train_eval_fn, "last_artifact", None)
            acc_d = _call_arm(train_eval_fn, block.X_train, block.Y_train_destroyed,
                              block.X_eval, block.Y_eval,
                              seed=seeds["seeds"][bid]["destroyed"], arm="destroyed",
                              block_id=bid)
            art_d = getattr(train_eval_fn, "last_artifact", None)
            record["artifacts"] = {"baseline": art_b, "destroyed": art_d}
            record["status"] = "VALID"
            baseline_acc.append(acc_b)
            destroyed_acc.append(acc_d)
            blocks.append(record)
        except Exception as exc:                    # noqa: BLE001 - recorded, not swallowed
            failures.append({"block_id": bid, "status": "FAILED",
                             "reason": type(exc).__name__, "detail": str(exc)})
            blocks.append({"block_id": bid, "status": "FAILED",
                           "reason": type(exc).__name__, "detail": str(exc)})

    n_valid = len(baseline_acc)
    if n_valid < min_valid_blocks:
        raise ValueError(
            f"only {n_valid} valid blocks ({len(failures)} failed); the frozen design "
            f"requires {min_valid_blocks}. Reporting a result below the preregistered "
            "minimum would change the design after the fact. Failures: "
            f"{[f['block_id'] for f in failures]}")
    # F1 invariant: every valid block must have used the SAME sealed set
    valid_records = [b for b in blocks if b.get("status") == "VALID"]
    if any(b.get("sealed_evaluation_sha256") for b in valid_records):
        shared = verify_shared_invariant(valid_records)
    else:
        shared = {"shared_evaluation_sha256": None,
                  "n_blocks_checked": len(valid_records)}
    results = paired_block_analysis(baseline_acc, destroyed_acc, alpha=CE1_SPEC.alpha)
    results["blocks"] = blocks
    results["min_valid_blocks"] = min_valid_blocks
    results["n_blocks_requested"] = n_blocks
    results["n_blocks_valid"] = n_valid
    results["failures"] = failures
    results["failure_tolerance"] = CE1_SPEC.failure_tolerance
    results["sealed_evaluation"] = shared
    results["seed_manifest"] = seeds
    results["reporting_rule"] = CE1_REPORTING_RULE
    results["frozen_parameters_verified"] = (
        validate_frozen_parameters(
            n_blocks=n_blocks, min_valid_blocks=min_valid_blocks, seed_base=seed_base)
        if production else {"production": False})
    results["training_protocol"] = {**PRODUCTION_TRAINING, **PRODUCTION_SAMPLES,
                                    "note": TRAINING_PROTOCOL_NOTE} if production else \
        {"note": "toy dry run - no real training performed"}
    cert = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_design_version": EXPERIMENT_DESIGN_VERSION,
        "design_specification_version": DESIGN_SPECIFICATION_VERSION,
        "reference_configuration": REFERENCE.to_dict(),
        "provenance": build_provenance(config_id=EXPERIMENT_ID, probe_seed=seed_base,
                                       checkpoint_info=pre["checkpoint"]),
        "preflight": pre,
        "results": results,
        "claim_scope": results["interpretation_scope"],
    }
    if not production:
        cert["non_evidentiary"] = True
        cert["claim_scope"] = "TOY DRY RUN - structural validation only, not scientific evidence"
    return write_certificate(cert, output_path, repo_root=repo_root), cert




# =====================================================================
# Production wiring: the real Gohr components
# =====================================================================

#: Reference training protocol. Declared explicitly - the repository's
#: GohrDataset defaults to rounds=7 and GohrModel to depth=5, so relying
#: on any constructor default here would silently reproduce the exact
#: class of error this audit exists to detect.
PRODUCTION_TRAINING = {
    "batch_size": 5000,
    "epochs": 200,
    "high_learning_rate": 0.002,
    "low_learning_rate": 0.0001,
    "optimizer": "adam",
    "loss": "mse",
    "save_best_only": True,
}
PRODUCTION_SAMPLES = {
    "train_samples": 10 ** 7,
    "validation_samples": 10 ** 6,
    "evaluation_samples": 10 ** 6,
}

TRAINING_PROTOCOL_NOTE = (
    "Within a block both arms share the training inputs, the training-time validation "
    "split and the held-out evaluation set; ONLY the training labels differ. The "
    "validation split used for checkpoint selection keeps INTACT labels in both arms "
    "(permuting it would change the destroyed arm's prediction target). The held-out "
    "evaluation set is generated separately and is never used for training or checkpoint "
    "selection, so the reported accuracies are not selected on."
)


def build_production_components(*, train_samples, validation_samples, evaluation_samples,
                                checkpoint_dir, sealed_dir):
    """
    Construct the real Gohr dataset/model/trainer/evaluator with EVERY
    scientifically important parameter passed explicitly.

    Returns (data_fn, train_eval_fn) for `run()`.
    """
    from audit.cryptography.gohr.dataset import GohrDataset
    from audit.cryptography.gohr.evaluate import GohrEvaluator
    from audit.cryptography.gohr.model import GohrModel
    from audit.cryptography.gohr.trainer import GohrTrainer
    from audit.cryptography.sealed_dataset import prepare_sealed_evaluation_set

    trainer = GohrTrainer(
        batch_size=PRODUCTION_TRAINING["batch_size"],
        epochs=PRODUCTION_TRAINING["epochs"],
        checkpoint_dir=checkpoint_dir,
        save_best_only=PRODUCTION_TRAINING["save_best_only"],
        high_learning_rate=PRODUCTION_TRAINING["high_learning_rate"],
        low_learning_rate=PRODUCTION_TRAINING["low_learning_rate"],
    )
    evaluator = GohrEvaluator(batch_size=PRODUCTION_TRAINING["batch_size"])

    # ---- F1 FIX: ONE sealed evaluation set, generated once, shared by all
    # blocks and both arms. Previously this was rebuilt inside data_fn, giving
    # every block a DIFFERENT evaluation sample - a direct violation of the
    # frozen design and of the "conditional on the fixed sealed test set"
    # inference statement.
    def _generate_eval(n):
        gen = GohrDataset(rounds=REFERENCE.rounds, differential=REFERENCE.differential,
                          train_samples=1, validation_samples=n)
        return gen.generate_baseline_dataset().validation

    sealed = prepare_sealed_evaluation_set(
        sealed_dir, generate_fn=_generate_eval, rounds=REFERENCE.rounds,
        differential=REFERENCE.differential, n_samples=evaluation_samples)

    def data_fn(block_index: int):
        """One independently generated training block + THE shared sealed set."""
        generator = GohrDataset(
            rounds=REFERENCE.rounds,                       # explicit: default is 7
            differential=REFERENCE.differential,
            train_samples=train_samples,
            validation_samples=validation_samples,
        )
        bundle = generator.generate_baseline_dataset()
        X_train, Y_train = bundle.train
        data_fn.last_validation = bundle.validation        # training-time validation split
        data_fn.sealed_sha256 = sealed["sha256"]
        return X_train, Y_train, sealed["X"], sealed["Y"]

    data_fn.sealed = sealed

    data_fn.last_validation = None
    train_eval_fn_artifact_holder = None

    def train_eval_fn(X_train, Y_train, X_eval, Y_eval, *, seed: int = 0,
                      arm: str = "arm", block_id: str = "block"):
        """
        Train ONE independent model on the supplied labels and score it.

        Seeding order is deliberate and load-bearing: set_seed -> build ->
        (trainer re-seeds) -> fit. See the F3 note below.
        """
        # F3 FIX: seed BEFORE the model is constructed. GohrModel(...).build()
        # initialises the weights immediately, but GohrTrainer.set_seed() runs
        # inside train() -- i.e. AFTER initialisation. The declared model seed
        # therefore controlled only shuffling/dropout, never the initial
        # weights, so the seed manifest overclaimed reproducibility. Seeding
        # here makes the declared seed govern initialisation as intended; the
        # trainer re-seeds identically before fit, which is harmless.
        GohrTrainer.set_seed(seed)
        model = GohrModel(
            depth=REFERENCE.depth,                         # explicit: default is 5
            regularization=REFERENCE.l2_reg,
            optimizer=PRODUCTION_TRAINING["optimizer"],
            loss=PRODUCTION_TRAINING["loss"],
        ).build()
        validation = data_fn.last_validation
        if validation is None:
            raise RuntimeError("no training-time validation split available for this block")
        trained, history = trainer.train(
            model, (X_train, Y_train), validation,
            seed=seed, checkpoint_name=f"{block_id}_{arm}_bestval_DEBUG_ONLY.keras",
        )
        # F5 FIX: the trainer's ModelCheckpoint artifact is a BEST_VAL_LOSS
        # model, but the frozen estimand is the TERMINAL-EPOCH model. Persist
        # the terminal model explicitly so the reported metric and the saved
        # artifact are the same object. The val_loss checkpoint is retained
        # only for debugging and is named accordingly.
        terminal_path = Path(checkpoint_dir) / f"{block_id}_{arm}_FINAL_EPOCH.keras"
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        trained.save(str(terminal_path))
        epochs_ran = len(history.history.get("loss", [])) if hasattr(history, "history") else None
        accuracy = float(evaluator.evaluate(trained, (X_eval, Y_eval)))
        train_eval_fn.last_artifact = {
            "checkpoint_rule": "FINAL_EPOCH",
            "terminal_model_path": str(terminal_path),
            "terminal_model_sha256": sha256_file(terminal_path),
            "epochs_completed": epochs_ran,
            "selection": "none - terminal epoch, no validation-based selection",
        }
        return accuracy

    return data_fn, train_eval_fn


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    def data_fn(i):
        r = np.random.default_rng(seed + i)
        return (r.integers(0, 2, (64, 8), dtype=np.uint8), r.integers(0, 2, 64, dtype=np.uint8),
                r.integers(0, 2, (32, 8), dtype=np.uint8), r.integers(0, 2, 32, dtype=np.uint8))
    def train_eval_fn(Xtr, Ytr, Xev, Yev):
        # deterministic stand-in: intact labels are learnable, permuted are not
        learnable = float(np.mean(Ytr[:len(Yev)] == Yev)) if len(Ytr) >= len(Yev) else 0.5
        return 0.5 + 0.4 * abs(learnable - 0.5) * 2
    return data_fn, train_eval_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CE1 - Distinguishing-Signal Destruction")
    ap.add_argument("--preflight", action="store_true", help="verify everything, run nothing")
    ap.add_argument("--dry-run", action="store_true", help="tiny CPU structural run")
    ap.add_argument("--n-blocks", type=int, default=CE1_SPEC.n_blocks,
                    help=f"FROZEN at {CE1_SPEC.n_blocks}")
    ap.add_argument("--min-valid-blocks", type=int, default=CE1_SPEC.min_valid_blocks,
                    help=f"FROZEN at {CE1_SPEC.min_valid_blocks}")
    ap.add_argument("--output", type=Path,
                    default=Path("evidence_current/ce1/certificate.json"))
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--seed-base", type=int, default=CE1_SPEC.seed_base,
                    help=f"FROZEN at {CE1_SPEC.seed_base}")
    ap.add_argument("--train-samples", type=int, default=CE1_SPEC.train_samples, help=f"FROZEN at {CE1_SPEC.train_samples}")
    ap.add_argument("--validation-samples", type=int,
                    default=CE1_SPEC.validation_samples,
                    help=f"FROZEN at {CE1_SPEC.validation_samples}")
    ap.add_argument("--evaluation-samples", type=int,
                    default=CE1_SPEC.evaluation_samples,
                    help=f"FROZEN at {CE1_SPEC.evaluation_samples}")
    args = ap.parse_args(argv)

    production = not args.dry_run
    if args.preflight:
        try:
            # F4 FIX: preflight now runs the SAME frozen-parameter validator as
            # the production gate. Previously it printed PREFLIGHT_OK for an
            # overridden --seed-base or sample count and only the production
            # branch refused, so the final gate before an expensive run could
            # bless a design nobody preregistered.
            if production:
                validate_frozen_parameters(
                    n_blocks=args.n_blocks, min_valid_blocks=args.min_valid_blocks,
                    seed_base=args.seed_base, train_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                    evaluation_samples=args.evaluation_samples)
            rep = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                            differential=REFERENCE.differential, depth=REFERENCE.depth,
                            l2_reg=REFERENCE.l2_reg, checkpoint=None,
                            output_path=args.output, repo_root=args.repo_root,
                            frozen={"n_blocks": args.n_blocks,
                                    "min_valid_blocks": args.min_valid_blocks,
                                    "seed_base": args.seed_base,
                                    "train_samples": args.train_samples,
                                    "validation_samples": args.validation_samples,
                                    "evaluation_samples": args.evaluation_samples,
                                    "checkpoint_rule": CE1_SPEC.checkpoint_rule,
                                    "terminal_epoch": CE1_SPEC.terminal_epoch},
                            production=production)
        except Exception as exc:
            print(f"PREFLIGHT FAILED: {exc}")
            return 1
        print(json.dumps(rep, indent=2, default=str))
        return 0

    if args.dry_run:
        data_fn, train_eval_fn = _toy()
        path, _ = run(n_blocks=args.n_blocks, min_valid_blocks=2,
                      output_path=args.output, train_eval_fn=train_eval_fn, data_fn=data_fn,
                      repo_root=args.repo_root, production=False)
        print(f"DRY RUN (non-evidentiary) wrote {path}")
        return 0

    # ---- genuine production entry point ----
    try:
        validate_frozen_parameters(
            n_blocks=args.n_blocks, min_valid_blocks=args.min_valid_blocks,
            seed_base=args.seed_base, train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            evaluation_samples=args.evaluation_samples)
    except ValueError as exc:
        print(exc)
        return 1

    checkpoint_dir = Path(args.output).parent / "checkpoints"
    sealed_dir = Path(args.output).parent / "sealed"
    data_fn, train_eval_fn = build_production_components(
        train_samples=args.train_samples, validation_samples=args.validation_samples,
        evaluation_samples=args.evaluation_samples, checkpoint_dir=checkpoint_dir,
        sealed_dir=sealed_dir,
    )
    path, cert = run(n_blocks=args.n_blocks, min_valid_blocks=args.min_valid_blocks,
                     output_path=args.output, train_eval_fn=train_eval_fn, data_fn=data_fn,
                     repo_root=args.repo_root, production=True, seed_base=args.seed_base)
    print(f"CE1 production certificate written: {path}")
    print(f"  baseline mean  {cert['results']['baseline_mean']:.6f}")
    print(f"  destroyed mean {cert['results']['destroyed_mean']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
