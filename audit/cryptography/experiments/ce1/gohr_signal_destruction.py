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

FROZEN PARAMETERS (frozen_design.CE1; validated, never chosen)
    n_blocks = 8, min_valid_blocks = 6, failure_tolerance = 2
    equivalence-to-chance DISABLED (no defensible margin at this test-set size)

PRODUCTION PATH: experiments/ce1/controller.py only. run() here remains for
non-evidentiary structural dry runs; run(production=True) is retired.
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
        # Frozen-parameter guards fire FIRST (defence in depth, unchanged)...
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
        # ...then the closure path is refused outright. It held training data only
        # in memory (os.urandom, not replayable) and passed validation through a
        # mutable attribute, so no interrupted arm could be resumed. Production
        # goes ONLY through the resumable controller (experiments/ce1/controller.py).
        raise RuntimeError(
            "run(production=True) is retired: use the CE1 controller "
            "(python -m audit.cryptography.experiments.ce1.gohr_signal_destruction "
            "--execute ...). run() remains only for non-evidentiary structural dry runs.")
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


# build_production_components() (data_fn / train_eval_fn closures with
# data_fn.last_validation) was REMOVED: see controller.BlockContext, which
# makes the seven block arrays explicit, persisted and hash-verified.


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
                    default=Path("audit/cryptography/evidence_current/ce1/certificate.json"))
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--validate-resume", type=Path, default=None,
                    help="read-only audit of an existing production run directory")
    ap.add_argument("--resume-dry-run", type=Path, default=None,
                    help="show the resume/GPU schedule for a run directory; trains nothing")
    ap.add_argument("--gpus", type=int, default=2, help="(ignored: GPUs are detected)")
    ap.add_argument("--run-dir", type=Path,
                    default=Path("audit/cryptography/evidence_current/ce1/production_resumable"))
    ap.add_argument("--legacy-run-dir", type=Path,
                    default=Path("audit/cryptography/evidence_current/ce1/production_20260929"))
    ap.add_argument("--sealed-source", type=Path,
                    default=Path("audit/cryptography/evidence_current/ce1/"
                                 "production_20260929/sealed"))
    ap.add_argument("--legacy-block0", choices=("RECOVER", "RETRAIN"), default=None,
                    help="recorded once in the run manifest; cannot be changed")
    ap.add_argument("--verify-legacy-block0", action="store_true",
                    help="read-only verification of the legacy terminal models (no accuracy)")
    ap.add_argument("--recover-legacy-block0", action="store_true",
                    help="seed block0 from the legacy FINAL_EPOCH models (no evaluation)")
    ap.add_argument("--status", action="store_true", help="read-only status + plan")
    ap.add_argument("--execute", action="store_true",
                    help="REQUIRED to train/evaluate in production")
    ap.add_argument("--operator-action", choices=("GRANT_RETRY", "REEVALUATE",
                                                  "RESTART_BLOCK"), default=None,
                    help="recorded engineering action on a NEEDS_OPERATOR block; cannot "
                         "fail, exclude or select a block")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--arm", choices=("baseline", "destroyed"), default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--finalize", action="store_true",
                    help="final validity gate + analysis of ALL valid blocks")
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
    from audit.cryptography.experiments.ce1 import controller as C
    from audit.cryptography.experiments.ce1.resume import ResumeRefused as R_err

    run_dir = args.run_dir
    legacy_dir = args.legacy_run_dir

    if args.verify_legacy_block0:
        from audit.cryptography.experiments.ce1.legacy_recovery import verify_legacy_block0
        rep = verify_legacy_block0(legacy_dir, C.CE1RunConfig.production_config())
        print(json.dumps(rep, indent=2, default=str))
        return 0 if rep["verified"] else 1

    if args.validate_resume or args.resume_dry_run or args.status:
        target = args.validate_resume or args.resume_dry_run or run_dir
        try:
            cfg = C.load_run_config(target)
            sealed = C.ensure_sealed(target, cfg)
        except Exception as exc:                     # noqa: BLE001
            print(f"RESUME REFUSED: {exc}")
            return 1
        print(json.dumps(C.run_status(target, cfg, sealed), indent=2))
        for i in range(cfg.n_blocks):
            plan = C.plan_block(target, cfg, i, sealed)
            acts = {a: v["action"] + (f" [{v['code']}]" if v.get("code") else "")
                    for a, v in plan["arms"].items()}
            print(f"  {plan['block_id']}: {acts}")
        if args.resume_dry_run:
            from audit.cryptography.experiments.ce1.worker import detect_gpus
            print(f"GPU detection: {detect_gpus()}")
            print("(dry run - nothing was trained)")
        return 0

    if args.preflight:
        try:
            # F4 FIX: preflight now runs the SAME frozen-parameter validator as
            # the production gate.
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

    # ---- genuine production entry points (controller only) ----
    try:
        validate_frozen_parameters(
            n_blocks=args.n_blocks, min_valid_blocks=args.min_valid_blocks,
            seed_base=args.seed_base, train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            evaluation_samples=args.evaluation_samples)
    except ValueError as exc:
        print(exc)
        return 1
    cfg = C.CE1RunConfig.production_config()

    if args.recover_legacy_block0:
        from audit.cryptography.experiments.ce1.legacy_recovery import recover_legacy_block0
        C.open_run(run_dir, cfg, repo_root=args.repo_root, legacy_block0_decision="RECOVER")
        C.ensure_sealed(run_dir, cfg, sealed_source=args.sealed_source)
        out = recover_legacy_block0(legacy_dir, run_dir, cfg)
        print(f"block0 seeded from legacy terminal models: {out['status']} "
              "(LEGACY_RECOVERED_PENDING_EVALUATION; evaluation happens in the next "
              "--execute pass; counting is decided by the final validity gate)")
        return 0

    if args.operator_action:
        if args.block is None or not args.reason:
            print("--operator-action requires --block and --reason")
            return 1
        try:
            out = C.operator_action(run_dir, C.load_run_config(run_dir), args.block,
                                    args.operator_action, reason=args.reason, arm=args.arm)
        except (C.ControllerError, R_err) as exc:
            print(f"OPERATOR ACTION REFUSED: {exc}")
            return 1
        print(json.dumps(out, indent=2))
        return 0

    if args.finalize:
        try:
            path, cert = C.finalize(run_dir, C.load_run_config(run_dir),
                                    repo_root=args.repo_root)
        except C.ControllerError as exc:
            print(f"NOT FINALIZED: {exc}")
            return 1
        print(f"CE1 certificate written: {path}")
        return 0

    if not args.execute:
        print("Nothing executed. Production training requires --execute "
              "(use --status / --resume-dry-run to inspect).")
        return 1
    C.open_run(run_dir, cfg, repo_root=args.repo_root,
               legacy_block0_decision=args.legacy_block0)
    from audit.cryptography.experiments.ce1.worker import run_auto
    out = run_auto(run_dir, cfg, sealed_source=args.sealed_source, repo_root=args.repo_root)
    print(json.dumps({"execution": out["execution"], "status": out["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
