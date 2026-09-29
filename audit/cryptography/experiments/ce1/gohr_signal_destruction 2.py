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
from audit.cryptography.preflight import preflight, require_frozen
from audit.cryptography.provenance import EXPERIMENT_DESIGN_VERSION, build_provenance

EXPERIMENT_ID = "CE1-SIGNAL-DESTRUCTION"


def run(*, n_blocks, equivalence_margin, output_path, train_eval_fn,
        data_fn, repo_root=None, production=True, seed_base=1000):
    """
    Execute CE1.

    `data_fn(block_index)` returns (X_train, Y_train, X_eval, Y_eval);
    `train_eval_fn(X_train, Y_train, X_eval, Y_eval)` trains one model and
    returns its accuracy on the untouched evaluation set. Injecting both
    keeps the scientific design testable on CPU without stubbing science.
    """
    if production:
        require_frozen("n_blocks", n_blocks,
                       why="the replication count fixes the design's power.")
        require_frozen("equivalence_margin", equivalence_margin,
                       why="equivalence to chance needs a predeclared margin; "
                           "p > alpha is not evidence of equality.")
    pre = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                    differential=REFERENCE.differential, depth=REFERENCE.depth,
                    l2_reg=REFERENCE.l2_reg, checkpoint=None, output_path=output_path,
                    repo_root=repo_root,
                    frozen={"n_blocks": n_blocks, "equivalence_margin": equivalence_margin},
                    production=production)

    baseline_acc, destroyed_acc, blocks = [], [], []
    for i in range(n_blocks):
        Xtr, Ytr, Xev, Yev = data_fn(i)
        block = build_ce1_block(Xtr, Ytr, Xev, Yev, block_id=f"block{i}",
                                permutation_seed=seed_base + i)
        blocks.append(block.verify_invariants())
        baseline_acc.append(train_eval_fn(block.X_train, block.Y_train_baseline,
                                          block.X_eval, block.Y_eval))
        destroyed_acc.append(train_eval_fn(block.X_train, block.Y_train_destroyed,
                                           block.X_eval, block.Y_eval))

    results = paired_block_analysis(baseline_acc, destroyed_acc,
                                    equivalence_margin=equivalence_margin)
    results["blocks"] = blocks
    cert = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_design_version": EXPERIMENT_DESIGN_VERSION,
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
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--equivalence-margin", type=float, default=None)
    ap.add_argument("--output", type=Path,
                    default=Path("evidence_current/ce1/certificate.json"))
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    production = not args.dry_run
    if args.preflight:
        try:
            rep = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                            differential=REFERENCE.differential, depth=REFERENCE.depth,
                            l2_reg=REFERENCE.l2_reg, checkpoint=None,
                            output_path=args.output, repo_root=args.repo_root,
                            frozen={"n_blocks": args.n_blocks,
                                    "equivalence_margin": args.equivalence_margin},
                            production=production)
        except Exception as exc:
            print(f"PREFLIGHT FAILED: {exc}")
            return 1
        print(json.dumps(rep, indent=2, default=str))
        return 0

    if args.dry_run:
        data_fn, train_eval_fn = _toy()
        path, _ = run(n_blocks=args.n_blocks or 4, equivalence_margin=args.equivalence_margin,
                      output_path=args.output, train_eval_fn=train_eval_fn, data_fn=data_fn,
                      repo_root=args.repo_root, production=False)
        print(f"DRY RUN (non-evidentiary) wrote {path}")
        return 0

    print("Production CE1 requires a GPU trainer; wire train_eval_fn/data_fn and supply "
          "--n-blocks and --equivalence-margin.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
