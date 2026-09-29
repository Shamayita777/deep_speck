"""
CE3 - Representation Decodability of the Analytical Target

QUESTION
    Is the analytical target decodable from the model's internal
    representation beyond a matched control?

DESIGN
    20 INDEPENDENT evaluation replicates, 5-fold CV nested inside each.
    The statistical unit is one stabilized selectivity per replicate -
    never the CV fold. Calibration is a methodological positive control
    for the probing pipeline, not direct cryptographic evidence.

Historical CE3 artifacts under evidence/ce3/ are preserved unchanged as
historical evidence; this file is the authoritative production path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit.cryptography.audit_config import REFERENCE
from audit.cryptography.certificate import CERTIFICATE_SCHEMA_VERSION, write_certificate
from audit.cryptography.preflight import preflight, require_frozen
from audit.cryptography.provenance import EXPERIMENT_DESIGN_VERSION, build_provenance
from audit.cryptography.experiments.ce3.design import DECISION_RULE, aggregate_replicates
from audit.cryptography.frozen_design import CE3 as CE3_SPEC
from audit.cryptography.frozen_design import DESIGN_SPECIFICATION_VERSION

EXPERIMENT_ID = "CE3-REPRESENTATION-INTERPRETATION"
REFERENCE_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "Archive"
    / "best5depth10.h5"
)


def _certificate(results, pre, *, production, seed, scope):
    cert = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_design_version": EXPERIMENT_DESIGN_VERSION,
        "design_specification_version": DESIGN_SPECIFICATION_VERSION,
        "reference_configuration": REFERENCE.to_dict(),
        "provenance": build_provenance(config_id=EXPERIMENT_ID, probe_seed=seed,
                                       checkpoint_info=pre["checkpoint"]),
        "preflight": pre,
        "results": results,
        "claim_scope": scope,
    }
    if not production:
        cert["non_evidentiary"] = True
        cert["claim_scope"] = ("TOY DRY RUN - structural validation only, "
                               "not scientific evidence")
    return cert


def run(*, selectivity_replicates, calibration_validated,
        n_splits_per_replicate=None, alpha=None, calibration_p_value=None,
        output_path, checkpoint=REFERENCE_CHECKPOINT, repo_root=None,
        production=True, seed=0):
    n_splits_per_replicate = (CE3_SPEC.n_splits_per_replicate
                              if n_splits_per_replicate is None else n_splits_per_replicate)
    alpha = CE3_SPEC.alpha if alpha is None else alpha
    if production:
        n_rep = len(list(selectivity_replicates))
        if n_rep != CE3_SPEC.n_replicates:
            raise ValueError(
                f"frozen design fixes n_replicates={CE3_SPEC.n_replicates}; got {n_rep}")
        if alpha != CE3_SPEC.alpha:
            raise ValueError(
                f"frozen design fixes alpha={CE3_SPEC.alpha} (fixed-sequence, no "
                f"adjustment); got {alpha}")
    pre = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                    differential=REFERENCE.differential, depth=REFERENCE.depth,
                    l2_reg=REFERENCE.l2_reg,
                    checkpoint=checkpoint if production else None,
                    output_path=output_path, repo_root=repo_root,
                    frozen={"alpha": alpha,
                            "multiplicity": CE3_SPEC.multiplicity,
                            "calibration_role": CE3_SPEC.calibration_role},
                    production=production)
    results = aggregate_replicates(
        selectivity_replicates, n_splits_per_replicate=n_splits_per_replicate,
        calibration_validated=calibration_validated, alpha=alpha,
        calibration_p_value=calibration_p_value)
    scope = ("Decodability of the analytical target from this frozen model's "
             "representation relative to a matched control, at the replicate level.")
    return write_certificate(_certificate(results, pre, production=production, seed=seed,
                                          scope=scope), output_path,
                             repo_root=repo_root), results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CE3 - representation decodability")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--alpha", type=float, default=CE3_SPEC.alpha,
                    help=f"FROZEN at {CE3_SPEC.alpha} (fixed-sequence, no adjustment)")
    ap.add_argument("--output", type=Path, default=Path("evidence_current/ce3/certificate.json"))
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)
    production = not args.dry_run
    if args.preflight:
        try:
            rep = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                            differential=REFERENCE.differential, depth=REFERENCE.depth,
                            l2_reg=REFERENCE.l2_reg,
                            checkpoint=REFERENCE_CHECKPOINT if production else None,
                            output_path=args.output, repo_root=args.repo_root,
                            frozen={"alpha": args.alpha,
                                    "multiplicity": CE3_SPEC.multiplicity},
                            production=production)
        except Exception as exc:
            print(f"PREFLIGHT FAILED: {exc}")
            return 1
        print(json.dumps(rep, indent=2, default=str)); return 0
    if args.dry_run:
        import numpy as np
        sel = list(np.linspace(0.05, 0.25, 20))
        path, _ = run(selectivity_replicates=sel, n_splits_per_replicate=5,
                      calibration_validated=True, calibration_p_value=0.001,
                      output_path=args.output, repo_root=args.repo_root, production=False)
        print(f"DRY RUN (non-evidentiary) wrote {path}"); return 0
    print("Production CE3 requires the probe pipeline outputs and a frozen corrected alpha.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
