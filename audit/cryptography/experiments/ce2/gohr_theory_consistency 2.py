"""
CE2 - Association with the Analytical Single-Trail Quantity

QUESTION
    Is the frozen model's output monotonically consistent with the
    independently computed analytical single-trail quantity?

DESIGN
    OBSERVATIONAL. Multiple independent theory datasets; one Spearman rho
    per run; uncertainty reported at the RUN level, because samples inside
    a run are not independent replicates of the model. Cannot establish
    causal use of the target.

Historical CE2 artifacts under evidence/ce2/ are preserved unchanged as
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
from audit.cryptography.experiments.ce2.design import ANALYTICAL_TARGET, run_level_association

EXPERIMENT_ID = "CE2-THEORY-CONSISTENCY"
REFERENCE_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "Archive" / "best5depth10.h5"
)


def _certificate(results, pre, *, production, seed, scope):
    cert = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_design_version": EXPERIMENT_DESIGN_VERSION,
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


def run(*, runs, output_path, checkpoint=REFERENCE_CHECKPOINT, repo_root=None,
        production=True, seed=0):
    pre = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                    differential=REFERENCE.differential, depth=REFERENCE.depth,
                    l2_reg=REFERENCE.l2_reg,
                    checkpoint=checkpoint if production else None,
                    output_path=output_path, repo_root=repo_root, production=production)
    results = run_level_association(runs)
    scope = ("Observational association between this frozen model's output and the "
             "analytical single-trail quantity. NOT causal; NOT the cipher's exact "
             "differential probability.")
    return write_certificate(_certificate(results, pre, production=production, seed=seed,
                                          scope=scope), output_path,
                             repo_root=repo_root), results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CE2 - analytical single-trail association")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("evidence_current/ce2/certificate.json"))
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
                            production=production)
        except Exception as exc:
            print(f"PREFLIGHT FAILED: {exc}")
            return 1
        print(json.dumps(rep, indent=2, default=str)); return 0
    if args.dry_run:
        runs = [{"run_id": f"toy{i}", "rho": r, "n": 500}
                for i, r in enumerate([-0.21, -0.19, -0.20, -0.22, -0.18])]
        path, _ = run(runs=runs, output_path=args.output, repo_root=args.repo_root,
                      production=False)
        print(f"DRY RUN (non-evidentiary) wrote {path}"); return 0
    print("Production CE2 requires the theory/model prediction runs; supply them to run().")
    return 2


if __name__ == "__main__":
    sys.exit(main())
