"""
CE4 - Ciphertext-Difference Intervention Sensitivity

QUESTION
    Is the frozen model's output more sensitive to the specified
    difference-bearing ciphertext structure than to a magnitude-matched,
    XOR-preserving control?

DESIGN
    Paired within-sample contrast on the eligible population. Both arms
    change the SAME number of bits; the control preserves the pair XOR
    exactly. Does NOT manipulate - and so cannot speak to - the analytical
    single-trail probability.

Historical CE4 artifacts under evidence/ce4/ are preserved unchanged as
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
from audit.cryptography.experiments.ce4.design import (
    ESTIMAND,
    EXPERIMENT_NAME,
    build_matched_interventions,
    eligible_mask,
    paired_contrast,
    population_accounting,
    verify_intervention_invariants,
)

EXPERIMENT_ID = "CE4-INTERVENTION-SENSITIVITY"
#: Resolved from the source tree (audit_config), not the working directory.
from audit.cryptography.audit_config import REFERENCE_CHECKPOINT_PATH as REFERENCE_CHECKPOINT  # noqa: E402


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


def run(*, structural_delta, control_delta, population, invariants, output_path,
        checkpoint=REFERENCE_CHECKPOINT, repo_root=None, production=True, seed=0):
    pre = preflight(experiment_id=EXPERIMENT_ID, rounds=REFERENCE.rounds,
                    differential=REFERENCE.differential, depth=REFERENCE.depth,
                    l2_reg=REFERENCE.l2_reg,
                    checkpoint=checkpoint if production else None,
                    output_path=output_path, repo_root=repo_root, production=production)
    results = paired_contrast(structural_delta, control_delta)
    results["population"] = population
    results["invariants"] = invariants
    return write_certificate(_certificate(results, pre, production=production, seed=seed,
                                          scope=results["claim_scope"]), output_path,
                             repo_root=repo_root), results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CE4 - intervention sensitivity")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("audit/cryptography/evidence_current/ce4/certificate.json"))
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
        import numpy as np
        rng = np.random.default_rng(5)
        n, d = 200, 32
        a = rng.integers(0, 2, (n, d), dtype=np.uint8); b = a.copy()
        for i in range(n):
            b[i, rng.choice(d, 4, replace=False)] ^= 1
        elig = eligible_mask(a, b, n_flips=2)
        acct = population_accounting(n, elig)
        ae, be = a[elig], b[elig]
        pos = np.array([np.flatnonzero(r)[:2] for r in (ae ^ be)])
        (sa, sb), (ca, cb) = build_matched_interventions(ae, be, pos)
        inv = verify_intervention_invariants(ae, be, sa, sb, ca, cb)
        path, _ = run(structural_delta=rng.normal(0.30, 0.05, ae.shape[0]),
                      control_delta=rng.normal(0.05, 0.05, ae.shape[0]),
                      population=acct, invariants=inv, output_path=args.output,
                      repo_root=args.repo_root, production=False)
        print(f"DRY RUN (non-evidentiary) wrote {path}"); return 0
    print("Production CE4 requires model output deltas from the intervention pipeline.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
