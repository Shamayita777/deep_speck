#!/usr/bin/env python3
"""
II-4 launcher (Implementation Integrity). Run from the repository root.

  Validate only (touches nothing):
      python -m audit.implementation.run_ii4 --mode production --output-dir <dir>

  Non-evidentiary smoke (tiny data, SMOKE mode):
      python -m audit.implementation.run_ii4 --mode smoke --output-dir <dir> --execute

  PRODUCTION (the real experiment) requires BOTH flags:
      python -m audit.implementation.run_ii4 --mode production --output-dir <dir> \
          --execute --confirm-production-ii4

Re-running the same production command resumes: nothing already produced
is regenerated, and every reused artifact is hash-verified.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EV_ROOT = REPO_ROOT / "audit" / "experimental_validity"
for p in (REPO_ROOT, EV_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit.implementation.ii4_experiment import (  # noqa: E402
    build_gohr_ii4_plan,
    build_gohr_ii4_smoke_plan,
)
from audit.implementation.ii4_runner import II4ExecutionError, II4Runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "production"), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--confirm-production-ii4", action="store_true")
    args = ap.parse_args()

    plan = (build_gohr_ii4_plan(output_dir=args.output_dir) if args.mode == "production"
            else build_gohr_ii4_smoke_plan(output_dir=args.output_dir))

    if args.execute:
        from audit.implementation.adapters.gohr_ii4 import GohrII4Adapter  # TF only when executing
        adapter = GohrII4Adapter(protocol=plan.fixed_protocol)
    else:
        adapter = None  # validation-only path never imports TensorFlow

    try:
        result = II4Runner(plan=plan, adapter=adapter).run(
            execute=args.execute, allow_production=args.confirm_production_ii4)
    except II4ExecutionError as exc:
        print(f"REFUSED: {exc}")
        return 1

    if not args.execute:
        print(json.dumps(result, indent=2))
        print(f"Validation only. experiment_id={plan.experiment_id} mode={plan.execution_mode.value}")
        return 0
    a = result["analysis"]
    print(f"{result['experiment_id']}  mode={result['execution_mode']}  "
          f"non_evidentiary={result['non_evidentiary']}")
    print(f"valid blocks {a['replication']['n_blocks_valid']}/{a['replication']['n_blocks_requested']}"
          f"  materiality={a['confirmatory']['materiality']['verdict']}"
          f"  equivalence={a['confirmatory']['equivalence']['verdict']}")
    print(f"written: {args.output_dir / 'ii4_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
