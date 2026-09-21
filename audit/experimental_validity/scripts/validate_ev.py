#!/usr/bin/env python3
"""
Pre-production validation gate.

Run before ANY production experiment. Exits non-zero (fails closed) if
any mandatory check fails. This is what Phase 6 ("freeze") of the
implementation order requires before confirmatory production runs may
begin.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from framework.provenance import software_provenance
from gohr import speck
from gohr.baseline import BASELINE, DOCUMENTED_SOFTWARE, compare_software_environment
from gohr.representation import generate_candidate1_permutation, run_full_validation


def main() -> int:
    print("=" * 70)
    print("EV PRE-PRODUCTION VALIDATION GATE")
    print("=" * 70)
    all_ok = True

    print("\n[1/3] Speck32/64 cipher correctness (check_testvector)...")
    if speck.check_testvector():
        print("    PASS - test vector matches Gohr's published value.")
    else:
        print("    FAIL - test vector mismatch. DO NOT PROCEED.")
        all_ok = False

    print("\n[2/3] Candidate-1 representation transform validation...")
    rng = np.random.default_rng(999)
    X, Y = speck.make_train_data(2000, BASELINE.rounds, diff=BASELINE.differential)
    permutation = generate_candidate1_permutation(rng)
    results = run_full_validation(X, Y, permutation)
    for key, value in results.items():
        if key == "bijection_problems" and value:
            print(f"    {key}: {value}")
    if results["all_passed"]:
        print(f"    PASS - all checks passed. permutation_hash={permutation.hash}")
    else:
        print(f"    FAIL - {results}. DO NOT PROCEED.")
        all_ok = False

    print("\n[3/3] Software environment comparison (actual vs. documented baseline)...")
    actual = software_provenance()
    comparison = compare_software_environment(actual)
    for key, info in comparison.items():
        status = "MATCH" if info["match"] else "MISMATCH (recorded, non-blocking)"
        print(f"    {key}: documented={info['documented']!r} actual={info['actual']!r} [{status}]")
    print("    NOTE: a software-version mismatch does not block validation, but it")
    print("    MUST be recorded in every certificate's provenance (it already is,")
    print("    via framework.provenance.software_provenance()).")

    print("\n" + "=" * 70)
    if all_ok:
        print("VALIDATION GATE: PASSED")
        print("=" * 70)
        return 0
    else:
        print("VALIDATION GATE: FAILED - production runs must not proceed.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
