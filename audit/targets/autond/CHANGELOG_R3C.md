# Changelog — r3b → r3c

Two statistical defects corrected before the preregistration freeze.
**The plan is still unfrozen and no confirmatory evaluation has been run.**
The approved margin Δ = 0.10 × (p_published − 0.5) is unchanged, and a
test asserts it.

The plan hash changed because the `PLAN` object changed:

```
r3b  fd8de5dd0e2779090e646b4d32b06a4b121e0ef2505772529655998f22e7f48d
r3c  b93fa50bb0914b17dc3423480466f18c6340225486c2dcbcb29c911d87207d40
```

Freeze against the **r3c** hash. Any sidecar written against the r3b hash
would be rejected by `is_frozen()`.

## Files

| file | change | why |
|---|---|---|
| `audit/common/equivalence.py` | **NEW** (generic) | TOST, outside-margin test, Holm, exact power, Student-t. Not target-specific, and the Gohr case study needs the same machinery, so a private copy in the adapter would be duplicated generic logic. **Pure addition — no pre-existing framework file is modified or shipped.** |
| `audit/targets/autond/preregistrations/autond_ii_v2.py` | modified | `decide()` no longer uses the point null for `DIFFERENT`; `power_at()` delegates to the exact calculation; `margin_sensitivity()` added; decision-rule prose, planning-SD provenance and SIMON limitation rewritten |
| `audit/targets/autond/evaluation_results.py` | modified | Student-t moved to the generic module and **re-exported** under the original names, so callers and tests are unaffected |
| `audit/targets/autond/run_autond_implementation_audit.py` | modified | `CODE_PATHS` binds `audit/common/equivalence.py` |
| `audit/targets/autond/tests/test_r3c_statistics.py` | **NEW** | 58 tests for both corrections |
| `audit/targets/autond/tests/test_corrections.py` | modified | two power tests referenced the removed `approx_power` |
| `audit/targets/autond/tests/test_packaging.py` | modified | contract widened by **exact path** for the one new generic module |
| `audit/targets/autond/STATISTICAL_NOTE.md` | **NEW** | assumptions, computed power, multiplicity, decision table |
| `audit/targets/autond/CHANGELOG_R3C.md` | **NEW** | this file |
| `audit/targets/autond/run_evidence/*` | added | `gohr_regression_v3c.log`, `r3c_statistics.log`, `r3c_suite_tf215.log`, `r3c_suite_portable.log`, `r3c_final_verify.log` |
| `audit/targets/autond/SHA256SUMS.txt` | regenerated | |

Nothing else changed. Verified by diffing the whole tree against a
pristine framework copy: the only difference outside
`audit/targets/autond/` is `Only in audit/common: equivalence.py`.

**Superseded logs retained, not replaced.** r3b's `gohr_regression_v3.log`,
`r3_suite_tf215.log` and `r3_suite_portable.log` still ship alongside the
r3c logs. An earlier build of this archive dropped them; that would have
discarded a passing record (Gohr 426 OK) to leave a tidier evidence
directory, which the standing constraints forbid. Both generations are
present so the r3b → r3c transition is auditable from the package alone.

## Rule change 1 — `DIFFERENT` requires evidence of an outside-margin discrepancy

**Was:** `DIFFERENT` when the Holm-adjusted two-sided test of `d = 0`
rejected and TOST failed.

**Now:** `DIFFERENT` when the Holm-adjusted `p_outside < α`, where
`p_outside = min(p_above, p_below)` tests `H₀: d ≤ +Δ` and `H₀: d ≥ −Δ`.
The point null is computed and reported as a diagnostic and takes no part
in the decision.

**Why it was wrong.** A discrepancy can be significantly non-zero while
lying inside the equivalence interval. Observed mean 0.515600 against
reference 0.5144 gives `d = +0.00120`, inside Δ = 0.00144, with
`p_tost = 0.2343` and `p_diff = 0.0161` — r3b returned `DIFFERENT`, r3c
returns `INDETERMINATE`. Failing to establish equivalence is not evidence
of a meaningful discrepancy.

**Error control.** For Δ > 0 the two directional rejection regions are
disjoint, so `min()` controls type-I error at the nominal level without
adjustment between directions. Holm is retained across the registered
family of 3 and applied independently to the equivalence and
outside-margin families, each with its own family-wise guarantee at
α = 0.05. No procedure was substituted for Holm, and no error-control
guarantees are mixed.

**`INDETERMINATE` unchanged in meaning:** neither null rejected.

## Rule change 2 — power with the variance estimated

**Was:** `2Φ(Δ/SE − t_crit) − 1` with SE treated as a known constant.

**Now:** `P(reject | s)` integrated over the χ distribution of `s`
(Simpson's rule), so the variance is estimated from the sample as the
analysis will estimate it. Cross-checked against a seeded Monte Carlo
simulation of the actual rule with a reported 99% interval.

**Why it was wrong.** It ignored the sampling variability of `s`
(4 df at n = 5), and went negative below the critical value, which was
clamped to zero and reported as a computed power of zero.

**Recomputed, at α/3:**

| fraction | r3b (clamped) | r3c exact |
|---|---|---|
| 0.02 | 0.000 | 0.0023 |
| 0.05 | 0.000 | 0.1291 |
| **0.10** | **0.895** | **0.7965** |
| 0.15 | 1.000 | 0.9931 |
| 0.20 | 1.000 | 1.0000 |

**The registered margin's power fell from a reported 0.895 to 0.7965** at
the conservative Holm step — marginally under the conventional 0.80. At
nominal α it is 0.9732, and the true value lies between the two because
the Holm threshold depends on the observed ordering. This is reported,
not smoothed over, and **the margin was not widened to recover power.**

α/3 is labelled a planning approximation throughout, distinct from the
registered step-down procedure.

## Documentation synchronised

- `DECISION_RULE["outside_margin_test"]` replaces `["difference_test"]`.
- `DECISION_RULE["point_null_is_diagnostic_only"]` records the
  correction, the counterexample and the regression test's name.
- `DECISION_RULE["mutual_exclusivity"]` states why both outcomes cannot
  co-occur.
- `PLANNING_SD_PROVENANCE` labels the SD as assumed, from a *different*
  artifact (the unpolished checkpoint), with its own 4 df uncertainty.
- `POWER["superseded_method"]` retains the old method rather than
  deleting it.
- `POWER["power_does_not_justify_a_margin"]` states that power informs
  the design but does not make a tolerance acceptable.
- SIMON limitation narrowed: the equivalence regions **overlap
  substantially near the published values**, so the experiment is **not
  expected** to resolve the 3e-4 discrepancy. The earlier absolute claim
  ("CANNOT identify") is gone. Both references and their provenance are
  preserved.

`TestWrittenRuleMatchesTheCode` asserts the prose and the executable rule
agree.

## Predictions

Unchanged. `H-AUTOND-SPECK-REPRO` predicts `SUPPORTED`;
`H-AUTOND-SIMON-REPRO` predicts `INCONCLUSIVE`. The SIMON rationale now
rests on the overlap argument rather than on an impossibility claim.

## Preserved, verified

- Evidence untouched: R1 manifest/result/logs, checkpoints, training
  bundle, census, load-verification. Hashes re-verified; see
  `run_evidence/`.
- The R1 findings stand: two returned differences against the README's
  one; `highest_non_random_round = 5` against 7 and 6.
- `.DS_Store` and `__pycache__` handling unchanged; nothing deleted.
- Freeze-gate behaviour unchanged: plan-file bytes bound, sidecar must be
  committed and match, repo-relative git paths, state-agnostic tests.
- Source-hash, checkpoint-hash, interface-verification and
  output-directory gates unchanged; none weakened.
- No training, search or evaluation was run.

## Freeze procedure

Unchanged from r3b except the hash. Install r3c, run the suite, then:

```bash
cd ~/Documents/deep_speck
PLAN=audit/targets/autond/preregistrations/autond_ii_v2.py
SIDE=audit/targets/autond/preregistrations/autond_ii_v2.frozen.json

git add "$PLAN"
git commit -m "Freeze autond-ii-v2" -- "$PLAN"
PLAN_COMMIT=$(git rev-parse HEAD)
PLAN_SHA=$(shasum -a 256 "$PLAN" | cut -d' ' -f1)
PLAN_HASH=$(python -c "from audit.targets.autond.preregistrations import autond_ii_v2 as p; print(p.plan_hash())")

cat > "$SIDE" <<EOF
{
  "preregistration": "autond-ii-v2",
  "plan_hash": "$PLAN_HASH",
  "frozen_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "plan_file_commit": "$PLAN_COMMIT",
  "plan_file_path": "$PLAN",
  "plan_file_sha256": "$PLAN_SHA",
  "approved_by": "Sham"
}
EOF

git add "$SIDE"
git commit -m "Record freeze record for autond-ii-v2"

python -m audit.targets.autond.preregistrations.autond_ii_v2 \
  --autond-root "$AUTOND_ROOT" --repo-root .
```

`$PLAN_HASH` must read `b93fa50bb091…`. Do not edit the plan file after
the first commit — the sidecar binds its bytes.
