# AutoND CE2 r3: validation record (2026-10-31)

- **Design:** `AutoND-CE2-frozen-design-2026-10-r3`, SHA-256
  `7e69ce6ccb0425dd496820e91dc990a15c23a239800b0a9382a18b621667ce0c`.
- **Preregistration:** `preregistration/preregistration.json`, SHA-256
  `3a053ea171474e7077a754ddd542d55750545e73bcf031a86f7dcf30859c6fca`; approval `PENDING_REVIEW`.
- **Supersedes:** r2 `ef3d764a8350e01a68fdcacf2fb38a15fd5622bfc8665f20e0e71df3f2ba4ba5`. r2 was never
  approved and never run in production.

Nothing here is confirmatory. No production run, no smoke run and no model evaluation on new data was
made under r3.

## 1. Newly executed under r3

| Check | Result | Where |
|---|---|---|
| r2 archive: 9 files copied to `history/r2/`, `SHA256SUMS` verified; the archived `design.py.r2` reproduces design hash `ef3d764a...`; the archived preregistration (`e273abe6...`) has that hash and an unapproved block; re-rendering r2 reproduces its bytes | pass | `history/r2/SHA256SUMS`; `tests/test_r3_specification.py` |
| r1 archive `SHA256SUMS`: 17/17 | pass | `history/r1/` |
| sign-test implementation checked against p_+: positives and negatives counted, exact zeros (including -0.0) removed and counted, p = min(1, 2 x smaller tail); one-sided control p = upper tail on n_effective | matches; **no analysis code changed** | `analysis.py`, `verify.py` |
| `--power` (43 s, AutoND environment) | `validation/r3/power.json`, SHA-256 `a23355fd...` | this directory |
| r3 joint-power grid, permutation-dependence rows and exact primary tables vs `validation/r2/power.json` | identical (seeded) | `test_r3_power_reproduces_r2_joint_tables` |
| zero-tie sensitivity: exact power by n_effective 20..5; joint power with 1-3 exact-zero primary runs | new results | `POWER_ANALYSIS.md` section 5 |
| pipeline-faithful check with zero counts (fresh targets, new sample) | 0 exact zeros of rho, rho_perm and D_perm in 3 x 400 replications | `POWER_ANALYSIS.md` section 4 |
| preregistration regenerated with `render_preregistration()` and `json.dumps(..., indent=1, sort_keys=True) + "\n"`; the design in the file equals `DESIGN`; stored hash = recomputed hash; `pending_items()` = `[]`; approval pending | pass | `test_markers_frozen_and_preregistration_matches_design` |
| test suites (section 4) | all pass | |

## 2. Reused from r2 (not re-run under r3)

These checks do not depend on the hypothesis wording and were not repeated. They were made on r2 code
paths that r3 leaves unchanged (`theory.py`, `adapter.py`, `runner.py`, `analysis.py`, model binding).

| Check | r2 result | File (unchanged) |
|---|---|---|
| `--check-model` | strict load, 68 arrays bit-identical, architecture as designed | `validation/r2/check_model.json` |
| `--validate-math` | gates C and D exhaustive at n = 4/6/8, 0 mismatches; E, F, encoding pass | `validation/r2/validate_math.json` |
| exploratory smoke run, 3 x 4,096 | INCONCLUSIVE; verifier 77/77 (78/78 with word 8), EXACT | `validation/r2/exploratory_smoke_summary.json` |
| pipeline-faithful check, r2 sample | see `POWER_ANALYSIS.md` section 4 | `validation/r2/power.json` |

The r2 smoke verification predates the two verifier checks added in r3
(`primary_zero_ties_and_n_effective`, `control_*_zero_ties_and_n_effective`). Those are exercised by the
test suite's own small smoke pipeline (`tests/test_pipeline.py`), not by a re-run of the r2 smoke.

## 3. Not affected by this correction

Data generation, target construction, gates A-F, score definition, model binding, behavioural binding,
evidence journal and quarantine, decision rule, Holm, serial gatekeeping, exclusion rule, run and pair
counts, alpha. r3 changes only text fields, the two status markers and the power-evidence binding in
`DESIGN`; `test_r2_to_r3_changes_are_exactly_the_declared_ones` enforces that no numeric or boolean
design value outside `revision` changed.

## 4. Code changed under r3 (with tests)

| File | Change |
|---|---|
| `design.py` | r3 design, `R2_DESIGN_SHA256`, `POWER_EVIDENCE_*`, `FROZEN`, `P_PLUS`; functions unchanged |
| `power.py` | additive: `zero_tie_sensitivity`, `n_zero_primary` (default path unchanged, tested), zero counts in `pipeline_faithful_check` (separate RNG stream), `clopper_pearson` overflow fix for about 1,030 < n <= 2,000 |
| `verify.py` | two checks: the certificate's `n_effective` and `n_zero_ties_removed`, primary and controls |
| `tests/test_r3_specification.py` | new, 45 tests |
| `tests/test_theory_statistics.py` | r1 comparison now against the archived r2 design; refusal message `PENDING_REVIEW` |
| `tests/test_pipeline.py` | tamper case `certificate_zero_ties` |

`analysis.py`, `runner.py`, `adapter.py`, `theory.py` and the CLI are unchanged.

Commands (repository root; `PYTHONDONTWRITEBYTECODE=1`):

```
AUTOND=~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python
$AUTOND -B -m audit.cryptography.kaggle.autond_ce2_kaggle --power --report audit/cryptography/experiments/autond/ce2/validation/r3/power.json
$AUTOND -B -m pytest -q -p no:cacheprovider audit/cryptography/experiments/autond/ce2/tests audit/cryptography/core/tests audit/cryptography/ce234/tests_authorised
$AUTOND -B -m pytest -q -p no:cacheprovider audit/cryptography/tests/test_autond_ce1.py
~/.venvs/gohr-tf221-validation-20261031/bin/python -B -m pytest -q -p no:cacheprovider audit/cryptography/experiments/autond/ce2/tests/test_r3_specification.py audit/cryptography/experiments/autond/ce2/tests/test_theory_statistics.py
~/.venvs/gohr-tf221-validation-20261031/bin/python -B -m pyflakes <changed files>
```

| Suite (environment) | Passed | Failed | Skipped |
|---|---|---|---|
| CE2: 126 (`test_pipeline` 43, `test_r3_specification` 45, `test_theory_statistics` 38) + core 67 + ce234 authorised 16 (AutoND) | 209 | 0 | 0 |
| AutoND CE1 regression, read-only (AutoND) | 97 | 0 | 0 |
| CE2 tests that need no TensorFlow (Gohr validation env, NumPy 2.4.6) | 83 | 0 | 0 |
| pyflakes on changed files | clean | | |

## 5. Still pending

- Your review and personal approval of the r3 preregistration (`docs/RUNBOOK.md` section 3).
- The production preflight in the pinned Kaggle environment (`--preflight`).
- The 10^6-sample behavioural binding, performed by `--run-all` initialisation.
- The probability of exact zero coefficients or differences for the real published model and twin:
  unknown. Production reports `n_zero_ties_removed` and `n_effective` for every test.
- The environment deviations of the local checks (pandas 2.2.3, SciPy 1.17.1, pytest 9.1.1, CPU): production
  enforces the pins.
