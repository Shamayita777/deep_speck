# Repair and hardening of the cryptography audit framework (2026-10-31)

Repository `deep_speck`, branch `audit-framework`, HEAD `fb539bc83d605cf1200fb2eed3640bc2dcbed9d2`
before and after. Nothing was committed, staged or pushed. No production experiment was run, no
approval was created, and no certificate was issued.

## A. Summary of changes

**Rule followed.** A file is frozen if its hash is bound by any of:
- `PACKAGE_MANIFEST.sha256` (96/96 verified);
- `ce234/CE234_MANIFEST.sha256`;
- the AutoND CE1 package `SHA256SUMS`;
- the CE2-CE4 certificates;
- the A3 adversarial freeze, which re-hashes 11 cryptography modules at run time and refuses to execute
  on any change.

Every repair to frozen behaviour is therefore additive. Files listed in `core/registry.py:FROZEN` are
byte-identical, as is all Gohr code, all Gohr CE1 code and evidence, and all AutoND CE1 files.

**Whole-repository audit** against the pre-change hash baseline of 1,874 files (venv and .git excluded):
- **Unchanged:** 1,856 files, byte-identical.
- **Changed:** 2.
  - `kaggle/autond_ce2_kaggle.py`: untracked, written in the previous session; rewired to the relocated
    CE2 modules, same CLI.
  - `ce234/README.md`: tracked; stale run counts (10 and 5, now 20 and 10) and the production route
    corrected; a status section added. Covered by `CE234_MANIFEST.v2.sha256`.
- **Removed:** 16. These were the r1 AutoND CE2 files at their old scattered paths (untracked, never
  approved, never run in production). All are kept byte-identical under
  `experiments/autond/ce2/history/r1/`, verified by `SHA256SUMS` before removal.
- **New:** listed in section G.

**What was built.**
- **Repaired:** AutoND CE2 evidence lifecycle, independent verification, hypothesis semantics,
  p-value numerics, power analysis, upstream import isolation; ce234 governance (additive), tamper-log
  audit, isolated frozen-suite runner.
- **Refactored or relocated:** AutoND CE2 moved into `experiments/autond/ce2/`. Target-level upstream
  and model binding moved to `targets/autond/`.
- **Retained unchanged:** the legacy framework (`engine.py`, `results.py`, `test/`, `adapters/`), the
  Gohr drivers, ce234 sources and all evidence.
- **Deprecated (documented, not edited):** `test/ce2/theory_consistency.py`, superseded by
  `core/theory_consistency.py`. The legacy CE2-CE4 drivers `experiments/ce{2,3,4}/gohr_*.py`, which are
  superseded by ce234. `ce234/production.py --production`, which is the historical route.

## B. Defect matrix

| # | Defect | Original behaviour | Correction | Tests | Result | Remaining risk |
|---|---|---|---|---|---|---|
| 3.1 | CE2 data-commit crash window | `data.npz` saved, then the record advanced; a crash left NOT_STARTED and an un-overwritable file | `core/evidence.py` commit protocol (intent, staged write, hash, validate, atomic link, transition); recovery quarantines | 7 fault points in core and 7 through the real runner, plus scoring interruption and committed-damage cases | pass | directory fsync is best effort on filesystems without it |
| 3.2 | record/journal inconsistency | record written first, then the journal; only the hash chain was checked | journal authoritative; record materialised and reconciled; verifier replays its own state machine | 13 mutation and reconciliation tests; 12 verifier tamper cases with named failure checks | pass | none known |
| 3.3 | rank-blind score verification | max abs difference within 1e-4, then rho from stored scores | inference from recomputed scores; ranks, ties and signs reported; fails on any decision-relevant difference | forged-scores test (consistent hashes, caught only by recomputation); small-rank-swap test | pass | recomputation uses the same DBitNet builder and TF (reproducibility, not an independent network) |
| 3.3b | recorded hashes not re-derived | `model_input_sha256` and weight hashes trusted | re-derived with own code | per-run checks | pass | |
| 3.4 | preflight scope overstated | preflight hash only | gates C/D re-executed by the verifier (n = 4, 6, and 8 optionally); scope table in `docs/VERIFICATION_REPORT.md` | gate tests; tamper case `preflight` | pass | the 16-bit 2^32 enumeration is recorded, not re-run |
| 4.3 | power presented as a guarantee | point estimates, one model, no Monte Carlo error | scenario-based sensitivity analysis: Monte Carlo SE and 95% intervals, wider dependence grid, pipeline-faithful check | power tests incl. null size and Clopper-Pearson vs SciPy | pass | twin share and dependence unknown; underpowered if the twin carries 75% of the effect |
| 4.4 | behavioural validation at 20,000 only | n/a | protocol extended (balance, finiteness, twin diagnostic); 10^6 not run | pipeline tests | n/a | **10^6 behavioural binding not run**; the r2 smoke lower bound is 0.5021 |
| 4.5 | two-sided hypothesis encoded as positive | legacy `TheoryConsistencyTest`, HIGHER_IS_BETTER | `core.hypothesis.HypothesisDirection` (TWO_SIDED etc.); CE2 records TWO_SIDED end to end | hypothesis tests; verifier check | pass | legacy class unchanged (manifest-bound); no production uses it |
| 4.6 | undefined becomes 0.0; p underflows to 0.0 | `test_score = 0.0`; `within_run_p = 0.0` | AssociationStatus; PValue states; exact p underflow reported with its log10 | numerics tests | pass | |
| 4.7 | single-twin generalisation | n/a | limitation added to the certificate and docs | n/a | n/a | design still uses one twin |
| 5.1 | ce234 production not gated | `--production` runs with no approval | `ce234/authorised_production.py` binds the plan hash, target, all decisions, every source hash, checkpoint and freeze | 16 tests (missing, incomplete, wrong hash, experiment or target, invalidated, unresolved, stale value, source changed, pending) | pass | the legacy CLI is still ungated; closing it needs a versioned edit of a frozen file and an A3 re-freeze (decision for the owner) |
| 5.2 | stale run counts / key names | README said 10 and 5; key `CE2.n_replicates` | README corrected; `DECISION_MAP` maps the keys and fails closed on unmapped keys | authorised tests | pass | the keys cannot be renamed (hashed plan) |
| 5.3 | tamper-log manifest mismatch | v1 `4af542c5` vs file `9f5a6823` | cause established: the tests append to the committed log; a fresh isolated pass reproduces `4af542c5` exactly. v1 kept, v2 issued, isolated runner, full-history audit | tamper-audit tests | pass | the frozen test still appends if run directly |
| 5.3b | matrix masks regressions | first record per id wins | `ce234/tamper_audit.py` | masked-regression test | pass | |
| 6.1 | competing Gohr production routes | the README and legacy docstrings call the legacy drivers authoritative | `ARCHITECTURE.md`, `core/registry.py` (tested); the legacy CLI cannot produce evidence | registry tests | pass | stale claims remain in manifest-bound README and docstrings |
| 6.2 | AutoND CE2 spread over 7 directories | n/a | co-located in `experiments/autond/ce2/`; the upstream stays as the single checkout | all CE2 tests | pass | |
| 6.3 | upstream import pollution | root left on `sys.path`, modules left in `sys.modules` | `targets/autond/upstream.py` isolated loader; stable identity; foreign-module test | isolation tests | pass | the CE1 adapter keeps its own loader (frozen) |
| 6.4 | unused components | n/a | usage verified by test; documented as retained or extension interface | registry tests | pass | |
| 6.5 | over-broad adapter interface | 15 scope-raising stubs | `core/capabilities.py` protocols; the CE2 adapter implements only its capability | capability tests | pass | the legacy base class is unchanged |
| X | packaging contamination | caches and macOS metadata in archives | scoped `.gitignore`; `tools/package_artifact.py` excludes them and refuses iCloud placeholders | packaging tests | pass | |

## C. Architecture

See `ARCHITECTURE.md` for the tree, ownership, routes, frozen files, evidence policy and environments.

## D. Tests

The two stacks need separate environments (Keras 3 for Gohr, Keras 2 for AutoND). See section F of
`ARCHITECTURE.md`.

**Static checks** (counted separately from the tests): 37 new or changed modules parse; every non-test
module imports in a fresh process; pyflakes 3.2.0 is clean after one fix (an unused import). No type
checker is configured in the repository, and none was run.

**AutoND stack** (`~/.venvs/autond-tf215-audit-20261015T050853Z`, run in the repository; tests write
only to temporary directories):

```
python -B -m pytest -p no:cacheprovider -q audit/cryptography/core/tests \
  audit/cryptography/experiments/autond/ce2/tests audit/cryptography/ce234/tests_authorised \
  audit/cryptography/tests/test_autond_ce1.py
```

| Suite | Collected | Passed | Failed | Skipped |
|---|---|---|---|---|
| core/tests | 67 | 67 | 0 | 0 |
| experiments/autond/ce2/tests | 80 | 80 | 0 | 0 |
| ce234/tests_authorised | 16 | 16 | 0 | 0 |
| tests/test_autond_ce1.py (CE1 regression, read-only) | 97 | 97 | 0 | 0 |
| **total** | **260** | **260** | **0** | **0** |

**Gohr stack** (`~/.venvs/gohr-tf221-validation-20261031`, in a cache-stripped copy-on-write clone so no
test can write into the repository):

| Suite | Collected | Passed | Failed | Skipped |
|---|---|---|---|---|
| tests/ (Gohr CE1-CE4 and framework; not autond) | 282 | 272 | 10 | 0 |
| core/tests and ce234/tests_authorised (cross-stack) | 83 | 81 | 0 | 2 |
| ce234/tests (frozen), via `run_frozen_suite_isolated.py` | 65 | 64 | 0 | 1 |

**Failures and skips.**
- The 10 failures are all in `tests/test_ce1_run_dir.py` and are identical to the pre-change baseline.
  The test assumes no `run_manifest.json` and no `production_20260930` exist in the tree, which is false
  now that production evidence is present (section H8).
- Clone skips: packaging needs a git tree; the adversarial package is not copied.
- The ce234 skip is the CE3 smoke run, which stops early by design.

**Pre-change baseline, same environments.**
- AutoND: 121 passed, 18 skipped. The skips were from the upstream walk-up in a clone.
- Gohr tests/: 272 passed, 10 failed.
- ce234: 64 passed, 1 skipped.

No previously passing test fails.

**Not run.**
- The production preflight in the pinned Kaggle environment.
- Any production run.
- `ce234/verify.py` on production directories (raw files evicted).
- The exact FREEZE_MANIFEST environment (Python 3.12.3, Keras 3.15.1).

**Validation evidence regenerated today** (`experiments/autond/ce2/validation/r2/`): model binding, all
gates at production sizes (re-executed, 291 s), power with the pipeline-faithful check, and an
exploratory smoke run verified with score recomputation and the n = 8 re-enumeration.

## E. Scientific validity

| Category | Established | Not established |
|---|---|---|
| Mathematical correctness | three xdp+ derivations and literal enumeration agree exhaustively for n <= 8; 8 sixteen-bit literal checks agree (re-executed today) | exhaustive 16-bit enumeration (the closed-form proof is relied on) |
| Implementation integrity | trajectories reproduce upstream encryption; strict checkpoint binding; weight hashes; isolated imports | |
| Dataset integrity | content hashes, commit protocol, journal replay, quarantine | AutoND CE2 data are not seed-replayable (`os.urandom`); retention is the guarantee |
| Behavioural validity | AutoND CE2: smoke n = 20,000, lower bound 0.5021 | the planned 10^6 binding |
| Statistical validity | exact tests, Holm, size <= alpha (simulated), Gaussian model checked under the null | power for the unknown twin share and dependence; variance under the real alternative at N = 10^6 |
| Experimental validity | controls on one frozen dataset; limitations recorded | one twin only; association, not mechanism |
| Independent verification | the AutoND CE2 verifier shares no producer code except the upstream DBitNet/TF inference | ce234 production directories not re-verified locally (raw files are iCloud placeholders) |
| Evidence integrity | manifests reconciled; the tamper-log history is explained | 47 Gohr evidence files evicted; `evidence_current/` untracked |
| Governance | fail-closed gates for AutoND CE2 (preregistration) and new ce234 production | no approval records exist; Gohr CE2-CE4 production ran without one |

## F. Production readiness

| Experiment | Status |
|---|---|
| **AutoND CE2** | **READY FOR APPROVAL** (design r2 pending review). Not approved, not run. Open before production: the approver's choice of run count given the power scenarios; the 10^6 behavioural binding; a preflight in the pinned Kaggle environment. |
| **Gohr CE234** | **PRODUCTION COMPLETE** for CE2, CE3 and CE4 (2026-10-03/04, SUPPORTED as recorded by the certificates), produced **without a recorded approval**, and **not independently re-verified here** because 24 of 30 raw files are iCloud placeholders. New production: NOT READY until an approval record exists. |

No approval for confirmatory production was issued by this repair.

## G. Git

- HEAD `fb539bc` (unchanged); no commit, no stage, no push.
- New tracked-file modification: `audit/cryptography/ce234/README.md`. All pre-existing modifications and
  deletions are preserved byte-identically.
- New untracked: `core/`, `targets/`, `tools/`, `experiments/autond/`, `ARCHITECTURE.md`,
  `REPAIR_REPORT_2026-10-31.md`, `.gitignore` (scoped), and in `ce234/`: `authorised_production.py`,
  `tamper_audit.py`, `run_frozen_suite_isolated.py`, `tests_authorised/`, `CE234_MANIFEST.v2.sha256`,
  `MANIFEST_RECONCILIATION_2026-10-31.md`.
- Removed untracked r1 paths: listed in section A.

## H. Remaining blockers, by severity

1. **Evidence preservation (critical).**
   - `evidence_current/` (Gohr CE1-CE4 production, including the multi-day CE1) is untracked, and 47 of
     its files are iCloud placeholders.
   - Materialise them, then archive with checksums off the sync service.
2. **Governance (high).**
   - No approval records exist for any production.
   - Gohr CE2-CE4 ran without one.
   - `ce234/production.py --production` is still ungated (frozen).
3. **Verification of completed production (high).** Re-run `ce234/verify.py` on the CE2-CE4 production
   directories once their raw files are local.
4. **A3 adversarial binding is broken (high, pre-existing).** Commit `50762ef` changed
   `adversarial/battery.py` and `sealed_sets.py` after the A3 freeze (`dbf02c9`), so A3 refuses to
   execute. All 11 cryptography modules it binds still match.
5. **AutoND CE2 (medium).**
   - The approver must decide the run count in light of the power table.
   - The 10^6 behavioural binding has not been run.
   - Production preflight is still needed in the pinned environment.
6. **Environments (medium).**
   - The repository `venv/` is evicted.
   - Neither production stack is reproduced exactly locally.
   - pandas, SciPy and pytest pins differ in the AutoND venv.
   - Gohr validation used a rebuilt venv.
7. **Stale but frozen text (low).** The root README and the legacy driver docstrings still claim the
   legacy drivers are authoritative. `ARCHITECTURE.md` supersedes them.
8. **Stale tests (low, pre-existing).** `tests/test_ce1_run_dir.py` (10 cases) assumes no
   `run_manifest.json` exists in the tree, which stopped being true once production evidence existed.
   `tests/test_autond_ce1.py` finds the upstream by walking to a sibling `AutoND-upstream/`, so it skips
   18 tests outside this machine's layout.
