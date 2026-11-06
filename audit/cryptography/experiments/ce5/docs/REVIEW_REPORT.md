# G-CE5 technical record (r3, 2026-11-05; updated 2026-11-06)

**Current state (2026-11-06).**
- Design `409e7dcb53c5eba7a623a584ec302f2aa3a26e84b3efcf9e6c021e5629fde5c5`.
- Calibration done and verified (R = 20).
- Current draft `preregistration/preregistration_409e7dcb_draft_20261106T050444Z.json` is not approved.
- No commit and no timestamp proof exist for CE5. The gate and production have not run.

Sections 1–6 record the state on 2026-11-05 before calibration; §7 and §8 record the later work.

## 1. Scope and boundary

Sections 1–6 cover the implementation, the tests, the non-confirmatory stages and the documentation of G-CE5 up to, but not including, preregistration.

Not performed at that point (calibration and preregistration drafts were done later; see §8):
- writing or rendering `preregistration.json`;
- approval;
- timestamp proofs;
- commits;
- the gate;
- calibration;
- production.

## 2. Corrections made in r3

| area | defect | correction | test |
|---|---|---|---|
| table key (Algorithm 1 L10) | r2 keyed the table on (δl, δr) | (δl, δv), δv = l ⊕ r ⊕ l′ ⊕ r′ | `test_delta_v_is…`, `test_table_is_indexed_by_dl_dv…`; the table check rejects a full-size (δl, δr) table |
| candidate loop (Algorithm 2 L4) | r2 averaged over 2^16 subkeys via 2^13 classes | literal loop 0 … 4095, divided by 4096 | `test_candidate_loop…`; a test with a 15-bit mask that fails under 2^13 or 2^16 |
| threshold (Algorithm 2 L11) | r2 used ≥ | strict > 2^−26 | boundary tests in both implementations |
| gate | the r2 interval showed separation only | reproduction of accuracy, TPR and TNR within τ and τ_r, required; separation reported | gate component tests; verifier |
| outcome rule | r2 accepted intervals crossing the margin as directional | directional verdicts need the whole interval beyond ±m | boundary table, exact-equality and qualifier tests |
| calibration | 5 replicates, plain SD, unblinded in r1 | 10 replicates, robust SD with an 80% bound, blinded | blinding test |
| stage outputs | stage records could be overwritten | every stage refuses existing paths; the verifier writes versioned files | overwrite tests |
| calibration size | pair count came from the command line | registered count enforced | refusal test |
| table loading | dtype and sum unchecked | both checked | refusal test |
| variability | tables and evaluation pairs not kept | tables, scores and ciphertexts stored with hashes; independent verifier | variability verifier test |
| verifier coverage | did not recompute intervals, secondary outcomes, qualifier, individual correlations or gate CP intervals; no label check | all recomputed; class-label sanity check added | 10 parametrised tampering cases, plus timestamp and approval cases |
| design text | stale r2 statements (source fields, "2^16" in verification, SD rule, anchoring) | corrected | none (text) |
| runner docstring | garbled | rewritten | none |

The r2 implementation, its tables, benchmark and 58 passing tests are archived in `historical/r2_superseded_20261105/`. They are validation of r2 only.

## 3. Tests and regression (final files, 2026-11-05)

**Command.** All commands are run from the repository root, using `~/.venvs/gohr-tf221-validation-20261031/bin/python` unless another environment is named.

| suite | command | result |
|---|---|---|
| CE5 compile and import | `py_compile` on every CE5 module and test; import of all six modules | pass |
| CE5, all tests (100 collected: crypto 30, statistics 29, pipeline 41) | `-m pytest -q -p no:cacheprovider -rs -W default audit/cryptography/experiments/ce5/tests` | **100 passed**, 0 failed, 0 skipped, no warnings, exit 0 (107 s) |
| frozen CE2–CE4, isolated | `-m audit.cryptography.ce234.run_frozen_suite_isolated --python <same>` | 64 passed, 1 skipped, exit 0; the suite appended 22 tamper-log lines to its temporary copy only |
| `audit/cryptography/core/tests` | pytest in a copy-on-write clone | 66 passed, 1 skipped (packaging needs a git working tree) |
| `audit/cryptography/ce234/tests_authorised` | same | 16 passed |
| AutoND CE2 tests (`~/.venvs/autond-ce2-prod`) | same | 138 passed, 1,164 warnings |
| AutoND CE3 tests | same | 122 passed |
| AutoND CE4 tests | same | 287 passed |
| `audit/cryptography/tests` | **not run to completion** | its fixtures copy the 7.2 GB `audit/cryptography` tree per test, which exhausted the 42 GB free disk; the run was stopped and its temporary directories deleted. CE5 does not import or modify anything the suite covers beyond read-only use of `gohr.speck`, `ce234.production` and `ce234.frozen_plan`, which the frozen suite covers |

**Protected files.**
- 580 files were hashed before work began: `ce234/`, `experiments/ce2–ce4`, `experiments/autond/`, `evidence_current/` and the checkpoint.
- All 580 were unchanged at the pause, after resumption, and at the end of the work.
- Tamper log: `9f5a68239b42463a…` throughout.

## 4. Non-confirmatory results

| stage | record | result |
|---|---|---|
| table build | `tables/tables_manifest.json` | aKR `09ff9388…`, LR `2ba703db…`; 10^7 uint32 counts each; table check PASS (p = 0.210, 0.995) |
| table re-verification, fresh seed | `validation/r3_final_20261105/table_validation.json` | PASS (p = 0.194, 0.641); wrong rounds and swapped difference p ≈ 0; (δl, δr) control rejected (χ² = 53,429, df 200); compaction bijective |
| benchmark | `validation/benchmark.json`, `validation/r3_final_20261105/benchmark.json` | 0.109 / 0.110 ms per real pair; 2^12 loop = 2^16 average bit for bit; scalar difference 0.0 |
| variability, run 1 | `validation/r3_final_20261105/` | accuracy 0.9284–0.9295, TPR 0.9072–0.9088, TNR 0.9491–0.9507; recomputes except re-scoring (no ciphertexts stored) |
| variability, run 2 | `validation/r3_final_20261105_variability_v2/` | accuracy 0.9284–0.9294, TPR 0.9060–0.9082, TNR 0.9495–0.9513; VERIFIED, with 80 scalar re-scorings |
| model binding | frozen `verify_reference_model` | SHA-256 match; 21 Conv1D layers, 10 residual merges; accuracy 0.93008 at 5 rounds (0.498 / 0.492 at 6 / 7); PASS |
| calibration | none at this point | not run on 2026-11-05 before `primary.margin` was set; done later (`validation/calibration_20261105T164019Z/`, R = 20; §8) |

## 5. Shared interpretation

- **Algorithm 2 line 9** is implemented as a lookup at the masked cell, an interpretation recorded in SOURCE_VERIFICATION.md §5.1. The scalar reference and the vectorised code share it.
- **Indirect support.** The variability runs place accuracy and TPR within about 0.1 point of Table 7 and TNR about 0.2 point below it, which supports this reading indirectly. They are non-confirmatory and are not the gate.

## 6. Open items

| # | item | where |
|---|---|---|
| 1 | margin m (proposal 0.10; sensitivity table in STATISTICS.md) | resolved: set to 0.10 (design `margin_decision`) |
| 2 | gate tolerance (registered formula; floors 0.005 and 0.010 proposed; observed 3 SD of 0.0012–0.0015 and 0.0024–0.0026) | resolved: 0.005 / 0.010 (design `tolerance_decision`) |
| 3 | replicate count (from blinded calibration after item 1) | resolved: R = 20 from the verified calibration (`replicate.n_replicates_record`) |
| 4 | stale draft preregistration file | resolved: versioned drafts in `preregistration/`; production takes an explicit `--prereg` path and SHA-256 |
| 5 | Table 7 evaluation-set size not stated in P1 | SOURCE_VERIFICATION.md §5 |
| 6 | Gohr 2019 not read directly; 0.911 and 0.929 are taken as quoted in P1 | SOURCE_VERIFICATION.md §6 |
| 7 | T verified only as T = ∏F (the frozen CE2 generator does not expose keys) | `verify.verify` |
| 8 | full OpenTimestamps verification against a block header needs a Bitcoin node | `design.production_authorisation` checks digest and attestation tag only |
| 9 | `audit/cryptography/tests` not run to completion (disk) | §3 |
| 10 | available memory varies (1.6–2.6 GB); calibration and production refuse below 2 GiB | `runner.require_memory` |

## 7. r3.1 corrections (2026-11-05, evening)

| area | correction | evidence |
|---|---|---|
| planning spread | `planning_sd`: σ_plan = max(s, 1.4826·MAD) · √(df/χ²_{0.20,df}). The claim is limited to P(σ ≤ σ_plan) ≥ 0.80 under normality, because σ_plan is never below the exact normal-theory bound | `analysis.py:227`; coverage simulation test |
| Monte Carlo selection | Bonferroni: one-sided Clopper–Pearson lower bound at 0.01/3 for each of the three requirements, giving a simultaneous ≥ 99% guarantee per R; nothing is claimed across R. 50,000 sets per (R, Δ) | `analysis.py:189, 249`; borderline and Bonferroni tests |
| common random numbers | implemented: one draw array per R, shared by all five conditions | `analysis.py:192`; sharing test and manual-recomputation test |
| table validation | fail-closed build; overall tablecheck decision with non-zero exit; registered relative record and hash; calibration and production require it; n_pairs checked against the registered 10^7 | `runner.py:154, 204, 209, 329` |
| crash recovery | manifest written before the gate; an orphan `gate_raw.npz` is adopted only if every score re-derives from its ciphertexts at the registered size; otherwise production refuses and keeps the file | `runner.py:276`; recovery, corrupt-orphan and crash tests |
| verifier | gate accuracy, TPR, TNR, n, exact per-class size, dtypes and finiteness; raw-array keys, shapes, dtypes and finiteness; T = 0 on infeasible trails; every stored primary and secondary summary; variability rows and metadata | `verify.py:143, 381`; 17 summary, 5 layout and 9 variability tampering tests |
| OpenTimestamps | `record_ots_verification` runs `ots -v verify` and records the command, exit code, output and the file and proof hashes. `ots_verification_ok` requires exit 0, a success line, no failure line, the file's digest in the output, and current hashes | `runner.py:298`, `design.py:355-375`; 8 record-rule tests, plus the real D2-N output rejected |

**Real `ots verify` run** (`validation/r3_1_ots_20261105/`), on the existing D2-N proof; the CE5 proof does not exist yet:
- The calendars returned Bitcoin attestations (blocks 974,635–974,654).
- Verification **failed (exit 1)**: no local Bitcoin node (`Could not connect to Bitcoin node`).
- File and proof bytes were unchanged.
- **A successful `ots verify` therefore requires a machine with a Bitcoin node.**

**Tests** (`validation/r3_1_tests_20261105/`; exit codes recorded):

| suite | result |
|---|---|
| compile | exit 0 |
| collection | 160 tests |
| `test_crypto.py` | 30 passed |
| `test_statistics.py` | 33 passed |
| `test_pipeline.py` | 97 passed |
| full CE5 | 160 passed, 0 failed, 0 skipped, no warnings |
| frozen CE2–CE4, isolated (`validation/r3_1_frozen_regression_20261105/`) | 64 passed, 1 skipped, exit 0; protected real files unchanged |
| `audit/cryptography/tests` (`validation/r3_1_broad_suite_20261105/`) | **11 failed**, 350 passed, 18 skipped, exit 1 |

**About the broad suite:**
- The run used an APFS copy-on-write version of `shutil.copytree`, installed only in the isolated clone (`clone_conftest.py`). The in-repo fixture `tests/test_ce1_run_dir.py:49-53` is hash-pinned in `PACKAGE_MANIFEST.sha256:91` and was not modified.
- Peak disk use was about 130 MB.
- A control run of one failing test with the original full copy (7.2 GB) failed identically. The failures depend on the state of the Gohr CE1 evidence, for example `production_20260930/blocks/block0 is not empty`, existing `run_manifest.json` files, and a Keras `TypeError` in `test_epoch_resume`. They are not caused by CE5 or by the copy strategy.

**iCloud placeholder loss (not caused by this work).**
- Between 19:49:27 and 19:52:55, 47 iCloud placeholder files disappeared from `audit/cryptography/evidence_current`:
  - 23 in `ce1/production_20260930`;
  - 14 in `ce2/production_20261003/raw`;
  - 10 in `ce4/production_20261004/raw`.
- The real files had never been downloaded to this Mac, and they are not tracked by git.
- No CE5 code or test references `evidence_current`, and every deletion in CE5 targets a test's own temporary directory.
- All 533 real protected files remain byte-identical. Two dataset placeholders also changed: one file was downloaded, and one placeholder disappeared for a file already deleted in the working tree.
- Recovery (iCloud on another device, icloud.com "Recently Deleted", or archived copies) remains to be decided.

## 8. Calibration, decisions and anchoring correction (2026-11-05 to 2026-11-06)

### Calibration and replicate count
- Calibration ran under design `b5c4ac4f` with 10 blinded replicates of the registered 10^6 pairs: `validation/calibration_20261105T164019Z/`.
- It is independently verified: σ_plan = 0.0016756.
- At R = 20, all three Bonferroni one-sided Clopper–Pearson lower bounds are ≥ 0.80.
- R = 20 is the smallest value on the registered grid, so n_replicates = 20.
- The failed first launch (`validation/calibration_20261105T162941Z`, exit 127 before any computation) is kept with its note.

### Design history

| design | change |
|---|---|
| `edb4bb88` | before margin and tolerance |
| `b5c4ac4f` | margin 0.10 and gate tolerance 0.005 / 0.010 set; calibration ran under this design |
| `f6f0ceb9` | n_replicates = 20 and its record |
| `654bc102` | gate.tolerance_rule wording; draft `preregistration_654bc102_draft_20261106T035130Z.json` (SHA-256 `41347a84…`) approved 2026-11-06T04:04:52Z |
| `409e7dcb` | anchoring wording: the text named `preregistration.json` literally, whereas `runner.py` and `verify.py` derive `<file>.ots` and `<file>.ots.verification.json` from the file passed with `--prereg`. The text now states that rule. One design_history entry added. `design.py` before the change: `historical/design_before_anchoring_fix_654bc102/` |

### Effect of the 409e7dcb correction

**Approval.**
- An approval binds one design hash, so the approval of the 654bc102 draft does not apply to design 409e7dcb.
- The approved 654bc102 file is unchanged. It was never timestamped and never used by production.
- The new draft `preregistration_409e7dcb_draft_20261106T050444Z.json` (SHA-256 `81991d2c4a8c3833442a0adc84bfc69bd88992ca51b49ddf4bb3efb26fc16a5e`) is unapproved. It registers the current source manifest (12 CE5 files, 3 frozen dependencies; production integrity check PASS).

**Calibration remains valid.**
- Against the calibration design, the only leaf differences are:
  - n_replicates and its record;
  - design_history;
  - the tolerance_rule wording;
  - the anchoring wording.
- No numerical parameter or procedure differs.
- `verify.verify_calibration` re-verifies the record under design 409e7dcb: VERIFIED, chosen R = 20.
- Calibration was not rerun.
- Check record: `validation/prereg_check_409e7dcb_draft_20261106T050444Z.txt` (23 / 23).

**Tests** (`validation/r3_6_tests_20261106/`, exit codes recorded):

| suite | result |
|---|---|
| compile | exit 0 |
| collection | 177 tests |
| `test_crypto.py` | 30 passed |
| `test_statistics.py` | 33 passed |
| `test_pipeline.py` | 114 passed, including two new anchoring tests |
| full CE5 | 177 passed, 0 failed, 0 skipped |

The tamper log is unchanged across the run. The frozen CE2–CE4 suite was not rerun, because no frozen file or CE5 dependency changed.

### Presentation copies of raw records
- Seven raw test records contain the local temporary-directory prefix.
- Presentation copies with that prefix replaced by `[LOCAL_TEMP_PATH_REDACTED]` are in `validation/redacted_for_presentation_20261106/`.
- `REDACTION_MANIFEST.json` lists the original and redacted hashes and 30 replacements.
- Each copy reverses to its original byte for byte. The originals are unchanged.

### Remaining before production
1. Approval of the 409e7dcb draft.
2. Commit.
3. `ots stamp`, then `ots upgrade`.
4. A successful `ots -v verify` recorded at `<approved file>.ots.verification.json`. This needs access to a Bitcoin node.
5. At least 2 GiB of available memory.
6. Production with `--prereg` and `--prereg-sha256`.

The gate and production have not run.
