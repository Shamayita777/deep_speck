# AutoND D5 — implementation findings (2026-10-29)

Repository `deep_speck`, branch `audit-framework`, base `3ad2607`. Each item is labelled:

| Label | Meaning |
|---|---|
| **[verified]** | Read or computed from the actual files during this pass |
| **[design]** | A recommendation or choice made in this protocol |
| **[assumption]** | Assumed and not checked |
| **[unverified]** | Not checked here |

No D5 data was generated and no production training ran. Unit tests used tiny sizes (§9).

## 0. Sources checked

**Methodology PDF.** `~/Downloads/31 (1).pdf`, SHA-256 `79cb7ca33d9c1fc12ee14873b8db3f40eb636c64ae43079336032cf9742e36c3` **[verified]**. This is the digest the D4 protocol records.
- The task named `31 (1)(20261029-063953).pdf`, but no file of that name exists under `~` (searched to depth 4). The suffix looks like an attachment timestamp, so I treated `31 (1).pdf` as the document **[assumption]**.
- Sections reviewed: §3.1–3.3, 3.6, 3.9, 3.10.2–3.10.3 and 3.11.

**Repository files.** SHA-256 values, all clean at HEAD **[verified]**:

| File | SHA-256 |
|---|---|
| Generic engine `audit/dataset/d5_training_data_scaling.py` | `04f3f319…a306` |
| `audit/dataset/adapters/autond.py` | `be81a5dc…9dd1` |
| `audit/dataset/experiments/d0_autond_canonical.py` (holds `AUTOND_SOURCE_PIN`) | `f5145ed2…8b20` |
| `audit/dataset/integrity.py` | `e1bf42cc…caff` |
| Gohr wrapper `audit/dataset/experiments/d5_gohr_training_data_scaling.py` (read only) | `bffd5751…4ce4` |

**Upstream pin.** The checkout is at `8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` with 0 tracked modifications. **[verified]**
- The SHA-256 values of `main.py`, `optimizer.py`, `train_nets.py`, `dbitnet.py`, `gohrnet.py` and `ciphers/speck3264.py` equal `AUTOND_SOURCE_PIN` exactly.
- Untracked `__pycache__/*.pyc` and `.DS_Store` are present. The adapter does not treat them as shadowing. Python uses a `.pyc` only when it matches its source.

**D4.** No D4 process runs on this machine; D4 production runs on Kaggle **[verified locally]**. No D4 file was read for modification or written.

## 1. The generic D5 engine API (reused unmodified) [verified]

| Function | What it does |
|---|---|
| `run_d5(root, training_sizes, test_samples, replicates, audit_seed, total_epochs, batch_size, train_dataset_factory(n, seed), test_dataset_factory(n, seed), model_factory(seed), training_callbacks_factory(seed)\|None, evaluate_model(model, x, y) -> (loss, acc), experiment_name, bootstrap_replicates, manifest, dataset_replay_reason, replicate_seed_role)` | Runs the experiment and returns a `D5Result` |
| `generate_certificate(result, …)` | Builds the engine's certificate |
| `load_test_dataset`, `load_replicate_dataset` | Re-verify file, logical and prefix hashes |
| `condition_paths`, `prefix_hashes_for` | Locate a condition's files and its prefix hashes |
| `_aggregate` | Reused by the target verifier to recompute the engine's statistics. It is not copied. |

**Design properties:**
- **Sizes and baseline.** Sizes are sorted and deduplicated; the smallest is the baseline.
- **Replicate seeds.** Each comes from `SeedSequence(audit_seed).spawn(R)`. A replicate uses **the same seed for every size**: `model_factory(seed)` is called once per condition with the replicate's seed.
- **Data.** One maximum-size dataset per replicate. Smaller sizes are prefixes `x[:n]`.
- **Test set.** **One fixed test set shared by every condition.** The engine's own manifest records `individual_test_examples_are_not_independent_replicates`.
- **Training.** `model.fit(x[:n], y[:n], epochs, batch_size)` runs with **no `validation_data`**. Evaluation uses the in-memory final-epoch model.
- **Statistics.** A paired t-test of each size against the baseline, Holm correction (α hard-coded at 0.05), percentile bootstrap CIs (seed `audit_seed + 991`), and per-replicate OLS slopes on log10 n.
- **Completeness.** A missing condition raises an error, so no result is produced.

## 2. AutoND binding and data generation [verified]

- **Binding.** `AutoNDDatasetAdapter(autond_root, cipher_name, delta_plain_int, num_rounds, delta_key_int, scenario, expected_source_digests, expected_upstream_commit)`.
  - The commit and every import-graph digest are checked **before** import.
  - A foreign or shadowing `main`/`optimizer`/`train_nets`/`dbitnet`/`gohrnet`/`ciphers` is refused, and nothing is evicted.
  - The digests of the files actually bound are recorded.
- **Generation.** `generate_partition(n)` calls `main.make_train_data(encrypt, 32, 64, n, 5, Δ_bits, 0)`.
  - X is `np.hstack([C0, C1])`: shape (n, 64), uint8, values {0, 1}.
  - Y is an independent `urandom & 1` bit per row. Rows with Y = 0 are replaced by random bits.
  - All randomness is `os.urandom`, with no seed parameter, so datasets cannot be replayed from a seed.
- **Model.** `dbitnet.make_model(64)` declares `Input(shape=(64, 1))`. Upstream feeds it (n, 64) arrays, which Keras expands, and D5 does the same.
  - `train_nets` compiles dbitnet with `Adam(amsgrad=True)`, loss `mse`, metric `acc`, and no LR schedule.
  - `EPOCHS = 10`, `BATCHSIZE = 5000`, `NUM_SAMPLES = 10⁷`.
  - Upstream selects the min-val_loss checkpoint per round. In the 12 October bundle, round 5's min-val_loss epoch was 10 of 10, i.e. the final epoch.

## 3. Safeguards reused without alteration

**From the engine:**
- Dataset and prefix hashing with re-verification.
- Refusal of partially persisted data and of non-matching resume.
- Immutable run manifest and config hash.
- Completeness check.

**From the adapter:**
- Pre-import pin enforcement.
- Shadowing refusal.
- Bound-file digests.

**From `integrity`:**
- `verify_anchored_committed_record` (anchor, tracked, committed, clean).
- `source_identity`.
- `audit_partition_overlap` (exact rows).
- `write_bytes_exclusive` / `write_json_exclusive`.

**From `d0_autond_canonical`:** `AUTOND_SOURCE_PIN`, re-checked against the files.

## 4. Incompatibilities and engine properties, handled at target level (engine not modified)

| # | Finding | Handling |
|---|---|---|
| 1 | The engine fits without validation data and evaluates the final epoch, so AutoND's validation-based selection, staging and stopping rule cannot be reproduced through the engine. | Declared as the design: a fixed round-5, fixed-budget experiment, not full-pipeline. Supporting evidence: in the reference run, round 5's selected epoch was the final one. |
| 2 | Checkpoint bytes are not bound to a digest. The engine also resumes from `latest.keras`. | The target never resumes: the run directory must be new or empty and is exclusively claimed. Each checkpoint's SHA-256 is logged at evaluation time and re-checked, and every checkpoint is reloaded and re-evaluated. |
| 3 | `_paired_t_test` without SciPy falls back to `erfc(|t|/√2)`. The comment calls this conservative, but it is **anti-conservative**: t = 2.5, df = 11 gives p = 0.0295 (t) versus 0.0124 (fallback). | A SciPy gate runs before any file is created. |
| 4 | Holm α is hard-coded at 0.05. | The registered α must equal 0.05 (single option). |
| 5 | The CIs are percentile bootstrap, which under-covers at k ≈ 12, and the only effect size is the mean difference. | Registered supplementary statistics: 95 % paired-t CI (interval of record), d_z, difference relative to baseline, exact Wilcoxon (sensitivity) and Shapiro–Wilk (diagnostic). The engine's output is unchanged and remains the primary test. |
| 6 | There is no practical-significance threshold. Methodology §3.6.4 Stage 6 requires effect sizes that are practically meaningful. | Registered decisions, now approved: `practical_effect_threshold` = 0.005 with `practical_effect_rule` = `t_interval_excludes_threshold_band`. |
| 7 | The engine's `test_loss` is Keras' compiled objective, **MSE + DBitNet's L2 penalties**, not pure MSE. Found by the tiny run, where the difference was about 0.0073. | Re-evaluation reports MSE, the regularization term and the objective, and compares the objective. Loss contrasts are exploratory. |
| 8 | With all differences identical, `_paired_t_test` returns t = ±inf and p = 0, which JSON can only store as a non-standard token. | Result files keep the engine's own serialisation (NaN/Infinity tokens) and are written exclusively. |
| 9 | Importing the engine imports Keras. | Driver and verifier import it lazily. A test confirms the driver import loads neither TF nor the engine. |

## 5. Statistical and reproducibility risks [design]

**Statistical:**
- **Design resolution.** No pilot estimate of σ_d (SD of paired differences) exists. At k = 12, 80 % power requires |Δ| ≥ 1.08 σ_d at Holm's strictest step.
- **Dependence on the test set.** All conditions share one test set, so contrasts are conditional on it. The binomial test-set SE (≈ 3·10⁻⁴ at 0.92) is reported.
- **Non-independence.** Prefixes are nested and therefore not independent. The replicate is the unit.

**Reproducibility:**
- **GPU nondeterminism.** Seeding fixes the initialisation and shuffle order, but not GPU kernels.
- **Datasets.** They cannot be regenerated; the persisted arrays are the object of record.
- **Runtime.** About 49 min per replicate at the owner-supplied T4 rate, roughly 9.8 h for k = 12 on one GPU **[assumption: rate]**.
- **Disk.** About 8.5 GB.

## 6. Estimand, scope, limitations

See the JSON `estimand`, `claim_scope` and `scope_and_limitations`, and the RATIONALE.

**The claim is:** a descriptive and inferential characterisation of accuracy against training size for this fixed protocol, conditional on the test set.

**Not claimed:**
- a scaling law;
- cryptographic learning;
- full-pipeline behaviour;
- equivalence from a null result;
- independence from the absence of duplicates.

## 7. Files created (all inside `audit/targets/autond/`)

| File | Purpose |
|---|---|
| `d5/__init__.py` | Package |
| `d5/training_data_scaling.py` | Driver: gates, no-resume claim, data checks, evaluation hook, engine call, certification; CLI `protocol-check` / `run` / `verify` |
| `d5/verification.py` | Independent verification, supplementary statistics, certificate |
| `d5/D5_AUTOND_IMPLEMENTATION_FINDINGS.md` | This report |
| `preregistrations/d5_autond_speck3264_r5_training_data_scaling.json` | Preregistration (APPROVED; anchor `62ef6959…`) |
| `preregistrations/d5_…RATIONALE.md`, `…APPROVAL_SHEET.md` | Rationale; approval record and runbook |
| `tests/test_d5_autond_training_data_scaling.py` | 103 target tests |

`SHA256SUMS.txt` is the II package's manifest, so it was **not** edited.

## 8. Must remain untouched

The following were **not modified**; their tracked files are unchanged against HEAD (§10):
- **Engine and framework:** the generic engine; `integrity.py`; `audit/common/`, `audit/implementation/`; shared schemas and certificates; generic tests.
- **Base adapter and pin:** `adapters/autond.py`; `d0_autond_canonical.py`.
- **Gohr:** the Gohr adapter, wrapper, preregistrations, evidence and tests.
- **AutoND D4:** code, preregistration, evidence and processes.
- **Other stages:** D0–D3, CE1–CE4, II.
- **Elsewhere:** the methodology PDF; the AutoND checkout.

## 9. Tests run safely (no production data or training)

`audit/targets/autond/tests/test_d5_autond_training_data_scaling.py`, run in its own process: **103 passed, 0 skipped** (after the §11–§13 correction passes).

**[real]** tests use the pinned DBitNet and generator on tiny sizes: 96/192 training rows, 1,024 test rows, 3 replicates, 1 epoch, CPU. No gate is stubbed: the end-to-end run is authorised by a throwaway git repository holding byte copies of the registered sources and an approved test-only preregistration. They cover:
- binding, and refusal of a wrong commit or digest before import;
- that imports resolve to the checkout, and that a foreign `dbitnet` is refused;
- seed-identical initial weights, and the compile configuration;
- the generator's representation;
- an end-to-end run giving `VERIFIED` with separate engine and target sections;
- nested-prefix hashes, and that replicates are independent;
- the same seed being used across sizes;
- detection of altered, missing or tampered artifacts;
- missing-condition refusal;
- no resume, including after a configuration change;
- `verify` detecting a change made after certification;
- protected files unchanged.

**[mock]** tests:
- partition-validation refusals;
- the train/test overlap policy;
- re-evaluation on a tiny Keras model;
- classification never claiming equivalence;
- paired-t interval and d_z against SciPy.

**[gate]** tests:
- the shipped preregistration being approved and well formed;
- `run` refusing without confirmation, with an unapproved preregistration, or without SciPy, in each case creating nothing;
- direct calls to `execute()` refusing, before any directory or training: a missing, unanchored or wrongly anchored preregistration; an unapproved protocol or a pending decision; values, commit or protocol id differing from the approved record; an uncommitted preregistration or source edit; a caller-supplied source identity that differs from HEAD; an unpinned binding;
- verification refusing a run whose claim lacks approval evidence, or whose preregistration, values, upstream pin or source identity do not match the approved, committed record;
- duplicate JSON keys being refused;
- validation of decision values.

## 10. Unresolved questions and blockers

1. **Practical threshold.** Approved: 0.005 with the t-interval rule.
2. **Replicate count.** Approved at 12 with the design resolution as stated; no σ_d estimate exists, so null contrasts may be inconclusive (RATIONALE §2).
3. **Approval.** All decisions and approval fields are approved; the protocol is committed with anchor `62ef695989d01c301b47cc21aa9d0313d2975fe294037878f66c49f530b3da68`.
4. **Untested paths.** Physical GPU execution and the full-scale runtime and memory are untested. Production must not share a session or GPU with D4.
5. **Cross-replicate duplicates.** These are not audited (recorded as a limitation).

## 11. Correction pass (2026-10-29)

These defects were found by re-reading the code. Each fix is target-local, and the engine is unchanged.

1. **Hash-only `verify` could report `VERIFIED`.** `verify --no-reevaluate` now reports at most `INTEGRITY_CONSISTENT_NOT_REEVALUATED`.
2. **Non-finite or degenerate statistics passed.** The engine's `_holm_adjust` drops non-finite p-values from the family, which shrinks m and weakens the correction. Its paired t returns ±inf with p = 0 when every difference is identical. Verification now refuses all of the following: a non-finite observation, a non-finite mean, t or p, a zero-variance contrast, a non-finite Holm p, and a Holm family that is not exactly the registered contrasts.
3. **Data audits were unbound.** The pre-training format and overlap records carried no hash. They now record the engine's logical SHA-256 of the audited arrays, and verification requires each record to exist and to match the persisted dataset.
4. **Verification did not require SciPy.** It now does; without SciPy it would recompute the engine's p-values with the anti-conservative fallback.
5. **Fixed-epoch confound undocumented.** The fixed-epoch policy is now stated in the driver's claim scope, the preregistration (`training_policy`, limitations) and the RATIONALE: larger n receive proportionally more updates, so data and compute are confounded.

## 12. Final targeted verification (2026-10-29)

Defects confirmed in the code and fixed:

1. **The SD check accepted `+inf`.** The check was `sd > 0`. Statistical validation now requires every field to be finite, not merely positive.
2. **The statistical check left several gaps.** It now also refuses:
   - raw or Holm-adjusted p-values outside [0, 1];
   - bootstrap CI bounds and per-size summaries that are missing or not finite;
   - Holm-adjusted values that differ from Holm recomputed on the raw p-values;
   - a wrong correction method or baseline;
   - per-contrast or per-size replicate counts that differ from the registered count;
   - observations that do not cover exactly the registered (replicate, size) cells;
   - malformed fields (`None`), which previously raised `TypeError`;
   - the target's own t-intervals and d_z when they are not finite.
3. **The SciPy gate was only on the CLI path.** It ran in `stage_run`, but `execute()`, which claims the directory and trains, did not call it. `execute()` now calls it first, and the preflight exercises the engine's exact `from scipy.stats import t` path against a reference value.

In addition, a certificate whose statistics were computed with the engine's no-SciPy fallback is now shown, by test, to fail verification through the recomputation check.

**Test fixture.** With the new zero-variance refusal, the earlier tiny fixture (2 replicates, 64 test rows) was itself occasionally degenerate. Its two paired differences sometimes coincided, giving SD = 0, and the run was then correctly refused; one in two full runs failed this way. The fixture now uses 3 replicates and 1,024 test rows, and five consecutive runs gave 76 passed. Production sizes are not affected.

## 13. Execution and provenance gates (2026-10-29)

1. **`execute()` trusted its caller for approval and clean sources.** Only the `run` stage enforced them. `execute()` now calls `authorize_execution` after the SciPy preflight and before claiming the directory. It reloads the preregistration from the recorded repository-relative path and anchor (approved, committed, clean, matching this driver), and requires the caller's values, commit and protocol id to equal the approved record's. It also requires the caller's source identity to equal the files clean at HEAD now, and the binding to be the verified pin. A missing preregistration file is now refused as an integrity error rather than raising `FileNotFoundError`. A preregistration whose top-level `status` is not `APPROVED` is now treated as pending.
2. **Verification did not check approval or source provenance.** `provenance_problems` now re-reads the preregistration from git at the recorded commit. It must hash to the recorded anchor and be an approved protocol whose values, protocol id and upstream pin equal the run's. The recorded source identity must list exactly the registered sources, all clean, each digest equal to its committed content at the recorded HEAD, and that HEAD must contain the preregistration commit.
3. **Tests.** The end-to-end fixture no longer stubs the gates. Direct-call tests show that an unapproved, pending, unanchored, uncommitted or mismatched protocol, dirty or mismatched sources, and an unpinned binding are each refused before the run directory exists and before the engine is called. Verification tests cover missing or mismatched provenance. The run-stage refusal test uses an unapproved test-only record, never the approved protocol with its real anchor.
