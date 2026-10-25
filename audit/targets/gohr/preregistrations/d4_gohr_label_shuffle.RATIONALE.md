# D4 Gohr label shuffle — design rationale, statistical review, traceability

Companion to `d4_gohr_label_shuffle.json` (protocol `d4-gohr-speck3264-r5-label-shuffle-v4`,
status **DRAFT, pending human approval**). Approval fields and commands are in
`d4_gohr_label_shuffle.APPROVAL_SHEET.md`.

## 0. Statistical design (rule `near_chance_baseline_gated_v4`, 2026-10-25)

### 0.1 Sources: overarching methodology vs D4 operationalization
There are two distinct sources. No provision of one is attributed to the other.

**Overarching audit methodology.** `~/Downloads/31 (1).pdf` (SHA-256 `79cb7ca3…42e36c3`, 44 pages; read in full via macOS PDFKit text extraction). It governs the general rules applied here:
- §3.6.4 Stages 4–6: controlled perturbation, comparative evaluation, and an inconclusive finding when the criteria are not all met.
- §3.9.4: a confidence interval for every primary result.
- §3.9.6: hypotheses fixed before results.
- §3.9.7: multiple-comparison control.
- §3.9.8 and §3.9.9 Rule III: non-rejection is not evidence of equivalence.
- §3.9.10: reporting requirements.

It does **not** prescribe D4 label shuffling, a near-chance expectation, the blunt-control limitation, a baseline floor, or any numerical margin.

**D4 planning source.** *Dataset Integrity — Redesign From First Principles* (dated 2026-08-27 in its filename). It is archived at `sources/d4_planning_notes_2026-08-27.md` (SHA-256 `0fd2854a…c3b4c7`, 31 047 bytes), byte-identical to the copy you supplied on 2026-10-25, and was read in full. The hash fixes the bytes. That this file is the 2026-08-27 note, earlier referred to as "Pasted markdown(20260827-073650).md", is your attestation (`planning_source_attestation`, pending).

What section F, "Controlled Perturbations — D4 critique and additions", supplies (verbatim quotes are in the JSON):
- D4 establishes "that the training pipeline does not produce above-chance accuracy when the label carries no information". Table L gives the decision basis as "near-chance under shuffle".
- Label shuffling is "a blunt instrument" that destroys genuine signal and artifacts together.
- The targeted generation-order and label-leakage screen (section E) is "a necessary complement". It is **not performed here**.
- A "verify, don't assume" item: does the shuffle touch only the label pairing? This is implemented as `test_label_shuffle_end_to_end_permutes_only_the_training_labels`.

What section F does **not** supply:
- any numerical D4 margin;
- the baseline-adequacy gate;
- the exact-binomial TOST for D4.

Section D recommends TOST for **D3**, with δ "grounded in domain reasoning … not a round number". That guidance concerns D3. The D4 margin is therefore disclosed as a stated practical tolerance (§0.7), not presented as derived. Section F's extra perturbations (partition swap, feature shuffle) and the positive control (for D3) are not implemented here.

Neither the hypotheses, the margin nor the floor was taken from the September `EFFECT_DETECTED` result. Its status is known. Its per-replicate arrays are not on disk and were not inspected as new data.

### 0.2 Why earlier drafts were insufficient
- **v2 (paired degradation only).** It is satisfied by a shuffled model at 0.70 against a 0.929 baseline, which is far from chance.
- **v3 (near-chance only).** It is satisfied if the baseline pipeline is also at chance, in which case the control is uninformative.
- **v4.** It requires both near-chance shuffled fits and adequate intact-label baselines.

Neither v2 nor v3 was approved or run.

### 0.3 Sampling design (verified in code) and the binomial model
- **Generator.** The test partition is one `speck.make_train_data(n, 5)` call. That function draws each label as an independent `os.urandom` bit (`Y = urandom(n) & 1`) and each row's key and plaintexts afresh. Rows are therefore i.i.d. and **class counts are not fixed** (`test_generator_samples_labels_iid_and_does_not_fix_class_counts`).
- **Independence from training.** The partition is generated separately from training and reaches only `predict()` of reloaded models.
- **Consequence.** Conditional on a fitted model m, its correctness indicators are i.i.d. Bernoulli(p_m), so X_m ~ Binomial(n, p_m) **exactly**. No stratified procedure is needed, because no stratum is fixed.
- **Shared test sample.** All 2k fits share one test sample, so their counts are dependent. They are never pooled as k·n independent examples. Every claim is made per fit and combined by IUT or Bonferroni, and neither needs independence.
- **Class diagnostics.** True-negative and true-positive rates per fit are reported descriptively and never gate.

### 0.4 Shuffled-label near-chance endpoint
Band: L = 0.5 − δ, U = 0.5 + δ, with δ = `chance_margin` (candidate 0.005, giving [0.495, 0.505]).

| Per fit | H0 | H1 | Exact test (reject iff p < level) | Level |
|---|---|---|---|---|
| EQUIVALENT | p ≤ L or p ≥ U | L < p < U | P(X ≥ x \| L) < α **and** P(X ≤ x \| U) < α, i.e. the 90 % CP interval lies strictly inside (L, U) | α |
| ABOVE_BAND / BELOW_BAND | L ≤ p ≤ U | p > U / p < L | P(X ≥ x \| U) < α/(2k) / P(X ≤ x \| L) < α/(2k) | α/(2k) each |

- **SUPPORTED:** all k fits are EQUIVALENT. This is an intersection-union test at level α with no adjustment.
- **NOT_SUPPORTED:** some fit departs. Bonferroni over the 2k one-sided tests gives familywise level α.
- **INCONCLUSIVE:** otherwise.

### 0.5 Intact-label baseline-adequacy gate
Floor t = `baseline_adequacy_threshold` (candidate 0.90). It uses the k registered baseline fits and the same test partition.

| Per fit | H0 | H1 | Exact test | Level |
|---|---|---|---|---|
| ADEQUATE | q ≤ t | q > t | P(X ≥ x \| t) < α, i.e. the one-sided 95 % CP lower bound is **strictly** above t | α |
| INADEQUATE | q ≥ t | q < t | P(X ≤ x \| t) < α/k, i.e. the one-sided 1−α/k upper bound is strictly below t | α/k |

- **PASSED:** all k fits are ADEQUATE. This is an IUT at level α.
- **FAILED:** some fit is INADEQUATE. Bonferroni gives familywise level α. This means a baseline is **demonstrably** below the floor.
- **INDETERMINATE:** otherwise. Adequacy is not established, but inadequacy is not established either.

No baseline is re-run, replaced or excluded after its accuracy is seen.

**Justification of 0.90.** It is a proposed reproduction-adequacy tolerance, **not** a value prescribed by Gohr or by the methodology.
- **Benchmark.** The published 5-round benchmark is 0.929 on 10⁶ test samples, as you cited it; the paper was not available offline.
- **Comparison with this repository.** It matches Gohr's uploaded code: `train_5_rounds.py` calls `train_speck_distinguisher(200, num_rounds=5, depth=10)`, and the run uses `make_resnet` with L2 1e-5, Adam, MSE loss, batch 5000, `cyclic_lr(10, 0.002, 0.0001)`, 10⁷ training rows, 10⁶ validation rows, and best-val_loss checkpointing.
- **Differences.**
  - `shuffle=False` here, against the Keras default `shuffle=True` in Gohr's code;
  - seeded initialisation;
  - the TF 2.15 runtime;
  - the local seed patch in `train_nets.py` (commit 6062a31).
- **So the paper does not validate the floor automatically.** 0.90 leaves 2.9 points of tolerance for those differences and for seed-to-seed training variation.
- **Test-set uncertainty is not seed variance.** The paper's binomial SE (about 0.00026 at n = 10⁶) describes one model's test sample, not training-run variation. No repository evidence measures this pipeline's seed spread, so the tolerance is a judgment.
- **Operating characteristics (design only).** ADEQUATE iff x ≥ 900 494, and INADEQUATE iff x ≤ 899 301.

| true q | P(ADEQUATE \| q) | P(INADEQUATE \| q) |
|---|---|---|
| 0.929 | 1.000 | 0.000 |
| 0.902 | 1.000 | 0.000 |
| 0.901 | 0.955 | 0.000 |
| 0.9005 | 0.509 | 0.00003 |
| 0.900 | 0.0499 (size) | 0.0100 (size) |
| 0.899 | 0.000 | 0.841 |

### 0.6 Overall D4 outcome and the secondary paired endpoint
Overall outcome (`fixed_design.overall_rule`, `combine_outcome`):

| Condition, checked in this order | Overall |
|---|---|
| any planned fit missing | `NOT_COMPLETED` |
| shuffled endpoint NOT_SUPPORTED | `NOT_SUPPORTED` |
| baseline gate FAILED | `BASELINE_INADEQUATE` |
| shuffled SUPPORTED **and** baseline PASSED | `SUPPORTED` |
| anything else | `INCONCLUSIVE` |

- **What SUPPORTED requires.** All 2k component tests must reject, so it is an IUT at level α.
- **What does not count.** A near-chance primary with a FAILED or INDETERMINATE baseline can never be SUPPORTED (`test_near_chance_primary_cannot_be_supported_without_adequate_baselines`, plus the full truth table in `test_composite_rule_truth_table`).
- **Failures stay distinguishable.** A demonstrably non-equivalent shuffled fit gives `NOT_SUPPORTED`, which is checked first. A baseline failure gives `BASELINE_INADEQUATE`. An unestablished baseline gives `INCONCLUSIVE`, with `baseline_adequacy_gate = INDETERMINATE`. The component fields always carry the underlying states, e.g. `NOT_SUPPORTED` together with `baseline_adequacy_gate = FAILED`.
- **Integrity is separate.** Integrity verification is reported separately (`verify` → `integrity: VERIFIED`) and is not scientific success.
- **Separate fields.** The certificate reports `overall_d4_outcome`, `shuffled_near_chance_endpoint`, `baseline_adequacy_gate` and `paired_degradation_endpoint` separately.

**Paired degradation (secondary, non-gating).**
- **Counts.** b = baseline wrong & perturbed right; c = baseline right & perturbed wrong; δ_r = (b − c)/n.
- **Test.** One-sided exact McNemar, P(B ≤ b | b + c, ½), equal to 1 when b + c = 0.
- **Rule.** Established iff every replicate has p < α and δ_r ≤ −0.01.
- **Why it doesn't gate.** It is secondary because a shuffled model can be much worse than its baseline and still far from chance. The old co-primary option's purpose is now served by the baseline gate.
- **Error rate.** Citing it as an additional confirmatory claim would raise the familywise error to ≤ 2α. The certificate marks it `secondary_non_gating`.

### 0.7 The near-chance margin (`chance_margin`, PENDING; candidate 0.005)
±0.5 percentage points is an **explicit practical design choice** of the protocol owner. It is not prescribed by the PDF, by section F of the planning note, or by Gohr's paper.

**What the band [0.495, 0.505] means here.** A shuffled fit counts as near chance if its accuracy is established to be within half a point of 0.5. That is at most about 1.2 % of the published 5-round signal (0.929 − 0.5 = 0.429). D4 therefore asks whether shuffling removes essentially all of the pipeline's advantage. A larger surviving residual is treated as a practically relevant leak. A residual inside the band is not ruled out by D4; it is left to the targeted screens.

**What the band does not depend on.**
- **The test-set size.** n = 10⁶ sets precision and operating characteristics, not acceptability, so the margin was not narrowed to 0.0025 merely because 0.0025 would also be powered.
- **D4 results.** The margin must not be selected or altered from any new D4 result.
- **The September result.** It does not determine the margin.

The design-only operating characteristics (below; `python -m audit.dataset.tools.d4_near_chance_operating_characteristics`) show:
- **At 0.005.** P(fit not EQUIVALENT | p = 0.5) = 6.5·10⁻¹⁷, so all five fits pass with probability ≥ 1 − 3.3·10⁻¹⁶. A fit at 0.5025 passes with probability 0.9996. The size at the band edge is ≤ α.
- **Power floor.** The test would be adequately powered for any δ ≥ 0.0025, and underpowered at δ ≤ 0.0015.

The 8-round accuracy is not used to bound the margin, because it is a different-round task.

| δ | P(fit EQ \| 0.5) | all-5 ≥ | P(EQ \| 0.5+δ/2) | P(EQ \| U) (size) | P(ABOVE \| 0.51) | P(ABOVE \| 0.5+2δ) |
|---|---|---|---|---|---|---|
| 0.001 | 0.277 | 0.000 | 0.172 | 0.041 | 1.000 | 0.282 |
| 0.0015 | 0.825 | 0.123 | 0.440 | 0.050 | 1.000 | 0.664 |
| 0.002 | 0.981 | 0.907 | 0.639 | 0.050 | 1.000 | 0.923 |
| 0.0025 | 0.9992 | 0.996 | 0.804 | 0.050 | 1.000 | 0.992 |
| 0.003 | 0.99999 | 0.9999 | 0.912 | 0.050 | 1.000 | 0.9997 |
| 0.005 | 1.000 | 1.000 | 0.9996 | 0.050 | 1.000 | 1.000 |
| 0.01 | 1.000 | 1.000 | 1.000 | 0.050 | 0.005 | 1.000 |

### 0.8 Verification and what it does not cover
- **Persisted.** The raw test predictions of every selected model, with SHA-256, dtype, shape and class counts in the certificate.
- **Recomputed by `verify`.**
  - every correct count, class count and (b, c), from those files and the preflight-bound test labels;
  - every p-value, bound, interval, state, endpoint and gate result and the overall outcome, from the integer counts;
  - all of these are compared by deep equality with the certificate.
- **Refused by `verify`.** Any margin, floor, α, rule, role, `effect_threshold`, test n or procedure that differs from the approved design.
- **Checked only with `--rerun-models`.** Each selected checkpoint, reloaded after a digest check, must reproduce the persisted thresholded predictions exactly.
- **Not covered.** Re-training (TensorFlow training is not bitwise reproducible); and the generator's assumptions, which are D3's subject.

## 1. Architecture

| Layer | Module | Owns |
|---|---|---|
| Shared integrity | `audit/dataset/integrity.py` | SHA-256, atomic/exclusive writes, all-or-nothing directory writes, canonical partition digest, exact-row overlap audit, git-anchored record check, source identity |
| Shared engine | `audit/dataset/d4_controlled_perturbation.py` | `D4Adapter` contract and its enforcement, replicate plan and seeds, paired evaluation, perturbation persistence/verification, `evaluate_decision` (used at run and verify time), certificate, `recompute_certificate_decision`, generic preregistration review |
| Perturbation | `audit/dataset/perturbations/label_shuffle.py` | `apply` and the definition check `verify` |
| Target adapter | `audit/dataset/adapters/gohr_d4.py` | Gohr model fit, hash-chained epoch records, crash recovery, selection and reload |
| Experiment | `audit/dataset/experiments/d4_gohr_label_shuffle.py` | fixed Gohr design, data generation, input encoding, overlap policy, stages `protocol-check / preflight / run / verify` |

`audit/dataset/adapters/gohr.py` is **pinned byte-for-byte** by the sealed Gohr D0 build attestation and the D3 certificate (SHA-256 `3c0b7ac5…`). Commits `2a92e47` and `822414f` had modified it, which broke those pins. This change restores the file exactly; the D4 lifecycle now lives in the `gohr_d4.py` subclass. All 19 source pins in the sealed D0/D3 records match the working tree again.

**What the tests show about genericity.** `test_d4_engine_contract.py` runs the shared engine with a second, non-Keras adapter on generic binary data. A subprocess check confirms the engine, integrity and perturbation modules import no Keras, TensorFlow, cipher or Gohr code. This shows the shared interface can be driven by another implementation. It does not show that any other real target (e.g. AutoND) works; that would need its own adapter and tests.

## 2. Checkpoint lifecycle and the orphan-epoch fix

Epoch commit order: Keras checkpoint → epoch record (checkpoint SHA-256, `val_loss`, fit identity, SHA-256 of previous record) → `state.json` (completed epoch, chain head). Only committed epochs are trusted.

`822414f` adopted an uncommitted ("orphan") epoch whenever its record chained to the committed head and its checkpoint hash matched. Nothing anchors an orphan's own bytes, so an altered `val_loss` would have been adopted. **Orphans are now never adopted.** Each of the eight situations is handled explicitly (table in the `gohr_d4.py` docstring):
- Cases 2–4 (incomplete write; checkpoint without record; complete orphan): files are moved into `quarantine/attempt_NN/` with a note, never deleted. The epoch is retrained from the last committed checkpoint and the retry is listed in `state.json` and in the selection record.
- Cases 5–8 (hash mismatch; altered record or broken chain; foreign identity; malformed state or record) and impossible orderings: refused, with nothing moved.

Fit identity = seed, condition, replicate, config hash, `train_digest` (features+labels actually trained on), `validation_digest`, `evaluation_digest` (test digest, opaque), and the model configuration. A resume against different data under the same configuration is refused.

**Limits of hash chaining.** A hash chain detects corruption, and substitution relative to a digest stored elsewhere. It is not a signature. Someone able to rewrite the epoch records, `state.json` and the selection record together is detected only by the copies held outside that directory: the certificate, which `verify` cross-checks against the files on disk, and the approver's anchor. Retraining a quarantined epoch is not bitwise reproducible on TensorFlow, which is why retries are recorded rather than hidden.

## 3. Data, perturbation and leakage

- **Preflight persistence** is crash-safe. Every output directory (run directory or refusal) is written and fsynced in a hidden `.<name>.staging_*` directory and renamed into place. An in-process failure removes the staging directory. A hard crash leaves it behind: it is never read as a result, and preflight and run then refuse until a human has looked at it. `load_preflight` requires `preflight.json` and exactly the recorded dataset files.
- **Partitions** are generated once at preflight by three independent `speck.make_train_data(n, 5)` calls (os.urandom, not seedable) and persisted. Each file's SHA-256 and a canonical digest are recorded; the digest covers role, dtype with byte order, shape and C-order bytes. Every later stage re-verifies them and re-checks the input encoding from the arrays themselves.
- **Label shuffle** persists only its permutation indices. Before reuse they must:
  - be a bijection of range(n), checked by bincount, as little-endian int64 of length n;
  - match their SHA-256;
  - equal the permutation the audit seed regenerates.

  The sidecar binds the indices to the training-data digest, the training-label digest, the condition (`perturbed`), replicate, model seed, audit seed and perturbation name. The labels are re-derived from the indices, checked against the recorded perturbed-data digest, and passed through `LabelShufflePerturbation.verify`. Non-index perturbations use the generic label/feature-file path, which is tested separately.
- **Leakage:** the test partition only ever reaches `predict()` of the reloaded selected models. `GohrD4Adapter` refuses to be constructed with test data. Checkpoint selection uses validation loss only, and no test statistic is computed before every fit has finished.

## 4. Partition overlap

`integrity.audit_partition_overlap` bit-packs binary rows into exact keys: big-endian uint64 for keys of 8 bytes or less, fixed-width void scalars otherwise. Peak memory is about 8 bytes per row plus one chunk. It reports, separately:
- within-partition duplicate rows and their label conflicts;
- for each cross-partition pair, shared rows, distinct shared vectors and label conflicts.

Tests cover high-bit and low-bit separation, row order, wide and non-binary rows, chunking, empty partitions and length mismatch.

The audit runs at preflight on the actual D4 partitions, before any training. The counts and digests are bound into `preflight.json` and the certificate. The policy (pending) decides refusal. A refusal writes one atomic `<run>.preflight_refused/` directory holding the measurements and the generated data; nothing is regenerated, deduplicated or altered.

**Prior measurements:**
- **Historical D4 partitions: not measurable.** Only `audit/dataset/evidence/d4/metadata.json` exists; the `.npy` files are not on disk. No overlap claim is made about them.
- The **D0 Gohr production instance** (a different dataset) measured zero duplicates, zero cross-partition overlap and zero label conflicts. That result does not transfer to D4.

## 5. Statistical review

| Item | Implementation | Assessment |
|---|---|---|
| Primary (v4) | Exact binomial TOST per shuffled fit against [0.5 − δ, 0.5 + δ]; IUT over all k; Bonferroni α/(2k) departure tests; gated by baseline adequacy (one-sided exact lower bound > t for all k) | See §0.4–0.6. Equivalence is established positively; non-rejection of p = 0.5 is never used. |
| Paired test | Exact McNemar = binomial on discordant pairs (b, c), per replicate | Exact given independent test rows (independent `make_train_data` draws). Conditional on the fitted pair: it tests these two models, not the training procedure. |
| Sidedness | v1: two-sided; v2: one-sided (perturbed worse) | H1 is directional. v1 would also count an accuracy *increase* as an effect (`test_directional_rule_rejects_an_improvement_that_two_sided_rule_accepts`). |
| Multiplicity | v1: Bonferroni α/k and all k must pass; v2: IUT, each at α | Requiring all k to pass is an intersection-union test, so it has level α without Bonferroni. v1 is therefore doubly conservative. |
| Replicate CI | Percentile bootstrap of mean Δ over k replicates | With k = 5 there are only 126 distinct resamples, and percentile coverage is poor. v1 gates on it; v2 reports it as descriptive only. |
| Practical threshold | v1: abs(mean Δ) ≥ thr; v2: every Δ_r ≤ −thr | `effect_threshold` is a **fraction** (0.01 = 1 pp). The September engine docstring said "percentage points"; that is corrected here, and the value and its comparison are unchanged. |
| Self-reported values | Decision recomputed from (b, c) and accuracies at run and verify time | A certificate whose p-values, differences, row count or outcome disagree is refused. |
| Failures | Any exception aborts the run; no exclusions or replacements | A partial run is `NOT_COMPLETED`, never a result on fewer replicates. |
| Calibration | None proposed | The exact test needs no simulation for its level. The synthetic-adapter tests check the software only and are **not** a calibration. |

All three rules are implemented in the engine; this runner accepts only `near_chance_baseline_gated_v4` (§0). The rows above for v1 and v2 describe the paired comparison, which v4 keeps as a secondary, non-gating endpoint (one-sided, IUT, as in v2).

## 6. Traceability

**The original "nine fixes" list could not be recovered.** I searched the repository, its history and every local Claude Code transcript under `~/.claude/projects/-Users-oishimoitra/`. The only transcript mentioning the D4 code (`e9599995…`) contains no such list, and every other "nine" refers to nine D3 test files. The list was therefore made in a conversation not stored on this machine. The table below lists every defect recoverable from history, diffs, reviews and code; no item is invented to reach nine.

| # | Defect / requirement | File / function | Change | Regression test(s) | Result |
|---|---|---|---|---|---|
| 1 | Last-epoch in-memory model evaluated, not the `val_loss`-selected checkpoint | `gohr_d4.select_checkpoint`, `GohrD4Adapter.train` | Selection over the complete committed history; reload from disk | `test_selected_checkpoint_is_reloaded_and_resume_agrees`, `test_selects_min_val_loss_…` | PASS |
| 2 | `best_*.keras` keyed by seed, shared by baseline and perturbed, reset on resume | `gohr_d4`, engine `checkpoint_paths` | Removed; per-condition directories; selection identity checked | `test_misattributed_selection_is_refused`, `test_record_from_another_fit_is_refused` | PASS |
| 3 | Persisted shuffled labels reused unverified | engine `save_/load_verified_perturbation`, `LabelShufflePerturbation.verify` | Sidecar binding, digest, regenerate-equality, definition check | `test_tampered_perturbed_labels_are_refused`, `test_perturbation_reused_across_…`, `test_label_shuffle_definition_check` | PASS |
| 4 | Dataset cache checked by shape/dtype only | driver `load_preflight` | Per-file SHA-256 and canonical digests, re-verified every stage | `test_verify_detects_tampering[dataset/test_y.npy]` | PASS |
| 5 | Frozen input encoding not enforced | driver `validate_input_encoding` | Checked from arrays at preflight, run and verify | `test_input_encoding_is_checked_from_arrays` | PASS |
| 6 | Output paths relative to the working directory | driver `REPOSITORY_ROOT`, `integrity.REPOSITORY_ROOT` | Repository-relative | stage tests (run from any cwd) | PASS |
| 7 | Certificate silently overwritten | `integrity.write_directory_exclusive`, `stage_run` | All-or-nothing exclusive result directory | `test_completed_run_and_preflight_are_never_replaced`, `test_directory_write_is_all_or_nothing` | PASS |
| 8 | Run config did not bind selection rule, encoding, approved design | driver `design_of`, `run_config_hash` | Config hash = hash of the approved design + prereg anchor | `test_end_to_end_verify_passes` | PASS |
| 9 | Only the selected checkpoint verified | `gohr_d4.verify_epoch_chain` | Every checkpoint verified, at selection, resume and verify | `test_any_tampered_checkpoint_is_refused[1,2,3]` | PASS |
| 10 | `val_loss` records alterable | `gohr_d4.verify_epoch_record` | Hash chain + committed head; schema and logs consistency | `test_any_altered_val_loss_is_refused`, `test_rewritten_consistent_chain_…` | PASS (limits in §2) |
| 11 | Orphan epoch adopted with unanchored `val_loss` (introduced in `822414f`) | `gohr_d4.classify_uncommitted`, `quarantine` | Never adopt; quarantine + retrain; suspicious cases refused | `test_orphan_with_altered_val_loss_is_not_trusted`, `test_crash_window_with_altered_orphan_val_loss_retrains_the_epoch`, `test_suspicious_uncommitted_artifacts_…` | PASS |
| 12 | Fit not bound to partition contents | `GohrD4Adapter.fit_identity` | Train/validation/evaluation digests in identity | `test_resume_against_substituted_partition_is_refused` | PASS |
| 13 | Partition overlap unaudited | `integrity.audit_partition_overlap`, `stage_preflight` | Read-only audit bound to preflight; policy-gated refusal | `test_overlap_*`, `test_preflight_overlap_refusal_keeps_data_…` | PASS (code); policy **BLOCKED** on approval; historical D4 data **not measurable** |
| 14 | `gohr.py` pinned by sealed D0/D3 evidence was modified | `adapters/gohr.py` | Restored byte-for-byte; D4 code moved to `gohr_d4.py` | 19/19 pins match (scripted check); D3 prereg tests pass | PASS |
| 15 | Decision and p-values trusted as self-reported; no independent verification | engine `evaluate_decision`, `recompute_certificate_decision`; driver `verify_run` | Recompute from (b, c); verify re-checks every artifact | `test_certificate_tampering_is_detected`, `test_verify_detects_tampering` | PASS |
| 16 | Adapter contract implicit; attribution unchecked | engine `D4Adapter`, `verify_adapter_selection` | Required selection fields checked against condition/replicate/seed/config/data | `test_adapter_without_selection_…`, `test_misattributed_selection_…`, `test_engine_imports_no_target_code` | PASS |
| 17 | Two-sided test for a directional hypothesis; Bonferroni plus k = 5 bootstrap gate | engine `evaluate_decision` | Superseded by row 26 | `test_directional_rule_*`, `test_identity_perturbation_yields_no_effect` | Superseded |
| 29 | D4 planning source unavailable and unhashed; clean checkout held a stale v3 draft | `sources/d4_planning_notes_2026-08-27.md`, JSON `methodology_source.d4_operationalization` | Archived byte-identical, SHA-256 recorded, section F quoted; v4 draft committed unapproved | `test_shipped_draft_registers_v4_and_leaves_every_scientific_choice_pending`, clean-checkout test run | PASS; attestation **BLOCKED** (human) |
| 28 | Near-chance result could be SUPPORTED with a baseline at chance; 8-round accuracy misused as a margin ceiling; planning source misattributed | engine `baseline_adequacy_replicate`, `evaluate_baseline_adequacy`, `combine_outcome`, `class_diagnostics`, v5 certificate; driver design and verify checks | Rule `near_chance_baseline_gated_v4`; baseline gate; composite outcome; paired role fixed to secondary; ceiling removed; source recorded by title/date, unhashed | `test_near_chance_primary_cannot_be_supported_without_adequate_baselines[3]`, `test_composite_rule_truth_table[11]`, `test_baseline_states_match_exact_rational_reference[3]`, `test_exact_level_of_the_baseline_tests[3]`, `test_generator_samples_labels_iid_and_does_not_fix_class_counts` | PASS (code); floor, margin, attestation **BLOCKED** (human) |
| 26 | Primary endpoint tested a reduction relative to baseline, not near-chance accuracy | engine `near_chance_replicate`, `evaluate_near_chance(_decision)`, `evaluate_paired_degradation`, v4 certificate, `recompute_certificate_decision`; driver `check_target_design`, `check_decision_values`, `verify_run`, `recount_from_predictions` | Rule `near_chance_tost_v3` (§0); separate `chance_margin`; paired endpoint kept separately; predictions persisted and recounted | `test_d4_near_chance.py` (63 tests incl. exact rational reference, size, duality), `test_verify_refuses_a_certificate_that_departs_from_the_approved_design[7]`, `test_verify_without_a_result_is_not_completed` | Superseded by row 28 (v3 → v4) |
| 18 | Threshold documented as percentage points but compared as a fraction | engine docs, certificate methodology | Documented as a fraction; value unchanged | — (documentation) | PASS |
| 19 | No D4 preregistration, approval gate or staged workflow | prereg JSON, engine `review_/require_approved_preregistration`, driver stages | Draft + gates; preflight/run refuse until approved, committed and anchored | `test_protocol_check_reports_pending_draft`, `test_preflight_refuses_the_unapproved_draft`, `test_anchored_record_*`, `test_shipped_gohr_draft_is_well_formed_and_entirely_pending` | PASS (code); approval **BLOCKED** (human) |
| 20 | Test data given to the training adapter | `GohrD4Adapter.__init__` | Refuses test arrays; test bound by digest only | `test_adapter_refuses_test_data` | PASS |
| 21 | `feature_shuffle`, `class_imbalance`, `metadata_removal` have no definition check | `perturbations/*.py` | Engine now refuses them (`verify` missing); not implemented | `test_perturbation_without_verify_is_refused` | NOT FIXED (out of D4 Gohr scope) |
| 23 | Persisted permutation not checked as a bijection or bound to the label digest, condition and seed | `integrity.verify_permutation_indices`, engine `save_/load_verified_perturbation`, `LabelShufflePerturbation.permutation_indices` | Indices persisted and verified (dtype, length, bounds, bijection, digest, regenerated plan); sidecar binds condition, seed, label digest | `test_persisted_indices_are_a_verified_int64_bijection`, `test_structurally_invalid_or_unplanned_indices_are_refused[6 cases]`, `test_indices_bound_to_training_labels`, `test_perturbation_reused_across_…[8 fields]` | PASS |
| 24 | Preflight dataset persistence not crash-safe; refusal written in two steps | `integrity.write_directory_exclusive`, `stale_staging_directories`; driver `stage_preflight`, `load_preflight` | fsync + atomic rename; stale staging refused; single refusal directory; exact file set | `test_failure_during_dataset_write_…`, `test_hard_crash_leftover_…`, `test_crash_during_rename_…`, `test_incomplete_or_altered_preflight_directory_is_refused[4]` | PASS |
| 25 | Experiment must train with the D4 adapter while the pinned base stays byte-identical | driver `make_adapter_factory`, `generate_gohr_partitions` | `GohrD4Adapter` trains; the pinned `GohrAdapter` only generates data | `test_experiment_trains_with_the_d4_adapter_…`, `test_completed_run_was_trained_by_the_d4_lifecycle`, `test_pinned_gohr_adapter_is_byte_identical`, `test_every_source_pinned_by_sealed_d0_d3_evidence_still_matches[2]` | PASS |
| 27 | Perturbation and checkpoint reload not verified end to end on the real pipeline; network output not re-checkable | driver `rerun_selected_models`, `verify --rerun-models` | Reload digest-checked selected checkpoint, re-predict test, require exact label agreement | `test_label_shuffle_end_to_end_permutes_only_the_training_labels`, `test_selected_checkpoint_is_the_val_loss_argmin_and_its_reload_reproduces_the_predictions`, `test_rerun_detects_predictions_not_produced_by_the_selected_checkpoint` | PASS |
| 22 | AutoND staged 5→7 training hand-off | AutoND implementation-integrity workflow | Not touched: it is a separate workflow with no shared D4 component, and no AutoND D4 design exists | — | **BLOCKED**: needs an AutoND D4 protocol before any code |

## 7. Residual engineering notes

- Preflight holds the generated partitions and their `.npy` bytes in memory (about 2–3 GB peak at 10⁷ rows).
- `run` is resumable: fits recover through the adapter, and perturbations through their sidecars. `result/` is written once.
- The September result in `audit/dataset/evidence/d4/` stays untouched and historical. New runs never write there; the run directory must be new.
