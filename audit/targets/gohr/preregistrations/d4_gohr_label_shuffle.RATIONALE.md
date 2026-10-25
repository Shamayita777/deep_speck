# D4 Gohr label shuffle — design rationale, statistical review, traceability

Companion to `d4_gohr_label_shuffle.json` (protocol `d4-gohr-speck3264-r5-label-shuffle-v3`,
status **DRAFT, pending human approval**). Approval fields and commands are in
`d4_gohr_label_shuffle.APPROVAL_SHEET.md`.

## 0. Statistical design correction (v3, 2026-10-25)

### 0.1 Source of the scientific question
The D4 question is whether the training pipeline can produce **above-chance** accuracy once permuting the training labels has destroyed the original feature/label correspondence. The expected diagnostic result is **near-chance** accuracy. Label shuffling is a **blunt control**: it destroys genuine signal and label-correlated artifacts together, so a clean result does not show that every form of leakage is absent. Targeted generation-order and label-leakage checks are complementary and are **not performed by this protocol** (`complementary_audits_not_in_scope` in the JSON).

**Discrepancy to resolve before approval.** I read the methodology PDF `~/Downloads/31 (1).pdf` (SHA-256 `79cb7ca3…42e36c3`, 44 pages) in full, via macOS PDFKit text extraction. Its Controlled Perturbation stage (§3.6.4 Stage 4) lists five interventions: regenerating with independent seeds, removing duplicates, changing partitioning, balancing classes, and alternative sampling. It does **not** mention label shuffling, the near-chance expectation, or the blunt-control caveat. Those three elements come from your written instruction of 2026-10-25, which attributes them to the original methodology. The PDF does supply the statistical rules applied here:
- §3.9.4: every primary result, distinguishing accuracy included, gets a confidence interval.
- §3.9.6: hypotheses are fixed before results.
- §3.9.7: multiple comparisons are controlled.
- §3.9.8 and §3.9.9 Rule III: failure to reject a null is never evidence of equivalence.
- §3.6.4 Stage 6: unmet criteria give an inconclusive finding.

Please confirm which document holds the D4 label-shuffle section.

Neither the hypothesis nor the margin was inferred from the September `EFFECT_DETECTED` result. Its per-replicate accuracies were not read while preparing this draft.

### 0.2 Why the v2 draft was insufficient
`directional_iut_v2` establishes only that every shuffled fit is worse than its baseline. A shuffled model at 0.70 against a 0.93 baseline satisfies it while being far from chance (`test_worse_than_baseline_but_far_from_chance_does_not_establish_the_primary_claim`). A reduction relative to baseline never substitutes for near-chance evidence. This runner therefore refuses v1 and v2; their code stays in the engine to reproduce historical evidence.

### 0.3 Primary endpoint: near-chance equivalence (rule `near_chance_tost_v3`)
For perturbed fit r, X_r is the number of correct test predictions of the reloaded selected model, out of n. Conditional on that model, X_r ~ Binomial(n, p_r) **if** the test rows are i.i.d. from the generator and independent of training. That holds by construction here (a separate `make_train_data` call; the test partition reaches only `predict()`). p_r is the model's accuracy over the test-generating distribution.

Chance reference: 0.5. It is the exact accuracy of any label-independent predictor when labels are fair coin flips independent of the features, which is how `speck.make_train_data` documents its labels. The observed test class counts are recorded (`provenance.test_label_counts`); class balance itself is D3's subject.

Band: L = 0.5 − δ, U = 0.5 + δ, where δ = `chance_margin` (a fraction of accuracy; a separate decision from `effect_threshold`).

| Claim per fit | H0 | H1 | Exact test (rejection iff p < level) | Level |
|---|---|---|---|---|
| EQUIVALENT | p_r ≤ L or p_r ≥ U | L < p_r < U | TOST: P(X ≥ x \| L) < α **and** P(X ≤ x \| U) < α, which is the same as the 90 % Clopper–Pearson interval lying strictly inside (L, U) | α (Berger & Hsu 1996) |
| ABOVE_BAND | p_r ≤ U | p_r > U | P(X ≥ x \| U) < α/(2k) | α/(2k) |
| BELOW_BAND | p_r ≥ L | p_r < L | P(X ≤ x \| L) < α/(2k) | α/(2k) |
| INDETERMINATE | — | — | none of the above | — |

The departure test is two-sided around the band because a reliably below-chance model also uses label information. A single fit cannot be both EQUIVALENT and departing; this is proved by the tail identities and asserted in code and tests.

**Joint outcome** over the k registered fits (all are required):
- `SUPPORTED`: all k are EQUIVALENT. This is an intersection-union test: the claim "every p_r is in the band" is rejected-null only if every component rejects, so the level is α with **no** multiplicity adjustment (Berger 1982). No independence between fits is needed.
- `NOT_SUPPORTED`: some fit is ABOVE_BAND or BELOW_BAND. This is a union over 2k one-sided tests, each at α/(2k), so Bonferroni gives a familywise level of α with no independence needed.
- `INCONCLUSIVE`: otherwise. The uncertainty establishes neither claim.
- `NOT_COMPLETED`: any planned fit is missing. A failed run writes no certificate; `verify` exits 5.

Only `SUPPORTED` is a scientific pass. `INCONCLUSIVE` and `NOT_COMPLETED` are never reported as passes (`primary_claim_supported` is false, and `run` and `verify` print "NOT a scientific pass").

**Scope of the claim.** Each statement is about one fitted model, conditional on that model and the shared test sample. The five fits share one test partition, so their accuracies are not five independent test sets and are never pooled. "All five" is a reproducibility statement over these five training seeds. It is not an estimate of the distribution of accuracies over seeds.

### 0.4 Supporting endpoint: paired degradation
- **Discordance and orientation.** On the same test rows: b = baseline wrong & perturbed right, and c = baseline right & perturbed wrong. Then δ_r = acc_pert − acc_base = (b − c)/n; this identity is checked by the verifier on the full 2×2 table.
- **Test.** The direction is perturbed worse. The exact conditional McNemar p-value is P(B ≤ b | b + c, ½). Concordant pairs do not enter. When b + c = 0 the p-value is 1, so the endpoint is not established.
- **Rule.** Established iff every replicate has p < α and δ_r ≤ −`effect_threshold`. This is an IUT at level α. `effect_threshold` (0.01) is used only here.
- **Interval.** A conditional exact interval for δ_r, given b + c, is reported for description only.

**Role** (decision `paired_endpoint_role`, pending):
- `secondary`: the outcome is the primary outcome. The paired result is reported beside it at its own level α. Citing both claims as confirmatory has familywise error ≤ 2α.
- `co_primary`: `SUPPORTED` additionally requires paired degradation. That is an IUT over all 2k component tests at α, so the joint level is α. Paired degradation that is not established gives `INCONCLUSIVE`, never `NOT_SUPPORTED`, because non-significance is not evidence of no degradation.

Trade-off: under `secondary`, a broken pipeline whose baseline is also at chance would still give primary `SUPPORTED`. The certificate shows this but the outcome does not change (`test_at_chance_without_paired_degradation_shows_the_endpoints_are_distinct`). I recommend `secondary` because the D4 question concerns the shuffled condition only. Choose `co_primary` if the outcome must stand alone.

### 0.5 The near-chance margin (`chance_margin`, PENDING)
The methodology fixes the qualitative expectation but not a number. I propose a **candidate of 0.005** (a fraction of accuracy, i.e. ±0.5 percentage points). It is **not approved**; the status stays `PENDING_HUMAN_APPROVAL`. The justification separates what can be derived from what is judgment:
- **Floor (design power, no outcome data).** With n_test = 10⁶, α = 0.05 and k = 5, δ ≥ 0.0025 gives P(all five EQUIVALENT | every p_r = 0.5) ≥ 0.996 by the union bound. δ = 0.002 gives ≥ 0.907, and δ ≤ 0.0015 is badly underpowered.
- **Ceiling (domain).** Gohr (CRYPTO 2019, Table 2) reports about 0.514 accuracy for the 8-round Speck32/64 distinguisher, which is used in published key recovery. An artifact of that strength must fall outside the band, so δ < 0.014. **This figure was recalled from the paper; it is not in the repository. Verify it before approval.**
- **Within (0.0025, 0.014) the choice is judgment.** 0.005 is a round value. It keeps an 8-round-strength artifact 0.009 (18 SE) outside the band and still declares equivalence with probability 0.9996 at a true p = 0.5025.

Any reproducible advantage is exploitable in principle, given enough data. The band therefore means "practically equivalent to chance for this pipeline control", not "cryptanalytically worthless".

Operating characteristics (exact; `python -m audit.dataset.tools.d4_near_chance_operating_characteristics`; the table is copied into the JSON):

| δ | P(fit EQ \| 0.5) | all-5 ≥ | P(EQ \| 0.5+δ/2) | P(EQ \| U) (size) | P(ABOVE \| 0.514) | P(ABOVE \| 0.5+2δ) |
|---|---|---|---|---|---|---|
| 0.001 | 0.277 | 0.000 | 0.172 | 0.041 | 1.000 | 0.282 |
| 0.0015 | 0.825 | 0.123 | 0.440 | 0.050 | 1.000 | 0.664 |
| 0.002 | 0.981 | 0.907 | 0.639 | 0.050 | 1.000 | 0.923 |
| 0.0025 | 0.9992 | 0.996 | 0.804 | 0.050 | 1.000 | 0.992 |
| 0.003 | 0.99999 | 0.9999 | 0.912 | 0.050 | 1.000 | 0.9997 |
| 0.005 | 1.000 | 1.000 | 0.9996 | 0.050 | 1.000 | 1.000 |
| 0.01 | 1.000 | 1.000 | 1.000 | 0.050 | 1.000 | 1.000 |

### 0.6 Verification and what it does not cover
- **Persisted.** The raw test predictions of every selected model (`fits/predictions/replicate_NN_<condition>.npy`, SHA-256 in the certificate). A resumed fit reuses an existing file only if its thresholded labels agree; otherwise the run is refused.
- **Recomputed by `verify`.** Every count and (b, c), from those files and the preflight-bound test labels. Then every p-value, interval, state, endpoint result and the outcome, from the integer counts, compared by deep equality with the certificate.
- **Refused by `verify`.** Any setting that differs from the approved design: margin, α, rule, role, `effect_threshold`, test n, or procedure (sidedness, aggregation, level).
- **Not covered.** `verify` does **not** rerun the network to confirm that the predictions are what the selected checkpoint outputs; it verifies the checkpoint and prediction digests only. Nor does it check the generator's sampling assumptions (D3).

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
| Primary (v3) | Exact binomial TOST per shuffled fit against [0.5 − δ, 0.5 + δ]; IUT over all k; Bonferroni α/(2k) departure tests | See §0.3. Equivalence is established positively; non-rejection of p = 0.5 is never used. |
| Paired test | Exact McNemar = binomial on discordant pairs (b, c), per replicate | Exact given independent test rows (independent `make_train_data` draws). Conditional on the fitted pair: it tests these two models, not the training procedure. |
| Sidedness | v1: two-sided; v2: one-sided (perturbed worse) | H1 is directional. v1 would also count an accuracy *increase* as an effect (`test_directional_rule_rejects_an_improvement_that_two_sided_rule_accepts`). |
| Multiplicity | v1: Bonferroni α/k and all k must pass; v2: IUT, each at α | Requiring all k to pass is an intersection-union test, so it has level α without Bonferroni. v1 is therefore doubly conservative. |
| Replicate CI | Percentile bootstrap of mean Δ over k replicates | With k = 5 there are only 126 distinct resamples, and percentile coverage is poor. v1 gates on it; v2 reports it as descriptive only. |
| Practical threshold | v1: abs(mean Δ) ≥ thr; v2: every Δ_r ≤ −thr | `effect_threshold` is a **fraction** (0.01 = 1 pp). The September engine docstring said "percentage points"; that is corrected here, and the value and its comparison are unchanged. |
| Self-reported values | Decision recomputed from (b, c) and accuracies at run and verify time | A certificate whose p-values, differences, row count or outcome disagree is refused. |
| Failures | Any exception aborts the run; no exclusions or replacements | A partial run is `NOT_COMPLETED`, never a result on fewer replicates. |
| Calibration | None proposed | The exact test needs no simulation for its level. The synthetic-adapter tests check the software only and are **not** a calibration. |

All three rules are implemented in the engine; this runner accepts only `near_chance_tost_v3` (§0). The rows above for v1 and v2 describe the paired comparison, which v3 keeps as its supporting endpoint (one-sided, IUT, as in v2).

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
| 26 | Primary endpoint tested a reduction relative to baseline, not near-chance accuracy | engine `near_chance_replicate`, `evaluate_near_chance(_decision)`, `evaluate_paired_degradation`, v4 certificate, `recompute_certificate_decision`; driver `check_target_design`, `check_decision_values`, `verify_run`, `recount_from_predictions` | Rule `near_chance_tost_v3` (§0); separate `chance_margin`; paired endpoint kept separately; predictions persisted and recounted | `test_d4_near_chance.py` (63 tests incl. exact rational reference, size, duality), `test_verify_refuses_a_certificate_that_departs_from_the_approved_design[7]`, `test_verify_without_a_result_is_not_completed` | PASS (code); `chance_margin`, `paired_endpoint_role` **BLOCKED** (human) |
| 18 | Threshold documented as percentage points but compared as a fraction | engine docs, certificate methodology | Documented as a fraction; value unchanged | — (documentation) | PASS |
| 19 | No D4 preregistration, approval gate or staged workflow | prereg JSON, engine `review_/require_approved_preregistration`, driver stages | Draft + gates; preflight/run refuse until approved, committed and anchored | `test_protocol_check_reports_pending_draft`, `test_preflight_refuses_the_unapproved_draft`, `test_anchored_record_*`, `test_shipped_gohr_draft_is_well_formed_and_entirely_pending` | PASS (code); approval **BLOCKED** (human) |
| 20 | Test data given to the training adapter | `GohrD4Adapter.__init__` | Refuses test arrays; test bound by digest only | `test_adapter_refuses_test_data` | PASS |
| 21 | `feature_shuffle`, `class_imbalance`, `metadata_removal` have no definition check | `perturbations/*.py` | Engine now refuses them (`verify` missing); not implemented | `test_perturbation_without_verify_is_refused` | NOT FIXED (out of D4 Gohr scope) |
| 23 | Persisted permutation not checked as a bijection or bound to the label digest, condition and seed | `integrity.verify_permutation_indices`, engine `save_/load_verified_perturbation`, `LabelShufflePerturbation.permutation_indices` | Indices persisted and verified (dtype, length, bounds, bijection, digest, regenerated plan); sidecar binds condition, seed, label digest | `test_persisted_indices_are_a_verified_int64_bijection`, `test_structurally_invalid_or_unplanned_indices_are_refused[6 cases]`, `test_indices_bound_to_training_labels`, `test_perturbation_reused_across_…[8 fields]` | PASS |
| 24 | Preflight dataset persistence not crash-safe; refusal written in two steps | `integrity.write_directory_exclusive`, `stale_staging_directories`; driver `stage_preflight`, `load_preflight` | fsync + atomic rename; stale staging refused; single refusal directory; exact file set | `test_failure_during_dataset_write_…`, `test_hard_crash_leftover_…`, `test_crash_during_rename_…`, `test_incomplete_or_altered_preflight_directory_is_refused[4]` | PASS |
| 25 | Experiment must train with the D4 adapter while the pinned base stays byte-identical | driver `make_adapter_factory`, `generate_gohr_partitions` | `GohrD4Adapter` trains; the pinned `GohrAdapter` only generates data | `test_experiment_trains_with_the_d4_adapter_…`, `test_completed_run_was_trained_by_the_d4_lifecycle`, `test_pinned_gohr_adapter_is_byte_identical`, `test_every_source_pinned_by_sealed_d0_d3_evidence_still_matches[2]` | PASS |
| 22 | AutoND staged 5→7 training hand-off | AutoND implementation-integrity workflow | Not touched: it is a separate workflow with no shared D4 component, and no AutoND D4 design exists | — | **BLOCKED**: needs an AutoND D4 protocol before any code |

## 7. Residual engineering notes

- Preflight holds the generated partitions and their `.npy` bytes in memory (about 2–3 GB peak at 10⁷ rows).
- `run` is resumable: fits recover through the adapter, and perturbations through their sidecars. `result/` is written once.
- The September result in `audit/dataset/evidence/d4/` stays untouched and historical. New runs never write there; the run directory must be new.
