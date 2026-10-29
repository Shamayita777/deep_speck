# AutoND D5 training-data scaling — design rationale

Companion to `d5_autond_speck3264_r5_training_data_scaling.json`, protocol `d5-autond-speck3264-single-key-0x400000-r5-dbitnet-training-data-scaling-v1`. Status: **APPROVED** (approver, timestamp, statement and pre-approval disclosure are in the JSON). No D5 data has been generated and no D5 training has run. The approved values and the commands are in the approval sheet.

## 0. What I am testing

I want to know how training-set size affects the held-out accuracy of AutoND's pinned DBitNet on SPECK32/64 (single-key, Δ = 0x400000) at **round 5**. I hold training to a **fixed protocol**:
- Adam(amsgrad), MSE, batch 5000;
- a fixed budget of 10 epochs;
- I evaluate the final-epoch model.

Within the framework this is not a new audit dimension. It is a target-specific use of the generic D5 engine under Dataset Integrity (methodology §3.6):
- I vary one dataset characteristic, size, and hold everything else fixed (Stages 4–5);
- I assess the result under §3.9 (Stage 6);
- it quantifies *Dataset Sensitivity* (§3.6.5).

**Fixed-epoch policy, so data and compute are confounded.** I train every size for the same number of epochs at batch 5000, so a size-n condition gets 10 · n / 5000 optimizer updates: 5,000 at 2.5 M and 20,000 at 10 M. What I estimate is accuracy as a function of training-set size *under this fixed-epoch policy*. I am not separating the effect of more data from the effect of more updates. A compute-matched design would be a different protocol with its own justification and approval.

This is **not** a full-pipeline experiment. I do not reproduce AutoND's staged loop, its validation data, its min-val_loss checkpoint or its stopping rule, because the engine fits without validation data. In my 12 October reference run, round 5's min-val_loss epoch was the final epoch (10 of 10), so there the final-epoch and upstream selections coincided. That is not guaranteed in general.

## 1. Design, and what the engine already provides

I use the engine `audit.dataset.d5_training_data_scaling`, schema 2.1, unmodified. It provides:

| Element | How the engine provides it |
|---|---|
| Replicates | One independent maximum-size dataset (10⁷ rows, os.urandom) per replicate; I run 12 |
| Size conditions | Nested prefixes of that replicate's dataset: 2.5, 5, 7.5 and 10 M |
| Initialisation | The same model seed for every size within a replicate; independent seeds across replicates |
| Evaluation | One fixed 10⁶-row test set shared by all conditions |
| Primary analysis | Paired t-test against the smallest size, Holm correction across the 3 contrasts, percentile bootstrap CIs |
| Secondary analysis | Replicate-wise slopes against log10 n (descriptive) |
| Integrity | File and logical SHA-256 for every dataset, prefix and the test set, re-verified on load and never regenerated |

My AutoND driver (`audit/targets/autond/d5/`) supplies the factories:
- **Data:** the pinned generator, through `AutoNDDatasetAdapter`. The commit and every import-graph digest are checked before import.
- **Model:** `tf.keras.utils.set_random_seed(seed)`, then the pinned `dbitnet.make_model(64)`, compiled exactly as `train_nets` compiles it.
- **Evaluation:** `model.evaluate` with batch 5000.

On top of the engine, the driver adds the safeguards in §3.

## 2. Statistics

- **Unit.** The replicate. Prefixes within a replicate are paired; test examples are not replicates. My inference is conditional on the one fixed test set.
- **Confirmatory test.** The engine's paired t-test with Holm correction at α = 0.05, applied to the three contrasts against the 2.5 M baseline.
- **SciPy is required.** Without SciPy the engine computes p-values by a normal approximation, which is *anti-conservative* at small k: at t = 2.5 with 11 degrees of freedom the t-test gives p = 0.0295 but the approximation gives 0.0124. The driver refuses to start, and verification refuses to pass, without SciPy.
- **Intervals.** I take the 95 % paired-t interval as the interval of record, because a percentile bootstrap under-covers at k = 12. The engine's bootstrap intervals are still reported.
- **Effect sizes.** For each contrast: the mean difference, that difference relative to the baseline, and Cohen's d_z.
- **Practical threshold.** I set `practical_effect_threshold` = **0.005** (half a percentage point of accuracy) with the rule **`t_interval_excludes_threshold_band`**. A contrast is practically meaningful only if it is Holm-significant *and* its whole 95 % paired-t interval lies beyond ±0.005. For scale, the binomial test-set SE at accuracy 0.92 is about 0.0003, so 0.005 is well above test-set noise. I chose the stricter of the two rules so that a point estimate just over the line does not count on its own.
- **Sensitivity and diagnostics.** An exact Wilcoxon signed-rank test (sensitivity to non-normality) and a Shapiro–Wilk diagnostic, which has low power at this k.
- **Classification.** Each contrast is `SIGNIFICANT_AND_PRACTICALLY_MEANINGFUL`, `SIGNIFICANT_BELOW_PRACTICAL_THRESHOLD` or `NOT_SIGNIFICANT_INCONCLUSIVE`. A non-significant contrast is inconclusive, never evidence of equivalence (§3.9.8, §3.9.9 Rule III).
- **Exploratory.** Test-loss contrasts and monotonicity counts. The engine's `test_loss` is Keras' compiled objective, MSE plus DBitNet's L2 penalties; the target re-evaluation reports the two terms separately.
- **Design resolution.** I have no pilot estimate of σ_d (the SD of paired differences). With k = 12, a two-sided paired t-test at 80 % power detects |Δ| ≥ 1.08 σ_d at Holm's strictest step, or 0.89 σ_d at α. Against my 0.005 threshold, a contrast of that size is reliably detectable only if σ_d is at most about 0.0046. If σ_d turns out larger, I will report null contrasts as inconclusive, not as "no effect".

## 3. Integrity safeguards added by the target

| Safeguard | What it does |
|---|---|
| Gates inside `execute()` | The approval, anchor, commit, run values, clean-source and pin checks are re-derived inside `execute()` itself, before the run directory is claimed or training starts. They do not depend on which entry point calls it. |
| No resume | The run directory must be new or empty, and is claimed with an exclusive `run_claim.json`. An interrupted run is preserved, reported and never resumed, so the engine's resume path is never used. |
| Binding | The claim records the preregistration anchor and commit, the source identity of the driver and engine files (all clean at HEAD), the upstream commit, the digests of the files actually bound, and the environment. |
| Provenance in verification | Verification re-reads the preregistration from git at the recorded commit. It must hash to the anchor and be approved with the run's values and pin. Every recorded source digest must be its committed content at the recorded HEAD, and that HEAD must contain the preregistration commit. |
| Data checks | Each partition must have shape (n, 64), dtype uint8 with values {0,1}, labels {0,1}, and the exact row count. Class counts are recorded descriptively. Train/test overlap refuses the run; within-partition duplicates are recorded. |
| Checkpoints | Each condition's final `latest.keras` SHA-256 is logged at evaluation time. Verification re-hashes it, reloads it, re-predicts the persisted test set, and compares within 1e-5 on accuracy and loss. |
| Certificate | The engine output and the target verification are kept in separate sections. The certificate is `VERIFIED` only if every condition, hash and check is present and consistent. |

This does not make the engine's resume implementation secure. The target simply never uses it.

## 4. Limitations

All of these are also listed in the JSON:
- **Data and compute confounded.** Larger sizes also get more optimizer updates, so I make no claim about data size independent of compute.
- **Fixed protocol, one round.** Results hold for round 5 under the fixed budget only, and are conditional on the fixed test set.
- **Not a law.** The scaling curve is descriptive and I do not extrapolate it.
- **No claim of learning.** A positive trend does not establish cryptographic learning.
- **No equivalence.** A null result does not establish equivalence.
- **Duplicates.** The exact-duplicate audits do not prove statistical independence, and cross-replicate duplicates are not audited.
- **Datasets cannot be regenerated.** They come from os.urandom; the persisted arrays are the reproducibility object.
- **GPU nondeterminism.** GPU training is not bit-deterministic.
- **Prior information.** Before approving, I had seen the round-5 reference accuracy of about 0.92 from an earlier run (disclosed in the JSON). No D5 outcome had been seen.
