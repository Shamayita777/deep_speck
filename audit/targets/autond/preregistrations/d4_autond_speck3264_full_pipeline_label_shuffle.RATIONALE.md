# AutoND D4 label shuffle — design rationale, statistical review, execution and traceability

Companion to `d4_autond_speck3264_full_pipeline_label_shuffle.json`, protocol `d4-autond-speck3264-single-key-0x400000-staged-dbitnet-full-pipeline-label-shuffle-v2`, status **APPROVED**. The approver, approval timestamp, approval statement and pre-approval disclosure are recorded in the JSON; the approved values and the runbook are in `d4_autond_speck3264_full_pipeline_label_shuffle.APPROVAL_SHEET.md`.

## 0. Design overview
This protocol audits AutoND's **full staged DBitNet pipeline, stopping rule included**, under a controlled label perturbation:
- intact-label baselines run the pipeline's own stopping progression from round 5 (in the reference run it stops at the round-9 sentinel);
- label-shuffled fits run the identical loop and are expected to stop at their first round;
- the two arms are compared at the **starting round**, the one round every fit trains regardless of where the stopping rule ends it.

Version 2 replaces v1, which fixed rounds 5–8, removed the stopping rule and compared round-8 accuracy. v1 audited a narrower object than the pipeline AutoND actually runs and was retired before any run; v2 audits the pipeline as it is.

## 1. Sources and evidence
| Source | Used for | Not used for |
|---|---|---|
| Overarching audit methodology (`methodology_source.overarching_methodology`) | General statistical rules: exact tests, IUT/Bonferroni aggregation, pre-registration and outcome discipline | Any D4-specific margin, floor or criterion |
| Label-permutation literature: Ojala and Garriga, JMLR 2010; Zhang et al., ICLR 2017 (`methodology_source.d4_basis`) | The control itself: permuting training labels destroys the feature/label dependence while keeping features and label marginals, so held-out accuracy must fall to chance | The numbers in §5, which are this protocol's design choices |
| Pinned AutoND checkout `8d66e0a4` | The staged loop, constants, stopping rule, sampling and model, read directly in the code (§2) | — |
| 12 October staged-training bundle (`audit/targets/autond/evidence/training/`, SHA256SUMS verified) | Round list, stopping sentinel, reference accuracies: per-round best val_acc 0.919 / 0.783 / 0.611 / 0.510 / 0.5007 for rounds 5–9; round-8 test mean 0.5097; runtime 5,878 s for rounds 5–9 on Kaggle T4×2 (117.6 s per epoch) | Inference: no bundle value enters a test |
| `autond_ii_v2` (frozen, approved) | Published reference values (unpolished 0.5096, polished 0.5144) and the AutoND reproduction claim | D4 inference |

No AutoND D4 result existed when this protocol was approved; no setting was chosen from a D4 outcome.

## 2. The audited pipeline (read in the pinned code)
`train_nets.train_neural_distinguisher(model_name='dbitnet')` runs the following loop.

**Each round:**
1. fresh `NUM_SAMPLES = 10⁷` / `NUM_VAL_SAMPLES = 10⁶` data;
2. DBitNet (`make_model(64)`, `Adam(amsgrad=True)`, MSE, `acc`) is fitted for `EPOCHS = 10` epochs, batch 5000, with Keras-default `shuffle=True`;
3. `ModelCheckpoint(monitor='val_loss', save_best_only=True)` keeps the best epoch;
4. `val_acc = max(history['val_acc'])` is computed.

**Then:**
- if `val_acc <= ABORT_TRAINING_BELOW_ACC = 0.505`, the loop stops;
- otherwise the round becomes `best_round`, and the next round starts from this round's best-val_loss weights via `load_weights` on the same model object, so the optimizer state carries over.

**Two statistics, not one.** The stop uses the round's maximum validation **accuracy**; the carried and evaluated checkpoint is the minimum validation **loss** epoch. The adapter keeps them separate (`round_summary`: `max_val_acc` versus `selected_epoch`), and `test_stop_uses_the_round_max_val_acc_not_the_selected_checkpoints` enforces it.

**Declared departures** (`fixed_design.training_procedure`), each required for an auditable, verifiable run:
1. Data are generated once at preflight and persisted, with **one distinct partition per round** (matching upstream's fresh data per round), shared by the paired fits and all replicates.
2. TensorFlow is seeded per fit and per (round, starting epoch).
3. Every epoch checkpoint is kept and hash-chained.
4. A registered cap, `max_round = 9`, bounds a fit that never stops.

## 3. Estimands, contrast and isolation
- **Contrast.** Baseline and perturbed share the seed, rounds, features, validation partitions (true labels), procedure and stopping rule; only the training labels differ.
- **Shuffle.** One permutation over the round-major concatenation of every registered round's training rows, so no round carries genuine signal into the next.
- **Comparison round.** The **starting round**, the only round every fit trains. A shuffled pipeline stops there with `best_round = None`; its selected round-5 checkpoint is the one evaluated.
- **Confirmatory endpoints** (engine rule `near_chance_baseline_gated_v4`), both at the comparison round:
  - near-chance equivalence of every shuffled fit;
  - adequacy of every baseline fit.
- **Pipeline progression (co-primary, §5).** Per fit: rounds trained, per-round max val_acc, stop round and `best_round`, recomputed from the committed records. The registered criterion is that every shuffled fit stops at the starting round with no `best_round`, and every baseline fit passes the starting round. It deliberately does not require baselines to reach the round-9 sentinel: round 8 cleared the stop by 0.0054 in the reference run, so that requirement would test reproduction of the late-round frontier, not label dependence. Each baseline's stop round is reported.
- **Isolation.** Test data reach only `predict()` of reloaded selected checkpoints. Stopping and selection use validation data only. The tests enforce this by digest.

## 4. Sampling and statistics
- **Sampling.** `main.make_train_data` gives each row its own key and plaintext (`os.urandom`), `Y = urandom & 1`, and replaces every `Y == 0` row with random bits. Rows are i.i.d. and class counts are not fixed; both properties are tested against the pinned source.
- **Consequence.** Given a fitted model, its correct count on the separate test partition is exactly Binomial(n, p). The 2k fits share one test sample and are never pooled; every claim is per fit, combined by IUT or Bonferroni, neither of which needs independence.
- **Exact tests:**
  - near chance: TOST at α, with departure tested at α/(2k) per side;
  - baseline: one-sided Clopper–Pearson lower bound above t at α, with inadequacy tested at α/k.
- **Outcome precedence:** `NOT_COMPLETED` > `NOT_SUPPORTED` > `BASELINE_INADEQUATE` > `SUPPORTED` > `INCONCLUSIVE`; under co-primary progression, an engine `SUPPORTED` with the criterion unmet becomes `PROGRESSION_NOT_CONFIRMED`.
- **Paired McNemar** is secondary and never gates.

## 5. Decisions
Operating characteristics are design-only (exact binomial, α = 0.05, k = 5, n_test = 10⁶), computed by `python -m audit.dataset.tools.d4_near_chance_operating_characteristics --n 1000000 --deltas 0.0025 0.005 --floor 0.90`.

| Decision | Value | Basis |
|---|---|---|
| `decision_rule` | `near_chance_baseline_gated_v4` | SUPPORTED requires near-chance equivalence of every shuffled fit **and** adequacy of every baseline |
| `chance_margin` | **0.005**, band (0.495, 0.505) | The pipeline's own "has not learned" region (advantage ≤ 0.005 ⇔ val_acc ≤ 0.505), tested on held-out accuracy with an exact equivalence test. P(fit EQ \| 0.5) = 1 − 6.5·10⁻¹⁷; P(EQ \| 0.5025) = 0.9996; size 0.050. Rejected: 10 % of the round-5 advantage (0.042), far too permissive for a null-model check. |
| `baseline_adequacy_threshold` | **0.90** | Below. ADEQUATE iff ≥ 900494 correct of 10⁶; P(ADEQUATE \| 0.900) = 0.050, (\| 0.901) = 0.955, (\| 0.902) = 1 − 2·10⁻⁷, (\| 0.919) = 1.000. |
| `pipeline_progression_role` | **co_primary** | Below |
| `effect_threshold` (secondary) | 0.01 | About 2.4 % of the round-5 reference advantage; the expected Δ at chance is about −0.42 |
| `starting_round` | 5 | The starting round of the provenance-bound staged reference run |
| `max_round` | 9 | That run's stopping sentinel (best val_acc 0.5007 ≤ 0.505) |
| `epochs_per_round`, `sample_sizes` | 10; 10⁷ / 10⁶ per round; test 10⁶ | Upstream constants; the test set matches one upstream evaluation set and powers the margin |
| `replicates`, `alpha`, `paired_endpoint_role` | 5, 0.05, secondary | Same α as the framework's Gohr D4 protocol |
| Policies | refuse on any cross-partition feature row; record within-partition duplicates; report label conflicts separately; abort the whole run with no exclusions; no calibration needed (all tests exact) | |

**`baseline_adequacy_threshold` = 0.90.**
- **Why not 0.505.** The near-chance band ends at 0.505. A 0.505 floor would let SUPPORTED stand for a baseline at about 0.506 and a shuffled fit at about 0.504 — two near-chance models shown to be near chance. The secondary paired endpoint would not prevent it.
- **What 0.90 secures.** Every adequate baseline has an advantage of at least 0.40, so the band admits at most 0.005 / 0.40 = 1.25 % of the intact pipeline's advantage surviving the shuffle. This is the same criterion as the framework's Gohr D4 protocol (floor 0.90 at round 5, margin 0.005), so one D4 standard applies across targets.
- **What 0.90 is.** A tolerance of 1.9 points below the reference run's round-5 validation maximum, 0.919 (epoch 10, which is also that round's min-val_loss epoch and therefore the checkpoint this protocol's selection rule picks). It is an adequacy floor for the control, not a reproduction claim; AutoND reproduction is established separately (`autond_ii_v2`).
- **Committed in advance.** A baseline demonstrably below 0.90 gives `BASELINE_INADEQUATE`; one whose bound straddles 0.90 gives `INCONCLUSIVE`. Neither is a pass, and no baseline is re-run, replaced or excluded after its accuracy is seen.

**`pipeline_progression_role` = co_primary.** The audited object is the full pipeline, stopping rule included, so the claim covers the pipeline's own decision on the shuffled data, not only one round's accuracy.
- **Rule.** `autond_overall_outcome = SUPPORTED` only if the engine outcome is SUPPORTED **and** the registered criterion holds. An engine SUPPORTED with the criterion unmet becomes `PROGRESSION_NOT_CONFIRMED`, which is not a pass. Every other engine outcome passes through unchanged.
- **Error control.** The criterion is an additional requirement on the SUPPORTED event, so the joint claim keeps level α. The criterion itself is upstream's point rule on a validation maximum, so `PROGRESSION_NOT_CONFIRMED` is reported as a descriptive failure together with the per-round validation maxima.
- **Power cost.** A shuffled fit at chance exceeds 0.505 in any of its 10 round-5 validation epochs (n = 10⁶) with probability at most 7.5·10⁻²³. The criterion therefore removes SUPPORTED only when a shuffled fit learns enough for the pipeline itself to continue — exactly the behaviour the audit exists to expose.

## 6. Implementation and reuse
| Layer | Module | Status |
|---|---|---|
| Engine, rule, certificate, recomputation | `audit/dataset/d4_controlled_perturbation.py` | shared framework engine, unmodified |
| Integrity primitives | `audit/dataset/integrity.py` | shared, unmodified |
| Pinned binding and generator | `audit/dataset/adapters/autond.py` | shared, unmodified |
| Label shuffle | `audit/dataset/perturbations/label_shuffle.py` | shared, unmodified |
| Staged lifecycle, stopping rule, recovery | `audit/dataset/adapters/autond_d4.py` | AutoND D4 |
| Stages, verifier, locks | `audit/dataset/experiments/d4_autond_label_shuffle.py` | AutoND D4 |
| Two-GPU scheduler and workers | `audit/dataset/experiments/d4_autond_two_gpu.py` | AutoND D4 |

**Interface fit.** `run_d4` takes one training array, so the rounds are passed round-major and split at registered offsets. The engine evaluates each fit's `selection`, which is the comparison round's selected checkpoint.

## 7. Checkpointing and resume (each fit independently)
**Fit isolation.** Each (replicate, condition) fit has its own directory, `fits/replicate_NN/<condition>/`.

**Fit identity.** Every record and `state.json` carry the fit identity:
- seed, condition, replicate and `config_hash` (preregistration anchor, upstream commit and digests, rounds, epochs, sizes);
- training digests, overall and per round;
- validation digests and the test digest;
- the model description, including the upstream commit, rounds, epochs and stop threshold;
- `run_binding` (the preflight SHA-256, the all-partitions digest and the run config hash);
- the adapter identity (module, schemas, selection rule).

**Per epoch, in order:**
1. the Keras `.keras` checkpoint (weights and optimizer state);
2. an atomic record holding the checkpoint SHA-256, `val_loss`, `val_acc`, the step, the identity and the previous record's SHA-256;
3. an atomic `state.json` commit.

**On restart:**
- the state, every record, every checkpoint digest and the identity are verified first;
- a fit with N committed steps resumes at step N+1 from step N's checkpoint;
- committed steps are never retrained, and a finished fit is verified and never retrained;
- orphans are quarantined (moved, never deleted, never adopted), and suspicious states are refused with nothing moved;
- the stop is recomputed from committed `val_acc` only, and evidence that training continued past a stop, or stopped early, is refused;
- **round boundary:** the boundary model takes its optimizer state from the round's last committed checkpoint and its weights from the round's selected checkpoint, exactly as in an uninterrupted run.

**Training finished, evaluation not.** The single-owner `run` stage calls `adapter.train`, which only verifies and reloads finished fits, then evaluates them.

**Resumed trajectories.** The Keras shuffle order and other RNG state are not checkpointed, so a resumed fit is not bit-identical to an uninterrupted one; it re-seeds TensorFlow from (seed, round, starting epoch), and every quarantine-and-retrain is recorded.

## 8. Two-GPU execution (`d4_autond_two_gpu.py`)
**Coordinator.** One process holding `.d4_autond_scheduler.lock` (created exclusively, carrying a token, never broken automatically). It writes only `execution/<id>/`: the plan manifest, worker logs, worker summaries and the execution manifest, which records the scheduler SHA-256, GPU inventory, assignment, exit codes and finalisation. It never imports TensorFlow.

**Workers.**
- **Assignment.** Baseline (intact labels) on **GPU 0** and perturbed (shuffled) on **GPU 1**, concurrently, one fit at a time each.
- **GPU check before TensorFlow.** `CUDA_VISIBLE_DEVICES` is set in each child's environment before Python starts. The worker refuses if it differs from the assignment, or if TensorFlow was already imported.
- **GPU check after TensorFlow.** The worker requires exactly one visible GPU and a probe op placed on it.
- **Isolation.** Each worker holds `fits/.arm_<condition>.lock` and writes only its own arm's fit directories (and, for the perturbed arm, the perturbation sidecars), through the engine's own save and verify functions.

**Single owners.** Preflight and the final `run` and `verify` each have one owner. `run` refuses while another scheduler holds the lock or any arm lock exists, and takes `.d4_autond_run.lock`; workers refuse while it runs or once a result exists.

**Failure and restart.** One worker's failure blocks finalisation and leaves all committed evidence untouched. Re-running `execute` verifies every fit, skips finished fits, resumes partial ones and starts absent ones, with no duplicate work.

**Modes.** `inspect` is a read-only dry run; `execute two-gpu` refuses unless GPUs 0 and 1 exist; `execute sequential` is the explicit single-device mode (GPU 0, or `--cpu`) and is recorded as sequential. Each mode needs its own confirmation string. The execution manifest records the GPU inventory and device assignment of the run, which is the record of how it was executed.

**Runtime.** At the measured 117.6 s per epoch: about 50 epochs per baseline fit and 10 per shuffled fit, so 300 epochs (about 9.8 h) sequentially, or about 8.2 h on two GPUs, where the baseline arm's 250 epochs are the critical path. The upper bound is 500 epochs.

## 9. Verification
- **`verify`:**
  - recounts every count from the persisted predictions;
  - recomputes every engine statistic, the progression and `autond_decision`;
  - re-verifies every partition (including per-round digests), every perturbation, and every chain and checkpoint;
  - checks each fit's full identity;
  - refuses artifacts beyond a fit's end (past a stop or past the cap).
- **`--rerun-models`** reloads each selected comparison-round checkpoint and requires identical thresholded predictions.
- **Scope.** Verification does not re-train; the generator's assumptions are audited separately (D3).

## 10. Limitations
- **Blunt control.** Label shuffling detects a pipeline that learns without the feature/label dependence; it does not by itself exclude every form of leakage. Targeted generation-order and label-leakage screens are separate audits.
- **One comparison round.** Confirmatory accuracy is tested at the starting round; later rounds are characterised by their validation maxima through the progression criterion, with no later-round test partitions.
- **Point-rule progression.** The stopping rule is upstream's point comparison; as a co-primary criterion it can only remove SUPPORTED, never create it.
- **Seeds, not population.** "All k" concerns these five seeds.
- **Reference values.** The AutoND paper PDF is not archived in the repository; published values are cited through `autond_ii_v2`.
- **Memory.** Preflight holds about 5.6×10⁷ rows plus their bytes (several GB); `run` holds the 5×10⁷-row training array.
