# AutoND D4 label shuffle — design rationale, statistical review, execution and traceability

Companion to `d4_autond_speck3264_full_pipeline_label_shuffle.json` (protocol `d4-autond-speck3264-single-key-0x400000-staged-dbitnet-full-pipeline-label-shuffle-v2`,
status **APPROVED**). The approver identity, approval timestamp, approval statement,
and pre-approval disclosure are recorded in the JSON and summarized in
`d4_autond_speck3264_full_pipeline_label_shuffle.APPROVAL_SHEET.md`.

No D4 preflight or production stage has been run under this protocol.

## 0. Why v2 replaces draft v1
Draft v1 fixed stages 5–8, removed the stopping rule, and compared round-8 accuracy. That design was a narrower estimand, chosen without the earlier full-pipeline design to hand. That design was not found in the repository or the session memory; v2 implements it as the protocol owner described it.

No concrete obstacle prevented implementing this.

## 1. The audited pipeline (read in the pinned code)
`train_nets.train_neural_distinguisher(model_name='dbitnet')` runs the following loop.

**Each round:**
1. fresh `NUM_SAMPLES = 10⁷` / `NUM_VAL_SAMPLES = 10⁶` data;
2. DBitNet (`make_model(64)`, `Adam(amsgrad=True)`, MSE, `acc`) is fitted for `EPOCHS = 10` epochs, batch 5000, with Keras-default `shuffle=True`;
3. `ModelCheckpoint(monitor='val_loss', save_best_only=True)` keeps the best epoch;
4. `val_acc = max(history['val_acc'])` is computed.

**Then:**
- if `val_acc <= ABORT_TRAINING_BELOW_ACC = 0.505`, the loop stops;
- otherwise the round becomes `best_round`, and the next round starts from this round's best-val_loss weights via `load_weights`, on the same model object, so the optimizer state carries over.

**Two statistics, not one.** The stop uses the round's maximum validation **accuracy**; the carried and evaluated checkpoint is the minimum validation **loss** epoch. The training bundle's README notes the same distinction. The adapter keeps them separate (`round_summary`: `max_val_acc` versus `selected_epoch`), and `test_stop_uses_the_round_max_val_acc_not_the_selected_checkpoints` checks it.

**Declared departures** (`fixed_design.training_procedure`):
1. Data are persisted once at preflight, with **one distinct partition per round** (as upstream's fresh data per round), shared by the paired fits and all replicates. The engine trains every replicate on one array.
2. TensorFlow is seeded.
3. Every epoch checkpoint is kept and hash-chained.
4. A registered cap `max_round` bounds a fit that never stops (candidate 9, the sentinel).

## 3. Estimands, contrast and isolation
- **Contrast.** Baseline and perturbed share the seed, rounds, features, validation partitions (true labels), procedure and stopping rule; only the training labels differ.
- **Shuffle.** One permutation over the round-major concatenation of every registered round's training rows, so no round carries genuine signal into the next.
- **Comparison round.** The **starting round**, the only round every fit trains. Under the stopping rule a shuffled pipeline is expected to stop there, with `best_round = None`. Its selected round-1 checkpoint is still the one evaluated.
- **Confirmatory endpoints** (engine rule `near_chance_baseline_gated_v4`, unchanged), both at the comparison round:
  - near-chance equivalence of every shuffled fit;
  - adequacy of every baseline fit.
- **Pipeline progression (co-primary, §5).** Per fit: rounds trained, per-round max val_acc, stop round, `best_round`, recomputed from committed records. Registered criterion: every shuffled fit stops at the starting round with no `best_round`, and every baseline fit passes the starting round. It does not require baselines to reach the round-9 sentinel: round 8 cleared the stop by only 0.0054 in the single reference run, so that would test reproduction of the late-round frontier, not label dependence.
- **Basis of the control.** Label permutation is the standard null control for a learning pipeline: it destroys the feature/label dependence while keeping the features and label marginals (Ojala and Garriga, JMLR 2010; Zhang et al., ICLR 2017). Applying it to AutoND, and every number in §5, are this protocol's design choices (`methodology_source.d4_basis`).
- **Isolation.** Test data reach only `predict()` of reloaded selected checkpoints. They are never used for training, stopping or selection: stopping and selection use validation only. Tests check this by digest.

## 4. Sampling and statistics
- **Sampling.** `main.make_train_data` gives each row its own key and plaintext (`os.urandom`), `Y = urandom & 1`, and every `Y == 0` row is replaced by random bits. Rows are i.i.d. and class counts are not fixed (both are tested against the pinned source).
- **Consequence.** Given a fitted model, its correct count on the separate test partition is exactly Binomial(n, p). The 2k fits share one test sample, so they are never pooled.
- **Exact tests:**
  - TOST at α, with departure at α/(2k) per side;
  - baseline: one-sided lower bound above t at α, and inadequacy at α/k.
- **Outcome rule:** `NOT_COMPLETED` > `NOT_SUPPORTED` > `BASELINE_INADEQUATE` > `SUPPORTED` > `INCONCLUSIVE`.
- **Paired McNemar** is secondary and never gates.

## 5. Decisions
The operating characteristics below are design-only (exact binomial, α = 0.05, k = 5, n_test = 10⁶), computed by `python -m audit.dataset.tools.d4_near_chance_operating_characteristics --n 1000000 --deltas 0.0025 0.005 --floor 0.90`.

| Decision | Value | Basis |
|---|---|---|
| `chance_margin` | **0.005**, band [0.495, 0.505] | The pipeline's own "has not learned" region (advantage ≤ 0.005 ⇔ val_acc ≤ 0.505), tested on test accuracy with an exact equivalence test. P(fit EQ \| 0.5) = 1 − 6.5·10⁻¹⁷; P(EQ \| 0.5025) = 0.9996; size 0.050. Alternatives: 0.0025; 10 % of the round-5 advantage (0.042), which is too permissive for a null-model check. |
| `baseline_adequacy_threshold` | **0.90** | See below. ADEQUATE iff ≥ 900494 correct of 10⁶; P(ADEQUATE \| 0.900) = 0.050, \| 0.901) = 0.955, \| 0.902) = 1 − 2·10⁻⁷, \| 0.919) = 1.000. |
| `effect_threshold` (secondary) | 0.01 | About 2.4 % of the round-5 reference advantage; the expected Δ at chance is about −0.42 |
| `starting_round` | 5 | The only provenance-bound staged run; the repo records disagreeing highest rounds (5 / 6 / 7) |
| `max_round` | 9 | The round-9 sentinel of that run |
| `epochs_per_round`, `sample_sizes` | 10; 10⁷ / 10⁶ per round; test 10⁶ | Upstream constants; the test set is the size of one upstream evaluation set and powers the margin |
| `replicates`, `alpha`, `paired_endpoint_role` | 5, 0.05, secondary | |
| **`pipeline_progression_role`** | **co_primary** | See below |
| Policies | as in the sheet | |

**`baseline_adequacy_threshold` = 0.90.**
- **Why not 0.505.** The near-chance band ends at 0.505. A 0.505 floor would let SUPPORTED stand for a baseline at about 0.506 and a shuffled fit at about 0.504: two near-chance models shown to be near chance. The secondary paired endpoint would not prevent it.
- **What 0.90 secures.** Every adequate baseline has an advantage of at least 0.40, so the band admits at most 0.005 / 0.40 = 1.25 % of the intact pipeline's advantage surviving the shuffle. This matches the approved Gohr D4 protocol (floor 0.90 at round 5, margin 0.005), so one D4 criterion applies across targets.
- **What 0.90 is not.** It is not a published AutoND test benchmark, and not a reproduction-fidelity claim (assessed separately, `autond_ii_v2`). It is a tolerance 1.9 points below the single reference run's round-5 *validation* maximum 0.919 (epoch 10, also that round's min-val_loss epoch). Round-5 test accuracy and seed spread are unmeasured, so the tolerance is a judgment, not an estimate.
- **Accepted in advance.** A baseline demonstrably below 0.90 gives `BASELINE_INADEQUATE`; one whose bound straddles 0.90 gives `INCONCLUSIVE`. Neither is a pass, and no baseline is re-run, replaced or excluded.

**`pipeline_progression_role` = co_primary.** The audited object is the full pipeline, stopping rule included, so the claim covers the pipeline's own decision on the shuffled data. The alternative, `reported_non_gating`, keeps the outcome purely error-controlled but leaves the stopping behaviour outside the claim.
- **Rule.** `autond_overall_outcome = SUPPORTED` only if the engine outcome is SUPPORTED **and** the registered criterion holds: every shuffled fit stops at the starting round with no `best_round`, and every baseline passes its starting round. An engine SUPPORTED with the criterion unmet becomes `PROGRESSION_NOT_CONFIRMED`, which is not a pass. Other engine outcomes pass through unchanged.
- **Error control.** An extra component can only shrink the SUPPORTED event, so co-primary keeps level α for the joint claim. But the progression component has no error control of its own (it is upstream's point rule on a validation maximum), so `PROGRESSION_NOT_CONFIRMED` is descriptive.
- **Power cost.** A shuffled fit at chance exceeds 0.505 in any of its 10 round-5 validation epochs (n = 10⁶) with probability at most 7.5·10⁻²³, so the criterion removes SUPPORTED only when a shuffled fit learns enough for the pipeline itself to continue.

## 6. Implementation and reuse
| Layer | Module | Status |
|---|---|---|
| Engine, rule, certificate, recomputation | `audit/dataset/d4_controlled_perturbation.py` | reused, **unmodified** |
| Integrity primitives | `audit/dataset/integrity.py` | reused, **unmodified** |
| Pinned binding and generator | `audit/dataset/adapters/autond.py` | reused, **unmodified** |
| Label shuffle | `audit/dataset/perturbations/label_shuffle.py` | reused, **unmodified** |
| Staged lifecycle, stopping rule, recovery | `audit/dataset/adapters/autond_d4.py` | **new** |
| Stages, verifier, locks, AutoND-local helpers | `audit/dataset/experiments/d4_autond_label_shuffle.py` | **new** |
| Two-GPU scheduler and workers | `audit/dataset/experiments/d4_autond_two_gpu.py` | **new** |

**Interface fit.** `run_d4` takes one training array, so the rounds are passed round-major and split at registered offsets. The engine evaluates each fit's `selection`, which is the comparison round's selected checkpoint.

## 7. Checkpointing and resume (each fit independently)
**Fit isolation.** Each (replicate, condition) fit has its own directory, `fits/replicate_NN/<condition>/`.

**Fit identity.** Every record and `state.json` carry the fit identity:
- seed, condition, replicate and `config_hash` (the design hash: preregistration anchor, upstream commit and digests, rounds, epochs, sizes);
- training digests, overall and per round;
- validation digests;
- the test digest;
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
- committed steps are never retrained, and a finished fit is verified and **never retrained** (the test patches `Model.fit` to fail);
- orphans are quarantined (moved, never deleted, never adopted), and suspicious states are refused with nothing moved;
- the stop is recomputed from committed `val_acc` only, and evidence that training continued past a stop, or stopped early, is refused;
- **round boundary:** the boundary model takes its optimizer state from the round's last committed checkpoint and its weights from the round's selected checkpoint, exactly as an uninterrupted run (tested with real checkpoints whose weights and optimizer iteration counts differ).

**Training finished, evaluation not.** The single-owner `run` stage calls `adapter.train`, which only verifies and reloads finished fits, then evaluates them.

**Not guaranteed: bit-identical trajectories.** The shuffle order and other RNG state are not checkpointed. A resumed fit re-seeds TensorFlow from (seed, round, starting epoch).

## 8. Two-GPU execution (`d4_autond_two_gpu.py`)
**Coordinator.** One process, holding `.d4_autond_scheduler.lock` (created exclusively, carrying a token, never broken automatically). It writes only `execution/<id>/`: the plan manifest, worker logs, worker summaries and the execution manifest, recording the scheduler SHA-256, GPU inventory, assignment, exit codes and finalisation. It never imports TensorFlow (tested).

**Workers.**
- **Assignment.** Baseline (intact labels) on **GPU 0** and perturbed (shuffled) on **GPU 1**, run concurrently, one fit at a time each.
- **GPU check before TensorFlow.** `CUDA_VISIBLE_DEVICES` is set in each child's environment before Python starts. The worker refuses if it differs from the assignment, or if TensorFlow was already imported (tested).
- **GPU check after TensorFlow.** It then requires exactly one visible GPU and that a probe op is placed on it.
- **Isolation.** Each worker holds `fits/.arm_<condition>.lock` and writes only its own arm's fit directories, plus, for the perturbed arm, the perturbation sidecars, through the engine's own save and verify functions.

**Single owners.** Preflight and the final `run` and `verify` each have one owner.
- `run` refuses while another scheduler holds the lock, or while any arm lock exists.
- It takes `.d4_autond_run.lock`.
- Workers refuse while it runs or once a result exists.

**Failure and restart.**
- One worker's failure blocks finalisation and leaves all committed evidence untouched (tested).
- Re-running `execute` verifies every fit, skips finished fits, resumes partial ones and starts absent ones (tested; no duplicate work).

**Modes.**
- `inspect` is a read-only dry run (tested by comparing every file's digest and mtime before and after).
- `execute two-gpu` refuses unless GPUs 0 and 1 exist.
- `execute sequential` is an explicit fallback (GPU 0, or `--cpu`), recorded as sequential.
- Each mode needs its own confirmation string.

**Not yet confirmed.** The manifest records `physical_two_gpu_confirmed: false`. A physical dual-T4 run has not been tested; it requires Kaggle. The two-GPU launch is tested synthetically, observing concurrency and `CUDA_VISIBLE_DEVICES`.

**Planning estimate** (from the owner-supplied 117.6 s/epoch):
- **Expected:** about 50 epochs per baseline fit and about 10 per shuffled fit, so 5 × 60 epochs, about 9.8 h if run sequentially.
- **Two-GPU, expected:** about the baseline arm's 250 epochs, about 8.2 h.
- **Upper bound:** 500 epochs.

This is not a guarantee.

## 9. Verification and its limits
- **`verify`:**
  - recounts every count from the persisted predictions;
  - recomputes every engine statistic, the progression and `autond_decision`;
  - re-verifies every partition (including per-round digests), every perturbation and every chain and checkpoint;
  - checks each fit's full identity;
  - refuses artifacts after a fit's end (past a stop or past the cap).
- **`--rerun-models`.** Reloads each selected comparison-round checkpoint and requires identical thresholded predictions.
- **Not covered.** Re-training, and the generator's assumptions (D3).

## 10. Limitations
- **Blunt control.** Label shuffling is blunt; the targeted screens are not performed here.
- **No later-round test.** Later-round accuracy is not tested; there are no later-round test partitions.
- **Stopping without error control.** The stopping rule is upstream's point rule, with no error control. As a co-primary criterion it can only remove SUPPORTED, never create it.
- **Seeds, not population.** "All k" concerns these seeds.
- **Unverified PDF digest.** The paper PDF digest is UNVERIFIED in this repository.
- **Memory.** Preflight holds about 5.6×10⁷ rows plus their bytes, several GB. Run holds the 5×10⁷-row training array.
