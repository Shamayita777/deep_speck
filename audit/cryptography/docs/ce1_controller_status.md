# CE1 resumable controller — status and operation

Production path: `experiments/ce1/controller.py` (entry point
`python -m audit.cryptography.experiments.ce1.gohr_signal_destruction`).
The closure path (`build_production_components`, `data_fn.last_validation`)
has been removed; `run(production=True)` is retired and refuses after the
frozen-parameter guards.

## Claims (kept separate)

| Claim | Status | Evidence |
|---|---|---|
| Infrastructure implemented | YES | controller, BlockContext, dataset commit, ledger, resume state schema 2 |
| Production trainer wired | YES | `train_arm` → `GohrTrainer.train(initial_epoch, terminal_epoch, extra_callbacks=[CE1EpochCheckpoint])` → `model.fit`; CLI routes only through the controller |
| Checkpoint/state resume — infrastructure | IMPLEMENTED | |
| Checkpoint/state resume — production code path | SUPPORTED BY INTEGRATION TEST | `test_real_sigkill_interruption_restart_resume`: real SIGKILL, new process, depth-10 GohrModel, resume with `initial_epoch=3`, optimizer iterations + LR verified. CPU, toy scale; **not yet exercised at 10^7 samples or on GPU** |
| Bit-exact trajectory resume | NOT CLAIMED | `shuffle_stream_continuity = false` on every resumed arm; shuffle not forced off |
| Controller SKIP/RESUME tested | YES | `test_controller_skip_evaluated_and_resume_at_138`, `test_training_complete_is_evaluated_not_retrained` |
| 2-GPU worker implemented | YES | `worker.py`; GPU visibility set in the child env before TF import; mode from detected GPUs |
| Worker process isolation / serial–parallel equivalence | TESTED ON CPU | `test_parallel_workers_isolated_and_equivalent_to_serial` (two concurrent CPU workers; parent never imports TF) |
| Physical 2-GPU execution | NOT TESTED | no GPU in the test environment |

## Block resolution (predeclared, outcome-blind; rules v2)

Recorded in every run manifest (`block_resolution_rules`) before any
training; a run refuses to reopen under different rules or an incompatible
`controller_protocol_version`. Resolution is DERIVED from artifacts/state on
every pass; no function of any evaluation result is ever used.

* Frozen counting rule (authoritative): 8 blocks, >= 6 VALID, <= 2 non-valid,
  analyse ALL valid blocks, no early stop.
* **VALID** — both arms EVALUATED and all six invariants evidenced:
  evaluation_bound, terminal_epoch (optimizer steps read from the terminal
  file + complete per-epoch record), assignment, shared_training_inputs,
  label_construction, shared_validation.
* **NOT_COUNTED** (final gate only; permanent evidence verdicts on completed
  blocks): N-INSUFFICIENT-EVIDENCE (invariant permanently unverifiable, e.g.
  legacy block0), N-FINAL-EPOCH (evaluated model is provably not the
  terminal model), N-PAIRING (arms provably not paired as frozen).
* **NEEDS_OPERATOR** (unresolved — never a failure, never consumes a slot):
  operational retry caps (MAX_ARM_FAILURES = 3, MAX_ARM_RERUNS = 2),
  evaluation-integrity failures, missing/altered terminal model, state file
  or committed dataset, identity mismatch, pairing guard.
* **PENDING** — normal progress. `finalize` refuses while any block is
  PENDING / NEEDS_OPERATOR / legacy-pending.
* Interruptions (SIGKILL / KeyboardInterrupt / SystemExit) never count.
* Operator actions (ledger-recorded, reason required): GRANT_RETRY,
  REEVALUATE (only after an integrity failure; must agree with every earlier
  ledger evaluation of the same model), RESTART_BLOCK (only if no arm of the
  block was ever evaluated). No action can fail, exclude or select a block.

### Evaluation binding

`sealed_predictions.npz` (sha256) ← `evaluation.json` (canonical sha256,
accuracy = n_correct / n_samples exactly, pred > 0.5 as in Keras) ← ledger
`EVALUATED` event carrying the same hashes, counts, model hash, sealed hash
and config fingerprint. Every gate pass recounts n_correct from the
predictions against the sealed labels. Any divergence pauses the arm.

### Path containment

Every recorded dataset/artifact path is validated on LOAD: absolute paths,
`..` components and symlink escapes outside the run (or data) directory are
refused.

## Working directory

Launch every production command from the **repository root** (the directory
that contains `audit/cryptography/`). With any other working directory the
output guard raises `WrongWorkingDirectoryError` before anything is written:
outputs are never redirected elsewhere. An explicit `--repo-root` is a
visible override (used by the tests for scratch directories).

## Operating sequence (do not run until authorised)

```
M=python -m audit.cryptography.experiments.ce1.gohr_signal_destruction
$M --verify-legacy-block0                  # read-only, no accuracy
$M --recover-legacy-block0                 # creates run dir, records RECOVER, seeds block0
$M --status                                # read-only plan
$M --execute                               # one restartable pass; repeat after interruptions
$M --operator-action GRANT_RETRY --block N --arm A --reason '...'   # only if NEEDS_OPERATOR
$M --finalize                              # final gate + analysis of ALL valid blocks
```

Disk: each block persists ~0.72 GB of training/validation data
(10^7 × 64 uint8 X_train). Keep the whole run directory between sessions;
paths inside it are relative, so it can be moved between hosts.
