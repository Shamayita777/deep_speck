# CE1 resumability — design rationale and gap analysis

Preserved as the rationale for the CE1 restart/resume implementation.

## EV is the reference for these six things only

persistent content-hashed datasets · reload-and-verify · config-hash-bound
resume · completed-unit skipping · append-only resume events · safe
process-level parallel execution.

## EV does NOT provide epoch-level resume

Verified in `framework/resumability.py` and `gohr/experiments.py`:
`RunState` carries `stage` / `completed_stages` / `config_hash` /
`checkpoint_hash` and **no epoch field**; the resume branches read
`if previous["stage"] == "complete": skip`, so an incomplete replicate is
**re-run from scratch**; `CONFIRMATORY_EVALUATION_PROTOCOL =
"final_epoch_weights"` with no `initial_epoch` and no optimizer-state
persistence. EV's resumable unit is the replicate — analogous to a CE1
block/arm, not to an epoch.

**CE1 epoch-level checkpoint/resume is therefore NEW ENGINEERING.** It is
not an EV port and is not justified by analogy to EV.

## What CE1 was missing

1. training/validation dataset persistence (generated from `os.urandom`,
   memory-only) — the decisive gap;
2. trainer wiring (single `fit()`, no per-epoch callback, no
   `initial_epoch`, no per-epoch checkpoint). Note: the legacy terminal
   `.keras` files DO contain optimizer state (Keras 3 stores it inside
   `model.weights.h5`); but the legacy run did not preserve the original
   training/validation arrays or per-epoch state, so exact
   continuation/reconstruction of the original training run is not provable;
3. an append-only ledger of resume events;
4. a validator for legacy (pre-resume) artifacts;
5. a per-GPU worker launcher.

## Resume semantics: two distinct claims

| claim | status |
|---|---|
| **checkpoint/state resume — infrastructure** | IMPLEMENTED. |
| **checkpoint/state resume — production code path** | SUPPORTED BY INTEGRATION TEST (`test_real_sigkill_interruption_restart_resume`: real SIGKILL, new process, real controller → GohrTrainer → `model.fit(initial_epoch=N)`; CPU, toy scale). Not yet exercised at production scale or on GPU. Restored: weights, optimizer state, epoch counter, LR-schedule position (a pure function of the epoch index under the frozen cycle-10 schedule) and the exact persisted datasets are restored. |
| **bit-exact trajectory-resumable** | **NOT CLAIMED.** Keras exposes no shuffle/dropout RNG state, so epochs N+1 onward do not follow the same minibatch stream an uninterrupted run would have. Every resumed arm records `shuffle_stream_continuity: false`. |

CE1's training behaviour is NOT altered to manufacture reproducibility:
`shuffle` is not forced to False and no deterministic input pipeline is
introduced. Either would be a protocol amendment requiring explicit
adjudication.

## Frozen scientific design — unchanged

CE-frozen-design-2026-03 · n_blocks 8 · min_valid 6 · tolerance 2 ·
seed_base 1000 · assignment_seed 20260101 with its prospectively frozen
randomized arm assignment · 10^7/10^6/10^6 · sealed shared evaluation set ·
FINAL_EPOCH 200 · exact_paired_sign_flip · paired-block structure.
EV's hypotheses, conditions and datasets are NOT imported.
