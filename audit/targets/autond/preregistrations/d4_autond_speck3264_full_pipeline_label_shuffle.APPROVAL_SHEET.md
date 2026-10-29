# AutoND D4 label shuffle — consolidated approval sheet and runbook

Protocol: `audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.json` (protocol id `…-full-pipeline-label-shuffle-v2`). Rationale: `d4_autond_speck3264_full_pipeline_label_shuffle.RATIONALE.md`.

**Status: APPROVED** (approver, timestamp and statement are in the JSON). No AutoND D4 preflight or production stage has been run. `protocol-check --require-approved` must exit 0.

**What is audited:** AutoND's full staged DBitNet pipeline, stopping rule included. The arms are compared at the shared starting round. The scientific result is `autond_overall_outcome`, and only `SUPPORTED` is a pass.

## A. Decisions to make

For each decision, set `"value"` and change `"status"` to `"APPROVED"` in `decisions.<name>`. Choose only from `options` where they are listed.

| Decision | Candidate | Note |
|---|---|---|
| `decision_rule` | `near_chance_baseline_gated_v4` | single option |
| `chance_margin` | `0.005` | **Your scientific decision.** The pipeline's own "has not learned" band; alternative 0.0025. Keep `"units": "accuracy_fraction"`. |
| `baseline_adequacy_threshold` | `0.90` | Gives the near-chance band meaning (≤ 1.25 % of the baseline advantage may survive); same criterion as Gohr D4; RATIONALE §5 |
| `pipeline_progression_role` | `co_primary` | The stopping behaviour is part of the claim; keeps level α (RATIONALE §5) |
| `effect_threshold` | `0.01` | secondary endpoint only |
| `starting_round` | `5` | **Your decision** (the repo records disagreeing highest rounds) |
| `max_round` | `9` | the round-9 stopping sentinel |
| `epochs_per_round` | `10` | upstream `EPOCHS` |
| `sample_sizes` | `{"train_per_round": 10000000, "validation_per_round": 1000000, "test": 1000000}` | upstream sizes; test at the starting round |
| `replicates` | `5` | |
| `alpha` | `0.05` | |
| `paired_endpoint_role` | `secondary` | single option |
| `partition_overlap_policy` | `refuse_on_any_cross_partition_feature_row` | or `record_only` |
| `within_partition_duplicates_policy` | `record_only` | or `refuse_on_any_within_partition_duplicate` |
| `label_conflict_reporting` | `report_separately_non_gating` | single option |
| `failure_handling` | `abort_whole_run_no_exclusions` | single option |
| `calibration` | `none_required` | single option |

**Top-level fields** (yours alone):

| Field | What to write |
|---|---|
| `approved` | `true` |
| `approved_by` | your approver identifier |
| `approved_utc` | `YYYY-MM-DDTHH:MM:SSZ`, at or before the commit time |
| `approval_statement` | what you approve, including the declared departures from upstream (RATIONALE §2) |
| `data_examined_before_approval` | at least the 12 October bundle results (round-by-round val_acc; round-8 test mean 0.5097), the AutoND II and D0–D3 results, and the design-only operating characteristics. No AutoND D4 result exists. |
| `status` | e.g. `APPROVED` (informational) |

## B. Commands (from `~/Documents/deep_speck`)

```bash
PY=~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python
P=audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.json

# 0. Draft review now (exit 0; pending fields listed; duplicate JSON keys are checked)
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle protocol-check --preregistration $P

# 1. After you approve every field: must exit 0
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle protocol-check --preregistration $P --require-approved

# 2. Commit the protocol, companions and every AutoND D4 source (preflight requires them clean at HEAD)
git add -- $P audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.RATIONALE.md \
    audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.APPROVAL_SHEET.md \
    audit/dataset/adapters/autond_d4.py audit/dataset/experiments/d4_autond_label_shuffle.py \
    audit/dataset/experiments/d4_autond_two_gpu.py \
    audit/dataset/tests/test_d4_autond_label_shuffle.py audit/dataset/tests/test_d4_autond_two_gpu.py
git commit -m "Approve AutoND D4 label-shuffle preregistration"

# 3. Anchor: record OUTSIDE the file
shasum -a 256 $P

# 4. USER-RUN GATE: preflight (generates rounds 5-9 train/validation and the round-5 test partition; no training)
RUN=audit/dataset/evidence/d4_autond_speck3264_$(git rev-parse --short=12 HEAD)
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle preflight \
    --preregistration $P --preregistration-sha256 <anchor> --run-dir $RUN --autond-root <AutoND checkout>

# 5. Dry run (read-only): verified status of every fit, locks, GPUs, what each worker would do
$PY -B -m audit.dataset.experiments.d4_autond_two_gpu inspect \
    --run-dir $RUN --preregistration $P --preregistration-sha256 <anchor> --autond-root <AutoND checkout>

# 6. USER-RUN GATE: production, two GPUs (baseline on GPU 0, shuffled on GPU 1; then run + verify --rerun-models)
$PY -B -m audit.dataset.experiments.d4_autond_two_gpu execute two-gpu --confirm EXECUTE_TWO_GPU_D4 \
    --run-dir $RUN --preregistration $P --preregistration-sha256 <anchor> --autond-root <AutoND checkout>
#    or, explicitly, one GPU / sequential (recorded as sequential, never as two-GPU):
#    ... execute sequential --confirm EXECUTE_SEQUENTIAL_D4 ...
#    or, equivalently for one GPU, the plain single-owner stage:  ... d4_autond_label_shuffle run ...

# 7. Independent verification (also run automatically at the end of step 6)
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle verify --run-dir $RUN --rerun-models
```

### What each gate does

- **Preflight refuses** if:
  - the preregistration is pending, malformed, has duplicate keys, is uncommitted or modified, or differs from the anchor;
  - the approval is dated after its commit;
  - any of the 8 source files is not clean at HEAD;
  - the checkout is not `8d66e0a4` with its pinned digests (checked **before** import), or its stop threshold is not 0.505;
  - `$RUN` exists.

  An overlap-policy refusal exits 4 and writes `$RUN.preflight_refused/`.
- **`execute` refuses** if:
  - the confirmation string is missing;
  - in two-GPU mode, GPUs 0 and 1 are not reported;
  - another scheduler's lock exists (locks are never broken automatically: after a crash, inspect the lock file and remove it by hand);
  - any fit's evidence is invalid.

  Rerunning it after an interruption verifies every fit, skips finished fits, resumes partial fits at their next committed epoch, and starts absent ones. A worker failure exits 2 with no result and all evidence preserved.
- **`run`** (the single owner of the result) refuses if a foreign scheduler lock or any arm lock exists, or if `$RUN/result` exists. Integrity refusals report `NOT_COMPLETED`.
- **`verify`:** exit 0 means integrity is verified. The printed **AutoND overall outcome** is the scientific result. Exit 5 means no result.

**Provenance:** `$RUN/execution/<id>/` holds `plan_manifest.json`, `<arm>_worker.log`, `<arm>_summary.json`, `run.log`, `verify.log` and `execution_manifest.json`. That covers the scheduler SHA-256, GPU inventory and assignment, exit codes, and `physical_two_gpu_confirmed: false` until a physical dual-T4 run is confirmed by you.

**Resources (planning, not a guarantee):**
- **Feasibility figure:** 117.6 s/epoch on Kaggle T4x2, supplied by you and not in the repository.
- **Expected:** about 300 epochs sequentially (about 9.8 h), or about 8.2 h with two GPUs.
- **Upper bound:** 500 epochs.
- **Memory:** preflight needs several GB.
