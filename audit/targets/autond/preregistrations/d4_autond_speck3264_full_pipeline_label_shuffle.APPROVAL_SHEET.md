# AutoND D4 label shuffle — approval record and runbook

Protocol: `audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.json` (protocol id `…-full-pipeline-label-shuffle-v2`). Rationale: `d4_autond_speck3264_full_pipeline_label_shuffle.RATIONALE.md`.

**Status: APPROVED.** The approver, timestamp, statement and pre-approval disclosure are recorded in the JSON. Preregistration anchor (SHA-256 of the JSON): `9bc14145739a6e7fc10f0a2dbb17a98fcd23b28b1bb635cd64c8ebffcad49f85`.

**What is audited:** AutoND's full staged DBitNet pipeline, stopping rule included. The arms are compared at the shared starting round. The scientific result is `autond_overall_outcome`; only `SUPPORTED` is a pass.

## A. Approved decisions

| Decision | Approved value | Basis (RATIONALE §5) |
|---|---|---|
| `decision_rule` | `near_chance_baseline_gated_v4` | near-chance shuffled fits **and** adequate baselines |
| `chance_margin` | `0.005` | the pipeline's own "has not learned" band, (0.495, 0.505) |
| `baseline_adequacy_threshold` | `0.90` | at most 1.25 % of the baseline advantage may survive the shuffle; same criterion as Gohr D4 |
| `pipeline_progression_role` | `co_primary` | the pipeline's stopping behaviour is part of the claim; keeps level α |
| `effect_threshold` | `0.01` | secondary paired endpoint only |
| `starting_round` | `5` | starting round of the staged reference run; the comparison round |
| `max_round` | `9` | the reference run's stopping sentinel |
| `epochs_per_round` | `10` | upstream `EPOCHS` |
| `sample_sizes` | `{"train_per_round": 10000000, "validation_per_round": 1000000, "test": 1000000}` | upstream sizes; test at the starting round |
| `replicates` | `5` | |
| `alpha` | `0.05` | |
| `paired_endpoint_role` | `secondary` | |
| `partition_overlap_policy` | `refuse_on_any_cross_partition_feature_row` | any shared row signals a generation defect |
| `within_partition_duplicates_policy` | `record_only` | does not cross the fit/selection/evaluation boundary |
| `label_conflict_reporting` | `report_separately_non_gating` | |
| `failure_handling` | `abort_whole_run_no_exclusions` | no replicate is excluded or replaced |
| `calibration` | `none_required` | every confirmatory test is exact |

## B. Runbook (from the repository root)

```bash
PY=<python 3.11 with tensorflow 2.15.1>
P=audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.json
ANCHOR=9bc14145739a6e7fc10f0a2dbb17a98fcd23b28b1bb635cd64c8ebffcad49f85
AUTOND_ROOT=<AutoND checkout at 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779>
RUN=audit/dataset/evidence/d4_autond_speck3264_$(git rev-parse --short=12 HEAD)

# 1. Protocol check (must exit 0)
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle protocol-check --preregistration $P --require-approved

# 2. Preflight: generates rounds 5-9 train/validation and the round-5 test partition; no training
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle preflight \
    --preregistration $P --preregistration-sha256 $ANCHOR --run-dir $RUN --autond-root $AUTOND_ROOT

# 3. Dry run (read-only): verified status of every fit, locks, GPUs, what each worker will do
$PY -B -m audit.dataset.experiments.d4_autond_two_gpu inspect \
    --run-dir $RUN --preregistration $P --preregistration-sha256 $ANCHOR --autond-root $AUTOND_ROOT

# 4. Production on two GPUs (baseline GPU 0, shuffled GPU 1), then run + verify --rerun-models
$PY -B -m audit.dataset.experiments.d4_autond_two_gpu execute two-gpu --confirm EXECUTE_TWO_GPU_D4 \
    --run-dir $RUN --preregistration $P --preregistration-sha256 $ANCHOR --autond-root $AUTOND_ROOT
#    single-device alternative, recorded as sequential:
#    ... d4_autond_two_gpu execute sequential --confirm EXECUTE_SEQUENTIAL_D4 ...

# 5. Independent verification (also run automatically at the end of step 4)
$PY -B -m audit.dataset.experiments.d4_autond_label_shuffle verify --run-dir $RUN --rerun-models
```

### What each stage enforces

- **Preflight refuses** if:
  - the preregistration is not approved, is malformed, has duplicate keys, is uncommitted or modified, or differs from the anchor;
  - the approval is dated after its commit;
  - any of the 8 source files is not clean at HEAD;
  - the checkout is not `8d66e0a4` with its pinned digests (checked **before** import), or its stop threshold is not 0.505;
  - `$RUN` exists.

  An overlap-policy refusal exits 4 and writes `$RUN.preflight_refused/`; the protocol then ends with the data kept.
- **`execute` refuses** if:
  - the confirmation string is missing;
  - in two-GPU mode, GPUs 0 and 1 are not reported;
  - another scheduler's lock exists (locks are never broken automatically; after a crash the lock file is inspected and removed by hand);
  - any fit's evidence is invalid.

  Re-running it after an interruption verifies every fit, skips finished fits, resumes partial fits at their next committed epoch and starts absent ones. A worker failure exits 2 with no result and all evidence preserved.
- **`run`** (the single owner of the result) refuses if a foreign scheduler lock or any arm lock exists, or if `$RUN/result` exists. Integrity refusals report `NOT_COMPLETED`.
- **`verify`:** exit 0 means integrity is verified; the printed **AutoND overall outcome** is the scientific result. Exit 5 means no result.

**Provenance:** `$RUN/execution/<id>/` holds `plan_manifest.json`, `<arm>_worker.log`, `<arm>_summary.json`, `run.log`, `verify.log` and `execution_manifest.json`: the scheduler SHA-256, GPU inventory and assignment, exit codes and finalisation.

**Resources:** 117.6 s per epoch on Kaggle T4×2 (measured in the 12 October reproduction). About 300 epochs in total, roughly 8.2 h on two GPUs (9.8 h sequentially); upper bound 500 epochs. Preflight needs several GB of memory.
