# AutoND D5 training-data scaling — approval record and runbook

Protocol: `audit/targets/autond/preregistrations/d5_autond_speck3264_r5_training_data_scaling.json` (protocol id `…-r5-dbitnet-training-data-scaling-v1`). Rationale: `d5_autond_speck3264_r5_training_data_scaling.RATIONALE.md`. Findings: `audit/targets/autond/d5/D5_AUTOND_IMPLEMENTATION_FINDINGS.md`.

**Status: APPROVED** on 2026-10-29 (09:17:53 UTC). The approver, timestamp, statement and pre-approval disclosure are recorded in the JSON. Preregistration anchor (SHA-256 of the JSON): `62ef695989d01c301b47cc21aa9d0313d2975fe294037878f66c49f530b3da68`.

**What I am auditing:** how training-set size affects the final-epoch held-out accuracy of the pinned DBitNet at round 5 under a fixed 10-epoch budget. Data and compute are confounded by design (RATIONALE §0). This is not a full-pipeline experiment.

## A. Approved decisions

| Decision | Approved value | Basis (RATIONALE) |
|---|---|---|
| `training_sizes` | `[2500000, 5000000, 7500000, 10000000]` | nested prefixes; the smallest is the baseline |
| `replicates` | `12` | design resolution in §2 and JSON `statistical_plan` |
| `epochs` | `10` | upstream `EPOCHS` |
| `batch_size` | `5000` | upstream `BATCHSIZE` |
| `test_samples` | `1000000` | binomial SE about 0.0003 at 0.92 |
| `alpha` | `0.05` | the engine's fixed Holm level |
| `bootstrap_replicates` | `5000` | descriptive CIs |
| `audit_seed` | `0` | spawns replicate seeds; does not control data |
| `primary_analysis` | `engine_paired_t_holm_vs_smallest_size` | the engine's test is confirmatory |
| `practical_effect_threshold` | `0.005` | half a point of accuracy, well above test-set noise (§2) |
| `practical_effect_rule` | `t_interval_excludes_threshold_band` | the whole 95 % paired-t interval must lie beyond ±0.005 |
| `partition_overlap_policy` | `refuse_on_any_train_test_feature_row` | any shared row signals a generation defect |
| `within_partition_duplicates_policy` | `record_only` | does not cross the train/test boundary |
| `failure_handling` | `no_resume_new_run_directory_no_exclusions` | no replicate is excluded, replaced or resumed |
| `verification_tolerance` | `{"accuracy_abs": 1e-05, "loss_abs": 1e-05}` | 10 of 10⁶ test examples |

## B. Runbook (from the repository root)

```bash
PY=<python 3.11 with tensorflow 2.15.1 and scipy>
P=audit/targets/autond/preregistrations/d5_autond_speck3264_r5_training_data_scaling.json
ANCHOR=62ef695989d01c301b47cc21aa9d0313d2975fe294037878f66c49f530b3da68
AUTOND_ROOT=<AutoND checkout at 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779>
RUN=audit/dataset/evidence/d5_autond_speck3264_r5_$(git rev-parse --short=12 HEAD)

# 1. Protocol check (must exit 0) and the anchor (must equal $ANCHOR)
$PY -B -m audit.targets.autond.d5.training_data_scaling protocol-check --preregistration $P --require-approved
shasum -a 256 $P

# 2. Tests, in their own process (must pass with 0 skipped)
$PY -B -m pytest -q -p no:cacheprovider audit/targets/autond/tests/test_d5_autond_training_data_scaling.py

# 3. Production: a NEW run directory every time; never in the same session or on the same GPU as D4
$PY -B -m audit.targets.autond.d5.training_data_scaling run --preregistration $P --preregistration-sha256 $ANCHOR \
    --run-dir $RUN --autond-root $AUTOND_ROOT --confirm RUN_AUTOND_D5_PRODUCTION

# 4. Read-only re-check of the finished run
$PY -B -m audit.targets.autond.d5.training_data_scaling verify --run-dir $RUN
```

`run` refuses before creating anything unless the working tree's D5 driver, verifier, engine, `integrity.py`, adapter and pin are all committed and clean, and the preregistration is the committed, anchored, approved record above. The same checks run again inside `execute()`.

## C. Exit codes

| Code | Meaning |
|---|---|
| 0 | `VERIFIED` |
| 2 | Refused before anything was created |
| 3 | `NOT_VERIFIED` (the problems are listed in the certificate) |
| 4 | `NOT_COMPLETED`: `run_failure.json` is written and the run directory is preserved. A restart uses a **new** directory. |

## D. Resources

Planning estimates only, from my 117.6 s per epoch of 10 M rows on a T4 (the D4 figure):

- **Training per replicate:** 10 epochs × 25 M rows, about 49 min.
- **For k = 12:** about 9.8 h on one GPU. The engine runs conditions sequentially in a single process.
- **Disk:** about 8.5 GB in total (datasets about 7.8 GB, plus checkpoints).
