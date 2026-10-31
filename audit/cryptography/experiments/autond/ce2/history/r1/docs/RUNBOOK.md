# AutoND CE2 — runbook

All commands run from the repository root (the directory containing `audit/`). On Kaggle, use one
foreground cell per step. No step trains a model.

## 1. Environment

The pinned stack is CPython 3.11 with tensorflow 2.15.1, keras 2.15.0, numpy 1.26.4, pandas 2.2.2,
scipy 1.11.4 and h5py 3.10.0, plus pytest for the tests. Production refuses any other version of
tensorflow, keras, numpy, pandas or scipy. Smoke records the deviation and continues.

```bash
uv venv --python 3.11 /tmp/ce2venv
uv pip install --python /tmp/ce2venv/bin/python "tensorflow[and-cuda]==2.15.1" keras==2.15.0 \
  tensorflow-estimator==2.15.0 numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 h5py==3.10.0 \
  ml-dtypes==0.3.2 protobuf==4.25.9 pytest==8.3.3
PY=/tmp/ce2venv/bin/python
CLI="$PY -B -m audit.cryptography.kaggle.autond_ce2_kaggle"
```

The pinned upstream is read from `audit/cryptography/autond_ce1/AutoND-upstream`. Set
`AUTOND_UPSTREAM` or pass `--upstream` to read it from elsewhere. Its hashes are verified before
import.

## 2. Checks (non-evidentiary)

```bash
$PY -B -m pytest -q -p no:cacheprovider audit/cryptography/tests/test_autond_ce2.py
$CLI --check-model                      # strict load, architecture, inference mode
$CLI --validate-math                    # gates C, D (exhaustive to word size 8, 16-bit literal), E, F (~5 min)
$CLI --preflight                        # production preflight, including the runtime gate
$CLI --smoke --out /kaggle/working      # creates /kaggle/working/autond_ce2_smoke_<utc>/, then verifies it
```

Smoke output is labelled `SMOKE_NOT_CONFIRMATORY`. It uses 3 runs × 4,096 pairs and is never
merged with production. Its decision is expected to be `INCONCLUSIVE`, because 3 runs cannot reach
p ≤ 0.05.

## 3. Freezing and approval (before any production)

1. Review `README.md`, `POWER_ANALYSIS.md`, `VERIFICATION_REPORT.md` and `preregistration.json`.
2. In `frozen_design_autond_ce2.py`, set the final run sizes and change both
   `PROPOSED_PENDING_APPROVAL` markers (`runs.status`, `statistics.power.status`) to `FROZEN`. If
   any other value changes, increase `DESIGN_VERSION`.
3. Regenerate `preregistration.json` with `render_preregistration()`, which gives a new
   `design_sha256`.
4. Record the approval in `preregistration.json`:
   - `approval.status = "APPROVED"`;
   - `approved_design_sha256` = the new hash;
   - `approved_by`;
   - `approved_utc`.
5. Commit. From this point the preregistration is never edited. Any change needs a new design
   version and a new approval.

`--run-all` refuses production while any marker is pending or the approval does not match the hash.

## 4. Production

```bash
OUT=/kaggle/working/autond_ce2_production_<utc>
$CLI --gpu 0 --run-all --out $OUT       # preflight, twin, behavioural binding, then every run
$CLI --gpu 0 --resume --out $OUT        # only after an interruption; terminal runs are never redone
$CLI --finalize --out $OUT              # certificate.json, once
$CLI --verify --out $OUT                # independent verification -> verification_<utc>.json
```

- **Initialisation:** a failed preflight or behavioural gate stops initialisation. `--finalize` then
  records `PRECHECK_FAILED`.
- **Interruption:** a run continues from its committed artifacts. Committed data are never
  regenerated, and outputs written but never recorded are set aside, not analysed.
- **Across Kaggle sessions:** attach the previous output as a dataset, copy it back to the same path,
  and run `--resume`.
- **Archiving:** keep the complete output directory: data, outputs, records, journal, twin weights
  and behavioural set.
