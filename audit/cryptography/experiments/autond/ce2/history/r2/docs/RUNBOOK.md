# AutoND CE2: runbook (design r2)

Run every command from the repository root, the directory that contains `audit/`. On Kaggle, use one
foreground cell per step. No step trains a model. Always run Python with `-B`.

## 1. Environment

Production requires the pinned stack: CPython 3.11, tensorflow 2.15.1, keras 2.15.0, numpy 1.26.4,
pandas 2.2.2, scipy 1.11.4 and h5py 3.10.0. It refuses any other version of tensorflow, keras,
numpy, pandas or scipy. Smoke runs record the deviation and continue.

```bash
uv venv --python 3.11 /tmp/ce2venv
uv pip install --python /tmp/ce2venv/bin/python "tensorflow[and-cuda]==2.15.1" keras==2.15.0 \
  tensorflow-estimator==2.15.0 numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 h5py==3.10.0 \
  ml-dtypes==0.3.2 protobuf==4.25.9 pytest==8.3.3
PY=/tmp/ce2venv/bin/python
CLI="$PY -B -m audit.cryptography.kaggle.autond_ce2_kaggle"
```

The single pinned upstream checkout (commit `8d66e0a4...`) is read from
`audit/cryptography/autond_ce1/AutoND-upstream`. Set `AUTOND_UPSTREAM` or pass `--upstream` to read
it from elsewhere. Every file is hash-verified before import. The import is isolated: the root is
removed from `sys.path`, and the upstream modules are removed from `sys.modules` afterwards.

Record `TF_DETERMINISTIC_OPS` and the GPU model. Both go into `preflight.json`.

## 2. Checks (non-evidentiary)

```bash
$PY -B -m pytest -q -p no:cacheprovider audit/cryptography/experiments/autond/ce2/tests audit/cryptography/core/tests
$CLI --check-model                      # strict load, architecture, inference mode
$CLI --validate-math                    # gates C, D (exhaustive to word size 8, 16-bit literal), E, F (~5 min)
$CLI --power --report <new.json>        # scenario-based sensitivity analysis + pipeline-faithful check
$CLI --preflight                        # production preflight, including the runtime gate
$CLI --smoke --out /kaggle/working      # creates /kaggle/working/autond_ce2_smoke_<utc>/, then verifies it
```

Smoke output is labelled `SMOKE_NOT_CONFIRMATORY`. It uses 3 runs of 4,096 pairs and is never merged
with production. Its decision is `INCONCLUSIVE` by construction, because 3 runs cannot reach p <= 0.05.

## 3. Freezing and approval (before any production)

1. Review `README.md`, `POWER_ANALYSIS.md`, `VERIFICATION_REPORT.md` and
   `preregistration/preregistration.json`. Also review the r1 to r2 changes in `design.py`
   (`DESIGN["revision"]`).
2. In `design.py`, change both `PROPOSED_PENDING_APPROVAL` markers (`runs.status`,
   `statistics.power.status`) to `FROZEN`. If any other value changes, increase `DESIGN_VERSION` and
   record the change in `DESIGN["revision"]`.
3. Keep the current `preregistration/preregistration.json` under `history/`. Then regenerate it:
   `json.dumps(render_preregistration(), indent=1, sort_keys=True) + "\n"`. This gives a new
   `design_sha256`.
4. Record the approval in the regenerated file, yourself:
   - `approval.status = "APPROVED"`;
   - `approved_design_sha256` = the new hash;
   - `approved_by`;
   - `approved_utc`.
5. Commit. From this point the preregistration is never edited. Any change needs a new design version
   and a new approval.

`--run-all` refuses production while any marker is pending or the approval does not match the hash.
The verifier checks the same conditions again.

**Assurance.** The approval block is an unsigned local record. It shows that the named person recorded
an approval in this repository. It is not a cryptographic signature and not evidence of independent
review.

## 4. Production

```bash
OUT=/kaggle/working/autond_ce2_production_<utc>
$CLI --gpu 0 --run-all --out $OUT       # preflight, twin, behavioural binding, then every run
$CLI --gpu 0 --resume --out $OUT        # only after an interruption
$CLI --finalize --out $OUT              # certificate.json, once
$CLI --verify --out $OUT --preflight-word8   # independent verification -> verification_<utc>.json
```

**Initialisation.**
- `run_manifest.json` is written last.
- If initialisation stops before that file exists, the directory is abandoned: `--resume` refuses it.
  Start a new directory, and keep and report the abandoned one.
- A failed preflight or behavioural gate is recorded as `PRECHECK_FAILED`, and `--finalize` records it.

**Interruption.**
- `journal.jsonl` is authoritative. A run continues from its last committed transition.
- An artifact without a committed transition goes to `runs/<id>/quarantine/`. It is hashed, journalled
  and never analysed, and that step runs again.
- A committed artifact that is missing or altered makes the run `FAILED`. It is never regenerated.
- A record cache that contradicts the journal stops the runner. Do not edit anything; report it.

**Across Kaggle sessions.** Attach the previous output as a dataset, copy it back to the same path, and
run `--resume`.

**Verification.** The verifier recomputes model scores and computes the decision from them. It reports
`numerical_status`:
- `EXACT`: the recomputed rho equals the stored-score rho to 1e-12 in every run;
- `DECISION_INVARIANT_NUMERICAL_DIFFERENCE`: some differ, but every sign the tests use and the
  decision are unchanged. This passes and is reported;
- `DECISION_RELEVANT_DIFFERENCE`: this fails.

Verify on the same hardware class as production where possible, and record both.

**Archiving.** Keep the complete output directory: data, outputs, records, journal, quarantine, twin
weights and behavioural set. Package it with `tools/package_artifact.py`, which refuses iCloud
placeholders. Do not keep production evidence only in a synced folder.
