# AutoND CE2: runbook (design r3)

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
PYTHONDONTWRITEBYTECODE=1 $PY -B -m pytest -q -p no:cacheprovider audit/cryptography/experiments/autond/ce2/tests audit/cryptography/core/tests
$CLI --check-model                      # strict load, architecture, inference mode
$CLI --validate-math                    # gates C, D (exhaustive to word size 8, 16-bit literal), E, F (~5 min)
$CLI --power --report <new.json>        # scenario-based sensitivity analysis + pipeline-faithful check
$CLI --preflight                        # production preflight, including the runtime gate
$CLI --smoke --out /kaggle/working      # creates /kaggle/working/autond_ce2_smoke_<utc>/, then verifies it
```

Smoke output is labelled `SMOKE_NOT_CONFIRMATORY`. It uses 3 runs of 4,096 pairs and is never merged
with production. Its decision is `INCONCLUSIVE` by construction, because 3 runs cannot reach p <= 0.05.

## 3. Approval (before any production)

The design is r3 and is fully specified. Both former `PROPOSED_PENDING_APPROVAL` markers
(`runs.status`, `statistics.power.status`) are `FROZEN`, and `preregistration/preregistration.json` has
been regenerated from `design.py`. Only the approval block is still open:

| | |
|---|---|
| design version | `AutoND-CE2-frozen-design-2026-10-r3` |
| design SHA-256 | `3f6340e8625749d1a1281ea44984ea4ee48521e00c6c774598398680b8a0635f` |
| preregistration file SHA-256 (as regenerated, approval pending) | `28ee038be530223bb9f34de983f2718b7125accb3ed6b2d45085de60a4cd1a41` |
| power evidence bound by the design | `validation/r3/power_final.json`, `c3eeb2326765fb6696d1f45c1789d61e039f0c045a91a51782f7057c1ecac161` |
| supersedes | r2 `ef3d764a...` (never approved; archived in `history/r2/`); r3 draft 1 `7e69ce6c...` (never approved; `history/r3-draft1/`) |

**1. Review.** Read `README.md`, `POWER_ANALYSIS.md`, `VERIFICATION_REPORT.md`, `DESIGN["revision"]`
in `design.py`, and `preregistration/preregistration.json`. The r2 to r3 change corrects the written
hypothesis and estimand, and reports a sign test with no nonzero value as NO_EFFECTIVE_OBSERVATIONS
instead of p = 1. Every decision outcome is unchanged.

**2. Check the hash yourself** (repository root):

```bash
PYTHONDONTWRITEBYTECODE=1 $PY -B -c "from audit.cryptography.experiments.autond.ce2 import design as FD; \
p = FD.load_preregistration('audit/cryptography/' + FD.PREREGISTRATION_RELPATH); \
print(FD.DESIGN_VERSION, FD.design_sha256(), p['design_sha256'], FD.pending_items(), p['approval'])"
```

It must print the design hash above twice, `[]`, and the pending approval block.

**3. Record the approval.** Edit only the `approval` block of `preregistration/preregistration.json`,
by hand. Do not re-render the file: `render_preregistration()` always writes a pending block. The block
currently reads:

```json
"approval": {
    "status": "PENDING_REVIEW",
    "approved_design_sha256": null,
    "approved_by": null,
    "approved_utc": null
}
```

To approve, replace it with:

```json
"approval": {
    "status": "APPROVED",
    "approved_design_sha256": "3f6340e8625749d1a1281ea44984ea4ee48521e00c6c774598398680b8a0635f",
    "approved_by": "<FULL_NAME_OF_ACCOUNTABLE_APPROVER>",
    "approved_utc": "<ACTUAL_ISO_8601_UTC_APPROVAL_TIMESTAMP>"
}
```

- `approved_design_sha256` must equal the r3 design hash exactly. Any other value, including the r2
  hash `ef3d764a...` and the r3 draft-1 hash `7e69ce6c...`, is refused.
- `approved_by` names the person who actually authorises the experiment.
- `approved_utc` is the actual time of approval in UTC (for example from `date -u +%Y-%m-%dT%H:%M:%SZ`
  at the moment of approval). Do not backdate it or generate it in advance.
- Keep exactly one `approval` block with exactly these four keys. The loader and the verifier parse the
  file strictly and refuse any duplicate JSON key (a second `"status"` would otherwise silently win).
- Do not change anything outside the approval block. Any change to the design needs a new
  `DESIGN_VERSION`, a new hash and a new approval.

This four-field block belongs to the AutoND CE2 preregistration only. It is not the
`core/approval.py` approval-record schema used by the ce234 authorised route, and the two are not
interchangeable.

**4. Confirm** that the approval is accepted:

```bash
PYTHONDONTWRITEBYTECODE=1 $PY -B -c "from audit.cryptography.experiments.autond.ce2 import design as FD; \
print(FD.production_authorisation(FD.load_preregistration('audit/cryptography/' + FD.PREREGISTRATION_RELPATH)))"
```

It must print `(True, 'approved')`. Then commit the approved file. From then on it is never edited.

`--run-all` refuses production unless every marker is frozen, the status is `APPROVED`, the approved
hash equals the design hash and both personal fields are filled. The verifier checks the same
conditions again on production directories.

**Assurance.** The approval block is an unsigned local record. It shows that the named person recorded
an approval in this repository. It is not a cryptographic signature and not evidence of independent
peer review.

**Not yet run:** the production preflight in the pinned Kaggle environment (`--preflight`), and the
10^6-sample behavioural binding, which `--run-all` performs during initialisation.

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
