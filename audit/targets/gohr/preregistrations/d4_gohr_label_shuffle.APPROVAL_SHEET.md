# D4 Gohr label shuffle — consolidated approval sheet and runbook

Protocol: `audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json`. Rationale: `d4_gohr_label_shuffle.RATIONALE.md`.

Nothing here has been approved. No D4 preflight or production stage has been run under this protocol.

## A. Decisions to make

For each decision, edit `decisions.<name>` in the JSON: set `"value"` and change `"status"` to `"APPROVED"`. Choose only from `options` where they are listed. The recommendation is in the JSON with its rationale.

| Decision | Recommended | Options / note |
|---|---|---|
| `decision_rule` | `directional_iut_v2` | or `conservative_two_sided_v1` (the September rule) |
| `alpha` | `0.05` | float |
| `effect_threshold` | `0.01` | float, a **fraction** of accuracy (0.01 = 1 pp) |
| `replicates` | `5` | integer ≥ 2 |
| `sample_sizes` | `{"train": 10000000, "validation": 1000000, "test": 1000000}` | positive integers |
| `epochs` | `200` | integer |
| `partition_overlap_policy` | `refuse_on_any_cross_partition_feature_row` | or `record_only` |
| `within_partition_duplicates_policy` | `record_only` | or `refuse_on_any_within_partition_duplicate` |
| `label_conflict_reporting` | `report_separately_non_gating` | single option |
| `failure_handling` | `abort_whole_run_no_exclusions` | single option |
| `calibration` | `none_required` | single option |

Top-level fields to fill:

| Field | What to write |
|---|---|
| `approved` | `true` |
| `approved_by` | your approver identifier |
| `approved_utc` | `YYYY-MM-DDTHH:MM:SSZ`, at or before the commit time (preflight checks this) |
| `approval_statement` | your statement of what you approve |
| `data_examined_before_approval` | what you have seen. At minimum, disclose the September D4 result (EFFECT_DETECTED) and the D0 overlap measurement (on a different dataset) |
| `status` | e.g. `APPROVED` (informational) |

The fixed design (`fixed_design`) was carried over from the existing D4 configuration and Gohr's procedure. It is approved as a whole by the top-level approval; `protocol-check` refuses any edit that departs from what the runner implements.

## B. Commands, in order (from `~/Documents/deep_speck`)

```bash
PY=~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python
P=audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json

# 1. After editing: structural check; exit 0 means well-formed and fully approved
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle protocol-check --preregistration $P --require-approved

# 2. Commit ONLY the protocol (and, if you edited them, its .md companions)
git add -- $P
git commit -m "Approve D4 Gohr label-shuffle preregistration"

# 3. Anchor: record this hash OUTSIDE the file (e.g. in your notes)
shasum -a 256 $P

# 4. USER-RUN GATE: preflight (generates and audits partitions; no training)
RUN=audit/dataset/evidence/d4_gohr_speck3264_r5_$(git rev-parse --short=12 HEAD)
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle preflight \
    --preregistration $P --preregistration-sha256 <anchor> --run-dir $RUN

# 5. USER-RUN GATE: production (resumable; result/ written once)
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle run \
    --preregistration $P --preregistration-sha256 <anchor> --run-dir $RUN

# 6. Independent verification (read-only)
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle verify --run-dir $RUN
```

### What each gate does

- **Preflight refuses** if:
  - the preregistration is pending, malformed, uncommitted, modified, or differs from the anchor;
  - the approval is dated after its commit;
  - any of the 8 D4 source files is not clean at HEAD;
  - `$RUN` or `$RUN.preflight_refused/` already exists;
  - a `.<run>.staging_*` directory from an interrupted write exists. Inspect it; it was never accepted.
- **Overlap policy:** if the approved overlap policy refuses, preflight exits 4. It writes one atomic directory, `$RUN.preflight_refused/`, holding the measurements and the generated data, kept as is. The protocol then stops. Do not re-run preflight to obtain a different dataset; a new dataset needs a documented protocol amendment.
- **Run refuses** if:
  - the approved design or anchor differs from preflight;
  - any partition file differs from preflight;
  - the sources differ from those recorded at preflight;
  - `$RUN/result` already exists.
- **Verify** recomputes the decision from (b, c) and the accuracies. It re-verifies every epoch chain and checkpoint, every perturbation sidecar and every partition file, and their bindings to preflight and the certificate.

**Interrupted runs:** if a run stops, rerun step 5 with the same arguments. Uncommitted epochs are quarantined and retrained, and every retry is recorded. An integrity refusal stops the run, and the outcome is NOT_COMPLETED.
