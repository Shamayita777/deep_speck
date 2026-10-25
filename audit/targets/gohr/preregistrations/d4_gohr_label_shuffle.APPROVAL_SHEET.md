# D4 Gohr label shuffle — consolidated approval sheet and runbook

Protocol: `audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json`. Rationale: `d4_gohr_label_shuffle.RATIONALE.md`.

**Approval status: APPROVED.** The approved design, approval identity,
timestamp, and pre-approval disclosure are recorded in
`d4_gohr_label_shuffle.json`.

No D4 preflight or production stage has been run under this protocol.

## A. Approved decisions

The table below records the decisions approved in the companion JSON.
The JSON is the authoritative machine-readable record.

| Decision | Approved value | Interpretation |
|---|---|---|
| `decision_rule` | `near_chance_baseline_gated_v4` | Near-chance equivalence gated by intact-label baseline adequacy |
| `chance_margin` | `0.005` | Accuracy band `[0.495, 0.505]` around chance `0.5` |
| `baseline_adequacy_threshold` | `0.90` | Every intact-label fit must have a one-sided 95% exact lower confidence bound strictly above `0.90` |
| `paired_endpoint_role` | `secondary` | Paired McNemar endpoint is secondary and never gates the outcome |
| `alpha` | `0.05` | Significance level |
| `effect_threshold` | `0.01` | Fraction of accuracy; `0.01` means one percentage point |
| `replicates` | `5` | Five planned paired replicates |
| `sample_sizes` | `{"train": 10000000, "validation": 1000000, "test": 1000000}` | Fixed train, validation and test sample sizes |
| `epochs` | `200` | Fixed epoch limit |
| `partition_overlap_policy` | `refuse_on_any_cross_partition_feature_row` | Refuse preflight if any cross-partition feature row overlaps |
| `within_partition_duplicates_policy` | `record_only` | Record within-partition duplicates without refusing solely for that reason |
| `label_conflict_reporting` | `report_separately_non_gating` | Report label conflicts separately; non-gating |
| `failure_handling` | `abort_whole_run_no_exclusions` | Abort the whole run; no selective exclusions |
| `calibration` | `none_required` | No calibration run required by this protocol |
| `planning_source_attestation` | `confirmed_by_protocol_owner` | Confirms the archived D4 planning note as the intended source; retain only if genuinely confirmed |

### Approval record

The JSON's top-level fields record the approver, approval timestamp, approval
statement, and what was examined before approval.

The approved `chance_margin` of `0.005` and baseline-adequacy floor of `0.90`
are protocol-owner-selected practical tolerances. They are not presented as
numerical values prescribed by the overarching methodology or the D4 planning
note, and they were not selected using the September 2026 historical D4 result.

The prior D0 overlap measurement concerns a different dataset. Its zero
duplicates, zero cross-partition overlap, and zero label conflicts do not
establish anything about the new D4 partitions. The new D4 partitions have
not yet been generated or audited by preflight.

The historical September D4 result was reported as `EFFECT_DETECTED`; its
per-replicate arrays were unavailable and were not used to choose the v4
thresholds. The v4 preflight and production run remain unexecuted.

The fixed design (`fixed_design`) is approved as a whole by the top-level
approval. `protocol-check` must pass before proceeding; it refuses edits that
depart from the design implemented by the runner.
## B. Commands, in order (from `~/Documents/deep_speck`)

```bash
PY=~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python
P=audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json

# 0. Current draft: exit 0 with fields listed as pending (--require-approved exits 3 until all are approved)
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle protocol-check --preregistration $P
# (optional) reproduce the margin operating characteristics; design inputs only
$PY -B -m audit.dataset.tools.d4_near_chance_operating_characteristics

# 1. After editing: structural check; exit 0 means well-formed and fully approved
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle protocol-check --preregistration $P --require-approved

# 2. Commit the protocol and both synchronized companion documents
git add -- "$P" \
    audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.RATIONALE.md \
    audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.APPROVAL_SHEET.md
git commit -m "Approve D4 Gohr label-shuffle preregistration and companion docs"

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
$PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle verify --run-dir $RUN --rerun-models
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
- **Verify**:
  - recounts every fit's correct count and (b, c) from the persisted predictions and the test labels;
  - recounts the class counts from the same files;
  - recomputes every TOST, departure, baseline-adequacy and McNemar statistic, and the overall outcome, from those counts;
  - refuses any margin, floor, α, rule, role, test n or procedure that differs from the approved design;
  - re-verifies every epoch chain and checkpoint, every perturbation sidecar and every partition file.

  Add `--rerun-models` to also reload every selected checkpoint and require it to reproduce the persisted test predictions. This needs TensorFlow and is recommended once on the production run.

  Exit 0 means integrity is verified. The printed **overall D4 outcome** is the scientific result, and only `SUPPORTED` is a pass. Exit 5 means there is no result (`NOT_COMPLETED`). Verify does not rerun the network.

**Interrupted runs:** if a run stops, rerun step 5 with the same arguments. Uncommitted epochs are quarantined and retrained, and every retry is recorded. An integrity refusal stops the run, and the outcome is NOT_COMPLETED.
