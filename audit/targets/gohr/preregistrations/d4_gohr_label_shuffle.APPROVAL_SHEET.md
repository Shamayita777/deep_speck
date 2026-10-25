# D4 Gohr label shuffle — consolidated approval sheet and runbook

Protocol: `audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json`. Rationale: `d4_gohr_label_shuffle.RATIONALE.md`.

Nothing here has been approved. No D4 preflight or production stage has been run under this protocol (v4, rule `near_chance_baseline_gated_v4`).

**Scientific decisions only you can make** (RATIONALE §0):
1. **`planning_source_attestation`.** The planning note *Dataset Integrity — Redesign From First Principles* (2026-08-27) is archived at `sources/d4_planning_notes_2026-08-27.md` (SHA-256 `0fd2854a49fd0676a24a69cdb5132063ad4c7e2a3bc50f842b1a3ae8c9c3b4c7`). Approve only if this is the original D4 planning note and section F is the D4 source. The PDF `~/Downloads/31 (1).pdf` is the overarching methodology only.
2. **`chance_margin`.** The candidate 0.005 gives the band [0.495, 0.505]. It is an explicit practical design choice that no source prescribes. It means a shuffled fit keeps at most about 1.2 % of the published 5-round advantage. It is not narrowed because n = 10⁶ would allow it, and it is not tuned to any D4 result.
3. **`baseline_adequacy_threshold`.** The candidate 0.90 is a reproduction-adequacy tolerance 2.9 points below the published 0.929. It is not prescribed by Gohr or by the methodology. Its pipeline differences are listed in `fixed_design.comparability_to_published_pipeline`.
4. **`paired_endpoint_role`.** `secondary`, the only option: reported, never gating.

**Overall outcome:** `SUPPORTED` only if every shuffled fit is near chance **and** every intact-label baseline is adequate. `BASELINE_INADEQUATE`, `INCONCLUSIVE` and `NOT_COMPLETED` are never passes.

The September 2026 `EFFECT_DETECTED` result is historical. Its status is known; its per-replicate arrays are not on disk and were not inspected. It did not set the margin or the floor.

## A. Decisions to make

For each decision, edit `decisions.<name>` in the JSON: set `"value"` and change `"status"` to `"APPROVED"`. Choose only from `options` where they are listed. The recommendation is in the JSON with its rationale.

| Decision | Recommended | Options / note |
|---|---|---|
| `decision_rule` | `near_chance_baseline_gated_v4` | single option; the runner refuses paired-only and ungated rules |
| `chance_margin` | `0.005` (candidate, **your scientific decision**) | float in (0, 0.5), a **fraction** of accuracy; must keep `"units": "accuracy_fraction"`; never copied from `effect_threshold` |
| `baseline_adequacy_threshold` | `0.90` (candidate, **your scientific decision**) | float in (0.5, 1), a **fraction** of accuracy; must keep `"units": "accuracy_fraction"` |
| `paired_endpoint_role` | `secondary` | single option; non-gating |
| `planning_source_attestation` | `confirmed_by_protocol_owner` | single option; only if the recorded description of section F is accurate |
| `alpha` | `0.05` | float; TOST level per one-sided test, baseline lower-bound level, IUT level, Bonferroni family level (α/(2k) per departure test, α/k per inadequacy test) |
| `effect_threshold` | `0.01` | float, a **fraction** of accuracy (0.01 = 1 pp); supporting paired endpoint only |
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
| `data_examined_before_approval` | what you have seen. At minimum, disclose: the September D4 status (EFFECT_DETECTED; its per-replicate arrays are unavailable and were not inspected); the D0 overlap measurement (on a different dataset); and the design-only operating characteristics in RATIONALE §0.5 and §0.7 |
| `status` | e.g. `APPROVED` (informational) |

The fixed design (`fixed_design`) was carried over from the existing D4 configuration and Gohr's procedure. It is approved as a whole by the top-level approval; `protocol-check` refuses any edit that departs from what the runner implements.

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
