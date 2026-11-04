# AutoND Experimental Validity

Four EV experiments for AutoND's DBitNet on SPECK32/64 (single-key, Δ = 0x400000), round 5, at the pinned upstream commit `8d66e0a4`. Configuration: `configs/autond_ev.yaml`. Code: `autond/`. Outputs: `results/autond/<mode>/<experiment>/`.

| Experiment | Design | What varies |
|---|---|---|
| `EV-AUTOND-BASELINE` | descriptive | fresh dataset and model seed per replicate |
| `EV-AUTOND-NOISE` | descriptive | nothing: one persisted dataset, one seed (initial weights verified identical), repeated runs |
| `H-EV-AUTOND-TRAIN-ORDER` | paired | `shuffle=True` (A) vs `shuffle=False` (B): example order only; not label shuffling, not D4 |
| `H-EV-AUTOND-REPRESENTATION` | paired | original encoding (A) vs a fixed within-word bit permutation (B) of train, validation and test features; labels unchanged |

## Protocol

Upstream `train_nets` for DBitNet, for one stage at round 5: `dbitnet.make_model(64)`, `Adam(amsgrad=True)`, MSE, `acc`, 10 epochs, batch 5000, 10⁷ training and 10⁶ validation rows, `model.fit` default `shuffle=True`. The EV-specific changes, recorded in every certificate (`autond.baseline.EV_SPECIFIC_CHANGES`):
- one stage from a fresh initialisation; no staged multi-round loop;
- the outcome is final-epoch accuracy on an independent 10⁶-row confirmatory test set, not max validation accuracy;
- TensorFlow is seeded per model, while GPU kernels stay nondeterministic;
- datasets are persisted and reloaded hash-verified;
- `shuffle` is passed explicitly;
- the two arms of a pair train in the frozen balanced order (`autond/order.py`), not always A first.

D5's approval does not cover any of this.

AutoND's 64 input columns, from the pinned source, are four 16-bit words, MSB first: C0 left, C0 right, C1 left, C1 right (`ciphers/speck3264.py` `convert_to_binary` and `encrypt`, `main.py` `make_train_data`). `tests/autond_tf_cases.py` checks this against the generator's own encrypt calls.

The representation arm (`autond/representation.py`, no Gohr code) permutes the bit positions within each word, keeping the word order. It uses seed 999, which realises the same mapping that was frozen for Gohr (SHA-256 `d7f9dd38…d427`). Before every calibration or production run, bijectivity, round trip, per-word content preservation, unchanged labels and determinism are checked on real AutoND data.

EV-AUTOND-BASELINE draws fresh training, validation and test data per replicate. Its spread is therefore end-to-end and includes test-set sampling; the certificate also reports the binomial test-set SD at the mean. EV-AUTOND-NOISE holds data, seed and configuration fixed, so it measures execution nondeterminism only.

## Statistics and decisions

These are the framework's, unchanged:
- **Effect.** D = B − A per pair.
- **Test.** Paired t-test and t-interval, with Cohen's d_z.
- **Equivalence.** TOST at ±ε.
- **Multiplicity.** Two Holm families across the AutoND primary family {TRAIN-ORDER, REPRESENTATION}: difference p-values and TOST p-values. A hypothesis without a p-value enters at p = 1.
- **Decision.** `decide_final`:
  - SUPPORTED needs a Holm-significant difference **and** a 95 % CI entirely outside ±ε;
  - NOT_SUPPORTED needs a non-significant difference **and** Holm-corrected equivalence;
  - anything else is INCONCLUSIVE.

The AutoND family is separate from Gohr's. `autond.experiments.apply_family_correction` takes production certificates only, and reads each certificate's CI from `statistics.confidence_interval` and ε from the TOST assessment. Gohr's `apply_primary_family_correction` reads `cert["confidence_interval"]` and `practical_significance["threshold"]`, which its certificates do not contain. That makes SUPPORTED unreachable after its Holm correction. Gohr is not changed here.

## Tiers

| | smoke | calibration | production |
|---|---|---|---|
| sizes | tiny | full protocol | full protocol |
| evidentiary | no | no | yes |
| binds | — | `design_hash` | `production_hash` |
| A/B order | derived on the fly | frozen schedule (10 pairs) | frozen schedule (6 pairs) |
| output | `results/autond/smoke/` (or `--output-root`) | `results/autond/calibration-balanced-ab-v1/` | `results/autond/production/` |
| gates | config only | no pending decision in representation/decisions/calibration; sources committed and clean; pin; permutation validated | as calibration, plus the production plan and a power plan derived under the same A/B protocol |

- `design_hash` covers the protocol, representation, decisions and calibration plan. So ε and the target effect are frozen before any pilot data exist.
- Each `(mode, experiment)` has its own directory, bound to its mode and held by a single-writer lock. Calibration and production seeds must not overlap.
- **Resumption.** A completed or failed unit is never re-run. An interrupted unit is re-run from scratch on its persisted data, with the same seed and the same scheduled A/B order. A unit whose confirmatory test set was consumed but whose result was not recorded is refused.

## Power planning

`plan-power` takes the calibration variance artifact, which holds σ_Δ and its one-sided 95 % chi-square upper bound per paired hypothesis, with no means or effects. It reuses the framework's joint simulation of the complete decision rule.

For each hypothesis, it requires P(SUPPORTED | effect = target) ≥ 80 % and P(NOT_SUPPORTED | effect = 0) ≥ 80 %. Each is taken as the worst case over the other hypothesis's state, because Holm couples the two. The smallest n meeting all four requirements is the production minimum.

The target effect must exceed ε. At a true effect equal to ε, P(SUPPORTED) stays below ½ for every n, so a plan that sets the target at the margin and sizes on "difference significant" alone, as the Gohr planner does, does not power the SUPPORTED decision.

## Balanced A/B execution order (`autond/order.py`, protocol `balanced-ab-v1`)

Within a pair, both arms share data, seed and verified initial weights, and run one after the other in one process on one GPU. Training them always A-then-B would confound B − A with execution position, for example through autotuning, caching or thermal state. So each pair has a frozen order, AB or BA, with ⌊n/2⌋ BA pairs:

    tag   = "autond-ev-ab-order-v1|<experiment>|design=<design_hash>|pairs=<n>"
    seed  = int(SHA-256(tag)[:16 hex digits], 16)
    order = numpy PCG64(seed).permutation(["AB"] * ceil(n/2) + ["BA"] * floor(n/2))

The schedules are frozen in `configs/autond_ev_order_schedule.json` (SHA-256 pinned in `autond/order.py`):

| Experiment | Production order, pairs 0–5 |
|---|---|
| `H-EV-AUTOND-TRAIN-ORDER` | BA, AB, AB, BA, AB, BA |
| `H-EV-AUTOND-REPRESENTATION` | BA, AB, AB, AB, BA, BA |

The file also holds the 10-pair calibration schedules.

**How the schedule is enforced:**
- **No draw at run time:** the runner never draws a schedule for calibration or production. It reads the frozen file and refuses if the file's hash, its design hash, its derivation, its length or its balance differ.
- **Recorded per unit:** each unit records `pair_id`, `execution_order`, `executed_conditions` and each arm's `execution_position`. The arms are stored in canonical A, B order, so the analysis is unchanged.
- **Bound to resume:** the order is part of the unit hash, so a schedule change refuses resumption.
- **Checked at aggregation:** a recorded order that differs from the schedule refuses aggregation.

The treatment definitions, statistics and decision rules are unchanged.

## Source and session provenance

**The source manifest** (`baseline.source_manifest`) is the SHA-256 of every file in `baseline.SOURCE_FILES`: the AutoND EV modules, the framework modules, both configs, and the dataset adapter and integrity code, plus one digest over them all. It is a code identity, distinct from the design and production hashes.

**How the manifest is bound:**
- **Launch:** every calibration or production invocation records git HEAD and checks that each file is clean.
- **Each unit:** every unit records `source_manifest_sha256`, `git_head`, `session_id` and `device`, and the manifest digest is part of the unit hash.
- **Each directory:** each output directory pins the manifest it was started under (`source_identity.json`). A later session with different code is refused (`SourceIdentityMismatch`), and so is aggregation of units carrying different digests.
- **Certificate:** it records the manifest, the single unit digest, the sessions and devices, and the frozen schedule.

**Session provenance** (`adapter.session_provenance`) records the following for every invocation:
- the GPU name, UUID and driver (`nvidia-smi`), TensorFlow build information (CUDA and cuDNN), visible devices and device details;
- the Python, TensorFlow, Keras, NumPy, SciPy, h5py, pandas and PyYAML versions;
- non-secret `KAGGLE_*` variables;
- a per-process session ID (boot ID plus a random token).

**NOISE scope.** EV-AUTOND-NOISE measures repeated-execution variability for one fixed dataset, seed, set of initial weights, and hardware and software environment, and nothing beyond that. Its reruns are refused if they would span two sessions or two devices (`SessionScopeViolation`). Run all ten in one session on one GPU.

## Calibration compatibility: fixed-order calibration replaced by a balanced one

The existing calibration (2026-10-30, Kaggle, design `84123aa4…`) trained every pair A then B. It is preserved unchanged in `results/autond/calibration/`:
- the 32 files of the original archive, byte-identical;
- a second backup outside the repository;
- no raw `.npz` datasets and no `.keras` models, so it cannot be fully recomputed;
- its GPU, CUDA, cuDNN and driver were not recorded and cannot be recovered.

**Why it can't size production.** Balanced order leaves the mean of B − A unbiased, but adds about δ² to its variance if training position has an effect δ. A fixed-order calibration cannot estimate that. So:
- `calibration_variance` accepts only certificates under the current protocol;
- `power_plan` records the protocol;
- the production gate refuses a power plan from another protocol.

The fixed-order plan `9fdf7244…` stays in `results/autond/calibration/` as a record, and the gate still refuses it.

**Balanced calibration (`results/autond/calibration-balanced-ab-v1/`).** Run on Kaggle on 2026-11-03, 10 pairs per hypothesis under the frozen schedule, one T4 per hypothesis, one session each, at commit `cd52db63`, with source manifest `81027ceb…`:

| Hypothesis | Valid | Order (AB / BA) | Mean B − A | SD | σ upper 95 % |
|---|---|---|---|---|---|
| H-EV-AUTOND-TRAIN-ORDER | 10/10 | 5 / 5 | −0.00024 | 0.00064 | 0.00105 |
| H-EV-AUTOND-REPRESENTATION | 10/10 | 5 / 5 | −0.13377 | 0.00045 | 0.00075 |

The resulting files are:
- variance artifact: `4c77681b…`;
- power plan: `c3db5cfe…`, with `required_n = 3`;
- what `production.power_plan` now declares: this plan.

The frozen production schedule has 6 pairs, so it covers that `n`. The raw `.npz` datasets and `.keras` models (2.5 GB) are kept outside the repository with the archive; the JSON records here carry their SHA-256s.

## Production design (`configs/autond_ev.yaml`)

| Experiment | Planned | Minimum valid | Trainings |
|---|---|---|---|
| EV-AUTOND-BASELINE | 10 | 8 | 10 |
| EV-AUTOND-NOISE | 10 | 8 | 10 |
| H-EV-AUTOND-TRAIN-ORDER | 6 pairs | 5 | 12 |
| H-EV-AUTOND-REPRESENTATION | 6 pairs | 5 | 12 |

That is 44 trainings in total.

**Power at 5 valid pairs (the minimum).** With the hypothesis-specific upper σ from the balanced calibration (TRAIN-ORDER 0.0010464, REPRESENTATION 0.0007468), the probability that both final decisions are correct is:

| σ × | Worst case over the four true-effect states |
|---|---|
| 1 | 0.949 |
| 1.25 | 0.950 |
| 1.5 | 0.950 |
| 2 | 0.948 |

At 6 valid pairs it is at least 0.949. The run is `power-sensitivity`, recorded in `results/autond/planning/power_sensitivity_balanced-ab-v1.json` (`5b0fc81f…`). The fixed-order version, `power_sensitivity.json`, is kept as a record.

**Fewer pairs:** 4 valid pairs keep at least 0.949 up to 2× σ. 3 pairs keep 0.945 up to 1.5× σ but drop to 0.894 at 2× σ.

**Non-normal differences (n = 5):**
- t₅, t₃ and contamination keep at least 0.955;
- skewed (lognormal) differences reduce the joint probability to about 0.82 (NOT_SUPPORTED about 0.87), because the small-sample t-test's false-significance rate rises.

## Commands (from `audit/experimental_validity`, pinned Python 3.11 / TF 2.15.1 with PyYAML, repository root importable)

```bash
python -B -m pytest -q -p no:cacheprovider tests                                 # full EV suite (TF cases run in a child process)
python -B -m autond.runner run --mode smoke --experiment all                     # pipeline check
python -B -m autond.runner preflight --mode calibration                          # gates only
# balanced recalibration (non-confirmatory), one process per GPU:
python -B -m autond.runner --gpu 0 run --mode calibration --experiment H-EV-AUTOND-TRAIN-ORDER --confirm-execute
python -B -m autond.runner --gpu 1 run --mode calibration --experiment H-EV-AUTOND-REPRESENTATION --confirm-execute
python -B -m autond.runner analyze-calibration                                   # -> results/autond/calibration-balanced-ab-v1/variance.json
python -B -m autond.runner plan-power                                            # -> results/autond/calibration-balanced-ab-v1/power_plan.json
python -B -m autond.runner power-sensitivity --variance <variance.json> --output <new.json>
python -B -m autond.runner preflight --mode production                           # refuses a plan from another execution-order protocol
python -B -m autond.runner --gpu 0 run --mode production --experiment <ID> --confirm-execute   # one process per GPU
python -B -m autond.runner family                                                # production Holm family
```

## Compute

Measured in calibration: 119–135 s per epoch, so about 20–22.5 min per training. Data generation, evaluation and saving were not measured separately.

| Run | Trainings | GPU time |
|---|---|---|
| Production | 44 | about 16.5–17.5 GPU-hours serial; about 8.5–9 h on 2 GPUs |
| Balanced recalibration | 40 | about 14–16 GPU-hours |
