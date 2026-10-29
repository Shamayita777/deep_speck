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
- `shuffle` is passed explicitly.

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
| gates | config only | no pending decision in representation/decisions/calibration; sources committed and clean; pin; permutation validated | as calibration, plus the production plan and power plan |

- `design_hash` covers the protocol, representation, decisions and calibration plan. So ε and the target effect are frozen before any pilot data exist.
- Each `(mode, experiment)` has its own directory, bound to its mode and held by a single-writer lock. Calibration and production seeds must not overlap.
- **Resumption.** A completed or failed unit is never re-run. An interrupted unit is re-run from scratch on its persisted data with the same seed. A unit whose confirmatory test set was consumed but whose result was not recorded is refused.

## Power planning

`plan-power` takes the calibration variance artifact, which holds σ_Δ and its one-sided 95 % chi-square upper bound per paired hypothesis, with no means or effects. It reuses the framework's joint simulation of the complete decision rule.

For each hypothesis, it requires P(SUPPORTED | effect = target) ≥ 80 % and P(NOT_SUPPORTED | effect = 0) ≥ 80 %. Each is taken as the worst case over the other hypothesis's state, because Holm couples the two. The smallest n meeting all four requirements is the production minimum.

The target effect must exceed ε. At a true effect equal to ε, P(SUPPORTED) stays below ½ for every n, so a plan that sets the target at the margin and sizes on "difference significant" alone, as the Gohr planner does, does not power the SUPPORTED decision.

## Decisions still required (placeholders in the config)

1. ε (`practical_threshold`) and its justification. The audit-wide precedent is 0.01 (Gohr EV, II-4); adopting it for AutoND is a decision, not a default.
2. `target_effect` (> ε) and its justification.
3. Calibration pairs per paired hypothesis.
4. Production units and `minimum_valid` for all four experiments, with the paired minimum ≥ the power plan's `required_n`.
5. The power plan's SHA-256 and `required_n`, after `plan-power`.

The seeds, the within-word permutation and the inherited α = 0.05, 80 % power, 50 000 simulations and σ upper bound are proposed in the config; review them with the rest.

## Commands (from `audit/experimental_validity`, pinned Python 3.11 / TF 2.15.1 with PyYAML, repository root importable)

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_autond_ev.py              # tiny, non-evidentiary; runs tests/autond_tf_cases.py in a child process
python -B -m autond.runner run --mode smoke --experiment all                     # pipeline check
python -B -m autond.runner preflight --mode calibration                          # gates only
python -B -m autond.runner --gpu 0 run --mode calibration --experiment H-EV-AUTOND-TRAIN-ORDER --confirm-execute
python -B -m autond.runner --gpu 1 run --mode calibration --experiment H-EV-AUTOND-REPRESENTATION --confirm-execute
python -B -m autond.runner analyze-calibration                                   # -> results/autond/calibration/variance.json
python -B -m autond.runner plan-power                                            # -> results/autond/calibration/power_plan.json
python -B -m autond.runner preflight --mode production
python -B -m autond.runner --gpu 0 run --mode production --experiment <ID> --confirm-execute   # one process per GPU
python -B -m autond.runner family                                                # production Holm family
```

Compute: 10 epochs per training against Gohr's 200. On a T4 at roughly 2 min per 10⁷-row epoch, that is about 20–25 min per training and about 45–50 min per pair. This is an estimate, not a measurement.
