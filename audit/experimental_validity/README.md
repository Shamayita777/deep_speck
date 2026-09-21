# CipherMind — Experimental Validity (EV)

Implements the Experimental Validity audit dimension for the CipherMind
cryptanalysis-auditing project, per the frozen scientific design
(5-round Gohr/Speck32/64 neural-distinguisher case study).

## Layout

```
framework/    Generic infrastructure - no Gohr-specific assumptions.
gohr/         Gohr/Speck32/64-specific adapter (dataset, model, training,
              evaluation, baseline, representation transform, the four
              frozen experiment definitions).
configs/      YAML configs for smoke and production runs.
tests/        pytest suite - see "Test tiers" below.
scripts/      validate_ev.py (pre-production gate), smoke_ev.py
              (non-evidentiary pipeline check), run_ev.py (production
              runner - fails closed on unresolved config),
              apply_family_correction.py (Holm correction across the
              two production certificates once both exist).
docs/         independence.md, discrepancies.md, statistical_plan.md.
results/      Output directory (smoke/ and production/ certificates).
Dockerfile    Target documented-baseline environment spec (see Issue 7
              honesty note inside it - not yet built/verified).
requirements.txt   Exact pins for the environment actually verified
              during development (see honesty note inside it - this is
              NOT the documented Gohr baseline environment).
```

## Test tiers (do not conflate these)

1. **Framework unit tests** (no TensorFlow needed) - statistics,
   effect sizes, multiplicity, power, certificate/decision logic,
   firewall, provenance. Fast (~1-2s total).
2. **TensorFlow-dependent integration tests** - real model
   construction/training/evaluation at smoke scale
   (`test_gohr_adapter_smoke.py`, `test_candidate1_firewall_lifecycle.py`,
   `test_resumability_integration.py`). Slower (real training runs);
   still not evidentiary - they exist to prove the pipeline is
   correct, not to produce scientific results.
3. **Smoke pipeline** (`scripts/smoke_ev.py`) - runs all four EV
   experiments end-to-end on a drastically reduced problem
   (rounds=3, depth=1, epochs=1, ~256 samples). Every certificate it
   produces carries `"non_evidentiary": true` / `"decision"` values
   that must never be read as scientific findings.
4. **Validation gate** (`scripts/validate_ev.py`) - cipher
   correctness, Candidate-1 transform validation, and software-version
   comparison against the documented baseline. Must pass before any
   production run.
5. **Production experiments** (`scripts/run_ev.py`) - not yet run;
   see "Production runs are currently blocked" below.

As of the last verified run in this repository's development
environment: **104 tests, all passing** (`python3 -m pytest tests/ -v`,
~85s). This count includes tiers 1 and 2 above; it does NOT include
tier 3 (the smoke pipeline is a script, not a pytest suite) or tier 5
(not executed). Do not treat a passing test suite, by itself, as
evidence that the scientific protocol is complete - see
`docs/statistical_plan.md` for the actual list of pre-production
blockers.

## Quick start

```bash
pip install -r requirements.txt   # exact pins - see file for what environment this verifies
python3 -m pytest tests/ -v                       # 104 tests, ~85s in the verified environment
python3 scripts/validate_ev.py                    # pre-production gate
python3 scripts/smoke_ev.py                       # non-evidentiary pipeline check (tier 3, NOT evidentiary)
python3 scripts/run_ev.py configs/gohr_ev_shuffle.yaml   # production (see below - currently refuses to run)
```

## Production runs are currently blocked, by design

Every production config in `configs/` contains explicit
`UNSPECIFIED_REQUIRES_...` placeholders for replicate counts and the
practical-significance threshold (epsilon). `scripts/run_ev.py`
refuses to run (exit code 1) until these are resolved via the
power-analysis procedure in `framework/power.py`, which itself
requires a defensible target effect size, a pilot noise estimate, and
a compute budget - none of which this implementation invents. A
`power_analysis.required_n` field, once populated, is additionally
checked against `minimum_valid_pairs` by
`framework.power.validate_replicate_plan_against_power` - an
evidentiary run cannot be configured with fewer minimum valid
replicates than its own declared power analysis requires, absent an
explicit, recorded justification. See `docs/statistical_plan.md`.

## The four frozen EV experiments

1. **EV-BASELINE** - independent replicates, descriptive noise floor.
2. **EV-NOISE** - identical-configuration reruns, descriptive execution variability.
3. **H-EV-SHUFFLE** - matched-paired, `shuffle=True` vs `False`.
4. **H-EV-REPRESENTATION** - matched-paired, original Gohr bit encoding
   vs. a fixed, information-preserving cross-word bit-alignment scramble
   ("Candidate 1").

No other core experiment is implemented (H-EV-ARCHITECTURE was
explicitly excluded from the frozen scope).
