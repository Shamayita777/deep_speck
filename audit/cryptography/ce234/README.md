# CE2–CE4 corrected production package

Isolated, additive package. It imports the existing `audit.cryptography`
modules read-only and modifies none of them, so an in-flight CE1 process that
imports those modules is unaffected. It never reads or writes any CE1 run
directory, CE1 checkpoint or CE1 sealed set; `--preflight` asserts this.

**Designed to support an IEEE S&P-quality research artifact.** No claim of
IEEE S&P compliance or acceptance is made.

## Commands (run from the repository root)

    # 1. preflight: binds the model by hash, architecture AND behaviour
    python -m audit.cryptography.ce234.production CE2 --preflight

    # 2. smoke: reduced data, output marked SMOKE ONLY — NOT SCIENTIFIC EVIDENCE
    python -m audit.cryptography.ce234.production CE2 --smoke
    python -m audit.cryptography.ce234.production CE3 --smoke
    python -m audit.cryptography.ce234.production CE4 --smoke

    # 3. production (only after the open decisions below are signed off)
    python -m audit.cryptography.ce234.production CE2 --production
    python -m audit.cryptography.ce234.production CE3 --production
    python -m audit.cryptography.ce234.production CE4 --production

    # 4. independent verification of any run directory
    python -m audit.cryptography.ce234.verify <run_dir>

    # 5. tests
    python -m pytest audit/cryptography/ce234/tests -q

Production evidence is written to
`audit/cryptography/evidence_current/ce{2,3,4}/production_<date>/`, never to
`audit/cryptography/evidence/` (historical, immutable) and never on top of an
existing non-empty directory.

## What each run directory contains

    frozen_plan.json     the design + SAP as executed, with its hash
    preflight.json       model binding, environment, source hashes
    raw/*.npz            every raw observation (targets, outputs, selectivities)
    certificate.json     statistics, decision, limitations, provenance

`verify.py` recomputes every reported statistic from `raw/` with its own
implementations (its own rank/Spearman code) and imports nothing from
`production.py`, so a bug in the production estimator cannot be reproduced by
the verifier. It detects a wrong model hash, wrong architecture, a missing
behavioural round binding, a wrong sample or replication count, an altered raw
file, an altered certificate number, and a flipped sign convention.

## Open decisions — production FAILS CLOSED until these are signed off

See `UNSPECIFIED_IN_SOURCE` in `frozen_plan.py`:

| Item | Status |
|---|---|
| CE2 replication count (10 runs) | ADDITIONAL SCIENTIFIC RECOMMENDATION |
| CE3 control model (untrained twin) | ADDITIONAL SCIENTIFIC RECOMMENDATION |
| CE4 run count (5) and `n_intervention_bits` (8) | RECOMMENDATION / inherited adapter default |

These must be fixed **before** production and never revised afterwards.
