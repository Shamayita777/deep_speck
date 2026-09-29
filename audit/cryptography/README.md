# Cryptographic Evidence Audit

Four experiments interrogate whether a frozen Gohr/Speck32/64 neural
distinguisher's behaviour reflects cryptanalytic structure. **They ask
different questions and are not interchangeable.**

| | Question | Implementation |
|---|---|---|
| **CE1** | Does destroying the differential training signal cause loss of distinguishing performance? | `experiments/ce1/` |
| **CE2** | Is model output associated with the analytical single-trail quantity? (observational) | `experiments/ce2/` |
| **CE3** | Is that quantity decodable from the internal representation beyond a control? | `experiments/ce3/` |
| **CE4** | Is output more sensitive to difference-bearing ciphertext structure than to an XOR-preserving control? | `experiments/ce4/` |

A negative CE2 alongside positive CE3/CE4 is **not** a contradiction: they
measure output agreement, representation decodability, and intervention
sensitivity respectively. Any conflict is preserved, never majority-voted.

## Layout

```
audit/cryptography/
├── audit_config.py      reference configuration (single source of truth)
├── integrity.py         checkpoint hash + realized-depth verification
├── output_policy.py     write protection for historical evidence
├── statistics.py        underflow-safe p-values, replicate-level inference
├── provenance.py        run provenance, EXPERIMENT_DESIGN_VERSION
├── certificate.py       certificate schema, validation, safe writing
├── preflight.py         final gate before expensive execution
├── gohr/ probe/ adapters/ engine.py evaluation.py results.py reporting.py
├── experiments/ce{1..4}/
│   ├── design.py        corrected scientific design for that CE
│   └── gohr_*.py        the authoritative production driver
├── tests/               CPU test suite
├── evidence/            HISTORICAL evidence - immutable
└── test/                historical CE test definitions - preserved
```

There is exactly **one** authoritative production path per CE.

## Reference configuration

Speck32/64, **5 rounds**, differential **(0x0040, 0x0000)**, depth **10**,
L2 **1e-5** — declared once in `audit_config.py`. Nothing relies on a
constructor default: the historical depth-5 result arose because a default
was never overridden.

Production loads only the verified checkpoint `Archive/best5depth10.h5`
(`sha256 256eb4a5…`). Depth is read from the file's structure and
cross-checked (`Conv1D == 1 + 2·depth` and residual-merge count). The
historical artifact named `best5depth10 (10).h5` actually contains a
depth-5 network and is **rejected by hash**.

## Historical vs current evidence

`evidence/` holds the historical CE1–CE4 results (depth-5 models). They are
immutable and are never rewritten to agree with the current design. Current
runs write only to `evidence_current/`; the write guard resolves paths and
refuses anything outside it, including `..` traversal and symlink escapes.

## Running

```bash
# CPU validation
python -m pytest audit/cryptography/tests -q

# preflight (verifies everything, runs nothing)
python -m audit.cryptography.experiments.ce2.gohr_theory_consistency --preflight

# structural dry run (toy, non-evidentiary)
python -m audit.cryptography.experiments.ce1.gohr_signal_destruction --dry-run
```

## Before the GPU experiment

Three parameters must be **frozen first** — the drivers refuse production
without them rather than defaulting:

- **CE1** `--n-blocks` (from a power rationale) and `--equivalence-margin`
- **CE3** `--corrected-alpha` (multiplicity-corrected threshold)

```bash
python -m audit.cryptography.experiments.ce1.gohr_signal_destruction \
    --preflight --n-blocks <N> --equivalence-margin <EPS>
```

## Reproducibility

Dataset generation draws from `os.urandom` and is **not seed-replayable**;
byte-for-byte replay requires persisting the generated data. The probe/CV
seed is separate and *is* reproducible. Provenance records both separately
and claims no replayability it does not have. Design version is recorded in
metadata (`EXPERIMENT_DESIGN_VERSION`), not in filenames.
