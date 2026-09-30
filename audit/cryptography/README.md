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

Production loads only the verified checkpoint `audit/cryptography/Archive/best5depth10.h5`
(resolved from the source tree via `audit_config.REFERENCE_CHECKPOINT_PATH`, never the working directory;
`sha256 256eb4a5…`). Depth is read from the file's structure and
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

## Working directory

Launch every production command from the **repository root** (the directory
that contains `audit/cryptography/`). With any other working directory the
output guard raises `WrongWorkingDirectoryError` before anything is written:
outputs are never redirected elsewhere. An explicit `--repo-root` is a
visible override (used by the tests for scratch directories).

## Frozen design and CE1 production

All scientific parameters are frozen in `frozen_design.py`
(`CE-frozen-design-2026-03`) and are validated, never chosen, at run time:

- **CE1** 8 blocks, >= 6 valid, 2 tolerated, exact paired sign-flip,
  FINAL_EPOCH 200, sealed set `33df1f95…`; equivalence-to-chance is
  DISABLED (no margin exists at this test-set size).
- **CE3** alpha 0.05 under the frozen fixed-sequence procedure.

CE1 production runs ONLY through the resumable controller
(`experiments/ce1/controller.py`); see `docs/ce1_controller_status.md` for
claims, block-resolution rules, legacy block0 recovery and the operating
sequence. Nothing trains without `--execute`.

## Reproducibility

Dataset generation draws from `os.urandom` and is **not seed-replayable**;
byte-for-byte replay requires persisting the generated data, which the CE1
controller does (content-hashed, committed per block, verified on every load). The probe/CV
seed is separate and *is* reproducible. Provenance records both separately
and claims no replayability it does not have. Design version is recorded in
metadata (`EXPERIMENT_DESIGN_VERSION`), not in filenames.
