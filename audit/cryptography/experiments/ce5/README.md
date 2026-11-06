# G-CE5: average-key-rank comparator for the Gohr depth-10 checkpoint

## Status (r3, 2026-11-06)

Design `Gohr-CE5-design-2026-11-r3`, SHA-256 `409e7dcb53c5eba7a623a584ec302f2aa3a26e84b3efcf9e6c021e5629fde5c5` (printed by `design.design_sha256()`).

| item | state |
|---|---|
| algorithm | P1 (ePrint 2021/287) Algorithms 1 and 2, literal: (δl‖δv) table, 2^12 candidates, strict threshold |
| primary source | read, transcribed and mapped (`docs/SOURCE_VERIFICATION.md`) |
| tables | built, hashed in the design; registered validation `validation/r3_final_20261105_tablecheck_v2/table_validation.json` (overall PASS) |
| benchmark | done (`docs/BENCHMARK.md`) |
| table variability | done, non-confirmatory; second run independently verified |
| model binding | frozen behavioural check PASS |
| design decisions | margin 0.10; gate tolerance 0.005 / 0.010; n_replicates 20 (from calibration); no field unset |
| calibration | done and independently verified: `validation/calibration_20261105T164019Z/` (R = 20). Run under design `b5c4ac4f`; the current design differs from it only in n_replicates, its record, design_history and two wording corrections (tolerance_rule, anchoring), and the record re-verifies under the current design |
| current draft | `preregistration/preregistration_409e7dcb_draft_20261106T050444Z.json`, SHA-256 `81991d2c4a8c3833442a0adc84bfc69bd88992ca51b49ddf4bb3efb26fc16a5e`; **not approved**; check record `validation/prereg_check_409e7dcb_draft_20261106T050444Z.txt` (23 / 23) |
| `preregistration/preregistration_654bc102_draft_20261106T035130Z.json` | approved 2026-11-06T04:04:52Z for design `654bc102`; superseded by the anchoring-wording correction, so its approval does not apply to the current design; kept unchanged, never timestamped, never used by production |
| `preregistration/preregistration_f6f0ceb9_draft_*.json` | earlier drafts; not approved; kept unchanged |
| `preregistration/preregistration.json` | stale draft from an earlier design state; kept unchanged and never used by production |
| gate, production | not run |
| commits, timestamp proofs | none for CE5 |

### Remaining before production

1. Approval of the current draft, recorded in its approval block (status, approver, UTC; the design hash is already filled).
2. Commit of the CE5 files.
3. `ots stamp` of the approved file, then `ots upgrade` once the Bitcoin attestation exists.
4. A successful `ots -v verify` recorded by `runner.record_ots_verification` at `<approved file>.ots.verification.json`. This needs a machine with access to a Bitcoin node; this Mac has none.
5. At least 2 GiB of available memory at launch.
6. Production with `--prereg <approved file> --prereg-sha256 <its SHA-256>`.

## Question

Within real 5-round SPECK32/64 pairs, does the bound network's output N agree more with the average-key-rank score D than with the CE2 single-trail probability T?

- Both are cryptanalytic accounts. The experiment compares them; it does not test whether the signal is cryptographic, and it cannot establish a mechanism.
- The comparison is restricted to real pairs because T exists only for them.
- D may be called a reproduction of Benamira et al.'s distinguisher only after the gate returns REPRODUCED.

## Files

| file | role |
|---|---|
| `akr.py` | Algorithm 1 (table), Algorithm 2 (literal 2^12 loop, strict decision), LR comparator |
| `reference.py` | independent scalar transcription of Algorithms 1 and 2, its own SPECK, and the independent table check |
| `design.py` | design, hash, pending items, stage gates, authorisation |
| `analysis.py` | per-replicate statistics, interval, outcomes, gate, power |
| `runner.py` | stages: tables, tablecheck, benchmark, variability, calibrate, production |
| `verify.py` | independent recomputation of production output and variability records (imports only `design.py` and `reference.py`) |
| `tables/` | registered tables and their build manifest |
| `validation/` | non-confirmatory records; every record carries `excluded_from_evidence`. `validation/redacted_for_presentation_20261106/` holds presentation copies of seven raw test records with the local temporary-directory prefix replaced (`REDACTION_MANIFEST.json`); the originals are unchanged |
| `docs/` | SOURCE_VERIFICATION, CRYPTO_CORRECTNESS, BASELINE_GATE, STATISTICS, BENCHMARK, REVIEW_REPORT |
| `historical/r2_superseded_20261105/` | superseded r2 code, tables and benchmark; validation of r2 only |
| `historical/design_before_*` | `design.py` as it stood before each design change (`edb4bb88`, `b5c4ac4f`, `f6f0ceb9`, `654bc102`) |

## Commands

Run every command from the repository root.
- `PY` is `~/.venvs/gohr-tf221-validation-20261031/bin/python`.
- `T` is `audit/cryptography/experiments/ce5/tables`.
- `M` is `-m audit.cryptography.experiments.ce5.runner`.

Every stage refuses to write to an existing path.

| step | command | status |
|---|---|---|
| tables | `PY M tables T` | done |
| table re-verification | `PY M tablecheck <new dir> --tables T` (exit 1 on FAIL) | done: `validation/r3_final_20261105`, `validation/r3_final_20261105_tablecheck_v2` (registered) |
| benchmark | `PY M benchmark <new dir> --tables T` | done: `validation/`, `validation/r3_final_20261105` |
| variability | `PY M variability <new dir>` | done: `validation/r3_final_20261105`, `validation/r3_final_20261105_variability_v2` |
| variability verification | `PY -c "from pathlib import Path; from audit.cryptography.experiments.ce5 import verify as V; print(V.verify_variability(Path('<dir>')))"` | done (`variability_verification.json` in each directory) |
| calibration | `PY M calibrate <new dir> --tables T` (uses the registered 10^6 pairs; 10 replicates; blinded) | done: `validation/calibration_20261105T164019Z` |
| timestamp verification record | `PY -c "from audit.cryptography.experiments.ce5 import runner as R; R.record_ots_verification('<approved preregistration file>', ots_bin='$HOME/.venvs/ots/bin/ots')"` on a machine with a Bitcoin node | after `ots upgrade`, on a machine with a Bitcoin node |
| production | `PY M production audit/cryptography/experiments/ce5/evidence/<UTC> --tables T --confirm <token> --prereg <approved file> --prereg-sha256 <its SHA-256>` | after approval, timestamping and a successful verification record |
| verification | `PY -m audit.cryptography.experiments.ce5.verify <evidence dir> --tables T --prereg <approved file> --model` | after production |

## Tests

```
PY -m pytest -q -p no:cacheprovider audit/cryptography/experiments/ce5/tests
```

| file | kind |
|---|---|
| `test_crypto.py` | cryptographic known answers, exhaustive key-bit and losslessness checks, implementation agreement |
| `test_statistics.py` | statistics software correctness |
| `test_pipeline.py` | synthetic end-to-end lifecycle, refusals and tampering |

Test counts and results are recorded in `docs/REVIEW_REPORT.md`. Passing tests show that the code computes what the documents specify. They are not evidence that Table 7 is reproduced; that is decided by the gate.
