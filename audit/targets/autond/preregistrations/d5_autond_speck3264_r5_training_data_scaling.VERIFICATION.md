# AutoND D5 — verification record

What has been checked for the D5 training-data scaling experiment, and whether it meets its integrity requirements. Protocol and approval: `d5_autond_speck3264_r5_training_data_scaling.json`. Scientific rationale: `d5_autond_speck3264_r5_training_data_scaling.RATIONALE.md`.

**Status (2026-10-29).** The implementation is verified at tiny scale on CPU. **The production experiment has not been run**, and tiny-scale tests do not show that it will run correctly on a GPU at full scale. Once it has run, its verification output is `result/d5_autond_certificate.json` in the run directory.

## 1. Layout

| File | Role |
|---|---|
| `audit/dataset/d5_training_data_scaling.py` | generic D5 engine (shared with Gohr D5), unmodified |
| `audit/dataset/adapters/autond.py` | pinned AutoND data binding, unmodified (data and provenance only, by design) |
| `audit/dataset/experiments/d5_autond_training_data_scaling.py` | AutoND D5 driver: gates, model factory, data checks, engine call, CLI |
| `audit/dataset/tools/d5_autond_postrun_verify.py` | post-run verifier and certificate (also called by the driver at the end of a run) |
| `audit/dataset/tests/test_d5_autond_training_data_scaling.py` | tests |

## 2. Inputs checked

| Item | Value | Result |
|---|---|---|
| Preregistration | approved; SHA-256 `ea0e03f63b03ce5563d5b48bad103b02650f0521e8a5fbb5aba85daaddc584fd`; its commit of record is the last commit that touches the JSON | `protocol-check --require-approved` exits 0; the anchored-commit gate accepts it; approval time precedes the commit |
| Generic engine | SHA-256 `04f3f319…`, schema 2.1 | unchanged |
| AutoND adapter | SHA-256 `be81a5dc…` | unchanged |
| AutoND upstream | commit `8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` | no tracked modifications; `main.py`, `optimizer.py`, `train_nets.py`, `dbitnet.py`, `gohrnet.py`, `ciphers/speck3264.py` equal `AUTOND_SOURCE_PIN` |
| Model and training | `dbitnet.make_model(64)`, `Adam(amsgrad=True)`, MSE, `acc`, 10 epochs, batch 5000, final-epoch model | equal to the registered fixed design and to upstream `train_nets` |
| Generator | `main.make_train_data`: X = `[C0, C1]` bits, (n, 64) uint8 {0,1}; Y random bits; `os.urandom` | equal to the registered representation; datasets cannot be replayed from a seed |

## 3. Integrity requirements

| Requirement | How it is enforced | Checked by |
|---|---|---|
| Only the approved, anchored, committed protocol runs, with exactly its values | `execute()` reloads the preregistration from its recorded path and anchor and compares the run's values, commit and protocol id, whichever entry point calls it | tests: unapproved, pending, missing, wrong-anchor, uncommitted, changed values or commit are refused before any directory or training |
| The code that runs is committed and recorded | every registered source must be clean at HEAD and equal the recorded identity | tests: dirty source, mismatched identity refused |
| Upstream code is the pinned code | commit and import-graph digests checked before import; foreign modules refused | tests |
| Valid p-values | SciPy required before a run and for verification; the engine's normal fallback is anti-conservative (t = 2.5, df = 11: 0.0295 vs 0.0124) | tests: refusal without SciPy; fallback statistics fail verification |
| Valid data | shape, dtype, values, labels and row count per partition; train/test overlap refuses the run; audits bound to the persisted arrays' hashes | tests |
| Design as registered | nested prefixes; independent replicate datasets; one seed across sizes within a replicate; fixed epochs, so updates scale with n | end-to-end test; JSON `training_policy` |
| No resume | new or empty run directory, claimed exclusively; an interrupted run is preserved and never resumed | tests |
| Results match what was trained | checkpoint SHA-256 logged at evaluation; reloaded and re-evaluated within 1e-5; engine aggregates recomputed | tests: altered, missing or incomplete checkpoints and tampered metrics are not verified |
| Statistics are well formed | finite values, p in [0, 1], Holm recomputed, complete pairing, registered baseline and replicate count | synthetic tests |
| Provenance of a finished run | the preregistration and sources are re-read from git at the recorded commits | tests: missing approval, wrong anchor or commit, changed values, dirty or altered sources, wrong upstream pin are not verified |
| A hash-only check is never "verified" | `verify --no-reevaluate` reports at most `INTEGRITY_CONSISTENT_NOT_REEVALUATED` | test |

Engine properties handled in the driver, without modifying the engine:
- it fits without validation data, so the final-epoch model is evaluated (declared in the design);
- its `test_loss` is MSE plus DBitNet's L2 penalties, so re-evaluation reports both;
- its Holm α is fixed at 0.05, so the registered α must equal it.

## 4. Tests

`audit/dataset/tests/test_d5_autond_training_data_scaling.py`, in its own process, Python 3.11 / TensorFlow 2.15.1 / Keras 2.15.0: **63 passed, 0 skipped**. Sizes are tiny: 96/192 training rows, 1,024 test rows, 3 replicates, 1 epoch, CPU. The end-to-end run uses the real DBitNet and generator, and no gate is stubbed.

All warnings are Python `DeprecationWarning`s raised inside Keras 2.15 (`keras/src/backend.py`, `randint(1, 1e9)` when seeding layer initialisers). They come from third-party code and do not affect results on Python 3.11.

## 5. Conclusion

The implementation meets the integrity requirements above as tested at tiny scale. Not yet checked:
- GPU execution, runtime (about 9.8 h for 12 replicates on one T4) and disk (about 8.5 GB) at production scale;
- the size of σ_d, which decides whether null contrasts are informative;
- exact duplicates across replicates (not audited).

GPU training is not bit-deterministic, so a re-run reproduces the procedure, not identical weights.
