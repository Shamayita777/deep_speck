# AutoND CE3: representation decodability

CE3 asks whether the AutoND CE2 target, coarsened into five frozen whole-value classes, is linearly decodable from the 64-unit penultimate ReLU layer of the published AutoND SPECK32/64 round-8 DBitNet. Decodability is tested beyond a within-replicate permuted-label control and beyond the fixed untrained twin of the CE2 production run, after the probe has shown that it recovers the genuine-versus-random task.

**Status:** design `AutoND-CE3-design-2026-11-r1`, `FROZEN`, design hash `849ffaa3…`; preregistration `APPROVED` for that hash (2026-11-01). No production run yet.

**Production is refused** until I fill in the approval block for this exact design hash.

No confirmatory run has been made.

## Layout

| Path | Contents |
|---|---|
| `design.py` | the single source of the design, its pins and the authorisation check |
| `adapter.py` | genuine and calibration data (via CE2 r3), frozen classes, graph binding of layer 32, extraction, head consistency |
| `target_bins.py` | whole-value partition rule, class assignment, coverage |
| `probes.py` | label-independent folds, odd-B permutations, the scaler and logistic probe, the convergence gate |
| `analysis.py` | per-replicate input checks, all arm fits, endpoints from committed predictions |
| `statistics.py` | exact sign tests, Clopper–Pearson, the fixed-sequence decision |
| `runner.py` | preflight, initialisation, journalled state machine, resume and quarantine, finalisation |
| `verify.py` | independent verifier, from raw pairs to the decision |
| `p1.py`, `benchmark.py`, `power.py` | P1 pilot, blinded benchmark, pipeline-faithful power |
| `diagnostics.py` | per-unit variance, sparsity, rank and spectrum of the published and twin representations on P1 |
| `provenance.py` | the CE2 source run and dependency pins |
| `preregistration/preregistration.json` | approved preregistration for design `849ffaa3…` |
| `validation/p1/`, `validation/power/`, `validation/benchmark/`, `validation/representation/` | non-evidentiary validation records, pinned by hash |
| `evidence/` | the only place production output may be written |
| `tests/` | isolated and integration tests |
| `../../../kaggle/autond_ce3_kaggle.py` | the command-line entry point |

**Read next:** `METHODOLOGY.md`, `POWER_ANALYSIS.md`, `VERIFICATION_REPORT.md`, `RUNBOOK.md` and `GOHR_COMPARISON.md`.

## What CE3 uses from elsewhere (read-only, hash-pinned)

- **CE2 r3:** `adapter.py`, `design.py`, `theory.py`, the approved preregistration, and `verify.py`.
- **Target and core:** `targets/autond/upstream.py` and `model_binding.py`; `core/evidence.py`; `upstream_sha256.json`.
- **CE2 production run:** `evidence/ce2/autond_ce2_production_20261031T155733Z`, for its certificate, verification report and untrained twin.

**What the CE3 verifier shares with the producer.** It is independent in its statistics, probe checks, permutation and fold reconstruction and journal replay, but it is not independent of everything:

- **CE2 verifier primitives** (`ce2/verify.py`): the SPECK trajectory, xdp+ weights, word-to-bit conversion, file, canonical and content hashing, `Findings`, upstream loading, strict HDF5 binding and the weights hash.
- **Model building and binding:** the pinned upstream `dbitnet.make_model` builder. The producer binds through `targets/autond/model_binding.py`; the verifier binds through the CE2 verifier's `strict_matches_hdf5` and `weights_hash`.
- **TensorFlow/Keras 2.15 inference**, for both representations and the model scores.
- **Evidence infrastructure:** it does not import `core/evidence.py`, but it reads the `core-journal-2` journal, record caches and quarantine layout that `core/evidence.py` writes.
- **Optionally scikit-learn 1.9.1**, with `--sklearn-refit`.

Nothing outside this directory and `kaggle/autond_ce3_kaggle.py` was added or changed.
