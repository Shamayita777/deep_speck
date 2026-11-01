# AutoND CE3: verification report (2026-11-01, frozen candidate)

Nothing in this report is confirmatory. No CE3 production run has been made. The design is `FROZEN` with hash `849ffaa3001e0d6c953625e22fb9523ab3eba2cf1d557b0509e8ec5df76da3be`, `pending_items()` is empty, and the preregistration was rendered with approval `PENDING_REVIEW` (`d4577547…`) and then approved for this exact hash on 2026-11-01. `production_authorisation` now returns `(True, 'approved')`.

## 1. Sources CE3 depends on

| Item | Result |
|---|---|
| CE2 r3 sources | unmodified, committed at `b39958a`, pinned: adapter `2726c663`, design `1cf3ca64`, theory `1b13c89f`, verify `5eedafca`, approved preregistration `b8629e15` (APPROVED, design `3f6340e8`) |
| targets, core and upstream hash file | pinned: `upstream.py` `99d70313`, `model_binding.py` `32e96468`, `core/evidence.py` `187cb6fa`, `upstream_sha256.json` `5c9fe811` |
| CE2 production run `evidence/ce2/autond_ce2_production_20261031T155733Z` | certificate `4eaa7ce8`, SUPPORTED, 20/20 VALID; manifest `8976f00c`; verification report `e27123a5`: passed, 311 checks, EXACT. Re-verified in memory earlier: 311 checks, 0 failures, EXACT |
| CE2 twin | file `adda08a3`, weights `23772278`; all 86 CE2 evidence-manifest files re-hashed intact by the production-mode preflight |

## 2. Validation records

| Record | SHA-256 | Result |
|---|---|---|
| P1 `validation/p1/p1_record.json` | `69b430ec` | unchanged; 10^6 pairs; edges [-68, -63, -59, -55] |
| power r2 `validation/power/power.json` | `ecc61ce0` | n = 200,000, B = 5 re-derived under the corrected S1 test (`POWER_ANALYSIS.md`) |
| power r1, draft 1 (superseded, kept) | `0789c5cf`, `b14dd0c6` | not used |
| benchmark r2 `validation/benchmark/benchmark.json` | `c7d06346` | cyclic-group permutations; 50/50 fits converged at max_iter 2000, at most 21 iterations, maximum gradient 9.9e-5, no warnings; no prediction, score or accuracy computed |
| benchmark r1 (superseded, kept) | `2f8444b1` | used the superseded permutations |
| representation diagnostics `validation/representation/representation_diagnostics.json` | `ffc19d39` | published: 14 non-constant units, effective rank 8.94, participation ratio 3.64; twin: 60, 40.1, 19.1 |
| production-mode preflight (report kept outside the repository) | | **ok**: runtime pins met, dependency pins unchanged, CE2 run intact, graph binding of both models, head difference 8.4e-8 / 8.1e-8, all four validation pins ok |

## 3. Verifier

`verify.py` imports nothing from the CE3 producer modules, `targets/autond` or `core/evidence.py`. It takes every pin from the run's own `design.json`, which must hash to the manifest; in production that file must also equal the approved preregistration.

**What it recomputes independently:**
- journal replay against its own state machine;
- KFold folds;
- the cyclic permutation group, with a closure check;
- the tie-break check;
- classes, P1 edges and coverage;
- the scaler, logistic gradient and multinomial class-sum;
- predictions from the stored parameters;
- float and exact rational balanced accuracies;
- the exact S1 rank;
- exact binomial tests in rational arithmetic, Clopper–Pearson by bisection, and the fixed-sequence decision;
- the certificate and evidence manifest.

**What it shares with the producer (corrected).** The earlier statement that it "shares only TensorFlow" was wrong. It shares:

- **CE2 verifier primitives** (`experiments/autond/ce2/verify.py`, pinned):
  - `_trajectory`, `_ror`, `xdp_weight`, `_words_to_bits`;
  - `sha256_file`, `canonical_sha256`, `content_hash`, `Findings`;
  - `load_upstream`, `strict_matches_hdf5`, `weights_hash`.

  The producer's cipher comes from CE2 `adapter.py`, so these are independent of the producer's cipher. They are still CE2 code, not CE3-specific code.
- **AutoND model building and binding:** the pinned upstream `dbitnet.make_model` builder, loaded through the CE2 verifier. The producer binds with `targets/autond/model_binding.py`; the verifier binds with the CE2 verifier's strict HDF5 binding and weights hash.
- **TensorFlow/Keras 2.15 inference**, for the representations and the model scores.
- **Shared evidence infrastructure:** the `core-journal-2` journal, record caches, quarantine and staging layout that `core/evidence.py` writes, and the content-hash convention shared with CE2.
- **Optionally, scikit-learn 1.9.1** with `--sklearn-refit`.

All dependency hashes in the design are unchanged.

## 4. Tests

| Suite (environment) | Passed | Failed |
|---|---|---|
| CE3, all 122 tests (`autond-ce2-prod`: Python 3.11.14, TF 2.15.1, sklearn 1.9.1) | 122 | 0 |
| — statistics (41), target and probes (22), design and validation (16) | 79 | 0 |
| — integration with the real model and twin (37) and end-to-end outcomes (6) | 43 | 0 |
| CE2 r3 + core + ce234 authorised, unchanged (`autond-tf215-audit` venv) | 221 | 0 |
| AutoND CE1 regression, unchanged | 97 | 0 |
| pyflakes on every CE3 module, test and the CLI | clean | |

**New S1 tests.**
- Exact enumeration that the tie-broken rank is uniform and the upper-half probability is exactly 1/2. It covers every multiset of B + 1 scores from three values, for B = 1, 3 and 5, over all arrangements and all tie-break permutations.
- The same enumeration run through the producer and verifier rank functions.
- An exact demonstration that the superseded median rule is not exact (0.687 and 0.313).
- Simulations with untied, heavily tied and all-tied scores, including a chi-square check of rank uniformity.
- Group closure, the cycle structure for every remainder, producer–verifier agreement, and the cyclic-shift identity used in the proof.
- Level checks for the binomial test and the intersection-union gate.
- Zero-policy checks: zeros never make rejection easier.

**Blinding tests.**
- The benchmark's arm loop is run with every `predict`, `predict_proba`, `decision_function`, `score`, balanced-accuracy and `fit_arm` routine patched to raise, and none is called.
- A static check confirms that `benchmark.py` calls none of them.

**End-to-end outcome fixtures** (`test_end_to_end_outcomes.py`). Each runs runner → journal → finalisation → verifier, with planted representations and real data:

| Fixture | Outcome | Verified |
|---|---|---|
| SUPPORTED | | ✓ |
| POSITIVE_CONTROL_FAILED | C_task rejects, C_twin is exactly 0 in every replicate | ✓ |
| INCONCLUSIVE | S1 not rejected; the observed assignment always ranks in the lower half | ✓ |
| TARGET_DECODABLE_TWIN_UNRESOLVED | S2 exactly 0 | ✓ |
| INCONCLUSIVE | every replicate FAILED, degenerate twin | ✓ |
| PRECHECK_FAILED | | decision recomputed; the only verifier failures are the expected missing twin copy and initialised event |

EXCLUDED replicates cannot arise from finite balanced accuracies, so that state is covered at the statistics level.

**Forgeries.** There are now 11 hash-consistent forgeries, adding a swapped tie-break. They join the forged representation, the forged endpoint, the six direct-tampering cases, the fault-point resume tests, the output-directory rules and the production refusal.

**CLI smoke** (2 × 6,000 pairs, written outside the repository): **VERIFIED, 88 checks, EXACT**, and 104 checks with `--sklearn-refit`. The smoke decision POSITIVE_CONTROL_FAILED is forced: two replicates cannot reach p < 0.05. The endpoint values were not inspected.

**Power uses the production group.** A test instruments a real power replication. It checks that the B = 5 arms are exactly `probes.permutations(n, index, 5)`, the production order-6 group, that the observed labels are the identity and arm j uses y[σ^(j+1)], and that no order-22 group is built. B = 21 is evaluated only in the parametric model, from the pooled permuted-arm moments of the order-6 and order-12 groups.

**Frozen-candidate runs** (after the freeze):
- CE3: 119/119; after approval 122/122. The production-refusal test now reads four unapproved or missing preregistrations from a temporary path instead of the repository's, which is now approved.
- Regressions: CE2/core/ce234-authorised 221/221 and CE1 97/97.
- Smoke: VERIFIED, 88 checks EXACT; 104 checks with `--sklearn-refit`.
- Production-mode preflight: ok. Runtime pins are met, there are no dependency problems, the CE2 run is intact, all validation pins are ok and the head differences are 8.8e-8 and 8.0e-8.

## 5. Still open before production

- Approval: done for design `849ffaa3…`.
- Next: `--preflight`, `--smoke` and the tests, then production with `--run-all`.
