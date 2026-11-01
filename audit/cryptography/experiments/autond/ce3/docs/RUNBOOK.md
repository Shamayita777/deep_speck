# AutoND CE3: runbook

**Where to run:** from the repository root (the directory that contains `audit/`), in the pinned environment: Python 3.11, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4, pandas 2.2.2, SciPy 1.11.4, scikit-learn 1.9.1, threadpoolctl 3.7.0. Locally that is `~/.venvs/autond-ce2-prod`.

In zsh, a function avoids word-splitting trouble:

```
PY=~/.venvs/autond-ce2-prod/bin/python
ce3() { PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m audit.cryptography.kaggle.autond_ce3_kaggle "$@" }
```

## 1. Validation records (already done; never repeat P1)

| Step | Command | Record |
|---|---|---|
| P1 | `ce3 --p1 --out audit/cryptography/experiments/autond/ce3/validation/p1` | `p1_record.json` `69b430ec…`, refuses to run twice |
| verify P1 | `ce3 --verify-p1 --report <new.json>` | 7/7 checks |
| power | `ce3 --power --workers 7 --report <new.json>` | `validation/power/power.json` (r2); r1 kept as `power_r1_superseded.json` |
| benchmark | `ce3 --benchmark --report <new.json>` | `validation/benchmark/benchmark.json` (r2); r1 kept as `benchmark_r1_superseded.json` |
| representation diagnostics | `ce3 --representation-diagnostics --report <new.json>` | `validation/representation/representation_diagnostics.json` `ffc19d39…` |
| model check | `ce3 --check-model --report <new.json>` | read-only |

The power run and the benchmark depend on the permutation construction, so both were redone after the S1 correction. The benchmark is run after the power selection because it uses the selected n and B.

## 2. Checks before freezing

```
ce3 --preflight --report /tmp/ce3_preflight.json        # must print ok: true; it hashes all CE2 evidence
ce3 --smoke --out /tmp/ce3_smoke                         # 2 x 6000 pairs, then the verifier must say VERIFIED
"$PY" -B -m pytest -q -p no:cacheprovider audit/cryptography/experiments/autond/ce3/tests
```

## 3. Freezing and approval (yours alone)

All four steps are done (2026-11-01).

1. **Done.** In `design.py`, `status`, `main_pairs_status` and `permutations_status` are `FROZEN`. The design hash is `849ffaa3001e0d6c953625e22fb9523ab3eba2cf1d557b0509e8ec5df76da3be`.
2. **Done.** The preregistration was re-rendered with `ce3 --render-preregistration`, and `preregistration/preregistration.json` (`d4577547…`) has pending items `[]`.
3. **Done.** The tests check the frozen state and that production is refused unless the approval block is complete for the exact hash.
4. **Done.** I approved the design: `status: APPROVED`, `approved_design_sha256` the hash above, `approved_utc` 2026-11-01T15:34:29Z. Before `--run-all`, run `--preflight`, `--smoke` and the tests.

## 4. Production

```
ce3 --run-all  --out audit/cryptography/experiments/autond/ce3/evidence/autond_ce3_production_<utc>
ce3 --resume   --out <same dir>          # after any interruption
ce3 --finalize --out <same dir>
ce3 --verify   --out <same dir>          # optional: --sklearn-refit
```

**Rules.**
- Production output must be under `experiments/autond/ce3/evidence/`.
- A directory whose initialisation stopped before `run_manifest.json` is abandoned. Start a new one.
- FAILED replicates are never replaced. With any non-VALID replicate the result is INCONCLUSIVE.
