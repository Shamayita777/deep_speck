# Changelog — r3c → r3d

**Registered sample size raised from n = 5 to n = 10. The equivalence
margin is UNCHANGED at Δ = 0.10 × (p_published − 0.5), and a test
asserts it.** The plan is still unfrozen and no confirmatory evaluation
has been run.

Operator decision, recorded verbatim in the request that produced this
revision: *"keep the margin at 10%; change the planned sample size to
n = 10 — before freezing"*, on the grounds that the margin must be
justified by the scientific question rather than widened to make the
experiment easier to pass, and conditional on the revised design's power
being computed by the plan's own registered procedure — including the
multiplicity correction — and passing its tests before the freeze.

The plan hash changed because the `PLAN` object changed:

```
r3c  b93fa50bb0914b17dc3423480466f18c6340225486c2dcbcb29c911d87207d40
r3d  f20cd9b12909d21dffd22aed83a6f303e8f445412490708f1f01c79a3c264bb7
```

Freeze against the **r3d** hash. A sidecar written against the r3c hash
would be rejected by `is_frozen()`.

## The finding that shaped this revision

`n = 10` is not a doubled constant, and could not be.

`nEval = 5` is a **module-level constant** of the target, at
`eval_nets.py:22`, and is **not a command-line argument**. One invocation
therefore produces exactly five accuracies. Reaching ten by editing that
constant would modify the pinned upstream repository, which this audit
forbids absolutely.

So the design invokes `eval_nets.py` **twice per checkpoint** with
distinct `--add_str` values and pools the two NPZs into one sample of
ten. Verified against the pinned source (`eval_nets.py`, sha256
`838587127c08834f6f9ecd7b27fc578da3942141fdcf5d9c513d46c646d63048`,
matching the recorded digest), not assumed:

| line | what it establishes |
|---|---|
| `:22` | `nEval = 5`, a module constant, not an argument |
| `:208-209` | each loop iteration draws a **fresh** 10⁶-sample dataset via `make_train_data` from `os.urandom` — so sets are independent within an invocation |
| `:27-42` | `evaluate_X_Y` is a deterministic function of `(model, X, Y)` — so a given dataset yields the same accuracy in either process |
| `:152` | the NPZ name is derived from `--add_str`, so distinct suffixes stop the second invocation overwriting the first |

The target is not modified, no default is changed, and the two
invocations are the same command but for the output-name suffix — which
a test asserts by stripping `--add_str` and comparing.

## Power, computed by the plan's own procedure

`margin_sensitivity()` recomputes on every call at `N_TEST_SETS`;
nothing below is transcribed.

| fraction | Δ | power @ α | power @ α/3 |
|---|---|---|---|
| 0.02 | 0.00029 | 0.0244 | 0.0030 |
| 0.05 | 0.00072 | 0.8620 | 0.6050 |
| **0.10 (registered)** | **0.00144** | **1.0000** | **0.9998** |
| 0.15 | 0.00216 | 1.0000 | 1.0000 |
| 0.20 | 0.00288 | 1.0000 | 1.0000 |

**Registered margin, conservative Holm step: 0.7965 → 0.9998.**

α/3 remains labelled a planning approximation, distinct from the
registered step-down, whose actual per-comparison threshold depends on
the observed ordering and is unknowable in advance.

The n = 5 figures are **retained, not deleted**, in
`POWER["superseded_design"]`: the margin did not change, so they remain
the honest record of what the earlier design could detect.

### Why n rose, stated as robustness rather than a point estimate

The planning SD carries 4 df, so its own 95% interval spans roughly
0.6×–2.9× the point estimate. The fragility, not the point estimate, is
the argument:

| true SD | n = 5 | n = 10 |
|---|---|---|
| as planned | 0.797 | 0.9998 |
| 1.5× | 0.339 | 0.996 |
| 2.0× | 0.129 | 0.905 |

At 2.9× — the pessimistic end — no feasible n rescues the design, and the
honest outcome is `INDETERMINATE`, which the decision rule returns rather
than defaulting to agreement. Recorded in `POWER["robustness_note"]`.

`POWER["power_does_not_justify_a_margin"]` is unchanged and still
asserted: raising n must not turn power into a margin argument.

## Files

| file | change | why |
|---|---|---|
| `preregistrations/autond_ii_v2.py` | modified | `SETS_PER_INVOCATION`, `INVOCATIONS_PER_CHECKPOINT`, `N_TEST_SETS`, `INVOCATION_SUFFIXES`; `DESIGN` gains `how_n_was_raised`, `why_two_invocations_and_not_a_changed_constant`, `pooling_rule`, `pooling_assumption`; SE/df prose → `sqrt(10)`/`df = 9`; exclusion and missing-data rules; `POWER` gains `superseded_design` and `robustness_note`; `power_at` and `margin_sensitivity` default to `N_TEST_SETS` |
| `evaluation_gate.py` | modified | two invocations per checkpoint with registered suffixes; per-invocation journal, log and NPZ paths; checkpoint re-hashed before **every** invocation |
| `evaluation_assemble.py` | modified | journals keyed by invocation; pooling step with independence checks |
| `evaluation_results.py` | modified | `EXPECTED_SETS_PER_INVOCATION` / `EXPECTED_N_POOLED`; `assert_matches_preregistration()`; `npz`/`npz_sha256` accept per-invocation lists; raw-NPZ cross-check verifies the **concatenation**; schema `v1` → `v2` |
| `tests/test_r3d_sample_size.py` | **NEW** | 35 tests for the design change |
| `tests/portable.py` | modified | `staged_run_dir` defaults to the registered design, with knobs to produce each malformed shape |
| `tests/test_r3c_statistics.py` | modified | `synth()` generalised to any n (it previously accepted `n` and ignored it); fixtures anchored to `N_TEST_SETS` |
| `tests/test_corrections.py`, `tests/test_r3_corrections.py` | modified | decision fixtures extended to ten values at the same SD scale; command count and NPZ check anchored to the design |
| `STATISTICAL_NOTE.md` | modified | design, power and robustness sections |
| `CHANGELOG_R3D.md` | **NEW** | this file |
| `SHA256SUMS.txt` | regenerated | |

`audit/common/equivalence.py` is **unchanged**: the generic statistics
needed no modification, because `decide()` already took n from the data
and df from the sample. That is the framework behaving as intended — a
target's design change did not reach the generic layer.

## What the new tests pin

- n = 10, derived from the two constants so they cannot disagree
- Δ still 0.10, and all three comparison margins unchanged
- the gate emits 2 × 2 = 4 invocations, with registered, distinct
  suffixes and therefore distinct NPZ paths
- the two invocations differ **only** in `--add_str`
- journal and log filenames are per-invocation, so the second cannot
  overwrite the first's record
- the validator's duplicated constants agree with the plan, and a
  simulated divergence raises
- **one surviving invocation is refused**, not analysed as an n = 5
  result
- byte-identical NPZs, repeated digests, colliding suffixes,
  unregistered suffixes and differing checkpoint bytes are each refused
- pooled accuracies equal the concatenation of the raw NPZs in
  registered order — pooling is checked, not assumed

Synthetic fixtures throughout. No real experimental result is used, and
no threshold is tuned against one.

## Honest limitations, recorded in the plan

- **Pooling is a recorded assumption, not a tested hypothesis.** With
  five sets per invocation a between-invocation effect could not be
  detected with useful power, so the design does not claim to have
  excluded one. The source-level argument above is why it is reasonable;
  the per-invocation accuracies are retained individually so a reader can
  inspect the two groups directly.
- The two per-invocation means are reported **for inspection only** and
  are not compared as a test, because no such test was registered.
- Ten sets depart from AutoND's own five-set protocol. The reference is a
  fixed published constant in the TOST either way, so more samples sharpen
  only our estimate — but the departure is real and is stated rather than
  presented as a closer reproduction.
- A failed invocation costs five sets at once. The remedy is to rerun the
  whole checkpoint into a fresh run directory, with the failure recorded;
  a single surviving invocation is never analysed.

## Preserved, verified

- Evidence untouched: R1 manifest/result/logs, checkpoints, training
  bundle, census, load-verification.
- The R1 findings stand: two returned differences against the README's
  one; `highest_non_random_round = 5` against 7 and 6.
- Freeze-gate behaviour unchanged; no integrity, provenance, hash,
  source-binding or preregistration gate was weakened.
- Superseded logs retained, as in r3c.
- No training, search or evaluation was run.

## Freeze procedure

Unchanged except the hash. Install r3d, run the suite, then follow the
procedure at the end of `CHANGELOG_R3C.md`, with
`$PLAN_HASH` reading **`f20cd9b12909d21d…`**.
