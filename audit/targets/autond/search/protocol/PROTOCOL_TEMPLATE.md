# Pre-run protocol — AutoND evolutionary input-difference search

> **Fill this in, commit it, and record the commit hash BEFORE launching a
> `--mode paper` run.** `run_search.py` refuses `--mode paper` without
> `--protocol`, and binds this file's SHA-256 into every result record.
>
> This protocol does **not** amend AutoND preregistration v1
> (plan hash `e51ece73deaa076a27cd1f490e7f4f6287f54785e932abdc1b5da49d118bf0ed`,
> file SHA-256 `0a226e5caf7ec49aa7740923558b880b17a24426c17f63cd6e13f58f0e38d9db`),
> which remains immutable and whose `scope_limits` explicitly exclude the search:
> *"a result about AutoND's shipped checkpoints is not a result about the AutoND
> SEARCH procedure, which this audit does not run."* This experiment sits
> **outside** that scope and must be reported as a separate, later addition, not
> as something v1 anticipated.

---

## 1. Identification

| Field | Value |
|---|---|
| Protocol id | `AUTOND-SEARCH-___` |
| Author | |
| Created (UTC) | |
| Git commit of this file | *(fill after committing; do not pre-date)* |
| Supersedes | — |

## 2. Target and revision

| Field | Value |
|---|---|
| Upstream | `https://github.com/Crypto-TII/AutoND` |
| Pinned commit | `8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` |
| `optimizer.py` SHA-256 | *(from the run header)* |
| Cipher module SHA-256 | *(from the run header)* |
| Cipher / scenario | `speck3264` / `single-key` |

## 3. The question this run answers

> State it as a question with a possible negative answer. A run whose result
> cannot come out "no" is not an experiment.

Suggested, to be edited rather than accepted unread:

**Does AutoND's automated search reliably return the input difference the
published Speck32/64 results depend on?**

The published pipeline's headline Speck32/64 result uses difference `0x400000`.
The search is stochastic and unseeded, so a single run establishes nothing about
reliability. The quantity of interest is the proportion of independent runs that
return `0x400000`, or a difference within the ε-close set, together with the
`highest_non_random_round` each run reports.

## 4. Estimand

| Field | Value |
|---|---|
| Primary quantity | proportion of independent search runs whose returned ε-close set contains `0x400000` |
| Experimental unit | one complete search run (not one generation, not one round) |
| Secondary | distribution of `highest_non_random_round`; distribution of returned set sizes |
| K (number of runs) | |
| Rationale for K | |

> K is a precision decision, not a budget decision. With K runs the standard
> error on a proportion near 1 is at most `0.5/sqrt(K)`; state the precision you
> need and derive K from it. Record the derivation here **before** running.

## 5. Configuration (fixed before the run)

| Parameter | Value | Source |
|---|---|---|
| `num_generations` | `50` | publication; `optimizer.py` ships `5` as a demonstration default |
| `num_samples` | `10000` | publication; `optimizer.py` ships `1000` |
| `epsilon` | `0.1` | `main.py` |
| `population_init` | `fresh` **or** `previous` — **choose one and say why** | see §6 |
| `numpy_seed` | | |
| `urandom_seed` | `null` unless you are deliberately deviating | see §7 |
| Config file SHA-256 | | |

## 6. The population-initialisation discrepancy — declare your choice

`optimizer.optimize` contains both variants, one commented out, with the
repository's own note that the shipped behaviour differs from the paper:

```python
#diffs, scores = evo(f=..., gen=None)
# The initial set of differences can be set to None, or to the differences
# returned for the previous round. We use the second option here, as opposed
# to the first (above) in the paper.
diffs, scores = evo(f=..., gen=diffs, verbose=1)
```

- `population_init="previous"` — the **shipped code's** behaviour. No patch to the
  loop. Answers: *what does the released implementation do?*
- `population_init="fresh"` — the **paper's described** procedure, via patch
  `P2-POPULATION-FRESH`. Answers: *what does the publication describe?*

**These are two different procedures. Their results are not pooled**, and
`verify_outputs.py` refuses an output directory that mixes them. If you intend to
compare them, say so here and give each its own directory and its own K.

Chosen: `____________`   Reason: ____________

## 7. Randomisation and reproducibility

The search draws from two independent sources:

- `numpy.random` — initial population, mutation selection, mutated bit positions. **Seedable.**
- `os.urandom` — the plaintexts and keys sampled each round. **Not seedable** without substituting the function.

Seeding numpy alone therefore makes a run **partially** reproducible. Do not
describe such a run as reproducible. Patch `P4-URANDOM` (`urandom_seed` in the
config) substitutes a seeded generator and makes a run bit-reproducible, but it is
a **deviation from the target's behaviour** and every result produced with it
carries that flag.

Declared position: ____________

## 8. Stopping and exclusion rules

| Rule | Value |
|---|---|
| Stopping condition | `optimize()`'s own: best score `< T = 0.05` (hardcoded upstream) |
| Wall-clock cap per run | |
| What counts as a failed run | |
| Are failed runs excluded? | **Default: no.** A run that crashed is reported, not dropped. |
| Maximum total runs | |

> A run may not be discarded after its result is seen. If a run is excluded, the
> reason must be technical, stated here in advance, and the excluded record kept.

## 9. Decision rule — written before the data exist

| Outcome | Interpretation |
|---|---|
| `0x400000` returned by ___ of K runs or more | |
| returned by fewer than ___ of K | |
| `highest_non_random_round` varies by more than ___ | |

> Fill these in now. A threshold chosen after seeing the proportion is not a
> decision rule.

## 10. What this run cannot establish

Fixed; do not delete.

- It does **not** reproduce AutoND's training pipeline, and produces no distinguisher.
- It says nothing about the shipped checkpoints, their accuracy, or their provenance.
- It is **not** evidence for II-4 or II-5, which require a completed
  `FORMAL_IMPLEMENTATION_COMPARISON` under `audit/implementation/stages.py`.
- It is **not** D4, D5 or CE1, none of which this run touches.
- A result at `population_init="previous"` describes the released code, not the
  publication's procedure, and vice versa.
- Unless `P4-URANDOM` was applied, the run is not bit-reproducible.

## 11. Evidence to preserve

`MANIFEST.json`, every `run_*/result.json`, every `run_*/generations.jsonl`, the
upstream-written `*_best_weighted_differences.csv` and search log, the config file,
this protocol, `verification.json` from `verify_outputs.py`, and the environment
record (`pip freeze`, `python -V`, CPU model).

## 12. Deviations encountered during execution

> Append only. Never edit a row after the fact; add a new one.

| UTC | Deviation | Why | Effect on interpretation |
|---|---|---|---|
| | | | |

## 13. Sign-off

Frozen by: ____________  Date (UTC): ____________

Git commit containing this frozen file: ____________

> Record the commit **after** committing, from `git rev-parse HEAD`. Do not write
> a commit hash you have not read from git, and do not back-date.
