# AutoND evolutionary input-difference search — reproduction harness

Runs the **search** stage of AutoND's pipeline against the pinned upstream source
without modifying it, and records enough about the execution to defend the result
later. It does not train anything.

Target: `https://github.com/Crypto-TII/AutoND` at
`8d66e0a43a9e6fbbb3624d68296c1b77aa2da779`.

**You run the experiment. I did not.** Everything below marked *performed during
packaging* was executed while preparing this package; everything marked *you must
do* has not been run by anyone yet. No experimental result for Speck32/64 or
Simon32/64 appears anywhere in this package, because none was produced.

---

## 1. The original implementation

### 1.1 Entry point and call graph

The published entry point is `main.py`, which runs the search and then trains:

```
main.py  __main__
  └── findGoodInputDifferences(cipher_name, scenario, output_dir, epsilon=0.1)   main.py:35
        └── optimizer.optimize(plain_bits, key_bits, encrypt,
                               nb_samples=NUM_SAMPLES, scenario, log_file, epsilon)
              ├── while True:                                      # one iteration per round
              │     ├── urandom  -> keys0, pt0                     # fresh each round
              │     ├── encrypt(pt0, keys0, current_round) -> C0
              │     ├── evo(f=lambda x: evaluate_multiple_differences(...),
              │     │        num_bits, L=32, gen=diffs, verbose=1)
              │     │     ├── f(gen)                               # score the population
              │     │     └── for generation in range(n):          # n = NUM_GENERATIONS
              │     │           ├── kids = pairwise XOR of survivors
              │     │           ├── mutate, de-duplicate against `explored`
              │     │           └── f(kids)                        # score the new candidates
              │     └── break when scores[-1] < T   (T = 0.05, hardcoded)
              ├── re-score all differences at every round 1..current_round-1
              ├── cumulative and round-weighted rankings
              └── return (epsilon-close differences as hex, highest_non_random_round)
  └── trainNeuralDistinguishers(...)                                # TRAINING — not run here
```

This harness calls `optimizer.optimize` **directly**, bypassing `main.py`. That is
what keeps TensorFlow out of the process (§4).

### 1.2 Paper versus shipped code

| Setting | Paper | Shipped `optimizer.py` | Source location | Reachable without editing upstream? |
|---|---|---|---|---|
| Generations | 50 | **5** | `NUM_GENERATIONS` line 12, commented *"50 in the paper, set to 5 here for demonstration"* | **No** — see §1.3 |
| Samples for the bias score | 10⁴ | **10³** | `NUM_SAMPLES` line 13, commented *"10**4 in the paper"* | Yes — `nb_samples` is a parameter of `optimize()` |
| Population init per round | fresh (`gen=None`) | **previous round's survivors** (`gen=diffs`) | `optimize()`, with the repository's own comment | Only via a declared patch |
| Population size L | 32 | 32 | `evo(..., L=32)` | n/a |
| Bias threshold T | — | 0.05 | `optimize()`, hardcoded local | n/a |
| ε for the close set | 0.1 | 0.1 | `main.findGoodInputDifferences` | Yes — parameter |
| Seeding | — | **none anywhere** | `np.random.*`, `os.urandom` | partially — §1.5 |

The first two are **disclosed demonstration defaults**: the code says so in its own
comments. They are a reproducibility gap, not a concealed discrepancy, and should
be written up that way.

### 1.3 The reproducibility trap — verified, not assumed

```python
NUM_GENERATIONS = 5
def evo(f, n=NUM_GENERATIONS, num_bits=32, L=32, gen=None, verbose=0):
```

`n` is bound to **5 at import time**. `optimize()` calls `evo(...)` without passing
`n`. So:

```python
optimizer.NUM_GENERATIONS = 50       # has NO effect
optimizer.evo.__defaults__[0]        # still 5
```

*Performed during packaging:* confirmed on the pinned checkout under Python 3.11.17
/ NumPy 1.26.4 — after assigning `NUM_GENERATIONS = 50`, `evo.__defaults__[0]`
remained `5`, and `optimize`'s `nb_samples` default remained `1000`.

Anyone who edits the constant at the top of the file gets 50; anyone who sets it
programmatically gets 5 **silently**. This harness rebinds the default explicitly
and records it as patch `P1-GENERATIONS`. A test asserts the trap still exists, so
if upstream ever changes how `evo` binds `n`, the patch is revisited rather than
silently misapplied.

### 1.4 The population-initialisation discrepancy (`gen=diffs`)

`optimize()` carries both variants, one commented out, with the repository's note:

```python
#diffs, scores = evo(f=..., gen=None)
# The initial set of differences can be set to None, or to the differences
# returned for the previous round. We use the second option here, as opposed
# to the first (above) in the paper.
diffs, scores = evo(f=..., gen=diffs, verbose=1)
```

**Implication.** The shipped search is not the search the paper describes. Seeding
round *N+1* from round *N*'s survivors is a warm start: it concentrates the search
around differences that were good at a lower round, which plausibly helps it reach
a high round quickly and equally plausibly narrows exploration. Which effect
dominates is an empirical question this harness can answer — run both and compare —
but it cannot be settled by reading the code, and **neither variant is "correct"**.
A result must name which one produced it. `verify_outputs.py` refuses an output
directory that mixes them.

This is a declared-versus-realised finding for **II-3**, establishable by source
citation alone, with no run at all.

### 1.5 Randomisation

| Source | Used for | Seedable |
|---|---|---|
| `numpy.random` | initial population, mutation selection, mutated bit positions | yes (`numpy_seed`) |
| `os.urandom` | plaintexts and keys, resampled every round | no, without substituting the function |

Seeding numpy makes a run **partially** reproducible and nothing more. The optional
`urandom_seed` substitutes a seeded generator (patch `P4-URANDOM`), making a run
bit-reproducible at the cost of **deviating from the target's behaviour**. It is off
by default and every result record states which applies.

### 1.6 Running the original faithfully

The upstream way is `python main.py speck3264 single-key -o results`, which runs the
search **and then trains a distinguisher for every difference found**. That training
is the expensive part and you have not authorised it. This harness runs the search
alone, by calling `optimizer.optimize` directly with `nb_samples` passed explicitly
and the generation default rebound.

---

## 2. Installation

### 2.1 Where to put it

```bash
cd ~/Documents/deep_speck
unzip -l /path/to/autond_search_harness.zip
mkdir -p /tmp/autond_search && unzip -q /path/to/autond_search_harness.zip -d /tmp/autond_search
( cd /tmp/autond_search && sha256sum -c SHA256SUMS.txt )     # verify before installing
cp -Rn /tmp/autond_search/audit/. audit/                      # -n never overwrites
```

Destination: `~/Documents/deep_speck/audit/targets/autond/search/`. Every path in
the archive is already its destination path. `cp -n` will not overwrite; to see
whether anything already existed with different content:

```bash
for f in $(cd /tmp/autond_search && find audit -type f); do
  cmp -s "/tmp/autond_search/$f" "$f" || echo "DIFFERS (not installed): $f"
done
```

### 2.2 Environment

**Use a separate environment from the TF 2.15 one.** This experiment needs no
TensorFlow, and installing it would only risk the `verify_no_tensorflow()` refusal.

```bash
cd ~/Documents/deep_speck
python3.11 -m venv ~/.venvs/autond-search       # or: uv venv --python 3.11 ~/.venvs/autond-search
source ~/.venvs/autond-search/bin/activate
pip install -r audit/targets/autond/search/requirements.txt
python -c "import numpy,pandas;print(numpy.__version__,pandas.__version__)"   # 1.26.4 2.2.3
```

Python 3.10 or 3.11. **NumPy must be 1.26.x** — see §4.2.

### 2.3 Verify the target

```bash
cd ~/Documents/AutoND-upstream
git rev-parse HEAD            # must print 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779
git status --short            # .DS_Store is harmless; anything else, investigate
shasum -a 256 optimizer.py ciphers/speck3264.py
```

The runner re-checks the commit itself and **refuses** to run against any other
revision.

---

## 3. Running it

### 3.1 R1 — the README's documented demonstration. Do this first.

**Claim under test:** the README states that `python main.py`, unmodified, returns
`Found 1 0.1-close differences: ['0x400000']`. R1 checks exactly that.

**R1 applies no behaviour-changing patch.** `num_generations` and `num_samples` are
both `null`, so `evo` keeps its import-time default of 5 and `optimize()` is called
without `nb_samples` — precisely as `main.findGoodInputDifferences` calls it. The
only patches applied are the two pass-through observers.

```bash
cd ~/Documents/deep_speck
source ~/.venvs/autond-search/bin/activate
python audit/targets/autond/search/run_search.py \
  --config audit/targets/autond/search/config/r1_readme_demo.json \
  --autond-root ~/Documents/AutoND-upstream \
  --output-dir ~/Documents/autond_search_runs/r1_$(date -u +%Y%m%dT%H%M%SZ)
```

**NOT YET EXECUTED.** No one has run this. Compare its `returned_differences_hex`
against the README's `['0x400000']`, and report `highest_non_random_round` as an
observation — the README contradicts itself on that value (prose says 7, its own log
sample and training output imply 6; see `PAPER_CODE_README_COMPARISON.md` §2.1).

The run is also your timing measurement for scaling to R2.

### 3.1b (superseded) Timing benchmark

```bash
cd ~/Documents/deep_speck
source ~/.venvs/autond-search/bin/activate
python audit/targets/autond/search/run_search.py \
  --config audit/targets/autond/search/config/benchmark_speck3264.json \
  --autond-root ~/Documents/AutoND-upstream \
  --output-dir ~/Documents/autond_search_runs/benchmark_$(date -u +%Y%m%dT%H%M%SZ)
```

This runs at the **shipped demonstration settings** (5 generations, 10³ samples).
The mode is tagged `BENCHMARK` in every record, the summary prints a reminder, and
`verify_outputs.py` refuses to pool it with a paper-configured run. **It is a
timing measurement on your machine, not a reproduction of anything.**

Use its `elapsed_seconds` to extrapolate, and treat the extrapolation as rough: the
work per round scales with `num_samples` and with the candidate count, the candidate
count grows with generations, and the number of rounds is data-dependent. Scale up
in one step first — `--runs 1` with generations 50 and samples 10³ via a copied
config in `benchmark` mode — before committing to the full configuration.

### 3.2 Freeze the protocol

```bash
cp audit/targets/autond/search/protocol/PROTOCOL_TEMPLATE.md \
   audit/targets/autond/search/protocol/AUTOND-SEARCH-001.md
# edit: K and its derivation, population_init and why, decision rule, exclusions
git add audit/targets/autond/search/protocol/AUTOND-SEARCH-001.md
git commit -m "Freeze AutoND search protocol AUTOND-SEARCH-001"
git rev-parse HEAD                                   # record this in the file's §13, then amend
shasum -a 256 audit/targets/autond/search/protocol/AUTOND-SEARCH-001.md
```

Record the commit hash you actually read from git. Do not write one you have not
read, and do not back-date.

### 3.3 R2 — the paper's parameters, for what the code exposes

```bash
python audit/targets/autond/search/run_search.py \
  --config audit/targets/autond/search/config/r2_paper_parameters.json \
  --autond-root ~/Documents/AutoND-upstream \
  --protocol audit/targets/autond/search/protocol/AUTOND-SEARCH-001.md \
  --output-dir ~/Documents/autond_search_runs/r2_$(date -u +%Y%m%dT%H%M%SZ) \
  --runs 10 \
  2>&1 | tee ~/Documents/autond_search_runs/r2.log
```

R2 **requires** `--protocol`, and the runner refuses a config carrying the R2 label
whose settings fall below the publication's 50 generations and t = 10⁴.

**R2 IS NOT A FULL PAPER-FAITHFUL REPRODUCTION, and must not be reported as one.**
The config's `_deviations` field lists what still differs, and the runner prints the
reminder after every R2 run:

| Paper §5.2 / Algorithm 1 | R2 as executed | Why |
|---|---|---|
| `T_b = 0.01` | **0.05** | hardcoded at `optimizer.py:138`, no parameter; reaching 0.01 needs editing upstream |
| generational replacement | **elitist** | `evo()` sorts parents and children together |
| `M = 1` | **99/100** | `randint(0,100) > 0` |
| no de-duplication, no early exit | **both present** | additions in `evo()` |
| fresh population per round | `previous` by default | set `population_init` to `fresh` to follow the paper, and record which you used |

R2 is therefore *"the paper's generation count and sample count, with the released
code's threshold, selection rule and mutation rate"*. Say that, not "paper-faithful".

### 3.4 Interruption and resume

Resume granularity is **one complete run**. A run interrupted mid-search cannot be
continued — the search holds its population in memory and upstream writes no
checkpoint — so that run index restarts from scratch.

```bash
python audit/targets/autond/search/run_search.py ... --resume
```

`--resume` skips indices whose `result.json` exists and says `COMPLETED`; a torn or
failed record is rerun. Ctrl-C stops after the current run, leaves its partial
record, and prints where it is.

---

## 4. Compute

### 4.1 No GPU, no TensorFlow — established from source and re-checked at runtime

`optimizer.py` imports `numpy`, `os`, `random`, `sys`, `copy`, `types`, `timeit`,
`pandas`. It does **not** import `main`, so it does not transitively import
`train_nets` or TensorFlow.

*Performed during packaging:* importing `optimizer` plus `ciphers.speck3264` left
`tensorflow`, `keras` and `main` all absent from `sys.modules`; the only heavy
dependencies pulled in were `numpy` and `pandas`.

`search_runner.verify_no_tensorflow()` re-checks this at the start of every run and
**refuses** if TensorFlow was imported, so the claim is re-established on your
machine rather than inherited from this file. The search is pure CPU: vectorised
cipher evaluations and bias-score averaging.

**No runtime estimate is offered.** I have not run the search on Speck32/64 at any
configuration, and a figure extrapolated from a toy fixture would be worthless.
Measure it with §3.1.

The training stage that follows the search in `main.py` is the GPU-bound part. It is
**not run** by this harness, and you have not authorised it.

### 4.2 NumPy 1.26 is mandatory

*Performed during packaging:* on NumPy 2.5.3, `import ciphers.speck3264` raises

```
OverflowError: Python integer 32768 out of bounds for uint8
  ciphers/speck3264.py:103 in convert_from_binary
    X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
```

The module calls `check_testvectors()` at import time, and NEP 50 integer promotion
rejects the expression. The cipher cannot be imported at all on NumPy 2. The runner
refuses early with this explanation rather than letting you discover it mid-setup.

---

## 5. Outputs

```
<output-dir>/
  MANIFEST.json                      sha256 of every file produced
  run_000/
    result.json                      config, patches, source binding, timings,
                                     returned differences, reproducibility statement
    generations.jsonl                one row per scoring call and per evo entry/exit
    speck3264_single-key             upstream's own search log
    speck3264_single-key_best_weighted_differences.csv    upstream's ranked CSV
  run_001/ ...
```

### Verification

```bash
python audit/targets/autond/search/verify_outputs.py \
  ~/Documents/autond_search_runs/paper_fresh_<stamp> \
  --json ~/Documents/autond_search_runs/paper_fresh_<stamp>_verification.json
```

Checks: every manifest hash recomputes; no file present that the manifest omits;
each run's `generations.jsonl` sequence is contiguous and matches the count in
`result.json`; every run names the pinned commit; TensorFlow absent; no mixing of
modes or of `population_init`. Exit 0 only if all pass.

It verifies **integrity**, not science. Whether the search reproduced the
publication is a judgement against your frozen protocol.

### Reading the result

- `highest_non_random_round` — the last round at which the best bias score cleared `T = 0.05`.
- `returned_differences_hex` — the ε-close set. **The question is whether `0x400000` is in it, and in how many of your K runs.**
- `elapsed_seconds` — wall clock for that run.
- `patches[].changes_behaviour` — which patches altered what was computed. `P1-GENERATIONS` always will; `P2-POPULATION-FRESH` and `P4-URANDOM` only if you selected them.
- `reproducibility.bit_reproducible` — `false` unless you set both seeds.

### Your own logs and hashes

```bash
cd ~/Documents/autond_search_runs/<stamp>
python -V > environment.txt
pip freeze >> environment.txt
( sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | head -20 ) >> environment.txt
find . -type f ! -name SHA256SUMS.txt | sort | xargs shasum -a 256 > SHA256SUMS.txt
```

---

## 6. Reporting failures and negative results

Report them. The experiment is designed so that a negative answer is informative.

- **A run crashes.** Its `result.json` holds the exception and traceback. Keep it, note it in protocol §12, rerun that index with `--resume`. Do not delete it.
- **`0x400000` is not returned.** That is a *result*, not a failure — it bears directly on the automation claim. Report the proportion with a confidence interval, and the differences that were returned instead.
- **Runs disagree with each other.** Expected: the search is unseeded. That variability is the measurement.
- **Deviation from the protocol.** Append a row to §12 with the UTC time, what changed, why, and the effect on interpretation. Never edit an earlier row.
- **Never** adjust K, ε or the decision rule after seeing results.

---

## 7. Separation from the rest of the audit

| This experiment | Relationship |
|---|---|
| **II-4 / II-5** | **No contribution.** `audit/implementation/stages.py` lets only a completed `FORMAL_IMPLEMENTATION_COMPARISON` make II-4 or II-5 `COMPLETED`. This is not one. It does not alter the frozen II-4 design, its arms, its block count or its training budget. |
| **D4 / D5** | Untouched. No dataset is perturbed or scaled; no training occurs. |
| **CE1** | Untouched. No labels are permuted; no model is trained. |
| **October 12 staged training** | **Not rerun, not replaced.** That experiment stands on its own evidence bundle. This one produces no checkpoint and no accuracy. |
| **Preregistration v1** | Unchanged. v1 explicitly excludes the search; this experiment is *outside* its scope and must be reported as a later addition, never as something v1 covered. |

Training counts for II-4, D4–D5 and CE1 are **unchanged by this experiment**, which
performs zero training runs.

## 8. Integrating the evidence

1. File the bundle under `audit/targets/autond/evidence/search/<protocol-id>/`.
2. In the Implementation Integrity certificate, record the `gen=diffs` discrepancy and
   the demonstration-default gap as **II-3 source-inspection findings**, citing
   `optimizer.py` by line. These need no run.
3. Record the search-reliability result under the **successor** preregistration, in
   Experimental Validity or Cryptographic Evidence — **not** Implementation
   Integrity, and **not** under v1.
4. In the manuscript, state the scope boundary explicitly: the audit covers the
   shipped checkpoints and the training pipeline; the search was examined
   separately, later, under its own protocol. Give the reason as **scope**, not
   compute — the search needs no GPU and claiming otherwise would be false.
5. Keep v1's `scope_limits` quoted verbatim so a reader can see the boundary was
   drawn in advance, not after the fact.

---

## 9. R3 — not implemented, and a scope decision for you

Algorithm 1 as published — generational replacement, no de-duplication, no early
exit, fresh population each round — is **not implemented here and must not be**.
Building it would mean writing from the paper while this package's author has
already read `evo()`, which is exactly the contamination II-2 exists to avoid.

**This is a scope decision, not a task.** If you want the paper's algorithm, it
belongs in the independent reimplementation workstream, written by someone working
from Algorithm 1 alone. Note the consequence in advance: a faithful II-2
implementation will differ from upstream *by construction* on the five algorithmic
points above, so that divergence is a known property of the design, not a finding to
be discovered later.

Nothing further about `optimizer.py`'s internals should be carried into the II-2
workstream.

## 10. Tests

*Performed during packaging*, Python 3.11.17 / NumPy 1.26.4 / pandas 2.2.3, against
the pinned checkout: **40 tests, 40 passed, 0 failed, 0 errors, 0 skipped.**

### The training-boundary proof — `tests/test_no_training_import.py`, 11 tests

**No fixture cipher is used in this proof.** Three independent angles, all against
the real pinned source:

- **Static.** `ast` walks the import graph of `optimizer.py` and
  `ciphers/speck3264.py` transitively through the checkout and asserts that
  `tensorflow`, `keras`, `train_nets`, `polish_nets`, `gohrnet`, `dbitnet`,
  `eval_nets` and `main` are all unreachable. Beyond the standard library only
  `numpy` and `pandas` appear.
- **Dynamic.** A *fresh subprocess* imports the search-only entry point and binds the
  real Speck32/64 cipher, then asserts `sys.modules`. A subprocess is required
  because the test session has already imported other things. No search is executed.
- **Controls.** The same static walk from `main.py` **must** reach `train_nets` and
  then `tensorflow`, and `train_nets.py` must import TensorFlow directly. Without
  these the first two would be consistent with the analyser finding nothing at all.
  Both controls pass, so the search-only path is demonstrably avoiding something
  real.

The runtime assertion is also exercised directly: injecting a `train_nets` module
into `sys.modules` makes `assert_no_training_imports` raise. That assertion runs
immediately before and immediately after every call to `optimizer.optimize`.

### The remaining 29 tests

`tests/test_search_runner.py` covers the harness: the source-derived facts, every
patch declared and reversible, logging and serialisation, manifest hashing,
`verify_outputs` catching tampering, and the CLI refusals (R1 carrying any override,
R2 below published settings, unknown mode, unknown `population_init`, missing keys,
a root without `optimizer.py`, an output directory inside the checkout, R2 without a
protocol).

The tests that need to *execute* a search to exercise logging and serialisation use
`tests/toy_cipher.py`, a 32-bit fixture. **It is a logging fixture and nothing more.
No test searches Speck32/64 or Simon32/64 at any setting, and no toy result is
evidence about AutoND.** The training-boundary proof above uses the real source
only.

Coverage: the four source-derived facts (NumPy pin, no TensorFlow, the
import-time default binding, `nb_samples` as a parameter); every patch declared,
effective and fully reversible, with `changes_behaviour` correct for each; a
complete toy run producing a contiguous log and an honest reproducibility
statement; the upstream module byte-identical after a run; manifest hashing;
`verify_outputs` catching a modified file, an added file and a missing manifest;
and the CLI refusing a `paper` label below published settings, an unknown mode, an
unknown `population_init`, missing keys, a root without `optimizer.py`, an output
directory inside the checkout, and `paper` mode without a protocol.

Run them yourself:

```bash
cd ~/Documents/deep_speck/audit/targets/autond/search
export AUTOND_UPSTREAM=~/Documents/AutoND-upstream
python -m unittest discover -s tests -t . -v
```

They fail rather than skip if `AUTOND_UPSTREAM` is unset.

One fixture note, recorded because it shaped the design: the first toy cipher was
XOR-linear, so every difference propagated deterministically, scores never fell
below `T`, and `optimize()`'s `while True` never terminated — it hung the suite.
The current fixture uses a non-linear integer mixer and terminates in about 2.4
seconds with `highest_non_random_round = 1`.

---

## 11. What I did, and what you must do

**Performed during packaging (software preparation and inspection only):**

- read `optimizer.py`, `main.py` and `ciphers/speck3264.py` at the pinned commit
- confirmed the search path imports neither TensorFlow nor `main`
- confirmed `ciphers/speck3264.py` cannot import under NumPy 2
- confirmed `evo`'s generation default is bound at import and is not overridable by assignment
- confirmed `sys.dont_write_bytecode` keeps the checkout byte-identical on import
- wrote and ran the 28-test suite against the toy fixture
- exercised the CLI refusal paths, which exit before any search begins

**You must do:**

- create the environment and install dependencies
- verify the upstream commit and source hashes on your machine
- run the timing benchmark
- fill in, commit and record the protocol
- launch the paper-configured runs
- verify the outputs and preserve the evidence
- interpret the result against your frozen decision rule

**Not done by anyone, and not to be inferred from this package:** the search has not
been run on Speck32/64 or Simon32/64 at any configuration. There is no runtime
figure, no returned difference and no reliability proportion in this package,
because none exists yet.
