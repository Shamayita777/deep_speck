# AutoND evolutionary search — paper vs README vs pinned code

Source-to-claim comparison for the **search** stage only. Established by reading
primary sources; no experiment was run to produce any of it.

| Source | Identification |
|---|---|
| Paper | Bellini, Gerault, Hambitzer, Rossi, *A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input Differences*, ePrint 2022/1467, FSE 2024. §5.1–5.4, Algorithm 1, Table 6. |
| README | `README.md` at the pinned commit |
| Code | `optimizer.py`, `main.py`, `ciphers/speck3264.py` at `8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` |

**Nothing below is silently resolved.** Where the three disagree, all three readings
are recorded and a faithful interpretation is recommended with its reason.

---

## 1. The documented reproduction commands

From the README, verbatim:

```bash
python main.py                      # demo: SPECK32, single-key
python3 main.py [cipher] [scenario] # scenario is 'single-key' or 'related-key'
```

`main.py.__main__` runs **PART 1 (search)** then **PART 2 (training)**. There is no
documented command that runs the search alone. Running `main.py` as documented
therefore also trains, which is the GPU-bound stage.

**Prerequisites**, README verbatim: *"standard Python modules, except for
`tensorflow version 2.15` (please note: at the moment, our code has
incompatibilities with TensorFlow 2.16)"*, plus `pandas` and `ray` via conda/pip.
This corroborates the TF 2.15 pin independently of our own finding.

## 2. The README's own expected output — a free reproduction target

This is the most useful thing in the README and it costs almost nothing to check.
At the **shipped demonstration settings**, the README states the search produces:

```
Found 1 0.1-close differences: ['0x400000'].
The highest round with a bias score above the threshold was 7.
```

So `python main.py`, unmodified, is claimed to find `0x400000` — the difference the
headline SPECK32 result uses. **That is a documented, checkable claim at
demonstration cost**, and it is a genuine reproduction of something the authors
assert, not merely a timing benchmark. It should be the first thing run.

### 2.1 But the README contradicts itself on the highest round

| Evidence inside the README | Implies |
|---|---|
| prose: *"The highest round … was 7"* | `highest_round = 7` |
| its own log sample: *"New log start, reached round 6"*, last section `Best at 6:` | `highest_round = 6` |
| its own demo output: *"Training dbitnet … starting from round 5"* | `highest_round = 6`, because `main.py:127` passes `max(1, highest_round-1)` |

Two of the three say 6, and the code's arithmetic confirms it. The log line is
written as `f'New log start, reached round {highest_non_random_round}'`, and the
re-scoring loop `for nr in range(1, current_round)` produces `Best at 1 … Best at 6`
only when `highest_non_random_round = 6`.

**Recommended interpretation: the prose "7" is the outlier**, most likely pasted
from a different run. Record the discrepancy; do not adopt either number as an
expectation. Treat *"`0x400000` is returned"* as the checkable claim and
`highest_round` as an observable to report, not a target to match.

### 2.2 Two further internal README inconsistencies

- **Epoch count.** The README says the shipped value is `EPOCHS = 5`. `train_nets.py:32`
  actually ships `EPOCHS = 10`. The README's pasted demo output says *"Training on
  10 epochs …"* and then shows `Epoch 2/5 … Epoch 5/5` — contradicting itself within
  one block.
- **Starting round.** The README's module-usage snippet uses
  `max(1, highest_round-2)`; `main.py:127` uses `max(1, highest_round-1)`.
- **A typo**, noted for completeness: *"`NUM_SAMPLES = 10**3 # 10**4 in the paper,
  set to 5 here"* — the shipped value is 10³, not 5.

None of these affect the search itself, but they establish that the README was not
kept in step with the code, which is relevant to how much weight its other claims
carry.

---

## 3. Search parameters: paper vs code

Paper §5.2: *"the initial population contains 1024 differences, the 32 best ones are
kept at each generation, and we set **M = 1, t = 10⁴, and T_b = 0.01**."*
Algorithm 1: `for iterations ← 0 to 50`.

| Parameter | Paper | Code | Code location | Disclosed by a comment? |
|---|---|---|---|---|
| Initial population | 1024 | 1024 (`L**2`, L=32) | `evo`, `gen = np.random.randint(2, size=(L**2, num_bits))` | — agrees |
| Survivors per generation | 32 | 32 (`L`) | `evo`, `good = idx[np.argsort(scores)][-L:]` | — agrees |
| Generations | **50** | **5** | `NUM_GENERATIONS`, line 12 | **yes** — *"50 in the paper, set to 5 here for demonstration"* |
| Samples per score, *t* | **10⁴** | **10³** | `NUM_SAMPLES`, line 13 | **yes** — *"10**4 in the paper"* |
| Threshold, *T_b* | **0.01** | **0.05** | `optimize()`, line 138: `T = 0.05 # The bias score threshold` | **NO** |
| Mutation probability, *M* | **1** | **≈0.99** | `evo`, `mutProb = 100`; `np.random.randint(0,100, …) > (100-mutProb)` | **NO** |
| ε for the close set | 0.1 | 0.1 | `main.findGoodInputDifferences` | — agrees |
| Weighted score | Σ_r r · b̃_{i,r} | `weightedScores += nr*finalScores[nr]` | `optimize()` | — agrees |

### 3.1 The threshold is the material one

`T_b` is the stopping rule: the search advances rounds until no difference scores
above it, and `highest_non_random_round` is the result. **The code's 0.05 is five
times the paper's 0.01**, so the code stops *earlier* and reports a *lower* highest
round than the paper's configuration would.

This matters beyond the search, because `main.py` feeds `highest_round` into
training as the starting round. A different threshold moves the whole downstream
pipeline.

It is also the one discrepancy with **no comment anywhere** — not in the code, not
in the README. Unlike the generation and sample counts, which are openly labelled
demonstration values, this one is undisclosed.

**Mitigating context, which must be reported alongside it.** Paper §5.4 says: *"we
find that reduced parameters, for instance **t = 10³, T_b = 0.05, and 10
generations**, provide faster results despite being slightly less robust."* So the
shipped configuration is close to a reduced parameter set the paper itself
discusses — `t = 10³` ✓ and `T_b = 0.05` ✓ — but **not** its generation count, which
the paper gives as 10 and the code ships as 5.

**Recommended interpretation.** The shipped configuration is the paper's *reduced*
setting with half its generations, not an arbitrary deviation. But `T_b` is a
registered parameter of the method that the code fixes to a different value than
§5.2 without saying so, and `optimize()` exposes no parameter for it: changing it
requires editing upstream. Report it as an **undisclosed hardcoded deviation with
an acknowledged precedent in §5.4**, and state which value any run used.

### 3.2 Mutation probability

Paper: `if RandomFloat(0, 1) < M then m ← 1 else m ← 0`, with M = 1 — every
candidate is mutated. Code: `np.random.randint(0, 100, …) > 0`, which is false for
the value 0, so roughly 1 candidate in 100 is not mutated. A small, undisclosed
difference; unlikely to be material, but it is a difference and it belongs in the
record.

---

## 4. Algorithm: paper's Algorithm 1 vs `evo()`

| Step | Paper, Algorithm 1 | Code, `evo()` | Same? |
|---|---|---|---|
| Initial population | 1024 random, sort, keep 32 | identical | ✔ |
| Candidate generation | pairwise `pop_i ⊕ pop_j ⊕ (m << RandomInt(0, n−1))` | pairwise XOR, then mutate selected rows at a random bit | ✔ |
| De-duplication | **absent** | `kids` filtered against an `explored` set and `np.unique` | ✘ **code adds this** |
| Selection | `Sort candidates`; `current_population ← first 32 of candidates` | `gen = vstack([gen, kids])`, sort the **union**, keep 32 | ✘ **code is elitist** |
| Early exit | **absent** | `if np.all(scores == 0.5): break` | ✘ **code adds this** |
| Return value | `candidates` | `gen, scores` — the 32 survivors | ✘ |

### 4.1 Elitism is the substantive algorithmic difference

The paper replaces the population with the best of the *new candidates* only —
generational replacement, in which a good parent can be lost. The code sorts
parents and children together and keeps the best 32 overall, so the best-ever
difference always survives.

Elitism generally converges faster and is more stable; it also explores less. This
is a real algorithmic divergence between the published description and the released
implementation, and **neither the code nor the README mentions it.**

**Recommended interpretation:** report the implementation as elitist, note that
Algorithm 1 as written is not, and do not describe a run of this code as executing
Algorithm 1. If you want the paper's algorithm, that is an II-2 reimplementation
task — written from Algorithm 1 without reading `evo()` — not a patch to upstream.

---

## 5. Cross-round population reuse (`gen=diffs`)

`optimize()` carries both variants, with the repository's own comment:

```python
#diffs, scores = evo(f=..., gen=None)
# The initial set of differences can be set to None, or to the differences returned
# for the previous round. We use the second option here, as opposed to the first
# (above) in the paper.
diffs, scores = evo(f=..., gen=diffs, verbose=1)
```

Paper §5.2: *"we run our optimizer iteratively from round 1 to round R + 1"*, and
Algorithm 1 begins from a random population each time. So the paper's procedure
restarts fresh per round; the code warm-starts from the previous round's survivors.

This is the **only** discrepancy of the five that the repository discloses itself.

**Recommended interpretation:** the shipped behaviour is `gen=diffs`, and a run of
the released code should use it, labelled as such. A `gen=None` run answers a
different question — what the paper describes — and the two are not pooled.

---

## 6. Randomisation

Paper: silent on seeding. Code: nothing is seeded anywhere. `np.random.randint`
drives the initial population, mutation selection and mutated bit positions;
`os.urandom` draws the plaintexts and keys afresh each round. **A single run
establishes nothing about reliability**, which is why the question worth asking is
the proportion of independent runs that return `0x400000`, not whether one run does.

---

## 7. Environment — the NumPy failure, investigated

Under NumPy 2, `import ciphers.speck3264` fails outright:

```
OverflowError: Python integer 32768 out of bounds for uint8
  ciphers/speck3264.py:103, in convert_from_binary
    X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
```

**Cause.** `X` is `uint16`; `arr[:, pos]` is `uint8`; `2**15 = 32768`. Under NumPy 1,
value-based casting promoted the Python int into the array's dtype. Under NumPy 2,
NEP 50 makes the Python int adopt the *operand's* dtype — `uint8` — and 32768 does
not fit. The module calls `check_testvectors()` at import time, so the failure
happens on import, before any audit code runs.

**Resolution: use the compatible environment, do not patch.** `numpy==1.26.4` is
exactly what AutoND's own published environment used and what the October 12
reproduction ran on. A patch here would alter cipher semantics to satisfy a newer
library, which is the opposite of faithful reproduction.

For completeness, the minimal patch *if it were ever unavoidable* would be casting
the multiplier to the array dtype at that one line. It is **not recommended, not
applied, and not supplied**. Were it ever needed it would have to be validated
against `check_testvectors()` — SPECK32/64 plaintext `(0x6574, 0x694c)`, key
`(0x1918, 0x1110, 0x0908, 0x0100)`, 22 rounds → ciphertext `(0xa868, 0x42f2)` —
which the module already asserts on every import, and that test vector agreement is
the condition any such patch must meet before it is trusted.

---

## 8. Summary of discrepancies

| # | Discrepancy | Disclosed? | Material? |
|---|---|---|---|
| D1 | Generations 5 vs paper 50 | yes, code comment + README | yes — determines search depth |
| D2 | Samples 10³ vs paper 10⁴ | yes, code comment + README | yes — determines score precision |
| D3 | **Threshold 0.05 vs paper 0.01** | **no** | **yes — sets the stopping round, and hence the training start round** |
| D4 | Mutation ≈0.99 vs paper M = 1 | no | unlikely |
| D5 | **Elitist selection vs Algorithm 1's generational replacement** | **no** | likely — changes convergence and exploration |
| D6 | De-duplication against `explored`; early exit on all-0.5 | no | unclear; both are additions |
| D7 | Cross-round warm start `gen=diffs` vs paper's fresh restart | **yes**, code comment | likely |
| D8 | README highest round 7 vs its own log and training output implying 6 | n/a | documentation only |
| D9 | README `EPOCHS = 5` vs shipped `10`; `highest_round-2` vs `-1` | n/a | documentation only |

D3 and D5 are the two findings that are both substantive and undisclosed. They are
**II-3 declared-versus-realised findings establishable by source citation alone**,
with no compute at all.

---

## 9. What a faithful reproduction means here

Three distinct runs, three distinct claims. They are not interchangeable.

**R1 — the README's documented demo.** `python main.py` unmodified, demonstration
settings. Checks the README's own claim that the search returns `0x400000`. Cheapest
and most directly checkable. Note it also triggers training; to check the search
alone, the run must stop after PART 1.

**R2 — the paper's §5.2 configuration.** 50 generations, t = 10⁴. Requires rebinding
`evo`'s import-time default (see the harness README). Whether to also set T_b = 0.01
is a **decision you must declare**: the code hardcodes 0.05 and exposes no parameter,
so reaching 0.01 means editing upstream, which R2 otherwise avoids. Running at
50 / 10⁴ / T_b = 0.05 is the paper's parameters for everything the code lets you
reach, and that is what it must be called — not "the paper's configuration".

**R3 — Algorithm 1 as published.** Generational replacement, no de-duplication, no
early exit, fresh population per round. This is **not** a run of upstream; it is an
independent reimplementation from the paper, and therefore II-2 work, written
without reading `evo()`.

R1 and R2 are reproduction (II-1): run their code, see whether you land where they
said. R3 is independent reimplementation (II-2): build from the paper alone. Mixing
them would contaminate both.
