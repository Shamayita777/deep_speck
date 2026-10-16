# II-2 Specification — the evolutionary optimizer and its round loop

> **Numbering is relative to the PDF supplied to II-2.** Every algorithm,
> line (L1–L19), section, definition, table and figure number in this package
> refers to that PDF, not to any other revision of ePrint 2022/1467.
>
> Identifying features of that PDF:
> - last printed page number is 29; I could not count physical pages from my
>   text rendering;
> - 2 figures: Fig. 1, Gohr's network; Fig. 2, dilated convolutions / DBitNet;
> - 9 tables;
> - 2 algorithms: **Algorithm 1 = evolutionary optimizer (§5.2)** and
>   Algorithm 2 = Gohr's optimizer (Appendix A);
> - the bias score is defined in §5.1.
>
> An automated extraction of the canonical revision reported the reverse
> algorithm numbering and the optimizer in §5.1. That report is unverified
> (AMBIGUITY_LOG A-031). In this package, "Algorithm 1" means the evolutionary
> optimizer in any case.

Written before any code. Each statement cites the paper or an entry in
AMBIGUITY_LOG.md. "L1–L19" are the printed lines of Algorithm 1 counted top to
bottom, `end` lines included. The numbering is mine; the paper does not number
its lines.

## 0. Source and scope

Source: the attached PDF, *A Cipher-Agnostic Neural Training Pipeline with
Automated Finding of Good Input Differences* (Bellini, Gerault, Hambitzer,
Rossi). The brief gives a different title (AMBIGUITY_LOG S-1), and I could not
check which revision this is (S-2).

In scope:
- the bias score, Definition 2 (§5.1);
- Algorithm 1, a single evolutionary search at a fixed number of rounds (§5.2);
- the outer procedure in §5.2, "Accounting for the Starting Round": iterate
  over rounds, stop on the threshold, take the union, rescore, compute the
  final weighted score.

Out of scope:
- ε-close selection (§5.3; A-027);
- neural training (§6);
- related-key and modular-addition settings (A-021);
- prepended-round scoring (§8, "Extending the Scope").

## 1. Notation

The paper's notation is used where it exists. Symbols marked (mine) are my own.

| Symbol | Meaning | Source |
|---|---|---|
| n | width of an input difference in bits | Alg. 1 L1 (`RandomInt(0, 2^n − 1)`); Def. 2 |
| n_out (mine) | width of the cipher output in bits; equals n for block ciphers | A-020 |
| k | key width in bits; 0 for keyless permutations | Def. 2 |
| E^r_K(X) | cipher reduced to r rounds | §5.2; r is implicit in Def. 2 |
| t | number of samples per bias-score evaluation | Def. 2; §5.2 |
| M | mutation probability | §5.2; L8 |
| T_b | bias-score threshold | §5.2; §5.4 |
| b̃^t_r(δ) | bias score of δ at r rounds | Def. 2 |
| Δ_r | list of differences returned for round r | §5.2 |
| R | last round whose run returned a score above T_b | §5.2 |
| b̃_{i,r} | rescored bias of union member δ_i at round r | §5.2 |
| S_δ | final score of δ | §5.2 |
| P₀ (mine) | initial population size = 1024 | L1; §5.2 |
| P (mine) | kept population size = 32 | L3, L17; §5.2 |
| G (mine) | number of generations = 50 | L4; §8 (A-002) |

## 2. The bias score b̃^t_r(δ)  (Definition 2, §5.1)

Formula (Def. 2, with r made explicit):

    b̃^t_r(δ) = (1/n_out) · Σ_{j=0}^{n_out−1} | 0.5 − (1/t) · Σ_{i=0}^{t−1} ( E^r_{K_i}(X_i) ⊕ E^r_{K_i}(X_i ⊕ δ) )_j |

Procedure for one evaluation:
1. Draw t plaintexts X_i uniformly from n bits. Draw t keys K_i uniformly from
   k bits, a separate key for each sample (Def. 2 writes K_i). Draws are
   independent, and each evaluation gets fresh draws (A-015).
2. Encrypt X_i and X_i ⊕ δ under K_i for r rounds. XOR the two outputs.
3. For each output bit j, compute p̂_j, the fraction of the t samples in which
   that bit is 1.
4. Return the mean over j of |0.5 − p̂_j|. The paper calls this a "sum" but
   its formula takes the mean (A-013); the vertical bars are read as absolute
   value (A-014). The result lies in [0, 0.5].

Properties I can derive from the definition. Tests use these, not results:
- b̃(0) = 0.5 exactly, because the output difference of a zero input
  difference is always zero. This is why δ = 0 must be excluded (A-004).
- For an ideal random function, p̂_j ≈ N(0.5, 1/(4t)). Then
  E[b̃] ≈ 1/√(2πt), which is 0.00399 at t = 10⁴. This matches the paper's
  "approximately 0.004" (§5.4).
- Under the same model, SD[b̃] ≈ (1/(2√t)) · √(1 − 2/π) / √n_out. That gives
  0.00053 for n_out = 32 and 0.00015 for n_out = 384, matching §5.4's
  0.00052 and 0.00015. The derivation is mine (normal approximation to the
  binomial). The agreement is evidence for my reading of the formula: mean,
  not sum, and absolute value.

## 3. Algorithm 1 — EVOLVE(E, r)

Input: a cipher E, a round count r, and the parameters in §5.
Output: Δ_r and its scores.
The pseudocode does not mention r; it enters only through the scoring function
(A-030).

**S1. Initialization (L1–L3).**
- Draw P₀ = 1024 integers uniformly from [0, 2^n − 1] (A-003).
- Discard zeros (A-004) and duplicates (A-005, A-024). Do not top up to 1024.
- Score each remaining difference with b̃^t_r.
- Sort descending with a stable sort (A-019) and keep the first P = 32 as the
  current population.

**S2. Generation loop (L4).**
Repeat G = 50 times. I read the bound as exclusive. This is cross-checked
against §8, which counts 496 per generation over 50 generations, 24,800 in
total (A-002).

**S3. Offspring (L5–L15).**
- Iterate over every unordered pair (i, j) with i < j among the current
  population. Indices run 0..|pop|−1, giving 496 pairs at |pop| = 32 (A-002).
- For each pair, draw u ~ U[0, 1). Set m = 1 if u < M, else m = 0 (L8–L12).
- Draw a bit position p uniformly from {0, …, n−1}. It is drawn even when
  m = 0 (A-003).
- The child is pop_i ⊕ pop_j ⊕ (m ≪ p), where ≪ is left shift (A-001).
  When m = 1, the child differs from pop_i ⊕ pop_j in exactly one bit.
  When m = 0, it equals pop_i ⊕ pop_j.

**S4. Candidate hygiene (my addition).**
- Discard any zero child (A-004).
- Discard repeated children, keeping the first occurrence in generation order
  (A-005). This can be switched off with `deduplicate`.

**S5. Selection (L16–L17).**
- Score each surviving candidate freshly with b̃^t_r.
- Sort descending (stable).
- The new population is the first min(P, |candidates|) candidates. Parents are
  not carried over (A-006); this can be switched on with `elitism`.

**S6. Return (L19 and §5.2).**
- L19 returns `candidates`, about 496 values. The prose says each round yields
  a list of 32 differences.
- I return Δ_r as the first 32 of the final sorted candidates, which is the
  final population, together with the scores from that final sort (A-007).
- The full final candidate list is also recorded.
- There is no early stopping inside Algorithm 1 (A-008).

**Small-population edge case (A-023).** If the population ever has fewer than
2 members, no offspring can be formed. The run then ends early and records how
many generations it completed.

## 4. Outer procedure — §5.2, "Accounting for the Starting Round"

**O1.** For r = 1, 2, …, run EVOLVE(E, r). Each run starts from a fresh random
population with no warm start (A-012).

**O2. Threshold stop.** If no score in the returned Δ_r is strictly greater
than T_b, stop. Then R = r − 1, and this round's list is discarded (A-009).
Every member of each kept Δ_r is retained, including members scoring ≤ T_b
(A-010).

**O3. Round cap (not in the paper).** If the run for round `max_rounds` still
returns a score above T_b, stop and set `terminated_by = "round_cap"`.
`max_rounds` defaults to the cipher's declared nominal round count; if the
cipher declares none, the caller must supply it (A-011).

**O4. Union.** U is the union of Δ_1, …, Δ_R as a set. Members are ordered by
first appearance: earlier round first, then rank within that round (A-018).

**O5. Rescoring.** For every δ_i in U and every r in 1..R, compute b̃_{i,r}
with fresh samples and the same t (A-016).

**O6. Final score.**

    S_{δi} = Σ_{r=1}^{R} w_r · b̃_{i,r}

- **What §5.2 prints:** the equation with no weight inside the parentheses.
- **What the prose says:** the score is "weighted", "providing higher weight to
  later rounds", and a simple sum is inadequate.
- **Default:** w_r = r (`round_weighting="linear"`). **NOT IN PAPER.** This is
  my construction to satisfy the prose (A-017).
- **Always reported:** the unweighted sum, available as
  `round_weighting="uniform"`.

**O7. Output.** Return U sorted by S descending (stable).
If R = 0 (round 1 already fails the threshold), U is empty.

**Single-round mode.** Calling `search(..., rounds=r)` runs only EVOLVE(E, r)
and returns Δ_r with its scores. This lets II-4 compare Algorithm 1 in
isolation from the outer procedure.

## 5. Parameters

| Parameter (code name) | Default | Source |
|---|---|---|
| initial_population (P₀) | 1024 | Alg. 1 L1; §5.2 "initial population contains 1024" |
| population_size (P) | 32 | L3, L17; §5.2 "32 best ones are kept" |
| generations (G) | 50 | L4 (A-002); §8 "over 50 generations" |
| mutation_probability (M) | 1.0 | §5.2 "we set M = 1" |
| samples (t) | 10_000 | §5.2 "t = 10⁴" |
| threshold (T_b) | 0.01 | §5.2; §5.4 |
| starting round | 1 | §5.2 "from round 1" |
| preset `paper_reduced` | t = 1000, T_b = 0.05, G = 10 | §5.4. **Not** assumed to be the repository's set (S-4) |
| max_rounds | cipher.nominal_rounds | NOT IN PAPER (A-011) |
| round_weighting | "linear" (w_r = r) | NOT IN PAPER: II-2 choice satisfying §5.2's prose; the printed equation is unweighted (A-017) |
| deduplicate | True | NOT IN PAPER (A-005) |
| elitism | False | per L17 (A-006) |

## 6. Cipher interface (cipher_interface.py)

The algorithm needs four things about a cipher:
- the difference width n (Alg. 1 L1, L13);
- a way to compute E^r_K(X) for random X and K at a chosen r (Def. 2);
- the output width;
- the key width.

I define a `Cipher` protocol:

- `name: str`
- `input_bits: int` — n
- `output_bits: int` — n_out (A-020)
- `key_bits: int` — k, which may be 0
- `nominal_rounds: int | None` — used only for the round cap (A-011)
- `encrypt(plaintexts, keys, rounds) -> ndarray`

`encrypt` takes `plaintexts`, a uint8 array of shape (t, input_bits);
`keys`, a uint8 array of shape (t, key_bits); and `rounds`, an int. It returns
a uint8 array of shape (t, output_bits). All values are 0/1, and the function
must be vectorized over rows. What "a round" means is the cipher's own
definition (A-022).

Bit convention: column 0 is the most significant bit of the integer view of a
difference (A-026). Differences are injected by XOR into the plaintext only;
this is the single-key setting (A-021).

Differences are Python `int`s, because n can exceed 64. Conversion between int
and bits goes through big-endian bytes and `np.unpackbits`, valid on
NumPy 1.26.4.

## 7. Output of `search(cipher, *, seed, **params) -> dict`

Keys required by the brief:

- `input_differences`: U in final rank order. In single-round mode, Δ_r.
- `score_per_difference`: S under the selected weighting. In single-round mode,
  the final-generation b̃ scores.
- `generations_completed`: total generations executed across all Algorithm 1
  runs, including the terminating run at round R+1 (A-025).
- `evaluations`: total calls to b̃, covering initial populations, every
  generation, and rescoring (A-025).
- `params_used`
- `seed`
- `paper_parameters`
- `paper_parameter_sources`

Extra keys:

- `rounds_reached` (R)
- `terminated_by` ("threshold", "round_cap", or "single_round")
- `per_round`: for each r, Δ_r, its scores, the full final candidate list, the
  generations completed, and the evaluations
- `rescored_matrix`: b̃_{i,r}
- `score_uniform` and `score_round_weighted`
- `encryptions`: 2·t·evaluations

Additional notes:

- `mode` is `"full"` or `"single_round"`.
- In single-round mode these keys are omitted, because that mode does not
  produce them: `rounds_reached`, `rescored_matrix`, `score_uniform`,
  `score_round_weighted`.
- `encryptions` is present only when the default scorer is used. With an
  injected scorer the number of encryptions is unknown, so the key is omitted.

## 8. Determinism

- No global RNG and no module-level mutable state.
- Each RNG stream is addressed directly as
  `SeedSequence(seed, spawn_key=key)`:
  - (1, r, 0): search stream for round r;
  - (1, r, 1): scoring stream for round r;
  - (2,): rescoring.
- Changed from "spawn in order" during implementation. Direct addressing means
  `search(..., rounds=r)` reproduces round r of a full run exactly, which a
  test checks.
- Within a generation, all u draws are made first, then all p draws (A-003).
- For a fixed seed and fixed parameters, the output is bit-for-bit identical.

## 9. Properties to test

These come from the paper or from the definitions. They are not results.

1. Determinism: the same seed gives identical output, and different seeds give
   (almost surely) different internal populations.
2. Population size never exceeds P, and each kept Δ_r has at most 32 members.
3. Before hygiene, a generation has C(|pop|, 2) candidates, which is 496 at
   |pop| = 32 (L6–L7; §8).
4. With M = 1, every child has Hamming distance 1 from pop_i ⊕ pop_j.
   With M = 0, the distance is 0.
5. Every candidate lies in [1, 2^n − 1]. Zero never appears.
6. b̃ lies in [0, 0.5], and b̃(0) = 0.5.
7. On a random-function cipher at t = 10⁴, the mean b̃ is about 0.004 (§5.4),
   tested with a tolerance.
8. A random-function cipher terminates at round 1 with R = 0.
   A degenerate identity cipher hits the round cap.
9. The threshold comparison is strict.
10. When nothing is removed by hygiene, the evaluation count equals the
    closed-form P₀ + G·C(P,2) per run.
11. Invalid parameters raise errors. Examples: P < 2, P > P₀, n < 1,
    M outside [0, 1], t < 1, a wrongly shaped `encrypt` output.
