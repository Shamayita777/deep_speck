# AutoND CE3: methodology

Design `AutoND-CE3-design-2026-11-r1`, status `FROZEN` (hash `849ffaa3…`). The design lives in `design.py`; `preregistration/preregistration.json` is its rendering, approved for that exact hash on 2026-11-01. Nothing here is confirmatory until the production run is made and verified.

## 1. Question

Across independent evaluation replicates, is the CE2 target, coarsened into frozen whole-value classes, linearly decodable from the 64-unit penultimate ReLU representation of the published AutoND SPECK32/64 round-8 DBitNet:

- beyond a within-replicate permuted-label control, and
- beyond one fixed untrained DBitNet,

once the probe pipeline has shown that it actually recovers the known genuine-versus-random task?

**What this is.** CE3 is observational, on one checkpoint, one layer and one twin.

**What this is not:**
- **Not a claim about use.** The claim is decodability; A3 already showed that a representation can carry a decodable target the network does not use.
- **Not independent of CE2.** The target is the CE2 construct.

## 2. Target

T is CE2's trajectory-derived Markov single-characteristic proxy: the sum over the eight rounds of log2 xdp+ along the reconstructed genuine-pair trajectory. It is computed by the pinned r3 `adapter.trajectory_and_weights`. T is a negative integer. It is not the fixed-key characteristic probability and not the full-cipher differential probability.

### Classes (P1)

- **Pilot.** The classes come from a separate pilot, P1: 10^6 fresh genuine pairs. P1 builds, loads and evaluates no network. The pinned upstream loader imports `main`, which imports TensorFlow as a library, so TensorFlow was imported but never used.
- **Rule.** Among all whole-value partitions whose every class holds at least 15% of P1, take the one minimising sum_j |5 * N(T <= e_j) - j * N|. Ties go to the smallest edge tuple.
- **Result.** P1 has 56 support values in [-91, -35]. The edges are **[-68, -63, -59, -55]**, with class fractions 0.1957, 0.2029, 0.1953, 0.1999, 0.2062.
- **Assignment.** class(T) = number of edges e with T > e. Equal T values are never split.
- **Exploratory pilot.** An earlier in-memory pilot (P0, also 10^6 pairs, never recorded) gave the same edges. I used it only to judge that five classes are feasible.

### The three-class fallback

This is resolved before production. Five classes are feasible on P1, so K = 5 and these edges are frozen and applied unchanged to every replicate. The confirmatory protocol has no three-class branch and no per-replicate rebinning. A replicate that violates the class-coverage minimum is FAILED.

### Coverage

The minimum is 15% of the replicate for each class, and at least 1,000 examples of every class in every training and every test fold. It is checked for:
- the actual labels;
- every permuted label vector;
- the calibration labels.

## 3. Representation and models

**Representation.**
- It is the output of layer 32 of `dbitnet.make_model(64)`.
- Identity is checked through the graph:
  - layer 32 is an `Activation(relu)` with output (None, 64);
  - it is fed only by `BatchNormalization` (31), which is fed by `Dense(64, linear)` (30);
  - its only consumer is the output layer `Dense(1, sigmoid)` (33).
- Layer names are not used. The published model's layer is called `activation_2`, but in the same process the twin's is called `activation_5`.
- Extraction is `keras.Model(model.input, layers[32].output).predict(X, batch_size=5000)`, in float32.
- **Head consistency.** For every main and calibration example, sigmoid(h W + b) of the output layer must reproduce `model.predict` to within 1e-5. This ties the representation to the model's actual output.

**Published model.** It is loaded with the pinned strict `load_published`: checkpoint `a7be788d…`, all 68 HDF5 arrays bit-identical.

**Untrained twin.**
- It is the exact twin saved by the verified CE2 production run: file `adda08a3…`, weights `23772278…`, seed 20261031.
- It is copied byte for byte into each CE3 output directory and loaded by its weights hash.
- It must have at least 2 non-constant features on every main and calibration set.
- Results are conditional on this one initialisation.

## 4. Sampling

| Item | Value |
|---|---|
| Replicates | 20, all required VALID; no reserves, no early stop, no interim look |
| Main set per replicate | n = 200,000 fresh genuine pairs (os.urandom, CE2 `generate_genuine_pairs`, gates E and F) |
| Calibration set per replicate | 50,000 genuine + 50,000 uniform 64-bit inputs, shuffled by PCG64(30,000,000 + i) |
| Folds | `KFold(5, shuffle=True)` on indices only; seeds 40,000,000 + i (main) and 60,000,000 + i (calibration) |
| Permutations | B = 5, the non-identity elements of a cyclic permutation group of order B + 1 (section 6) |
| S1 tie-break | a uniform random permutation of {0, ..., B}, drawn from os.urandom and committed in `data.npz` before any probe is fitted |

How n and B were chosen is in `POWER_ANALYSIS.md`.

## 5. Probe

**Configuration.**
- `StandardScaler` followed by `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", tol=1e-4, max_iter=2000, fit_intercept=True, class_weight=None)`.
- The scaler is fitted on the training fold only.
- Features are float64 and BLAS runs on one thread, for determinism. scikit-learn is pinned at 1.9.1.

**Convergence.** A fit counts as converged when all three hold:
- `n_iter_ < max_iter`;
- no `ConvergenceWarning` is raised;
- the maximum absolute gradient of the training objective at the fitted parameters is at most 1e-3. The gradient is computed outside scikit-learn.

A non-converged fit makes the replicate FAILED; nothing is retried. The only allowed alternative is max_iter = 5000, and only if the blinded benchmark had needed it (`POWER_ANALYSIS.md`, section 7).

## 6. Arms and endpoints

Every main-set arm uses the same examples and folds. BA is the balanced accuracy of the pooled held-out predictions: BA = (1/K) sum_c (correct predictions of class c) / (examples of class c). It is a rational number; S1 compares it exactly as a fraction.

| Arm | Features | Labels |
|---|---|---|
| A | trained representation | actual classes y |
| P_b | trained representation | y o sigma^(b+1), b = 0 .. B-1 |
| T | twin representation | actual classes |
| R | raw 64-bit input | actual classes (descriptive only) |
| CAL_trained / CAL_twin | trained / twin representation | genuine vs random (calibration folds) |
| CAL_model | none: the published model's own decision 1[score > 0.5] | genuine vs random |

### 6.1 The permutation group

For replicate i, o = PCG64(50,000,000 + i).permutation(n) is cut in order into floor(n / (B + 1)) blocks of length B + 1. The remainder is cut into blocks whose lengths divide B + 1, largest first. Each block c = (c_0, ..., c_{L-1}) becomes a cycle with sigma(c_j) = c_{(j+1) mod L}. Every cycle length divides B + 1, so sigma^(B+1) = id and G = {id, sigma, ..., sigma^B} is a group of order B + 1. The observed assignment is the identity element of G. Each P_b keeps the global class counts exactly. With n = 200,000 and B = 5 there are 33,333 six-cycles and one two-cycle.

The earlier draft drew B independent seeded permutations. Those do not form a group, and a fixed set of non-group permutations does not make the B + 1 assignments exactly exchangeable. That is why it was replaced.

### 6.2 Endpoints per replicate

- **C_task** = BA(CAL_trained) − [0.5 + κ · max(BA(CAL_model) − 0.5, 0)], with κ = 0.5. The bracket is a per-replicate recovery threshold set by the published model's own balanced accuracy on the same calibration examples. At the published reference accuracy 0.5144 it is 0.5 + 0.5 × 0.0144 = **0.5072**.
- **C_twin** = BA(CAL_trained) − BA(CAL_twin).
- **S1** = R − (B + 2)/2. R ∈ {1, ..., B + 1} is the rank of the observed assignment among the B + 1 assignments of G, ordered ascending by the pair (exact BA, recorded tie-break value). S1 > 0 exactly when R is one of the upper (B + 1)/2 ranks: for B = 5, R ∈ {4, 5, 6}. S1 takes the values ±1/2, ..., ±B/2 and is never 0.
- **S2** = BA(A) − BA(T).
- **raw_gap** = BA(A) − BA(R) and **S1_margin** = BA(A) − median_b BA(P_b). Both are descriptive only and never tested.

### 6.3 The published reference accuracy and κ

**Source.** 0.5144 is from the pinned upstream `README.md`, lines 151–181. It is `eval_nets.py` run on `speck3264_single-key_0x400000_round8_polished.h5`. There are five evaluation sets of 10^6 samples from `main.make_train_data`, with label Y ~ Bernoulli(1/2) and Y = 0 inputs uniform. The metric is the accuracy of 1[score > 0.5] (`evaluate_X_Y`). The README reports 0.5144 ± 0.0003, the mean and standard deviation over the five sets. The per-set TPR (0.512–0.514) and TNR (0.515–0.517) make the balanced accuracy equal to the accuracy at that precision. The calibration set's exact 50/50 balance makes BA(CAL_model) equal to the model's accuracy on it.

**Role of 0.5144.** CE3 does not use it as a fixed threshold. The threshold is computed per replicate from the model's own BA on the same calibration examples, which removes that replicate's sampling noise in the model's accuracy from the comparison (the standard error at 10^5 examples is about 0.0016). 0.5144 sets the synthetic task strength in the power analysis and gives the reference threshold 0.5072.

**Why κ = 0.5.**
- The model's output is sigmoid(h W + b), a linear function of this same layer. So a linear probe can represent the model's decision exactly, and κ = 1 is attainable in the population.
- A finite-sample probe loses part of that. In the pipeline-faithful power check with 50,000 examples per class it recovered a mean fraction of about 0.76–0.79 of the synthetic model's advantage.
- κ = 0.5 asks that at least half the known signal survive the probe pipeline. A pipeline that recovers only a small fraction of a known effect therefore cannot pass, while ordinary estimation loss does not fail a working pipeline.
- κ is a prospective judgment made before any CE3 measurement. The power evidence reports κ ∈ {0, 0.25, 0.5, 0.75, 0.9}.

## 7. Inference

### 7.1 Per-endpoint test

For each endpoint D ∈ {C_task, C_twin, S1, S2}, let q(D) = P(D_i > 0) for a valid replicate i. The hypotheses are:

- H0_D: q(D) ≤ 1/2;
- H1_D: q(D) > 1/2.

N+ = #{i : D_i > 0} counts over **all** n valid replicates, and p = P(X ≥ N+) with X ~ Bin(n, 1/2). H0_D is rejected when p < α = 0.05. P(Bin(n, q) ≥ k) increases in q, so the size is at most α for every q ≤ 1/2. With n = 20, rejection needs N+ ≥ 15 (p = 0.0207); N+ = 14 gives 0.0577. The interval is the exact two-sided 95% Clopper–Pearson interval for q.

**Zeros.** An exact 0.0, including −0.0, is a non-positive outcome. It stays in the denominator and counts against rejection, and no zero is removed. This is conservative: for C_task, C_twin and S2 the test is of P(D > 0), the stronger statement. S1 is never 0.

**Other cases.**
- n = 0 gives NO_VALID_OBSERVATIONS: no p-value and no rejection.
- A non-finite endpoint makes the replicate EXCLUDED, never zero.
- A FAILED replicate is retained and never replaced. Any non-VALID replicate makes the decision INCONCLUSIVE.

### 7.2 Exact null distribution of S1

**H0_S1 (conditional exchangeability).** Condition on the trained representation matrix H, the label-independent folds F and the class-count vector N = (N_0, ..., N_4) of the replicate's main-set labels. Then the label vector y is exchangeable: every arrangement of labels with counts N is equally likely, whatever H is. This holds in particular when the rows (h_j, y_j) are i.i.d. with y_j independent of h_j, but it is weaker than that and is all the proof needs.

**Claim.** Under H0_S1, P(R = r | H, F, N) = 1/(B + 1) for r = 1, ..., B + 1, so P(S1 > 0 | H, F, N) = 1/2 exactly, whatever ties the balanced accuracies produce. The unconditional statement follows by averaging.

**Proof.**
1. Condition on (H, F, N). Every element of G preserves the class counts N, so y ∘ σ is again an arrangement with counts N. Under H0_S1 it has the same conditional law as y.
2. The probe and the BA are deterministic, so s_j = f(H, y ∘ σ^j, F) for j = 0, ..., B. Replacing y by y ∘ σ maps (y, y∘σ, ..., y∘σ^B) to (y∘σ, ..., y∘σ^B, y), because σ^(B+1) = id. The score tuple (s_0, ..., s_B) therefore has the same law as its cyclic shift (s_1, ..., s_B, s_0). This holds for any distribution of s, including atoms (ties).
3. The tie-break t is a uniform random permutation of {0, ..., B}, independent of (H, F, N, y). Its cyclic shift is also uniform and independent, so the pairs (s_j, t_j) are jointly shift-invariant in law. They are all distinct because the t_j are distinct.
4. Let R_j be the rank of position j under the lexicographic order. By shift-invariance, R_0, R_1, ..., R_B all have the same law. They always form a permutation of {1, ..., B + 1}, so for each r, sum_j P(R_j = r) = 1. Hence P(R_0 = r) = 1/(B + 1).
5. The upper half {(B+1)/2 + 1, ..., B + 1} has (B + 1)/2 of the B + 1 equally likely ranks, so P(S1 > 0) = 1/2.

**Conditioning.** Let E be the event that coverage holds and the probe converges for all B + 1 assignments of G. E depends only on (H, F) and the set of assignments, so it is symmetric under the shift, so the law in step 4 also holds given E, and R is defined whenever E holds. Other VALID checks need not be shift-invariant; the T arm, for example, uses y itself. But they are only extra requirements: {S1 rejects and every replicate VALID} is contained in {S1 rejects and E holds in every replicate}. So P(S1 rejects ∧ all VALID) ≤ P(S1 rejects | E in every replicate) ≤ α, and any non-VALID replicate makes the decision INCONCLUSIVE.

**Across replicates.** Replicates use independent data, so the S1 indicators are i.i.d. Bernoulli(1/2) and N+ ~ Bin(20, 1/2) exactly.

**Checked by:**
- exact enumeration of every multiset of B + 1 scores from three values (B = 1, 3, 5), over all arrangements and all tie-break permutations;
- simulation with untied, heavily tied and all-tied scores;
- a pipeline-faithful null run of the real probe, group and tie-break (`POWER_ANALYSIS.md`).

**Why the superseded rule failed.** The superseded rule was S1 = BA(A) − median of B permuted BAs, with zeros dropped. It is not exact. With exchangeable two-valued scores, P(s = 1) = 0.3 and B = 5, it gives P(S1 > 0 | S1 ≠ 0) = 0.687, which is anti-conservative (0.313 for P(s = 1) = 0.7). The test suite enumerates this case.

### 7.3 Calibration gate C

- **Component hypotheses:**
  - H0_task: q(C_task) ≤ 1/2 against H1_task: q(C_task) > 1/2;
  - H0_twin: q(C_twin) ≤ 1/2 against H1_twin: q(C_twin) > 1/2.
- **Component tests.** Each is the exact binomial test of 7.1 over all 20 valid replicates, with zeros non-positive.
- **Gate.** H0_C = H0_task ∪ H0_twin. C is rejected (the gate passes) only when **both** component tests reject at α = 0.05: N+(C_task) ≥ 15 **and** N+(C_twin) ≥ 15.
- **Level.** This is Berger's intersection-union test. Under H0_C at least one component null is true, and that component rejects with probability at most α. So P(gate passes | H0_C) ≤ α, with no adjustment.

### 7.4 Fixed sequence

At α = 0.05 per step:
1. C (the gate above);
2. S1;
3. S2.

A step is tested only if every earlier step rejected, so the FWER is at most α under arbitrary dependence.

| Decision | When |
|---|---|
| PRECHECK_FAILED | preflight, binding, provenance, pins, P1 or benchmark check failed |
| INCONCLUSIVE | fewer than 20 VALID, no valid observations, or S1 not rejected |
| POSITIVE_CONTROL_FAILED | C not rejected (S1 and S2 reported, marked not tested) |
| TARGET_DECODABLE_TWIN_UNRESOLVED | C and S1 rejected, S2 not |
| SUPPORTED | C, S1 and S2 rejected |

**Worked example.** C is positive in 20/20 replicates for both components. S1 has 17 upper-half ranks and 3 lower-half ranks, so p = 1351/2^20 = 1.3e-3 and it rejects. S2 is 13 positive and 7 non-positive, so p = 137980/2^20 = 0.132 and it does not reject. The decision is TARGET_DECODABLE_TWIN_UNRESOLVED.

## 7.5 Representation diagnostics (P1 inputs, 10^6 pairs, layer 32 unchanged)

`validation/representation/representation_diagnostics.json` gives, for both models, per-unit variance, the exact-zero fraction, numerical rank, the singular-value spectrum and effective rank. Only the P1 keys and plaintexts were read; the target was not read and no probe was fitted.

| | published | twin |
|---|---|---|
| units with nonzero variance | 14 | 60 |
| units exactly 0 on all 10^6 inputs | 50 | 4 |
| overall fraction of exact zeros | 0.905 | 0.461 |
| numerical rank, centred (numpy rule, float64 eps) | 14 | 60 |
| numerical rank, singular values above 1e-2 of the largest | 12 | 53 |
| effective rank (entropy of singular values) | 8.94 | 40.1 |
| participation ratio of covariance eigenvalues | 3.64 | 19.1 |
| variance in the top 5 directions | 91.3% | 41.4% |
| effective rank after the probe's standardisation | 11.5 | 55.7 |

**What the earlier note got wrong.** It said "effectively 14-dimensional", from the 14 non-constant units on 2,048 pairs. That is not supported:
- On 10^6 inputs the same 14 units are the only non-constant ones, and they span 14 numerical dimensions under the float64 rule.
- Two of them have variances of only 8.4e-8 and 8.2e-11, and the variance is concentrated: the entropy effective rank is 8.9 and the participation ratio 3.6.
- The StandardScaler rescales every non-constant unit to unit variance, including the two near-silent ones, so the geometry the probe sees has effective rank 11.5.

The representation therefore has 14 non-constant units but no single "dimension". These diagnostics describe the layer and do not change it.

## 8. Probe accessibility versus information

Every comparison is between balanced accuracies of one fixed linear probe. A higher score means the target is more *linearly accessible* from that representation, not that the representation holds more information.

The representation is a deterministic function of the 64-bit input X. By the data-processing inequality, I(T; h(X)) <= I(T; X). So `raw_gap > 0` can never mean "the network adds information"; it can only mean "a linear probe finds more of it". The certificate and these documents use only the accessibility wording.

## 9. What a result does and does not show

**SUPPORTED shows:** consistent linear decodability of the coarsened CE2 target from this layer, beyond its marginal distribution and beyond one fixed untrained twin, by a probe that demonstrably recovers the genuine-versus-random task.

**It does not show:**
- that the network uses T;
- that it learned a cryptanalytic strategy;
- anything about other checkpoints, layers or twins.

**A non-rejection** is not evidence of absence: a linear probe can miss non-linear structure.
