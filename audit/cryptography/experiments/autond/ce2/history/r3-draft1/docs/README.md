# AutoND CE2: theory consistency of the published AutoND SPECK32/64 round-8 DBitNet

**Status (2026-10-31):**
- design r3 `7e69ce6ccb0425dd496820e91dc990a15c23a239800b0a9382a18b621667ce0c`; all design markers
  `FROZEN`; preregistration `PENDING_REVIEW` (not approved);
- no production run and no certificate;
- exploratory smoke runs only, all under r2 (disclosed in `POWER_ANALYSIS.md` section 7).

r3 supersedes r2 (`ef3d764a...`, never approved, archived in `history/r2/`). It corrects the written
hypothesis and estimand so that they describe the sign test the code already performs. No computation,
parameter, sample size, sampling rule, checkpoint, alpha or control changed. See
`DESIGN["revision"]` and section "What r3 corrected" below.

**Question.** Across independent evaluation runs, does the run-level Spearman association between the
published model's output scores and the trajectory-derived Markov-model single-characteristic
probability proxy show a systematic imbalance between positive and negative nonzero coefficients? The
study is observational. The primary hypothesis is TWO-SIDED: no direction is prespecified.

## What is measured and what is tested

- **rho_i (per-run statistic).** In run i, the Spearman rank correlation (average ranks, no jitter)
  between the published model's sigmoid scores and the target T over that run's 10^6 genuine pairs.
- **p_+ (primary estimand).** p_+ = Pr(rho_i > 0 | rho_i != 0, run i is valid): the probability that a
  valid run's nonzero coefficient is positive.
- **Primary test.** An exact two-sided binomial sign test of H0: p_+ = 1/2 against H1: p_+ != 1/2, over
  the nonzero rho_i of the valid runs; p = min(1, 2 x smaller binomial tail).
- **What a rejection establishes.** Positive and negative nonzero run-level coefficients do not occur
  with equal probability: the sign of the association is systematically imbalanced across runs. It is
  not a test of E[rho_i] = 0 and says nothing about the shape of the rho_i distribution beyond its sign
  balance. A non-rejection establishes neither symmetry nor zero association.
- **Descriptive summaries.** The individual rho_i, their median (with a distribution-free order-statistic
  interval and a run-level bootstrap as sensitivity), interquartile range, minimum, maximum and sign
  counts are reported. They describe magnitude; they do not replace p_+.

**Zero coefficients.** A defined rho_i that equals 0.0 exactly is a tie:
- the run stays VALID and counts in `n_valid`;
- it is excluded from the sign-test denominator and counted in `n_zero_ties_removed`;
- it is never turned into a positive or negative sign;
- `n_effective` = `n_valid` - `n_zero_ties_removed` is the sign test's sample size. With 20 valid runs it
  is below 20 whenever a zero occurs, and the analysis reports both numbers.

**Undefined coefficients.** If any of the three coefficients of a run (primary, permuted target, twin)
is undefined, the run is EXCLUDED with its recorded reason. "Undefined" means the conditions in
`analysis.definedness()`: constant score, constant target, non-finite values, shape mismatch, fewer
than 3 pairs, or a non-finite coefficient. An undefined coefficient is never treated as zero and never
given a value. All 20 runs must be valid (no reserve runs). With an excluded run the decision is
INCONCLUSIVE.

**Control tests.** Only if the primary test rejects, for each control c (permuted target, untrained
twin), D_i,c = |rho_primary,i| - |rho_c,i| is tested with an exact one-sided sign test:
H0,c: Pr(D_i,c > 0 | D_i,c != 0) = 1/2 against H1,c: Pr(D_i,c > 0 | D_i,c != 0) > 1/2, with Holm across
the two at alpha = 0.05.
- A zero difference is a tie: removed from that test's denominator and counted.
- A run with rho_primary,i = 0 and a nonzero control coefficient gives D_i,c < 0, a negative sign.
- **What separation establishes:** across runs, the published model's coefficient is more often larger
  in magnitude than the coefficient obtained with a permuted target (a target-alignment control) and with
  one fixed untrained DBitNet (a training control). It does not establish what the training changed
  inside the network.

**Decision.** SUPPORTED needs the primary rejection and both Holm-adjusted control rejections. A
primary rejection with an unresolved control is ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED. Anything else
is INCONCLUSIVE or PRECHECK_FAILED, never evidence of zero association.

## What r3 corrected

r2 wrote the null as full-distribution symmetry of the run-level population Spearman coefficient, and
named that coefficient as the estimand. The implemented test removes zero coefficients and tests only
whether nonzero signs are equally likely. These are different hypotheses. Symmetry about 0 implies
p_+ = 1/2, but not conversely: a skewed distribution with median 0 has p_+ = 1/2 and is not symmetric.
The sign test's null is therefore the larger hypothesis p_+ = 1/2. Its size is controlled over all of
it, it has no power against asymmetric distributions with p_+ = 1/2, and a non-rejection cannot support
symmetry. r3 states p_+ throughout.
The computation is unchanged, so any data would produce the same p-values and decision under r2 and r3.
What a result is allowed to mean has been corrected.

## Files (one place)

| | |
|---|---|
| `design.py` | the design (single source); r3 supersedes r2 (inferential wording corrected, no computational change) |
| `preregistration/preregistration.json` | deterministic serialisation of the design plus the approval block |
| `theory.py` | xdp+ by Lipmaa-Moriai closed form, carry-pair automaton and literal enumeration |
| `adapter.py` | genuine pairs, trajectories, frozen run data, theory-consistency capability |
| `analysis.py` | Spearman, exact sign tests, Holm, decision rule |
| `runner.py` | preflight, behavioural binding, journalled run state machine, finalisation |
| `verify.py` | independent verifier (imports nothing from the producer) |
| `power.py` | scenario-based sensitivity analysis and pipeline-faithful check |
| `validation/r2/` | r2 check outputs: model binding, gates at production sizes, power, exploratory smoke summary (kept unchanged) |
| `validation/r3/` | r3 power and zero-tie sensitivity (`power.json`, bound by hash in the design) and `VALIDATION_RECORD.md` (what was re-run, reused, unaffected or pending) |
| `history/r2/` | r2 design, preregistration, documents and the r2 versions of the files r3 changed, byte-identical (`SHA256SUMS`) |
| `history/r1/` | r1 design, preregistration, implementation and documents, byte-identical (`SHA256SUMS`) |
| `docs/` | this README, `RUNBOOK.md`, `POWER_ANALYSIS.md`, `VERIFICATION_REPORT.md` |
| `tests/` | `test_theory_statistics.py` and `test_r3_specification.py` (no TensorFlow), `test_pipeline.py` |

The CLI is `audit/cryptography/kaggle/autond_ce2_kaggle.py`. The target layer
(`audit/cryptography/targets/autond/`) holds the hash-verified, import-isolated upstream loader and the
strict checkpoint binding. The pinned upstream itself stays at `autond_ce1/AutoND-upstream/`, the only
copy.

## Design in brief

- **Data.** Each run draws 10^6 genuine pairs with the upstream `make_train_data` construction: a fresh
  64-bit master key per pair, shared by both members, and P1 = P0 XOR 0x400000. The source is
  `os.urandom`, so runs are not seed-replayable. Every raw array is retained and hashed.
- **Target.** T = -sum_r k_r, where xdp+ = 2^-k_r on the realised round transitions. Every factor
  is checked by two independent derivations (gate E).
- **Score.** The sigmoid output of the published checkpoint (SHA-256 `a7be788d...`), strictly bound:
  23 weight-bearing layers, 68 arrays bit-identical, 298,113 total parameters = 296,065 trainable
  + 2,048 non-trainable.
- **Controls, on the same frozen run dataset.**
  - **Permuted target:** a seeded permutation.
  - **Untrained twin:** one seeded, untrained DBitNet. A control result is conditional on this one
    twin, not a claim about untrained DBitNets in general.
- **Inference.** The unit is the run (20 runs, all required valid). See "What is measured and what is
  tested" above. Within-run p-values are descriptive only, with explicit UNDERFLOW and DEGENERATE
  states.

## Behavioural binding (model binding, not learning evidence)

The binding set is a fresh balanced set: n/2 genuine pairs and n/2 uniform random 64-bit inputs (the
upstream negative class), shuffled by a fixed seed. It requires:
- a one-sided 95% Wilson lower bound on accuracy above 0.5;
- exact label balance;
- all scores finite;
- generation gates passed.

The untrained twin's accuracy on the same set is recorded as a diagnostic, not as a gate.

The design uses n = 10^6. **This has not been run.** The only behavioural results so far are smoke
runs at n = 20,000:
- r1: accuracy 0.519, lower bound 0.513;
- r2: accuracy 0.5079, lower bound 0.5021 (twin 0.4975).

These do not give the assurance of the planned 10^6 set. The r2 margin at 20,000 is thin. The binding
shows that the checkpoint and input pipeline behave as a distinguisher. It is not evidence about what the
model learned.

## What the experiment does not establish

- No causal claim: the study is observational association only.
- Not that the network learned differential cryptanalysis, uses the characteristic, or has any internal
  cryptographic mechanism or representation.
- Not symmetry of the distribution of rho_i, not any property of its shape beyond the sign balance,
  and not E[rho_i] != 0: the primary test is about p_+.
- Not the magnitude of the association: that is described, not tested.
- The target is a proxy: not the fixed-key characteristic probability, and not the differential
  probability of the cipher.
- The results are conditional on one published checkpoint and one twin.
- Only genuine pairs are evaluated.
- The results do not amend or explain AutoND CE1.
