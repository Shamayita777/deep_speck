# G-CE5 statistical protocol (r3)

## Quantities

- **N:** output of the bound depth-10 network (`256eb4a5…6737`).
- **D:** continuous aKR score, score / 2^12.
- **T:** CE2 single-trail probability.
- **LR:** probability of the ciphertexts' masked (δl‖δv) cell in the 5-round Algorithm-1 table.

## Primary estimand

Δ_r = atanh ρ_s(N, D | real) − atanh ρ_s(N, T | real), computed on the same real pairs of independent replicate r. ρ_s is Spearman's rank correlation with average ranks.

- **Real pairs only.** T is the probability of the trail a real pair followed, and is undefined for random pairs. ρ_s(N, D | random) is descriptive.
- **Dependence.** The two correlations share N and the pairs. Taking the paired difference within each replicate, and inferring only across independent replicates, absorbs that dependence. Pairs are never units of inference.

## Interval and outcomes (revised in r3)

The interval for the median of Δ_r is the order-statistic interval (Δ_(k), Δ_(R−k+1)), with k the largest integer such that P(Bin(R, ½) ≤ k − 1) ≤ 0.025. Exact values on the registered grid (tested in both implementations):

| R | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| k | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
| coverage | 0.9586 | 0.9567 | 0.9572 | 0.9590 | 0.9615 | 0.9643 | 0.9672 | 0.9700 | 0.9727 |

| outcome | condition |
|---|---|
| DISTRIBUTION_GREATER | lower ≥ m |
| TRAIL_GREATER | upper ≤ −m |
| EQUIVALENT | −m < lower and upper < m |
| INCONCLUSIVE | otherwise |

**What changed from r2.** r2 declared DISTRIBUTION_GREATER when lower > 0 and upper ≥ m. That treated an interval such as (0.01, 0.20) as a practically meaningful difference, although it is compatible with a difference of 0.01. Under r3:
- a directional verdict requires the **whole** interval beyond the margin;
- an interval that excludes 0 but crosses the margin is INCONCLUSIVE, with the descriptive qualifier DIRECTION_D or DIRECTION_T;
- the qualifier is never reported as practical significance;
- an interval that excludes 0 inside (−m, m) is EQUIVALENT: a real but negligible difference;
- a non-significant interval wider than the margin is INCONCLUSIVE.

All are tested, including boundary values, and the verifier implements the same rule independently. All registered replicates must be valid.

## Margin m = 0.10 (set on 2026-11-05, before calibration)

**Proposal:** 0.10 on the Fisher-z scale. The Fisher z transform approximately stabilises the sampling variance of a correlation, so a fixed margin has roughly constant meaning across levels of ρ_s.

Under r3, m is the smallest difference the experiment can certify. A directional verdict needs the whole interval beyond ±m. A large m makes directional verdicts hard to obtain and equivalence easy; a small m does the reverse.

What each candidate margin means at the G-CE2 level, ρ_s = 0.807 (z = 1.1183):

| m (Fisher z) | ρ_s at z + m | ρ_s at z − m | change in ρ_s² at z + m |
|---|---|---|---|
| 0.05 | 0.8237 (+0.0167) | 0.7888 (−0.0182) | +0.027 |
| 0.10 (proposal) | 0.8392 (+0.0322) | 0.7692 (−0.0378) | +0.053 |
| 0.20 | 0.8664 (+0.0594) | 0.7251 (−0.0819) | +0.099 |

ρ_s² is shown only as a scale reference. For a rank correlation it is not a proportion of variance explained in the regression sense.

The primary outcome is also reported at 0.05 and 0.20 (`primary.margin_sensitivity`, descriptive). No data have informed the proposal, and the blinded calibration cannot inform it.

## Secondary analyses (descriptive or Holm-adjusted; none replaces the primary)

1. **Resolution-matched Δ.** D re-ranked with T's tie groups (T takes few values, powers of two), so that resolution alone cannot favour D. It uses the same interval and outcomes.
2. **Δ_LR.** The LR score in place of D.
3. **Incremental predictive value.**
   - Model: OLS of the normal scores of N on the normal scores of T, of D, and of both (cubic terms, plus T·D in the combined model). The model class is fixed, with no hyperparameters.
   - Fit on the first half of each replicate's real pairs; evaluate MSE on the second half. Normal scores are computed within each half, and all three models use the same evaluation pairs.
   - Δ_D = L(M_T) − L(M_TD) and Δ_T = L(M_D) − L(M_TD), each tested with an exact two-sided sign test over replicates and Holm-adjusted over the two.
   - A positive Δ_D means D carries predictive information about N beyond T. It is not causal reliance or mechanism. A leakage test is included.
4. **Descriptive correlations:** ρ(N, D | real), ρ(N, D | random), ρ(N, T | real), ρ(N, LR | real), ρ(N, D_matched | real), and the number of distinct T values.

**Multiplicity:** the primary is a single test, the incremental family has two tests under Holm, and everything else is descriptive.

## Sample size

**Pairs per replicate: 10^6 (recorded in the design).**
- Rule, fixed before calibration: the largest of 10^4, 3·10^4, 10^5, 3·10^5 and 10^6 whose projected time is at most 12 h and whose raw storage is at most 10 GB, at the worst case R = 60.
- Measured rate (BENCHMARK.md): 0.109–0.110 ms per real pair, where one real pair covers one real and one random pair (generation, inference, aKR and LR scoring).
- At 10^6 pairs and R = 60: 1.84 h and 2.2 GB.

**Replicates: unset.**
- **Planning model:** Δ_r ~ N(Δ, σ_plan²).
- **Rule:** choose the smallest R in {20, 25, …, 60} for which, for all three requirements, the one-sided Clopper–Pearson lower bound of the Monte Carlo estimate at level 0.01/3 (Bonferroni) is ≥ 0.80. The requirements are:
  - P(EQUIVALENT | Δ = 0);
  - P(DISTRIBUTION_GREATER | Δ = 2m), which requires the lower limit ≥ m;
  - P(TRAIL_GREATER | Δ = −2m).
- **Default:** if no R qualifies, R = 60 and the shortfall is reported.
- **Also reported:** outcome probabilities at Δ = m and Δ = m/2. At Δ = m, a directional verdict is expected at most about half the time.

**Monte Carlo** (`analysis.planning_draws`, `outcome_probabilities`, `choose_replicates`):
- **Draws.** For each R, one array Z of 50,000 × R standard-normal draws is generated (seed [0, R]). All five conditions at that R use x = Δ + σ_plan·Z on that same array; these are common random numbers. A test checks that each R draws once and that all five conditions receive that array, and another checks the outcome counts against a manual computation from Z.
- **Precision.** Each estimate is reported with its standard error (≤ 0.0018 near 0.8) and its count.
- **Guarantee.** With the Bonferroni level, for a given R the probability that any of the three true probabilities lies below its bound is at most 0.01. This is a simultaneous guarantee over the three requirements at one R. **No joint guarantee is claimed across the R grid.**
- **Borderline case.** A point estimate marginally above 0.80 cannot select R. This is tested directly.

**Planning spread σ_plan** (`analysis.planning_sd`):
- σ_plan = max(s, 1.4826·MAD) · √(df / χ²_{0.20, df}), with df = 9 for 10 calibration replicates.
- s · √(df / χ²_{0.20, df}) is the exact one-sided 80% upper confidence bound for σ when the values are i.i.d. normal.
- σ_plan is never below that bound, so P(σ ≤ σ_plan) ≥ 0.80 under normality. This is checked by simulation in the tests, with coverage ≥ 0.79 over 4,000 trials.
- The MAD term carries no coverage guarantee of its own; it only protects against a sample SD deflated by the data. No coverage is claimed for non-normal data.

**Calibration.**
- 10 replicates of 10^6 real pairs plus their random class, on fresh data, taking about 18 minutes. (r2 specified 5.)
- **Blinded.** Only centred Δ values and the planning-spread record (`planning_sd`) are written. The location of Δ and every correlation are discarded, and a test checks that `calibration.json` contains neither.
- **Refusals.** The stage refuses:
  - while the margin is unset;
  - without the registered, passing table-validation record;
  - for any pair count other than the registered one;
  - if `calibration.json` already exists;
  - below 2 GiB of available memory.
- **Status:** not run, because `primary.margin` is unset.

**Assumptions.**
- Replicates are independent and identically distributed: the data are fresh draws and the model is fixed.
- The normal model is used for planning only. The registered inference is distribution-free.
