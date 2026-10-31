# AutoND CE2: power, as a scenario-based sensitivity analysis (design r3)

**This is not a guarantee of power.** It is a scenario-based sensitivity analysis, conditional on its
assumptions. Every number below is the probability that the complete preregistered procedure returns
SUPPORTED (or, where stated, that the primary test rejects) *under a stated model* of the run-level
coefficients. The true effect is unknown, and so is the dependence between the published-model and
untrained-twin coefficients. No coefficient observed on the published model was used. The 90% target
is **not** reached uniformly: whether it is reached depends on the effect, on the twin's share of the
association and on the twin-primary error correlation (section 3).

Source: `validation/r3/power_final.json` (SHA-256 `c3eeb232...`, bound in
`DESIGN["statistics"]["power"]`), produced by `--power` on 2026-10-31 for the final r3 design. The r3
draft-1 output `validation/r3/power.json` (`a23355fd...`) is kept unchanged. Code: `power.py`. 50,000
Monte Carlo replications per simulated scenario. Each estimate carries its Monte Carlo standard error
and a 95% interval.

**Interval method.** `clopper_pearson()` has two branches. For n <= 2000 it is exact Clopper-Pearson
inversion. For n > 2000 it is **not** exact Clopper-Pearson: it inverts a continuity-corrected normal
approximation to the binomial tail. Every simulated power estimate here has n_sim = 50,000, so every
interval in sections 3 and 5 (and the "lower 95% bound" used in section 3) comes from the approximate
branch. Each output row records its method in `ci95_method`. The exact branch is used only for the
zero-frequency bounds of section 4.
- **Reproduced, not new:** the joint-power grid, the permutation-dependence rows and the exact primary
  power tables are seeded. The r3 run reproduces `validation/r2/power.json` exactly (tested).
- **Reproduced from r3 draft 1:** the zero-primary-run scenarios and the exact power by n_effective
  (tested).
- **New in the final r3 run:** the control zero-difference scenarios (section 5).
- **New sample each run:** the pipeline-faithful check uses fresh `os.urandom` targets, so its numbers
  are a new sample in each of r2, r3 draft 1 and final r3, not a reproduction.

## 1. Procedure simulated

Per scenario, 20 (or 10, 30) runs:
1. an exact two-sided sign test of H0: p_+ = 1/2, with p_+ = Pr(rho_i > 0 | rho_i != 0, run i is valid);
2. only if step 1 rejects at alpha = 0.05, two exact one-sided sign tests of
   H0,c: Pr(D_i,c > 0 | D_i,c != 0) = 1/2 against "> 1/2", D_i,c = |rho_primary,i| - |rho_c,i|, for the
   permuted target and the twin, with Holm at alpha.

This is the analysis in `analysis.py`. `tests/test_theory_statistics.py` checks that it agrees with the
verifier's independent decision rule on 200 random cases.

## 2. Run-level model and its assumptions

Effects are in units of se = (N - 1)^-1/2. At N = 10^6, one unit is a rho of 0.001.

    rho_primary = delta + e_p
    rho_perm    = e_q,                  corr(e_p, e_q) in {-0.3, 0, 0.3}
    rho_twin    = twin_ratio * delta + e_t,  twin_ratio in {0, 0.25, 0.5, 0.75}, corr(e_p, e_t) in {-0.5, 0, 0.5, 0.9}

**Assumptions.**
- The errors are jointly Gaussian. The model is continuous, so it assigns probability 0 to an exact zero
  run-level coefficient and to an exact zero paired control difference. Every simulated run therefore
  enters every sign test: n_effective = n_runs. The real analysis removes exact zeros; section 5 shows
  what that costs.
- Runs are independent and identically distributed.
- The variance of rho_hat is (N - 1)^-1 under every scenario (checked in section 4, under the null and
  small synthetic alternatives only).

## 3. Smallest simulated effect whose Monte Carlo lower 95% bound for P(SUPPORTED) is at least 0.90

The grid of delta is {0.5, 1, 1.5, 2, 2.5, 3, 4}. "none" means no value up to delta = 4 reaches the bound.

| runs | twin_ratio | corr -0.5 | corr 0 | corr 0.5 | corr 0.9 |
|---|---|---|---|---|---|
| 10 | 0 | 3.0 | 3.0 | 3.0 | 4.0 |
| 10 | 0.25 | 4.0 | 4.0 | 3.0 | 3.0 |
| 10 | 0.5 | none | none | 4.0 | 3.0 |
| 10 | 0.75 | none | none | none | 4.0 |
| **20** | 0 | 2.5 | 2.5 | 2.0 | 2.5 |
| **20** | 0.25 | 2.5 | 2.5 | 2.0 | 2.0 |
| **20** | 0.5 | 4.0 | 3.0 | 2.5 | 2.0 |
| **20** | 0.75 | none | none | 4.0 | 2.5 |
| 30 | 0 | 2.0 | 2.0 | 2.0 | 2.0 |
| 30 | 0.25 | 2.0 | 2.0 | 2.0 | 2.0 |
| 30 | 0.5 | 2.5 | 2.5 | 2.0 | 2.0 |
| 30 | 0.75 | none | 4.0 | 3.0 | 2.0 |

**Dependence of the permuted control** (20 runs, twin_ratio 0.25, twin corr 0.5): at delta = 2,
P(SUPPORTED) is 0.917, 0.912 and 0.915 for perm corr -0.3, 0 and 0.3. At delta = 3 it is 1.000 in all
three. Each estimate has a Monte Carlo SE of about 0.0013.


**Correction to r1 (kept from r2).** r1 stated that 20 runs give about 90% joint power at |rho| of about
0.0020-0.0028. That holds for twin_ratio <= 0.25, and at twin_ratio 0.5 only when the twin's errors are
positively correlated with the primary's (corr >= 0.5).
- If the untrained twin shares half the association and its errors are uncorrelated or negatively
  correlated with the primary's, the requirement rises to delta = 3-4, i.e. |rho| of about
  0.003-0.004.
- If it shares three quarters, 20 runs may not reach 90% at any effect up to delta = 4.
- The run count is frozen at 20 in r3. It was not changed to improve these numbers.

## 4. Is the Gaussian run-level model adequate? (pipeline-faithful check)

The model assumes that, for the real CE2 target (an integer with heavy ties), Spearman's rho of N pairs
has variance (N - 1)^-1 and is close to normal. The check simulates the actual estimator (average ranks,
no jitter) on the real model-free target of 200,000 freshly generated genuine pairs, subsampled to
N = 50,000, with synthetic scores `lam * standardised(target) + N(0, 1)` and 400 replications per row.
Under r3 it also computes, from the same synthetic scores, the permuted-target coefficient, and counts
exact zeros.

**Final r3 run (`validation/r3/power_final.json`, new sample):**

| lam | mean rho | (N-1) var(rho) | Monte Carlo SE | skewness | P(rho > 0), simulated | P(rho > 0), Gaussian | exact zeros: rho / rho_perm / D_perm |
|---|---|---|---|---|---|---|---|
| 0 | 0.0001 | 0.966 | 0.068 | 0.07 | 0.488 | 0.509 | 0 / 0 / 0 of 400 |
| 0.002 | 0.0021 | 1.087 | 0.077 | 0.11 | 0.668 | 0.674 | 0 / 0 / 0 of 400 |
| 0.005 | 0.0044 | 1.017 | 0.072 | -0.05 | 0.833 | 0.834 | 0 / 0 / 0 of 400 |

**r3 draft-1 run (`validation/r3/power.json`, new sample, kept unchanged):**

| lam | mean rho | (N-1) var(rho) | Monte Carlo SE | skewness | P(rho > 0), simulated | P(rho > 0), Gaussian | exact zeros: rho / rho_perm / D_perm |
|---|---|---|---|---|---|---|---|
| 0 | 0.0002 | 1.040 | 0.074 | 0.12 | 0.525 | 0.518 | 0 / 0 / 0 of 400 |
| 0.002 | 0.0019 | 1.146 | 0.081 | 0.13 | 0.645 | 0.653 | 0 / 0 / 0 of 400 |
| 0.005 | 0.0046 | 1.028 | 0.073 | -0.11 | 0.873 | 0.847 | 0 / 0 / 0 of 400 |

**r2 run (`validation/r2/power.json`, reused, unchanged):**

| lam | mean rho | (N-1) var(rho) | Monte Carlo SE | skewness | P(rho > 0), simulated | P(rho > 0), Gaussian |
|---|---|---|---|---|---|---|
| 0 | -0.0001 | 0.943 | 0.067 | -0.02 | 0.508 | 0.487 |
| 0.002 | 0.0018 | 1.126 | 0.080 | 0.34 | 0.633 | 0.647 |
| 0.005 | 0.0047 | 1.100 | 0.078 | 0.03 | 0.860 | 0.843 |

**Reading.**
- Under the null, the variance agrees with (N - 1)^-1 in all three runs (0.94, 1.04, 0.97, each within
  about one Monte Carlo SE of 1).
- Under small alternatives the six values are 1.13 and 1.10 (r2), 1.15 and 1.03 (draft 1), 1.09 and
  1.02 (final): 2-15% above 1, each within about 1.8 SE. All six are above 1, which suggests a small
  real excess, but it is not resolved. If real, the required delta grows by up to about 7%.
- Simulated and Gaussian sign probabilities agree within about 1.5 Monte Carlo SE (SE about 0.017-0.025
  per row).
- The r2 skewness of 0.34 at lam = 0.002 (2.8 SE) was not reproduced (0.13 and 0.11, about 1 SE). It is
  consistent with Monte Carlo noise; it is not explained.
- No exact zero (of rho, rho_perm or D_perm) occurred in 2,400 synthetic replications (draft 1 and
  final). The exact one-sided 95% upper bound is 0.0075 per replication for each row of 400 (0.00125
  pooled). **This bounds the synthetic-score estimator only.** The twin, and D_twin, are not simulated.

**Limits.**
- N is 5 x 10^4, not 10^6.
- The score model is synthetic. The real published-model and twin scores (float32 sigmoid outputs, with
  their own tie structure) have a different joint structure, which cannot be simulated without the
  models. The check validates properties of the Spearman estimator with the real target. It does not
  validate the joint published-model / twin / permutation model of section 2.
- The twin-primary dependence (corr) is therefore empirically unknown, and the table spans it.

## 5. Zero ties (new in r3)

The sign tests remove exact zeros. A defined rho_i = 0 keeps the run VALID (it counts in n_valid) but
removes it from the primary denominator, so n_effective can be below 20 even when all 20 runs are valid.
In a control test, a zero paired difference D_i,c = 0 is removed in the same way. A run with
rho_primary,i = 0 and a nonzero control coefficient has D_i,c = -|rho_c,i| < 0: it counts *against*
separation.

**Exact primary power conditional on n_effective** (no simulation; P(reject) for sign probability p_+;
the threshold is the number of same-sign coefficients needed):

| n_effective | threshold | p_+ = 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|
| 20 | 15 | 0.246 | 0.416 | 0.617 | 0.804 | 0.933 |
| 19 | 15 | 0.150 | 0.282 | 0.465 | 0.673 | 0.856 |
| 18 | 14 | 0.189 | 0.333 | 0.519 | 0.716 | 0.879 |
| 17 | 13 | 0.235 | 0.389 | 0.574 | 0.758 | 0.901 |
| 16 | 13 | 0.134 | 0.246 | 0.405 | 0.598 | 0.790 |
| 15 | 12 | 0.173 | 0.297 | 0.461 | 0.648 | 0.823 |
| 14 | 12 | 0.084 | 0.161 | 0.281 | 0.448 | 0.648 |

Power is not monotone in n_effective, because the discrete binomial test's attained size jumps
(19 needs the same 15 as 20). Below n_effective = 6 the primary test cannot reject at alpha = 0.05
(smallest p = 2/2^5 = 0.0625). The first Holm step of a control test needs n_effective >= 6 as well.

**Joint power when k of the 20 primary coefficients are exactly 0** (simulated, 50,000 replications,
Monte Carlo SE <= 0.0023; k = 0 is the section 3 grid):

| twin_ratio | twin corr | delta | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|---|---|
| 0 | 0 | 2.0 | 0.896 | 0.780 | 0.596 | 0.362 |
| 0 | 0 | 2.5 | 0.995 | 0.981 | 0.926 | 0.780 |
| 0 | 0 | 3.0 | 1.000 | 0.999 | 0.995 | 0.967 |
| 0 | 0.5 | 2.0 | 0.903 | 0.791 | 0.615 | 0.380 |
| 0 | 0.5 | 2.5 | 0.994 | 0.975 | 0.921 | 0.770 |
| 0.25 | 0 | 2.0 | 0.849 | 0.714 | 0.516 | 0.289 |
| 0.25 | 0 | 2.5 | 0.985 | 0.952 | 0.865 | 0.677 |
| 0.25 | 0 | 3.0 | 0.999 | 0.995 | 0.978 | 0.910 |
| 0.25 | 0.5 | 2.0 | 0.912 | 0.808 | 0.636 | 0.393 |
| 0.25 | 0.5 | 2.5 | 0.997 | 0.983 | 0.940 | 0.811 |
| 0.5 | 0 | 3.0 | 0.942 | 0.871 | 0.741 | 0.539 |
| 0.5 | 0 | 4.0 | 0.996 | 0.985 | 0.951 | 0.857 |
| 0.5 | 0.5 | 2.5 | 0.975 | 0.929 | 0.819 | 0.614 |
| 0.5 | 0.5 | 3.0 | 0.998 | 0.991 | 0.963 | 0.873 |

(The full grid, including delta = 4 for every row, is in `power_final.json`.) A fixed k is a scenario,
not an estimate. Under the Gaussian model of section 2 the k-zero runs are forced: the model has no
mechanism that produces them.

**Joint power when k_perm paired differences D_perm and k_twin differences D_twin are exactly 0**
(simulated, 50,000 replications, Monte Carlo SE <= 0.0023; the primary coefficients are untouched, so
the primary test keeps n_effective = 20). The zero-D_perm runs are the first k_perm runs and the
zero-D_twin runs the last k_twin, so they are disjoint. The control tests then have
n_effective = 20 - k_perm and 20 - k_twin (recorded per row in the output). Columns are (k_perm, k_twin);
(0, 0) is the section 3 grid:

| twin_ratio | twin corr | delta | (0,0) | (1,0) | (2,0) | (3,0) | (0,1) | (0,2) | (0,3) | (1,1) | (2,2) | (3,3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2.0 | 0.896 | 0.908 | 0.912 | 0.867 | 0.904 | 0.913 | 0.871 | 0.904 | 0.922 | 0.846 |
| 0 | 0 | 2.5 | 0.995 | 0.995 | 0.996 | 0.991 | 0.996 | 0.996 | 0.991 | 0.995 | 0.997 | 0.987 |
| 0 | 0.5 | 2.0 | 0.903 | 0.908 | 0.919 | 0.875 | 0.910 | 0.918 | 0.880 | 0.909 | 0.928 | 0.853 |
| 0 | 0.5 | 2.5 | 0.994 | 0.995 | 0.995 | 0.991 | 0.994 | 0.996 | 0.988 | 0.994 | 0.996 | 0.985 |
| 0.25 | 0 | 2.0 | 0.849 | 0.854 | 0.864 | 0.821 | 0.865 | 0.883 | 0.811 | 0.858 | 0.887 | 0.786 |
| 0.25 | 0 | 2.5 | 0.985 | 0.986 | 0.985 | 0.982 | 0.987 | 0.990 | 0.974 | 0.986 | 0.990 | 0.969 |
| 0.25 | 0.5 | 2.0 | 0.912 | 0.923 | 0.932 | 0.886 | 0.917 | 0.925 | 0.891 | 0.918 | 0.935 | 0.864 |
| 0.25 | 0.5 | 2.5 | 0.997 | 0.997 | 0.997 | 0.992 | 0.997 | 0.997 | 0.994 | 0.997 | 0.998 | 0.990 |
| 0.5 | 0 | 2.0 | 0.604 | 0.610 | 0.616 | 0.589 | 0.647 | 0.695 | 0.563 | 0.638 | 0.691 | 0.550 |
| 0.5 | 0 | 2.5 | 0.832 | 0.834 | 0.831 | 0.829 | 0.859 | 0.887 | 0.786 | 0.858 | 0.886 | 0.786 |
| 0.5 | 0 | 3.0 | 0.942 | 0.941 | 0.941 | 0.941 | 0.954 | 0.963 | 0.913 | 0.953 | 0.964 | 0.913 |
| 0.5 | 0.5 | 2.0 | 0.804 | 0.820 | 0.822 | 0.781 | 0.834 | 0.853 | 0.765 | 0.823 | 0.856 | 0.747 |
| 0.5 | 0.5 | 2.5 | 0.975 | 0.974 | 0.975 | 0.970 | 0.979 | 0.984 | 0.958 | 0.978 | 0.984 | 0.953 |
| 0.5 | 0.5 | 3.0 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.999 | 0.995 | 0.998 | 0.999 | 0.994 |

(The full grid, delta 2, 2.5, 3 and 4 for every row, is in `power_final.json`.) Across all 216
scenarios the change from (0, 0) lies between -0.063 and +0.091. The effect of zero differences is
small and not monotone. A zero difference is not a "loss": the run leaves the control's denominator, and
the discrete one-sided thresholds move. At alpha = 0.05 (the second Holm step), 20 runs need 15
positives (75%), 19 need 14 (74%), 18 need 13 (72%) and 17 need 13 (76%). At alpha/2 (the first step),
20 need 15, 19 need 15, 18 need 14 and 17 need 13. The increases seen with (0, 2) when the twin is the
harder control come from these thresholds. They are not a reason to expect, or to want, zero
differences. With every difference of a control exactly 0 (n_effective = 0), that control is
NO_EFFECTIVE_OBSERVATIONS and SUPPORTED is impossible (`test_control_zero_difference_scenarios`).

These scenarios fix the number of zeros. They do not model a mechanism that would produce zeros, nor
any dependence between a zero and the sign the run would otherwise have had.

**What remains unknown.** The probability that a real run gives an exact float64 zero for rho_primary,
for a control coefficient, or for a paired difference D_perm or D_twin. It depends on the joint tie
structure of the real published-model and twin scores with the target at N = 10^6, which no simulation
here reproduces. The synthetic check found none in 2,400 replications at N = 5 x 10^4. It did not
simulate the twin at all. That is not evidence about the real models. The production analysis reports n_zero_ties_removed and n_effective for every
test, so any zero tie will be visible.

## 6. What is established and what is not

- **Established:**
  - The decision procedure has size at most alpha under the null (tested).
  - Power is monotone in delta and decreases as the twin shares more of the association (tested).
  - The Monte Carlo error of every simulated number above is small (SE <= 0.0023; intervals from the
    normal-tail branch, see "Interval method").
  - The exact conditional power of the primary test for every n_effective from 20 down to 5.
- **Not established:**
  - the true effect;
  - the twin's share of the association;
  - the twin-primary error correlation;
  - whether the variance under the real alternative matches the model at N = 10^6;
  - the probability of exact zero coefficients or differences, primary, D_perm or D_twin, for the
    real models (section 5);
  - joint power under any real zero mechanism (the scenarios fix counts only).

None of these may be estimated from the exploratory coefficients (section 7).

## 7. Exploratory coefficients already seen (disclosed, not used)

Development checks computed the real model's coefficient on small samples:
- r1: 4,096 pairs gave primary 0.105, permuted -0.005 and twin 0.020; a separate smoke run gave 0.077;
- r2 smoke (3 x 4,096 pairs, `validation/r2/exploratory_smoke_summary.json`): primary 0.080, 0.070 and
  0.093.

No smoke run was performed under r3. These coefficients were not used to choose any size, threshold or
scenario, and must not be. They are noisy small-sample estimates on a population the confirmatory runs
resample.
