# AutoND CE2: power, as a scenario-based sensitivity analysis (design r2)

**This is not a guarantee of power.** It is a scenario-based sensitivity analysis. Every number below
is the probability that the complete preregistered procedure returns SUPPORTED *under a stated model*
of the run-level coefficients. The true effect is unknown, and so is the dependence between the
published-model and untrained-twin coefficients. No coefficient observed on the published model was
used.

Source: `validation/r2/power.json`, produced by `--power`. Code: `power.py`. 50,000 Monte Carlo
replications per scenario. Each estimate carries its Monte Carlo standard error and a 95% interval.

## 1. Procedure simulated

Per scenario, 20 (or 10, 30) runs:
1. an exact two-sided sign test on rho_primary;
2. only if step 1 rejects at alpha = 0.05, two exact one-sided sign tests on
   D_c = |rho_primary| - |rho_c| for the permuted target and the twin, with Holm at alpha.

This is the analysis in `analysis.py`. `tests/test_theory_statistics.py` checks that it agrees with the
verifier's independent decision rule on 200 random cases.

## 2. Run-level model

Effects are in units of se = (N - 1)^-1/2. At N = 10^6, one unit is a rho of 0.001.

    rho_primary = delta + e_p
    rho_perm    = e_q,                  corr(e_p, e_q) in {-0.3, 0, 0.3}
    rho_twin    = twin_ratio * delta + e_t,  twin_ratio in {0, 0.25, 0.5, 0.75}, corr(e_p, e_t) in {-0.5, 0, 0.5, 0.9}

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

**Correction to r1.** r1 stated that 20 runs give about 90% joint power at |rho| of about
0.0020-0.0028. That holds for twin_ratio <= 0.25, and at twin_ratio 0.5 only when the twin's errors are
positively correlated with the primary's (corr >= 0.5).
- If the untrained twin shares half the association and its errors are uncorrelated or negatively
  correlated with the primary's, the requirement rises to delta = 3-4, i.e. |rho| of about
  0.003-0.004.
- If it shares three quarters, 20 runs may not reach 90% at any effect up to delta = 4.
- The run count is still PROPOSED_PENDING_APPROVAL. These scenarios are for the approver.

## 4. Is the Gaussian run-level model adequate? (pipeline-faithful check)

The model assumes that, for the real CE2 target (an integer with heavy ties), Spearman's rho of N pairs
has variance (N - 1)^-1 and is close to normal. The check simulated the actual estimator (average ranks,
no jitter) on the real model-free target of 200,000 freshly generated genuine pairs, subsampled to
N = 50,000. It used synthetic scores `lam * standardised(target) + N(0, 1)` and 400 replications per
row:

| lam | mean rho | (N-1) var(rho) | Monte Carlo SE | skewness | P(rho > 0), simulated | P(rho > 0), Gaussian |
|---|---|---|---|---|---|---|
| 0 | -0.0001 | 0.943 | 0.067 | -0.02 | 0.508 | 0.487 |
| 0.002 | 0.0018 | 1.126 | 0.080 | 0.34 | 0.633 | 0.647 |
| 0.005 | 0.0047 | 1.100 | 0.078 | 0.03 | 0.860 | 0.843 |

**Reading.**
- Under the null, the variance agrees with (N - 1)^-1, which is exact for the permutation distribution
  with ties.
- Under small alternatives, the variance is 10-13% above it. That is about 1.5 standard errors, so it is
  not resolved by this check. If real, the required delta grows by about 5-6%.
- The sign probabilities, which are what the sign tests use, agree with the Gaussian approximation
  within Monte Carlo error.
- The skewness of 0.34 at lam = 0.002 is about 2.8 standard errors (SE about 0.12) and is unexplained at
  this precision.

**Limits.**
- N is 5 x 10^4, not 10^6.
- The score model is synthetic. The real published-model and twin scores have a different joint
  structure, which cannot be simulated without the models.
- The twin-primary dependence (corr) is therefore empirically unknown, and the table spans it.

## 5. What is established and what is not

- **Established:**
  - The decision procedure has size at most alpha under the null (tested).
  - Power is monotone in delta and decreases as the twin shares more of the association (tested).
  - The Monte Carlo error of every number above is small (SE <= 0.0022).
- **Not established:**
  - the true effect;
  - the twin's share of the association;
  - the twin-primary error correlation;
  - whether the variance under the real alternative matches the model at N = 10^6.

None of these may be estimated from the exploratory coefficients (Section 6).

## 6. Exploratory coefficients already seen (disclosed, not used)

Development checks computed the real model's coefficient on small samples:
- r1: 4,096 pairs gave primary 0.105, permuted -0.005 and twin 0.020; a separate smoke run gave 0.077;
- r2 smoke (3 x 4,096 pairs, `validation/r2/exploratory_smoke_summary.json`): primary 0.080, 0.070 and
  0.093.

They were not used to choose any size, threshold or scenario, and must not be. They are noisy
small-sample estimates on a population the confirmatory runs resample.
