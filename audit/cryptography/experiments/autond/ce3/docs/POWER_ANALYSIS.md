# AutoND CE3: power analysis and the choice of n and B (r2, after the S1 correction)

**Evidence.**
- `validation/power/power.json`, SHA-256 `ecc61ce0…`, pinned in the design.
- `power_r1_superseded.json` (`0789c5cf…`) is kept. It used the superseded median S1 rule, seeded non-group permutations, a calibration set of n/2 per class instead of the design's fixed 50,000, a synthetic model accuracy of about 0.516 instead of 0.5144, and only 12–16 replications per n.
- `power_draft1_superseded.json` (`b14dd0c6…`) is kept. It fixed the S1 and S2 means for every n, ignoring estimation loss.

**What was not used.** No real AutoND representation and no CE2 or CE3 result entered this analysis. The only empirical inputs are the model-free P1 class fractions and the published accuracy 0.5144, which sets the strength of the synthetic calibration task.

## 1. Two levels, kept apart

| Level | Count | What it is |
|---|---|---|
| independent synthetic pipeline replications | 48 per n, 144 in total (1,844 s on 7 workers) | each draws its own geometry and data and runs the unchanged CE3 folds, permutation group, tie-break, probe and exact BA code |
| null pipeline replications | 400 at n = 5,000, B = 5 | the same code with labels independent of the representation |
| inner draws per power evaluation | 100,000 (50,000 inside the bootstrap) | draws from the fitted per-replicate model, to get the per-replicate outcome cells |
| bootstrap resamples | 400 per n | resampling of the 48 replications; every mean, variance and covariance is re-estimated each time |
| decision power | exact | the 20-replicate counts of positive signs are multinomial in the per-replicate cells; P(C) and P(S1 and S2) are summed exactly over all count vectors, with no outer Monte Carlo |
| outer decision simulation (check only) | 20,000 | the complete `statistics.analyse` procedure on simulated experiments, to check the exact calculator |

C uses calibration data that are independent of the main data, so P(SUPPORTED) = P(all valid) × P(C) × P(S1 and S2).

## 2. Per-replicate model

From the replications at each n I estimate:
- the means, variances and covariances of BA(A), BA(T) and BA(P_b), with the P arms treated as exchangeable;
- the mean and covariance of the calibration triple (probe BA, twin BA, model BA).

S1 is positive in a simulated replicate when at least (B + 1)/2 of the B simulated permuted BAs lie below BA(A). That is the upper-half rank event; continuous draws have no ties. For B = 5 and B = 11 the actual group and tie-break were also run inside every replication, which gives the model-free check in section 4.

**Measured inputs** (means; the model BA was set to 0.5144):

| n | BA(A) − BA(P) | BA(A) − BA(T) | SD of BA(A) | probe BA (cal) | model BA (cal) | recovery fraction |
|---|---|---|---|---|---|---|
| 50,000 | 0.00147 | 0.00181 | 0.00231 | 0.51175 | 0.51492 | 0.79 |
| 100,000 | 0.00245 | 0.00233 | 0.00140 | 0.51100 | 0.51441 | 0.76 |
| 200,000 | 0.00304 | 0.00326 | 0.00113 | 0.51123 | 0.51441 | 0.78 |

## 3. Power after the S1 correction (reference scenario: oracle effect 0.005, κ = 0.5, τ = 0, no ties, no failures)

| n | B | plug-in P(SUPPORTED) | bootstrap median | bootstrap 5% bound | P(C) | P(S1 and S2) |
|---|---|---|---|---|---|---|
| 50,000 | 5 | 0.295 | 0.284 | 0.115 | 1.000 | 0.295 |
| 50,000 | 21 | 0.315 | 0.326 | 0.138 | 1.000 | 0.315 |
| 100,000 | 5 | 0.980 | 0.983 | **0.879** | 1.000 | 0.980 |
| 100,000 | 21 | 0.980 | 0.983 | **0.877** | 1.000 | 0.980 |
| 200,000 | 5 | 1.0000 | 1.0000 | 0.9996 | 1.000 | 1.0000 |
| 200,000 | 11 | 1.0000 | 1.0000 | 0.9996 | 1.000 | 1.0000 |
| 200,000 | 21 | 1.0000 | 1.0000 | 0.9996 | 1.000 | 1.0000 |

**Inner Monte Carlo variation.** A repeat with another inner seed moves the plug-in values by at most 0.01 (for example 0.9798 against 0.9804 at 100k, B = 5).

**Direct check.** Simulating the complete decision procedure 20,000 times at 200k, B = 5, gives P(SUPPORTED) = 1.0 (95% interval [0.99982, 1]), in agreement with the exact calculation.

## 4. Model-free check, and where the parametric model is optimistic

For B = 5 and 11, each replication's actual outcome cells (S1 upper half × S2 > 0, and C_task × C_twin) were counted, given a Jeffreys Dirichlet posterior, and turned into exact power.

| n | B | empirical S1 upper-half rate | parametric q(S1) | empirical plug-in | posterior median | posterior 5% |
|---|---|---|---|---|---|---|
| 100,000 | 5 | 0.896 | 0.937 | 0.891 | 0.852 | 0.491 |
| 200,000 | 5 | 0.979 | 0.993 | 0.9999 | 0.999 | 0.964 |
| 200,000 | 11 | 0.979 | 0.995 | 0.9999 | 0.999 | 0.966 |

At 100k the parametric model is optimistic: it overstates the S1 upper-half rate (0.937 against an observed 0.896). The 100k candidate fails the rule on both methods. At 200k both methods clear 0.90.

**Ties.** No replication had a tie between the observed and a permuted exact BA, nor among the permuted BAs.

## 5. Exactness of S1 in the real pipeline (null run)

400 null replications gave 193 upper-half ranks: 0.483, 95% interval [0.433, 0.533]. The rank counts were 66, 77, 64, 58, 72, 63, consistent with uniform. No ties occurred.

## 6. Selection (rules fixed in the design)

- **n:** the smallest n whose bootstrap 5% bound at B = 21 is at least 0.90. 100k gives 0.877, so **n = 200,000**.
- **B:** the smallest B whose plug-in is within 0.01 of B = 21 and whose own bound meets the rule, so **B = 5**.

These are the same values as r1. They were not carried over: they were re-derived under the corrected S1 test, with more replications and propagated parameter uncertainty.

## 7. Sensitivity (B = 5; exact decision power at the plug-in parameters)

| varied | value | n = 100k | n = 200k |
|---|---|---|---|
| κ | 0 / 0.25 / 0.5 | 0.980 | 1.000 |
| κ | 0.75 | 0.064 | 0.163 |
| κ | 0.9 | 0.000 | 0.000 |
| effect scale (probe-level S1 and S2 means) | 0.5 | 0.441 | 0.917 |
| effect scale | 0.75 | 0.845 | 0.999 |
| effect scale | 1.25 | 0.999 | 1.000 |
| between-replicate SD τ | 0.001 | 0.939 | 1.000 |
| τ | 0.002 | 0.756 | 0.981 |
| τ | 0.003 | 0.499 | 0.822 |
| degenerate (all-tied) replicate rate | 0.05 | 0.917 | 0.998 |
| tie rate | 0.10 | 0.796 | 0.976 |
| tie rate | 0.20 | 0.451 | 0.749 |
| per-replicate failure rate | 0.005 | 0.886 | 0.905 |
| failure rate | 0.01 | 0.801 | 0.818 |
| C_task–C_twin correlation | 0 or 0.9 | 0.980 | 1.000 |
| combined pessimistic (scale 0.75, τ 0.002, ties 0.05, failures 0.005) | | 0.315 | 0.655 |

**Endpoint dependence.** Each row also stores the Fréchet lower bound, max(0, sum of the four marginal rejection probabilities − 3), which holds for any dependence between the four endpoints. At 200k in the reference scenario it is 1.000.

**Reading the table.**
- **κ is the sharpest sensitivity.** The synthetic probe recovers about 0.78 of the model's advantage, so κ = 0.75 would usually fail the gate even with a working pipeline. κ = 0.5 leaves a margin.
- **Failures cap power.** All 20 replicates are required, so a per-replicate failure probability f caps P(SUPPORTED) at (1 − f)^20.
- **Ties.** Degenerate tied replicates make S1 a coin flip and S2 non-positive, so they cost power but never level.

## 8. Cost at 200,000

The blinded benchmark r2 (`benchmark.json` `c7d06346…`) fitted all 50 probes of one replicate in 22 s, with at most 21 iterations, maximum gradient 9.9e-5 and no convergence warnings. 20 replicates take roughly 20–40 minutes, with about 0.2 GB of retained evidence per replicate.

## 9. Limits

- **Synthetic geometry.** It is an assumption: Gaussian-ReLU features, which are not the real 14-unit representation (see the representation diagnostics).
- **Normal model.** The per-replicate model is normal. The model-free check shows it is optimistic at 100k, so the selection rests on the larger n, where both agree.
- **Oracle-effect sensitivity.** It is approximated by scaling the probe-level effect. No separate synthetic runs were made at other oracle effects.
- **SESOI and κ** are prospective judgments.
