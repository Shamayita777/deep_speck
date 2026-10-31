# AutoND CE2 — power and computational analysis

Status: **proposal for review**. The run count and pairs per run in the design are marked
`PROPOSED_PENDING_APPROVAL` and production refuses to start until they are frozen and approved.

Raw outputs: `validation/power_analysis.json` (from `--power`) and `validation/benchmark_cpu.json`.

## Model of a run

All runs evaluate one fixed checkpoint, so the population coefficient ρ is the same in every run, and
run-to-run variation is sampling variation only. Under independence, the permutation variance of a
correlation between two fixed vectors is exactly 1/(N − 1), with or without ties, so the within-run
standard error is se = 1/√(N − 1): 0.0010 at N = 10⁶.

Effects are expressed as δ = ρ / se. A run's coefficient is modelled as N(δ, 1) in se units, so
Pr[ρᵢ > 0] = Φ(δ).

The controls are modelled as follows:
- **Permuted target:** distributed N(0, 1) and independent of the primary.
- **Untrained twin:** distributed N(r·δ, 1), with r ∈ {0, 0.25, 0.5}. Its sampling-error correlation
  with the primary is c ∈ {0, 0.5}, because the twin is evaluated on the same pairs.

The joint power of the complete hierarchical rule was simulated with 50,000 replications per
scenario:
- the primary exact two-sided sign test;
- then both exact one-sided paired sign tests, with Holm correction.

## Primary test alone (exact)

The table gives power at α = 0.05 for the run-level sign probabilities requested.

| runs | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | reject if positives ≥ |
|---|---|---|---|---|---|---|
| 10 | 0.086 | 0.149 | 0.244 | 0.376 | 0.544 | 9 |
| 15 | 0.173 | 0.297 | 0.461 | 0.648 | 0.823 | 12 |
| 20 | 0.246 | 0.416 | 0.617 | 0.804 | 0.933 | 15 |
| 25 | 0.306 | 0.512 | 0.727 | 0.891 | 0.975 | 18 |
| 30 | 0.358 | 0.589 | 0.803 | 0.939 | 0.990 | 21 |
| 40 | 0.441 | 0.703 | 0.897 | 0.981 | 0.999 | 27 |

The sign probabilities 0.65–0.85 correspond to δ = 0.39–1.04: |ρ| ≈ 0.0004–0.0010 at N = 10⁶.

## Complete decision rule (simulated): P(SUPPORTED)

At 20 runs, with c = 0:

| δ (se units) | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| twin r = 0 | 0.004 | 0.075 | 0.457 | 0.895 | 0.995 | 1.000 | 1.000 |
| twin r = 0.25 | 0.004 | 0.071 | 0.413 | 0.845 | 0.985 | 0.999 | 1.000 |
| twin r = 0.5 | 0.004 | 0.050 | 0.270 | 0.606 | 0.833 | 0.942 | 0.996 |

At the run level, the probability that a control is beaten is q = Pr[|ρ_primary| > |ρ_control|]. It
rises from 0.54 at δ = 0.5 to 0.86 at δ = 2 and 0.995 at δ = 4, which is the paired
control-superiority range considered. Control separation, not the primary test, limits the joint
power.

The table gives δ at 90% joint power, with the corresponding |ρ| at N = 10⁶ in brackets. The "—" cell
means 90% power is not reached within δ ≤ 4.

| runs | r = 0, c = 0 | r = 0.25, c = 0 | r = 0.5, c = 0 | r = 0.5, c = 0.5 |
|---|---|---|---|---|
| 10 | 2.94 (0.0029) | 3.33 (0.0033) | — | 3.52 (0.0035) |
| 20 | 2.02 (0.0020) | 2.20 (0.0022) | 2.81 (0.0028) | 2.28 (0.0023) |
| 30 | 1.73 (0.0017) | 1.79 (0.0018) | 2.01 (0.0020) | 1.83 (0.0018) |

The detectable |ρ| scales as 1/√N, so doubling the pairs per run divides every entry by √2.

## What the analysis can and cannot justify

- **No prior effect size.** No published or prior AutoND analysis gives a defensible expected ρ for
  this target, and none is assumed. The analysis is a sensitivity analysis, and the design is
  characterised by its minimum detectable effect.
- **Run validity.** A run is excluded only for an undefined coefficient. That requires a constant
  score, a constant target or non-finite values over 10⁶ pairs. In the smoke corpora (4,096 pairs), the
  target took 43 distinct integer values and over 99% of the scores were distinct, so exclusions are
  not expected. Failures (gate violations or exceptions) indicate a defect and are retained as such.
- **Twin association is not a power problem.** If the untrained twin's association is comparable to
  the primary's, more runs or pairs cannot produce SUPPORTED. The correct outcome is then
  `ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED`.
- **Significance is not magnitude.** At these sizes, very small associations are detectable. The
  certificate therefore reports the run-level coefficients, the median and its order-statistic
  interval, and the decision must be read together with them.

## Computation (non-evidentiary benchmark)

The benchmark ran one 10⁶-pair run on an 8-core arm64 CPU, without a GPU, under TF 2.15.1. No
correlation was computed.

| Step | Time / size |
|---|---|
| Generation, trajectory and gates E/F | 29 s |
| Published-model inference | 17.5 s |
| Twin inference | 17.4 s |
| Compressed run data | 53.7 MB |
| Compressed outputs | 8.0 MB |

The runner re-verifies committed data (about another 30 s), so a run takes about 2 minutes on this
CPU.

At 20 runs, the totals are about 40 minutes and about 1.25 GB. The behavioural set (10⁶ pairs) and
the preflight (mathematical gates about 5 min, dominated by exhaustive literal enumeration at word
size 8) are additional. These costs do not constrain the design.

## Recommendation

**20 runs × 10⁶ genuine pairs, all 20 required valid, no reserve runs.**

1. The design gives at least 90% joint power for |ρ| ≥ 0.0020 when the twin carries no association,
   and for |ρ| ≥ 0.0028 when the twin carries half the primary's association. Both are a small
   fraction of a percent of rank variance.
2. With 20 runs, the smallest attainable two-sided p is 1.9 × 10⁻⁶, and the 95% order-statistic
   interval uses ranks 6 and 15 (exact coverage 0.959).
3. The cost is roughly 40 CPU minutes and 1.25 GB.

**Alternatives for the reviewer:**
- 30 runs lowers the detectable |ρ| to 0.0017–0.0020 at 1.5 times the cost.
- 2 × 10⁶ pairs per run lowers it by a factor of √2 at twice the cost.
- Requiring 18 of 20 valid runs would add tolerance for execution failures. It would also weaken the
  rule that a failure indicates a defect, so it is not recommended.

## Disclosure of exploratory observations

During pipeline development, two non-evidentiary checks printed coefficients for the real model:
- **Adapter check, 4,096 pairs:** primary 0.105, permuted −0.005, twin 0.020.
- **CPU smoke run `run_00`, 4,096 pairs:** primary 0.077, permuted 0.031, twin −0.010.

At N = 4,096, se ≈ 0.016. These values were not used to choose the run count, the pairs per run or
any threshold; the recommendation above follows from the effect-agnostic analysis alone. They are
`NOT_CONFIRMATORY` and play no part in the confirmatory analysis.
