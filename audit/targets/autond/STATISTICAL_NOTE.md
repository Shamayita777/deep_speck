# Statistical note — AutoND shipped-checkpoint evaluation (plan `autond-ii-v2`)

Revision 3c. Plan hash `b93fa50bb0914b17dc3423480466f18c6340225486c2dcbcb29c911d87207d40`.
**Not frozen. No confirmatory evaluation has been run.**

Every number below is computed by
`audit/common/equivalence.py` and reproduced by
`audit.targets.autond.preregistrations.autond_ii_v2.margin_sensitivity()`.
None is transcribed, and
`tests/test_r3c_statistics.py::TestPlanPowerReporting::test_margin_sensitivity_is_computed_not_transcribed`
recomputes the table against the module on every run.

## 1. The question

For each shipped checkpoint, does its accuracy on ten freshly generated
10⁶-sample test sets agree with the published value, to within a
tolerance fixed before the data existed?

Three registered comparisons:

| id | checkpoint | reference | source |
|---|---|---|---|
| `CMP-SPECK-PAPER` | speck3264 r8 | 0.5144 | paper Table 6 and README (they agree) |
| `CMP-SIMON-PAPER` | simon3264 r11 | 0.5179 | paper Table 6 |
| `CMP-SIMON-README` | simon3264 r11 | 0.5182 | repository README |

Both SIMON references are retained. The plan does not choose between them.

## 2. The margin

**Δ = 0.10 × (p_published − 0.5)**, approved by the operator and unchanged
in this revision.

| reference | Δ |
|---|---|
| 0.5144 | 0.00144 |
| 0.5179 | 0.00179 |
| 0.5182 | 0.00182 |

The tolerance is a fraction of the **distinguishing advantage** over
random guessing, not of the accuracy, because the advantage is the
quantity of scientific interest: 0.5144 carries an advantage of 0.0144,
and a tolerance on 0.5144 itself would be meaningless at this scale.

Four justifications, recorded in `EQUIVALENCE_MARGIN["justification"]`:

1. **Scale.** One rule applies across ciphers and rounds whose advantages
   differ by an order of magnitude, without retuning.
2. **Magnitude.** 10% of the advantage changes no cryptanalytic
   conclusion in the paper — not the number of rounds distinguishable,
   not the ordering of the architectures compared, not whether a
   distinguisher beats random.
3. **Above reporting granularity.** Published accuracies are given to
   four decimals, so rounding alone admits ±5e-5. The smallest margin
   here is 0.00144, about 29× that.
4. **Above the sources' own disagreement.** See §5.

**What power does not do.** Power tells you whether a design can detect
what it is looking for. It does **not** make a tolerance scientifically
acceptable — a margin wide enough to guarantee power could be wide enough
to admit a discrepancy that matters. The justification above rests on the
published advantage, the reporting granularity and the sources'
disagreement. Power only rules out margins so tight the experiment could
conclude nothing. Recorded in
`POWER["power_does_not_justify_a_margin"]`.

### Limitations of the margin

- It is a **tolerance, not a measurement**. A result equivalent within Δ
  may still differ from the published value by up to Δ. Equivalence is
  not identity.
- 10% of the advantage is a **convention**, not a derived constant. A
  reviewer preferring 5% would be making a defensible choice that this
  design cannot support at n = 10 (§4).
- It is fixed as a function of **published values only**, so nothing
  observed can move it. A successor plan would be required.

## 2b. The sample: ten sets from two invocations

`n = 10` is **not** a changed constant, and could not be.

`nEval = 5` is a module-level constant of the target at
`eval_nets.py:22`, not a command-line argument, so one invocation yields
exactly five accuracies. Editing it would modify the pinned upstream
repository, which this audit forbids. The design therefore invokes
`eval_nets.py` **twice per checkpoint** with distinct `--add_str` values
and pools the two NPZs into one sample of ten. The two commands are
identical but for that suffix.

Pooling treats all ten sets as exchangeable. That is argued from the
pinned source, not assumed:

- `:208-209` — each loop iteration draws a **fresh** 10⁶-sample dataset
  via `make_train_data` from `os.urandom`, so sets are independent within
  an invocation;
- `:27-42` — `evaluate_X_Y` is a deterministic function of
  `(model, X, Y)`, so a given dataset yields the same accuracy in either
  process;
- the checkpoint is a fixed file whose sha256 is verified before **every**
  invocation.

The two invocations therefore differ only in their random draws.

**This is a recorded assumption, not a tested hypothesis.** With five
sets per invocation, a between-invocation effect could not be detected
with useful power, so the design does not claim to have excluded one. The
per-invocation accuracies are retained individually so a reader can
inspect the two groups directly; the two per-invocation means are
reported **for inspection only** and are not compared as a test, because
no such test was registered.

**Departure from the original protocol, stated.** AutoND's own Table 6
figure is a five-set mean. Ten sets are therefore *not* a closer
reproduction of their protocol — they are a more precise estimate of our
own quantity. The reference remains a fixed published constant in the
TOST either way, so the extra sets sharpen only our side of the
comparison.

**Missing data.** A failed invocation costs five sets at once. If fewer
than ten sets complete, the comparison is `INDETERMINATE`; a single
surviving invocation is **never** analysed as an n = 5 result, which
would be an unregistered fallback design chosen after seeing which run
failed. The remedy is to rerun the whole checkpoint into a fresh run
directory with the failure recorded.

## 3. The decision rule

With `d = mean − p_ref`, `SE = s/√10`, `df = 9`, per comparison:

| outcome | condition |
|---|---|
| `EQUIVALENT` | Holm-adjusted `p_tost < α` over the equivalence family |
| `DIFFERENT` | Holm-adjusted `p_outside < α` over the outside-margin family |
| `INDETERMINATE` | neither null rejected |

**Equivalence (TOST).** `H₀_lower: d ≤ −Δ` via `t = (d+Δ)/SE` upper tail;
`H₀_upper: d ≥ +Δ` via `t = (d−Δ)/SE` lower tail. `p_tost = max(p_lower,
p_upper)`. Both must reject — intersection-union, so no correction is
needed *between the pair*.

**Outside the margin.** `H₀_above: d ≤ +Δ` via `t = (d−Δ)/SE` upper tail;
`H₀_below: d ≥ −Δ` via `t = (d+Δ)/SE` lower tail. `p_outside =
min(p_above, p_below)`. For Δ > 0 the two rejection regions are
**disjoint** — rejecting both would require `d > Δ` and `d < −Δ` at once —
so the minimum controls type-I error at the nominal level without further
adjustment between the directions. Asserted by
`TestOutsideMarginTest::test_rejection_regions_are_disjoint`.

**The point null `d = 0` is computed and reported, and takes no part in
the decision.**

### Correction recorded: why the point null was removed

Revision 3b classified a comparison `DIFFERENT` when the point-null test
rejected and TOST failed. That is unsound: a discrepancy can be
significantly non-zero while lying **inside** the equivalence interval.

The counterexample, now a named regression test
(`TestDifferentRequiresOutsideMargin::test_the_exact_r3b_counterexample_is_now_indeterminate`):

| quantity | value |
|---|---|
| observed mean | 0.515600 |
| reference | 0.5144 |
| `d` | +0.00120 |
| Δ | 0.00144 → **inside the margin** |
| `p_tost` | 0.2343 → equivalence not established |
| `p_diff` (point null) | 0.0161 → significant |
| `p_outside` | 0.7657 → outside-margin **not** established |
| r3b outcome | `DIFFERENT` ← wrong |
| r3c outcome | `INDETERMINATE` ← correct |

Failing to establish equivalence is not evidence of a meaningful
discrepancy. `TestDifferentRequiresOutsideMargin::test_no_inside_margin_mean_is_ever_different`
sweeps 7 offsets × 6 precisions across the whole inside-margin range and
asserts `DIFFERENT` never appears.

**Mutual exclusivity.** `EQUIVALENT` needs the interval for `d` inside
(−Δ, +Δ); `DIFFERENT` needs it wholly outside. No dataset satisfies both.
`equivalence.classify` **raises** rather than silently preferring one, so
a future change that broke the property would fail loudly.

### Multiplicity

Holm-Bonferroni over the family of 3, at α = 0.05. Sort ascending; compare
`p_(i)` against `α/(m−i+1)` = α/3, α/2, α; stop at the first failure and
reject nothing from that rank onward.

Applied **independently** to the equivalence family and the
outside-margin family, because they test different null hypotheses. Each
family carries its own family-wise guarantee at α. A p-value is never
carried between them. One implementation: `equivalence.holm`.

## 4. Power

Computed by `equivalence.tost_power_exact`: `P(reject | s)` integrated
over the χ distribution of `s` by Simpson's rule, so **the variance is
estimated from the sample as the analysis will estimate it**, not assumed
known. Cross-checked by `tost_power_mc`, a Monte Carlo simulation of the
actual rule (200 000 replications, seed 20261015, 99% interval reported).

### Correction recorded: the superseded calculation

Revision 3b computed `2Φ(Δ/SE − t_crit) − 1` with SE held fixed. Two
faults: it treated the SD as known, when at n = 5 it is estimated on four
degrees of freedom; and the expression goes **negative** whenever
Δ/SE < t_crit, which was clamped to zero and then reported as a computed
power of zero. Fractions 0.02 and 0.05 were both reported as `0.000`.
Their real values at α/3 are **0.0023** and **0.1291**.

`TestPowerIsExactNotApproximated::test_no_fraction_is_reported_as_exactly_zero_power`
pins this.

### Margin sensitivity (computed)

Planning SD 6.7e-4, n = 10 (the registered design), true difference 0.

| fraction | Δ (SPECK) | power @ α=0.05 | power @ α/3 |
|---|---|---|---|
| 0.02 | 0.00029 | 0.0244 | 0.0030 |
| 0.05 | 0.00072 | 0.8620 | 0.6050 |
| **0.10 (registered)** | **0.00144** | **1.0000** | **0.9998** |
| 0.15 | 0.00216 | 1.0000 | 1.0000 |
| 0.20 | 0.00288 | 1.0000 | 1.0000 |

At the superseded n = 5 the registered margin gave 0.9732 / **0.7965**.
Those figures are retained in `POWER["superseded_design"]`.

### Power at the registered margin

| comparison | Δ | @ α | @ α/3 | Monte Carlo @ α/3 (99% CI) |
|---|---|---|---|---|
| SPECK 0.5144 | 0.00144 | 1.0000 | 0.9998 | 0.9998 [0.9997, 0.9998] |
| SIMON 0.5179 | 0.00179 | 1.0000 | 1.0000 | 1.0000 [1.0000, 1.0000] |
| SIMON 0.5182 | 0.00182 | 1.0000 | 1.0000 | 1.0000 [1.0000, 1.0000] |

Exact and Monte Carlo agree to simulation error throughout.

**Reading these honestly.** α/3 is the most conservative Holm step and is
a **planning approximation**, not the registered procedure: the actual
per-comparison threshold depends on the observed ordering and is
unknowable in advance. True power therefore lies **between the α/3 and α
columns**. The revision-3b figure of 0.895 came from the broken formula
and should be disregarded.

**Why n rose rather than Δ.** At n = 5, SPECK's power at α/3 was 0.7965 —
marginally below the conventional 0.80 — and it was *fragile*: because the
planning SD is itself a 4-df estimate (95% interval ≈ 0.6×–2.9×), a true
SD of 1.5× dropped it to 0.339 and 2.0× to 0.129. The margin was **not**
widened, because Δ is justified by the question (§2) and widening it would
relax the criterion for agreement. Ten independent test sets per
checkpoint were registered instead:

| true SD | n = 5 | n = 10 |
|---|---|---|
| as planned | 0.797 | 0.9998 |
| 1.5× | 0.339 | 0.996 |
| 2.0× | 0.129 | 0.905 |

At 2.9× — the pessimistic end of the SD interval — no feasible n rescues
the design, and the honest outcome is `INDETERMINATE`, which the rule
returns rather than defaulting to agreement.

### Power assumptions, stated

1. The planning SD 6.7e-4 is **assumed**, not known. It is the sample SD
   of this project's own five-set evaluation of the **staged-training
   round-8 (unpolished)** checkpoint — a *different artifact* from the
   shipped polished checkpoints under test. It is itself estimated on
   4 df; a 95% interval for an SD on 4 df spans roughly 0.6× to 2.9× the
   point estimate, so these powers are indicative of the design, not
   guarantees. Recorded verbatim in `PLANNING_SD_PROVENANCE`.
2. Design use only. It never enters the analysis, shifts the margin or
   conditions an outcome. The observed replicate SD of the confirmatory
   run is computed from that run's own ten accuracies and reported
   separately.
3. Accuracies are treated as normal around the true mean — reasonable for
   a mean of 10⁶ Bernoulli trials.
4. Power is reported at a true difference of zero. It falls as the true
   difference approaches Δ, asserted by
   `test_power_falls_as_the_true_difference_approaches_the_margin`.

## 5. The SIMON limitation, stated narrowly

The paper and README differ by 0.00030; the SIMON margin is 0.00179.
Because the margin exceeds the gap, the two registered equivalence
regions **overlap over most of their extent**, so near the published
values a result equivalent to one is very likely equivalent to the other.

**This experiment is therefore not expected to resolve which published
SIMON value is correct**, and `H-AUTOND-SIMON-REPRO` predicts
`INCONCLUSIVE` on that basis.

It is **not** claimed that the design is incapable of separating them
under every conceivable outcome. A mean falling in the non-overlapping
sliver, or far enough out to be `DIFFERENT` from one reference while
`INDETERMINATE` against the other, would discriminate. Those outcomes are
possible but not expected. Asserted by
`TestWrittenRuleMatchesTheCode::test_simon_limitation_is_narrow_not_absolute`.

## 6. Outcome decision table

### Per comparison

| `p_tost` (Holm) | `p_outside` (Holm) | outcome |
|---|---|---|
| < α | ≥ α | `EQUIVALENT` |
| ≥ α | < α | `DIFFERENT` |
| ≥ α | ≥ α | `INDETERMINATE` |
| < α | < α | impossible — `classify` raises |

Plus, before any test: fewer than 5 completed sets, non-numeric or
non-finite accuracies, or zero variance → `INDETERMINATE` with a recorded
reason. A partial mean is never substituted.

### Per hypothesis

| hypothesis | comparisons | `SUPPORTED` | `NOT_SUPPORTED` | `INCONCLUSIVE` | predicted |
|---|---|---|---|---|---|
| `H-AUTOND-SPECK-REPRO` | 1 | `EQUIVALENT` | `DIFFERENT` | otherwise | `SUPPORTED` |
| `H-AUTOND-SIMON-REPRO` | 2 | exactly one `EQUIVALENT`, none `INDETERMINATE` | both `DIFFERENT` | otherwise | `INCONCLUSIVE` |

Predictions are never revised after seeing results. A wrong prediction is
retained and reported as wrong (`PLAN["immutability"]`).

## 7. What this experiment cannot establish

- It is inference on shipped weights: it says nothing about the training
  pipeline that produced them, nor about the search that chose their
  input difference.
- Five fresh sets estimate *this* checkpoint's accuracy. They say nothing
  about variation across independently trained models, of which there is
  exactly one here.
- Loading and evaluating a checkpoint is not evidence of its provenance.
- Equivalence within Δ is not identity.
- The paper/README SIMON discrepancy is not expected to be resolved (§5).
