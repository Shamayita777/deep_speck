# EV Statistical Plan (Round 6 revision)

## Statistical unit
The independent training run / model replicate. Never an individual
ciphertext example, batch, or prediction.

## Primary estimand
mu_D = E[D], D = accuracy_condition_B - accuracy_condition_A, for
H-EV-SHUFFLE and H-EV-REPRESENTATION.

## Primary test
**Paired t-test** (`scipy.stats.ttest_1samp` on paired differences),
H0: mu_D = 0, H1: mu_D != 0.

Historical note (Round 5): the primary p-value previously came from
`scipy.stats.wilcoxon`, whose actual null hypothesis - per SciPy's own
documentation - is that the distribution of differences is *symmetric
about zero*, not that the mean equals zero. Since the frozen estimand
is explicitly the mean, this was an estimand/test mismatch, corrected
in Round 5 and unchanged since. The paired t-test targets mu_D
directly, requires no symmetry assumption, and its underlying
statistic is exactly Cohen's d_z * sqrt(n) - a fully coherent chain:
estimand -> effect size -> test -> CI, all in the same units, all
targeting the same quantity. **This design is not revisited in Round
6 and Wilcoxon is not reintroduced.**

## Primary confidence interval
t-distribution CI for mu_D: `d_bar +/- t_(n-1, alpha/2) * SE`. This
matches the primary test exactly. The percentile bootstrap CI is
retained as a **secondary sensitivity check** (robust to non-normality)
and is reported in every certificate, but does not drive any decision.

## Effect size
Cohen's d_z = mean_diff / sd_diff.

## Equivalence method: TOST
Schuirmann (1987); Lakens (2017, *Social Psychological and Personality
Science*, 8(4):355-362). Two one-sided t-tests against -epsilon and
+epsilon:

    H0_lower: mu_D <= -epsilon   vs.  H1_lower: mu_D > -epsilon
    H0_upper: mu_D >= +epsilon   vs.  H1_upper: mu_D < +epsilon

Equivalence is concluded iff `max(p_lower, p_upper) < alpha`. No
additional correction is needed within TOST itself (intersection-union
principle). Computed directly (`framework.statistics.tost_equivalence`)
rather than via CI-containment (an earlier CI-containment approximation
using the wrong confidence level was corrected in Round 5).

## TOST computation policy (clarified, Round 6)
TOST is computed as a **prespecified secondary equivalence analysis
whenever sufficient valid replicate pairs exist**, regardless of
whether the primary difference-detection test is significant
(`gohr.experiments._finalize_paired_certificate` calls
`assess_practical_equivalence` unconditionally once `sufficient` is
True - this was already the actual implementation; an earlier draft of
this document incorrectly described TOST as being run "only when the
primary test is non-significant", which did not match the code. That
mismatch is corrected here, in the documentation, per the Round-6
audit's explicit instruction not to alter decision semantics merely to
make the wording convenient). TOST's result is, however, only
**decision-relevant** when the primary (Holm-corrected) test does not
detect a difference: `framework.certificate.decide_final` returns
SUPPORTED whenever the primary test is significant regardless of what
TOST says, and only consults the TOST outcome
(EQUIVALENT_WITHIN_THRESHOLD -> NOT_SUPPORTED) when the primary
decision is INCONCLUSIVE. This is "Option B" from the Round-6 audit:
always compute, decision-gate on the primary test's outcome.

## Practical significance (epsilon) - FROZEN

**epsilon = 0.01 absolute accuracy on the confirmatory test set.**

Justification: 0.01 is the audit-wide smallest effect treated as
practically meaningful for accuracy on the confirmatory test set. The
SAME margin is frozen for II-4, so the audit applies a single
predeclared practical-significance scale. This is a **methodological
judgment, not a literature-derived universal threshold**.

SUPERSEDED (statistical plan v1): earlier text required epsilon to be
"sanity-checked against pilot replicate-to-replicate spread". That is
withdrawn. Validating a practical-significance margin against observed
noise confuses measurement precision with scientific consequence, and
would let pilot data influence a confirmatory decision rule. **Pilot
data estimate nuisance variance ONLY**; epsilon is never derived from,
selected by, or validated against them.

## Alpha
0.05, both for the primary difference-detection test and for TOST.

## Multiplicity - TWO SEPARATE HOLM FAMILIES

1. **Difference-detection family** - Holm across the two primary
   difference-detection p-values, alpha=0.05.
2. **Equivalence family** - Holm across the two OVERALL TOST p-values
   (`p_tost = max(p_lower, p_upper)`), alpha=0.05. NOT_SUPPORTED is a
   formal conclusion asserted for two hypotheses, so it carries its own
   multiplicity risk. **No additional correction is applied inside each
   TOST**: its intersection-union structure already controls the
   individual equivalence claim, and correcting twice would
   double-penalise.

Prospective planning simulates **both Holm families EXACTLY** inside a
joint simulation of the complete decision rule (paired t-test ->
difference Holm -> TOST -> equivalence Holm -> `decide_final()`). The
earlier alpha/2 Bonferroni-style approximation has been removed: the two
hypotheses are simulated together because Holm couples them, and
NOT_SUPPORTED power is computed for the JOINT requirement (difference
non-significant after its family AND TOST equivalent after its family),
not for TOST alone. Powering TOST alone overstated the achievable
NOT_SUPPORTED rate.

## Multiplicity (detail)
Holm step-down across the frozen primary family
{H-EV-SHUFFLE, H-EV-REPRESENTATION} at family alpha=0.05, applied to
the **primary difference-detection p-values only**
(`framework/multiplicity.py`, invoked via
`gohr.experiments.apply_primary_family_correction`, which must be run
once both certificates exist - see `scripts/apply_family_correction.py`).
TOST is a logically separate analysis, computed regardless of the
primary test's significance (see "TOST computation policy" above) but
**decision-relevant only** when the primary test does not detect a
difference. It has its own built-in Type-I-error control (the
intersection-union principle) and is not folded into the Holm family.
This scoping choice is this project's own methodological call, not
something mandated by Lakens/Schuirmann, and is flagged as such. The
primary family remains exactly these two hypotheses; no additional
hypothesis (e.g. an architecture experiment) is folded in.

## Target power
80% (Cohen, 1988 convention), retained as the default. No authoritative
source reviewed mandates 90% specifically for this experiment type;
90% remains an available upgrade if compute budget allows, not a
requirement.

## Power analysis

**Simulation assumptions (explicit).** The prospective power simulation
assumes INDEPENDENT paired differences that are NORMALLY distributed with
mean equal to the target effect (0 for the equivalence/null scenario) and
standard deviation equal to `sigma_Delta_upper_95`. Both primary
hypotheses are drawn in the same trial so the Holm families are applied
to jointly-realised p-values. Departures from
normality or independence change the required n; the simulation cannot
detect them.

Prospective sizing uses each hypothesis's one-sided **95% UPPER
confidence bound** for sigma_Delta (chi-square), not the point estimate:
a ~10-pair pilot leaves sigma uncertain enough that point-estimate
sizing under-powers the experiment roughly half the time.

Planning applies **alpha/2 per hypothesis** for difference detection as
a CONSERVATIVE BONFERRONI-STYLE APPROXIMATION to Holm. This is **not
exact Holm power**: Holm's step-down tests the larger p-value at alpha,
so alpha/2 for both yields an upper bound on required n. TOST is planned
at full alpha at true effect 0 and is excluded from the Holm family
(intersection-union control).

The common production replicate count K is the **maximum** over four
requirements: difference detection and TOST equivalence, for each of the
two primary hypotheses.

SUPERSEDED: earlier text implied sigma_Delta could be obtained from
EV-BASELINE/EV-NOISE. It cannot. Both primary hypotheses are PAIRED with
a shared dataset and shared model seed; EV-BASELINE measures variability
across exactly the components pairing removes (overestimating n), and
EV-NOISE holds the manipulated factor fixed (underestimating it).
sigma_Delta must come from a PAIRED calibration run of the same
hypothesis.

Simulates the **actual final procedures**:
`framework.power.simulate_power(procedure="difference_ttest", ...)`
for the primary test, and `procedure="equivalence_tost"` for power to
correctly conclude equivalence when truly equivalent (typically
simulated at `target_effect=0.0`,
`target_effect_source="pilot_variance_only"`). Anti-misuse guards:
`target_effect_source="observed_production"` is refused (`ValueError`);
no defensible target effect -> `power_unavailable()`. Unchanged since
Round 5; not weakened in Round 6.

## Replicate count vs. prospective power (Round 6, Issue 3)
`framework.power.validate_replicate_plan_against_power` guards against
an evidentiary paired experiment being configured with
`minimum_valid_pairs` below its own declared `power_analysis.required_n`
(see the `power_analysis` block added to `configs/gohr_ev_shuffle.yaml`
and `configs/gohr_ev_representation.yaml`). If `minimum_valid_pairs <
required_n` and no explicit, non-empty `underpowered_justification` is
recorded, `scripts/run_ev.py` refuses to run
(`UnderpoweredReplicatePlanError`). `requested_pairs` may still exceed
`minimum_valid_pairs` to predeclare tolerance for replicate attrition -
that relationship is intentionally unrestricted. `required_n: null`
(the current state of both configs) does not itself trigger this
guard; production remains blocked by the pre-existing placeholder
checks on `requested_pairs`/`minimum_valid_pairs` until real values -
including a real, power-simulation-derived `required_n` - are supplied.

## Confirmatory evaluation / checkpoint protocol (Round 6, Issue 5)
**Frozen decision: final-epoch weights, not the reloaded best-val-loss
checkpoint ("Option B").** `gohr/adapter.py` evaluates the same
in-memory model object that `model.fit()` left in place after its last
epoch; `ModelCheckpoint`'s saved artifact is never reloaded before
evaluation. This was already the actual behavior prior to Round 6 - it
is now an explicit, named constant
(`gohr.baseline.CONFIRMATORY_EVALUATION_PROTOCOL = "final_epoch_weights"`),
recorded in every replicate's manifest
(`confirmatory_evaluation_protocol` field). The saved checkpoint file's
hash is still recorded for provenance/failure-detection purposes only
(a missing checkpoint file is treated as a failed replicate) - it is
not the model-selection mechanism for the confirmatory metric.

## Replicate-count determination - NOT YET RUN
Blocked on: (1) a non-confirmatory pilot run to estimate s_D (requires
real compute, not executed in this sandbox), (2) confirming epsilon
against that pilot per the gating requirement above, (3) a stated
compute budget to bound the feasible search range in
`required_replicates_for_power`, (4) a production-scale runtime
benchmark (Issue 10) to confirm the resulting plan is executable at
all within the available compute/time budget.

## Decision semantics

    SUPPORTED       Holm-corrected difference test significant AND the
                    95% CI for the signed mean difference lies entirely
                    outside [-epsilon, +epsilon].
    NOT_SUPPORTED   difference test not significant AND TOST establishes
                    equivalence within +/-epsilon.
    INCONCLUSIVE    everything else.

Vocabulary: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / NOT_RUN / INVALID
(`framework/certificate.py`), combined by `decide_final()`. A
non-significant difference-detection result is INCONCLUSIVE unless TOST
separately and formally establishes equivalence - never NOT_SUPPORTED on
its own.

CHANGE FROM v1 (v1 wording removed entirely, not retained alongside):
v1 returned SUPPORTED on any significant difference. With
enough replicates an accuracy difference of 1e-4 is detectable, and
reporting that as support would conflate statistical with practical
significance. **A tiny statistically significant effect that lies inside
epsilon is NOT material** and is now INCONCLUSIVE. Where epsilon or the
CI is unavailable, SUPPORTED is unreachable.

