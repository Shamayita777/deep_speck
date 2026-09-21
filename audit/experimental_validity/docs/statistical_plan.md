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

## Practical significance (epsilon)

**Candidate: epsilon = 0.01 absolute confirmatory accuracy.**

Research basis: Picard (2021), *"torch.manual_seed(3407) is all you
need"*, reports ~0.1% SD and ~0.5% max-min spread in ResNet-50/ImageNet
test accuracy from random seed alone across 50 seeds, and describes
that gap as "widely considered significant" in that literature. This
does **not** establish 0.01 as correct for Speck32/64 specifically (no
published seed-variance study exists for this task), but it shows 0.01
is larger than typical pure-seed noise in a comparable, more mature
setting - the right qualitative property for a margin.

**Gating requirement (not yet satisfied):** before this value is used
in a confirmatory production run, it must be checked against this
project's own EV-BASELINE/EV-NOISE pilot replicate-to-replicate spread.
If pilot s_D turns out comparable to or larger than 0.01, the margin is
not resolvable at any feasible replicate count and must be revisited -
not silently kept. Both `configs/gohr_ev_shuffle.yaml` and
`configs/gohr_ev_representation.yaml` currently carry
`practical_threshold: UNSPECIFIED_REQUIRES_PILOT_SANITY_CHECK`, which
`scripts/run_ev.py`'s fail-closed placeholder check refuses to run
against, by construction - not merely by convention. This value is
**not** presented as a universal ML or cryptanalysis standard; if the
pilot fails to support it, the margin must be revised before freeze,
not after.

## Alpha
0.05, both for the primary difference-detection test and for TOST.

## Multiplicity
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
See `framework/certificate.py`'s module docstring for the fixed
vocabulary (SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / NOT_RUN /
INVALID) and `decide_final()` for how the Holm-corrected
difference-detection decision and the TOST equivalence assessment are
combined. A non-significant difference-detection result is INCONCLUSIVE
unless TOST separately and formally establishes equivalence - it is
never NOT_SUPPORTED on its own.
