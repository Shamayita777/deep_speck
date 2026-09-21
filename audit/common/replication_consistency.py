"""
Generic replication-consistency assessment.

PURPOSE
-------
When a NEW independent replication of an experiment is produced under
an IDENTICAL preregistered protocol as a PRIOR accepted result, this
module classifies how the new result relates to the old one. It is
generic: it knows nothing about any pillar, cipher, model, or metric,
runs no experiments, and computes no new experimental data - it
operates only on already-computed summary quantities.

CHOSEN METHODOLOGY: PREDICTION INTERVAL
---------------------------------------
Statistical consistency is assessed with a PREDICTION INTERVAL around
the prior estimate, following the established replication-assessment
literature (Patil, Peng & Leek 2016, *Perspect Psychol Sci* 11(4);
Spence & Stanley 2016, *PLOS ONE* 11(9):e0162874; Spence & Stanley
2024, *AMPPS* 7(1)). The prediction interval answers precisely the
question asked here - "is the new result within the range expected
from sampling error alone, given the prior result?" - by combining the
sampling variability of BOTH results:

    PI = theta_prior  +/-  z_crit * sqrt(se_prior^2 + se_new^2)

A replication estimate falling inside this interval is statistically
consistent with the prior result; one falling outside is not.

WHY NOT CONFIDENCE-INTERVAL OVERLAP
-----------------------------------
An earlier draft of this module used CI overlap/non-overlap as a
consistency signal. That was REJECTED as methodologically wrong. Two
95% CIs can overlap substantially while the difference between the
estimates is significant (Greenland et al. 2016, *Eur J Epidemiol*
31(4), misinterpretation #23 - their worked example gives overlapping
intervals with p = 0.03). Overlap is a conservative, biased proxy:
non-overlap implies significance, but overlap implies nothing. The
variances of the two estimates add, so the correct comparison is built
on sqrt(se_1^2 + se_2^2) - which is exactly what the prediction
interval does and what CI-overlap eyeballing does not. CI overlap is
therefore retained ONLY as an optional descriptive field, explicitly
labelled as not inferential, and it NEVER drives a classification.

WHY NOT EQUIVALENCE TESTING (TOST) AS THE PRIMARY MECHANISM
-----------------------------------------------------------
TOST answers "is the difference smaller than a predeclared margin?",
which is a genuine and useful question - but it is a DIFFERENT question
from "is this replication consistent with sampling error?", and it
requires a justified equivalence margin that may not exist for an
arbitrary metric. The practical threshold, where the caller supplies
one, is used here for the separate MATERIALITY judgement (is the shift
big enough to matter?), which is the role it is suited to. Statistical
consistency (prediction interval) and practical materiality (threshold)
are kept as orthogonal axes rather than conflated.

WHY NOT A BARE SIGNIFICANCE TEST OF THE DIFFERENCE
--------------------------------------------------
A two-sample test of prior-vs-new would detect arbitrarily small
differences at large N, and "significantly different" is not the
question a replication assessment asks. The prediction interval is
scaled to what sampling error can produce, which is the relevant
yardstick.

THE THREE CONCEPTUAL OUTCOMES, AND A FOURTH
--------------------------------------------
    ORDINARY_VARIATION_SAME_DECISION - decision unchanged, new estimate
        inside the prediction interval, and (if a threshold was given)
        the shift is within it.
    MAGNITUDE_SHIFT_SAME_DECISION - decision unchanged, but the shift
        is either outside the prediction interval or exceeds the
        predeclared practical threshold. The qualitative conclusion
        held; the numbers moved enough to report rather than absorb.
    MATERIALLY_CONTRADICTORY - the decision itself changed, or the
        estimate is both statistically inconsistent AND practically
        material. Requires investigation/reassessment.
    INSUFFICIENT_EVIDENCE - added because the three above cannot be
        honestly distinguished without uncertainty information. If
        neither standard errors nor a practical threshold are supplied,
        this module REFUSES to certify "ordinary variation" from a
        decision match alone - a matching decision with unknown
        uncertainty is not evidence of consistency, and manufacturing
        that certainty would be exactly the kind of unsupported
        upgrade this framework exists to prevent.

ASSUMPTIONS
-----------
- The prediction interval as implemented assumes the two estimates are
  independent and approximately normally distributed around a common
  true value, and uses a normal (z) critical value. Callers working
  with small replicate counts should supply a t-based critical value
  via `critical_value` rather than relying on the normal default.
- Standard errors must be on the SAME scale as the point estimates.
- Protocol identity is the caller's assertion (see below); this module
  verifies the identity fields it is given, not the world.

LIMITATIONS
-----------
- A wide prior interval (imprecise prior result) yields a wide
  prediction interval, so consistency becomes easy to achieve and
  correspondingly uninformative. Patil et al. note this explicitly.
  The `prediction_interval_width` is therefore always returned so a
  reader can judge how demanding the test actually was.
- This module classifies a comparison; it does not decide whether a
  replication SHOULD be run, and it never edits prior evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReplicationConsistencyError(ValueError):
    pass


class ReplicationComparisonClass(str, Enum):
    ORDINARY_VARIATION_SAME_DECISION = "ORDINARY_VARIATION_SAME_DECISION"
    MAGNITUDE_SHIFT_SAME_DECISION = "MAGNITUDE_SHIFT_SAME_DECISION"
    MATERIALLY_CONTRADICTORY = "MATERIALLY_CONTRADICTORY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ProtocolIdentity:
    """
    Fields that establish that two results were produced under the SAME
    PROTOCOL - i.e. the same preregistered experimental specification
    and configuration.

    IMPORTANT (corrected): `dataset_snapshot_id` is deliberately NOT a
    protocol-identity field. A dataset snapshot identifies a particular
    INPUT REALIZATION, not the protocol. Requiring it to match would
    have forbidden exactly the case this module exists to serve: a
    genuine independent replication normally draws FRESH data under the
    same protocol, and would therefore carry a different snapshot id.
    Conversely two runs on the identical snapshot are a re-run
    (reproduction), not an independent replication.

    The snapshot id is therefore carried alongside as INFORMATIONAL
    context (see `input_realization`), and the classifier reports
    whether the input realization was identical so a reader can tell a
    reproduction from a replication - but it never blocks the
    comparison.
    """
    experiment_id: str
    config_hash: str
    preregistration_hash: Optional[str]

    def mismatches(self, other: "ProtocolIdentity") -> list[str]:
        fields = ["experiment_id", "config_hash", "preregistration_hash"]
        return [f for f in fields if getattr(self, f) != getattr(other, f)]


class ConsistencyMethod(str, Enum):
    """
    How statistical consistency is assessed. The normal-approximation
    prediction interval is the default, but it is NOT valid for every
    metric type, so the method is explicit rather than assumed.

    NORMAL_PREDICTION_INTERVAL
        theta_prior +/- crit * sqrt(se_prior^2 + se_new^2). Valid when
        the estimator is approximately normally distributed and
        unbounded on the relevant scale (means, mean differences,
        log-odds, Fisher-z-transformed correlations).

    CALLER_SUPPLIED
        The caller has already determined statistical consistency by a
        method appropriate to its metric (e.g. a bootstrap predictive
        distribution for a bounded proportion near 0 or 1, or an
        exact method for a rate) and passes the verdict in directly.
        Used so that this generic module never silently applies a
        normal approximation to a metric that violates it.

    Metrics known to need CALLER_SUPPLIED rather than the default:
    proportions/accuracies near the 0/1 boundary, raw correlation
    coefficients (transform to Fisher z first), variance ratios, and
    any bounded or strongly skewed statistic.
    """
    NORMAL_PREDICTION_INTERVAL = "NORMAL_PREDICTION_INTERVAL"
    CALLER_SUPPLIED = "CALLER_SUPPLIED"


@dataclass(frozen=True)
class SuppliedConsistencyVerdict:
    """
    A statistical-consistency verdict computed by the caller with a
    method appropriate to its metric.

    CORRECTED: the verdict alone is not auditable - a bare boolean
    gives a reviewer no way to check HOW consistency was determined.
    A method description is therefore REQUIRED, and an evidence
    reference (the artifact/analysis that produced it) is recorded
    where one exists. This preserves the generic abstraction while
    keeping the resulting classification traceable.
    """
    statistically_consistent: bool
    method_description: str
    evidence_reference: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.method_description or not self.method_description.strip():
            raise ReplicationConsistencyError(
                "SuppliedConsistencyVerdict requires a non-empty method_description. A bare "
                "boolean verdict is not auditable: a reviewer must be able to see by what "
                "statistical method consistency was determined."
            )

    def to_dict(self) -> dict:
        return {
            "statistically_consistent": self.statistically_consistent,
            "method_description": self.method_description,
            "evidence_reference": self.evidence_reference,
        }


@dataclass(frozen=True)
class InputRealization:
    """Informational context distinguishing a reproduction from a replication."""
    prior_dataset_snapshot_id: Optional[str]
    new_dataset_snapshot_id: Optional[str]

    @property
    def identical(self) -> Optional[bool]:
        if self.prior_dataset_snapshot_id is None or self.new_dataset_snapshot_id is None:
            return None
        return self.prior_dataset_snapshot_id == self.new_dataset_snapshot_id

    @property
    def comparison_kind(self) -> str:
        """
        NEUTRAL description of the input relationship only.

        CORRECTED: this deliberately does NOT classify the scientific
        activity as a "reproduction" or a "replication". Snapshot
        equality is a fact about inputs; whether a given comparison
        constitutes reproduction, replication, or something else is a
        scientific judgement that depends on what else varied (fresh
        randomness, independent regeneration, different operators) and
        is left to the caller. Inferring it here from snapshot equality
        alone would assert more than the data supports.
        """
        if self.identical is None:
            return "UNKNOWN"
        return "SAME_INPUT_REALIZATION" if self.identical else "DIFFERENT_INPUT_REALIZATION"


@dataclass(frozen=True)
class ReplicationConsistencyResult:
    comparison_class: ReplicationComparisonClass
    decision_changed: bool
    point_estimate_delta: float
    prediction_interval: Optional[tuple[float, float]]
    prediction_interval_width: Optional[float]
    within_prediction_interval: Optional[bool]
    exceeds_practical_threshold: Optional[bool]
    descriptive_confidence_intervals_overlap: Optional[bool]
    input_realization_kind: str
    consistency_method: str
    supplied_consistency_verdict: Optional[dict]
    rationale: str
    methodology: str = (
        "Prediction interval (Patil/Peng/Leek 2016; Spence & Stanley 2016) for statistical "
        "consistency; predeclared practical threshold for materiality. CI overlap is "
        "descriptive only and does not drive classification."
    )

    def to_dict(self) -> dict:
        return {
            "comparison_class": self.comparison_class.value,
            "decision_changed": self.decision_changed,
            "point_estimate_delta": self.point_estimate_delta,
            "prediction_interval": list(self.prediction_interval) if self.prediction_interval else None,
            "prediction_interval_width": self.prediction_interval_width,
            "within_prediction_interval": self.within_prediction_interval,
            "exceeds_practical_threshold": self.exceeds_practical_threshold,
            "descriptive_confidence_intervals_overlap": self.descriptive_confidence_intervals_overlap,
            "input_realization_kind": self.input_realization_kind,
            "consistency_method": self.consistency_method,
            "supplied_consistency_verdict": self.supplied_consistency_verdict,
            "rationale": self.rationale,
            "methodology": self.methodology,
        }


class EstimateDependence(str, Enum):
    """
    Whether the two estimates being compared are statistically
    independent.

    INDEPENDENT
        The estimates come from independently generated data. The
        standard prediction-interval formula
        sqrt(se_prior^2 + se_new^2) applies.

    DEPENDENT
        The estimates share input data (e.g. both evaluated on the SAME
        dataset realization), so their sampling errors are correlated.
        The independent-estimate formula is NOT valid here: ignoring a
        positive covariance OVERSTATES the combined variance, producing
        an interval that is too wide and therefore too permissive -
        it would call genuinely discrepant results "consistent".
        Correcting it requires the covariance (or the SE of the paired
        difference), which this generic module cannot derive. Callers
        in this situation must compute consistency themselves and pass
        the verdict via ConsistencyMethod.CALLER_SUPPLIED.

    UNKNOWN
        Dependence has not been established. Treated as DEPENDENT for
        safety: the module refuses rather than silently assuming
        independence.
    """
    INDEPENDENT = "INDEPENDENT"
    DEPENDENT = "DEPENDENT"
    UNKNOWN = "UNKNOWN"


def compute_prediction_interval(
    *, prior_estimate: float, prior_se: float, new_se: float,
    critical_value: float = 1.959963985,
    dependence: EstimateDependence = EstimateDependence.INDEPENDENT,
) -> tuple[float, float]:
    """
    Prediction interval around `prior_estimate` for a replication whose
    own standard error is `new_se`.

    ASSUMPTIONS (now explicit and enforced):
      - The two estimates are INDEPENDENT. The formula
        spread = crit * sqrt(se_prior^2 + se_new^2)
        adds variances, which is only valid with zero covariance.
        Passing dependence=DEPENDENT or UNKNOWN raises rather than
        silently applying the independent formula (see
        EstimateDependence).
      - Each estimate is approximately normally distributed on the
        scale supplied, and `critical_value` matches the intended
        coverage (default: two-sided normal 95%). Use a t-based value
        for small replicate counts.

    Default critical_value is the two-sided normal 95% value.
    """
    if dependence is not EstimateDependence.INDEPENDENT:
        raise ReplicationConsistencyError(
            f"compute_prediction_interval assumes INDEPENDENT estimates, but dependence="
            f"{dependence.value}. Adding variances is invalid when sampling errors are "
            "correlated (it overstates the combined variance and yields an interval that is "
            "too permissive). Compute consistency with a method appropriate to the dependent "
            "design and supply it via ConsistencyMethod.CALLER_SUPPLIED."
        )
    if prior_se < 0 or new_se < 0:
        raise ReplicationConsistencyError("Standard errors must be non-negative.")
    if critical_value <= 0:
        raise ReplicationConsistencyError(
            f"critical_value must be positive, got {critical_value!r}. A non-positive "
            "critical value would produce a degenerate or inverted interval."
        )
    if not all(math.isfinite(v) for v in (prior_estimate, prior_se, new_se, critical_value)):
        raise ReplicationConsistencyError("Non-finite input to prediction interval computation.")
    spread = critical_value * math.sqrt(prior_se ** 2 + new_se ** 2)
    return (prior_estimate - spread, prior_estimate + spread)


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def classify_replication_consistency(
    *,
    prior_identity: ProtocolIdentity,
    new_identity: ProtocolIdentity,
    prior_decision: str,
    new_decision: str,
    prior_point_estimate: float,
    new_point_estimate: float,
    consistency_method: ConsistencyMethod = ConsistencyMethod.NORMAL_PREDICTION_INTERVAL,
    dependence: EstimateDependence = EstimateDependence.UNKNOWN,
    prior_standard_error: Optional[float] = None,
    new_standard_error: Optional[float] = None,
    supplied_consistency_verdict: Optional[SuppliedConsistencyVerdict] = None,
    practical_threshold: Optional[float] = None,
    critical_value: float = 1.959963985,
    input_realization: Optional[InputRealization] = None,
    prior_confidence_interval: Optional[tuple[float, float]] = None,
    new_confidence_interval: Optional[tuple[float, float]] = None,
) -> ReplicationConsistencyResult:
    """
    Classify a new replication against a prior accepted result under an
    identical protocol.

    TWO INDEPENDENT AXES (never conflated):

      STATISTICAL CONSISTENCY - is the new estimate within what sampling
        error can produce given the prior estimate? Determined EITHER by
        the normal-approximation prediction interval (requires BOTH
        standard errors) OR by a caller-supplied verdict computed with a
        method appropriate to the metric.

      PRACTICAL MATERIALITY - is the shift larger than a predeclared
        threshold that matters scientifically?

    CORRECTED RULE: a practical threshold ALONE can never establish
    ORDINARY_VARIATION_SAME_DECISION. "The shift is smaller than a
    threshold I chose" is a statement about importance, not about
    whether the difference is attributable to sampling error, and the
    two are not interchangeable. Without a statistical basis the result
    is INSUFFICIENT_EVIDENCE - though a threshold alone CAN still
    establish a MAGNITUDE_SHIFT when it is exceeded, since that
    direction needs no sampling-error argument.

    Raises ReplicationConsistencyError if the PROTOCOL identities differ.
    Differing dataset snapshots do NOT block the comparison - see
    ProtocolIdentity's docstring.
    """
    mismatched = prior_identity.mismatches(new_identity)
    if mismatched:
        raise ReplicationConsistencyError(
            f"Cannot classify replication consistency: protocol identity differs in "
            f"{mismatched}. A changed protocol is a different experiment, not a replication."
        )

    for name, value in (("prior_point_estimate", prior_point_estimate),
                        ("new_point_estimate", new_point_estimate)):
        if not math.isfinite(value):
            raise ReplicationConsistencyError(f"{name} must be finite, got {value!r}.")

    if (
        dependence is EstimateDependence.INDEPENDENT
        and input_realization is not None
        and input_realization.identical is True
    ):
        raise ReplicationConsistencyError(
            "dependence=INDEPENDENT was asserted, but the supplied input_realization shows "
            "both results used the SAME input realization, under which sampling errors are "
            "plausibly correlated. Either justify and supply a dependence-appropriate verdict "
            "via ConsistencyMethod.CALLER_SUPPLIED, or correct the dependence declaration."
        )

    decision_changed = prior_decision != new_decision
    delta = new_point_estimate - prior_point_estimate

    # --- Axis 1: statistical consistency ---
    prediction_interval: Optional[tuple[float, float]] = None
    pi_width: Optional[float] = None
    statistically_consistent: Optional[bool] = None

    if consistency_method is ConsistencyMethod.CALLER_SUPPLIED:
        statistically_consistent = (
            supplied_consistency_verdict.statistically_consistent
            if supplied_consistency_verdict is not None else None
        )
    elif prior_standard_error is not None and new_standard_error is not None:
        # CORRECTED: dependence must be an EXPLICIT caller decision. It
        # defaults to UNKNOWN, and compute_prediction_interval refuses
        # anything other than INDEPENDENT - so the independent-estimate
        # formula can no longer be reached by silent default. Previously
        # this call omitted `dependence` entirely, which meant a
        # SAME_INPUT_REALIZATION comparison (where sampling errors are
        # plausibly correlated) silently used the independent formula.
        prediction_interval = compute_prediction_interval(
            prior_estimate=prior_point_estimate, prior_se=prior_standard_error,
            new_se=new_standard_error, critical_value=critical_value,
            dependence=dependence,
        )
        pi_width = prediction_interval[1] - prediction_interval[0]
        statistically_consistent = (
            prediction_interval[0] <= new_point_estimate <= prediction_interval[1]
        )

    # --- Axis 2: practical materiality ---
    exceeds_threshold: Optional[bool] = None
    if practical_threshold is not None:
        if practical_threshold < 0:
            raise ReplicationConsistencyError("practical_threshold must be non-negative.")
        exceeds_threshold = abs(delta) > practical_threshold

    # --- Descriptive only; never used for classification ---
    ci_overlap: Optional[bool] = None
    if prior_confidence_interval is not None and new_confidence_interval is not None:
        ci_overlap = _intervals_overlap(prior_confidence_interval, new_confidence_interval)

    def build(cls: ReplicationComparisonClass, rationale: str) -> ReplicationConsistencyResult:
        return ReplicationConsistencyResult(
            comparison_class=cls, decision_changed=decision_changed, point_estimate_delta=delta,
            prediction_interval=prediction_interval, prediction_interval_width=pi_width,
            within_prediction_interval=statistically_consistent,
            exceeds_practical_threshold=exceeds_threshold,
            descriptive_confidence_intervals_overlap=ci_overlap,
            supplied_consistency_verdict=(
                supplied_consistency_verdict.to_dict()
                if supplied_consistency_verdict is not None else None
            ),
            input_realization_kind=(
                input_realization.comparison_kind if input_realization else "UNKNOWN"
            ),
            consistency_method=consistency_method.value,
            rationale=rationale,
        )

    if decision_changed:
        return build(
            ReplicationComparisonClass.MATERIALLY_CONTRADICTORY,
            f"Decision changed from {prior_decision!r} to {new_decision!r} under an identical "
            "protocol. A decision flip is definitionally material and requires investigation, "
            "independent of the magnitude of the underlying numerical shift.",
        )

    statistically_inconsistent = (statistically_consistent is False)
    practically_material = (exceeds_threshold is True)

    if statistically_inconsistent and practically_material:
        return build(
            ReplicationComparisonClass.MATERIALLY_CONTRADICTORY,
            "The decision held, but the replication estimate is outside what sampling error "
            "can explain AND the shift exceeds the predeclared practical threshold. Both axes "
            "indicate a real, material discrepancy requiring investigation despite the "
            "unchanged decision label.",
        )

    if statistically_inconsistent or practically_material:
        reasons = []
        if statistically_inconsistent:
            reasons.append("is outside what sampling error can explain")
        if practically_material:
            reasons.append("exceeds the predeclared practical threshold")
        return build(
            ReplicationComparisonClass.MAGNITUDE_SHIFT_SAME_DECISION,
            "The qualitative decision held, but the replication estimate "
            + " and ".join(reasons)
            + ". This shift should be reported explicitly rather than absorbed into 'no change'.",
        )

    # Nothing fired. ORDINARY_VARIATION requires a POSITIVE statistical
    # basis - not merely the absence of a threshold breach.
    if statistically_consistent is not True:
        missing = (
            "no caller-supplied consistency verdict was provided (a "
                 "SuppliedConsistencyVerdict with a method_description is required)"
            if consistency_method is ConsistencyMethod.CALLER_SUPPLIED
            else "standard errors for both results were not supplied, so no prediction "
                 "interval could be computed"
        )
        return build(
            ReplicationComparisonClass.INSUFFICIENT_EVIDENCE,
            f"The decision matched and no practical threshold was breached, but {missing}. "
            "A within-threshold shift is a statement about importance, not about whether the "
            "difference is attributable to sampling error; classification as ordinary variation "
            "is withheld rather than manufacturing an unsupported statistical claim.",
        )

    return build(
        ReplicationComparisonClass.ORDINARY_VARIATION_SAME_DECISION,
        "The decision held and the replication estimate is within the range expected from "
        "sampling error alone"
        + (" and within the predeclared practical threshold" if exceeds_threshold is False else "")
        + " - consistent with ordinary replicate-to-replicate variation.",
    )
