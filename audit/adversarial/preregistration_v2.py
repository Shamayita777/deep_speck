"""
ADVERSARIAL BATTERY - PRE-REGISTRATION v2.

v1 (preregistration.py) is NOT modified and NOT deleted. A1 and its wrong
prediction stay in the record. This file registers a new battery whose design
was corrected after v1's construction failure, and it is a NEW registration
with its own hash, not an edit of the old one.

WHY v1 FAILED, AND WHY THE FIX IS NOT COSMETIC
----------------------------------------------
A1 was registered as a "theory-mimic that is not a distinguisher". It
regressed the analytical single-trail quantity and then scored 0.88 on the
real/random task, so the registered construction-validity rule voided its
dimension verdicts. The post-mortem is that the construction was close to
self-contradictory: the analytical quantity is defined through the realised
differential trail of a REAL pair, and the thing that separates real from
random 5-round Speck pairs IS the differential structure of the ciphertext
pair (Gohr, CRYPTO 2019; Benamira et al., EUROCRYPT 2021). A model that
predicts the quantity well must encode that structure, and encoding it is
most of distinguishing.

The correction is structural rather than a parameter tweak, but it is NOT
that every v2 adversary carries an analytic guarantee. The rule is:

  * an ANALYTIC non-distinguishing guarantee is PREFERRED wherever the
    construction admits one, and where it exists the empirical check only
    confirms that the implementation matches the specification (A3: the
    output is a constant function of the input, so ROC AUC is exactly 0.5 for
    every dataset);
  * where no analytic guarantee is available, non-distinguishing status is
    established EMPIRICALLY against a preregistered equivalence criterion
    fixed before the confirmatory run (A4).

A4 therefore has no analytic guarantee of being a non-distinguisher. Its
output depends on the input only through the ciphertext-pair XOR, and the
real/random XOR distributions differ, so some residual separation is possible
in principle; whether it is small enough is decided by the registered
equivalence criterion and may fail.

DEVELOPMENT SCREENING (screening.py, DEVELOPMENT ONLY - NOT EVIDENCE) was run
before this registration to characterise candidate features. It measured, on
300,000 mixed real/random 5-round pairs and 120,000 real pairs:

    feature                         AUC(real/random)   Spearman vs CE2 target
    popcount(pair XOR), 32 bits     0.5610 [0.5590, 0.5628]        -0.206
    popcount(pair XOR), left word   0.5231 [0.5214, 0.5249]        -0.127
    popcount(C0) only               0.5007 [0.5000, 0.5025]        -0.001
    parity(pair XOR)                0.5005 [0.5000, 0.5025]        +0.007
    keyed BLAKE2b map(pair XOR)     0.5020 [0.5003, 0.5039]        -0.003
    audited Gohr model (reference)  0.9760                         +0.806

Two conclusions follow, and both are registered here BEFORE any confirmatory
run:

(1) The previously proposed A1' - "train a model to predict the Hamming
    weight of the ciphertext-pair XOR" - is REJECTED. Its AUC is 0.561 with a
    confidence interval excluding 0.5 by ~30 standard errors, so it is a
    genuine, if weak, distinguisher. It is also not a "nuisance" feature in
    any defensible sense: the Hamming weight of the ciphertext-pair XOR is a
    coarse statistic OF THE DIFFERENTIAL, the exact object differential
    cryptanalysis studies. Calling it non-cryptographic would be a label of
    convenience.

(2) The LOW-ASSOCIATION / LOW-DISTINGUISHABILITY screened features -
    popcount of a single ciphertext, parity of the pair XOR, and a keyed
    BLAKE2b-derived map of the pair XOR - measured AUC <= 0.502 with
    |rho| <= 0.01 against the CE2 target. Note that parity and the keyed map
    ARE functions of the ciphertext-pair differential; the claim is about
    their measured association and distinguishability, NOT that they carry no
    differential content, which would require a mathematical argument this
    registration does not make. In the screened family, association with
    CE2's target and real/random distinguishing power moved together. That is
    a development observation, not a theorem, and it is registered as the
    motivation for the hypothesis below rather than as a finding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

PREREG_VERSION = "ciphermind-adversarial-prereg-v2-2026-10-06"
SUPERSEDES = "ciphermind-adversarial-prereg-2026-10-06"   # v1 retained, not edited

# ---------------------------------------------------------------------------
# THREE SEPARATE AXES. No value on one axis may be derived from another.
# ---------------------------------------------------------------------------

#: Axis 1 - the raw decision the CE dimension returned, copied verbatim.
SCIENTIFIC_DECISIONS = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE")

#: Axis 2 - adversarial interpretation. UNDETERMINED exists precisely so that
#: INCONCLUSIVE never has to be forced into CAUGHT or NOT_CAUGHT.
CAUGHT, NOT_CAUGHT, NOT_APPLICABLE, UNDETERMINED = (
    "CAUGHT", "NOT_CAUGHT", "NOT_APPLICABLE", "UNDETERMINED")

#: Axis 3 - was the adversary actually built as registered?
CONSTRUCTION_VALID, CONSTRUCTION_FAILED = "VALID", "FAILED"

#: The ONLY permitted mapping from axis 1 to axis 2, fixed here.
#: SUPPORTED      -> the dimension accepted a model lacking the property  -> NOT_CAUGHT
#: NOT_SUPPORTED  -> the dimension declined to accept it                  -> CAUGHT
#: INCONCLUSIVE   -> the dimension rendered no verdict                    -> UNDETERMINED
#: An INCONCLUSIVE caused by insufficient replication is an absence of
#: evidence about the dimension, never evidence that the dimension works.
DECISION_TO_ADVERSARIAL = {
    "SUPPORTED": NOT_CAUGHT,
    "NOT_SUPPORTED": CAUGHT,
    "INCONCLUSIVE": UNDETERMINED,
}

#: If construction is FAILED, every adversarial outcome is NOT_APPLICABLE
#: regardless of what the dimensions returned. The raw scientific decisions
#: are still recorded verbatim; only their adversarial interpretation is
#: suppressed.
CONSTRUCTION_FAILURE_SUPPRESSES_INTERPRETATION = True


# ---------------------------------------------------------------------------
# Replication: the adversarial protocol has its OWN estimand and its OWN rules
# ---------------------------------------------------------------------------

#: v1's error was to call the production dimension estimators at reduced
#: replication and then read their production decision rule, which returns
#: INCONCLUSIVE below the production minimum - and v1's code then mapped that
#: to CAUGHT. v2 fixes this at the level of the design, not the mapping:
#: the adversarial protocol declares its own replication counts AND its own
#: decision rule, and never reuses a production decision object.
ADVERSARIAL_PROTOCOL = {
    "estimand": (
        "For a model M lacking the scientific property a dimension is meant to "
        "evidence, does that dimension's PRIMARY ENDPOINT fall on the accepting side "
        "of its own preregistered criterion? This is a question about the dimension's "
        "discriminative power, not about the audited model, and it is not the "
        "production estimand."),
    "unit_of_replication": "independent evaluation replicate, as in production",
    "replication": {"CE2": 20, "CE3": 20, "CE4": 10},
    "replication_justification": (
        "Matched to the production counts. A reduced count was considered and "
        "rejected: the adversarial claim is that a dimension ACCEPTS a model it "
        "should not, and accepting is exactly the regime where the production "
        "decision rule's replication minimum binds. Running fewer replicates than "
        "production would make the dimension return INCONCLUSIVE for reasons that "
        "have nothing to do with the adversary, which is the defect v1 contained. "
        "The cost is modest because the adversaries here are analytic or cheap to "
        "construct; CE3 dominates at roughly 20 x the per-replicate probe cost."),
    "sample_counts": {"CE2": 1_000_000, "CE3": 100_000, "CE4": 100_000},
    "decision_rule": (
        "Adversarial acceptance is read from the dimension's own primary endpoint "
        "and its own preregistered criterion, evaluated by this protocol: CE2, the "
        "exact two-sided sign test over replicate-level rho with the preregistered "
        "control-separation requirement; CE3, the fixed-sequence calibration gate "
        "then the one-sided Wilcoxon over replicate-level selectivity; CE4, the "
        "across-replicate sign test at the primary magnitude with all manipulation "
        "checks passing. A dimension that cannot evaluate its criterion returns "
        "INCONCLUSIVE and maps to UNDETERMINED."),
    "exclusions": "a failed replicate is recorded and retained; it is never replaced",
    "stopping_rule": "exactly the registered replicate count; no interim looks",
}


# ---------------------------------------------------------------------------
# Construction-validity criteria, with justification for every number
# ---------------------------------------------------------------------------

#: v1 used "approximately 0.50" and then |acc - 0.5| < 0.05, neither of which
#: was defensible: 0.05 was a round number, and accuracy depends on an
#: arbitrary threshold. v2 uses AUC, which is threshold-free, plus a
#: two-sided equivalence region justified below, plus - for adversaries that
#: admit one - an ANALYTIC argument that makes the empirical check a
#: confirmation rather than the load-bearing step.
DISTINGUISHER_CRITERION = {
    "primary_metric": "ROC AUC on a sealed, hash-bound, mixed real/random test set",
    "why_auc": (
        "threshold-free, so it cannot be inflated by selecting a cut on the same "
        "data; and it is the probability that a random real pair is scored above a "
        "random random pair, which is the quantity 'does this model separate the "
        "classes at all' asks."),
    "equivalence_region": [0.49, 0.51],
    "equivalence_rule": (
        "CONSERVATIVE 95% CONFIDENCE-INTERVAL EQUIVALENCE CRITERION: the two-sided "
        "95% bootstrap confidence interval for AUC must lie ENTIRELY inside "
        "[0.49, 0.51]. This is NOT a conventional alpha = 0.05 TOST, which "
        "corresponds to containment of the 90% interval; requiring the 95% interval "
        "is strictly more conservative, so a construction admitted under this rule "
        "would also be admitted by a 0.05 TOST, while the converse does not hold. It "
        "remains an equivalence-style criterion and NOT a non-significance test: a "
        "wide interval straddling 0.5 FAILS, because absence of evidence of "
        "separation is not evidence of absence."),
    "equivalence_criterion_name": "conservative 95% CI equivalence criterion",
    "relation_to_tost": "strictly more conservative than alpha = 0.05 TOST "
                        "(which would require only 90% CI containment)",
    "margin_justification": (
        "The margin is set from measured properties of this problem, not chosen for "
        "convenience. Development screening put the weakest DIFFERENTIAL feature "
        "examined - left-word XOR popcount - at AUC 0.523, and the full pair-XOR "
        "popcount at 0.561, while features with no differential content sat at "
        "0.5005-0.5020 with intervals of half-width ~0.002. A margin of 0.01 is "
        "therefore (a) wider than the sampling noise of the measurement at the "
        "registered n, and (b) less than half the distance to the weakest "
        "differential candidate among the screened features, so a model admitted by "
        "this rule is separated from that candidate by a factor of at least two. "
        "A model whose interval lies inside [0.49, 0.51] is not usefully a "
        "distinguisher at this sample size, and the certificate records the bound "
        "rather than claiming exact chance."),
    "secondary_descriptive": ["accuracy at the fixed 0.5 threshold",
                              "accuracy at a threshold calibrated on a DISJOINT "
                              "calibration split",
                              "balanced accuracy at that calibrated threshold"],
    "sealed_set": {
        "n": 1_000_000, "composition": "balanced real/random, as make_train_data emits",
        "persistence": "generated once, written to disk, SHA256 recorded, reloaded "
                       "for every evaluation; never regenerated during scoring",
        "calibration_split": "a SEPARATE persisted set, used only to choose a "
                             "threshold; the sealed set is never used to select "
                             "anything",
    },
}

FIDELITY_CRITERION = {
    "rule": "each adversary declares its own fidelity statistic and threshold; the "
            "threshold must be justified by the construction, not by convention",
    "note": "v1's 0.90 was arbitrary. Where an adversary's target is computed "
            "exactly rather than learned, fidelity is exact by construction and the "
            "check is an assertion rather than a threshold.",
}

CONFIRMATORY_PRECONDITIONS = {
    "clean_working_tree": (
        "a confirmatory run REFUSES to start if the git working tree is dirty: a "
        "certificate whose source hashes cannot be tied to a committed state is not "
        "independently reconstructable"),
    "git_required": "if git metadata is unavailable the run refuses unless the "
                    "operator passes --allow-no-git, which is recorded in the "
                    "certificate as a provenance limitation",
    "source_hash_coverage": (
        "every source artifact that materially affects execution is hashed: the "
        "adversarial package, the CE234 plan/production/verifier, and the Gohr "
        "cipher, dataset, model, trainer and evaluator, the probe evaluator and the "
        "CE4 design module"),
    "smoke_and_build_exempt": "only --run is gated; --smoke and --build are not "
                              "confirmatory",
    "frozen_registration_hash": (
        "a confirmatory run recomputes plan_hash() and refuses unless it equals the "
        "frozen constant EXPECTED_PREREG_HASH. A clean commit containing a silently "
        "edited registration is therefore not accepted as the same registration"),
    "no_construction_during_confirmation": (
        "--run never trains, rebuilds or reconstructs an adversary. A missing "
        "artifact is a hard failure, not a trigger to build one"),
    "frozen_model_artifact": (
        "--build writes a freeze manifest recording the model SHA256, training "
        "record, source hashes, registration hash and environment; --run verifies "
        "the loaded artifact against it and refuses on mismatch. No selection among "
        "candidate artifacts is permitted"),
}

#: EXECUTION SCOPE RULE. `targets` is authoritative: a dimension is executed
#: only if it appears in the adversary's registered targets. A prediction
#: entry alone must never cause execution - that is how an earlier draft would
#: have run CE2 for an adversary registered against CE4 only.
EXECUTION_SCOPE_RULE = (
    "execute dimension D for adversary A iff D in A.targets; every other dimension "
    "is recorded NOT_APPLICABLE with its registered rationale and is not run")

MODEL_BINDING_CRITERION = {
    "applicability": "the gate presupposes a checkpoint artifact; for an adversary "
                     "without one the binding is recorded as NOT_APPLICABLE with a "
                     "reason, never as CAUGHT",
    "rule": "every adversary WITH A CHECKPOINT is submitted to the UNMODIFIED "
            "production binding "
            "(checkpoint SHA256 plus behavioural round binding) and the outcome is "
            "recorded before any dimension runs",
    "expected": "refusal, on the hash alone",
    "if_not_refused": "a defect in the binding: report it as a finding, fix it in "
                      "the open, do not patch silently",
    "do_not_weaken": True,
}


# ---------------------------------------------------------------------------
# Adversaries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Adversary:
    adversary_id: str
    version: str
    name: str
    targets: tuple
    construction: str
    non_distinguisher_argument: str
    analytic_guarantee: bool
    #: Does the production model-binding gate even apply to this adversary?
    #: The gate verifies a checkpoint SHA256 and a behavioural round binding,
    #: both of which presuppose a trained checkpoint artifact.
    binding_applicable: bool = True
    binding_not_applicable_reason: str = ""
    fidelity: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    prediction_rationale: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    status: str = "REGISTERED"


A1_HISTORICAL = Adversary(
    adversary_id="A1", version="1.0", name="theory-mimic (HISTORICAL, FAILED)",
    targets=("CE2", "CE3", "CE4"),
    construction="see preregistration.py v1 - NOT modified",
    non_distinguisher_argument="asserted, not analytic - this is why it failed",
    analytic_guarantee=False,
    status="CONSTRUCTION_FAILED - retained permanently; its dimension verdicts are "
           "void and its wrong prediction stands in the record",
)

A3 = Adversary(
    adversary_id="A3", version="1.0", name="representation adversary (CE3-targeted)",
    targets=("CE3",),
    construction=(
        "Reference architecture. The penultimate 64-unit layer is trained to encode "
        "CE3's target (the quintile-discretised analytical quantity) by attaching a "
        "temporary classification head during construction; the head is then "
        "DISCARDED and the model's output layer is replaced by a constant map. The "
        "shipped adversary therefore emits an identical value for every input."),
    non_distinguisher_argument=(
        "ANALYTIC. The output is a constant function of the input, so the induced "
        "ranking of samples is empty and ROC AUC is exactly 0.5 by construction, for "
        "every dataset, with no sampling uncertainty. No empirical argument is "
        "needed; the empirical check exists only to confirm the implementation "
        "matches the specification."),
    analytic_guarantee=True,
    fidelity={"statistic": "balanced accuracy of a probe trained on A3's penultimate "
                           "representation for CE3's target, on held-out folds",
              "threshold": "must exceed the untrained-twin control by a positive "
                           "margin on every registered replicate; no fixed numeric "
                           "threshold is imposed because CE3's own criterion is "
                           "the comparison, not an absolute level"},
    predictions={
        "distinguisher_auc": "exactly 0.5 (analytic)",
        "model_binding_gate": CAUGHT,
        "CE3_representation_decodability": NOT_CAUGHT,
        "CE2_theory_consistency": NOT_APPLICABLE,
        "CE4_intervention_sensitivity": NOT_APPLICABLE,
    },
    prediction_rationale={
        "CE3_representation_decodability": (
            "CE3 probes layers[-2], not the output. A3's penultimate layer is trained "
            "to encode the target, so the probe should recover it well above the "
            "untrained twin. If CE3 nevertheless returns NOT_SUPPORTED, CE3 "
            "discriminates on grounds beyond decodability and that is a finding in "
            "CE3's favour."),
        "CE2_theory_consistency": (
            "NOT_APPLICABLE by construction: a constant output makes Spearman rho "
            "undefined, so CE2 cannot be evaluated on A3 and no claim about CE2 is "
            "made from it. The certificate must record 'undefined', not 0."),
        "CE4_intervention_sensitivity": (
            "NOT_APPLICABLE: a constant output gives an identically zero intervention "
            "effect in both arms."),
    },
    confidence={"CE3_representation_decodability": "high",
                "distinguisher_auc": "certain (analytic)"},
)

A4 = Adversary(
    adversary_id="A4", version="1.0", name="XOR-response adversary (CE4-targeted)",
    targets=("CE4",),
    construction=(
        "Not a trained network: a closed-form response function. The model maps a "
        "ciphertext pair to h(C0 XOR C1), where h is a keyed BLAKE2b-derived "
        "deterministic map into [0, 1] (BLAKE2b of the packed 32-bit XOR, keyed by a "
        "registered constant, divided by 2^64). No formal pseudorandom-function "
        "claim is made or needed: all the construction requires is determinism and "
        "dependence on the XOR alone. Implemented behind the same predict() "
        "interface the dimensions use."),
    non_distinguisher_argument=(
        "EMPIRICAL, against the registered equivalence criterion. NO analytic "
        "guarantee is claimed. h depends on the input ONLY through the "
        "ciphertext-pair XOR, and h is a keyed BLAKE2b-derived deterministic map, so "
        "it carries no monotone information about the XOR's distribution; but the "
        "real and random XOR distributions do differ, so residual separation is "
        "possible in principle. Development screening measured AUC 0.5020 "
        "[0.5003, 0.5039] at n = 300,000, which lies ENTIRELY INSIDE the registered "
        "region [0.49, 0.51]. That is encouraging but not decisive: the screening "
        "set is not the sealed set, and interval width scales with n, so A4 may "
        "still fail its own construction criterion on the confirmatory data. A "
        "small-n illustration of that failure mode was observed during pipeline "
        "smoke at n = 8,000, where the interval [0.5024, 0.5266] fell outside the "
        "region. The confirmatory run re-measures on the sealed set and the "
        "equivalence criterion decides; failure is a registered possibility, not a "
        "defect to be designed away."),
    analytic_guarantee=False,
    binding_applicable=False,
    binding_not_applicable_reason=(
        "A4 is a closed-form response function with no checkpoint artifact. The "
        "production binding verifies a checkpoint SHA256 and a behavioural round "
        "binding; neither is defined for A4. Submitting a non-existent path to the "
        "gate and recording the resulting error as CAUGHT would be a fabricated "
        "result. A3 is the genuine model-binding adversary."),
    fidelity={"statistic": "exactness of the XOR dependence",
              "rule": "assert f(x) == f(x') whenever the two inputs have identical "
                      "pair XOR; exact by construction, verified on sampled pairs "
                      "rather than estimated"},
    predictions={
        "distinguisher_auc": "95% CI inside [0.49, 0.51]",
        "model_binding_gate": NOT_APPLICABLE,
        "CE4_intervention_sensitivity": NOT_CAUGHT,
        "CE2_theory_consistency": NOT_APPLICABLE,
        "CE3_representation_decodability": NOT_APPLICABLE,
    },
    prediction_rationale={
        "CE4_intervention_sensitivity": (
            "CE4's control arm preserves the pair XOR exactly, so A4's output is "
            "EXACTLY unchanged under the control and the control effect is "
            "identically zero. The structural arm changes the XOR at 2k positions, so "
            "the output changes. The gap is therefore positive by construction and "
            "CE4 should accept a model that is demonstrably not cryptanalytic. This "
            "is the sharpest test in the battery."),
        "CE2_theory_consistency": (
            "CE2 is outside A4's registered target scope and is NOT executed. A4 "
            "targets CE4 only; v2 registers no adversary against CE2, as stated "
            "under HYPOTHESES.not_tested_here. An earlier draft carried a CE2 "
            "prediction for A4, which would have executed a dimension outside the "
            "registered scope; that is corrected here."),
        "CE3_representation_decodability": (
            "NOT_APPLICABLE: A4 is a closed-form function with no hidden layer, so "
            "CE3 has nothing to probe. Recorded as not applicable, not as a pass."),
    },
    confidence={"CE4_intervention_sensitivity": "high (analytic for the control arm)",
                "distinguisher_auc": "moderate - the screening estimate is from "
                                     "development data and must be re-established on "
                                     "the sealed set"},
)

ADVERSARIES = {a.adversary_id: a for a in (A3, A4)}
HISTORICAL = {A1_HISTORICAL.adversary_id: A1_HISTORICAL}

REJECTED_CANDIDATES = {
    "A1'-hamming-weight": {
        "proposal": "train a model to predict popcount(C0 XOR C1)",
        "rejected_because": (
            "measured AUC 0.5610, 95% CI [0.5590, 0.5628], far outside the "
            "equivalence region, so it is a real distinguisher; and the Hamming "
            "weight of the ciphertext-pair XOR is a coarse statistic of the "
            "differential itself, so describing it as a non-cryptographic nuisance "
            "feature would not survive review"),
        "evidence": "screening.py, DEVELOPMENT ONLY - NOT SCIENTIFIC EVIDENCE",
    },
}

# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

HYPOTHESES = {
    "H-ADV-2": (
        "CE3's criterion can be satisfied by a model whose output is constant, i.e. "
        "one that performs no cryptanalytic task at all. If so, CE3 evidences "
        "DECODABILITY of its target from a representation and nothing further, and "
        "CE3 verdicts must always be reported together with the model binding that "
        "establishes the model is a distinguisher."),
    "H-ADV-3": (
        "CE4's criterion can be satisfied by a closed-form response function of the "
        "ciphertext-pair XOR. If so, CE4 evidences SENSITIVITY TO THE INTERVENTION "
        "CONTRAST and not causal dependence on cryptographic structure, and its "
        "wording must be bounded accordingly."),
    "not_tested_here": (
        "H-ADV-1 as stated in v1 - about CE2, CE3 and CE4 jointly - is NOT tested by "
        "this battery. A1 failed, and no adversary in v2 is designed to defeat CE2. "
        "The screening observation that association with CE2's target and "
        "distinguishing power moved together in the screened family is a development "
        "observation that motivates leaving CE2 untested, not a result about CE2."),
}

INTERPRETATION_RULES = {
    "construction FAILED": "all adversarial outcomes NOT_APPLICABLE; raw scientific "
                           "decisions still recorded; no hypothesis is tested",
    "dimension returns INCONCLUSIVE": "UNDETERMINED; record the dimension's own stated "
                                      "reason; never read as the dimension working",
    "dimension returns SUPPORTED with construction VALID": "NOT_CAUGHT; the "
                                                           "corresponding hypothesis "
                                                           "is supported for that "
                                                           "dimension",
    "dimension returns NOT_SUPPORTED with construction VALID": "CAUGHT; that dimension "
                                                               "discriminates against "
                                                               "this adversary, which "
                                                               "strengthens it",
    "binding gate does not refuse": "framework defect; report and fix in the open",
    "scope": "every result here concerns the DIMENSIONS, never the audited model. "
             "Nothing in this battery may be cited as evidence about what the audited "
             "Gohr model does or does not use.",
}

GLOBAL_CONCLUSION_UNCHANGED = (
    "Claim C1 remains INCONCLUSIVE / LEVEL_1_PREDICTIVE. This battery cannot change "
    "it: it evidences properties of the audit dimensions, not of the audited model.")


#: FROZEN REGISTRATION HASH. Computed over prereg_dict() AFTER the design was
#: finalised and BEFORE any confirmatory run. This constant is deliberately NOT
#: included in prereg_dict(), so declaring it does not perturb the value it
#: pins. A confirmatory run refuses unless plan_hash() equals this.
EXPECTED_PREREG_HASH = (
    "ed079688bc7e84e73e550ea28c9f44ea838feb46bdbfa4e05cc0a033f8cbc4df")


def prereg_dict() -> dict:
    return {
        "prereg_version": PREREG_VERSION, "supersedes": SUPERSEDES,
        "axes": {"scientific_decisions": list(SCIENTIFIC_DECISIONS),
                 "adversarial_outcomes": [CAUGHT, NOT_CAUGHT, NOT_APPLICABLE,
                                          UNDETERMINED],
                 "construction_status": [CONSTRUCTION_VALID, CONSTRUCTION_FAILED]},
        "decision_to_adversarial": DECISION_TO_ADVERSARIAL,
        "construction_failure_suppresses_interpretation":
            CONSTRUCTION_FAILURE_SUPPRESSES_INTERPRETATION,
        "protocol": ADVERSARIAL_PROTOCOL,
        "confirmatory_preconditions": CONFIRMATORY_PRECONDITIONS,
        "execution_scope_rule": EXECUTION_SCOPE_RULE,
        "distinguisher_criterion": DISTINGUISHER_CRITERION,
        "fidelity_criterion": FIDELITY_CRITERION,
        "model_binding_criterion": MODEL_BINDING_CRITERION,
        "adversaries": {k: asdict(v) for k, v in ADVERSARIES.items()},
        "historical": {k: asdict(v) for k, v in HISTORICAL.items()},
        "rejected_candidates": REJECTED_CANDIDATES,
        "hypotheses": HYPOTHESES,
        "interpretation_rules": INTERPRETATION_RULES,
        "global_conclusion_unchanged": GLOBAL_CONCLUSION_UNCHANGED,
    }


def plan_hash() -> str:
    return hashlib.sha256(
        json.dumps(prereg_dict(), sort_keys=True, default=str).encode()).hexdigest()


def map_decision(scientific_decision: str, construction_status: str) -> str:
    """
    The ONLY path from a scientific decision to an adversarial outcome.
    INCONCLUSIVE can never become CAUGHT; a failed construction suppresses
    interpretation entirely.
    """
    if scientific_decision not in SCIENTIFIC_DECISIONS:
        raise ValueError(f"unknown scientific decision {scientific_decision!r}")
    if construction_status == CONSTRUCTION_FAILED:
        return NOT_APPLICABLE
    return DECISION_TO_ADVERSARIAL[scientific_decision]
