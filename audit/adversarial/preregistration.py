"""
ADVERSARIAL BATTERY - FROZEN PRE-REGISTRATION.

The CipherMind dimensions have so far been exercised on one model that
passes them. That demonstrates the framework can validate; it does not
demonstrate the framework can DISCRIMINATE. An instrument that only ever
confirms is not falsifiable.

This battery closes that gap by constructing models that are known, by
construction, not to be cryptanalytic distinguishers, and asking which
components of the framework notice. The outcome of interest is not "the
framework catches them" - it is the full map of which component catches
which adversary, INCLUDING the cells where a dimension fails to catch.
Those failures measure the discriminative power of each dimension and are
reported as findings, not as defects to be hidden.

This file is frozen BEFORE any adversary is evaluated. `plan_hash()` binds
every adversarial certificate to these predictions, so a prediction cannot
be revised after the result is seen. A wrong prediction is a result; a
silently edited prediction is misconduct.

The dataset pillar already validates its DETECTORS against planted defects
(tests/test_d{1,2,3,4}_adversarial_validation.py). This battery extends the
same pattern from detectors to whole models passing through the inferential
dimensions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

PREREG_VERSION = "ciphermind-adversarial-prereg-2026-10-06"

#: SCALE. The battery runs at a REDUCED, DECLARED scale, not at the frozen
#: production scale of CE2/CE3/CE4. Justification: the adversarial claim is
#: qualitative - "dimension X cannot discriminate adversary A from a genuine
#: distinguisher" - and an adversary engineered to maximise a dimension's
#: estimand produces an effect far larger than the production design was
#: powered to detect. Adversarial certificates are therefore NOT comparable
#: in statistical power to production certificates and must never be
#: presented alongside them as if they were.
ADVERSARIAL_SCALE = {
    "label": "ADVERSARIAL-SCALE (reduced; NOT production scale)",
    "ce2": {"n_runs": 5, "n_samples": 100_000},
    "ce3": {"n_replicates": 5, "n_samples": 20_000, "n_splits": 5},
    "ce4": {"n_runs": 3, "n_samples": 20_000},
    "comparability": "not comparable with the production CE2/CE3/CE4 certificates",
}

#: Outcomes a dimension can return for an adversary.
CAUGHT = "CAUGHT"          # the component refused or returned a negative verdict
NOT_CAUGHT = "NOT_CAUGHT"  # the component returned a positive verdict anyway
NA = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Adversary:
    adversary_id: str
    name: str
    construction: str
    why_not_a_distinguisher: str
    #: Component -> predicted outcome. Frozen before execution.
    predictions: dict = field(default_factory=dict)
    prediction_rationale: dict = field(default_factory=dict)
    #: Treatment-fidelity gate: was this adversary actually built?
    manipulation_check: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)


A1 = Adversary(
    adversary_id="A1",
    name="theory-mimic",
    construction=(
        "A network with the reference architecture (depth 10, L2 1e-5, identical "
        "64-bit input encoding and identical 64-unit penultimate layer) trained by "
        "regression onto a monotone rescaling of the analytical single-trail "
        "probability of each REAL difference-bearing ciphertext pair. It never sees "
        "a random-class sample and is never trained on the real-vs-random label."),
    why_not_a_distinguisher=(
        "The real/random decision problem is never posed to it. Its output is a "
        "monotone function of a quantity computed from the realised trail, so on the "
        "mixed real/random population it carries no class information beyond whatever "
        "the trail statistic incidentally provides. It is a theory regressor wearing a "
        "distinguisher's architecture."),
    predictions={
        "distinguisher_accuracy_on_sealed_set": "approximately 0.50 (chance)",
        "model_binding_gate": CAUGHT,
        "CE2_theory_consistency": NOT_CAUGHT,
        "CE3_representation_decodability": NOT_CAUGHT,
        "CE4_intervention_sensitivity": NOT_CAUGHT,
    },
    prediction_rationale={
        "model_binding_gate": (
            "preflight binds by checkpoint hash AND by behavioural round binding "
            "(accuracy >= 0.85 at 5 rounds). A1 fails both, so the gate should refuse "
            "it before any dimension executes."),
        "CE2_theory_consistency": (
            "CE2's estimand is the rank association between model output and the "
            "analytical quantity. A1 is trained to BE a monotone function of that "
            "quantity, so it should maximise CE2's estimand - predicted Spearman rho "
            ">= +0.90, i.e. higher than the audited distinguisher's +0.807."),
        "CE3_representation_decodability": (
            "to regress the quantity, the penultimate layer must encode it, so the "
            "probe should recover it at least as well as from the audited model. "
            "Predicted replicate-level selectivity > +0.20 and SUPPORTED."),
        "CE4_intervention_sensitivity": (
            "the structural intervention changes the ciphertext-pair XOR and hence the "
            "trail quantity A1 regresses; the control preserves the XOR and should "
            "leave it nearly unchanged. Predicted positive gap and SUPPORTED. This "
            "prediction is the least certain of the three."),
    },
    #: MANIPULATION CHECK (treatment fidelity). Before any dimension verdict is
    #: interpreted, A1 must actually BE a theory-mimic: its output must track
    #: the analytical quantity on held-out real pairs. An under-trained network
    #: is not the adversary this battery registered, and its dimension verdicts
    #: say nothing about the hypothesis. The threshold is outcome-blind - it
    #: concerns whether the treatment was delivered, not what the dimensions
    #: then conclude - and mirrors the manipulation checks CE4 already requires.
    manipulation_check={
        "requirement": "Spearman rho between A1's output and the analytical target, "
                       "on held-out real pairs, must be >= 0.90",
        "rationale": "A1's whole construction is 'a monotone function of the "
                     "analytical quantity'. If it does not achieve that, the "
                     "adversary was not built, and no conclusion about CE2/CE3/CE4 "
                     "may be drawn from it.",
        "on_failure": "CONSTRUCTION FAILED - retrain at the registered scale and "
                      "re-run; do not interpret the dimension verdicts.",
        "threshold": 0.90,
    },
    confidence={
        "model_binding_gate": "high",
        "CE2_theory_consistency": "high",
        "CE3_representation_decodability": "high",
        "CE4_intervention_sensitivity": "moderate - stated explicitly because it may "
                                        "be wrong, and a wrong prediction here is "
                                        "itself informative",
    },
)

#: The claim the battery is designed to test. Stated before execution so the
#: conclusion cannot be reshaped around whatever the data happen to show.
HYPOTHESIS = (
    "H-ADV-1: the inferential dimensions CE2, CE3 and CE4, taken on their own, do "
    "not establish that an audited model is a cryptanalytic distinguisher. If A1 is "
    "SUPPORTED by them while scoring at chance on the real/random task, then the "
    "model-binding gate - not the inferential dimensions - is what carries that part "
    "of the argument, and the framework's conclusions must be stated accordingly.")

#: What each possible result would mean. Fixed in advance so neither outcome
#: can be spun after the fact.
INTERPRETATION_RULES = {
    "A1 refused by the gate AND SUPPORTED by CE2/CE3/CE4 when the gate is bypassed": (
        "H-ADV-1 is supported. The gate is load-bearing rather than decorative, and "
        "CE2/CE3/CE4 verdicts must always be reported as conditional on it."),
    "A1 refused by the gate AND rejected by one or more dimensions": (
        "that dimension discriminates on its own; record which, and the framework's "
        "claim strengthens correspondingly."),
    "A1 NOT refused by the gate": (
        "a defect in the binding, which would be a finding about the framework and "
        "must be fixed and reported, not quietly patched."),
    "A1 scores well above chance on the real/random task": (
        "the construction failed - A1 is accidentally a distinguisher - and no "
        "conclusion about the dimensions may be drawn from it. The adversary would "
        "have to be rebuilt and re-registered."),
}

#: The counterfactual is explicit: dimensions are run on A1 only with the
#: binding deliberately bypassed, and every resulting certificate is stamped
#: non-evidentiary. No adversarial run may ever be presented as evidence
#: about the audited model.
EXECUTION_PROTOCOL = {
    "step_1": "verify A1 is not a distinguisher: accuracy on a sealed real/random set",
    "step_2": "submit A1 to the ordinary model-binding gate and record the refusal",
    "step_3": "bypass the gate EXPLICITLY and run CE2, CE3, CE4 on A1 at adversarial "
              "scale, recording each verdict",
    "step_4": "compare every outcome with the frozen prediction above",
    "certificates": "stamped ADVERSARIAL - NOT EVIDENCE ABOUT THE AUDITED MODEL",
}

ADVERSARIES = {A1.adversary_id: A1}


def prereg_dict() -> dict:
    return {
        "prereg_version": PREREG_VERSION,
        "hypothesis": HYPOTHESIS,
        "scale": ADVERSARIAL_SCALE,
        "execution_protocol": EXECUTION_PROTOCOL,
        "interpretation_rules": INTERPRETATION_RULES,
        "adversaries": {k: asdict(v) for k, v in ADVERSARIES.items()},
        "outcome_vocabulary": [CAUGHT, NOT_CAUGHT, NA],
    }


def plan_hash() -> str:
    return hashlib.sha256(
        json.dumps(prereg_dict(), sort_keys=True, default=str).encode()).hexdigest()
