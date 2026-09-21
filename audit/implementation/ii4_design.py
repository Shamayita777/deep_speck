"""
FROZEN II-4 design specification.

Every value in this module is a preregistered methodological decision.
Changing any of them after execution begins invalidates the confirmatory
inference; a changed design must be issued as a NEW versioned
specification, never an edit of this one.

The Gohr instantiation appears at the bottom and is clearly separated
from the generic design parameters above it, so a second audit reuses
the generic block and supplies only its own ConformanceFactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from audit.common.provenance import sha256_bytes
from audit.common.strict_json import dumps_strict
from audit.implementation.conformance import ConformanceFactor

II4_SPEC_VERSION = "II4-SPEC-V1"


@dataclass(frozen=True)
class II4Design:
    """Generic, case-study-independent II-4 design parameters."""

    # --- design ---
    n_blocks: int                       # K: dataset blocks = experimental units
    min_valid_blocks: int               # below this -> INCONCLUSIVE, never a verdict

    # --- outcome ---
    primary_endpoint: str               # what Y is
    evaluation_rule: str                # which model object is evaluated
    epochs: int                         # fixed schedule; no early stopping

    # --- pairing semantics (declared, not assumed) ---
    same_dataset: bool
    same_model_initialization: bool
    same_training_shuffle_stream: bool
    same_evaluation_data: bool

    # --- inference ---
    alpha: float
    delta_materiality: float            # predeclared practical margin
    delta_is_equivalence_margin: bool

    # --- confirmatory evaluation data ---
    confirmatory_test_set: str = "GLOBAL_SHARED"   # ONE set for all blocks and arms
    confirmatory_data_mode: str = "SHARED"         # firewall provenance mode

    spec_version: str = II4_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.min_valid_blocks > self.n_blocks:
            raise ValueError("min_valid_blocks cannot exceed n_blocks.")
        if self.n_blocks < 2:
            raise ValueError("A paired design requires at least 2 blocks.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        if self.delta_materiality <= 0:
            raise ValueError("delta_materiality must be positive.")
        if self.confirmatory_test_set != "GLOBAL_SHARED" or self.confirmatory_data_mode != "SHARED":
            raise ValueError(
                "II-4 uses ONE global shared confirmatory test set "
                "(confirmatory_test_set='GLOBAL_SHARED', confirmatory_data_mode='SHARED').")
        if self.same_model_initialization:
            # Guard against silently reintroducing the assumption that a
            # paired design means a shared model seed. For a conformance
            # comparison the two arms have different parameter shapes, so a
            # shared seed produces non-corresponding weights and yields no
            # variance reduction; declaring it would misstate the design.
            raise ValueError(
                "II-4 declares same_model_initialization=False. Arms differing in an "
                "architectural conformance factor have different parameter shapes, so a "
                "shared seed does not constitute paired initialization."
            )

    @property
    def sign_flip_feasible(self) -> bool:
        """
        Exact paired sign-flip resolution: 2^K assignments give a minimum
        two-sided p of 2/2^K. Below K=6 the test cannot reach alpha=0.05
        at any effect size, so it must not be reported as if it could.
        """
        return (2 / (2 ** self.n_blocks)) < self.alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "n_blocks": self.n_blocks,
            "min_valid_blocks": self.min_valid_blocks,
            "primary_endpoint": self.primary_endpoint,
            "evaluation_rule": self.evaluation_rule,
            "epochs": self.epochs,
            "pairing": {
                "same_dataset": self.same_dataset,
                "same_model_initialization": self.same_model_initialization,
                "same_training_shuffle_stream": self.same_training_shuffle_stream,
                "same_evaluation_data": self.same_evaluation_data,
            },
            "alpha": self.alpha,
            "delta_materiality": self.delta_materiality,
            "delta_is_equivalence_margin": self.delta_is_equivalence_margin,
            "confirmatory_test_set": self.confirmatory_test_set,
            "confirmatory_data_mode": self.confirmatory_data_mode,
            "sign_flip_feasible": self.sign_flip_feasible,
        }

    def preregistration_hash(self) -> str:
        return sha256_bytes(dumps_strict(self.to_dict(), sort_keys=True).encode("utf-8"))


# =====================================================================
# FROZEN DESIGN (generic parameters)
# =====================================================================

II4_DESIGN = II4Design(
    n_blocks=10,
    min_valid_blocks=9,
    primary_endpoint="sealed_confirmatory_test_accuracy",
    evaluation_rule="terminal_epoch",
    epochs=200,
    same_dataset=True,
    same_model_initialization=False,
    same_training_shuffle_stream=False,
    same_evaluation_data=True,
    alpha=0.05,
    delta_materiality=0.01,
    delta_is_equivalence_margin=True,
)

EVALUATION_RULE_RATIONALE = (
    "TERMINAL EPOCH. The model is trained for exactly the reference 200 epochs and the "
    "in-memory model at the terminal epoch is evaluated. No epoch is selected using "
    "validation performance for the primary endpoint.\n\n"
    "What this does and does not achieve: it ELIMINATES validation-based checkpoint-"
    "selection as a mechanism by which the two arms could be measured at differently "
    "chosen points. It does NOT make the estimator unbiased for 'the effect of the "
    "conformance factor on capability' in general: the terminal epoch may represent "
    "different convergence states across arms, so the estimand is specifically "
    "TERMINAL-EPOCH PERFORMANCE UNDER THE FIXED 200-EPOCH SCHEDULE, and must be reported "
    "with that qualification.\n\n"
    "Best-val-loss checkpointing may remain enabled for provenance/debugging. It must not "
    "determine the primary endpoint. Note that validation-based selection would not have "
    "introduced optimism into a SEALED TEST measurement in any case (selection used "
    "validation data, not test data); the reason for fixing the terminal epoch is "
    "comparability of the evaluation point across arms, not test-side bias."
)

ESTIMAND_STATEMENT = (
    "Delta_k = Y_k(declared) - Y_k(realized), the SIGNED per-block difference in sealed "
    "confirmatory test accuracy. The target estimand is E[Delta] taken over training/"
    "validation dataset generation, model initialization and training shuffle, under the "
    "frozen 200-epoch terminal-evaluation rule. The primary estimator is the mean of the K "
    "block differences; the experimental unit is the DATASET BLOCK, not the training run "
    "and not the test example."
)

CONDITIONAL_INFERENCE_STATEMENT = (
    "INFERENCE IS CONDITIONAL ON THE FIXED SEALED EVALUATION SET. II-4 uses ONE global "
    "1,000,000-example confirmatory test set, generated once before block 0, persisted, "
    "hashed, sealed, and reused for all 10 blocks and both arms. Every Delta_k is therefore "
    "measured on the same test sample. This induces correlated evaluation measurements and "
    "removes independent test-sampling variation from the between-block variability of "
    "Delta; it does NOT make test-sampling error 'cancel exactly', and the resulting "
    "confidence interval does not account for variation that a different test sample would "
    "produce. Conclusions are stated conditional on this sealed set."
)

STATISTICAL_UNIT_STATEMENT = (
    "The dataset block is the experimental unit (K blocks -> K observations, df = K-1). The "
    "1,000,000 sealed test examples are evaluation observations NESTED INSIDE each block; "
    "they are not independent experimental replicates for the conformance comparison and "
    "must never be treated as such."
)


# =====================================================================
# PREREGISTERED INTERPRETATION RULE (frozen before execution)
#
# Maps the II-4 confirmatory verdicts onto the Implementation Integrity
# outcome. Recorded in the preregistration and therefore in the config
# fingerprint; the certificate converter reads the rule FROM THE
# PREREGISTRATION FILE, never from this constant, so a post-hoc edit here
# cannot change the interpretation of an already-run experiment.
# =====================================================================

INTERPRETATION_RULE_VERSION = "II4-INTERPRETATION-V1"

INTERPRETATION_RULE = {
    "version": INTERPRETATION_RULE_VERSION,
    "rules": [
        {"when": {"materiality": "MATERIALLY_HIGHER"},
         "scientific_impact": "DEMONSTRATED", "ii_outcome": "FAIL",
         "meaning": ("II-4 demonstrates a material performance difference under the tested "
                     "protocol; Implementation Integrity FAILS for depth-5-scoped evidence.")},
        {"when": {"materiality": "MATERIALLY_LOWER"},
         "scientific_impact": "DEMONSTRATED", "ii_outcome": "FAIL",
         "meaning": ("II-4 demonstrates a material performance difference under the tested "
                     "protocol; Implementation Integrity FAILS for depth-5-scoped evidence.")},
        {"when": {"equivalence": "EQUIVALENT_WITHIN_MARGIN"},
         "scientific_impact": "REFUTED", "ii_outcome": "CONDITIONAL_PASS",
         "meaning": ("CONDITIONAL_PASS for ACCURACY CONFORMANCE ONLY. This does NOT validate "
                     "transferability of Cryptographic Evidence obtained on the depth-5 model: "
                     "II-4 measures accuracy, not internal representation or intervention "
                     "sensitivity.")},
        {"when": {"materiality": "NO_DIRECTIONAL_CLAIM", "equivalence_not": "EQUIVALENT_WITHIN_MARGIN"},
         "scientific_impact": "UNDEMONSTRATED", "ii_outcome": "INCONCLUSIVE",
         "meaning": "No directional materiality claim and equivalence not established."},
        {"when": {"materiality": "INSUFFICIENT_BLOCKS"},
         "scientific_impact": "UNDEMONSTRATED", "ii_outcome": "INCONCLUSIVE",
         "meaning": "Fewer than the preregistered minimum valid blocks (Rule IV)."},
    ],
    "post_hoc_prohibition": ("This rule is fixed before execution and must never be inferred, "
                             "altered or selected after results are observed."),
}


def apply_interpretation_rule(rule: dict, materiality: str, equivalence: str) -> dict:
    """
    Apply a PREREGISTERED rule (passed in, typically loaded from the
    preregistration file) to the confirmatory verdicts. Exactly one rule
    must match; zero or several matches is a hard error rather than a
    silent default.
    """
    matches = []
    for r in rule["rules"]:
        w = r["when"]
        ok = True
        if "materiality" in w and w["materiality"] != materiality:
            ok = False
        if "equivalence" in w and w["equivalence"] != equivalence:
            ok = False
        if "equivalence_not" in w and w["equivalence_not"] == equivalence:
            ok = False
        if ok:
            matches.append(r)
    if len(matches) != 1:
        raise ValueError(
            f"Interpretation rule {rule.get('version')} matched {len(matches)} clauses for "
            f"materiality={materiality!r}, equivalence={equivalence!r}; exactly one is required.")
    return matches[0]


# =====================================================================
# GOHR INSTANTIATION (case-study specific; swap this for another audit)
# =====================================================================

GOHR_DEPTH_FACTOR = ConformanceFactor(
    name="residual_depth",
    declared_value=10,
    realized_value=5,
    declaration_source=(
        "audit/implementation/reference/train_5_rounds.py: "
        "train_speck_distinguisher(200, num_rounds=5, depth=10)"
    ),
    verification_method=(
        "Embedded Keras model_config read directly from the historical checkpoint "
        "audit/evidence_bundle/cryptography/ce1/best5depth10 (10).h5 "
        "(sha256 de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea, identical "
        "to its record in audit/evidence_bundle/BUNDLE_MANIFEST.json): 11 Conv1D layers and "
        "5 Add (residual merge) layers. Independently corroborated by constructing the reference "
        "architecture at depths 1/2/5/10 and counting layers, giving depth = (n_conv1d-1)/2 "
        "= number of Add layers. Filename was not relied upon."
    ),
    finding_id="II-FINDING-DEPTH-V1",
)

GOHR_FIXED_PROTOCOL = {
    "cipher": "speck32_64",
    "num_rounds": 5,
    "differential": (0x0040, 0x0000),
    "epochs": 200,
    "batch_size": 5000,
    "optimizer": "adam",
    "loss": "mse",
    "lr_schedule": "cyclic_lr(10, 0.002, 0.0001)",
    "lr_period": 10,
    "lr_high": 0.002,
    "lr_low": 0.0001,
    # Reference: train_nets.py builds make_resnet(depth=depth, reg_param=10**-5).
    # The EV adapter omits reg_param and silently uses the 1e-4 default; II-4
    # passes the reference value explicitly.
    "reg_param": 1e-5,
    "shuffle": True,                      # reference-conformant (Keras default in train_nets)
    "train_samples": 10_000_000,
    "validation_samples": 1_000_000,
    "sealed_test_samples": 1_000_000,
}

FROZEN_BASE_SEED = 700_000
