#!/usr/bin/env python3
"""
Frozen design, interpretation rule and execution plan for a paired-block
II-4 comparison - GENERIC.

No Gohr, Speck, Keras or neural-network knowledge. A case study supplies
a ComparisonAxis, a fixed protocol dict and an adapter; this module
supplies the design semantics, the decision rule, the preregistration
document and its hash.

RELATIONSHIP TO THE EXISTING II-4 DESIGN MODULE
-----------------------------------------------
`ii4_design.II4Design` is frozen and bound by source hash into the
completed conformance experiment's evidence. It is NOT modified and NOT
imported for mutation. The design parameters here are deliberately the
same quantities with the same meanings, so that two II-4 experiments on
different axes remain comparable, and the test suite asserts field-level
agreement with the frozen II4_DESIGN rather than leaving the claim of
parity to prose.

Two things are added that the conformance design did not need:

  REPLACEMENT BLOCKS. A design with K blocks and a minimum of M valid
  blocks tolerates K-M losses and then returns INCONCLUSIVE. Simulation
  of the exact binomial survival probability showed that K=7 with M=6
  retains only ~0.45 probability of reaching M at the one-sided 95%
  upper bound on the per-arm failure rate implied by the evidence so
  far (0 failures in 21 observed arm-trainings => rate <= 0.133). The
  design therefore pre-declares a bounded number of REPLACEMENT blocks.

  Replacement is admissible ONLY for a technical failure occurring
  BEFORE the sealed-test evaluation of that block, i.e. when the block
  has no endpoint value at all. It is therefore independent of the
  outcome: there is no outcome to be influenced by. Replacing a block
  because its difference was unwelcome is excluded structurally, since
  the firewall consumes the sealed test set exactly once per (arm,
  block) and a block that reached evaluation can never be replaced.
  The bound is frozen in advance so that the procedure is not
  outcome-adaptive sample-size inflation.

  ARM-STRUCTURE PREMISE. An implementation comparison presupposes that
  both implementations realize the SAME model structure; if they do
  not, the comparison confounds implementation with architecture. The
  design records this as an explicit precondition the runner must
  verify before training, rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit.common.outcomes import ExecutionMode
from audit.common.provenance import sha256_bytes, utc_timestamp
from audit.common.strict_json import dumps_strict
from audit.implementation.comparison import (
    ArmSeedManifest,
    AxisKind,
    ComparisonAxis,
)

II4_COMPARISON_SPEC_VERSION = "II4-COMPARISON-SPEC-V1"


# =====================================================================
# design
# =====================================================================

@dataclass(frozen=True)
class ComparisonDesign:
    """Generic paired-block II-4 design parameters."""

    n_blocks: int
    min_valid_blocks: int
    max_replacement_blocks: int

    primary_endpoint: str
    evaluation_rule: str
    epochs: int

    # pairing semantics - declared, never assumed
    same_dataset: bool
    same_model_initialization: bool
    same_training_shuffle_stream: bool
    same_evaluation_data: bool

    alpha: float
    delta_materiality: float
    delta_is_equivalence_margin: bool

    confirmatory_test_set: str = "GLOBAL_SHARED"
    confirmatory_data_mode: str = "SHARED"
    spec_version: str = II4_COMPARISON_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.n_blocks < 2:
            raise ValueError("A paired design requires at least 2 blocks.")
        if self.min_valid_blocks > self.n_blocks:
            raise ValueError("min_valid_blocks cannot exceed n_blocks.")
        if self.max_replacement_blocks < 0:
            raise ValueError("max_replacement_blocks cannot be negative.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        if self.delta_materiality <= 0:
            raise ValueError("delta_materiality must be positive.")
        if self.confirmatory_test_set != "GLOBAL_SHARED" or self.confirmatory_data_mode != "SHARED":
            raise ValueError(
                "II-4 uses ONE global shared confirmatory test set "
                "(confirmatory_test_set='GLOBAL_SHARED', confirmatory_data_mode='SHARED').")
        if self.same_model_initialization:
            raise ValueError(
                "same_model_initialization must be False. Two arms that differ in an "
                "architectural parameter have different parameter shapes; two arms that are "
                "different implementations execute different RNG draw sequences. In neither "
                "case does a shared integer seed produce corresponding initial weights, so "
                "declaring paired initialization would misstate the design.")
        if not self.same_dataset:
            raise ValueError("same_dataset must be True: the block design depends on it.")
        if not self.same_evaluation_data:
            raise ValueError(
                "same_evaluation_data must be True: the arms are compared on one sealed set.")

    @property
    def max_total_blocks(self) -> int:
        return self.n_blocks + self.max_replacement_blocks

    @property
    def sign_flip_feasible_at_min_valid(self) -> bool:
        """
        Exact paired sign-flip resolution at the MINIMUM number of valid
        blocks, not at the planned number. A design whose secondary test
        is feasible only if every block survives has no feasible
        secondary test; stating it at min_valid is the honest form.
        """
        return (2 / (2 ** self.min_valid_blocks)) < self.alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "n_blocks": self.n_blocks,
            "min_valid_blocks": self.min_valid_blocks,
            "max_replacement_blocks": self.max_replacement_blocks,
            "max_total_blocks": self.max_total_blocks,
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
            "sign_flip_feasible_at_min_valid": self.sign_flip_feasible_at_min_valid,
        }

    def preregistration_hash(self) -> str:
        return sha256_bytes(dumps_strict(self.to_dict(), sort_keys=True).encode("utf-8"))


REPLACEMENT_BLOCK_RULE = {
    "rule_version": "II4-REPLACEMENT-V1",
    "target": (
        "replacement restores the PREREGISTERED block count n_blocks, not merely "
        "min_valid_blocks: a run that stopped at the minimum would carry less power than the "
        "design that was preregistered while leaving its replacement budget unused"),
    "admissible_when": (
        "a block is invalidated by a TECHNICAL failure of one or both arms that occurs "
        "BEFORE the sealed-test evaluation of that block, so the block carries no endpoint "
        "value for either arm"),
    "inadmissible_when": [
        "any arm of the block reached sealed-test evaluation (the firewall has recorded a "
        "consumption for that arm and block)",
        "the replacement budget max_replacement_blocks is already exhausted",
        "the failure is a disagreement with an expected result rather than a technical fault",
    ],
    "why_this_is_not_outcome_adaptive": (
        "A replaceable block has no endpoint value, so the decision to replace it cannot "
        "depend on the value it would have had. The budget is frozen before execution and "
        "cannot be raised during the run. Both conditions are enforced by the runner, not "
        "left to the operator."),
    "recorded": (
        "every replacement records the failed block id it replaces, the failure reason, and "
        "its own fresh dataset and seeds; the failed block is RETAINED in the artifact"),
}


ARM_STRUCTURE_PRECONDITION = (
    "For an IMPLEMENTATION axis the runner verifies, before any training, that both arms "
    "construct models of identical structure (layer classes, shapes and parameter count). "
    "Two implementations that realize different structures cannot be compared on this design: "
    "the measured difference would confound implementation with architecture. A mismatch "
    "ABORTS the run rather than being recorded as a result.")

ESTIMAND_STATEMENT = (
    "Delta = E[Y(arm_a) - Y(arm_b)], expectation taken over dataset-block generation, model "
    "initialization and training shuffle, at a FIXED checkpoint rule, on a FIXED sealed test "
    "set. The estimand is a difference of expected endpoint values, not a difference of best "
    "achievable values; no selection over epochs or runs enters it.")

CONDITIONAL_INFERENCE_STATEMENT = (
    "All inference is CONDITIONAL on the single sealed confirmatory test set. The common test "
    "set induces correlated evaluation measurements across arms and blocks and removes "
    "independent test-sampling variation between arms; it does not cancel exactly. "
    "Generalisation beyond this test set is not claimed.")

STATISTICAL_UNIT_STATEMENT = (
    "The experimental unit is the DATASET BLOCK, not the training example and not the model. "
    "K valid blocks give K paired differences and df = K-1. Test examples are not units: they "
    "are shared across every arm and block, so treating them as units would be "
    "pseudo-replication and would understate the standard error by orders of magnitude.")


# =====================================================================
# interpretation rule
# =====================================================================

INTERPRETATION_RULE_VERSION = "II4-IMPL-INTERPRETATION-V1"

#: Frozen mapping from (materiality, equivalence) to a scientific
#: conclusion, for an IMPLEMENTATION axis. Written as data so that the
#: rule is hashed into the preregistration and cannot be re-interpreted
#: by editing code after a run. The conclusion is a CONJUNCTION: TOST
#: alone never licenses an equivalence claim here.
IMPLEMENTATION_INTERPRETATION_RULE = {
    "rule_version": INTERPRETATION_RULE_VERSION,
    "applies_to_axis_kind": AxisKind.IMPLEMENTATION.value,
    "methodology_reference": "Section 3.5.6 Rule I",
    "conjunction_note": (
        "IMPLEMENTATIONS_EQUIVALENT requires BOTH a TOST rejection at alpha AND a signed 95% "
        "CI that supports no directional claim. A TOST p-value alone is not sufficient, and "
        "a non-significant difference test is never evidence of equivalence."),
    "key_format": "<materiality>|<equivalence>",
    "map": {
        "NO_DIRECTIONAL_CLAIM|EQUIVALENT_WITHIN_MARGIN": {
            "conclusion": "IMPLEMENTATIONS_EQUIVALENT",
            "meaning": (
                "The two implementations produce statistically equivalent experimental "
                "outcomes within the preregistered margin, on this endpoint and this sealed "
                "test set. This is the condition methodology Section 3.5.6 Rule I states."),
        },
        "NO_DIRECTIONAL_CLAIM|INCONCLUSIVE": {
            "conclusion": "INCONCLUSIVE",
            "meaning": (
                "No directional difference is supported and equivalence is not established. "
                "The data are compatible with both; this is an absence of evidence, not "
                "evidence of agreement."),
        },
        "NO_DIRECTIONAL_CLAIM|NOT_EQUIVALENT": {
            "conclusion": "INCONCLUSIVE",
            "meaning": (
                "Equivalence rejected without a supported direction: the data are too "
                "dispersed to place the difference, which is an INCONCLUSIVE result rather "
                "than a finding."),
        },
        "MATERIALLY_HIGHER|NOT_EQUIVALENT": {
            "conclusion": "IMPLEMENTATIONS_DIFFER",
            "meaning": (
                "arm_a exceeds arm_b by more than the preregistered margin. The implementations "
                "do not produce equivalent outcomes; Section 3.5.6 Rule I is not satisfied and "
                "the discrepancy is itself the finding."),
        },
        "MATERIALLY_LOWER|NOT_EQUIVALENT": {
            "conclusion": "IMPLEMENTATIONS_DIFFER",
            "meaning": (
                "arm_b exceeds arm_a by more than the preregistered margin. As above, with the "
                "direction reversed."),
        },
        "INSUFFICIENT_BLOCKS|NOT_ASSESSED": {
            "conclusion": "INCONCLUSIVE",
            "meaning": (
                "Fewer than min_valid_blocks blocks survived. INCONCLUSIVE by the "
                "preregistered rule, decided by the block count and not by inspecting the "
                "data."),
        },
    },
    "unmapped_combination": {
        "conclusion": "INCONCLUSIVE",
        "meaning": (
            "A (materiality, equivalence) pair not present in the frozen map is reported as "
            "INCONCLUSIVE with the pair recorded verbatim. An unanticipated combination is "
            "never resolved by choosing the more favourable reading."),
    },
}


def apply_implementation_interpretation_rule(
    rule: dict, materiality: str, equivalence: str) -> dict[str, Any]:
    """Apply the frozen rule. Never invents a conclusion for an unmapped pair."""
    entry = rule["map"].get(f"{materiality}|{equivalence}")
    if entry is None:
        entry = dict(rule["unmapped_combination"])
        entry["unmapped_pair"] = [materiality, equivalence]
    return {
        "rule_version": rule["rule_version"],
        "materiality": materiality,
        "equivalence": equivalence,
        **entry,
    }


# =====================================================================
# plan
# =====================================================================

@dataclass
class ComparisonPlan:
    """A fully specified, hash-stable II-4 comparison execution plan."""
    experiment_id: str
    design: ComparisonDesign
    axis: ComparisonAxis
    fixed_protocol: dict[str, Any]
    seeds: ArmSeedManifest
    output_dir: Path
    shared_components: dict[str, str] = field(default_factory=dict)
    design_rationale: dict[str, Any] = field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION
    created_at_utc: str = field(default_factory=utc_timestamp)

    def preregistration(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at_utc": self.created_at_utc,
            "execution_mode": self.execution_mode.value,
            "design": self.design.to_dict(),
            "comparison_axis": self.axis.to_dict(),
            "fixed_protocol": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in self.fixed_protocol.items()},
            "seed_manifest": self.seeds.to_dict(),
            "shared_components": dict(self.shared_components),
            "estimand": ESTIMAND_STATEMENT,
            "conditional_inference": CONDITIONAL_INFERENCE_STATEMENT,
            "statistical_unit": STATISTICAL_UNIT_STATEMENT,
            "arm_structure_precondition": ARM_STRUCTURE_PRECONDITION,
            "replacement_block_rule": REPLACEMENT_BLOCK_RULE,
            "interpretation_rule": IMPLEMENTATION_INTERPRETATION_RULE,
            "confirmatory_test_set": {
                "scope": "GLOBAL_SHARED",
                "firewall_mode": "SHARED",
                "lifecycle": ("generated ONCE before block 0 -> persisted -> sha256 -> sealed "
                              "under every block id INCLUDING replacement block ids -> reused "
                              "for all blocks and both arms -> consumed exactly once per "
                              "(arm, block)"),
                "per_block_test_sets": False,
            },
            "confirmatory_endpoint": self.design.primary_endpoint,
            "secondary_descriptive": [
                "max_val_acc (historical comparability only; optimistically biased as a "
                "generalisation estimate and NOT the confirmatory endpoint)",
                "final_val_acc",
                "best_val_loss_checkpoint_hash (provenance only; does NOT determine the "
                "primary endpoint)",
            ],
            "design_rationale": dict(self.design_rationale),
        }

    def preregistration_hash(self) -> str:
        return sha256_bytes(
            dumps_strict(self.preregistration(), sort_keys=True, default=str).encode("utf-8"))

    def config_fingerprint(self) -> str:
        """
        Hash of everything defining the SCIENTIFIC experiment, excluding
        wall-clock creation time. A resumed run whose fingerprint differs
        from the recorded one is a different experiment and fails loudly.
        """
        body = {k: v for k, v in self.preregistration().items() if k != "created_at_utc"}
        return sha256_bytes(dumps_strict(body, sort_keys=True, default=str).encode("utf-8"))

    def validate(self) -> list[str]:
        """Pre-flight checks. Must be empty before execution is permitted."""
        problems: list[str] = []
        d = self.design
        if d.evaluation_rule != "terminal_epoch":
            problems.append(
                f"evaluation_rule is {d.evaluation_rule!r}; the frozen II-4 primary rule is "
                "'terminal_epoch'. A best-validation rule would make the endpoint depend on a "
                "selection over epochs.")
        if not d.primary_endpoint.startswith("sealed_"):
            problems.append(
                f"primary_endpoint is {d.primary_endpoint!r}; a validation-derived metric must "
                "not be the confirmatory endpoint.")
        if self.fixed_protocol.get("epochs") != d.epochs:
            problems.append("fixed_protocol epochs disagree with the design epochs.")
        if self.fixed_protocol.get("shuffle") is not True:
            problems.append("shuffle must be True to match the reference training behaviour.")
        if tuple(self.seeds.arm_ids) != tuple(self.axis.arm_ids):
            problems.append(
                f"seed manifest arms {self.seeds.arm_ids} disagree with axis arms "
                f"{self.axis.arm_ids}.")
        if self.seeds.n_blocks < d.max_total_blocks:
            problems.append(
                f"seed manifest covers {self.seeds.n_blocks} blocks but the design permits up "
                f"to {d.max_total_blocks} including replacements; replacement blocks would "
                "have no predetermined seeds.")
        problems.extend(self.seeds.problems())
        if self.axis.kind is AxisKind.IMPLEMENTATION and not self.shared_components:
            problems.append(
                "an IMPLEMENTATION comparison must declare shared_components: any component "
                "used identically by both arms is excluded from the comparison, and a reader "
                "must be told which, and on what evidential basis.")
        if "SMOKE" in self.experiment_id.upper() and \
                self.execution_mode is ExecutionMode.PRODUCTION:
            problems.append("a PRODUCTION experiment_id must not contain 'SMOKE'.")
        return problems

    def validate_production(self, *, frozen_hash: Optional[str] = None) -> list[str]:
        """
        Production-only checks: the plan must be exactly the frozen
        specification, so no smoke setting can leak into evidence.
        """
        problems = list(self.validate())
        if self.execution_mode is not ExecutionMode.PRODUCTION:
            problems.append("execution_mode is not PRODUCTION.")
        if not self.design.sign_flip_feasible_at_min_valid:
            problems.append(
                f"the exact sign-flip secondary cannot reach alpha={self.design.alpha} at "
                f"min_valid_blocks={self.design.min_valid_blocks} (minimum attainable "
                f"two-sided p = {2 / (2 ** self.design.min_valid_blocks):.4f}). A production "
                "design must not carry a secondary test that cannot resolve.")
        if frozen_hash is not None and self.config_fingerprint() != frozen_hash:
            problems.append(
                f"config fingerprint {self.config_fingerprint()} does not match the frozen "
                f"preregistration hash {frozen_hash}.")
        return problems
