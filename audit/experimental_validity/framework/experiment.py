"""
Generic experiment specification for Experimental Validity.

Contains no Gohr-specific assumptions. A concrete experiment (e.g.
H-EV-SHUFFLE) is an instance of `Experiment` constructed in
gohr.experiments; this module only defines the reusable shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DesignType(str, Enum):
    DESCRIPTIVE = "descriptive"          # e.g. EV-BASELINE, EV-NOISE
    MATCHED_PAIRED = "matched_paired"    # e.g. H-EV-SHUFFLE, H-EV-REPRESENTATION
    INDEPENDENT_SAMPLES = "independent_samples"  # reserved; not used by the frozen core


class RunMode(str, Enum):
    SMOKE = "smoke"
    PRODUCTION = "production"


@dataclass(frozen=True)
class Factor:
    """A single controllable experimental factor and its role."""
    name: str
    essential: bool  # True: changing it changes the cryptographic task itself
    rationale: str


@dataclass(frozen=True)
class Condition:
    """
    One arm of an experiment (e.g. "shuffle=True" or "shuffle=False").
    `overrides` holds only the parameters that differ from the frozen
    baseline configuration; everything else is inherited from the
    baseline, making deviations explicit and diffable.
    """
    condition_id: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairingDeclaration:
    """
    Explicit statement of exactly what is matched between the arms of a
    paired experiment. Per the frozen randomness rules, a shared
    model_seed does not by itself guarantee an identical shuffle stream
    if a condition changes RNG consumption upstream - so each of these
    is declared independently rather than inferred from "same seed".
    """
    same_dataset: bool
    same_model_initialization: bool
    same_training_shuffle_stream: bool
    same_evaluation_data: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "same_dataset": self.same_dataset,
            "same_model_initialization": self.same_model_initialization,
            "same_training_shuffle_stream": self.same_training_shuffle_stream,
            "same_evaluation_data": self.same_evaluation_data,
        }

    def describe_unit(self) -> str:
        """
        Render the replication unit FROM THE DECLARED FIELDS rather than
        from a hard-coded sentence.

        Previously the Gohr orchestration hard-coded "matched pair of
        independently trained models (same dataset, same model seed)"
        into every paired certificate. That sentence is true of the EV
        shuffle/representation experiments but FALSE of a conformance
        comparison such as II-4, where the two arms have different
        parameter shapes and initialization is deliberately independent.
        Deriving the prose guarantees the certificate narrative cannot
        contradict the declaration it sits beside.
        """
        shared, independent = [], []
        for label, value in (
            ("dataset", self.same_dataset),
            ("model initialization", self.same_model_initialization),
            ("training shuffle stream", self.same_training_shuffle_stream),
            ("evaluation data", self.same_evaluation_data),
        ):
            (shared if value else independent).append(label)
        parts = []
        if shared:
            parts.append("shared " + ", ".join(shared))
        if independent:
            parts.append("independent " + ", ".join(independent))
        return "matched pair of independently trained models (" + "; ".join(parts) + ")"


@dataclass(frozen=True)
class PracticalSignificance:
    """
    Explicit, optional practical-significance threshold (epsilon).

    threshold=None means "no predeclared practical-significance
    threshold exists". The framework must never fabricate one; the
    decision layer (framework.certificate) is required to report
    formal practical-equivalence certification as NOT_AVAILABLE
    whenever threshold is None, regardless of the p-value observed.
    """
    threshold: Optional[float]
    predeclared: bool
    justification: Optional[str] = None

    def is_available(self) -> bool:
        return self.threshold is not None and self.predeclared


@dataclass(frozen=True)
class ReplicatePlan:
    requested_replicates: int
    minimum_valid_replicates: int


@dataclass(frozen=True)
class Experiment:
    """Generic, reusable experiment specification."""
    id: str
    version: str
    hypothesis_id: Optional[str]
    scientific_claim: str
    hypothesized_confounding_mechanism: Optional[str]
    rationale: str
    factors: list[Factor]
    conditions: list[Condition]
    controls: list[str]
    independent_variable: Optional[str]
    dependent_variable: str
    unit_of_replication: str
    design_type: DesignType
    replicate_plan: ReplicatePlan
    pairing: Optional[PairingDeclaration]
    practical_significance: PracticalSignificance
    multiplicity_family: Optional[str]
    expected_observation_if_alternative: Optional[str] = None
    expected_observation_if_null: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "hypothesis_id": self.hypothesis_id,
            "scientific_claim": self.scientific_claim,
            "hypothesized_confounding_mechanism": self.hypothesized_confounding_mechanism,
            "rationale": self.rationale,
            "factors": [f.__dict__ for f in self.factors],
            "conditions": [{"condition_id": c.condition_id, "description": c.description, "overrides": c.overrides} for c in self.conditions],
            "controls": self.controls,
            "independent_variable": self.independent_variable,
            "dependent_variable": self.dependent_variable,
            "unit_of_replication": self.unit_of_replication,
            "design_type": self.design_type.value,
            "replicate_plan": self.replicate_plan.__dict__,
            "pairing": self.pairing.to_dict() if self.pairing else None,
            "practical_significance": {
                "threshold": self.practical_significance.threshold,
                "predeclared": self.practical_significance.predeclared,
                "justification": self.practical_significance.justification,
            },
            "multiplicity_family": self.multiplicity_family,
            "expected_observation_if_alternative": self.expected_observation_if_alternative,
            "expected_observation_if_null": self.expected_observation_if_null,
        }
