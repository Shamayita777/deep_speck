#!/usr/bin/env python3
"""
Generic comparison machinery for Implementation Integrity Stage II-4.

NO knowledge of Gohr, Speck, Keras, TensorFlow, neural networks or
residual depth appears in this module. It is the abstraction that lets
one II-4 engine serve both questions the methodology distinguishes.

WHY THIS MODULE EXISTS
----------------------
Methodology Section 3.5.4 Stage 4 specifies that "both implementations
are evaluated using identical experimental protocols", and Section 3.5.3
states the null hypothesis over "independent implementations". The
project's first II-4 experiment did something different and useful: it
varied one declared-vs-realized parameter WITHIN a single
implementation. Both are paired-block comparisons with the same design,
the same analysis and the same firewall; they differ only in WHAT
DISTINGUISHES THE TWO ARMS.

That difference is named here as a COMPARISON AXIS, and it is given
consequences:

    AxisKind.CONFORMANCE_PARAMETER
        Arms differ by the value of one implementation parameter.
        One implementation, two configurations.
        Answers: "does this discrepancy change the outcome?"
        -> ComponentKind.CONFORMANCE_IMPACT_EXPERIMENT
        -> CANNOT satisfy Stage II-4.

    AxisKind.IMPLEMENTATION
        Arms differ by which independently written implementation
        builds and trains the model. Two implementations, one protocol.
        Answers Section 3.5.3's H0 directly.
        -> ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON
        -> CAN satisfy Stage II-4.

The axis therefore carries the stage-satisfaction rule as DATA rather
than leaving it to a runner, a reviewer or a docstring. A third audit
adding a new axis kind must state which ComponentKind it produces, and
`stages.validate_stage_record` then enforces the consequence
automatically.

WHAT THIS BUYS A SECOND CASE STUDY
----------------------------------
A new target supplies a ComparisonAxis, two ImplementationArms and an
adapter. Design, seed policy, block protocol, firewall discipline,
analysis, interpretation rule, provenance, resume and certificate
assembly are reused unchanged. Nothing in this module needs editing to
audit a different cipher, a different architecture or a different
framework.

ARM NAMING. The first II-4 experiment hardcoded the arm names
"declared" and "realized" into its seed manifest, block outcome and
runner. Those names are meaningless for an implementation comparison.
Arms here are identified by caller-supplied ids, and every structure
below is keyed by those ids, so the axis names its own arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

COMPARISON_SCHEMA_VERSION = "ii4-comparison-v1"


class AxisKind(str, Enum):
    """What distinguishes the two arms of a II-4 comparison."""
    CONFORMANCE_PARAMETER = "CONFORMANCE_PARAMETER"
    IMPLEMENTATION = "IMPLEMENTATION"


#: Axis kind -> the stages.ComponentKind a run along that axis produces.
#: Held here, beside the axis definition, so that the mapping is part of
#: the specification rather than a decision made at certificate time.
AXIS_COMPONENT_KIND: dict[AxisKind, str] = {
    AxisKind.CONFORMANCE_PARAMETER: "CONFORMANCE_IMPACT_EXPERIMENT",
    AxisKind.IMPLEMENTATION: "FORMAL_IMPLEMENTATION_COMPARISON",
}

#: Axis kinds whose completed run can satisfy methodology Stage II-4.
#: Only a comparison of two implementations answers Section 3.5.3's H0.
AXIS_SATISFIES_STAGE_II4: frozenset[AxisKind] = frozenset({AxisKind.IMPLEMENTATION})


@dataclass(frozen=True)
class ComparisonArm:
    """
    One arm of a II-4 comparison.

    `identity` is what makes this arm the arm it is, in whatever terms
    the axis uses: a parameter value for a conformance axis, or the
    provenance of an implementation for an implementation axis. It is
    recorded in the preregistration and re-checked against the
    constructed model and the persisted artifact, so a run cannot
    silently measure something other than what it declared.

    `source_paths` must be non-empty for an implementation arm: an
    implementation that cannot be pinned by file hash cannot be the
    subject of a reproducible comparison.
    """
    arm_id: str
    label: str
    identity: Any
    source_paths: tuple[str, ...] = ()
    provenance_note: str = ""

    def __post_init__(self) -> None:
        if not self.arm_id or not self.arm_id.strip():
            raise ValueError("ComparisonArm requires a non-empty arm_id.")
        if not self.label or not self.label.strip():
            raise ValueError(f"arm {self.arm_id!r} requires a human-readable label.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id, "label": self.label, "identity": self.identity,
            "source_paths": list(self.source_paths),
            "provenance_note": self.provenance_note,
        }


@dataclass(frozen=True)
class ComparisonAxis:
    """
    The axis along which two arms differ, plus the evidential
    consequence of that choice.

    `sign_convention` is mandatory and is stated as "<arm_a> minus
    <arm_b>" so that the direction of every reported effect is
    unambiguous in the artifact itself, not inferred from code.
    """
    kind: AxisKind
    name: str
    arm_a: ComparisonArm
    arm_b: ComparisonArm
    scientific_question: str
    verification_method: str
    finding_id: Optional[str] = None
    schema_version: str = COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.arm_a.arm_id == self.arm_b.arm_id:
            raise ValueError("A comparison needs two distinct arm ids.")
        if not self.scientific_question.strip():
            raise ValueError(
                f"axis {self.name!r} requires scientific_question: an experiment that cannot "
                "state the question it answers cannot have its answer interpreted.")
        if not self.verification_method.strip():
            raise ValueError(
                f"axis {self.name!r} requires verification_method: how the arms were "
                "established to be what they claim to be.")
        if self.kind is AxisKind.CONFORMANCE_PARAMETER:
            if self.arm_a.identity == self.arm_b.identity:
                raise ValueError(
                    f"conformance axis {self.name!r} has equal arm identities "
                    f"({self.arm_a.identity!r}); there is no discrepancy to assess.")
        if self.kind is AxisKind.IMPLEMENTATION:
            for arm in (self.arm_a, self.arm_b):
                if not arm.source_paths:
                    raise ValueError(
                        f"implementation arm {arm.arm_id!r} declares no source_paths. An "
                        "implementation comparison must pin both implementations by file "
                        "hash; an unpinned arm cannot be reproduced or audited.")
            if set(self.arm_a.source_paths) & set(self.arm_b.source_paths):
                shared = sorted(set(self.arm_a.source_paths) & set(self.arm_b.source_paths))
                raise ValueError(
                    f"implementation arms share source file(s) {shared}. Arms that execute the "
                    "same code are not independent implementations, and comparing them would "
                    "answer no question. If a file is genuinely shared infrastructure (for "
                    "example a shared data generator), it belongs in the plan's shared_"
                    "components, not in an arm's source_paths.")

    @property
    def arm_ids(self) -> tuple[str, str]:
        return (self.arm_a.arm_id, self.arm_b.arm_id)

    @property
    def arms(self) -> dict[str, ComparisonArm]:
        return {self.arm_a.arm_id: self.arm_a, self.arm_b.arm_id: self.arm_b}

    @property
    def arm_identities(self) -> dict[str, Any]:
        return {self.arm_a.arm_id: self.arm_a.identity, self.arm_b.arm_id: self.arm_b.identity}

    @property
    def sign_convention(self) -> str:
        return f"{self.arm_a.arm_id} minus {self.arm_b.arm_id}"

    @property
    def component_kind(self) -> str:
        return AXIS_COMPONENT_KIND[self.kind]

    @property
    def satisfies_stage_ii4(self) -> bool:
        """
        Whether a COMPLETED run along this axis can satisfy methodology
        Stage II-4. False for a conformance axis: it answers a different
        question, however well it is executed.
        """
        return self.kind in AXIS_SATISFIES_STAGE_II4

    def stage_consequence(self) -> str:
        if self.satisfies_stage_ii4:
            return (
                "A completed run along this axis compares two independently written "
                "implementations under identical experimental protocols and therefore addresses "
                "methodology Section 3.5.3's null hypothesis directly. It can satisfy Stage "
                "II-4 and supply the statistical assessment for Stage II-5.")
        return (
            "A completed run along this axis varies one parameter within a single "
            "implementation. It answers whether that discrepancy changes the outcome, NOT "
            "whether two implementations agree, so it can never satisfy Stage II-4 or II-5 "
            "however many blocks it completes (enforced by stages.validate_stage_record).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "name": self.name,
            "arm_a": self.arm_a.to_dict(),
            "arm_b": self.arm_b.to_dict(),
            "arm_ids": list(self.arm_ids),
            "sign_convention": self.sign_convention,
            "scientific_question": self.scientific_question,
            "verification_method": self.verification_method,
            "finding_id": self.finding_id,
            "component_kind": self.component_kind,
            "satisfies_stage_ii4": self.satisfies_stage_ii4,
            "stage_consequence": self.stage_consequence(),
        }


def conformance_axis_from_factor(factor, *, scientific_question: str = "") -> ComparisonAxis:
    """
    Bridge: express an existing `conformance.ConformanceFactor` as a
    ComparisonAxis.

    This exists to DEMONSTRATE the generality rather than assert it. The
    project's first II-4 experiment is already frozen and must never be
    re-run, so the only honest way to show that the generic axis
    subsumes it is to reconstruct its arm semantics from its own frozen
    factor and check that they agree - which the test suite does. The
    arm ids deliberately keep that experiment's original names so the
    reconstruction is comparable to its recorded artifacts.
    """
    return ComparisonAxis(
        kind=AxisKind.CONFORMANCE_PARAMETER,
        name=factor.name,
        arm_a=ComparisonArm(
            arm_id="declared", label=f"{factor.name} = {factor.declared_value!r} (declared)",
            identity=factor.declared_value,
            provenance_note=f"declared by {factor.declaration_source}"),
        arm_b=ComparisonArm(
            arm_id="realized", label=f"{factor.name} = {factor.realized_value!r} (realized)",
            identity=factor.realized_value,
            provenance_note="realized value read from the produced artifact"),
        scientific_question=scientific_question or (
            f"Does setting {factor.name} to its declared value {factor.declared_value!r} "
            f"instead of the realized {factor.realized_value!r} change the primary endpoint?"),
        verification_method=factor.verification_method,
        finding_id=getattr(factor, "finding_id", None),
    )


# =====================================================================
# seeds
# =====================================================================

@dataclass(frozen=True)
class ArmSeedManifest:
    """
    Predetermined, recorded training seeds: one per (block, arm).

    NEVER one seed shared by both arms. For a conformance axis the arms
    have different parameter shapes, so a shared integer yields
    non-corresponding weights. For an implementation axis the arms are
    different code, so the same integer drives different numbers of RNG
    draws in a different order and again yields non-corresponding
    weights. In both cases a shared seed would create the appearance of
    paired initialization without the fact of it, which is why the
    design declares same_model_initialization=False and this manifest
    makes that declaration structurally true.

    Seeds are derived arithmetically from a frozen base so that the
    whole manifest is reproducible from one integer, and are checked for
    pairwise distinctness.
    """
    base_seed: int
    n_blocks: int
    arm_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.arm_ids)) != len(self.arm_ids):
            raise ValueError(f"arm_ids must be distinct, got {self.arm_ids}.")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be positive.")

    def seed_for(self, block_index: int, arm_id: str) -> int:
        if arm_id not in self.arm_ids:
            raise ValueError(f"unknown arm {arm_id!r}; expected one of {self.arm_ids}.")
        if not 0 <= block_index < self.n_blocks:
            raise ValueError(f"block_index {block_index} outside 0..{self.n_blocks - 1}.")
        return self.base_seed + len(self.arm_ids) * block_index + self.arm_ids.index(arm_id)

    def all_seeds(self) -> dict[str, dict[str, int]]:
        return {
            f"block{i}": {a: self.seed_for(i, a) for a in self.arm_ids}
            for i in range(self.n_blocks)
        }

    def problems(self) -> list[str]:
        seen: dict[int, str] = {}
        out: list[str] = []
        for i in range(self.n_blocks):
            for arm in self.arm_ids:
                s = self.seed_for(i, arm)
                where = f"block{i}/{arm}"
                if s in seen:
                    out.append(f"duplicate seed {s}: {seen[s]} and {where}.")
                seen[s] = where
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed, "n_blocks": self.n_blocks,
            "arm_ids": list(self.arm_ids),
            "policy": ("one distinct seed per (block, arm); the arms are NEVER given the same "
                       "seed - see design.same_model_initialization=False"),
            "seeds": self.all_seeds(),
        }


# =====================================================================
# block outcome
# =====================================================================

@dataclass
class ComparisonBlockOutcome:
    """
    One dataset block: an endpoint value per arm, plus validity.

    BLOCK-LEVEL INVALIDATION (frozen): if ANY arm fails, the whole block
    is invalid for the paired analysis. Keeping the surviving arm would
    silently convert a paired observation into an unpaired one and
    destroy the blocking the design depends on. This mirrors the rule
    already frozen for the conformance experiment, generalized to
    arbitrary arm ids.
    """
    block_id: str
    arm_ids: tuple[str, str]
    dataset_hash: str
    values: dict[str, float] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    secondary: dict[str, Any] = field(default_factory=dict)
    replaces_block_id: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return (not self.failures
                and all(self.values.get(a) is not None for a in self.arm_ids))

    @property
    def signed_difference(self) -> Optional[float]:
        """Delta_k = Y(arm_a) - Y(arm_b), matching axis.sign_convention."""
        if not self.is_valid:
            return None
        a, b = self.arm_ids
        return float(self.values[a]) - float(self.values[b])

    def record_failure(self, *, arm: str, reason: str, detail: str,
                       seed: Optional[int] = None) -> None:
        self.failures.append({"arm": arm, "reason": reason, "detail": detail, "seed": seed})

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "arm_ids": list(self.arm_ids),
            "dataset_hash": self.dataset_hash,
            "values": dict(self.values),
            "seeds": dict(self.seeds),
            "signed_difference": self.signed_difference,
            "is_valid": self.is_valid,
            "failures": list(self.failures),
            "secondary": dict(self.secondary),
            "replaces_block_id": self.replaces_block_id,
        }


def collect_valid_differences(blocks: list[ComparisonBlockOutcome]) -> list[float]:
    """
    Signed differences of the valid blocks, in block order.

    Invalid blocks are DROPPED but never deleted: the runner retains
    every block record, including failed ones, so the analysis is run on
    what survived while the artifact still shows what did not. Silently
    omitting failures from the record is how an attrition-biased result
    is made to look clean.
    """
    return [b.signed_difference for b in blocks if b.is_valid]


# =====================================================================
# adapter contract
# =====================================================================

@runtime_checkable
class ComparisonAdapter(Protocol):
    """
    What a case study must supply. Everything else is reused.

    The arm-aware methods are the generalization over the conformance
    adapter: `build_arm_model` and `train_arm` both receive the arm id,
    so an implementation axis can dispatch to a DIFFERENT code path per
    arm, rather than one code path parameterized by a value.

    SHARED COMPONENTS. `generate_block_datasets` is deliberately NOT
    arm-aware: both arms of a block train on the same data, which is
    what makes the design paired. For an implementation comparison this
    means data generation is a SHARED component and therefore not a
    compared one; the adapter must say which implementation's generator
    is used and on what basis, via `shared_components`.
    """

    def provenance_sources(self) -> dict[str, str]: ...
    def environment_details(self) -> dict[str, Any]: ...

    def shared_components(self) -> dict[str, str]:
        """
        Components used identically by both arms, and therefore excluded
        from the comparison. Must be non-empty for an implementation
        axis if any component is shared, and each entry must state the
        evidential basis for treating it as equivalent across arms.
        """

    def generate_sealed_test_set(self, directory) -> tuple: ...
    def load_sealed_test_set(self, path: str, expected_hash: str) -> Any: ...
    def generate_block_datasets(self, block_id: str, directory) -> dict: ...
    def load_block_datasets(self, records: dict) -> Any: ...

    def build_arm_model(self, arm_id: str, *, seed: int) -> Any: ...
    def probe_arm_model(self, arm_id: str, model: Any) -> Any:
        """
        Structural facts about the constructed model, compared by the
        runner against the arm's expected structure. For an
        implementation comparison the two arms must realize the SAME
        structure - that is the premise of comparing them - so this is
        where a divergence is caught before 9 GPU-hours are spent.
        """

    def expected_arm_structure(self, arm_id: str) -> Any:
        """The structure `probe_arm_model` must return for this arm."""

    def train_arm(self, arm_id: str, model: Any, block_data: Any, *,
                  checkpoint_path: str, seed: int) -> Any: ...
    def save_terminal_model(self, arm_id: str, model: Any, path) -> dict: ...
    def load_terminal_model(self, arm_id: str, path: str, expected_hash: str) -> Any: ...
    def evaluate_terminal(self, arm_id: str, model: Any, sealed_test: Any) -> float: ...
