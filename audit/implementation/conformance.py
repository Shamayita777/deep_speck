"""
Generic conformance-impact machinery for Implementation Integrity II-4.

SCOPE OF GENERALITY
-------------------
This module contains NO knowledge of Gohr, Speck, neural networks, or
residual depth. II-4 asks a question that is the same for every audited
system:

    Given a CONFIRMED discrepancy between the DECLARED value of some
    implementation parameter P and the value actually REALIZED in the
    artifacts, does setting P to its declared value materially change
    the scientific outcome?

"Residual depth 10 vs 5" is one instantiation. For another audited
paper P might be an optimizer, a preprocessing step, a sampling rule, a
tokenizer, or a numerical precision. The abstractions here are written
so that a second audit supplies a new ConformanceFactor and a new
adapter, and reuses the design, analysis, provenance and firewall
machinery unchanged.

WHAT THE ADAPTER MUST SUPPLY
----------------------------
See ConformanceAdapter. Crucially it must include
`probe_realized_value`: the generic form of "read the architecture out
of the checkpoint and count residual merges". Every audit needs some
way to establish what an artifact ACTUALLY realizes, independently of
what its filename or documentation claims; II-4 cannot be trusted
without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

CONFORMANCE_SCHEMA_VERSION = "conformance-v1"


@dataclass(frozen=True)
class ConformanceFactor:
    """
    A confirmed declared-vs-realized discrepancy in one implementation
    parameter, together with how the realized value was established.

    `verification_method` is mandatory prose: an audit may not assert a
    discrepancy without saying how it was verified. `declaration_source`
    must point at the artifact that declares the intended value (e.g. a
    reference script), not at documentation about it.
    """

    name: str
    declared_value: Any
    realized_value: Any
    declaration_source: str
    verification_method: str
    finding_id: Optional[str] = None
    schema_version: str = CONFORMANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConformanceFactor requires a name.")
        if not self.verification_method or not self.verification_method.strip():
            raise ValueError(
                "ConformanceFactor requires a non-empty verification_method: an audit may "
                "not assert a declared-vs-realized discrepancy without recording how the "
                "realized value was established."
            )
        if not self.declaration_source or not self.declaration_source.strip():
            raise ValueError(
                "ConformanceFactor requires a declaration_source identifying the artifact "
                "that declares the intended value."
            )
        if self.declared_value == self.realized_value:
            raise ValueError(
                f"ConformanceFactor {self.name!r} has declared == realized "
                f"({self.declared_value!r}); there is no discrepancy to audit. II-4 exists to "
                "assess the impact of a CONFIRMED discrepancy."
            )

    @property
    def arm_values(self) -> dict[str, Any]:
        """The two experimental arms, named by role rather than by value."""
        return {"declared": self.declared_value, "realized": self.realized_value}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "declared_value": self.declared_value,
            "realized_value": self.realized_value,
            "declaration_source": self.declaration_source,
            "verification_method": self.verification_method,
            "finding_id": self.finding_id,
        }


@runtime_checkable
class ConformanceAdapter(Protocol):
    """
    Contract an audited system implements so the generic II-4 runner can
    execute against it. Implementations are case-study-specific; the
    runner is not. Every method below is called by the runner; nothing
    else about the audited system is assumed.

    Two probes are required, deliberately:
      probe_model            - realized value of an IN-MEMORY model
      probe_realized_value   - realized value read back from a SAVED
                               artifact on disk
    The runner checks both against the REQUESTED value. Reading the
    persisted artifact is the independent check: it is what establishes
    that the thing evaluated really is what the arm claims to be, rather
    than trusting the constructor argument.
    """

    def provenance_sources(self) -> dict:
        """Repository-relative paths of every case-specific file determining the run."""

    def environment_details(self) -> dict:
        """Framework/hardware details (e.g. accelerator, CUDA) as observed; None if unknown."""

    def generate_sealed_test_set(self, directory) -> tuple:
        """Generate + persist ONE confirmatory test set. Returns (path, hash, n)."""

    def load_sealed_test_set(self, path, expected_hash: str) -> Any:
        """Load the sealed set, failing if its hash differs from the record."""

    def generate_block_datasets(self, block_id: str, directory) -> dict:
        """Generate + persist one block's training/validation data. Returns records."""

    def load_block_datasets(self, records: dict) -> Any:
        """Reload a block's data, failing on any hash mismatch."""

    def build_model(self, factor_value: Any, *, seed: int) -> Any:
        """Construct a model with the conformance factor set to `factor_value`."""

    def probe_model(self, model: Any) -> Any:
        """Realized factor value of an in-memory model."""

    def probe_realized_value(self, artifact_path: str) -> Any:
        """Realized factor value read from a saved artifact, not from config."""

    def train(self, model: Any, block_data: Any, *, checkpoint_path: str) -> Any:
        """Train under the frozen protocol. Must never touch the sealed test set."""

    def save_terminal_model(self, model: Any, path) -> dict:
        """Persist the terminal-epoch model; report hash and reload identity."""

    def load_terminal_model(self, path: str, expected_hash: str) -> Any:
        """Reload a terminal model, failing on hash mismatch."""

    def evaluate_terminal(self, model: Any, sealed_test: Any) -> float:
        """Evaluate a terminal-epoch model on the sealed confirmatory set."""


@dataclass
class BlockOutcome:
    """
    One dataset block: both arms, plus validity.

    BLOCK-LEVEL INVALIDATION (frozen protocol): if either arm fails, the
    whole block is invalid for the paired analysis. Retaining the
    surviving arm would silently convert a paired observation into an
    unpaired one and break the blocking that the design depends on.
    """

    block_id: str
    dataset_hash: str
    declared_arm_value: Optional[float] = None
    realized_arm_value: Optional[float] = None
    declared_seed: Optional[int] = None
    realized_seed: Optional[int] = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    secondary: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return (
            self.declared_arm_value is not None
            and self.realized_arm_value is not None
            and not self.failures
        )

    @property
    def signed_difference(self) -> Optional[float]:
        """Delta_k = Y(declared) - Y(realized). Signed; direction is meaningful."""
        if not self.is_valid:
            return None
        return self.declared_arm_value - self.realized_arm_value

    def record_failure(self, *, arm: str, reason: str, detail: str, seed: Optional[int]) -> None:
        self.failures.append({"arm": arm, "reason": reason, "detail": detail, "seed": seed})

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "dataset_hash": self.dataset_hash,
            "declared_arm_value": self.declared_arm_value,
            "realized_arm_value": self.realized_arm_value,
            "declared_seed": self.declared_seed,
            "realized_seed": self.realized_seed,
            "signed_difference": self.signed_difference,
            "is_valid": self.is_valid,
            "failures": list(self.failures),
            "secondary": dict(self.secondary),
        }
