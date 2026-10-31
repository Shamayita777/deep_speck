"""
Capability-specific adapter interfaces.

The legacy ``adapters.base.CryptographicAdapter`` requires every adapter to implement the CE1,
CE2, CE3 and CE4 surfaces at once, so an adapter for one experiment must stub out the others.
These protocols split that surface by capability. An adapter implements only what its experiment
measures; a measurement checks the capability it needs with ``require``.

They are structural (``typing.Protocol``): no inheritance is needed, and an existing adapter that
already has the methods satisfies them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SignalDestructionCapability(Protocol):
    def generate_baseline_dataset(self) -> Any: ...
    def generate_signal_destroyed_dataset(self) -> Any: ...
    def train(self, dataset: Any, *args, **kwargs) -> Any: ...
    def evaluate(self, model: Any, dataset: Any) -> Any: ...


@runtime_checkable
class TheoryConsistencyCapability(Protocol):
    """A frozen model, one immutable dataset, a model-free reference, and model predictions."""
    def load(self) -> Any: ...
    def generate_theory_dataset(self) -> Any: ...
    def compute_theoretical_reference(self, dataset: Any) -> Any: ...
    def compute_model_predictions(self, model: Any, dataset: Any) -> Any: ...


@runtime_checkable
class RepresentationCapability(Protocol):
    def generate_representation_tasks(self) -> Any: ...
    def extract_representations(self, model: Any, dataset: Any) -> Any: ...


@runtime_checkable
class InterventionCapability(Protocol):
    def generate_intervention_tasks(self) -> Any: ...
    def apply_structural_intervention(self, *args, **kwargs) -> Any: ...
    def apply_control_intervention(self, *args, **kwargs) -> Any: ...
    def compute_intervention_magnitude(self, *args, **kwargs) -> Any: ...


class CapabilityError(TypeError):
    pass


def require(adapter: Any, capability: type) -> None:
    if not isinstance(adapter, capability):
        missing = [m for m in capability.__dict__ if not m.startswith("_") and callable(getattr(capability, m, None))
                   and not hasattr(adapter, m)]
        raise CapabilityError(f"{type(adapter).__name__} lacks {capability.__name__}: missing {missing}")
