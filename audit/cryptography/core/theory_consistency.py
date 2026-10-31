"""
Theory-consistency measurement with an explicit hypothesis direction.

Replaces, for new experiments, the legacy ``test.ce2.theory_consistency.TheoryConsistencyTest``,
which (i) always reports ``HIGHER_IS_BETTER`` / a positive hypothesis and (ii) converts any
``ValueError`` from the correlation into ``test_score = 0.0``. Here:

* the hypothesis direction is a constructor argument (TWO_SIDED, POSITIVE, NEGATIVE) and is
  recorded in the result;
* an undefined coefficient is recorded with ``AssociationStatus`` and ``statistic = None``,
  never as 0.0;
* the within-sample p-value is a ``numerics.PValue`` (underflow and degeneracy are explicit) and
  is descriptive only.

The legacy class is kept byte-identical because ``PACKAGE_MANIFEST.sha256`` binds it; no
production route uses it (see ``core/registry.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .capabilities import TheoryConsistencyCapability, require
from .hypothesis import AssociationStatus, HypothesisDirection
from .numerics import PValue, correlation_t_approximation


class UndefinedAssociation(Exception):
    """Raised by a coefficient function when the coefficient does not exist. Carries the status."""

    def __init__(self, status: AssociationStatus, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class AssociationResult:
    test_name: str
    hypothesis: HypothesisDirection
    status: AssociationStatus
    statistic: float | None
    n: int
    within_sample_p: PValue
    reason: str = ""
    runtime_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def defined(self) -> bool:
        return self.status is AssociationStatus.DEFINED

    def to_dict(self) -> dict:
        return {"test_name": self.test_name, "hypothesis": self.hypothesis.value,
                "status": self.status.value, "statistic": self.statistic, "n": self.n,
                "within_sample_p": self.within_sample_p.to_dict(),
                "within_sample_p_role": "descriptive only, never confirmatory",
                "reason": self.reason, "runtime_s": self.runtime_s, "metadata": self.metadata}


class TheoryConsistencyMeasurement:
    """One association measurement between a frozen model and a model-free reference."""

    name = "Theory Consistency (core, direction-explicit)"

    def __init__(self, hypothesis: HypothesisDirection,
                 coefficient: Callable[[np.ndarray, np.ndarray], float]):
        if not isinstance(hypothesis, HypothesisDirection):
            raise TypeError("hypothesis must be a HypothesisDirection")
        self.hypothesis = hypothesis
        self.coefficient = coefficient

    def run(self, adapter: Any) -> AssociationResult:
        require(adapter, TheoryConsistencyCapability)
        start = time.perf_counter()
        model = adapter.load()
        dataset = adapter.generate_theory_dataset()
        reference = np.asarray(adapter.compute_theoretical_reference(dataset))
        predictions = np.asarray(adapter.compute_model_predictions(model, dataset))
        n = int(predictions.size)
        try:
            stat = float(self.coefficient(predictions, reference))
        except UndefinedAssociation as exc:
            return AssociationResult(self.name, self.hypothesis, exc.status, None, n,
                                     PValue("UNDEFINED", None, None, "none", exc.reason), exc.reason,
                                     time.perf_counter() - start)
        if not np.isfinite(stat):
            return AssociationResult(self.name, self.hypothesis, AssociationStatus.UNDEFINED, None, n,
                                     PValue("UNDEFINED", None, None, "none", "non-finite coefficient"),
                                     "non-finite coefficient", time.perf_counter() - start)
        return AssociationResult(self.name, self.hypothesis, AssociationStatus.DEFINED, stat, n,
                                 correlation_t_approximation(stat, n), "", time.perf_counter() - start)
