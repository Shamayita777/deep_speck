from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from audit.cryptography.test.ce3.types import (
    TargetSpecification,
)

@dataclass(frozen=True, slots=True)
class InterventionTask:
    """
    One causal intervention task executed during CE4.

    A task couples an evaluation dataset with the
    cryptographic target whose causal necessity will
    be evaluated.

    Tasks marked as primary provide the principal
    scientific evidence for CE4, while auxiliary
    tasks may validate intervention behaviour.
    """

    dataset: Any
    """
    Evaluation dataset.
    """

    target: TargetSpecification
    """
    Cryptographic quantity whose causal necessity
    is being evaluated.
    """

    is_primary: bool = False

@dataclass(frozen=True, slots=True)
class InterventionEvaluation:
    """
    Intermediate result produced by CE4 causal
    intervention analysis.

    Records the paired intervention effects before
    statistical interpretation.
    """

    targeted_effect_mean: float
    """
    Mean output change produced by perturbing the
    theoretically relevant structure.
    """

    control_effect_mean: float
    """
    Mean output change produced by an equally-sized
    theoretically irrelevant perturbation.
    """

    necessity_gap_mean: float
    """
    Mean causal necessity gap.

    Defined as

        targeted_effect
        -
        control_effect
    """

    necessity_gap_values: list[float]
    """
    Per-sample paired necessity gaps.

    These are later passed directly to the shared
    paired significance procedure.
    """

    n_samples: int
    """
    Number of evaluated samples.
    """

    target: TargetSpecification
    """
    Cryptographic quantity evaluated.
    """

    metric_name: str