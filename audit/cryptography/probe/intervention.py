"""
Cryptographic Evidence CE4 Causal Intervention Evaluation
============================================================

Framework-level implementation of the CE4 evidence pipeline:
paired targeted-vs-control intervention effects in, causal
necessity and its statistical significance out.

Unlike CE3 (probe.evaluation), nothing is fit here. CE4 measures
a direct paired output difference per evaluation sample, under
two interventions applied to the same frozen model -- there is
no training, and therefore no train/test split or cross-
validation fold structure. The paired unit for CE4 is the
per-sample necessity gap, not a per-fold score difference.

The statistical procedure (Wilcoxon signed-rank, Cohen's d_z,
bootstrap CI) is shared with CE3 via
`audit.cryptography.probe.significance.paired_significance`,
not reimplemented here.
"""

from __future__ import annotations

import numpy as np

from audit.cryptography.test.ce3.types import (
    PairedComparisonStatistic,
    TargetSpecification,
)
from audit.cryptography.test.ce4.types import InterventionEvaluation

from .significance import paired_significance


def _validate_magnitude_match(
    structural_magnitude: np.ndarray,
    control_magnitude: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> None:
    """
    Enforce that structural and control interventions perturbed
    each sample by the same magnitude (Framework Principle 2:
    only WHICH positions are perturbed should differ between
    arms, not HOW MUCH). Magnitude is adapter-defined (see
    CryptographicAdapter.compute_intervention_magnitude); this
    function only enforces equality generically.
    """

    if structural_magnitude.shape != control_magnitude.shape:
        raise ValueError(
            "Structural and control intervention magnitude "
            "arrays must have matching shape: got "
            f"{structural_magnitude.shape} and "
            f"{control_magnitude.shape}."
        )

    mismatch = np.abs(structural_magnitude - control_magnitude) > tolerance

    if np.any(mismatch):
        n_mismatched = int(np.sum(mismatch))
        raise ValueError(
            "Structural and control interventions are not "
            f"magnitude-matched for {n_mismatched} of "
            f"{structural_magnitude.shape[0]} samples "
            f"(tolerance={tolerance}). A control intervention "
            "must perturb each sample by the same magnitude as "
            "the structural intervention, differing only in "
            "which positions are perturbed."
        )


def evaluate_necessity(
    original_predictions: np.ndarray,
    structural_predictions: np.ndarray,
    control_predictions: np.ndarray,
    structural_magnitude: np.ndarray,
    control_magnitude: np.ndarray,
    target: TargetSpecification,
    *,
    magnitude_tolerance: float = 0.0,
    metric_name: str = "Absolute Output Probability Difference",
) -> InterventionEvaluation:
    """
    Compute CE4's causal necessity evidence for one
    InterventionTask.

    Parameters
    ----------
    original_predictions, structural_predictions,
    control_predictions
        Model outputs on the unperturbed, structurally-
        perturbed, and control-perturbed datasets respectively.
        All three must be index-aligned to the same samples.

    structural_magnitude, control_magnitude
        Per-sample perturbation magnitude for each intervention
        (see CryptographicAdapter.compute_intervention_magnitude).
        Used only to enforce the magnitude-match constraint --
        not part of the returned evidence.

    Returns
    -------
    InterventionEvaluation
        Raw per-sample measurements only. Statistical
        significance is computed separately via
        `compute_significance`, matching the same evidence /
        interpretation separation used in CE3.
    """

    if not (
        original_predictions.shape
        == structural_predictions.shape
        == control_predictions.shape
    ):
        raise ValueError(
            "original_predictions, structural_predictions, and "
            "control_predictions must have identical shape: got "
            f"{original_predictions.shape}, "
            f"{structural_predictions.shape}, "
            f"{control_predictions.shape}."
        )

    _validate_magnitude_match(
        structural_magnitude, control_magnitude,
        tolerance=magnitude_tolerance,
    )

    targeted_effect = np.abs(original_predictions - structural_predictions)
    control_effect = np.abs(original_predictions - control_predictions)
    necessity_gap = targeted_effect - control_effect

    return InterventionEvaluation(
        targeted_effect_mean=float(np.mean(targeted_effect)),
        control_effect_mean=float(np.mean(control_effect)),
        necessity_gap_mean=float(np.mean(necessity_gap)),
        necessity_gap_values=necessity_gap.tolist(),
        n_samples=int(necessity_gap.shape[0]),
        target=target,
        metric_name=metric_name,
    )


def compute_significance(
    evaluation: InterventionEvaluation,
    *,
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedComparisonStatistic:
    """
    Test whether mean necessity gap is significantly greater
    than zero. Delegates to the same paired-significance
    procedure CE3 uses; only the pairing unit differs.
    """

    diffs = np.asarray(evaluation.necessity_gap_values, dtype=np.float64)

    return paired_significance(
        diffs, alternative=alternative, n_bootstrap=n_bootstrap, seed=seed,
    )