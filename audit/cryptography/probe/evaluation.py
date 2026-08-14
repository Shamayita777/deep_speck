"""
Cryptographic Evidence CE3 Probing Evaluation
==============================================

Framework-level implementation of the CE3 evidence pipeline:
representations in, selectivity and its statistical significance
out.

This module owns everything the framework was agreed to own:

    • fold generation (repeated stratified k-fold / repeated
      k-fold, selected by target type)
    • the random seed shared between the real and control arms
      of every fold
    • the scoring metric (selected by target type)
    • the default probe algorithm (selected by target type)
    • the statistical test used to assess selectivity

Probes themselves (see probe.base.CryptographicProbe) know only
`fit` and `predict`. Nothing in this module assumes anything
about cryptographic primitives, neural architectures, or which
paper produced the representations under evaluation.

Multiple-comparisons correction across several declared targets
and/or representation layers is intentionally NOT performed
here. This module evaluates one target against one pair of
representation matrices and returns one p-value; correction
across the full set of targets/layers evaluated in a single CE3
audit run is the responsibility of the CE3 test orchestrator,
which is the only component that sees the complete set of
p-values being generated.
"""

from __future__ import annotations
from typing import Callable

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from ..test.ce3.types import (
    PairedComparisonStatistic,
    ProbeEvaluation,
    ReplicatedProbeEvaluation,
    TargetSpecification,
    TargetType,
)
from .base import CryptographicProbe
from .significance import paired_significance

# ============================================================
# Default Probes (Q1)
# ============================================================


class LogisticRegressionProbe(CryptographicProbe):
    """
    Default CE3 probe for BINARY and MULTICLASS targets.

    Wraps scikit-learn's LogisticRegression with a fixed
    regularization strength, identical across the real and
    control arms of every fold, so that hyperparameter choice
    cannot itself become an uncontrolled variable in the
    selectivity comparison (Framework Principle 2).
    """

    def __init__(
        self, *, random_state: int | None = None, C: float = 1.0,
    ) -> None:
        self._C = C
        self._random_state = random_state
        self._model = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=random_state,
        )

    def fit(
        self, train_representations: np.ndarray, train_labels: np.ndarray,
    ) -> None:
        self._model.fit(train_representations, train_labels)

    def predict(self, test_representations: np.ndarray) -> np.ndarray:
        return self._model.predict(test_representations)

    @property
    def name(self) -> str:
        return "Logistic Regression"

    @property
    def description(self) -> str:
        return (
            f"scikit-learn LogisticRegression (C={self._C}, "
            "max_iter=1000), the CE3 framework default probe for "
            "binary and multiclass targets."
        )


class RidgeRegressionProbe(CryptographicProbe):
    """
    Default CE3 probe for CONTINUOUS targets.
    """

    def __init__(
        self, *, random_state: int | None = None, alpha: float = 1.0,
    ) -> None:
        self._alpha = alpha
        self._random_state = random_state
        self._model = Ridge(alpha=alpha, random_state=random_state)

    def fit(
        self, train_representations: np.ndarray, train_labels: np.ndarray,
    ) -> None:
        self._model.fit(train_representations, train_labels)

    def predict(self, test_representations: np.ndarray) -> np.ndarray:
        return self._model.predict(test_representations)

    @property
    def name(self) -> str:
        return "Ridge Regression"

    @property
    def description(self) -> str:
        return (
            f"scikit-learn Ridge (alpha={self._alpha}), the CE3 "
            "framework default probe for continuous targets."
        )


_METRIC_DISPLAY_NAMES: dict[TargetType, str] = {
    TargetType.BINARY: "Balanced Accuracy",
    TargetType.MULTICLASS: "Balanced Accuracy",
    TargetType.CONTINUOUS: "R\u00b2",
}


def _default_probe_factory(
    target_type: TargetType,
) -> Callable[[int], CryptographicProbe]:
    if target_type in (TargetType.BINARY, TargetType.MULTICLASS):
        return lambda seed: LogisticRegressionProbe(random_state=seed)
    if target_type is TargetType.CONTINUOUS:
        return lambda seed: RidgeRegressionProbe(random_state=seed)
    raise ValueError(f"No default probe defined for target type: {target_type}")


def _default_metric(
    target_type: TargetType,
) -> Callable[[np.ndarray, np.ndarray], float]:
    if target_type in (TargetType.BINARY, TargetType.MULTICLASS):
        return balanced_accuracy_score
    if target_type is TargetType.CONTINUOUS:
        return r2_score
    raise ValueError(f"No default metric defined for target type: {target_type}")


def _make_splitter(
    target_type: TargetType, n_splits: int, n_repeats: int, seed: int,
):
    if target_type in (TargetType.BINARY, TargetType.MULTICLASS):
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed,
        )
    if target_type is TargetType.CONTINUOUS:
        return RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed,
        )
    raise ValueError(f"No default splitter defined for target type: {target_type}")


# ============================================================
# Target Admissibility (Q10, non-degeneracy check)
# ============================================================


def _validate_target(
    target: TargetSpecification,
    real_representations: np.ndarray,
    control_representations: np.ndarray,
) -> None:
    if real_representations.shape[0] != control_representations.shape[0]:
        raise ValueError(
            "real_representations and control_representations must "
            "have the same number of samples: got "
            f"{real_representations.shape[0]} and "
            f"{control_representations.shape[0]}."
        )

    if real_representations.shape[0] != target.labels.shape[0]:
        raise ValueError(
            "target.labels must have one entry per representation "
            f"sample: got {target.labels.shape[0]} labels for "
            f"{real_representations.shape[0]} samples."
        )

    if not target.theoretical_interpretation.strip():
        raise ValueError(
            f"Target '{target.name}' has no theoretical_interpretation. "
            "CE3 requires every declared target to state its "
            "cryptographic meaning independent of this experiment "
            "before probing may proceed."
        )

    if target.target_type in (TargetType.BINARY, TargetType.MULTICLASS):
        n_unique = len(np.unique(target.labels))
        if n_unique < 2:
            raise ValueError(
                f"Target '{target.name}' is degenerate: only "
                f"{n_unique} unique label value(s) found."
            )
    else:
        if np.std(target.labels) == 0.0:
            raise ValueError(
                f"Target '{target.name}' is degenerate: labels have "
                "zero variance."
            )


# ============================================================
# Selectivity Evaluation (Q3, Q4, Q5, Q8, Q9)
# ============================================================


def evaluate_selectivity(
    real_representations: np.ndarray,
    control_representations: np.ndarray,
    target: TargetSpecification,
    *,
    probe_factory: Callable[[int], CryptographicProbe] | None = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> ProbeEvaluation:
    """
    Evaluate control-normalized decodability (selectivity) of
    `target` from `real_representations`, relative to
    `control_representations`.

    Uses repeated stratified k-fold (BINARY/MULTICLASS) or
    repeated k-fold (CONTINUOUS) cross-validation. The default
    is n_splits=5 repeated n_repeats=10 times, giving 50 paired
    per-fold selectivity observations -- deliberately more than
    a single 5-fold pass, since 5 paired observations alone
    cannot reach conventional significance under a Wilcoxon
    signed-rank test regardless of effect size.

    The same (train_idx, test_idx) partition and the same
    per-fold probe seed are used for both the real and control
    arms of every fold, so that only the representation source
    differs between arms (Framework Principle 2).

    Representations are assumed already extracted for the full
    dataset before this function is called; extraction is a
    deterministic forward pass through a frozen, non-trainable
    model with no dependence on the label, so there is no
    leakage risk in splitting only at the probing stage.

    Returns
    -------
    ProbeEvaluation
        Raw per-fold measurements only. Statistical significance
        is computed separately by `compute_significance`.
    """

    _validate_target(target, real_representations, control_representations)

    if probe_factory is None:
        probe_factory = _default_probe_factory(target.target_type)

    metric_fn = _default_metric(target.target_type)
    metric_name = _METRIC_DISPLAY_NAMES[target.target_type]

    splitter = _make_splitter(target.target_type, n_splits, n_repeats, seed)

    real_scores: list[float] = []
    control_scores: list[float] = []
    selectivity_values: list[float] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(real_representations, target.labels)
    ):
        fold_seed = seed + fold_index

        y_train = target.labels[train_idx]
        y_test = target.labels[test_idx]

        real_probe = probe_factory(fold_seed)
        real_probe.fit(real_representations[train_idx], y_train)
        real_pred = real_probe.predict(real_representations[test_idx])
        real_score = float(metric_fn(y_test, real_pred))

        control_probe = probe_factory(fold_seed)
        control_probe.fit(control_representations[train_idx], y_train)
        control_pred = control_probe.predict(control_representations[test_idx])
        control_score = float(metric_fn(y_test, control_pred))

        real_scores.append(real_score)
        control_scores.append(control_score)
        selectivity_values.append(real_score - control_score)

    return ProbeEvaluation(
        real_score_mean=float(np.mean(real_scores)),
        control_score_mean=float(np.mean(control_scores)),
        selectivity_mean=float(np.mean(selectivity_values)),
        selectivity_values=selectivity_values,
        # NOTE: this is the TOTAL number of folds actually run
        # (n_splits * n_repeats), matching len(selectivity_values)
        # exactly -- not the bare n_splits parameter. Flagging
        # this because the field's docstring in types.py doesn't
        # currently distinguish the two; worth a one-line
        # clarification there.
        n_splits=n_splits * n_repeats,
        metric_name=metric_name,
        target=target,
    )


# ============================================================
# Statistical Significance 
# ============================================================
def compute_significance(
    evaluation: ProbeEvaluation,
    *,
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedComparisonStatistic:
    """
    Test whether mean selectivity is significantly greater than
    zero.

    CE3-specific wrapper around the shared paired-significance
    procedure. The pairing unit for CE3 is the per-fold
    selectivity value.
    """

    diffs = np.asarray(
        evaluation.selectivity_values,
        dtype=np.float64,
    )

    return paired_significance(
        diffs,
        alternative=alternative,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def evaluate_selectivity_replicates(
    task_factory: Callable[[int], "RepresentationTask"],
    adapter,
    model,
    control_model,
    *,
    n_replicates: int = 20,
    n_splits: int = 5,
    seed: int = 0,
) -> "ReplicatedProbeEvaluation":
    """
    Estimate selectivity across `n_replicates` independently
    generated evaluation datasets, each internally stabilized
    by `n_splits`-fold CV.

    The replicate is the statistical unit here, not the fold:
    each call to `task_factory` must produce a freshly generated
    dataset and a target derived from that same dataset, so that
    target consistency holds per replicate (Framework Principle
    2 extended across replicates, not just across folds).

    `evaluate_selectivity` is called once per replicate with
    n_repeats=1: the folds inside a replicate exist only to
    stabilize that replicate's single point estimate, and are
    never themselves treated as independent observations.
    """

    replicate_evaluations: list[ProbeEvaluation] = []
    selectivity_replicates: list[float] = []

    for replicate_index in range(n_replicates):
        replicate_seed = seed + replicate_index

        task = task_factory(replicate_seed)

        real_representations = adapter.extract_representations(
            model, task.dataset,
        )
        control_representations = adapter.extract_representations(
            control_model, task.dataset,
        )

        evaluation = evaluate_selectivity(
            real_representations,
            control_representations,
            task.target,
            n_splits=n_splits,
            n_repeats=1,
            seed=replicate_seed,
        )

        replicate_evaluations.append(evaluation)
        selectivity_replicates.append(evaluation.selectivity_mean)

    return ReplicatedProbeEvaluation(
        replicate_evaluations=replicate_evaluations,
        selectivity_replicates=selectivity_replicates,
        n_replicates=n_replicates,
        n_splits_per_replicate=n_splits,
        metric_name=replicate_evaluations[0].metric_name,
        target_name=replicate_evaluations[0].target.name,
    )


def compute_significance_over_replicates(
    replicated_evaluation: "ReplicatedProbeEvaluation",
    *,
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedComparisonStatistic:
    """
    CE3 significance test over independent replicate-level
    selectivity values. Mirrors compute_significance exactly;
    the only difference is which array is the statistical unit.
    """

    diffs = np.asarray(
        replicated_evaluation.selectivity_replicates,
        dtype=np.float64,
    )

    return paired_significance(
        diffs,
        alternative=alternative,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )