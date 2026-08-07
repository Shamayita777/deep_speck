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

from ..test.ce3.types import PairedComparisonStatistic, ProbeEvaluation, TargetSpecification, TargetType
from .base import CryptographicProbe


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
    zero, using the Wilcoxon signed-rank test on the paired
    per-fold selectivity values in `evaluation.selectivity_values`.

    `alternative="greater"` (default) matches CE3's directional
    hypothesis -- real representations decode the target better
    than the control -- rather than testing a non-directional
    "different from zero" hypothesis, which would needlessly
    halve the test's power for the question actually being asked.

    Effect size is Cohen's d_z (matched-pairs standardized mean
    difference: mean(diff) / std(diff)), reported independently
    of the significance test chosen, since Wilcoxon's own
    rank-biserial effect size requires internals scipy does not
    expose directly.

    The confidence interval is a percentile bootstrap over the
    per-fold selectivity values rather than a normal-
    approximation interval, since using Wilcoxon here already
    signals that normality of the per-fold differences should
    not be assumed.
    """

    diffs = np.asarray(evaluation.selectivity_values, dtype=np.float64)
    n = diffs.shape[0]

    if n < 2:
        raise ValueError(
            "At least 2 paired folds are required to compute "
            "statistical significance."
        )

    if np.all(diffs == diffs[0]):
        raise ValueError(
            "Wilcoxon signed-rank test is undefined: all per-fold "
            "selectivity values are identical (zero variance "
            "across folds)."
        )

    statistic, p_value = wilcoxon(diffs, alternative=alternative)

    std = np.std(diffs, ddof=1)
    effect_size = float(np.mean(diffs) / std) if std > 0 else float("nan")

    rng = np.random.default_rng(seed)
    resamples = rng.choice(diffs, size=(n_bootstrap, n), replace=True)
    bootstrap_means = resamples.mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    print("\n----- CE3 statistics -----")
    print("Mean:", np.mean(diffs))
    print("Std :", np.std)
    print("Min :", np.min(diffs))
    print("Max :", np.max(diffs))
    print("First 10:", diffs[:10])
    print("--------------------------\n")
    return PairedComparisonStatistic(
        test_name="wilcoxon_signed_rank",
        alternative=alternative,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=effect_size,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_pairs=int(n),
    )