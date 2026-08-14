"""
Cryptographic Evidence CE3 Datatypes
====================================

Framework-level data structures for the Representation Interpretation
audit (CE3).

These dataclasses describe the information exchanged between the
Cryptographic Evidence framework and adapter implementations during
representation analysis.

The module is intentionally independent of any particular
cryptographic primitive, machine-learning architecture, probing
algorithm, or research paper.

Responsibilities
----------------
• Describe representation datasets
• Describe cryptographic target specifications
• Describe probe evaluation results

No experimental logic is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


# ============================================================
# Target Types
# ============================================================


class TargetType(Enum):
    """
    Type of cryptographic prediction task.

    The target type allows the framework to select an appropriate
    generic probing procedure without requiring any knowledge of the
    underlying cryptographic quantity.

    BINARY
        Two-class prediction.

    MULTICLASS
        Finite set of discrete classes.

    CONTINUOUS
        Continuous-valued cryptographic quantity.
    """

    BINARY = "binary"

    MULTICLASS = "multiclass"

    CONTINUOUS = "continuous"


# ============================================================
# Target Specification
# ============================================================


@dataclass(frozen=True, slots=True)
class TargetSpecification:
    """
    Cryptographic quantity to be decoded from a learned
    representation.

    The framework deliberately does not prescribe what this quantity
    should be. Instead, each adapter declares the quantity together
    with sufficient metadata for scientific interpretation.

    Framework-level admissibility constraints (validated elsewhere)
    include:

    • non-degenerate target
    • model-independent target
    • independently interpretable cryptographic meaning
    """

    name: str
    """
    Human-readable target name.

    Examples
    --------
    Differential Class
    Trail Probability
    Round Count
    """

    description: str
    """
    Brief explanation of the target.
    """

    target_type: TargetType
    """
    Binary, multiclass, or continuous.
    """

    labels: np.ndarray
    """
    Ground-truth target values.

    These labels must be generated independently of the model under
    evaluation.
    """

    theoretical_interpretation: str
    """
    Scientific meaning of the target.

    This field explains why the target corresponds to a meaningful
    cryptographic quantity.
    """


# ============================================================
# Representation Dataset
# ============================================================


@dataclass(frozen=True, slots=True)
class RepresentationDataset:
    """
    Dataset used for representation probing.

    The framework views a representation dataset simply as a matrix of
    learned representations paired with a declared cryptographic
    target.

    It makes no assumptions regarding

    • network architecture
    • hidden layer
    • feature dimensionality
    • probing algorithm
    """

    representations: np.ndarray
    """
    Matrix of learned representations.

    Shape
    -----
    (num_samples, representation_dimension)
    """

    target: TargetSpecification
    """
    Target declared by the adapter.
    """

    @property
    def num_samples(self) -> int:
        """
        Number of samples.
        """

        return int(self.representations.shape[0])

    @property
    def representation_dimension(self) -> int:
        """
        Representation dimensionality.
        """

        return int(self.representations.shape[1])

@dataclass(frozen=True, slots=True)
class RepresentationTask:
    """
    One representation probing task executed during CE3.

    A task couples the evaluation dataset with the
    cryptographic target declared for that dataset.

    Different targets may require different evaluation
    datasets. For example,

        • differential-class calibration requires a
          dataset containing both real and random pairs

        • analytical trail probability is defined only
          for real differential pairs.

    Bundling the dataset and target together avoids
    assuming that one evaluation dataset is appropriate
    for every probing target.
    """

    dataset: Any
    """
    Evaluation dataset associated with this probing task.
    """

    target: TargetSpecification
    """
    Cryptographic target evaluated on the supplied dataset.
    """

    is_primary: bool = False
    """
    Whether this task provides the primary scientific
    evidence used for the CE3 decision.

    Exactly one task should be marked as primary.
    """

# ============================================================
# Probe Evaluation
# ============================================================

@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    """
    Intermediate result produced by representation probing.

    ProbeEvaluation is not the final CE3 audit result.

    Instead, it records the probing statistics produced during
    cross-validated probing. These statistics are later converted
    into the generic CryptographicTestResult used by the
    Cryptographic Evidence framework.
    """

    real_score_mean: float
    """
    Mean probe performance on the trained model
    representations across all folds.
    """

    control_score_mean: float
    """
    Mean probe performance on the control model
    representations across all folds.
    """

    selectivity_mean: float
    """
    Mean selectivity across all folds.

    Selectivity is defined as

        real_score - control_score

    computed independently for each fold and then averaged.
    """

    selectivity_values: list[float]
    """
    Per-fold selectivity values.

    These values are retained so that downstream statistical
    procedures (e.g., paired t-test or Wilcoxon signed-rank
    test) may be performed without repeating the probing
    experiment.
    """

    n_splits: int
    """
    Number of cross-validation folds.
    """
    metric_name: str
    """
    Performance metric used during probing.

    Examples
    --------
    Accuracy
    R²
    """

    target: TargetSpecification
    """
    Cryptographic target evaluated by the probe.
    """
@dataclass(frozen=True, slots=True)
class ReplicatedProbeEvaluation:
    """
    CE3 evidence across independent replicate-level observations.

    This is the object that should feed statistical significance
    testing. `ProbeEvaluation.selectivity_values` (per-fold) is
    never itself a set of independent observations -- it exists
    only to produce one stabilized point estimate per replicate,
    retained here in `replicate_evaluations` purely for audit
    trail / debugging, not for significance testing.
    """

    replicate_evaluations: list[ProbeEvaluation]
    """
    Full per-replicate CV detail, retained for audit trail.
    NOT the statistical unit -- do not feed to paired_significance.
    """

    selectivity_replicates: list[float]
    """
    One selectivity value per independent replicate. THIS is the
    array that should be passed to paired_significance -- these
    are the actual statistical observations.
    """

    n_replicates: int
    """
    Number of independently generated evaluation datasets.
    """

    n_splits_per_replicate: int
    """
    Number of CV folds used internally within each replicate to
    stabilize its point estimate. NOT a count of independent
    observations.
    """

    metric_name: str

    target_name: str

    @property
    def selectivity_mean(self) -> float:
        return float(np.mean(self.selectivity_replicates))
# ============================================================
# Statistical Significance
# ============================================================

@dataclass(frozen=True, slots=True)
class PairedComparisonStatistic:
    """
    Statistical significance of a CE3 selectivity result.

    This dataclass represents the statistical interpretation of a
    ProbeEvaluation. It is produced after probing has completed and
    therefore remains a distinct stage from evidence generation,
    following Framework Principle 5 (Evidence and Interpretation).
    """

    test_name: str
    """
    Name of the statistical test.

    Example
    -------
    wilcoxon_signed_rank
    """

    alternative: str
    """
    Alternative hypothesis used by the statistical test.
    """

    statistic: float
    """
    Test statistic.
    """

    p_value: float
    """
    Statistical significance.
    """

    effect_size: float
    """
    Standardized effect size.
    """

    ci_low: float
    """
    Lower confidence interval bound.
    """

    ci_high: float
    """
    Upper confidence interval bound.
    """

    n_pairs: int
    """
    Number of paired observations used by the statistical test.
    """