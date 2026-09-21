"""
D4 - Controlled Perturbation Audit.

Generic framework for repeated controlled perturbation experiments.

Scientific design
-----------------
D4 estimates whether a controlled perturbation changes predictive
performance relative to an unperturbed baseline.

For binary classification, inference is performed on predictions
from the same held-out test examples, giving a paired comparison.
McNemar's exact test is used for the paired binary correctness
outcomes, while replicate-level performance differences quantify
training-run variability.

D4 deliberately does NOT interpret a label-shuffle effect as proof
of cryptographic dependence. A label shuffle tests whether the
original feature/label correspondence is necessary for predictive
performance; the observed effect may arise from cryptographic
structure, implementation artifacts, dataset-construction artifacts,
or other label-correlated structure.

The framework therefore reports:

- replicate-level baseline accuracy;
- replicate-level perturbed accuracy;
- paired accuracy difference;
- replicate mean and standard deviation;
- bootstrap confidence interval for the replicate-level effect;
- exact McNemar p-value for each paired test-set comparison;
- multiplicity-adjusted McNemar inference across replicates;
- effect-detection decision based on a pre-specified effect size
  and inferential criterion.

No result is labelled "statistically significant" merely because
an effect exceeds a percentage threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import json
from pathlib import Path
from scipy.stats import binomtest

# ============================================================
# Perturbation interface
# ============================================================

class Perturbation:
    """
    Abstract interface for a controlled dataset perturbation.

    A perturbation receives a feature matrix and label array and
    returns the perturbed feature/label pair.

    Perturbation randomness must be supplied explicitly through
    the NumPy Generator argument. This prevents accidental use
    of global RNG state.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
    ) -> None:
        if not name:
            raise ValueError(
                "Perturbation name must be non-empty."
            )

        if not description:
            raise ValueError(
                "Perturbation description must be non-empty."
            )

        self.name = name
        self.description = description

    def apply(
        self,
        features: Any,
        labels: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the perturbation.

        Subclasses must implement this method.
        """

        raise NotImplementedError(
            "Perturbation subclasses must implement apply()."
        )

# ============================================================
# Result structures
# ============================================================

@dataclass
class ReplicateResult:
    """One independent baseline/perturbation training replicate."""

    replicate: int
    seed: int

    baseline_score: float
    perturbed_score: float

    absolute_difference: float
    relative_difference_percent: float

    mcnemar_b: int
    mcnemar_c: int
    mcnemar_pvalue: float


@dataclass
class D4Result:
    """Aggregated repeated D4 experiment."""

    perturbation: str

    replicates: list[ReplicateResult]

    mean_baseline_score: float
    mean_perturbed_score: float

    mean_absolute_difference: float
    sd_absolute_difference: float

    mean_relative_difference_percent: float

    bootstrap_ci_low: float
    bootstrap_ci_high: float

    min_mcnemar_pvalue: float
    max_mcnemar_pvalue: float

    adjusted_alpha: float

    effect_threshold: float

    effect_detected: bool

    inference_supported: bool

    interpretation: str


# ============================================================
# Paired binary inference
# ============================================================

def mcnemar_exact_pvalue(
    baseline_correct: np.ndarray,
    perturbed_correct: np.ndarray,
) -> tuple[int, int, float]:
    """
    Perform exact McNemar inference on paired binary outcomes.

    Parameters
    ----------
    baseline_correct:
        Boolean correctness indicators for the baseline model.

    perturbed_correct:
        Boolean correctness indicators for the perturbed model.

    Returns
    -------
    (b, c, p_value)

    b:
        Baseline incorrect / perturbed correct.

    c:
        Baseline correct / perturbed incorrect.

    Notes
    -----
    Only discordant pairs contribute to the McNemar test.
    """

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool,
    )

    perturbed_correct = np.asarray(
        perturbed_correct,
        dtype=bool,
    )

    if baseline_correct.shape != perturbed_correct.shape:
        raise ValueError(
            "Paired correctness arrays must have identical shapes."
        )

    b = int(
        np.sum(
            (~baseline_correct)
            & perturbed_correct
        )
    )

    c = int(
        np.sum(
            baseline_correct
            & (~perturbed_correct)
        )
    )

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided McNemar test.
        #
        # Under H0, conditional on the discordant total,
        # b ~ Binomial(discordant, 0.5).
        p_value = float(
            binomtest(
                b,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return b, c, p_value


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    bootstrap_replicates: int = 5000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap percentile confidence interval for the mean.

    The bootstrap resamples independent training replicates,
    not individual test examples.

    This distinction is important: test examples are paired
    observations within a replicate, whereas independent model
    fits are the experimental replicates.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) < 2:
        raise ValueError(
            "At least two replicate-level observations are required."
        )

    if bootstrap_replicates < 1000:
        raise ValueError(
            "bootstrap_replicates must be >= 1000."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie in (0, 1)."
        )

    indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_replicates, len(values)),
    )

    bootstrap_means = np.mean(
        values[indices],
        axis=1,
    )

    alpha = 1.0 - confidence_level

    low = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    high = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return low, high


# ============================================================
# Prediction handling
# ============================================================

def binary_predictions_to_correctness(
    predictions: Any,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Convert binary model predictions into correctness indicators.

    Predictions may be shaped:

        (n,)
        (n, 1)

    and are interpreted as probabilities/logits thresholded at 0.5.

    This function is intentionally explicit because D4's paired
    inference is defined for binary classification.
    """

    labels = np.asarray(labels).reshape(-1)

    predictions = np.asarray(predictions)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    if predictions.ndim != 1:
        raise ValueError(
            "Binary predictions must have shape (n,) or (n, 1)."
        )

    if len(predictions) != len(labels):
        raise ValueError(
            "Prediction and label counts must match."
        )

    predicted_labels = (
        predictions >= 0.5
    ).astype(np.int64)

    return predicted_labels == labels.astype(np.int64)

def checkpoint_paths(
    checkpoint_root: Path,
    replicate: int,
    condition: str,
) -> tuple[Path, Path]:
    """Return the persistent condition directory and state path.

    The first return value is intentionally a directory rather than a
    single rolling checkpoint.  The Gohr adapter stores immutable
    epoch checkpoints inside this directory as:

        checkpoint_epoch_001.keras
        checkpoint_epoch_002.keras
        ...

    The state sidecar records the latest completed epoch.
    """
    directory = (
        checkpoint_root
        / f"replicate_{replicate:02d}"
        / condition
    )
    return directory, directory / "state.json"


def load_completed_state(
    state_path: Path,
) -> dict[str, Any] | None:

    if not state_path.exists():
        return None

    with state_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)
# ============================================================
# One repeated experiment
# ============================================================

def run_d4(
    *,
    perturbation: Any,
    train_features: Any,
    train_labels: Any,
    test_features: Any,
    test_labels: Any,
    adapter_factory: Callable[[int], Any],
    checkpoint_root: Path,
    replicates: int,
    audit_seed: int,
    effect_threshold: float,
    bootstrap_replicates: int = 5000,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
    run_config_hash: str | None = None,
) -> D4Result:
    """
    Execute repeated controlled perturbation inference.

    Parameters
    ----------
    perturbation:
        Perturbation object with an apply(features, labels, rng)
        method.

    train_features, train_labels:
        Original training data.

    test_features, test_labels:
        Fixed held-out test data. Every model is evaluated on
        exactly this same test partition.

    adapter_factory:
        Callable receiving an independent integer seed and
        returning a fresh dataset/model adapter.

    replicates:
        Number of independent model-training replicates.

    audit_seed:
        Seed controlling the audit-side RNG streams.

    effect_threshold:
        Minimum absolute accuracy difference, expressed in
        percentage points, regarded as practically meaningful.

    bootstrap_replicates:
        Number of bootstrap resamples over independent
        training replicates.

    confidence_level:
        Confidence level for the replicate-level bootstrap CI.

    alpha:
        Family-wise significance level.

    Returns
    -------
    D4Result
    """

    if replicates < 2:
        raise ValueError(
            "At least two independent training replicates are required."
        )

    if effect_threshold < 0:
        raise ValueError(
            "effect_threshold must be non-negative."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must lie in (0, 1)."
        )

    seed_sequence = np.random.SeedSequence(
        audit_seed
    )

    child_sequences = seed_sequence.spawn(
        replicates
    )

    replicate_results: list[ReplicateResult] = []

    for replicate_index, child_sequence in enumerate(
        child_sequences,
        start=1,
    ):
        rng = np.random.default_rng(
            child_sequence
        )

        seed_value = int(
            child_sequence.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )

        print()
        print(
            f"Running D4 replicate "
            f"{replicate_index}/{replicates} "
            f"(seed={seed_value})..."
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_adapter = adapter_factory(
            seed_value
        )

        baseline_checkpoint, baseline_state = (
            checkpoint_paths(
                checkpoint_root,
                replicate_index,
                "baseline",
            )
        )

        baseline_model = baseline_adapter.train(
            train_features,
            train_labels,
            checkpoint_path=baseline_checkpoint,
            state_path=baseline_state,
            replicate=replicate_index,
            condition="baseline",
            run_config_hash=run_config_hash,
        )

        baseline_predictions = baseline_adapter.predict(
            test_features
        )

        baseline_correct = (
            binary_predictions_to_correctness(
                baseline_predictions,
                test_labels,
            )
        )

        baseline_score = float(
            np.mean(baseline_correct)
        )

        # ----------------------------------------------------
        # Perturbation
        # ----------------------------------------------------

        perturbation_directory = (
            checkpoint_root
            / "perturbations"
        )

        perturbation_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        perturbation_path = (
            perturbation_directory
            / f"replicate_{replicate_index:02d}_labels.npy"
        )

        if perturbation_path.exists():

            print(
                f"Loading persisted perturbation for "
                f"replicate {replicate_index}..."
            )

            perturbed_features = train_features

            perturbed_labels = np.load(
                perturbation_path,
                mmap_mode="r",
            )

        else:

            print(
                f"Generating perturbation for "
                f"replicate {replicate_index}..."
            )

            perturbed_features, perturbed_labels = (
                perturbation.apply(
                    train_features,
                    train_labels,
                    rng=rng,
                )
            )

            np.save(
                perturbation_path,
                perturbed_labels,
            )

        perturbed_adapter = adapter_factory(
            seed_value
        )

        perturbed_checkpoint, perturbed_state = (
            checkpoint_paths(
                checkpoint_root,
                replicate_index,
                "perturbed",
            )
        )

        perturbed_model = perturbed_adapter.train(
            perturbed_features,
            perturbed_labels,
            checkpoint_path=perturbed_checkpoint,
            state_path=perturbed_state,
            replicate=replicate_index,
            condition="perturbed",
            run_config_hash=run_config_hash,
        )

        perturbed_predictions = (
            perturbed_adapter.predict(
                test_features
            )
        )

        perturbed_correct = (
            binary_predictions_to_correctness(
                perturbed_predictions,
                test_labels,
            )
        )

        perturbed_score = float(
            np.mean(perturbed_correct)
        )

        # ----------------------------------------------------
        # Paired inference
        # ----------------------------------------------------

        b, c, p_value = mcnemar_exact_pvalue(
            baseline_correct,
            perturbed_correct,
        )

        absolute_difference = (
            perturbed_score
            - baseline_score
        )

        if baseline_score == 0.0:
            relative_difference_percent = float("nan")
        else:
            relative_difference_percent = (
                absolute_difference
                / baseline_score
                * 100.0
            )

        replicate_results.append(
            ReplicateResult(
                replicate=replicate_index,
                seed=seed_value,
                baseline_score=baseline_score,
                perturbed_score=perturbed_score,
                absolute_difference=absolute_difference,
                relative_difference_percent=(
                    relative_difference_percent
                ),
                mcnemar_b=b,
                mcnemar_c=c,
                mcnemar_pvalue=p_value,
            )
        )

        # Explicitly delete models before next replicate.
        del baseline_model
        del perturbed_model

    # ========================================================
    # Aggregate independent training replicates
    # ========================================================

    baseline_scores = np.asarray(
        [
            r.baseline_score
            for r in replicate_results
        ],
        dtype=np.float64,
    )

    perturbed_scores = np.asarray(
        [
            r.perturbed_score
            for r in replicate_results
        ],
        dtype=np.float64,
    )

    absolute_differences = np.asarray(
        [
            r.absolute_difference
            for r in replicate_results
        ],
        dtype=np.float64,
    )

    relative_differences = np.asarray(
        [
            r.relative_difference_percent
            for r in replicate_results
        ],
        dtype=np.float64,
    )

    bootstrap_rng = np.random.default_rng(
        np.random.SeedSequence(
            [audit_seed, 0xD4]
        )
    )

    ci_low, ci_high = bootstrap_mean_ci(
        absolute_differences,
        rng=bootstrap_rng,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )

    # --------------------------------------------------------
    # Multiplicity control
    # --------------------------------------------------------

    # Each replicate produces one paired McNemar test.
    # Bonferroni controls family-wise error conservatively.
    adjusted_alpha = (
        alpha / replicates
    )

    mcnemar_passes = sum(
        r.mcnemar_pvalue < adjusted_alpha
        for r in replicate_results
    )

    # A robust decision requires:
    #
    # 1. practical effect threshold met;
    # 2. replicate-level bootstrap CI excludes zero;
    # 3. all independent paired comparisons meet the
    #    conservative multiplicity-adjusted criterion.
    #
    # This is intentionally conservative.
    ci_excludes_zero = (
        ci_low > 0.0
        or ci_high < 0.0
    )

    practical_effect = (
        abs(
            float(
                np.mean(
                    absolute_differences
                )
            )
        )
        >= effect_threshold
    )

    inference_supported = (
        mcnemar_passes == replicates
        and ci_excludes_zero
    )

    effect_detected = (
        practical_effect
        and inference_supported
    )

    interpretation = (
        "A reproducible performance effect was detected under "
        "the specified perturbation, with the pre-specified "
        "practical-effect criterion and conservative paired "
        "inference criterion both satisfied."
        if effect_detected
        else
        "The experiment did not satisfy all pre-specified "
        "criteria for a reproducible performance effect."
    )

    return D4Result(
        perturbation=perturbation.name,
        replicates=replicate_results,
        mean_baseline_score=float(
            np.mean(baseline_scores)
        ),
        mean_perturbed_score=float(
            np.mean(perturbed_scores)
        ),
        mean_absolute_difference=float(
            np.mean(absolute_differences)
        ),
        sd_absolute_difference=float(
            np.std(
                absolute_differences,
                ddof=1,
            )
        ),
        mean_relative_difference_percent=float(
            np.nanmean(relative_differences)
        ),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        min_mcnemar_pvalue=float(
            np.min(
                [
                    r.mcnemar_pvalue
                    for r in replicate_results
                ]
            )
        ),
        max_mcnemar_pvalue=float(
            np.max(
                [
                    r.mcnemar_pvalue
                    for r in replicate_results
                ]
            )
        ),
        adjusted_alpha=adjusted_alpha,
        effect_threshold=effect_threshold,
        effect_detected=effect_detected,
        inference_supported=inference_supported,
        interpretation=interpretation,
    )


# ============================================================
# Certificate
# ============================================================

def generate_certificate(
    result: D4Result,
    *,
    dataset_id: str,
    dataset_version: str,
    generation_procedure: str,
    generation_parameters: Mapping[str, Any],
    audit_seed: int,
    confidence_level: float,
    alpha: float,
) -> dict[str, Any]:
    """
    Construct a machine-readable D4 certificate.
    """

    return {
        "audit": {
            "id": "D4",
            "name": "Controlled Perturbation Audit",
            "claim": (
                "Whether the specified controlled perturbation "
                "produces a reproducible change in predictive "
                "performance under the stated experimental "
                "and inferential design."
            ),
        },
        "decision": {
            "outcome": (
                "EFFECT_DETECTED"
                if result.effect_detected
                else "NO_REPRODUCIBLE_EFFECT_DETECTED"
            ),
            "effect_detected": result.effect_detected,
            "inference_supported": result.inference_supported,
        },
        "findings": {
            "perturbation": result.perturbation,
            "replicates": [
                {
                    "replicate": r.replicate,
                    "seed": r.seed,
                    "baseline_accuracy": r.baseline_score,
                    "perturbed_accuracy": r.perturbed_score,
                    "accuracy_difference": (
                        r.absolute_difference
                    ),
                    "relative_difference_percent": (
                        r.relative_difference_percent
                    ),
                    "mcnemar_b": r.mcnemar_b,
                    "mcnemar_c": r.mcnemar_c,
                    "mcnemar_pvalue": r.mcnemar_pvalue,
                }
                for r in result.replicates
            ],
            "mean_baseline_accuracy": (
                result.mean_baseline_score
            ),
            "mean_perturbed_accuracy": (
                result.mean_perturbed_score
            ),
            "mean_accuracy_difference": (
                result.mean_absolute_difference
            ),
            "sd_accuracy_difference": (
                result.sd_absolute_difference
            ),
            "mean_relative_difference_percent": (
                result.mean_relative_difference_percent
            ),
            "bootstrap_ci": {
                "confidence_level": confidence_level,
                "low": result.bootstrap_ci_low,
                "high": result.bootstrap_ci_high,
            },
            "mcnemar": {
                "minimum_pvalue": (
                    result.min_mcnemar_pvalue
                ),
                "maximum_pvalue": (
                    result.max_mcnemar_pvalue
                ),
                "familywise_alpha": alpha,
                "bonferroni_adjusted_alpha": (
                    result.adjusted_alpha
                ),
            },
        },
        "methodology": {
            "unit_of_independent_replication": (
                "independent model training replicate"
            ),
            "evaluation_design": (
                "same fixed held-out test partition for "
                "baseline and perturbed models"
            ),
            "paired_test": (
                "exact two-sided McNemar test on paired "
                "test-example correctness"
            ),
            "multiple_comparison_control": (
                "Bonferroni correction across independent "
                "replicate-level McNemar tests"
            ),
            "effect_size": (
                "absolute accuracy difference in percentage "
                "points"
            ),
            "practical_effect_threshold": (
                result.effect_threshold
            ),
            "confidence_interval": (
                "percentile bootstrap over independent "
                "training replicates"
            ),
            "bootstrap_replicates": 5000,
            "decision_rule": (
                "EFFECT_DETECTED iff the mean absolute accuracy "
                "difference meets the practical-effect threshold, "
                "the replicate-level bootstrap confidence interval "
                "excludes zero, and every replicate satisfies the "
                "Bonferroni-adjusted paired McNemar criterion."
            ),
        },
        "provenance": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_procedure": generation_procedure,
            "generation_parameters": dict(
                generation_parameters
            ),
            "audit_seed": audit_seed,
        },
        "limitations": [
            (
                "The perturbation result establishes dependence "
                "on the original feature/label correspondence, "
                "not cryptographic dependence specifically."
            ),
            (
                "A label shuffle can disrupt any predictive "
                "relationship between features and labels, "
                "including relationships arising from dataset "
                "construction or implementation artifacts."
            ),
            (
                "Inference is conditional on the specified "
                "model architecture, optimization procedure, "
                "dataset generation procedure, perturbation, "
                "and held-out test partition."
            ),
            (
                "The bootstrap confidence interval quantifies "
                "uncertainty across independent training "
                "replicates; it is not a confidence interval for "
                "the population of all possible datasets."
            ),
            (
                "The experiment does not by itself establish "
                "security, cryptographic hardness, or absence "
                "of alternative predictive shortcuts."
            ),
        ],
        "interpretation": result.interpretation,
    }


# ============================================================
# Reporting
# ============================================================

def print_report(
    result: D4Result,
) -> None:
    """Print the D4 statistical report."""

    print("=" * 72)
    print("Dataset Integrity Audit")
    print("D4 - Controlled Perturbation Audit")
    print("=" * 72)
    print()

    print(
        f"Perturbation                 : "
        f"{result.perturbation}"
    )

    print(
        f"Independent replicates      : "
        f"{len(result.replicates)}"
    )

    print()

    print("Aggregate performance")
    print("-" * 72)

    print(
        f"Mean baseline accuracy      : "
        f"{result.mean_baseline_score:.8f}"
    )

    print(
        f"Mean perturbed accuracy     : "
        f"{result.mean_perturbed_score:.8f}"
    )

    print(
        f"Mean accuracy difference    : "
        f"{result.mean_absolute_difference:.8f}"
    )

    print(
        f"SD accuracy difference      : "
        f"{result.sd_absolute_difference:.8f}"
    )

    print(
        f"Mean relative difference    : "
        f"{result.mean_relative_difference_percent:.4f}%"
    )

    print()

    print("Replicate-level paired inference")
    print("-" * 72)

    for replicate in result.replicates:
        print(
            f"Replicate {replicate.replicate:2d} | "
            f"baseline={replicate.baseline_score:.6f} | "
            f"perturbed={replicate.perturbed_score:.6f} | "
            f"diff={replicate.absolute_difference:+.6f} | "
            f"McNemar p={replicate.mcnemar_pvalue:.6g}"
        )

    print()

    print("Replicate-level bootstrap")
    print("-" * 72)

    print(
        f"95% CI for mean accuracy difference: "
        f"[{result.bootstrap_ci_low:.8f}, "
        f"{result.bootstrap_ci_high:.8f}]"
    )

    print()

    print("Decision")
    print("-" * 72)

    print(
        f"Practical-effect threshold    : "
        f"{result.effect_threshold:.8f}"
    )

    print(
        f"Bonferroni-adjusted alpha     : "
        f"{result.adjusted_alpha:.8g}"
    )

    print(
        f"Paired inference supported    : "
        f"{result.inference_supported}"
    )

    print(
        f"Effect detected               : "
        f"{result.effect_detected}"
    )

    print(
        f"Interpretation                : "
        f"{result.interpretation}"
    )

    print("=" * 72)