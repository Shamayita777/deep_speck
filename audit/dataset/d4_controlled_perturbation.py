"""
D4 - Controlled Perturbation Audit

Architecture

Perturbation
        ↓
PerturbationResult
        ↓
Dataset Adapter
        ↓
Generic Perturbation Engine
        ↓
Evaluation
        ↓
Certificate
        ↓
Reporting
        ↓
JSON
        ↓
main()
"""

# ============================================================
# Imports
# ============================================================

from abc import ABC, abstractmethod

from typing import Any

from dataclasses import dataclass
# ============================================================
# Phase 1
# Perturbation Interface
# ============================================================

@dataclass
class Perturbation(ABC):
    """
    Abstract base class for controlled dataset perturbations.

    Every perturbation must define:
        • a name
        • a description
        • an apply() method that returns the perturbed
          features and labels.
    """

    name: str
    description: str

    @abstractmethod
    def apply(
        self,
        features: Any,
        labels: Any,
    ) -> tuple[Any, Any]:
        """
        Apply the perturbation.

        Parameters
        ----------
        features
            Original feature matrix.

        labels
            Original target labels.

        Returns
        -------
        tuple
            Perturbed (features, labels).
        """
        raise NotImplementedError


@dataclass
class PerturbationResult:
    """
    Result produced by a controlled perturbation audit.
    """

    perturbation: str

    baseline_score: float

    perturbed_score: float

    absolute_difference: float

    relative_difference: float

    runtime: float

    notes: str

# ============================================================
# Phase 2
# Generic Perturbation Engine
# ============================================================

import time

def run_perturbation(
    *,
    perturbation: Perturbation,
    features: Any,
    labels: Any,
    adapter: Any,
    notes: str = "",
) -> PerturbationResult:
    """
    Execute a controlled perturbation experiment.

    The framework is intentionally independent of the
    underlying dataset, machine-learning framework,
    evaluation metric, and cryptographic primitive.

    Parameters
    ----------
    perturbation
        Perturbation to apply.

    features
        Original feature matrix.

    labels
        Original labels.

    adapter
        Dataset adapter for training and evaluating models.

    notes
        Optional experiment notes.

    Returns
    -------
    PerturbationResult
        Result of the perturbation experiment.
    """

    start = time.perf_counter()

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    import json

    from pathlib import Path

    evidence_dir = Path("audit/dataset/evidence/d4")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    status_file = evidence_dir / "baseline_status.json"
    model_file = evidence_dir / "baseline_model.keras"

    if status_file.exists() and model_file.exists():

        print("Loading existing baseline...")

        baseline_model = adapter.load(model_file)

        with open(status_file) as fp:
            status = json.load(fp)

        baseline_score = status["baseline_score"]

    else:

        print("Training baseline...")

        baseline_model = adapter.train(
            features,
            labels,
        )

        baseline_score = adapter.evaluate(
            baseline_model,
        )

        adapter.save(model_file)

        with open(status_file, "w") as fp:
            json.dump(
                {
                    "completed": True,
                    "baseline_score": baseline_score,
                    "num_rounds": adapter.num_rounds,
                    "depth": adapter.depth,
                    "epochs": adapter.epochs,
                    "batch_size": adapter.batch_size,
                    "seed": adapter.seed,
                },
                fp,
                indent=4,
            )
    # --------------------------------------------------------
    # Controlled perturbation
    # --------------------------------------------------------

    perturbed_features, perturbed_labels = (
        perturbation.apply(
            features,
            labels,
        )
    )

    perturbed_model = adapter.train(
        perturbed_features,
        perturbed_labels,
    )

    perturbed_score = adapter.evaluate(
        perturbed_model,
    )

    # --------------------------------------------------------
    # Performance difference
    # --------------------------------------------------------

    absolute_difference = (
        perturbed_score
        - baseline_score
    )

    if baseline_score == 0:

        relative_difference = 0.0

    else:

        relative_difference = (
            absolute_difference
            / baseline_score
        ) * 100.0

    runtime = (
        time.perf_counter()
        - start
    )

    return PerturbationResult(

        perturbation=perturbation.name,

        baseline_score=baseline_score,

        perturbed_score=perturbed_score,

        absolute_difference=absolute_difference,

        relative_difference=relative_difference,

        runtime=runtime,

        notes=notes,
    )

# ============================================================
# Evaluation
# ============================================================

@dataclass
class EvaluationResult:
    """
    Decision produced from a perturbation result.

    The framework does not assume any particular
    machine-learning metric or expected direction of
    change. The interpretation of the observed effect
    is left to the dataset adapter.
    """

    observed_effect: bool

    threshold: float

    decision: str

    rationale: str


def evaluate_result(
    result: PerturbationResult,
    *,
    effect_threshold: float,
) -> EvaluationResult:
    """
    Evaluate whether a perturbation produced a
    meaningful performance change.

    Parameters
    ----------
    result
        Result returned by the perturbation engine.

    effect_threshold
        Minimum absolute relative change (%) required
        to consider the perturbation effective.

    Returns
    -------
    EvaluationResult
    """

    observed_effect = (
        abs(result.relative_difference)
        >= effect_threshold
    )

    decision = (
        "SIGNIFICANT_CHANGE"
        if observed_effect
        else "NO_SIGNIFICANT_CHANGE"
    )

    rationale = (
        "Perturbation produced a measurable "
        "performance change."
        if observed_effect
        else
        "Perturbation produced no substantial "
        "performance change."
    )

    return EvaluationResult(

        observed_effect=observed_effect,

        threshold=effect_threshold,

        decision=decision,

        rationale=rationale,
    )

# ============================================================
# Certificate
# ============================================================

def generate_certificate(
    result: PerturbationResult,
    evaluation: EvaluationResult,
) -> dict:
    """
    Generate a machine-readable audit certificate.

    Returns
    -------
    dict
        Certificate summarizing the perturbation audit.
    """

    return {

        "audit_dimension": "Controlled Perturbation",

        "perturbation": result.perturbation,

        "baseline_score": result.baseline_score,

        "perturbed_score": result.perturbed_score,

        "absolute_difference": result.absolute_difference,

        "relative_difference": result.relative_difference,

        "runtime": result.runtime,

        "decision": evaluation.decision,

        "observed_effect": evaluation.observed_effect,

        "threshold": evaluation.threshold,

        "rationale": evaluation.rationale,

        "notes": result.notes,
    }

# ============================================================
# Reporting
# ============================================================

import json
from pathlib import Path


def print_report(
    result: PerturbationResult,
    evaluation: EvaluationResult,
) -> None:
    """
    Print a human-readable perturbation audit report.
    """

    print("=" * 60)
    print("Controlled Perturbation Audit")
    print("=" * 60)

    print(f"Perturbation        : {result.perturbation}")
    print(f"Baseline Score      : {result.baseline_score:.6f}")
    print(f"Perturbed Score     : {result.perturbed_score:.6f}")
    print(f"Absolute Difference : {result.absolute_difference:.6f}")
    print(f"Relative Difference : {result.relative_difference:.2f}%")
    print(f"Runtime             : {result.runtime:.2f} seconds")
    print()

    print(f"Decision            : {evaluation.decision}")
    print(f"Observed Effect     : {evaluation.observed_effect}")
    print(f"Threshold           : {evaluation.threshold:.2f}%")
    print(f"Rationale           : {evaluation.rationale}")

    if result.notes:
        print(f"Notes               : {result.notes}")

    print("=" * 60)


def save_json(
    certificate: dict,
    output_path: str | Path,
) -> None:
    """
    Save the audit certificate as a JSON file.
    """

    output_path = Path(output_path)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            certificate,
            fp,
            indent=4,
        )

