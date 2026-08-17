"""
Causal Intervention Test
==========================

Cryptographic Evidence Phase CE4.

Scientific Goal
---------------
Determine whether the same cryptographic quantity CE2 and CE3
established as theoretically consistent and representationally
decodable is CAUSALLY NECESSARY for the trained model's output --
completing the evidential chain from correlation (CE2) through
decodability (CE3) to causal dependence (CE4).

Evidence Logic
--------------
For each declared target, the frozen model's output is compared
across three conditions: the original input, an input with the
theoretically-relevant structure perturbed (structural
intervention), and an input with a magnitude-matched but
theoretically-irrelevant region perturbed (control intervention).
The paired difference between these two effects -- the necessity
gap -- is CE4's evidence. A positive, statistically significant
necessity gap supports causal dependence on the declared
structure specifically, rather than on perturbation magnitude
alone.

This module is framework-level only. No intervention mechanism,
magnitude definition, or cryptographic knowledge lives here --
that is entirely adapter-owned, per `probe.intervention`.

Scope
-----
Mirrors test/ce3/representation_interpretation.py: the adapter
declares a list of InterventionTask objects, exactly one marked
is_primary, driving the CE4 decision; remaining tasks report as
auxiliary evidence in metadata.
"""

from __future__ import annotations

import dataclasses
import time

from audit.cryptography.adapters.base import CryptographicAdapter
from audit.cryptography.probe.intervention import (
    compute_significance,
    evaluate_necessity,
)
from audit.cryptography.results import CryptographicTestResult
from audit.cryptography.test.base import CryptographicTest, EvidenceDirection


class CausalInterventionTest(CryptographicTest):
    """
    Cryptographic Evidence Test CE4.

    Compares paired structural-vs-control intervention effects
    on a frozen model's output for an adapter-declared
    cryptographic target. Positive, statistically significant
    necessity gap supports the hypothesis that the model's
    output causally depends on that target, not merely
    correlates with or encodes it.
    """

    def __init__(
        self,
        *,
        magnitude_tolerance: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            name="Causal Intervention",
            description=(
                "Evaluate whether perturbing an independently "
                "declared cryptographic structure changes the "
                "frozen model's output more than a magnitude-"
                "matched, theoretically irrelevant perturbation."
            ),
            hypothesis=(
                "The targeted structural intervention should "
                "produce a larger output change than the matched "
                "control intervention."
            ),
            evidence_direction=EvidenceDirection.HIGHER_IS_BETTER,
        )
        self._magnitude_tolerance = magnitude_tolerance
        self._seed = seed

    def validate(self, adapter: CryptographicAdapter) -> None:
        if not isinstance(adapter, CryptographicAdapter):
            raise TypeError(
                "adapter must implement CryptographicAdapter."
            )

    def run(self, adapter: CryptographicAdapter) -> CryptographicTestResult:
        self.validate(adapter)

        start = time.perf_counter()

        model = adapter.load()

        tasks = adapter.generate_intervention_tasks()

        primary_tasks = [t for t in tasks if t.is_primary]
        if len(primary_tasks) != 1:
            raise ValueError(
                "Exactly one InterventionTask must be marked "
                f"is_primary=True; found {len(primary_tasks)}."
            )

        task_results: dict[str, dict] = {}

        for task in tasks:

            filtered_dataset, n_excluded, n_total = (
                adapter.select_intervenable_samples(task.dataset, task.target)
            )

            if n_excluded > 0:
                exclusion_rate = n_excluded / n_total
                # 1% is a placeholder sanity bound, same spirit as CE4's
                # effect-size thresholds -- should be justified/reported,
                # not treated as self-evidently correct.
                if exclusion_rate > 0.01:
                    raise ValueError(
                        f"select_intervenable_samples excluded "
                        f"{n_excluded}/{n_total} samples "
                        f"({exclusion_rate:.2%}) for target "
                        f"'{task.target.name}'. This exceeds the 1% "
                        "sanity bound and likely indicates a "
                        "misconfigured intervention magnitude rather "
                        "than an expected rare boundary condition -- "
                        "investigate before proceeding."
                    )

            task = dataclasses.replace(task, dataset=filtered_dataset)

            structural_dataset = adapter.apply_structural_intervention(
                task.dataset, task.target,
            )
            control_dataset = adapter.apply_control_intervention(
                task.dataset, task.target,
            )

            structural_magnitude = adapter.compute_intervention_magnitude(
                task.dataset, structural_dataset,
            )
            control_magnitude = adapter.compute_intervention_magnitude(
                task.dataset, control_dataset,
            )

            original_predictions = adapter.compute_model_predictions(
                model, task.dataset,
            )
            structural_predictions = adapter.compute_model_predictions(
                model, structural_dataset,
            )
            control_predictions = adapter.compute_model_predictions(
                model, control_dataset,
            )

            try:
                evaluation = evaluate_necessity(
                    original_predictions,
                    structural_predictions,
                    control_predictions,
                    structural_magnitude,
                    control_magnitude,
                    task.target,
                    magnitude_tolerance=self._magnitude_tolerance,
                )
                significance = compute_significance(
                    evaluation, seed=self._seed,
                )

                task_results[task.target.name] = {
                    "is_primary": task.is_primary,
                    "metric_name": evaluation.metric_name,
                    "targeted_effect_mean": evaluation.targeted_effect_mean,
                    "control_effect_mean": evaluation.control_effect_mean,
                    "necessity_gap_mean": evaluation.necessity_gap_mean,
                    "effect_size": significance.effect_size,
                    "ci_low": significance.ci_low,
                    "ci_high": significance.ci_high,
                    "p_value": significance.p_value,
                    "statistical_test": significance.test_name,
                    "n_samples": evaluation.n_samples,
                    "n_excluded": n_excluded,
                    "n_total_before_filtering": n_total,
                    "exclusion_rate": (
                        n_excluded / n_total if n_total else 0.0
                    ),
                }

            except ValueError as exc:
                task_results[task.target.name] = {
                    "is_primary": task.is_primary,
                    "error": str(exc),
                }

        primary_name = primary_tasks[0].target.name
        primary_result = task_results[primary_name]
        runtime = time.perf_counter() - start

        if "error" in primary_result:
            test_score, p_value, sample_size = 0.0, None, None
            notes = f"Primary target undefined: {primary_result['error']}"
        else:
            test_score = primary_result["necessity_gap_mean"]
            p_value = primary_result["p_value"]
            sample_size = primary_result["n_samples"]
            notes = f"Primary target: {primary_name}."

        metadata = {"primary_target": primary_name, "tasks": task_results}
        if "error" not in primary_result:
            metadata.update(
                {k: v for k, v in primary_result.items() if k != "is_primary"}
            )

        return CryptographicTestResult(
            test_name=self.name,
            evidence_direction=self.evidence_direction,
            baseline_score=0.0,
            test_score=test_score,
            performance_drop=test_score,
            relative_difference=test_score,
            runtime=runtime,
            p_value=p_value,
            sample_size=sample_size,
            notes=notes,
            metadata=metadata,
        )

    @property
    def supported_rationale(self) -> str:
        return (
            "Perturbing the declared structure produced a "
            "significantly larger output change than a matched "
            "control perturbation, supporting causal necessity "
            "of the structure identified by CE2/CE3."
        )

    @property
    def inconclusive_rationale(self) -> str:
        return (
            "Necessity gap is positive but the evidence "
            "was not statistically significant."
        )

    @property
    def unsupported_rationale(self) -> str:
        return (
            "The structural intervention did not produce a "
            "larger output change than the matched "
            "control intervention."
        )