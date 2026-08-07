"""
Representation Interpretation Test
====================================

Cryptographic Evidence Phase CE3.

Scientific Goal
---------------
Determine whether a specific, independently-defined cryptographic
quantity is decodable from a trained model's hidden representation
to a degree that exceeds what an appropriately controlled baseline
representation achieves for the same quantity.

Evidence Logic
--------------
Representations are extracted from the trained ("real") model and
from an adapter-supplied control model, over the same evaluation
dataset. Both representation sets are probed for the same
adapter-declared target using an identical cross-validation
protocol (same folds, same per-fold seeds), and the resulting
selectivity (real probe score - control probe score, computed
per fold) is the evidence CE3 reports.

This module is framework-level only. It contains no knowledge of
Speck, differential cryptanalysis, neural architectures, or which
research paper's model is being audited. All probing methodology
(fold generation, default probe selection, scoring metric,
significance testing) lives in `probe.evaluation`, per Framework
Principle 5 (evidence generation is separated from scientific
interpretation, and separated again from this orchestration
layer).

Scope
-----
The adapter supplies one or more RepresentationTask objects,
each coupling an evaluation dataset with its corresponding
cryptographic target.

Each task is evaluated independently. Tasks marked as
`is_primary=True` provide the primary scientific evidence
reported by CE3, while auxiliary tasks (e.g. calibration
targets) validate the probing methodology.
"""

from __future__ import annotations

import time

from audit.cryptography.adapters.base import CryptographicAdapter
from audit.cryptography.probe.evaluation import (
    compute_significance,
    evaluate_selectivity,
)
from audit.cryptography.results import CryptographicTestResult
from audit.cryptography.test.base import CryptographicTest, EvidenceDirection


class RepresentationInterpretationTest(CryptographicTest):
    """
    Cryptographic Evidence Test CE3.

    Compares control-normalized decodability (selectivity) of an
    adapter-declared cryptographic target from a frozen model's
    hidden representation, relative to an adapter-supplied control
    model's representation. Positive, statistically significant
    selectivity supports the hypothesis that the trained model's
    representation encodes the declared quantity beyond what the
    control representation does.
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_repeats: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__(
            name="Representation Interpretation",
            description=(
                "Evaluate whether a frozen model's hidden "
                "representation encodes an independently declared "
                "cryptographic quantity beyond what a control "
                "representation encodes."
            ),
            hypothesis=(
                "The declared cryptographic target should be more "
                "decodable from the trained model's representation "
                "than from the control model's representation."
            ),
            evidence_direction=EvidenceDirection.HIGHER_IS_BETTER,
        )
        self._n_splits = n_splits
        self._n_repeats = n_repeats
        self._seed = seed

    def validate(self, adapter: CryptographicAdapter) -> None:
        if not isinstance(adapter, CryptographicAdapter):
            raise TypeError(
                "adapter must implement CryptographicAdapter."
            )

    def run(self, adapter: CryptographicAdapter) -> CryptographicTestResult:
        self.validate(adapter)

        start = time.perf_counter()

        # ---------------------------------------------
        # Frozen model and control model
        # ---------------------------------------------

        model = adapter.load()
        control_model = adapter.provide_control_model()
        test_score = 0.0
        p_value = None
        sample_size = None
        notes = ""
        metadata = {}
        tasks = adapter.generate_representation_tasks()
        calibration_evaluation = None
        calibration_significance = None

        primary_evaluation = None
        primary_significance = None

        for task in tasks:

            dataset = task.dataset
            target = task.target

            real_representations = adapter.extract_representations(
                model,
                dataset,
            )

            control_representations = adapter.extract_representations(
                control_model,
                dataset,
            )

            try:
                evaluation = evaluate_selectivity(
                    real_representations,
                    control_representations,
                    target,
                    n_splits=self._n_splits,
                    n_repeats=self._n_repeats,
                    seed=self._seed,
                )

                significance = compute_significance(
                    evaluation,
                    seed=self._seed,
                )

                if task.is_primary:

                    primary_evaluation = evaluation
                    primary_significance = significance

                else:

                    calibration_evaluation = evaluation
                    calibration_significance = significance

            except ValueError as exc:

                if not task.is_primary:
                    raise RuntimeError(
                        "Calibration probing failed."
                    ) from exc

                test_score = 0.0
                p_value = None
                sample_size = int(real_representations.shape[0])

                notes = (
                    "Selectivity is undefined "
                    f"({exc}); recorded as no evidence of "
                    "control-normalized decodability."
                )

                metadata = {}
        if calibration_evaluation is None:
            raise RuntimeError(
                "Calibration representation task was not executed."
            )

        if primary_evaluation is None:
            raise RuntimeError(
                "Primary representation task was not executed."
            )

        calibration_supported = (
            calibration_evaluation.selectivity_mean > 0.0
            and calibration_significance.p_value < 0.05
        )

        primary_supported = (
            primary_evaluation.selectivity_mean > 0.0
            and primary_significance.p_value < 0.05
        )

        test_score = primary_evaluation.selectivity_mean

        p_value = primary_significance.p_value

        sample_size = primary_significance.n_pairs

        notes = "Representation probing completed successfully."

        metadata = {

            "calibration_selectivity":
                calibration_evaluation.selectivity_mean,

            "calibration_p_value":
                calibration_significance.p_value,

            "calibration_validated":
                calibration_supported,

            "target_name":
                primary_evaluation.target.name,

            "target_type":
                primary_evaluation.target.target_type.value,

            "metric_name":
                primary_evaluation.metric_name,

            "real_probe_score":
                primary_evaluation.real_score_mean,

            "control_probe_score":
                primary_evaluation.control_score_mean,

            "effect_size":
                primary_significance.effect_size,

            "ci_low":
                primary_significance.ci_low,

            "ci_high":
                primary_significance.ci_high,

            "statistical_test":
                primary_significance.test_name,

            "statistical_alternative":
                primary_significance.alternative,

            "n_folds":
                primary_evaluation.n_splits,

            "primary_supported":
                primary_supported,
        }

        runtime = time.perf_counter() - start
        baseline_score = 0.0
        performance_drop = test_score - baseline_score
        relative_difference = test_score

        return CryptographicTestResult(
            test_name=self.name,
            evidence_direction=self.evidence_direction,
            baseline_score=baseline_score,
            test_score=test_score,
            performance_drop=performance_drop,
            relative_difference=relative_difference,
            runtime=runtime,
            p_value=p_value,
            sample_size=sample_size,
            notes=notes,
            metadata=metadata,
        )

    @property
    def supported_rationale(self) -> str:
        return (
            "Observed selectivity indicates the declared target "
            "is more decodable from the trained model's "
            "representation than from the control representation."
        )

    @property
    def inconclusive_rationale(self) -> str:
        return (
            "Observed selectivity is positive but weaker than "
            "the required threshold."
        )

    @property
    def unsupported_rationale(self) -> str:
        return (
            "Observed selectivity is negative or zero, indicating "
            "no evidence that the trained model's representation "
            "encodes the declared target beyond the control."
        )