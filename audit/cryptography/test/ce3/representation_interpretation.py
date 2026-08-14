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
For each independently generated evaluation replicate,
representations are extracted from the trained ("real") model
and from an adapter-supplied control model over the same
replicate-specific dataset. Both representation sets are
probed for the same adapter-declared target using identical
cross-validation folds and per-fold probe seeds.

The mean selectivity within each replicate is retained as one
replicate-level observation. Statistical significance is then
computed across independently generated evaluation replicates,
not across individual cross-validation folds.

The cross-validation folds therefore stabilize each replicate's
point estimate but are not treated as independent statistical
observations.

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
    compute_significance_over_replicates,
    evaluate_selectivity_replicates,
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
        n_replicates: int = 20,
        n_splits_per_replicate: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(
            name="Representation Interpretation",
            description=(
                "Evaluate whether a frozen model's hidden "
                "representation encodes an independently declared "
                "cryptographic quantity beyond what a control "
                "representation encodes, across independently "
                "generated evaluation replicates."
            ),
            hypothesis=(
                "The declared cryptographic target should be more "
                "decodable from the trained model's representation "
                "than from the control model's representation, "
                "consistently across independent replicates."
            ),
            evidence_direction=EvidenceDirection.HIGHER_IS_BETTER,
        )

        self._n_replicates = n_replicates
        self._n_splits_per_replicate = n_splits_per_replicate
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
        control_model = adapter.provide_control_model()

        primary_replicated = evaluate_selectivity_replicates(
            task_factory=adapter.generate_primary_representation_task,
            adapter=adapter,
            model=model,
            control_model=control_model,
            n_replicates=self._n_replicates,
            n_splits=self._n_splits_per_replicate,
            seed=self._seed,
        )

        calibration_replicated = evaluate_selectivity_replicates(
            task_factory=adapter.generate_calibration_representation_task,
            adapter=adapter,
            model=model,
            control_model=control_model,
            n_replicates=self._n_replicates,
            n_splits=self._n_splits_per_replicate,
            seed=self._seed + 10_000,
        )

        primary_significance = compute_significance_over_replicates(
            primary_replicated, seed=self._seed,
        )
        calibration_significance = compute_significance_over_replicates(
            calibration_replicated, seed=self._seed,
        )

        calibration_supported = (
            calibration_replicated.selectivity_mean > 0.0
            and calibration_significance.p_value < 0.025  # Bonferroni, m=2
        )
        primary_supported = (
            primary_replicated.selectivity_mean > 0.0
            and primary_significance.p_value < 0.025
        )

        test_score = primary_replicated.selectivity_mean
        p_value = primary_significance.p_value
        sample_size = primary_significance.n_pairs  # == n_replicates, automatically

        metadata = {
            "calibration_selectivity": calibration_replicated.selectivity_mean,
            "calibration_p_value": calibration_significance.p_value,
            "calibration_validated": calibration_supported,
            "target_name": primary_replicated.target_name,
            "metric_name": primary_replicated.metric_name,
            "effect_size": primary_significance.effect_size,
            "ci_low": primary_significance.ci_low,
            "ci_high": primary_significance.ci_high,
            "statistical_test": primary_significance.test_name,
            "statistical_alternative": primary_significance.alternative,
            "n_replicates": primary_replicated.n_replicates,
            "n_splits_per_replicate": primary_replicated.n_splits_per_replicate,
            "statistical_unit": "independent_evaluation_replicate",
            "primary_supported": primary_supported,
        }

        runtime = time.perf_counter() - start
        baseline_score = 0.0

        return CryptographicTestResult(
            test_name=self.name,
            evidence_direction=self.evidence_direction,
            baseline_score=baseline_score,
            test_score=test_score,
            performance_drop=test_score - baseline_score,
            relative_difference=test_score,
            runtime=runtime,
            p_value=p_value,
            sample_size=sample_size,
            notes="Representation probing completed across independent replicates.",
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
            "Observed selectivity was positive, but the evidence "
            "did not reach the required statistical significance "
            "threshold."
        )

    @property
    def unsupported_rationale(self) -> str:
        return (
            "Observed selectivity is negative or zero, indicating "
            "no evidence that the trained model's representation "
            "encodes the declared target beyond the control."
        )