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
The current adapter interface declares exactly one target per
`declare_target` call and returns exactly one representation
matrix per `extract_representations` call. This test therefore
evaluates one target against one representation layer per run.
Running CE3 against multiple declared targets or multiple hidden
layers means instantiating and running this test once per
target/layer from the experiment script -- it is not handled
inside this module, since the adapter interface does not
currently expose either as a list.
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

        # ---------------------------------------------
        # Evaluation dataset and declared target.
        #
        # declare_target receives only the dataset, never the
        # model, enforcing model-independence of the target at
        # the interface level rather than by adapter convention.
        # ---------------------------------------------

        dataset = adapter.generate_representation_dataset()
        target = adapter.declare_targets(dataset)

        # ---------------------------------------------
        # Representation extraction -- real and control arms,
        # over the identical evaluation dataset.
        # ---------------------------------------------

        real_representations = adapter.extract_representations(
            model, dataset,
        )
        control_representations = adapter.extract_representations(
            control_model, dataset,
        )

        # ---------------------------------------------
        # Selectivity and its statistical significance.
        #
        # Only these two calls are guarded: extraction above
        # either succeeds or is an adapter-level failure that
        # should propagate, not be silently absorbed here. What
        # can legitimately be undefined is the *evidence itself*
        # -- e.g. a degenerate target, or all-identical per-fold
        # selectivity values -- which is exactly what
        # evaluate_selectivity / compute_significance raise
        # ValueError for.
        # ---------------------------------------------

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
                evaluation, seed=self._seed,
            )

            test_score = evaluation.selectivity_mean
            p_value = significance.p_value
            sample_size = significance.n_pairs
            notes = "Selectivity computed successfully."
            metadata = {
                "target_name": target.name,
                "target_type": target.target_type.value,
                "metric_name": evaluation.metric_name,
                "real_probe_score": evaluation.real_score_mean,
                "control_probe_score": evaluation.control_score_mean,
                "effect_size": significance.effect_size,
                "ci_low": significance.ci_low,
                "ci_high": significance.ci_high,
                "statistical_test": significance.test_name,
                "statistical_alternative": significance.alternative,
                "n_folds": evaluation.n_splits,
            }

        except ValueError as exc:

            test_score = 0.0
            p_value = None
            sample_size = int(real_representations.shape[0])
            notes = (
                "Selectivity is undefined "
                f"({exc}); recorded as no evidence of "
                "control-normalized decodability."
            )
            metadata = {}

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