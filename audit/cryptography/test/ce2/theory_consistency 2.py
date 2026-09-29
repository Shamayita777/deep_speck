"""
Theory Consistency Test
========================

Cryptographic Evidence Phase CE2.

Scientific Goal
---------------
Determine whether the behaviour of an already-trained
cryptographic model agrees with an independent, model-free
cryptographic theory prediction.

The model is frozen: it is loaded, not retrained. This isolates
the learned representation from optimization dynamics (random
initialization, SGD randomness, learning-rate schedule), which
are irrelevant to what the frozen function has already learned.

This module is framework-level only. It contains no knowledge of
Speck, differential cryptanalysis, or neural architectures.
"""

from __future__ import annotations

import time

from audit.cryptography.adapters.base import CryptographicAdapter
from audit.cryptography.results import CryptographicTestResult
from audit.cryptography.test.base import CryptographicTest, EvidenceDirection


class TheoryConsistencyTest(CryptographicTest):
    """
    Cryptographic Evidence Test CE2.

    Compares a frozen model's predictions against an independent,
    model-free theoretical reference, using rank correlation. A
    positive correlation supports the hypothesis that the model's
    learned function agrees with established cryptographic theory
    for the property under test.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Theory Consistency",
            description=(
                "Evaluate whether a frozen model's predictions "
                "agree with an independent theoretical reference."
            ),
            hypothesis=(
                "The model's predictions should be positively "
                "rank-correlated with the theoretical reference "
                "derived independently of the model."
            ),
            evidence_direction=EvidenceDirection.HIGHER_IS_BETTER,
        )

    def validate(self, adapter: CryptographicAdapter) -> None:
        if not isinstance(adapter, CryptographicAdapter):
            raise TypeError(
                "adapter must implement CryptographicAdapter."
            )

    def run(self, adapter: CryptographicAdapter) -> CryptographicTestResult:
        self.validate(adapter)

        start = time.perf_counter()

        # ---------------------------------------------
        # Frozen model
        # ---------------------------------------------

        model = adapter.load()

        # ---------------------------------------------
        # Theory dataset (samples + theoretical reference
        # generated together, as required for correct
        # per-sample pairing)
        # ---------------------------------------------

        dataset = adapter.generate_theory_dataset()

        theoretical_reference = adapter.compute_theoretical_reference(
            dataset
        )

        model_predictions = adapter.compute_model_predictions(
            model, dataset,
        )

        # ---------------------------------------------
        # Theory Consistency Score
        # ---------------------------------------------

        try:
            correlation = adapter.compute_theory_consistency(
                theoretical_reference, model_predictions,
            )

            test_score = correlation.statistic
            p_value = correlation.p_value
            sample_size = correlation.n
            notes = (
                "Theory Consistency Score computed successfully."
            )

        except ValueError as exc:

            test_score = 0.0
            p_value = None
            sample_size = int(len(theoretical_reference))
            notes = (
                "Theory Consistency Score is undefined "
                f"({exc}); recorded as no evidence of "
                "theory-consistent structure."
            )

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
        )

    @property
    def supported_rationale(self) -> str:
        return (
            "Observed rank correlation supports "
            "agreement between the frozen model "
            "and the theoretical reference."
        )

    @property
    def inconclusive_rationale(self) -> str:
        return (
            "Observed positive rank correlation "
            "is weaker than the required threshold."
        )

    @property
    def unsupported_rationale(self) -> str:
        return (
            "Observed rank correlation is negative, "
            "indicating disagreement between the "
            "frozen model and the theoretical "
            "reference."
        )