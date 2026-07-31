"""
Signal Destruction Test
=======================

Cryptographic Evidence Phase CE1.

This module implements the Signal Destruction audit.

Scientific Goal
---------------
Determine whether predictive performance depends upon the
presence of genuine cryptographic signal.

The audit compares model performance on:

    • a baseline experiment

and

    • a signal-destroyed experiment

If performance degrades substantially after destroying the
cryptographic relationship, the model is likely exploiting
meaningful cryptographic structure rather than unrelated
statistical artefacts.

This module is framework-level only.

It contains no knowledge of

    • Gohr
    • Speck
    • AES
    • plaintexts
    • ciphertexts
    • differential cryptanalysis
    • neural architectures
"""

from __future__ import annotations

import time

from audit.cryptography.adapters.base import CryptographicAdapter
from audit.cryptography.results import CryptographicTestResult
from audit.cryptography.test.base import CryptographicTest


class SignalDestructionTest(CryptographicTest):
    """
    Cryptographic Evidence Test CE1.

    Signal Destruction evaluates whether model performance
    depends upon the cryptographic relationship present in
    the training data.

    The experiment consists of

        baseline experiment

            versus

        signal-destroyed experiment

    while keeping all remaining conditions unchanged.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Signal Destruction",
            description=(
                "Evaluate whether predictive performance "
                "depends on meaningful cryptographic signal."
            ),
            hypothesis=(
                "Destroying the cryptographic signal should "
                "cause measurable degradation in model "
                "performance."
            ),
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate(
        self,
        adapter: CryptographicAdapter,
    ) -> None:
        """
        Validate that the supplied adapter satisfies the
        CryptographicAdapter interface.
        """

        if not isinstance(adapter, CryptographicAdapter):
            raise TypeError(
                "adapter must implement CryptographicAdapter."
            )

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------

    def run(
        self,
        adapter: CryptographicAdapter,
    ) -> CryptographicTestResult:
        """
        Execute the Signal Destruction audit.

        Parameters
        ----------
        adapter
            Concrete cryptographic adapter.

        Returns
        -------
        CryptographicTestResult
            Immutable experimental result.
        """

        self.validate(adapter)

        start = time.perf_counter()

        # ---------------------------------------------
        # Baseline experiment
        # ---------------------------------------------

        baseline_dataset = adapter.generate_baseline_dataset()

        baseline_model = adapter.train(
            baseline_dataset,
        )

        baseline_score = adapter.evaluate(
            baseline_model,
        )

        # ---------------------------------------------
        # Signal-destroyed experiment
        # ---------------------------------------------

        destroyed_dataset = (
            adapter.generate_signal_destroyed_dataset()
        )

        destroyed_model = adapter.train(
            destroyed_dataset,
        )

        destroyed_score = adapter.evaluate(
            destroyed_model,
        )

        runtime = time.perf_counter() - start

        performance_drop = (
            baseline_score - destroyed_score
        )

        if baseline_score == 0.0:
            relative_difference = 0.0
        else:
            relative_difference = (
                performance_drop
                / baseline_score
            ) 

        return CryptographicTestResult(
            test_name=self.name,
            baseline_score=baseline_score,
            test_score=destroyed_score,
            performance_drop=performance_drop,
            relative_difference=relative_difference,
            runtime=runtime,
            notes=(
                "Baseline and signal-destroyed experiments "
                "completed successfully."
            ),
        )