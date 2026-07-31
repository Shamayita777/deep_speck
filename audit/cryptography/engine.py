"""
Cryptographic Evidence Engine
=============================

Generic execution engine for Cryptographic Evidence tests.

The engine is responsible for orchestrating test execution while
remaining completely independent of any specific cryptographic
experiment.

Responsibilities
----------------
• Execute a cryptographic test
• Measure execution time
• Log execution events
• Handle execution errors
• Return the test result

The engine contains no knowledge of any specific audit such as
Signal Destruction, Theory Consistency, or Representation
Interpretation.
"""

from __future__ import annotations

import logging
import time

from .adapters.base import CryptographicAdapter
from .results import CryptographicTestResult
from .test.base import CryptographicTest


logger = logging.getLogger(__name__)


class CryptographicEngine:
    """
    Generic execution engine for cryptographic evidence tests.
    """

    def __init__(
        self,
        *,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        """
        Initialise the execution engine.
        """

        self.logger = logger_instance or logger

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------

    def execute(
        self,
        test: CryptographicTest,
        adapter: CryptographicAdapter,
    ) -> CryptographicTestResult:
        """
        Execute a cryptographic test.

        Parameters
        ----------
        test
            Test to execute.

        adapter
            Cryptographic adapter.

        Returns
        -------
        CryptographicTestResult
            Result produced by the test.

        Raises
        ------
        Exception
            Re-raises any exception produced during execution
            after logging it.
        """

        self.logger.info(
            "Starting test: %s",
            test.name,
        )

        start_time = time.perf_counter()

        try:

            test.validate(adapter)

            result = test.run(adapter)

        except Exception:

            self.logger.exception(
                "Test '%s' failed.",
                test.name,
            )

            raise

        elapsed = time.perf_counter() - start_time

        self.logger.info(
            "Completed test '%s' in %.3f seconds.",
            test.name,
            elapsed,
        )

        return result

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "Cryptographic Engine"

    @property
    def description(self) -> str:
        return (
            "Generic execution engine for "
            "Cryptographic Evidence tests."
        )