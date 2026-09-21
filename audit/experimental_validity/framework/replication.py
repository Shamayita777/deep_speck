"""
Replicate-set management for Experimental Validity.

The statistical unit is the independent training run / model replicate.
This module collects ReplicateOutcome records (framework.failures) for
one condition and exposes them for statistics, with no code path that
could construct a "replicate" from individual ciphertext examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from framework.failures import ReplicateOutcome, ReplicateStatus, enforce_minimum_valid_replicates


@dataclass
class ReplicateSet:
    condition_id: str
    outcomes: list[ReplicateOutcome] = field(default_factory=list)

    def add(self, outcome: ReplicateOutcome) -> None:
        if outcome.condition_id != self.condition_id:
            raise ValueError(
                f"Outcome condition_id {outcome.condition_id!r} does not match "
                f"ReplicateSet condition_id {self.condition_id!r}."
            )
        self.outcomes.append(outcome)

    def valid_outcomes(self) -> list[ReplicateOutcome]:
        return [o for o in self.outcomes if o.status == ReplicateStatus.VALID]

    def failed_outcomes(self) -> list[ReplicateOutcome]:
        return [o for o in self.outcomes if o.status == ReplicateStatus.FAILED]

    def valid_metric_array(self) -> np.ndarray:
        """
        Array of metric values from VALID replicates only, in the order
        they were added. Failed replicates are excluded from the
        analysis array but remain fully visible via failed_outcomes()
        and the summary from check_sufficiency() - they are never
        removed from the reporting, only from the numeric array fed to
        statistics functions (which cannot accept a None metric).
        """
        return np.array([o.metric_value for o in self.valid_outcomes()], dtype=float)

    def check_sufficiency(self, minimum_required: int) -> tuple[bool, dict]:
        return enforce_minimum_valid_replicates(self.outcomes, minimum_required)
