"""
Failure handling for Experimental Validity.

The system must fail closed: a failed replicate is recorded, never
silently dropped from the denominator, and remains visible in every
downstream report. This module defines the taxonomy of failure reasons
and the replicate-level outcome record used throughout the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FailureReason(str, Enum):
    NAN = "nan"
    OOM = "oom"
    TIMEOUT = "timeout"
    DIVERGENCE = "divergence"
    MISSING_CHECKPOINT = "missing_checkpoint"
    CORRUPTED_DATASET = "corrupted_dataset"
    INVALID_METRIC = "invalid_metric"
    INTERRUPTED = "interrupted"
    MISSING_PROVENANCE = "missing_provenance"


class ReplicateStatus(str, Enum):
    VALID = "valid"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass
class ReplicateOutcome:
    """
    Outcome of one independent training-run / model replicate.

    This is the atomic unit that all EV statistics operate on. Nothing
    in this framework is permitted to construct a "replicate" out of
    individual ciphertext examples, batches, or predictions - the
    statistical unit is always one independently trained model.
    """

    replicate_id: str
    condition_id: str
    run_id: str
    status: ReplicateStatus
    metric_value: Optional[float] = None
    failure_reason: Optional[FailureReason] = None
    failure_detail: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == ReplicateStatus.VALID and self.metric_value is None:
            raise ValueError(
                f"Replicate {self.replicate_id} marked VALID but has no metric_value. "
                "Refusing to construct an inconsistent replicate outcome."
            )
        if self.status == ReplicateStatus.FAILED and self.failure_reason is None:
            raise ValueError(
                f"Replicate {self.replicate_id} marked FAILED but no failure_reason "
                "was supplied. Failure reasons must always be recorded."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicate_id": self.replicate_id,
            "condition_id": self.condition_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "metric_value": self.metric_value,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "failure_detail": self.failure_detail,
            "provenance": self.provenance,
        }


def detect_failure_from_history(history: dict[str, list[float]]) -> Optional[FailureReason]:
    """
    Inspect a Keras-style training history dict for NaN/divergence.

    Returns the FailureReason if the run should be treated as failed,
    or None if the history looks valid. This is a conservative check:
    it only flags unambiguous NaN/inf occurrences, and does not attempt
    to guess at "soft" divergence (e.g. slow degradation), which is a
    scientific question for the experiment's own analysis, not a
    software-level failure.
    """
    import math

    for key, values in history.items():
        for v in values:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return FailureReason.INVALID_METRIC
            if math.isnan(fv):
                return FailureReason.NAN
            if math.isinf(fv):
                return FailureReason.DIVERGENCE
    return None


def enforce_minimum_valid_replicates(
    outcomes: list[ReplicateOutcome],
    minimum_required: int,
) -> tuple[bool, dict[str, Any]]:
    """
    Check whether the minimum required number of VALID replicates was
    reached. Never reduces the denominator: `requested`, `valid`, and
    `failed` counts are always computed against the full outcome list.

    Returns (sufficient, summary_dict). Callers must map
    `sufficient=False` to decision=INCONCLUSIVE, never to PASS/FAIL/
    SUPPORTED/NOT_SUPPORTED.
    """
    valid = [o for o in outcomes if o.status == ReplicateStatus.VALID]
    failed = [o for o in outcomes if o.status == ReplicateStatus.FAILED]
    not_run = [o for o in outcomes if o.status == ReplicateStatus.NOT_RUN]

    summary = {
        "requested_replicates": len(outcomes),
        "valid_replicates": len(valid),
        "failed_replicates": len(failed),
        "not_run_replicates": len(not_run),
        "minimum_required": minimum_required,
        "failure_reasons": [
            {"replicate_id": o.replicate_id, "reason": o.failure_reason.value if o.failure_reason else None}
            for o in failed
        ],
    }
    return len(valid) >= minimum_required, summary
