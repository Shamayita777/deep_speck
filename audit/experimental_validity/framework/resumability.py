"""
Resumability for Experimental Validity.

A crash/restart must never silently change the scientific experiment:
resuming replays the same configuration hash, the same seeds, and the
same replicate plan. If the recorded configuration hash of a resume
attempt does not match the original run's, resumption is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from framework.provenance import utc_timestamp


class ResumeViolation(RuntimeError):
    pass


@dataclass
class RunState:
    experiment_id: str
    replicate_id: str
    stage: str
    completed_stages: list[str]
    config_hash: str
    checkpoint_hash: Optional[str]
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "replicate_id": self.replicate_id,
            "stage": self.stage,
            "completed_stages": self.completed_stages,
            "config_hash": self.config_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "timestamp": self.timestamp,
        }


class ResumeLedger:
    """Append-only, on-disk record of run states and resume events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def record_state(self, state: RunState) -> None:
        self._append({"event": "state", **state.to_dict()})

    def record_resume_event(
        self,
        *,
        experiment_id: str,
        replicate_id: str,
        previous_config_hash: str,
        resumed_config_hash: str,
    ) -> None:
        """
        Record a resume event. Raises ResumeViolation if the
        configuration hash has changed since the run was last
        checkpointed - a resumed run must use the identical
        configuration, never a silently modified one.
        """
        if previous_config_hash != resumed_config_hash:
            raise ResumeViolation(
                f"Refusing to resume {experiment_id}/{replicate_id}: "
                f"configuration hash changed from {previous_config_hash} to "
                f"{resumed_config_hash}. A resumed run must not change the "
                "scientific experiment."
            )
        self._append(
            {
                "event": "resume",
                "experiment_id": experiment_id,
                "replicate_id": replicate_id,
                "previous_config_hash": previous_config_hash,
                "resumed_config_hash": resumed_config_hash,
                "timestamp": utc_timestamp(),
            }
        )

    def last_state_for(self, experiment_id: str, replicate_id: str) -> Optional[dict[str, Any]]:
        last: Optional[dict[str, Any]] = None
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if (
                    record.get("event") == "state"
                    and record.get("experiment_id") == experiment_id
                    and record.get("replicate_id") == replicate_id
                ):
                    last = record
        return last
