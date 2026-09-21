"""
Generic run manifest - the record every individual run (a training run,
an evaluation run, a statistical-analysis run) must produce, regardless
of which pillar or case study it belongs to.

Knows nothing about Gohr/Speck/TensorFlow specifically, though the
environment capture helper here will happily report TensorFlow's
version if it happens to be importable - it does not assume it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from audit.common.ids import validate_identifier
from audit.common.outcomes import ExecutionMode, RunStatus
from audit.common.provenance import utc_timestamp


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def working_tree_dirty(repo_dir: Optional[str] = None) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0:
            return None
        return len(result.stdout.strip()) > 0
    except Exception:
        return None


def capture_environment(repo_dir: Optional[str] = None) -> dict[str, Any]:
    """
    Capture ACTUAL installed/observed environment info. Never guesses at
    a framework version - if a package is not importable, its field is
    explicitly None, not omitted and not assumed.
    """
    env: dict[str, Any] = {
        "python_version": sys.version,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,  # populated by a pillar-specific probe if relevant (e.g. tf.config.list_physical_devices)
        "git_commit": git_commit(repo_dir),
        "working_tree_dirty": working_tree_dirty(repo_dir),
    }
    for pkg in ("numpy", "scipy", "tensorflow", "keras"):
        try:
            module = __import__(pkg)
            env[f"{pkg}_version"] = getattr(module, "__version__", None)
        except Exception:
            env[f"{pkg}_version"] = None
    return env


@dataclass
class RunManifest:
    run_id: str
    experiment_id: str
    condition_id: Optional[str]
    replicate_id: Optional[str]
    claim_ids: list[str]
    hypothesis_ids: list[str]
    execution_mode: ExecutionMode
    config_hash: str
    preregistration_hash: Optional[str]
    dataset_snapshot_id: Optional[str]
    dataset_hashes: dict[str, str]
    code_hashes: dict[str, str]
    model_source_hash: Optional[str]
    environment: dict[str, Any]
    seeds: dict[str, Optional[int]]
    command: str
    start_time_utc: str
    end_time_utc: Optional[str] = None
    exit_code: Optional[int] = None
    status: RunStatus = RunStatus.SKIPPED
    checkpoint: Optional[dict[str, Any]] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, context="run_id")
        validate_identifier(self.experiment_id, context="experiment_id")

    def mark_completed(self, *, exit_code: int, metrics: dict[str, Any]) -> None:
        self.end_time_utc = utc_timestamp()
        self.exit_code = exit_code
        self.metrics = metrics
        self.status = RunStatus.COMPLETED if exit_code == 0 else RunStatus.FAILED

    def mark_aborted(self, *, reason: str) -> None:
        self.end_time_utc = utc_timestamp()
        self.status = RunStatus.ABORTED
        self.limitations.append(f"Run aborted: {reason}")

    def mark_invalid(self, *, reason: str) -> None:
        self.end_time_utc = utc_timestamp()
        self.status = RunStatus.INVALID
        self.limitations.append(f"Run marked invalid: {reason}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "experiment_id": self.experiment_id,
            "condition_id": self.condition_id, "replicate_id": self.replicate_id,
            "claim_ids": list(self.claim_ids), "hypothesis_ids": list(self.hypothesis_ids),
            "execution_mode": self.execution_mode.value, "config_hash": self.config_hash,
            "preregistration_hash": self.preregistration_hash,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "dataset_hashes": dict(self.dataset_hashes), "code_hashes": dict(self.code_hashes),
            "model_source_hash": self.model_source_hash, "environment": self.environment,
            "seeds": dict(self.seeds), "command": self.command,
            "start_time_utc": self.start_time_utc, "end_time_utc": self.end_time_utc,
            "exit_code": self.exit_code, "status": self.status.value,
            "checkpoint": self.checkpoint, "metrics": self.metrics, "statistics": self.statistics,
            "stdout_path": self.stdout_path, "stderr_path": self.stderr_path,
            "warnings": list(self.warnings), "limitations": list(self.limitations),
        }
