"""
Generic provenance infrastructure for Experimental Validity (EV).

Contains no Gohr-specific assumptions. Mirrors the hashing conventions
already established in the project's Dataset Integrity (D1-D5) common
module (audit.dataset.common.provenance) for cross-dimension consistency,
but is implemented independently so that EV does not import D's or CE's
code (see docs/independence.md for the rationale).

Per the frozen randomness rules for this project:
    - dataset generation may use os.urandom() and is therefore NOT
      exactly replayable from a recorded seed. This module never
      fabricates a seed for such data; callers must pass
      exact_replay_available=False and a reason in that case.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def utc_timestamp() -> str:
    """Current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 digest of arbitrary bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 digest of a file, read incrementally."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """
    SHA-256 digest committing to dtype, shape, and raw C-contiguous bytes
    of an array. Two arrays with the same numeric values but different
    dtype/shape are treated as different artifacts, since dtype/shape
    affect the actual computational representation fed to a model.
    """
    array = np.asarray(array)
    contiguous = np.ascontiguousarray(array)

    digest = hashlib.sha256()
    dtype_bytes = str(contiguous.dtype).encode("utf-8")
    shape_bytes = json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8")

    digest.update(len(dtype_bytes).to_bytes(8, "big"))
    digest.update(dtype_bytes)
    digest.update(len(shape_bytes).to_bytes(8, "big"))
    digest.update(shape_bytes)
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """
    Deterministic hash of a configuration mapping.

    Uses a canonical JSON serialization (sorted keys, fixed separators)
    so that the same logical configuration always hashes identically
    regardless of key insertion order.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    """
    Return the current git commit hash, or None if unavailable
    (e.g. not a git repository). Never raises.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def software_provenance(repo_dir: Optional[str] = None) -> dict[str, Any]:
    """
    Capture *actual installed* software versions rather than documented
    or assumed ones. Callers that also want to compare against a
    documented reference baseline must do so explicitly (see
    gohr.baseline.compare_software_environment) - this function does
    not silently assume a match.
    """
    versions: dict[str, Any] = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "os": platform.platform(),
        "hardware": platform.processor() or platform.machine(),
    }

    try:
        import tensorflow as tf  # type: ignore

        versions["tensorflow_version"] = tf.__version__
    except Exception:
        versions["tensorflow_version"] = None

    try:
        import keras  # type: ignore

        versions["keras_version"] = keras.__version__
    except Exception:
        versions["keras_version"] = None

    versions["git_commit"] = git_commit(repo_dir)
    return versions


def dataset_provenance(
    *,
    dataset_id: str,
    role: str,
    array_hashes: Mapping[str, str],
    generation_procedure: str,
    generation_parameters: Mapping[str, Any],
    exact_replay_available: bool,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build a standardized dataset provenance record.

    exact_replay_available must be an explicit boolean; when False,
    `reason` must be supplied (e.g. "dataset generation uses
    os.urandom()"). This function refuses to silently imply
    reproducibility that does not exist.
    """
    if not exact_replay_available and not reason:
        raise ValueError(
            "exact_replay_available=False requires an explicit reason; "
            "refusing to silently omit why exact replay is unavailable."
        )
    return {
        "dataset_id": dataset_id,
        "role": role,
        "array_hashes": dict(array_hashes),
        "generation_procedure": generation_procedure,
        "generation_parameters": dict(generation_parameters),
        "exact_replay_available": bool(exact_replay_available),
        "reason": reason,
        "recorded_at_utc": utc_timestamp(),
    }
