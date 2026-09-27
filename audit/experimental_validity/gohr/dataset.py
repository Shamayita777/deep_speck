"""
Dataset generation, persistence, and provenance for the Gohr adapter.

Wraps gohr.speck.make_train_data. Per the frozen randomness rules,
dataset generation uses os.urandom() and is NOT seedable - every
DatasetBundle produced here carries exact_replay_available=False with
an explicit reason. This module never fabricates a dataset_seed.

Two distinct usage patterns are supported and must never be confused:

    - MATCHED datasets: one dataset instance generated once and reused
      across both arms of a paired experiment (H-EV-SHUFFLE,
      H-EV-REPRESENTATION), so that the only difference between arms is
      the manipulated factor.
    - INDEPENDENT datasets: a fresh instance generated per replicate
      (EV-BASELINE), each persisted and hashed separately.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from framework.provenance import array_sha256, dataset_provenance
from gohr import speck

EXACT_REPLAY_REASON = "dataset generation uses os.urandom(), which is not seedable"


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    role: str  # "train" | "validation" | "confirmatory_test" | "calibration"
    X: np.ndarray
    Y: np.ndarray
    rounds: int
    differential: tuple[int, int]
    n: int
    provenance: dict[str, Any]

    @property
    def x_hash(self) -> str:
        return array_sha256(self.X)

    @property
    def y_hash(self) -> str:
        return array_sha256(self.Y)

    @property
    def combined_hash(self) -> str:
        # A single hash over both arrays is what firewall.py seals against.
        return array_sha256(np.concatenate([self.X.astype(np.uint8), self.Y.reshape(-1, 1).astype(np.uint8)], axis=1))


def generate_dataset(
    *,
    n: int,
    rounds: int,
    differential: tuple[int, int],
    role: str,
    dataset_id: Optional[str] = None,
) -> DatasetBundle:
    """
    Generate one fresh dataset instance via gohr.speck.make_train_data.

    dataset_id is auto-generated (uuid4) if not supplied - it identifies
    this specific generated instance, not a reproducible seed (none
    exists for this generator).
    """
    dataset_id = dataset_id or str(uuid.uuid4())
    X, Y = speck.make_train_data(n, rounds, diff=differential)

    provenance = dataset_provenance(
        dataset_id=dataset_id,
        role=role,
        array_hashes={"X": array_sha256(X), "Y": array_sha256(Y)},
        generation_procedure="gohr.speck.make_train_data (ported verbatim from archive/speck.py)",
        generation_parameters={"n": n, "rounds": rounds, "differential": list(differential)},
        exact_replay_available=False,
        reason=EXACT_REPLAY_REASON,
    )
    return DatasetBundle(
        dataset_id=dataset_id,
        role=role,
        X=X,
        Y=Y,
        rounds=rounds,
        differential=differential,
        n=n,
        provenance=provenance,
    )


def persist_dataset(bundle: DatasetBundle, directory: str | Path) -> Path:
    """
    Persist a dataset instance to disk, named by its content hash so
    that accidental overwrites with different content are impossible
    (a hash collision would be required, which np.savez will not
    produce for distinct content).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bundle.role}_{bundle.dataset_id}_{bundle.combined_hash[:16]}.npz"
    np.savez_compressed(path, X=bundle.X, Y=bundle.Y)
    return path


def load_dataset(
    path: str | Path,
    *,
    dataset_id: str,
    role: str,
    rounds: int,
    differential: tuple[int, int],
    expected_combined_hash: Optional[str] = None,
) -> DatasetBundle:
    """
    Load a persisted dataset instance. If expected_combined_hash is
    supplied, verifies it and raises on mismatch (fail closed) rather
    than silently loading substituted or corrupted data.
    """
    path = Path(path)
    with np.load(path) as data:
        X, Y = data["X"], data["Y"]

    provenance = dataset_provenance(
        dataset_id=dataset_id,
        role=role,
        array_hashes={"X": array_sha256(X), "Y": array_sha256(Y)},
        generation_procedure="gohr.speck.make_train_data (loaded from persisted file)",
        generation_parameters={"n": len(Y), "rounds": rounds, "differential": list(differential)},
        exact_replay_available=False,
        reason=EXACT_REPLAY_REASON,
    )
    bundle = DatasetBundle(
        dataset_id=dataset_id, role=role, X=X, Y=Y, rounds=rounds,
        differential=differential, n=len(Y), provenance=provenance,
    )
    if expected_combined_hash is not None and bundle.combined_hash != expected_combined_hash:
        raise ValueError(
            f"Loaded dataset at {path} has combined hash {bundle.combined_hash}, "
            f"expected {expected_combined_hash}. Refusing to use a dataset that does "
            "not match its recorded provenance."
        )
    return bundle
