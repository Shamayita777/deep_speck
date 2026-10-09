"""
D5 — Training-Data Scaling Audit.

Scientific design
------------------
D5 estimates how predictive performance changes with training-set size
while holding the dataset-generation mechanism, model architecture,
optimizer/training procedure, epoch budget, and fixed held-out test set
constant.

The strengthened design uses BOTH:

1. Independent replicate datasets:
   each replicate generates one independent maximum-size dataset.

2. Nested prefixes within each replicate:
   the 2.5M, 5M, 7.5M and 10M conditions are the prefixes of that same
   replicate's 10M dataset.

This separates:
    - within-dataset sample-size effects; and
    - between-dataset/model-seed variation.

The model initialization seed is held fixed across sizes within a replicate,
so paired differences within a replicate are directly interpretable as
sample-size effects. Replicates have independent seeds.

A single fixed held-out test set is shared by all conditions.

D5 does NOT establish cryptographic learning, a universal scaling law, or
causality beyond the controlled experimental factors. The smallest declared
training size is the baseline condition for all paired contrasts. The
replicate-wise scaling slope against log10(training samples) is a
SECONDARY, DESCRIPTIVE diagnostic; it does not override or replace the
primary paired-contrast analysis, and it is not a causal or
cryptographic-learning claim.

Persistence and integrity contract (schema 2.1)
------------------------------------------------
D5's dataset generator (as used by the Gohr/Speck adapter) is NOT guaranteed
deterministic with respect to the audit seed: it may draw from an OS
entropy source. Consequently:

    - the integer audit/replicate seed does NOT control dataset content;
    - exact historical dataset replay from that seed is NOT available;
    - the persisted dataset files, together with cryptographic hashes over
      both their file bytes and their logical (dtype+shape+content) bytes,
      ARE the replay/resume object.

Once a replicate's maximum-size dataset, or the fixed test dataset, has been
persisted in full, it is treated as immutable for the run:

    - it is loaded and hash-verified, never regenerated;
    - every declared nested-prefix size has its own persisted content hash,
      computed from the persisted maximum-size arrays and re-verified on
      every load;
    - a PARTIALLY persisted dataset (some but not all of its files present)
      is an integrity failure, not a signal to regenerate -- the run
      refuses to proceed and asks for explicit cleanup or a new output
      directory;
    - a per-condition checkpoint/state file records the exact configuration
      hash and the exact dataset/test content hashes it was produced
      against, and resuming a condition re-verifies all of them before any
      further training occurs.

Nothing above weakens or replaces the existing D5 statistical engine
(bootstrap mean confidence intervals, paired t-tests against the baseline
size, Holm step-down correction across the size contrasts, and replicate-
wise scaling slopes), which is preserved as-is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import math
import os

import numpy as np
from keras.callbacks import Callback
from keras.models import load_model


SCHEMA_VERSION = "2.1"

# Number of rows hashed per chunk when computing a "logical" content hash of
# a (possibly very large, possibly memory-mapped) array. Bounds peak extra
# memory regardless of total array size; does not change the hash value.
_HASH_CHUNK_ROWS = 200_000

# Generic (adapter-independent) default provenance text. Callers (e.g. the
# Gohr driver) are expected to override these with an accurate, specific
# description of their own generator's determinism properties. The generic
# framework defaults are deliberately conservative: they never claim
# deterministic replay unless a caller explicitly says so.
_DEFAULT_DATASET_REPLAY_REASON = (
    "This D5 run was not given an explicit dataset-determinism statement by "
    "its caller. Treat exact historical dataset replay as unavailable unless "
    "the adapter/driver documentation states otherwise."
)
_DEFAULT_REPLICATE_SEED_ROLE = (
    "audit_RNG_stream; the caller did not declare whether this seed controls "
    "dataset generation -- treat it as not controlling dataset content unless "
    "documented otherwise."
)


@dataclass
class D5Observation:
    replicate: int
    training_samples: int
    seed: int
    test_samples: int
    test_accuracy: float
    test_loss: float
    completed_epochs: int
    total_epochs: int


@dataclass
class D5Result:
    experiment: str
    training_sizes: list[int]
    replicates: int
    observations: list[D5Observation]
    summary: dict[str, Any]
    pairwise: dict[str, Any]
    scaling: dict[str, Any]
    manifest: dict[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash of the file's raw on-disk bytes (including the .npy header)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _array_logical_sha256(array: np.ndarray, chunk_rows: int = _HASH_CHUNK_ROWS) -> str:
    """Content hash of an array's logical value: dtype + shape + raw
    contiguous bytes -- independent of .npy file-format/header details, and
    therefore able to detect a content change even if two files happen to
    differ only in header encoding, and able to be recomputed identically
    from an in-memory array or a memory-mapped one.

    Chunked along axis 0 so a 10,000,000-row persisted replicate dataset can
    be hashed with bounded peak memory. Never mutates or reorders the input;
    only reads it.
    """
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    if arr.ndim == 0:
        h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
    n = arr.shape[0]
    step = max(1, int(chunk_rows))
    for start in range(0, n, step):
        chunk = arr[start : start + step]
        h.update(np.ascontiguousarray(chunk).tobytes())
    return h.hexdigest()


def dataset_paths(root: Path, replicate: int) -> dict[str, Path]:
    d = root / "datasets" / f"replicate_{replicate:02d}"
    return {"train_x": d / "train_x.npy", "train_y": d / "train_y.npy",
            "metadata": d / "metadata.json"}


def test_dataset_paths(root: Path) -> dict[str, Path]:
    d = root / "datasets" / "test"
    return {"test_x": d / "test_x.npy", "test_y": d / "test_y.npy",
            "metadata": d / "metadata.json"}


def condition_paths(root: Path, replicate: int, size: int) -> dict[str, Path]:
    d = root / "runs" / f"replicate_{replicate:02d}" / f"n_{size}"
    return {"checkpoint": d / "latest.keras", "state": d / "state.json",
            "history": d / "history.json"}


def _persisted_group_state(paths: Mapping[str, Path]) -> str:
    """Classify a group of files that must always be persisted together
    (e.g. train_x.npy + train_y.npy + metadata.json) as:

        "absent"   -- none of the files exist: safe to generate fresh.
        "complete" -- every file exists: safe to load and verify.
        "partial"  -- some but not all files exist: an integrity failure.
                      This must never be treated as either of the above.
    """
    existing = [p for p in paths.values() if p.exists()]
    if len(existing) == 0:
        return "absent"
    if len(existing) == len(paths):
        return "complete"
    return "partial"


def _partial_state_error(kind: str, paths: Mapping[str, Path]) -> RuntimeError:
    present = sorted(str(p) for p in paths.values() if p.exists())
    missing = sorted(str(p) for p in paths.values() if not p.exists())
    return RuntimeError(
        f"Integrity failure: the persisted {kind} is INCOMPLETE (partially "
        "written), not absent and not complete. "
        f"Present: {present}. Missing: {missing}. "
        "Dataset generation for this experiment is not guaranteed "
        "deterministic, so an incomplete dataset cannot be safely completed "
        "or regenerated in place without silently producing a different "
        "dataset than whatever partial state exists. Refusing to proceed. "
        "Remove the incomplete files explicitly (after confirming they are "
        "genuinely abandoned) and rerun, or use a new output directory."
    )


def dataset_exists(root: Path, replicate: int) -> bool:
    return _persisted_group_state(dataset_paths(root, replicate)) == "complete"


def test_exists(root: Path) -> bool:
    return _persisted_group_state(test_dataset_paths(root)) == "complete"


# ---------------------------------------------------------------------------
# Fixed test dataset: generate-once, hash-verify-always, never regenerate.
# ---------------------------------------------------------------------------

def ensure_test_dataset(
    root: Path,
    *,
    test_samples: int,
    audit_seed: int,
    config_hash: str,
    test_dataset_factory: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    dataset_replay_reason: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = test_dataset_paths(root)
    state = _persisted_group_state(paths)
    if state == "partial":
        raise _partial_state_error("fixed test dataset", paths)
    if state == "absent":
        print("Generating fixed held-out test dataset...")
        x, y = test_dataset_factory(test_samples, audit_seed)
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError("Generated test feature/label counts differ.")
        if len(x) != int(test_samples):
            raise ValueError(
                f"Generated test dataset has {len(x):,} rows, expected {test_samples:,}."
            )
        _atomic_npy(paths["test_x"], x)
        _atomic_npy(paths["test_y"], y)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "samples": int(len(x)),
            "dtype": {"test_x": str(x.dtype), "test_y": str(y.dtype)},
            "test_x_shape": list(x.shape),
            "test_y_shape": list(y.shape),
            "test_x_file_sha256": _file_sha256(paths["test_x"]),
            "test_y_file_sha256": _file_sha256(paths["test_y"]),
            "test_x_logical_sha256": _array_logical_sha256(x),
            "test_y_logical_sha256": _array_logical_sha256(y),
            "dataset_seed": None,
            "dataset_replay_available": False,
            "dataset_replay_reason": dataset_replay_reason,
            "audit_seed": int(audit_seed),
            "config_hash": config_hash,
        }
        _atomic_json(paths["metadata"], metadata)
        print("Fixed test dataset persisted and hashed.")
        del x, y
        return load_test_dataset(root, config_hash=config_hash)
    print("Loading and verifying existing fixed test dataset (never regenerated)...")
    return load_test_dataset(root, config_hash=config_hash)


def load_test_dataset(root: Path, *, config_hash: str | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = test_dataset_paths(root)
    state = _persisted_group_state(paths)
    if state != "complete":
        if state == "partial":
            raise _partial_state_error("fixed test dataset", paths)
        raise FileNotFoundError("Fixed test dataset has not been generated yet.")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "Fixed test dataset metadata schema_version="
            f"{metadata.get('schema_version')!r} does not match the required "
            f"{SCHEMA_VERSION!r}. Old-schema run directories are not migrated "
            "automatically. Start a new output directory, or explicitly "
            "migrate this metadata to the current schema before rerunning."
        )
    required_keys = (
        "test_x_file_sha256", "test_y_file_sha256",
        "test_x_logical_sha256", "test_y_logical_sha256",
        "test_x_shape", "test_y_shape", "samples",
    )
    missing_keys = [k for k in required_keys if k not in metadata]
    if missing_keys:
        raise RuntimeError(
            f"Fixed test dataset metadata is missing required integrity "
            f"fields {missing_keys}. Refusing to load. Start a new output "
            "directory or explicitly migrate this metadata to the current schema."
        )
    if config_hash is not None and metadata.get("config_hash") != config_hash:
        raise RuntimeError(
            "Fixed test dataset metadata config_hash does not match the "
            "active run configuration. Refusing to load."
        )

    actual_x_file_hash = _file_sha256(paths["test_x"])
    if actual_x_file_hash != metadata["test_x_file_sha256"]:
        raise RuntimeError("Fixed test dataset test_x.npy file hash mismatch: integrity failure.")
    actual_y_file_hash = _file_sha256(paths["test_y"])
    if actual_y_file_hash != metadata["test_y_file_sha256"]:
        raise RuntimeError("Fixed test dataset test_y.npy file hash mismatch: integrity failure.")

    x = np.load(paths["test_x"], mmap_mode="r")
    y = np.load(paths["test_y"], mmap_mode="r")

    if _array_logical_sha256(x) != metadata["test_x_logical_sha256"]:
        raise RuntimeError("Fixed test dataset test_x logical content hash mismatch: integrity failure.")
    if _array_logical_sha256(y) != metadata["test_y_logical_sha256"]:
        raise RuntimeError("Fixed test dataset test_y logical content hash mismatch: integrity failure.")
    if len(x) != len(y):
        raise RuntimeError("Fixed test dataset persisted test_x/test_y length mismatch.")
    if len(x) != int(metadata["samples"]):
        raise RuntimeError("Fixed test dataset persisted sample count does not match metadata.")

    return x, y, metadata


# ---------------------------------------------------------------------------
# Per-replicate maximum-size dataset: generate-once, hash-verify-always,
# never regenerate. Every declared nested-prefix size gets its own persisted
# and re-verified content hash.
# ---------------------------------------------------------------------------

def ensure_replicate_dataset(
    root: Path,
    replicate: int,
    *,
    max_size: int,
    sizes: Sequence[int],
    audit_seed: int,
    replicate_seed: int,
    config_hash: str,
    train_dataset_factory: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    dataset_replay_reason: str,
    replicate_seed_role: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = dataset_paths(root, replicate)
    state = _persisted_group_state(paths)
    if state == "partial":
        raise _partial_state_error(f"replicate {replicate} dataset", paths)
    if state == "absent":
        print(f"Generating independent {max_size:,}-sample dataset for replicate {replicate}...")
        x, y = train_dataset_factory(max_size, replicate_seed)
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError(f"Replicate {replicate}: generated feature/label counts differ.")
        if len(x) != int(max_size):
            raise ValueError(
                f"Replicate {replicate}: generated dataset has {len(x):,} rows, "
                f"expected {max_size:,}."
            )
        prefix_hashes: dict[str, Any] = {}
        for n in sizes:
            n = int(n)
            if n > len(x):
                raise ValueError(
                    f"Replicate {replicate}: declared nested prefix size {n:,} "
                    f"exceeds generated dataset size {len(x):,}."
                )
            prefix_hashes[str(n)] = {
                "samples": n,
                "train_x_logical_sha256": _array_logical_sha256(x[:n]),
                "train_y_logical_sha256": _array_logical_sha256(y[:n]),
            }
        _atomic_npy(paths["train_x"], x)
        _atomic_npy(paths["train_y"], y)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "replicate": int(replicate),
            "max_samples": int(max_size),
            "samples": int(len(x)),
            "dtype": {"train_x": str(x.dtype), "train_y": str(y.dtype)},
            "train_x_shape": list(x.shape),
            "train_y_shape": list(y.shape),
            "train_x_file_sha256": _file_sha256(paths["train_x"]),
            "train_y_file_sha256": _file_sha256(paths["train_y"]),
            "train_x_logical_sha256": _array_logical_sha256(x),
            "train_y_logical_sha256": _array_logical_sha256(y),
            "nested_prefix_sizes": [int(n) for n in sizes],
            "nested_prefix_hashes": prefix_hashes,
            "dataset_seed": None,
            "dataset_replay_available": False,
            "dataset_replay_reason": dataset_replay_reason,
            "audit_seed": int(audit_seed),
            "replicate_seed": int(replicate_seed),
            "replicate_seed_role": replicate_seed_role,
            "config_hash": config_hash,
        }
        _atomic_json(paths["metadata"], metadata)
        print(f"Persisted and hashed maximum-size dataset for replicate {replicate}.")
        del x, y
        return load_replicate_dataset(root, replicate, sizes=sizes, config_hash=config_hash)
    print(f"Loading and verifying existing dataset for replicate {replicate} (never regenerated)...")
    return load_replicate_dataset(root, replicate, sizes=sizes, config_hash=config_hash)


def load_replicate_dataset(
    root: Path,
    replicate: int,
    *,
    sizes: Sequence[int],
    config_hash: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = dataset_paths(root, replicate)
    state = _persisted_group_state(paths)
    if state != "complete":
        if state == "partial":
            raise _partial_state_error(f"replicate {replicate} dataset", paths)
        raise FileNotFoundError(f"Dataset for replicate {replicate} has not been generated yet.")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Replicate {replicate} dataset metadata schema_version="
            f"{metadata.get('schema_version')!r} does not match the required "
            f"{SCHEMA_VERSION!r}. Old-schema run directories are not migrated "
            "automatically. Start a new output directory, or explicitly "
            "migrate this metadata to the current schema before rerunning."
        )
    required_keys = (
        "train_x_file_sha256", "train_y_file_sha256",
        "train_x_logical_sha256", "train_y_logical_sha256",
        "train_x_shape", "train_y_shape", "samples",
        "nested_prefix_hashes", "nested_prefix_sizes",
    )
    missing_keys = [k for k in required_keys if k not in metadata]
    if missing_keys:
        raise RuntimeError(
            f"Replicate {replicate} dataset metadata is missing required "
            f"integrity fields {missing_keys}. Refusing to load. Start a new "
            "output directory or explicitly migrate this metadata to the "
            "current schema."
        )
    if config_hash is not None and metadata.get("config_hash") != config_hash:
        raise RuntimeError(
            f"Replicate {replicate} dataset metadata config_hash does not "
            "match the active run configuration. Refusing to load."
        )

    actual_x_file_hash = _file_sha256(paths["train_x"])
    if actual_x_file_hash != metadata["train_x_file_sha256"]:
        raise RuntimeError(f"Replicate {replicate} train_x.npy file hash mismatch: integrity failure.")
    actual_y_file_hash = _file_sha256(paths["train_y"])
    if actual_y_file_hash != metadata["train_y_file_sha256"]:
        raise RuntimeError(f"Replicate {replicate} train_y.npy file hash mismatch: integrity failure.")

    x = np.load(paths["train_x"], mmap_mode="r")
    y = np.load(paths["train_y"], mmap_mode="r")

    if _array_logical_sha256(x) != metadata["train_x_logical_sha256"]:
        raise RuntimeError(f"Replicate {replicate} train_x logical content hash mismatch: integrity failure.")
    if _array_logical_sha256(y) != metadata["train_y_logical_sha256"]:
        raise RuntimeError(f"Replicate {replicate} train_y logical content hash mismatch: integrity failure.")
    if len(x) != len(y):
        raise RuntimeError(f"Replicate {replicate} persisted train_x/train_y length mismatch.")
    if len(x) != int(metadata["samples"]):
        raise RuntimeError(f"Replicate {replicate} persisted sample count does not match metadata.")

    prefix_hashes = metadata["nested_prefix_hashes"]
    for n in sizes:
        n = int(n)
        key = str(n)
        if key not in prefix_hashes:
            raise RuntimeError(
                f"Replicate {replicate} metadata has no persisted hash for "
                f"declared nested prefix size {n:,}. Refusing to proceed "
                "rather than assume prefix slicing is still valid."
            )
        entry = prefix_hashes[key]
        if n > len(x):
            raise RuntimeError(
                f"Declared nested prefix size {n:,} exceeds persisted dataset "
                f"size {len(x):,} for replicate {replicate}."
            )
        if int(entry.get("samples", -1)) != n:
            raise RuntimeError(
                f"Replicate {replicate} nested-prefix {n:,} metadata sample "
                "count mismatch."
            )
        actual_prefix_x = _array_logical_sha256(x[:n])
        if actual_prefix_x != entry.get("train_x_logical_sha256"):
            raise RuntimeError(
                f"Replicate {replicate} nested-prefix {n:,} train_x hash "
                "mismatch: integrity failure."
            )
        actual_prefix_y = _array_logical_sha256(y[:n])
        if actual_prefix_y != entry.get("train_y_logical_sha256"):
            raise RuntimeError(
                f"Replicate {replicate} nested-prefix {n:,} train_y hash "
                "mismatch: integrity failure."
            )

    return x, y, metadata


def prefix_hashes_for(metadata: Mapping[str, Any], n: int) -> tuple[str, str]:
    """Return (train_x_logical_sha256, train_y_logical_sha256) for a
    declared nested prefix size, reading only already-verified metadata
    (see load_replicate_dataset / ensure_replicate_dataset)."""
    entry = metadata["nested_prefix_hashes"][str(int(n))]
    return entry["train_x_logical_sha256"], entry["train_y_logical_sha256"]


# ---------------------------------------------------------------------------
# Per-condition checkpoint/state.
# ---------------------------------------------------------------------------

class EpochCheckpointCallback(Callback):
    """Persist a complete model and state after every completed epoch.

    The state file identifies not only the training configuration
    (replicate, training_samples, seed, epoch counters) but the exact
    persisted-data content it was computed against: the configuration hash,
    the nested-prefix train_x/train_y content hashes, and the fixed test
    set's content hashes. Resuming this condition re-verifies all of these
    against the currently persisted data before any further training.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        state_path: Path,
        condition: str,
        replicate: int,
        training_samples: int,
        seed: int,
        total_epochs: int,
        *,
        config_hash: str,
        train_x_prefix_hash: str,
        train_y_prefix_hash: str,
        test_x_hash: str,
        test_y_hash: str,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.state_path = state_path
        self.condition = condition
        self.replicate = replicate
        self.training_samples = training_samples
        self.seed = seed
        self.total_epochs = total_epochs
        self.config_hash = config_hash
        self.train_x_prefix_hash = train_x_prefix_hash
        self.train_y_prefix_hash = train_y_prefix_hash
        self.test_x_hash = test_x_hash
        self.test_y_hash = test_y_hash

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_name("checkpoint.tmp.keras")
        self.model.save(tmp)
        os.replace(tmp, self.checkpoint_path)
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if epoch + 1 >= self.total_epochs else "in_progress",
            "condition": self.condition,
            "replicate": self.replicate,
            "training_samples": self.training_samples,
            "seed": self.seed,
            "total_epochs": self.total_epochs,
            "completed_epochs": int(epoch + 1),
            "config_hash": self.config_hash,
            "train_x_prefix_hash": self.train_x_prefix_hash,
            "train_y_prefix_hash": self.train_y_prefix_hash,
            "test_x_hash": self.test_x_hash,
            "test_y_hash": self.test_y_hash,
            "last_epoch_logs": {
                str(k): float(v) for k, v in logs.items()
                if np.isscalar(v) and np.isfinite(float(v))
            },
        }
        _atomic_json(self.state_path, state)

def _validate_checkpoint_state_pair(
    paths: Mapping[str, Path],
    condition: str,
) -> bool:
    """Validate the all-or-nothing checkpoint/state persistence invariant.

    Returns False when neither checkpoint nor state exists, meaning the
    condition may start fresh.

    Returns True when both checkpoint and state exist, meaning the persisted
    condition may be validated and resumed/skipped.

    Raises RuntimeError when exactly one of the two files exists. Orphaned
    checkpoint/state persistence must never be silently discarded or
    overwritten.
    """
    checkpoint_exists = paths["checkpoint"].exists()
    state_exists = paths["state"].exists()

    if not checkpoint_exists and not state_exists:
        return False

    if checkpoint_exists and state_exists:
        return True

    if checkpoint_exists and not state_exists:
        raise RuntimeError(
            f"Integrity failure for {condition}: "
            "state.json is missing while latest.keras exists. "
            "Checkpoint/state persistence must be all-or-nothing; "
            "refusing to silently discard or overwrite the orphaned "
            "checkpoint."
        )

    raise RuntimeError(
        f"Integrity failure for {condition}: "
        "latest.keras is missing while state.json exists. "
        "Checkpoint/state persistence must be all-or-nothing; "
        "refusing to silently accept or resume the orphaned state."
    )


def _validate_condition_state_matches_current_data(
    state: Mapping[str, Any],
    *,
    condition: str,
    config_hash: str,
    expected_train_x_hash: str,
    expected_train_y_hash: str,
    expected_test_x_hash: str,
    expected_test_y_hash: str,
) -> None:
    """Refuse to resume a condition whose recorded state does not match the
    data/configuration that is currently persisted. This is the resume-time
    half of the integrity contract; EpochCheckpointCallback is the write-time
    half.
    """
    checks = {
        "config_hash": (state.get("config_hash"), config_hash),
        "train_x_prefix_hash": (state.get("train_x_prefix_hash"), expected_train_x_hash),
        "train_y_prefix_hash": (state.get("train_y_prefix_hash"), expected_train_y_hash),
        "test_x_hash": (state.get("test_x_hash"), expected_test_x_hash),
        "test_y_hash": (state.get("test_y_hash"), expected_test_y_hash),
    }
    mismatches = {k: (a, b) for k, (a, b) in checks.items() if a != b}
    if mismatches:
        detail = "; ".join(f"{k}: state={a!r} != current={b!r}" for k, (a, b) in mismatches.items())
        raise RuntimeError(
            f"Refusing to resume {condition}: its checkpoint state was computed "
            "against different data or configuration than what is currently "
            f"persisted ({detail}). A checkpoint must never be resumed against "
            "data it was not trained on. Use a new output directory if the "
            "underlying data or configuration was intentionally changed."
        )


# ---------------------------------------------------------------------------
# Statistical engine (bootstrap CIs, paired t-tests, Holm correction,
# scaling slopes) -- UNCHANGED except for explicit baseline/role labeling.
# ---------------------------------------------------------------------------

def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                       n_boot: int, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = (1.0 - confidence) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))


def _paired_t_test(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t statistic and p-value; scipy is used if available."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2:
        return float("nan"), float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return (float("inf"), 0.0) if np.mean(diffs) != 0 else (0.0, 1.0)
    t = float(np.mean(diffs) / (sd / math.sqrt(len(diffs))))
    try:
        from scipy.stats import t as t_dist
        p = float(2.0 * t_dist.sf(abs(t), df=len(diffs) - 1))
    except Exception:
        # Conservative normal approximation fallback.
        p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return t, p


def _holm_adjust(pairs: list[tuple[str, float]], alpha: float) -> dict[str, Any]:
    valid = [(k, p) for k, p in pairs if np.isfinite(p)]
    ordered = sorted(valid, key=lambda z: z[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for i, (k, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[k] = running
    return {
        "alpha": alpha,
        "method": "Holm step-down",
        "adjusted_p_values": adjusted,
        "rejections": {k: bool(v < alpha) for k, v in adjusted.items()},
    }


def _aggregate(observations: list[D5Observation], sizes: Sequence[int],
                replicates: int, bootstrap_replicates: int,
                seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_size = {}
    for n in sizes:
        vals = np.array([o.test_accuracy for o in observations if o.training_samples == n], dtype=float)
        losses = np.array([o.test_loss for o in observations if o.training_samples == n], dtype=float)
        lo, hi = _bootstrap_mean_ci(vals, rng, bootstrap_replicates)
        by_size[str(n)] = {
            "n": int(n), "replicates": int(len(vals)),
            "accuracy_mean": float(vals.mean()),
            "accuracy_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "accuracy_ci95_bootstrap": [lo, hi],
            "accuracy_min": float(vals.min()), "accuracy_max": float(vals.max()),
            "loss_mean": float(losses.mean()),
            "loss_sd": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
        }

    # Paired contrasts against the smallest declared training size (the
    # explicit baseline condition -- see "baseline_training_size" below).
    base_n = int(sizes[0])
    paired: dict[str, Any] = {
        "baseline_training_size": base_n,
        "baseline_role": (
            "The smallest declared training size is the fixed baseline "
            "condition. Every paired contrast below is (comparison size) "
            "minus (this baseline), computed within-replicate."
        ),
    }
    pvals = []
    lookup = {(o.replicate, o.training_samples): o for o in observations}
    for n in sizes[1:]:
        diffs = np.array([
            lookup[(r, n)].test_accuracy - lookup[(r, base_n)].test_accuracy
            for r in range(1, replicates + 1)
            if (r, n) in lookup and (r, base_n) in lookup
        ], dtype=float)
        t, p = _paired_t_test(diffs)
        lo, hi = _bootstrap_mean_ci(diffs, rng, bootstrap_replicates)
        key = f"{base_n}_vs_{n}"
        paired[key] = {
            "baseline_size": base_n, "comparison_size": int(n),
            "replicates": int(len(diffs)),
            "mean_accuracy_difference": float(diffs.mean()) if len(diffs) else float("nan"),
            "sd_accuracy_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "bootstrap_ci95": [lo, hi],
            "paired_t_statistic": t, "paired_p_value": p,
        }
        pvals.append((key, p))
    paired["multiple_comparison_correction"] = _holm_adjust(pvals, 0.05)

    # Replicate-wise slopes against log10(sample count). SECONDARY/
    # DESCRIPTIVE diagnostic only -- see "role" field below. This is not a
    # causal or cryptographic-learning claim and does not override the
    # paired-contrast analysis above, which remains the primary analysis.
    x = np.log10(np.asarray(sizes, dtype=float))
    slopes = []
    for r in range(1, replicates + 1):
        y = np.array([lookup[(r, n)].test_accuracy for n in sizes if (r, n) in lookup], dtype=float)
        if len(y) == len(sizes):
            slope = float(np.polyfit(x, y, 1)[0])
            slopes.append(slope)
    slopes = np.asarray(slopes, dtype=float)
    slo, shi = _bootstrap_mean_ci(slopes, rng, bootstrap_replicates)
    monotone = []
    for r in range(1, replicates + 1):
        y = [lookup[(r, n)].test_accuracy for n in sizes if (r, n) in lookup]
        if len(y) == len(sizes):
            monotone.append(all(y[i+1] >= y[i] for i in range(len(y)-1)))
    scaling = {
        "role": (
            "SECONDARY/DESCRIPTIVE diagnostic. Does not override, replace, or "
            "supersede the primary paired-contrast analysis above. Not a "
            "causal claim and not a claim of a universal scaling law or of "
            "cryptographic learning."
        ),
        "model": "ordinary least squares accuracy ~ log10(training_samples), fit separately per replicate",
        "replicate_slopes": slopes.tolist(),
        "mean_slope": float(slopes.mean()) if len(slopes) else float("nan"),
        "mean_slope_bootstrap_ci95": [slo, shi],
        "monotone_non_decreasing_replicates": int(sum(monotone)),
        "replicates_with_complete_curves": int(len(monotone)),
    }
    return by_size, paired, scaling


def run_d5(
    *,
    root: Path,
    training_sizes: Sequence[int],
    test_samples: int,
    replicates: int,
    audit_seed: int,
    total_epochs: int,
    batch_size: int,
    train_dataset_factory: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    test_dataset_factory: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[int], Any],
    training_callbacks_factory: Callable[[int], list[Any]] | None,
    evaluate_model: Callable[[Any, np.ndarray, np.ndarray], tuple[float, float]],
    experiment_name: str,
    bootstrap_replicates: int = 5000,
    manifest: Mapping[str, Any] | None = None,
    dataset_replay_reason: str = _DEFAULT_DATASET_REPLAY_REASON,
    replicate_seed_role: str = _DEFAULT_REPLICATE_SEED_ROLE,
) -> D5Result:
    root.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(n) for n in training_sizes})
    if len(sizes) < 2:
        raise ValueError("At least two training sizes are required.")
    if replicates < 2:
        raise ValueError("At least two independent replicates are required.")
    if test_samples < 1 or total_epochs < 1 or batch_size < 1:
        raise ValueError("test_samples, total_epochs and batch_size must be positive.")

    max_size = max(sizes)
    base_manifest = dict(manifest or {})
    base_manifest.update({
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_name,
        "training_sizes": sizes,
        "baseline_training_size": sizes[0],
        "replicates": replicates,
        "test_samples": test_samples,
        "total_epochs": total_epochs,
        "batch_size": batch_size,
        "audit_seed": audit_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "design": {
            "independent_replicate_datasets": True,
            "nested_prefixes_within_replicate": True,
            "same_model_seed_across_sizes_within_replicate": True,
            "fixed_test_partition": True,
            "nested_prefixes_are_not_independent_datasets": True,
            "individual_test_examples_are_not_independent_replicates": True,
            "independent_experimental_unit": "replicate",
        },
        "dataset_seed": None,
        "dataset_replay_available": False,
        "dataset_replay_reason": dataset_replay_reason,
        "replicate_seed_role": replicate_seed_role,
    })
    config_hash = _sha256_json(base_manifest)
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash:
            raise RuntimeError(
                "D5 run configuration mismatch against the existing "
                f"run_manifest.json at {manifest_path}. Refusing to resume "
                "and refusing to overwrite the existing manifest. Use a new "
                "output directory for a changed experiment configuration "
                "(this includes SCHEMA_VERSION upgrades)."
            )
    else:
        payload = dict(base_manifest)
        payload["config_hash"] = config_hash
        _atomic_json(manifest_path, payload)

    # Fixed test set: generate once, otherwise load and hash-verify. Never
    # regenerated once persisted in full; a partial persisted test dataset
    # is a hard failure, not something to silently overwrite.
    test_x, test_y, test_meta = ensure_test_dataset(
        root,
        test_samples=test_samples,
        audit_seed=audit_seed,
        config_hash=config_hash,
        test_dataset_factory=test_dataset_factory,
        dataset_replay_reason=dataset_replay_reason,
    )
    expected_test_x_hash = test_meta["test_x_logical_sha256"]
    expected_test_y_hash = test_meta["test_y_logical_sha256"]

    # Independent replicate seeds. These control model initialization and
    # the audit's own RNG stream -- NOT dataset generation (see
    # dataset_replay_reason / replicate_seed_role above and in persisted
    # metadata).
    seq = np.random.SeedSequence(audit_seed)
    child = seq.spawn(replicates)
    replicate_seeds = [
        int(s.generate_state(1, dtype=np.uint32)[0]) for s in child
    ]

    observations: list[D5Observation] = []
    for r in range(1, replicates + 1):
        seed = replicate_seeds[r - 1]
        print()
        print("=" * 72)
        print(f"D5 replicate {r}/{replicates} | model_initialization_and_audit_RNG_stream seed={seed}")
        print("=" * 72)

        train_x_full, train_y_full, train_meta = ensure_replicate_dataset(
            root, r,
            max_size=max_size,
            sizes=sizes,
            audit_seed=audit_seed,
            replicate_seed=seed,
            config_hash=config_hash,
            train_dataset_factory=train_dataset_factory,
            dataset_replay_reason=dataset_replay_reason,
            replicate_seed_role=replicate_seed_role,
        )

        for n in sizes:
            p = condition_paths(root, r, n)
            condition = f"replicate_{r:02d}/n_{n}"
            expected_train_x_hash, expected_train_y_hash = prefix_hashes_for(train_meta, n)

            checkpoint_state_pair_valid = _validate_checkpoint_state_pair(
                p,
                condition,
            )

            state = None
            if checkpoint_state_pair_valid:
                state = json.loads(p["state"].read_text(encoding="utf-8"))
                _validate_condition_state_matches_current_data(
                    state,
                    condition=condition,
                    config_hash=config_hash,
                    expected_train_x_hash=expected_train_x_hash,
                    expected_train_y_hash=expected_train_y_hash,
                    expected_test_x_hash=expected_test_x_hash,
                    expected_test_y_hash=expected_test_y_hash,
                )
            completed = int(state.get("completed_epochs", 0)) if state else 0
            if completed >= total_epochs and p["checkpoint"].exists():
                print(f"Skipping completed {condition}.")
                model = load_model(p["checkpoint"], compile=True)
            else:
                if p["checkpoint"].exists() and completed > 0:
                    print(f"Resuming {condition} from epoch {completed}/{total_epochs}.")
                    model = load_model(p["checkpoint"], compile=True)
                else:
                    print(f"Starting {condition} from epoch 0/{total_epochs}.")
                    model = model_factory(seed)
                    completed = 0

                callbacks = [
                    EpochCheckpointCallback(
                        p["checkpoint"], p["state"], condition, r, n, seed, total_epochs,
                        config_hash=config_hash,
                        train_x_prefix_hash=expected_train_x_hash,
                        train_y_prefix_hash=expected_train_y_hash,
                        test_x_hash=expected_test_x_hash,
                        test_y_hash=expected_test_y_hash,
                    )
                ]
                if training_callbacks_factory is not None:
                    callbacks.extend(training_callbacks_factory(seed))

                # Critical nested-design operation: prefix only. The prefix
                # content hash was already verified in ensure_replicate_dataset
                # / load_replicate_dataset above.
                x = train_x_full[:n]
                y = train_y_full[:n]
                model.fit(
                    x, y,
                    initial_epoch=completed,
                    epochs=total_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1,
                )
                del x, y

            loss, acc = evaluate_model(model, test_x, test_y)
            _atomic_json(p["state"], {
                "schema_version": SCHEMA_VERSION,
                "config_hash": config_hash,
                "status": "complete",
                "condition": condition,
                "replicate": r,
                "training_samples": n,
                "seed": seed,
                "total_epochs": total_epochs,
                "completed_epochs": total_epochs,
                "train_x_prefix_hash": expected_train_x_hash,
                "train_y_prefix_hash": expected_train_y_hash,
                "test_x_hash": expected_test_x_hash,
                "test_y_hash": expected_test_y_hash,
                "test_loss": float(loss),
                "test_accuracy": float(acc),
            })
            _atomic_json(p["history"], {
                "replicate": r, "training_samples": n,
                "test_loss": float(loss), "test_accuracy": float(acc),
            })
            observations.append(D5Observation(
                replicate=r, training_samples=n, seed=seed,
                test_samples=test_samples, test_accuracy=float(acc),
                test_loss=float(loss), completed_epochs=total_epochs,
                total_epochs=total_epochs,
            ))
            del model

        del train_x_full, train_y_full

    # Reconstruct all completed observations from state files, including
    # resumed sessions, rather than trusting only the in-memory loop above.
    observations = []
    for r in range(1, replicates + 1):
        seed = replicate_seeds[r - 1]
        for n in sizes:
            p = condition_paths(root, r, n)
            if not p["state"].exists():
                continue
            st = json.loads(p["state"].read_text(encoding="utf-8"))
            if st.get("status") == "complete":
                observations.append(D5Observation(
                    replicate=r, training_samples=n, seed=seed,
                    test_samples=test_samples,
                    test_accuracy=float(st["test_accuracy"]),
                    test_loss=float(st["test_loss"]),
                    completed_epochs=int(st["completed_epochs"]),
                    total_epochs=total_epochs,
                ))

    expected_conditions = {(r, n) for r in range(1, replicates + 1) for n in sizes}
    found_conditions = {(o.replicate, o.training_samples) for o in observations}
    missing_conditions = sorted(expected_conditions - found_conditions)
    if missing_conditions:
        missing_desc = ", ".join(f"replicate_{r:02d}/n_{n}" for r, n in missing_conditions)
        raise RuntimeError(
            "D5 finished without a complete, valid observation for every "
            "declared replicate x training-size condition. A production D5 "
            f"result is not emitted. Incomplete or missing conditions: {missing_desc}. "
            "Their state files remain on disk (visible, not deleted) for "
            "inspection; rerun to complete them."
        )

    summary, pairwise, scaling = _aggregate(
        observations, sizes, replicates, bootstrap_replicates, audit_seed + 991
    )
    return D5Result(
        experiment=experiment_name,
        training_sizes=sizes,
        replicates=replicates,
        observations=observations,
        summary=summary,
        pairwise=pairwise,
        scaling=scaling,
        manifest={**base_manifest, "config_hash": config_hash},
    )


def generate_certificate(result: D5Result, *, dataset_id: str,
                         dataset_version: str, generation_procedure: str,
                         generation_parameters: Mapping[str, Any],
                         audit_seed: int) -> dict[str, Any]:
    return {
        "audit": {
            "id": "D5",
            "name": "Training-Data Scaling Audit",
            "scope": "sample-size scaling under a fixed dataset/model/training protocol",
            "claim": (
                "Characterizes how predictive performance changes with training-set size; "
                "it does not by itself establish cryptographic learning or a universal "
                "dataset-size law."
            ),
        },
        "design": result.manifest["design"],
        "configuration": result.manifest,
        "findings": {
            "observations": [asdict(x) for x in result.observations],
            "summary_by_training_size": result.summary,
            "baseline_training_size": result.manifest.get("baseline_training_size"),
            "paired_contrasts": result.pairwise,
            "scaling_analysis": result.scaling,
        },
        "provenance": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_procedure": generation_procedure,
            "generation_parameters": dict(generation_parameters),
            "audit_seed": audit_seed,
            "dataset_seed": result.manifest.get("dataset_seed"),
            "dataset_replay_available": result.manifest.get("dataset_replay_available"),
            "dataset_replay_reason": result.manifest.get("dataset_replay_reason"),
            "replicate_seed_role": result.manifest.get("replicate_seed_role"),
            "resume_integrity_contract": (
                "Persisted dataset files (full replicate datasets, every declared "
                "nested prefix, and the fixed test set) are identified by both a "
                "file SHA-256 and a logical content SHA-256 (dtype+shape+bytes). "
                "Per-condition checkpoints additionally record the exact "
                "configuration hash and dataset/test content hashes they were "
                "produced against. On resume, all of these are re-verified before "
                "any further computation; a mismatch or an incomplete persisted "
                "dataset is a hard failure, never a silent regeneration."
            ),
        },
        "interpretation": (
            "D5 is a descriptive/estimative scaling experiment. "
            "The nested-prefix component controls the sampled population within each "
            "replicate, while independent replicate maximum datasets quantify "
            "between-dataset and training stochastic variation. The primary analysis "
            "is the paired contrast of each training size against the smallest "
            "declared size (the baseline); the scaling slope is a secondary, "
            "descriptive diagnostic only."
        ),
        "limitations": [
            "D5 is conditional on the specified model architecture, optimizer, "
            "training procedure, epoch budget, batch size, and dataset generator; "
            "it does not generalize beyond them.",
            "Nested prefixes are not independent datasets; their purpose is paired "
            "sample-size comparison within a replicate, not additional independent "
            "sampling.",
            "Independent maximum-size replicate datasets provide the between-dataset "
            "(and between-model-initialization) variation; the nested prefixes alone "
            "do not.",
            "The fixed test set is deliberately shared across every condition.",
            "Individual test examples are therefore NOT treated as independent "
            "experimental replicates; the independent experimental unit is the "
            "replicate, and the nested prefixes within a replicate are paired "
            "conditions, not additional independent units.",
            "Dataset generation for this experiment is not guaranteed deterministic "
            "with respect to the audit seed (see provenance.dataset_replay_reason); "
            "exact historical dataset replay from the integer audit seed is not "
            "available.",
            "Exact historical dataset replay is unavailable from the integer audit "
            "seed; the persisted dataset files and their hashes are the "
            "reproducibility/resumability object instead.",
            "Persisted dataset hashes (file and logical/content, for the full "
            "replicate dataset, every declared nested prefix, and the fixed test "
            "set) are therefore part of the resumability contract, not merely "
            "diagnostic metadata.",
            "The scaling curve is a secondary, descriptive diagnostic; it does not "
            "establish a universal scaling law and does not override the primary "
            "paired-contrast analysis.",
            "D5 does not by itself establish cryptographic learning.",
        ],
    }


def print_report(result: D5Result) -> None:
    print()
    print("=" * 72)
    print("Dataset Integrity Audit")
    print("D5 — Training-Data Scaling Audit")
    print("=" * 72)
    print(f"Replicates                  : {result.replicates}")
    print(f"Training sizes              : {', '.join(f'{n:,}' for n in result.training_sizes)}")
    print(f"Baseline training size      : {result.manifest.get('baseline_training_size'):,}")
    print("Design                      : independent replicate datasets + nested prefixes")
    print(
        "Dataset replay               : "
        f"{'AVAILABLE' if result.manifest.get('dataset_replay_available') else 'NOT AVAILABLE'} "
        f"({result.manifest.get('dataset_replay_reason')})"
    )
    print()
    print("Learning curve summary")
    print("-" * 72)
    for n in result.training_sizes:
        s = result.summary[str(n)]
        ci = s["accuracy_ci95_bootstrap"]
        print(
            f"{n:>12,d} | mean accuracy={s['accuracy_mean']:.8f} "
            f"| SD={s['accuracy_sd']:.8f} "
            f"| 95% bootstrap CI=[{ci[0]:.8f}, {ci[1]:.8f}]"
        )
    print()
    print(f"Paired contrasts vs baseline training size ({result.pairwise['baseline_training_size']:,})")
    print("-" * 72)
    for k, v in result.pairwise.items():
        if k in ("multiple_comparison_correction", "baseline_training_size", "baseline_role"):
            continue
        ci = v["bootstrap_ci95"]
        print(
            f"{k:>28} | mean Δ={v['mean_accuracy_difference']:+.8f} "
            f"| 95% CI=[{ci[0]:+.8f}, {ci[1]:+.8f}] "
            f"| paired p={v['paired_p_value']:.6g}"
        )
    print()
    print("Scaling summary (SECONDARY / DESCRIPTIVE -- see result.scaling['role'])")
    print("-" * 72)
    s = result.scaling
    ci = s["mean_slope_bootstrap_ci95"]
    print(f"Mean accuracy slope vs log10(N): {s['mean_slope']:+.8f}")
    print(f"95% bootstrap CI              : [{ci[0]:+.8f}, {ci[1]:+.8f}]")
    print(
        f"Monotone non-decreasing curves : "
        f"{s['monotone_non_decreasing_replicates']}/"
        f"{s['replicates_with_complete_curves']}"
    )
    print("=" * 72)


def write_learning_curve_plot(result: D5Result, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = np.array(result.training_sizes, dtype=float)
    means = np.array([result.summary[str(n)]["accuracy_mean"] for n in result.training_sizes])
    lo = np.array([result.summary[str(n)]["accuracy_ci95_bootstrap"][0] for n in result.training_sizes])
    hi = np.array([result.summary[str(n)]["accuracy_ci95_bootstrap"][1] for n in result.training_sizes])
    ax.plot(sizes, means, marker="o", label="Mean test accuracy")
    ax.fill_between(sizes, lo, hi, alpha=0.2, label="95% bootstrap CI")
    for r in range(1, result.replicates + 1):
        ys = []
        for n in result.training_sizes:
            obs = next(o for o in result.observations
                       if o.replicate == r and o.training_samples == n)
            ys.append(obs.test_accuracy)
        ax.plot(sizes, ys, marker=".", alpha=0.35, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title("D5 Training-Data Scaling")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)