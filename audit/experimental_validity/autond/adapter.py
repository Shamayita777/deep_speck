"""
Pinned AutoND binding, persisted datasets, and DBitNet construction/training/evaluation.

Data: audit.dataset.adapters.autond.AutoNDDatasetAdapter with the canonical
AUTOND_SOURCE_PIN (commit and every import-graph digest checked BEFORE any AutoND module
is imported). That adapter supplies data and provenance only, by design, so the model is
built here from the pinned `dbitnet` module, compiled exactly as upstream train_nets
compiles it for DBitNet: Adam(amsgrad=True), loss "mse", metric "acc".
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:      # audit.dataset.* must be importable from scripts and tests
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.failures import FailureReason, detect_failure_from_history  # noqa: E402
from framework.provenance import array_sha256, config_hash, sha256_file  # noqa: E402

CIPHER, SCENARIO, DELTA_PLAIN, ROUNDS, FEATURE_BITS = "speck3264", "single-key", 0x400000, 5, 64
PREDICT_BATCH = 5000
EXACT_REPLAY_REASON = "main.make_train_data draws from os.urandom; there is no seed on this path"


class AutoNDBindingError(RuntimeError):
    pass


def bind(autond_root) -> Any:
    """The pinned AutoND binding; refuses before import on any commit or digest mismatch."""
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    binding = AutoNDDatasetAdapter(
        autond_root=autond_root, cipher_name=CIPHER, delta_plain_int=DELTA_PLAIN, num_rounds=ROUNDS,
        scenario=SCENARIO, expected_source_digests=AUTOND_SOURCE_PIN["source_digests"],
        expected_upstream_commit=AUTOND_SOURCE_PIN["upstream_commit"])
    if not binding.pin_verified_preimport or binding.feature_bits != FEATURE_BITS:
        raise AutoNDBindingError("AutoND pin not verified before import, or feature width is not 64.")
    return binding


def pinned_module(binding, name: str):
    """A module from the pinned import graph, refused unless it resolves inside the checkout."""
    from audit.dataset.adapters.autond import _is_within, _module_locations

    module = sys.modules.get(name)
    locations = _module_locations(module) if module is not None else []
    if not locations or not all(_is_within(loc, binding.autond_root) for loc in locations):
        raise AutoNDBindingError(f"module {name!r} resolves to {locations}, outside {binding.autond_root}")
    return module


# ============================================================ data


@dataclass(frozen=True)
class Partition:
    X: np.ndarray
    Y: np.ndarray

    @property
    def sha256(self) -> str:
        """One hash over features and labels: what the firewall seals and pairing compares."""
        return array_sha256(np.concatenate([self.X, self.Y.reshape(-1, 1)], axis=1))


def validate_partition(X, Y, rows: int) -> None:
    X, Y = np.asarray(X), np.asarray(Y)
    problems = []
    if X.shape != (rows, FEATURE_BITS) or Y.shape != (rows,):
        problems.append(f"shapes {X.shape}/{Y.shape}, expected ({rows}, {FEATURE_BITS})/({rows},)")
    elif X.dtype != np.uint8 or Y.dtype != np.uint8:
        problems.append(f"dtypes {X.dtype}/{Y.dtype}, expected uint8")
    elif int(X.max(initial=0)) > 1 or int(Y.max(initial=0)) > 1:
        problems.append("values outside {0, 1}")
    if problems:
        raise AutoNDBindingError(f"generated partition refused: {problems}")


def load_or_generate(binding, path: Path, sizes: dict[str, int]) -> dict[str, Partition]:
    """Partitions persisted once and reloaded hash-verified on every later use (never regenerated).

    The generator is os.urandom-based, so persistence is what makes a pair's data the same for
    both arms and across an interrupted run. A new set also has its train/validation/test
    partitions audited for exact shared feature rows, which would indicate a generation defect.
    """
    path = Path(path)
    manifest = path.with_suffix(".json")
    if manifest.is_file():
        recorded = json.loads(manifest.read_text())
        with np.load(path) as data:
            parts = {role: Partition(data[f"{role}_X"], data[f"{role}_Y"]) for role in sizes}
        for role, part in parts.items():
            if part.sha256 != recorded["sha256"][role]:
                raise AutoNDBindingError(f"{path}: persisted {role} partition does not match its recorded hash")
        return parts

    from audit.dataset.integrity import audit_partition_overlap

    parts = {}
    for role, rows in sizes.items():
        X, Y = binding.generate_partition(int(rows))
        validate_partition(X, Y, int(rows))
        parts[role] = Partition(X, Y)
    overlap = audit_partition_overlap({role: (p.X, p.Y) for role, p in parts.items()})
    shared = {pair: entry["rows"] for pair, entry in overlap["cross"].items() if entry["rows"]}
    if shared:
        raise AutoNDBindingError(f"generated partitions share feature rows {shared}; refusing")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{r}_{k}": getattr(p, k) for r, p in parts.items() for k in ("X", "Y")})
    manifest.write_text(json.dumps({
        "sha256": {role: p.sha256 for role, p in parts.items()}, "rows": dict(sizes),
        "file_sha256": sha256_file(path), "overlap_method": overlap["method"],
        "generation": "AutoNDDatasetAdapter.generate_partition -> main.make_train_data",
        "exact_replay_available": False, "reason": EXACT_REPLAY_REASON}, indent=2, sort_keys=True))
    return parts


# ============================================================ model


def weights_sha256(model) -> str:
    """SHA-256 over every weight tensor in layer order: the evidence that two arms start identically."""
    return config_hash([array_sha256(w) for w in model.get_weights()])     # each entry binds dtype and shape


def build_model(binding, seed: int):
    """Seeded DBitNet compiled as upstream train_nets compiles it for 'dbitnet'."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    model = pinned_module(binding, "dbitnet").make_model(2 * int(binding.plain_bits))
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss="mse", metrics=["acc"])
    return model


@dataclass(frozen=True)
class TrainingOutcome:
    failure: Optional[FailureReason]
    history: dict[str, list[float]]


def train(model, train: Partition, validation: Partition, *, epochs: int, batch_size: int,
          shuffle: bool) -> TrainingOutcome:
    """model.fit as upstream calls it (validation data, fixed epochs), with `shuffle` explicit."""
    import tensorflow as tf

    try:
        fit = model.fit(train.X, train.Y, epochs=int(epochs), batch_size=int(batch_size), shuffle=bool(shuffle),
                        validation_data=(validation.X, validation.Y), verbose=2)
    except tf.errors.ResourceExhaustedError:
        return TrainingOutcome(FailureReason.OOM, {})
    history = {k: [float(v) for v in values] for k, values in fit.history.items()}
    return TrainingOutcome(detect_failure_from_history(history), history)


def accuracy(model, test: Partition) -> float:
    """Confirmatory accuracy of the final-epoch model: fraction with (prediction > 0.5) == label."""
    predictions = model.predict(test.X, batch_size=PREDICT_BATCH, verbose=0).reshape(-1)
    return float(np.mean((predictions > 0.5).astype(np.uint8) == test.Y))
