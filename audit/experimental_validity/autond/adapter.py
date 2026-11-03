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


#: The pinned dbitnet.make_model applies tf.subtract to its Keras Input, which Keras 3 rejects.
SUPPORTED = {"tensorflow": "2.15.", "keras": "2.15."}
TENSORFLOW_DISTRIBUTIONS = ("tensorflow", "tensorflow-cpu", "tensorflow-macos", "tensorflow-intel")


def require_supported_environment() -> dict:
    """Refuse, before AutoND or TensorFlow is imported, unless TensorFlow and Keras are 2.15.x."""
    from importlib import metadata

    def version(names):
        for name in names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return None

    found = {"tensorflow": version(TENSORFLOW_DISTRIBUTIONS), "keras": version(("keras",))}
    bad = {k: v for k, v in found.items() if v is None or not v.startswith(SUPPORTED[k])}
    if bad:
        raise AutoNDBindingError(
            f"unsupported environment {found}: the pinned AutoND DBitNet needs TensorFlow 2.15 with Keras 2.15 "
            "(e.g. tensorflow==2.15.1 keras==2.15.0); Keras 3 rejects its tf.subtract on the model input")
    return found


def bind(autond_root) -> Any:
    """The pinned AutoND binding; refuses on an unsupported TensorFlow/Keras and, before import, on any
    commit or digest mismatch."""
    require_supported_environment()
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


_SESSION_ID: Optional[str] = None
_SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def session_id() -> str:
    """One identifier per process: the boot id (Linux) plus a random token drawn when first asked for."""
    global _SESSION_ID
    if _SESSION_ID is None:
        import uuid
        boot = Path("/proc/sys/kernel/random/boot_id")
        _SESSION_ID = f"{boot.read_text().strip() if boot.is_file() else 'no-boot-id'}:{uuid.uuid4().hex}"
    return _SESSION_ID


def _nvidia_smi() -> dict:
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False}
    fields = "index,name,uuid,driver_version,memory.total,pci.bus_id"
    out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader"], capture_output=True, text=True)
    gpus = [dict(zip(fields.split(","), (c.strip() for c in line.split(","))))
            for line in out.stdout.strip().splitlines() if line.strip()] if out.returncode == 0 else []
    return {"available": out.returncode == 0, "gpus": gpus, "stderr": out.stderr.strip()[:500]}


def session_provenance() -> dict:
    """Hardware and software identity of this process, recorded per run invocation and per unit."""
    import os
    import platform
    from importlib import metadata
    import tensorflow as tf
    build = {k: str(v) for k, v in tf.sysconfig.get_build_info().items()}
    physical = [d.name for d in tf.config.list_physical_devices("GPU")]
    details = []
    for d in tf.config.list_physical_devices("GPU"):
        try:
            details.append({k: str(v) for k, v in tf.config.experimental.get_device_details(d).items()})
        except Exception as exc:              # device details are best effort; their absence is recorded
            details.append({"error": f"{type(exc).__name__}: {exc}"})
    versions = {}
    for pkg in ("tensorflow", "keras", "numpy", "scipy", "h5py", "pandas", "PyYAML"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    kaggle = {k: v for k, v in sorted(os.environ.items())
              if k.startswith("KAGGLE_") and not any(m in k.upper() for m in _SECRET_MARKERS)}
    return {"session_id": session_id(), "hostname": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "versions": versions,
            "tensorflow_build": build, "tensorflow_gpus": physical, "tensorflow_gpu_details": details,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "nvidia_smi": _nvidia_smi(),
            "kaggle_environment": kaggle}


def device_identity(session: dict) -> dict:
    """The GPU this process trains on: the first visible GPU's nvidia-smi record, or the CPU."""
    gpus = (session.get("nvidia_smi") or {}).get("gpus") or []
    visible = session.get("cuda_visible_devices")
    if not session.get("tensorflow_gpus"):
        return {"device": "CPU", "gpu_uuid": None, "gpu_name": None}
    if visible not in (None, ""):
        first = visible.split(",")[0].strip()
        match = [g for g in gpus if g.get("index") == first or g.get("uuid") == first]
        gpus = match or gpus
    g = gpus[0] if gpus else {}
    return {"device": "GPU", "gpu_uuid": g.get("uuid"), "gpu_name": g.get("name"),
            "driver_version": g.get("driver_version")}
