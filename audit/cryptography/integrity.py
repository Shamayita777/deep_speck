"""
Programmatic model / checkpoint integrity.

Depth is determined from the ACTUAL saved architecture, never from the
filename. The historical artifact named `best5depth10 (10).h5` contains a
depth-5 network; a filename is not evidence.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from audit.cryptography.audit_config import (
    HISTORICAL_DEPTH5_CHECKPOINTS,
    REFERENCE,
    REFERENCE_CHECKPOINT_SHA256,
)


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _layer_census(path) -> dict:
    """Layer-class counts read from the persisted model configuration."""
    path = Path(path)
    if not path.exists():
        raise ModelIntegrityError(f"checkpoint not found: {path}")
    cfg = None
    if path.suffix == ".keras":
        try:
            with zipfile.ZipFile(path) as zf:
                cfg = json.loads(zf.read("config.json"))
        except Exception as exc:
            raise ModelIntegrityError(f"unreadable .keras archive {path}: {exc}") from exc
    else:
        try:
            import h5py
            with h5py.File(path, "r") as h:
                raw = h.attrs.get("model_config")
                if raw is None:
                    raise ModelIntegrityError(
                        f"{path} stores no model_config; architecture cannot be verified "
                        "(weights-only files are not acceptable for the current audit).")
                cfg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except ModelIntegrityError:
            raise
        except Exception as exc:
            raise ModelIntegrityError(f"unreadable HDF5 checkpoint {path}: {exc}") from exc
    layers = cfg.get("config", {}).get("layers", [])
    if not layers:
        raise ModelIntegrityError(f"{path} contains no layer configuration.")
    census: dict = {}
    l2 = set()
    for layer in layers:
        census[layer["class_name"]] = census.get(layer["class_name"], 0) + 1
        reg = layer.get("config", {}).get("kernel_regularizer")
        if reg:
            l2.add(reg.get("config", {}).get("l2"))
    census["_l2"] = sorted(v for v in l2 if v is not None)
    return census


def detect_depth(path) -> int:
    """
    Realized depth of a saved Gohr network.

    Two independent structural estimators must agree:
        residual merges   depth = #Add
        convolutions      depth = (#Conv1D - 1) / 2
    Disagreement means the file is not the reference architecture, and we
    refuse rather than guess.
    """
    census = _layer_census(path)
    adds = census.get("Add", 0)
    convs = census.get("Conv1D", 0)
    if convs < 1 or (convs - 1) % 2 != 0:
        raise ModelIntegrityError(
            f"{path}: Conv1D count {convs} is not of the form 1 + 2*depth; not the "
            "reference Gohr architecture.")
    by_conv = (convs - 1) // 2
    if adds != by_conv:
        raise ModelIntegrityError(
            f"{path}: inconsistent architecture - {adds} residual merges but Conv1D "
            f"count implies depth {by_conv}.")
    return adds


def verify_reference_checkpoint(path, *, expected_sha256: str = REFERENCE_CHECKPOINT_SHA256,
                                expected_depth: int = REFERENCE.depth) -> dict:
    """
    Full gate for a checkpoint the current audit intends to load.

    Checks content hash AND realized depth AND L2, and explicitly refuses
    any known historical depth-5 artifact.
    """
    path = Path(path)
    digest = sha256_file(path)
    if digest in HISTORICAL_DEPTH5_CHECKPOINTS:
        raise ModelIntegrityError(
            f"{path} is a HISTORICAL depth-5 checkpoint "
            f"({HISTORICAL_DEPTH5_CHECKPOINTS[digest]}). the current audit must use the verified "
            "depth-10 reference checkpoint; historical artifacts are evidence, not inputs.")
    depth = detect_depth(path)
    census = _layer_census(path)
    problems = []
    if expected_sha256 is not None and digest != expected_sha256:
        problems.append(f"sha256 {digest} != expected {expected_sha256}")
    if depth != expected_depth:
        problems.append(f"realized depth {depth} != expected {expected_depth}")
    if census["_l2"] and REFERENCE.l2_reg not in census["_l2"]:
        problems.append(f"L2 {census['_l2']} != reference {REFERENCE.l2_reg}")
    if problems:
        raise ModelIntegrityError(f"{path}: " + "; ".join(problems))
    return {"path": str(path), "sha256": digest, "realized_depth": depth,
            "conv1d": census.get("Conv1D"), "residual_merges": census.get("Add"),
            "l2": census["_l2"]}
