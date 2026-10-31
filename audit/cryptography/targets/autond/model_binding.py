"""
AutoND checkpoint binding, weight hashing, architecture description and inference.

* ``load_published`` hash-verifies the published HDF5 file before building the model, loads it,
  then checks every array of every weight-bearing layer against the HDF5 datasets bit for bit, in
  order, and the architecture against an expected description.
* ``weights_sha256`` hashes dtype, shape and bytes of every weight in model order. Keras variable
  names are excluded because they depend on how many models the process has built.
* ``predict`` is upstream inference: ``model.predict`` (Keras inference mode), batch 5000.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .upstream import CHECKPOINT_SHA256, Upstream, sha256_file

PREDICT_BATCH = 5000


class BindingError(RuntimeError):
    pass


def weights_sha256(model) -> str:
    h = hashlib.sha256()
    for v in model.weights:
        a = np.ascontiguousarray(v.numpy())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def architecture(model) -> dict:
    return {"layers": len(model.layers),
            "weight_bearing_layers": len([l for l in model.layers if l.weights]),
            "params_total": int(model.count_params()),
            "params_trainable": int(sum(int(np.prod(v.shape)) for v in model.trainable_weights)),
            "params_non_trainable": int(sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)),
            "input_shape": list(model.input_shape), "output_shape": list(model.output_shape)}


def strict_compare(model, path) -> tuple[int, int, list, str | None]:
    """(weight groups in file, arrays in file, mismatches, HDF5 keras_version) for a loaded model."""
    import h5py
    wl = [l for l in model.layers if l.weights]
    mismatched = []
    with h5py.File(path, "r") as f:
        names = [x.decode() if isinstance(x, bytes) else x for x in f.attrs["layer_names"]]
        groups = []
        for nm in names:
            wn = [x.decode() if isinstance(x, bytes) else x for x in f[nm].attrs["weight_names"]]
            if wn:
                groups.append((nm, wn))
        n_arrays = sum(len(g[1]) for g in groups)
        if len(groups) != len(wl):
            mismatched.append(f"{len(groups)} weight groups in file vs {len(wl)} weight-bearing layers")
        for layer, (nm, wn) in zip(wl, groups):
            if len(layer.weights) != len(wn):
                mismatched.append(f"{layer.name}: {len(layer.weights)} vs {len(wn)} arrays")
                continue
            for v, w in zip(layer.weights, wn):
                ref = f[nm][w][()]
                if tuple(v.shape) != ref.shape or not np.array_equal(v.numpy(), ref):
                    mismatched.append(f"{layer.name}/{v.name} != {nm}/{w}")
        keras_version = f.attrs.get("keras_version")
    kv = keras_version.decode() if isinstance(keras_version, bytes) else keras_version
    return len(groups), n_arrays, mismatched, kv


def load_published(up: Upstream, expected_architecture: dict):
    path = up.checkpoint
    observed = sha256_file(path)
    if observed != CHECKPOINT_SHA256:
        raise BindingError(f"published checkpoint SHA-256 {observed} != {CHECKPOINT_SHA256}")
    model = up.dbitnet.make_model(2 * up.cipher.plain_bits)
    model.load_weights(str(path))
    n_groups, n_arrays, mismatched, kv = strict_compare(model, path)
    arch = architecture(model)
    arch["hdf5_weight_arrays"] = n_arrays
    diff = {k: {"observed": arch.get(k), "design": v} for k, v in expected_architecture.items() if arch.get(k) != v}
    binding = {"checkpoint": str(path), "sha256": observed, "sha256_verified": True, "hdf5_keras_version": kv,
               "strict_load": {"weight_groups_in_file": n_groups, "weight_arrays_in_file": n_arrays,
                               "all_arrays_bit_identical": not mismatched, "mismatches": mismatched},
               "architecture": arch, "architecture_matches_design": not diff, "architecture_differences": diff,
               "weights_sha256": weights_sha256(model)}
    if mismatched or diff:
        raise BindingError(f"published model binding failed: {json.dumps(binding, default=str)}")
    return model, binding


def build_untrained_twin(up: Upstream, seed: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return up.dbitnet.make_model(2 * up.cipher.plain_bits)


def load_untrained_twin(up: Upstream, path, expected_weights_sha256: str):
    model = up.dbitnet.make_model(2 * up.cipher.plain_bits)
    model.load_weights(str(path))
    got = weights_sha256(model)
    if got != expected_weights_sha256:
        raise BindingError(f"untrained twin weights hash {got} != recorded {expected_weights_sha256}")
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X, batch_size=PREDICT_BATCH, verbose=0), dtype=np.float32).ravel()


def inference_mode_check(model, X: np.ndarray) -> dict:
    a = predict(model, X)
    b = np.asarray(model(X.astype(np.float32), training=False), dtype=np.float32).ravel()
    return {"n": int(X.shape[0]), "finite": bool(np.all(np.isfinite(a))),
            "predict_vs_call_training_false_max_abs_diff": float(np.max(np.abs(a - b))),
            "min": float(a.min()), "max": float(a.max())}
