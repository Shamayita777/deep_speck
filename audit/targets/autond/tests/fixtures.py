#!/usr/bin/env python3
"""
Synthetic HDF5 fixtures for the architecture-census tests.

WHY SYNTHETIC. The failure modes under test are a missing layer group, an
unrecognised layout, a zero-parameter file and a digest collision between
structurally different models. Producing those by damaging a real
research artifact would destroy evidence, so every failure fixture is
built here from scratch. The real checkpoints are exercised separately,
read-only, by test_architecture_census_real.py.

FAITHFULNESS. The layouts below reproduce what Keras actually writes, as
observed in AutoND's shipped SPECK checkpoint:

  weights-only   root attrs: backend, keras_version, layer_names
                 one group per layer, named by layer, carrying attr
                 weight_names; datasets live INSIDE that group at the
                 path "<layer>/<weight>:0", so a group named "conv1d"
                 contains "conv1d/kernel:0" and "conv1d/bias:0"
                 weightless layers are present as EMPTY groups whose
                 weight_names attribute is an empty list

  full model     the same structure under a /model_weights group, plus a
                 root `model_config` attribute, optionally
                 /optimizer_weights

Nothing here imports TensorFlow.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

#: A small two-block network. Shapes are arbitrary but internally
#: consistent; the tests assert arithmetic over them, not their realism.
SPEC_A = [
    ("input_1", []),
    ("conv1d", [("conv1d/kernel:0", (2, 1, 8)), ("conv1d/bias:0", (8,))]),
    ("batch_normalization", [("batch_normalization/gamma:0", (8,)),
                             ("batch_normalization/beta:0", (8,)),
                             ("batch_normalization/moving_mean:0", (8,)),
                             ("batch_normalization/moving_variance:0", (8,))]),
    ("add", []),
    ("dense", [("dense/kernel:0", (8, 4)), ("dense/bias:0", (4,))]),
]

#: Structurally DIFFERENT from SPEC_A: wider conv, different dense shape.
#: Used for the regression test that a collector which cannot read a file
#: must not hand two different architectures one digest.
SPEC_B = [
    ("input_1", []),
    ("conv1d", [("conv1d/kernel:0", (2, 1, 16)), ("conv1d/bias:0", (16,))]),
    ("batch_normalization", [("batch_normalization/gamma:0", (16,)),
                             ("batch_normalization/beta:0", (16,)),
                             ("batch_normalization/moving_mean:0", (16,)),
                             ("batch_normalization/moving_variance:0", (16,))]),
    ("add", []),
    ("dense", [("dense/kernel:0", (16, 4)), ("dense/bias:0", (4,))]),
]


def _bytes_array(names):
    return np.array([n.encode("utf-8") for n in names], dtype="S")


def spec_parameters(spec) -> tuple[int, int]:
    """(total, non_trainable) implied by a spec, computed independently."""
    total = non_trainable = 0
    for _layer, weights in spec:
        for wname, shape in weights:
            n = int(np.prod(shape)) if shape else 1
            total += n
            if wname.rsplit("/", 1)[-1].split(":")[0] in ("moving_mean", "moving_variance"):
                non_trainable += n
    return total, non_trainable


def _write_layers(root, spec, *, drop_groups=(), seed=0):
    rng = np.random.default_rng(seed)
    root.attrs["layer_names"] = _bytes_array([name for name, _ in spec])
    for name, weights in spec:
        if name in drop_groups:
            continue                      # declared in layer_names, absent here
        g = root.create_group(name)
        g.attrs["weight_names"] = _bytes_array([w for w, _ in weights])
        for wname, shape in weights:
            g.create_dataset(wname, data=rng.standard_normal(shape).astype(np.float32))


def write_weights_only(path: Path, spec=SPEC_A, *, drop_groups=(), seed=0,
                       keras_version="2.7.0") -> Path:
    with h5py.File(path, "w") as f:
        f.attrs["backend"] = b"tensorflow"
        f.attrs["keras_version"] = keras_version.encode()
        _write_layers(f, spec, drop_groups=drop_groups, seed=seed)
    return path


def write_full_model(path: Path, spec=SPEC_A, *, seed=0, with_optimizer=True,
                     keras_version="2.15.0") -> Path:
    with h5py.File(path, "w") as f:
        f.attrs["backend"] = b"tensorflow"
        f.attrs["keras_version"] = keras_version.encode()
        f.attrs["model_config"] = json.dumps(
            {"class_name": "Functional", "config": {"name": "fixture"}}).encode()
        f.attrs["training_config"] = json.dumps({"loss": "mse"}).encode()
        _write_layers(f.create_group("model_weights"), spec, seed=seed)
        if with_optimizer:
            og = f.create_group("optimizer_weights")
            og.create_dataset("iteration:0", data=np.int64(1234))
            og.create_dataset("Adam/m/dense/kernel:0",
                              data=np.zeros((8, 4), dtype=np.float32))
    return path


def write_unrecognised_hdf5(path: Path) -> Path:
    """Valid HDF5, but neither Keras layout: no layer_names, no model_weights."""
    with h5py.File(path, "w") as f:
        f.attrs["something_else"] = b"not a keras checkpoint"
        f.create_dataset("payload", data=np.arange(10, dtype=np.float32))
    return path


def write_zero_parameter(path: Path) -> Path:
    """Recognised weights-only layout whose every layer is weightless."""
    spec = [("input_1", []), ("add", []), ("flatten", [])]
    return write_weights_only(path, spec)


def write_not_hdf5(path: Path) -> Path:
    path.write_bytes(b"PK\x03\x04 this is a zip archive, not HDF5\n")
    return path
