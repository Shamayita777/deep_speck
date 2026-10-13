#!/usr/bin/env python3
"""
Ordered architecture signature - GENERIC.

No Gohr, Speck or cipher knowledge. Keras is imported lazily inside the
one function that needs a live model, so this module is importable (and
its digest comparable) without TensorFlow installed.

WHY COUNTS ARE NOT ENOUGH
-------------------------
The first version of the pre-training probe compared counts by layer
class plus the total parameter count. That is too weak to establish the
premise of an implementation comparison, and demonstrably so: two models
can agree on (21 Conv1D, 3 Dense, 23 BatchNormalization, 10 Add,
102,497 parameters) while differing in

  * ORDER - a residual Add placed after one convolution instead of two;
  * ACTIVATION PLACEMENT - `Conv1D(activation='relu')` versus a separate
    `Activation('relu')` layer, which changes where the non-linearity
    sits relative to BatchNormalization and is invisible to a count of
    Conv1D layers;
  * PADDING, STRIDES or DILATION - 'same' versus 'causal' changes the
    receptive field with no change to any count or shape;
  * BATCHNORM SETTINGS - axis, momentum, epsilon, center, scale;
  * REGULARIZATION - an L2 of 1e-4 instead of 1e-5 changes the training
    objective and leaves every count and shape identical. This project
    has already been bitten by exactly that: the EV adapter omitted
    reg_param and silently used the 1e-4 default.
  * TENSOR SHAPES - two architectures can share a parameter total while
    distributing it differently.

A measured difference between two models that differ in any of these is
not a difference between implementations; it is a difference between
architectures wearing the same parameter count. This module makes that
confusion impossible to reach by accident.

WHAT THE SIGNATURE IS
---------------------
An ordered list, one entry per layer in `model.layers` order, of
class name, output shape, the class-relevant configuration, and the
ordered shapes and dtypes of that layer's weights - plus the model's
input and output shapes and its parameter counts. Hashed to one digest.

LAYER NAMES ARE DELIBERATELY EXCLUDED. Keras auto-generates names
('conv1d', 'conv1d_1', ...) and two independent implementations will
not agree on them even when they build the identical graph. Including
names would make every comparison fail for a cosmetic reason. WEIGHT
VALUES are excluded for the same reason in reverse: the signature must
be identical for two freshly initialized models and must stay identical
across training, so that it can be checked both before training and
against the persisted artifact afterwards.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

SIGNATURE_VERSION = "architecture-signature-v2"

#: Configuration keys captured for every layer that declares them. The
#: list is deliberately broad: a key a layer does not have is skipped,
#: so adding a key here can only make the signature stricter.
CONFIG_KEYS = (
    # convolution / dense geometry
    "filters", "units", "kernel_size", "strides", "padding", "dilation_rate",
    "groups", "data_format",
    # non-linearity and bias
    "activation", "use_bias",
    # initialization - two models identical in every other respect can
    # train differently if one uses a different kernel initializer
    "kernel_initializer", "bias_initializer", "beta_initializer", "gamma_initializer",
    "moving_mean_initializer", "moving_variance_initializer", "depthwise_initializer",
    # normalization
    "axis", "momentum", "epsilon", "center", "scale",
    # shape manipulation
    "target_shape", "dims", "rate", "pool_size",
    # output
    "final_activation",
)

#: Regularizer / initializer / constraint slots whose configuration is
#: captured. An L2 difference changes the training objective while
#: leaving every count, shape and activation identical.
REGULARIZER_KEYS = (
    "kernel_regularizer", "bias_regularizer", "activity_regularizer",
    "beta_regularizer", "gamma_regularizer",
)
CONSTRAINT_KEYS = ("kernel_constraint", "bias_constraint")


class SignatureMismatch(RuntimeError):
    """Two architecture signatures that were required to agree do not."""


def _plain(value: Any) -> Any:
    """Canonicalize a config value so equal configurations compare equal."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        # Keras serializes regularizers as {'module':..., 'class_name':...,
        # 'config': {...}, 'registered_name':...}. Only class_name and
        # config carry meaning; module and registered_name vary across
        # Keras versions and would make the digest version-dependent.
        if "class_name" in value:
            return {"class_name": value.get("class_name"),
                    "config": _plain(value.get("config") or {})}
        return {k: _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        # Round to a fixed precision so that 1e-05 and 1.0000000001e-05
        # from different serialization paths do not differ.
        return float(f"{value:.12g}")
    return str(value)


def _shape(obj) -> Optional[list]:
    try:
        return [None if d is None else int(d) for d in tuple(obj.shape)]
    except Exception:                                            # noqa: BLE001
        return None


def layer_signature(layer) -> dict[str, Any]:
    """Signature of one layer. Excludes the layer's name and weight values."""
    config = {}
    try:
        config = layer.get_config() or {}
    except Exception:                                            # noqa: BLE001
        config = {}

    entry: dict[str, Any] = {"class_name": type(layer).__name__}
    for key in CONFIG_KEYS:
        if key in config:
            entry[key] = _plain(config[key])
    for key in REGULARIZER_KEYS + CONSTRAINT_KEYS:
        if key in config and config[key] is not None:
            entry[key] = _plain(config[key])

    try:
        entry["output_shape"] = _shape(layer.output)
    except Exception:                                            # noqa: BLE001
        entry["output_shape"] = None

    weights = []
    try:
        for w in layer.weights:
            weights.append({"shape": _shape(w), "dtype": str(w.dtype),
                            "trainable": bool(getattr(w, "trainable", True))})
    except Exception:                                            # noqa: BLE001
        pass
    entry["weights"] = weights
    return entry


def graph_connectivity(model) -> Optional[list[list[list[int]]]]:
    """
    Per layer, in `model.layers` order, the ordered indices of the layers
    feeding it.

    WHY THIS IS SEPARATE FROM THE LAYER LIST. Two models can agree on
    every layer's class, configuration, output shape and weight shapes,
    and on the total parameter count, while differing in WHICH TENSOR
    feeds a residual merge. That difference was demonstrated to be
    invisible to the v1 signature: a 13-layer model whose `Add` was fed
    from the post-activation tensor of block 1 instead of the block
    input produced a byte-identical digest. For a residual architecture
    that is the architecture's defining property, so a signature blind
    to it cannot establish the premise of an implementation comparison.

    NAME-INDEPENDENT BY CONSTRUCTION. Keras records connectivity by
    layer NAME, and two independent implementations never agree on
    auto-generated names ('conv1d_7' and so on). Names are therefore
    mapped to positional indices before hashing, so the connectivity
    compares equal exactly when the graph shape is equal.

    Returns None for a model whose config carries no connectivity (a
    plain Sequential), in which case the layer order already determines
    the graph and nothing is lost.
    """
    try:
        config = model.get_config()
    except Exception:                                            # noqa: BLE001
        return None
    entries = config.get("layers")
    if not isinstance(entries, list) or not entries:
        return None

    index_of: dict[str, int] = {}
    for i, entry in enumerate(entries):
        name = (entry.get("config") or {}).get("name") or entry.get("name")
        if name is not None:
            index_of[name] = i

    def sources(node) -> list[list[int]]:
        """Pull every keras_history reference out of one inbound node."""
        found: list[list[int]] = []

        def walk(obj):
            if isinstance(obj, dict):
                history = (obj.get("config") or {}).get("keras_history")
                if isinstance(history, (list, tuple)) and history:
                    src = index_of.get(history[0])
                    tail = [int(x) for x in history[1:] if isinstance(x, int)]
                    found.append([-1 if src is None else src, *tail])
                    return
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    walk(value)

        walk(node)
        return found

    connectivity: list[list[list[int]]] = []
    any_edges = False
    for entry in entries:
        inbound = entry.get("inbound_nodes") or []
        edges: list[list[int]] = []
        for node in inbound:
            edges.extend(sources(node))
        if edges:
            any_edges = True
        connectivity.append(edges)
    return connectivity if any_edges else None


def architecture_signature(model) -> dict[str, Any]:
    """
    Full ordered signature of a built Keras model, plus its digest.

    Callable both on a freshly constructed model (before training) and
    on a model reloaded from disk (after training): the signature is
    invariant to weight VALUES, so the same digest must appear at both
    points, and a mismatch means the persisted artifact is not the model
    that was trained.
    """
    layers = [layer_signature(layer) for layer in model.layers]
    try:
        trainable = int(sum(int(w.shape.num_elements() or 0)
                            for w in model.trainable_weights))
    except Exception:                                            # noqa: BLE001
        import numpy as _np
        trainable = int(sum(int(_np.prod(w.shape)) for w in model.trainable_weights))
    total = int(model.count_params())

    body = {
        "signature_version": SIGNATURE_VERSION,
        "layer_count": len(layers),
        "layers": layers,
        "connectivity": graph_connectivity(model),
        "input_shape": _plain(getattr(model, "input_shape", None)),
        "output_shape": _plain(getattr(model, "output_shape", None)),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }
    body["digest"] = signature_digest(body)
    return body


def signature_digest(signature: dict[str, Any]) -> str:
    """Digest over everything except the digest field itself."""
    body = {k: v for k, v in signature.items() if k != "digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode("utf-8")).hexdigest()


def verify_signature(signature: dict[str, Any]) -> None:
    """Recompute and check a signature's own digest."""
    recomputed = signature_digest(signature)
    if recomputed != signature.get("digest"):
        raise SignatureMismatch(
            f"signature digest {signature.get('digest')} does not match its content "
            f"(recomputed {recomputed}); the record has been altered.")


def diff_signatures(a: dict[str, Any], b: dict[str, Any], *,
                    label_a: str = "a", label_b: str = "b",
                    max_report: int = 12) -> list[str]:
    """
    Human-readable differences between two signatures.

    Exists so that an abort message names WHAT differs. "Signatures
    differ" sends an operator to read 84 layers by hand; "layer 12:
    padding same vs causal" is actionable, and for an implementation
    comparison it is very often the actual finding.
    """
    problems: list[str] = []
    for key in ("input_shape", "output_shape", "total_parameters",
                "trainable_parameters", "non_trainable_parameters", "layer_count"):
        if a.get(key) != b.get(key):
            problems.append(f"{key}: {label_a}={a.get(key)!r} {label_b}={b.get(key)!r}")

    ca, cb = a.get("connectivity"), b.get("connectivity")
    if ca != cb:
        if ca is None or cb is None:
            problems.append(
                f"connectivity: {label_a}={'absent' if ca is None else 'present'} "
                f"{label_b}={'absent' if cb is None else 'present'}")
        else:
            for i in range(min(len(ca), len(cb))):
                if ca[i] != cb[i]:
                    cls = (a.get("layers") or [{}])[i].get("class_name") if i < len(
                        a.get("layers") or []) else "?"
                    problems.append(
                        f"layer {i} ({cls}) INPUT WIRING: {label_a} is fed by layer(s) "
                        f"{[e[0] for e in ca[i]]}, {label_b} by {[e[0] for e in cb[i]]}")

    la, lb = a.get("layers", []), b.get("layers", [])
    for i in range(min(len(la), len(lb))):
        if la[i] == lb[i]:
            continue
        keys = sorted(set(la[i]) | set(lb[i]))
        for k in keys:
            if la[i].get(k) != lb[i].get(k):
                problems.append(
                    f"layer {i} ({la[i].get('class_name')}/{lb[i].get('class_name')}) "
                    f"{k}: {label_a}={la[i].get(k)!r} {label_b}={lb[i].get(k)!r}")
        if len(problems) >= max_report:
            problems.append(f"... further differences suppressed after {max_report}")
            break
    if len(la) != len(lb):
        problems.append(f"layer count: {label_a}={len(la)} {label_b}={len(lb)}")
    return problems


def require_identical(a: dict[str, Any], b: dict[str, Any], *,
                      label_a: str, label_b: str, context: str = "") -> None:
    """Raise SignatureMismatch, naming the differences, unless equal."""
    if a.get("digest") == b.get("digest"):
        return
    problems = diff_signatures(a, b, label_a=label_a, label_b=label_b)
    raise SignatureMismatch(
        f"{context}architecture signatures of {label_a} and {label_b} differ "
        f"({label_a}={a.get('digest')}, {label_b}={b.get('digest')}):\n  "
        + "\n  ".join(problems if problems
                      else ["digests differ but no field-level difference was located; "
                            "the signature schema may have changed"]))
