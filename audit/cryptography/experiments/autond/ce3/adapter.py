from __future__ import annotations

import hashlib
import os
from typing import Callable

import numpy as np

from ....targets.autond import model_binding as MB
from ....targets.autond import upstream as UP
from ..ce2 import adapter as CE2A
from ..ce2 import design as CE2D
from . import design as D
from .target_bins import apply_edges


class AdapterError(RuntimeError):
    pass


content_sha256 = CE2A.content_sha256
model_input = CE2A.model_input
generate_genuine_pairs = CE2A.generate_genuine_pairs
trajectory_and_weights = CE2A.trajectory_and_weights
random_pairs_input = CE2A.random_pairs_input


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frozen_edges() -> list[int]:
    if D.TARGET_EDGES is None:
        raise AdapterError("target edges are not recorded in the design; run and record P1 first")
    return list(D.TARGET_EDGES)


def classes_for(target, edges=None) -> np.ndarray:
    return apply_edges(target, frozen_edges() if edges is None else edges, D.N_CLASSES)


def main_dataset(up: UP.Upstream, n: int, randbytes: Callable[[int], bytes] = os.urandom) -> dict:
    gen = generate_genuine_pairs(up, n, randbytes=randbytes)
    return {"arrays": gen["arrays"], "gates": gen["gates"]}


def calibration_dataset(up: UP.Upstream, n_per_class: int, replicate_index: int,
                        randbytes: Callable[[int], bytes] = os.urandom) -> dict:
    gen = generate_genuine_pairs(up, n_per_class, randbytes=randbytes)
    g = gen["arrays"]
    Xr = random_pairs_input(n_per_class, randbytes=randbytes)
    n = 2 * n_per_class
    order = np.random.Generator(np.random.PCG64(D.CALIBRATION_SHUFFLE_SEED_BASE + int(replicate_index))).permutation(n)
    labels = np.concatenate([np.ones(n_per_class, np.int8), np.zeros(n_per_class, np.int8)])[order]
    arrays = {"key_words": g["key_words"], "pt0": g["pt0"], "pt1": g["pt1"], "c0": g["c0"], "c1": g["c1"],
              "random_inputs_packed": np.packbits(Xr, axis=1), "order": order.astype(np.int64), "labels": labels}
    return {"arrays": arrays, "gates": gen["gates"]}


def calibration_input(up: UP.Upstream, cal: dict) -> np.ndarray:
    Xg = model_input(up, cal["c0"], cal["c1"])
    Xr = np.unpackbits(cal["random_inputs_packed"], axis=1)[:, :64]
    return np.concatenate([Xg, Xr])[cal["order"]]


def load_models(up: UP.Upstream, twin_path):
    model, binding = MB.load_published(up, CE2D.DESIGN["reference"]["observed_architecture"])
    twin = MB.load_untrained_twin(up, twin_path, D.CE2_TWIN_WEIGHTS_SHA256)
    return model, twin, binding


def _layer_info(layer) -> dict:
    cfg = layer.get_config()
    return {"class": type(layer).__name__, "name": layer.name, "activation": cfg.get("activation"),
            "units": cfg.get("units"), "output_shape": list(layer.output_shape)}


def bind_representation(model) -> dict:
    r = D.REPRESENTATION
    layers = model.layers
    problems = []
    if len(layers) != r["n_layers"]:
        problems.append(f"{len(layers)} layers, expected {r['n_layers']}")
        return {"ok": False, "problems": problems}
    L, B, P, H = (layers[r["layer_index"]], layers[r["inbound"]["index"]], layers[r["pre_activation_dense"]["index"]],
                  layers[r["consumer"]["index"]])
    info = {k: _layer_info(v) for k, v in (("representation", L), ("inbound", B), ("pre_activation_dense", P),
                                            ("consumer", H))}
    if info["representation"]["class"] != r["layer_class"] or info["representation"]["activation"] != r["activation"]:
        problems.append(f"layer {r['layer_index']} is {info['representation']}")
    if info["representation"]["output_shape"] != r["output_shape"]:
        problems.append(f"representation output shape {info['representation']['output_shape']}")
    if info["inbound"]["class"] != r["inbound"]["class"]:
        problems.append(f"inbound layer is {info['inbound']}")
    p = r["pre_activation_dense"]
    if (info["pre_activation_dense"]["class"], info["pre_activation_dense"]["units"],
            info["pre_activation_dense"]["activation"]) != (p["class"], p["units"], p["activation"]):
        problems.append(f"pre-activation layer is {info['pre_activation_dense']}")
    c = r["consumer"]
    if (info["consumer"]["class"], info["consumer"]["units"], info["consumer"]["activation"]) != \
            (c["class"], c["units"], c["activation"]):
        problems.append(f"consumer layer is {info['consumer']}")
    if layers[-1] is not H or model.outputs[0] is not H.output:
        problems.append("the consumer is not the model output")

    def inbound(layer):
        return [x for node in layer._inbound_nodes for x in (node.inbound_layers if isinstance(node.inbound_layers, list)
                                                             else [node.inbound_layers])]
    if inbound(L) != [B]:
        problems.append("representation is not fed only by the batch-normalisation layer")
    if inbound(B) != [P]:
        problems.append("batch normalisation is not fed only by the 64-unit dense layer")
    if inbound(H) != [L]:
        problems.append("output layer is not fed only by the representation")
    consumers = [n.outbound_layer for n in L._outbound_nodes]
    if consumers != [H]:
        problems.append("representation has consumers other than the output layer")
    return {"ok": not problems, "problems": problems, "layers": info, "layer_index": r["layer_index"],
            "name_used_for_identity": False}


def extractor(model):
    import tensorflow as tf
    b = bind_representation(model)
    if not b["ok"]:
        raise AdapterError(f"representation binding failed: {b['problems']}")
    return tf.keras.Model(model.input, model.layers[D.REPRESENTATION["layer_index"]].output)


def extract(ext, X: np.ndarray) -> np.ndarray:
    return np.asarray(ext.predict(X, batch_size=MB.PREDICT_BATCH, verbose=0), dtype=np.float32)


def head(model):
    W, b = model.layers[D.REPRESENTATION["consumer"]["index"]].get_weights()
    return np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64)


def head_difference(model, h: np.ndarray, score: np.ndarray) -> float:
    W, b = head(model)
    z = h.astype(np.float64) @ W[:, 0] + b[0]
    return float(np.max(np.abs(1.0 / (1.0 + np.exp(-z)) - score.astype(np.float64)))) if h.shape[0] else 0.0


def n_nonconstant(h: np.ndarray) -> int:
    return int(np.count_nonzero(np.var(h.astype(np.float64), axis=0) > 0.0))
