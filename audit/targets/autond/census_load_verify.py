#!/usr/bin/env python3
"""
TensorFlow-side load verification for AutoND checkpoints.

WHY THIS EXISTS, AND WHY IT IS A SEPARATE INSTRUMENT FROM THE CENSUS.
`architecture_census_v2.py` reads the HDF5 file with h5py and never
imports TensorFlow, deliberately: a framework version mismatch must not
be mistakable for an architecture mismatch. That design means it cannot
answer one question the audit also needs - whether the checkpoint
actually LOADS into the model the pinned source constructs, consuming
every tensor. This file answers that, and only that.

WHAT A SUCCESSFUL load_weights() DOES NOT PROVE.
Keras's HDF5 topological load path matches the file's layer inventory
against the model's weighted layers in order. It raises on a count or
shape mismatch, but a call that returns without raising is NOT by itself
evidence that every tensor in the file reached a variable with the right
value: the call is a side effect whose success is reported by the absence
of an exception. This tool therefore does not infer consumption from the
call. After loading it compares EVERY model variable element-wise
against the dataset in the file that should have populated it, and
separately records whether the variable changed from its initialised
value. Exact equality with the file is the positive statement;
`changed_on_load` is the corroborating one.

MAPPING, AND WHY IT IS POSITIONAL RATHER THAN BY NAME. In the
weights-only layout the root attribute `layer_names` lists every layer,
including weightless ones, and each layer group carries `weight_names`.
Keras's HDF5 topological load filters both sides to layers that own
weights and zips them IN ORDER. This tool reproduces that filtering and
that ordering explicitly.

It deliberately does NOT join on layer names. Keras auto-names layers
from a counter that is global to the Python session, so the first
`make_model()` call in a process produces `conv1d`, `batch_normalization`
and so on, while a later call in the same process produces `conv1d_128`,
`batch_normalization_176`. A name-keyed verifier therefore works on the
first model built and silently fails to match on every subsequent one -
which is exactly the kind of accidental, state-dependent agreement this
instrument exists to rule out. The names on both sides are RECORDED, and
their agreement is reported as an observation, but the correspondence
that is checked is the positional one Keras itself uses.
`clear_session()` is called before building so that a single-model run
also produces the canonical names.

FAIL CLOSED. Any missing tensor, extra tensor, shape disagreement, value
disagreement, non-finite value, or layer-sequence mismatch is a REFUSAL.
There is no "succeeded with warnings" result.

ENVIRONMENT. Requires the TensorFlow/Keras stack the target itself
requires. AutoND's `dbitnet.make_model` applies `tf.subtract` and
`tf.divide` to a KerasTensor (dbitnet.py lines 35-36), which Keras 3
rejects, so this runs under TensorFlow 2.15.x / Keras 2.15.x. The stack
actually used is recorded in the output rather than asserted here.

Read-only with respect to the target: opens the checkpoint with mode='r',
imports the pinned source without modifying it, and writes only its own
output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np

VERIFIER_VERSION = "autond-census-load-verify-v1"

#: Layers whose groups legitimately carry no weights. Recorded rather
#: than hard-coded as an expectation: the file states which layers have
#: weights via `weight_names`, and that statement is what is used.
WEIGHTLESS_IS_NORMAL = True


class LoadVerificationRefusal(RuntimeError):
    """Raised instead of returning a partial or unverified result."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return [_decode(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_decode(x) for x in v]
    return v


def file_inventory(path: Path) -> dict:
    """
    Read the file's own statement of what it contains, with h5py only.

    Returns the ordered layer inventory, the per-layer declared weight
    names, and the dataset for each, keyed by (layer, weight_name).
    """
    if not path.is_file():
        raise LoadVerificationRefusal(f"checkpoint not found: {path}")
    if not h5py.is_hdf5(path):
        raise LoadVerificationRefusal(f"{path.name}: not an HDF5 file.")

    inv = {"layer_names": [], "weighted_layers": [], "weighted": [],
           "tensor_count": 0, "layout": None}
    with h5py.File(path, "r") as f:
        if "model_weights" in f and "model_config" in f.attrs:
            inv["layout"] = "full_model"
            root = f["model_weights"]
        elif "layer_names" in f.attrs:
            inv["layout"] = "weights_only"
            root = f
        else:
            raise LoadVerificationRefusal(
                f"{path.name}: unrecognised layout; no /model_weights group and no "
                "root 'layer_names' attribute.")

        names = (_decode(root.attrs["layer_names"]) if "layer_names" in root.attrs
                 else _decode(f.attrs["layer_names"]))
        inv["layer_names"] = list(names)
        missing = [n for n in names if n not in root]
        if missing:
            raise LoadVerificationRefusal(
                f"{path.name}: {len(missing)} layer(s) named in 'layer_names' are "
                f"absent from the file: {missing[:8]}. Refusing.")

        declared_paths = set()
        for name in names:
            g = root[name]
            wnames = _decode(g.attrs["weight_names"]) if "weight_names" in g.attrs else []
            if not wnames:
                continue
            inv["weighted_layers"].append(name)
            weights = []
            for w in wnames:
                if w not in g:
                    raise LoadVerificationRefusal(
                        f"{path.name}: layer {name!r} declares weight {w!r} in "
                        "'weight_names' but the dataset is absent. Refusing.")
                d = g[w]
                declared_paths.add(f"{name}/{w}")
                weights.append({
                    "name": w,
                    "shape": tuple(int(x) for x in d.shape),
                    "dtype": str(d.dtype),
                    "value": np.asarray(d),
                })
            inv["weighted"].append({"layer": name, "weights": weights})
            inv["tensor_count"] += len(weights)

        # Any dataset not declared by a weight_names entry is unexpected.
        seen = []

        def visit(p, obj):
            if isinstance(obj, h5py.Dataset):
                seen.append(p)

        root.visititems(visit)
        inv["undeclared_datasets"] = sorted(p for p in seen if p not in declared_paths)
        inv["declared_count"] = len(declared_paths)
    return inv


def weight_fingerprint(pairs) -> str:
    """
    sha256 over the loaded state in canonical order. Binds a specific set
    of weight VALUES, so a later tool can assert it evaluated the same
    loaded model rather than merely the same file.
    """
    h = hashlib.sha256()
    for layer, wname, arr in sorted(pairs, key=lambda t: (t[0], t[1])):
        h.update(f"{layer}|{wname}|{tuple(arr.shape)}|{arr.dtype}".encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def verify(checkpoint: Path, autond_root: Path, input_size: int = 64) -> dict:
    sys.path.insert(0, str(autond_root))
    try:
        import tensorflow as tf  # noqa: F401  (imported for version record)
        import dbitnet
    except Exception as exc:                                        # noqa: BLE001
        raise LoadVerificationRefusal(
            f"cannot import the pinned AutoND source from {autond_root}: "
            f"{type(exc).__name__}: {exc}. dbitnet.make_model applies tf.subtract/"
            "tf.divide to a KerasTensor, which Keras 3 rejects; this tool requires "
            "the TensorFlow 2.15.x / Keras 2.15.x stack the target itself requires."
        ) from exc

    import keras

    rec = {
        "verifier": VERIFIER_VERSION,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "autond_root": str(autond_root),
        "input_size": input_size,
        "environment": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "problems": [],
    }

    inv = file_inventory(checkpoint)
    rec["layout"] = inv["layout"]
    rec["file_layer_names"] = inv["layer_names"]
    rec["file_weighted_layers"] = inv["weighted_layers"]
    rec["file_undeclared_datasets"] = inv["undeclared_datasets"]

    # Reset the session-global layer-name counter so a single-model run
    # produces Keras's canonical names. The verification below does not
    # depend on this; it is for legible output.
    keras.backend.clear_session()
    model = dbitnet.make_model(input_size)
    model_weighted = [l for l in model.layers if l.weights]
    rec["model_layer_count"] = len(model.layers)
    rec["model_weighted_layers"] = [l.name for l in model_weighted]

    # Names are an observation, never the join key: Keras's auto-naming
    # counter is global to the session, so names disagree whenever more
    # than one model has been built in this process.
    rec["layer_names_agree"] = rec["model_weighted_layers"] == inv["weighted_layers"]

    if len(model_weighted) != len(inv["weighted"]):
        raise LoadVerificationRefusal(
            f"{checkpoint.name}: the model constructed from the pinned source has "
            f"{len(model_weighted)} weighted layer(s); the file declares "
            f"{len(inv['weighted'])}. These cannot correspond. Refusing.")

    # Values before the load, so 'did the load actually write' is answerable.
    before = [[np.array(w.numpy(), copy=True) for w in layer.weights]
              for layer in model_weighted]

    try:
        model.load_weights(str(checkpoint), by_name=False)
    except Exception as exc:                                        # noqa: BLE001
        raise LoadVerificationRefusal(
            f"load_weights(by_name=False) raised {type(exc).__name__}: {exc}") from exc
    rec["load_weights_raised"] = False

    tensors = []
    consumed = 0
    for idx, (layer, fl) in enumerate(zip(model_weighted, inv["weighted"])):
        if len(layer.weights) != len(fl["weights"]):
            rec["problems"].append(
                f"position {idx}: model layer {layer.name!r} owns {len(layer.weights)} "
                f"variable(s); file layer {fl['layer']!r} declares "
                f"{len(fl['weights'])}.")
            continue
        for j, (w, src) in enumerate(zip(layer.weights, fl["weights"])):
            got = np.array(w.numpy(), copy=True)
            row = {"position": [idx, j],
                   "model_layer": layer.name, "model_weight": w.name,
                   "file_layer": fl["layer"], "file_weight": src["name"],
                   "model_shape": list(got.shape), "model_dtype": str(got.dtype),
                   "file_shape": list(src["shape"]), "file_dtype": src["dtype"]}
            if tuple(got.shape) != src["shape"]:
                row["status"] = "SHAPE_MISMATCH"
                rec["problems"].append(
                    f"{layer.name}/{w.name}: model shape {tuple(got.shape)} vs "
                    f"file shape {src['shape']} at position {[idx, j]}.")
            elif not np.array_equal(got, src["value"]):
                row["status"] = "VALUE_MISMATCH"
                row["max_abs_difference"] = float(
                    np.max(np.abs(got.astype(np.float64)
                                  - src["value"].astype(np.float64))))
                rec["problems"].append(
                    f"{layer.name}/{w.name}: loaded value differs from the file. "
                    "load_weights returned without raising, so this is exactly the "
                    "silent non-consumption the call cannot rule out.")
            else:
                row["status"] = "CONSUMED_EXACTLY"
                consumed += 1
            row["changed_on_load"] = not np.array_equal(before[idx][j], got)
            if not np.all(np.isfinite(got)):
                row["status"] = "NON_FINITE"
                rec["problems"].append(
                    f"{layer.name}/{w.name}: loaded value contains NaN or Inf.")
            tensors.append(row)

    if inv["undeclared_datasets"]:
        rec["problems"].append(
            f"{len(inv['undeclared_datasets'])} dataset(s) in the file are not declared "
            f"by any layer's 'weight_names': {inv['undeclared_datasets'][:8]}")

    rec["tensors"] = tensors
    rec["tensor_count_model"] = sum(len(l.weights) for l in model_weighted)
    rec["tensor_count_file"] = inv["tensor_count"]
    if rec["tensor_count_model"] != rec["tensor_count_file"]:
        rec["problems"].append(
            f"tensor count differs: model {rec['tensor_count_model']} vs file "
            f"{rec['tensor_count_file']}.")
    rec["unconsumed_file_tensors"] = sorted(
        f"{t['file_layer']}/{t['file_weight']}" for t in tensors
        if t.get("status") != "CONSUMED_EXACTLY")
    rec["consumed_exactly"] = consumed
    rec["unchanged_on_load"] = sorted(
        f"{t['model_layer']}/{t['model_weight']}" for t in tensors
        if t.get("changed_on_load") is False)

    total = int(sum(np.prod(t["model_shape"]) for t in tensors)) if tensors else 0
    rec["total_parameters_loaded"] = total
    rec["loaded_weight_fingerprint"] = weight_fingerprint(
        [(l.name, w.name, np.array(w.numpy(), copy=True))
         for l in model_weighted for w in l.weights])

    rec["verified"] = not rec["problems"]
    rec["establishes"] = [
        "the pinned source constructs a model whose weighted-layer sequence and "
        "per-variable shapes match this file's own declared inventory",
        "every model variable holds a value element-wise identical to the dataset in "
        "the file that should have populated it",
        "no dataset in the file was left unconsumed and none is undeclared",
    ] if rec["verified"] else []
    rec["does_not_establish"] = [
        "that the checkpoint was PRODUCED by this source or by the published training "
        "procedure; file identity and loadability are not provenance",
        "anything about the model's predictive behaviour, which is a separate "
        "inference measurement",
        "that the computation graph matches the publication's prose description, "
        "which is a textual comparison this tool does not perform",
        "that the layer NAMES correspond: Keras's auto-naming counter is global to the "
        "session, so names are recorded as an observation and the checked "
        "correspondence is the positional one Keras's own loader uses",
    ]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify that an AutoND checkpoint loads completely into the "
                    "model the pinned source constructs.")
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--autond-root", type=Path, required=True,
                    help="path to the pinned AutoND checkout (contains dbitnet.py)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-size", type=int, default=64,
                    help="2 * plain_bits; 64 for Speck32/64 and Simon32/64")
    args = ap.parse_args()

    records, refusals = [], []
    for p in args.checkpoints:
        try:
            records.append(verify(p, args.autond_root, args.input_size))
        except LoadVerificationRefusal as exc:
            refusals.append({"checkpoint": str(p), "refusal": str(exc)})
        except Exception as exc:                                    # noqa: BLE001
            refusals.append({"checkpoint": str(p),
                             "refusal": f"{type(exc).__name__}: {exc}"})

    out = {"artifact": VERIFIER_VERSION, "models": records, "refusals": refusals,
           "all_verified": bool(records) and not refusals
                           and all(r["verified"] for r in records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))

    print(f"Wrote {args.output}\n")
    for r in records:
        print(f"  {r['checkpoint']}  [{r['layout']}]")
        print(f"    sha256              {r['checkpoint_sha256']}")
        print(f"    stack               tf {r['environment']['tensorflow']} / "
              f"keras {r['environment']['keras']}")
        print(f"    weighted layers     model {len(r['model_weighted_layers'])} / "
              f"file {len(r['file_weighted_layers'])}  "
              f"(names agree: {r['layer_names_agree']}; matched positionally)")
        print(f"    tensors             model {r['tensor_count_model']} / "
              f"file {r['tensor_count_file']}")
        print(f"    consumed exactly    {r['consumed_exactly']} / {r['tensor_count_model']}")
        print(f"    parameters loaded   {r['total_parameters_loaded']:,}")
        print(f"    weight fingerprint  {r['loaded_weight_fingerprint']}")
        if r["unchanged_on_load"]:
            print(f"    UNCHANGED ON LOAD   {r['unchanged_on_load']}")
        print(f"    VERIFIED            {r['verified']}")
        for p in r["problems"]:
            print(f"      PROBLEM {p}")
        print()
    for x in refusals:
        print(f"  REFUSED {x['checkpoint']}\n          {x['refusal']}\n")
    return 0 if out["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
