#!/usr/bin/env python3
"""
Derive DBitNet's expected tensor inventory from the PINNED SOURCE, then
check the architecture census against it.

WHY THIS EXISTS. The census reports what the shipped HDF5 files contain.
By itself that cannot adjudicate H-AUTOND-ARCH, whose question is whether
the artifact instantiates the architecture its description specifies. Two
things are needed on the other side of that comparison:

  (1) what the pinned source CONSTRUCTS  - derived here, analytically,
      by reimplementing dbitnet.make_model's shape algebra in pure Python
      with no TensorFlow import, so a framework version mismatch cannot
      be mistaken for an architecture mismatch; and
  (2) what the PAPER STATES              - a textual comparison, not
      derivable from code, and deliberately NOT asserted by this script.

This script establishes (1) and reconciles it with the census. It does
not adjudicate the hypothesis: it supplies the source side of the
comparison, and records that the paper side is outstanding.

PROVENANCE. Shape algebra transcribed from
  AutoND @ 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779, dbitnet.py lines 1-74
  get_dilation_rates: while input_size >= 8: append(input_size//2 - 1)
  per block: Conv1D(k=2, padding='valid', dilation=d) -> BN -> skip
             Conv1D(k=2, padding='causal', dilation=1) -> Add -> BN
             n_filters += n_add_filters
  head:      Flatten -> [Dense(d1) -> BN -> ReLU] x2 -> Dense(d2) -> BN
             -> ReLU -> Dense(1, sigmoid)
A Conv1D with kernel_size=2 and padding='valid' at dilation d maps a
length-L sequence to length L-d; padding='causal' preserves length.

Read-only. Imports no TensorFlow and no Keras.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"
DERIVATION_VERSION = "autond-dbitnet-source-derivation-v1"


def dilation_rates(input_size: int) -> list[int]:
    """Transcription of dbitnet.get_dilation_rates."""
    drs = []
    while input_size >= 8:
        drs.append(int(input_size / 2 - 1))
        input_size = input_size // 2
    return drs


def derive(input_size: int = 64, n_filters: int = 32, n_add_filters: int = 16,
           d1: int = 256, d2: int = 64) -> dict:
    """
    Walk make_model in construction order, emitting the Keras auto-generated
    layer name, the weight tensors it owns and their shapes.

    Layer names are deterministic given construction order, so predicting
    them makes the check sharper than a shape-multiset comparison: a
    reordered graph with the same tensor shapes would still disagree.
    """
    drs = dilation_rates(input_size)
    layers: list[dict] = []
    seq_chain = [input_size]
    counts = {"conv1d": 0, "batch_normalization": 0, "dense": 0,
              "add": 0, "activation": 0}

    def name(base: str) -> str:
        n = counts[base]
        counts[base] += 1
        return base if n == 0 else f"{base}_{n}"

    def conv(c_in: int, c_out: int) -> None:
        layers.append({"layer": name("conv1d"), "class": "Conv1D",
                       "tensors": [("kernel:0", [2, c_in, c_out]),
                                   ("bias:0", [c_out])]})

    def bn(width: int) -> None:
        layers.append({"layer": name("batch_normalization"),
                       "class": "BatchNormalization",
                       "tensors": [("gamma:0", [width]), ("beta:0", [width]),
                                   ("moving_mean:0", [width]),
                                   ("moving_variance:0", [width])],
                       "non_trainable": ["moving_mean:0", "moving_variance:0"]})

    def dense(f_in: int, f_out: int) -> None:
        layers.append({"layer": name("dense"), "class": "Dense",
                       "tensors": [("kernel:0", [f_in, f_out]),
                                   ("bias:0", [f_out])]})

    def bare(layer_name: str, cls: str) -> None:
        layers.append({"layer": layer_name, "class": cls, "tensors": []})

    # Input and the two normalisation ops (x-0.5)/0.5. These are raw tf ops
    # on a KerasTensor, which is also why make_model does not build under
    # Keras 3; they appear in the layer inventory carrying no weights.
    bare("input_1", "InputLayer")
    bare("tf.math.subtract", "TFOpLambda")
    bare("tf.math.truediv", "TFOpLambda")

    length, c_in, f = input_size, 1, n_filters
    for d in drs:
        conv(c_in, f)                    # padding='valid', dilation=d
        length -= d                      # kernel_size=2 -> L - d*(2-1)
        seq_chain.append(length)
        bn(f)
        conv(f, f)                       # padding='causal' -> length kept
        bare(name("add"), "Add")
        bn(f)
        c_in = f
        f += n_add_filters

    flat = length * c_in
    bare("flatten", "Flatten")
    dense(flat, d1); bn(d1); bare(name("activation"), "Activation")
    dense(d1, d1);  bn(d1); bare(name("activation"), "Activation")
    dense(d1, d2);  bn(d2); bare(name("activation"), "Activation")
    dense(d2, 1)                         # sigmoid output unit

    total = trainable = 0
    for l in layers:
        nt = set(l.get("non_trainable", ()))
        for tname, shape in l["tensors"]:
            n = 1
            for s in shape:
                n *= s
            l.setdefault("parameters", 0)
            l["parameters"] += n
            total += n
            if tname not in nt:
                trainable += n
        l.setdefault("parameters", 0)

    return {
        "derivation": DERIVATION_VERSION,
        "pinned_commit": PINNED_COMMIT,
        "method": ("shape algebra transcribed from dbitnet.py and evaluated in "
                   "pure Python; TensorFlow and Keras are not imported"),
        "hyperparameters": {"input_size": input_size, "n_filters": n_filters,
                            "n_add_filters": n_add_filters, "d1": d1, "d2": d2},
        "dilation_rates": drs,
        "sequence_length_chain": seq_chain,
        "flatten_width": flat,
        "layers": layers,
        "layer_count": len(layers),
        "tensor_count": sum(len(l["tensors"]) for l in layers),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "dense_layers": {
            "hidden": [d1, d1, d2],
            "output": "1 unit, sigmoid activation",
            "note": ("FOUR Dense layers are constructed: three hidden "
                     "(256, 256, 64) each followed by BatchNormalization and "
                     "ReLU, then a single-unit sigmoid output. A description "
                     "reading 'three dense layers 256/256/64' omits the output "
                     "unit and should be stated as three HIDDEN dense layers."),
        },
        "not_established_here": [
            "whether the PAPER's stated architecture matches this derivation "
            "(a textual comparison against the publication, outstanding)",
            "the numerical weight VALUES, which this derivation says nothing about",
            "that the shipped file was produced by this code rather than by any "
            "other code emitting the same tensor inventory",
        ],
    }


def reconcile(derived: dict, census_path: Path) -> dict:
    census = json.loads(census_path.read_text())
    expected_tensors = sorted(
        f"{l['layer']}|{t}|{shape}"
        for l in derived["layers"] for t, shape in l["tensors"])
    expected_names = [l["layer"] for l in derived["layers"]]

    results = []
    for m in census.get("models", []):
        got_tensors = sorted(
            f"{l['layer']}|{t['name'].split('/')[-1]}|{t['shape']}"
            for l in m["layers"] for t in l["tensors"])
        got_names = [l["layer"] for l in m["layers"]]
        d = m["derived"]
        checks = {
            "layer_names_and_order": got_names == expected_names,
            "tensor_names_and_shapes": got_tensors == expected_tensors,
            "layer_count": d["layer_count"] == derived["layer_count"],
            "tensor_count": d["tensor_count"] == derived["tensor_count"],
            "total_parameters": d["total_parameters"] == derived["total_parameters"],
            "trainable_parameters":
                d["trainable_parameters"] == derived["trainable_parameters"],
            "non_trainable_parameters":
                d["non_trainable_parameters"] == derived["non_trainable_parameters"],
        }
        mismatch = {
            "only_in_file": [t for t in got_tensors if t not in expected_tensors],
            "only_in_source": [t for t in expected_tensors if t not in got_tensors],
        } if not checks["tensor_names_and_shapes"] else {}
        results.append({
            "path": m["path"], "sha256": m["sha256"], "layout": m["layout"],
            "keras_version": m["observed"]["keras_version"],
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "tensor_mismatch": mismatch,
            "structural_digest": d["structural_digest"],
        })
    return {"census_file": census_path.name,
            "census_artifact": census.get("artifact"),
            "models": results,
            "all_models_reconcile": bool(results) and all(
                r["all_checks_pass"] for r in results)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive DBitNet from pinned source; reconcile with the census.")
    ap.add_argument("--census", type=Path, required=True,
                    help="JSON written by architecture_census_v2.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-size", type=int, default=64,
                    help="2 * plain_bits; 64 for Speck32/64 and Simon32/64")
    args = ap.parse_args()

    derived = derive(input_size=args.input_size)
    rec = reconcile(derived, args.census)
    out = {"artifact": DERIVATION_VERSION, "source_derivation": derived,
           "reconciliation": rec}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))

    d = derived
    print(f"DBitNet derived from source @ {PINNED_COMMIT[:12]} "
          f"(input_size={d['hyperparameters']['input_size']})\n")
    print(f"  dilation rates        {d['dilation_rates']}")
    print(f"  sequence length       {' -> '.join(map(str, d['sequence_length_chain']))}")
    print(f"  flatten width         {d['flatten_width']}")
    print(f"  layers / tensors      {d['layer_count']} / {d['tensor_count']}")
    print(f"  parameters            {d['total_parameters']:,} "
          f"({d['trainable_parameters']:,} trainable + "
          f"{d['non_trainable_parameters']:,} non-trainable)")
    print(f"\nReconciliation against {rec['census_file']}:\n")
    for r in rec["models"]:
        verdict = "RECONCILES" if r["all_checks_pass"] else "DISAGREES"
        print(f"  {verdict}  {r['path']}  [{r['layout']}, keras {r['keras_version']}]")
        for k, v in r["checks"].items():
            if not v:
                print(f"            FAIL {k}")
        for side, items in r.get("tensor_mismatch", {}).items():
            for t in items[:10]:
                print(f"            {side}: {t}")
    print(f"\n  all models reconcile: {rec['all_models_reconcile']}")
    print(f"\nWrote {args.output}")
    print("\nNOT established by this script: whether the PAPER's stated "
          "architecture matches this derivation. That is a textual comparison "
          "against the publication and remains outstanding; H-AUTOND-ARCH must "
          "not be adjudicated until it is done.")
    return 0 if rec["all_models_reconcile"] else 1


if __name__ == "__main__":
    sys.exit(main())
