#!/usr/bin/env python3
"""
Architecture census v2 - repaired collector for Keras HDF5 checkpoints.

WHY v2 EXISTS. v1 assumed a single serialization layout: a `/model_weights`
group plus a `model_config` root attribute. AutoND's shipped polished
checkpoints use neither - their layer groups sit at the HDF5 root. Given
those files v1 returned `layers: 0`, `parameters: 0` and a structural
digest equal to SHA-256 of the empty string, identically for structurally
DIFFERENT models, while reporting success and exiting zero. That is a
silent wrong answer with a digest collision. v1 and its outputs are
retained as evidence of the instrumentation failure; nothing it produced
may be used to adjudicate a hypothesis.

WHAT v2 CHANGES.
  * Two layouts are recognised explicitly and recorded by name:
      full_model   - /model_weights group, `model_config` root attribute,
                     optionally /optimizer_weights  (Keras ModelCheckpoint
                     with save_weights_only=False)
      weights_only - layer groups at the HDF5 root, `layer_names` root
                     attribute, no model_config  (model.save_weights)
  * An unrecognised layout, a readable file with no weight tensors, or a
    zero parameter count is a REFUSAL, never a result. An audit instrument
    that can return an empty census without erroring is defective by
    construction.
  * Every reported quantity is tagged with how it was established:
      observed      - read out of this file
      derived       - computed from observed quantities
      not_in_file   - NOT serialised here; it can only come from the
                      pinned source, and this collector will not assert it

WHAT A WEIGHTS FILE CANNOT TELL YOU. Dilation rates, kernel padding mode,
activation functions, the residual wiring and the input normalisation are
NOT recoverable from a weights-only HDF5. They are listed under
`not_in_file` so that a reader cannot mistake architecture conformance on
tensor shapes for conformance on the full computation graph. A full_model
file does carry `model_config`, so for that layout they move to observed -
and the layout field records which case applies.

Read-only. Opens every file with mode='r' and writes only its own output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

CENSUS_VERSION = "autond-architecture-census-v2"

#: Weight-name suffixes that Keras treats as non-trainable state.
NON_TRAINABLE_SUFFIXES = ("moving_mean", "moving_variance")

#: Graph properties a weights-only HDF5 does not serialise.
NOT_IN_WEIGHTS_FILE = (
    "dilation_rate", "padding", "kernel_size_semantics", "activation",
    "residual_wiring", "input_normalisation", "layer_connectivity",
)


class CensusRefusal(RuntimeError):
    """Raised instead of returning an empty or unverifiable census."""


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
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


def detect_layout(f: h5py.File) -> str:
    """Name the serialization layout, or refuse."""
    has_mw = "model_weights" in f
    has_cfg = "model_config" in f.attrs
    has_root_layers = "layer_names" in f.attrs
    if has_mw and has_cfg:
        return "full_model"
    if has_root_layers and not has_mw:
        return "weights_only"
    if has_mw:
        return "full_model_no_config"
    raise CensusRefusal(
        "unrecognised HDF5 layout: no /model_weights group and no root "
        "'layer_names' attribute. This collector handles the Keras "
        "full-model and weights-only layouts only; it will not guess.")


def _walk_layers(group: h5py.Group, order) -> list:
    """
    Walk the declared layer inventory.

    A layer named in `layer_names` but absent from the file is a REFUSAL,
    not a row with a flag. The inventory is the file's own statement of
    what it contains; a gap between that statement and the actual groups
    means the census is incomplete, and an incomplete census must not be
    admissible. Recording it as `missing_from_file: True` and continuing
    would reproduce the v1 defect in a quieter form - a census that looks
    successful while silently under-counting layers, tensors, parameters
    and the structural digest.
    """
    rows = []
    missing = [n for n in order if n not in group]
    if missing:
        raise CensusRefusal(
            f"layer inventory incomplete: {len(missing)} layer(s) named in "
            f"'layer_names' are absent from the file: {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}. "
            "A census over a partial layer set would under-count parameters "
            "and produce a structural digest for a model the file does not "
            "contain. Refusing.")
    for name in order:
        g = group[name]
        declared = (_decode(g.attrs["weight_names"])
                    if "weight_names" in g.attrs else None)
        tensors, total, trainable = [], 0, 0
        def visit(path, obj):
            nonlocal total, trainable
            if isinstance(obj, h5py.Dataset):
                n = int(np.prod(obj.shape)) if obj.shape else 1
                nt = any(s in path for s in NON_TRAINABLE_SUFFIXES)
                total += n
                if not nt:
                    trainable += n
                tensors.append({"name": path, "shape": list(obj.shape),
                                "dtype": str(obj.dtype), "parameters": n,
                                "trainable": not nt})
        g.visititems(visit)
        tensors.sort(key=lambda t: t["name"])
        rows.append({"layer": name, "declared_weight_names": declared,
                     "tensors": tensors, "parameters": total,
                     "trainable_parameters": trainable})
    return rows


def structural_digest(layers: list, layout: str) -> str:
    """
    Hash of the architecture as the FILE records it: invariant to weight
    values, sensitive to layer set, tensor names, shapes and dtypes.

    The layout is NOT hashed: the same architecture saved both ways must
    produce the same digest, which is what makes the format-control test
    meaningful.
    """
    parts = []
    for layer in layers:
        for t in layer["tensors"]:
            parts.append(f"{layer['layer']}|{t['name'].split('/')[-1]}|"
                         f"{t['shape']}|{t['dtype']}")
    if not parts:
        raise CensusRefusal("no weight tensors found; refusing to emit a digest "
                            "over an empty census (this was the v1 defect).")
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def census(path: Path) -> dict:
    if not path.is_file():
        raise CensusRefusal(f"file not found: {path}")
    if not h5py.is_hdf5(path):
        raise CensusRefusal(
            f"{path.name}: not an HDF5 file. A .keras file is a zip archive; "
            "this collector reads HDF5 only.")

    rec = {"path": path.name, "sha256": sha256_file(path),
           "bytes": path.stat().st_size}

    with h5py.File(path, "r") as f:
        layout = detect_layout(f)
        rec["layout"] = layout
        rec["observed"] = {
            "keras_version": _decode(f.attrs.get("keras_version")),
            "backend": _decode(f.attrs.get("backend")),
            "root_groups": sorted(f.keys()),
            "has_model_config": "model_config" in f.attrs,
            "has_training_config": "training_config" in f.attrs,
            "has_optimizer_state": "optimizer_weights" in f,
        }

        if layout.startswith("full_model"):
            wgroup = f["model_weights"]
        else:
            wgroup = f
        order = (_decode(wgroup.attrs["layer_names"])
                 if "layer_names" in wgroup.attrs
                 else (_decode(f.attrs["layer_names"])
                       if "layer_names" in f.attrs else sorted(wgroup.keys())))
        layers = _walk_layers(wgroup, order)

        total = sum(l["parameters"] for l in layers)
        trainable = sum(l["trainable_parameters"] for l in layers)
        if total == 0:
            raise CensusRefusal(
                f"{path.name}: layout {layout} recognised but zero weight "
                "parameters found. Refusing: an empty census is not a result.")

        rec["layers"] = layers
        rec["derived"] = {
            "layer_count": len(layers),
            "tensor_count": sum(len(l["tensors"]) for l in layers),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": total - trainable,
            "structural_digest": structural_digest(layers, layout),
        }
        if "optimizer_weights" in f:
            n = 0
            def count(_p, o):
                nonlocal n
                if isinstance(o, h5py.Dataset):
                    n += 1
            f["optimizer_weights"].visititems(count)
            rec["derived"]["optimizer_slot_tensors"] = n

        if layout == "weights_only":
            rec["not_in_file"] = {
                "fields": list(NOT_IN_WEIGHTS_FILE),
                "note": ("A weights-only HDF5 serialises tensor names, shapes "
                         "and values only. These graph properties cannot be "
                         "established from this file and must come from the "
                         "pinned source; this collector does not assert them."),
            }
        else:
            rec["not_in_file"] = {
                "fields": [],
                "note": ("model_config is present, so the computation graph is "
                         "serialised in this file. This collector still reports "
                         "only tensor-level facts; parsing model_config is a "
                         "separate step."),
            }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repaired read-only architecture census (v2).")
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-digest-match", action="store_true",
                    help=("require that every input share one structural "
                          "digest; use for the format-control test, where the "
                          "same architecture is saved in both layouts"))
    args = ap.parse_args()

    records, refusals = [], []
    for p in args.models:
        try:
            records.append(census(p))
        except CensusRefusal as exc:
            refusals.append({"path": str(p), "refusal": str(exc)})
        except Exception as exc:                                  # noqa: BLE001
            refusals.append({"path": str(p),
                             "refusal": f"{type(exc).__name__}: {exc}"})

    out = {
        "artifact": CENSUS_VERSION,
        "supersedes": ("architecture_census.py (v1), which returned an empty "
                       "census with a constant digest on the weights-only "
                       "layout; v1 and its outputs are retained as evidence of "
                       "the instrumentation failure and are not usable as "
                       "hypothesis evidence"),
        "method": "h5py read-only; TensorFlow deliberately not imported",
        "models": records,
        "refusals": refusals,
    }

    digests = {r["derived"]["structural_digest"] for r in records}
    out["structural_digests_distinct"] = len(digests)
    if args.expect_digest_match:
        out["digest_match"] = (len(digests) == 1 and not refusals)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))

    print(f"Wrote {args.output}\n")
    for r in records:
        d = r["derived"]
        print(f"  {r['path']}")
        print(f"    layout            {r['layout']}")
        print(f"    sha256            {r['sha256']}")
        print(f"    keras / backend   {r['observed']['keras_version']} / "
              f"{r['observed']['backend']}")
        print(f"    layers / tensors  {d['layer_count']} / {d['tensor_count']}")
        print(f"    parameters        {d['total_parameters']:,} "
              f"({d['trainable_parameters']:,} trainable + "
              f"{d['non_trainable_parameters']:,} non-trainable)")
        if "optimizer_slot_tensors" in d:
            print(f"    optimizer slots   {d['optimizer_slot_tensors']}")
        print(f"    structural digest {d['structural_digest']}")
        if r["not_in_file"]["fields"]:
            print(f"    NOT in this file  {', '.join(r['not_in_file']['fields'])}")
        print()
    for x in refusals:
        print(f"  REFUSED {x['path']}\n          {x['refusal']}")
    if args.expect_digest_match:
        print(f"  format control: {'MATCH' if out.get('digest_match') else 'MISMATCH'} "
              f"({out['structural_digests_distinct']} distinct digest(s))")
    return 1 if refusals else 0


if __name__ == "__main__":
    sys.exit(main())
