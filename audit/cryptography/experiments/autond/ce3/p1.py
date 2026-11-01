from __future__ import annotations

import io
import json
import platform
import sys
from pathlib import Path

import numpy as np

from ....core import evidence as EV
from ....targets.autond import upstream as UP
from . import adapter as A
from . import design as D
from .target_bins import choose_target_binning

SCHEMA = "autond-ce3-p1-target-pilot-1"
DATA_NAME = "p1_pairs.npz"
RECORD_NAME = "p1_record.json"


class P1Error(RuntimeError):
    pass


def _npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def generate(out_dir: Path, *, n: int = D.P1_PAIRS, upstream=None, randbytes=None) -> dict:
    out_dir = Path(out_dir)
    if (out_dir / RECORD_NAME).exists() or (out_dir / DATA_NAME).exists():
        raise P1Error(f"{out_dir} already holds a P1 record; P1 is generated once")
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise P1Error("P1 is model-free: TensorFlow or Keras is already imported in this process")
    up = UP.load(upstream)
    gen = A.main_dataset(up, n, **({"randbytes": randbytes} if randbytes else {}))
    a = gen["arrays"]
    if not (gen["gates"]["E"]["passed"] and gen["gates"]["F"]["passed"]):
        raise P1Error(f"P1 generation gates failed: {gen['gates']}")
    stored = {k: a[k] for k in ("key_words", "pt0", "pt1", "target")}
    selection = choose_target_binning(a["target"], D.MIN_CLASS_FRACTION)
    data = _npz_bytes(stored)
    sha = EV.atomic_write_bytes(out_dir / DATA_NAME, data)
    support, counts = np.unique(a["target"], return_counts=True)
    record = {
        "schema": SCHEMA, "experiment_id": D.EXPERIMENT_ID, "evidentiary": False, "model_loaded": False,
        "role": ("model-free target discretisation pilot; never part of a CE3 replicate; fixes the class count and "
                 "whole-value edges before any probe is fitted"),
        "generated_utc": EV.utc(), "n": int(n), "data_file": DATA_NAME, "data_sha256": sha,
        "data_content_sha256": A.content_sha256(stored),
        "stored_arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in stored.items()},
        "generation_gates": gen["gates"],
        "generator": {"function": "experiments/autond/ce2/adapter.py generate_genuine_pairs",
                      "dependencies": {k: D.DEPENDENCIES[k] for k in ("experiments/autond/ce2/adapter.py",
                                                                     "experiments/autond/ce2/theory.py",
                                                                     "targets/autond/upstream.py",
                                                                     "upstream_sha256.json")},
                      "upstream_verification_all_verified": up.verification["all_verified"],
                      "randomness": "os.urandom"},
        "support": {"values": [int(v) for v in support], "counts": [int(c) for c in counts]},
        "min_class_fraction": D.MIN_CLASS_FRACTION,
        "rule": D.DESIGN["target"]["pilot"]["rule"],
        "selection": selection,
        "selected_n_classes": selection["selected"]["n_classes"],
        "selected_edges": selection["selected"]["edges"],
        "class_rule": D.DESIGN["target"]["class_rule"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    EV.write_json(out_dir / RECORD_NAME, record)
    return record


def load_record(path: Path) -> dict:
    return json.loads(Path(path).read_text(), object_pairs_hook=lambda p: D._reject_duplicate_keys(p))
