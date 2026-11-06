"""
Post-run diagnostic of the B1 observation H1 = 0.4159 versus K3 = 0.9290. Not part of the registered
benchmark; it reads the same checkpoints and writes h1_diagnostics.json. No cause is assumed: each
check records raw values for H1 and, as a control, K3.

  checks: file identity and architecture; Keras-loaded weights against the raw HDF5 datasets; the
  training history saved beside the H1 file; accuracy by round count 3-7; per-class accuracy,
  inverted-label accuracy, AUC and the prediction distribution; and accuracy under every ordering
  of the four 16-bit input words, with and without bit reversal inside each word.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[4]
MODELS = {"H1": REPO / "audit/cryptography/evidence/ce1/best5depth10 (10).h5",
          "K3": REPO / "audit/cryptography/Archive/best5depth10.h5"}
HISTORY = REPO / "audit/cryptography/evidence/ce1/h5r_depth10 (10).npy"
N = 100_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats(y, p):
    c = (p > 0.5) == y
    return {"accuracy": float(c.mean()), "accuracy_label_1": float(c[y == 1].mean()),
            "accuracy_label_0": float(c[y == 0].mean()), "inverted_label_accuracy": float(((p <= 0.5) == y).mean()),
            "auc": auc(y, p), "mean_prediction": float(p.mean()), "fraction_above_half": float((p > 0.5).mean()),
            "label_1_fraction": float(y.mean())}


def main() -> int:
    import keras
    spec = importlib.util.spec_from_file_location("released_speck", REPO / "speck.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    out = {"n_per_check": N, "threshold": "> 0.5, as in B1", "generator": "speck.py make_train_data (os.urandom)"}
    models = {}
    for k, path in MODELS.items():
        with h5py.File(path, "r") as f:
            cfg = json.loads(f.attrs["model_config"])
            raw = {}
            f["model_weights"].visititems(lambda n, o: raw.__setitem__(n, np.array(o)) if isinstance(o, h5py.Dataset) else None)
            meta = {a: (f.attrs[a].decode() if isinstance(f.attrs[a], bytes) else str(f.attrs[a]))
                    for a in f.attrs if a != "model_config"}
        m = keras.models.load_model(path, compile=False)
        loaded = [w for w in m.weights]
        unmatched = []
        for w in loaded:
            cands = [v for n, v in raw.items() if v.shape == tuple(w.shape) and np.array_equal(v, np.asarray(w))]
            if not cands:
                unmatched.append(w.path)
        layers = [l["class_name"] for l in cfg["config"]["layers"]]
        acts = sorted({l["config"].get("activation") for l in cfg["config"]["layers"] if "activation" in l["config"]})
        out[k] = {"path": str(path.relative_to(REPO)), "sha256": sha(path), "h5_attributes": meta,
                  "conv1d": layers.count("Conv1D"), "add": layers.count("Add"), "activations": acts,
                  "input_shape": list(m.input_shape), "output_shape": list(m.output_shape),
                  "loaded_weight_tensors": len(loaded), "loaded_tensors_without_identical_h5_dataset": unmatched,
                  "rounds": {}, "word_orders_round5": {}}
        models[k] = m
    if HISTORY.is_file():
        try:
            h = np.load(HISTORY, allow_pickle=False)
            out["H1_companion_history"] = {"path": str(HISTORY.relative_to(REPO)), "sha256": sha(HISTORY),
                                           "shape": list(h.shape), "max": float(h.max()), "last": float(h.ravel()[-1]),
                                           "first": float(h.ravel()[0])}
        except ValueError as exc:
            out["H1_companion_history"] = {"path": str(HISTORY.relative_to(REPO)), "not_loadable_without_pickle": str(exc)}
    for nr in (3, 4, 5, 6, 7):
        X, Y = sp.make_train_data(N, nr)
        for k, m in models.items():
            out[k]["rounds"][str(nr)] = stats(Y, m.predict(X, batch_size=5000, verbose=0).ravel())
    X, Y = sp.make_train_data(N, 5)
    words = X.reshape(N, 4, 16)
    for perm in itertools.permutations(range(4)):
        for rev in (False, True):
            w = words[:, list(perm), :]
            if rev:
                w = w[:, :, ::-1]
            Xp = np.ascontiguousarray(w.reshape(N, 64))
            for k, m in models.items():
                s = stats(Y, m.predict(Xp, batch_size=5000, verbose=0).ravel())
                out[k]["word_orders_round5"][f"{''.join(map(str, perm))}{'r' if rev else ''}"] = {
                    "accuracy": s["accuracy"], "auc": s["auc"]}
    for k in models:
        wo = out[k]["word_orders_round5"]
        best = max(wo, key=lambda x: wo[x]["accuracy"])
        out[k]["best_word_order_round5"] = {"order": best, **wo[best]}
    path = Path(__file__).with_name("h1_diagnostics.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: {"rounds_accuracy": {r: round(v["accuracy"], 4) for r, v in out[k]["rounds"].items()},
                          "r5": {x: round(out[k]["rounds"]["5"][x], 4) for x in ("accuracy_label_1", "accuracy_label_0",
                                                                                "inverted_label_accuracy", "auc",
                                                                                "mean_prediction", "fraction_above_half")},
                          "best_word_order": out[k]["best_word_order_round5"],
                          "identity_order": out[k]["word_orders_round5"]["0123"],
                          "unmatched_weights": len(out[k]["loaded_tensors_without_identical_h5_dataset"]),
                          "activations": out[k]["activations"], "attrs": out[k]["h5_attributes"]} for k in models},
                     indent=1))
    print("history:", out.get("H1_companion_history"))
    print(path, sha(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
