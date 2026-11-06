"""
SDB-V2 baselines. None is mapped to claim support, and none is forced to a verdict.

B1   accuracy-only reproduction by FRESH-DATA INFERENCE with an existing model: H1, S2, K3 only.
     Data from the released generator (root speck.py make_train_data, which draws from os.urandom, so
     the draw is recorded rather than reseedable); predictions thresholded `> 0.5` as in the frozen
     ce234 binding. Output is descriptive: accuracy, delta from the reported 0.9291, its standard error
     and a 95% Wilson interval. No REPRODUCED/NOT_REPRODUCED threshold is applied.
B1-S stored-prediction rescoring, H2 only: the archived A-D4 prediction arrays against test_y.npy,
     `>= 0.5` as in d4_controlled_perturbation.py. Not model re-execution; reported separately.
B2   the executed checklist C1-C8, item applicability fixed in each bundle (E / NS / NA).
B1-naive is an illustrative policy described in the report; it is not executed or scored.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from audit.framework_validation.sdb_v2.cases_v2 import FIX, H2_BUNDLE, REPO, bundle

B1_CASES = ("H1", "S2", "K3")
B1_N = 200_000
Z95 = 1.959963984540054
SPECK_VECTOR = {"key": (0x1918, 0x1110, 0x0908, 0x0100), "pt": (0x6574, 0x694C), "ct": (0xA868, 0x42F2),
                "rounds": 22}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wilson(k: int, n: int, z: float = Z95):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def b1(case: str) -> tuple[dict, dict]:
    import keras
    b = bundle(case)
    sp = _module(REPO / b["cipher_module"], f"sdb_v2_released_speck_{case}")
    model = keras.models.load_model(REPO / b["model"], compile=False)
    X, Y = sp.make_train_data(B1_N, 5)
    pred = model.predict(X, batch_size=5000, verbose=0).ravel()
    correct = (pred > 0.5) == Y
    k, n = int(correct.sum()), int(len(Y))
    acc = k / n
    lo, hi = wilson(k, n)
    out = {"status": "EXECUTED", "mode": "fresh-data inference with an existing model", "n": n, "correct": k,
           "accuracy": acc, "reported_accuracy": b["reported_accuracy"], "delta": acc - b["reported_accuracy"],
           "se": math.sqrt(acc * (1 - acc) / n), "wilson95": [lo, hi],
           "delta_wilson95": [lo - b["reported_accuracy"], hi - b["reported_accuracy"]],
           "reported_accuracy_uncertainty": "none published", "decision_threshold": None}
    raw = {"correct_packbits": np.packbits(correct.astype(np.uint8)), "X_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
           "Y_sha256": hashlib.sha256(Y.astype(np.uint8).tobytes()).hexdigest(), "generator": b["generator"],
           "model_sha256": hashlib.sha256((REPO / b["model"]).read_bytes()).hexdigest()}
    return out, raw


def b1_status(case: str) -> str:
    if case in B1_CASES:
        return "EXECUTED"
    if case == "H2":
        return "B1-S"
    return "NOT_ASSESSED" if bundle(case).get("reported_accuracy") is not None else "NOT_APPLICABLE"


def b1s() -> tuple[dict, dict]:
    y = np.load(H2_BUNDLE / "dataset" / "test_y.npy", allow_pickle=False)
    out, raw = {}, {}
    for p in sorted((H2_BUNDLE / "fits" / "predictions").glob("*.npy")):
        pred = np.load(p, allow_pickle=False)
        c = (pred >= 0.5) == y
        out[p.name] = {"correct": int(c.sum()), "test_examples": int(len(y)),
                       "label_0_correct": int(c[y == 0].sum()), "label_1_correct": int(c[y == 1].sum())}
        raw[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out, raw


# --- B2 ------------------------------------------------------------------------------------------------

def _h5_architecture(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        cfg = f.attrs.get("model_config")
        cfg = json.loads(cfg.decode() if isinstance(cfg, bytes) else cfg)
        names = []

        def walk(o):
            if isinstance(o, dict):
                if "class_name" in o:
                    names.append(o["class_name"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(cfg.get("config", {}).get("layers", []))
        n = [0]
        f["model_weights"].visititems(lambda _, o: n.__setitem__(0, n[0] + o.size) if isinstance(o, h5py.Dataset) else None)
    return {"conv1d": names.count("Conv1D"), "add": names.count("Add"), "params": n[0]}


def _load(path: Path) -> None:
    s = path.suffix
    if s == ".h5":
        import h5py
        import keras
        with h5py.File(path, "r"):
            pass
        keras.models.load_model(path, compile=False)
    elif s in (".npz", ".npy"):
        d = np.load(path, allow_pickle=False)
        if s == ".npz":
            for k in d.files:
                d[k]
    elif s == ".py":
        _module(path, f"sdb_v2_b2_{abs(hash(str(path)))}")
    elif s == ".json":
        json.loads(path.read_text())
    else:
        path.read_bytes()


def b2(case: str) -> dict:
    b = bundle(case)
    app = b["b2"]
    items = {}
    for item, spec in app.items():
        if spec["status"] != "E":
            items[item] = {"result": "NOT_ASSESSED" if spec["status"] == "NS" else "NOT_APPLICABLE",
                           "reason": spec["reason"]}
            continue
        try:
            items[item] = _item(item, b)
        except Exception as exc:                                        # noqa: BLE001
            items[item] = {"result": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}
    res = [v["result"] for v in items.values()]
    output = ("FLAGGED" if "FAIL" in res else
              "NO_ISSUE_FOUND" if "PASS" in res and all(r in ("PASS", "NOT_APPLICABLE") for r in res) else
              "NOT_ASSESSED")
    return {"items": items, "output": output}


def _item(item: str, b: dict) -> dict:
    files = [REPO / f for f in b["files"]]
    if item == "C1":
        missing = [str(f) for f in files if not f.is_file()] + ([] if (FIX / b["case"] / "bundle.json").is_file()
                                                                else ["bundle.json"])
        return {"result": "FAIL" if missing else "PASS", "missing": missing}
    if item == "C2":
        for f in files:
            _load(f)
        return {"result": "PASS", "loaded": len(files)}
    if item == "C3":
        env = b.get("environment") or {}
        need = [k for k in ("python", "numpy") if not env.get(k)]
        return {"result": "FAIL" if need else "PASS", "missing": need}
    if item == "C5":
        m = _module(REPO / b["cipher_module"], f"sdb_v2_b2_cipher_{b['case']}")
        ks = m.expand_key(SPECK_VECTOR["key"], SPECK_VECTOR["rounds"])
        ct = tuple(int(x) for x in m.encrypt(SPECK_VECTOR["pt"], ks))
        return {"result": "PASS" if ct == SPECK_VECTOR["ct"] else "FAIL", "ciphertext": [hex(x) for x in ct]}
    if item == "C6":
        p = np.load(REPO / b["partitions"], allow_pickle=False)
        parts = {k: {r.tobytes() for r in p[k]} for k in p.files if not k.endswith("_y")}
        names = sorted(parts)
        overlaps = {f"{x}/{y}": len(parts[x] & parts[y]) for i, x in enumerate(names) for y in names[i + 1:]}
        return {"result": "FAIL" if any(overlaps.values()) else "PASS", "overlaps": overlaps}
    if item == "C7":
        got = {k: hashlib.sha256((REPO / k).read_bytes()).hexdigest() for k in b["published_checksums"]}
        bad = [k for k, h in b["published_checksums"].items() if got[k] != h]
        if b["case"] == "H2":
            cert = json.loads((REPO / b["certificate"]).read_text())
            for row in cert["findings"]["replicates"]:
                for cond in ("baseline", "perturbed"):
                    m = row[f"{cond}_predictions"]
                    f = H2_BUNDLE / "fits" / "predictions" / m["file"]
                    if hashlib.sha256(f.read_bytes()).hexdigest() != m["sha256"]:
                        bad.append(m["file"])
        return {"result": "FAIL" if bad else "PASS", "mismatched": bad}
    if item == "C8":
        arch = _h5_architecture(REPO / b["model"])
        pub = b["published_architecture"]
        bad = {k: (arch[k], pub[k]) for k in ("conv1d", "add", "params") if arch[k] != pub[k]}
        return {"result": "FAIL" if bad else "PASS", "observed": arch, "differs": bad}
    raise KeyError(item)
