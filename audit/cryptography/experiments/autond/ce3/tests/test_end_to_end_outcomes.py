import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from audit.cryptography.experiments.autond.ce3 import adapter as A
from audit.cryptography.experiments.autond.ce3 import design as D
from audit.cryptography.experiments.autond.ce3 import probes as PR
from audit.cryptography.experiments.autond.ce3 import runner as R
from audit.cryptography.experiments.autond.ce3 import verify as V
from audit.cryptography.targets.autond import model_binding as MB
from audit.cryptography.targets.autond import upstream as UP

SIZES = {"n_replicates": 5, "main_pairs_per_replicate": 3000, "calibration_per_class": 1500,
         "min_class_count_per_fold": 20, "permutations_per_replicate": 5}

SCENARIOS = {
    "SUPPORTED": {"trained_main": "signal", "twin_main": "noise", "trained_cal": "signal", "twin_cal": "noise"},
    "POSITIVE_CONTROL_FAILED": {"trained_main": "signal", "twin_main": "noise", "trained_cal": "signal",
                                "twin_cal": "same_as_trained"},
    "INCONCLUSIVE_S1": {"trained_main": "permuted_signal", "twin_main": "noise", "trained_cal": "signal",
                        "twin_cal": "noise"},
    "TARGET_DECODABLE_TWIN_UNRESOLVED": {"trained_main": "signal", "twin_main": "same_as_trained",
                                         "trained_cal": "signal", "twin_cal": "noise"},
    "INCONCLUSIVE_FAILED_REPLICATES": {"trained_main": "signal", "twin_main": "constant", "trained_cal": "signal",
                                       "twin_cal": "noise"},
}


class Plant:
    def __init__(self, scenario: dict):
        self.scenario, self.main, self.cal, self.roles, self.sets = scenario, {}, {}, {}, {}

    def features(self, role: str, X: np.ndarray) -> np.ndarray:
        keys = [r.tobytes() for r in np.asarray(X, dtype=np.uint8)]
        is_main = keys[0] in self.main
        labels = np.array([(self.main if is_main else self.cal)[k] for k in keys], dtype=np.int64)
        part = f"{'trained' if role == 'published' else 'twin'}_{'main' if is_main else 'cal'}"
        kind = self.scenario[part]
        if kind == "same_as_trained":
            return self.features("published", X)
        rng = np.random.default_rng(len(keys) * 7 + (1 if role == "twin" else 0) + (2 if is_main else 0))
        h = 0.3 * rng.standard_normal((len(keys), 64))
        if kind == "signal":
            h[np.arange(len(keys)), labels if is_main else 10 + labels] += 2.0
        elif kind == "permuted_signal":
            index, y = self.sets[keys[0]]
            perms = PR.permutations(y.size, index, SIZES["permutations_per_replicate"])
            for j in range(3):
                h[np.arange(len(keys)), 5 * j + y[perms[j]]] += 2.0
        elif kind == "constant":
            h[:] = 0.0
        return np.maximum(h, 0.0).astype(np.float32)

    def install(self, mp):
        orig_main, orig_cal, orig_load = A.main_dataset, A.calibration_dataset, A.load_models
        plant = self

        def main_dataset(up, n, *a, **k):
            g = orig_main(up, n, *a, **k)
            X = A.model_input(up, g["arrays"]["c0"], g["arrays"]["c1"])
            y = A.classes_for(g["arrays"]["target"])
            plant.main.update({X[i].tobytes(): int(y[i]) for i in range(X.shape[0])})
            plant.sets[X[0].tobytes()] = (len(plant.sets), y.astype(np.int64))
            return g

        def calibration_dataset(up, n, idx, *a, **k):
            c = orig_cal(up, n, idx, *a, **k)
            Xc = A.calibration_input(up, c["arrays"])
            plant.cal.update({Xc[i].tobytes(): int(c["arrays"]["labels"][i]) for i in range(Xc.shape[0])})
            return c

        def load_models(up, path):
            model, twin, binding = orig_load(up, path)
            plant.roles[id(model)], plant.roles[id(twin)] = "published", "twin"
            plant.model = model
            return model, twin, binding

        def predict(model, X):
            W, b = A.head(model)
            z = plant.features("published", X).astype(np.float64) @ W[:, 0] + b[0]
            return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

        mp.setattr(A, "main_dataset", main_dataset)
        mp.setattr(A, "calibration_dataset", calibration_dataset)
        mp.setattr(A, "load_models", load_models)
        mp.setattr(A, "extractor", lambda m: ("PLANTED", plant.roles[id(m)]))
        mp.setattr(A, "extract", lambda ext, X: plant.features(ext[1], X))
        mp.setattr(MB, "predict", predict)


@pytest.fixture(scope="module")
def real_preflight():
    return R.preflight(UP.load(), "smoke")


def _run(tmp_path, monkeypatch, real_preflight, scenario, *, precheck_ok=True):
    monkeypatch.setattr(D, "SMOKE", dict(SIZES))
    monkeypatch.setattr(R, "preflight", lambda up, mode: {**real_preflight, "ok": precheck_ok})
    Plant(scenario).install(monkeypatch)
    out = tmp_path / "autond_ce3_smoke_outcome"
    man = R.initialise(out, "smoke")
    if man["status"] == "INITIALISED":
        R.run_all(out)
    cert = R.finalize(out)
    rep = V.verify(out, recompute_models=False)
    return out, cert, rep


@pytest.mark.parametrize("name,expected", [
    ("SUPPORTED", "SUPPORTED"),
    ("POSITIVE_CONTROL_FAILED", "POSITIVE_CONTROL_FAILED"),
    ("INCONCLUSIVE_S1", "INCONCLUSIVE"),
    ("TARGET_DECODABLE_TWIN_UNRESOLVED", "TARGET_DECODABLE_TWIN_UNRESOLVED"),
    ("INCONCLUSIVE_FAILED_REPLICATES", "INCONCLUSIVE"),
])
def test_end_to_end_decision_outcomes_verify(tmp_path, monkeypatch, real_preflight, name, expected):
    out, cert, rep = _run(tmp_path, monkeypatch, real_preflight, SCENARIOS[name])
    assert cert["decision"] == expected, cert["decision_reason"]
    assert rep["passed"], rep["failures"]
    assert rep["decision_recomputed"] == expected
    an = cert["analysis"]
    if name == "INCONCLUSIVE_FAILED_REPLICATES":
        assert set(cert["run_states"].values()) == {"FAILED"} and an["n_valid"] == 0
        for rid in cert["run_states"]:
            rec = json.loads((out / "runs" / rid / "record.json").read_text())
            assert "degenerate" in rec["failure"]
        return
    assert set(cert["run_states"].values()) == {"VALID"}
    steps = {s["step"]: s["rejected"] for s in an["sequence"]}
    if name == "SUPPORTED":
        assert steps == {"C": True, "S1": True, "S2": True}
        assert all(an["endpoints"][k]["test"]["n_positive"] == 5 for k in ("C_task", "C_twin", "S1", "S2"))
        assert all(v["S1"] == 2.5 for v in cert["replicate_endpoints"].values())
    elif name == "POSITIVE_CONTROL_FAILED":
        assert steps == {"C": False}
        assert an["sequence"][0]["rejected_components"] == {"C_task": True, "C_twin": False}
        assert an["endpoints"]["C_twin"]["test"]["n_zero"] == 5
        assert all(v["C_twin"] == 0.0 for v in cert["replicate_endpoints"].values())
    elif name == "INCONCLUSIVE_S1":
        assert steps == {"C": True, "S1": False}
        for rid in cert["run_states"]:
            m = json.loads((out / "runs" / rid / "measurement.json").read_text())
            ba = m["balanced_accuracy"]
            assert min(ba["P00"], ba["P01"], ba["P02"]) > 0.9 > 0.4 > max(ba["A"], ba["P03"], ba["P04"])
            assert m["s1_rank"]["rank"] <= 3 and m["endpoints"]["S1"] < 0
    elif name == "TARGET_DECODABLE_TWIN_UNRESOLVED":
        assert steps == {"C": True, "S1": True, "S2": False}
        assert an["endpoints"]["S2"]["test"]["n_zero"] == 5


def test_end_to_end_precheck_failed(tmp_path, monkeypatch, real_preflight):
    out, cert, rep = _run(tmp_path, monkeypatch, real_preflight, SCENARIOS["SUPPORTED"], precheck_ok=False)
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["status"] == "PRECHECK_FAILED" and cert["decision"] == "PRECHECK_FAILED"
    assert rep["decision_recomputed"] == "PRECHECK_FAILED"
    assert not (out / "runs").exists() or not any((out / "runs").rglob("*.npz"))
    assert set(rep["failures"]) <= {"run_manifest_hash_matches_initialised_event", "twin_copy_equals_ce2_twin"}
