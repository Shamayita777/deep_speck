import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from audit.cryptography.core import evidence as EV
from audit.cryptography.experiments.autond.ce3 import adapter as A
from audit.cryptography.experiments.autond.ce3 import design as D
from audit.cryptography.experiments.autond.ce3 import runner as R
from audit.cryptography.experiments.autond.ce3 import verify as V
from audit.cryptography.targets.autond import model_binding as MB
from audit.cryptography.targets.autond import upstream as UP

TWIN = R.CRYPTO_ROOT / D.CE2_RUN_RELPATH / D.CE2_TWIN_RELPATH


@pytest.fixture(scope="session")
def up():
    return UP.load()


@pytest.fixture(scope="session")
def models(up):
    return A.load_models(up, TWIN)


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    out = tmp_path_factory.mktemp("ce3") / "autond_ce3_smoke_base"
    man = R.initialise(out, "smoke")
    assert man["status"] == "INITIALISED"
    R.run_all(out)
    R.finalize(out)
    return out


@pytest.fixture(scope="session")
def base_preflight(base):
    return json.loads((base / "preflight.json").read_text())


def test_representation_binding_by_graph_for_published_and_twin(models):
    model, twin, _ = models
    for m in (model, twin):
        b = A.bind_representation(m)
        assert b["ok"], b["problems"]
    assert model.layers[32].name.startswith("activation_") and type(model.layers[32]).__name__ == "Activation"
    assert MB.weights_sha256(twin) == D.CE2_TWIN_WEIGHTS_SHA256


def test_binding_rejects_another_layer(models, monkeypatch):
    rep = dict(D.REPRESENTATION, layer_index=29)
    monkeypatch.setattr(D, "REPRESENTATION", rep)
    assert not A.bind_representation(models[0])["ok"]


def test_head_reproduces_model_output(up, models):
    model = models[0]
    g = A.main_dataset(up, 1024)["arrays"]
    X = A.model_input(up, g["c0"], g["c1"])
    h = A.extract(A.extractor(model), X)
    assert h.shape == (1024, 64) and h.dtype == np.float32 and np.all(h >= 0)
    assert A.head_difference(model, h, MB.predict(model, X)) <= D.HEAD_TOLERANCE


def test_calibration_set_is_exactly_balanced_and_reconstructible(up):
    c = A.calibration_dataset(up, 500, 3)
    a = c["arrays"]
    assert int(a["labels"].sum()) == 500 and a["labels"].size == 1000
    X = A.calibration_input(up, a)
    assert X.shape == (1000, 64) and set(np.unique(X).tolist()) <= {0, 1}
    assert np.array_equal(X[a["labels"] == 1], A.model_input(up, a["c0"], a["c1"])[a["order"][a["order"] < 500]])
    order = np.random.Generator(np.random.PCG64(D.CALIBRATION_SHUFFLE_SEED_BASE + 3)).permutation(1000)
    assert np.array_equal(order, a["order"])


def test_smoke_lifecycle_verifies_with_models_recomputed(base):
    rep = V.verify(base, recompute_models=True)
    assert rep["passed"], rep["failures"]
    assert rep["numerical_status"] == "EXACT"
    cert = json.loads((base / "certificate.json").read_text())
    assert cert["label"] == "SMOKE_NOT_CONFIRMATORY" and cert["evidentiary"] is False
    assert set(cert["run_states"].values()) == {"VALID"}
    for rid in cert["run_states"]:
        for name in R.ARTIFACTS:
            assert (base / "runs" / rid / name).is_file()


def test_twin_copy_and_manifest(base):
    man = json.loads((base / "run_manifest.json").read_text())
    assert EV.sha256_file(base / man["untrained_twin"]["file"]) == D.CE2_TWIN_FILE_SHA256 == EV.sha256_file(TWIN)
    assert man["producer_sources"] == R.source_hashes()


@pytest.mark.parametrize("point", EV.FAULT_POINTS)
def test_interruption_at_every_fault_point_resumes_and_verifies(tmp_path, base_preflight, monkeypatch, point):
    monkeypatch.setattr(R, "preflight", lambda up, mode: base_preflight)
    out = tmp_path / "autond_ce3_smoke_fault"
    R.initialise(out, "smoke")
    if point == "during_quarantine":
        EV.arm_fault("after_promote")
        with pytest.raises(EV.InjectedFault):
            R.run_all(out, only=0)
    EV.arm_fault(point)
    try:
        with pytest.raises(EV.InjectedFault):
            R.run_all(out, only=0)
    finally:
        EV.disarm_faults()
    recs = R.run_all(out)
    assert all(r["state"] == "VALID" for r in recs)
    R.finalize(out)
    rep = V.verify(out, recompute_models=False)
    assert rep["passed"], rep["failures"]


def _reseal(out: Path, rid: str, name: str, arrays: dict | None = None, obj=None):
    p = out / "runs" / rid / name
    if arrays is not None:
        p.write_bytes(R.npz_bytes(arrays))
    else:
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    key = name.split(".")[0]
    lines = (out / "journal.jsonl").read_bytes().splitlines()
    entries = [json.loads(x) for x in lines]
    prev = EV.GENESIS
    for e in entries:
        fl = e.get("fields", {})
        if e.get("run_id") == rid and f"{key}_sha256" in fl:
            fl[f"{key}_sha256"] = EV.sha256_file(p)
            if arrays is not None:
                fl[f"{key}_content_sha256"] = A.content_sha256(arrays)
        if e.get("run_id") == rid and e.get("to") == "VALID" and obj is not None:
            fl["endpoints"] = obj["endpoints"]
        e["prev_sha256"] = prev
        prev = hashlib.sha256(EV.canonical(e)).hexdigest()
    (out / "journal.jsonl").write_bytes(b"".join(EV.canonical(e) + b"\n" for e in entries))
    rec = EV.materialise(entries, rid, R.MACHINE, record_schema=R.RECORD_SCHEMA)
    (out / "runs" / rid / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    cert = json.loads((out / "certificate.json").read_text())
    cert["evidence_manifest"] = R.evidence_manifest(out)
    if obj is not None:
        cert["replicate_endpoints"][rid] = obj["endpoints"]
    (out / "certificate.json").write_text(json.dumps(cert, indent=1, sort_keys=True) + "\n")


def _npz(p):
    with np.load(p) as z:
        return {k: z[k].copy() for k in z.files}


def _copy(base, tmp_path):
    out = tmp_path / base.name
    shutil.copytree(base, out)
    for v in out.glob("verification_*.json"):
        v.unlink()
    return out


def _flip_pred(a):
    a["A__pred"][0] = (a["A__pred"][0] + 1) % 5


def _perturb_coef(a):
    a["T__coef"][1] += 0.05


def _perturb_one_row(a):
    a["T__coef"][1, 2] += 0.05


def _shift_mean(a):
    a["CAL_trained__mean"][2] += 0.01


def _swap_fold(a):
    i, j = np.flatnonzero(a["fold_main"] == 0)[0], np.flatnonzero(a["fold_main"] == 1)[0]
    a["fold_main"][i], a["fold_main"][j] = 1, 0


def _swap_perm(a):
    a["perm_index"][0, [0, 1]] = a["perm_index"][0, [1, 0]]


def _swap_tiebreak(a):
    a["s1_tiebreak"][[0, 1]] = a["s1_tiebreak"][[1, 0]]


FORGERIES = {
    "prediction": ("probes.npz", _flip_pred, "predictions_follow_from_stored_parameters"),
    "coefficients_softmax_invariant": ("probes.npz", _perturb_coef, "multinomial_coefficients_sum_to_zero_across_classes"),
    "coefficients_one_class": ("probes.npz", _perturb_one_row, "stored_probe_parameters_satisfy_objective_gradient"),
    "scaler": ("probes.npz", _shift_mean, "scalers_fitted_on_training_folds_only"),
    "folds": ("probes.npz", _swap_fold, "folds_reconstructed_label_independent"),
    "permutation": ("probes.npz", _swap_perm, "permutations_reconstructed"),
    "target": ("data.npz", lambda a: a["target"].__setitem__(0, a["target"][0] - 1), "trajectory_weights_target"),
    "s1_tiebreak": ("data.npz", _swap_tiebreak, "s1_rank_recomputed_exactly"),
    "classes": ("data.npz", lambda a: a["classes"].__setitem__(0, (a["classes"][0] + 1) % 5), "classes_from_frozen_edges"),
    "calibration_label": ("calibration.npz", lambda a: a["labels"].__setitem__(0, 1 - a["labels"][0]),
                          "calibration_order_labels_balance"),
    "ciphertext": ("data.npz", lambda a: a["c0"].__setitem__((0, 0), a["c0"][0, 0] ^ 1),
                   "ciphertexts_and_input_difference_own_cipher"),
}


@pytest.mark.parametrize("case", sorted(FORGERIES))
def test_hash_consistent_forgery_is_caught_by_recomputation(base, tmp_path, case):
    out = _copy(base, tmp_path)
    name, mutate, check = FORGERIES[case]
    a = _npz(out / "runs" / "rep_00" / name)
    mutate(a)
    _reseal(out, "rep_00", name, arrays=a)
    rep = V.verify(out, recompute_models=False)
    assert "journal_chain_sequence_and_identity" not in rep["failures"]
    assert "evidence_manifest_hashes" not in rep["failures"]
    assert f"rep_00:{check}" in rep["failures"], rep["failures"]


def test_forged_representation_is_caught_by_reextraction(base, tmp_path):
    out = _copy(base, tmp_path)
    a = _npz(out / "runs" / "rep_00" / "representations.npz")
    a["trained_main"] += 0.5
    _reseal(out, "rep_00", "representations.npz", arrays=a)
    rep = V.verify(out, recompute_models=True)
    assert "rep_00:representations_reextracted" in rep["failures"]
    assert "rep_00:head_consistency_main" in rep["failures"]


def test_forged_endpoint_is_caught(base, tmp_path):
    out = _copy(base, tmp_path)
    m = json.loads((out / "runs" / "rep_00" / "measurement.json").read_text())
    m["endpoints"]["S1"] = 0.25
    _reseal(out, "rep_00", "measurement.json", obj=m)
    rep = V.verify(out, recompute_models=False)
    assert "rep_00:endpoints_recomputed" in rep["failures"]


@pytest.mark.parametrize("case", ["raw_artifact", "journal", "certificate", "twin", "stray", "design"])
def test_direct_tampering_is_caught(base, tmp_path, case):
    out = _copy(base, tmp_path)
    if case == "raw_artifact":
        p = out / "runs" / "rep_01" / "probes.npz"
        p.write_bytes(p.read_bytes() + b"\0")
        want = "journal_replay_state_machine_and_commits"
    elif case == "journal":
        lines = (out / "journal.jsonl").read_bytes().splitlines()
        lines[3] = lines[3].replace(b'"seq":3', b'"seq":3 ')
        (out / "journal.jsonl").write_bytes(b"\n".join(lines) + b"\n")
        want = "journal_chain_sequence_and_identity"
    elif case == "certificate":
        c = json.loads((out / "certificate.json").read_text())
        c["decision"] = "SUPPORTED"
        (out / "certificate.json").write_text(json.dumps(c))
        want = "certificate_statistics_and_decision_recomputed"
    elif case == "twin":
        p = out / "controls" / "untrained_twin.weights.h5"
        p.write_bytes(p.read_bytes()[:-1] + b"\1")
        want = "twin_copy_equals_ce2_twin"
    elif case == "stray":
        (out / "runs" / "rep_00" / "extra.npz").write_bytes(b"x")
        want = "every_run_file_committed_cached_or_quarantined"
    else:
        d = json.loads((out / "design.json").read_text())
        d["endpoints"]["kappa"] = 0.0
        (out / "design.json").write_text(json.dumps(d))
        want = "design_file_hash_matches_manifest"
    rep = V.verify(out, recompute_models=False)
    assert want in rep["failures"], rep["failures"]


def test_output_directory_rules(tmp_path):
    with pytest.raises(R.RunnerError):
        R.initialise(R.CRYPTO_ROOT / "evidence" / "autond_ce3_smoke_x", "smoke")
    with pytest.raises(R.RunnerError):
        R.initialise(tmp_path / "not_a_smoke_name", "smoke")
    with pytest.raises(R.RunnerError):
        R.initialise(tmp_path / "prod", "production")
    assert not (R.CRYPTO_ROOT / "evidence" / "autond_ce3_smoke_x").exists()


@pytest.mark.parametrize("case", ["missing", "pending", "wrong_hash", "no_approver"])
def test_production_refused_without_frozen_approved_preregistration(tmp_path, monkeypatch, case):
    def must_not_initialise(*a, **k):
        raise AssertionError("production initialisation was not refused")

    monkeypatch.setattr(R, "store_for", must_not_initialise)
    pr = D.render_preregistration()
    good = {"status": "APPROVED", "approved_design_sha256": D.design_sha256(), "approved_by": "x",
            "approved_utc": "2026-01-01T00:00:00Z"}
    pr["approval"] = {"pending": pr["approval"], "wrong_hash": {**good, "approved_design_sha256": "0" * 64},
                      "no_approver": {**good, "approved_by": None}}.get(case, good)
    path = tmp_path / "preregistration.json"
    if case != "missing":
        path.write_text(json.dumps(pr))
    out = R.EVIDENCE_ROOT / f"autond_ce3_production_refusal_test_{case}"
    assert not out.exists()
    try:
        with pytest.raises(R.RunnerError, match="production refused"):
            R.initialise(out, "production", prereg_path=path)
        assert not out.exists()
    finally:
        if out.is_dir() and not any(out.iterdir()):
            out.rmdir()
