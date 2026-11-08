"""End-to-end tests of the production lifecycle and the verifier on synthetic
data from known distributions. The network, the trail proxy and the
preregistration are replaced by test doubles; ciphertexts, tables and aKR
scores are real computations on toy-sized tables."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from audit.cryptography.experiments.ce5 import akr as A
from audit.cryptography.experiments.ce5 import analysis as AN
from audit.cryptography.experiments.ce5 import design as DS
from audit.cryptography.experiments.ce5 import runner as RU
from audit.cryptography.experiments.ce5 import verify as V

DIFF = (0x0040, 0x0000)
R, N_PAIRS = 8, 120
TOY_TABLE_PAIRS = 2_000_000


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    tdir = tmp_path_factory.mktemp("tables")
    hashes = {}
    for name, rounds, n in (("akr", 4, TOY_TABLE_PAIRS), ("lr", 5, TOY_TABLE_PAIRS)):
        counts = A.build_table(n, rounds, DIFF, rng=np.random.default_rng(rounds))
        hashes[name] = RU.write_npz(tdir / RU.TABLE_FILES[name], counts=counts, n_pairs=np.int64(n),
                                    rounds=np.int64(rounds), mask=np.array(A.MASK, dtype=np.uint16))
    rec = {"decision": "PASS", "tables": {k: {"sha256": v} for k, v in hashes.items()}}
    hashes["validation_record"] = "table_validation.json"
    hashes["validation_sha256"] = RU.write_json(tdir / "table_validation.json", rec)
    return tdir, hashes


@pytest.fixture(autouse=True)
def _ce5_root_is_the_toy_table_dir(tables, monkeypatch):
    monkeypatch.setattr(RU, "CE5_ROOT", tables[0])


def toy_design(hashes, *, tolerance=0.5):
    d = DS.design()
    for path in DS.pending_items(d):
        node, keys = d, path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = "test value"
    d["baseline_D"]["table_sha256"] = hashes["akr"]
    d["secondary_comparator_LR"]["table_sha256"] = hashes["lr"]
    d["gate"]["tolerance"] = {"accuracy": tolerance, "rates": tolerance}
    d["gate"]["pairs_per_class"] = 100
    d["primary"]["margin"] = 0.1
    d["baseline_D"]["table_pairs"] = TOY_TABLE_PAIRS
    d["baseline_D"]["table_validation"].update(record=hashes["validation_record"], sha256=hashes["validation_sha256"])
    d["replicate"]["n_replicates"] = R
    d["replicate"]["real_pairs_per_replicate"] = N_PAIRS
    return d


def write_prereg(path, d, *, approve=True, ots=True, bitcoin=True, verified=True):
    pre = DS.render_preregistration(d)
    pre["registered_identities"] = RU.source_manifest()
    if approve:
        pre["approval"].update(status="APPROVED", approved_by="test-approver", approved_utc="2026-11-05T00:00:00Z")
    path.write_text(json.dumps(pre, indent=2))
    if ots:
        digest = hashlib.sha256(path.read_bytes()).digest()
        tail = DS.OTS_PENDING_TAG + (DS.OTS_BITCOIN_TAG if bitcoin else b"")
        (path.parent / (path.name + ".ots")).write_bytes(DS.OTS_MAGIC + b"\x01\x08" + digest + b"\xf0\x10" + tail)
        if verified:
            write_ots_record(path)
    return path


def PLENTY():
    return 8 << 30


def write_ots_record(path, *, exit_code=0, extra="", digest=None, proof_sha=None):
    fsha = hashlib.sha256(path.read_bytes()).hexdigest()
    proof = path.parent / (path.name + ".ots")
    rec = {"command": ["ots", "-v", "verify", str(proof)], "exit_code": exit_code, "stdout": "",
           "stderr": (f"Assuming target filename is '{path}'\nHashing file, algorithm sha256\n"
                      f"Got digest {digest or fsha}\n{extra}"),
           "file_sha256": fsha, "proof_sha256": proof_sha or hashlib.sha256(proof.read_bytes()).hexdigest()}
    if exit_code == 0 and not extra:
        rec["stderr"] += "Success! Bitcoin block 900000 attests existence as of 2026-11-06 UTC\n"
    (path.parent / (path.name + ".ots.verification.json")).write_text(json.dumps(rec))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binder_ok():
    return {"sha256": DS.REFERENCE_CHECKPOINT_SHA256, "verdict": "PASS"}, None


def synthetic_replicate(kind: str, seed: int = 0):
    rng = np.random.default_rng(seed)

    def make(model, tables, n, index):
        out = {}
        for cls, real in (("real", True), ("random", False)):
            c = A.ciphertext_pairs(n, 5, DIFF, real=real, rng=rng)
            D = A.akr_scores(*c, tables["akr"][0], tables["akr"][1], index=index)
            out[cls] = {"c0l": c[0], "c0r": c[1], "c1l": c[2], "c1r": c[3], "D": D}
        F = rng.uniform(0.05, 1.0, size=(5, n))
        T = np.prod(F, axis=0)
        zD, zT = AN.normal_scores(out["real"]["D"] + 1e-12 * rng.normal(size=n)), AN.normal_scores(T)
        signal = {"D": zD, "T": zT}[kind]
        out["real"].update(N=(signal + 0.3 * rng.normal(size=n)).astype(np.float32), T=T, F=F,
                           LR=A.difference_scores(*(out["real"][w] for w in ("c0l", "c0r", "c1l", "c1r")),
                                                  tables["lr"][0], tables["lr"][1], index=index))
        out["random"]["N"] = rng.normal(size=n).astype(np.float32)
        return out
    return make


@pytest.fixture
def lifecycle(tables, tmp_path, monkeypatch):
    tdir, hashes = tables

    def run(kind="D", d=None, **kw):
        d = d or toy_design(hashes)
        monkeypatch.setattr(DS, "_DESIGN", d)
        pre = write_prereg(tmp_path / "preregistration.json", d, **kw)
        out = tmp_path / "run"
        man = RU.production(out, tdir, DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                            replicate=synthetic_replicate(kind), memory_probe=PLENTY)
        return man, out, pre
    return run


def _verify(out, tdir, pre):
    return V.verify(out, tdir, prereg_path=pre, subsample=2, table_sample=20_000)


def test_distribution_signal_is_detected_and_verified(lifecycle, tables):
    man, out, pre = lifecycle("D")
    assert man["gate"]["decision"] == "PASS"
    assert man["decision"] == "DISTRIBUTION_GREATER"
    v = _verify(out, tables[0], pre)
    assert v["status"] == "VERIFIED", v["problems"]
    assert v["recomputed_decision"] == "DISTRIBUTION_GREATER"


def test_trail_signal_is_detected_and_verified(lifecycle, tables):
    man, out, pre = lifecycle("T")
    assert man["decision"] == "TRAIL_GREATER"
    assert _verify(out, tables[0], pre)["status"] == "VERIFIED"


def test_failed_gate_is_unadjudicated_and_runs_no_replicate(lifecycle, tables):
    d = toy_design(tables[1], tolerance=1e-6)
    man, out, pre = lifecycle("D", d=d)
    assert man["gate"]["decision"] == "FAIL" and man["decision"] == "UNADJUDICATED"
    assert not (out / "raw").exists()
    assert _verify(out, tables[0], pre)["recomputed_decision"] == "UNADJUDICATED"


def test_verifier_detects_tampered_raw_file(lifecycle, tables):
    man, out, pre = lifecycle("D")
    p = out / "raw" / "rep03.npz"
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    arr["real_N"] = arr["real_N"][::-1].copy()
    p.unlink()
    np.savez_compressed(p, **arr)
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED" and any("rep03" in x for x in v["problems"])


def test_verifier_detects_resealed_tampering(lifecycle, tables):
    man, out, pre = lifecycle("D")
    p = out / "raw" / "rep02.npz"
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    arr["real_D"] = arr["real_D"] * 1.5
    p.unlink()
    np.savez_compressed(p, **arr)
    m = json.loads((out / "manifest.json").read_text())
    m["replicates"]["rep02"]["raw_sha256"] = RU.sha256_file(p)
    (out / "manifest.json").write_text(json.dumps(m))
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED"


def test_verifier_detects_edited_summary(lifecycle, tables):
    man, out, pre = lifecycle("D")
    m = json.loads((out / "manifest.json").read_text())
    m["replicates"]["rep01"]["delta"] += 0.01
    m["decision"] = "EQUIVALENT"
    (out / "manifest.json").write_text(json.dumps(m))
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED"
    assert any("rep01 delta" in x for x in v["problems"]) and any("decision" in x for x in v["problems"])


def test_verifier_detects_missing_file(lifecycle, tables):
    man, out, pre = lifecycle("D")
    (out / "raw" / "rep05.npz").unlink()
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED" and any("rep05" in x for x in v["problems"])


def test_verifier_detects_modified_table(lifecycle, tables, tmp_path):
    man, out, pre = lifecycle("D")
    t2 = tmp_path / "t2"
    t2.mkdir()
    for name, fn in RU.TABLE_FILES.items():
        (t2 / fn).write_bytes((tables[0] / fn).read_bytes())
    with np.load(t2 / RU.TABLE_FILES["akr"]) as z:
        arr = {k: z[k] for k in z.files}
    arr["counts"] = arr["counts"].copy()
    arr["counts"][0] += 1
    (t2 / RU.TABLE_FILES["akr"]).unlink()
    np.savez_compressed(t2 / RU.TABLE_FILES["akr"], **arr)
    v = _verify(out, t2, pre)
    assert v["status"] == "NOT_VERIFIED"


def test_production_refuses_without_approval(lifecycle):
    with pytest.raises(SystemExit, match="not APPROVED"):
        lifecycle("D", approve=False)


def test_production_refuses_without_timestamp(lifecycle):
    with pytest.raises(SystemExit, match="OpenTimestamps"):
        lifecycle("D", ots=False)


def test_production_refuses_wrong_token(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "p.json", d)
    for tok in (None, "run", "RUN_GCE5_PRODUCTIO"):
        with pytest.raises(SystemExit, match="token"):
            RU.production(tmp_path / "o", tables[0], tok, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok, memory_probe=PLENTY)


def test_production_refuses_modified_configuration(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    pre = write_prereg(tmp_path / "p.json", d)
    d2 = copy.deepcopy(d)
    d2["primary"]["margin"] = 0.2
    monkeypatch.setattr(DS, "_DESIGN", d2)
    with pytest.raises(SystemExit, match="refused"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok, memory_probe=PLENTY)


def test_production_refuses_unregistered_table(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    d["baseline_D"]["table_sha256"] = "0" * 64
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "p.json", d)
    with pytest.raises(SystemExit, match="does not name the registered tables"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                      memory_probe=PLENTY)
    with pytest.raises(RuntimeError, match="not the registered"):
        RU.registered_tables(tables[0], d)


def test_production_refuses_modified_checkpoint(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "p.json", d)

    def bad_binder():
        raise RuntimeError("checkpoint sha256 differs from frozen")
    with pytest.raises(RuntimeError, match="checkpoint"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=bad_binder, memory_probe=PLENTY)
    assert not (tmp_path / "o" / "gate_raw.npz").exists()


def test_production_refuses_the_real_draft_preregistration():
    pre = RU.CRYPTO_ROOT / DS.PREREGISTRATION_RELPATH
    with pytest.raises(SystemExit, match="unset items"):
        RU.production(RU.CE5_ROOT / "evidence" / "must_not_exist", RU.CE5_ROOT / "tables",
                      DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok, memory_probe=PLENTY)
    assert not (RU.CE5_ROOT / "evidence" / "must_not_exist").exists()


def test_raw_files_are_never_overwritten(tmp_path):
    RU.write_npz(tmp_path / "x.npz", a=np.zeros(3))
    with pytest.raises(FileExistsError):
        RU.write_npz(tmp_path / "x.npz", a=np.ones(3))


def test_resume_skips_completed_replicates_and_refuses_other_design(lifecycle, tables, monkeypatch, tmp_path):
    man, out, pre = lifecycle("D")
    before = {k: v["raw_sha256"] for k, v in man["replicates"].items()}
    again = RU.production(out, tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                          replicate=synthetic_replicate("D", seed=99), memory_probe=PLENTY)
    assert {k: v["raw_sha256"] for k, v in again["replicates"].items()} == before
    d2 = copy.deepcopy(DS._DESIGN)
    d2["primary"]["margin"] = 0.3
    monkeypatch.setattr(DS, "_DESIGN", d2)
    pre2 = write_prereg(tmp_path / "p2.json", d2)
    with pytest.raises(SystemExit, match="different design"):
        RU.production(out, tables[0], DS.CONFIRM_TOKEN, prereg_path=pre2, prereg_sha256=_sha(pre2), binder=binder_ok, memory_probe=PLENTY)


def test_calibration_is_blinded(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    d["replicate"]["real_pairs_per_replicate"] = 80
    rec = RU.calibrate(tmp_path, tables[0], 80, replicates=4, binder=binder_ok,
                       replicate=synthetic_replicate("D", seed=3), memory_probe=PLENTY)
    text = (tmp_path / "calibration.json").read_text()
    assert rec["blinded"] and abs(sum(rec["centred_deltas"])) < 1e-12
    assert "rho" not in text and '"delta"' not in text and "mean" not in text
    ps = rec["planning_spread"]
    assert ps["sigma_plan"] >= ps["u_sd"] > ps["sample_sd"] > 0 and ps["n"] == 4
    assert rec["table_validation"]["decision"] == "PASS"


def test_ots_parser_reads_an_existing_audit_timestamp():
    prereg = RU.CRYPTO_ROOT.parent / "dataset" / "preregistrations" / "d2n_fresh_draw_null.json"
    proof = prereg.parent / (prereg.name + ".ots")
    if not proof.exists():
        pytest.skip("D2-N timestamp proof not present")
    assert DS.ots_digest(proof) == hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert DS.ots_digest(prereg) is None


def test_production_refuses_pending_only_timestamp(lifecycle):
    with pytest.raises(SystemExit, match="Bitcoin attestation"):
        lifecycle("D", bitcoin=False)


def test_production_refuses_low_memory(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "p.json", d)
    with pytest.raises(SystemExit, match="available memory"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                      memory_probe=lambda: 64 << 20)
    with pytest.raises(SystemExit, match="available memory"):
        RU.production(tmp_path / "o2", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                      memory_probe=lambda: None)


def test_stages_refuse_while_their_fields_are_unset(tmp_path, monkeypatch):
    d = DS.design()
    d["baseline_D"]["algorithm_source_text"] = DS.UNSET
    monkeypatch.setattr(DS, "_DESIGN", d)
    for fn in (lambda: RU.build_tables(tmp_path / "t", 10), lambda: RU.table_variability(tmp_path / "v")):
        with pytest.raises(SystemExit, match="algorithm_source_text"):
            fn()
    assert not (tmp_path / "t").exists()


def test_calibration_refuses_while_margin_is_unset(tmp_path, monkeypatch):
    d = DS.design()
    d["primary"]["margin"] = DS.UNSET
    monkeypatch.setattr(DS, "_DESIGN", d)
    with pytest.raises(SystemExit, match="primary.margin"):
        RU.calibrate(tmp_path, tmp_path, 10, binder=binder_ok, memory_probe=PLENTY)
    assert not (tmp_path / "calibration.json").exists()


def test_verifier_rejects_pending_only_timestamp(lifecycle, tables):
    man, out, pre = lifecycle("D")
    proof = pre.parent / (pre.name + ".ots")
    proof.write_bytes(proof.read_bytes().replace(DS.OTS_BITCOIN_TAG, b""))
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED" and any("Bitcoin" in x for x in v["problems"])


def _reseal(out, rid, edit):
    p = out / "raw" / f"{rid}.npz"
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    edit(arr)
    p.unlink()
    np.savez_compressed(p, **arr)
    m = json.loads((out / "manifest.json").read_text())
    m["replicates"][rid]["raw_sha256"] = RU.sha256_file(p)
    (out / "manifest.json").write_text(json.dumps(m))


def _edit_manifest(out, edit):
    m = json.loads((out / "manifest.json").read_text())
    edit(m)
    (out / "manifest.json").write_text(json.dumps(m))


@pytest.mark.parametrize("case", ["trail", "labels", "network", "interval", "correlation", "outcome",
                                  "design_identity", "checkpoint_identity", "secondary", "gate_ci"])
def test_verifier_detects_each_tampering(lifecycle, tables, case):
    man, out, pre = lifecycle("D")

    def swap(a):
        for w in ("c0l", "c0r", "c1l", "c1r", "N", "D"):
            a[f"real_{w}"], a[f"random_{w}"] = a[f"random_{w}"].copy(), a[f"real_{w}"].copy()
    edits = {
        "trail": lambda: _reseal(out, "rep01", lambda a: a.__setitem__("real_T", a["real_T"][::-1].copy())),
        "labels": lambda: _reseal(out, "rep01", swap),
        "network": lambda: _reseal(out, "rep01", lambda a: a.__setitem__("real_N", a["real_N"][::-1].copy())),
        "interval": lambda: _edit_manifest(out, lambda m: m["analysis"]["primary"]["interval"].__setitem__(
            "lower", m["analysis"]["primary"]["interval"]["lower"] - 0.01)),
        "correlation": lambda: _edit_manifest(out, lambda m: m["replicates"]["rep02"].__setitem__(
            "rho_N_T_real", m["replicates"]["rep02"]["rho_N_T_real"] + 0.01)),
        "outcome": lambda: _edit_manifest(out, lambda m: m.__setitem__("decision", "TRAIL_GREATER")),
        "design_identity": lambda: _edit_manifest(out, lambda m: m.__setitem__("design_sha256", "0" * 64)),
        "checkpoint_identity": lambda: _edit_manifest(out, lambda m: m["model_binding"].__setitem__("sha256", "0" * 64)),
        "secondary": lambda: _edit_manifest(out, lambda m: m["analysis"]["secondary"]["delta_LR"].__setitem__(
            "outcome", "EQUIVALENT" if m["analysis"]["secondary"]["delta_LR"]["outcome"] != "EQUIVALENT" else "INCONCLUSIVE")),
        "gate_ci": lambda: _edit_manifest(out, lambda m: m["gate"]["cp95"]["accuracy"].__setitem__(0, 0.5)),
    }
    assert _verify(out, tables[0], pre)["status"] == "VERIFIED"
    edits[case]()
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED", case


def test_verifier_detects_unapproved_or_mismatched_timestamp(lifecycle, tables):
    man, out, pre = lifecycle("D")
    proof = pre.parent / (pre.name + ".ots")
    b = bytearray(proof.read_bytes())
    b[len(DS.OTS_MAGIC) + 2] ^= 0xFF
    proof.write_bytes(bytes(b))
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED" and any("timestamp" in x for x in v["problems"])
    p = json.loads(pre.read_text())
    p["approval"]["status"] = "PROPOSED_PENDING_APPROVAL"
    pre.write_text(json.dumps(p))
    v = _verify(out, tables[0], pre)
    assert any("not APPROVED" in x for x in v["problems"])


def test_verifier_never_overwrites_its_output(lifecycle, tables):
    man, out, pre = lifecycle("D")
    a = _verify(out, tables[0], pre)
    b = _verify(out, tables[0], pre)
    assert a["written_to"] == "verification.json" and b["written_to"] == "verification_1.json"


def test_stage_outputs_are_never_overwritten(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    d["replicate"]["real_pairs_per_replicate"] = 80
    monkeypatch.setattr(DS, "_DESIGN", d)
    (tmp_path / "calibration.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        RU.calibrate(tmp_path, tables[0], 80, binder=binder_ok, memory_probe=PLENTY)
    with pytest.raises(SystemExit, match="already exists"):
        RU.build_tables(tables[0], 10)


def test_calibration_refuses_an_unregistered_pair_count(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    d["replicate"]["real_pairs_per_replicate"] = 80
    monkeypatch.setattr(DS, "_DESIGN", d)
    with pytest.raises(SystemExit, match="not the registered"):
        RU.calibrate(tmp_path, tables[0], 81, binder=binder_ok, memory_probe=PLENTY)


def test_variability_preserves_tables_scores_and_refuses_rerun(tmp_path, monkeypatch):
    d = DS.design()
    d["baseline_D"]["table_pairs"] = 20_000
    monkeypatch.setattr(DS, "_DESIGN", d)
    rec = RU.table_variability(tmp_path, n_tables=3, n_eval=200)
    assert len(rec["rows"]) == 3 and rec["status"].startswith("proposal")
    for r in rec["rows"]:
        assert RU.sha256_file(tmp_path / "variability" / f"table_{r['table']}.npz") == r["table_sha256"]
        assert RU.sha256_file(tmp_path / "variability" / f"scores_{r['table']}.npz") == r["scores_sha256"]
        assert r["confusion"]["tp"] + r["confusion"]["fn"] == 200
    assert rec["proposed_tolerance"]["accuracy"] >= 0.005 and rec["proposed_tolerance"]["rates"] >= 0.010
    with pytest.raises(SystemExit, match="already exists"):
        RU.table_variability(tmp_path, n_tables=3, n_eval=200)


def test_load_tables_rejects_wrong_dtype_or_sum(tables, tmp_path):
    for fn in RU.TABLE_FILES.values():
        (tmp_path / fn).write_bytes((tables[0] / fn).read_bytes())
    with np.load(tmp_path / RU.TABLE_FILES["akr"]) as z:
        arr = {k: z[k] for k in z.files}
    arr["counts"] = arr["counts"].astype(np.int64)
    (tmp_path / RU.TABLE_FILES["akr"]).unlink()
    np.savez_compressed(tmp_path / RU.TABLE_FILES["akr"], **arr)
    with pytest.raises(RuntimeError, match="uint32"):
        RU.load_tables(tmp_path)


def test_variability_verifier_recomputes_and_detects_tampering(tmp_path, monkeypatch):
    d = DS.design()
    d["baseline_D"]["table_pairs"] = 20_000
    monkeypatch.setattr(DS, "_DESIGN", d)
    RU.table_variability(tmp_path, n_tables=3, n_eval=200)
    v = V.verify_variability(tmp_path, per_table=2)
    assert v["status"] == "VERIFIED" and v["scalar_rescored_pairs"] == 12, v["problems"]
    p = tmp_path / "variability" / "scores_1.npz"
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    arr["real_score"] = arr["real_score"] * 1.01
    p.unlink()
    np.savez_compressed(p, **arr)
    rec = json.loads((tmp_path / "table_variability.json").read_text())
    rec["rows"][1]["scores_sha256"] = RU.sha256_file(p)
    (tmp_path / "table_variability.json").write_text(json.dumps(rec))
    assert V.verify_variability(tmp_path, per_table=50)["status"] == "NOT_VERIFIED"


# ---- fail-closed table validation

def test_tables_stage_stops_and_writes_no_table_if_its_check_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(RU.REF, "table_check", lambda *a, **k: {"p_value": 0.0, "cells": 0})
    with pytest.raises(SystemExit, match="independent check"):
        RU.build_tables(tmp_path, 20_000, rng=np.random.default_rng(0))
    assert not any((tmp_path / f).exists() for f in RU.TABLE_FILES.values())
    assert list(tmp_path.glob("tables_manifest_FAILED_*.json"))


def test_tablecheck_exit_code_is_nonzero_on_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(RU, "validate_tables", lambda out, tdir: {"decision": "FAIL", "components": {"x": False}})
    assert RU.main(["tablecheck", str(tmp_path), "--tables", str(tmp_path)]) == 1
    monkeypatch.setattr(RU, "validate_tables", lambda out, tdir: {"decision": "PASS", "components": {"x": True}})
    assert RU.main(["tablecheck", str(tmp_path), "--tables", str(tmp_path)]) == 0


@pytest.mark.parametrize("problem", ["missing", "hash", "fail", "other_tables"])
def test_calibration_and_production_require_a_passing_registered_validation(tables, tmp_path, monkeypatch, problem):
    d = toy_design(tables[1])
    d["replicate"]["real_pairs_per_replicate"] = 80
    rec = {"decision": "PASS", "tables": {"akr": {"sha256": tables[1]["akr"]}, "lr": {"sha256": tables[1]["lr"]}}}
    if problem == "fail":
        rec["decision"] = "FAIL"
    if problem == "other_tables":
        rec["tables"]["akr"]["sha256"] = "1" * 64
    monkeypatch.setattr(RU, "CE5_ROOT", tmp_path)
    path = tmp_path / "tv.json"
    digest = RU.write_json(path, rec)
    if problem == "missing":
        path.unlink()
    d["baseline_D"]["table_validation"].update(record="tv.json", sha256="0" * 64 if problem == "hash" else digest)
    monkeypatch.setattr(DS, "_DESIGN", d)
    with pytest.raises(SystemExit, match="table-validation|table validation"):
        RU.calibrate(tmp_path / "cal", tables[0], 80, binder=binder_ok, memory_probe=PLENTY)
    pre = write_prereg(tmp_path / "p.json", d)
    with pytest.raises(SystemExit, match="table-validation|table validation"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                      memory_probe=PLENTY)
    assert not (tmp_path / "o" / "gate_raw.npz").exists()


def test_registered_tables_checks_the_registered_pair_count(tables):
    d = toy_design(tables[1])
    d["baseline_D"]["table_pairs"] = 10_000_000
    with pytest.raises(RuntimeError, match="not the registered"):
        RU.registered_tables(tables[0], d)


def test_production_refuses_without_recorded_ots_verify(lifecycle):
    with pytest.raises(SystemExit, match="ots verify"):
        lifecycle("D", verified=False)


# ---- crash recovery

def _crash_copy(out, tmp_path, name):
    new = tmp_path / name
    new.mkdir()
    (new / "gate_raw.npz").write_bytes((out / "gate_raw.npz").read_bytes())
    m = json.loads((out / "manifest.json").read_text())
    for k in ("gate", "analysis", "decision", "finished_utc"):
        m.pop(k, None)
    m["replicates"] = {}
    (new / "manifest.json").write_text(json.dumps(m))
    return new


def test_production_recovers_a_verified_orphan_gate_file(lifecycle, tables, tmp_path):
    man, out, pre = lifecycle("D")
    new = _crash_copy(out, tmp_path, "crashed")
    before = RU.sha256_file(new / "gate_raw.npz")
    m2 = RU.production(new, tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                       replicate=synthetic_replicate("D", seed=5), memory_probe=PLENTY)
    assert "recovered" in m2["gate"] and m2["gate"]["confusion"] == man["gate"]["confusion"]
    assert RU.sha256_file(new / "gate_raw.npz") == before == m2["gate"]["raw_sha256"]
    assert m2["decision"] in AN.OUTCOMES


def test_production_refuses_a_corrupt_orphan_gate_file_and_keeps_it(lifecycle, tables, tmp_path):
    man, out, pre = lifecycle("D")
    new = _crash_copy(out, tmp_path, "corrupt")
    with np.load(new / "gate_raw.npz") as z:
        arr = {k: z[k] for k in z.files}
    arr["real_score"] = arr["real_score"] * 1.0001
    (new / "gate_raw.npz").unlink()
    np.savez_compressed(new / "gate_raw.npz", **arr)
    kept = RU.sha256_file(new / "gate_raw.npz")
    with pytest.raises(SystemExit, match="do not re-derive"):
        RU.production(new, tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok, memory_probe=PLENTY)
    assert RU.sha256_file(new / "gate_raw.npz") == kept and not (new / "raw").exists()


def test_initial_manifest_is_written_before_the_gate(lifecycle, tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "p.json", d)

    def boom(*a, **k):
        raise OSError("simulated crash")
    monkeypatch.setattr(RU, "run_gate", boom)
    with pytest.raises(OSError):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok,
                      memory_probe=PLENTY)
    m = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert "gate" not in m and m["design_sha256"] == DS.design_sha256(d)


# ---- verifier: summaries, gate fields, raw layout

def _gate_edit(key, val):
    return lambda m: m["gate"].__setitem__(key, val)


@pytest.mark.parametrize("case", ["gate_accuracy", "gate_tpr", "gate_tnr", "gate_n", "gate_reproduced",
                                  "primary_deltas", "primary_median", "primary_k", "sign_p", "sign_npos",
                                  "n_valid", "sec_rho_median", "sec_rho_max", "sec_deltaD_median", "sec_deltaT_p",
                                  "rep_z", "rep_distinct_T"])
def test_verifier_detects_summary_tampering_with_raw_files_unchanged(lifecycle, tables, case):
    man, out, pre = lifecycle("D")
    edits = {
        "gate_accuracy": _gate_edit("accuracy", 0.5), "gate_tpr": _gate_edit("tpr", 0.5),
        "gate_tnr": _gate_edit("tnr", 0.5), "gate_n": _gate_edit("n", 123), "gate_reproduced": _gate_edit("reproduced", False),
        "primary_deltas": lambda m: m["analysis"]["primary"]["deltas"].__setitem__(0, 9.0),
        "primary_median": lambda m: m["analysis"]["primary"]["interval"].__setitem__("median", 9.0),
        "primary_k": lambda m: m["analysis"]["primary"]["interval"].__setitem__("k", 2),
        "sign_p": lambda m: m["analysis"]["primary"].__setitem__("p_value", 0.5),
        "sign_npos": lambda m: m["analysis"]["primary"].__setitem__("n_positive", 0),
        "n_valid": lambda m: m["analysis"].__setitem__("n_valid", 1),
        "sec_rho_median": lambda m: m["analysis"]["secondary"]["rho_N_T_real"].__setitem__("median", 0.0),
        "sec_rho_max": lambda m: m["analysis"]["secondary"]["rho_N_D_random"].__setitem__("max", 0.99),
        "sec_deltaD_median": lambda m: m["analysis"]["secondary"]["delta_D"].__setitem__("median", 9.0),
        "sec_deltaT_p": lambda m: m["analysis"]["secondary"]["delta_T"].__setitem__("p_value", 0.123),
        "rep_z": lambda m: m["replicates"]["rep03"].__setitem__("z_T", 0.0),
        "rep_distinct_T": lambda m: m["replicates"]["rep03"].__setitem__("distinct_T", 1),
    }
    _edit_manifest(out, edits[case])
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED", case


@pytest.mark.parametrize("case", ["dtype", "nonfinite", "shape", "extra_key", "infeasible_T"])
def test_verifier_detects_raw_layout_violations(lifecycle, tables, case):
    man, out, pre = lifecycle("D")

    def infeasible(a):
        a["real_F"] = a["real_F"].copy()
        a["real_F"][0, 0] = 0.0
    edits = {"dtype": lambda a: a.__setitem__("real_N", a["real_N"].astype(np.float64)),
             "nonfinite": lambda a: a["random_D"].__setitem__(0, np.nan),
             "shape": lambda a: a.__setitem__("real_T", a["real_T"][:-1].copy()),
             "extra_key": lambda a: a.__setitem__("random_T", np.zeros(N_PAIRS)),
             "infeasible_T": infeasible}
    _reseal(out, "rep04", edits[case])
    v = _verify(out, tables[0], pre)
    assert v["status"] == "NOT_VERIFIED", case


def test_verifier_requires_the_recorded_ots_verify_output(lifecycle, tables):
    man, out, pre = lifecycle("D")
    (pre.parent / (pre.name + ".ots.verification.json")).unlink()
    v = _verify(out, tables[0], pre)
    assert any("ots verify" in x for x in v["problems"])


def test_verifier_checks_the_exact_gate_size(lifecycle, tables, tmp_path, monkeypatch):
    man, out, pre = lifecycle("D")
    d = copy.deepcopy(DS._DESIGN)
    d["gate"]["pairs_per_class"] = 99
    monkeypatch.setattr(DS, "_DESIGN", d)
    v = _verify(out, tables[0], pre)
    assert any("pairs per class" in x for x in v["problems"])


# ---- variability record tampering

@pytest.mark.parametrize("case", ["row_accuracy", "row_tpr", "row_tnr", "three_sd", "proposed", "rule", "n_tables",
                                  "table_pairs", "sd"])
def test_variability_verifier_detects_record_tampering(tmp_path, monkeypatch, case):
    d = DS.design()
    d["baseline_D"]["table_pairs"] = 20_000
    monkeypatch.setattr(DS, "_DESIGN", d)
    RU.table_variability(tmp_path, n_tables=3, n_eval=200)
    assert V.verify_variability(tmp_path, per_table=1)["status"] == "VERIFIED"
    rec = json.loads((tmp_path / "table_variability.json").read_text())
    {"row_accuracy": lambda: rec["rows"][0].__setitem__("accuracy", rec["rows"][0]["accuracy"] - 0.25),
     "row_tpr": lambda: rec["rows"][1].__setitem__("tpr", rec["rows"][1]["tpr"] - 0.25),
     "row_tnr": lambda: rec["rows"][2].__setitem__("tnr", rec["rows"][2]["tnr"] - 0.25),
     "three_sd": lambda: rec["three_sd"].__setitem__("accuracy", 0.9),
     "proposed": lambda: rec["proposed_tolerance"].__setitem__("rates", 0.5),
     "rule": lambda: rec.__setitem__("rule", "tau = 0"),
     "n_tables": lambda: rec.__setitem__("n_tables", 5),
     "table_pairs": lambda: rec.__setitem__("table_pairs", 10),
     "sd": lambda: rec["sd"].__setitem__("tnr", rec["sd"]["tnr"] + 0.5)}[case]()
    (tmp_path / "table_variability.json").write_text(json.dumps(rec))
    assert V.verify_variability(tmp_path, per_table=1)["status"] == "NOT_VERIFIED", case


def test_table_validation_record_must_be_repository_relative(tables, monkeypatch):
    d = toy_design(tables[1])
    d["baseline_D"]["table_validation"]["record"] = str(tables[0] / "table_validation.json")
    with pytest.raises(SystemExit, match="relative"):
        RU.require_table_validation(d)


def test_registered_design_records_a_relative_validation_path():
    rec = DS.design()["baseline_D"]["table_validation"]["record"]
    assert not rec.startswith("/") and rec.startswith("validation/")


# ---- OpenTimestamps verification record

def _prereg_with_proof(tmp_path):
    p = tmp_path / "preregistration.json"
    p.write_text('{"x": 1}')
    digest = hashlib.sha256(p.read_bytes()).digest()
    (tmp_path / "preregistration.json.ots").write_bytes(DS.OTS_MAGIC + b"\x01\x08" + digest + DS.OTS_BITCOIN_TAG)
    return p


@pytest.mark.parametrize("case,want", [
    ("ok", True), ("exit1", False), ("no_success_line", False), ("wrong_digest", False),
    ("failure_marker", False), ("stale_file", False), ("stale_proof", False), ("missing", False)])
def test_ots_verification_record_rules(tmp_path, case, want):
    p = _prereg_with_proof(tmp_path)
    if case == "ok":
        write_ots_record(p)
    elif case == "exit1":
        write_ots_record(p, exit_code=1)
    elif case == "no_success_line":
        write_ots_record(p, extra="Got 1 attestation(s)\n")
    elif case == "wrong_digest":
        write_ots_record(p, digest="0" * 64)
        rec = json.loads((tmp_path / "preregistration.json.ots.verification.json").read_text())
        rec["stderr"] += "Success! Bitcoin block 1 attests existence\n"
        (tmp_path / "preregistration.json.ots.verification.json").write_text(json.dumps(rec))
    elif case == "failure_marker":
        write_ots_record(p)
        rec = json.loads((tmp_path / "preregistration.json.ots.verification.json").read_text())
        rec["stderr"] += "Bitcoin verification failed: bad merkle root\n"
        (tmp_path / "preregistration.json.ots.verification.json").write_text(json.dumps(rec))
    elif case == "stale_file":
        write_ots_record(p)
        p.write_text('{"x": 2}')
    elif case == "stale_proof":
        write_ots_record(p, proof_sha="0" * 64)
    ok, why = DS.ots_verification_ok(tmp_path / "preregistration.json.ots.verification.json", p)
    assert ok is want, (case, why)


def test_the_real_unsuccessful_ots_verify_output_is_rejected(tmp_path):
    src = Path(__file__).resolve().parents[1] / "validation" / "r3_1_ots_20261105"
    if not (src / "stdout.txt").exists():
        pytest.skip("recorded D2-N ots verify output not present")
    p = _prereg_with_proof(tmp_path)
    rec = {"exit_code": int((src / "exit_code.txt").read_text()), "stdout": (src / "stdout.txt").read_text(),
           "stderr": (src / "stderr.txt").read_text(), "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
           "proof_sha256": hashlib.sha256((tmp_path / "preregistration.json.ots").read_bytes()).hexdigest()}
    (tmp_path / "preregistration.json.ots.verification.json").write_text(json.dumps(rec))
    ok, why = DS.ots_verification_ok(tmp_path / "preregistration.json.ots.verification.json", p)
    assert not ok and "exited 1" in why


def test_record_ots_verification_runs_the_command_and_never_overwrites(tmp_path):
    p = _prereg_with_proof(tmp_path)
    fsha = hashlib.sha256(p.read_bytes()).hexdigest()
    fake = tmp_path / "ots"
    fake.write_text("#!/bin/sh\necho 'Got digest " + fsha + "' >&2\necho 'Success! Bitcoin block 5 attests existence' >&2\nexit 0\n")
    fake.chmod(0o755)
    rec = RU.record_ots_verification(p, ots_bin=str(fake))
    assert rec["exit_code"] == 0 and rec["command"][1:3] == ["-v", "verify"] and rec["file_sha256"] == fsha
    assert DS.ots_verification_ok(tmp_path / "preregistration.json.ots.verification.json", p)[0]
    with pytest.raises(SystemExit, match="already exists"):
        RU.record_ots_verification(p, ots_bin=str(fake))


def test_anchoring_text_names_the_prereg_argument_and_its_derived_paths():
    text = DS.design()["anchoring"]
    assert "--prereg" in text and "--prereg-sha256" in text
    assert "'.ots' appended" in text and "'.verification.json' appended" in text
    assert "preregistration.json" not in text


def test_anchoring_paths_follow_the_supplied_file_name(tmp_path):
    p = tmp_path / "preregistration_abcd1234_draft_20990101T000000Z.json"
    p.write_text('{"x": 1}')
    digest = hashlib.sha256(p.read_bytes()).digest()
    Path(str(p) + ".ots").write_bytes(DS.OTS_MAGIC + b"\x01\x08" + digest + DS.OTS_BITCOIN_TAG)
    fsha = digest.hex()
    fake = tmp_path / "ots"
    fake.write_text("#!/bin/sh\necho 'Got digest " + fsha + "' >&2\necho 'Success! Bitcoin block 5 attests existence' >&2\nexit 0\n")
    fake.chmod(0o755)
    RU.record_ots_verification(p, ots_bin=str(fake))
    record = Path(str(p) + ".ots.verification.json")
    assert record.exists() and not (tmp_path / "preregistration.json.ots.verification.json").exists()
    assert DS.ots_verification_ok(record, p)[0]
    d = DS.design()
    pre = {"design": d, "approval": {"status": "APPROVED", "design_sha256": DS.design_sha256(d),
                                     "approved_by": "x", "approved_utc": "2099-01-01T00:00:00Z"}}
    assert DS.production_authorisation(pre, DS.CONFIRM_TOKEN, prereg_path=p) == (True, "approved")
    record.unlink()
    ok, why = DS.production_authorisation(pre, DS.CONFIRM_TOKEN, prereg_path=p)
    assert not ok and p.name + ".ots.verification.json" in why


def test_calibration_record_is_independently_verified_and_tampering_detected(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    d["replicate"]["real_pairs_per_replicate"] = 80
    monkeypatch.setattr(DS, "_DESIGN", d)
    RU.calibrate(tmp_path, tables[0], 80, replicates=10, binder=binder_ok,
                 replicate=synthetic_replicate("D", seed=7), memory_probe=PLENTY)
    v = V.verify_calibration(tmp_path)
    assert v["status"] == "VERIFIED", v["problems"]
    rec = json.loads((tmp_path / "calibration.json").read_text())
    rec["replicate_rule"]["chosen"] = 60 if rec["replicate_rule"]["chosen"] != 60 else 20
    (tmp_path / "calibration.json").write_text(json.dumps(rec))
    assert V.verify_calibration(tmp_path)["status"] == "NOT_VERIFIED"


# ---- explicit preregistration selection

def _approved(tables, tmp_path, monkeypatch, name="p.json"):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    return write_prereg(tmp_path / name, d)


def test_production_requires_an_explicit_preregistration_and_hash(tables, tmp_path, monkeypatch):
    pre = _approved(tables, tmp_path, monkeypatch)
    for kwargs in ({}, {"prereg_path": pre}, {"prereg_sha256": _sha(pre)}):
        with pytest.raises(SystemExit, match="explicit preregistration path"):
            RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, binder=binder_ok, memory_probe=PLENTY, **kwargs)
    assert not (tmp_path / "o").exists()


def test_production_rejects_a_hash_that_is_not_the_approved_file(tables, tmp_path, monkeypatch):
    pre = _approved(tables, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="is not the approved"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256="0" * 64,
                      binder=binder_ok, memory_probe=PLENTY)
    with pytest.raises(SystemExit, match="does not exist"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=tmp_path / "nope.json",
                      prereg_sha256=_sha(pre), binder=binder_ok, memory_probe=PLENTY)


def test_production_rejects_the_stale_default_preregistration():
    stale = RU.CRYPTO_ROOT / DS.PREREGISTRATION_RELPATH
    with pytest.raises(SystemExit, match="production refused"):
        RU.production(RU.CE5_ROOT / "o", RU.CE5_ROOT, DS.CONFIRM_TOKEN, prereg_path=stale, prereg_sha256=_sha(stale),
                      binder=binder_ok, memory_probe=PLENTY)


def test_production_rejects_an_unapproved_draft(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "p.json", d, approve=False)
    with pytest.raises(SystemExit, match="not APPROVED"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre),
                      binder=binder_ok, memory_probe=PLENTY)


def test_production_rejects_a_mismatched_design_hash(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = DS.render_preregistration(d)
    pre["registered_identities"] = RU.source_manifest()
    pre["approval"].update(status="APPROVED", approved_by="test-approver", approved_utc="2026-11-05T00:00:00Z",
                           design_sha256="f" * 64)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(pre))
    with pytest.raises(SystemExit, match="approval hash does not match"):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=path, prereg_sha256=_sha(path),
                      binder=binder_ok, memory_probe=PLENTY)


def test_resume_refuses_a_different_preregistration(lifecycle, tables, tmp_path):
    man, out, pre = lifecycle("D")
    other = tmp_path / "copy.json"
    other.write_text(pre.read_text() + " ")
    (tmp_path / "copy.json.ots").write_bytes(
        DS.OTS_MAGIC + b"\x01\x08" + hashlib.sha256(other.read_bytes()).digest() + b"\xf0\x10" + DS.OTS_BITCOIN_TAG)
    write_ots_record(other)
    with pytest.raises(SystemExit, match="different preregistration"):
        RU.production(out, tables[0], DS.CONFIRM_TOKEN, prereg_path=other, prereg_sha256=_sha(other),
                      binder=binder_ok, memory_probe=PLENTY)


# ---- source-code integrity

def _source_copy(tmp_path):
    dst = tmp_path / "src"
    for p in RU.SOURCE_ROOT.rglob("*.py"):
        rel = p.relative_to(RU.SOURCE_ROOT)
        if "historical" in rel.parts or "__pycache__" in rel.parts:
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / rel).write_bytes(p.read_bytes())
    return dst


@pytest.mark.parametrize("case,match", [("changed", "changed"), ("missing", "missing"), ("unexpected", "unexpected")])
def test_production_rejects_source_tampering_before_any_computation(tables, tmp_path, monkeypatch, case, match):
    pre = _approved(tables, tmp_path, monkeypatch)
    src = _source_copy(tmp_path)
    if case == "changed":
        (src / "akr.py").write_bytes((src / "akr.py").read_bytes() + b"\n")
    elif case == "missing":
        (src / "reference.py").unlink()
    else:
        (src / "extra_module.py").write_text("x = 1\n")
    calls = []
    with pytest.raises(SystemExit, match=match):
        RU.production(tmp_path / "o", tables[0], DS.CONFIRM_TOKEN, prereg_path=pre, prereg_sha256=_sha(pre),
                      binder=lambda: calls.append(1), memory_probe=PLENTY, source_root=src)
    assert not calls and not (tmp_path / "o").exists()


def test_unchanged_source_copy_passes_integrity(tables, tmp_path, monkeypatch):
    pre = _approved(tables, tmp_path, monkeypatch)
    src = _source_copy(tmp_path)
    assert RU.source_integrity(json.loads(pre.read_text()), source_root=src)["decision"] == "PASS"


def test_frozen_dependency_change_is_rejected(tmp_path):
    reg = {"registered_identities": RU.source_manifest()}
    crypto = tmp_path / "crypto"
    for f in RU.FROZEN_DEPENDENCIES:
        (crypto / f).parent.mkdir(parents=True, exist_ok=True)
        (crypto / f).write_bytes((RU.CRYPTO_ROOT / f).read_bytes())
    assert RU.source_integrity(reg, crypto_root=crypto)["decision"] == "PASS"
    (crypto / "gohr/speck.py").write_bytes((crypto / "gohr/speck.py").read_bytes() + b"#\n")
    with pytest.raises(SystemExit, match="dependency_changed"):
        RU.source_integrity(reg, crypto_root=crypto)


def test_preregistration_without_a_source_manifest_is_rejected(tables, tmp_path, monkeypatch):
    d = toy_design(tables[1])
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = DS.render_preregistration(d)
    with pytest.raises(SystemExit, match="no source manifest"):
        RU.source_integrity(pre)


def test_the_approved_preregistration_and_unchanged_sources_pass_end_to_end(lifecycle, tables):
    man, out, pre = lifecycle("D")
    assert man["source_integrity"]["decision"] == "PASS" and man["preregistration_sha256"] == _sha(pre)
    assert man["decision"] in AN.OUTCOMES


def test_verifier_detects_source_change_after_production(lifecycle, tables, tmp_path):
    man, out, pre = lifecycle("D")
    src = _source_copy(tmp_path)
    (src / "analysis.py").write_bytes((src / "analysis.py").read_bytes() + b"\n")
    v = V.verify(out, tables[0], prereg_path=pre, subsample=2, table_sample=20_000, source_root=src)
    assert any("sources on disk differ" in x for x in v["problems"])


# ---- amendment 01 (OTS verification method), end to end

def _leb128(n):
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _amend(pre, tmp_path, height=900000, time=1790000000):
    """Self-consistent amendment-01 evidence for a toy preregistration: a proof
    committing sha256(file digest) at a Bitcoin height, an 80-byte header
    carrying that Merkle root, and Java output and endpoint replies for it."""
    digest = hashlib.sha256(pre.read_bytes()).digest()
    payload = _leb128(height)
    proof = Path(str(pre) + ".ots")
    proof.write_bytes(DS.OTS_MAGIC + b"\x01\x08" + digest + b"\x08\x00" + DS.OTS_BITCOIN_TAG
                      + _leb128(len(payload)) + payload)
    msg = hashlib.sha256(digest).digest()
    hdr = b"\x00" * 4 + b"\x11" * 32 + msg + time.to_bytes(4, "little") + b"\x00" * 8
    bh, root = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()[::-1].hex(), msg[::-1].hex()
    a = {"schema": DS.AMENDMENT_SCHEMA, "amendment_id": DS.AMENDMENT_ID,
         "amends": {"preregistration_sha256": _sha(pre), "design_sha256": DS.design_sha256(),
                    "proof_sha256": _sha(proof)},
         "method": {"name": DS.JAVA_METHOD_NAME, "java_repository": DS.JAVA_OTS_REPOSITORY,
                    "java_commit": DS.JAVA_OTS_COMMIT, "java_jar_sha256": DS.JAVA_OTS_JAR_SHA256,
                    "java_runtime": DS.JAVA_RUNTIME, "java_endpoint": DS.JAVA_ENDPOINT,
                    "second_endpoint": DS.SECOND_ENDPOINT, "evidence_schema": DS.JAVA_EVIDENCE_SCHEMA},
         "registered_identities": RU.source_manifest()}
    a["approval"] = {"status": "APPROVED", "approved_by": "test-approver", "approved_utc": "2026-11-08T00:00:00Z",
                     "amendment_body_sha256": DS.amendment_body_sha256(a)}
    apath = tmp_path / "amendment.json"
    apath.write_text(json.dumps(a))
    err = (f"{DS.JAVA_ENDPOINT}/block-height/{height} {bh}\n{DS.JAVA_ENDPOINT}/block/{bh} BlockHeader{{merkleroot="
           f"'{root}', blockHash='{bh}', time='{time}'}}\nLite-client verification, assuming block {bh} is valid\n")
    rec = {"schema": DS.JAVA_EVIDENCE_SCHEMA, "method": DS.JAVA_METHOD_NAME, "amendment_sha256": _sha(apath),
           "java": {"source_commit": DS.JAVA_OTS_COMMIT, "jar_sha256": DS.JAVA_OTS_JAR_SHA256,
                    "runtime_version_output": f"OpenJDK Runtime Environment {DS.JAVA_RUNTIME}\n",
                    "command": ["java", "-jar", "OtsCli.jar", "-v", "verify", "-f", str(pre), str(proof)],
                    "exit_code": 0, "stdout": f"Success! Bitcoin block {height} attests data existed\n", "stderr": err},
           "second_endpoint": {"base": DS.SECOND_ENDPOINT, "responses": [
               {"path": f"/block-height/{height}", "body": bh}, {"path": f"/block/{bh}/header", "body": hdr.hex()}]},
           "file_sha256": _sha(pre), "proof_sha256": _sha(proof)}
    Path(str(proof) + DS.JAVA_EVIDENCE_SUFFIX).write_text(json.dumps(rec))
    return apath


def test_production_and_verifier_under_amendment_01(tables, tmp_path, monkeypatch):
    tdir, hashes = tables
    d = toy_design(hashes)
    monkeypatch.setattr(DS, "_DESIGN", d)
    pre = write_prereg(tmp_path / "preregistration.json", d, verified=False)
    apath = _amend(pre, tmp_path)
    kw = dict(prereg_path=pre, prereg_sha256=_sha(pre), binder=binder_ok, replicate=synthetic_replicate("D"),
              memory_probe=PLENTY)
    with pytest.raises(SystemExit, match="no recorded 'ots verify' run"):
        RU.production(tmp_path / "plain", tdir, DS.CONFIRM_TOKEN, **kw)      # the registered gate is unchanged
    with pytest.raises(SystemExit, match="both its path and its SHA-256"):
        RU.production(tmp_path / "half", tdir, DS.CONFIRM_TOKEN, amendment_path=apath, **kw)
    out = tmp_path / "run"
    man = RU.production(out, tdir, DS.CONFIRM_TOKEN, amendment_path=apath, amendment_sha256=_sha(apath), **kw)
    assert man["amendment_sha256"] == _sha(apath) and man["source_integrity"]["decision"] == "PASS"
    assert V.verify(out, tdir, prereg_path=pre, subsample=2, table_sample=20_000,
                    amendment_path=apath)["status"] == "VERIFIED"
    v = V.verify(out, tdir, prereg_path=pre, subsample=2, table_sample=20_000)
    assert any("names an amendment but none was supplied" in x for x in v["problems"])
