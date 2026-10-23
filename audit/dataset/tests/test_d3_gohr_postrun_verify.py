"""Post-run verifier for Gohr D3 outputs (audit/dataset/tools/d3_gohr_postrun_verify.py).

FIXTURES ARE SYNTHETIC-SCALE: one small real Gohr instance (attested build), a
throwaway committed preregistration and fabricated calibration evidence bound
to the Gohr tool, all in temporary directories; the runner's PRODUCTION_SIZES
is reduced to the fixture's. A passing verifier here is not evidence about any
production output.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from audit.dataset import canonical as C
from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d1_gohr_canonical_census as D1R
from audit.dataset.experiments import d3_gohr_confirmatory as R
from audit.dataset.tests import test_d3_method_validity as MV
from audit.dataset.tools import d0_gohr_attested_build as ATT
from audit.dataset.tools import d3_gohr_postrun_verify as V

SMALL = {"train": 6000, "validation": 3000, "test": 3000}


def _calibration(dest: Path, run_id: str = "fixture-run") -> Path:
    tool_sha = hashlib.sha256(R.CALIBRATION_TOOL_PATH.read_bytes()).hexdigest()

    def mutate(ev):
        ev["run_id"] = run_id
        ev["identity"]["tool_sha256"] = tool_sha
        ev["design"].update(family_roles={f: "confirmatory" for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES},
                            exploratory_alpha=0.01)
    return MV._write_cal(dest, MV._evidence({"design": {"partition_sizes": dict(SMALL), "feature_dimension": 64},
                                             "configuration": {"family_alpha_allocation": dict(R.EXPECTED_ALLOCATION)}},
                                            mutate=mutate))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "PRODUCTION_SIZES", dict(SMALL))
        root = tmp_path_factory.mktemp("postrun")
        inst = root / "d0"
        ATT.attested_build(inst, spec=dict(C.CANONICAL_SPEC, **{f"{n}_samples": v for n, v in SMALL.items()}),
                           command=["test-fixture"], log=lambda *a, **k: None)
        rec = MV._prereg_record(family_alpha_allocation=dict(R.EXPECTED_ALLOCATION),
                                family_roles=dict(R.REQUIRED_ROLES), tolerances=R.derived_tolerances(),
                                tolerance_justification={k: "test fixture justification text, not a claim"
                                                         for k in d3.TOLERANCE_KEYS})
        path, anchor = MV._commit_prereg(MV._git_repo(root), rec)
        mp.setattr(D1R, "PRODUCTION_SIZES", dict(SMALL))
        d1 = root / "d1"
        assert D1R.main(["--instance-dir", str(inst), "--out-dir", str(d1)]) == 0
        cal = _calibration(root / "cal")
        out = root / "d3"
        assert R.main(["--instance-dir", str(inst), "--d1-dir", str(d1), "--out-dir", str(out),
                       "--preregistration", str(path), "--preregistration-sha256", anchor,
                       "--calibration-dir", str(cal)]) == 0
        yield {"root": root, "out": out, "inst": inst, "d1": d1, "prereg": str(path), "anchor": anchor,
               "cal": str(cal), "mp": mp}


def _args(r, **over):
    a = {"out": r["out"], "inst": r["inst"], "d1": r["d1"], "prereg": r["prereg"], "anchor": r["anchor"],
         "cal": r["cal"], **over}
    return [str(a["out"]), str(a["inst"]), str(a["d1"]), a["prereg"], a["anchor"], str(a["cal"])]


def _snapshot(*dirs):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs for p in sorted(Path(d).rglob("*"))
            if p.is_file()}


def test_genuine_output_passes_and_nothing_is_written(run, capsys) -> None:
    before = _snapshot(run["out"], run["inst"], run["d1"])
    assert V.main(_args(run)) == 0
    text = capsys.readouterr().out
    assert "ALL CHECKS PASSED" in text and "FAIL " not in text
    assert "PASS  D1 evidence complete, bound to this instance, outcome PASS" in text
    for fam in ("R1", "R2", "A", "B", "C", "D", "E"):
        assert f"PASS  family {fam}:" in text
    assert _snapshot(run["out"], run["inst"], run["d1"]) == before


@pytest.mark.parametrize("edit", ["verdict_fail", "certificate_edited"])
def test_d1_evidence_changed_after_the_run_fails(run, tmp_path, capsys, edit) -> None:
    d = tmp_path / "d1_copy"
    shutil.copytree(run["d1"], d)
    p = d / D1R.CERTIFICATE_NAME
    c = json.loads(p.read_text())
    if edit == "verdict_fail":
        c["verdict"]["outcome"] = "FAIL"
        p.write_text(json.dumps(c))
        _edit_json(d / D1R.RUN_MANIFEST_NAME,
                   lambda m: m["files"].update({D1R.CERTIFICATE_NAME: hashlib.sha256(p.read_bytes()).hexdigest()}))
    else:
        c["note"] = "edited"
        p.write_text(json.dumps(c))
    assert V.main(_args(run, d1=d)) == 1
    assert "FAIL  D1 evidence" in capsys.readouterr().out


def test_a_protocol_mismatch_is_reported_as_fail_not_an_unhandled_exit(run, tmp_path, capsys) -> None:
    """check_protocol raises SystemExit; the verifier must record it as FAIL and continue, then exit 1."""
    def mutate_design(ev_dir):
        p = ev_dir / d3.CALIBRATION_EVIDENCE_NAME
        ev = json.loads(p.read_text())
        ev["design"]["exploratory_alpha"] = 0.05
        p.write_text(json.dumps(ev))
        _edit_json(ev_dir / d3.CALIBRATION_MANIFEST_NAME,
                   lambda m: m["files"].update({d3.CALIBRATION_EVIDENCE_NAME: hashlib.sha256(p.read_bytes()).hexdigest()}))
        return ev_dir
    other = mutate_design(_calibration(tmp_path / "cal_mismatch"))
    assert V.main(_args(run, cal=other)) == 1
    text = capsys.readouterr().out
    assert "FAIL  preregistration" in text and "SystemExit" in text and "SOME CHECKS FAILED" in text


def _copy_out(run, tmp_path) -> Path:
    d = tmp_path / "out_copy"
    shutil.copytree(run["out"], d)
    return d


def _edit_json(path: Path, mutate) -> None:
    obj = json.loads(path.read_text())
    mutate(obj)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


@pytest.mark.parametrize("damage", [
    "certificate_edited", "sealed_audit_edited", "manifest_seal_edited", "extra_file", "missing_file",
    "run_id_mismatch", "outcome_edited", "family_outcome_edited",
])
def test_tampered_output_fails(run, tmp_path, capsys, damage) -> None:
    d = _copy_out(run, tmp_path)
    if damage == "certificate_edited":
        _edit_json(d / R.CERTIFICATE_NAME, lambda c: c.update(interpretation="edited"))
    elif damage == "sealed_audit_edited":
        (d / R.AUDIT_FILE_NAME).write_bytes((d / R.AUDIT_FILE_NAME).read_bytes().replace(b'"schema_version"', b'"schema_versioN"', 1))
    elif damage == "manifest_seal_edited":
        _edit_json(d / R.RUN_MANIFEST_NAME, lambda m: m.update(audit_seal_sha256="0" * 64))
    elif damage == "extra_file":
        (d / "extra.txt").write_text("x")
    elif damage == "missing_file":
        (d / R.AUDIT_FILE_NAME).unlink()
    elif damage == "run_id_mismatch":
        _edit_json(d / R.RUN_MANIFEST_NAME, lambda m: m.update(run_id="0" * 32))
    elif damage == "outcome_edited":
        _edit_json(d / R.CERTIFICATE_NAME, lambda c: c["decision"].update(outcome="PASS", provisional_outcome="PASS"))
    elif damage == "family_outcome_edited":
        _edit_json(d / R.CERTIFICATE_NAME, lambda c: c["decision"]["family_outcomes"].update(B="PASS"))
    assert V.main(_args(run, out=d)) == 1
    assert "SOME CHECKS FAILED" in capsys.readouterr().out


def test_wrong_anchor_fails(run, capsys) -> None:
    assert V.main(_args(run, anchor="0" * 64)) == 1
    assert "FAIL  preregistration" in capsys.readouterr().out


def test_a_different_calibration_fails(run, tmp_path, capsys) -> None:
    other = _calibration(tmp_path / "other_cal", run_id="another-run")
    assert V.main(_args(run, cal=other)) == 1
    assert "FAIL  calibration" in capsys.readouterr().out


def test_tampered_d0_instance_fails(run, tmp_path, capsys) -> None:
    d = tmp_path / "d0_copy"
    shutil.copytree(run["inst"], d)
    data = bytearray((d / "validation.npz").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (d / "validation.npz").write_bytes(bytes(data))
    assert V.main(_args(run, inst=d)) == 1
    assert "FAIL  D0 instance" in capsys.readouterr().out


def test_a_missing_output_directory_fails_closed(run, tmp_path, capsys) -> None:
    assert V.main(_args(run, out=tmp_path / "absent")) == 1
    assert "FAIL  output files" in capsys.readouterr().out
