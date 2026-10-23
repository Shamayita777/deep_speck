"""Gohr confirmatory D3 runner (audit/dataset/experiments/d3_gohr_confirmatory.py).

FIXTURES ARE SYNTHETIC-SCALE: a small real Gohr instance (6,000 / 3,000 /
3,000 rows) is built through the attested builder into a temporary directory;
preregistrations are committed in throwaway git repositories; calibration
evidence is fabricated (test fixture, bound to the Gohr calibration tool's
hash). The runner's PRODUCTION_SIZES is reduced to the fixture's sizes ONLY in
the tests that need it. No production data is generated or read, and no
calibration is run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from audit.dataset import canonical as C
from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d1_gohr_canonical_census as D1R
from audit.dataset.experiments import d3_gohr_confirmatory as R
from audit.dataset.tests import test_d3_method_validity as MV
from audit.dataset.tools import d0_gohr_attested_build as ATT
from audit.dataset.tools import d3_gohr_method_validity_calibration as G

SMALL = {"train": 6000, "validation": 3000, "test": 3000}
SMALL_SPEC = dict(C.CANONICAL_SPEC, **{f"{n}_samples": v for n, v in SMALL.items()})
GOHR_TOOL_SHA = hashlib.sha256(R.CALIBRATION_TOOL_PATH.read_bytes()).hexdigest()


def _quiet(*a, **k):
    return None


#: The instance-bound D1 evidence of the module fixture instance (set by the fixture).
D1_EVIDENCE: dict = {}


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    root = tmp_path_factory.mktemp("gohr_d0")
    out = root / "instance"
    ATT.attested_build(out, spec=SMALL_SPEC, command=["test-fixture"], log=_quiet)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D1R, "PRODUCTION_SIZES", dict(SMALL))
        assert D1R.main(["--instance-dir", str(out), "--out-dir", str(root / "d1")]) == 0
    D1_EVIDENCE["dir"] = root / "d1"
    return out


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(R, "PRODUCTION_SIZES", dict(SMALL))


class Spy:
    def __init__(self, passthrough: bool):
        self.calls, self.passthrough = 0, passthrough

    def __call__(self, partitions, **kw):
        self.calls += 1
        if not self.passthrough:
            raise AssertionError("the D3 engine was invoked")
        return d3.class_stratified_audit(partitions, **kw)


@pytest.fixture
def blocking_spy(monkeypatch):
    s = Spy(passthrough=False)
    monkeypatch.setattr(R, "D3_ENGINE", s)
    return s


def _prereg(**over):
    return MV._prereg_record(**{"family_alpha_allocation": dict(R.EXPECTED_ALLOCATION),
                                "family_roles": dict(R.REQUIRED_ROLES), "tolerances": R.derived_tolerances(),
                                "tolerance_justification": {k: "test fixture justification text, not a claim"
                                                            for k in d3.TOLERANCE_KEYS}, **over})


def _calibration(tmp_path, *, sizes=None, alloc=None, roles=None, supported=(0.45, 0.55), tool_sha=GOHR_TOOL_SHA,
                 extra=None):
    alloc = dict(alloc or R.EXPECTED_ALLOCATION)
    payload = {"design": {"partition_sizes": dict(sizes or SMALL), "feature_dimension": 64},
               "configuration": {"family_alpha_allocation": alloc}}

    def mutate(ev):
        ev["identity"]["tool_sha256"] = tool_sha
        ev["design"].update(family_roles=dict(roles or {f: R.REQUIRED_ROLES[f]
                                                        for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}),
                            exploratory_alpha=0.01)
        if extra:
            extra(ev)
    return MV._write_cal(tmp_path / "cal", MV._evidence(payload, supported=supported, mutate=mutate))


def _gates(tmp_path, *, prereg=None, commit=True, **cal):
    path, sha = MV._commit_prereg(MV._git_repo(tmp_path), prereg or _prereg(), commit=commit)
    return path, sha, _calibration(tmp_path, **cal)


def _run(inst, out, path, sha, cal_dir, *extra, d1=None):
    return R.main(["--instance-dir", str(inst), "--d1-dir", str(d1 or D1_EVIDENCE["dir"]), "--out-dir", str(out),
                   "--preregistration", str(path), "--preregistration-sha256", sha, "--calibration-dir", str(cal_dir),
                   *extra])


def _d1_copy(tmp_path, mutate_cert=None, mutate_manifest=None, rehash=True) -> Path:
    """A copy of the fixture's D1 evidence, optionally edited (and re-hashed into its run manifest)."""
    d = tmp_path / "d1_copy"
    shutil.copytree(D1_EVIDENCE["dir"], d)
    cp, mp_ = d / D1R.CERTIFICATE_NAME, d / D1R.RUN_MANIFEST_NAME
    if mutate_cert:
        c = json.loads(cp.read_text())
        mutate_cert(c)
        cp.write_text(json.dumps(c, indent=2, sort_keys=True))
    m = json.loads(mp_.read_text())
    if rehash:
        m["files"][D1R.CERTIFICATE_NAME] = hashlib.sha256(cp.read_bytes()).hexdigest()
    if mutate_manifest:
        mutate_manifest(m)
    mp_.write_text(json.dumps(m, indent=2, sort_keys=True))
    return d


def _copy(instance, tmp_path) -> Path:
    dst = tmp_path / "instance_copy"
    shutil.copytree(instance, dst)
    return dst


def _no_output(out: Path) -> None:
    assert not out.exists(), "a refused run must write nothing"
    assert not list(out.parent.glob(".d3_gohr_staging_*")), "no staging leftovers"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_runner_calls_only_the_generic_confirmatory_engine() -> None:
    import ast
    tree = ast.parse(Path(R.__file__).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert R.D3_ENGINE is d3.class_stratified_audit
    assert not any("d3_gohr_distribution_statistics" in m for m in imported)
    assert not {"audit_distribution", "evaluate_decision", "print_report"} & (called | imported)
    assert {"verify_preregistration", "verify_calibration_evidence", "evaluate_confirmatory_decision",
            "verify_audit_result"} <= called


def test_declared_protocol_constants() -> None:
    assert R.EXPECTED_SPEC == C.CANONICAL_SPEC and R.EXPECTED_SPEC["num_rounds"] == 5
    assert R.EXPECTED_SPEC["input_difference"] == {"left_word": "0x0040", "right_word": "0x0000"}
    assert R.PRODUCTION_SIZES == {"train": 10_000_000, "validation": 1_000_000, "test": 1_000_000}
    assert R.REQUIRED_ROLES == {"R1": "confirmatory", "R2": "confirmatory", "A": "confirmatory",
                                "B": "confirmatory", "C": "confirmatory", "D": "confirmatory", "E": "exploratory"}
    assert R.CONFIRMATORY_FAMILIES == ("R1", "R2", "A", "B", "C", "D")
    assert R.EXPECTED_ALLOCATION == {f: 0.005 / 6 for f in R.CONFIRMATORY_FAMILIES}
    assert R.FEATURE_GROUPS == {"C0": list(range(32)), "C1": list(range(32, 64))}
    assert R.CALIBRATION_TOOL_PATH.name == "d3_gohr_method_validity_calibration.py"
    assert R.CALIBRATION_SUITE_PATH.name == "test_d3_method_validity.py"
    assert R.EXPLORATORY_ALPHA == 0.01 and R.GLOBAL_ALPHA == 0.01


def test_feature_layout_matches_the_generator_on_real_rows(instance) -> None:
    """C0 = columns 0-31: within-half pairs are uniform in both classes, as declared."""
    x, y = C.load_partition(instance, "train", spec=SMALL_SPEC)
    assert x.shape[1] == 64 and set(np.unique(x)) <= {0, 1}
    import speck as sp
    words = sp.convert_to_binary([np.array([0x8000], dtype=np.uint16), np.array([0], dtype=np.uint16),
                                  np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16)])
    assert words[0, 0] == 1 and words[0, 63] == 1 and words[0].sum() == 2


def test_generator_source_verifies_against_the_pin() -> None:
    rec = R.verify_generator_source()
    assert rec["generator_pin"] == ATT.GENERATOR_PIN
    assert rec["imported_generator"]["resolved_file_is_pinned_path"] is True


def test_wrong_generator_digest_is_refused(monkeypatch) -> None:
    monkeypatch.setitem(ATT.GENERATOR_PIN, "sha256", "0" * 64)
    with pytest.raises(SystemExit, match="not the pinned source"):
        R.verify_generator_source()


def test_wrong_imported_module_path_is_refused(monkeypatch) -> None:
    real = ATT.generator_module_record
    monkeypatch.setattr(ATT, "generator_module_record",
                        lambda repo: dict(real(repo), code_object_file_is_pinned_path=False))
    with pytest.raises(SystemExit, match="not compiled from the repository-root speck.py"):
        R.verify_generator_source()


def test_cited_generator_lines_are_present(monkeypatch) -> None:
    monkeypatch.setattr(R, "GENERATOR_SOURCE_LINES", R.GENERATOR_SOURCE_LINES + ("this line is not in speck.py",))
    with pytest.raises(SystemExit, match="no longer contains"):
        R.verify_generator_source()


# ---------------------------------------------------------------------------
# End to end (synthetic-scale fixtures)
# ---------------------------------------------------------------------------

def test_confirmatory_run_end_to_end_records_a_bound_chain(instance, tmp_path, small) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    assert _run(instance, out, path, sha, cal_dir) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([R.CERTIFICATE_NAME, R.AUDIT_FILE_NAME,
                                                            R.RUN_MANIFEST_NAME])
    cert = json.loads((out / R.CERTIFICATE_NAME).read_text())
    dec, chain = cert["decision"], cert["chain"]
    assert dec["outcome_overridden_because"] == []
    assert dec["gating_families"] == ["R1", "R2", "A", "B", "C", "D"] and dec["non_gating_families"] == ["E"]
    assert dec["family_outcomes"]["E"] == "NOT_GATED" and dec["family_outcomes"]["R1"] != "NOT_GATED"
    assert dec["guarantees"]["alpha_total"] == pytest.approx(0.005) and dec["guarantees"]["joint_error_bound"] == pytest.approx(0.01)
    assert cert["configuration"]["tolerances"] == R.derived_tolerances()
    assert cert["configuration"]["family_roles"] == R.REQUIRED_ROLES
    assert cert["configuration"]["family_alpha_allocation"] == R.EXPECTED_ALLOCATION
    # Chain: preregistration -> calibration -> D0 -> attestation -> code -> seal -> manifest.
    assert chain["preregistration"]["sha256"] == sha and chain["preregistration"]["commit"]
    assert chain["calibration"]["run_id"] == "fixture-run" and chain["calibration"]["tool_sha256"] == GOHR_TOOL_SHA
    man = C.load_manifest(instance)
    assert chain["d0_instance"]["canonical_instance_id"] == man["canonical_instance_id"]
    assert chain["d0_instance"]["manifest_sha256"] == hashlib.sha256((instance / C.MANIFEST_NAME).read_bytes()).hexdigest()
    assert chain["build_attestation"]["sha256"] == hashlib.sha256((instance / ATT.ATTESTATION_NAME).read_bytes()).hexdigest()
    assert cert["code_identity"]["engine"]["sha256"] == d3.engine_sha256()
    assert cert["code_identity"]["generator"]["sha256"] == ATT.GENERATOR_PIN["sha256"]
    sealed = (out / R.AUDIT_FILE_NAME).read_bytes()
    seal = hashlib.sha256(sealed).hexdigest()
    assert seal == chain["audit_seal_sha256"] == dec["audit_seal_sha256"]
    assert json.loads(sealed)["input_provenance"] == cert["loaded_partition_hashes"] == cert["audited_input_provenance"]
    rm = json.loads((out / R.RUN_MANIFEST_NAME).read_text())
    assert rm["files"][R.AUDIT_FILE_NAME] == seal
    assert rm["files"][R.CERTIFICATE_NAME] == hashlib.sha256((out / R.CERTIFICATE_NAME).read_bytes()).hexdigest()
    assert rm["preregistration_sha256"] == sha and rm["build_attestation_sha256"] == chain["build_attestation"]["sha256"]
    d1_cert_sha = hashlib.sha256((D1_EVIDENCE["dir"] / D1R.CERTIFICATE_NAME).read_bytes()).hexdigest()
    assert chain["d1_census"]["outcome"] == "PASS" and chain["d1_census"]["certificate_sha256"] == d1_cert_sha
    assert chain["d1_census"] == R.verify_d1_evidence(D1_EVIDENCE["dir"], instance)
    assert rm["d1_certificate_sha256"] == d1_cert_sha
    for phrase in ("not a mathematical conversion", "not validated practical-equivalence margins",
                   "conditional on i.i.d. rows", "PRACTICAL_UNSUPPORTED is not a deviation", "E is exploratory"):
        assert phrase in cert["interpretation"]
    own = json.dumps({k: v for k, v in cert.items() if k != "decision"})
    assert "/Users/" not in own and str(Path.home()) not in own, "the runner's own fields carry no home path"
    assert not list(tmp_path.glob(".d3_gohr_staging_*"))


def test_preflight_only_runs_every_gate_and_writes_nothing(instance, tmp_path, small, blocking_spy, capsys) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    assert _run(instance, out, path, sha, cal_dir, "--preflight-only") == 0
    assert blocking_spy.calls == 0 and "PREFLIGHT PASSED" in capsys.readouterr().out
    _no_output(out)


# ---------------------------------------------------------------------------
# Refusals: nothing written, engine never reached
# ---------------------------------------------------------------------------

def test_existing_output_directory_is_refused_and_untouched(instance, tmp_path, small, blocking_spy) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("existing evidence")
    with pytest.raises(SystemExit, match="exists"):
        _run(instance, out, path, sha, cal_dir)
    assert [p.name for p in out.iterdir()] == ["keep.txt"] and blocking_spy.calls == 0


@pytest.mark.parametrize("case", ["unapproved", "uncommitted", "wrong_anchor", "edited_after_commit"])
def test_preregistration_refusals(instance, tmp_path, small, blocking_spy, case) -> None:
    rec = _prereg(approved=False) if case == "unapproved" else _prereg()
    path, sha, cal_dir = _gates(tmp_path, prereg=rec, commit=(case != "uncommitted"))
    if case == "wrong_anchor":
        sha = "0" * 64
    if case == "edited_after_commit":
        path.write_text(path.read_text().replace("Fixture Approver", "Someone Else"))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="preregistration rejected"):
        _run(instance, out, path, sha, cal_dir)
    _no_output(out)
    assert blocking_spy.calls == 0


@pytest.mark.parametrize("over, message", [
    ({"tolerances": {"class_ratio": 0.00144, "class_conditional_bit": 0.00288, "pairwise_tvd": 0.00144,
                     "cross_class_bit": 0.00288}}, "II-4-derived"),
    ({"family_roles": {**R.REQUIRED_ROLES, "R1": "exploratory"},
      "family_alpha_allocation": {f: 0.001 for f in ("R2", "A", "B", "C", "D")}}, "family roles"),
    ({"family_alpha_allocation": {"R1": 0.0025, "R2": 0.0005, "A": 0.0005, "B": 0.0005, "C": 0.0005,
                                  "D": 0.0005}}, "equal six-family split"),
])
def test_a_preregistration_that_is_not_this_protocol_is_refused(instance, tmp_path, small, blocking_spy,
                                                                 over, message) -> None:
    path, sha, cal_dir = _gates(tmp_path, prereg=_prereg(**over))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        _run(instance, out, path, sha, cal_dir)
    _no_output(out)
    assert blocking_spy.calls == 0


@pytest.mark.parametrize("cal, message", [
    ({"tool_sha": hashlib.sha256((R.PACKAGE_DIR / "tools" / "d3_method_validity_calibration.py").read_bytes()).hexdigest()},
     "calibration rejected"),
    ({"alloc": {f: 0.001 for f in ("R2", "A", "B", "C", "D")}}, "family_alpha_allocation"),
    ({"roles": {**{f: "confirmatory" for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}, "R1": "exploratory"}},
     "family_roles"),
    ({"sizes": {"train": 10_000_000, "validation": 1_000_000, "test": 1_000_000}}, "partition_sizes"),
    ({"supported": (0.4, 0.6)}, "supported_class_fraction"),
    ({"extra": lambda ev: ev["identity"].update(engine_sha256="0" * 64)}, "calibration rejected"),
    ({"extra": lambda ev: ev["S0_unit_suite"].update(suite_sha256="0" * 64)}, "calibration rejected"),
    ({"extra": lambda ev: ev.update(status="NOT_CONSISTENT")}, "calibration rejected"),
])
def test_incompatible_or_stale_calibration_is_refused(instance, tmp_path, small, blocking_spy, cal, message) -> None:
    path, sha, cal_dir = _gates(tmp_path, **cal)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        _run(instance, out, path, sha, cal_dir)
    _no_output(out)
    assert blocking_spy.calls == 0


D1_FAILURES = [
    ("missing", "is missing; the frozen D1 gate is not met"),
    ("outcome_fail", "requires D1 outcome PASS"),
    ("other_instance", "names a different canonical instance"),
    ("other_manifest", "names a different D0 manifest"),
    ("other_attestation", "names a different build attestation"),
    ("certificate_not_hashed", "does not hash the certificate"),
    ("incomplete", "does not hash the certificate"),
    ("extra_file", "must hold exactly the D1 certificate and run manifest"),
    ("legacy_certificate", "not written by the instance-bound Gohr D1 runner"),
]


def _bad_d1(kind, tmp_path) -> Path:
    if kind == "missing":
        return tmp_path / "no_d1_here"
    if kind == "outcome_fail":
        return _d1_copy(tmp_path, mutate_cert=lambda c: c["verdict"].update(outcome="FAIL"))
    if kind == "other_instance":
        return _d1_copy(tmp_path, mutate_cert=lambda c: c["canonical_instance"].update(canonical_instance_id="0" * 64),
                        mutate_manifest=lambda m: m.update(canonical_instance_id="0" * 64))
    if kind == "other_manifest":
        return _d1_copy(tmp_path, mutate_cert=lambda c: c["canonical_instance"].update(manifest_sha256="0" * 64),
                        mutate_manifest=lambda m: m.update(d0_manifest_sha256="0" * 64))
    if kind == "other_attestation":
        return _d1_copy(tmp_path, mutate_cert=lambda c: c["build_attestation"].update(sha256="0" * 64),
                        mutate_manifest=lambda m: m.update(build_attestation_sha256="0" * 64))
    if kind == "certificate_not_hashed":
        return _d1_copy(tmp_path, mutate_cert=lambda c: c.update(note="edited"), rehash=False)
    if kind == "incomplete":
        return _d1_copy(tmp_path, mutate_manifest=lambda m: m.update(run_status="RUNNING"))
    if kind == "extra_file":
        d = _d1_copy(tmp_path)
        (d / "extra.json").write_text("{}")
        return d
    if kind == "legacy_certificate":
        return _d1_copy(tmp_path, mutate_cert=lambda c: c.pop("runner"))
    raise AssertionError(kind)


@pytest.mark.parametrize("kind, message", D1_FAILURES)
def test_d1_gate_refuses_production_without_bound_passing_d1(instance, tmp_path, small, blocking_spy,
                                                             kind, message) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        _run(instance, out, path, sha, cal_dir, d1=_bad_d1(kind, tmp_path))
    _no_output(out)
    assert blocking_spy.calls == 0


@pytest.mark.parametrize("kind, message", [D1_FAILURES[0], D1_FAILURES[1], D1_FAILURES[2]])
def test_d1_gate_refuses_preflight_too(instance, tmp_path, small, blocking_spy, kind, message) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        _run(instance, out, path, sha, cal_dir, "--preflight-only", d1=_bad_d1(kind, tmp_path))
    _no_output(out)


def test_d1_dir_is_a_required_argument(instance, tmp_path, small) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    with pytest.raises(SystemExit):
        R.main(["--instance-dir", str(instance), "--out-dir", str(tmp_path / "out"), "--preregistration", str(path),
                "--preregistration-sha256", sha, "--calibration-dir", str(cal_dir)])
    _no_output(tmp_path / "out")


def test_production_sizes_are_enforced(instance, tmp_path, blocking_spy) -> None:
    """Without the test-only size reduction, the small fixture instance is not the frozen configuration."""
    path, sha, cal_dir = _gates(tmp_path, sizes=R.PRODUCTION_SIZES)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not the frozen Gohr D0 configuration"):
        _run(instance, out, path, sha, cal_dir)
    _no_output(out)


def _reseal(d: Path, mutate) -> None:
    man = json.loads((d / C.MANIFEST_NAME).read_text())
    mutate(man)
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(man["specification"])
    (d / C.MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m["specification"].update(num_rounds=6), "num_rounds"),
    (lambda m: m["specification"].update(input_difference={"left_word": "0x0080", "right_word": "0x0000"}),
     "input_difference"),
    (lambda m: m["specification"].update(target="autond"), "target"),
])
def test_wrong_specification_is_refused(instance, tmp_path, small, blocking_spy, mutate, message) -> None:
    d = _copy(instance, tmp_path)
    _reseal(d, mutate)
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        _run(d, out, path, sha, cal_dir)
    _no_output(out)


def test_missing_attestation_is_refused(instance, tmp_path, small, blocking_spy) -> None:
    d = _copy(instance, tmp_path)
    (d / ATT.ATTESTATION_NAME).unlink()
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="no build attestation"):
        _run(d, out, path, sha, cal_dir)
    _no_output(out)


def test_mismatched_attestation_is_refused(instance, tmp_path, small, blocking_spy) -> None:
    d = _copy(instance, tmp_path)
    att = json.loads((d / ATT.ATTESTATION_NAME).read_text())
    att["partitions"]["validation"]["labels_sha256"] = "0" * 64
    (d / ATT.ATTESTATION_NAME).write_text(json.dumps(att))
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="partition validation hashes differ"):
        _run(d, out, path, sha, cal_dir)
    _no_output(out)


def test_tampered_partition_file_is_refused(instance, tmp_path, small, blocking_spy) -> None:
    d = _copy(instance, tmp_path)
    p = d / "test.npz"
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF
    p.write_bytes(bytes(data))
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        _run(d, out, path, sha, cal_dir)
    _no_output(out)
    assert blocking_spy.calls == 0


def test_class_fraction_outside_the_calibrated_range_is_refused(instance, tmp_path, small, blocking_spy,
                                                                monkeypatch) -> None:
    # A VALID calibrated range (it must contain 1/2) chosen to exclude this fixture's train fraction.
    man = C.load_manifest(instance)
    f = man["partitions"]["train"]["label_counts"]["positive"] / SMALL["train"]
    if f == 0.5:
        pytest.skip("fixture train fraction is exactly 1/2; no valid range excludes it")
    rng = (0.5, 0.9) if f < 0.5 else (0.1, 0.5)
    monkeypatch.setattr(R, "SUPPORTED_CLASS_FRACTION", list(rng))
    path, sha, cal_dir = _gates(tmp_path, supported=rng)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="class fraction outside the calibrated range"):
        _run(instance, out, path, sha, cal_dir)
    _no_output(out)


def test_a_failure_while_writing_leaves_no_partial_evidence(instance, tmp_path, small, monkeypatch) -> None:
    path, sha, cal_dir = _gates(tmp_path)
    out = tmp_path / "out"

    def boom(*a, **k):
        raise OSError("simulated rename failure")
    monkeypatch.setattr(R.os, "rename", boom)
    with pytest.raises(OSError, match="simulated"):
        _run(instance, out, path, sha, cal_dir)
    _no_output(out)


# ---------------------------------------------------------------------------
# Regression protection: nothing outside the new files changed
# ---------------------------------------------------------------------------

#: SHA-256 of files this work must not change (recorded before it began, 2026-10-23).
PROTECTED = {
    "audit/dataset/d3_distribution_statistics.py": "6d1ebe9e1857318642f1a022d18289078f5eaeefb3a5d9268e9d9b36275bfff7",
    "audit/dataset/experiments/d3_autond_distribution.py": "30b94415b11707633228cc3b17c5769e5b2ce0b1b30c6ebbf5fdab2e04b8bb0d",
    "audit/dataset/experiments/d3_gohr_distribution_statistics.py": "095ae3255e844ae4ac4a5025b86df32af706ac98a5e0ad19b0d3c70931cf1e4c",
    "audit/dataset/tools/d3_method_validity_calibration.py": "3f9b46ca278658646a3ace4f0423647db0aba3683a661ee03636abe4e7ae8e04",
    "audit/dataset/tests/test_d3_method_validity.py": "808dedc1aeb0b32ab8dc30fc72710e5c867bb01f2c241be36a931df300a29d32",
    "audit/dataset/canonical.py": "7fc650f5b451850cd461761b0ac117d7052707dd9fec1c045cab4d754b73ea1d",
    "audit/dataset/adapters/gohr.py": "3c0b7ac5327c68233ac792062f3559232cfc965b8a403312e552bcb2b516b488",
    "speck.py": "59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce",
    "audit/targets/autond/preregistrations/d3_autond_speck3264_r8.json": "cbd56c4aed8a9304b8ef5600948aa9c1cf456aefa7e623b10d4c1c1be5e3d114",
    "audit/dataset/evidence/d3/d3_gohr_10000000_1000000_1000000_5r.json": "757e62bfebc237f806134ba0e036ee23797048fe891b58a17cb52ce435633ed2",
    "audit/dataset/evidence/d3/d3_gohr_distribution_consistency_10000000_1000000_1000000_5r.json": "8b59754008e6a5e8e760f6e3361316d0c60d128066a2dd6d5aaa5a7dca485e8d",
    "audit/dataset/evidence/d3_method_validity_20261022_c39726e5/d3_method_validity.json": "64deedb75fb1a070c0c79e261992bd42b260ae096332a7d1e1ccf92a73f15e1b",
    "audit/dataset/evidence/d3_method_validity_20261022_c39726e5/d3_method_validity_run_manifest.json": "8f1188cf691935edcf87bff1aae01880b373a05f05e9d7b158cfcae26231a126",
    "audit/dataset/evidence/d3_autond_speck3264_r8_confirmatory_6974fba6/d3_autond_sealed_audit.json": "3b802eeafd73f7da90650447abed658a3fc8d99dca3be76253f8733d37e375fd",
    "audit/dataset/evidence/d3_autond_speck3264_r8_confirmatory_6974fba6/d3_autond_certificate.json": "cd0317f578e78ea1b67ba0f692a51065a81e3b5035f09cc572b48c10bb6fda31",
}


@pytest.mark.parametrize("rel", sorted(PROTECTED))
def test_protected_file_is_unchanged(rel) -> None:
    p = R.REPO_ROOT / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present in this checkout (SKIPPED IS NOT PASSED)")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == PROTECTED[rel]
