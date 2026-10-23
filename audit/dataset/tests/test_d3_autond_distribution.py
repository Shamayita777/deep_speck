"""Integration tests for experiments/d3_autond_distribution.py (the AutoND D3 runner).

The complete path is exercised end to end on a SMALL REAL canonical D0
instance, built through the real D0 builder and the pinned generator (the
D2 runner's test convention). Nothing here reads or writes the production
instance. Tests that need the pinned AutoND checkout are SKIPPED (reported,
not passed) where it is absent.

Covered: target and configuration correctness, D0 integrity and provenance
binding, use of the generic confirmatory engine only (never the legacy path),
fail-closed behaviour (refusals before the engine is reached, no partial or
overwritten output), blinded integration runs that record no statistics, and
confirmatory runs whose gates use real preregistration and calibration
bindings (fixtures: a throwaway git repository and fabricated, internally
consistent calibration evidence in tmp_path).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from audit.dataset import canonical as C
from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.adapters.autond import AutoNDDatasetAdapter
from audit.dataset.common.provenance import array_sha256
from audit.dataset.experiments import d3_autond_distribution as R
from audit.dataset.tests import test_d3_method_validity as MV

ADAPTERS = Path(R.__file__).resolve().parents[1] / "adapters"


def _pinned_root() -> Path:
    for cand in (os.environ.get("AUTOND_ROOT"), "~/Documents/AutoND-upstream"):
        if cand and (Path(cand).expanduser() / "main.py").is_file():
            return Path(cand).expanduser().resolve()
    pytest.skip("pinned AutoND checkout not found; set AUTOND_ROOT. SKIPPED IS NOT PASSED.")


@pytest.fixture(scope="module")
def root() -> Path:
    return _pinned_root()


@pytest.fixture(scope="module")
def instance(root, tmp_path_factory):
    """A small REAL canonical D0 instance, built through the real builder."""
    import argparse
    from audit.dataset.experiments import d0_autond_canonical as D0
    out = tmp_path_factory.mktemp("d0") / "instance"
    ns = argparse.Namespace(autond_root=str(root), num_rounds=8, cipher=None, delta_plain_hex=None,
                            train_samples=8000, validation_samples=4000, test_samples=4000,
                            out_dir=out, repo_root=Path("."))
    man = C.build(out, repo_root=Path("."), spec=D0.build_spec(ns), log=lambda *a, **k: None)
    return out, man


def _copy(instance, tmp_path) -> Path:
    dst = tmp_path / "instance_copy"
    shutil.copytree(instance[0], dst)
    return dst


def _reseal(d: Path, mutate) -> None:
    man = json.loads((d / C.MANIFEST_NAME).read_text())
    mutate(man)
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(man["specification"])
    (d / C.MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))


class Spy:
    def __init__(self, passthrough: bool):
        self.calls, self.kwargs, self.passthrough = 0, None, passthrough

    def __call__(self, partitions, **kw):
        self.calls += 1
        self.partitions, self.kwargs = partitions, kw
        if not self.passthrough:
            raise AssertionError("the D3 engine was invoked")
        return d3.class_stratified_audit(partitions, **kw)


@pytest.fixture
def blocking_spy(monkeypatch):
    s = Spy(passthrough=False)
    monkeypatch.setattr(R, "D3_ENGINE", s)
    return s


@pytest.fixture
def recording_spy(monkeypatch):
    s = Spy(passthrough=True)
    monkeypatch.setattr(R, "D3_ENGINE", s)
    return s


def _blinded(inst_dir, out):
    return R.main(["--mode", "blinded-integration", "--instance-dir", str(inst_dir), "--out-dir", str(out)])


def _cert(out) -> dict:
    return json.loads((out / R.CERTIFICATE_NAME).read_text())


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def test_autond_has_exactly_the_two_established_adapters() -> None:
    assert sorted(p.name for p in ADAPTERS.glob("autond*.py")) == ["autond.py", "autond_d2.py"]


LEGACY_NAMES = {"audit_distribution", "evaluate_decision", "compare_pairwise_joint_distributions",
                "compare_partitions", "summarize_partition", "compare_with_intended_distribution",
                "print_report", "save_json", "class_balance_test"}
CONFIRMATORY_API = {"class_stratified_audit", "evaluate_confirmatory_decision", "verify_audit_result",
                    "verify_preregistration", "verify_calibration_evidence", "validate_preregistration_content",
                    "allocate_familywise_alpha", "TOLERANCE_KEYS", "CLASS_STRATIFIED_CONFIRMATORY_FAMILIES",
                    "PreregistrationRejected", "CalibrationRejected", "D3_METHOD_VERSION",
                    "DECISION_RULE_VERSION", "engine_sha256", "joint_alpha_problem", "GUARANTEE_COUNT"}


def test_runner_uses_only_the_generic_confirmatory_engine() -> None:
    tree = ast.parse(Path(R.__file__).read_text())
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "d3"}
    assert used <= CONFIRMATORY_API, used - CONFIRMATORY_API
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | used
    assert not names & LEGACY_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "d3_distribution_statistics" in node.module:
            pytest.fail("the runner must use the engine through the module, not import names from it")
    assert R.D3_ENGINE is d3.class_stratified_audit
    assert R.ADAPTER_FACTORY is AutoNDDatasetAdapter


def test_runner_has_no_default_tolerances() -> None:
    """Candidate tolerances are documentation; main() never calls the derivation."""
    tree = ast.parse(Path(R.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "margin_derived_tolerance_candidates" not in called


# ---------------------------------------------------------------------------
# Blinded integration run on a real D0 instance
# ---------------------------------------------------------------------------

def test_blinded_run_passes_autond_configuration_and_the_d0_arrays(instance, tmp_path, recording_spy) -> None:
    out = tmp_path / "out"
    assert _blinded(instance[0], out) == 0
    kw, parts = recording_spy.kwargs, recording_spy.partitions
    assert kw["expected_class_ratio"] == 0.5 and kw["expected_bit_probability"] == {0: 0.5, 1: 0.5}
    assert kw["feature_groups"] == {"C0": list(range(32)), "C1": list(range(32, 64))}
    assert kw["tolerances"] == {k: None for k in d3.TOLERANCE_KEYS} and kw["mc_seed"] == R.MC_SEED
    assert kw["family_roles"] == RUNNER_ROLES and kw["family_alpha_allocation"] == RUNNER_ALLOC
    assert kw["exploratory_alpha"] == 0.01 and d3.joint_alpha_problem(0.01, kw["family_alpha_allocation"]) is None
    man = instance[1]
    for n in C.PARTITIONS:
        x, y = parts[n]
        assert array_sha256(x) == man["partitions"][n]["features_sha256"]
        assert array_sha256(y) == man["partitions"][n]["labels_sha256"]


def test_blinded_certificate_records_provenance_and_no_statistics(instance, tmp_path) -> None:
    out = tmp_path / "out"
    assert _blinded(instance[0], out) == 0
    cert = _cert(out)
    text = (out / R.CERTIFICATE_NAME).read_text()
    for leak in ('"p_value"', '"tvd', '"category', '"outcome"', 'provisional', '"significant"',
                 '"absolute_difference', '"decision"'):
        assert leak not in text, leak
    assert cert["mode"] == "blinded-integration" and cert["blinded"] is True
    assert cert["audit"]["method_version"] == d3.D3_METHOD_VERSION
    man = instance[1]
    want = {n: {"features_sha256": man["partitions"][n]["features_sha256"],
                "labels_sha256": man["partitions"][n]["labels_sha256"]} for n in C.PARTITIONS}
    assert cert["audited_input_provenance"] == want == cert["loaded_partition_hashes"]
    assert cert["canonical_instance"]["canonical_instance_id"] == man["canonical_instance_id"]
    assert cert["adapter"]["class"].endswith("AutoNDDatasetAdapter") and cert["adapter"]["dataset_id"] == "autond-speck3264"
    assert [f["family_id"] for f in cert["families"]] == ["R1", "R2", "A", "B", "C", "D", "E"]
    assert {f["family_id"]: f["role"] for f in cert["families"]} == RUNNER_ROLES
    assert cert["code_identity"]["engine_sha256"] == d3.engine_sha256()
    assert not (out / R.AUDIT_FILE_NAME).exists()       # sealed results are not persisted when blinded
    rm = json.loads((out / R.RUN_MANIFEST_NAME).read_text())
    assert rm["run_status"] == "COMPLETE" and set(rm["files"]) == {R.CERTIFICATE_NAME}
    assert rm["files"][R.CERTIFICATE_NAME] == hashlib.sha256((out / R.CERTIFICATE_NAME).read_bytes()).hexdigest()
    assert not list(tmp_path.glob(".d3_autond_staging_*"))


# ---------------------------------------------------------------------------
# Confirmatory runs with real gate bindings (fixtures)
# ---------------------------------------------------------------------------

#: The runner's production roles and joint allocation (0.001 per confirmatory family).
RUNNER_ROLES = {**R.REQUIRED_ROLES, "E": "exploratory"}
RUNNER_ALLOC = d3.allocate_familywise_alpha(0.01 / d3.GUARANTEE_COUNT, R.CONFIRMATORY_FAMILIES)


def _runner_prereg(**over):
    return MV._prereg_record(**{"family_alpha_allocation": dict(RUNNER_ALLOC),
                                "family_roles": dict(RUNNER_ROLES), **over})


def _gates(tmp_path, instance, *, sizes=None):
    repo = MV._git_repo(tmp_path)
    path, sha = MV._commit_prereg(repo, _runner_prereg())
    design = {"partition_sizes": sizes or {n: instance[1]["partitions"][n]["samples"] for n in C.PARTITIONS},
              "feature_dimension": 64}
    cal_dir = MV._write_cal(tmp_path / "cal", MV._evidence({"design": design,
                                                           "configuration": {"family_alpha_allocation": RUNNER_ALLOC}}))
    return path, sha, cal_dir


def _confirmatory(inst_dir, out, path, sha, cal_dir):
    return R.main(["--mode", "confirmatory", "--instance-dir", str(inst_dir), "--out-dir", str(out),
                   "--preregistration", str(path), "--preregistration-sha256", sha,
                   "--calibration-dir", str(cal_dir)])


def test_confirmatory_run_with_satisfied_gates_records_a_bound_decision(instance, tmp_path) -> None:
    path, sha, cal_dir = _gates(tmp_path, instance)
    out = tmp_path / "out"
    assert _confirmatory(instance[0], out, path, sha, cal_dir) == 0
    cert = _cert(out)
    dec = cert["decision"]
    assert dec["outcome_overridden_because"] == [] and dec["outcome"] in {"PASS", "CONDITIONAL_PASS", "INCONCLUSIVE", "FAIL"}
    assert dec["method_version"] == d3.D3_METHOD_VERSION
    assert cert["configuration"]["tolerances"] == MV.WIDE_TOL
    assert cert["configuration"]["family_alpha_allocation"] == RUNNER_ALLOC
    assert cert["configuration"]["family_roles"] == RUNNER_ROLES
    assert dec["gating_families"] == list(R.CONFIRMATORY_FAMILIES) and set(dec["non_gating_families"]) == {"R1", "E"}
    assert dec["family_outcomes"]["R1"] == "NOT_GATED"
    assert dec["guarantees"]["joint_error_bound"] == pytest.approx(0.01)
    assert "PASS or equivalence claim is not made" in cert["interpretation"]
    assert cert["preregistration"]["sha256"] == sha and cert["preregistration"]["commit"]
    assert cert["calibration"]["run_id"] == "fixture-run"
    sealed = (out / R.AUDIT_FILE_NAME).read_bytes()
    assert hashlib.sha256(sealed).hexdigest() == cert["audit_seal_sha256"] == dec["audit_seal_sha256"]
    assert json.loads(sealed)["input_provenance"] == cert["loaded_partition_hashes"]
    rm = json.loads((out / R.RUN_MANIFEST_NAME).read_text())
    assert rm["files"][R.AUDIT_FILE_NAME] == cert["audit_seal_sha256"]


def test_calibration_for_a_different_design_cannot_authorise_the_decision(instance, tmp_path) -> None:
    path, sha, cal_dir = _gates(tmp_path, instance,
                                sizes={"train": 10_000_000, "validation": 1_000_000, "test": 1_000_000})
    out = tmp_path / "out"
    assert _confirmatory(instance[0], out, path, sha, cal_dir) == 0
    dec = _cert(out)["decision"]
    assert dec["outcome"] == "INCONCLUSIVE" and any("partition sizes" in r for r in dec["outcome_overridden_because"])


def test_confirmatory_gates_are_verified_before_any_data_is_loaded(instance, tmp_path, blocking_spy, monkeypatch) -> None:
    path, sha, cal_dir = _gates(tmp_path, instance)
    out = tmp_path / "out"
    loads = []
    monkeypatch.setattr(R, "load_partitions", lambda *a, **k: loads.append(1) or (_ for _ in ()).throw(AssertionError))
    for args in ([], ["--preregistration", str(path)],
                 ["--preregistration", str(path), "--preregistration-sha256", "0" * 64, "--calibration-dir", str(cal_dir)],
                 ["--preregistration", str(path), "--preregistration-sha256", sha, "--calibration-dir", str(tmp_path / "none")]):
        with pytest.raises(SystemExit):
            R.main(["--mode", "confirmatory", "--instance-dir", str(instance[0]), "--out-dir", str(out), *args])
    stale = MV._write_cal(tmp_path / "stale", MV._evidence(
        {"design": {"partition_sizes": {"train": 8000, "validation": 4000, "test": 4000}, "feature_dimension": 64},
         "configuration": {"family_alpha_allocation": RUNNER_ALLOC}},
        mutate=lambda e: e["identity"].__setitem__("engine_sha256", "0" * 64)))
    with pytest.raises(SystemExit):
        _confirmatory(instance[0], out, path, sha, stale)
    assert blocking_spy.calls == 0 and not loads and not out.exists()


def test_blinded_mode_refuses_gate_arguments(instance, tmp_path, blocking_spy) -> None:
    with pytest.raises(SystemExit):
        R.main(["--mode", "blinded-integration", "--instance-dir", str(instance[0]),
                "--out-dir", str(tmp_path / "o"), "--calibration-dir", str(tmp_path)])
    assert blocking_spy.calls == 0


def test_mode_is_required(instance, tmp_path) -> None:
    with pytest.raises(SystemExit):
        R.main(["--instance-dir", str(instance[0]), "--out-dir", str(tmp_path / "o")])


# ---------------------------------------------------------------------------
# Fail-closed: integrity, configuration, source pin, output
# ---------------------------------------------------------------------------

def test_existing_output_directory_is_refused_and_left_untouched(instance, tmp_path, blocking_spy) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "prior.json").write_text("{}")
    with pytest.raises(SystemExit):
        _blinded(instance[0], out)
    assert [p.name for p in out.iterdir()] == ["prior.json"] and blocking_spy.calls == 0


def test_tampered_partition_bytes_stop_the_run(instance, tmp_path, blocking_spy) -> None:
    d = _copy(instance, tmp_path)
    f = d / "test.npz"
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(SystemExit):
        _blinded(d, tmp_path / "out")
    assert blocking_spy.calls == 0 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutate", [
    lambda m: m["specification"].__setitem__("num_rounds", 7),
    lambda m: m["specification"].__setitem__("cipher", "simon3264"),
    lambda m: m["specification"].__setitem__("delta_plain_hex", "0x200000"),
    lambda m: m["specification"].__setitem__("source_commit_observed", "0" * 40),
    lambda m: m["specification"]["adapter_provenance"]["source_digests"].__setitem__("main.py", "0" * 64),
    lambda m: m["specification"]["representation"].__setitem__("feature_bits", 63),
])
def test_wrong_configuration_or_source_stops_the_run(instance, tmp_path, blocking_spy, mutate) -> None:
    d = _copy(instance, tmp_path)
    _reseal(d, mutate)
    with pytest.raises(SystemExit):
        _blinded(d, tmp_path / "out")
    assert blocking_spy.calls == 0 and not (tmp_path / "out").exists()


def test_tampered_instance_identity_stops_the_run(instance, tmp_path, blocking_spy) -> None:
    d = _copy(instance, tmp_path)
    man = json.loads((d / C.MANIFEST_NAME).read_text())
    man["canonical_instance_id"] = "0" * 64
    (d / C.MANIFEST_NAME).write_text(json.dumps(man))
    with pytest.raises(SystemExit):
        _blinded(d, tmp_path / "out")
    assert blocking_spy.calls == 0


def test_missing_instance_is_refused(tmp_path, blocking_spy) -> None:
    with pytest.raises(SystemExit):
        _blinded(tmp_path / "absent", tmp_path / "out")
    assert blocking_spy.calls == 0


def test_adapter_disagreeing_with_d0_stops_the_run(instance, tmp_path, blocking_spy, monkeypatch) -> None:
    class Wrong:
        def __init__(self, **kw):
            self.dataset_id, self.feature_bits, self.half_width = "autond-simon3264", 64, 32
            self.source_digests = dict(R.AUTOND_SOURCE_PIN["source_digests"])
    monkeypatch.setattr(R, "ADAPTER_FACTORY", Wrong)
    with pytest.raises(SystemExit):
        _blinded(instance[0], tmp_path / "out")
    assert blocking_spy.calls == 0


def test_preflight_constructs_the_real_pinned_adapter(instance) -> None:
    pre = R.preflight(instance[0], log=lambda *a, **k: None)
    assert isinstance(pre["adapter"], AutoNDDatasetAdapter)
    assert pre["adapter"].source_digests["main.py"] == R.AUTOND_SOURCE_PIN["source_digests"]["main.py"]


# ---------------------------------------------------------------------------
# Target declarations (moved from the retired adapters/autond_d3.py)
# ---------------------------------------------------------------------------

def _pinned_copy(root, tmp_path) -> Path:
    d = tmp_path / "autond"
    (d / "ciphers").mkdir(parents=True)
    shutil.copy(root / "main.py", d / "main.py")
    shutil.copy(root / "ciphers" / "speck3264.py", d / "ciphers" / "speck3264.py")
    return d


def test_generator_source_binding_accepts_the_pin(root, tmp_path) -> None:
    rec = R.verify_generator_source(_pinned_copy(root, tmp_path))
    assert rec["source_digests_verified"]["main.py"] == R.AUTOND_SOURCE_PIN["source_digests"]["main.py"]


def test_generator_source_binding_refuses_edits_and_missing_files(root, tmp_path) -> None:
    d = _pinned_copy(root, tmp_path)
    (d / "main.py").write_text((d / "main.py").read_text() + "\n# edit\n")
    with pytest.raises(SystemExit):
        R.verify_generator_source(d)
    d2 = _pinned_copy(root, tmp_path / "b")
    (d2 / "ciphers" / "speck3264.py").unlink()
    with pytest.raises(SystemExit):
        R.verify_generator_source(d2)


def test_generator_source_binding_refuses_a_stale_cited_line(root, tmp_path, monkeypatch) -> None:
    d = _pinned_copy(root, tmp_path)
    text = (d / "main.py").read_text().replace("C = np.hstack([C0, C1])", "C = np.hstack([C1, C0])")
    (d / "main.py").write_text(text)
    pin = copy.deepcopy(R.AUTOND_SOURCE_PIN)
    pin["source_digests"]["main.py"] = hashlib.sha256(text.encode()).hexdigest()
    monkeypatch.setattr(R, "AUTOND_SOURCE_PIN", pin)
    with pytest.raises(SystemExit, match="stale"):
        R.verify_generator_source(d)


def test_reference_declaration_is_the_generator_derived_law() -> None:
    assert R.REFERENCE["expected_class_ratio"] == 0.5
    assert R.REFERENCE["expected_bit_probability"] == {0: 0.5, 1: 0.5}
    assert "the 64-bit joint law of a Y=1 row is uniform" in R.REFERENCE["not_claimed"]


def test_published_accuracy_is_strict() -> None:
    from audit.targets.autond.preregistrations.autond_ii_v2 import PUBLISHED_VALUES, equivalence_margin
    v = R.published_accuracy(PUBLISHED_VALUES)
    assert v == PUBLISHED_VALUES["speck3264_round8"]["paper_table_6"]
    assert R.margin_derived_tolerance_candidates()["accuracy_margin"] == equivalence_margin(v)
    for mutate in (lambda pv: pv.pop("speck3264_round8"),
                   lambda pv: pv["speck3264_round8"].pop("paper_table_6"),
                   lambda pv: pv["speck3264_round8"].__setitem__("paper_table_6", "0.5144"),
                   lambda pv: pv["speck3264_round8"].__setitem__("paper_table_6", 0.4),
                   lambda pv: pv["speck3264_round8"].__setitem__("agree", False)):
        pv = copy.deepcopy(PUBLISHED_VALUES)
        mutate(pv)
        with pytest.raises(ValueError):
            R.published_accuracy(pv)


# ---------------------------------------------------------------------------
# Pre-load gate compatibility and output exclusivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("over", [
    dict(family_alpha_allocation={f: 0.0009 for f in RUNNER_ALLOC}),                        # not the calibrated split
    dict(global_familywise_alpha=0.02),                                                     # not the calibrated alpha
    dict(family_roles={**RUNNER_ROLES, "E": "confirmatory"},
         family_alpha_allocation={**{f: 0.005 / 6 for f in RUNNER_ALLOC}, "E": 0.005 / 6}),  # E may not gate
    dict(family_roles=dict(MV.ROLES), family_alpha_allocation=dict(MV.GATE_ALLOC)),         # R1 may not gate here
    dict(tolerances={**MV.WIDE_TOL, "invented": 0.1},
         tolerance_justification={k: "test fixture justification text, not a scientific claim"
                                  for k in [*MV.WIDE_TOL, "invented"]}),                    # unknown tolerance key
])
def test_incompatible_preregistration_is_refused_before_data_load(instance, tmp_path, blocking_spy, monkeypatch, over) -> None:
    repo = MV._git_repo(tmp_path)
    path, sha = MV._commit_prereg(repo, _runner_prereg(**over))
    (tmp_path / "g").mkdir()
    _, _, cal_dir = _gates(tmp_path / "g", instance)
    loads = []
    monkeypatch.setattr(R, "load_partitions", lambda *a, **k: loads.append(1))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="incompatible"):
        _confirmatory(instance[0], out, path, sha, cal_dir)
    assert not loads and blocking_spy.calls == 0 and not out.exists()


PER_GUARANTEE_ALLOC = {f: 0.01 / len(R.CONFIRMATORY_FAMILIES) for f in R.CONFIRMATORY_FAMILIES}   # sums to 0.01


def test_runner_gate_check_applies_the_engine_joint_alpha_rule() -> None:
    """The runner's check and the engine's verifiers use one rule: d3.joint_alpha_problem."""
    content = {"tolerances": dict(MV.WIDE_TOL), "family_alpha_allocation": dict(PER_GUARANTEE_ALLOC),
               "family_roles": dict(RUNNER_ROLES), "global_familywise_alpha": 0.01}
    design = {"global_alpha": 0.01, "family_alpha_allocation": dict(PER_GUARANTEE_ALLOC)}
    with pytest.raises(SystemExit, match="joint alpha"):
        R.check_gate_compatibility(content, design)
    ok = dict(content, family_alpha_allocation=dict(RUNNER_ALLOC))
    R.check_gate_compatibility(ok, dict(design, family_alpha_allocation=dict(RUNNER_ALLOC)))


def test_per_guarantee_split_is_refused_at_preregistration_and_calibration(instance, tmp_path, blocking_spy) -> None:
    """A full 0.01 split per guarantee cannot reach the data through either binding."""
    (tmp_path / "g").mkdir()
    good_path, good_sha, good_cal = _gates(tmp_path / "g", instance)
    out = tmp_path / "out"
    path, sha = MV._commit_prereg(MV._git_repo(tmp_path), _runner_prereg(
        family_alpha_allocation=dict(PER_GUARANTEE_ALLOC)))
    with pytest.raises(SystemExit, match="joint"):
        _confirmatory(instance[0], out, path, sha, good_cal)
    design = {"partition_sizes": {n: instance[1]["partitions"][n]["samples"] for n in C.PARTITIONS},
              "feature_dimension": 64}
    bad_cal = MV._write_cal(tmp_path / "badcal", MV._evidence(
        {"design": design, "configuration": {"family_alpha_allocation": PER_GUARANTEE_ALLOC}}))
    with pytest.raises(SystemExit, match="joint"):
        _confirmatory(instance[0], out, good_path, good_sha, bad_cal)
    assert blocking_spy.calls == 0 and not out.exists()


@pytest.mark.parametrize("prior", [None, "file"])
def test_output_writer_never_replaces_an_existing_directory(tmp_path, prior) -> None:
    out = tmp_path / "out"
    out.mkdir()
    if prior:
        (out / "prior.json").write_text("{}")
    with pytest.raises(SystemExit):
        R._write_atomically(out, {"x.json": b"{}"}, {"run_id": "t"})
    assert sorted(p.name for p in out.iterdir()) == ([] if prior is None else ["prior.json"])
    assert not list(tmp_path.glob(".d3_autond_staging_*"))


def test_output_writer_creates_a_complete_set_or_nothing(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    R._write_atomically(out, {"x.json": b"{}"}, {"run_id": "t"})
    assert sorted(p.name for p in out.iterdir()) == sorted(["x.json", R.RUN_MANIFEST_NAME])
    out2 = tmp_path / "out2"
    real = os.rename
    monkeypatch.setattr(os, "rename", lambda *a: (_ for _ in ()).throw(OSError("simulated failure")))
    with pytest.raises(OSError):
        R._write_atomically(out2, {"x.json": b"{}"}, {"run_id": "t"})
    monkeypatch.setattr(os, "rename", real)
    assert not out2.exists() and not list(tmp_path.glob(".d3_autond_staging_*"))
