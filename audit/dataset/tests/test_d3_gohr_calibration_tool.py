"""The Gohr method-validity calibration tool (audit/dataset/tools/d3_gohr_method_validity_calibration.py).

These tests check the tool's DESIGN and its identity with the AutoND tool. They
do not run the calibration (S0-S8); only a few cheap exact computations are
evaluated at the new per-test alphas as spot checks.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d3_gohr_confirmatory as R
from audit.dataset.tests import test_d3_method_validity as MV
from audit.dataset.tools import d3_gohr_method_validity_calibration as G
from audit.dataset.tools import d3_method_validity_calibration as AUTOND

GOHR_TOOL = Path(G.__file__)
AUTOND_TOOL = Path(AUTOND.__file__)
#: Functions whose bodies legitimately differ (report text and labels; repository-relative paths).
ALLOWED_FUNCTION_DIFFERENCES = {"s5", "main", "s0_unit_suite"}
#: Names the Gohr tool adds (repository-relative path and environment records).
ALLOWED_NEW_NAMES = {"const:REPO_ROOT", "_repo_relative", "_environment_record"}
#: Module-level assignments that legitimately differ.
ALLOWED_CONSTANT_DIFFERENCES = {"ROOT_SEED", "FAMILY_ROLES", "UNAPPROVED_DESIGN_INPUTS"}


def _defs(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    out[f"const:{t.id}"] = ast.dump(node.value, include_attributes=False)
    return out


def test_tool_is_the_autond_tool_except_the_declared_design() -> None:
    a, g = _defs(AUTOND_TOOL), _defs(GOHR_TOOL)
    assert set(g) - set(a) == ALLOWED_NEW_NAMES and set(a) <= set(g), "only the declared names are added"
    differing = {k for k in a if a[k] != g[k]}
    allowed = ALLOWED_FUNCTION_DIFFERENCES | {f"const:{c}" for c in ALLOWED_CONSTANT_DIFFERENCES}
    assert differing <= allowed, f"unexpected differences: {sorted(differing - allowed)}"
    assert {f"const:{c}" for c in ALLOWED_CONSTANT_DIFFERENCES} <= differing


def test_design_is_six_confirmatory_families_with_an_equal_split() -> None:
    assert G.FAMILY_ROLES == {f: "confirmatory" for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
    assert G.CONFIRMATORY == ("R1", "R2", "A", "B", "C", "D")
    assert G.FAMILY_ALPHA_ALLOCATION == {f: 0.005 / 6 for f in G.CONFIRMATORY} == R.EXPECTED_ALLOCATION
    assert sum(G.FAMILY_ALPHA_ALLOCATION.values()) == pytest.approx(0.005, rel=1e-15)
    assert d3.joint_alpha_problem(G.GLOBAL_ALPHA, G.FAMILY_ALPHA_ALLOCATION) is None
    assert G.GLOBAL_ALPHA == R.GLOBAL_ALPHA == 0.01 and G.GUARANTEE_ALPHA == 0.005
    assert G.EXPLORATORY_ALPHA == R.EXPLORATORY_ALPHA == 0.01
    assert G.PRODUCTION_SIZES == R.PRODUCTION_SIZES == {"train": 10_000_000, "validation": 1_000_000,
                                                        "test": 1_000_000}
    assert G.FEATURE_DIMENSION == R.FEATURE_DIMENSION == 64
    assert list(G.SUPPORTED_CLASS_FRACTION) == R.SUPPORTED_CLASS_FRACTION
    assert G.ROOT_SEED != AUTOND.ROOT_SEED
    assert G.SUITE_PATH == R.CALIBRATION_SUITE_PATH and R.CALIBRATION_TOOL_PATH == GOHR_TOOL


def test_per_test_alpha_arithmetic() -> None:
    fam = 0.005 / 6
    assert G.COUNTS == {"R1": 3, "R2": 384, "A": 6048, "B": 6048, "C": 6048, "D": 192, "E": 6048}
    expect_h = {"R1": fam / 3, "R2": fam / 384, "A": fam / 6048, "B": fam / 6048, "C": fam / 6048, "D": fam / 192}
    assert G.PER_HYPOTHESIS_ALPHA == pytest.approx(expect_h, rel=1e-15)
    used = G.ALPHAS_USED
    assert used == d3._alphas_used(G.FAMILY_ALPHA_ALLOCATION, G.COUNTS)
    assert used["binomial"] == pytest.approx(sorted([fam / 384, fam / 3]), rel=1e-15)
    # R2's interval alpha (fam/384) equals D's per-proportion alpha (fam/192/2): one value.
    assert used["clopper_pearson"] == pytest.approx(sorted([fam / 6048 / 8, fam / 384, fam / 3]), rel=1e-15)
    assert used["fisher_cell"] == pytest.approx(sorted([fam / 6048 / 4, fam / 192]), rel=1e-15)
    autond = d3._alphas_used(AUTOND.FAMILY_ALPHA_ALLOCATION, AUTOND.COUNTS)
    assert not set(used["clopper_pearson"]) & set(autond["clopper_pearson"]), "every interval alpha changed"


def test_s1_examines_the_confirmatory_r1_and_r2_alphas() -> None:
    src = (GOHR_TOOL.read_text())
    assert "PER_HYPOTHESIS_ALPHA[fam]" in src and G.PER_HYPOTHESIS_ALPHA["R1"] == pytest.approx(0.005 / 18)


def _fixture_evidence(alloc, used_from=None, *, tool=GOHR_TOOL):
    used = d3._alphas_used(used_from or alloc, d3.expected_hypothesis_counts(3, 64))
    payload = {"design": {"partition_sizes": dict(G.PRODUCTION_SIZES), "feature_dimension": 64},
               "configuration": {"family_alpha_allocation": dict(alloc)}}

    def mutate(ev):
        ev["identity"]["tool_sha256"] = hashlib.sha256(tool.read_bytes()).hexdigest()
        ev["S6_exact_clopper_pearson"]["rows"] = [{"alpha": a, "valid": True} for a in used["clopper_pearson"]]
        ev["S8_exact_fisher"]["rows"] = [{"alpha": a, "valid": True} for a in used["fisher_cell"]]
    return MV._evidence(payload, mutate=mutate)


def test_verifier_accepts_evidence_covering_the_gohr_alphas(tmp_path) -> None:
    d = MV._write_cal(tmp_path / "cal", _fixture_evidence(G.FAMILY_ALPHA_ALLOCATION))
    b = d3.verify_calibration_evidence(d, tool_path=GOHR_TOOL, test_suite_path=G.SUITE_PATH)
    assert b.design()["family_alpha_allocation"] == G.FAMILY_ALPHA_ALLOCATION


def test_evidence_computed_at_the_autond_alphas_cannot_cover_the_gohr_design(tmp_path) -> None:
    d = MV._write_cal(tmp_path / "cal", _fixture_evidence(G.FAMILY_ALPHA_ALLOCATION,
                                                          used_from=AUTOND.FAMILY_ALPHA_ALLOCATION))
    with pytest.raises(d3.CalibrationRejected, match="does not cover every"):
        d3.verify_calibration_evidence(d, tool_path=GOHR_TOOL, test_suite_path=G.SUITE_PATH)


def test_evidence_produced_by_the_autond_tool_is_refused_for_the_gohr_tool(tmp_path) -> None:
    d = MV._write_cal(tmp_path / "cal", _fixture_evidence(G.FAMILY_ALPHA_ALLOCATION, tool=AUTOND_TOOL))
    with pytest.raises(d3.CalibrationRejected, match="tool_sha256"):
        d3.verify_calibration_evidence(d, tool_path=GOHR_TOOL, test_suite_path=G.SUITE_PATH)


def test_real_autond_calibration_is_refused_for_the_gohr_tool() -> None:
    d = MV.SUITE_PATH.parents[1] / "evidence" / "d3_method_validity_20261022_c39726e5"
    if not d.is_dir():
        pytest.skip("AutoND calibration evidence not present (SKIPPED IS NOT PASSED)")
    with pytest.raises(d3.CalibrationRejected, match="tool_sha256"):
        d3.verify_calibration_evidence(d, tool_path=GOHR_TOOL, test_suite_path=G.SUITE_PATH)
    design = json.loads((d / d3.CALIBRATION_EVIDENCE_NAME).read_text())["design"]
    with pytest.raises(SystemExit, match="calibration design"):
        R.check_protocol({"family_roles": R.REQUIRED_ROLES, "family_alpha_allocation": R.EXPECTED_ALLOCATION,
                          "global_familywise_alpha": 0.01, "tolerances": R.derived_tolerances()}, design)


def test_iid_limitation_is_stated_conditionally_and_not_overclaimed() -> None:
    strings = [n.value for n in ast.walk(ast.parse(GOHR_TOOL.read_text()))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for overclaim in ("D2 tests it", "D2 tests dependence"):
        assert not any(overclaim in s for s in strings), overclaim
    assert ("Every guarantee assumes i.i.d. rows within each sample and, for the exact tests, independent compared "
            "samples. D1 and D2 probe specified duplicate and dependence patterns only; they do not establish these "
            "assumptions, and existing Gohr D1/D2 results concern other instances.") in strings
    assert any("neither establishes i.i.d. rows" in s for s in strings)
    # The AutoND tool keeps its original wording (unchanged, bound to calibration c39726e5).
    assert "Row independence is assumed throughout; D2 tests it." in AUTOND_TOOL.read_text()


def test_s0_runs_the_bound_suite_by_repository_relative_path(monkeypatch) -> None:
    """The S0 record holds no home-directory path; the same suite file runs from the repository root."""
    seen = {}

    class Done:
        returncode, stdout = 0, "278 passed\n"

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, cwd=kw.get("cwd"))
        return Done()
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    rec = G.s0_unit_suite()
    assert seen["cmd"][-1] == "audit/dataset/tests/test_d3_method_validity.py"
    assert Path(seen["cwd"]) == G.REPO_ROOT and (G.REPO_ROOT / seen["cmd"][-1]).resolve() == G.SUITE_PATH
    assert rec["suite_sha256"] == hashlib.sha256(G.SUITE_PATH.read_bytes()).hexdigest() and rec["exit_code"] == 0
    assert str(Path.home()) not in json.dumps(rec) and "/Users/" not in json.dumps(rec)


def test_evidence_path_and_environment_records_are_not_absolute() -> None:
    assert G._repo_relative(G.ENGINE_PATH) == "audit/dataset/d3_distribution_statistics.py"
    env = G._environment_record()
    assert set(env) == {"python", "numpy", "scipy", "platform", "executable", "virtual_environment"}
    assert "/" not in env["executable"] and "/" not in env["virtual_environment"]
    assert str(Path.home()) not in json.dumps(env)
    main = next(n for n in ast.parse(GOHR_TOOL.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    attrs = {(n.value.id, n.attr) for n in ast.walk(main) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    assert ("sys", "executable") not in attrs, "main() must not record the interpreter's absolute path"
    calls = [ast.dump(n) for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "str"]
    assert not any("ENGINE_PATH" in c for c in calls), "engine_path must be repository-relative"


def test_main_refuses_an_existing_output_directory_before_any_work(tmp_path) -> None:
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        G.main(["--out-dir", str(out)])
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("n, p", [(50, 0.5), (1000, 0.25), (1000, 1e-3)])
def test_exact_clopper_pearson_spot_checks_at_the_gohr_alphas(n, p) -> None:
    for a in G.ALPHAS_USED["clopper_pearson"]:
        assert G._exact_cp_noncoverage(n, p, a) <= a * (1 + 1e-9)


def test_exact_fisher_spot_checks_at_the_gohr_alphas() -> None:
    for a in G.ALPHAS_USED["fisher_cell"]:
        for row in G._s8_task((a, 50, 40)) + G._s8_task((a, 1000, 10)):
            assert row["valid"], row
