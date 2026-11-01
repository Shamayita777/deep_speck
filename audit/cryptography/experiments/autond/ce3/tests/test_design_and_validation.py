import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np

import pytest

from audit.cryptography.core import evidence as EV
from audit.cryptography.experiments.autond.ce3 import design as D
from audit.cryptography.experiments.autond.ce3 import power as PW
from audit.cryptography.experiments.autond.ce3 import provenance as PV

CE3 = Path(__file__).resolve().parents[1]
CRYPTO = CE3.parents[2]
PRODUCER = ["adapter.py", "analysis.py", "benchmark.py", "design.py", "diagnostics.py", "p1.py", "power.py", "probes.py",
            "provenance.py", "runner.py", "statistics.py", "target_bins.py"]


def imports(path):
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


def test_frozen_preregistration_is_approved_for_the_exact_design_hash():
    pr = D.load_preregistration(CRYPTO / D.PREREGISTRATION_RELPATH)
    rendered = json.loads(json.dumps(D.render_preregistration()))
    assert {k: v for k, v in pr.items() if k != "approval"} == {k: v for k, v in rendered.items() if k != "approval"}
    assert pr["design"]["status"] == D.FROZEN and D.pending_items() == [] and D.pending_items(pr["design"]) == []
    assert pr["design"]["sampling"]["main_pairs_status"] == D.FROZEN
    assert pr["design"]["sampling"]["permutations_status"] == D.FROZEN
    assert set(pr["approval"]) == set(rendered["approval"])
    assert pr["approval"]["status"] == "APPROVED"
    assert pr["approval"]["approved_design_sha256"] == D.design_sha256() == pr["design_sha256"]
    assert pr["approval"]["approved_by"] and pr["approval"]["approved_utc"]
    assert D.production_authorisation(pr) == (True, "approved")


def test_authorisation_requires_every_approval_field_and_the_exact_hash():
    good = {"status": "APPROVED", "approved_design_sha256": D.design_sha256(), "approved_by": "x",
            "approved_utc": "2026-01-01T00:00:00Z"}
    for change in ({"status": "PENDING_REVIEW"}, {"approved_design_sha256": "0" * 64}, {"approved_by": None},
                   {"approved_utc": None}):
        pr = D.render_preregistration()
        pr["approval"] = {**good, **change}
        assert D.production_authorisation(pr)[0] is False, change
    pr = D.render_preregistration()
    pr["design"] = {**pr["design"], "status": D.PROPOSED}
    pr["approval"] = good
    assert D.production_authorisation(pr)[0] is False


def test_duplicate_keys_are_rejected():
    with pytest.raises(D.DesignError):
        D.parse_json_strict('{"approval": {"status": "PENDING_REVIEW", "status": "APPROVED"}}')


def test_design_hash_is_canonical_and_json_serialisable():
    assert D.design_sha256() == D.design_sha256(json.loads(json.dumps(D.DESIGN)))


def test_dependency_pins_match_the_live_tree():
    assert PV.dependency_problems(CRYPTO) == []


def test_ce2_source_run_provenance_without_full_rehash():
    r = PV.verify_ce2_source_run(CRYPTO, hash_all_evidence=False)
    assert all(r["checks"].values()) and r["twin_file_sha256"] == D.CE2_TWIN_FILE_SHA256


def test_validation_evidence_pins_and_consistency():
    for rel, sha in ((D.P1_RECORD_RELPATH, D.P1_RECORD_SHA256), (D.POWER_EVIDENCE_RELPATH, D.POWER_EVIDENCE_SHA256),
                     (D.BENCHMARK_RELPATH, D.BENCHMARK_SHA256),
                     (D.REPRESENTATION_DIAGNOSTICS_RELPATH, D.REPRESENTATION_DIAGNOSTICS_SHA256)):
        assert EV.sha256_file(CRYPTO / rel) == sha
    p1 = json.loads((CRYPTO / D.P1_RECORD_RELPATH).read_text())
    assert p1["selected_edges"] == D.TARGET_EDGES and p1["selected_n_classes"] == D.N_CLASSES
    assert p1["evidentiary"] is False and p1["model_loaded"] is False
    pw = json.loads((CRYPTO / D.POWER_EVIDENCE_RELPATH).read_text())
    assert pw["selection"]["status"] == "SELECTED"
    assert (pw["selection"]["main_pairs"], pw["selection"]["permutations"]) == (D.N_MAIN_PAIRS, D.N_PERMUTATIONS)
    assert pw["real_representations_used"] is False and pw["s1_procedure"] == D.DESIGN["endpoints"]["S1"]
    assert pw["permutation_construction"] == D.DESIGN["sampling"]["permutation_construction"]
    assert pw["pipeline_faithful"]["calibration_per_class"] == D.N_CALIBRATION_PER_CLASS
    bm = json.loads((CRYPTO / D.BENCHMARK_RELPATH).read_text())
    assert bm["scores_computed"] is False and bm["predictions_computed"] is False and bm["all_converged"] is True
    assert bm["selected_max_iter"] == D.PROBE["logistic"]["max_iter"] and bm["main_pairs"] == D.N_MAIN_PAIRS
    assert bm["permutations"] == D.N_PERMUTATIONS and bm["edges"] == D.TARGET_EDGES and bm["runtime_problems"] == []
    assert bm["true_target_labels_used_in_fitting"] == ["A", "T", "R"]
    assert bm["permutation_construction"] == D.DESIGN["sampling"]["permutation_construction"]
    assert all("fits" not in r for r in bm["runs"].values())
    dg = json.loads((CRYPTO / D.REPRESENTATION_DIAGNOSTICS_RELPATH).read_text())
    assert dg["target_used"] is False and dg["layer_changed"] is False and dg["inputs"]["target_read"] is False
    assert dg["inputs"]["n"] == D.P1_PAIRS and dg["layer"]["layer_index"] == D.REPRESENTATION["layer_index"]


def test_power_record_is_reproducible_from_its_own_replications():
    pw = json.loads((CRYPTO / D.POWER_EVIDENCE_RELPATH).read_text())
    by_n = {}
    for r in pw["pipeline_faithful"]["replicates"]:
        by_n.setdefault(r["n"], []).append(r)
    assert {str(n): len(v) for n, v in by_n.items()} == pw["pipeline_faithful"]["replications_per_n"]
    for n, reps in by_n.items():
        rec = pw["per_n"][str(n)]
        th = PW.estimate(reps)
        assert th == rec["estimates"]
        for b in PW.B_CANDIDATES:
            assert PW.power_at(th, b)["p_supported"] == rec["plug_in"][str(b)]["p_supported"]
    chosen = pw["selection"]["main_pairs"]
    boot = pw["per_n"][str(chosen)]["bootstrap"]
    assert boot[str(max(PW.B_CANDIDATES))]["lower"] >= PW.POWER_TARGET
    assert boot[str(pw["selection"]["permutations"])]["lower"] >= PW.POWER_TARGET


def test_power_s1_null_check_is_consistent_with_one_half():
    pw = json.loads((CRYPTO / D.POWER_EVIDENCE_RELPATH).read_text())
    nc = pw["s1_null_check"]
    assert nc["ci95"][0] <= 0.5 <= nc["ci95"][1] and sum(nc["rank_counts"]) == nc["replications"]


def test_power_exact_count_calculation_matches_enumeration():
    rng = np.random.default_rng(5)
    for _ in range(5):
        cells = rng.dirichlet(np.ones(4))
        n, crit = 6, 4
        brute = 0.0
        for seq in itertools.product(range(4), repeat=n):
            n1 = sum(1 for c in seq if c in (0, 1))
            n2 = sum(1 for c in seq if c in (0, 2))
            if n1 >= crit and n2 >= crit:
                brute += math.prod(cells[c] for c in seq)
        assert math.isclose(PW.joint_count_power(cells, n=n, crit=crit), brute, rel_tol=1e-12)


def test_power_critical_counts():
    assert PW.critical_count(20) == 15 and PW.critical_count(19) == 14 and PW.critical_count(5) == 5


def test_permutation_count_is_odd_and_sizes_respect_minimums():
    assert D.N_PERMUTATIONS % 2 == 1 and D.SMOKE["permutations_per_replicate"] % 2 == 1
    assert min(json.loads((CRYPTO / D.P1_RECORD_RELPATH).read_text())["selection"]["selected"]["class_fractions"]) \
        * D.N_MAIN_PAIRS * 0.2 > 5 * D.MIN_CLASS_COUNT_PER_FOLD


def test_verifier_imports_no_ce3_producer_module_or_core_evidence():
    imp = imports(CE3 / "verify.py")
    assert not any(i.lstrip(".") in {m[:-3] for m in PRODUCER} for i in imp)
    assert not any("core" in i for i in imp)
    assert "..ce2" in imp


def test_sources_do_not_import_gohr_ce1_or_legacy_framework():
    banned = ("gohr", "ce234", "adapters", "frozen_design", "probe", "experiments.ce1", "experiments.ce3", "autond_ce1")
    for name in PRODUCER + ["verify.py"]:
        for i in imports(CE3 / name):
            leaf = i.lstrip(".")
            assert not any(leaf == b or leaf.startswith(b + ".") for b in banned), (name, i)


def test_no_comments_or_docstrings_in_ce3_code():
    for p in list(CE3.glob("*.py")) + list((CE3 / "tests").glob("*.py")) + [CRYPTO / "kaggle" / "autond_ce3_kaggle.py"]:
        src = p.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node) is None, p
        import io
        import tokenize
        assert not [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT], p


def test_power_replications_use_the_production_order_six_group_for_b5(monkeypatch):
    from audit.cryptography.experiments.autond.ce3 import probes as PR
    calls, fits = [], []
    real_perm, real_arm = PR.permutations, PW._arm

    def perm(n, index, b):
        p = real_perm(n, index, b)
        calls.append((n, index, b, p))
        return p

    def arm(X, y, folds, k):
        fits.append(np.asarray(y).copy())
        return real_arm(X, y, folds, k)
    monkeypatch.setattr(PR, "permutations", perm)
    monkeypatch.setattr(PW, "_arm", arm)
    monkeypatch.setattr(PW, "calibration_lambda", lambda *a, **k: 0.08)
    fr = json.loads((CRYPTO / D.P1_RECORD_RELPATH).read_text())["selection"]["selected"]["class_fractions"]
    rep = PW._pipeline_replicate((1200, 300, fr, 0.03, 4242, 7))
    assert [c[2] for c in calls] == list(PW.B_EMPIRICAL) == [5, 11]
    n, index, b, p = calls[0]
    assert np.array_equal(p, real_perm(n, index, 5)) and np.array_equal(p[0][p[4]], np.arange(n))
    y = fits[0]
    assert all(np.array_equal(fits[1 + j], y[p[j]]) for j in range(5))
    assert len(rep["ba_perm"]["5"]) == 5 and rep["s1"]["5"]["rank"] in range(1, 7)
    assert all(c[2] + 1 != 22 for c in calls)
    nc = PW.null_replicate((600, 5, fr, 99, 3))
    assert calls[-1][2] == 5 and nc["rank"] in range(1, 7)
