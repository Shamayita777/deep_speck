"""Guarantees the v2 adversarial layer must not lose."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from audit.adversarial import preregistration_v2 as P2      # noqa: E402
from audit.adversarial import preregistration as P1         # noqa: E402
from audit.adversarial import sealed_sets as SS             # noqa: E402

tf = pytest.importorskip("tensorflow")
from audit.adversarial import battery as B                  # noqa: E402


# ---- axis separation -------------------------------------------------------

def test_inconclusive_never_becomes_caught():
    assert P2.map_decision("INCONCLUSIVE", P2.CONSTRUCTION_VALID) == P2.UNDETERMINED
    for status in (P2.CONSTRUCTION_VALID, P2.CONSTRUCTION_FAILED):
        assert P2.map_decision("INCONCLUSIVE", status) != P2.CAUGHT


def test_mapping_is_total_and_explicit():
    assert set(P2.DECISION_TO_ADVERSARIAL) == set(P2.SCIENTIFIC_DECISIONS)
    assert P2.map_decision("SUPPORTED", P2.CONSTRUCTION_VALID) == P2.NOT_CAUGHT
    assert P2.map_decision("NOT_SUPPORTED", P2.CONSTRUCTION_VALID) == P2.CAUGHT
    with pytest.raises(ValueError):
        P2.map_decision("PASS", P2.CONSTRUCTION_VALID)


def test_construction_failure_suppresses_interpretation():
    for d in P2.SCIENTIFIC_DECISIONS:
        assert P2.map_decision(d, P2.CONSTRUCTION_FAILED) == P2.NOT_APPLICABLE


def test_v1_bug_is_gone():
    """v1 mapped anything != SUPPORTED to CAUGHT. That must not reappear."""
    src = (Path(B.__file__).read_text() + Path(P2.__file__).read_text())
    assert 'NOT_CAUGHT if d == "SUPPORTED" else' not in src
    assert P2.map_decision("INCONCLUSIVE", P2.CONSTRUCTION_VALID) == P2.UNDETERMINED


def test_raw_scientific_decision_is_preserved(tmp_path):
    rows = B.evaluate_dimensions.__doc__
    assert "NOT_APPLICABLE" in rows
    # structural: the certificate keeps axis 1 verbatim alongside axis 2
    import inspect
    src = inspect.getsource(B.evaluate_dimensions)
    assert '"scientific_decision": decision' in src
    assert '"adversarial_outcome": PRE2.map_decision(' in src


# ---- sealed data -----------------------------------------------------------

def test_sealed_set_is_persisted_hash_bound_and_not_regenerated(tmp_path):
    a = SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))
    assert not a["loaded_from_disk"] and (tmp_path / "sealed.npz").exists()
    b = SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))
    assert b["loaded_from_disk"]
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["Y"], b["Y"])
    assert a["content_sha256"] == b["content_sha256"]
    assert a["exact_replay_available"] is False      # stated, not pretended


def test_tampered_sealed_set_is_refused(tmp_path):
    SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))
    with np.load(tmp_path / "sealed.npz") as z:
        d = {k: z[k] for k in z.files}
    d["Y"] = 1 - d["Y"]
    np.savez_compressed(tmp_path / "sealed.npz", **d)
    with pytest.raises(ValueError, match="content hash"):
        SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))


def test_missing_sealed_file_is_never_silently_regenerated(tmp_path):
    SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))
    (tmp_path / "sealed.npz").unlink()
    with pytest.raises(FileNotFoundError, match="NOT regenerated"):
        SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))


# ---- threshold hygiene -----------------------------------------------------

def test_threshold_is_never_selected_on_the_sealed_set():
    import inspect
    src = inspect.getsource(B.assess_distinguisher)
    pick = src.split("# threshold chosen on CALIBRATION ONLY")[1].split("def acc_at")[0]
    assert "s_cal" in pick and "s_seal" not in pick, pick
    assert "never used to select anything" in src


def test_constant_output_model_scores_exactly_chance():
    class Const:
        def predict(self, X, batch_size=None, verbose=0):
            return np.full((len(X), 1), 0.5, dtype=np.float32)
    y = np.array([0, 1] * 50)
    assert B._auc(Const().predict(np.zeros((100, 64))).ravel(), y) == 0.5


# ---- preregistration integrity --------------------------------------------

def test_prereg_hash_changes_when_a_prediction_changes(monkeypatch):
    before = P2.plan_hash()
    original = P2.ADVERSARIES["A4"].predictions["CE4_intervention_sensitivity"]
    monkeypatch.setitem(P2.ADVERSARIES["A4"].predictions,
                        "CE4_intervention_sensitivity", P2.CAUGHT)  # any change
    assert P2.plan_hash() != before
    monkeypatch.setitem(P2.ADVERSARIES["A4"].predictions,
                        "CE4_intervention_sensitivity", original)
    assert P2.plan_hash() == before


def test_v1_preregistration_is_preserved_unmodified():
    assert P1.PREREG_VERSION == P2.SUPERSEDES
    assert "A1" in P1.ADVERSARIES
    assert P1.A1.predictions["CE2_theory_consistency"] == "NOT_CAUGHT"   # wrong, kept
    assert "CONSTRUCTION_FAILED" in P2.HISTORICAL["A1"].status


def test_rejected_candidate_is_recorded_with_its_evidence():
    r = P2.REJECTED_CANDIDATES["A1'-hamming-weight"]
    assert "0.5610" in r["rejected_because"]
    assert "DEVELOPMENT ONLY" in r["evidence"]


def test_reduced_scale_cannot_masquerade_as_production():
    prot = P2.ADVERSARIAL_PROTOCOL
    from audit.cryptography.ce234 import frozen_plan as FP
    assert prot["replication"]["CE2"] == FP.CE2.n_runs
    assert prot["replication"]["CE3"] == FP.CE3.n_replicates
    assert prot["replication"]["CE4"] == FP.CE4.n_runs
    assert "not the production estimand" in prot["estimand"].lower()


def test_equivalence_region_is_justified_not_asserted():
    crit = P2.DISTINGUISHER_CRITERION
    assert crit["equivalence_region"] == [0.49, 0.51]
    j = crit["margin_justification"]
    assert "0.523" in j and "0.561" in j        # tied to measured quantities
    assert "not a non-significance test" in crit["equivalence_rule"].lower() or \
           "TOST" in crit["equivalence_rule"]


def test_global_conclusion_is_declared_unchanged():
    assert "INCONCLUSIVE / LEVEL_1_PREDICTIVE" in P2.GLOBAL_CONCLUSION_UNCHANGED


# ---- A4 construction -------------------------------------------------------

def test_a4_depends_only_on_the_pair_xor():
    m = B.A4XorResponse()
    rep = B.verify_a4_xor_dependence(m, n=2000)
    assert rep["pair_xor_preserved"] and rep["outputs_identical"] and rep["delivered"]


# ===================================================================
# Mandatory patch set - one test per required change
# ===================================================================

def test_change1_a4_binding_is_not_applicable_and_no_fake_path():
    """A4 has no checkpoint: the gate must record NOT_APPLICABLE, never CAUGHT."""
    assert P2.ADVERSARIES["A4"].binding_applicable is False
    assert P2.ADVERSARIES["A3"].binding_applicable is True
    assert P2.ADVERSARIES["A4"].predictions["model_binding_gate"] == P2.NOT_APPLICABLE
    rep = B.binding_check(P2.ADVERSARIES["A4"], Path("/does/not/matter"))
    assert rep["outcome"] == P2.NOT_APPLICABLE and rep["executed"] is False
    assert "closed-form" in rep["reason"]
    src = Path(B.__file__).read_text()
    assert "/nonexistent" not in src          # the fabricated probe path is gone


def test_change2_three_axes_remain_separate_in_the_certificate():
    import inspect
    src = inspect.getsource(B.evaluate_dimensions)
    assert '"scientific_decision": decision' in src
    assert '"adversarial_outcome": PRE2.map_decision(' in src
    assert P2.map_decision("INCONCLUSIVE", P2.CONSTRUCTION_VALID) == P2.UNDETERMINED
    cert_src = inspect.getsource(B.main)
    assert '"construction_status": construction' in cert_src


def test_change3_no_prf_terminology_without_a_formal_claim():
    for mod in (P2, B):
        text = Path(mod.__file__).read_text()
        assert "pseudorandom map" not in text
        assert "PRF" not in text or "no formal pseudorandom-function" in text
    assert "BLAKE2b-derived deterministic map" in \
        P2.ADVERSARIES["A4"].construction
    assert "No formal pseudorandom-function claim" in P2.ADVERSARIES["A4"].construction


def test_change4_intro_does_not_claim_analytic_guarantee_for_all():
    doc = Path(P2.__file__).read_text()
    assert "guarantee is PREFERRED wherever the" in doc
    assert "A4 therefore has no analytic guarantee" in doc
    assert P2.ADVERSARIES["A3"].analytic_guarantee is True
    assert P2.ADVERSARIES["A4"].analytic_guarantee is False
    assert "EMPIRICAL, against the registered equivalence criterion" in \
        P2.ADVERSARIES["A4"].non_distinguisher_argument


def test_change5_equivalence_criterion_is_named_correctly():
    crit = P2.DISTINGUISHER_CRITERION
    assert crit["equivalence_region"] == [0.49, 0.51]          # unchanged
    assert crit["equivalence_criterion_name"] == "conservative 95% CI equivalence criterion"
    assert "NOT a conventional alpha = 0.05 TOST" in crit["equivalence_rule"]
    assert "more conservative" in crit["relation_to_tost"]


def test_change6_screened_candidate_wording():
    j = P2.DISTINGUISHER_CRITERION["margin_justification"]
    assert "differential candidate among the screened features" in j
    assert "weakest genuine differential signal observed" not in j


def test_change7_confirmatory_run_refuses_a_dirty_tree():
    assert B.confirmatory_preflight({"working_tree": "dirty"}, False)["ok"] is False
    assert "dirty" in B.confirmatory_preflight({"working_tree": "dirty"}, True)["reason"]
    assert B.confirmatory_preflight({"working_tree": "clean"}, False)["ok"] is True
    assert B.confirmatory_preflight({"working_tree": "UNAVAILABLE"}, False)["ok"] is False
    ok = B.confirmatory_preflight({"working_tree": "UNAVAILABLE"}, True)
    assert ok["ok"] is True and "provenance_limitation" in ok
    assert "smoke_and_build_exempt" in P2.CONFIRMATORY_PRECONDITIONS


def test_change8_source_hashes_cover_every_material_artifact():
    h = B.source_hashes()
    assert "_unhashed_modules" not in h, h.get("_unhashed_modules")
    need = ["ce234/frozen_plan.py", "ce234/production.py", "ce234/verify.py",
            "gohr/speck.py", "gohr/dataset.py", "gohr/model.py", "gohr/trainer.py",
            "gohr/evaluate.py", "probe/evaluation.py", "ce4/design.py",
            "adversarial/battery.py", "adversarial/preregistration_v2.py",
            "adversarial/sealed_sets.py", "adversarial/preregistration.py"]
    for frag in need:
        assert any(frag in k for k in h), frag
    assert all(len(v) == 64 for k, v in h.items() if k != "_unhashed_modules")


def test_change9_sealed_set_specification_mismatch_hard_fails(tmp_path):
    SS.build_or_load(tmp_path, "sealed", 2000, rounds=5, differential=(0x0040, 0))
    for bad in ({"n": 3000}, {"rounds": 7}, {"differential": (0x0020, 0)}):
        kw = {"n": 2000, "rounds": 5, "differential": (0x0040, 0), **bad}
        with pytest.raises(ValueError, match="does not match the requested"):
            SS.build_or_load(tmp_path, "sealed", kw["n"], rounds=kw["rounds"],
                             differential=kw["differential"])
    # the matching request still loads
    again = SS.build_or_load(tmp_path, "sealed", 2000, rounds=5,
                             differential=(0x0040, 0))
    assert again["loaded_from_disk"]


def test_change11_v1_artifacts_untouched():
    v1 = Path(P1.__file__).read_text()
    assert "A1 = Adversary(" in v1 and 'adversary_id="A1"' in v1
    assert P1.A1.predictions["CE2_theory_consistency"] == "NOT_CAUGHT"
    assert P1.A1.manipulation_check["threshold"] == 0.90
    assert (Path(P1.__file__).parent / "a1_theory_mimic.py").exists()


# ===================================================================
# Second patch set
# ===================================================================

class _StubAdv:
    """Minimal stand-in so scope can be tested without running a dimension."""
    def __init__(self, targets, predictions):
        self.targets = targets
        self.predictions = predictions
        self.prediction_rationale = {}


def test_p2_1_a4_executes_ce4_only(monkeypatch):
    """`targets` is authoritative: a prediction entry must not cause execution."""
    a4 = P2.ADVERSARIES["A4"]
    assert a4.targets == ("CE4",)
    assert a4.predictions["CE2_theory_consistency"] == P2.NOT_APPLICABLE
    assert a4.predictions["CE3_representation_decodability"] == P2.NOT_APPLICABLE
    assert "outside A4's registered target scope" in \
        a4.prediction_rationale["CE2_theory_consistency"]

    ran = []
    for name in ("run_ce2", "run_ce3", "run_ce4"):
        monkeypatch.setattr(B.PR, name,
                            (lambda n: (lambda *a, **k: (ran.append(n),
                                                         {"decision": "SUPPORTED"})[1]))(name))
    res = B.evaluate_dimensions(object(), Path("/tmp/x"), a4,
                                {"CE2": 1, "CE3": 1, "CE4": 1},
                                {"CE2": 10, "CE3": 10, "CE4": 10}, seed=0,
                                construction_status=P2.CONSTRUCTION_VALID,
                                log=lambda *a: None)
    assert ran == ["run_ce4"], ran
    assert res["CE2"]["executed"] is False and res["CE3"]["executed"] is False
    assert res["CE4"]["executed"] is True
    assert res["CE2"]["in_registered_targets"] is False


def test_p2_1_scope_rule_beats_a_stray_prediction(monkeypatch):
    """Even a non-NOT_APPLICABLE prediction cannot execute an out-of-scope dim."""
    stray = _StubAdv(("CE4",), {"CE2_theory_consistency": P2.CAUGHT,
                                "CE3_representation_decodability": P2.NOT_APPLICABLE,
                                "CE4_intervention_sensitivity": P2.NOT_CAUGHT})
    ran = []
    for name in ("run_ce2", "run_ce3", "run_ce4"):
        monkeypatch.setattr(B.PR, name,
                            (lambda n: (lambda *a, **k: (ran.append(n),
                                                         {"decision": "SUPPORTED"})[1]))(name))
    B.evaluate_dimensions(object(), Path("/tmp/x"), stray, {"CE2": 1, "CE3": 1, "CE4": 1},
                          {"CE2": 10, "CE3": 10, "CE4": 10}, seed=0,
                          construction_status=P2.CONSTRUCTION_VALID, log=lambda *a: None)
    assert ran == ["run_ce4"]
    assert "targets is authoritative" in \
        __import__("inspect").getsource(B.evaluate_dimensions).replace("`", "")


def test_p2_2_screening_ci_wording_is_mathematically_consistent():
    arg = P2.ADVERSARIES["A4"].non_distinguisher_argument
    lo, hi = 0.5003, 0.5039
    region = P2.DISTINGUISHER_CRITERION["equivalence_region"]
    assert region[0] <= lo and hi <= region[1]          # the CI IS inside
    assert "ENTIRELY INSIDE the registered" in arg
    assert "lies OUTSIDE the registered region" not in arg
    assert "0.5024" in arg and "0.5266" in arg          # the smoke failure, correctly
    assert region == [0.49, 0.51]                       # region unchanged


def test_p2_3_no_unjustified_no_differential_content_claim():
    doc = Path(P2.__file__).read_text()
    assert "Features with no differential content" not in doc
    assert "LOW-ASSOCIATION / LOW-DISTINGUISHABILITY screened features" in doc
    assert "ARE functions of the ciphertext-pair differential" in doc
    for frag in ("0.5007", "0.5005", "0.5020", "0.5610"):
        assert frag in doc                              # measurements preserved


def test_p2_4_run_never_builds_a3():
    import inspect
    src = inspect.getsource(B.main)
    # the A3 construction branch, not the earlier precondition block
    a3_branch = src.split('if a.adversary == "A3":')[1].split("else:\n        model = A4")[0]
    run_branch = a3_branch.split("if a.run:")[1].split("elif a.build")[0]
    assert "build_a3" not in run_branch, run_branch
    assert "load_frozen_a3" in run_branch
    assert "never builds" in inspect.getsource(B.load_frozen_a3)


def test_p2_4_run_hard_fails_without_a_frozen_artifact(tmp_path):
    with pytest.raises(SystemExit, match="no freeze manifest"):
        B.load_frozen_a3(tmp_path, tmp_path / "a3_representation.keras")


def test_p2_6_frozen_model_hash_is_enforced(tmp_path):
    mp = tmp_path / "a3_representation.keras"
    mp.write_bytes(b"pretend-model")
    head = "c0ffee00" + "0" * 32
    srcs = {"audit/adversarial/battery.py": "ab" * 32}
    man = {"schema": "adversary-freeze-manifest-1", "model_sha256": SS.sha256_file(mp),
           "training_record": {"adversary": "A3"}, "prereg_hash": P2.plan_hash(),
           "freeze_git_commit": head, "source_hashes": srcs}
    (tmp_path / B.A3_FREEZE_NAME).write_text(json.dumps(man))
    assert B.load_frozen_a3(tmp_path, mp,
                            current_sources=srcs)["model_sha256"] == man["model_sha256"]
    mp.write_bytes(b"different-model")
    with pytest.raises(SystemExit, match="does not match the frozen manifest"):
        B.load_frozen_a3(tmp_path, mp, current_sources=srcs)
    mp.unlink()
    with pytest.raises(SystemExit, match="must be built, frozen and committed"):
        B.load_frozen_a3(tmp_path, mp, current_sources=srcs)


def test_p2_5_frozen_registration_hash_gates_confirmatory_run():
    assert P2.plan_hash() == P2.EXPECTED_PREREG_HASH
    ok = B.confirmatory_preflight({"working_tree": "clean"}, False)
    assert ok["ok"] is True
    bad = B.confirmatory_preflight({"working_tree": "clean"}, False,
                                   plan_hash="00" * 32)
    assert bad["ok"] is False and "frozen confirmatory registration" in bad["reason"]
    # the VALUE is not part of the hashed document, so declaring it cannot
    # perturb the hash it pins (the NAME may appear in prose, which is fine)
    assert P2.EXPECTED_PREREG_HASH not in json.dumps(P2.prereg_dict(), default=str)


def test_p2_8_invariants_unchanged():
    assert P2.DISTINGUISHER_CRITERION["equivalence_region"] == [0.49, 0.51]
    assert set(P2.HYPOTHESES) >= {"H-ADV-2", "H-ADV-3", "not_tested_here"}
    assert "INCONCLUSIVE / LEVEL_1_PREDICTIVE" in P2.GLOBAL_CONCLUSION_UNCHANGED
    assert P1.A1.predictions["CE2_theory_consistency"] == "NOT_CAUGHT"
    assert P1.PREREG_VERSION == P2.SUPERSEDES


# ===================================================================
# Patch set 4 - commit identity records provenance;
#               source hashes enforce source integrity
# ===================================================================

_FROZEN_HEAD = "a3f1c0de" + "0" * 32        # commit A: where A3 was built
_LATER_HEAD = "b4d0c0de" + "1" * 32         # commit B: adds the artifact + manifest


def _fixture_freeze(tmp_path, monkeypatch, head=_FROZEN_HEAD, payload=b"pretend-a3"):
    """Controlled A3 freeze fixture. Never rebuilds the real adversary."""
    monkeypatch.setattr(B, "_git", lambda: {"commit": head, "working_tree": "clean"})
    mp = tmp_path / "a3_representation.keras"
    mp.write_bytes(payload)
    man = B.write_freeze_manifest(tmp_path, mp, {"adversary": "A3", "fixture": True})
    return mp, man


def test_1_build_records_a_full_freeze_commit(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    assert man["freeze_git_commit"] == _FROZEN_HEAD
    assert len(man["freeze_git_commit"]) == 40
    assert all(c in "0123456789abcdef" for c in man["freeze_git_commit"])
    on_disk = json.loads((tmp_path / B.A3_FREEZE_NAME).read_text())
    assert on_disk["freeze_git_commit"] == _FROZEN_HEAD
    for k in ("model_sha256", "source_hashes", "prereg_hash", "prereg_version",
              "environment", "git"):
        assert k in on_disk
    assert B.verify_freeze_commit_recorded(man) == _FROZEN_HEAD


def test_2_descendant_head_with_unchanged_sources_is_allowed(tmp_path, monkeypatch):
    """Commit A builds; commit B adds the artifact; sources unchanged -> allowed."""
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "_git",
                        lambda: {"commit": _LATER_HEAD, "working_tree": "clean"})
    assert B.current_head() == _LATER_HEAD != man["freeze_git_commit"]
    loaded = B.load_frozen_a3(tmp_path, mp, current_sources=man["source_hashes"])
    assert loaded["model_sha256"] == man["model_sha256"]
    assert B.confirmatory_preflight({"working_tree": "clean"}, False)["ok"] is True
    assert B.verify_source_integrity(man, man["source_hashes"])["verified"] is True


def test_3_modified_source_is_refused(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "_git",
                        lambda: {"commit": _LATER_HEAD, "working_tree": "clean"})
    tampered = dict(man["source_hashes"])
    victim = sorted(tampered)[0]
    tampered[victim] = "ff" * 32
    with pytest.raises(SystemExit) as e:
        B.load_frozen_a3(tmp_path, mp, current_sources=tampered)
    msg = str(e.value)
    assert "Source-hash mismatch" in msg and victim in msg
    assert "not the code that produced it" in msg
    # a source file appearing that was not hashed at freeze time is also refused
    extra = {**man["source_hashes"], "audit/new_module.py": "aa" * 32}
    with pytest.raises(SystemExit, match="present now but not at freeze time"):
        B.verify_source_integrity(man, extra)


def test_4_dirty_tree_still_refused(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    assert B.verify_source_integrity(man, man["source_hashes"])["verified"]
    assert B.verify_freeze_commit_recorded(man) == _FROZEN_HEAD
    assert B.load_frozen_a3(tmp_path, mp, current_sources=man["source_hashes"])
    pre = B.confirmatory_preflight({"working_tree": "dirty"}, False)
    assert pre["ok"] is False and "dirty" in pre["reason"]


def test_5_missing_or_invalid_freeze_commit_refused(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    absent = {k: v for k, v in man.items() if k != "freeze_git_commit"}
    with pytest.raises(SystemExit, match="records no git commit"):
        B.verify_freeze_commit_recorded(absent)             # key missing entirely
    for bad in ("UNAVAILABLE", ""):
        with pytest.raises(SystemExit, match="records no git commit"):
            B.verify_freeze_commit_recorded({**man, "freeze_git_commit": bad})
    with pytest.raises(SystemExit, match="not a full 40-"):
        B.verify_freeze_commit_recorded({**man, "freeze_git_commit": "a3f1c0de"})
    broken = {**man, "freeze_git_commit": "UNAVAILABLE"}
    (tmp_path / B.A3_FREEZE_NAME).write_text(json.dumps(broken, default=str))
    with pytest.raises(SystemExit, match="records no git commit"):
        B.load_frozen_a3(tmp_path, mp, current_sources=man["source_hashes"])


def test_6_missing_git_metadata_refused_unless_allow_no_git():
    assert B.confirmatory_preflight({"working_tree": "UNAVAILABLE"}, False)["ok"] is False
    ok = B.confirmatory_preflight({"working_tree": "UNAVAILABLE"}, True)
    assert ok["ok"] is True and "provenance_limitation" in ok


def test_7_model_hash_mismatch_refused_despite_valid_provenance(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    mp.write_bytes(b"tampered-after-freezing")
    with pytest.raises(SystemExit, match="does not match the frozen manifest"):
        B.load_frozen_a3(tmp_path, mp, current_sources=man["source_hashes"])
    mp.unlink()
    with pytest.raises(SystemExit, match="must be built, frozen and committed"):
        B.load_frozen_a3(tmp_path, mp, current_sources=man["source_hashes"])


def test_8_prereg_mismatch_refused_despite_valid_provenance(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    assert B.verify_source_integrity(man, man["source_hashes"])["verified"]
    pre = B.confirmatory_preflight({"working_tree": "clean"}, False,
                                   plan_hash="ff" * 32)
    assert pre["ok"] is False and "frozen confirmatory registration" in pre["reason"]


def test_9_freeze_commit_remains_provenance_from_a_later_commit(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "_git",
                        lambda: {"commit": _LATER_HEAD, "working_tree": "clean"})
    loaded = B.load_frozen_a3(tmp_path, mp, current_sources=man["source_hashes"])
    assert loaded["freeze_git_commit"] == _FROZEN_HEAD      # still commit A
    assert B.current_head() == _LATER_HEAD                  # running from B
    on_disk = json.loads((tmp_path / B.A3_FREEZE_NAME).read_text())
    assert on_disk["freeze_git_commit"] == _FROZEN_HEAD     # manifest not refreshed


def test_run_never_refreshes_the_manifest_or_rebuilds():
    import inspect
    src = inspect.getsource(B.load_frozen_a3)
    assert "write_freeze_manifest" not in src and "build_a3" not in src
    assert "write_text" not in src                          # nothing is mutated
    main = inspect.getsource(B.main)
    a3 = main.split('if a.adversary == "A3":')[1].split("else:\n        model = A4")[0]
    run_branch = a3.split("if a.run:")[1].split("elif a.build")[0]
    assert "build_a3" not in run_branch and "write_freeze_manifest" not in run_branch


def test_four_confirmatory_conditions_remain_independent(tmp_path, monkeypatch):
    mp, man = _fixture_freeze(tmp_path, monkeypatch)
    good = man["source_hashes"]
    assert B.confirmatory_preflight({"working_tree": "clean"}, False)["ok"] is True
    assert B.load_frozen_a3(tmp_path, mp, current_sources=good)["model_sha256"]
    failures = []
    if not B.confirmatory_preflight({"working_tree": "dirty"}, False)["ok"]:
        failures.append("dirty_tree")
    if not B.confirmatory_preflight({"working_tree": "clean"}, False,
                                    plan_hash="00" * 32)["ok"]:
        failures.append("prereg_hash")
    try:
        B.load_frozen_a3(tmp_path, mp, current_sources={**good, sorted(good)[0]: "00" * 32})
    except SystemExit:
        failures.append("source_hashes")
    mp.write_bytes(b"x")
    try:
        B.load_frozen_a3(tmp_path, mp, current_sources=good)
    except SystemExit:
        failures.append("model_hash")
    assert failures == ["dirty_tree", "prereg_hash", "source_hashes", "model_hash"]


def test_freeze_commit_is_not_part_of_the_preregistration():
    doc = json.dumps(P2.prereg_dict(), default=str)
    assert "freeze_git_commit" not in doc
    assert P2.plan_hash() == P2.EXPECTED_PREREG_HASH == \
        "ed079688bc7e84e73e550ea28c9f44ea838feb46bdbfa4e05cc0a033f8cbc4df"


def test_a1_historical_record_remains_untouched():
    assert P1.PREREG_VERSION == P2.SUPERSEDES
    assert P1.A1.predictions["CE2_theory_consistency"] == "NOT_CAUGHT"   # wrong, kept
    assert P1.A1.manipulation_check["threshold"] == 0.90
    assert (Path(P1.__file__).parent / "a1_theory_mimic.py").exists()
    assert "CONSTRUCTION_FAILED" in P2.HISTORICAL["A1"].status
    assert "A1" not in P2.ADVERSARIES                   # not merged into v2


# ---- patch set 5: repository-root and git-state reporting ----

def test_repo_root_is_the_directory_containing_audit():
    assert (B.REPO_ROOT / "audit" / "adversarial" / "battery.py").exists(), B.REPO_ROOT
    assert B.HERE.parents[1] == B.REPO_ROOT
    # rel() must not carry a spurious leading directory
    assert B.rel(B.HERE / "battery.py") == "audit/adversarial/battery.py"


def test_failed_git_is_never_reported_as_clean(monkeypatch):
    class _R:
        def __init__(self, rc, out="", err="fatal: not a git repository"):
            self.returncode, self.stdout, self.stderr = rc, out, err
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: _R(128))
    g = B._git()
    assert g["commit"] == "UNAVAILABLE" and g["working_tree"] == "UNAVAILABLE"
    assert "repo_root" in g
    # success: rev-parse returns a SHA, status --porcelain returns nothing
    def _ok(cmd, *a, **k):
        return _R(0, "db42d91fdf384a58f028e3a20e3108b7dbe8c37e\n"
                  if "rev-parse" in cmd else "")
    monkeypatch.setattr(B.subprocess, "run", _ok)
    g = B._git()
    assert g["working_tree"] == "clean"
    assert g["commit"] == "db42d91fdf384a58f028e3a20e3108b7dbe8c37e"

    # a modified file makes status non-empty -> dirty, not clean
    def _dirty(cmd, *a, **k):
        return _R(0, "db42d91fdf384a58f028e3a20e3108b7dbe8c37e\n"
                  if "rev-parse" in cmd else " M audit/adversarial/battery.py")
    monkeypatch.setattr(B.subprocess, "run", _dirty)
    assert B._git()["working_tree"] == "dirty"


def test_build_warns_when_provenance_is_incomplete(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(B, "_git",
                        lambda: {"commit": "UNAVAILABLE", "working_tree": "UNAVAILABLE"})
    mp = tmp_path / "a3_representation.keras"; mp.write_bytes(b"x")
    man = B.write_freeze_manifest(tmp_path, mp, {"adversary": "A3"})
    assert "provenance_warning" in man
    assert "WARNING:" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="records no git commit"):
        B.verify_freeze_commit_recorded(man)
