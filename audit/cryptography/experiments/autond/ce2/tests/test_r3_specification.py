"""AutoND CE2 r3: the written hypothesis matches the implemented sign test; zero and undefined
coefficients are kept apart; r2 is archived; the r3 preregistration is frozen but not approved.
No TensorFlow needed."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from audit.cryptography.experiments.autond.ce2 import analysis as S
from audit.cryptography.experiments.autond.ce2 import design as FD
from audit.cryptography.experiments.autond.ce2 import power as PW
from audit.cryptography.experiments.autond.ce2 import verify as V

CE2 = Path(__file__).resolve().parents[1]
CRYPTO = CE2.parents[2]
R2 = CE2 / "history" / "r2"
COND = "Pr(rho_i > 0 | rho_i != 0, run i is valid)"
OBSOLETE = ("distributed symmetrically", "symmetrically about 0", "symmetric about 0",
            "population Spearman rho is", "population Spearman rank correlation", "Pr[rho_i > 0]")
ACTIVE_DOCS = ("README.md", "RUNBOOK.md", "POWER_ANALYSIS.md", "VERIFICATION_REPORT.md")


def _flat(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        q = f"{p}.{k}" if p else k
        out.update(_flat(v, q) if isinstance(v, dict) else {q: v})
    return out


def _r2_design() -> dict:
    return json.loads((R2 / "preregistration.json").read_text())["design"]


def _sums_verify(root: Path) -> list[str]:
    bad = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != h:
            bad.append(rel)
    return bad


def _outcomes(primary, perm=None, twin=None):
    perm = perm if perm is not None else [0.0001] * len(primary)
    twin = twin if twin is not None else [0.0001] * len(primary)
    return [S.RunOutcome(f"run_{i:02d}", "VALID", {"primary": p, "permuted_target": q, "untrained_twin": t})
            for i, (p, q, t) in enumerate(zip(primary, perm, twin))]


def _analyse(outcomes):
    return S.analyse(outcomes, alpha=0.05, min_valid_runs=len(outcomes), n_planned=len(outcomes),
                     bootstrap_resamples=200, bootstrap_seed=1)


def _verifier_decide(outcomes):
    rhos = {o.run_id: o.rho for o in outcomes if o.status == "VALID"}
    return V.decide(rhos, statuses={o.run_id: o.status for o in outcomes}, n_runs=len(outcomes),
                    min_valid=len(outcomes), alpha=0.05, precheck_ok=True)


# ================================================= hypothesis consistency


def test_primary_hypothesis_is_the_conditional_sign_probability_and_two_sided():
    h, t = FD.DESIGN["hypothesis"], FD.DESIGN["statistics"]["primary_test"]
    assert h["direction"] == "TWO_SIDED" and "two-sided" in t["name"]
    assert t["h0"] == f"{COND} = 1/2" and t["h1"] == f"{COND} != 1/2"
    assert t["h0"] in h["h0"] and t["h1"] in h["h1"]
    assert "No direction is prespecified." in h["h1"]
    assert h["measurement"] == "core.theory_consistency.TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED)"
    assert FD.P_PLUS == COND and COND in FD.DESIGN["estimand"]["primary"]
    assert "do not replace the primary sign-probability estimand" in FD.DESIGN["estimand"]["descriptive"]


def test_control_hypotheses_are_explicit_and_one_sided():
    c = FD.DESIGN["statistics"]["control_tests"]
    assert c["h0"].startswith("Pr(D_i,c > 0 | D_i,c != 0) = 1/2")
    assert c["h1"].startswith("Pr(D_i,c > 0 | D_i,c != 0) > 1/2")
    assert "Holm" in c["multiplicity"] and "only if the primary test rejects" in c["gate"]
    assert FD.DESIGN["statistics"]["alpha"] == 0.05


def test_no_active_text_describes_the_null_as_full_distribution_symmetry():
    active = {k: v for k, v in FD.DESIGN.items() if k != "revision"}
    blob = json.dumps(active)
    assert not [o for o in OBSOLETE if o in blob]
    for name in ACTIVE_DOCS:
        text = (CE2 / "docs" / name).read_text()
        assert not [o for o in OBSOLETE if o in text], name
    # the only remaining occurrence is the revision note that quotes and corrects the r2 wording
    assert "symmetrically about 0" in " ".join(FD.DESIGN["revision"]["changes"])


def test_research_question_and_estimand_are_observational_and_about_sign_imbalance():
    q = FD.DESIGN["research_question"]
    assert "systematic imbalance between positive and negative nonzero coefficients" in q
    assert "observational" in FD.DESIGN["study_type"]
    for word in ("cause", "causal effect", "learned", "mechanism"):
        assert word not in q.lower()


# ========================================================== zero handling


@pytest.mark.parametrize("values,pos,neg,zeros", [
    ([0.01] * 20, 20, 0, 0),                                  # all positive
    ([-0.01] * 20, 0, 20, 0),                                 # all negative
    ([0.01] * 10 + [-0.01] * 10, 10, 10, 0),                  # balanced
    ([0.01] * 17 + [0.0] * 3, 17, 0, 3),                      # some exact zeros
    ([0.0] * 20, 0, 0, 20),                                   # all zeros
    ([0.02] * 9 + [-0.02] * 6 + [0.0, -0.0] * 2 + [0.0], 9, 6, 5),   # mixture; -0.0 is a zero
])
def test_primary_sign_test_removes_and_counts_zeros(values, pos, neg, zeros):
    r = S.sign_test_two_sided(values)
    n = pos + neg
    assert (r["n_positive"], r["n_negative"], r["n_zero_ties_removed"], r["n_effective"]) == (pos, neg, zeros, n)
    expected = None if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)
    assert r["p_value"] == expected
    assert r["status"] == (S.NO_EFFECTIVE_OBSERVATIONS if n == 0 else S.TESTED)
    vp, vn, vpv = V.sign_two_sided(values)                    # the verifier's independent implementation
    assert (vp, vn) == (pos, neg) and vpv == expected


def test_zero_coefficients_are_valid_runs_and_reduce_only_n_effective():
    a = _analyse(_outcomes([0.01] * 17 + [0.0] * 3))
    assert a["n_valid"] == 20 and a["not_valid"] == []
    assert a["primary"]["n_effective"] == 17 and a["primary"]["n_zero_ties_removed"] == 3
    assert a["primary"]["p_value"] == 2 / 2 ** 17 and a["primary"]["rejected"]
    v = _verifier_decide(_outcomes([0.01] * 17 + [0.0] * 3))
    assert v["n_valid"] == 20 and (v["primary"]["pos"], v["primary"]["neg"]) == (17, 0)
    assert v["primary"]["p"] == a["primary"]["p_value"]


def test_all_zero_coefficients_are_not_testable_inconclusive_and_not_excluded():
    a = _analyse(_outcomes([0.0] * 20))
    assert a["n_valid"] == 20 and a["not_valid"] == []
    pr = a["primary"]
    assert (pr["n_effective"], pr["n_zero_ties_removed"], pr["p_value"]) == (0, 20, None)
    assert pr["status"] == S.NO_EFFECTIVE_OBSERVATIONS and pr["rejected"] is False
    assert a["decision"] == S.INCONCLUSIVE and "not testable" in a["decision_reason"]
    assert "separation_test" not in a["control_permuted_target"]
    v = _verifier_decide(_outcomes([0.0] * 20))
    assert v["decision"] == "INCONCLUSIVE" and v["primary"]["status"] == "NO_EFFECTIVE_OBSERVATIONS"
    assert v["primary"]["p"] is None and v["n_valid"] == 20
    assert V.sign_two_sided([0.0, -0.0]) == (0, 0, None)


def test_one_nonzero_coefficient_is_testable_but_cannot_reject():
    a = _analyse(_outcomes([0.0] * 19 + [0.01]))
    assert a["primary"]["status"] == S.TESTED and a["primary"]["n_effective"] == 1
    assert a["primary"]["p_value"] == 1.0 and a["decision"] == S.INCONCLUSIVE


def test_undefined_coefficient_is_never_converted_to_zero():
    score, target = np.full(100, 0.5), np.arange(100.0)
    assert S.definedness(score, target) == "constant model output"
    with pytest.raises(S.UndefinedCorrelation):
        S.spearman(score, target)
    assert V.rho(score, target) is None                       # the verifier: None, not 0.0
    with pytest.raises(S.UndefinedCorrelation):
        S.spearman(np.arange(100.0), np.full(100, -3.0))
    assert V.rho(np.r_[np.arange(99.0), np.nan], np.arange(100.0)) is None
    # a defined coefficient can be exactly zero; it is a number (a tie), not an exclusion
    x, y = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 2.0, 1.0])
    assert S.definedness(x, y) is None and S.spearman(x, y) == 0.0 and V.rho(x, y) == 0.0


def test_excluded_runs_are_distinct_from_zero_runs():
    outs = _outcomes([0.0] * 19)
    outs.append(S.RunOutcome("run_19", "EXCLUDED", {}, "untrained_twin: constant model output"))
    a = _analyse(outs)
    assert a["n_valid"] == 19 and a["not_valid"][0]["status"] == "EXCLUDED"
    assert a["decision"] == S.INCONCLUSIVE and "19 valid runs" in a["decision_reason"]


# ======================================================== control tests


def test_control_test_removes_and_counts_zero_differences():
    r = S.sign_test_one_sided_greater([0.01] * 14 + [0.0] * 4 + [-0.01] * 2)
    assert (r["n_positive"], r["n_negative"], r["n_zero_ties_removed"], r["n_effective"]) == (14, 2, 4, 16)
    assert r["p_value"] == sum(math.comb(16, i) for i in range(14, 17)) / 2 ** 16
    assert V.sign_greater([0.01] * 14 + [0.0] * 4 + [-0.01] * 2)[2] == pytest.approx(r["p_value"], rel=1e-15)
    assert "H1: Pr[D > 0] > 1/2" in r["test"]


def test_control_alternative_is_one_sided_greater():
    assert S.sign_test_one_sided_greater([-0.01] * 20)["p_value"] == 1.0
    assert S.sign_test_one_sided_greater([0.01] * 20)["p_value"] == 1 / 2 ** 20
    z = S.sign_test_one_sided_greater([0.0] * 20)
    assert z["status"] == S.NO_EFFECTIVE_OBSERVATIONS and z["p_value"] is None and z["n_zero_ties_removed"] == 20
    assert V.sign_greater([0.0] * 20) == (0, 0, None)


def test_zero_paired_differences_in_the_full_procedure():
    primary = [0.01] * 20
    perm = [0.01] * 3 + [0.0] * 17                           # |0.01| - |0.01| = 0 exactly: three ties
    a = _analyse(_outcomes(primary, perm=perm, twin=[0.0] * 20))
    t = a["control_permuted_target"]["separation_test"]
    assert (t["n_positive"], t["n_zero_ties_removed"], t["n_effective"]) == (17, 3, 17)
    v = _verifier_decide(_outcomes(primary, perm=perm, twin=[0.0] * 20))
    assert (v["controls"]["permuted_target"]["pos"], v["controls"]["permuted_target"]["neg"]) == (17, 0)
    assert v["decision"] == a["decision"] == S.SUPPORTED


def test_untestable_control_is_reported_and_cannot_support():
    primary = [0.01] * 20
    a = _analyse(_outcomes(primary, perm=primary, twin=[0.0001] * 20))   # every D_perm exactly 0
    c = a["control_permuted_target"]
    assert c["separation_test"]["status"] == S.NO_EFFECTIVE_OBSERVATIONS
    assert c["separation_test"]["n_effective"] == 0 and c["separation_test"]["n_zero_ties_removed"] == 20
    assert c["holm"] == {"p_value": None, "holm_adjusted_p": None, "threshold": None, "rejected": False,
                         "status": S.NO_EFFECTIVE_OBSERVATIONS}
    t = a["control_untrained_twin"]["holm"]
    assert t["rejected"] and t["threshold"] == 0.025          # the family stays two: alpha / 2 first
    assert a["decision"] == S.ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED and "not testable" in a["decision_reason"]
    v = _verifier_decide(_outcomes(primary, perm=primary, twin=[0.0001] * 20))
    assert v["controls"]["permuted_target"]["status"] == "NO_EFFECTIVE_OBSERVATIONS"
    assert v["holm"]["permuted_target"] == {"adjusted": None, "rejected": False}
    assert v["holm"]["untrained_twin"]["rejected"] and v["decision"] == a["decision"]


def test_producer_and_verifier_agree_with_zeros_and_untestable_tests():
    rng = np.random.default_rng(5)
    for _ in range(300):
        p = np.where(rng.random(20) < rng.choice([0.0, 0.3, 1.0]), 0.0, rng.normal(0.01, 0.01, 20))
        q = np.where(rng.random(20) < 0.3, np.abs(p), rng.normal(0, 0.005, 20))
        t = np.where(rng.random(20) < rng.choice([0.0, 1.0]), -np.abs(p), rng.normal(0.002, 0.01, 20))
        outs = _outcomes(list(p), perm=list(q), twin=list(t))
        a, v = _analyse(outs), _verifier_decide(outs)
        assert a["decision"] == v["decision"]
        assert a["primary"]["status"] == v["primary"]["status"]
        assert a["primary"]["p_value"] == v["primary"]["p"] or abs(a["primary"]["p_value"] - v["primary"]["p"]) < 1e-15
        for c in ("permuted_target", "untrained_twin"):
            if "separation_test" in a[f"control_{c}"]:
                assert a[f"control_{c}"]["separation_test"]["status"] == v["controls"][c]["status"]
                assert a[f"control_{c}"]["holm"]["rejected"] == v["holm"][c]["rejected"]


def test_zero_primary_run_counts_as_negative_in_control_tests():
    a = _analyse(_outcomes([0.01] * 18 + [0.0] * 2, perm=[0.001] * 20, twin=[0.001] * 20))
    t = a["control_untrained_twin"]["separation_test"]
    assert (t["n_positive"], t["n_negative"], t["n_zero_ties_removed"]) == (18, 2, 0)
    assert "-|rho_c,i| < 0" in FD.DESIGN["statistics"]["control_tests"]["zero_primary_run"]


# ======================================================= design integrity


def test_r2_archive_verifies_and_reproduces_the_r2_design_hash():
    assert _sums_verify(R2) == []
    names = {ln.split("  ", 1)[1] for ln in (R2 / "SHA256SUMS").read_text().splitlines()}
    assert {"design.py.r2", "preregistration.json", "implementation/verify.py.r2",
            "implementation/tests_test_pipeline.py.r2", "implementation/analysis.py.r2",
            "implementation/power.py.r2", "implementation/tests_test_theory_statistics.py.r2",
            "related/REPAIR_REPORT_2026-10-31.md.r2"} <= names
    # the repair report was only appended to by r3
    assert (CRYPTO / "REPAIR_REPORT_2026-10-31.md").read_bytes().startswith(
        (R2 / "related" / "REPAIR_REPORT_2026-10-31.md.r2").read_bytes())
    assert 'DESIGN_VERSION = "AutoND-CE2-frozen-design-2026-10-r2"' in (R2 / "design.py.r2").read_text()
    pre = json.loads((R2 / "preregistration.json").read_text())
    assert pre["design_sha256"] == FD.R2_DESIGN_SHA256 == FD.design_sha256(pre["design"])
    assert pre["design"]["design_version"] == "AutoND-CE2-frozen-design-2026-10-r2"
    assert pre["approval"]["status"] == "PENDING_REVIEW" and pre["approval"]["approved_by"] is None
    assert hashlib.sha256((R2 / "preregistration.json").read_bytes()).hexdigest() == (
        "e273abe656d0abe9aff0a64ac41b43e26fe6395d5624b5262a60f8486ba6f8d9")
    src = (R2 / "design.py.r2").read_text()
    assert 'DESIGN_VERSION = "AutoND-CE2-frozen-design-2026-10-r2"' in src


def test_r3_draft1_archive_verifies():
    d1 = CE2 / "history" / "r3-draft1"
    assert _sums_verify(d1) == []
    pre = FD.parse_preregistration_json((d1 / "preregistration.json").read_text())
    assert pre["design_sha256"] == FD.R3_DRAFT1_DESIGN_SHA256 == FD.design_sha256(pre["design"])
    assert pre["approval"]["status"] == "PENDING_REVIEW"
    assert hashlib.sha256((CE2 / "validation" / "r3" / "power.json").read_bytes()).hexdigest() == (
        "a23355fd92d20bd8234ef0b18a5fb1408d3e8f84ed9aacb6121cabeb852e18ea")


def test_r1_archive_unchanged():
    assert _sums_verify(CE2 / "history" / "r1") == []
    assert hashlib.sha256((CE2 / "history" / "r1" / "SHA256SUMS").read_bytes()).hexdigest() == (
        "70fb1d15732548e0cc284ca9b9e3944789e4da9a57b1e68d7fa10c9d3ba8347e")


def test_r3_revision_record():
    r = FD.DESIGN["revision"]
    assert FD.DESIGN["design_version"] == "AutoND-CE2-frozen-design-2026-10-r3"
    assert r["supersedes"]["design_sha256"] == FD.R2_DESIGN_SHA256
    assert "never approved" in r["supersedes"]["status"] and "smoke" in r["supersedes"]["status"]
    assert r["scientific_parameters_changed"] is False and r["decision_outcomes_changed"] is False
    assert r["computational_procedure_changed"] is True and "n_effective = 0" in r["computational_change"]
    assert r["inferential_specification_changed"] is True
    assert r["review_history"][0]["design_sha256"] == FD.R3_DRAFT1_DESIGN_SHA256 != FD.design_sha256()
    assert r["kind"] == "inferential-specification clarification"


def test_r2_to_r3_changes_are_exactly_the_declared_ones():
    a, b = _flat(_r2_design()), _flat(FD.DESIGN)
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed_prefixes = ("revision.", "estimand", "hypothesis.h0", "hypothesis.h1", "research_question",
                        "design_version", "runs.status", "statistics.power.status",
                        "statistics.power.assumptions", "statistics.power.evidence.",
                        "statistics.primary_test.", "statistics.primary_endpoint_role",
                        "statistics.control_tests.h0", "statistics.control_tests.h1",
                        "statistics.control_tests.p_value", "statistics.control_tests.zeros",
                        "statistics.control_tests.zero_primary_run", "exclusion.zero_coefficient",
                        "statistics.control_tests.no_effective_observations", "decision.INCONCLUSIVE")
    assert all(k.startswith(allowed_prefixes) for k in changed), sorted(changed)
    # no numeric or boolean value outside the revision record changed or appeared
    for k in changed:
        if not k.startswith("revision."):
            for v in (a.get(k), b.get(k)):
                assert v is None or isinstance(v, (str, list)), k
    # the primary test's p-value rule and the control statistic and test are unchanged
    for k in ("statistics.primary_test.p_value", "statistics.control_tests.statistic",
              "statistics.control_tests.test", "statistics.control_tests.multiplicity",
              "statistics.control_tests.gate", "statistics.primary_endpoint"):
        assert a[k] == b[k], k


@pytest.mark.parametrize("path,value", [
    ("runs.n_runs", 20), ("runs.pairs_per_run", 1_000_000), ("runs.min_valid_runs", 20),
    ("runs.reserve_runs", 0), ("behavioural_binding.n", 1_000_000),
    ("behavioural_binding.requirement", "lower bound > 0.5"), ("statistics.alpha", 0.05),
    ("conduct.stopping", "all planned runs are executed; no early stop"),
    ("exclusion.replacement", "none; no reserve runs"),
    ("reference.checkpoint_sha256", "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26"),
    ("reference.upstream_commit", "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"),
    ("controls.untrained_twin.twin_seed", 20261031),
])
def test_frozen_scientific_parameters(path, value):
    d = FD.DESIGN
    for k in path.split("."):
        d = d[k]
    assert d == value
    assert "one-sided 95% Wilson lower bound" in FD.DESIGN["behavioural_binding"]["interval"]


def test_markers_frozen_and_preregistration_matches_design():
    assert FD.DESIGN["runs"]["status"] == FD.DESIGN["statistics"]["power"]["status"] == "FROZEN"
    assert FD.pending_items() == []
    path = CRYPTO / FD.PREREGISTRATION_RELPATH
    pre = FD.load_preregistration(path)
    assert pre["design"] == FD.DESIGN and pre["design_sha256"] == FD.design_sha256()
    assert pre["schema"] == FD.PREREGISTRATION_SCHEMA == "autond-ce2-preregistration-1"
    # the file is the deterministic rendering, with only the approval block filled in by the approver
    expected = {**FD.render_preregistration(), "approval": pre["approval"]}
    assert path.read_bytes() == (json.dumps(expected, indent=1, sort_keys=True) + "\n").encode()


def test_single_design_version_assignment():
    src = (CE2 / "design.py").read_text()
    assert src.count("DESIGN_VERSION = ") == 1


def test_preregistration_parses_strictly_and_duplicate_keys_are_refused(tmp_path):
    text = (CRYPTO / FD.PREREGISTRATION_RELPATH).read_text()
    assert FD.parse_preregistration_json(text) == json.loads(text)
    status = json.loads(text)["approval"]["status"]
    dup = text.replace(f'"status": "{status}"', f'"status": "FORGED",\n  "status": "{status}"', 1)
    assert dup != text and json.loads(dup)["approval"]["status"] == status   # json.loads hides it
    f = tmp_path / "dup.json"
    f.write_text(dup)
    with pytest.raises(FD.PreregistrationError, match="duplicate JSON key 'status'"):
        FD.load_preregistration(f)
    with pytest.raises(V.VerificationError, match="duplicate"):
        json.loads(dup, object_pairs_hook=V._no_duplicate_keys)


def test_preregistration_approval_is_complete_and_bound_to_the_r3_hash():
    from datetime import datetime
    pre = FD.load_preregistration(CRYPTO / FD.PREREGISTRATION_RELPATH)
    a = pre["approval"]
    assert set(a) == {"status", "approved_design_sha256", "approved_by", "approved_utc"}
    assert a["status"] == "APPROVED" and a["approved_design_sha256"] == FD.design_sha256()
    assert isinstance(a["approved_by"], str) and a["approved_by"].strip()
    assert a["approved_utc"].endswith("Z") and datetime.fromisoformat(a["approved_utc"][:-1])
    assert FD.production_authorisation(pre) == (True, "approved")
    # the renderer still writes an empty block: an approval is only ever recorded by hand
    assert FD.render_preregistration()["approval"] == {"status": "PENDING_REVIEW", "approved_design_sha256": None,
                                                       "approved_by": None, "approved_utc": None}


def test_production_authorisation_fails_closed():
    pre = FD.render_preregistration()
    ok, why = FD.production_authorisation(pre)
    assert not ok and "PENDING_REVIEW" in why
    stale = {**pre, "approval": {"status": "APPROVED", "approved_design_sha256": FD.R2_DESIGN_SHA256,
                                 "approved_by": "x", "approved_utc": "y"}}
    ok, why = FD.production_authorisation(stale)
    assert not ok and "does not equal" in why
    draft = {**pre, "approval": {**stale["approval"], "approved_design_sha256": FD.R3_DRAFT1_DESIGN_SHA256}}
    assert not FD.production_authorisation(draft)[0]
    for missing in ("approved_by", "approved_utc"):
        a = {"status": "APPROVED", "approved_design_sha256": FD.design_sha256(), "approved_by": "x",
             "approved_utc": "y", missing: None}
        assert not FD.production_authorisation({**pre, "approval": a})[0]
    pending = FD.design()
    pending["runs"]["status"] = FD.PROPOSED
    a = {"status": "APPROVED", "approved_design_sha256": FD.design_sha256(), "approved_by": "x", "approved_utc": "y"}
    assert not FD.production_authorisation({"design": pending, "approval": a})[0]


def test_r2_preregistration_is_rejected_by_the_r3_module():
    with pytest.raises(FD.PreregistrationError):
        FD.load_preregistration(R2 / "preregistration.json")


def test_power_evidence_is_bound_by_hash():
    ev = FD.DESIGN["statistics"]["power"]["evidence"]
    data = (CRYPTO / ev["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ev["sha256"] == FD.POWER_EVIDENCE_SHA256
    p = json.loads(data)
    assert "zero_tie_sensitivity" in p and all("n_exact_zero_rho" in r for r in p["pipeline_faithful_check"]["rows"])
    assert ev["path"] == "experiments/autond/ce2/validation/r3/power_final.json"


def test_runbook_carries_the_current_hash_and_an_empty_approval_template():
    text = (CE2 / "docs" / "RUNBOOK.md").read_text()
    assert FD.design_sha256() in text
    assert '"approved_by": "<FULL_NAME_OF_ACCOUNTABLE_APPROVER>"' in text
    assert '"approved_utc": "<ACTUAL_ISO_8601_UTC_APPROVAL_TIMESTAMP>"' in text


# =============================================================== power


def test_zero_tie_sensitivity_exact_and_simulated():
    z = PW.zero_tie_sensitivity(n_sim=1000, seed=5)
    by = z["primary_exact_power_by_n_effective"]
    assert by["20"]["power"]["0.8"] == PW.primary_power_exact(20, 0.8, 0.05)
    assert by["20"]["rejection_threshold"] == 15 and by["19"]["rejection_threshold"] == 15
    assert by["19"]["power"]["0.8"] < by["20"]["power"]["0.8"]        # one zero tie costs power here
    assert z["smallest_n_effective_that_can_reject"]["primary_two_sided"] == 6
    rows = z["joint_power_with_exact_zero_primary_runs"]
    assert {r["n_zero_primary"] for r in rows} == {1, 2, 3}
    assert all(r["n_effective_primary"] == 20 - r["n_zero_primary"] for r in rows)


def test_zero_primary_runs_scenario_reduces_power_and_default_path_is_unchanged():
    kw = dict(n_runs=20, delta=2.5, twin_ratio=0.25, twin_error_corr=0.5, alpha=0.05, n_sim=4000, seed=9)
    base = PW.simulate_joint_power(**kw)
    assert PW.simulate_joint_power(**kw, n_zero_primary=0) == base
    assert "n_zero_primary" not in base
    three = PW.simulate_joint_power(**kw, n_zero_primary=3)
    assert three["power_supported"]["estimate"] < base["power_supported"]["estimate"]


def test_r3_power_reproduces_r2_joint_tables_and_draft1_zero_rows():
    r2 = json.loads((CE2 / "validation" / "r2" / "power.json").read_text())
    d1 = json.loads((CE2 / "validation" / "r3" / "power.json").read_text())
    r3 = json.loads((CRYPTO / FD.POWER_EVIDENCE_RELPATH).read_text())
    for k in ("joint_power_simulation", "permutation_dependence_sensitivity", "primary_exact_power",
              "rejection_threshold_positive_runs", "smallest_attainable_two_sided_p"):
        assert r2[k] == r3[k], k
    for k in ("joint_power_with_exact_zero_primary_runs", "primary_exact_power_by_n_effective"):
        assert d1["zero_tie_sensitivity"][k] == r3["zero_tie_sensitivity"][k], k


def test_power_evidence_has_control_zero_scenarios_and_records_interval_methods():
    r3 = json.loads((CRYPTO / FD.POWER_EVIDENCE_RELPATH).read_text())
    rows = r3["zero_tie_sensitivity"]["joint_power_with_exact_zero_control_differences"]
    pairs = {(r["n_zero_diff_perm"], r["n_zero_diff_twin"]) for r in rows}
    assert {(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 2), (3, 3)} == pairs
    for r in rows:
        assert r["n_effective_perm"] == [20 - r["n_zero_diff_perm"]]
        assert r["n_effective_twin"] == [20 - r["n_zero_diff_twin"]]
    every = (r3["joint_power_simulation"] + r3["permutation_dependence_sensitivity"] + rows
             + r3["zero_tie_sensitivity"]["joint_power_with_exact_zero_primary_runs"])
    for r in every:
        for k in ("power_primary", "power_supported"):
            assert r[k]["ci95_method"] == PW.CI_METHOD_NORMAL and r[k]["n_sim"] > 2000
    for r in r3["pipeline_faithful_check"]["rows"]:
        assert "Clopper-Pearson" in r["exact_zero_rho_upper95_method"]


def test_control_zero_difference_scenarios():
    kw = dict(n_runs=20, delta=2.5, twin_ratio=0.25, twin_error_corr=0.5, alpha=0.05, n_sim=3000, seed=4)
    base = PW.simulate_joint_power(**kw)
    assert PW.simulate_joint_power(**kw, n_zero_diff_perm=0, n_zero_diff_twin=0) == base
    r = PW.simulate_joint_power(**kw, n_zero_diff_perm=2, n_zero_diff_twin=3)
    assert (r["n_effective_perm"], r["n_effective_twin"]) == ([18], [17])
    assert r["power_primary"] == base["power_primary"]      # primary coefficients untouched
    none = PW.simulate_joint_power(**kw, n_zero_diff_perm=20)
    assert none["n_effective_perm"] == [0] and none["power_supported"]["estimate"] == 0.0


def test_interval_methods_are_described_for_both_branches():
    doc = PW.clopper_pearson.__doc__
    assert "exact Clopper-Pearson" in doc and "NOT exact Clopper-Pearson" in doc and "normal" in doc
    assert PW._estimate(np.ones(100, dtype=bool))["ci95_method"] == PW.CI_METHOD_EXACT
    assert PW._estimate(np.ones(2001, dtype=bool))["ci95_method"] == PW.CI_METHOD_NORMAL


def test_clopper_pearson_does_not_overflow_for_moderate_n():
    lo, hi = PW.clopper_pearson(1000, 2000)
    assert 0.47 < lo < 0.5 < hi < 0.53
    lo, hi = PW.clopper_pearson(3, 10)
    assert lo == pytest.approx(0.06674, abs=1e-5) and hi == pytest.approx(0.65245, abs=1e-5)


def test_pipeline_faithful_check_counts_exact_zeros():
    rng = np.random.default_rng(0)
    target = -rng.integers(10, 40, 20_000)
    r = PW.pipeline_faithful_check(target, n=2_000, lambdas=(0.0,), n_sim=50, seed=3)["rows"][0]
    for k in ("n_exact_zero_rho", "n_exact_zero_rho_permuted", "n_exact_zero_paired_difference_permuted"):
        assert isinstance(r[k], int) and 0 <= r[k] <= 50
    assert 0 < r["exact_zero_rho_upper95_per_replication"] < 1
