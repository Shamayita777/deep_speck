"""Validation of the Gohr D3 preregistration against the real engine, runner and Gohr calibration design.

The canonical record (audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json)
is either a clean UNAPPROVED draft or, once the approver signs it, a fully
approved record; these tests hold in both states. Draft and approved states are
simulated ONLY on copies, the approved one in a throwaway git repository.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d3_gohr_confirmatory as R
from audit.dataset.tests import test_d3_method_validity as MV
from audit.dataset.tools import d3_gohr_method_validity_calibration as G

REPO = R.REPO_ROOT
DRAFT = REPO / "audit" / "targets" / "gohr" / "preregistrations" / "d3_gohr_speck3264_r5.json"
RATIONALE = DRAFT.with_suffix(".RATIONALE.md")
JUSTIFICATION_PREFIX = "OPERATIONAL THRESHOLD by design convention"
REQUIRED_DISCLAIMERS = ("not a mathematical conversion from accuracy to a distributional effect",
                        "not a validated practical-equivalence margin",
                        "not a certificate of full-distribution equivalence")


@pytest.fixture(scope="module")
def draft() -> dict:
    return json.loads(DRAFT.read_text())


def _approved_copy(record: dict) -> dict:
    rec = copy.deepcopy(record)
    rec["data_examined_before_approval"] = rec["data_examined_before_approval"].replace(
        d3.PENDING_MARKER, "fixture answer (test only)")
    rec.update(approved=True, approved_by="Fixture Approver (test only)",
               approved_utc="2026-10-23T00:00:00+00:00", approval_statement=d3.APPROVAL_STATEMENT)
    return rec


def _unapproved_draft(record: dict) -> dict:
    rec = copy.deepcopy(record)
    rec.update(approved=False, approved_by="", approved_utc="", approval_statement="")
    return rec


def _calibration_design() -> dict:
    """The design block the Gohr calibration tool records (G.main), as the runner reads it."""
    return {"global_alpha": G.GLOBAL_ALPHA, "family_alpha_allocation": G.FAMILY_ALPHA_ALLOCATION,
            "family_roles": G.FAMILY_ROLES, "exploratory_alpha": G.EXPLORATORY_ALPHA,
            "partition_sizes": dict(G.PRODUCTION_SIZES), "feature_dimension": G.FEATURE_DIMENSION,
            "supported_class_fraction": list(G.SUPPORTED_CLASS_FRACTION)}


def test_unapproved_draft_is_refused(draft) -> None:
    rec = _unapproved_draft(_approved_copy(draft))
    with pytest.raises(d3.PreregistrationRejected, match="not approved"):
        d3.validate_preregistration_content(rec)


def test_canonical_record_is_a_clean_draft_or_fully_approved(draft) -> None:
    if draft["approved"] is False:
        assert draft["approved_by"] == "" and draft["approved_utc"] == "" and draft["approval_statement"] == ""
        with pytest.raises(d3.PreregistrationRejected):
            d3.validate_preregistration_content(draft)
    else:
        d3.validate_preregistration_content(draft)


def test_open_attestation_blocks_approval(draft) -> None:
    rec = _approved_copy(draft)
    rec["data_examined_before_approval"] += f" Open item: {d3.PENDING_MARKER}."
    with pytest.raises(d3.PreregistrationRejected, match="open attestation"):
        d3.validate_preregistration_content(rec)
    text = draft["data_examined_before_approval"]
    assert text.count(d3.PENDING_MARKER) <= 1
    if d3.PENDING_MARKER in text:
        before = text.split(d3.PENDING_MARKER)[0]
        assert "(3b) The Gohr D2 detection-sensitivity calibration" in before[-1500:], \
            "the only open attestation concerns the D2 calibration results (item 3b)"
        assert before.endswith("what the approver examined of its results before this protocol review: ")


def test_draft_discloses_prior_observations_and_timing(draft) -> None:
    text = draft["data_examined_before_approval"]
    for needle in ("had NOT been generated", "R1's sufficient statistic",
                   "d3_gohr_distribution_consistency_10000000_1000000_1000000_5r.json", "outcome PASS",
                   "DESCRIPTIVE_ONLY", "3434a362d66c432aba3e41923e279982", "outcome INCONCLUSIVE",
                   "d1_gohr_10000000_1000000_1000000_5r.json", "d2_gohr_speck_r6_rerun.json",
                   "II-4-GOHR-IMPL-V1", "fixed before the Gohr D0 instance was generated"):
        assert needle in text, needle


def test_draft_is_complete_for_the_engine_schema_once_approved(draft) -> None:
    content = d3.validate_preregistration_content(_approved_copy(draft))
    assert draft["schema_version"] == d3.PREREGISTRATION_SCHEMA_VERSION
    assert set(content["tolerances"]) == set(d3.TOLERANCE_KEYS)


def test_roles_allocation_and_joint_alpha(draft) -> None:
    assert draft["family_roles"] == R.REQUIRED_ROLES
    assert draft["family_roles"]["R1"] == "confirmatory" and draft["family_roles"]["E"] == "exploratory"
    assert draft["global_familywise_alpha"] == 0.01
    assert draft["family_alpha_allocation"] == R.EXPECTED_ALLOCATION == G.FAMILY_ALPHA_ALLOCATION
    assert set(draft["family_alpha_allocation"]) == {"R1", "R2", "A", "B", "C", "D"}
    assert sum(draft["family_alpha_allocation"].values()) == pytest.approx(0.005, rel=1e-15)
    assert d3.joint_alpha_problem(0.01, draft["family_alpha_allocation"]) is None


def test_tolerances_are_derived_from_the_frozen_ii4_margin(draft) -> None:
    assert R.materiality_margin() == 0.01
    assert draft["tolerances"] == R.derived_tolerances() == {
        "class_ratio": 0.01, "class_conditional_bit": 0.02, "pairwise_tvd": 0.01, "cross_class_bit": 0.02}
    rec = json.loads((REPO / R.MARGIN_SOURCE["frozen_record"]).read_text())
    src = REPO / R.MARGIN_SOURCE["preregistration"]
    assert rec["source_manifest"]["preregistration"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()


def test_margin_source_is_refused_when_it_no_longer_matches_its_frozen_record(tmp_path) -> None:
    for rel in (R.MARGIN_SOURCE["frozen_record"], R.MARGIN_SOURCE["preregistration"]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
    p = tmp_path / R.MARGIN_SOURCE["preregistration"]
    p.write_text(p.read_text().replace("delta_materiality=0.01", "delta_materiality=0.02"))
    with pytest.raises(SystemExit, match="no longer matches its frozen record"):
        R.materiality_margin(tmp_path)


def test_thresholds_are_operational_conventions_with_explicit_limits(draft) -> None:
    for k, text in draft["tolerance_justification"].items():
        assert text.startswith(JUSTIFICATION_PREFIX), k
        for d in REQUIRED_DISCLAIMERS:
            assert d in text, (k, d)
        assert "Reusing it here is a convention" in text, k
    j = draft["tolerance_justification"]
    assert "does NOT bound a label shortcut on its own" in j["class_conditional_bit"]
    assert "does NOT bound the 64-bit model" in j["cross_class_bit"]
    assert "more than two bits" in j["pairwise_tvd"]
    assert "Expected uniform-cell design-resolution illustration" in j["pairwise_tvd"]
    assert "This is not a universal impossibility proof" in j["pairwise_tvd"]


def test_quoted_design_resolution_matches_the_calibration_s5() -> None:
    rows = {(r["family"], r["design"]): r["minimum_certifiable_tolerance"] for r in G.s5()["rows"]}
    quoted = {("R1", "train"): "0.00057", ("R1", "validation"): "0.00182",
              ("R2", "train per class"): "0.00106", ("R2", "validation per class"): "0.00335",
              ("D", "train Y1 vs Y0"): "0.00212", ("D", "validation Y1 vs Y0"): "0.00670",
              ("A", "train vs validation"): "0.00644", ("A", "validation vs test"): "0.00977",
              ("B/C", "train vs validation"): "0.00911", ("B/C", "validation vs test"): "0.01381"}
    rationale = RATIONALE.read_text()
    for key, text in quoted.items():
        assert f"{rows[key]:.5f}" == text, (key, rows[key])
        assert text in rationale, text
    pair = json.loads(DRAFT.read_text())["tolerance_justification"]["pairwise_tvd"]
    for key in (("A", "train vs validation"), ("A", "validation vs test"), ("B/C", "train vs validation"),
                ("B/C", "validation vs test")):
        assert quoted[key] in pair


AH_PAIR = 0.005 / 6 / 6048     # per-hypothesis alpha of A, B and C


def test_expected_uniform_cell_resolution_illustrations(draft) -> None:
    """The figures quoted as illustrations: expected uniform cells, stated class fractions."""
    assert AH_PAIR == pytest.approx(1.377866e-7, rel=1e-6)
    res = lambda na, nb: d3.design_resolution(na, nb, per_hypothesis_alpha=AH_PAIR)["minimum_certifiable_tolerance"]
    assert f"{res(1_000_000, 1_000_000):.7f}" == "0.0097669"       # A validation vs test, pooled
    assert f"{res(500_000, 500_000):.7f}" == "0.0138135"           # B/C at 50% class fractions
    assert f"{res(450_000, 450_000):.6f}" == "0.014561"            # B/C at 45%
    rationale = RATIONALE.read_text()
    for fig in ("0.0097669", "0.0138135", "0.014561", "Expected uniform-cell design-resolution illustrations"):
        assert fig in rationale, fig
    t = draft["tolerances"]
    rows = {(r["family"], r["design"]): r["minimum_certifiable_tolerance"] for r in G.s5()["rows"]}
    assert rows[("R1", "validation")] < t["class_ratio"] and rows[("R2", "validation per class")] < t["class_conditional_bit"]
    assert rows[("D", "validation Y1 vs Y0")] < t["cross_class_bit"] and rows[("B/C", "validation vs test")] > t["pairwise_tvd"]


def test_resolution_is_not_data_independent_so_no_universal_impossibility() -> None:
    """A table far from the reference law (all rows in one cell) has an upper bound far below m."""
    k = np.array([[500_000, 0, 0, 0]], dtype=np.int64)
    assert d3.tvd_confidence_bounds(k, k, alpha=AH_PAIR)["upper"][0] < 0.001


def test_width_only_floor_holds_for_every_table_examined() -> None:
    """upper >= 1/4 * sum(widths) (max(x, y) >= (x + y)/2) on random tables, and the floor exceeds m on the
    near-uniform grid the RATIONALE quotes (a numerical evaluation, not a proof)."""
    rng = np.random.default_rng(7)
    per = AH_PAIR / 8
    for _ in range(200):
        n_a, n_b = int(rng.integers(1_000, 600_000)), int(rng.integers(1_000, 600_000))
        ta = rng.multinomial(n_a, rng.dirichlet([1, 1, 1, 1]))[None]
        tb = rng.multinomial(n_b, rng.dirichlet([1, 1, 1, 1]))[None]
        up = d3.tvd_confidence_bounds(ta, tb, alpha=AH_PAIR)["upper"][0]
        la, ua = d3.clopper_pearson_bounds(ta, np.full_like(ta, n_a), alpha=per)
        lb, ub = d3.clopper_pearson_bounds(tb, np.full_like(tb, n_b), alpha=per)
        assert up + 1e-12 >= min(1.0, 0.25 * float((ua - la).sum() + (ub - lb).sum()))
    floors = []
    for n in (450_000, 500_000, 550_000):
        for p in (0.20, 0.25, 0.30):
            k = np.full(4, int(round(p * n)), dtype=np.int64)
            lo, hi = d3.clopper_pearson_bounds(k, np.full(4, n, dtype=np.int64), alpha=per)
            floors.append(2 * float((hi - lo)[0]))
    assert min(floors) > 0.01 and f"{min(floors):.4f}" == "0.0122" and f"{max(floors):.4f}" == "0.0154"


@pytest.mark.parametrize("path", [DRAFT, RATIONALE])
def test_no_universal_impossibility_claim(path) -> None:
    text = path.read_text()
    assert text.count("impossib") == text.count("not a universal impossibility proof")
    for phrase in ("cannot PASS", "unreachable", "can never PASS", "impossible"):
        assert phrase not in text, phrase


def test_draft_matches_the_gohr_calibration_design(draft) -> None:
    R.check_protocol(d3.validate_preregistration_content(_approved_copy(draft)), _calibration_design())


def test_r1_exploratory_with_r1_still_allocated_is_refused_by_the_engine_itself(draft) -> None:
    rec = _approved_copy(draft)
    rec["family_roles"] = {**R.REQUIRED_ROLES, "R1": "exploratory"}
    with pytest.raises(d3.PreregistrationRejected, match="cover exactly the confirmatory families"):
        d3.validate_preregistration_content(rec)


@pytest.mark.parametrize("change", [
    {"family_alpha_allocation": {"R1": 0.0025, "R2": 0.0005, "A": 0.0005, "B": 0.0005, "C": 0.0005, "D": 0.0005}},
    {"family_alpha_allocation": {f: 0.001 for f in ("R2", "A", "B", "C", "D")},
     "family_roles": {**R.REQUIRED_ROLES, "R1": "exploratory"}},
    {"tolerances": {"class_ratio": 0.00144, "class_conditional_bit": 0.00288, "pairwise_tvd": 0.00144,
                    "cross_class_bit": 0.00288}},
])
def test_a_different_protocol_is_refused_by_the_runner(draft, change) -> None:
    rec = _approved_copy(draft)
    rec.update(change)
    content = d3.validate_preregistration_content(rec)
    with pytest.raises(SystemExit, match="not this protocol"):
        R.check_protocol(content, _calibration_design())


def test_approved_and_committed_copy_passes_verification(draft, tmp_path) -> None:
    path, sha = MV._commit_prereg(MV._git_repo(tmp_path), _approved_copy(draft))
    binding = d3.verify_preregistration(path, expected_sha256=sha)
    assert binding.record()["tolerances"] == draft["tolerances"]


def test_an_unapproved_draft_cannot_be_used_by_the_runner(draft, tmp_path) -> None:
    path, sha = MV._commit_prereg(MV._git_repo(tmp_path), _unapproved_draft(_approved_copy(draft)))
    with pytest.raises(d3.PreregistrationRejected, match="not approved"):
        d3.verify_preregistration(path, expected_sha256=sha)
    if draft["approved"] is False:
        with pytest.raises(d3.PreregistrationRejected):
            d3.verify_preregistration(DRAFT, expected_sha256=hashlib.sha256(DRAFT.read_bytes()).hexdigest())


def test_rationale_states_the_claim_boundaries() -> None:
    text = RATIONALE.read_text()
    for needle in ("not a mathematical conversion", "not validated practical-equivalence margins",
                   "not a certificate of full-distribution equivalence", "does not establish i.i.d. sampling",
                   "is not a deviation", "Illustration only", "arbitrary dependence between tests",
                   "c39726e5", "the **same** uniform key", "permutation of {0,1}³²"):
        assert needle in text, needle


def test_rationale_states_the_iid_condition_precisely() -> None:
    text = RATIONALE.read_text()
    assert ("Every guarantee assumes i.i.d. rows within each sample. The exact tests additionally assume that the "
            "two compared samples are independent. **D1 and D2 do not prove these assumptions.**") in text
    assert "the existing Gohr D1/D2 results concern other instances" in text
    for overclaim in ("D2 tests it", "D2 tests dependence", "establishes i.i.d", "proves independence"):
        assert overclaim not in text, overclaim


@pytest.mark.parametrize("path", [DRAFT, RATIONALE])
def test_no_assistant_self_reference_but_disclosure_is_kept(path) -> None:
    text = path.read_text()
    for word in ("assistant", "Claude", "ChatGPT", "LLM"):
        assert word.lower() not in text.lower(), word
    assert "protocol drafter" in text and "protocol development" in text


def test_disclosure_separates_protocol_development_from_the_approver(draft) -> None:
    text = draft["data_examined_before_approval"]
    assert ("Human approver: before approval, saw these outcome labels only (D1 PASS; D2 INCONCLUSIVE, "
            "DESCRIPTIVE_ONLY, NOT_RUN) and did not inspect the underlying records or their statistics; during this "
            "protocol review (2026-10-23) the approver was also shown the protocol drafter's written summary of the "
            "records' contents (the findings below), not the records themselves.") in text
    assert ("During this protocol review (2026-10-23) the approver was shown the protocol drafter's written summary "
            "of the artifact") in text and "not the artifact itself" in text
    assert ("Protocol drafter: inspected the complete underlying records on 2026-10-23, during this protocol review "
            "and before approval.") in text
    assert "this protocol was not drafted blind to them" in text
    for full_hash in ("cffc1d6416b5bd62fb0a4d19358b87ea297afb1cfdecb8925b5c35bc72c0bdca",
                      "c8b434c26dd6fee91d4b02447df85a093c7af21371d965778daaacd2cbc09603"):
        assert full_hash in text
    assert "examined by the approver (attested 2026-10-23 for the AutoND D3 protocol)" in text
    assert "Whether the approver examined them" not in text
    item3 = text.split("(3) The legacy Gohr D1 certificate")[1].split("(3b)")[0]
    assert d3.PENDING_MARKER not in item3, "the D1/D2 attestation (item 3) is answered"


def test_current_review_is_recorded_without_inferring_prior_exposure(draft) -> None:
    text = draft["data_examined_before_approval"]
    item6 = text.split("(6) ")[1]
    assert item6.startswith("Approver's review during this protocol review: on 2026-10-23")
    assert "it implies nothing about what the approver examined before this protocol review" in item6
    item3b = text.split("(3b)")[1].split("(4)")[0]
    assert ("what the approver examined of its results before this protocol review: the approver opened the calibration "
            "JSON or examined its individual per-condition detection rates.") in item3b
    assert d3.PENDING_MARKER not in text, "every attestation is answered"
    assert "realized_exact_radius_fraction metadata reads 0.0" in item3b and "only that metadata field is wrong" in item3b


def test_no_d2_gate_decision_and_its_limitation_are_frozen() -> None:
    text = RATIONALE.read_text()
    for needle in ("**No D2 gate (approver decision, 2026-10-23).**", "D2 cannot certify i.i.d. sampling",
                   "concern another, unpersisted dataset", "D1 PASS detects exact duplicates only",
                   "independence is assumed, not proved"):
        assert needle in text, needle
    assert "PENDING APPROVER ANSWER" not in text and "open attestation" not in text


def test_calibration_disclosure_is_factual_and_separate_from_the_rerun(draft) -> None:
    text = draft["data_examined_before_approval"]
    item3b = text.split("(3b)")[1].split("(4)")[0]
    for needle in ("audit/dataset/evidence/d2/calibration/d2_calibration_final_v1.json",
                   "8b987a0de9df8ee1f440d1a82e3f7e7eb2e3d45b0c2274efa10fc7ec2939822f", "status COMPLETED, 2026-09-14",
                   "synthetic fault-injection calibration", "D2 engine schema 6.4 at commit 95f5fa58",
                   "it audits no clean dataset", "whose own calibration slot is NOT_RUN",
                   "does not demonstrably calibrate that rerun's detector", "The approver ran this calibration",
                   "Protocol drafter: inspected the complete artifact on 2026-10-23",
                   "6 met the 0.95 Wilson-lower-bound detection target"):
        assert needle in item3b, needle
    p = REPO / "audit/dataset/evidence/d2/calibration/d2_calibration_final_v1.json"
    if p.is_file():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == "8b987a0de9df8ee1f440d1a82e3f7e7eb2e3d45b0c2274efa10fc7ec2939822f"
        c = json.loads(p.read_text())["calibration"]
        conds = [v for r in c["near_duplicate_injection"].values() for v in r.values()] + list(c["lag_copy_injection"].values())
        assert len(conds) == 25 and sum(v["confidence_lower_bound_target_met"] for v in conds) == 6


def test_disclosed_record_hashes_match_the_records_on_disk() -> None:
    for rel, h in (("audit/dataset/evidence/d1/d1_gohr_10000000_1000000_1000000_5r.json",
                    "cffc1d6416b5bd62fb0a4d19358b87ea297afb1cfdecb8925b5c35bc72c0bdca"),
                   ("audit/dataset/evidence/d2_gohr_speck_r6_rerun.json",
                    "c8b434c26dd6fee91d4b02447df85a093c7af21371d965778daaacd2cbc09603")):
        p = REPO / rel
        if not p.is_file():
            pytest.skip(f"{rel} not present (SKIPPED IS NOT PASSED)")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h


def test_d1_gate_policy_is_frozen_without_overclaiming() -> None:
    text = RATIONALE.read_text()
    assert ("If D1 does not return the preregistered PASS outcome, stop before D3 production, preserve all generated "
            "data and evidence, and investigate. Do not regenerate the dataset or choose another instance merely to "
            "obtain a PASS. A D1 PASS does not prove i.i.d. sampling") in text
    assert "under the D1 gate of §5 (no D3 production unless D1 returns PASS)" in text
    assert "The D3 runner enforces this gate itself" in text and "`--d1-dir`" in text
    assert "not yet implemented" not in text


def test_rationale_explains_the_frozen_decision_semantics() -> None:
    text = RATIONALE.read_text()
    for needle in ("`classify_hypothesis`, lines 1317–1350", "`CATEGORY_OUTCOME` (line 930)",
                   "`_OUTCOME_ORDER`, line 940", "`evaluate_confirmatory_decision`, lines 2252–2338",
                   "| lower > t (strict) | `CONFIRMED_DEVIATION` | **FAIL** |",
                   "| upper ≤ t and an exact detection | `STATISTICAL_ONLY` | **CONDITIONAL_PASS** |",
                   "| upper ≤ t and no detection | `EQUIVALENT_WITHIN_TOLERANCE` | **PASS** |",
                   "It does **not** state equality of the full 64-bit joint distribution",
                   "It is not altered here"):
        assert needle in text, needle
    src = Path(d3.__file__).read_text().splitlines()
    assert src[1316].startswith("def classify_hypothesis") and src[929].startswith("CATEGORY_OUTCOME = {")
    assert src[939].startswith("_OUTCOME_ORDER = ") and src[2251].startswith("def evaluate_confirmatory_decision")
    assert "provisional = _combine_outcomes" in src[2282] and 'outcome = provisional if not reasons' in src[2337]


def test_rationale_documents_paths_and_the_d1_prerequisite() -> None:
    text = RATIONALE.read_text()
    for needle in ("`decision.preregistration.path` will therefore hold an absolute path",
                   "Sealed and production evidence is never edited in place", "redaction manifest",
                   "d1_gohr_exact_census.py` always regenerates data", "separate Gohr D1 runner"):
        assert needle in text, needle
    src = Path(d3.__file__).read_text().splitlines()
    assert "_Path(path).resolve()" in src[2215] and '"preregistration": ({"path": pr.path' in src[2361]
