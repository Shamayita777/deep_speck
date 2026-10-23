"""Validation of the AutoND D3 preregistration against the real engine and calibration.

The canonical record (audit/targets/autond/preregistrations/d3_autond_speck3264_r8.json)
is either a clean UNAPPROVED draft or, once the approver signs it, a fully
approved record; these tests hold in both states. They check that it is
complete and consistent with the engine schema, the runner's roles and
joint-alpha allocation, the bound calibration and the calibration's own
design-resolution evidence; that its disclosure names the blinded runs by
their exact seals; that it makes no PASS or equivalence claim; and that a
draft (approval fields blank, or an attestation still open) is refused.
Draft and approved states are simulated ONLY on copies, the approved one in a
throwaway git repository.

They also recompute, from the D2 certificates, the radius-8 arithmetic quoted
in the RATIONALE. That arithmetic is descriptive and approximate; nothing
here tests row independence.
"""
from __future__ import annotations

import copy
import json
import re
from math import comb
from pathlib import Path

import pytest

from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d3_autond_distribution as R
from audit.dataset.tests import test_d3_method_validity as MV

REPO = Path(R.__file__).resolve().parents[3]
DRAFT = REPO / "audit" / "targets" / "autond" / "preregistrations" / "d3_autond_speck3264_r8.json"
RATIONALE = DRAFT.with_suffix(".RATIONALE.md")
#: The calibration run bound to this draft (fresh directory; never a superseded run).
CALIBRATION = REPO / "audit" / "dataset" / "evidence" / "d3_method_validity_20261022_c39726e5"
SUPERSEDED_CALIBRATIONS = ("d3_method_validity_20261022_976983a3", "d3_method_validity_20261022_6d81cf20",
                           "d3_method_validity_20261022_00b48038")
D2_DIR = REPO / "audit" / "dataset" / "evidence" / "d2_autond_speck3264_r8_production"
D0_MANIFEST = REPO / "audit" / "dataset" / "evidence" / "d0_autond_speck3264_r8_production" / "d0_canonical_manifest.json"
#: The two blinded integration runs disclosed in data_examined_before_approval,
#: run_id -> full audit_seal_sha256, as recorded by each run's certificate and manifest.
BLINDED_RUN_SEALS = {
    "6601b309567e4758ab458df55895ce45": "2803b0db8e0e2f5e7469f9b1b7eb664db2cfb275ffcbf57abe199b24f28a4e90",
    "1c04fe4cfcf5408aaee1fdb2f8da19e2": "0f01e61f4fb0824dc95a1d41a2ad10cc50404f0b808d7dd9b07c5981b09436fb",
}
BLINDED_DIRS = sorted((REPO / "audit" / "dataset" / "evidence").glob("d3_autond_speck3264_r8_blinded_integration_*/"))


@pytest.fixture(scope="module")
def draft() -> dict:
    return json.loads(DRAFT.read_text())


def _approved_copy(draft: dict) -> dict:
    rec = copy.deepcopy(draft)
    rec["data_examined_before_approval"] = rec["data_examined_before_approval"].replace(
        d3.PENDING_MARKER, "fixture answer (test only)")
    rec.update(approved=True, approved_by="Fixture Approver (test only)",
               approved_utc="2026-10-22T00:00:00+00:00", approval_statement=d3.APPROVAL_STATEMENT)
    return rec


def _unapproved_draft(record: dict) -> dict:
    """The record in its draft state: every approval field blank or false, content unchanged."""
    rec = copy.deepcopy(record)
    rec.update(approved=False, approved_by="", approved_utc="", approval_statement="")
    return rec


def _open_attestation_copy(record: dict) -> dict:
    """An otherwise approved copy whose disclosure still carries an unanswered attestation."""
    rec = _approved_copy(record)
    rec["data_examined_before_approval"] += f" Whether the approver examined X: {d3.PENDING_MARKER}."
    return rec


def test_unapproved_draft_is_refused(draft) -> None:
    with pytest.raises(d3.PreregistrationRejected, match="not approved"):
        d3.validate_preregistration_content(_unapproved_draft(draft))


def test_canonical_record_is_a_clean_draft_or_fully_approved(draft) -> None:
    """No half-filled approval: either every approval field is blank, or the whole record validates."""
    if draft["approved"] is False:
        assert draft["approved_by"] == "" and draft["approved_utc"] == "" and draft["approval_statement"] == ""
        with pytest.raises(d3.PreregistrationRejected, match="not approved"):
            d3.validate_preregistration_content(draft)
    else:
        d3.validate_preregistration_content(draft)


def test_open_attestation_blocks_approval(draft) -> None:
    """An unanswered attestation is refused even with every approval field filled; the canonical one is answered."""
    with pytest.raises(d3.PreregistrationRejected, match="open attestation"):
        d3.validate_preregistration_content(_open_attestation_copy(draft))
    assert d3.PENDING_MARKER not in json.dumps(draft)


def test_draft_discloses_what_was_examined(draft) -> None:
    text = draft["data_examined_before_approval"]
    assert "[Approver" not in text
    for needle in ("D0 canonical manifest", "label counts", "R1's complete sufficient statistic", "D1 certificate",
                   "D2 certificates", "INCONCLUSIVE", "blinded D3 integration runs", "6601b309567e4758ab458df55895ce45",
                   "1c04fe4cfcf5408aaee1fdb2f8da19e2", "No D3 effect, bound, p-value, category or outcome",
                   "Gohr D3 results", "by the drafting assistant (on 2026-10-22) and by the approver",
                   "outcome PASS", "DESCRIPTIVE_ONLY",
                   "The Gohr PASS does not validate this record's thresholds, alphas, family roles or statistical design",
                   "not a preregistration made before the data existed"):
        assert needle in text, needle
    d0 = json.loads(D0_MANIFEST.read_text())
    for part in ("train", "validation", "test"):
        lc = d0["partitions"][part]["label_counts"]
        assert f"{lc['positive']:,}/{lc['negative']:,}" in text


def test_blinded_runs_are_disclosed_with_their_exact_seals(draft) -> None:
    """Each disclosed blinded run is named with its full seal, and the pairs match the runs' own records."""
    text = draft["data_examined_before_approval"]
    disclosed = dict(re.findall(r"run ([0-9a-f]{32}), seal ([0-9a-f]{64})\b", text))
    assert disclosed == BLINDED_RUN_SEALS
    assert not re.search(r"seal [0-9a-f]{1,63}(?![0-9a-f])", text), "abbreviated or malformed seal"
    if not BLINDED_DIRS:
        pytest.skip("blinded integration evidence not present (SKIPPED IS NOT PASSED)")
    recorded = {}
    for d in BLINDED_DIRS:
        manifest = json.loads((d / "d3_autond_run_manifest.json").read_text())
        cert = json.loads((d / "d3_autond_certificate.json").read_text())
        assert cert["mode"] == "blinded-integration" and cert["blinded"] is True
        assert manifest["run_id"] == cert["run_identity"]["run_id"]
        assert manifest["audit_seal_sha256"] == cert["audit_seal_sha256"]
        recorded[manifest["run_id"]] = manifest["audit_seal_sha256"]
    assert recorded == BLINDED_RUN_SEALS


def test_draft_is_complete_for_the_engine_schema_once_approved(draft) -> None:
    content = d3.validate_preregistration_content(_approved_copy(draft))
    assert content["global_familywise_alpha"] == 0.01
    assert set(content["tolerances"]) == set(d3.TOLERANCE_KEYS)
    assert draft["schema_version"] == d3.PREREGISTRATION_SCHEMA_VERSION


def test_tolerances_are_the_source_derived_values_not_transcribed(draft) -> None:
    assert draft["tolerances"] == R.margin_derived_tolerance_candidates()["tolerances"]


def test_thresholds_are_operational_and_claim_no_pass(draft) -> None:
    for k, text in draft["tolerance_justification"].items():
        assert text.startswith("OPERATIONAL FAIL THRESHOLD, not a validated equivalence margin"), k
        assert "no PASS or equivalence claim is made" in text, k
    just = draft["tolerance_justification"]
    assert "does NOT bound a label shortcut on its own" in just["class_conditional_bit"]
    assert "does NOT bound the 64-bit model" in just["cross_class_bit"]
    assert "more than two bits" in just["pairwise_tvd"]
    assert "NON-GATING" in just["class_ratio"]


def test_allocation_roles_and_joint_alpha(draft) -> None:
    assert draft["global_familywise_alpha"] == 0.01
    assert draft["family_alpha_allocation"] == d3.allocate_familywise_alpha(0.01 / d3.GUARANTEE_COUNT,
                                                                             R.CONFIRMATORY_FAMILIES)
    assert draft["family_alpha_allocation"] == {f: 0.001 for f in ("R2", "A", "B", "C", "D")}
    assert d3.joint_alpha_problem(draft["global_familywise_alpha"], draft["family_alpha_allocation"]) is None
    assert draft["family_roles"] == {**R.REQUIRED_ROLES, "E": "exploratory"}
    assert draft["family_roles"]["R1"] == "exploratory"


def test_calibration_is_fresh_not_a_superseded_run() -> None:
    assert CALIBRATION.name not in SUPERSEDED_CALIBRATIONS


def _calibration():
    if not CALIBRATION.is_dir():
        pytest.skip(f"calibration evidence {CALIBRATION.name} not present (SKIPPED IS NOT PASSED)")
    return d3.verify_calibration_evidence(CALIBRATION, tool_path=R.CALIBRATION_TOOL_PATH,
                                          test_suite_path=R.CALIBRATION_SUITE_PATH)


def test_draft_is_compatible_with_the_bound_calibration(draft) -> None:
    cal = _calibration()
    R.check_gate_compatibility(d3.validate_preregistration_content(_approved_copy(draft)), cal.design())
    d0 = json.loads(D0_MANIFEST.read_text())
    assert cal.design()["partition_sizes"] == {p: d0["partitions"][p]["samples"] for p in ("train", "validation", "test")}
    assert cal.design()["family_roles"] == R.REQUIRED_ROLES


def _s5():
    _calibration()
    rows = json.loads((CALIBRATION / "d3_method_validity.json").read_text())["S5_design_resolution"]["rows"]
    return {(r["family"], r["design"]): r["minimum_certifiable_tolerance"] for r in rows}


def test_no_confirmatory_family_can_pass_at_these_thresholds(draft) -> None:
    """Each confirmatory family contains hypotheses whose resolution exceeds the threshold."""
    res, t = _s5(), draft["tolerances"]
    assert res[("R2", "train per class")] <= t["class_conditional_bit"] < res[("R2", "validation per class")]
    assert res[("D", "train Y1 vs Y0")] <= t["cross_class_bit"] < res[("D", "validation Y1 vs Y0")]
    pair = [v for (f, _), v in res.items() if f in ("A", "B/C")]
    assert t["pairwise_tvd"] < min(pair)


def test_quoted_resolution_figures_match_the_calibration(draft) -> None:
    res = _s5()
    quoted = {("R2", "train per class"): "0.00105", ("R2", "validation per class"): "0.00332",
              ("D", "train Y1 vs Y0"): "0.00210", ("D", "validation Y1 vs Y0"): "0.00665",
              ("A", "train vs validation"): "0.0064", ("A", "validation vs test"): "0.0097",
              ("B/C", "train vs validation"): "0.0091", ("B/C", "validation vs test"): "0.0137",
              ("R1", "train"): "0.00046", ("R1", "validation"): "0.00147"}
    rationale = RATIONALE.read_text()
    for key, text in quoted.items():
        digits = len(text.split(".")[1])
        assert f"{res[key]:.{digits}f}" == text, (key, res[key])
        assert text in rationale, text
    for key in [k for k in quoted if k[0] in ("R2", "D", "A", "B/C")]:
        fam_text = json.dumps(draft["tolerance_justification"])
        assert quoted[key] in fam_text, key


def test_approved_and_committed_copy_passes_verification(draft, tmp_path) -> None:
    path, sha = MV._commit_prereg(MV._git_repo(tmp_path), _approved_copy(draft))
    binding = d3.verify_preregistration(path, expected_sha256=sha)
    assert binding.record()["tolerances"] == draft["tolerances"]


def test_an_unapproved_draft_cannot_be_used_by_the_runner(draft, tmp_path) -> None:
    """Even committed and anchored, a draft is refused; so is the canonical file while it is still a draft."""
    import hashlib
    path, sha = MV._commit_prereg(MV._git_repo(tmp_path), _unapproved_draft(draft))
    with pytest.raises(d3.PreregistrationRejected, match="not approved"):
        d3.verify_preregistration(path, expected_sha256=sha)
    if draft["approved"] is False:
        with pytest.raises(d3.PreregistrationRejected):
            d3.verify_preregistration(DRAFT, expected_sha256=hashlib.sha256(DRAFT.read_bytes()).hexdigest())


def _p8() -> float:
    return sum(comb(64, k) for k in range(9)) / 2 ** 64


def _approx_recall(radius=8, bits=64, sampled=16, tables=8) -> float:
    """APPROXIMATE recall of D2's retrieval for pairs within `radius`, weighted by the uniform
    distance distribution. Assumes independent position draws per table and a non-binding bucket cap."""
    mass = {d: comb(bits, d) for d in range(radius + 1)}
    rec = {d: 1 - (1 - comb(bits - d, sampled) / comb(bits, sampled)) ** tables for d in mass}
    return sum(mass[d] * rec[d] for d in mass) / sum(mass.values())


def test_radius8_arithmetic_quoted_in_the_rationale_is_reproducible() -> None:
    """Recomputes the DESCRIPTIVE figures in RATIONALE §5. Asserts no statement about independence:
    the full-recall expectation is an upper bound on what D2 could observe, and the recall-adjusted
    figure is approximate and assumption-dependent."""
    if not D2_DIR.is_dir():
        pytest.skip("D2 production evidence not present")
    rationale = RATIONALE.read_text()
    assert "2.7813 × 10⁻¹⁰" in rationale and f"{_p8():.4e}" == "2.7813e-10"
    assert "not a formal test of independence" in rationale and "Approximate, assumption-dependent" in rationale
    d0 = json.loads(D0_MANIFEST.read_text())
    cert = json.loads((D2_DIR / "d2_autond_c_certificate.json").read_text())
    assert cert["findings"]["decision"]["outcome"] == "INCONCLUSIVE"
    assert cert["findings"]["decision"]["not_proof_of_independence"] is True
    recall = _approx_recall()
    assert f"{recall:.2f}" == "0.53"
    full, adjusted, observed = {}, {}, {}
    for part, rec in cert["findings"]["results"]["d2_1_near_duplicate_structure"].items():
        r = rec["8"]
        name = part.split(":")[1]
        assert r["null_pair_probability"] == 0.0 and r["excess_ratio_status"] == "UNBOUNDED"
        assert r["projection_tables"] == 8 and r["projection_bits"] == 16 and r["exhaustive"] is False
        n_ref = d0["partitions"][name]["label_counts"]["negative"]
        full[name] = r["queried_rows"] * (1 - (1 - _p8()) ** (n_ref - 1))
        adjusted[name] = full[name] * recall
        observed[name] = r["rows_with_verified_neighbor_within_radius"]
    assert [f"{full[p]:.2f}" for p in ("train", "validation", "test")] == ["28.45", "2.85", "2.85"]
    assert [f"{adjusted[p]:.1f}" for p in ("train", "validation", "test")] == ["15.1", "1.5", "1.5"]
    assert [observed[p] for p in ("train", "validation", "test")] == [14, 2, 1]
    assert re.search(r"28\.45 \(train\), 2\.85 \(validation\), 2\.85 \(test\)", rationale)
    assert "≈ 15.1 / 1.5 / 1.5" in rationale and "14 / 2 / 1" in rationale
