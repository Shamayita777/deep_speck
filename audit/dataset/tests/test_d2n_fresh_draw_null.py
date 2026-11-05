"""D2-N fresh-draw null: decision rule, gates and record integrity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.dataset.experiments import d2n_fresh_draw_null as N

PREREG = Path(__file__).resolve().parents[1] / "preregistrations" / "d2n_fresh_draw_null.json"


def design():
    doc, sha = N.load_prereg(PREREG)
    return doc, sha


def test_rank_p_counts_ties_against_the_observed():
    assert N.rank_p(10, [1, 2, 3]) == 1 / 4
    assert N.rank_p(3, [1, 2, 3]) == 2 / 4
    assert N.rank_p(0, [0, 0, 0]) == 1.0


def test_rejection_needs_the_observed_count_above_every_null_count():
    null = list(range(40, 99))           # 59 values
    alpha_per = 0.05 / 3
    assert N.classify(99, null, alpha_per=alpha_per, ratio_gate=1.25)["statistical_gate"]
    assert not N.classify(98, null, alpha_per=alpha_per, ratio_gate=1.25)["statistical_gate"]


def test_both_gates_give_excess_one_gate_gives_inconclusive():
    alpha_per = 0.05 / 3
    big = N.classify(200, [100] * 59, alpha_per=alpha_per, ratio_gate=1.25)
    small = N.classify(110, [100] * 59, alpha_per=alpha_per, ratio_gate=1.25)
    none = N.classify(100, [100] * 59, alpha_per=alpha_per, ratio_gate=1.25)
    assert big["outcome"] == "EXCESS"
    assert small["outcome"] == "INCONCLUSIVE"
    assert none["outcome"] == "NO_EXCESS"


def test_design_is_59_nulls_bonferroni_over_three_partitions():
    doc, _ = design()
    d = doc["design"]
    assert d["null_datasets"] == 59 and d["primary_radius"] == 8
    assert 1 / (d["null_datasets"] + 1) <= d["alpha"] / 3


def _write_records(tmp: Path, d: dict, sha: str, obs: dict, null: dict, M=None):
    M = M or d["null_datasets"]
    for j in range(M + 1):
        rec = {"schema": N.RECORD_SCHEMA, "pipeline": "autond", "index": j,
               "design_sha256": sha, "source_hashes": {"x": "y"}, "partitions": {}}
        for p in N.PARTITIONS:
            src = obs if j == 0 else null
            rec["partitions"][p] = {"per_radius": {str(r): {"hits": src[p].get(r, 0),
                                                            "queried_rows": 20480}
                                                   for r in d["radii"]}}
        rec["record_sha256"] = N.canonical_sha256(rec)
        N.write_json_atomic(tmp / "records" / f"{j:03d}.json", rec)


def test_analysis_pass_fail_and_detection_flag(tmp_path):
    doc, sha = design()
    d = doc["design"]
    flat = {p: {8: 50} for p in N.PARTITIONS}
    _write_records(tmp_path, d, sha, flat, flat)
    assert N.analyse(tmp_path, d, sha, "autond")["d2n_outcome"] == "PASS"

    excess = {"train": {8: 200}, "validation": {8: 50}, "test": {8: 50}}
    _write_records(tmp_path, d, sha, excess, flat)
    assert N.analyse(tmp_path, d, sha, "autond")["d2n_outcome"] == "FAIL"

    flag = {"train": {8: 50, 2: 1}, "validation": {8: 50}, "test": {8: 50}}
    _write_records(tmp_path, d, sha, flag, flat)
    cert = N.analyse(tmp_path, d, sha, "autond")
    assert cert["d2n_outcome"] == "INCONCLUSIVE" and cert["detection_flags"]


def test_incomplete_run_is_not_analysed(tmp_path):
    doc, sha = design()
    d = doc["design"]
    flat = {p: {8: 50} for p in N.PARTITIONS}
    _write_records(tmp_path, d, sha, flat, flat, M=10)
    assert N.analyse(tmp_path, d, sha, "autond")["outcome"] == "INCOMPLETE"


def test_tampered_record_is_refused(tmp_path):
    doc, sha = design()
    d = doc["design"]
    flat = {p: {8: 50} for p in N.PARTITIONS}
    _write_records(tmp_path, d, sha, flat, flat)
    f = tmp_path / "records" / "000.json"
    r = json.loads(f.read_text())
    r["partitions"]["train"]["per_radius"]["8"]["hits"] = 1
    f.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="fails its own hash"):
        N.analyse(tmp_path, d, sha, "autond")


def test_record_from_another_design_is_refused(tmp_path):
    doc, sha = design()
    d = doc["design"]
    flat = {p: {8: 50} for p in N.PARTITIONS}
    _write_records(tmp_path, d, "0" * 64, flat, flat)
    with pytest.raises(SystemExit, match="another design"):
        N.analyse(tmp_path, d, sha, "autond")


def test_production_refuses_without_approval_and_without_token(tmp_path):
    with pytest.raises(SystemExit, match="--confirm"):
        N.main(["run", "--prereg", str(PREREG), "--pipeline", "gohr",
                "--instance-dir", str(PREREG.parents[1] / "evidence" /
                                      "d0_gohr_speck_r5_production_730516c79981"),
                "--out-dir", str(tmp_path)])
    doc, sha = design()
    with pytest.raises(SystemExit, match="not APPROVED|approval names"):
        N.require_approval(doc, sha)


def test_approval_must_name_the_current_design():
    doc, sha = design()
    doc = json.loads(json.dumps(doc))
    doc["approval"].update(status="APPROVED", design_sha256="f" * 64,
                           approved_by="x", approved_utc="2026-11-05T00:00:00Z")
    with pytest.raises(SystemExit, match="approval names design"):
        N.require_approval(doc, sha)
