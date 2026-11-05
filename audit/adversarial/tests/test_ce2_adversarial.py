"""E6: outcome mapping, interpretation and production gates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.adversarial import ce2_adversarial as E

PREREG = Path(__file__).resolve().parents[1] / "preregistration_ce2_e6.json"


def test_only_supported_counts_as_satisfying_the_criterion():
    assert E.adversarial_outcome("SUPPORTED") == "CE2_CRITERION_SATISFIED"
    for d in ("INCONCLUSIVE", "NOT_SUPPORTED", None):
        assert E.adversarial_outcome(d) == "CE2_CRITERION_NOT_SATISFIED"


def test_any_satisfied_adversary_blocks_admission():
    doc, _ = E.load_prereg(PREREG)
    d = doc["design"]
    ok = {"A3": {"adversarial_outcome": "CE2_CRITERION_NOT_SATISFIED"},
          "A4": {"adversarial_outcome": "CE2_CRITERION_NOT_SATISFIED"}}
    bad = dict(ok, A4={"adversarial_outcome": "CE2_CRITERION_SATISFIED"})
    assert E.interpret(ok, d)["ce2_admission"] == "ADMITTED"
    assert E.interpret(bad, d) == {"ce2_admission": "NOT_ADMITTED",
                                   "statement": d["interpretation"]["if_any_satisfied"],
                                   "satisfied_by": ["A4"]}


def test_registered_replication_is_the_frozen_ce2_replication():
    doc, _ = E.load_prereg(PREREG)
    assert doc["design"]["ce2_replication"] == {"n_runs": 20, "samples_per_run": 1_000_000}


def test_production_needs_token_and_approval(tmp_path):
    with pytest.raises(SystemExit, match="--confirm"):
        E.main(["run", "--prereg", str(PREREG), "--out-dir", str(tmp_path)])
    doc, sha = E.load_prereg(PREREG)
    with pytest.raises(SystemExit, match="not APPROVED|approval names"):
        E.require_approval(doc, sha)
    doc = json.loads(json.dumps(doc))
    doc["approval"].update(status="APPROVED", design_sha256="0" * 64, approved_by="x",
                           approved_utc="2026-11-05T00:00:00Z")
    with pytest.raises(SystemExit, match="approval names design"):
        E.require_approval(doc, sha)
