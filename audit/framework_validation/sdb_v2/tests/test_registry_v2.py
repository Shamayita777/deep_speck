"""Static checks on the SDB-V2 registry, the verifier's independence and the V1 files."""
from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path

from audit.framework_validation.sdb_v2 import cases_v2 as C

HERE = Path(__file__).resolve().parents[1]
V1 = {"registration_sdb_v1.json": "d9df4816ac10ee6d9e4b584ac00b1158e8449742ec5b3746e32c60b9b1f5d048",
      "seeded_defects.py": "c4d5ffd8", "claim_engine.py": "821c8e29"}


def test_twenty_three_unique_cases_in_the_registered_groups():
    assert len(C.ORDER) == len(set(C.ORDER)) == 23
    assert Counter(C.GROUPS.values()) == {"historical_defect": 1, "historical_replay": 1, "synthetic_defect": 6,
                                          "clean_control": 7, "positive_decision_rule_control": 1,
                                          "abstention_control": 7}
    assert not {"K1", "U8"} & set(C.ORDER) and not any(c.startswith("A") for c in C.ORDER)


def test_detector_only_cases_have_no_claim_level_expectation():
    exp = C.expected_outcomes()
    for c in ("U4", "S5", "S4", "S2", "H1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"):
        assert set(exp[c]) == {"C"}, c
    assert set(exp["H2"]) == {"replay", "D", "R"}
    assert set(exp) - {"_H2_B1S"} == set(C.ORDER)


def test_h2_records_the_three_round_quantities_and_three_of_five():
    h2 = C.expected_outcomes()["H2"]["replay"]
    pert = {k: v for k, v in h2["fits"].items() if k.startswith("perturbed")}
    assert all({"selected_stage_round", "best_round", "stop_round"} <= set(v) for v in pert.values())
    assert sum(v["best_round"] == 5 and v["stop_round"] == 6 for v in pert.values()) == 3
    assert h2["perturbed_passed_round5_and_stopped_round6"] == 3 and h2["criterion"]["met"] is False


def test_the_verifier_imports_no_project_module():
    tree = ast.parse((HERE / "verify_sdb_v2.py").read_text())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods <= {"__future__", "argparse", "hashlib", "json", "math", "re", "subprocess", "sys", "datetime",
                    "pathlib", "h5py", "numpy", "scipy"}, mods


def test_v1_is_unchanged():
    for name, h in V1.items():
        assert hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest().startswith(h), name
