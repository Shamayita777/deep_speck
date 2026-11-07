"""
The verifier's own re-implementations against the frozen components they recheck, on inputs that are
NOT the registered fixtures (fresh seeds, temporary files). The verifier itself imports no project code;
only this test brings the two together.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("sdb_v2_verifier_under_test", HERE / "verify_sdb_v2.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def test_m1_against_verify_m1(tmp_path):
    from audit.implementation.ii2_independent import verify_m1
    ref = REPO / "audit/implementation/reference/speck.py"
    assert V.m1_outcomes(ref)["published_vector"] == verify_m1.test_published_vector(verify_m1.load_reference_speck())["status"]
    mut = tmp_path / "m.py"
    mut.write_text(ref.read_text().replace("def ALPHA():\n    return(7);", "def ALPHA():\n    return(9);"))
    s = importlib.util.spec_from_file_location("sdb_v2_tmp_mut", mut)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    with pytest.raises((AssertionError, RuntimeError)):
        verify_m1.test_published_vector(m)
    assert V.m1_outcomes(mut) == {"published_vector": "FAIL", "key_schedule": "FAIL", "round_traces": "FAIL", "alpha": 9}


def test_h5_architecture_against_the_probe(tmp_path):
    sys.path.insert(0, str(REPO))
    import train_nets
    from audit.implementation.gohr_adapter import probe_checkpoint_architecture
    p = tmp_path / "net.h5"
    train_nets.make_resnet(depth=3, reg_param=1e-5).save(p)
    pr = probe_checkpoint_architecture(p).to_dict()
    a = V.h5_arch(p)
    assert (a["conv1d"], a["add"], a["depth"]) == (pr["n_conv1d_layers"], pr["n_add_layers"], pr["derived_depth"]) == (7, 3, 3)


def _fit(round_max, cond="perturbed", rep=1):
    from audit.framework_validation.sdb_v2.fixtures_v2 import _fit
    return _fit(cond, rep, round_max)


def test_progression_and_criterion_against_autond_d4():
    from audit.dataset.adapters.autond_d4 import progression
    from audit.dataset.experiments.d4_autond_label_shuffle import progression_criterion
    rng = random.Random(5)
    for _ in range(200):
        maxes, k = [], 0
        while k < 5:
            m = rng.choice([0.501, 0.505, 0.5051, 0.51, 0.7])
            maxes.append(m)
            k += 1
            if m <= 0.505:
                break
        recs = _fit(maxes)
        mine, theirs = V.progression(recs), progression(recs, rounds=[5, 6, 7, 8, 9], epochs_per_round=10, threshold=0.505)
        for key in ("best_round", "stop_round", "rounds_trained", "rounds_max_val_acc", "status"):
            assert mine[key] == theirs[key], key
        assert (mine["selected_stage_round"], mine["selected_epoch"]) == (theirs["summaries"][0]["stage_round"],
                                                                           theirs["summaries"][0]["selected_epoch"])
    fits = [{"condition": c, "best_round": b, "stop_round": s} for c, b, s in
            (("perturbed", None, 5), ("perturbed", 5, 6), ("baseline", 8, 9))]
    t = progression_criterion({"fits": fits}, 5)
    assert V.criterion(fits) == {"met": t["met"], "perturbed": t["perturbed_stop_at_starting_round_without_best_round"],
                                 "baseline": t["baseline_pass_the_starting_round"]}


def test_claim_precedence_against_the_engine():
    from audit.framework_validation.sdb_v2 import engine_v2 as E
    rng = random.Random(9)
    for _ in range(5000):
        req = [d for d in E.DIMENSIONS if rng.random() < 0.6]
        dims = {d: rng.choice(["PASS", "FAIL", "INCONCLUSIVE"]) for d in E.DIMENSIONS if rng.random() < 0.9}
        ce = [{"id": f"t{i}", "outcome": rng.choice(["SUPPORTED", "NOT_SUPPORTED"]), "required": rng.random() < 0.3,
               "prereg": rng.random() < 0.7, "powered": rng.random() < 0.7, "control": rng.random() < 0.3}
              for i in range(rng.randint(0, 2))]
        L, pred = rng.randint(0, 3), rng.random() < 0.9
        comp = E.Component("c", frozenset(req), L, "COMPLETE", pred, {d: E.DimRecord(E.Dim(v)) for d, v in dims.items()},
                           tuple(E.CETest(t["id"], E.CE(t["outcome"]), t["required"], t["prereg"], t["powered"], t["control"])
                                 for t in ce))
        e, v = E.decide(comp), V.decide(req, dims, L, ce, pred)
        assert (e.verdict.value, e.rule.split(":")[0]) == (v["verdict"], v["rule"])
        if v["rule"] in ("3", "4"):
            assert e.level == v["level"]


def test_ev_against_the_frozen_chain(tmp_path, monkeypatch):
    from audit.framework_validation.sdb_v2 import cases_v2 as C
    rng = np.random.default_rng(3)
    for i in range(40):
        a = 0.92 + rng.normal(0, 0.001, 6)
        delta = rng.choice([0.0, -0.004, -0.012, -0.05])
        b = a + delta + rng.normal(0, rng.choice([0.0002, 0.002, 0.01]), 6)
        valid = int(rng.choice([2, 5, 6]))
        units = [{"unit": f"u{k}", "a": float(a[k]), "b": float(b[k]), "status": "valid" if k < valid else "failed",
                  "failure_reason": None if k < valid else "divergence"} for k in range(6)]
        f = {"units": units, "epsilon": 0.01, "epsilon_predeclared": True, "alpha": 0.05, "minimum_valid": 5,
             "statistics_seed": 1}
        d = tmp_path / f"c{i}"
        d.mkdir()
        (d / "u.json").write_text(json.dumps(f))
        (d / "bundle.json").write_text(json.dumps({"records": str(d / "u.json")}))
        monkeypatch.setattr(C, "FIX", tmp_path)
        frozen = C._ev(f"c{i}")[0]["C"]
        assert V.ev_decision(f) == frozen, (f, frozen)


def test_d4_against_evaluate_near_chance_decision():
    from audit.dataset.d4_controlled_perturbation import evaluate_near_chance_decision
    rng = random.Random(4)
    for _ in range(300):
        rows = [{"replicate": k, "test_examples": 1_000_000,
                 "perturbed_correct": rng.choice([495_100, 497_000, 500_000, 503_000, 504_400, 505_000, 505_300, 506_500]),
                 "baseline_correct": rng.choice([899_200, 900_500, 901_500, 918_000]),
                 "mcnemar_b": 46_000, "mcnemar_c": 459_000} for k in range(5)]
        f = {"rows": rows, "planned_replicates": 5, "chance": 0.5, "margin": 0.005, "alpha": 0.05,
             "effect_threshold": 0.01, "baseline_threshold": 0.9}
        t = evaluate_near_chance_decision(rows, planned_replicates=5, chance=0.5, margin=0.005, alpha=0.05,
                                          effect_threshold=0.01, baseline_threshold=0.9, paired_endpoint_role="secondary")
        mine = V.d4_decision(f)
        assert mine == {"d4_outcome": t["outcome"], "primary": t["primary"]["outcome"], "baseline": t["baseline"]["outcome"],
                        "fit_states": [p["state"] for p in t["primary"]["replicates"]]}


def test_d2n_against_classify():
    from audit.dataset.experiments.d2n_fresh_draw_null import classify
    rng = random.Random(2)
    for _ in range(2000):
        null = [rng.choice([0, 0, 0, 1, 2, 5]) for _ in range(59)]
        obs = rng.choice([0, 1, 2, 5, 6, 9, 40])
        assert V.d2n(obs, null, 0.05 / 3, 1.25) == classify(obs, null, alpha_per=0.05 / 3, ratio_gate=1.25)["outcome"]


def test_exact_neighbour_count_against_brute_force():
    rng = np.random.default_rng(8)
    x = rng.integers(0, 2, size=(300, 64), dtype=np.uint8)
    x[10] = x[20]
    x[10, 3] ^= 1
    brute = sum(any(int(np.sum(x[i] != x[j])) <= 8 for j in range(len(x)) if j != i) for i in range(len(x)))
    assert V.rows_with_neighbour(x, 8) == brute >= 2
