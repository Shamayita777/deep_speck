"""
SDB-V2 case registry (23 cases) and the real-component adapters that execute them.

Each case names the level(s) it tests:
  C       component conformance: the real framework component returns the registered output
  D       dimension-rule conformance: the manuscript's roll-up applied to that output, inside a fixture
  R       claim-rule conformance: the V2 engine on declared or derived dimension inputs
  replay  H2 only: recorded historical values reproduced from raw evidence
No case establishes real dimension or claim evidence about Gohr or AutoND.

Adapters call the frozen components unchanged; they never reimplement a decision. Expected
outputs are fixed in the registration before execution (H2's are copied from the A-D4 certificate
by `h2_expected`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from audit.framework_validation.sdb_v2 import engine_v2 as E

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
import os
FIX = Path(os.environ.get("SDB_V2_FIXTURES_ROOT", HERE / "fixtures"))   # override only for the smoke run

GROUPS = {
    "H1": "historical_defect", "H2": "historical_replay",
    **{f"S{i}": "synthetic_defect" for i in range(1, 7)},
    **{f"K{i}": "clean_control" for i in range(2, 9)},
    "P1": "positive_decision_rule_control",
    **{f"U{i}": "abstention_control" for i in range(1, 8)},
}
LEVELS = {"H1": ["C"], "H2": ["replay", "D", "R"], "S1": ["C", "D", "R"], "S2": ["C"], "S3": ["C", "D", "R"],
          "S4": ["C"], "S5": ["C"], "S6": ["C", "D", "R"], **{f"K{i}": ["C"] for i in range(2, 9)}, "P1": ["R"],
          "U1": ["C", "D", "R"], "U2": ["R"], "U3": ["R"], "U4": ["C"], "U5": ["R"], "U6": ["C", "D", "R"],
          "U7": ["R"]}
ORDER = ["H1", "H2", "S1", "S2", "S3", "S4", "S5", "S6", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "P1",
         "U1", "U2", "U3", "U4", "U5", "U6", "U7"]
assert len(ORDER) == 23 and set(ORDER) == set(GROUPS) == set(LEVELS)


def bundle(case: str) -> dict:
    return json.loads((FIX / case / "bundle.json").read_text())


#: Fixture claim template (registered): R_c = {dim}, the other dimensions N/A, L* = L2, reproduction
#: COMPLETE and predictive advantage declared. Used only to exercise roll-up and precedence.
FIXTURE_CLAIM = {"L": 2, "reproduction": "COMPLETE", "predictive": True, "others": "N/A"}


def _fixture_claim(case: str, dim: str, outcome: str) -> dict:
    dims = {d: E.DimRecord(E.Dim(FIXTURE_CLAIM["others"])) for d in E.DIMENSIONS if d != dim}
    dims[dim] = E.DimRecord(E.Dim(outcome))
    d = E.decide(E.Component(case, frozenset({dim}), FIXTURE_CLAIM["L"], FIXTURE_CLAIM["reproduction"],
                             FIXTURE_CLAIM["predictive"], dims))
    return {"verdict": d.verdict.value, "rule": d.rule.split(":")[0]}


def _decision(d: E.Decision) -> dict:
    return {"verdict": d.verdict.value, "rule": d.rule.split(":")[0], "level": d.level, "blocking": d.blocking}


# --- dimension roll-ups, each the manuscript's written rule -------------------------------------------

def rollup_ii(m1: dict) -> str:
    """l.929: FAIL when a required criterion is violated. The fixture registers all three M1 checks."""
    return "FAIL" if any(m1[k] == "FAIL" for k in ("published_vector", "key_schedule", "round_traces")) else "NOT_ROLLED_UP"


def rollup_di_d1(outcome: str) -> str:
    """l.2002: DI is FAIL if any experiment gives FAIL. A single D1 PASS is not rolled up to DI PASS."""
    return "FAIL" if outcome == "FAIL" else "NOT_ROLLED_UP"


def rollup_di_d4(outcome: str) -> str:
    """l.2002: success needs every shuffled fit in the band and every baseline adequate (PASS); a fit
    outside the band or a baseline below the floor is FAIL; anything else INCONCLUSIVE."""
    return {"SUPPORTED": "PASS", "NOT_SUPPORTED": "FAIL", "BASELINE_INADEQUATE": "FAIL"}.get(outcome, "INCONCLUSIVE")


def rollup_ev(decision: str) -> str:
    """l.2003: EQUIVALENT gives PASS and EFFECT gives FAIL. A single equivalence is not rolled up to EV PASS
    (K2 is component-level); EFFECT and INCONCLUSIVE are."""
    return {"EFFECT": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE"}.get(decision, "NOT_ROLLED_UP")


# --- adapters --------------------------------------------------------------------------------------------

def _checkpoint(case):
    from audit.cryptography.ce234.production import PreflightError, verify_reference_model
    from audit.implementation.gohr_adapter import probe_checkpoint_architecture
    b = bundle(case)
    path = REPO / b["model"]
    probe = probe_checkpoint_architecture(path).to_dict()
    raw = {"probe": probe}
    try:
        rep = verify_reference_model(checkpoint=path, behavioural=True)
        out = {"binding": "ACCEPTED", "behavioural_verdict": rep.get("behavioural_binding", {}).get("verdict")}
        raw["binding_report"] = rep
    except PreflightError as exc:
        msg = str(exc)
        out = {"binding": "REFUSED", "binding_refusal_kind": "sha256" if "sha256" in msg else "other"}
        raw["binding_refusal"] = msg
    out.update({"probe_n_conv1d": probe["n_conv1d_layers"], "probe_n_add": probe["n_add_layers"],
                "probe_derived_depth": probe["derived_depth"]})
    return {"C": out}, raw


def _speck(case):
    from audit.implementation.ii2_independent import verify_m1
    b = bundle(case)
    if case == "K4":
        ref = verify_m1.load_reference_speck()
    else:
        spec = importlib.util.spec_from_file_location("sdb_v2_s1_mutated_speck", REPO / b["cipher_module"])
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
    out, raw = {}, {}
    for key, fn in (("published_vector", verify_m1.test_published_vector),
                    ("key_schedule", verify_m1.compare_key_schedules),
                    ("round_traces", verify_m1.compare_round_traces)):
        try:
            raw[key] = fn(ref)
            out[key] = raw[key].get("status", "PASS")
        except (AssertionError, RuntimeError) as exc:
            out[key], raw[key] = "FAIL", {"error": f"{type(exc).__name__}: {exc}"}
    res = {"C": out}
    if case == "S1":
        ii = rollup_ii(out)
        res["D"] = {"II": ii}
        res["R"] = _fixture_claim(case, "II", ii)
    return res, raw


def _d1(case):
    from audit.dataset.d1_duplicate_detection import audit_duplicates, evaluate_d1
    p = np.load(REPO / bundle(case)["partitions"], allow_pickle=False)
    results = audit_duplicates(p["train"], p["validation"], p["test"], p["train_y"], p["validation_y"], p["test_y"])
    ev = evaluate_d1(results)
    res = {"C": {"d1_outcome": ev["outcome"]}}
    if case == "S3":
        di = rollup_di_d1(ev["outcome"])
        res["D"] = {"DI": di}
        res["R"] = _fixture_claim(case, "DI", di)
    return res, {"audit_duplicates": results, "evaluate_d1": ev}


def _d2n(case):
    from audit.dataset.experiments.d2n_fresh_draw_null import classify, scan
    b = bundle(case)
    cfg = b["d2n"]
    test = np.load(REPO / b["partitions"], allow_pickle=False)["test"]
    r = cfg["radius"]
    obs = scan(test, radii=[r], queries=cfg["queries"], seed=cfg["query_seed_base"])
    nulls = []
    for i in range(cfg["null_draws"]):
        x = np.random.default_rng(cfg["null_seed_base"] + i).integers(0, 2, size=test.shape, dtype=np.uint8)
        nulls.append(scan(x, radii=[r], queries=cfg["queries"], seed=cfg["query_seed_base"] + 1 + i))
    observed = obs["per_radius"][str(r)]["hits"]
    null_hits = [n["per_radius"][str(r)]["hits"] for n in nulls]
    c = classify(observed, null_hits, alpha_per=cfg["alpha_per"], ratio_gate=cfg["ratio_gate"])
    return {"C": {"d2n_outcome": c["outcome"]}}, {"observed_scan": obs, "null_scans": nulls, "classify": c,
                                                  "observed_hits": observed, "null_hits": null_hits}


def _progression_fixture(case):
    from audit.dataset.adapters.autond_d4 import progression
    from audit.dataset.experiments.d4_autond_label_shuffle import progression_criterion
    f = json.loads((REPO / bundle(case)["records"]).read_text())
    fits, raw = [], []
    for recs in f["fits"]:
        pr = progression(recs, rounds=f["rounds"], epochs_per_round=f["epochs_per_round"], threshold=f["threshold"])
        fits.append({"condition": recs[0]["condition"], "replicate": recs[0]["replicate"],
                     "best_round": pr["best_round"], "stop_round": pr["stop_round"]})
        raw.append({k: v for k, v in pr.items() if k != "summaries"} | {"condition": recs[0]["condition"],
                                                                         "replicate": recs[0]["replicate"]})
    crit = progression_criterion({"fits": fits}, f["starting_round"])
    pick = lambda c, k: [x[k] for x in fits if x["condition"] == c]
    return {"C": {"perturbed_best_rounds": pick("perturbed", "best_round"),
                  "perturbed_stop_rounds": pick("perturbed", "stop_round"),
                  "baseline_best_rounds": pick("baseline", "best_round"),
                  "baseline_stop_rounds": pick("baseline", "stop_round"),
                  "criterion_met": crit["met"],
                  "perturbed_component": crit["perturbed_stop_at_starting_round_without_best_round"],
                  "baseline_component": crit["baseline_pass_the_starting_round"]}}, {"fits": raw, "criterion": crit}


def _ev(case):
    sys.path.insert(0, str(REPO / "audit/experimental_validity"))
    from framework.certificate import (DecisionState, assess_practical_equivalence, decide_difference_detection,
                                       decide_final)
    from framework.experiment import PracticalSignificance
    from framework.failures import FailureReason, ReplicateOutcome, ReplicateStatus
    from framework.replication import ReplicateSet
    from framework.seeds import statistics_rng
    from framework.statistics import paired_analysis
    f = json.loads((REPO / bundle(case)["records"]).read_text())
    sets = {"a": ReplicateSet("a"), "b": ReplicateSet("b")}
    for u in f["units"]:
        for k in ("a", "b"):
            if u["status"] == "valid":
                sets[k].add(ReplicateOutcome(u["unit"], k, u["unit"], ReplicateStatus.VALID, metric_value=u[k]))
            else:
                sets[k].add(ReplicateOutcome(u["unit"], k, u["unit"], ReplicateStatus.FAILED,
                                             failure_reason=FailureReason(u["failure_reason"])))
    # the frozen AutoND EV chain (experiments._certificate): sufficiency first, then the paired decision
    sufficient, summary = sets["a"].check_sufficiency(max(f["minimum_valid"], 2))
    raw: dict[str, Any] = {"sufficiency": summary}
    decision = DecisionState.INCONCLUSIVE
    if sufficient:
        ps = PracticalSignificance(f["epsilon"], f["epsilon_predeclared"], "SDB-V2 fixture margin")
        paired = paired_analysis(sets["a"].valid_metric_array(), sets["b"].valid_metric_array(),
                                 rng=statistics_rng(f["statistics_seed"]))
        assessment = assess_practical_equivalence(paired, ps)
        provisional = decide_difference_detection(adjusted_p_value=paired.p_value, alpha=f["alpha"],
                                                  sufficient_replicates=True)
        decision = decide_final(difference_decision=provisional, practical_equivalence=assessment,
                                confidence_interval={"low": paired.ci_low, "high": paired.ci_high}, epsilon=ps.threshold)
        raw.update({"paired": paired.to_dict(), "assessment": assessment, "difference": provisional.value})
    name = {"SUPPORTED": "EFFECT", "NOT_SUPPORTED": "EQUIVALENT"}.get(decision.value, decision.value)
    raw["decide_final"] = decision.value
    res = {"C": {"ev_decision": name, "sufficient": bool(sufficient)}}
    if case in ("S6", "U6"):
        ev = rollup_ev(name)
        res["D"] = {"EV": ev}
        res["R"] = _fixture_claim(case, "EV", ev)
    return res, raw


def _d4(case):
    from audit.dataset.d4_controlled_perturbation import evaluate_near_chance_decision
    f = json.loads((REPO / bundle(case)["records"]).read_text())
    dec = evaluate_near_chance_decision(f["rows"], planned_replicates=f["planned_replicates"], chance=f["chance"],
                                        margin=f["margin"], alpha=f["alpha"], effect_threshold=f["effect_threshold"],
                                        baseline_threshold=f["baseline_threshold"],
                                        paired_endpoint_role=f["paired_endpoint_role"])
    di = rollup_di_d4(dec["outcome"])
    return {"C": {"d4_outcome": dec["outcome"], "primary": dec["primary"]["outcome"],
                  "baseline": dec["baseline"]["outcome"],
                  "fit_states": [p["state"] for p in dec["primary"]["replicates"]]},
            "D": {"DI": di}, "R": _fixture_claim(case, "DI", di)}, {"decision": dec}


def _par1(case):
    from audit.implementation.ii4_parallel import verify as V
    rep = V.verify(REPO / bundle(case)["run_dir"], repo_root=REPO, require_dual_gpu=True)
    out = {"par1_verifier": rep["status"]}
    if case == "U4":
        out["problem_mentions"] = ("analysis.json changed after the run"
                                   if any("analysis.json changed after the run" in p for p in rep["problems"])
                                   else "other")
    return {"C": out}, {"verifier_report": rep}


def _field(spec: dict):
    obj = json.loads((REPO / spec["certificate"]).read_text())
    for k in spec["field"]:
        obj = obj[k]
    return obj


def component_from_declared(c: dict) -> tuple[E.Component, dict]:
    """A Component from a declared.json entry; certificate-sourced fields are read, never asserted."""
    read = {}
    tests = []
    for t in c["ce"]:
        outcome = t.get("outcome")
        if "outcome_from" in t:
            outcome = read[f"{t['id']}.outcome"] = _field(t["outcome_from"])
        control = t.get("control", False)
        if "control_from" in t:
            v = read[f"{t['id']}.control_field"] = _field(t["control_from"])
            cf = t["control_from"]
            control = (v == cf["control_satisfied_if"]) if "control_satisfied_if" in cf else \
                (v != cf["control_satisfied_unless"])
        tests.append(E.CETest(t["id"], E.CE(outcome), required=t["required"], preregistered=t["prereg"],
                              adequately_powered=t["powered"], satisfied_by_executed_control=control))
    dims = {d: E.DimRecord(E.Dim(v)) for d, v in c["dims"].items()}
    return E.Component(c["id"], frozenset(c["required"]), c["L"], c["reproduction"], c["predictive"], dims,
                       tuple(tests)), read


def _engine_case(case):
    entries = json.loads((FIX / case / "declared.json").read_text())["asserted_inputs"]
    out, raw = {}, {}
    for c in entries:
        comp, read = component_from_declared(c)
        out[c["id"]] = _decision(E.decide(comp))
        raw[c["id"]] = {"certificate_fields_read": read}
    if case == "P1":
        res = out["P1"] | {"input_e6_ce2_admission": raw["P1"]["certificate_fields_read"]["CE2.control_field"]}
    elif case == "U2":
        r = raw["U2"]["certificate_fields_read"]
        res = out["U2"] | {"input_a3_ce3_scientific_decision": r["CE3.outcome"],
                           "input_a3_ce3_adversarial_outcome": r["CE3.control_field"]}
    elif case == "U7":
        res = out
    else:
        res = out[case]
    return {"R": res}, raw


# --- H2 ----------------------------------------------------------------------------------------------------

H2_BUNDLE = Path(os.environ.get("SDB_V2_H2_BUNDLE", HERE / "evidence" / "h2_ad4"))
AD4_CERT = REPO / "audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/result/d4_certificate.json"
A1_RECORDED = {  # manuscript l.1535, l.1586: A1's recorded dimension outcomes under each protocol
    "v1": {"II": "FAIL", "DI": "INCONCLUSIVE", "EV": "FAIL"},
    "v2": {"II": "INCONCLUSIVE", "DI": "INCONCLUSIVE", "EV": "FAIL"},
}


def _h2_fit_records() -> dict[str, list]:
    fits: dict[str, list] = {}
    for p in sorted((H2_BUNDLE / "fits").glob("replicate_*/*/stage_r*/*.metrics.json")):
        r = json.loads(p.read_text())
        fits.setdefault(f"{r['condition']}#{r['replicate']}", []).append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in fits.items()}


def h2_expected() -> dict:
    """Copied from the A-D4 certificate by script, never retyped."""
    cert = json.loads(AD4_CERT.read_text())
    fits = {}
    for row in cert["findings"]["replicates"]:
        for cond in ("baseline", "perturbed"):
            sel = row[f"{cond}_selection"]
            pr = sel["progression"]
            fits[f"{cond}#{row['replicate']}"] = {
                "selected_stage_round": sel["selected_stage_round"], "selected_epoch": sel["selected_epoch"],
                "best_round": pr["best_round"], "stop_round": pr["stop_round"], "status": pr["status"],
                "rounds_trained": pr["rounds_trained"], "rounds_max_val_acc": pr["rounds_max_val_acc"]}
    rescoring = {}
    for row in cert["findings"]["replicates"]:
        for cond in ("baseline", "perturbed"):
            m = row[f"{cond}_predictions"]
            rescoring[m["file"]] = {"correct": row[f"{cond}_correct"], "test_examples": row["test_examples"],
                                    "label_0_correct": m["class_counts"]["label_0"]["correct"],
                                    "label_1_correct": m["class_counts"]["label_1"]["correct"]}
    crit = cert["autond_decision"]["progression_criterion"]
    return {"replay": {"fits": fits,
                       "criterion": {"met": crit["met"],
                                     "perturbed": crit["perturbed_stop_at_starting_round_without_best_round"],
                                     "baseline": crit["baseline_pass_the_starting_round"]},
                       "engine_overall_d4_outcome": cert["decision"]["overall_d4_outcome"],
                       "autond_overall_outcome": cert["autond_decision"]["autond_overall_outcome"],
                       "baseline_adequacy_gate": cert["decision"]["baseline_adequacy_gate"],
                       "perturbed_passed_round5_and_stopped_round6": 3},
            "D": {"DI_for_A1": "INCONCLUSIVE"},
            "R": {"v1": {"verdict": "NOT_SUPPORTED", "blocking": ["EV", "II"]},
                  "v2": {"verdict": "NOT_SUPPORTED", "blocking": ["EV"]}},
            "B1S": rescoring}


def _h2(case):
    from audit.dataset.adapters.autond_d4 import progression
    from audit.dataset.experiments.d4_autond_label_shuffle import autond_decision, pipeline_progression, \
        progression_criterion
    cert = json.loads(AD4_CERT.read_text())
    recs = _h2_fit_records()
    fits, crit_in, raw = {}, [], {}
    for key, rs in sorted(recs.items()):
        pr = progression(rs, rounds=[5, 6, 7, 8, 9], epochs_per_round=10, threshold=0.505)
        first = pr["summaries"][0]
        fits[key] = {"selected_stage_round": first["stage_round"], "selected_epoch": first["selected_epoch"],
                     "best_round": pr["best_round"], "stop_round": pr["stop_round"], "status": pr["status"],
                     "rounds_trained": pr["rounds_trained"], "rounds_max_val_acc": pr["rounds_max_val_acc"]}
        cond, rep = key.split("#")
        crit_in.append({"condition": cond, "replicate": int(rep), "best_round": pr["best_round"],
                        "stop_round": pr["stop_round"]})
        raw[key] = {"n_records": len(rs), "selected_checkpoint": first["checkpoint"],
                    "selected_checkpoint_sha256": first["checkpoint_sha256"]}
    crit = progression_criterion({"fits": crit_in}, 5)
    pp = pipeline_progression(cert, "co_primary")
    dec = autond_decision(cert, "co_primary", 5)
    n36 = sum(1 for k, f in fits.items() if k.startswith("perturbed") and f["best_round"] == 5 and f["stop_round"] == 6)
    replay = {"fits": fits, "criterion": {"met": crit["met"],
                                          "perturbed": crit["perturbed_stop_at_starting_round_without_best_round"],
                                          "baseline": crit["baseline_pass_the_starting_round"]},
              "engine_overall_d4_outcome": dec["engine_overall_d4_outcome"],
              "autond_overall_outcome": dec["autond_overall_outcome"],
              "baseline_adequacy_gate": cert["decision"]["baseline_adequacy_gate"],
              "perturbed_passed_round5_and_stopped_round6": n36}
    di = rollup_di_d4(dec["autond_overall_outcome"])
    claim = {}
    for proto, dims in A1_RECORDED.items():
        d = E.decide(E.Component("A1", frozenset(E.DIMENSIONS), 2, "COMPLETE", True,
                                 {k: E.DimRecord(E.Dim(v)) for k, v in dims.items()}))
        claim[proto] = {"verdict": d.verdict.value, "blocking": d.blocking}
    return ({"replay": replay, "D": {"DI_for_A1": di}, "R": claim},
            {"records_per_fit": raw, "criterion_recomputed": crit, "pipeline_progression_on_certificate": pp,
             "autond_decision_on_certificate": dec,
             "claim_mapping_provenance": ("A-D4 bears on A1 only under v2, first recorded after the A-D4 result "
                                          "(D2-N registration 5fb47d3, 2026-11-05; A-D4 evidence ad8c5d8, 2026-11-04). "
                                          "Under v1 every AutoND DI experiment bears on A1 and A2. Not preregistered.")})


ADAPTERS = {"H1": _checkpoint, "S2": _checkpoint, "K3": _checkpoint, "S1": _speck, "K4": _speck, "S3": _d1,
            "K5": _d1, "S4": _d2n, "K6": _d2n, "S5": _progression_fixture, "K8": _progression_fixture, "S6": _ev,
            "K2": _ev, "U6": _ev, "U1": _d4, "K7": _par1, "U4": _par1, "H2": _h2,
            **{c: _engine_case for c in ("P1", "U2", "U3", "U5", "U7")}}


def expected_outcomes() -> dict:
    """The registered expected response of every case, fixed before execution."""
    X = {
        "H1": {"C": {"probe_n_conv1d": 11, "probe_n_add": 5, "probe_derived_depth": 5, "binding": "REFUSED",
                     "binding_refusal_kind": "sha256"}},
        "S1": {"C": {"published_vector": "FAIL", "key_schedule": "FAIL", "round_traces": "FAIL"},
               "D": {"II": "FAIL"}, "R": {"verdict": "NOT_SUPPORTED", "rule": "1"}},
        "S2": {"C": {"binding": "REFUSED", "binding_refusal_kind": "sha256"}},
        "S3": {"C": {"d1_outcome": "FAIL"}, "D": {"DI": "FAIL"}, "R": {"verdict": "NOT_SUPPORTED", "rule": "1"}},
        "S4": {"C": {"d2n_outcome": "EXCESS"}},
        "S5": {"C": {"perturbed_best_rounds": [5] * 5, "perturbed_stop_rounds": [6] * 5,
                     "baseline_best_rounds": [8] * 5, "baseline_stop_rounds": [9] * 5, "criterion_met": False,
                     "perturbed_component": False, "baseline_component": True}},
        "S6": {"C": {"ev_decision": "EFFECT", "sufficient": True}, "D": {"EV": "FAIL"},
               "R": {"verdict": "NOT_SUPPORTED", "rule": "1"}},
        "K2": {"C": {"ev_decision": "EQUIVALENT", "sufficient": True}},
        "K3": {"C": {"binding": "ACCEPTED", "behavioural_verdict": "PASS", "probe_n_conv1d": 21, "probe_n_add": 10}},
        "K4": {"C": {"published_vector": "PASS", "key_schedule": "PASS", "round_traces": "PASS"}},
        "K5": {"C": {"d1_outcome": "PASS"}},
        "K6": {"C": {"d2n_outcome": "NO_EXCESS"}},
        "K7": {"C": {"par1_verifier": "VERIFIED"}},
        "K8": {"C": {"perturbed_best_rounds": [None] * 5, "perturbed_stop_rounds": [5] * 5,
                     "baseline_best_rounds": [8] * 5, "baseline_stop_rounds": [9] * 5, "criterion_met": True,
                     "perturbed_component": True, "baseline_component": True}},
        "P1": {"R": {"verdict": "SUPPORTED", "rule": "4", "level": 3, "input_e6_ce2_admission": "ADMITTED"}},
        "U1": {"C": {"d4_outcome": "INCONCLUSIVE", "primary": "INCONCLUSIVE", "baseline": "PASSED",
                     "fit_states": ["INDETERMINATE"] * 5},
               "D": {"DI": "INCONCLUSIVE"}, "R": {"verdict": "INCONCLUSIVE", "rule": "2"}},
        "U2": {"R": {"verdict": "INCONCLUSIVE", "rule": "3", "level": 2,
                     "input_a3_ce3_scientific_decision": "SUPPORTED", "input_a3_ce3_adversarial_outcome": "NOT_CAUGHT"}},
        "U3": {"R": {"verdict": "INCONCLUSIVE", "rule": "2"}},
        "U4": {"C": {"par1_verifier": "NOT_VERIFIED", "problem_mentions": "analysis.json changed after the run"}},
        "U5": {"R": {"verdict": "INCONCLUSIVE", "rule": "2", "blocking": ["EV: missing_prerequisite"]}},
        "U6": {"C": {"ev_decision": "INCONCLUSIVE", "sufficient": False}, "D": {"EV": "INCONCLUSIVE"},
               "R": {"verdict": "INCONCLUSIVE", "rule": "2"}},
        "U7": {"R": {"a": {"verdict": "INCONCLUSIVE", "rule": "3", "level": 2},
                     "b": {"verdict": "INCONCLUSIVE", "rule": "3", "level": 2}}},
    }
    h2 = h2_expected()
    X["H2"] = {k: h2[k] for k in ("replay", "D", "R")}
    X["_H2_B1S"] = h2["B1S"]
    return X


def matches(expected: Any, observed: Any) -> bool:
    """Every registered key must be present with an equal value; extra observed keys are ignored."""
    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(k in observed and matches(v, observed[k]) for k, v in expected.items())
    return expected == observed
