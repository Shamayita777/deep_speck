"""
CHANGE 1: parallel-execution safety for two hypothesis-level calibration
jobs (one per GPU).
CHANGE 2: prospective power for the ACTUAL joint final decision rule.

Mocked/lightweight only - no calibration or production experiment is run.
"""

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

EV = Path(__file__).resolve().parents[1]
CONFIGS = EV / "configs"
PRIMARY = ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_pp_{name}", EV / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# CHANGE 1 - parallel execution safety
# =====================================================================

def test_two_calibration_jobs_have_disjoint_output_roots():
    dirs = {h: yaml.safe_load((CONFIGS / f"gohr_ev_{n}_calibration.yaml").read_text())["output_dir"]
            for h, n in zip(PRIMARY, ("shuffle", "representation"))}
    assert len(set(dirs.values())) == 2
    a, b = dirs[PRIMARY[0]], dirs[PRIMARY[1]]
    assert not a.startswith(b) and not b.startswith(a)   # neither nests inside the other


def test_two_calibration_jobs_have_disjoint_seed_ranges():
    cfgs = [yaml.safe_load((CONFIGS / f"gohr_ev_{n}_calibration.yaml").read_text())
            for n in ("shuffle", "representation")]
    seeds = []
    for c in cfgs:
        base = c["base_model_seed"]
        seeds.append(set(range(base, base + c["requested_pairs"])))
    assert seeds[0].isdisjoint(seeds[1])


def test_state_files_are_per_output_dir_so_jobs_cannot_read_each_other():
    """Ledger, sidecar, firewall and datasets all live under the job's own root."""
    import inspect

    from gohr import experiments as ex
    for fn in (ex._resume_ledger_for, ex._pair_results_sidecar_path,
               ex._pair_dataset_manifest_path):
        params = list(inspect.signature(fn).parameters)
        assert params[0] == "output_dir"


def test_output_isolation_refuses_another_experiments_root():
    run_ev = _load("run_ev")
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle_calibration.yaml").read_text())
    cfg["output_dir"] = "results/calibration/ev_representation"     # the other job's root
    with pytest.raises(run_ev.ConfigNotFrozenError, match="reserved output root"):
        run_ev._check_output_isolation(cfg, "H-EV-SHUFFLE")


def test_output_isolation_accepts_a_jobs_own_root():
    run_ev = _load("run_ev")
    for hyp, name in zip(PRIMARY, ("shuffle", "representation")):
        cfg = yaml.safe_load((CONFIGS / f"gohr_ev_{name}_calibration.yaml").read_text())
        run_ev._check_output_isolation(cfg, hyp)        # must not raise


def test_run_lock_prevents_a_second_writer(tmp_path):
    run_ev = _load("run_ev")
    d = tmp_path / "job"
    run_ev._acquire_output_lock(d, "H-EV-SHUFFLE")
    assert (d / "RUN_LOCK").exists()
    with pytest.raises(run_ev.ConcurrentRunConflictError, match="locked"):
        run_ev._acquire_output_lock(d, "H-EV-SHUFFLE")


def test_run_lock_records_pid_experiment_and_gpu(tmp_path):
    run_ev = _load("run_ev")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        run_ev._acquire_output_lock(tmp_path / "j", "H-EV-REPRESENTATION")
        text = (tmp_path / "j" / "RUN_LOCK").read_text()
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert "pid=" in text and "H-EV-REPRESENTATION" in text and "gpu=1" in text


def test_two_jobs_can_hold_locks_on_their_own_roots(tmp_path):
    run_ev = _load("run_ev")
    run_ev._acquire_output_lock(tmp_path / "shuffle", "H-EV-SHUFFLE")
    run_ev._acquire_output_lock(tmp_path / "representation", "H-EV-REPRESENTATION")
    assert (tmp_path / "shuffle" / "RUN_LOCK").exists()
    assert (tmp_path / "representation" / "RUN_LOCK").exists()


@pytest.mark.parametrize("argv,expected", [
    (["run_ev.py", "cfg.yaml", "--gpu", "0"], "0"),
    (["run_ev.py", "cfg.yaml", "--gpu=1"], "1"),
    (["run_ev.py", "cfg.yaml"], None),
])
def test_gpu_pinning_is_explicit_and_deterministic(argv, expected, monkeypatch):
    """CUDA_VISIBLE_DEVICES must be set from --gpu BEFORE TensorFlow import."""
    run_ev = _load("run_ev")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    got = run_ev._pin_gpu_before_tensorflow_import()
    assert got == expected
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected


def test_gpu_pinning_runs_before_any_heavy_import():
    """The pin must execute at module top, ahead of gohr/tensorflow imports."""
    src = (EV / "scripts" / "run_ev.py").read_text()
    pin = src.index("_PINNED_GPU = _pin_gpu_before_tensorflow_import()")
    assert pin < src.index("from gohr")
    assert pin < src.index("from framework")


def test_gpu_is_not_a_config_field_so_resume_state_is_unaffected():
    """
    GPU choice must never enter the config: the resume ledger compares a
    config hash, so a config-level GPU field would invalidate an in-flight
    calibration (e.g. an existing pair0/pair1).
    """
    for name in ("shuffle_calibration", "representation_calibration"):
        cfg = yaml.safe_load((CONFIGS / f"gohr_ev_{name}.yaml").read_text())
        assert not any("gpu" in str(k).lower() or "cuda" in str(k).lower() for k in cfg)


def test_resume_config_hash_excludes_gpu_and_output_dir():
    import inspect

    from gohr import experiments as ex
    src = inspect.getsource(ex._run_paired_experiment)
    call = src[src.index("cfg_hash = _replicate_config_hash("):]
    call = call[:call.index(")")]
    for forbidden in ("gpu", "cuda", "output_dir"):
        assert forbidden not in call.lower()


def test_sequential_execution_remains_valid():
    """Nothing added is mandatory: --gpu is optional and defaults to unset."""
    run_ev = _load("run_ev")
    import argparse
    parser = argparse.ArgumentParser()
    # mirror of the production parser flags we depend on
    src = (EV / "scripts" / "run_ev.py").read_text()
    assert '"--gpu", default=None' in src
    assert "--confirm-execute" in src


# =====================================================================
# CHANGE 2 - joint decision-rule power
# =====================================================================

def _sigmas():
    return {"H-EV-SHUFFLE": 0.0049, "H-EV-REPRESENTATION": 0.0081}


def test_joint_simulation_reproduces_the_decision_semantics():
    from framework.power import simulate_joint_primary_family
    from framework.seeds import statistics_rng
    r = simulate_joint_primary_family(
        hypotheses=PRIMARY, sigmas=_sigmas(), true_effects={h: 0.0 for h in PRIMARY},
        n_pairs=20, alpha=0.05, epsilon=0.01, n_simulations=2000,
        rng=statistics_rng(1), target_effect_source="predeclared")
    for hyp in PRIMARY:
        # NOT_SUPPORTED requires BOTH conditions, so it can never exceed
        # either component - this is the bug the change fixes.
        assert r.power_not_supported[hyp] <= r.power_equivalence_after_holm[hyp]
        assert r.power_not_supported[hyp] <= 1.0 - r.power_difference_significant[hyp] + 1e-9
        # at true effect 0, SUPPORTED must be essentially unreachable
        assert r.power_supported[hyp] < 0.02


def test_joint_power_is_strictly_below_tost_only_power():
    """Proves the planner cannot silently revert to TOST-only power."""
    from framework.power import simulate_joint_primary_family
    from framework.seeds import statistics_rng
    r = simulate_joint_primary_family(
        hypotheses=PRIMARY, sigmas=_sigmas(), true_effects={h: 0.0 for h in PRIMARY},
        n_pairs=20, alpha=0.05, epsilon=0.01, n_simulations=3000,
        rng=statistics_rng(7), target_effect_source="predeclared")
    assert any(r.power_not_supported[h] < r.power_equivalence_after_holm[h] for h in PRIMARY)


def test_holm_two_is_the_exact_step_down_procedure():
    import numpy as np
    from framework.power import _holm_two
    p_a = np.array([0.01, 0.30, 0.04])
    p_b = np.array([0.30, 0.01, 0.04])
    adj_a, adj_b = _holm_two(p_a, p_b, 0.05)
    # smaller p adjusted to min(1, 2p); larger to max(that, p_larger)
    assert adj_a[0] == pytest.approx(0.02) and adj_b[0] == pytest.approx(0.30)
    assert adj_b[1] == pytest.approx(0.02) and adj_a[1] == pytest.approx(0.30)
    assert adj_a[2] == pytest.approx(0.08) and adj_b[2] == pytest.approx(0.08)


def test_planner_simulates_the_joint_rule_not_tost_alone(tmp_path):
    planner = _load("plan_ev_power")
    art = {"artifact": "ev-calibration-variance-v2", "non_evidentiary": True,
           "inputs": [{"experiment_id": h, "certificate_sha256": "a" * 64, "n_valid": 10}
                      for h in PRIMARY],
           "sigma_Delta_estimates": {"H-EV-SHUFFLE": 0.003, "H-EV-REPRESENTATION": 0.005},
           "sigma_Delta_upper_95": _sigmas()}
    ap = tmp_path / "v.json"; ap.write_text(json.dumps(art))
    plan = planner.build_power_plan(
        art, variance_artifact=ap, epsilon=0.01, target_effect=0.01,
        target_effect_source="predeclared", alpha=0.05, target_power=0.80,
        simulations=600, max_n=14)
    reqs = plan["requirements"]
    assert len(reqs) == 4
    eq = [r for r in reqs if r["procedure"] == "equivalence"]
    assert all("NOT_SUPPORTED" in r["decision_simulated"] for r in eq)
    assert all("equivalence-Holm" in r["decision_simulated"] for r in eq)
    assert all(r["multiplicity_simulated"] == "holm_exact" for r in reqs)
    # exact Holm simulated -> full alpha used, no alpha/2 approximation
    assert {r["alpha_used"] for r in reqs} == {0.05}
    assert "COMPLETE final decision rule" in plan["procedure_simulated"]
    assert plan["common_design"]["required_n"] == max(r["required_n"] for r in reqs)
    assert plan["inputs"]["sigma_source"] == "upper95"


def test_planner_is_deterministic_under_fixed_seeding(tmp_path):
    planner = _load("plan_ev_power")
    art = {"artifact": "ev-calibration-variance-v2", "non_evidentiary": True,
           "inputs": [{"experiment_id": h, "certificate_sha256": "a" * 64, "n_valid": 10}
                      for h in PRIMARY],
           "sigma_Delta_estimates": {"H-EV-SHUFFLE": 0.003, "H-EV-REPRESENTATION": 0.005},
           "sigma_Delta_upper_95": _sigmas()}
    ap = tmp_path / "v.json"; ap.write_text(json.dumps(art))
    kw = dict(variance_artifact=ap, epsilon=0.01, target_effect=0.01,
              target_effect_source="predeclared", alpha=0.05, target_power=0.80,
              simulations=400, max_n=10)
    a = planner.build_power_plan(art, **kw)
    b = planner.build_power_plan(art, **kw)
    assert [r["required_n"] for r in a["requirements"]] == \
           [r["required_n"] for r in b["requirements"]]
    assert [r["achieved_power"] for r in a["requirements"]] == \
           [r["achieved_power"] for r in b["requirements"]]


def test_production_gate_still_requires_exactly_50000_simulations():
    planner = _load("plan_ev_power")
    assert planner.FROZEN["simulations"] == 50_000
    assert planner.DEFAULT_SIMULATIONS == 50_000

    class _A:
        epsilon = 0.01; target_effect = 0.01; target_effect_source = "predeclared"
        alpha = 0.05; target_power = 0.80; simulations = 1000; use_sigma = "upper95"
    art = {"artifact": "ev-calibration-variance-v2", "non_evidentiary": True,
           "inputs": [{"experiment_id": h, "certificate_sha256": "a" * 64, "n_valid": 10}
                      for h in PRIMARY],
           "sigma_Delta_upper_95": _sigmas()}
    with pytest.raises(planner.FrozenDesignViolation, match="simulations"):
        planner._require_frozen(_A(), art)
    _A.simulations = 50_000
    planner._require_frozen(_A(), art)          # accepted only at the frozen budget


def test_frozen_statistical_parameters_unchanged():
    planner = _load("plan_ev_power")
    assert planner.FROZEN["epsilon"] == 0.01
    assert planner.FROZEN["target_effect"] == 0.01
    assert planner.FROZEN["alpha"] == 0.05
    assert planner.FROZEN["target_power"] == 0.80
    assert planner.FROZEN["sigma_source"] == "upper95"
    assert planner.FROZEN["calibration_pairs_per_hypothesis"] == 10
    assert tuple(planner.PRIMARY_FAMILY) == PRIMARY
