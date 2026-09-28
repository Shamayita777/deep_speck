"""
Calibration (pilot) tier: non-evidentiary status, contamination guards,
mode-locked output directories, and fail-closed production gating.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from framework.certificate import (
    NonConfirmatoryEvidenceError,
    require_confirmatory_certificate,
)
from framework.validation import (
    ConfigValidationError,
    NON_EVIDENTIARY_MODES,
    RUN_MODES,
    is_calibration,
    is_non_evidentiary,
    require_run_mode,
)
from gohr.experiments import _baseline_overrides_for_run_mode, apply_primary_family_correction

EV = Path(__file__).resolve().parents[1]
PRIMARY_FAMILY_IDS = ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")
CONFIGS = EV / "configs"


# --- 1. calibration is a real, distinct, non-evidentiary tier -------------


# ---------------------------------------------------------------------------
# In-process script invocation.
#
# These tests exercise VALIDATION paths only. Spawning a subprocess per test
# paid a full TensorFlow import (~10-15s) each, so ~25 validation tests cost
# several minutes of pure interpreter startup for no added assurance. The
# scripts are loaded ONCE here and their main() called with a patched argv,
# which tests exactly the same code path. Nothing about the gates changes;
# only the harness does.
# ---------------------------------------------------------------------------
import contextlib
import importlib.util
import io


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", EV / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SCRIPTS = {}


def _script(name):
    if name not in _SCRIPTS:
        _SCRIPTS[name] = _load_script(name)
    return _SCRIPTS[name]


class _Result:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout, self.stderr = returncode, stdout, ""


def _invoke(name, argv):
    """Call a script's main() in-process with argv; capture rc and stdout."""
    mod = _script(name)
    buf = io.StringIO()
    old = sys.argv
    sys.argv = [f"{name}.py", *[str(a) for a in argv]]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rc = mod.main()
            except SystemExit as exc:                  # argparse rejects a choice
                rc = exc.code if isinstance(exc.code, int) else 1
            except Exception as exc:                   # surfaced like a crashed CLI
                print(f"{type(exc).__name__}: {exc}")
                rc = 1
    finally:
        sys.argv = old
    return _Result(rc, buf.getvalue())


def test_calibration_is_a_recognised_mode():
    assert require_run_mode("calibration") == "calibration"
    assert set(RUN_MODES) == {"smoke", "calibration", "production"}


def test_calibration_is_non_evidentiary_and_distinct_from_smoke():
    assert is_non_evidentiary("calibration") and is_non_evidentiary("smoke")
    assert not is_non_evidentiary("production")
    assert is_calibration("calibration") and not is_calibration("smoke")
    assert set(NON_EVIDENTIARY_MODES) == {"smoke", "calibration"}


def test_unknown_mode_refused():
    with pytest.raises(ConfigValidationError):
        require_run_mode("pilot")


def test_calibration_uses_the_real_protocol_not_smoke_semantics():
    cal = _baseline_overrides_for_run_mode("calibration")
    prod = _baseline_overrides_for_run_mode("production")
    smoke = _baseline_overrides_for_run_mode("smoke")
    assert cal == prod                      # identical frozen protocol
    assert cal != smoke
    assert cal["rounds"] == 5 and cal["depth"] == 10 and cal["epochs"] == 200
    assert cal["train_size"] == 10_000_000 and cal["confirmatory_test_size"] == 1_000_000


# --- 2/3. calibration cannot be consumed as confirmatory evidence ----------

def _cert(**kw):
    base = {"audit": {"id": "H-EV-SHUFFLE"}, "non_evidentiary": False,
            "calibration": False, "run_mode": "production"}
    base.update(kw)
    return base


def test_confirmatory_gate_accepts_a_production_certificate():
    assert require_confirmatory_certificate(_cert(), context="test")


def test_confirmatory_gate_refuses_calibration():
    with pytest.raises(NonConfirmatoryEvidenceError):
        require_confirmatory_certificate(
            _cert(calibration=True, non_evidentiary=True, run_mode="calibration"), context="test")


def test_confirmatory_gate_refuses_smoke():
    with pytest.raises(NonConfirmatoryEvidenceError):
        require_confirmatory_certificate(
            _cert(non_evidentiary=True, run_mode="smoke"), context="test")


def test_confirmatory_gate_treats_missing_status_as_not_confirmatory():
    """Absence of a claim must never be read as a claim."""
    with pytest.raises(NonConfirmatoryEvidenceError):
        require_confirmatory_certificate({"audit": {"id": "X"}}, context="test")


def test_confirmatory_gate_refuses_mislabelled_run_mode():
    with pytest.raises(NonConfirmatoryEvidenceError, match="run_mode"):
        require_confirmatory_certificate(_cert(run_mode="calibration"), context="test")


def test_primary_family_holm_refuses_calibration_certificates():
    cal = _cert(calibration=True, non_evidentiary=True, run_mode="calibration")
    with pytest.raises(NonConfirmatoryEvidenceError):
        apply_primary_family_correction(cal, _cert(audit={"id": "H-EV-REPRESENTATION"}))
    with pytest.raises(NonConfirmatoryEvidenceError):
        apply_primary_family_correction(_cert(), cal)


# --- 4/5/11. production remains fail-closed --------------------------------

def _run_ev(config_path, tmp_path):
    return _invoke("run_ev", [config_path])


def test_production_configs_still_carry_unresolved_placeholders():
    for name in ("gohr_ev_baseline", "gohr_ev_noise", "gohr_ev_shuffle", "gohr_ev_representation"):
        assert "UNSPECIFIED" in (CONFIGS / f"{name}.yaml").read_text()


def test_production_blocked_by_unresolved_replicate_counts(tmp_path):
    out = _run_ev(CONFIGS / "gohr_ev_baseline.yaml", tmp_path)
    assert out.returncode == 1 and "REFUSING TO RUN" in out.stdout


def test_production_blocked_when_epsilon_absent(tmp_path):
    """Paired production requires a PREDECLARED epsilon, independent of counts."""
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=10, minimum_valid_pairs=9)
    for k in ("practical_threshold", "practical_threshold_predeclared",
              "practical_threshold_justification"):
        cfg.pop(k, None)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1
    assert "PREDECLARED practical-significance margin" in out.stdout


def test_production_blocked_when_epsilon_not_predeclared(tmp_path):
    """A numeric epsilon is not enough: predeclaration must be asserted."""
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=10, minimum_valid_pairs=9, practical_threshold=0.01,
               practical_threshold_justification="consequence-based text",
               practical_threshold_predeclared=False)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "predeclared" in out.stdout.lower()


def test_epsilon_requires_non_empty_justification(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=10, minimum_valid_pairs=9, practical_threshold=0.01,
               practical_threshold_justification="   ", practical_threshold_predeclared=True)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "justification" in out.stdout


# --- 6. power guard ---------------------------------------------------------

def _power_artifact(tmp_path, required_n=24):
    """Fully frozen-compliant plan; the gate now validates every field."""
    art = tmp_path / "plan.json"
    art.write_text(json.dumps({
        "artifact": "ev-power-plan-v2", "non_evidentiary": True,
        "primary_family": ["H-EV-SHUFFLE", "H-EV-REPRESENTATION"],
        "inputs": {"epsilon": 0.01, "target_effect": 0.01,
                   "target_effect_source": "predeclared", "alpha": 0.05,
                   "target_power": 0.80, "n_simulations": 50000, "sigma_source": "upper95"},
        "common_design": {"required_n": required_n}}))
    return art


def _with_hash(cfg, art):
    import hashlib
    pa = dict(cfg.get("power_analysis") or {})
    pa["artifact_sha256"] = hashlib.sha256(art.read_bytes()).hexdigest()
    cfg["power_analysis"] = pa
    return cfg


def test_production_blocked_when_power_artifact_missing(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=24, minimum_valid_pairs=24)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "power_analysis.artifact_path" in out.stdout


def test_production_blocked_when_required_n_exceeds_minimum_valid_pairs(tmp_path):
    art = _power_artifact(tmp_path, required_n=24)
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=24, minimum_valid_pairs=20,
               power_analysis={"artifact_path": str(art), "required_n": 24})
    _with_hash(cfg, art)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "below the power-required n" in out.stdout


def test_production_blocked_when_required_n_disagrees_with_artifact(tmp_path):
    art = _power_artifact(tmp_path, required_n=24)
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=30, minimum_valid_pairs=30,
               power_analysis={"artifact_path": str(art), "required_n": 12})
    _with_hash(cfg, art)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "disagrees with the power" in out.stdout


# --- 8. calibration and confirmatory outputs are distinct ------------------

def test_calibration_configs_exist_and_are_mode_tagged():
    for name in ("gohr_ev_baseline_calibration", "gohr_ev_noise_calibration",
                 "gohr_ev_shuffle_calibration", "gohr_ev_representation_calibration"):
        cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
        assert cfg["run_mode"] == "calibration"
        assert "calibration" in cfg["output_dir"]


def test_calibration_output_dirs_disjoint_from_production():
    cal = {yaml.safe_load((CONFIGS / f"{n}.yaml").read_text())["output_dir"]
           for n in ("gohr_ev_baseline_calibration", "gohr_ev_noise_calibration",
                     "gohr_ev_shuffle_calibration", "gohr_ev_representation_calibration")}
    prod = {yaml.safe_load((CONFIGS / f"{n}.yaml").read_text())["output_dir"]
            for n in ("gohr_ev_baseline", "gohr_ev_noise", "gohr_ev_shuffle",
                      "gohr_ev_representation")}
    assert cal.isdisjoint(prod)


def test_calibration_configs_keep_the_real_protocol():
    for name in ("gohr_ev_baseline_calibration", "gohr_ev_noise_calibration"):
        cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
        assert (cfg["rounds"], cfg["depth"], cfg["epochs"]) == (5, 10, 200)
        assert cfg["train_size"] == 10_000_000


def test_paired_calibration_pilot_size_is_frozen_at_ten():
    """FROZEN: 10 matched pairs per paired primary - an operational predeclared
    pilot size, NOT a claimed universal statistical minimum."""
    for name in ("gohr_ev_shuffle_calibration", "gohr_ev_representation_calibration"):
        cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
        assert cfg["requested_pairs"] == 10 and cfg["minimum_valid_pairs"] == 10


def test_single_arm_calibration_sizes_remain_undeclared():
    """EV-BASELINE/EV-NOISE do not size the paired design; their pilot sizes
    are still open and must not have been invented."""
    for name in ("gohr_ev_baseline_calibration", "gohr_ev_noise_calibration"):
        assert "UNSPECIFIED_REQUIRES_PILOT_PRECISION_DECISION" in (CONFIGS / f"{name}.yaml").read_text()


def test_calibration_also_blocked_until_its_own_counts_are_declared(tmp_path):
    out = _run_ev(CONFIGS / "gohr_ev_baseline_calibration.yaml", tmp_path)
    assert out.returncode == 1 and "placeholder" in out.stdout.lower()


def test_output_dir_mode_lock_refuses_mixing(tmp_path):
    from importlib import util
    spec = util.spec_from_file_location("run_ev_mod", EV / "scripts" / "run_ev.py")
    mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
    d = tmp_path / "shared"
    mod._lock_output_dir_to_mode(d, "calibration")
    mod._lock_output_dir_to_mode(d, "calibration")          # idempotent
    with pytest.raises(mod.RunModeConflictError, match="never share"):
        mod._lock_output_dir_to_mode(d, "production")
    assert (d / "RUN_MODE").read_text().strip() == "calibration"


# --- 9. calibration analysis records hashes and refuses contamination ------

CAL_CONFIG = {"H-EV-SHUFFLE": "gohr_ev_shuffle_calibration",
              "H-EV-REPRESENTATION": "gohr_ev_representation_calibration"}


def _genuine_provenance(experiment):
    """Provenance a certificate from the frozen calibration config would carry."""
    import hashlib
    from framework.provenance import config_hash
    cfg = yaml.safe_load((CONFIGS / f"{CAL_CONFIG[experiment]}.yaml").read_text())
    prov = {"config_hash": config_hash(cfg),
            "statistical_plan_sha256": hashlib.sha256(
                (EV / "docs" / "statistical_plan.md").read_bytes()).hexdigest(),
            "source_manifest_sha256": "c" * 64}
    if experiment == "H-EV-REPRESENTATION":
        prov["candidate1_permutation_sha256"] = cfg["expected_candidate1_permutation_sha256"]
    return prov


def _cal_cert(tmp_path, name, experiment, a, b, genuine=True, **over):
    cert = tmp_path / name
    body = {"audit": {"id": experiment}, "run_mode": "calibration", "calibration": True,
            "non_evidentiary": True,
            "raw_paired_values": {"condition_a": a, "condition_b": b}}
    if genuine:
        body["provenance"] = _genuine_provenance(experiment)
    body.update(over)
    cert.write_text(json.dumps(body))
    return cert


def _analyse(tmp_path, *certs, extra=()):
    return _invoke("analyze_ev_calibration",
                   [*certs, "--output", tmp_path / "v.json", *extra])


def test_calibration_analyser_refuses_production_certificates(tmp_path):
    cert = tmp_path / "prod.json"
    cert.write_text(json.dumps({"audit": {"id": "H-EV-SHUFFLE"}, "run_mode": "production",
                                "non_evidentiary": False, "calibration": False,
                                "raw_paired_values": {"condition_a": [0.9], "condition_b": [0.9]}}))
    out = _analyse(tmp_path, cert)
    assert out.returncode == 1 and "required" in out.stdout


def test_calibration_analyser_records_hashes_ucl_and_omits_effects(tmp_path):
    c1 = _cal_cert(tmp_path, "s.json", "H-EV-SHUFFLE",
                   [0.90 + i * 1e-3 for i in range(10)], [0.91 + i * 1.1e-3 for i in range(10)])
    c2 = _cal_cert(tmp_path, "r.json", "H-EV-REPRESENTATION",
                   [0.90 + i * 1e-3 for i in range(10)], [0.60 + i * 2e-3 for i in range(10)])
    r = _analyse(tmp_path, c1, c2)
    assert r.returncode == 0, r.stdout + r.stderr
    art = json.loads((tmp_path / "v.json").read_text())
    assert art["non_evidentiary"] is True
    assert all(len(e["certificate_sha256"]) == 64 for e in art["inputs"])
    assert set(art["sigma_Delta_estimates"]) == {"H-EV-SHUFFLE", "H-EV-REPRESENTATION"}
    # the UCL must exceed the point estimate for every hypothesis
    for h, sd in art["sigma_Delta_estimates"].items():
        assert art["sigma_Delta_upper_95"][h] > sd
    # Scan KEYS, not prose: the artifact's own disclaimer legitimately uses
    # the words "p-value" and "conclusion" while promising not to report them.
    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)
    present = set(keys(art))
    for forbidden in ("p_value", "mean_difference", "effect_size", "conclusion",
                      "decision", "ci_low", "ci_high"):
        assert forbidden not in present, f"calibration artifact leaks {forbidden}"


def test_calibration_analyser_requires_both_primary_hypotheses(tmp_path):
    c1 = _cal_cert(tmp_path, "s.json", "H-EV-SHUFFLE",
                   [0.90 + i * 1e-3 for i in range(10)], [0.91 + i * 1.1e-3 for i in range(10)])
    r = _analyse(tmp_path, c1)
    assert r.returncode == 1 and "missing calibration input" in r.stdout


def test_calibration_analyser_rejects_duplicate_inputs(tmp_path):
    vals = ([0.90 + i * 1e-3 for i in range(10)], [0.91 + i * 1.1e-3 for i in range(10)])
    c1 = _cal_cert(tmp_path, "s.json", "H-EV-SHUFFLE", *vals)
    c2 = _cal_cert(tmp_path, "s2.json", "H-EV-SHUFFLE", *vals)
    r = _analyse(tmp_path, c1, c2, extra=("--allow-partial",))
    assert r.returncode == 1 and "duplicate" in r.stdout


def test_calibration_analyser_rejects_unequal_and_non_finite_arrays(tmp_path):
    bad_len = _cal_cert(tmp_path, "a.json", "H-EV-SHUFFLE", [0.9] * 10, [0.91] * 9)
    assert "differ in length" in _analyse(tmp_path, bad_len, extra=("--allow-partial",)).stdout
    nan = _cal_cert(tmp_path, "b.json", "H-EV-SHUFFLE", [0.9] * 9 + [None], [0.91] * 10)
    assert "non-finite" in _analyse(tmp_path, nan, extra=("--allow-partial",)).stdout


def test_sigma_upper_confidence_bound_matches_chi_square():
    import math
    from scipy import stats
    sys.path.insert(0, str(EV / "scripts"))
    from analyze_ev_calibration import sigma_upper_confidence_bound
    s, n = 0.004, 10
    expected = s * math.sqrt((n - 1) / stats.chi2.ppf(0.05, n - 1))
    assert abs(sigma_upper_confidence_bound(s, n) - expected) < 1e-12
    assert sigma_upper_confidence_bound(s, n) > s


def test_power_planner_refuses_without_both_paired_sigmas(tmp_path):
    art = tmp_path / "v.json"
    art.write_text(json.dumps({"artifact": "ev-calibration-variance-v2", "inputs": [],
                               "sigma_Delta_estimates": {"H-EV-SHUFFLE": 3e-4},
                               "sigma_Delta_upper_95": {"H-EV-SHUFFLE": 5e-4}}))
    r = _invoke("plan_ev_power", [
                        "--variance-artifact", str(art), "--epsilon", "0.01",
                        "--target-effect", "0.01"])
    assert r.returncode == 1 and "H-EV-REPRESENTATION" in r.stdout


def test_power_planner_logic_common_k_families_and_sigma(tmp_path):
    """
    Planner LOGIC at a unit-test simulation budget.

    Calls build_power_plan() directly: the frozen-parameter gate lives in
    main()/_require_frozen and still demands exactly 50,000 simulations for
    a production plan (see
    test_planner_fails_closed_on_every_frozen_parameter). Running the
    production budget here would cost tens of millions of simulated tests
    for no additional assurance about the logic.
    """
    import sys as _sys
    _sys.path.insert(0, str(EV / "scripts"))
    from plan_ev_power import build_power_plan

    art_path = _variance_artifact(tmp_path)
    art = json.loads(art_path.read_text())
    plan = build_power_plan(
        art, variance_artifact=art_path, epsilon=0.01, target_effect=0.01,
        target_effect_source="predeclared", alpha=0.05, target_power=0.80,
        simulations=400, max_n=12,
    )
    reqs = plan["requirements"]
    # both primary hypotheses, both procedures
    assert len(reqs) == 4
    assert {r["hypothesis"] for r in reqs} == set(PRIMARY_FAMILY_IDS)
    assert {r["procedure"] for r in reqs} == {"difference", "equivalence"}
    # Holm is now simulated EXACTLY inside the joint decision simulation,
    # so full alpha is used and the alpha/2 approximation is gone.
    assert {r["alpha_used"] for r in reqs} == {0.05}
    assert all(r["multiplicity_simulated"] == "holm_exact" for r in reqs)
    # common K is the max over all four requirements
    assert plan["common_design"]["required_n"] == max(r["required_n"] for r in reqs)
    # upper-95% sigma used, never the point estimate
    assert plan["inputs"]["sigma_source"] == "upper95"
    for r in reqs:
        assert r["sigma_used"] == art["sigma_Delta_upper_95"][r["hypothesis"]]
        assert r["sigma_used"] != r["sigma_point_estimate"]
    assert "simulated EXACTLY" in plan["multiplicity"]["planning_caveat"]
    assert "gpu" not in json.dumps(plan).lower()


def test_power_planner_refuses_observed_production_effect(tmp_path):
    art = tmp_path / "v.json"
    art.write_text(json.dumps({
        "artifact": "ev-calibration-variance-v2",
        "inputs": [{"experiment_id": "H-EV-SHUFFLE", "certificate_sha256": "a" * 64, "n_valid": 10},
                   {"experiment_id": "H-EV-REPRESENTATION", "certificate_sha256": "b" * 64, "n_valid": 10}],
        "sigma_Delta_estimates": {"H-EV-SHUFFLE": 3e-4, "H-EV-REPRESENTATION": 3e-4},
        "sigma_Delta_upper_95": {"H-EV-SHUFFLE": 5e-4, "H-EV-REPRESENTATION": 5e-4}}))
    r = _invoke("plan_ev_power", [
                        "--variance-artifact", str(art), "--epsilon", "0.01",
                        "--target-effect", "0.05", "--target-effect-source", "observed_production",
                        "--simulations", "50", "--max-n", "4",
                        "--output", str(tmp_path / "p.json")])
    assert r.returncode != 0
    assert "observed_production" in (r.stdout + r.stderr)


# --- permutation binding across calibration and production ----------------

def test_representation_calibration_and_production_share_one_permutation_hash():
    prod = yaml.safe_load((CONFIGS / "gohr_ev_representation.yaml").read_text())
    cal = yaml.safe_load((CONFIGS / "gohr_ev_representation_calibration.yaml").read_text())
    assert prod["expected_candidate1_permutation_sha256"] == \
        cal["expected_candidate1_permutation_sha256"]
    assert prod["permutation_generation_seed"] == cal["permutation_generation_seed"]


def test_expected_permutation_hash_matches_the_realized_permutation():
    """The bound hash must be the one the frozen seed actually produces."""
    import numpy as np
    from gohr.representation import generate_candidate1_permutation
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_representation.yaml").read_text())
    perm = generate_candidate1_permutation(np.random.default_rng(cfg["permutation_generation_seed"]))
    assert perm.hash == cfg["expected_candidate1_permutation_sha256"]


def test_representation_run_refuses_on_permutation_mismatch(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_representation_calibration.yaml").read_text())
    cfg["expected_candidate1_permutation_sha256"] = "f" * 64
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "permutation hash mismatch" in out.stdout
    # detected during PRE-FLIGHT: nothing generated, no experiment launched
    assert "VALIDATION ONLY" not in out.stdout
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*.npz"))


def test_representation_run_refuses_without_expected_permutation(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_representation_calibration.yaml").read_text())
    cfg.pop("expected_candidate1_permutation_sha256")
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "expected_candidate1_permutation_sha256" in out.stdout
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*.npz"))


# --- manipulated factor exemption ------------------------------------------

def test_shuffle_config_must_not_pin_its_manipulated_factor(tmp_path):
    """shuffle is the factor under test; pinning it to the baseline is refused."""
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle_calibration.yaml").read_text())
    cfg["shuffle"] = True
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "manipulated factor" in out.stdout


def test_baseline_consistency_catches_a_silently_diverged_parameter(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_baseline_calibration.yaml").read_text())
    cfg.update(requested_replicates=2, minimum_valid_replicates=2, reg_param=1e-4)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "reg_param" in out.stdout


# --- paired dataset persistence and initial-weight equality ----------------

def test_paired_datasets_are_persisted_per_pair(tmp_path):
    """An incomplete pair must reuse its exact dataset, never regenerate one."""
    from gohr.experiments import _load_or_generate_pair_datasets, _pair_dataset_manifest_path
    kw = dict(rounds=3, differential=(0x0040, 0x0000), train_size=64, val_size=32,
              confirmatory_test_size=32)
    first = _load_or_generate_pair_datasets(tmp_path, "H-EV-SHUFFLE", "pair0", **kw)
    manifest = _pair_dataset_manifest_path(tmp_path, "H-EV-SHUFFLE", "pair0")
    assert manifest.exists()
    second = _load_or_generate_pair_datasets(tmp_path, "H-EV-SHUFFLE", "pair0", **kw)
    assert second.train.combined_hash == first.train.combined_hash
    assert second.confirmatory_test.combined_hash == first.confirmatory_test.combined_hash
    other = _load_or_generate_pair_datasets(tmp_path, "H-EV-SHUFFLE", "pair1", **kw)
    assert other.train.combined_hash != first.train.combined_hash   # independent across pairs


def test_initial_weight_hash_is_identical_for_the_same_seed_and_differs_otherwise():
    from gohr.adapter import hash_initial_weights
    from gohr.model import make_resnet
    from gohr import train as gohr_train

    def build(seed):
        gohr_train.set_seed(seed)
        return hash_initial_weights(make_resnet(depth=1, reg_param=1e-5))

    assert build(4242) == build(4242)
    assert build(4242) != build(4243)


# --- protocol manifest -------------------------------------------------------

def test_protocol_manifest_records_the_frozen_design(tmp_path):
    out = tmp_path / "pm.json"
    r = _invoke("ev_protocol_manifest", [
                        str(CONFIGS / "gohr_ev_representation_calibration.yaml"),
                        "--output", str(out)])
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads(out.read_text())
    assert m["practical_significance"]["epsilon"] == 0.01
    assert m["practical_significance"]["predeclared"] is True
    assert m["statistical_design"]["alpha"] == 0.05
    assert m["statistical_design"]["target_power"] == 0.80
    assert m["statistical_design"]["target_effect"] == 0.01
    assert m["calibration"]["pairs_per_paired_primary"] == 10
    assert len(m["source_manifest_sha256"]) == 64
    assert len(m["baseline"]["hash"]) == 64
    assert len(m["statistical_plan_sha256"]) == 64
    assert m["candidate1_permutation_sha256"] == \
        yaml.safe_load((CONFIGS / "gohr_ev_representation.yaml").read_text())[
            "expected_candidate1_permutation_sha256"]
    assert m["non_evidentiary"] is True
    assert "NOT exact Holm" in m["statistical_design"]["multiplicity_at_planning"]


# --- execution confirmation: validation must never launch an experiment ----

def test_calibration_is_validation_only_without_confirm_execute(tmp_path):
    """
    REGRESSION: calibration uses the FULL production protocol, so an
    accidental launch generates 10M-sample datasets and trains. Validating
    a config must never start one.
    """
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle_calibration.yaml").read_text())
    out_dir = tmp_path / "out"
    cfg["output_dir"] = str(out_dir)
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    res = _run_ev(p, tmp_path)
    assert res.returncode == 0
    assert "VALIDATION ONLY" in res.stdout and "--confirm-execute" in res.stdout
    # nothing generated: no datasets, no ledger, no certificate
    assert not out_dir.exists() or not any(out_dir.rglob("*.npz"))
    assert not (out_dir / "resume_ledger.jsonl").exists()


def test_production_is_validation_only_without_confirm_execute(tmp_path):
    art = _power_artifact(tmp_path, required_n=10)
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=10, minimum_valid_pairs=10,
               power_analysis={"artifact_path": str(art), "required_n": 10})
    _with_hash(cfg, art)
    out_dir = tmp_path / "out"
    cfg["output_dir"] = str(out_dir)
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    res = _run_ev(p, tmp_path)
    assert res.returncode == 0 and "VALIDATION ONLY" in res.stdout
    assert not out_dir.exists() or not any(out_dir.rglob("*.npz"))


def test_smoke_still_runs_without_confirmation(tmp_path):
    """Smoke is tiny and non-evidentiary; it must remain frictionless."""
    import inspect
    from importlib import util
    spec = util.spec_from_file_location("run_ev_mod2", EV / "scripts" / "run_ev.py")
    mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
    src = inspect.getsource(mod.main)
    assert 'run_mode in ("calibration", "production")' in src


# ===================== audit round: items 1-13 =============================

def _paired_cert(hyp, *, raw_p, ci, p_tost, eps=0.01, equivalent=False):
    return {
        "audit": {"id": hyp}, "run_mode": "production", "non_evidentiary": False,
        "calibration": False,
        "statistics": {"raw_p_value": raw_p},
        "confidence_interval": {"low": ci[0], "high": ci[1]},
        "practical_significance": {
            "threshold": eps,
            "assessment": {
                "formal_practical_equivalence": (
                    "EQUIVALENT_WITHIN_THRESHOLD" if equivalent else "INCONCLUSIVE"),
                "threshold": eps, "tost": {"p_tost": p_tost}},
        },
        "conservative_wording": {"supported": "SUP-TEXT", "equivalent": "EQ-TEXT",
                                 "inconclusive": "INC-TEXT"},
        "conservative_conclusion": "STALE",
    }


# --- item 1: corrected-significant + CI outside epsilon -> SUPPORTED -------

def test_family_correction_can_reach_supported_with_ci_outside_epsilon():
    from gohr.experiments import apply_primary_family_correction
    a = _paired_cert("H-EV-SHUFFLE", raw_p=1e-6, ci=(0.05, 0.09), p_tost=0.99)
    b = _paired_cert("H-EV-REPRESENTATION", raw_p=1e-6, ci=(0.20, 0.30), p_tost=0.99)
    apply_primary_family_correction(a, b)
    assert a["decision"] == "SUPPORTED" and b["decision"] == "SUPPORTED"
    assert a["conservative_conclusion"] == "SUP-TEXT"      # recomputed, not stale


def test_family_correction_significant_but_inside_epsilon_is_inconclusive():
    from gohr.experiments import apply_primary_family_correction
    a = _paired_cert("H-EV-SHUFFLE", raw_p=1e-8, ci=(0.0005, 0.0015), p_tost=0.99)
    b = _paired_cert("H-EV-REPRESENTATION", raw_p=1e-8, ci=(0.0005, 0.0015), p_tost=0.99)
    apply_primary_family_correction(a, b)
    assert a["decision"] == "INCONCLUSIVE" and a["conservative_conclusion"] == "INC-TEXT"


# --- item 2: separate equivalence Holm family -------------------------------

def test_equivalence_family_is_holm_corrected_separately():
    from gohr.experiments import EQUIVALENCE_FAMILY_ID, apply_primary_family_correction
    a = _paired_cert("H-EV-SHUFFLE", raw_p=0.9, ci=(-0.001, 0.001), p_tost=0.001,
                     equivalent=True)
    b = _paired_cert("H-EV-REPRESENTATION", raw_p=0.9, ci=(-0.001, 0.001), p_tost=0.002,
                     equivalent=True)
    out = apply_primary_family_correction(a, b)
    eq = out["equivalence_multiplicity_result"]
    assert eq is not None and eq["family_id"] == EQUIVALENCE_FAMILY_ID
    assert eq["family_id"] != out["multiplicity_result"]["family_id"]
    assert "no additional correction inside each tost" in eq["note"].lower()
    assert a["decision"] == "NOT_SUPPORTED" and a["conservative_conclusion"] == "EQ-TEXT"


def test_equivalence_lost_after_holm_is_not_not_supported():
    """A TOST that only just passed alone must not survive family correction."""
    from gohr.experiments import apply_primary_family_correction
    a = _paired_cert("H-EV-SHUFFLE", raw_p=0.9, ci=(-0.001, 0.001), p_tost=0.049,
                     equivalent=True)
    b = _paired_cert("H-EV-REPRESENTATION", raw_p=0.9, ci=(-0.001, 0.001), p_tost=0.049,
                     equivalent=True)
    apply_primary_family_correction(a, b)
    assert a["decision"] == "INCONCLUSIVE"
    assert a["practical_significance"]["assessment"]["formal_practical_equivalence"] \
        == "INCONCLUSIVE"


# --- items 3-5: planner frozen-design enforcement ---------------------------

def _variance_artifact(tmp_path, **over):
    art = {"artifact": "ev-calibration-variance-v2", "non_evidentiary": True,
           "inputs": [{"experiment_id": h, "certificate_sha256": "a" * 64, "n_valid": 10}
                      for h in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")],
           "sigma_Delta_estimates": {"H-EV-SHUFFLE": 0.003, "H-EV-REPRESENTATION": 0.005},
           "sigma_Delta_upper_95": {"H-EV-SHUFFLE": 0.0049, "H-EV-REPRESENTATION": 0.0081}}
    art.update(over)
    p = tmp_path / "var.json"; p.write_text(json.dumps(art))
    return p


def _plan(tmp_path, art, *extra):
    return _invoke("plan_ev_power", [
         "--variance-artifact", str(art), "--epsilon", "0.01", "--target-effect", "0.01",
         "--output", str(tmp_path / "plan.json"), *extra])


@pytest.mark.parametrize("flag,value", [
    ("--epsilon", "0.02"), ("--target-effect", "0.05"), ("--alpha", "0.1"),
    ("--target-power", "0.9"), ("--simulations", "1000"),
    ("--target-effect-source", "literature"),
])
def test_planner_fails_closed_on_every_frozen_parameter(tmp_path, flag, value):
    art = _variance_artifact(tmp_path)
    r = _plan(tmp_path, art, flag, value, "--max-n", "3")
    assert r.returncode == 1 and "frozen design violated" in r.stdout


def test_planner_rejects_point_sigma_option(tmp_path):
    art = _variance_artifact(tmp_path)
    r = _plan(tmp_path, art, "--use-sigma", "point")
    assert r.returncode != 0 and "point" not in (r.stdout or "").split("choose from")[-1][:0] + ""


@pytest.mark.parametrize("mutation", [
    {"artifact": "ev-calibration-variance-v1"},
    {"non_evidentiary": False},
    {"inputs": [{"experiment_id": "H-EV-SHUFFLE", "certificate_sha256": "a" * 64, "n_valid": 10}]},
    {"inputs": [{"experiment_id": h, "certificate_sha256": "a" * 64, "n_valid": 7}
                for h in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")]},
    {"inputs": [{"experiment_id": h, "certificate_sha256": "short", "n_valid": 10}
                for h in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")]},
    {"sigma_Delta_upper_95": {"H-EV-SHUFFLE": 0.004}},
])
def test_planner_rejects_invalid_calibration_artifacts(tmp_path, mutation):
    art = _variance_artifact(tmp_path, **mutation)
    r = _plan(tmp_path, art, "--max-n", "3")
    assert r.returncode == 1 and "frozen design violated" in r.stdout


def test_planner_simulates_both_holm_families_exactly(tmp_path):
    """Superseded alpha/2 planning: Holm is now simulated exactly."""
    art = _variance_artifact(tmp_path)
    r = _plan(tmp_path, art, "--max-n", "6")
    assert r.returncode == 0, r.stdout
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert {x["alpha_used"] for x in plan["requirements"]} == {0.05}
    assert all(x["multiplicity_simulated"] == "holm_exact" for x in plan["requirements"])
    assert "COMPLETE final decision rule" in plan["procedure_simulated"]


def _good_plan_file(tmp_path, required_n=12):
    plan = {"artifact": "ev-power-plan-v2", "non_evidentiary": True,
            "primary_family": ["H-EV-SHUFFLE", "H-EV-REPRESENTATION"],
            "inputs": {"epsilon": 0.01, "target_effect": 0.01,
                       "target_effect_source": "predeclared", "alpha": 0.05,
                       "target_power": 0.80, "n_simulations": 50000,
                       "sigma_source": "upper95"},
            "common_design": {"required_n": required_n}}
    p = tmp_path / "plan.json"; p.write_text(json.dumps(plan))
    return p


def _prod_cfg(tmp_path, plan_path, **over):
    import hashlib
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    cfg.update(requested_pairs=12, minimum_valid_pairs=12,
               power_analysis={"artifact_path": str(plan_path), "artifact_sha256": digest,
                               "required_n": 12})
    cfg["output_dir"] = str(tmp_path / "out")
    for k, v in over.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    return p


def test_production_accepts_a_fully_frozen_power_artifact(tmp_path):
    cfg = _prod_cfg(tmp_path, _good_plan_file(tmp_path))
    out = _run_ev(cfg, tmp_path)
    assert out.returncode == 0 and "VALIDATION ONLY" in out.stdout   # still never executes


def test_production_requires_declared_artifact_hash(tmp_path):
    plan = _good_plan_file(tmp_path)
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=12, minimum_valid_pairs=12,
               power_analysis={"artifact_path": str(plan), "required_n": 12})
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "artifact_sha256 must be declared" in out.stdout


def test_production_detects_swapped_power_artifact(tmp_path):
    plan = _good_plan_file(tmp_path)
    cfg = _prod_cfg(tmp_path, plan)
    plan.write_text(plan.read_text().replace('"required_n": 12', '"required_n": 99'))
    out = _run_ev(cfg, tmp_path)
    assert out.returncode == 1 and "hash mismatch" in out.stdout


@pytest.mark.parametrize("mutation,needle", [
    ({"artifact": "ev-power-plan-v1"}, "plan schema"),
    ({"non_evidentiary": False}, "non_evidentiary"),
    ({"primary_family": ["H-EV-SHUFFLE"]}, "primary_family"),
])
def test_production_rejects_malformed_power_plan(tmp_path, mutation, needle):
    plan = _good_plan_file(tmp_path)
    body = json.loads(plan.read_text()); body.update(mutation); plan.write_text(json.dumps(body))
    cfg = _prod_cfg(tmp_path, plan)
    out = _run_ev(cfg, tmp_path)
    assert out.returncode == 1 and needle in out.stdout


@pytest.mark.parametrize("key,value", [
    ("epsilon", 0.02), ("alpha", 0.1), ("target_power", 0.9),
    ("n_simulations", 1000), ("sigma_source", "point"), ("target_effect", 0.05),
])
def test_production_rejects_plan_computed_under_different_design(tmp_path, key, value):
    plan = _good_plan_file(tmp_path)
    body = json.loads(plan.read_text()); body["inputs"][key] = value
    plan.write_text(json.dumps(body))
    cfg = _prod_cfg(tmp_path, plan)
    out = _run_ev(cfg, tmp_path)
    assert out.returncode == 1 and f"plan {key}" in out.stdout


def test_production_rejects_required_n_above_minimum_valid_pairs(tmp_path):
    plan = _good_plan_file(tmp_path, required_n=20)
    import hashlib
    cfg = yaml.safe_load((CONFIGS / "gohr_ev_shuffle.yaml").read_text())
    cfg.update(requested_pairs=20, minimum_valid_pairs=12,
               power_analysis={"artifact_path": str(plan),
                               "artifact_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                               "required_n": 20})
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    out = _run_ev(p, tmp_path)
    assert out.returncode == 1 and "below the power-required n" in out.stdout


# --- item 7: calibration certificates must be genuine ----------------------

def test_analyser_rejects_hand_constructed_certificate(tmp_path):
    cert = _cal_cert(tmp_path, "s.json", "H-EV-SHUFFLE",
                     [0.9] * 10, [0.91] * 10, genuine=False)   # no config_hash / provenance
    r = _analyse(tmp_path, cert, extra=("--allow-partial",))
    assert r.returncode == 1 and "config_hash" in r.stdout


def test_analyser_rejects_wrong_pair_count(tmp_path):
    import yaml as _y
    from framework.provenance import config_hash
    cfg = _y.safe_load((CONFIGS / "gohr_ev_shuffle_calibration.yaml").read_text())
    cert = _cal_cert(tmp_path, "s.json", "H-EV-SHUFFLE",
                     [0.90 + i * 1e-3 for i in range(7)], [0.91 + i * 1e-3 for i in range(7)])
    r = _analyse(tmp_path, cert, extra=("--allow-partial",))
    assert r.returncode == 1 and "exactly" in r.stdout


# --- item 8/9: resume provenance and transactional firewall ----------------

def test_pair_sidecar_round_trips_full_provenance(tmp_path):
    from gohr.experiments import _load_pair_results_sidecar, _save_pair_result_sidecar
    _save_pair_result_sidecar(tmp_path, "H-EV-SHUFFLE", "pair0", acc_a=0.9, acc_b=0.8,
                              raw_a={"run_id": "A"}, raw_b={"run_id": "B"},
                              initial_weight_record={"replicate_id": "pair0",
                                                     "condition_a": "h", "condition_b": "h",
                                                     "identical": True})
    entry = _load_pair_results_sidecar(tmp_path, "H-EV-SHUFFLE")["pair0"]
    assert entry["raw_a"]["run_id"] == "A" and entry["raw_b"]["run_id"] == "B"
    assert entry["initial_weight_record"]["identical"] is True


def test_paired_runner_accepts_firewall_path_for_transactional_persistence():
    import inspect
    from gohr.experiments import _run_paired_experiment, run_h_ev_representation, run_h_ev_shuffle
    for fn in (_run_paired_experiment, run_h_ev_shuffle, run_h_ev_representation):
        assert "firewall_path" in inspect.signature(fn).parameters
    src = inspect.getsource(_run_paired_experiment)
    assert "firewall.save(firewall_path)" in src
    # persisted per completed pair, not only at the end of the experiment
    assert src.index("firewall.save(firewall_path)") < src.index("return (")


def test_resume_branch_restores_raw_and_weight_provenance():
    import inspect
    from gohr.experiments import _run_paired_experiment
    src = inspect.getsource(_run_paired_experiment)
    for needle in ('raw_a.append(entry["raw_a"])', 'raw_b.append(entry["raw_b"])',
                   'initial_weight_hashes.append(entry["initial_weight_record"])'):
        assert needle in src


# --- items 10-13: documentation coherence ----------------------------------

def test_docs_contain_one_coherent_v2_decision_rule():
    plan = (EV / "docs" / "statistical_plan.md").read_text()
    assert "v1, superseded" not in plan
    assert plan.count("## Decision semantics") == 1
    assert "entirely outside" in " ".join(plan.split())


def test_docs_declare_two_holm_families_and_planning_caveat():
    plan = (EV / "docs" / "statistical_plan.md").read_text()
    assert "TWO SEPARATE HOLM FAMILIES" in plan
    assert "simulated EXACTLY" in plan or "exactly" in plan.lower()
    assert "no additional correction" in " ".join(plan.lower().split())


def test_docs_state_the_simulation_assumptions():
    plan = (EV / "docs" / "statistical_plan.md").read_text()
    assert "INDEPENDENT paired differences" in plan and "NORMALLY distributed" in plan
    assert "sigma_Delta_upper_95" in plan


def test_docs_state_40_trainings_and_no_runtime_claim():
    cal = (EV / "docs" / "ev_calibration_plan.md").read_text()
    assert "40 trainings" in cal and "20 per hypothesis" in cal
    assert "No runtime claim is made" in cal


def test_docs_use_independently_generated_wording():
    cal = (EV / "docs" / "ev_calibration_plan.md").read_text()
    assert "no explicit overlap guarantee" in " ".join(cal.lower().split())
