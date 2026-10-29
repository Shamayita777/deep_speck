"""
AutoND Experimental Validity tests that run in the shared pytest process: configuration gates, the
representation transform on synthetic data, the environment gate, statistics, family correction and
power planning. Nothing here binds AutoND, imports TensorFlow models or seeds Keras.

Everything that does is in tests/autond_tf_cases.py, run in a child process by
test_tensorflow_cases_pass_in_an_isolated_process: tf.keras.utils.set_random_seed leaves process-wide
Keras seeding state that would otherwise change later Gohr tests' same-seed initialisation. Set
AUTOND_ROOT if the checkout is not at ~/Documents/AutoND-upstream (SKIPPED IS NOT PASSED).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autond import adapter as A
from autond import baseline as B
from autond import calibration as C
from autond import experiments as E
from autond import representation as REP
from autond import runner as R
from framework.certificate import DecisionState, NonConfirmatoryEvidenceError, decide_final
from framework.multiplicity import holm_correction
from framework.power import simulate_joint_primary_family
from framework.seeds import statistics_rng

CASES = Path(__file__).with_name("autond_tf_cases.py")


def _root():
    for candidate in (os.environ.get("AUTOND_ROOT"), "~/Documents/AutoND-upstream"):
        if candidate and (Path(candidate).expanduser() / "main.py").is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


ROOT = _root()
needs_checkout = pytest.mark.skipif(ROOT is None, reason="pinned AutoND checkout not found; SKIPPED IS NOT PASSED")
H1, H2 = B.PRIMARY_FAMILY


def filled(**decisions):
    """The shipped config with every placeholder resolved by TEST-ONLY values (never a real decision)."""
    config = copy.deepcopy(B.load_config())
    config["decisions"].update({"practical_threshold": 0.01, "practical_threshold_justification": "TEST-ONLY",
                                "target_effect": 0.02, "target_effect_justification": "TEST-ONLY", **decisions})
    for h in B.PRIMARY_FAMILY:
        config["calibration"][h]["units"] = 3
    for e in B.EXPERIMENT_IDS:
        config["production"][e].update(units=4, minimum_valid=4)
    config["production"]["power_plan"].update(sha256="0" * 64, required_n=4)
    return config


# ============================================================ process isolation


@needs_checkout
def test_tensorflow_cases_pass_in_an_isolated_process():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(A.REPOSITORY_ROOT), env.get("PYTHONPATH")) if p)
    out = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", str(CASES)],
                         cwd=CASES.parents[1], env=env, capture_output=True, text=True, timeout=3600)
    summary = [line for line in out.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line]
    report = out.stdout[-6000:] + out.stderr[-2000:]
    assert out.returncode == 0 and summary, report
    assert "failed" not in summary[-1] and "error" not in summary[-1] and "skipped" not in summary[-1], report
    print(f"isolated AutoND TensorFlow cases: {summary[-1]}")


def test_importing_autond_loads_neither_tensorflow_nor_autond_modules():
    code = ("import sys, autond.runner, autond.experiments, autond.representation\n"
            "bad = [m for m in ('tensorflow', 'keras', 'dbitnet', 'train_nets', 'main', 'gohr', "
            "'gohr.representation') if m in sys.modules]\nprint(bad); sys.exit(1 if bad else 0)")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(CASES.parents[1]), env.get("PYTHONPATH")) if p)
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=CASES.parents[1], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


# ============================================================ environment gate


@pytest.mark.parametrize("versions", [{"tensorflow": "2.18.0", "keras": "3.6.0"},
                                      {"tensorflow": "2.15.1", "keras": "3.6.0"},
                                      {"tensorflow": None, "keras": "2.15.0"}])
def test_an_unsupported_tensorflow_or_keras_is_refused_before_anything_is_imported(versions, monkeypatch, tmp_path):
    from importlib import metadata

    def version(name):
        value = versions["keras"] if name == "keras" else versions["tensorflow"] if name == "tensorflow" else None
        if value is None:
            raise metadata.PackageNotFoundError(name)
        return value
    monkeypatch.setattr(metadata, "version", version)
    with pytest.raises(A.AutoNDBindingError, match="TensorFlow 2.15 with Keras 2.15"):
        A.bind(tmp_path / "no-checkout")
    out = tmp_path / "smoke-out"
    assert R.main(["run", "--mode", "smoke", "--autond-root", str(tmp_path), "--output-root", str(out)]) == 2
    assert not out.exists()


def test_the_supported_environment_is_accepted():
    found = A.require_supported_environment()
    assert found["tensorflow"].startswith("2.15.") and found["keras"].startswith("2.15.")


# ============================================================ representation (synthetic data; no Gohr code)


def test_the_frozen_permutation_reproduces_the_declared_hash_and_a_different_one_is_refused():
    config = B.load_config()
    permutation = E.frozen_permutation(config)
    assert permutation.hash == config["representation"]["expected_sha256"] == REP.generate(999).hash
    config["representation"]["generation_seed"] = 1000
    with pytest.raises(B.ConfigError, match="not the declared"):
        E.frozen_permutation(config)


def test_the_mapping_moves_bits_only_within_their_word_in_the_declared_direction():
    permutation = REP.generate(999)
    columns = np.arange(64, dtype=np.int64)[None, :]          # each column carries its own index
    moved = permutation.apply(columns)[0]
    for w in range(4):
        for j in range(16):
            assert moved[16 * w + j] == 16 * w + permutation.mapping[w, j]      # destination j <- source mapping[w, j]
    assert np.array_equal(permutation.invert(permutation.apply(columns)), columns)


def test_invariants_and_refusals_on_synthetic_bits(tmp_path):
    rng = np.random.default_rng(3)
    X, Y = rng.integers(0, 2, (500, 64)).astype(np.uint8), rng.integers(0, 2, 500).astype(np.uint8)
    report = REP.validate(X, Y, REP.generate(999), 999)
    assert report["all_passed"] and all(report[k] for k in ("bijection", "round_trip", "content_preserved",
                                                           "labels_unchanged", "deterministic"))
    assert REP.validate(X, Y, REP.generate(999), 998)["deterministic"] is False      # a different seed is detected
    for mapping, message in ((np.zeros((4, 16), np.int64), "C0_left is not a bijection"),
                             (np.tile(np.arange(16), (3, 1)), "shape"), (np.tile(np.arange(16.0), (4, 1)), "integer")):
        with pytest.raises(ValueError, match=message):
            REP.WordPermutation(mapping)
    with pytest.raises(ValueError, match="shape"):
        REP.generate(999).apply(X[:, :63])
    path = tmp_path / "permutation.json"
    REP.generate(999).save(path)
    assert REP.WordPermutation.load(path).hash == REP.generate(999).hash
    record = json.loads(path.read_text())
    record["mapping"][0][0], record["mapping"][0][1] = record["mapping"][0][1], record["mapping"][0][0]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recorded SHA-256"):
        REP.WordPermutation.load(path)


@pytest.mark.parametrize("X, Y", [(np.zeros((8, 63), np.uint8), np.zeros(8, np.uint8)),
                                  (np.zeros((8, 64), np.float32), np.zeros(8, np.uint8)),
                                  (np.full((8, 64), 2, np.uint8), np.zeros(8, np.uint8))])
def test_invalid_partitions_are_refused(X, Y):
    with pytest.raises(A.AutoNDBindingError):
        A.validate_partition(X, Y, 8)


def test_output_directories_are_bound_to_one_mode_and_one_writer(tmp_path):
    R.lock_mode(tmp_path / "x", "calibration")
    with pytest.raises(R.Refused, match="belongs to run_mode"):
        R.lock_mode(tmp_path / "x", "production")
    lock = R.acquire_lock(tmp_path / "x")
    with pytest.raises(R.Refused, match="RUN_LOCK"):
        R.acquire_lock(tmp_path / "x")
    lock.unlink()
    assert R.output_dir("calibration", H1) != R.output_dir("production", H1)


# ============================================================ configuration gates


def test_the_shipped_config_is_frozen_except_for_the_pending_decisions():
    config = B.load_config()
    assert config["protocol"] == B.FROZEN_PROTOCOL and B.check_config(config, "smoke") == []
    pending = B.check_config(config, "calibration")
    assert len(pending) == 1 and "decisions.practical_threshold" in pending[0] and "calibration." in pending[0]
    assert "production.power_plan.sha256" in B.check_config(config, "production")[0]


def test_production_refuses_through_the_entry_point_while_decisions_are_pending(tmp_path, capsys):
    assert R.main(["run", "--mode", "production", "--experiment", H1, "--autond-root", str(tmp_path)]) == 2
    assert "unresolved decisions" in capsys.readouterr().err


@pytest.mark.parametrize("edit, message", [
    (lambda c: c["protocol"].update(rounds=6), "protocol differs"),
    (lambda c: c["protocol"].update(epochs=200), "protocol differs"),
    (lambda c: c["decisions"].update(alpha=0.1), "statistical plan fixes"),
    (lambda c: c["decisions"].update(target_effect=0.01), "must exceed practical_threshold"),
    (lambda c: c["decisions"].update(practical_threshold_justification=""), "non-empty justification"),
    (lambda c: c["production"][H2].update(base_seed=91000), "reuses model seeds"),
    (lambda c: c["production"][H1].update(minimum_valid=9), "minimum_valid <= units"),
])
def test_invalid_configurations_are_refused(edit, message):
    config = filled()
    assert B.check_config(config, "production") == []
    edit(config)
    assert any(message in p for p in B.check_config(config, "production")), B.check_config(config, "production")


def test_hash_scopes_bind_each_tier_to_what_was_frozen_before_it():
    config = filled()
    design, production = B.design_hash(config), B.production_hash(config)
    config["smoke"]["protocol"]["epochs"] = 3
    assert (B.design_hash(config), B.production_hash(config)) == (design, production)
    config["production"][H1]["units"] = 9
    assert B.design_hash(config) == design and B.production_hash(config) != production
    config["decisions"]["practical_threshold"] = 0.005
    assert B.design_hash(config) != design


# ============================================================ statistics, family correction, power


def _cert(diffs, *, epsilon=0.01, sufficient=True, run_mode="production"):
    """A minimal paired certificate built with the framework's own analysis (synthetic values)."""
    from framework.certificate import assess_practical_equivalence
    from framework.experiment import PracticalSignificance
    from framework.statistics import paired_analysis

    a = np.full(len(diffs), 0.9)
    res = paired_analysis(a, a + np.asarray(diffs), rng=statistics_rng(1))
    assessment = assess_practical_equivalence(res, PracticalSignificance(epsilon, True, "TEST"))
    return {"statistics": {"raw_p_value": res.p_value, "confidence_interval": {"low": res.ci_low, "high": res.ci_high}},
            "practical_significance": {"assessment": assessment}, "sufficient": sufficient, "run_mode": run_mode,
            "non_evidentiary": run_mode != "production", "calibration": run_mode == "calibration",
            "production_hash": "h", "experiment_id": "x"}


def test_family_correction_reaches_supported_and_not_supported_with_the_framework_rule():
    big = _cert([0.05, 0.052, 0.048, 0.051, 0.049, 0.05])          # CI far outside +/-0.01
    null = _cert([0.0005, -0.0004, 0.0003, -0.0002, 0.0001, 0.0])  # tightly inside the margin
    family = E.holm_family({H1: big, H2: null})
    assert family["decisions"] == {H1: "SUPPORTED", H2: "NOT_SUPPORTED"}
    expected = holm_correction(family_id="x", hypothesis_ids=[H1, H2],
                               p_values=[big["statistics"]["raw_p_value"], null["statistics"]["raw_p_value"]])
    assert family["certificates"][H1]["statistics"]["adjusted_p_value"] == expected.adjusted_p_values[0]
    for h in (H1, H2):
        cert = family["certificates"][h]
        assert cert["decision"] == decide_final(
            difference_decision=(DecisionState.SUPPORTED if cert["statistics"]["adjusted_p_value"] < 0.05
                                 else DecisionState.INCONCLUSIVE),
            practical_equivalence=cert["practical_significance"]["assessment"],
            confidence_interval=cert["statistics"]["confidence_interval"], epsilon=0.01).value


def test_a_significant_effect_inside_the_margin_and_a_missing_p_value_are_handled_conservatively():
    small = _cert([0.004, 0.0042, 0.0038, 0.0041, 0.0039, 0.004])   # significant but inside +/-0.01
    absent = _cert([0.0, 0.0]) | {"sufficient": False}
    absent["statistics"]["raw_p_value"] = None
    family = E.holm_family({H1: small, H2: absent})
    assert family["decisions"][H1] == "INCONCLUSIVE" and family["decisions"][H2] == "INCONCLUSIVE"
    p = small["statistics"]["raw_p_value"]
    assert family["certificates"][H1]["statistics"]["adjusted_p_value"] == pytest.approx(min(1.0, 2 * p))


def test_only_production_certificates_enter_the_confirmatory_family():
    for mode in ("smoke", "calibration"):
        with pytest.raises(NonConfirmatoryEvidenceError):
            E.apply_family_correction({H1: _cert([0.05, 0.051, 0.049], run_mode=mode), H2: _cert([0.05, 0.051, 0.049])})
    family = E.apply_family_correction({H1: _cert([0.05, 0.051, 0.049]), H2: _cert([0.0, 0.001, -0.001])})
    assert family["run_mode"] == "production"


def test_a_target_effect_at_the_margin_cannot_power_supported():
    """The Gohr planner's frozen target equals epsilon; at that effect SUPPORTED stays below 1/2 for every n."""
    for n in (10, 50, 200):
        res = simulate_joint_primary_family(hypotheses=(H1, H2), sigmas={H1: 0.003, H2: 0.003},
                                            true_effects={H1: 0.01, H2: 0.01}, n_pairs=n, alpha=0.05, epsilon=0.01,
                                            n_simulations=4000, rng=statistics_rng(n),
                                            target_effect_source="predeclared")
        assert res.power_difference_significant[H1] > 0.99 and res.power_supported[H1] < 0.5


def _variance(config, sigma=0.004):
    return {"artifact": C.VARIANCE_ARTIFACT, "design_hash": B.design_hash(config),
            "sigma_Delta_upper_95": {H1: sigma, H2: sigma}}


def test_power_plan_sizes_the_actual_final_decisions_under_the_worst_other_hypothesis():
    config = filled(power_simulations=2000)
    plan = E.power_plan(_variance(config), config, variance_sha256="v", max_n=60)
    n = plan["required_n"]
    assert n is not None and plan["power_curve"][-1]["n"] == n
    assert all(v >= 0.80 for k, v in plan["power_curve"][-1].items() if k != "n")
    assert any(v < 0.80 for k, v in plan["power_curve"][-2].items() if k != "n")
    # the worst case is never better than the both-effects scenario the Gohr planner used
    both = simulate_joint_primary_family(hypotheses=(H1, H2), sigmas={H1: 0.004, H2: 0.004},
                                         true_effects={H1: 0.02, H2: 0.02}, n_pairs=n, alpha=0.05, epsilon=0.01,
                                         n_simulations=2000, rng=statistics_rng(70_000_000 + 1000 * n),
                                         target_effect_source="predeclared")
    assert plan["power_curve"][-1][f"{H1}:supported"] <= both.power_supported[H1]
    with pytest.raises(ValueError, match="must exceed"):
        E.power_plan(_variance(filled(target_effect=0.01)), filled(target_effect=0.01), variance_sha256="v")
    with pytest.raises(ValueError, match="this design"):
        E.power_plan(_variance(filled()), filled(practical_threshold=0.005), variance_sha256="v")


def test_production_requires_the_declared_power_plan_and_enough_pairs(tmp_path, monkeypatch):
    config = filled(power_simulations=2000)
    plan_path = tmp_path / "plan.json"
    plan = E.power_plan(_variance(config), config, variance_sha256="v", max_n=60)
    plan_path.write_text(json.dumps(plan))
    config["production"]["power_plan"] = {"path": str(plan_path), "sha256": E.sha256_file(plan_path),
                                          "required_n": plan["required_n"]}
    for h in B.PRIMARY_FAMILY:
        config["production"][h].update(units=plan["required_n"] + 2, minimum_valid=plan["required_n"])
    assert E.check_power_plan(config) == []
    low = copy.deepcopy(config)
    low["production"][H1]["minimum_valid"] = plan["required_n"] - 1
    assert any("below" in p for p in E.check_power_plan(low))
    plan_path.write_text(json.dumps(plan | {"required_n": 2}))
    assert "differs from the SHA-256" in E.check_power_plan(config)[0]


# Lower 5% points of chi-square (standard published tables), fixed here rather than computed by SciPy.
CHI2_LOWER_5_PERCENT = {1: 0.00393214, 4: 0.710723, 9: 3.325113, 11: 4.574813, 19: 10.117013, 29: 17.708366}


@pytest.mark.parametrize("df", sorted(CHI2_LOWER_5_PERCENT))
def test_sigma_bound_matches_the_tabulated_chi_square_formula(df):
    expected = 0.004 * np.sqrt(df / CHI2_LOWER_5_PERCENT[df])
    assert C.sigma_upper_confidence_bound(0.004, df + 1) == pytest.approx(expected, rel=2e-6)


def test_sigma_bound_reference_factors_and_shape():
    assert C.sigma_upper_confidence_bound(1.0, 10) == pytest.approx(1.645198, abs=2e-6)       # sqrt(9/3.325113)
    assert C.sigma_upper_confidence_bound(1.0, 2) == pytest.approx(15.9472, abs=2e-4)         # sqrt(1/0.00393214)
    assert C.sigma_upper_confidence_bound(2.5, 12) == pytest.approx(2.5 * C.sigma_upper_confidence_bound(1.0, 12))
    factors = [C.sigma_upper_confidence_bound(1.0, n) for n in (2, 3, 5, 10, 20, 50, 200)]
    assert all(f > 1 for f in factors) and factors == sorted(factors, reverse=True)           # tighter with more pairs
    assert C.sigma_upper_confidence_bound(0.0, 10) == 0.0
    assert C.sigma_upper_confidence_bound(1.0, np.int64(10)) == C.sigma_upper_confidence_bound(1.0, 10)
    assert C.SIGMA_UCL_LEVEL == 0.95


def test_sigma_bound_covers_the_true_sigma_95_percent_of_the_time():
    rng = np.random.default_rng(20261029)
    for n in (5, 12):
        samples = rng.normal(0.0, 1.0, size=(40_000, n))
        bounds = np.array([C.sigma_upper_confidence_bound(float(s), n) for s in samples.std(axis=1, ddof=1)])
        assert abs(np.mean(bounds >= 1.0) - 0.95) < 0.006          # one-sided: P(sigma <= bound) = 0.95


@pytest.mark.parametrize("sd, n, level", [(1.0, 1, 0.95), (1.0, 0, 0.95), (1.0, True, 0.95), (1.0, 2.0, 0.95),
                                          (-0.1, 10, 0.95), (float("nan"), 10, 0.95), (float("inf"), 10, 0.95),
                                          (True, 10, 0.95), ("0.1", 10, 0.95), (1.0, 10, 0.0), (1.0, 10, 1.0)])
def test_sigma_bound_refuses_invalid_input(sd, n, level):
    with pytest.raises(ValueError):
        C.sigma_upper_confidence_bound(sd, n, level)


def test_calibration_variance_uses_only_matching_calibration_certificates(tmp_path):
    config = filled()
    paths = {}
    for h, diffs in ((H1, [0.001, -0.002, 0.003]), (H2, [0.002, 0.0, -0.001])):
        cert = {"run_mode": "calibration", "calibration": True, "non_evidentiary": True, "experiment_id": h,
                "design_hash": B.design_hash(config),
                "raw_paired_values": {"condition_a": [0.9] * 3, "condition_b": [0.9 + d for d in diffs]}}
        paths[h] = tmp_path / f"{h}.json"
        paths[h].write_text(json.dumps(cert))
    base = json.loads(paths[H1].read_text())
    artifact = C.calibration_variance(list(paths.values()), config)
    assert set(artifact["inputs"]) == set(B.PRIMARY_FAMILY) and "mean" not in json.dumps(artifact)
    sd = float(np.std(np.array([0.001, -0.002, 0.003]), ddof=1))
    assert artifact["inputs"][H1]["sigma_Delta"] == pytest.approx(sd)
    assert artifact["sigma_Delta_upper_95"][H1] == pytest.approx(sd * np.sqrt(2 / 0.102587), rel=2e-5)   # df = 2
    assert artifact["artifact"] == "autond-ev-calibration-variance-1" and artifact["non_evidentiary"] is True
    assert set(artifact) == {"artifact", "non_evidentiary", "design_hash", "generated_at_utc", "inputs",
                             "sigma_Delta_upper_95", "prohibition"}
    assert set(artifact["inputs"][H1]) == {"certificate", "certificate_sha256", "n_valid", "sigma_Delta",
                                           "sigma_Delta_upper_95"}
    production = {"run_mode": "production", "calibration": False, "non_evidentiary": False}
    for bad, message in ((production, "not a calibration"),
                         ({"design_hash": "other"}, "different design")):
        paths[H1].write_text(json.dumps(base | bad))
        with pytest.raises(ValueError, match=message):
            C.calibration_variance(list(paths.values()), config)
    paths[H1].write_text(json.dumps(base))
    config["calibration"][H1]["units"] = 4
    with pytest.raises(ValueError, match="different design"):     # the plan is part of the design hash
        C.calibration_variance(list(paths.values()), config)
