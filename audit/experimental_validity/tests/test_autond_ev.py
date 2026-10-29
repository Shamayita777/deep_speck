"""
AutoND Experimental Validity tests. Every run here is tiny and NON-EVIDENTIARY.

The pipeline tests bind the pinned AutoND checkout and use its real generator and DBitNet on CPU; set
AUTOND_ROOT if the checkout is not at ~/Documents/AutoND-upstream (SKIPPED IS NOT PASSED). Run in the
pinned TensorFlow 2.15 environment with the repository root importable.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from autond import adapter as A  # noqa: E402
from autond import baseline as B  # noqa: E402
from autond import experiments as E  # noqa: E402
from autond import runner as R  # noqa: E402
from framework.certificate import DecisionState, NonConfirmatoryEvidenceError, decide_final  # noqa: E402
from framework.firewall import FirewallViolation  # noqa: E402
from framework.multiplicity import holm_correction  # noqa: E402
from framework.power import simulate_joint_primary_family  # noqa: E402
from framework.resumability import ResumeViolation  # noqa: E402
from framework.seeds import statistics_rng  # noqa: E402
from gohr.representation import apply_candidate1, invert_candidate1, run_full_validation  # noqa: E402


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


@pytest.fixture(scope="module", autouse=True)
def _restore_keras_seed_state():
    """Leave the process as found. tf.keras.utils.set_random_seed (the AutoND seeding) installs a thread-local
    Keras seed generator that every later initialiser draws from; the Gohr tests sharing this pytest process
    seed only Python, NumPy and TensorFlow, so without this their same-seed models would differ."""
    try:
        from keras.src import backend
    except ImportError:
        backend = None
    state = getattr(backend, "_SEED_GENERATOR", None)      # Keras 2 only
    if state is None:
        yield
        return
    sentinel = object()
    saved = getattr(state, "generator", sentinel)
    yield
    if saved is sentinel:
        if hasattr(state, "generator"):
            del state.generator
    else:
        state.generator = saved


@pytest.fixture(scope="module")
def binding():
    if ROOT is None:
        pytest.skip("pinned AutoND checkout not found; SKIPPED IS NOT PASSED")
    return A.bind(ROOT)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """One tiny end-to-end smoke run of all four experiments plus the family step, through the entry point."""
    if ROOT is None:
        pytest.skip("pinned AutoND checkout not found; SKIPPED IS NOT PASSED")
    root = tmp_path_factory.mktemp("smoke")
    assert R.main(["run", "--mode", "smoke", "--experiment", "all", "--autond-root", ROOT,
                   "--output-root", str(root)]) == 0
    return {e: json.loads((root / "smoke" / e / "certificate.json").read_text()) for e in B.EXPERIMENT_IDS} | {
        "_root": root, "_family": json.loads((root / "smoke" / "family_certificate.json").read_text())}


def tiny(experiment, binding, out, config=None, mode="smoke", units=2):
    config = copy.deepcopy(config or B.load_config())
    config["smoke"][experiment]["units"] = units
    return E.run_experiment(experiment, config=config, mode=mode, output_dir=out, binding=binding,
                            permutation=E.frozen_permutation(config))


# ============================================================ pin, data, model


@needs_checkout
@pytest.mark.parametrize("change, message", [("commit", "revision"), ("digest", "pin mismatch")])
def test_a_wrong_pin_is_refused_before_import(change, message, monkeypatch):
    from audit.dataset.adapters.autond import SourceBindingError
    from audit.dataset.experiments import d0_autond_canonical as C

    pin = copy.deepcopy(C.AUTOND_SOURCE_PIN)
    if change == "commit":
        pin["upstream_commit"] = "0" * 40
    else:
        pin["source_digests"]["dbitnet.py"] = "f" * 64
    monkeypatch.setattr(C, "AUTOND_SOURCE_PIN", pin)
    with pytest.raises(SourceBindingError, match=message):
        A.bind(ROOT)


def test_real_generator_and_dbitnet_follow_the_upstream_protocol(binding, monkeypatch):
    X, Y = binding.generate_partition(256)
    A.validate_partition(X, Y, 256)
    assert X.shape == (256, 64) and X.dtype == np.uint8 and set(np.unique(Y)) <= {0, 1}
    a, b, c = A.build_model(binding, 7), A.build_model(binding, 7), A.build_model(binding, 8)
    assert A.weights_sha256(a) == A.weights_sha256(b) != A.weights_sha256(c)
    assert a.input_shape == (None, 64, 1) and a.loss == "mse"
    assert type(a.optimizer).__name__ == "Adam" and a.optimizer.get_config()["amsgrad"] is True
    fake = type(A)("dbitnet")
    fake.__file__ = "/elsewhere/dbitnet.py"
    monkeypatch.setitem(__import__("sys").modules, "dbitnet", fake)
    with pytest.raises(A.AutoNDBindingError, match="outside"):
        A.pinned_module(binding, "dbitnet")


@pytest.mark.parametrize("X, Y", [(np.zeros((8, 63), np.uint8), np.zeros(8, np.uint8)),
                                  (np.zeros((8, 64), np.float32), np.zeros(8, np.uint8)),
                                  (np.full((8, 64), 2, np.uint8), np.zeros(8, np.uint8))])
def test_invalid_partitions_are_refused(X, Y):
    with pytest.raises(A.AutoNDBindingError):
        A.validate_partition(X, Y, 8)


# ============================================================ representation


def test_the_frozen_permutation_is_bound_and_a_different_one_refused():
    config = B.load_config()
    assert E.frozen_permutation(config).hash == config["representation"]["expected_sha256"]
    config["representation"]["generation_seed"] = 1000
    with pytest.raises(B.ConfigError, match="not the declared"):
        E.frozen_permutation(config)


def test_representation_invariants_on_real_autond_data(binding):
    X, Y = binding.generate_partition(2000)
    permutation = E.frozen_permutation(B.load_config())
    report = run_full_validation(X, Y, permutation)
    assert report["all_passed"] and report["bijection"] and report["round_trip"] and report["labels_unchanged"]
    Xp = apply_candidate1(X, permutation)
    assert not np.array_equal(Xp, X) and np.array_equal(invert_candidate1(Xp, permutation), X)
    for w in range(4):          # bits never leave their 16-bit word
        assert np.array_equal(np.sort(Xp[:, 16 * w:16 * (w + 1)], axis=1), np.sort(X[:, 16 * w:16 * (w + 1)], axis=1))


# ============================================================ end to end and pairing (smoke, real)


def test_tiny_end_to_end_smoke_is_non_evidentiary(smoke):
    for e in B.EXPERIMENT_IDS:
        cert = smoke[e]
        assert cert["run_mode"] == "smoke" and cert["non_evidentiary"] is True and cert["is_evidentiary"] is False
        assert cert["sufficiency_summary"]["valid_replicates"] == 3 and "manifest" in cert
    assert smoke["_family"]["non_evidentiary"] is True and set(smoke["_family"]["decisions"]) == set(B.PRIMARY_FAMILY)
    with pytest.raises(NonConfirmatoryEvidenceError):
        E.apply_family_correction({h: smoke[h] for h in B.PRIMARY_FAMILY})


def test_pairs_share_persisted_data_and_verified_initial_weights(smoke):
    for h in B.PRIMARY_FAMILY:
        for unit in smoke[h]["replication"]["raw_replicate_results"]:
            a, b = unit["arms"]
            assert a["dataset_sha256"] == b["dataset_sha256"]
            assert a["initial_weight_sha256"] == b["initial_weight_sha256"]
            assert a["model_seed"] == b["model_seed"]
    order = smoke[H1]["replication"]["raw_replicate_results"][0]["arms"]
    assert [a["shuffle"] for a in order] == [True, False] and order[0]["input_sha256"] == order[1]["input_sha256"]
    rep = smoke[H2]["replication"]["raw_replicate_results"][0]["arms"]
    assert rep[0]["input_sha256"]["test"] != rep[1]["input_sha256"]["test"]
    assert rep[1]["permutation_sha256"] == B.load_config()["representation"]["expected_sha256"]


def test_the_representation_arm_sees_exactly_the_permuted_persisted_data(smoke):
    directory = smoke["_root"] / "smoke" / H2
    with np.load(directory / "datasets" / "pair000.npz") as data:
        X, Y = data["test_X"], data["test_Y"]
    arm_b = smoke[H2]["replication"]["raw_replicate_results"][0]["arms"][1]
    expected = A.Partition(apply_candidate1(X, E.frozen_permutation(B.load_config())), Y).sha256
    assert arm_b["input_sha256"]["test"] == expected and arm_b["dataset_sha256"]["test"] == A.Partition(X, Y).sha256


def test_noise_reruns_share_data_and_weights_and_baseline_replicates_do_not(smoke):
    noise = smoke["EV-AUTOND-NOISE"]["replication"]["raw_replicate_results"]
    assert len({u["arms"][0]["dataset_sha256"]["train"] for u in noise}) == 1
    assert len({u["arms"][0]["initial_weight_sha256"] for u in noise}) == 1
    base = smoke["EV-AUTOND-BASELINE"]["replication"]["raw_replicate_results"]
    assert len({u["arms"][0]["dataset_sha256"]["train"] for u in base}) == 3
    assert len({u["arms"][0]["initial_weight_sha256"] for u in base}) == 3


def test_paired_firewall_consumes_each_slot_once(smoke):
    fw = smoke[H2]["firewall"]
    assert fw["frozen"] and len(fw["sealed_dataset_hashes"]) == 3 and len(fw["consumed_evaluation_keys"]) == 6
    assert fw["freeze_record"]["frozen_permutation_hash"] == B.load_config()["representation"]["expected_sha256"]


@needs_checkout
def test_a_pair_whose_initial_weights_differ_is_refused(binding, tmp_path, monkeypatch):
    build, calls = A.build_model, []

    def mismatched(binding_, seed):
        calls.append(seed)
        return build(binding_, seed + len(calls) % 2)
    monkeypatch.setattr(A, "build_model", mismatched)
    with pytest.raises(RuntimeError, match="different realised initial weights"):
        tiny(H1, binding, tmp_path / "r", units=1)


# ============================================================ resumability and isolation (real, tiny)


@needs_checkout
def test_resume_reuses_results_and_data_and_refuses_a_changed_configuration(binding, tmp_path, monkeypatch):
    out = tmp_path / "r"
    first = tiny(H1, binding, out)
    stamp = (out / "datasets" / "pair000.npz").stat().st_mtime_ns
    monkeypatch.setattr(A, "train", lambda *a, **k: pytest.fail("a completed pair must not be retrained"))
    again = tiny(H1, binding, out)
    assert again["raw_paired_values"] == first["raw_paired_values"]
    assert (out / "datasets" / "pair000.npz").stat().st_mtime_ns == stamp
    assert any('"event": "resume"' in line for line in (out / "resume_ledger.jsonl").read_text().splitlines())
    changed = B.load_config()
    changed["smoke"]["protocol"]["epochs"] = 2
    with pytest.raises(ResumeViolation):
        tiny(H1, binding, out, config=changed)


@needs_checkout
def test_an_interrupted_unit_is_rerun_on_its_persisted_data(binding, tmp_path, monkeypatch):
    out, train, calls = tmp_path / "r", A.train, []

    def interrupt_second_pair(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return train(*args, **kwargs)
    monkeypatch.setattr(A, "train", interrupt_second_pair)
    with pytest.raises(KeyboardInterrupt):
        tiny(H2, binding, out)
    recorded = json.loads((out / "datasets" / "pair001.json").read_text())["sha256"]
    monkeypatch.setattr(A, "train", train)
    cert = tiny(H2, binding, out)
    assert cert["sufficiency_summary"]["valid_replicates"] == 2
    assert cert["replication"]["raw_replicate_results"][1]["arms"][0]["dataset_sha256"] == recorded


@needs_checkout
def test_consumed_but_unrecorded_confirmatory_data_are_never_evaluated_again(binding, tmp_path, monkeypatch):
    out = tmp_path / "r"
    monkeypatch.setattr(A, "accuracy", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        tiny(H1, binding, out, units=1)
    monkeypatch.undo()
    with pytest.raises(FirewallViolation, match="consumed"):
        tiny(H1, binding, out, units=1)


@needs_checkout
def test_a_failed_unit_stays_failed_and_counts_against_the_minimum(binding, tmp_path, monkeypatch):
    from framework.failures import FailureReason

    out, train, calls = tmp_path / "r", A.train, []

    def nan_first(*args, **kwargs):
        calls.append(1)
        return A.TrainingOutcome(FailureReason.NAN, {}) if len(calls) == 1 else train(*args, **kwargs)
    monkeypatch.setattr(A, "train", nan_first)
    cert = tiny("EV-AUTOND-BASELINE", binding, out)
    assert cert["sufficiency_summary"]["failed_replicates"] == 1 and cert["sufficient"] is False
    monkeypatch.setattr(A, "train", lambda *a, **k: pytest.fail("a failed unit must not be re-run"))
    assert tiny("EV-AUTOND-BASELINE", binding, out)["sufficiency_summary"]["failed_replicates"] == 1


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


@needs_checkout
def test_calibration_requires_clean_committed_sources_and_confirmation(binding, monkeypatch, tmp_path):
    config = filled()
    monkeypatch.setattr(R, "load_config", lambda *a: config)
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    dirty = {"git_head": "x", "files": {"audit/experimental_validity/autond/experiments.py": {"clean_at_head": False}}}
    monkeypatch.setattr(R, "source_identity", lambda: dirty)
    assert R.main(["run", "--mode", "calibration", "--autond-root", ROOT]) == 2
    monkeypatch.setattr(R, "source_identity", lambda: {"git_head": "x", "files": {}})
    assert R.main(["run", "--mode", "calibration", "--autond-root", ROOT]) == 0     # dry run
    assert not (tmp_path / "results").exists()


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
    return {"artifact": E.VARIANCE_ARTIFACT, "design_hash": B.design_hash(config),
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
    artifact = E.calibration_variance(list(paths.values()), config)
    assert set(artifact["inputs"]) == set(B.PRIMARY_FAMILY) and "mean" not in json.dumps(artifact)
    sd = float(np.std(np.array([0.001, -0.002, 0.003]), ddof=1))
    assert artifact["inputs"][H1]["sigma_Delta"] == pytest.approx(sd)
    assert artifact["sigma_Delta_upper_95"][H1] > sd
    production = {"run_mode": "production", "calibration": False, "non_evidentiary": False}
    for bad, message in ((production, "not a calibration"),
                         ({"design_hash": "other"}, "different design")):
        paths[H1].write_text(json.dumps(base | bad))
        with pytest.raises(ValueError, match=message):
            E.calibration_variance(list(paths.values()), config)
    paths[H1].write_text(json.dumps(base))
    config["calibration"][H1]["units"] = 4
    with pytest.raises(ValueError, match="different design"):     # the plan is part of the design hash
        E.calibration_variance(list(paths.values()), config)
