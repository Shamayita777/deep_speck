"""
AutoND D5 tests.  Tiny sizes only; no production data or training.

The end-to-end run binds the pinned AutoND checkout and uses the real DBitNet and generator on CPU.  No gate is
stubbed: it is authorised by a throwaway git repository holding byte copies of the D5 sources and an approved
test-only preregistration.  Run in its own pytest process (the binding refuses a foreign `main` already loaded).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from audit.dataset.integrity import REPOSITORY_ROOT, IntegrityError  # noqa: E402
from audit.dataset.experiments import d5_autond_training_data_scaling as D  # noqa: E402
from audit.dataset.tools import d5_autond_postrun_verify as V  # noqa: E402


def _find_root():
    for candidate in (os.environ.get("AUTOND_ROOT"), "~/Documents/AutoND-upstream"):
        if candidate and (Path(candidate).expanduser() / "main.py").is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


ROOT = _find_root()
needs_checkout = pytest.mark.skipif(ROOT is None, reason="pinned AutoND checkout not found; set AUTOND_ROOT. "
                                                         "SKIPPED IS NOT PASSED.")

#: 3 replicates x 1024 test rows; with fewer, paired differences can coincide and the run is (rightly) degenerate.
TINY = {"training_sizes": [96, 192], "replicates": 3, "epochs": 1, "batch_size": 32, "test_samples": 1024,
        "alpha": 0.05, "bootstrap_replicates": 1000, "audit_seed": 0,
        "primary_analysis": "engine_paired_t_holm_vs_smallest_size", "practical_effect_threshold": 0.01,
        "practical_effect_rule": "point_estimate_at_least_threshold",
        "partition_overlap_policy": "refuse_on_any_train_test_feature_row",
        "within_partition_duplicates_policy": "record_only",
        "failure_handling": "no_resume_new_run_directory_no_exclusions",
        "verification_tolerance": {"accuracy_abs": 1e-5, "loss_abs": 1e-5}}
STUB_PREREG = {"path": "TEST-ONLY", "sha256": "0" * 64, "commit": None, "protocol_id": "TEST-ONLY"}
PREREG_REL = "audit/targets/autond/preregistrations/d5_test_only_tiny.json"
GIT = ["git", "-c", "user.name=test-only", "-c", "user.email=test-only@invalid", "-c", "commit.gpgsign=false"]


def shipped():
    return D.load_json_strict(D.DEFAULT_PREREGISTRATION)


def _copy(run_dir: Path, tmp_path: Path) -> Path:
    shutil.copytree(run_dir, tmp_path / "copy")
    return tmp_path / "copy"


def tiny_record(**changes):
    """The shipped protocol with every decision approved at its TINY value (test-only, never a real approval)."""
    record = shipped()
    for name, value in TINY.items():
        record["decisions"][name].update(status="APPROVED", value=copy.deepcopy(value))
    record.update(approved=True, approved_by="TEST-ONLY", approved_utc="2026-01-01T00:00:00Z",
                  approval_statement="TEST-ONLY", data_examined_before_approval="TEST-ONLY")
    record.update(changes)
    return record


def make_repo(path: Path, record: dict) -> tuple[Path, dict]:
    """A git repository with byte copies of the registered sources and `record` committed at PREREG_REL."""
    path.mkdir(parents=True)
    for rel in D.SOURCE_PATHS:
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / rel, path / rel)
    (path / PREREG_REL).parent.mkdir(parents=True, exist_ok=True)
    (path / PREREG_REL).write_text(json.dumps(record, indent=2))
    for args in (["init", "-q"], ["add", "--", *D.SOURCE_PATHS, PREREG_REL], ["commit", "-q", "-m", "test-only"]):
        subprocess.run(GIT + args, cwd=str(path), check=True, capture_output=True)
    anchor = hashlib.sha256((path / PREREG_REL).read_bytes()).hexdigest()
    head = subprocess.run(["git", "log", "-1", "--format=%H%x00%cI"], cwd=str(path), check=True,
                          capture_output=True, text=True).stdout.strip().split("\x00")
    return path, {"path": PREREG_REL, "sha256": anchor, "protocol_id": record["protocol_id"],
                  "commit": {"sha": head[0], "committed_utc": head[1], "path": PREREG_REL}}


@pytest.fixture(scope="module", autouse=True)
def mini_repo(tmp_path_factory):
    """Every gate and provenance check in this module resolves against the throwaway repository."""
    repo, prereg = make_repo(tmp_path_factory.mktemp("repo") / "repo", tiny_record())
    patch = pytest.MonkeyPatch()
    patch.setattr(V, "REPOSITORY_ROOT", repo)
    yield repo, prereg
    patch.undo()


@pytest.fixture
def engine_never_runs(monkeypatch):
    from audit.dataset import d5_training_data_scaling as engine

    calls = []
    monkeypatch.setattr(engine, "run_d5", lambda **kw: calls.append(kw))
    return calls


@pytest.fixture(scope="module")
def binding():
    if ROOT is None:
        pytest.skip("pinned AutoND checkout not found; SKIPPED IS NOT PASSED.")
    return D.bind_autond(ROOT)


@pytest.fixture(scope="module")
def completed(binding, mini_repo, tmp_path_factory):
    """One tiny end-to-end run through the unmodified engine."""
    run_dir = tmp_path_factory.mktemp("d5") / "run"
    code = D.execute(copy.deepcopy(TINY), run_dir=run_dir, binding=binding, prereg=mini_repo[1],
                     sources=D.require_clean_sources(), environment={"test": True})
    return run_dir, code


# ============================================================ preregistration


def test_shipped_preregistration_is_approved_and_matches_the_driver():
    record = shipped()
    review = D.review_preregistration(record)
    assert review["problems"] == [] and review["pending"] == [] and D.check_fixed_design(record) == []
    assert D.check_decision_values(review["values"]) == [] and set(review["values"]) == set(D.DECISIONS)
    assert record["claim_scope"] == D.CLAIM_SCOPE and "not separated" in record["training_policy"]
    assert D.main(["protocol-check", "--require-approved"]) == 0


@pytest.mark.parametrize("section, key, value", [
    ("target", "cipher", "simon3264"), ("target", "num_rounds", 6), ("upstream", "commit", "0" * 40),
    ("engine", "schema_version", "2.0"), ("training", "model_selection", "min_val_loss")])
def test_a_different_design_in_the_preregistration_is_refused(section, key, value):
    record = shipped()
    record["fixed_design"][section][key] = value
    assert any(f"fixed_design.{section}" in p for p in D.check_fixed_design(record))


def test_decision_values_are_validated(tmp_path):
    assert D.check_decision_values(TINY) == []
    for change in ({"training_sizes": [192, 96]}, {"replicates": 1}, {"alpha": 0.01},
                   {"practical_effect_threshold": 0}, {"partition_overlap_policy": "ignore"}):
        assert D.check_decision_values({**TINY, **change}), change
    (tmp_path / "dup.json").write_text('{"a": 1, "a": 2}')
    with pytest.raises(IntegrityError, match="duplicate"):
        D.load_json_strict(tmp_path / "dup.json")


# ============================================================ gates before anything is created


def test_run_refuses_without_confirmation_or_approval(tmp_path, engine_never_runs):
    assert D.main(["run", "--preregistration-sha256", "0" * 64, "--run-dir", str(tmp_path / "r")]) == D.EXIT_REFUSED
    # never the shipped (approved) protocol with its real anchor: that would start production
    repo, prereg = make_repo(tmp_path / "repo", tiny_record(approved=False, approved_by=None))
    code = D.main(["run", "--preregistration", str(repo / PREREG_REL), "--preregistration-sha256", prereg["sha256"],
                   "--run-dir", str(tmp_path / "r"), "--confirm", D.CONFIRM_RUN, "--autond-root", str(ROOT or tmp_path)])
    assert code == D.EXIT_REFUSED and not (tmp_path / "r").exists() and engine_never_runs == []


def test_run_and_execute_refuse_without_scipy(binding, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    code = D.main(["run", "--preregistration-sha256", "0" * 64, "--run-dir", str(tmp_path / "r"),
                   "--confirm", D.CONFIRM_RUN])
    assert code == D.EXIT_REFUSED
    with pytest.raises(IntegrityError, match="SciPy"):
        D.execute(copy.deepcopy(TINY), run_dir=tmp_path / "r", binding=binding, prereg=STUB_PREREG,
                  sources={}, environment={})
    assert not (tmp_path / "r").exists()


@needs_checkout
@pytest.mark.parametrize("case, message", [
    ("missing", "AutoND D5 preregistration is missing"),
    ("wrong_anchor", "differs from the approver's anchor"),
    ("values", "values differ from the approved preregistration: ['epochs']"),
    ("commit", "commit or protocol id"),
    ("sources", "source identity supplied by the caller differs"),
    ("binding", "not the verified pin"),
])
def test_execute_refuses_a_caller_that_does_not_match_the_approved_record(case, message, binding, mini_repo,
                                                                            tmp_path, engine_never_runs):
    values, prereg, sources = copy.deepcopy(TINY), copy.deepcopy(mini_repo[1]), D.require_clean_sources()
    use_binding = binding
    if case == "missing":
        prereg = STUB_PREREG
    elif case == "wrong_anchor":
        prereg["sha256"] = "f" * 64
    elif case == "values":
        values["epochs"] = 2
    elif case == "commit":
        prereg["commit"] = {**prereg["commit"], "sha": "0" * 40}
    elif case == "sources":
        sources["files"][D.SOURCE_PATHS[-1]]["sha256"] = "0" * 64
    elif case == "binding":
        use_binding = types.SimpleNamespace(pin_verified_preimport=True, expected_upstream_commit="0" * 40,
                                            source_digests=binding.source_digests)
    with pytest.raises(IntegrityError, match=re.escape(message)):
        D.execute(values, run_dir=tmp_path / "r", binding=use_binding, prereg=prereg, sources=sources,
                  environment={})
    assert not (tmp_path / "r").exists() and engine_never_runs == []


@needs_checkout
@pytest.mark.parametrize("case, message", [
    ("unapproved", "not approved"),
    ("pending_decision", "decisions.practical_effect_threshold"),
    ("uncommitted_preregistration", "uncommitted modifications"),
    ("dirty_source", "not clean at HEAD"),
])
def test_execute_refuses_an_unapproved_protocol_or_dirty_tree(case, message, binding, tmp_path, monkeypatch,
                                                               engine_never_runs):
    record = tiny_record(approved=False) if case == "unapproved" else tiny_record()
    if case == "pending_decision":
        record["decisions"]["practical_effect_threshold"].update(status=D.PENDING, value=None)
    repo, prereg = make_repo(tmp_path / "repo", record)
    monkeypatch.setattr(V, "REPOSITORY_ROOT", repo)
    sources = D.require_clean_sources()
    if case == "uncommitted_preregistration":
        (repo / PREREG_REL).write_text((repo / PREREG_REL).read_text() + "\n")
        prereg["sha256"] = hashlib.sha256((repo / PREREG_REL).read_bytes()).hexdigest()
    elif case == "dirty_source":
        with open(repo / D.SOURCE_PATHS[-1], "a") as handle:
            handle.write("\n# local edit\n")
    with pytest.raises(IntegrityError, match=message):
        D.execute(copy.deepcopy(TINY), run_dir=tmp_path / "r", binding=binding, prereg=prereg, sources=sources,
                  environment={})
    assert not (tmp_path / "r").exists() and engine_never_runs == []


# ============================================================ AutoND binding


@needs_checkout
@pytest.mark.parametrize("change, message", [("commit", "revision"), ("digest", "pin mismatch")])
def test_a_wrong_upstream_pin_is_refused_before_import(change, message):
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter, SourceBindingError
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    commit, digests = AUTOND_SOURCE_PIN["upstream_commit"], dict(AUTOND_SOURCE_PIN["source_digests"])
    if change == "commit":
        commit = "0" * 40
    else:
        digests["dbitnet.py"] = "f" * 64
    with pytest.raises(SourceBindingError, match=message):
        AutoNDDatasetAdapter(autond_root=ROOT, cipher_name=D.CIPHER, delta_plain_int=D.DELTA_PLAIN,
                             num_rounds=D.NUM_ROUNDS, expected_upstream_commit=commit,
                             expected_source_digests=digests)


def test_imports_resolve_to_the_pinned_checkout(binding, monkeypatch):
    from audit.dataset.adapters.autond import _is_within, _module_locations

    assert binding.pin_verified_preimport and binding.feature_bits == 64 and binding.num_rounds == 5
    for name in ("main", "optimizer", "train_nets", "dbitnet", "gohrnet", "ciphers.speck3264"):
        assert all(_is_within(loc, Path(ROOT)) for loc in _module_locations(sys.modules[name])), name
    assert binding.source_digests == shipped()["fixed_design"]["upstream"]["source_digests"]
    fake = types.ModuleType("dbitnet")
    fake.__file__ = "/elsewhere/dbitnet.py"
    monkeypatch.setitem(sys.modules, "dbitnet", fake)
    with pytest.raises(IntegrityError, match="outside"):
        D.pinned_module(binding, "dbitnet")


def test_same_seed_gives_identical_initial_weights_and_the_upstream_compile(binding):
    factory = D.make_model_factory(binding)
    a, b, c = factory(7), factory(7), factory(8)
    assert all(np.array_equal(u, v) for u, v in zip(a.get_weights(), b.get_weights()))
    assert not all(np.array_equal(u, v) for u, v in zip(a.get_weights(), c.get_weights()))
    assert a.input_shape == (None, 64, 1) and a.loss == "mse"
    assert type(a.optimizer).__name__ == "Adam" and a.optimizer.get_config()["amsgrad"] is True


def test_the_generator_produces_the_registered_representation(binding):
    x, y = binding.generate_partition(256)
    assert D.validate_partition(x, y, 256)["rows"] == 256
    assert binding.delta_plain_int == 0x400000 and binding.delta_key_int == 0 and binding.scenario == "single-key"


# ============================================================ data checks and run directory


@pytest.mark.parametrize("x, y, message", [
    (np.zeros((10, 63), np.uint8), np.zeros(10, np.uint8), "shape"),
    (np.zeros((10, 64), np.float32), np.zeros(10, np.uint8), "dtypes"),
    (np.full((10, 64), 2, np.uint8), np.zeros(10, np.uint8), "outside"),
    (np.zeros((10, 64), np.uint8), np.full(10, 3, np.uint8), "labels contain"),
])
def test_invalid_partitions_are_refused(x, y, message):
    with pytest.raises(IntegrityError, match=message):
        D.validate_partition(x, y, 10)


class _FakeBinding:
    def __init__(self, partitions):
        self.partitions = list(partitions)

    def generate_partition(self, samples):
        return self.partitions.pop(0)


def test_train_test_overlap_is_refused_under_the_registered_policy(tmp_path):
    rng = np.random.default_rng(0)
    test = (rng.integers(0, 2, (32, 64), dtype=np.uint8), rng.integers(0, 2, 32, dtype=np.uint8))
    train_x = rng.integers(0, 2, (64, 64), dtype=np.uint8)
    train_x[5] = test[0][3]
    train = (train_x, rng.integers(0, 2, 64, dtype=np.uint8))
    gate = D.DataGate(_FakeBinding([test, train]), tmp_path / "a", TINY)
    gate.test_factory(32, 0)
    with pytest.raises(IntegrityError, match="equal a test row"):
        gate.train_factory(64, 0)
    assert json.loads((tmp_path / "a" / "data_audit" / "replicate_01.json").read_text())["train_test_overlap"]["rows"] == 1
    lenient = D.DataGate(_FakeBinding([test, train]), tmp_path / "b", {**TINY, "partition_overlap_policy": "record_only"})
    lenient.test_factory(32, 0)
    assert len(lenient.train_factory(64, 0)[0]) == 64


def test_only_a_new_or_empty_run_directory_is_claimed(tmp_path):
    D.claim_run_directory(tmp_path / "new", {"run_id": "a"})
    (tmp_path / "empty").mkdir()
    D.claim_run_directory(tmp_path / "empty", {"run_id": "b"})
    with pytest.raises(IntegrityError, match="not empty"):
        D.claim_run_directory(tmp_path / "new", {"run_id": "c"})


def test_an_interrupted_run_is_never_resumed(completed, binding, mini_repo, tmp_path):
    interrupted = _copy(completed[0], tmp_path)
    shutil.rmtree(interrupted / V.RESULT_DIR)
    with pytest.raises(IntegrityError, match="never resumed"):
        D.execute(copy.deepcopy(TINY), run_dir=interrupted, binding=binding, prereg=mini_repo[1],
                  sources=D.require_clean_sources(), environment={})


# ============================================================ end to end (tiny, real)


def test_tiny_end_to_end_run_is_verified(completed):
    run_dir, code = completed
    assert code == D.EXIT_OK
    certificate = json.loads((run_dir / V.RESULT_DIR / V.TARGET_CERTIFICATE).read_text())
    assert certificate["status"] == "VERIFIED" and certificate["problems"] == []
    engine_section = certificate["engine_generated_result"]
    assert engine_section["sha256"] == hashlib.sha256((run_dir / engine_section["path"]).read_bytes()).hexdigest()
    target = certificate["target_independent_verification"]
    assert set(target["conditions"]) == {V.condition_name(r, n) for r, n in V.expected_conditions([96, 192], 3)}
    for entry in target["conditions"].values():
        assert entry["absolute_differences"]["accuracy"] <= 1e-5 and entry["absolute_differences"]["loss"] <= 1e-5
        assert entry["independent_evaluation"]["regularization_loss"] > 0      # engine test_loss = MSE + L2
    assert target["supplementary_statistics"]["contrasts_vs_baseline"]["96_vs_192"]["classification"] in V.CONTRAST_CLASSES


def test_prefixes_are_nested_replicates_independent_and_seeds_shared_across_sizes(completed):
    from audit.dataset import d5_training_data_scaling as engine

    root = completed[0] / V.ENGINE_DIR
    datasets = []
    for r in (1, 2):
        x, y, meta = engine.load_replicate_dataset(root, r, sizes=[96, 192])
        datasets.append(np.asarray(x))
        seeds = set()
        for n in (96, 192):
            hashes = meta["nested_prefix_hashes"][str(n)]
            assert engine._array_logical_sha256(x[:n]) == hashes["train_x_logical_sha256"]
            assert engine._array_logical_sha256(y[:n]) == hashes["train_y_logical_sha256"]
            state = json.loads(engine.condition_paths(root, r, n)["state"].read_text())
            assert state["train_x_prefix_hash"] == hashes["train_x_logical_sha256"]
            seeds.add(state["seed"])
        assert len(seeds) == 1
    assert not np.array_equal(*datasets)
    audit = completed[0] / "data_audit"
    assert sorted(p.name for p in audit.iterdir()) == ["replicate_01.json", "replicate_02.json", "replicate_03.json",
                                                      "test.json"]


def _append_byte(run_dir):
    with open(run_dir / V.ENGINE_DIR / "runs" / "replicate_01" / "n_96" / "latest.keras", "ab") as handle:
        handle.write(b"\0")


def _shift_accuracy(run_dir):
    path = run_dir / V.ENGINE_DIR / "runs" / "replicate_01" / "n_192" / "state.json"
    state = json.loads(path.read_text())
    state["test_accuracy"] += 0.25
    path.write_text(json.dumps(state))


def _unbind_audit(run_dir):
    path = run_dir / "data_audit" / "replicate_01.json"
    record = json.loads(path.read_text())
    record["x_logical_sha256"] = "0" * 64
    path.write_text(json.dumps(record))


@pytest.mark.parametrize("tamper, message", [
    (_append_byte, "replicate_01/n_96: checkpoint bytes differ"),
    (lambda d: (d / V.ENGINE_DIR / "runs" / "replicate_02" / "n_192" / "latest.keras").unlink(),
     "replicate_02/n_192: checkpoint latest.keras is missing"),
    (lambda d: shutil.rmtree(d / V.ENGINE_DIR / "runs" / "replicate_02" / "n_96"), "replicate_02/n_96"),
    (_shift_accuracy, "differ from the metrics logged"),
    (_unbind_audit, "replicate_01.json does not describe the persisted"),
])
def test_tampered_or_incomplete_runs_are_not_verified(tamper, message, completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    tamper(altered)
    report = V.verify_run(altered)
    assert report["status"] == "NOT_VERIFIED" and V.build_certificate(altered, report)["status"] == "NOT_VERIFIED"
    assert any(message in p for p in report["problems"]), report["problems"]


def test_hash_only_verification_is_never_verified_and_detects_later_changes(completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    assert D.main(["verify", "--run-dir", str(altered), "--no-reevaluate"]) == D.EXIT_OK
    assert V.verify_run(altered, reevaluate=False)["status"] == "INTEGRITY_CONSISTENT_NOT_REEVALUATED"
    (altered / V.ENGINE_DIR / "runs" / "replicate_01" / "n_192" / "latest.keras").write_bytes(b"replaced")
    assert D.main(["verify", "--run-dir", str(altered), "--no-reevaluate"]) == D.EXIT_NOT_VERIFIED


def _claim_edit(case, claim):
    if case == "no_preregistration":
        claim.pop("preregistration")
    elif case == "anchor":
        claim["preregistration"]["sha256"] = "f" * 64
    elif case == "commit":
        claim["preregistration"]["commit"]["sha"] = "0" * 40
    elif case == "values":
        claim["values"]["practical_effect_threshold"] = 0.5
    elif case == "dirty_source":
        claim["source_identity"]["files"][D.SOURCE_PATHS[0]]["clean_at_head"] = False
    elif case == "source_digest":
        claim["source_identity"]["files"][D.SOURCE_PATHS[-1]]["sha256"] = "0" * 64
    elif case == "upstream":
        claim["upstream"]["commit"] = "0" * 40


@pytest.mark.parametrize("case, message", [
    ("no_preregistration", "no anchored, committed preregistration record"),
    ("anchor", "does not match the recorded anchor"),
    ("commit", "is not available at commit"),
    ("values", "run values differ from the approved"),
    ("dirty_source", "were not clean at HEAD"),
    ("source_digest", "recorded SHA-256 is not its committed content"),
    ("upstream", "differ from the approved pin"),
])
def test_runs_without_matching_approval_or_source_provenance_are_not_verified(case, message, completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    claim = json.loads((altered / V.RUN_CLAIM).read_text())
    _claim_edit(case, claim)
    (altered / V.RUN_CLAIM).write_text(json.dumps(claim))
    report = V.verify_run(altered, reevaluate=False)
    assert report["status"] == "NOT_VERIFIED"
    assert any(message in p for p in report["problems"]), report["problems"]


def test_verification_requires_scipy_and_rejects_fallback_statistics(completed, tmp_path, monkeypatch):
    """Recompute the engine's aggregates with SciPy blocked (its normal fallback) and put them in a copy."""
    from audit.dataset import d5_training_data_scaling as engine

    altered = _copy(completed[0], tmp_path)
    path = altered / V.RESULT_DIR / V.ENGINE_CERTIFICATE
    certificate = json.loads(path.read_text())
    observations = [engine.D5Observation(**o) for o in certificate["findings"]["observations"]]
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "scipy", None)
        m.setitem(sys.modules, "scipy.stats", None)
        assert "SciPy" in V.verify_run(completed[0], reevaluate=False)["problems"][0]
        _, fallback, _ = engine._aggregate(observations, [96, 192], 3, 1000, 0 + 991)
    certificate["findings"]["paired_contrasts"] = json.loads(json.dumps(fallback))
    path.write_text(json.dumps(certificate))
    problems = V.verify_run(altered, reevaluate=False)["problems"]
    assert any("paired contrasts differ from their recomputation" in p for p in problems)


# ============================================================ statistics (synthetic)


def test_independent_reevaluation_matches_keras(tmp_path):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(3)
    model = tf.keras.Sequential([tf.keras.Input((64,)), tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="mse", metrics=["acc"])
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, (300, 64), dtype=np.uint8)
    y = rng.integers(0, 2, 300, dtype=np.uint8)
    model.fit(x, y, epochs=1, batch_size=50, verbose=0)
    loss, accuracy = model.evaluate(x, y, batch_size=50, verbose=0)
    model.save(tmp_path / "m.keras")
    again = V.reevaluate_checkpoint(tmp_path / "m.keras", x, y, batch_size=64)
    assert abs(again["accuracy"] - accuracy) <= 1e-6 and abs(again["loss_objective"] - loss) <= 1e-6


def test_classification_never_claims_equivalence():
    t = {"lower": -0.001, "upper": 0.02}

    def classify(holm_p, mean, rule):
        return V.classify_contrast(holm_p=holm_p, mean_difference=mean, t_interval=t, alpha=0.05,
                                   practical_threshold=0.01, practical_rule=rule)

    assert classify(0.2, 0.05, "point_estimate_at_least_threshold") == "NOT_SIGNIFICANT_INCONCLUSIVE"
    assert classify(0.01, 0.015, "t_interval_excludes_threshold_band") == "SIGNIFICANT_BELOW_PRACTICAL_THRESHOLD"
    assert classify(0.01, 0.015, "point_estimate_at_least_threshold") == "SIGNIFICANT_AND_PRACTICALLY_MEANINGFUL"


def test_paired_supplement_matches_scipy():
    from scipy import stats

    d = [0.01, 0.012, 0.008, 0.011, 0.009, 0.013]
    out = V.paired_supplement(d)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=stats.sem(d))
    assert out["t_interval"]["lower"] == pytest.approx(lo) and out["t_interval"]["upper"] == pytest.approx(hi)
    assert out["cohens_dz"] == pytest.approx(np.mean(d) / np.std(d, ddof=1))


SIZES3 = [96, 192, 288]


def _findings(sd=0.004, p=0.01, holm=None, accuracy=0.9, replicates=2):
    """A complete, well-formed engine 'findings' block for 2 replicates x 3 sizes."""
    contrasts = {f"96_vs_{n}": {"replicates": replicates, "mean_accuracy_difference": 0.01,
                                "sd_accuracy_difference": sd, "bootstrap_ci95": [0.005, 0.015],
                                "paired_t_statistic": 3.0, "paired_p_value": p} for n in (192, 288)}
    adjusted = holm if holm is not None else {k: min(1.0, 2 * p) for k in contrasts}
    observations = [{"replicate": r, "training_samples": n, "test_accuracy": accuracy, "test_loss": 0.2}
                    for r in (1, 2) for n in SIZES3]
    summary = {str(n): {"replicates": replicates, "accuracy_mean": 0.9, "accuracy_sd": 0.01,
                        "accuracy_ci95_bootstrap": [0.89, 0.91]} for n in SIZES3}
    return {"observations": observations, "summary_by_training_size": summary,
            "paired_contrasts": {**contrasts, "baseline_training_size": 96,
                                 "multiple_comparison_correction": {"method": "Holm step-down",
                                                                    "adjusted_p_values": adjusted}}}


def _edit(path, value):
    findings = _findings()
    target = findings
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return findings


def _problems(findings):
    return V.statistical_integrity_problems(findings, sizes=SIZES3, replicates=2)


def test_complete_finite_statistics_pass():
    assert _problems(_findings()) == []


@pytest.mark.parametrize("findings, message", [
    (_findings(p=float("nan")), "is not a probability"),
    (_findings(p=1.5), "raw p 1.5 is not a probability"),
    (_findings(sd=0.0), "zero variance"),
    (_findings(sd=float("inf")), "sd_accuracy_difference is not finite"),
    (_findings(holm={"96_vs_192": 0.02}), "expected Holm"),
    (_findings(holm={"96_vs_192": 0.01, "96_vs_288": 0.01}), "differs from Holm"),
    (_findings(accuracy=float("nan")), "non-finite observation"),
    (_findings(replicates=1), "pairs 1 replicates, expected 2"),
    (_edit(["paired_contrasts", "96_vs_192", "mean_accuracy_difference"], None), "mean_accuracy_difference is not finite"),
    (_edit(["paired_contrasts", "baseline_training_size"], 192), "expected the smallest size"),
    (_edit(["observations"], _findings()["observations"][:-1]), "pairing is incomplete or duplicated"),
])
def test_non_finite_degenerate_or_incomplete_statistics_are_refused(findings, message):
    assert any(message in problem for problem in _problems(findings)), _problems(findings)


def test_non_finite_supplementary_intervals_are_refused():
    good = {"contrasts_vs_baseline": {"96_vs_192": {"mean": 0.01, "cohens_dz": 2.0,
                                                    "t_interval": {"lower": 0.0, "upper": 0.02}}},
            "per_training_size_accuracy": {"96": {"lower": 0.8, "upper": 0.9}}}
    assert V.supplementary_problems(good) == []
    bad = copy.deepcopy(good)
    bad["contrasts_vs_baseline"]["96_vs_192"]["t_interval"]["upper"] = float("inf")
    bad["per_training_size_accuracy"]["96"]["lower"] = float("nan")
    assert len(V.supplementary_problems(bad)) == 2
