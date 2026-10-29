"""
AutoND D5 target tests.  No production-sized data or training: every size here is tiny.

Labelled scope of each group:
  [real]  binds the pinned AutoND checkout and uses the real DBitNet model and main.make_train_data on
          tiny sizes (CPU).  No gate is stubbed: the end-to-end run is authorised by a throwaway git
          repository holding byte copies of the D5 sources and an approved TINY preregistration (test-only).
  [mock]  synthetic arrays or a tiny Keras model; does not exercise the AutoND model or data path.
  [gate]  checks a refusal before anything is created.

Run in its own pytest process: the AutoND binding refuses a foreign `main`/`train_nets` already loaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from audit.dataset.integrity import REPOSITORY_ROOT, IntegrityError  # noqa: E402
from audit.targets.autond.d5 import training_data_scaling as D  # noqa: E402
from audit.targets.autond.d5 import verification as V  # noqa: E402


def _find_root():
    for candidate in (os.environ.get("AUTOND_ROOT"), "~/Documents/AutoND-upstream"):
        if candidate and (Path(candidate).expanduser() / "main.py").is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


ROOT = _find_root()
needs_checkout = pytest.mark.skipif(ROOT is None, reason="pinned AutoND checkout not found; set AUTOND_ROOT. "
                                                         "SKIPPED IS NOT PASSED.")

#: Files this target must never modify; digested at import, before any fixture runs.
PROTECTED = [REPOSITORY_ROOT / p for p in (
    "audit/dataset/d5_training_data_scaling.py", "audit/dataset/integrity.py", "audit/dataset/adapters/autond.py",
    "audit/dataset/adapters/gohr.py", "audit/dataset/adapters/autond_d4.py",
    "audit/dataset/experiments/d5_gohr_training_data_scaling.py", "audit/dataset/experiments/d0_autond_canonical.py",
    "audit/dataset/experiments/d4_autond_label_shuffle.py", "audit/dataset/experiments/d4_autond_two_gpu.py",
    "audit/dataset/d4_controlled_perturbation.py",
    "audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.json",
    "audit/targets/autond/SHA256SUMS.txt",
)]
PROTECTED_DIGESTS = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED if p.is_file()}

#: 3 replicates x 1024 test rows: with 2 x 64 the two paired differences coincide often enough (steps of 1/64)
#: that the zero-variance refusal fires and the fixture itself becomes a flaky, degenerate run.
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


def _copy(run_dir: Path, tmp_path: Path, name="copy") -> Path:
    target = tmp_path / name
    shutil.copytree(run_dir, target)
    return target


def tiny_record(**changes):
    """The shipped protocol with every decision approved at its TINY value (test-only, never a real approval)."""
    record = shipped()
    for name, value in TINY.items():
        record["decisions"][name].update(status="APPROVED", value=copy.deepcopy(value))
    record.update(approved=True, approved_by="TEST-ONLY", approved_utc="2026-01-01T00:00:00Z",
                  approval_statement="TEST-ONLY fixture", data_examined_before_approval="TEST-ONLY fixture")
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
    """[real] One tiny end-to-end run through the unmodified engine, authorised by the throwaway repository."""
    run_dir = tmp_path_factory.mktemp("d5") / "run"
    code = D.execute(copy.deepcopy(TINY), run_dir=run_dir, binding=binding, prereg=mini_repo[1],
                     sources=D.require_clean_sources(), environment={"stubbed_in_test": True})
    return run_dir, code


# ============================================================ preregistration [gate]


def test_shipped_preregistration_is_approved_and_well_formed():
    record = shipped()
    review = D.review_preregistration(record)
    assert review["problems"] == [] and review["pending"] == [] and D.check_fixed_design(record) == []
    assert D.check_decision_values(review["values"]) == [] and set(review["values"]) == set(D.DECISIONS)
    assert record["approved"] is True and record["status"] == "APPROVED"
    assert D.main(["protocol-check"]) == 0
    assert D.main(["protocol-check", "--require-approved"]) == 0


def test_recommended_values_are_the_frozen_proposed_design():
    d = shipped()["decisions"]
    assert d["training_sizes"]["recommended"] == [2_500_000, 5_000_000, 7_500_000, 10_000_000]
    assert (d["replicates"]["recommended"], d["epochs"]["recommended"], d["batch_size"]["recommended"],
            d["test_samples"]["recommended"]) == (12, 10, 5000, 1_000_000)
    assert d["practical_effect_threshold"]["recommended"] is None      # left to the protocol owner
    assert d["alpha"]["options"] == [D.ENGINE_HOLM_ALPHA]


@pytest.mark.parametrize("section, key, value", [
    ("target", "cipher", "simon3264"), ("target", "delta_plain_hex", "0x400"), ("target", "scenario", "related-key"),
    ("target", "num_rounds", 6), ("target", "delta_key_hex", "0x1"), ("upstream", "commit", "0" * 40),
    ("engine", "schema_version", "2.0")])
def test_wrong_target_or_pin_in_the_preregistration_is_refused(section, key, value):
    record = shipped()
    record["fixed_design"][section][key] = value
    assert any(f"fixed_design.{section}.{key}" in p for p in D.check_fixed_design(record))


def test_wrong_training_semantics_in_the_preregistration_is_refused():
    record = shipped()
    record["fixed_design"]["training"]["model_selection"] = "min_val_loss"
    assert any("fixed_design.training" in p for p in D.check_fixed_design(record))


def test_decision_values_are_validated():
    assert D.check_decision_values(TINY) == []
    for change in ({"training_sizes": [192, 96]}, {"training_sizes": [96]}, {"replicates": 1}, {"alpha": 0.01},
                   {"practical_effect_threshold": 0}, {"partition_overlap_policy": "ignore"},
                   {"verification_tolerance": {"accuracy_abs": 0.0, "loss_abs": 1e-5}}):
        assert D.check_decision_values({**TINY, **change}), change


def test_duplicate_json_keys_are_refused(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(IntegrityError, match="duplicate"):
        D.load_json_strict(path)


def test_run_refuses_without_confirmation(tmp_path):
    assert D.main(["run", "--preregistration-sha256", "0" * 64, "--run-dir", str(tmp_path / "r")]) == D.EXIT_REFUSED
    assert not (tmp_path / "r").exists()


def test_run_refuses_an_unapproved_preregistration_before_creating_anything(tmp_path, engine_never_runs):
    # never the shipped (approved) protocol with its real anchor: that would start production
    repo, prereg = make_repo(tmp_path / "repo", tiny_record(approved=False, approved_by=None))
    code = D.main(["run", "--preregistration", str(repo / PREREG_REL), "--preregistration-sha256", prereg["sha256"],
                   "--run-dir", str(tmp_path / "r"), "--confirm", D.CONFIRM_RUN, "--autond-root", str(ROOT or tmp_path)])
    assert code == D.EXIT_REFUSED and not (tmp_path / "r").exists() and engine_never_runs == []


def test_statistical_dependencies_are_checked_before_any_run_begins(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(D, "load_approved_preregistration", lambda *a: calls.append("prereg"))
    monkeypatch.setitem(sys.modules, "scipy", None)          # import scipy -> ImportError
    with pytest.raises(IntegrityError, match="SciPy"):
        V.require_statistics_dependencies()
    code = D.main(["run", "--preregistration-sha256", "0" * 64, "--run-dir", str(tmp_path / "r"),
                   "--confirm", D.CONFIRM_RUN])
    assert code == D.EXIT_REFUSED and calls == [] and not (tmp_path / "r").exists()


# ============================================================ source binding [real]


@needs_checkout
def test_wrong_upstream_commit_is_refused_before_import():
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter, SourceBindingError
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    with pytest.raises(SourceBindingError, match="revision"):
        AutoNDDatasetAdapter(autond_root=ROOT, cipher_name=D.CIPHER, delta_plain_int=D.DELTA_PLAIN,
                             num_rounds=D.NUM_ROUNDS, expected_upstream_commit="0" * 40,
                             expected_source_digests=AUTOND_SOURCE_PIN["source_digests"])


@needs_checkout
def test_wrong_source_digest_is_refused_before_import():
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter, SourceBindingError
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    digests = dict(AUTOND_SOURCE_PIN["source_digests"], **{"dbitnet.py": "f" * 64})
    with pytest.raises(SourceBindingError, match="pin mismatch"):
        AutoNDDatasetAdapter(autond_root=ROOT, cipher_name=D.CIPHER, delta_plain_int=D.DELTA_PLAIN,
                             num_rounds=D.NUM_ROUNDS, expected_upstream_commit=AUTOND_SOURCE_PIN["upstream_commit"],
                             expected_source_digests=digests)


def test_the_pin_used_by_the_driver_matches_the_checkout_files():
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    if ROOT is None:
        pytest.skip("pinned AutoND checkout not found; SKIPPED IS NOT PASSED.")
    for rel, digest in AUTOND_SOURCE_PIN["source_digests"].items():
        assert hashlib.sha256((Path(ROOT) / rel).read_bytes()).hexdigest() == digest, rel


def test_imports_resolve_to_the_pinned_checkout(binding):
    from audit.dataset.adapters.autond import _is_within, _module_locations

    assert binding.pin_verified_preimport and binding.feature_bits == 64 and binding.num_rounds == 5
    for name in ("main", "optimizer", "train_nets", "dbitnet", "gohrnet", "ciphers.speck3264"):
        assert all(_is_within(loc, Path(ROOT)) for loc in _module_locations(sys.modules[name])), name
    assert binding.source_digests == shipped()["fixed_design"]["upstream"]["source_digests"]


def test_a_foreign_model_module_is_refused(binding, monkeypatch):
    fake = types.ModuleType("dbitnet")
    fake.__file__ = "/elsewhere/dbitnet.py"
    monkeypatch.setitem(sys.modules, "dbitnet", fake)
    with pytest.raises(IntegrityError, match="outside"):
        D.pinned_module(binding, "dbitnet")


def test_same_seed_gives_identical_initial_weights_and_the_verified_compile(binding):
    factory = D.make_model_factory(binding)
    a, b, c = factory(7), factory(7), factory(8)
    assert all(np.array_equal(u, v) for u, v in zip(a.get_weights(), b.get_weights()))
    assert not all(np.array_equal(u, v) for u, v in zip(a.get_weights(), c.get_weights()))
    # Upstream declares Input(shape=(64, 1)) and feeds (n, 64) arrays, which Keras expands; D5 does the same.
    assert a.input_shape == (None, 64, 1) and a.loss == "mse"
    assert a.predict(np.zeros((2, 64), np.uint8), verbose=0).shape == (2, 1)
    assert type(a.optimizer).__name__ == "Adam" and a.optimizer.get_config()["amsgrad"] is True


def test_the_generator_produces_the_verified_representation(binding):
    x, y = binding.generate_partition(256)
    record = D.validate_partition(x, y, 256)
    assert record["rows"] == 256 and record["label_ones"] + record["label_zeros"] == 256
    assert binding.delta_plain_int == 0x400000 and binding.delta_key_int == 0 and binding.scenario == "single-key"


# ============================================================ data checks [mock]


@pytest.mark.parametrize("x, y, message", [
    (np.zeros((10, 63), np.uint8), np.zeros(10, np.uint8), "shape"),
    (np.zeros((9, 64), np.uint8), np.zeros(9, np.uint8), "shape"),
    (np.zeros((10, 64), np.float32), np.zeros(10, np.uint8), "dtypes"),
    (np.full((10, 64), 2, np.uint8), np.zeros(10, np.uint8), "outside"),
    (np.zeros((10, 64), np.uint8), np.full(10, 3, np.uint8), "labels contain"),
    (np.zeros((10, 64), np.uint8), np.zeros((10, 1), np.uint8), "labels have shape"),
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
    record = json.loads((tmp_path / "a" / "data_audit" / "replicate_01.json").read_text())
    assert record["train_test_overlap"]["rows"] == 1
    lenient = D.DataGate(_FakeBinding([test, train]), tmp_path / "b", {**TINY, "partition_overlap_policy": "record_only"})
    lenient.test_factory(32, 0)
    assert len(lenient.train_factory(64, 0)[0]) == 64


def test_training_data_before_the_test_set_is_refused(tmp_path):
    gate = D.DataGate(_FakeBinding([]), tmp_path, TINY)
    with pytest.raises(IntegrityError, match="before the fixed test set"):
        gate.train_factory(10, 0)


# ============================================================ run directory policy [gate]


def test_nonempty_run_directory_is_refused(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "stray.txt").write_text("x")
    with pytest.raises(IntegrityError, match="not empty"):
        D.claim_run_directory(tmp_path / "r", {"run_id": "x"})


def test_new_or_empty_run_directory_is_claimed_exclusively(tmp_path):
    D.claim_run_directory(tmp_path / "new", {"run_id": "a"})
    (tmp_path / "empty").mkdir()
    D.claim_run_directory(tmp_path / "empty", {"run_id": "b"})
    with pytest.raises(IntegrityError):
        D.claim_run_directory(tmp_path / "new", {"run_id": "c"})


def test_an_interrupted_run_is_never_resumed_even_with_a_changed_configuration(completed, binding, mini_repo,
                                                                               tmp_path):
    run_dir, _ = completed
    interrupted = _copy(run_dir, tmp_path)
    shutil.rmtree(interrupted / V.RESULT_DIR)
    with pytest.raises(IntegrityError, match="never resumed"):          # approved values: reaches the claim
        D.execute(copy.deepcopy(TINY), run_dir=interrupted, binding=binding, prereg=mini_repo[1],
                  sources=D.require_clean_sources(), environment={})
    with pytest.raises(IntegrityError, match="differ from the approved"):  # changed values: refused earlier
        D.execute({**TINY, "epochs": 2}, run_dir=interrupted, binding=binding, prereg=mini_repo[1],
                  sources=D.require_clean_sources(), environment={})


# ============================================================ end to end [real, tiny]


def test_tiny_end_to_end_run_is_verified_with_separate_sections(completed):
    run_dir, code = completed
    assert code == D.EXIT_OK
    certificate = json.loads((run_dir / V.RESULT_DIR / V.TARGET_CERTIFICATE).read_text())
    assert certificate["status"] == "VERIFIED" and certificate["problems"] == []
    engine_section = certificate["engine_generated_result"]
    assert engine_section["sha256"] == hashlib.sha256((run_dir / engine_section["path"]).read_bytes()).hexdigest()
    target = certificate["target_independent_verification"]
    assert set(target["conditions"]) == {V.condition_name(r, n) for r, n in V.expected_conditions([96, 192], TINY["replicates"])}
    for entry in target["conditions"].values():
        assert len(entry["checkpoint_sha256"]) == 64 and entry["independent_evaluation"]["n"] == TINY["test_samples"]
        assert entry["absolute_differences"]["accuracy"] <= 1e-5 and entry["absolute_differences"]["loss"] <= 1e-5
        # the engine's test_loss is Keras' compiled objective: MSE plus DBitNet's L2 penalties
        assert entry["independent_evaluation"]["regularization_loss"] > 0
    contrast = target["supplementary_statistics"]["contrasts_vs_baseline"]["96_vs_192"]
    assert contrast["classification"] in V.CONTRAST_CLASSES and "t_interval" in contrast


def test_nested_prefixes_are_prefixes_with_correct_hashes(completed):
    from audit.dataset import d5_training_data_scaling as engine

    run_dir, _ = completed
    root = run_dir / V.ENGINE_DIR
    for r in (1, 2):
        x, y, meta = engine.load_replicate_dataset(root, r, sizes=[96, 192])
        assert len(x) == 192
        for n in (96, 192):
            assert engine._array_logical_sha256(x[:n]) == meta["nested_prefix_hashes"][str(n)]["train_x_logical_sha256"]
            assert engine._array_logical_sha256(y[:n]) == meta["nested_prefix_hashes"][str(n)]["train_y_logical_sha256"]
            state = json.loads(engine.condition_paths(root, r, n)["state"].read_text())
            assert state["train_x_prefix_hash"] == meta["nested_prefix_hashes"][str(n)]["train_x_logical_sha256"]
    one, _, _ = engine.load_replicate_dataset(root, 1, sizes=[96, 192])
    two, _, _ = engine.load_replicate_dataset(root, 2, sizes=[96, 192])
    assert not np.array_equal(np.asarray(one), np.asarray(two))          # independent replicate datasets


def test_the_same_seed_is_used_across_sizes_within_a_replicate(completed):
    from audit.dataset import d5_training_data_scaling as engine

    root = completed[0] / V.ENGINE_DIR
    for r in (1, 2):
        seeds = {json.loads(engine.condition_paths(root, r, n)["state"].read_text())["seed"] for n in (96, 192)}
        assert len(seeds) == 1


def test_data_audit_records_exist_for_every_partition(completed):
    audit = completed[0] / "data_audit"
    assert sorted(p.name for p in audit.iterdir()) == ["replicate_01.json", "replicate_02.json", "replicate_03.json", "test.json"]
    assert json.loads((audit / "replicate_01.json").read_text())["train_test_overlap"]["rows"] == 0


def test_altered_checkpoint_bytes_are_detected(completed, tmp_path):
    run_dir, _ = completed
    altered = _copy(run_dir, tmp_path)
    checkpoint = altered / V.ENGINE_DIR / "runs" / "replicate_01" / "n_96" / "latest.keras"
    with open(checkpoint, "ab") as handle:
        handle.write(b"\0")
    report = V.verify_run(altered)
    assert report["status"] == "NOT_VERIFIED"
    assert any("replicate_01/n_96: checkpoint bytes differ" in p for p in report["problems"])


def test_missing_checkpoint_is_detected(completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    (altered / V.ENGINE_DIR / "runs" / "replicate_02" / "n_192" / "latest.keras").unlink()
    report = V.verify_run(altered)
    assert any("replicate_02/n_192: checkpoint latest.keras is missing" in p for p in report["problems"])


def test_a_missing_condition_prevents_a_verified_certificate(completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    shutil.rmtree(altered / V.ENGINE_DIR / "runs" / "replicate_02" / "n_96")
    report = V.verify_run(altered)
    assert report["status"] == "NOT_VERIFIED" and "supplementary_statistics" not in report
    assert V.build_certificate(altered, report)["status"] == "NOT_VERIFIED"


def test_tampered_engine_metrics_are_detected(completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    state_path = altered / V.ENGINE_DIR / "runs" / "replicate_01" / "n_192" / "state.json"
    state = json.loads(state_path.read_text())
    state["test_accuracy"] = state["test_accuracy"] + 0.25
    state_path.write_text(json.dumps(state))
    problems = V.verify_run(altered)["problems"]
    assert any("differ from the metrics logged" in p for p in problems)
    assert any("engine certificate observations differ" in p for p in problems)


def test_verify_stage_reports_a_change_after_certification(completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    assert D.main(["verify", "--run-dir", str(altered), "--no-reevaluate"]) == D.EXIT_OK
    assert V.verify_run(altered, reevaluate=False)["status"] == "INTEGRITY_CONSISTENT_NOT_REEVALUATED"  # never VERIFIED
    (altered / V.ENGINE_DIR / "runs" / "replicate_01" / "n_192" / "latest.keras").write_bytes(b"replaced")
    assert D.main(["verify", "--run-dir", str(altered), "--no-reevaluate"]) == D.EXIT_NOT_VERIFIED


def test_no_protected_file_was_modified(completed):
    assert completed[1] == D.EXIT_OK
    now = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED if p.is_file()}
    assert now == PROTECTED_DIGESTS


# ============================================================ independent re-evaluation [mock]


def test_independent_reevaluation_matches_keras_on_a_tiny_mock_model(tmp_path):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(3)
    model = tf.keras.Sequential([tf.keras.Input((64,)), tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="mse", metrics=["acc"])
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, (300, 64), dtype=np.uint8)
    y = rng.integers(0, 2, 300, dtype=np.uint8)
    model.fit(x, y, epochs=1, batch_size=50, verbose=0)
    loss, accuracy = model.evaluate(x, y, batch_size=50, verbose=0)
    path = tmp_path / "m.keras"
    model.save(path)
    again = V.reevaluate_checkpoint(path, x, y, batch_size=64)
    assert abs(again["accuracy"] - accuracy) <= 1e-6 and abs(again["loss_objective"] - loss) <= 1e-6
    assert again["regularization_loss"] == 0.0 and again["loss_mse"] == again["loss_objective"]


def test_supplementary_classification_never_claims_equivalence():
    t = {"lower": -0.001, "upper": 0.02}
    assert V.classify_contrast(holm_p=0.2, mean_difference=0.05, t_interval=t, alpha=0.05, practical_threshold=0.01,
                               practical_rule="point_estimate_at_least_threshold") == "NOT_SIGNIFICANT_INCONCLUSIVE"
    assert V.classify_contrast(holm_p=0.01, mean_difference=0.015, t_interval=t, alpha=0.05,
                               practical_threshold=0.01, practical_rule="t_interval_excludes_threshold_band") \
        == "SIGNIFICANT_BELOW_PRACTICAL_THRESHOLD"
    assert V.classify_contrast(holm_p=0.01, mean_difference=0.015, t_interval=t, alpha=0.05,
                               practical_threshold=0.01, practical_rule="point_estimate_at_least_threshold") \
        == "SIGNIFICANT_AND_PRACTICALLY_MEANINGFUL"


def test_paired_supplement_matches_scipy():
    from scipy import stats

    d = [0.01, 0.012, 0.008, 0.011, 0.009, 0.013]
    out = V.paired_supplement(d)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=stats.sem(d))
    assert out["t_interval"]["lower"] == pytest.approx(lo) and out["t_interval"]["upper"] == pytest.approx(hi)
    assert out["cohens_dz"] == pytest.approx(np.mean(d) / np.std(d, ddof=1))


def test_importing_the_driver_loads_neither_tensorflow_nor_the_engine():
    code = ("import sys; import audit.targets.autond.d5.training_data_scaling\n"
            "bad = [m for m in ('tensorflow', 'keras', 'audit.dataset.d5_training_data_scaling', 'main') "
            "if m in sys.modules]\nprint(bad); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPOSITORY_ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


# ============================================================ correction pass: decision integrity


SIZES3 = [96, 192, 288]


def _findings(sd=0.004, p=0.01, holm=None, accuracy=0.9, replicates=2):
    """A complete, well-formed engine 'findings' block for 2 replicates x 3 sizes (synthetic)."""
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


def _problems(findings):
    return V.statistical_integrity_problems(findings, sizes=SIZES3, replicates=2)


def _edit(path, value, **kwargs):
    findings = _findings(**kwargs)
    target = findings
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return findings


def test_finite_complete_statistics_pass_the_integrity_check():  # [mock]
    assert _problems(_findings()) == []


@pytest.mark.parametrize("findings, message", [
    (_findings(p=float("nan")), "is not a probability"),
    (_findings(sd=0.0), "zero variance"),
    (_findings(sd=float("inf")), "sd_accuracy_difference is not finite"),          # sd > 0 alone accepts +inf
    (_findings(sd=float("nan")), "sd_accuracy_difference is not finite"),
    (_findings(p=1.5), "raw p 1.5 is not a probability"),
    (_findings(p=-0.01), "is not a probability"),
    (_findings(holm={"96_vs_192": 0.02}), "expected Holm"),                         # engine drops non-finite p
    (_findings(holm={"96_vs_192": 0.02, "96_vs_288": float("nan")}), "Holm-adjusted p nan is not a probability"),
    (_findings(holm={"96_vs_192": 1.2, "96_vs_288": 0.02}), "Holm-adjusted p 1.2 is not a probability"),
    (_findings(holm={"96_vs_192": 0.01, "96_vs_288": 0.01}), "differs from Holm"),  # unadjusted values
    (_findings(accuracy=float("nan")), "non-finite observation"),
    (_findings(replicates=1), "pairs 1 replicates, expected 2"),
    (_edit(["paired_contrasts", "96_vs_192", "paired_t_statistic"], float("inf")), "paired_t_statistic is not finite"),
    (_edit(["paired_contrasts", "96_vs_192", "mean_accuracy_difference"], None), "mean_accuracy_difference is not finite"),
    (_edit(["paired_contrasts", "96_vs_192", "bootstrap_ci95"], [0.0, float("nan")]), "bootstrap CI"),
    (_edit(["paired_contrasts", "96_vs_288"], None), "contrast 96_vs_288 is missing"),
    (_edit(["paired_contrasts", "multiple_comparison_correction", "method"], "Bonferroni"), "expected Holm"),
    (_edit(["paired_contrasts", "baseline_training_size"], 192), "expected the smallest size"),
    (_edit(["summary_by_training_size", "192", "accuracy_ci95_bootstrap"], [0.8, float("inf")]), "summary for n=192"),
    (_edit(["observations"], _findings()["observations"][:-1]), "pairing is incomplete or duplicated"),
    (_edit(["observations"], _findings()["observations"] + _findings()["observations"][:1]), "pairing is incomplete"),
])
def test_non_finite_degenerate_or_incomplete_statistics_are_refused(findings, message):  # [mock]
    assert any(message in problem for problem in _problems(findings)), _problems(findings)


def test_non_finite_supplementary_intervals_are_refused():  # [mock]
    good = {"contrasts_vs_baseline": {"96_vs_192": {"mean": 0.01, "cohens_dz": 2.0,
                                                    "t_interval": {"lower": 0.0, "upper": 0.02}}},
            "per_training_size_accuracy": {"96": {"lower": 0.8, "upper": 0.9}}}
    assert V.supplementary_problems(good) == []
    bad = copy.deepcopy(good)
    bad["contrasts_vs_baseline"]["96_vs_192"]["t_interval"]["upper"] = float("inf")
    bad["per_training_size_accuracy"]["96"]["lower"] = float("nan")
    assert len(V.supplementary_problems(bad)) == 2


def test_data_audits_are_bound_to_the_persisted_datasets(completed, tmp_path):
    audit_path = completed[0] / "data_audit" / "replicate_01.json"
    assert len(json.loads(audit_path.read_text())["x_logical_sha256"]) == 64
    altered = _copy(completed[0], tmp_path)
    record = json.loads((altered / "data_audit" / "replicate_01.json").read_text())
    record["x_logical_sha256"] = "0" * 64
    (altered / "data_audit" / "replicate_01.json").write_text(json.dumps(record))
    (altered / "data_audit" / "test.json").unlink()
    problems = V.verify_run(altered, reevaluate=False)["problems"]
    assert any("replicate_01.json does not describe the persisted" in p for p in problems)
    assert any("data audit test.json is missing" in p for p in problems)


def test_verification_requires_scipy(completed, monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    report = V.verify_run(completed[0], reevaluate=False)
    assert report["status"] == "NOT_VERIFIED" and "SciPy" in report["problems"][0]


def test_fixed_epoch_policy_is_the_documented_estimand():
    record = shipped()
    assert record["claim_scope"] == D.CLAIM_SCOPE
    assert "Fixed-EPOCH" in record["training_policy"] and "not separated" in record["training_policy"]
    assert any("compute" in item for item in D.CLAIM_SCOPE["does_not_establish"])
    rationale = D.DEFAULT_PREREGISTRATION.with_suffix(".RATIONALE.md").read_text(encoding="utf-8")
    assert "Fixed-epoch policy, so data and compute are confounded" in rationale
    assert any("compute" in item for item in record["scope_and_limitations"])
    assert record["decisions"]["replicates"]["status"] == "APPROVED"
    assert record["decisions"]["practical_effect_threshold"]["value"] == 0.005


# ============================================================ SciPy on the execution path


def test_scipy_preflight_passes_when_scipy_is_available():  # [gate]
    assert V.require_statistics_dependencies()["scipy"]


def test_execute_refuses_without_scipy_before_claiming_the_directory(binding, tmp_path, monkeypatch):  # [gate]
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    with pytest.raises(IntegrityError, match="SciPy"):
        D.execute(copy.deepcopy(TINY), run_dir=tmp_path / "r", binding=binding, prereg=STUB_PREREG,
                  sources={}, environment={})
    assert not (tmp_path / "r").exists()


def test_a_certificate_computed_with_the_engine_fallback_is_not_verified(completed, tmp_path, monkeypatch):
    """[real] Recompute the engine's aggregates with SciPy blocked (its fallback) and put them in a copy."""
    from audit.dataset import d5_training_data_scaling as engine

    altered = _copy(completed[0], tmp_path)
    path = altered / V.RESULT_DIR / V.ENGINE_CERTIFICATE
    certificate = json.loads(path.read_text())
    observations = [engine.D5Observation(**o) for o in certificate["findings"]["observations"]]
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "scipy", None)
        m.setitem(sys.modules, "scipy.stats", None)
        _, fallback, _ = engine._aggregate(observations, [96, 192], TINY["replicates"], 1000, 0 + 991)
    assert fallback["96_vs_192"]["paired_p_value"] != certificate["findings"]["paired_contrasts"]["96_vs_192"][
        "paired_p_value"]
    certificate["findings"]["paired_contrasts"] = json.loads(json.dumps(fallback))
    path.write_text(json.dumps(certificate))
    problems = V.verify_run(altered, reevaluate=False)["problems"]
    assert any("paired contrasts differ from their recomputation" in p for p in problems)


# ============================================================ approval and source gates on execute() itself [gate]


def _authorised(mini_repo):
    return {"values": copy.deepcopy(TINY), "prereg": copy.deepcopy(mini_repo[1]), "sources": D.require_clean_sources()}


@needs_checkout
@pytest.mark.parametrize("case, message", [
    ("unanchored", "anchor must be a 64-hex"),
    ("missing", "AutoND D5 preregistration is missing"),
    ("wrong_anchor", "differs from the approver's anchor"),
    ("absolute_path", "not repository-relative"),
    ("values", "values differ from the approved preregistration: ['epochs']"),
    ("commit", "commit or protocol id"),
    ("protocol_id", "commit or protocol id"),
    ("sources_mismatch", "source identity supplied by the caller differs"),
    ("sources_missing", "source identity supplied by the caller differs"),
    ("binding", "not the verified pin"),
])
def test_execute_refuses_a_bad_caller_before_claiming_or_training(case, message, binding, mini_repo, tmp_path,
                                                                    engine_never_runs):
    args = _authorised(mini_repo)
    use_binding = binding
    if case == "unanchored":
        args["prereg"]["sha256"] = None
    elif case == "missing":
        args["prereg"] = STUB_PREREG
    elif case == "wrong_anchor":
        args["prereg"]["sha256"] = "f" * 64
    elif case == "absolute_path":
        args["prereg"]["path"] = str(mini_repo[0] / PREREG_REL)
    elif case == "values":
        args["values"]["epochs"] = 2
    elif case == "commit":
        args["prereg"]["commit"] = {**args["prereg"]["commit"], "sha": "0" * 40}
    elif case == "protocol_id":
        args["prereg"]["protocol_id"] = "another-protocol"
    elif case == "sources_mismatch":
        rel = D.SOURCE_PATHS[-1]
        args["sources"]["files"][rel]["sha256"] = "0" * 64
    elif case == "sources_missing":
        args["sources"] = {}
    elif case == "binding":
        use_binding = types.SimpleNamespace(pin_verified_preimport=True, expected_upstream_commit="0" * 40,
                                            source_digests=binding.source_digests)
    with pytest.raises(IntegrityError, match=__import__("re").escape(message)):
        D.execute(args["values"], run_dir=tmp_path / "r", binding=use_binding, prereg=args["prereg"],
                  sources=args["sources"], environment={})
    assert not (tmp_path / "r").exists() and engine_never_runs == []


@needs_checkout
@pytest.mark.parametrize("changes, message", [
    ({"approved": False}, "not approved"),
    ({"approved_by": None}, "not approved"),
    ({"status": "DRAFT_PENDING_HUMAN_APPROVAL", "approved": False, "approved_utc": None}, "not approved"),
])
def test_execute_refuses_an_unapproved_protocol(changes, message, binding, tmp_path, monkeypatch, engine_never_runs):
    repo, prereg = make_repo(tmp_path / "repo", tiny_record(**changes))
    monkeypatch.setattr(V, "REPOSITORY_ROOT", repo)
    with pytest.raises(IntegrityError, match=message):
        D.execute(copy.deepcopy(TINY), run_dir=tmp_path / "r", binding=binding, prereg=prereg,
                  sources=D.require_clean_sources(), environment={})
    assert not (tmp_path / "r").exists() and engine_never_runs == []


@needs_checkout
def test_execute_refuses_a_pending_decision(binding, tmp_path, monkeypatch, engine_never_runs):
    record = tiny_record()
    record["decisions"]["practical_effect_threshold"].update(status=D.PENDING, value=None)
    repo, prereg = make_repo(tmp_path / "repo", record)
    monkeypatch.setattr(V, "REPOSITORY_ROOT", repo)
    with pytest.raises(IntegrityError, match="decisions.practical_effect_threshold"):
        D.execute(copy.deepcopy(TINY), run_dir=tmp_path / "r", binding=binding, prereg=prereg,
                  sources=D.require_clean_sources(), environment={})
    assert not (tmp_path / "r").exists() and engine_never_runs == []


@needs_checkout
@pytest.mark.parametrize("dirty", ["preregistration", "source"])
def test_execute_refuses_uncommitted_changes(dirty, binding, tmp_path, monkeypatch, engine_never_runs):
    repo, prereg = make_repo(tmp_path / "repo", tiny_record())
    monkeypatch.setattr(V, "REPOSITORY_ROOT", repo)
    sources = D.require_clean_sources()
    if dirty == "preregistration":
        (repo / PREREG_REL).write_text((repo / PREREG_REL).read_text() + "\n")
        prereg["sha256"] = hashlib.sha256((repo / PREREG_REL).read_bytes()).hexdigest()
        message = "uncommitted modifications"
    else:
        with open(repo / D.SOURCE_PATHS[-1], "a") as handle:
            handle.write("\n# local edit\n")
        message = "not clean at HEAD"
    with pytest.raises(IntegrityError, match=message):
        D.execute(copy.deepcopy(TINY), run_dir=tmp_path / "r", binding=binding, prereg=prereg, sources=sources,
                  environment={})
    assert not (tmp_path / "r").exists() and engine_never_runs == []


# ============================================================ provenance in verification [real]


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
    elif case == "source_list":
        claim["source_identity"]["files"].pop(D.SOURCE_PATHS[0])
    elif case == "no_source_identity":
        claim["source_identity"] = {"stubbed": True}
    elif case == "upstream":
        claim["upstream"]["commit"] = "0" * 40


@pytest.mark.parametrize("case, message", [
    ("no_preregistration", "no anchored, committed preregistration record"),
    ("anchor", "does not match the recorded anchor"),
    ("commit", "is not available at commit"),
    ("values", "run values differ from the approved"),
    ("dirty_source", "were not clean at HEAD"),
    ("source_digest", "recorded SHA-256 is not its committed content"),
    ("source_list", "not the registered sources"),
    ("no_source_identity", "no source identity"),
    ("upstream", "differ from the approved pin"),
])
def test_verification_rejects_missing_or_mismatched_provenance(case, message, completed, tmp_path):
    altered = _copy(completed[0], tmp_path)
    claim = json.loads((altered / V.RUN_CLAIM).read_text())
    _claim_edit(case, claim)
    (altered / V.RUN_CLAIM).write_text(json.dumps(claim))
    report = V.verify_run(altered, reevaluate=False)
    assert report["status"] == "NOT_VERIFIED"
    assert any(message in p for p in report["problems"]), report["problems"]


def test_verification_rejects_a_run_whose_protocol_is_not_in_this_repository(completed, tmp_path, monkeypatch):
    other, _ = make_repo(tmp_path / "other", tiny_record())
    monkeypatch.setattr(V, "REPOSITORY_ROOT", other)
    report = V.verify_run(completed[0], reevaluate=False)
    assert report["status"] == "NOT_VERIFIED"
    assert any("is not available at commit" in p for p in report["problems"])


def test_an_authorised_run_records_verifiable_provenance(completed):
    claim = json.loads((completed[0] / V.RUN_CLAIM).read_text())
    assert V.provenance_problems(claim) == []
