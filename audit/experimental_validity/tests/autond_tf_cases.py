"""
AutoND EV cases that bind the pinned checkout, seed Keras or train DBitNet. Tiny, CPU, NON-EVIDENTIARY.

Not collected with the shared suite (the name does not match test_*.py): tf.keras.utils.set_random_seed
leaves process-wide Keras seeding state behind, so these run in their own pytest process, launched by
tests/test_autond_ev.py::test_tensorflow_cases_pass_in_an_isolated_process. They can also be run directly:
    python -B -m pytest -q -p no:cacheprovider tests/autond_tf_cases.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from autond import adapter as A  # noqa: E402
from autond import baseline as B  # noqa: E402
from autond import experiments as E  # noqa: E402
from autond import representation as REP  # noqa: E402
from autond import runner as R  # noqa: E402
from framework.certificate import NonConfirmatoryEvidenceError  # noqa: E402
from framework.failures import FailureReason  # noqa: E402
from framework.firewall import FirewallViolation  # noqa: E402
from framework.resumability import ResumeViolation  # noqa: E402
from test_autond_ev import H1, H2, ROOT, filled  # noqa: E402

pytestmark = pytest.mark.skipif(ROOT is None, reason="pinned AutoND checkout not found; SKIPPED IS NOT PASSED")


@pytest.fixture(scope="module")
def binding():
    return A.bind(ROOT)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """One tiny end-to-end smoke run of all four experiments plus the family step, through the entry point."""
    root = tmp_path_factory.mktemp("smoke")
    assert R.main(["run", "--mode", "smoke", "--experiment", "all", "--autond-root", ROOT,
                   "--output-root", str(root)]) == 0
    return {e: json.loads((root / "smoke" / e / "certificate.json").read_text()) for e in B.EXPERIMENT_IDS} | {
        "_root": root, "_family": json.loads((root / "smoke" / "family_certificate.json").read_text())}


def tiny(experiment, binding, out, config=None, units=2):
    config = copy.deepcopy(config or B.load_config())
    config["smoke"][experiment]["units"] = units
    return E.run_experiment(experiment, config=config, mode="smoke", output_dir=out, binding=binding,
                            permutation=E.frozen_permutation(config))


def bits(word: int) -> list[int]:
    return [(word >> (15 - j)) & 1 for j in range(16)]


# ============================================================ pin, layout, model


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


def test_the_cipher_writes_each_word_msb_first_left_word_first(binding):
    cipher = A.pinned_module(binding, "ciphers.speck3264")
    p = cipher.convert_to_binary(np.uint16([0x6574, 0x694c]).reshape(-1, 1))
    k = cipher.convert_to_binary(np.uint16([0x1918, 0x1110, 0x0908, 0x0100]).reshape(-1, 1))
    assert p[0].tolist() == bits(0x6574) + bits(0x694c)
    assert cipher.encrypt(p, k, 22)[0].tolist() == bits(0xa868) + bits(0x42f2)      # SPECK32/64 test vector


def test_generated_columns_are_c0_left_c0_right_c1_left_c1_right(binding, monkeypatch):
    """Record the generator's actual encrypt calls and recompute both ciphertexts with native-integer rounds."""
    cipher = A.pinned_module(binding, "ciphers.speck3264")
    calls, encrypt = [], cipher.encrypt

    def recording(p, k, r):
        calls.append((p.copy(), k.copy(), r))
        return encrypt(p, k, r)
    monkeypatch.setattr(cipher, "encrypt", recording)
    X, Y = binding.generate_partition(512)
    (p0, k0, r0), (p1, k1, r1) = calls
    assert r0 == r1 == 5 and np.array_equal(k0, k1)
    delta = np.array(bits(0x0040) + bits(0x0000), dtype=np.uint8)              # 0x400000 = left word 0x0040
    assert np.array_equal(p1, p0 ^ delta)

    def native(p_bits, k_bits):
        to_int = lambda b: int("".join(map(str, b)), 2)                          # noqa: E731
        x, y = to_int(p_bits[:16]), to_int(p_bits[16:])
        ks = cipher.expand_key([to_int(k_bits[16 * i:16 * (i + 1)]) for i in range(4)], 5)
        for rk in ks:
            x, y = cipher.enc_one_round((x, y), rk)
        return bits(int(x)) + bits(int(y))
    real = np.flatnonzero(Y == 1)
    assert len(real) > 100
    for i in real[:64]:
        assert X[i, :32].tolist() == native(p0[i], k0[i]) and X[i, 32:].tolist() == native(p1[i], k1[i])
    random_rows = np.flatnonzero(Y == 0)
    assert not all(np.array_equal(X[i, :32], cipher.encrypt(p0[i:i + 1], k0[i:i + 1], 5)[0]) for i in random_rows[:32])


def test_real_generator_and_dbitnet_follow_the_upstream_protocol(binding, monkeypatch):
    X, Y = binding.generate_partition(256)
    A.validate_partition(X, Y, 256)
    a, b, c = A.build_model(binding, 7), A.build_model(binding, 7), A.build_model(binding, 8)
    assert A.weights_sha256(a) == A.weights_sha256(b) != A.weights_sha256(c)
    assert a.input_shape == (None, 64, 1) and a.loss == "mse"
    assert type(a.optimizer).__name__ == "Adam" and a.optimizer.get_config()["amsgrad"] is True
    fake = type(A)("dbitnet")
    fake.__file__ = "/elsewhere/dbitnet.py"
    monkeypatch.setitem(sys.modules, "dbitnet", fake)
    with pytest.raises(A.AutoNDBindingError, match="outside"):
        A.pinned_module(binding, "dbitnet")


def test_representation_invariants_on_real_autond_data(binding):
    X, Y = binding.generate_partition(2000)
    permutation = REP.frozen(B.load_config())
    report = REP.validate(X, Y, permutation, B.load_config()["representation"]["generation_seed"])
    assert report["all_passed"] and report["sha256"] == B.load_config()["representation"]["expected_sha256"]
    Xp = permutation.apply(X)
    assert not np.array_equal(Xp, X) and np.array_equal(permutation.invert(Xp), X)
    for w in range(4):          # each row keeps exactly its bits per word
        block = slice(16 * w, 16 * (w + 1))
        assert np.array_equal(np.sort(Xp[:, block], axis=1), np.sort(X[:, block], axis=1))


# ============================================================ end to end and pairing (smoke)


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
            assert a["dataset_sha256"] == b["dataset_sha256"] and a["model_seed"] == b["model_seed"]
            assert a["initial_weight_sha256"] == b["initial_weight_sha256"]
    order = smoke[H1]["replication"]["raw_replicate_results"][0]["arms"]
    assert [a["shuffle"] for a in order] == [True, False] and order[0]["input_sha256"] == order[1]["input_sha256"]
    assert order[0]["permutation_sha256"] is None is order[1]["permutation_sha256"]
    rep = smoke[H2]["replication"]["raw_replicate_results"][0]["arms"]
    assert [a["shuffle"] for a in rep] == [True, True]
    assert rep[0]["input_sha256"]["test"] != rep[1]["input_sha256"]["test"]
    assert rep[1]["permutation_sha256"] == B.load_config()["representation"]["expected_sha256"]


def test_the_representation_arm_sees_exactly_the_permuted_persisted_data(smoke):
    directory = smoke["_root"] / "smoke" / H2
    arm_a, arm_b = smoke[H2]["replication"]["raw_replicate_results"][0]["arms"]
    permutation = REP.frozen(B.load_config())
    with np.load(directory / "datasets" / "pair000.npz") as data:
        for role in ("train", "validation", "test"):
            X, Y = data[f"{role}_X"], data[f"{role}_Y"]
            assert arm_a["input_sha256"][role] == A.Partition(X, Y).sha256 == arm_b["dataset_sha256"][role]
            assert arm_b["input_sha256"][role] == A.Partition(permutation.apply(X), Y).sha256


def test_noise_and_baseline_record_their_data_sharing_semantics(smoke):
    noise = smoke["EV-AUTOND-NOISE"]
    units = noise["replication"]["raw_replicate_results"]
    assert len({json.dumps(u["arms"][0]["dataset_sha256"], sort_keys=True) for u in units}) == 1
    assert len({u["arms"][0]["initial_weight_sha256"] for u in units}) == 1
    assert noise["manifest"]["dataset_role"] == "shared"
    assert all(len(u["arms"][0]["final_weight_sha256"]) == 64 for u in units)   # content hash, not the archive
    base = smoke["EV-AUTOND-BASELINE"]
    units = base["replication"]["raw_replicate_results"]
    assert len({u["arms"][0]["dataset_sha256"]["test"] for u in units}) == 3        # fresh test set per replicate
    assert len({u["arms"][0]["initial_weight_sha256"] for u in units}) == 3
    assert "test-set sampling" in base["narrative"]["what_was_tested"]
    assert not any("dataset" in c or "seed" in c for c in base["narrative"]["controls"])
    assert "dataset (one for every rerun)" in noise["narrative"]["controls"]
    for h, other in ((H1, "input representation (original)"), (H2, "shuffle=True")):
        controls = smoke[h]["narrative"]["controls"]
        assert "dataset (shared by both arms of a pair)" in controls and other in controls
    assert base["statistics_detail"]["test_set_sampling_sd_at_mean"] > 0


def test_paired_firewall_consumes_each_slot_once(smoke):
    fw = smoke[H2]["firewall"]
    assert fw["frozen"] and len(fw["sealed_dataset_hashes"]) == 3 and len(fw["consumed_evaluation_keys"]) == 6
    assert fw["freeze_record"]["frozen_permutation_hash"] == B.load_config()["representation"]["expected_sha256"]


def test_a_pair_whose_initial_weights_differ_is_refused(binding, tmp_path, monkeypatch):
    build, calls = A.build_model, []

    def mismatched(binding_, seed):
        calls.append(seed)
        return build(binding_, seed + len(calls) % 2)
    monkeypatch.setattr(A, "build_model", mismatched)
    with pytest.raises(RuntimeError, match="different realised initial weights"):
        tiny(H1, binding, tmp_path / "r", units=1)


# ============================================================ resumability, failures, gates


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


def test_consumed_but_unrecorded_confirmatory_data_are_never_evaluated_again(binding, tmp_path, monkeypatch):
    out = tmp_path / "r"
    monkeypatch.setattr(A, "accuracy", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        tiny(H1, binding, out, units=1)
    monkeypatch.undo()
    with pytest.raises(FirewallViolation, match="consumed"):
        tiny(H1, binding, out, units=1)


def test_a_failed_unit_stays_failed_and_counts_against_the_minimum(binding, tmp_path, monkeypatch):
    out, train, calls = tmp_path / "r", A.train, []

    def nan_first(*args, **kwargs):
        calls.append(1)
        return A.TrainingOutcome(FailureReason.NAN, {}) if len(calls) == 1 else train(*args, **kwargs)
    monkeypatch.setattr(A, "train", nan_first)
    cert = tiny("EV-AUTOND-BASELINE", binding, out)
    assert cert["sufficiency_summary"]["failed_replicates"] == 1 and cert["sufficient"] is False
    monkeypatch.setattr(A, "train", lambda *a, **k: pytest.fail("a failed unit must not be re-run"))
    assert tiny("EV-AUTOND-BASELINE", binding, out)["sufficiency_summary"]["failed_replicates"] == 1


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
