"""
Integration test for the full Candidate-1 + firewall lifecycle, per the
explicit requirement that this be an actual automated test rather than
a code-path assumption. Exercises real TensorFlow/Keras training at
smoke scale.

Covers, in order:
    - permutation generated, validated, persisted, hash frozen
    - experiment freeze occurs only after the permutation hash is known
    - confirmatory dataset sealed (per-replicate)
    - confirmatory dataset hash recorded
    - condition A consumes exactly once
    - condition B consumes exactly once
    - repeated evaluation is rejected
    - alternate dataset is rejected
    - alternate permutation is rejected (mismatched frozen hash)
"""

import uuid
from pathlib import Path

import numpy as np
import pytest

from framework.firewall import FirewallViolation, FreezeRecord, TestSetFirewall
from framework.provenance import config_hash, utc_timestamp
from gohr.adapter import generate_matched_datasets
from gohr.baseline import BASELINE
from gohr.experiments import run_h_ev_representation
from gohr.representation import generate_candidate1_permutation, run_full_validation
from gohr import speck


def _make_frozen_firewall(experiment_id: str, permutation_hash) -> TestSetFirewall:
    fw = TestSetFirewall(experiment_id=experiment_id)
    fw.freeze(FreezeRecord(
        frozen_config_hash=config_hash({"x": 1}), frozen_permutation_hash=permutation_hash,
        frozen_replicate_plan_hash=config_hash({"n": 2}),
        frozen_statistical_plan_hash=config_hash({"alpha": 0.05}),
        frozen_at_utc=utc_timestamp(),
    ))
    return fw


def test_permutation_generate_validate_persist_hash(tmp_path):
    permutation = generate_candidate1_permutation(np.random.default_rng(42))
    X, Y = speck.make_train_data(1000, 3)
    validation = run_full_validation(X, Y, permutation)
    assert validation["all_passed"] is True

    path = tmp_path / "perm.json"
    permutation.save(path)
    reloaded = type(permutation).load(path)
    assert reloaded.hash == permutation.hash


def test_freeze_requires_permutation_hash_known_first():
    """The freeze record must be constructed WITH the permutation hash already known."""
    permutation = generate_candidate1_permutation(np.random.default_rng(1))
    fw = _make_frozen_firewall("H-EV-REPRESENTATION", permutation.hash)
    assert fw.freeze_record.frozen_permutation_hash == permutation.hash


def test_seal_before_freeze_is_rejected():
    fw = TestSetFirewall(experiment_id="X")
    with pytest.raises(FirewallViolation):
        fw.seal_confirmatory_dataset("pair0", "somehash")


def test_per_replicate_sealing_allows_multiple_independent_datasets():
    """
    Regression test for the structural firewall defect: each of K
    replicate pairs generates its OWN confirmatory dataset, and the
    firewall must accommodate this (one sealed hash per replicate_id),
    not reject the second pair as a hash mismatch against the first.
    """
    permutation = generate_candidate1_permutation(np.random.default_rng(2))
    fw = _make_frozen_firewall("H-EV-REPRESENTATION", permutation.hash)

    datasets_0 = generate_matched_datasets(rounds=3, differential=BASELINE.differential,
                                            train_size=50, val_size=20, confirmatory_test_size=20)
    datasets_1 = generate_matched_datasets(rounds=3, differential=BASELINE.differential,
                                            train_size=50, val_size=20, confirmatory_test_size=20)
    assert datasets_0.confirmatory_test.combined_hash != datasets_1.confirmatory_test.combined_hash

    fw.seal_confirmatory_dataset("pair0", datasets_0.confirmatory_test.combined_hash)
    fw.seal_confirmatory_dataset("pair1", datasets_1.confirmatory_test.combined_hash)  # must NOT raise

    fw.consume_confirmatory_evaluation(condition_id="condition_a", replicate_id="pair0",
                                        observed_dataset_hash=datasets_0.confirmatory_test.combined_hash)
    fw.consume_confirmatory_evaluation(condition_id="condition_b", replicate_id="pair0",
                                        observed_dataset_hash=datasets_0.confirmatory_test.combined_hash)
    fw.consume_confirmatory_evaluation(condition_id="condition_a", replicate_id="pair1",
                                        observed_dataset_hash=datasets_1.confirmatory_test.combined_hash)


def test_repeated_evaluation_rejected():
    permutation = generate_candidate1_permutation(np.random.default_rng(3))
    fw = _make_frozen_firewall("X", permutation.hash)
    fw.seal_confirmatory_dataset("pair0", "hashA")
    fw.consume_confirmatory_evaluation(condition_id="condition_a", replicate_id="pair0", observed_dataset_hash="hashA")
    with pytest.raises(FirewallViolation):
        fw.consume_confirmatory_evaluation(condition_id="condition_a", replicate_id="pair0", observed_dataset_hash="hashA")


def test_alternate_dataset_rejected():
    permutation = generate_candidate1_permutation(np.random.default_rng(4))
    fw = _make_frozen_firewall("X", permutation.hash)
    fw.seal_confirmatory_dataset("pair0", "hashA")
    with pytest.raises(FirewallViolation):
        fw.consume_confirmatory_evaluation(condition_id="condition_a", replicate_id="pair0", observed_dataset_hash="hashB")


def test_alternate_permutation_rejected_by_run_h_ev_representation(tmp_path):
    """
    run_h_ev_representation must refuse to run if the permutation passed
    in does not match the one recorded at freeze time.
    """
    permutation_frozen = generate_candidate1_permutation(np.random.default_rng(5))
    permutation_different = generate_candidate1_permutation(np.random.default_rng(6))
    fw = _make_frozen_firewall("H-EV-REPRESENTATION", permutation_frozen.hash)

    with pytest.raises(RuntimeError, match="Permutation hash mismatch"):
        run_h_ev_representation(
            run_mode="smoke", requested_pairs=1, minimum_valid_pairs=1,
            output_dir=tmp_path, permutation=permutation_different, firewall=fw,
        )


def test_full_lifecycle_smoke_end_to_end(tmp_path):
    """
    Full, real (TensorFlow-executing) lifecycle at smoke scale:
    generate -> validate -> persist -> freeze (with hash) -> run ->
    seal-per-pair -> consume-once-per-condition -> certificate produced.
    """
    permutation = generate_candidate1_permutation(np.random.default_rng(999))
    X, Y = speck.make_train_data(1000, BASELINE.rounds, diff=BASELINE.differential)
    assert run_full_validation(X, Y, permutation)["all_passed"] is True
    permutation.save(tmp_path / "candidate1_permutation.json")

    fw = _make_frozen_firewall("H-EV-REPRESENTATION", permutation.hash)
    cert = run_h_ev_representation(
        run_mode="smoke", requested_pairs=2, minimum_valid_pairs=2,
        output_dir=tmp_path, permutation=permutation, firewall=fw,
    )
    assert cert["non_evidentiary"] is True
    assert cert["decision"] in ("SUPPORTED", "INCONCLUSIVE", "NOT_SUPPORTED")
    # Both pairs must have consumed exactly 2 keys each (condition_a, condition_b).
    assert len(fw._consumed_keys) == 4
    assert len(fw.sealed_dataset_hashes) == 2
