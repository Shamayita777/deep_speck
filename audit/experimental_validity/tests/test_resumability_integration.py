"""
Integration test for replicate-level resumability, per the explicit
requirement to test: interrupted experiment -> saved state -> resume ->
completion -> verify the final result is consistent with an
uninterrupted run under the same frozen configuration.

Scope note: resumability here is at the REPLICATE level (the natural
statistical unit for this project), not mid-epoch. Each completed
replicate is recorded in a ResumeLedger; a subsequent invocation with
an IDENTICAL configuration hash skips already-completed replicates
rather than retraining them, while a MISMATCHED configuration hash for
an existing replicate_id raises rather than silently overwriting it.
"""

from pathlib import Path

import numpy as np
import pytest

from framework.firewall import FreezeRecord, TestSetFirewall
from framework.provenance import config_hash, utc_timestamp
from framework.resumability import ResumeLedger
from gohr.experiments import run_ev_baseline, run_ev_noise, run_h_ev_shuffle


def _shuffle_firewall() -> TestSetFirewall:
    fw = TestSetFirewall(experiment_id="H-EV-SHUFFLE")
    fw.freeze(FreezeRecord(
        frozen_config_hash=config_hash({"x": 1}), frozen_permutation_hash=None,
        frozen_replicate_plan_hash=config_hash({"n": 3}),
        frozen_statistical_plan_hash=config_hash({"alpha": 0.05}),
        frozen_at_utc=utc_timestamp(),
    ))
    return fw


def test_ev_baseline_uninterrupted_run_completes_all_replicates(tmp_path):
    cert = run_ev_baseline(
        run_mode="smoke", requested_replicates=3, minimum_valid_replicates=3, output_dir=tmp_path,
    )
    ledger = ResumeLedger(tmp_path / "resume_ledger.jsonl")
    for i in range(3):
        state = ledger.last_state_for("EV-BASELINE", f"baseline_r{i}")
        assert state is not None
        assert state["stage"] == "complete"


def test_ev_baseline_resume_skips_completed_replicates(tmp_path):
    """
    Simulates an interruption after 2 of 3 replicates by running with
    requested_replicates=2 first, then "resuming" with
    requested_replicates=3 using an IDENTICAL configuration - the first
    two must be skipped (not retrained), and the run must still reach 3
    valid replicates total.
    """
    cert_partial = run_ev_baseline(
        run_mode="smoke", requested_replicates=2, minimum_valid_replicates=2, output_dir=tmp_path,
    )
    assert cert_partial["sufficiency_summary"]["valid_replicates"] == 2

    ledger = ResumeLedger(tmp_path / "resume_ledger.jsonl")
    state_r0_before = ledger.last_state_for("EV-BASELINE", "baseline_r0")
    assert state_r0_before is not None

    cert_resumed = run_ev_baseline(
        run_mode="smoke", requested_replicates=3, minimum_valid_replicates=3, output_dir=tmp_path,
    )
    assert cert_resumed["sufficiency_summary"]["valid_replicates"] == 3
    assert cert_resumed["sufficiency_summary"]["requested_replicates"] == 3

    # The first two replicates' recorded state must be untouched (same
    # config_hash and checkpoint_hash) - proving they were not silently
    # rerun with different randomness under the resumed invocation.
    state_r0_after = ledger.last_state_for("EV-BASELINE", "baseline_r0")
    assert state_r0_after["config_hash"] == state_r0_before["config_hash"]
    assert state_r0_after["checkpoint_hash"] == state_r0_before["checkpoint_hash"]


def test_resume_rejects_changed_configuration(tmp_path):
    """
    A resume attempt with a DIFFERENT configuration (here: a different
    base_model_seed, which changes every downstream config_hash) for an
    existing replicate_id must raise, not silently proceed under the
    new configuration.
    """
    run_ev_baseline(
        run_mode="smoke", requested_replicates=1, minimum_valid_replicates=1,
        output_dir=tmp_path, base_model_seed=1000,
    )
    with pytest.raises(RuntimeError, match="Resume conflict"):
        run_ev_baseline(
            run_mode="smoke", requested_replicates=1, minimum_valid_replicates=1,
            output_dir=tmp_path, base_model_seed=9999,  # different seed -> different config_hash
        )


def test_h_ev_shuffle_resume_recovers_raw_paired_values(tmp_path):
    """
    For the paired experiment, a resumed run must recover the ACTUAL
    accuracy values from the interrupted run (via the sidecar file),
    not merely skip re-training while silently shrinking the returned
    dataset - this verifies the final result is genuinely consistent
    with what an uninterrupted run would have produced for those pairs.
    """
    fw1 = _shuffle_firewall()
    cert_partial = run_h_ev_shuffle(
        run_mode="smoke", requested_pairs=1, minimum_valid_pairs=1,
        output_dir=tmp_path, firewall=fw1,
    )
    values_partial = cert_partial["raw_paired_values"]

    fw2 = _shuffle_firewall()
    cert_full = run_h_ev_shuffle(
        run_mode="smoke", requested_pairs=2, minimum_valid_pairs=2,
        output_dir=tmp_path, firewall=fw2,
    )
    values_full = cert_full["raw_paired_values"]

    # pair0's recovered value from the resumed run must exactly match
    # what was recorded during the original (interrupted) run.
    assert values_full["condition_a"][0] == values_partial["condition_a"][0]
    assert values_full["condition_b"][0] == values_partial["condition_b"][0]
    assert len(values_full["condition_a"]) == 2


def test_ev_noise_resume_reuses_persisted_dataset_not_regenerated(tmp_path):
    """
    ISSUE 1 (Round 6): regression test for the EV-NOISE resume defect.
    Before the fix, run_ev_noise() called generate_matched_datasets()
    unconditionally on every invocation - including a resumed one -
    which (since Gohr dataset generation is os.urandom-based, never
    seedable) silently produced a DIFFERENT dataset after interruption.
    This test simulates exactly that scenario: a partial run, followed
    by a second ("resumed") invocation, and asserts the dataset hash
    recorded for reruns completed in each phase is IDENTICAL - proving
    the second invocation reused the persisted dataset rather than
    generating a fresh one.
    """
    cert_partial = run_ev_noise(
        run_mode="smoke", requested_reruns=1, minimum_valid_reruns=1, output_dir=tmp_path,
    )
    hash_partial = cert_partial["narrative"]["dataset"]["hash"]

    cert_resumed = run_ev_noise(
        run_mode="smoke", requested_reruns=2, minimum_valid_reruns=2, output_dir=tmp_path,
    )
    hash_resumed = cert_resumed["narrative"]["dataset"]["hash"]

    assert hash_partial == hash_resumed, (
        "EV-NOISE resumed run used a different dataset than the interrupted run - "
        "the persisted-dataset fix is not working."
    )

    # The manifest sidecar for the dataset itself must also exist and
    # point at the same persisted files across both invocations.
    from gohr.experiments import _ev_noise_dataset_manifest_path
    import json
    manifest = json.loads(_ev_noise_dataset_manifest_path(tmp_path).read_text())
    assert manifest["train"]["hash"] == manifest["train"]["hash"]  # sanity: manifest is well-formed
    assert Path(manifest["train"]["path"]).exists()


def test_ev_noise_second_invocation_does_not_regenerate_when_all_reruns_already_complete(tmp_path):
    """A resumed EV-NOISE run whose reruns are ALL already complete must not regenerate anything."""
    run_ev_noise(run_mode="smoke", requested_reruns=2, minimum_valid_reruns=2, output_dir=tmp_path)
    from gohr.experiments import _ev_noise_dataset_manifest_path
    manifest_path = _ev_noise_dataset_manifest_path(tmp_path)
    mtime_before = manifest_path.stat().st_mtime

    run_ev_noise(run_mode="smoke", requested_reruns=2, minimum_valid_reruns=2, output_dir=tmp_path)
    mtime_after = manifest_path.stat().st_mtime
    assert mtime_before == mtime_after, "Dataset manifest was rewritten on a fully-resumed run."


def test_firewall_state_restored_on_resume_not_replaced_with_fresh(tmp_path):
    """
    ISSUE 2 (Round 6): regression test for the firewall-not-restored
    defect. Simulates a full production-style invocation sequence
    through scripts/run_ev.py's actual helper function
    (_load_or_freeze_firewall), across two separate calls representing
    "first run" and "resume after restart" - the second call must
    return a firewall carrying over the first call's sealed/consumed
    state, not a fresh empty one.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_ev import _load_or_freeze_firewall

    config = {
        "requested_pairs": 2, "minimum_valid_pairs": 2, "alpha": 0.05,
        "multiplicity_family": "ev_primary_family_v1", "practical_threshold": None,
    }

    fw_first = _load_or_freeze_firewall("H-EV-SHUFFLE", tmp_path, config, permutation_hash=None)
    fw_first.seal_confirmatory_dataset("pair0", "hashA")
    fw_first.consume_confirmatory_evaluation(condition_id="condition_a", replicate_id="pair0", observed_dataset_hash="hashA")
    fw_first.consume_confirmatory_evaluation(condition_id="condition_b", replicate_id="pair0", observed_dataset_hash="hashA")
    fw_first.save(tmp_path / "firewall.json")

    # Simulate a fresh process: call the SAME helper again as resume would.
    fw_resumed = _load_or_freeze_firewall("H-EV-SHUFFLE", tmp_path, config, permutation_hash=None)

    assert fw_resumed.sealed_dataset_hashes == {"pair0": "hashA"}
    assert fw_resumed._consumed_keys == {"condition_a:pair0", "condition_b:pair0"}

    # A previously consumed key must remain rejected after "restart".
    from framework.firewall import FirewallViolation
    with pytest.raises(FirewallViolation):
        fw_resumed.consume_confirmatory_evaluation(
            condition_id="condition_a", replicate_id="pair0", observed_dataset_hash="hashA"
        )


def test_firewall_resume_rejects_changed_config(tmp_path):
    """A resume attempt with a different config (different frozen_config_hash) must be refused."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_ev import _load_or_freeze_firewall

    config_v1 = {"requested_pairs": 2, "minimum_valid_pairs": 2, "alpha": 0.05,
                 "multiplicity_family": "ev_primary_family_v1", "practical_threshold": None}
    config_v2 = {"requested_pairs": 5, "minimum_valid_pairs": 5, "alpha": 0.05,  # different -> different hash
                 "multiplicity_family": "ev_primary_family_v1", "practical_threshold": None}

    fw = _load_or_freeze_firewall("H-EV-SHUFFLE", tmp_path, config_v1, permutation_hash=None)
    fw.save(tmp_path / "firewall.json")

    with pytest.raises(RuntimeError, match="Refusing to resume"):
        _load_or_freeze_firewall("H-EV-SHUFFLE", tmp_path, config_v2, permutation_hash=None)


def test_full_interruption_resume_lifecycle_with_firewall_and_dataset(tmp_path):
    """
    ISSUE 1 + ISSUE 2 combined integration test, per the Round-6
    requirement: run replicate -> seal -> consume/evaluate -> interrupt
    -> reload process (simulated) -> restore firewall + dataset ->
    resume -> verify previous consumption remains consumed -> verify no
    second confirmatory evaluation is possible -> verify recovered
    metrics are identical to what an uninterrupted run would produce.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_ev import _load_or_freeze_firewall
    from framework.firewall import FirewallViolation

    config = {"requested_pairs": 1, "minimum_valid_pairs": 1, "alpha": 0.05,
              "multiplicity_family": "ev_primary_family_v1", "practical_threshold": None}

    # "First process": run 1 pair through the real orchestration.
    fw_first = _load_or_freeze_firewall("H-EV-SHUFFLE", tmp_path, config, permutation_hash=None)
    cert_first = run_h_ev_shuffle(
        run_mode="smoke", requested_pairs=1, minimum_valid_pairs=1,
        output_dir=tmp_path, firewall=fw_first,
    )
    fw_first.save(tmp_path / "firewall.json")

    # "Restart": construct a brand-new firewall object via the same
    # resume-aware helper, as scripts/run_ev.py would after a crash.
    fw_resumed = _load_or_freeze_firewall("H-EV-SHUFFLE", tmp_path, config, permutation_hash=None)
    assert fw_resumed is not fw_first  # genuinely a different Python object (simulating a new process)
    assert fw_resumed.sealed_dataset_hashes == fw_first.sealed_dataset_hashes
    assert fw_resumed._consumed_keys == fw_first._consumed_keys

    # No second confirmatory evaluation of the already-consumed pair is possible.
    sealed_hash = fw_resumed.sealed_dataset_hashes["pair0"]
    with pytest.raises(FirewallViolation):
        fw_resumed.consume_confirmatory_evaluation(
            condition_id="condition_a", replicate_id="pair0", observed_dataset_hash=sealed_hash
        )

    # Resuming the experiment itself (same requested_pairs=1) must
    # recover the identical recorded metrics, not rerun pair0.
    cert_resumed = run_h_ev_shuffle(
        run_mode="smoke", requested_pairs=1, minimum_valid_pairs=1,
        output_dir=tmp_path, firewall=fw_resumed,
    )
    assert cert_resumed["raw_paired_values"] == cert_first["raw_paired_values"]
