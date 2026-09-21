import numpy as np
import pytest

from framework.firewall import FirewallViolation, FreezeRecord, TestSetFirewall
from framework.provenance import array_sha256, config_hash, dataset_provenance


def test_array_sha256_deterministic():
    a = np.array([1, 2, 3], dtype=np.uint8)
    b = np.array([1, 2, 3], dtype=np.uint8)
    assert array_sha256(a) == array_sha256(b)


def test_array_sha256_sensitive_to_dtype():
    a = np.array([1, 2, 3], dtype=np.uint8)
    b = np.array([1, 2, 3], dtype=np.int64)
    assert array_sha256(a) != array_sha256(b)


def test_config_hash_order_independent():
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_dataset_provenance_requires_reason_when_not_replayable():
    with pytest.raises(ValueError):
        dataset_provenance(
            dataset_id="x", role="train", array_hashes={}, generation_procedure="p",
            generation_parameters={}, exact_replay_available=False, reason=None,
        )


def test_dataset_provenance_allows_true_replay_without_reason():
    record = dataset_provenance(
        dataset_id="x", role="train", array_hashes={}, generation_procedure="p",
        generation_parameters={}, exact_replay_available=True,
    )
    assert record["exact_replay_available"] is True


def test_firewall_seal_requires_freeze_first():
    fw = TestSetFirewall(experiment_id="EXP1")
    with pytest.raises(FirewallViolation):
        fw.seal_confirmatory_dataset("pair0", "somehash")


def test_firewall_cannot_reseal_same_replicate_with_different_dataset():
    fw = TestSetFirewall(experiment_id="EXP1")
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.seal_confirmatory_dataset("pair0", "hashA")
    with pytest.raises(FirewallViolation):
        fw.seal_confirmatory_dataset("pair0", "hashB")


def test_firewall_allows_sealing_different_replicates_with_different_datasets():
    """Each replicate_id gets its own independent confirmatory dataset."""
    fw = TestSetFirewall(experiment_id="EXP1")
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.seal_confirmatory_dataset("pair0", "hashA")
    fw.seal_confirmatory_dataset("pair1", "hashB")  # must NOT raise
    assert fw.sealed_dataset_hashes == {"pair0": "hashA", "pair1": "hashB"}


def test_firewall_confirmatory_evaluation_single_use():
    fw = TestSetFirewall(experiment_id="EXP1")
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.seal_confirmatory_dataset("r1", "hashA")
    fw.consume_confirmatory_evaluation(condition_id="a", replicate_id="r1", observed_dataset_hash="hashA")
    with pytest.raises(FirewallViolation):
        fw.consume_confirmatory_evaluation(condition_id="a", replicate_id="r1", observed_dataset_hash="hashA")


def test_firewall_rejects_hash_mismatch():
    fw = TestSetFirewall(experiment_id="EXP1")
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.seal_confirmatory_dataset("r1", "hashA")
    with pytest.raises(FirewallViolation):
        fw.consume_confirmatory_evaluation(condition_id="a", replicate_id="r1", observed_dataset_hash="WRONG")


def test_firewall_cannot_refreeze():
    fw = TestSetFirewall(experiment_id="EXP1")
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    with pytest.raises(FirewallViolation):
        fw.freeze(FreezeRecord("cfg2", None, "rep", "stat", "2026-01-01T00:00:01Z"))


# --- SHARED confirmatory-data provenance mode (II-4) ---

def test_default_mode_is_per_replicate_unchanged_behaviour():
    from framework.firewall import ConfirmatoryDataMode
    fw = TestSetFirewall(experiment_id="EXP")
    assert fw.confirmatory_data_mode is ConfirmatoryDataMode.PER_REPLICATE


def test_per_replicate_mode_still_accepts_one_hash_under_many_replicates():
    """Confirms the firewall was ALREADY capable; SHARED is semantics, not capability."""
    fw = TestSetFirewall(experiment_id="EXP")
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    for k in range(10):
        fw.seal_confirmatory_dataset(f"block{k}", "h" * 64)
    assert len(set(fw.sealed_dataset_hashes.values())) == 1


def test_shared_mode_rejects_a_divergent_hash():
    from framework.firewall import ConfirmatoryDataMode
    fw = TestSetFirewall(experiment_id="EXP",
                         confirmatory_data_mode=ConfirmatoryDataMode.SHARED)
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.seal_confirmatory_dataset("block0", "a" * 64)
    with pytest.raises(FirewallViolation, match="SHARED"):
        fw.seal_confirmatory_dataset("block1", "b" * 64)


def test_shared_mode_still_enforces_exactly_once_consumption():
    from framework.firewall import ConfirmatoryDataMode
    fw = TestSetFirewall(experiment_id="EXP",
                         confirmatory_data_mode=ConfirmatoryDataMode.SHARED)
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.seal_confirmatory_dataset("block0", "a" * 64)
    fw.consume_confirmatory_evaluation(condition_id="depth10", replicate_id="block0",
                                       observed_dataset_hash="a" * 64)
    with pytest.raises(FirewallViolation):
        fw.consume_confirmatory_evaluation(condition_id="depth10", replicate_id="block0",
                                           observed_dataset_hash="a" * 64)


def test_shared_mode_round_trips_through_persistence(tmp_path):
    from framework.firewall import ConfirmatoryDataMode
    fw = TestSetFirewall(experiment_id="EXP",
                         confirmatory_data_mode=ConfirmatoryDataMode.SHARED)
    fw.freeze(FreezeRecord("cfg", None, "rep", "stat", "2026-01-01T00:00:00Z"))
    fw.save(tmp_path / "fw.json")
    assert TestSetFirewall.load(tmp_path / "fw.json").confirmatory_data_mode \
        is ConfirmatoryDataMode.SHARED


def test_legacy_firewall_file_without_mode_loads_as_per_replicate(tmp_path):
    import json
    from framework.firewall import ConfirmatoryDataMode
    fw = TestSetFirewall(experiment_id="EXP")
    fw.save(tmp_path / "fw.json")
    data = json.loads((tmp_path / "fw.json").read_text())
    data.pop("confirmatory_data_mode")
    (tmp_path / "fw.json").write_text(json.dumps(data))
    assert TestSetFirewall.load(tmp_path / "fw.json").confirmatory_data_mode \
        is ConfirmatoryDataMode.PER_REPLICATE


# --- pairing prose derived, not hard-coded ---

def test_describe_unit_reflects_independent_initialization():
    from framework.experiment import PairingDeclaration
    ii4 = PairingDeclaration(True, False, False, True).describe_unit()
    assert "independent model initialization" in ii4
    assert "same model seed" not in ii4


def test_no_hardcoded_same_model_seed_prose_remains():
    import inspect
    import gohr.experiments as ex
    assert "same dataset, same model seed" not in inspect.getsource(ex)
