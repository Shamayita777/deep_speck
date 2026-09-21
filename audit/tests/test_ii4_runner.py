"""
II-4 runner tests with a FAKE, TensorFlow-free adapter.

These exercise the full frozen execution sequence, resumability, the
shared sealed test set and block invalidation. Because the runner is
driven entirely through the ConformanceAdapter contract, a fake adapter
suffices - which is itself evidence that the runner carries no Gohr
knowledge and can be reused for another audited system.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("framework.firewall")

from audit.implementation.conformance import ConformanceAdapter
from audit.implementation.ii4_experiment import build_gohr_ii4_smoke_plan
from audit.implementation.ii4_runner import II4ExecutionError, II4Runner


class SimulatedCrash(BaseException):
    """Escapes the runner's per-arm `except Exception`, like a real kill."""


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class _TrainResult:
    n_epochs_completed: int
    final_val_acc: float = 0.9
    max_val_acc: float = 0.91
    checkpoint_hash: str = "c" * 64
    failure_reason: object = None


class FakeAdapter:
    def __init__(self, epochs, *, fail_arm=None, crash_at=None, misbuild=False,
                 short_epochs=False, base=(0.93, 0.61), noise=1e-4, sources=None):
        self.epochs = epochs
        self.fail_arm, self.crash_at = fail_arm, crash_at
        self.misbuild, self.short_epochs = misbuild, short_epochs
        self.base, self.noise = base, noise
        self._sources = sources or {"fake_adapter": "audit/tests/test_ii4_runner.py"}
        self.calls = {"test_set": 0, "blocks": 0, "train": 0, "evaluate": 0}

    # provenance
    def provenance_sources(self):
        return dict(self._sources)

    def environment_details(self):
        return {"gpus": [], "note": "fake adapter - no accelerator"}

    # data
    def generate_sealed_test_set(self, directory):
        self.calls["test_set"] += 1
        Path(directory).mkdir(parents=True, exist_ok=True)
        p = Path(directory) / "sealed.bin"
        p.write_bytes(b"sealed-test-set")
        return p, _sha(p), 1000

    def load_sealed_test_set(self, path, expected_hash):
        if _sha(path) != expected_hash:
            raise RuntimeError("sealed test set hash mismatch")
        return {"path": path}

    def generate_block_datasets(self, block_id, directory):
        self.calls["blocks"] += 1
        Path(directory).mkdir(parents=True, exist_ok=True)
        out = {}
        for role in ("train", "validation"):
            p = Path(directory) / f"{role}.bin"
            p.write_bytes(f"{block_id}-{role}".encode())
            out[role] = {"path": str(p), "hash": _sha(p), "n": 10, "dataset_id": f"{block_id}-{role}"}
        return out

    def load_block_datasets(self, records):
        for rec in records.values():
            if _sha(rec["path"]) != rec["hash"]:
                raise RuntimeError("block data hash mismatch")
        return records

    # model
    def build_model(self, factor_value, *, seed):
        depth = factor_value + 1 if self.misbuild else factor_value
        return {"depth": depth, "seed": seed}

    def probe_model(self, model):
        return model["depth"]

    def probe_realized_value(self, artifact_path):
        return json.loads(Path(artifact_path).read_text())["depth"]

    def train(self, model, block_data, *, checkpoint_path):
        self.calls["train"] += 1
        key = (Path(checkpoint_path).parents[1].name, Path(checkpoint_path).parent.name)
        if self.crash_at == key:
            raise SimulatedCrash()
        if self.fail_arm == key:
            raise RuntimeError("simulated NaN divergence")
        return _TrainResult(n_epochs_completed=self.epochs - (1 if self.short_epochs else 0))

    def save_terminal_model(self, model, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(model))
        return {"path": str(path), "hash": _sha(path), "reload_weight_identical": True}

    def load_terminal_model(self, path, expected_hash):
        if _sha(path) != expected_hash:
            raise RuntimeError("terminal model hash mismatch")
        return json.loads(Path(path).read_text())

    def evaluate_terminal(self, model, sealed_test):
        self.calls["evaluate"] += 1
        base = self.base[0] if model["depth"] == 10 else self.base[1]
        return base + (model["seed"] % 7) * self.noise


def _plan(tmp_path, n_blocks=10, epochs=2):
    return build_gohr_ii4_smoke_plan(output_dir=tmp_path / "run", n_blocks=n_blocks, epochs=epochs)


def _run(tmp_path, adapter, **kw):
    return II4Runner(plan=_plan(tmp_path, **kw), adapter=adapter).run(execute=True)


# --- genericity ---------------------------------------------------------

def test_fake_adapter_satisfies_generic_contract():
    assert isinstance(FakeAdapter(2), ConformanceAdapter)


# --- dry run -------------------------------------------------------------

def test_dry_run_touches_nothing(tmp_path):
    out = II4Runner(plan=_plan(tmp_path), adapter=FakeAdapter(2)).run(execute=False)
    assert out["executed"] is False
    assert not (tmp_path / "run").exists()


def test_production_plan_refuses_without_explicit_allow(tmp_path):
    from audit.implementation.ii4_experiment import build_gohr_ii4_plan
    plan = build_gohr_ii4_plan(output_dir=tmp_path / "prod")
    with pytest.raises(II4ExecutionError, match="allow_production"):
        II4Runner(plan=plan, adapter=FakeAdapter(200)).run(execute=True)
    assert not (tmp_path / "prod").exists()


# --- global shared sealed test set ----------------------------------------

def test_one_global_test_set_reused_across_all_10_blocks(tmp_path):
    adapter = FakeAdapter(2)
    report = _run(tmp_path, adapter)
    assert adapter.calls["test_set"] == 1
    sealed = set(report["firewall"]["sealed_dataset_hashes"].values())
    assert len(report["firewall"]["sealed_dataset_hashes"]) == 10
    assert sealed == {report["sealed_test_set"]["hash"]}
    assert report["firewall"]["confirmatory_data_mode"] == "SHARED"


def test_exactly_once_consumption_per_arm_block(tmp_path):
    report = _run(tmp_path, FakeAdapter(2))
    keys = report["firewall"]["consumed_evaluation_keys"]
    assert len(keys) == 20 and len(set(keys)) == 20
    assert {k.split(":")[0] for k in keys} == {"declared", "realized"}


def test_both_arms_of_every_block_evaluated_on_same_hash(tmp_path):
    report = _run(tmp_path, FakeAdapter(2))
    state = json.loads((tmp_path / "run" / "ii4_state.json").read_text())
    hashes = {state["blocks"][b]["arms"][a]["sealed_test_hash"]
              for b in state["blocks"] for a in ("declared", "realized")}
    assert hashes == {report["sealed_test_set"]["hash"]}


def test_per_block_data_is_independently_generated(tmp_path):
    adapter = FakeAdapter(2)
    _run(tmp_path, adapter)
    state = json.loads((tmp_path / "run" / "ii4_state.json").read_text())
    train_hashes = {state["blocks"][b]["dataset"]["train"]["hash"] for b in state["blocks"]}
    assert adapter.calls["blocks"] == 10 and len(train_hashes) == 10


# --- seeds -----------------------------------------------------------------

def test_20_distinct_seeds_recorded_and_used(tmp_path):
    _run(tmp_path, FakeAdapter(2))
    state = json.loads((tmp_path / "run" / "ii4_state.json").read_text())
    seeds = [state["blocks"][b]["arms"][a]["seed"] for b in state["blocks"]
             for a in ("declared", "realized")]
    assert len(seeds) == 20 and len(set(seeds)) == 20
    assert min(seeds) == 700000 and max(seeds) == 700019
    prereg = json.loads((tmp_path / "run" / "ii4_preregistration.json").read_text())
    assert len(prereg["seed_manifest"]["seeds"]) == 10


# --- architecture probes ------------------------------------------------------

def test_probes_recorded_for_both_arms(tmp_path):
    _run(tmp_path, FakeAdapter(2))
    state = json.loads((tmp_path / "run" / "ii4_state.json").read_text())
    for b in state["blocks"].values():
        assert b["arms"]["declared"]["probe_in_memory"] == 10
        assert b["arms"]["declared"]["probe_artifact"] == 10
        assert b["arms"]["realized"]["probe_in_memory"] == 5
        assert b["arms"]["realized"]["probe_artifact"] == 5


def test_probe_mismatch_aborts_whole_experiment(tmp_path):
    with pytest.raises(II4ExecutionError, match="realizes"):
        _run(tmp_path, FakeAdapter(2, misbuild=True))


# --- terminal epoch -------------------------------------------------------------

def test_terminal_epoch_endpoint_recorded(tmp_path):
    _run(tmp_path, FakeAdapter(2))
    state = json.loads((tmp_path / "run" / "ii4_state.json").read_text())
    for b in state["blocks"].values():
        for a in b["arms"].values():
            assert a["evaluation_rule"] == "terminal_epoch"
            assert a["terminal_epoch"] == 2
            assert "max_val_acc_historical_descriptive_only" in a["secondary"]


def test_incomplete_epochs_fail_the_arm(tmp_path):
    report = _run(tmp_path, FakeAdapter(2, short_epochs=True))
    assert all(not b["is_valid"] for b in report["blocks"])
    assert report["analysis"]["replication"]["sufficient"] is False


# --- failure handling -----------------------------------------------------------

def test_failed_arm_invalidates_whole_block(tmp_path):
    report = _run(tmp_path, FakeAdapter(2, fail_arm=("block3", "realized")))
    b3 = next(b for b in report["blocks"] if b["block_id"] == "block3")
    assert b3["is_valid"] is False and b3["signed_difference"] is None
    assert b3["declared_arm_value"] is None       # surviving arm NOT retained
    assert report["analysis"]["replication"]["n_blocks_valid"] == 9
    assert report["analysis"]["replication"]["sufficient"] is True


def test_two_failed_blocks_is_inconclusive(tmp_path):
    adapter = FakeAdapter(2, fail_arm=("block1", "declared"))
    plan = _plan(tmp_path)
    runner = II4Runner(plan=plan, adapter=adapter)
    runner.run(execute=True)
    # second failure via state edit is artificial; instead verify the rule directly:
    from audit.implementation.ii4_analysis import analyse_ii4
    r = analyse_ii4([0.3] * 8, n_blocks_requested=10, min_valid_blocks=9, delta=0.01)
    assert r.sufficient is False


def test_min_valid_is_nine(tmp_path):
    from audit.implementation.ii4_design import II4_DESIGN
    assert II4_DESIGN.min_valid_blocks == 9


def test_failed_block_is_not_evaluated_on_sealed_set(tmp_path):
    adapter = FakeAdapter(2, fail_arm=("block0", "realized"))
    _run(tmp_path, adapter)
    assert adapter.calls["evaluate"] == 18   # 9 valid blocks x 2 arms; block0 never evaluated


# --- resumability ------------------------------------------------------------------

def test_resume_does_not_regenerate_test_set_blocks_or_completed_arms(tmp_path):
    plan = _plan(tmp_path)
    first = FakeAdapter(2, crash_at=("block4", "declared"))
    with pytest.raises(SimulatedCrash):
        II4Runner(plan=plan, adapter=first).run(execute=True)
    assert first.calls["test_set"] == 1 and first.calls["blocks"] == 5

    second = FakeAdapter(2)
    report = II4Runner(plan=_plan(tmp_path), adapter=second).run(execute=True)
    assert second.calls["test_set"] == 0          # sealed set reused
    assert second.calls["blocks"] == 5            # only blocks 5..9 generated
    assert second.calls["train"] == 12            # block4 x2 + blocks5-9 x2
    assert report["analysis"]["replication"]["n_blocks_valid"] == 10
    assert len(report["firewall"]["consumed_evaluation_keys"]) == 20


def test_resume_cannot_substitute_sealed_test_set(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(SimulatedCrash):
        II4Runner(plan=plan, adapter=FakeAdapter(2, crash_at=("block2", "declared"))).run(execute=True)
    sealed = tmp_path / "run" / "sealed_test" / "sealed.bin"
    sealed.write_bytes(b"a different test set")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        II4Runner(plan=_plan(tmp_path), adapter=FakeAdapter(2)).run(execute=True)


def test_resume_rejects_changed_configuration(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(SimulatedCrash):
        II4Runner(plan=plan, adapter=FakeAdapter(2, crash_at=("block1", "declared"))).run(execute=True)
    changed = build_gohr_ii4_smoke_plan(output_dir=tmp_path / "run", n_blocks=10, epochs=3)
    with pytest.raises(II4ExecutionError, match="fingerprint"):
        II4Runner(plan=changed, adapter=FakeAdapter(3)).run(execute=True)


def test_resume_detects_tampered_completed_arm(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(SimulatedCrash):
        II4Runner(plan=plan, adapter=FakeAdapter(2, crash_at=("block2", "declared"))).run(execute=True)
    tm = tmp_path / "run" / "blocks" / "block0" / "declared" / "terminal_model.keras"
    tm.write_text(json.dumps({"depth": 10, "seed": 1}))
    with pytest.raises(RuntimeError, match="terminal model hash mismatch"):
        II4Runner(plan=_plan(tmp_path), adapter=FakeAdapter(2)).run(execute=True)


# --- preregistration -----------------------------------------------------------------

def test_preregistration_records_global_shared_semantics(tmp_path):
    _run(tmp_path, FakeAdapter(2))
    prereg = json.loads((tmp_path / "run" / "ii4_preregistration.json").read_text())
    assert prereg["confirmatory_test_set"]["scope"] == "GLOBAL_SHARED"
    assert prereg["confirmatory_test_set"]["firewall_mode"] == "SHARED"
    assert prereg["confirmatory_test_set"]["per_block_test_sets"] is False
    assert "CONDITIONAL ON THE FIXED SEALED EVALUATION SET" in prereg["conditional_inference"]
    assert "cancel exactly" in prereg["conditional_inference"]   # stated as NOT holding
    assert prereg["design"]["pairing"]["same_model_initialization"] is False


def test_smoke_report_is_marked_non_evidentiary(tmp_path):
    report = _run(tmp_path, FakeAdapter(2))
    assert report["non_evidentiary"] is True
    assert report["execution_mode"] == "SMOKE"
    assert "SMOKE" in report["experiment_id"]
