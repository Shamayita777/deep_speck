"""
NON-EVIDENTIARY end-to-end smoke test of the REAL Gohr II-4 adapter
driven by the generic runner. Tiny data (2k/500/2k), 2 epochs, 3 blocks.
It verifies mechanics - real training, the on-disk architecture probe,
terminal-model reload identity, the shared sealed test set - and can
never be confused with II-4 evidence: execution_mode=SMOKE and the
experiment_id carries 'SMOKE-NONEVIDENTIARY'.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from audit.implementation.conformance import ConformanceAdapter          # noqa: E402
from audit.implementation.ii4_experiment import build_gohr_ii4_smoke_plan  # noqa: E402
from audit.implementation.ii4_runner import II4Runner                    # noqa: E402
from audit.implementation.adapters.gohr_ii4 import GohrII4Adapter      # noqa: E402


@pytest.fixture
def smoke(tmp_path):
    plan = build_gohr_ii4_smoke_plan(output_dir=tmp_path / "run", n_blocks=3, epochs=2,
                                     train_samples=2000, validation_samples=500,
                                     test_samples=2000)
    adapter = GohrII4Adapter(protocol=plan.fixed_protocol)
    return plan, adapter, tmp_path / "run"


def test_real_adapter_satisfies_generic_contract():
    assert isinstance(GohrII4Adapter(protocol={}), ConformanceAdapter)


def test_real_adapter_passes_reference_reg_param(smoke):
    plan, adapter, _ = smoke
    model = adapter.build_model(10, seed=1)
    regs = {float(l.kernel_regularizer.l2) for l in model.layers
            if getattr(l, "kernel_regularizer", None) is not None}
    assert len(regs) == 1
    assert next(iter(regs)) == pytest.approx(1e-5)   # reference value, NOT the 1e-4 default


def test_real_gohr_ii4_smoke_end_to_end(smoke):
    plan, adapter, root = smoke
    report = II4Runner(plan=plan, adapter=adapter).run(execute=True)

    assert report["non_evidentiary"] is True and report["execution_mode"] == "SMOKE"
    assert report["firewall"]["confirmatory_data_mode"] == "SHARED"
    assert set(report["firewall"]["sealed_dataset_hashes"].values()) == \
        {report["sealed_test_set"]["hash"]}
    assert len(report["firewall"]["consumed_evaluation_keys"]) == 6

    state = json.loads((root / "ii4_state.json").read_text())
    for blk in state["blocks"].values():
        d, r = blk["arms"]["declared"], blk["arms"]["realized"]
        assert (d["probe_in_memory"], d["probe_artifact"]) == (10, 10)   # real Add count
        assert (r["probe_in_memory"], r["probe_artifact"]) == (5, 5)
        assert d["terminal_model"]["reload_weight_identical"] is True
        assert r["terminal_model"]["reload_weight_identical"] is True
        assert d["seed"] != r["seed"]
        assert d["terminal_epoch"] == 2 and d["evaluation_rule"] == "terminal_epoch"
        assert 0.0 <= d["sealed_test_accuracy"] <= 1.0
