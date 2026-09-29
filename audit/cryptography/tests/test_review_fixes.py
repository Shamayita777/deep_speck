"""Regression tests for the independent-review findings (F1, F3, F4, F5, F7)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from audit.cryptography import sealed_dataset as sds
from audit.cryptography.experiments.ce1 import gohr_signal_destruction as ce1_driver
from audit.cryptography.frozen_design import CE1 as CE1_SPEC


# ============ F1: ONE sealed evaluation set shared by all blocks ============

def _gen(n, tag=0):
    r = np.random.default_rng(tag)
    return r.integers(0, 2, (n, 64), dtype=np.uint8), r.integers(0, 2, n, dtype=np.uint8)


def test_sealed_set_generated_once_and_reused(tmp_path):
    calls = {"n": 0}
    def gen(n):
        calls["n"] += 1
        return _gen(n, calls["n"])                # different data each call
    a = sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=gen, rounds=5,
                                          differential=(0x0040, 0x0000), n_samples=32)
    b = sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=gen, rounds=5,
                                          differential=(0x0040, 0x0000), n_samples=32)
    assert calls["n"] == 1                        # generator NOT called again
    assert a["sha256"] == b["sha256"]
    assert b["reused"] is True
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["Y"], b["Y"])


def test_sealed_set_is_persisted_with_manifest_and_hash(tmp_path):
    m = sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n), rounds=5,
                                          differential=(0x0040, 0x0000), n_samples=32)
    assert sds.data_path(tmp_path).exists() and sds.manifest_path(tmp_path).exists()
    rec = json.loads(sds.manifest_path(tmp_path).read_text())
    assert rec["sha256"] == m["sha256"] and len(rec["sha256"]) == 64
    assert rec["role"] == "sealed_evaluation_set"


def test_tampered_sealed_set_is_refused(tmp_path):
    sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n), rounds=5,
                                      differential=(0x0040, 0x0000), n_samples=32)
    X, Y = _gen(32, 99)
    np.savez(sds.data_path(tmp_path), X=X, Y=Y)   # substitute a different set
    with pytest.raises(sds.SealedDatasetError, match="has hash"):
        sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n), rounds=5,
                                          differential=(0x0040, 0x0000), n_samples=32)


def test_missing_data_file_is_not_silently_regenerated(tmp_path):
    sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n), rounds=5,
                                      differential=(0x0040, 0x0000), n_samples=32)
    sds.data_path(tmp_path).unlink()
    with pytest.raises(sds.SealedDatasetError, match="refusing to regenerate"):
        sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n), rounds=5,
                                          differential=(0x0040, 0x0000), n_samples=32)


@pytest.mark.parametrize("kw,msg", [
    ({"rounds": 7}, "rounds"),
    ({"n_samples": 64}, "n_samples"),
    ({"differential": (0x0080, 0x0000)}, "differential"),
])
def test_sealed_set_config_mismatch_refused(tmp_path, kw, msg):
    base = dict(rounds=5, differential=(0x0040, 0x0000), n_samples=32)
    sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n), **base)
    with pytest.raises(sds.SealedDatasetError, match=msg):
        sds.prepare_sealed_evaluation_set(tmp_path, generate_fn=lambda n: _gen(n),
                                          **{**base, **kw})


def test_shared_invariant_detects_divergent_blocks():
    ok = [{"sealed_evaluation_sha256": "a" * 64} for _ in range(8)]
    assert sds.verify_shared_invariant(ok)["n_blocks_checked"] == 8
    bad = ok[:-1] + [{"sealed_evaluation_sha256": "b" * 64}]
    with pytest.raises(sds.SealedDatasetError, match="distinct evaluation artifacts"):
        sds.verify_shared_invariant(bad)


def test_all_blocks_reference_the_same_sealed_hash(tmp_path):
    """End-to-end: every block in a run must carry one identical hash."""
    sealed = sds.prepare_sealed_evaluation_set(
        tmp_path / "sealed", generate_fn=lambda n: _gen(n), rounds=5,
        differential=(0x0040, 0x0000), n_samples=16)

    def data_fn(i):
        r = np.random.default_rng(i)
        data_fn.sealed_sha256 = sealed["sha256"]
        return (r.integers(0, 2, (16, 8), np.uint8), r.integers(0, 2, 16, np.uint8),
                sealed["X"], sealed["Y"])
    data_fn.sealed_sha256 = sealed["sha256"]

    _, cert = ce1_driver.run(n_blocks=3, min_valid_blocks=2, production=False,
                             output_path=tmp_path / "evidence_current/ce1/c.json",
                             train_eval_fn=lambda *a, **k: 0.6, data_fn=data_fn,
                             repo_root=tmp_path)
    r = cert["results"]
    hashes = {b["sealed_evaluation_sha256"] for b in r["blocks"] if b.get("status") == "VALID"}
    assert hashes == {sealed["sha256"]}
    assert r["sealed_evaluation"]["shared_evaluation_sha256"] == sealed["sha256"]


# ============ F3: failure tolerance ============

def _run_with_failures(tmp_path, failing_blocks, n_blocks=8, min_valid=6):
    sealed = sds.prepare_sealed_evaluation_set(
        tmp_path / "sealed", generate_fn=lambda n: _gen(n), rounds=5,
        differential=(0x0040, 0x0000), n_samples=16)

    def data_fn(i):
        if i in failing_blocks:
            raise RuntimeError(f"simulated failure in block{i}")
        r = np.random.default_rng(i)
        data_fn.sealed_sha256 = sealed["sha256"]
        return (r.integers(0, 2, (16, 8), np.uint8), r.integers(0, 2, 16, np.uint8),
                sealed["X"], sealed["Y"])
    data_fn.sealed_sha256 = sealed["sha256"]
    calls = {"n": 0}
    def train(*a, **k):
        calls["n"] += 1
        return 0.9 if calls["n"] % 2 else 0.5
    return ce1_driver.run(n_blocks=n_blocks, min_valid_blocks=min_valid, production=False,
                          output_path=tmp_path / "evidence_current/ce1/c.json",
                          train_eval_fn=train, data_fn=data_fn, repo_root=tmp_path)


@pytest.mark.parametrize("failing,expected_valid", [(set(), 8), ({2}, 7), ({1, 5}, 6)])
def test_failures_within_tolerance_still_certify(tmp_path, failing, expected_valid):
    _, cert = _run_with_failures(tmp_path, failing)
    r = cert["results"]
    assert r["n_blocks_valid"] == expected_valid
    assert len(r["failures"]) == len(failing)
    assert r["n_blocks"] == expected_valid                 # analysis uses valid blocks only


def test_failure_does_not_abort_later_blocks(tmp_path):
    """An early failure must not discard blocks 4-8."""
    _, cert = _run_with_failures(tmp_path, {0})
    ids = [b["block_id"] for b in cert["results"]["blocks"]]
    assert ids == [f"block{i}" for i in range(8)]          # all attempted
    assert cert["results"]["blocks"][0]["status"] == "FAILED"
    assert cert["results"]["n_blocks_valid"] == 7


def test_three_failures_refuse_certification(tmp_path):
    with pytest.raises(ValueError, match="only 5 valid blocks"):
        _run_with_failures(tmp_path, {0, 3, 6})


def test_failure_reason_is_persisted(tmp_path):
    _, cert = _run_with_failures(tmp_path, {4})
    f = cert["results"]["failures"][0]
    assert f["block_id"] == "block4" and f["reason"] == "RuntimeError"
    assert "simulated failure" in f["detail"]


# ============ F4: frozen parameters not operator-modifiable ============

@pytest.mark.parametrize("override", [
    {"n_blocks": 10}, {"min_valid_blocks": 4}, {"seed_base": 999},
    {"train_samples": 1000}, {"validation_samples": 1000}, {"evaluation_samples": 1000},
])
def test_frozen_parameter_override_is_refused(override):
    with pytest.raises(ValueError, match="frozen scientific parameters were overridden"):
        ce1_driver.validate_frozen_parameters(**override)


def test_frozen_parameters_accept_the_frozen_values():
    out = ce1_driver.validate_frozen_parameters(
        n_blocks=CE1_SPEC.n_blocks, min_valid_blocks=CE1_SPEC.min_valid_blocks,
        seed_base=CE1_SPEC.seed_base, train_samples=CE1_SPEC.train_samples,
        validation_samples=CE1_SPEC.validation_samples,
        evaluation_samples=CE1_SPEC.evaluation_samples)
    assert out["n_blocks"] == 8 and out["evaluation_samples"] == 10 ** 6


def test_cli_override_refused_end_to_end(tmp_path, capsys):
    rc = ce1_driver.main(["--n-blocks", "10", "--output",
                          str(tmp_path / "evidence_current/ce1/c.json"),
                          "--repo-root", str(tmp_path)])
    assert rc == 1
    assert "frozen scientific parameters were overridden" in capsys.readouterr().out


# ============ F5: persisted artifact is the terminal-epoch model ============

def test_production_persists_final_epoch_artifact_not_best_val():
    import inspect
    src = inspect.getsource(ce1_driver.build_production_components)
    assert "_FINAL_EPOCH.keras" in src
    assert "terminal_model_sha256" in src
    assert "bestval_DEBUG_ONLY" in src          # val-loss artifact clearly demoted
    assert '"checkpoint_rule": "FINAL_EPOCH"' in src


# ============ F7: orphaned CE3 threshold quarantined ============

def test_supported_threshold_not_used_by_any_current_ce_path():
    import inspect
    from audit.cryptography.experiments.ce3 import design as ce3
    assert "supported_threshold" not in inspect.signature(ce3.aggregate_replicates).parameters
    from audit.cryptography import evaluation
    doc = inspect.getdoc(evaluation.CryptographicEvaluator) or ""
    assert "LEGACY / QUARANTINED" in doc
    assert "must not be used by any current CE production path" in " ".join(doc.split())
