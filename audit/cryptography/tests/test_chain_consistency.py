"""
Layer-consistency trace: config -> dataset -> model/checkpoint -> experiment
-> statistical unit -> statistics -> certificate -> report.

For each CE, one complete toy execution is traced and the quantities in
the final certificate are asserted to be the quantities computed upstream.
"""
import json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from audit.cryptography import provenance
from audit.cryptography.provenance import EXPERIMENT_DESIGN_VERSION as DESIGN_VERSION
from audit.cryptography import audit_config as config, certificate, integrity, statistics as stats
from audit.cryptography.experiments.ce1 import design as ce1
from audit.cryptography.experiments.ce2 import design as ce2
from audit.cryptography.experiments.ce3 import design as ce3
from audit.cryptography.experiments.ce4 import design as ce4

ARCHIVE = ROOT / "Archive"            # audit/cryptography/Archive


def _cert(experiment_id, results, ckpt=None):
    c = {
        "certificate_schema_version": certificate.CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": experiment_id, "experiment_design_version": DESIGN_VERSION,
        "reference_configuration": config.REFERENCE.to_dict(),
        "provenance": provenance.build_provenance(config_id=f"{experiment_id}-toy",
                                                   probe_seed=11, checkpoint_info=ckpt),
        "results": results,
        "claim_scope": "TOY EXECUTION - structural validation only, no scientific inference",
    }
    certificate.validate_certificate(c)
    return c


@pytest.mark.skipif(not (ARCHIVE / "best5depth10.h5").exists(), reason="archive absent")
def test_chain_checkpoint_layer_is_consistent():
    info = integrity.verify_reference_checkpoint(ARCHIVE / "best5depth10.h5")
    c = _cert("CE2", {"placeholder": True}, ckpt=info)
    # the depth the certificate advertises is the depth physically verified
    assert c["provenance"]["checkpoint"]["realized_depth"] == 10
    assert c["reference_configuration"]["depth"] == 10
    assert c["provenance"]["checkpoint"]["sha256"] == config.REFERENCE_CHECKPOINT_SHA256


def test_chain_ce1_toy_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    blocks, base, dest = [], [], []
    for i in range(4):
        Xtr = rng.integers(0, 2, (64, 8), dtype=np.uint8)
        Ytr = rng.integers(0, 2, 64, dtype=np.uint8)
        Xev = rng.integers(0, 2, (32, 8), dtype=np.uint8)
        Yev = rng.integers(0, 2, 32, dtype=np.uint8)
        blk = ce1.build_ce1_block(Xtr, Ytr, Xev, Yev, block_id=f"blk{i}", permutation_seed=i)
        blocks.append(blk)
        base.append(0.90 + 0.01 * i); dest.append(0.50 + 0.001 * i)   # stand-in metrics
    res = ce1.paired_block_analysis(base, dest)
    c = _cert("CE1", res)
    p = certificate.write_certificate(c, tmp_path / "audit/cryptography/evidence_current" / "ce1" / "cert.json", repo_root=tmp_path)
    loaded = json.loads(p.read_text())
    # UPSTREAM  ->  CERTIFICATE consistency
    assert loaded["results"]["n_blocks"] == len(blocks) == 4
    # genuinely enforced: the per-block differences stored in the certificate
    # must equal the upstream baseline-minus-destroyed values, elementwise
    assert np.allclose(loaded["results"]["paired_difference"]["raw_block_differences"],
                       [b - d for b, d in zip(base, dest)], rtol=0, atol=1e-12)
    assert np.isclose(loaded["results"]["baseline_mean"], np.mean(base))
    assert np.isclose(loaded["results"]["destroyed_mean"], np.mean(dest))
    assert np.isclose(loaded["results"]["paired_difference"]["mean"],
                      np.mean(np.array(base) - np.array(dest)))
    assert "block" in loaded["results"]["statistical_unit"]


def test_chain_ce2_toy_end_to_end(tmp_path):
    runs = [{"run_id": f"r{i}", "rho": r, "n": 1000}
            for i, r in enumerate([-0.21, -0.19, -0.20, -0.22, -0.18])]
    res = ce2.run_level_association(runs)
    c = _cert("CE2", res)
    loaded = json.loads(certificate.write_certificate(c, tmp_path / "audit/cryptography/evidence_current" / "ce2" / "cert.json", repo_root=tmp_path).read_text())
    assert loaded["results"]["n_runs"] == 5
    assert len(loaded["results"]["per_run"]) == 5
    assert np.isclose(loaded["results"]["run_level_inference"]["mean"],
                      np.mean([r["rho"] for r in runs]))
    assert loaded["results"]["causal_interpretation_permitted"] is False


def test_chain_ce3_toy_end_to_end(tmp_path):
    sel = list(np.linspace(0.05, 0.25, 20))
    res = ce3.aggregate_replicates(sel, n_splits_per_replicate=5,
                                   calibration_validated=True, calibration_p_value=0.001)
    c = _cert("CE3", res)
    loaded = json.loads(certificate.write_certificate(c, tmp_path / "audit/cryptography/evidence_current" / "ce3" / "cert.json", repo_root=tmp_path).read_text())
    r = loaded["results"]
    assert r["n_replicates"] == 20 and r["n_splits_per_replicate"] == 5
    assert len(r["selectivity_replicates"]) == 20            # replicate unit preserved
    assert np.isclose(r["mean_selectivity"], np.mean(sel))
    assert "n_folds" not in json.dumps(loaded)
    assert r["statistical_unit"] == "independent evaluation replicate"


def test_chain_ce4_toy_end_to_end(tmp_path):
    rng = np.random.default_rng(5)
    n, d = 300, 32
    a = rng.integers(0, 2, (n, d), dtype=np.uint8)
    b = a.copy()
    for i in range(n):
        b[i, rng.choice(d, 4, replace=False)] ^= 1
    elig = ce4.eligible_mask(a, b, n_flips=2)
    acct = ce4.population_accounting(n, elig)
    ae, be = a[elig], b[elig]
    pos = np.array([np.flatnonzero(row)[:2] for row in (ae ^ be)])
    (sa, sb), (ca, cb) = ce4.build_matched_interventions(ae, be, pos)
    inv = ce4.verify_intervention_invariants(ae, be, sa, sb, ca, cb)
    s_delta = rng.normal(0.30, 0.05, ae.shape[0])
    c_delta = rng.normal(0.05, 0.05, ae.shape[0])
    res = ce4.paired_contrast(s_delta, c_delta)
    res.update({"population": acct, "invariants": inv})
    c = _cert("CE4", res)
    loaded = json.loads(certificate.write_certificate(c, tmp_path / "audit/cryptography/evidence_current" / "ce4" / "cert.json", repo_root=tmp_path).read_text())
    r = loaded["results"]
    # population accounting flows through unchanged
    assert r["population"]["N_total"] == n
    assert r["population"]["N_analyzed"] == int(elig.sum())
    assert r["population"]["N_analyzed"] + r["population"]["N_excluded"] == n
    # the contrast was computed on exactly the analyzed population
    assert len(s_delta) == r["population"]["N_analyzed"]
    assert r["invariants"]["magnitude_matched"] is True
    assert r["invariants"]["control_preserves_xor"] is True
    assert np.isclose(r["mean_gap"], float(np.mean(s_delta - c_delta)))
    assert r["experiment"] == "Ciphertext-Difference Intervention Sensitivity"
    assert "causal necessity" not in json.dumps(loaded).lower()
