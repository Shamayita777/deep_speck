"""CE-v2 CPU test suite: unit, integration on tiny fixtures, and negative tests."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]      # .../audit/cryptography
sys.path.insert(0, str(ROOT.parents[1]))   # repo root, so `audit.` resolves

from audit.cryptography import provenance
from audit.cryptography.provenance import EXPERIMENT_DESIGN_VERSION as DESIGN_VERSION
from audit.cryptography import audit_config as config, certificate, integrity, output_policy as paths, statistics as stats
from audit.cryptography.experiments.ce1 import design as ce1
from audit.cryptography.experiments.ce2 import design as ce2
from audit.cryptography.experiments.ce3 import design as ce3
from audit.cryptography.experiments.ce4 import design as ce4

ARCHIVE = ROOT / "Archive"            # audit/cryptography/Archive

# ---------------------------------------------------------------------------
# ENVIRONMENT DEPENDENCE
#
# The v2 package is self-contained, but four tests exercise components that
# ship OUTSIDE this bundle:
#   * three `gohr` tests need cryptography/gohr/speck.py (original repo);
#   * one historical-immutability test needs cryptography/evidence/.
# They are skipped - never silently passed - when those are absent, and the
# reported count therefore differs between a bundle-only checkout and the
# full repository. Both counts are stated in the report.
# ---------------------------------------------------------------------------
HAVE_GOHR = (ROOT / "gohr" / "speck.py").exists()
HAVE_HISTORICAL_EVIDENCE = (ROOT / "evidence" / "ce3").is_dir()
requires_gohr = pytest.mark.skipif(
    not HAVE_GOHR, reason="requires cryptography/gohr/speck.py from the original repository")
requires_historical = pytest.mark.skipif(
    not HAVE_HISTORICAL_EVIDENCE,
    reason="requires cryptography/evidence/ from the original repository")

# ---------- configuration ----------
def test_reference_configuration_is_frozen():
    assert config.REFERENCE.rounds == 5
    assert config.REFERENCE.differential == (0x0040, 0x0000)
    assert config.REFERENCE.depth == 10
    assert config.REFERENCE.l2_reg == 1e-5

@pytest.mark.parametrize("kw", [
    dict(rounds=7, differential=(0x0040, 0x0000), depth=10),
    dict(rounds=5, differential=(0x0080, 0x0000), depth=10),
    dict(rounds=5, differential=(0x0040, 0x0000), depth=5),
    dict(rounds=5, differential=(0x0040, 0x0000), depth=10, l2_reg=1e-4),
])
def test_wrong_configuration_is_rejected(kw):
    with pytest.raises(config.ConfigurationError):
        config.require_reference_config(**kw)

def test_reference_configuration_accepted():
    config.require_reference_config(rounds=5, differential=(0x0040, 0x0000), depth=10, l2_reg=1e-5)

# ---------- model integrity ----------
@pytest.mark.skipif(not (ARCHIVE / "best5depth10.h5").exists(), reason="archive absent")
def test_depth10_checkpoint_verifies():
    info = integrity.verify_reference_checkpoint(ARCHIVE / "best5depth10.h5")
    assert info["realized_depth"] == 10
    assert info["conv1d"] == 1 + 2 * 10
    assert info["residual_merges"] == 10
    assert info["sha256"] == config.REFERENCE_CHECKPOINT_SHA256

@pytest.mark.skipif(not (ROOT / "evidence/ce1/best5depth10 (10).h5").exists(), reason="absent")
def test_misleadingly_named_depth5_checkpoint_is_refused():
    """Filename says depth10; the file is depth 5. Must be rejected."""
    p = ROOT / "evidence/ce1/best5depth10 (10).h5"
    assert integrity.detect_depth(p) == 5
    with pytest.raises(integrity.ModelIntegrityError, match="HISTORICAL depth-5"):
        integrity.verify_reference_checkpoint(p)

def test_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(integrity.ModelIntegrityError, match="not found"):
        integrity.detect_depth(tmp_path / "nope.h5")

def test_corrupted_checkpoint_fails_loudly(tmp_path):
    bad = tmp_path / "corrupt.h5"; bad.write_bytes(b"not an hdf5 file")
    with pytest.raises(integrity.ModelIntegrityError):
        integrity.detect_depth(bad)

@pytest.mark.skipif(not (ARCHIVE / "best5depth10.h5").exists(), reason="archive absent")
def test_wrong_expected_hash_is_refused():
    with pytest.raises(integrity.ModelIntegrityError, match="sha256"):
        integrity.verify_reference_checkpoint(ARCHIVE / "best5depth10.h5",
                                              expected_sha256="0" * 64)

@pytest.mark.skipif(not (ARCHIVE / "best5depth10.h5").exists(), reason="archive absent")
def test_wrong_expected_depth_is_refused():
    with pytest.raises(integrity.ModelIntegrityError, match="depth"):
        integrity.verify_reference_checkpoint(ARCHIVE / "best5depth10.h5", expected_depth=5)

# ---------- xdp_plus regression ----------
def _brute(n, a, b, g):
    M = (1 << n) - 1; c = 0
    for x in range(1 << n):
        for y in range(1 << n):
            if (((x ^ a) + (y ^ b)) & M) ^ ((x + y) & M) == g: c += 1
    return c / float(1 << (2 * n))

@requires_gohr
def test_xdp_plus_exhaustive_small_width():
    from audit.cryptography.gohr import speck as sp
    orig_ws, orig_mask = sp.WORD_SIZE, sp.MASK_VAL
    try:
        n = 4; sp.WORD_SIZE = lambda: n; sp.MASK_VAL = (1 << n) - 1
        for a in range(1 << n):
            for b in range(1 << n):
                g = np.arange(1 << n, dtype=np.uint32)
                got = sp.xdp_plus(np.full(1 << n, a, np.uint32), np.full(1 << n, b, np.uint32), g)
                for gi in range(1 << n):
                    assert abs(float(got[gi]) - _brute(n, a, b, gi)) < 1e-12
    finally:
        sp.WORD_SIZE, sp.MASK_VAL = orig_ws, orig_mask

@requires_gohr
def test_xdp_plus_randomized_16bit_against_independent_formula():
    from audit.cryptography.gohr import speck as sp
    def ref(a, b, g, n=16):
        M = (1 << n) - 1; a &= M; b &= M; g &= M
        eq = (~a ^ b) & (~a ^ g) & M
        eq_sh = (~(a << 1) ^ (b << 1)) & (~(a << 1) ^ (g << 1)) & M
        if eq_sh & (a ^ b ^ g ^ ((b << 1) & M)) & M: return 0.0
        return 2.0 ** (-bin((~eq) & (M >> 1) & M).count("1"))
    rng = np.random.default_rng(7)
    A = rng.integers(0, 1 << 16, 5000, dtype=np.uint64).astype(np.uint32)
    B = rng.integers(0, 1 << 16, 5000, dtype=np.uint64).astype(np.uint32)
    G = rng.integers(0, 1 << 16, 5000, dtype=np.uint64).astype(np.uint32)
    got = sp.xdp_plus(A, B, G)
    for i in range(len(A)):
        assert abs(float(got[i]) - ref(int(A[i]), int(B[i]), int(G[i]))) < 1e-15

@requires_gohr
def test_speck_reference_test_vector():
    from audit.cryptography.gohr import speck as sp
    assert sp.check_testvector()

# ---------- p-value reporting ----------
def test_p_value_never_reported_as_literal_zero():
    r = stats.p_value_report(0.0)
    assert r["p_value"] is None                      # NOT 0.0
    assert r["p_underflow"] is True
    assert r["p_upper_bound"] == 1e-300
    assert r["p_representation"] == "p < 1e-300"
    assert r["log10_p"] is None


def test_underflowed_p_cannot_be_used_as_a_number():
    """A consumer doing arithmetic on p_value must fail, not silently see 0."""
    r = stats.p_value_report(0.0)
    with pytest.raises(TypeError):
        _ = r["p_value"] < 0.05


def test_certificate_rejects_underflow_marked_with_numeric_p():
    c = _valid_cert()
    c["results"]["inference"] = {"p_value": 1e-310, "p_underflow": True}
    with pytest.raises(certificate.CertificateSchemaError, match="not null"):
        certificate.validate_certificate(c)

def test_normal_p_value_reported_with_log10():
    r = stats.p_value_report(1e-12)
    assert r["p_underflow"] is False and r["log10_p"] == pytest.approx(-12)

# ---------- CE1-v2 ----------
def _toy(n_train=64, n_eval=32, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, (n_train, d), dtype=np.uint8), rng.integers(0, 2, n_train, dtype=np.uint8),
            rng.integers(0, 2, (n_eval, d), dtype=np.uint8), rng.integers(0, 2, n_eval, dtype=np.uint8))

def test_ce1_permutes_only_training_labels():
    Xtr, Ytr, Xev, Yev = _toy()
    blk = ce1.build_ce1_block(Xtr, Ytr, Xev, Yev, block_id="b0", permutation_seed=1)
    assert not np.array_equal(blk.Y_train_baseline, blk.Y_train_destroyed)
    assert np.array_equal(np.sort(blk.Y_train_baseline), np.sort(blk.Y_train_destroyed))
    assert np.array_equal(blk.Y_eval, Yev)          # evaluation labels intact
    assert np.array_equal(blk.X_eval, Xev)

def test_ce1_evaluation_set_is_shared_and_untouched():
    Xtr, Ytr, Xev, Yev = _toy()
    before = Yev.copy()
    blk = ce1.build_ce1_block(Xtr, Ytr, Xev, Yev, block_id="b0", permutation_seed=2)
    inv = blk.verify_invariants()
    assert inv["evaluation_labels_intact"] and inv["evaluation_shared_between_arms"]
    assert np.array_equal(Yev, before)

def test_ce1_blocks_have_distinct_ids_and_seeds():
    Xtr, Ytr, Xev, Yev = _toy()
    blocks = [ce1.build_ce1_block(Xtr, Ytr, Xev, Yev, block_id=f"b{i}", permutation_seed=100 + i)
              for i in range(4)]
    assert len({b.block_id for b in blocks}) == 4
    assert len({b.permutation_seed for b in blocks}) == 4

def test_ce1_rejects_mismatched_lengths():
    Xtr, Ytr, Xev, Yev = _toy()
    with pytest.raises(ce1.BlockConstructionError):
        ce1.build_ce1_block(Xtr, Ytr[:-1], Xev, Yev, block_id="bad", permutation_seed=1)

def test_ce1_paired_analysis_unit_is_the_block():
    res = ce1.paired_block_analysis([0.93, 0.92, 0.94, 0.93], [0.50, 0.51, 0.49, 0.50])
    assert res["n_blocks"] == 4
    assert "block" in res["statistical_unit"]
    assert res["paired_difference"]["mean"] > 0.4

def test_ce1_non_significance_is_not_equivalence():
    """Frozen design: equivalence is DISABLED, never inferred from non-significance."""
    res = ce1.paired_block_analysis([0.93] * 4, [0.50, 0.51, 0.49, 0.50])
    eq = res["destroyed_vs_chance"]["equivalence"]
    assert eq["enabled"] is False and eq["verdict"] == "NOT_ASSESSED"

def test_ce1_equivalence_margin_is_not_accepted():
    """The rejected eps=0.01 must not be reintroducible through the API."""
    import inspect
    assert "equivalence_margin" not in inspect.signature(ce1.paired_block_analysis).parameters

# ---------- CE3-v2 ----------
def test_ce3_statistical_unit_is_replicate_not_fold():
    out = ce3.aggregate_replicates([0.1] * 19 + [0.12], n_splits_per_replicate=5,
                                   calibration_validated=True, calibration_p_value=0.001)
    assert out["n_replicates"] == 20
    assert out["n_splits_per_replicate"] == 5
    assert out["statistical_unit"] == "independent evaluation replicate"
    assert len(out["selectivity_replicates"]) == 20     # not 100

def test_ce3_dead_thresholds_absent_from_v2_api():
    import inspect
    sig = inspect.signature(ce3.aggregate_replicates).parameters
    assert "supported_threshold" not in sig and "inconclusive_threshold" not in sig
    code = _code_without_docstrings(ce3)
    assert "supported_threshold" not in code and "inconclusive_threshold" not in code

def test_ce3_calibration_failure_forces_inconclusive():
    out = ce3.aggregate_replicates([0.3] * 20, n_splits_per_replicate=5,
                                   calibration_validated=False, calibration_p_value=0.001)
    assert out["decision"] == "INCONCLUSIVE" and "calibration" in out["decision_reason"]

def test_ce3_negative_selectivity_is_not_supported():
    out = ce3.aggregate_replicates([-0.2] * 20, n_splits_per_replicate=5,
                                   calibration_validated=True, calibration_p_value=0.001)
    assert out["decision"] == "NOT_SUPPORTED"

def test_ce3_reports_real_values_not_null():
    out = ce3.aggregate_replicates(list(np.linspace(0.05, 0.2, 20)), n_splits_per_replicate=5,
                                   calibration_validated=True, calibration_p_value=0.001)
    for k in ("mean_selectivity", "ci95", "effect_size_dz", "inference"):
        assert out[k] is not None
    assert out["calibration"]["role"].startswith("methodological positive control")

def test_ce3_rejects_single_replicate():
    with pytest.raises(ValueError):
        ce3.aggregate_replicates([0.1], n_splits_per_replicate=5,
                                 calibration_validated=True, calibration_p_value=0.001)

# ---------- CE4-v2 ----------
def _pair(n=200, d=32, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, (n, d), dtype=np.uint8)
    b = a.copy()
    flip = rng.integers(0, d, (n, 3))
    for i in range(n): b[i, flip[i]] ^= 1
    return a, b

def test_ce4_control_preserves_xor_and_matches_magnitude():
    """Both arms must change the SAME number of bits (2k), not k vs 2k."""
    a, b = _pair()
    diff = a ^ b
    pos = np.array([np.flatnonzero(row)[:2] for row in diff])      # 2k = 2
    (sa, sb), (ca, cb) = ce4.build_matched_interventions(a, b, pos)
    inv = ce4.verify_intervention_invariants(a, b, sa, sb, ca, cb)
    assert inv["bits_changed_per_sample"] == 2
    assert inv["control_preserves_xor"] and inv["magnitude_matched"]
    assert np.array_equal(ca ^ cb, a ^ b)
    assert not np.array_equal(sa ^ sb, a ^ b)

def test_ce4_xor_identity_holds_elementwise():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 1000, dtype=np.uint8); b = rng.integers(0, 2, 1000, dtype=np.uint8)
    assert np.array_equal((a ^ 1) ^ (b ^ 1), a ^ b)

def test_ce4_invariant_check_detects_magnitude_mismatch():
    a, b = _pair()
    diff = a ^ b
    pos = np.array([np.flatnonzero(row)[:2] for row in diff])
    (sa, sb), (ca, cb) = ce4.build_matched_interventions(a, b, pos)
    ca[0, (pos[0, 0] + 5) % ca.shape[1]] ^= 1          # extra bit in control only
    with pytest.raises(ce4.InterventionError):
        ce4.verify_intervention_invariants(a, b, sa, sb, ca, cb)

def test_ce4_eligibility_and_exclusion_accounting():
    a, b = _pair(n=100)
    elig = ce4.eligible_mask(a, b, n_flips=3)
    acct = ce4.population_accounting(100, elig)
    assert acct["N_total"] == 100
    assert acct["N_analyzed"] + acct["N_excluded"] == 100
    assert 0.0 <= acct["exclusion_rate"] <= 1.0
    assert "part of the estimand" in acct["note"]

def _code_without_docstrings(mod):
    """Source with docstrings stripped: explanatory history is not a leftover."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_ce4_has_no_arbitrary_one_percent_gate():
    code = _code_without_docstrings(ce4)
    assert "0.01" not in code
    assert "causal necessity" not in code.lower()
    assert ce4.EXPERIMENT_NAME == "Ciphertext-Difference Intervention Sensitivity"

def test_ce4_control_wording_is_narrow():
    assert "declared XOR-difference target" in ce4.CONTROL_DESCRIPTION
    assert "cryptographically inert" not in ce4.CONTROL_DESCRIPTION

def test_ce4_paired_contrast_reports_estimand_and_scope():
    rng = np.random.default_rng(0)
    s = rng.normal(0.30, 0.05, 400); c = rng.normal(0.05, 0.05, 400)
    out = ce4.paired_contrast(s, c)
    assert out["mean_gap"] > 0 and out["ci95"][0] > 0
    assert "NOT establish" in out["claim_scope"]
    assert out["statistical_unit"].startswith("eligible sample")

def test_ce4_rejects_unpaired_inputs():
    with pytest.raises(ce4.InterventionError):
        ce4.paired_contrast([1, 2, 3], [1, 2])

# ---------- CE2-v2 ----------
def test_ce2_is_observational_and_run_level():
    out = ce2.run_level_association([{"run_id": f"r{i}", "rho": r, "n": 100000}
                                     for i, r in enumerate([-0.21, -0.19, -0.20, -0.22, -0.18])])
    assert out["causal_interpretation_permitted"] is False
    assert out["n_runs"] == 5
    assert out["run_level_inference"]["statistical_unit"].startswith("independent run")
    assert "single-trail" in out["analytical_target"].lower()

def test_ce2_target_is_not_overclaimed():
    t = ce2.ANALYTICAL_TARGET.lower()
    assert "not the cipher's exact differential probability" in t
    assert "not 'what the network learned'" in t

# ---------- write protection ----------
@pytest.mark.parametrize("bad", [
    "evidence/ce1/cert.json", "audit/cryptography/evidence/ce3/x.json",
    "some/frozen/x.json", "audit/evidence_bundle/x.json", "results/cert.json",
])
def test_historical_output_paths_are_refused(bad):
    with pytest.raises(paths.HistoricalWriteError):
        paths.assert_audit_output_path(bad)

def test_v2_output_path_accepted(tmp_path):
    # mock repository at tmp_path, passed EXPLICITLY: an implicit (cwd) root must be
    # the real repository root of this package (see test_wrong_cwd_fails_closed_*)
    assert paths.assert_audit_output_path("audit/cryptography/evidence_current/ce3/certificate.json",
                                          repo_root=tmp_path)


@pytest.mark.parametrize("traversal", [
    "audit/cryptography/evidence_current/../evidence/ce1/cert.json",
    "audit/cryptography/evidence_current/../../escape.json",
    "audit/cryptography/evidence_current/ce3/../../evidence/ce4/cert.json",
    "audit/cryptography/evidence_current/./../evidence/cert.json",
])
def test_path_traversal_cannot_bypass_protection(traversal, tmp_path, monkeypatch):
    """`..` must not smuggle a write out of evidence_current/ despite the token."""
    with pytest.raises(paths.HistoricalWriteError):
        paths.assert_audit_output_path(traversal, repo_root=tmp_path)


def test_absolute_path_outside_v2_root_is_refused(tmp_path, monkeypatch):
    with pytest.raises(paths.HistoricalWriteError):
        paths.assert_audit_output_path(tmp_path / "evidence" / "ce1" / "cert.json",
                                       repo_root=tmp_path)


def test_symlink_escape_is_refused(tmp_path, monkeypatch):
    (tmp_path / "audit/cryptography/evidence_current").mkdir(parents=True)
    (tmp_path / "historical").mkdir()
    link = tmp_path / "audit/cryptography/evidence_current" / "sneaky"
    link.symlink_to(tmp_path / "historical", target_is_directory=True)
    with pytest.raises(paths.HistoricalWriteError):
        paths.assert_audit_output_path("audit/cryptography/evidence_current/sneaky/cert.json",
                                       repo_root=tmp_path)

@requires_historical
def test_historical_evidence_unchanged_after_v2_write(tmp_path):
    hist = ROOT / "evidence" / "ce3"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in hist.rglob("*") if p.is_file()}
    cert = _valid_cert()
    certificate.write_certificate(cert, tmp_path / "audit/cryptography/evidence_current" / "ce3" / "c.json",
                                  repo_root=tmp_path)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in hist.rglob("*") if p.is_file()}
    assert before == after and before

# ---------- certificate schema ----------
def _valid_cert():
    return {
        "certificate_schema_version": certificate.CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": "CE3", "experiment_design_version": DESIGN_VERSION,
        "reference_configuration": config.REFERENCE.to_dict(),
        "provenance": provenance.build_provenance(config_id="toy", probe_seed=7),
        "results": {"mean": 0.1, "inference": stats.p_value_report(1e-9)},
        "claim_scope": "toy",
    }

def test_certificate_roundtrips_and_validates(tmp_path):
    p = certificate.write_certificate(_valid_cert(), tmp_path / "audit/cryptography/evidence_current" / "c.json", repo_root=tmp_path)
    loaded = json.loads(p.read_text())
    certificate.validate_certificate(loaded)
    assert loaded["reference_configuration"]["depth"] == 10

def test_certificate_rejects_missing_fields():
    c = _valid_cert(); del c["provenance"]
    with pytest.raises(certificate.CertificateSchemaError, match="missing"):
        certificate.validate_certificate(c)

def test_certificate_rejects_stale_n_folds():
    c = _valid_cert(); c["results"]["n_folds"] = 5
    with pytest.raises(certificate.CertificateSchemaError, match="n_folds"):
        certificate.validate_certificate(c)

def test_certificate_rejects_nan_and_inf():
    for bad in (float("nan"), float("inf")):
        c = _valid_cert(); c["results"]["mean"] = bad
        with pytest.raises(certificate.CertificateSchemaError, match="non-finite"):
            certificate.validate_certificate(c)

def test_certificate_rejects_literal_zero_p_value():
    c = _valid_cert(); c["results"]["inference"] = {"p_value": 0.0, "p_underflow": False}
    with pytest.raises(certificate.CertificateSchemaError, match="literal number 0.0"):
        certificate.validate_certificate(c)

def test_certificate_rejects_wrong_reference_configuration():
    c = _valid_cert(); c["reference_configuration"]["depth"] = 5
    with pytest.raises(certificate.CertificateSchemaError, match="reference_configuration"):
        certificate.validate_certificate(c)

def test_provenance_separates_probe_seed_from_dataset_randomness():
    prov = provenance.build_provenance(config_id="x", probe_seed=42)
    sp = prov["seed_policy"]
    assert sp["probe_cv_seed"] == 42
    assert sp["dataset_generation_randomness"] == "os.urandom"
    assert sp["exact_dataset_replay_available"] is False
