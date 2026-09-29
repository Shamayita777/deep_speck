"""Final pre-production gate: F2 re-freeze, F3 seeding order, F4 preflight, F13."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from audit.cryptography import certificate
from audit.cryptography.experiments.ce1 import design as ce1
from audit.cryptography.experiments.ce1 import gohr_signal_destruction as ce1_driver
from audit.cryptography.frozen_design import CE1 as CE1_SPEC
from audit.cryptography.frozen_design import DESIGN_SPECIFICATION_VERSION


# ---------------- F2: null matches the procedure ----------------

def test_null_is_the_sharp_null_not_the_weak_null():
    h = CE1_SPEC.primary_hypothesis
    assert "sharp" in h.lower() and "exchangeable" in h.lower()
    assert h != "H0: E[Delta] = 0 (two-sided)"
    assert "implies E[Delta] = 0" in h          # estimand unchanged


def test_exactness_assumption_is_stated():
    a = CE1_SPEC.exactness_assumption
    assert "2^K" in a and "EXACT" in a
    assert "coin flip" in a          # amendment -03: randomization, not assumption


def test_rejection_interpretation_does_not_overclaim():
    r = CE1_SPEC.rejection_interpretation
    assert "SOME effect" in r
    assert "does NOT by itself license the weak-null" in r


def test_design_version_records_the_refreeze():
    assert DESIGN_SPECIFICATION_VERSION == "CE-frozen-design-2026-03"


def test_procedure_alpha_and_k_unchanged_by_refreeze():
    assert CE1_SPEC.primary_test == "exact_paired_sign_flip"
    assert CE1_SPEC.alpha == 0.05 and CE1_SPEC.n_blocks == 8
    assert CE1_SPEC.min_valid_blocks == 6 and CE1_SPEC.equivalence_enabled is False


def test_analysis_output_carries_the_null_and_assumption():
    rng = np.random.default_rng(0)
    r = ce1.paired_block_analysis(0.93 + rng.normal(0, 3e-4, 8),
                                  0.50 + rng.normal(0, 3e-4, 8))
    pt = r["paired_difference"]["primary_test"]
    assert "sharp" in pt["null_hypothesis"].lower()
    assert pt["exactness_assumption"] and pt["rejection_interpretation"]


def test_certificate_rejects_the_superseded_weak_null():
    from audit.cryptography.audit_config import REFERENCE
    from audit.cryptography import provenance
    cert = {
        "certificate_schema_version": certificate.CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": "CE1-SIGNAL-DESTRUCTION",
        "experiment_design_version": provenance.EXPERIMENT_DESIGN_VERSION,
        "reference_configuration": REFERENCE.to_dict(),
        "provenance": provenance.build_provenance(config_id="t"),
        "results": {"primary_test": {"null_hypothesis": "H0: E[Delta] = 0 (two-sided)"}},
        "claim_scope": "t",
    }
    with pytest.raises(certificate.CertificateSchemaError, match="superseded weak null"):
        certificate.validate_certificate(cert)


# ---------------- F3: seed applied before model initialisation ----------------

def test_seed_is_set_before_model_construction_in_source():
    import inspect
    src = inspect.getsource(ce1_driver.build_production_components)
    assert src.index("GohrTrainer.set_seed(seed)") < src.index("model = GohrModel(")


@pytest.mark.skipif(not (ROOT / "gohr" / "model.py").exists(), reason="needs gohr")
def test_same_seed_gives_identical_initial_weights():
    pytest.importorskip("tensorflow")
    from audit.cryptography.gohr.model import GohrModel
    from audit.cryptography.gohr.trainer import GohrTrainer

    def init(seed):
        GohrTrainer.set_seed(seed)
        return GohrModel(depth=1, regularization=1e-5).build().get_weights()

    a, b, c = init(4242), init(4242), init(4243)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))       # deterministic
    assert any(not np.array_equal(x, y) for x, y in zip(a, c))   # seed-sensitive


def test_arms_never_share_a_model_training_seed():
    m = ce1_driver.seed_manifest(8)
    for blk in m["seeds"].values():
        assert blk["baseline"] != blk["destroyed"]
    assert m["model_training_seeds_pairwise_distinct"] is True
    assert m["n_model_training_seeds"] == 16


# ---------------- F13: honest seed semantics ----------------

def test_seed_manifest_scopes_its_reproducibility_claim():
    m = ce1_driver.seed_manifest(8)
    assert "does NOT make the generated Speck datasets replayable" in m["reproducibility_scope"]
    assert "not claimed distinct" in m["distinctness_claim"]


# ---------------- F4: preflight enforces the frozen design ----------------

def _preflight(tmp_path, extra):
    return ce1_driver.main(["--preflight", "--output",
                            str(tmp_path / "evidence_current/ce1/c.json"),
                            "--repo-root", str(tmp_path), *extra])


def test_preflight_ok_with_frozen_values(tmp_path, capsys):
    assert _preflight(tmp_path, []) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PREFLIGHT_OK"


@pytest.mark.parametrize("flag,value", [
    ("--seed-base", "999"), ("--train-samples", "1000"),
    ("--validation-samples", "1000"), ("--evaluation-samples", "1000"),
    ("--n-blocks", "10"), ("--min-valid-blocks", "4"),
])
def test_preflight_refuses_overridden_scientific_parameter(tmp_path, capsys, flag, value):
    rc = _preflight(tmp_path, [flag, value])
    out = capsys.readouterr().out
    assert rc == 1
    assert "PREFLIGHT FAILED" in out and "frozen scientific parameters were overridden" in out
    assert "PREFLIGHT_OK" not in out


def test_preflight_and_production_share_one_validator():
    import inspect
    src = inspect.getsource(ce1_driver.main)
    assert src.count("validate_frozen_parameters(") >= 2       # preflight AND production


def test_operational_parameters_remain_configurable(tmp_path, capsys):
    assert _preflight(tmp_path, []) == 0                        # output/repo-root vary freely
    assert json.loads(capsys.readouterr().out)["output_path"].endswith("c.json")


def test_preflight_reports_the_frozen_values_it_enforced(tmp_path, capsys):
    _preflight(tmp_path, [])
    fp = json.loads(capsys.readouterr().out)["frozen_parameters"]
    assert fp["n_blocks"] == 8 and fp["min_valid_blocks"] == 6
    assert fp["seed_base"] == CE1_SPEC.seed_base
    assert fp["evaluation_samples"] == CE1_SPEC.evaluation_samples
    assert fp["checkpoint_rule"] == "FINAL_EPOCH" and fp["terminal_epoch"] == 200


# ============ Randomized arm assignment (freeze amendment -03) ============

def test_arm_assignment_is_actually_randomized():
    """Not deterministic parity: both assignment outcomes must occur."""
    m = ce1_driver.seed_manifest(8)
    flips = [b["assignment_flip"] for b in m["seeds"].values()]
    assert set(flips) == {0, 1}
    for b in m["seeds"].values():
        assert {b["baseline"], b["destroyed"]} == {b["seed_A"], b["seed_B"]}


def test_intact_arm_is_not_confounded_with_seed_parity():
    """The old design always gave intact the EVEN seed - a confound."""
    m = ce1_driver.seed_manifest(8)
    parities = {b["baseline"] % 2 for b in m["seeds"].values()}
    assert parities == {0, 1}


def test_assignment_is_reproducible_and_prospectively_declared():
    a, b = ce1_driver.seed_manifest(8), ce1_driver.seed_manifest(8)
    assert a == b
    assert a["assignment_seed"] == CE1_SPEC.assignment_seed
    assert "randomization" in a and "coin flip" in a["randomization"]


def test_assignment_seed_is_separate_from_model_seeds():
    m = ce1_driver.seed_manifest(8)
    model_seeds = {b[k] for b in m["seeds"].values() for k in ("seed_A", "seed_B")}
    assert CE1_SPEC.assignment_seed not in model_seeds


def test_different_assignment_seed_gives_a_different_randomization():
    a = ce1_driver.seed_manifest(8)
    c = ce1_driver.seed_manifest(8, assignment_seed=CE1_SPEC.assignment_seed + 1)
    assert [x["assignment_flip"] for x in a["seeds"].values()] != \
           [x["assignment_flip"] for x in c["seeds"].values()]


def test_frozen_design_records_the_randomization():
    assert CE1_SPEC.arm_assignment_randomized is True
    assert DESIGN_SPECIFICATION_VERSION == "CE-frozen-design-2026-03"
    a = CE1_SPEC.exactness_assumption
    assert "coin flip" in a and "probability 1/2" in a
    assert "EXACT by DESIGN, not by assumption" in a


def test_analysis_records_the_randomization_basis():
    rng = np.random.default_rng(0)
    r = ce1.paired_block_analysis(0.93 + rng.normal(0, 3e-4, 8),
                                  0.50 + rng.normal(0, 3e-4, 8))
    pt = r["paired_difference"]["primary_test"]
    assert pt["arm_assignment_randomized"] is True
    assert "randomization distribution" in pt["randomization_basis"]


def test_certificate_refuses_signflip_without_recorded_randomization():
    from audit.cryptography import provenance
    from audit.cryptography.audit_config import REFERENCE
    cert = {
        "certificate_schema_version": certificate.CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": "CE1-SIGNAL-DESTRUCTION",
        "experiment_design_version": provenance.EXPERIMENT_DESIGN_VERSION,
        "reference_configuration": REFERENCE.to_dict(),
        "provenance": provenance.build_provenance(config_id="t"),
        "results": {"primary_test": {"test": "exact_paired_sign_flip",
                                     "arm_assignment_randomized": False}},
        "claim_scope": "t",
    }
    with pytest.raises(certificate.CertificateSchemaError, match="arm assignment was randomized"):
        certificate.validate_certificate(cert)


def test_production_run_uses_the_randomized_assignment(tmp_path):
    """Production code path, not a mock: seeds reach arms per the coin flip."""
    from audit.cryptography import sealed_dataset as sds
    sealed = sds.prepare_sealed_evaluation_set(
        tmp_path / "sealed",
        generate_fn=lambda n: (np.zeros((n, 64), np.uint8), np.zeros(n, np.uint8)),
        rounds=5, differential=(0x0040, 0x0000), n_samples=8)
    seen = []

    def data_fn(i):
        r = np.random.default_rng(i)
        data_fn.sealed_sha256 = sealed["sha256"]
        return (r.integers(0, 2, (8, 8), np.uint8), r.integers(0, 2, 8, np.uint8),
                sealed["X"], sealed["Y"])
    data_fn.sealed_sha256 = sealed["sha256"]

    def spy(X, Y, Xe, Ye, *, seed, arm, block_id):
        seen.append((block_id, arm, seed))
        return 0.9 if arm == "baseline" else 0.5

    ce1_driver.run(n_blocks=3, min_valid_blocks=2, production=False,
                   output_path=tmp_path / "evidence_current/ce1/c.json",
                   train_eval_fn=spy, data_fn=data_fn, repo_root=tmp_path)
    m = ce1_driver.seed_manifest(3)
    for bid, arm, seed in seen:
        assert seed == m["seeds"][bid]["baseline" if arm == "baseline" else "destroyed"]
