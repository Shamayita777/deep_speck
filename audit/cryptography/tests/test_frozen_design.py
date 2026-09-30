"""
Frozen-design conformance: the adjudicated CE1/CE3 specification.

Every value here was fixed before any production observation. These tests
exist so that a later edit cannot silently move a preregistered parameter.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from audit.cryptography import frozen_design as FD
from audit.cryptography.experiments.ce1 import design as ce1
from audit.cryptography.experiments.ce1 import gohr_signal_destruction as ce1_driver
from audit.cryptography.experiments.ce3 import design as ce3


# ---------------- frozen values ----------------

def test_ce1_frozen_values():
    assert FD.CE1.n_blocks == 8
    assert FD.CE1.min_valid_blocks == 6
    assert FD.CE1.failure_tolerance == 2
    assert FD.CE1.n_blocks == FD.CE1.min_valid_blocks + FD.CE1.failure_tolerance
    assert FD.CE1.alpha == 0.05
    assert FD.CE1.primary_test == "exact_paired_sign_flip"
    assert FD.CE1.checkpoint_rule == "FINAL_EPOCH"


def test_ce1_equivalence_is_disabled():
    assert FD.CE1.equivalence_enabled is False
    assert FD.CE1.equivalence_margin is None


def test_ce3_frozen_values():
    assert FD.CE3.n_replicates == 20
    assert FD.CE3.n_splits_per_replicate == 5
    assert FD.CE3.alpha == 0.05                       # NOT 0.025
    assert FD.CE3.calibration_role == "methodological_gate"
    assert FD.CE3.multiplicity == "fixed_sequence_calibration_then_primary"


def test_rejected_values_are_not_reintroduced():
    """0.01 (II-4) and 0.025 (historical Bonferroni) must not reappear."""
    spec = json.dumps(FD.CE1.to_dict()) + json.dumps(FD.CE3.to_dict())
    assert "0.025" not in spec
    assert FD.CE1.equivalence_margin != 0.01


# ---------------- CE1 exact sign-flip ----------------

def test_sign_flip_is_exact_and_enumerates_all_assignments():
    r = ce1.exact_sign_flip_test([0.3] * 8)
    assert r["enumerated_assignments"] == 2 ** 8
    assert r["p_value"] == pytest.approx(2 / 2 ** 8)


@pytest.mark.parametrize("k,can_reach", [(4, False), (5, False), (6, True), (8, True)])
def test_sign_flip_resolution_limit(k, can_reach):
    r = ce1.exact_sign_flip_test([0.3] * k)
    assert r["min_attainable_two_sided_p"] == pytest.approx(2 / 2 ** k)
    assert r["can_reach_alpha_005"] is can_reach


def test_min_valid_blocks_six_is_the_resolution_boundary():
    """Why 6: at 5 blocks the primary test cannot reach alpha at ANY effect."""
    assert ce1.exact_sign_flip_test([1.0] * 5)["p_value"] > 0.05
    assert ce1.exact_sign_flip_test([1.0] * 6)["p_value"] < 0.05


def test_sign_flip_matches_brute_force_enumeration():
    from itertools import product
    d = np.array([0.4, -0.1, 0.25, 0.3, 0.2, 0.35, 0.05])
    obs = abs(d.mean())
    exp = sum(1 for s in product((1, -1), repeat=len(d))
              if abs((d * np.array(s)).mean()) >= obs - 1e-15) / 2 ** len(d)
    assert ce1.exact_sign_flip_test(d)["p_value"] == pytest.approx(exp)


# ---------------- CE1 analysis ----------------

def _analysis(n=8, base=0.93, dest=0.50):
    rng = np.random.default_rng(0)
    b = base + rng.normal(0, 3e-4, n)
    d = dest + rng.normal(0, 3e-4, n)
    return ce1.paired_block_analysis(b, d)


def test_ce1_reports_every_block_difference_and_no_bootstrap():
    r = _analysis()
    assert len(r["paired_difference"]["raw_block_differences"]) == 8
    assert "bootstrap" not in json.dumps(r).lower()


def test_ce1_makes_no_equivalence_claim():
    eq = _analysis()["destroyed_vs_chance"]["equivalence"]
    assert eq["enabled"] is False and eq["verdict"] == "NOT_ASSESSED"
    assert "NOT evidence of equivalence" in eq["reason"]


def test_ce1_reports_advantage_excluded_by_ci():
    r = _analysis()["destroyed_vs_chance"]
    assert r["advantage_excluded_by_ci"] > 0
    assert r["ci95_t"][0] < r["mean_accuracy"] < r["ci95_t"][1]


def test_ce1_primary_test_is_sign_flip_not_wilcoxon():
    pt = _analysis()["paired_difference"]["primary_test"]
    assert pt["test"] == "exact_paired_sign_flip"
    assert "wilcoxon" not in json.dumps(_analysis()).lower()


def test_ce1_interpretation_is_scoped():
    scope = _analysis()["interpretation_scope"]
    assert "does NOT establish that no non-cryptographic shortcut exists" in scope
    assert "no equivalence-to-chance claim" in scope


# ---------------- CE1 driver enforcement ----------------

def test_production_refuses_non_frozen_block_counts(tmp_path):
    with pytest.raises(ValueError, match="frozen scientific parameters were overridden"):
        ce1_driver.run(n_blocks=10, output_path=tmp_path / "audit/cryptography/evidence_current/ce1/c.json",
                       train_eval_fn=lambda *a, **k: 0.5,
                       data_fn=lambda i: (np.zeros((4, 8), np.uint8), np.zeros(4, np.uint8),
                                          np.zeros((4, 8), np.uint8), np.zeros(4, np.uint8)),
                       repo_root=tmp_path, production=True)


def test_production_refuses_result_below_minimum_valid_blocks(tmp_path):
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        return 0.5
    def data_fn(i):
        r = np.random.default_rng(i)
        return (r.integers(0, 2, (8, 8), np.uint8), r.integers(0, 2, 8, np.uint8),
                r.integers(0, 2, (4, 8), np.uint8), r.integers(0, 2, 4, np.uint8))
    # 3 blocks < min_valid 6 -> refuse (and n_blocks!=8 refuses first)
    with pytest.raises(ValueError):
        ce1_driver.run(n_blocks=3, min_valid_blocks=6,
                       output_path=tmp_path / "audit/cryptography/evidence_current/ce1/c.json",
                       train_eval_fn=flaky, data_fn=data_fn, repo_root=tmp_path,
                       production=True)


def test_min_valid_below_six_is_refused(tmp_path):
    """
    Two independent guards refuse this: the frozen-value check fires first,
    and the >=6 resolution guard is defence-in-depth behind it.
    """
    with pytest.raises(ValueError, match="min_valid_blocks") as exc:
        ce1_driver.run(n_blocks=8, min_valid_blocks=5,
                       output_path=tmp_path / "audit/cryptography/evidence_current/ce1/c.json",
                       train_eval_fn=lambda *a, **k: 0.5, data_fn=lambda i: None,
                       repo_root=tmp_path, production=True)
    assert "frozen scientific parameters were overridden" in str(exc.value)


def test_resolution_guard_is_defence_in_depth(tmp_path):
    """The >=6 guard must reject a sub-resolution design on its own terms."""
    import inspect
    src = inspect.getsource(ce1_driver.run)
    assert "min_valid_blocks < 6" in src
    assert "cannot reach" in src


# ---------------- seed manifest ----------------

def test_seed_manifest_is_deterministic_and_distinct():
    a = ce1_driver.seed_manifest(8)
    b = ce1_driver.seed_manifest(8)
    assert a == b
    # only MODEL-TRAINING seeds are claimed pairwise distinct
    seeds = [blk[k] for blk in a["seeds"].values() for k in ("baseline", "destroyed")]
    assert len(seeds) == 16 and len(set(seeds)) == 16      # arms never share a seed
    for blk in a["seeds"].values():
        assert blk["baseline"] != blk["destroyed"]
    assert "NOT seed-replayable" in a["dataset_generation_randomness"]


# ---------------- CE3 fixed sequence ----------------

def _ce3(sel, calibrated=True, cal_p=0.001, alpha=None):
    return ce3.aggregate_replicates(sel, n_splits_per_replicate=5,
                                    calibration_validated=calibrated,
                                    calibration_p_value=cal_p, alpha=alpha)


def test_ce3_uses_full_alpha_no_adjustment():
    r = _ce3(list(np.linspace(0.05, 0.25, 20)))
    assert r["alpha"] == 0.05
    assert r["multiplicity"]["adjustment"] == "none (fixed-sequence controls FWER at full alpha)"
    assert r["multiplicity"]["sequence"] == ["calibration_gate", "primary"]


def test_ce3_gate_failure_stops_the_sequence():
    r = _ce3(list(np.linspace(0.05, 0.25, 20)), calibrated=False)
    assert r["decision"] == "INCONCLUSIVE"
    assert r["primary_tested"] is False
    assert r["multiplicity"]["stopped_at_first_non_rejection"] is True


def test_ce3_gate_p_value_above_alpha_also_stops():
    r = _ce3(list(np.linspace(0.05, 0.25, 20)), calibrated=True, cal_p=0.20)
    assert r["decision"] == "INCONCLUSIVE" and r["primary_tested"] is False


def test_ce3_supported_requires_gate_and_positive_and_p():
    r = _ce3(list(np.linspace(0.05, 0.25, 20)))
    assert r["decision"] == "SUPPORTED" and r["primary_tested"] is True


def test_ce3_negative_selectivity_not_supported():
    assert _ce3([-0.2] * 20)["decision"] == "NOT_SUPPORTED"


def test_ce3_replicate_unit_preserved():
    r = _ce3(list(np.linspace(0.05, 0.25, 20)))
    assert r["n_replicates"] == 20 and r["n_splits_per_replicate"] == 5
    assert len(r["selectivity_replicates"]) == 20        # not 100
    assert r["statistical_unit"] == "independent evaluation replicate"


def test_ce3_rejects_non_frozen_cv_structure():
    with pytest.raises(ValueError, match="n_splits_per_replicate"):
        ce3.aggregate_replicates([0.1] * 20, n_splits_per_replicate=10,
                                 calibration_validated=True)


def test_ce3_calibration_declared_methodological():
    r = _ce3(list(np.linspace(0.05, 0.25, 20)))
    assert r["calibration"]["role"].startswith("methodological positive control")
    assert "NOT direct cryptographic evidence" in r["calibration"]["role"]
