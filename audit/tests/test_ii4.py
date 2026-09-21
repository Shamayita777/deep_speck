import math
from pathlib import Path

import numpy as np
import pytest

from audit.implementation.conformance import BlockOutcome, ConformanceFactor
from audit.implementation.ii4_analysis import (
    EquivalenceVerdict,
    MaterialityVerdict,
    analyse_ii4,
    exact_sign_flip_p,
)
from audit.implementation.ii4_design import (
    GOHR_DEPTH_FACTOR,
    GOHR_FIXED_PROTOCOL,
    II4_DESIGN,
    II4Design,
)
from audit.implementation.ii4_experiment import (
    build_gohr_ii4_plan,
    collect_valid_differences,
    new_block,
)


# --- frozen design values -------------------------------------------

def test_frozen_design_values():
    assert II4_DESIGN.n_blocks == 10
    assert II4_DESIGN.delta_materiality == 0.01
    assert II4_DESIGN.epochs == 200
    assert II4_DESIGN.alpha == 0.05
    assert II4_DESIGN.primary_endpoint == "sealed_confirmatory_test_accuracy"
    assert II4_DESIGN.evaluation_rule == "terminal_epoch"


def test_frozen_pairing_semantics_are_asymmetric():
    assert II4_DESIGN.same_dataset is True
    assert II4_DESIGN.same_model_initialization is False
    assert II4_DESIGN.same_training_shuffle_stream is False
    assert II4_DESIGN.same_evaluation_data is True


def test_design_refuses_same_model_initialization():
    with pytest.raises(ValueError, match="same_model_initialization"):
        II4Design(
            n_blocks=10, min_valid_blocks=9,
            primary_endpoint="sealed_confirmatory_test_accuracy",
            evaluation_rule="terminal_epoch", epochs=200,
            same_dataset=True, same_model_initialization=True,
            same_training_shuffle_stream=False, same_evaluation_data=True,
            alpha=0.05, delta_materiality=0.01, delta_is_equivalence_margin=True,
        )


def test_k10_supports_exact_sign_flip():
    assert II4_DESIGN.sign_flip_feasible is True
    assert 2 / (2 ** 10) < 0.05


def test_gohr_protocol_is_reference_conformant():
    assert GOHR_FIXED_PROTOCOL["shuffle"] is True
    assert GOHR_FIXED_PROTOCOL["epochs"] == 200
    assert GOHR_FIXED_PROTOCOL["num_rounds"] == 5
    assert GOHR_FIXED_PROTOCOL["differential"] == (0x0040, 0x0000)


# --- conformance factor ---------------------------------------------

def test_conformance_factor_requires_a_real_discrepancy():
    with pytest.raises(ValueError, match="no discrepancy"):
        ConformanceFactor(name="p", declared_value=10, realized_value=10,
                          declaration_source="src", verification_method="probe")


def test_conformance_factor_requires_verification_method():
    with pytest.raises(ValueError, match="verification_method"):
        ConformanceFactor(name="p", declared_value=10, realized_value=5,
                          declaration_source="src", verification_method="  ")


def test_gohr_factor_records_declared_10_realized_5():
    assert GOHR_DEPTH_FACTOR.declared_value == 10
    assert GOHR_DEPTH_FACTOR.realized_value == 5
    assert GOHR_DEPTH_FACTOR.arm_values == {"declared": 10, "realized": 5}
    assert "Add" in GOHR_DEPTH_FACTOR.verification_method


# --- block invalidation ---------------------------------------------

def test_failed_arm_invalidates_whole_block():
    b = BlockOutcome(block_id="block0", dataset_hash="h")
    b.declared_arm_value = 0.93
    b.record_failure(arm="realized", reason="nan", detail="loss NaN", seed=7)
    assert b.is_valid is False
    assert b.signed_difference is None


def test_surviving_arm_is_not_retained_in_the_analysis_set():
    good = BlockOutcome("block0", "h0"); good.declared_arm_value = 0.93; good.realized_arm_value = 0.61
    half = BlockOutcome("block1", "h1"); half.declared_arm_value = 0.93
    half.record_failure(arm="realized", reason="oom", detail="", seed=1)
    diffs = collect_valid_differences([good, half])
    assert len(diffs) == 1
    assert diffs[0] == pytest.approx(0.32)


def test_signed_difference_is_declared_minus_realized():
    b = BlockOutcome("b", "h"); b.declared_arm_value = 0.90; b.realized_arm_value = 0.60
    assert b.signed_difference == pytest.approx(0.30)


# --- seeds ------------------------------------------------------------

def test_arms_never_share_a_seed(tmp_path):
    plan = build_gohr_ii4_plan(output_dir=tmp_path)
    for i in range(plan.design.n_blocks):
        assert plan.seeds.seed_for(i, "declared") != plan.seeds.seed_for(i, "realized")
    all_seeds = [plan.seeds.seed_for(i, a)
                 for i in range(plan.design.n_blocks) for a in ("declared", "realized")]
    assert len(set(all_seeds)) == len(all_seeds) == 20


def test_plan_validates_clean(tmp_path):
    assert build_gohr_ii4_plan(output_dir=tmp_path).validate() == []


def test_plan_preregistration_hash_is_stable(tmp_path):
    p1 = build_gohr_ii4_plan(output_dir=tmp_path, experiment_id="X")
    p2 = build_gohr_ii4_plan(output_dir=tmp_path, experiment_id="X")
    # created_at differs, so hashes differ; the DESIGN portion must not.
    assert p1.preregistration()["design"] == p2.preregistration()["design"]


def test_new_block_carries_predetermined_seeds(tmp_path):
    plan = build_gohr_ii4_plan(output_dir=tmp_path)
    b = new_block(3, "hash3", plan)
    assert b.declared_seed == plan.seeds.seed_for(3, "declared")
    assert b.realized_seed == plan.seeds.seed_for(3, "realized")


# --- materiality on the SIGNED CI ------------------------------------

def _analyse(diffs, delta=0.01, n_req=10, min_valid=9):
    return analyse_ii4(diffs, n_blocks_requested=n_req,
                       min_valid_blocks=min_valid, delta=delta, alpha=0.05)


def test_materially_higher_requires_ci_lower_bound_above_delta():
    r = _analyse([0.30] * 9 + [0.31])          # huge, tight
    assert r.materiality is MaterialityVerdict.MATERIALLY_HIGHER
    assert r.ci95_low > 0.01


def test_materially_lower_requires_ci_upper_bound_below_minus_delta():
    r = _analyse([-0.30] * 9 + [-0.31])
    assert r.materiality is MaterialityVerdict.MATERIALLY_LOWER
    assert r.ci95_high < -0.01


def test_significant_but_below_delta_is_not_material():
    """Statistically significant yet practically small must NOT be 'material'."""
    r = _analyse([0.002] * 10)
    assert r.p_value is not None and r.p_value < 0.05
    assert r.materiality is MaterialityVerdict.NO_DIRECTIONAL_CLAIM


def test_point_estimate_above_delta_but_wide_ci_is_not_material():
    """A point-estimate rule would call this material; the signed-CI rule must not."""
    rng = np.random.default_rng(0)
    diffs = list(0.012 + rng.normal(0, 0.05, size=10))
    r = _analyse(diffs)
    assert r.mean_difference > 0.01          # point estimate exceeds delta
    assert r.ci95_low < 0.01                 # but the CI does not clear it
    assert r.materiality is MaterialityVerdict.NO_DIRECTIONAL_CLAIM


def test_materiality_uses_signed_not_absolute_ci():
    """Direction must be preserved: a large negative effect is LOWER, not HIGHER."""
    r = _analyse([-0.30] * 10)
    assert r.materiality is MaterialityVerdict.MATERIALLY_LOWER
    assert r.materiality is not MaterialityVerdict.MATERIALLY_HIGHER


# --- equivalence, separate from non-significance ----------------------

def test_equivalence_requires_tost_not_mere_non_significance():
    """Noisy data centred on zero: non-significant, but NOT equivalent."""
    rng = np.random.default_rng(1)
    diffs = list(rng.normal(0, 0.05, size=10))
    r = _analyse(diffs)
    assert r.p_value > 0.05                                  # difference test not significant
    assert r.equivalence is not EquivalenceVerdict.EQUIVALENT_WITHIN_MARGIN
    assert r.equivalence is EquivalenceVerdict.INCONCLUSIVE


def test_tight_near_zero_yields_equivalence():
    r = _analyse([0.0005, -0.0003, 0.0002, 0.0004, -0.0001,
                  0.0003, 0.0000, 0.0002, -0.0002, 0.0001])
    assert r.equivalence is EquivalenceVerdict.EQUIVALENT_WITHIN_MARGIN
    assert r.tost_p < 0.05


def test_large_effect_is_not_equivalent():
    r = _analyse([0.30] * 9 + [0.31])
    assert r.equivalence is EquivalenceVerdict.NOT_EQUIVALENT


def test_materiality_and_equivalence_are_computed_independently():
    r = _analyse([0.30] * 9 + [0.31])
    d = r.to_dict()["confirmatory"]
    assert "p_tost" in d["equivalence"] and d["equivalence"]["p_tost"] is not None
    assert d["materiality"]["verdict"] != d["equivalence"]["verdict"]
    assert "never inferred from a non-significant" in d["equivalence"]["note"]


# --- replication sufficiency ------------------------------------------

def test_insufficient_blocks_is_inconclusive_not_a_verdict():
    r = _analyse([0.30] * 5, min_valid=9)     # huge effect, too few blocks
    assert r.sufficient is False
    assert r.materiality is MaterialityVerdict.INSUFFICIENT_BLOCKS
    assert r.equivalence is EquivalenceVerdict.NOT_ASSESSED
    assert r.mean_difference is None


def test_nine_valid_blocks_still_sufficient():
    assert _analyse([0.30] * 9, min_valid=9).sufficient is True


# --- sign-flip secondary ---------------------------------------------

def test_sign_flip_exact_p_bounds():
    p = exact_sign_flip_p(np.array([0.3] * 10))
    assert p == pytest.approx(2 / 1024)


def test_sign_flip_reported_infeasible_below_k6():
    r = analyse_ii4([0.30, 0.31, 0.29, 0.30, 0.32],
                    n_blocks_requested=5, min_valid_blocks=5, delta=0.01)
    assert r.sign_flip_feasible is False
    assert r.sign_flip_p is None
    assert any("cannot reach alpha" in w for w in r.warnings)


def test_sign_flip_feasible_at_k10():
    r = _analyse([0.30] * 10)
    assert r.sign_flip_feasible is True
    assert r.sign_flip_p is not None


# --- schema separation -------------------------------------------------

def test_output_schema_separates_confirmatory_from_secondary():
    d = _analyse([0.30] * 10).to_dict()
    assert "confirmatory" in d and "secondary_robustness" in d
    assert "materiality" in d["confirmatory"] and "equivalence" in d["confirmatory"]
    assert d["confirmatory"]["experimental_unit"] == "dataset block"


def test_no_automatic_pass_field_anywhere():
    d = _analyse([0.30] * 10).to_dict()
    flat = str(d).upper()
    assert "'PASS'" not in flat and '"PASS"' not in flat


# --- frozen decisions of the final pass ---------------------------------

def test_global_shared_confirmatory_test_set_is_frozen():
    assert II4_DESIGN.confirmatory_test_set == "GLOBAL_SHARED"
    assert II4_DESIGN.confirmatory_data_mode == "SHARED"


def test_design_rejects_per_block_test_sets():
    with pytest.raises(ValueError, match="GLOBAL"):
        II4Design(
            n_blocks=10, min_valid_blocks=9,
            primary_endpoint="sealed_confirmatory_test_accuracy",
            evaluation_rule="terminal_epoch", epochs=200,
            same_dataset=True, same_model_initialization=False,
            same_training_shuffle_stream=False, same_evaluation_data=True,
            alpha=0.05, delta_materiality=0.01, delta_is_equivalence_margin=True,
            confirmatory_test_set="PER_BLOCK", confirmatory_data_mode="PER_REPLICATE",
        )


def test_base_seed_frozen_at_700000():
    from audit.implementation.ii4_design import FROZEN_BASE_SEED
    assert FROZEN_BASE_SEED == 700000


def test_reference_reg_param_in_frozen_protocol():
    assert GOHR_FIXED_PROTOCOL["reg_param"] == 1e-5
    assert (GOHR_FIXED_PROTOCOL["lr_period"], GOHR_FIXED_PROTOCOL["lr_high"],
            GOHR_FIXED_PROTOCOL["lr_low"]) == (10, 0.002, 0.0001)


def test_historical_hash_is_full_and_bound_to_bundle():
    full = "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea"
    assert full in GOHR_DEPTH_FACTOR.verification_method
    assert "de630afc..." not in GOHR_DEPTH_FACTOR.verification_method
    assert "evidence_bundle" in GOHR_DEPTH_FACTOR.verification_method


def test_preregistration_states_conditional_inference(tmp_path):
    prereg = build_gohr_ii4_plan(output_dir=tmp_path).preregistration()
    assert "CONDITIONAL ON THE FIXED SEALED EVALUATION SET" in prereg["conditional_inference"]
    assert prereg["confirmatory_test_set"]["per_block_test_sets"] is False


def test_production_plan_validates_against_frozen_spec(tmp_path):
    assert build_gohr_ii4_plan(output_dir=tmp_path).validate_production() == []


def test_production_validation_rejects_changed_seed(tmp_path):
    p = build_gohr_ii4_plan(output_dir=tmp_path, base_seed=1)
    assert any("base_seed" in x for x in p.validate_production())


def test_config_fingerprint_ignores_creation_time(tmp_path):
    a = build_gohr_ii4_plan(output_dir=tmp_path)
    b = build_gohr_ii4_plan(output_dir=tmp_path)
    assert a.config_fingerprint() == b.config_fingerprint()
