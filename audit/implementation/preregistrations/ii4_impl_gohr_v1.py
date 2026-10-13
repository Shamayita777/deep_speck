#!/usr/bin/env python3
"""
PREREGISTRATION - II-4-GOHR-IMPL-V1
Formal reference-versus-independent implementation comparison.

Frozen BEFORE any block of this experiment is executed. Running
`python3 -m audit.implementation.preregistrations.ii4_impl_gohr_v1`
validates the plan and prints the hash that the runner must be given as
--frozen-hash; the runner refuses a production execution whose
configuration does not reproduce it.

WHAT THIS EXPERIMENT IS FOR
---------------------------
Methodology Section 3.5.4 Stage 4 requires that "both implementations
are evaluated using identical experimental protocols", and Section
3.5.3's null hypothesis is stated over "independent implementations".
Implementation certificate V6 records Stage II-4 and II-5 as PARTIAL and
the pillar decision as INCONCLUSIVE precisely because no such comparison
exists: the completed II-4-GOHR-DEPTH-V1 experiment trained both of its
arms with the SAME implementation, so it answers whether the depth
discrepancy matters, not whether two implementations agree.

This experiment supplies the missing comparison. It is the only kind of
run that `stages.validate_stage_record` permits to carry Stage II-4 to
COMPLETED.

WHAT IS NOT RE-RUN
------------------
II-2, II-3 and II-4-GOHR-DEPTH-V1 are complete and their evidence is
read-only input here. Nothing in this plan regenerates, re-evaluates or
supersedes them. The depth experiment's bound artifacts, certificates
V1-V6 and the version registry are untouched; this run writes only into
its own fresh output directory.

PREDICTION, RECORDED IN ADVANCE
-------------------------------
IMPLEMENTATIONS_EQUIVALENT. Basis: the II-2 independent run reached
fresh-test accuracy 0.9291 against the reference-reported 0.9291, and
II-2's M1 verification established exact deterministic agreement of the
cipher, key schedule, encoding, dataset construction and architecture.
If this prediction is wrong it STAYS wrong in the record, as the A1
preregistration's incorrect NOT_CAUGHT prediction does.

A wrong prediction here would be a substantive finding, not a failure of
the audit: it would mean two implementations that agree exactly on every
deterministic component nevertheless diverge materially once training is
involved, which is a result about the reproducibility of neural
cryptanalysis rather than about either implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from audit.common.outcomes import ExecutionMode
from audit.implementation.adapters.gohr_impl_comparison import (
    ARM_INDEPENDENT,
    ARM_REFERENCE,
    EXPECTED_COUNTS,
    GOHR_PROTOCOL,
    SHARED_COMPONENTS,
)
from audit.implementation.comparison import (
    ArmSeedManifest,
    AxisKind,
    ComparisonArm,
    ComparisonAxis,
)
from audit.implementation.arm_recovery import RECOVERY_EXACTNESS
from audit.implementation.ii4_comparison_design import ComparisonDesign, ComparisonPlan
from audit.implementation.preflight import TRAINING_TOPOLOGY

EXPERIMENT_ID = "II-4-GOHR-IMPL-V1"

#: Distinct from the depth experiment's 700_000 so that no (block, arm)
#: seed is shared between the two experiments.
FROZEN_BASE_SEED = 800_000

# =====================================================================
# DESIGN
# =====================================================================

DESIGN = ComparisonDesign(
    n_blocks=7,
    min_valid_blocks=6,
    max_replacement_blocks=2,
    primary_endpoint="sealed_confirmatory_test_accuracy",
    evaluation_rule="terminal_epoch",
    epochs=GOHR_PROTOCOL["epochs"],
    same_dataset=True,
    same_model_initialization=False,
    same_training_shuffle_stream=False,
    same_evaluation_data=True,
    alpha=0.05,
    delta_materiality=0.01,
    delta_is_equivalence_margin=True,
)

# =====================================================================
# DESIGN RATIONALE - every number below was computed, not asserted
# =====================================================================

DESIGN_RATIONALE = {
    "reference_variance": {
        "source": "II-4-GOHR-DEPTH-V1",
        "between_block_sd_of_differences": 3.16683e-4,
        "derivation": (
            "Recovered from that experiment's own PUBLISHED 95% CI rather than taken from "
            "prose: half-width 2.26541e-4 / t(9)=2.262157 = se 1.00144e-4; "
            "se * sqrt(10) = sd 3.16683e-4. Agrees with the 3.2e-4 quoted in certificate V6."),
        "caveat": (
            "This is the SD of a conformance comparison within one implementation. The SD of a "
            "between-implementation comparison is unknown and could be larger, which is why "
            "the power below is reported as a curve over SD multipliers rather than at a "
            "single assumed value."),
    },
    "power": {
        "method": (
            "Monte Carlo over the COMPLETE frozen decision rule - the conjunction of a TOST "
            "rejection and a signed-95%-CI no-directional-claim - by drawing block differences "
            "and running audit.implementation.ii4_analysis.analyse_ii4 itself. TOST was NOT "
            "powered in isolation; the Experimental Validity planner was corrected for exactly "
            "that error and the mistake is not repeated here."),
        "simulations": 10000,
        "true_delta": 0.0,
        "equivalence_power_by_sd_multiplier": {
            "K=6": {"1x": 1.000, "3x": 1.000, "10x": 1.000, "20x": 0.904,
                    "30x": 0.471, "40x": 0.189, "60x": 0.034},
            "K=7": {"1x": 1.000, "10x": 1.000, "20x": 0.958, "30x": 0.597, "40x": 0.257},
            "K=8": {"1x": 1.000, "10x": 1.000, "20x": 0.985, "30x": 0.709, "40x": 0.342},
        },
        "validates_v6_claim": (
            "Certificate V6's proposal stated that TOST power at margin 0.01 is approximately "
            "1 at K=6 even at 10x the depth-run SD. CONFIRMED: 1.000 at 10x. The simulation "
            "additionally establishes the boundary the proposal did not state - power holds to "
            "about 20x and collapses by 30x, so the design tolerates roughly twenty times the "
            "spread previously measured and no more."),
        "specificity_at_the_margin_boundary": {
            "K=6": 0.0505, "K=7": 0.0478,
            "note": ("With the true difference set to exactly the margin the design declares "
                     "equivalence at approximately alpha, as a correctly calibrated TOST must. "
                     "This checks the rule as IMPLEMENTED, not as described."),
        },
    },
    "replication_and_failure_tolerance": {
        "finding": (
            "K=7 with min_valid_blocks=6 tolerates ONE lost block. The per-arm failure "
            "probability is not zero merely because no failure has been observed: 0 failures "
            "in 21 observed arm-trainings (20 from II-4-GOHR-DEPTH-V1's ten valid blocks, 1 "
            "from the II-2 independent run) gives an exact one-sided 95% upper bound of 0.133 "
            "by Clopper-Pearson. At that bound, P(at least 6 valid blocks from 7) = 0.449 - a "
            "design more likely than not to return INCONCLUSIVE for want of surviving blocks."),
        "p_enough_valid_blocks": {
            "K=7,min=6": {"0%": 1.000, "2%": 0.971, "5%": 0.856, "10%": 0.604, "@0.133": 0.449},
            "K=9,min=6": {"0%": 1.000, "2%": 1.000, "5%": 0.992, "10%": 0.927, "@0.133": 0.838},
        },
        "resolution": (
            "The planned design stays K=7 with min_valid_blocks=6 as specified, and the "
            "shortfall is addressed by a frozen budget of TWO replacement blocks (effective "
            "maximum 9 blocks, matching the K=9,min=6 row above). A replacement is admissible "
            "only for a technical failure occurring BEFORE any sealed-test evaluation of that "
            "block, so the block replaced has no endpoint value and the decision cannot depend "
            "on one. The budget is frozen here and the runner cannot raise it during "
            "execution, so this is not outcome-adaptive sample-size inflation."),
        "independence_caveat": (
            "Arm failures are modelled as independent. A systematic cause - out-of-memory at "
            "batch 5000, a driver fault, a session or quota expiry - fails arms together, so "
            "every survival probability above is an UPPER bound."),
    },
    "sign_flip_secondary": {
        "min_attainable_two_sided_p_at_6_valid_blocks": 2 / (2 ** 6),
        "feasible_at_min_valid": (2 / (2 ** 6)) < DESIGN.alpha,
        "rationale": (
            "6 valid blocks is the smallest K at which the exact paired sign-flip test can "
            "reach two-sided p < 0.05 (2/2^6 = 0.03125). Feasibility is stated at "
            "min_valid_blocks rather than at n_blocks: a secondary test that resolves only if "
            "every block survives is not a feasible secondary test."),
    },
    "compute_estimate": {
        "basis": (
            "Measured, not estimated from first principles: the depth-10 arms of "
            "II-4-GOHR-DEPTH-V1 took 3.7-4.6 h per arm on 2x Tesla T4, and the II-2 "
            "independent 200-epoch run took 4.5 h."),
        "per_block_hours": "2 arms x 3.7-4.6 h + ~0.15 h data generation and evaluation",
        "planned_K7": {"gpu_hours_low": 52.9, "gpu_hours_high": 65.5, "midpoint": 59.2},
        "worst_case_K9_both_replacements_used": {
            "gpu_hours_low": 68.1, "gpu_hours_high": 84.3, "midpoint": 76.2},
        "storage_measured": {
            "basis": ("MEASURED, not estimated: 10^5 real samples from reference "
                      "speck.make_train_data give X as uint8 (n, 64), 6.50 MB in memory and "
                      "1.03 MB as a compressed npz. An earlier estimate of 2.8-3.6 GB was "
                      "wrong by roughly 25x because it ignored how well uint8 compresses."),
            "per_block_compressed_gb": 0.11,
            "nine_blocks_gb": 1.0,
            "sealed_test_gb": 0.01,
            "checkpoints_gb": 0.13,
            "total_estimate_gb": 1.2,
            "recommended_free_gb": 5.0,
        },
        "checkpoint_overhead_measured": {
            "basis": ("MEASURED on the production architecture (102,497 params) after a real "
                      "training step, so the optimizer slots are populated: 8 consecutive "
                      "two-slot commits, median 0.333 s, max 0.360 s per epoch."),
            "seconds_per_epoch_median": 0.333,
            "minutes_per_arm_200_epochs": 1.1,
            "gpu_hours_K7": 0.26,
            "gpu_hours_K9": 0.33,
            "share_of_total_runtime": "under 0.5 per cent; it does not move the estimate",
            "slot_size_mb": 1.59,
            "disk_two_slots_two_arms_nine_blocks_mb": 57.4,
            "disk_all_checkpoints_mb": 114.8,
        },
        "memory_measured": {
            "basis": ("X_train is uint8 on disk (640 MB at 10^7 x 64) but is cast to float32 "
                      "for training, which is 2.56 GB, plus 0.26 GB for validation."),
            "peak_host_gb_estimate": 4.0,
            "recommended_free_host_gb": 8.0,
        },
    },
    "gpu_utilisation": dict(TRAINING_TOPOLOGY),
    "interruption_recovery": dict(RECOVERY_EXACTNESS),
    "wall_clock_note": (
        "The GPU-hour figures are also the WALL-CLOCK figures. The runner trains arms and "
        "blocks sequentially on one device and declares no distribution strategy, so a "
        "two-GPU machine does not finish sooner; the second GPU idles throughout. Any "
        "schedule built on an assumed 2x speedup is wrong. See gpu_utilisation."),
    "what_this_experiment_cannot_establish": [
        "Training-seed reproducibility of either implementation: each arm is trained once per "
        "block from a single recorded seed, so between-seed variance within an arm is not "
        "estimated.",
        "Equivalence of the DATA GENERATORS: generation is shared, by design, and its "
        "equivalence rests on the II-2 M1 deterministic verification rather than on this run.",
        "Anything about the paper-prose architecture variant: both arms follow the public "
        "supplementary code path, which includes BatchNormalization after the second dense "
        "layer. The prose variant is not built and not tested.",
        "Generalisation beyond the single sealed confirmatory test set, on which all inference "
        "here is conditional.",
        "That the reference arm is a byte-identical execution of the 2019 code: the reference "
        "files are unmodified, but they are driven by an external loop that supplies block "
        "data and a seed, because the reference's own entry point generates its data "
        "internally and a paired block design requires shared data.",
        "That the independent arm's seeding is its own code: train_independent.train() is "
        "bypassed for the same reason, so its seeding block is reproduced rather than invoked.",
        "Bit-exact continuation of an interrupted arm on a TensorFlow or Keras version other "
        "than the one the exactness test was measured on. On TF 2.20.0 / Keras 3.15.1 a "
        "reseeded resume reproduces the uninterrupted run's batch sequence and final weights "
        "bit-for-bit (measured); the test re-measures this, so another stack fails the suite "
        "rather than invalidating the claim silently.",
    ],
}

# =====================================================================
# AXIS
# =====================================================================

AXIS = ComparisonAxis(
    kind=AxisKind.IMPLEMENTATION,
    name="gohr_reference_vs_independent_implementation",
    arm_a=ComparisonArm(
        arm_id=ARM_REFERENCE,
        label="original Gohr reference implementation (CRYPTO 2019 supplementary code)",
        identity={"implementation": "reference", "builder": "train_nets.make_resnet",
                  "depth": GOHR_PROTOCOL["depth"], "reg_param": GOHR_PROTOCOL["reg_param"]},
        source_paths=(
            "audit/implementation/reference/train_nets.py",
            "audit/implementation/reference/train_5_rounds.py",
        ),
        provenance_note=(
            "Hash-verified in II-1 and never modified. Loaded by file path; make_resnet, "
            "cyclic_lr and set_seed are the reference's own."),
    ),
    arm_b=ComparisonArm(
        arm_id=ARM_INDEPENDENT,
        label="II-2 independent reimplementation",
        identity={"implementation": "independent", "builder": "model_independent.build_gohr_model",
                  "depth": GOHR_PROTOCOL["depth"], "reg_param": GOHR_PROTOCOL["reg_param"]},
        source_paths=(
            "audit/implementation/ii2_independent/model_independent.py",
            "audit/implementation/ii2_independent/train_independent.py",
            "audit/implementation/ii2_independent/speck_independent.py",
            "audit/implementation/ii2_independent/dataset_independent.py",
        ),
        provenance_note=(
            "The implementation assessed as COMPLETED under Stage II-2 in certificate V6, "
            "bound there by source hash; written without reference to the reference model, "
            "cipher or dataset generator."),
    ),
    scientific_question=(
        "Under identical experimental protocols, do the reference and independent "
        "implementations reach the same sealed-test accuracy when training is replicated "
        "across dataset blocks?"),
    verification_method=(
        "Both implementations are pinned by file sha256 in the run's source manifest. Their "
        "deterministic components were verified to agree exactly in II-2 (M1: published test "
        "vector, key schedule over 1000 keys, round traces over 10000 word states, binary "
        "encoding, dataset construction, architecture; M7: differential theory by an "
        "independent carry-difference automaton). Before any training, the runner verifies "
        "that both arms construct models of identical structure "
        f"({EXPECTED_COUNTS['conv1d']} Conv1D, {EXPECTED_COUNTS['add']} Add, "
        f"{EXPECTED_COUNTS['parameters']:,} parameters, and identical ordered layer "
        "configurations, activations, padding, strides, BatchNormalization settings, "
        "regularizer configuration and weight shapes) and aborts otherwise. The ordered "
        "signature of both implementations was measured and found identical; see "
        "adapters/gohr_declared_architecture.json."),
    finding_id="II-FINDING-DEPTH-V3",
)

PREDICTION = {
    "hypothesis": "H-II4-IMPL",
    "null": ("independent implementations consistent with the published methodology produce "
             "statistically equivalent experimental outcomes (methodology Section 3.5.3)"),
    "predicted_conclusion": "IMPLEMENTATIONS_EQUIVALENT",
    "basis": ("The II-2 independent run reached fresh-test accuracy 0.9291 against the "
              "reference-reported 0.9291, and M1 established exact deterministic agreement on "
              "every non-stochastic component."),
    "falsified_if": ("the signed 95% CI of the paired difference lies wholly beyond "
                     f"+/-{DESIGN.delta_materiality}, giving IMPLEMENTATIONS_DIFFER"),
    "immutability": ("This prediction is never revised in response to results. A wrong "
                     "prediction is retained and reported as wrong."),
}


def build_plan(output_dir: Path, *, mode: ExecutionMode = ExecutionMode.PRODUCTION,
               experiment_id: str = EXPERIMENT_ID,
               design: ComparisonDesign = DESIGN) -> ComparisonPlan:
    """Construct (never execute) the frozen plan."""
    return ComparisonPlan(
        experiment_id=experiment_id,
        design=design,
        axis=AXIS,
        fixed_protocol={**GOHR_PROTOCOL, "epochs": design.epochs},
        seeds=ArmSeedManifest(base_seed=FROZEN_BASE_SEED,
                              n_blocks=design.max_total_blocks,
                              arm_ids=AXIS.arm_ids),
        output_dir=Path(output_dir),
        shared_components=dict(SHARED_COMPONENTS),
        design_rationale={**DESIGN_RATIONALE, "prediction": PREDICTION},
        execution_mode=mode,
        created_at_utc="FROZEN",
    )


def freeze() -> int:
    """
    Validate the plan and print the hash downstream tools must pin.

    `created_at_utc` is the literal string FROZEN so that the printed
    preregistration hash is reproducible by anyone re-running this
    module, rather than changing with the wall clock.
    """
    plan = build_plan(Path("audit/implementation/runs") / EXPERIMENT_ID.lower())
    problems = plan.validate_production()
    if problems:
        print("REFUSED: this preregistration does not validate as a production plan:\n")
        for p in problems:
            print(f"    {p}")
        return 1

    d = plan.design
    print(f"{EXPERIMENT_ID} - READY TO FREEZE\n")
    print(f"  axis kind              {plan.axis.kind.value}")
    print(f"  satisfies Stage II-4   {plan.axis.satisfies_stage_ii4}")
    print(f"  component kind         {plan.axis.component_kind}")
    print(f"  arms                   {plan.axis.arm_ids[0]} vs {plan.axis.arm_ids[1]}")
    print(f"  sign convention        {plan.axis.sign_convention}")
    print(f"  blocks                 K={d.n_blocks}, min valid {d.min_valid_blocks}, "
          f"up to {d.max_replacement_blocks} replacements (max {d.max_total_blocks})")
    print(f"  epochs / batch         {d.epochs} / {GOHR_PROTOCOL['batch_size']}")
    print(f"  alpha / margin         {d.alpha} / {d.delta_materiality}")
    print(f"  sign-flip at min valid {d.sign_flip_feasible_at_min_valid} "
          f"(min p = {2 / (2 ** d.min_valid_blocks):.5f})")
    print(f"  base seed              {FROZEN_BASE_SEED}")
    print(f"  prediction             {PREDICTION['predicted_conclusion']}")
    est = DESIGN_RATIONALE["compute_estimate"]
    print(f"\n  compute (K=7)          {est['planned_K7']['gpu_hours_low']}-"
          f"{est['planned_K7']['gpu_hours_high']} GPU-h "
          f"(midpoint {est['planned_K7']['midpoint']})")
    print(f"  compute (K=9 worst)    "
          f"{est['worst_case_K9_both_replacements_used']['gpu_hours_low']}-"
          f"{est['worst_case_K9_both_replacements_used']['gpu_hours_high']} GPU-h")
    print(f"\n  design hash            {d.preregistration_hash()}")
    print(f"  preregistration hash   {plan.preregistration_hash()}")
    print(f"  config fingerprint     {plan.config_fingerprint()}")
    print("\nCommit this file unchanged, then pass the config fingerprint to the runner as")
    print("--frozen-hash. The runner refuses a production execution whose configuration does")
    print("not reproduce it.")
    return 0


if __name__ == "__main__":
    sys.exit(freeze())
