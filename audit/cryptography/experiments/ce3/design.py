"""
CE3: representation decodability, replicate-level inference.

STRUCTURE  20 independent evaluation replicates x 5-fold CV within each.
UNIT       one stabilized selectivity value per INDEPENDENT REPLICATE.

CV folds are a variance-reduction device nested inside a replicate, never
independent experimental units: treating 20x5 as 100 observations would be
pseudo-replication. The historical `supported_threshold=0.20` /
`inconclusive_threshold=0.05` parameters are NOT part of the decision rule
and are absent here.

The calibration arm is a METHODOLOGICAL POSITIVE CONTROL validating the
probing pipeline. It is not direct cryptographic evidence for the primary
target.
"""

from __future__ import annotations

DECISION_RULE = (
    "CE3 decision rule: (1) the calibration positive control must validate the probing "
    "pipeline; (2) the primary replicate-level mean selectivity must be positive; (3) the "
    "primary replicate-level p-value must satisfy the predeclared multiplicity-corrected "
    "threshold. No additional practical-magnitude threshold is applied."
)


def aggregate_replicates(selectivity_replicates, *, n_splits_per_replicate: int,
                         calibration_validated: bool, alpha: float | None = None,
                         calibration_p_value: float | None = None) -> dict:
    """
    Replicate-level aggregation under the FROZEN fixed-sequence design.

    MULTIPLICITY: pre-ordered hypotheses (calibration gate -> primary) are
    each tested at FULL alpha = 0.05 with strong FWER control and no
    adjustment, stopping at the first non-rejection. The historical
    Bonferroni alpha = 0.025 is unnecessarily conservative under both the
    gate reading and the confirmatory reading and is superseded.

    The statistical unit is the independent replicate; the CV splits
    nested inside a replicate are a variance-reduction device and are
    never treated as independent observations.
    """
    from audit.cryptography.frozen_design import CE3 as CE3_SPEC
    from audit.cryptography.frozen_design import CE3_DECISION_RULE
    from audit.cryptography.statistics import replicate_level_summary

    alpha = CE3_SPEC.alpha if alpha is None else alpha
    if n_splits_per_replicate != CE3_SPEC.n_splits_per_replicate:
        raise ValueError(
            f"frozen design fixes n_splits_per_replicate="
            f"{CE3_SPEC.n_splits_per_replicate}; got {n_splits_per_replicate}")

    summary = replicate_level_summary(selectivity_replicates, alpha=alpha)
    wil = summary.get("wilcoxon_signed_rank") or {}
    p = wil.get("p_value")
    positive = summary["mean"] > 0

    # --- STEP 1: calibration gate (fixed sequence stops here on failure) ---
    gate_passed = bool(calibration_validated) and (
        calibration_p_value is None or calibration_p_value < alpha)
    if not gate_passed:
        decision = "INCONCLUSIVE"
        reason = ("fixed-sequence step 1 failed: the calibration positive control did not "
                  "validate the probing pipeline, so the primary hypothesis is not tested")
        primary_tested = False
    else:
        primary_tested = True
        if positive and p is not None and p < alpha:
            decision, reason = "SUPPORTED", (
                "fixed-sequence step 1 passed; mean replicate-level selectivity positive "
                f"and p < alpha={alpha}")
        elif not positive:
            decision, reason = "NOT_SUPPORTED", "mean replicate-level selectivity is not positive"
        else:
            decision, reason = "INCONCLUSIVE", f"p does not meet alpha={alpha}"

    return {
        "design_specification": CE3_SPEC.to_dict(),
        "decision": decision,
        "decision_reason": reason,
        "decision_rule": CE3_DECISION_RULE,
        "statistical_unit": CE3_SPEC.statistical_unit,
        "n_replicates": summary["n_replicates"],
        "n_splits_per_replicate": int(n_splits_per_replicate),
        "alpha": alpha,
        "multiplicity": {
            "procedure": CE3_SPEC.multiplicity,
            "family_alpha": alpha,
            "adjustment": "none (fixed-sequence controls FWER at full alpha)",
            "sequence": ["calibration_gate", "primary"],
            "stopped_at_first_non_rejection": not primary_tested,
        },
        "calibration": {
            "validated": bool(calibration_validated),
            "p_value": calibration_p_value,
            "gate_passed": gate_passed,
            "role": ("methodological positive control for the probing pipeline; NOT direct "
                     "cryptographic evidence for the primary target"),
        },
        "primary_tested": primary_tested,
        "selectivity_replicates": summary["raw_replicate_values"],
        "mean_selectivity": summary["mean"],
        "ci95": summary["ci95_t"],
        "ci95_bootstrap": summary["ci95_bootstrap"],
        "effect_size_dz": summary["effect_size_cohens_dz"],
        "inference": summary["wilcoxon_signed_rank"],
    }
