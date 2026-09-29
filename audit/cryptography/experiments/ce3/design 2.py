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
                         calibration_validated: bool, corrected_alpha: float,
                         alpha: float = 0.05) -> dict:
    """Replicate-level aggregation plus the explicit, singular decision rule."""
    from audit.cryptography.statistics import replicate_level_summary

    summary = replicate_level_summary(selectivity_replicates, alpha=alpha)
    wil = summary.get("wilcoxon_signed_rank") or {}
    p = wil.get("p_value")
    positive = summary["mean"] > 0
    p_ok = (p is not None and p < corrected_alpha)
    if not calibration_validated:
        decision, reason = "INCONCLUSIVE", "calibration positive control did not validate"
    elif positive and p_ok:
        decision, reason = "SUPPORTED", "calibration validated; mean selectivity positive; p < corrected alpha"
    elif not positive:
        decision, reason = "NOT_SUPPORTED", "mean replicate-level selectivity is not positive"
    else:
        decision, reason = "INCONCLUSIVE", "p does not meet the predeclared corrected threshold"
    return {
        "decision": decision,
        "decision_reason": reason,
        "decision_rule": DECISION_RULE,
        "statistical_unit": "independent evaluation replicate",
        "n_replicates": summary["n_replicates"],
        "n_splits_per_replicate": int(n_splits_per_replicate),
        "corrected_alpha": corrected_alpha,
        "calibration": {"validated": bool(calibration_validated),
                        "role": ("methodological positive control for the probing pipeline; "
                                 "NOT direct cryptographic evidence for the primary target")},
        "selectivity_replicates": summary["raw_replicate_values"],
        "mean_selectivity": summary["mean"],
        "ci95": summary["ci95_t"],
        "ci95_bootstrap": summary["ci95_bootstrap"],
        "effect_size_dz": summary["effect_size_cohens_dz"],
        "inference": summary["wilcoxon_signed_rank"],
    }
