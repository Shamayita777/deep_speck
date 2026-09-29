"""
CE2: observational association with the analytical single-trail quantity.

TARGET (narrow, explicit)
    Lipmaa-Moriai closed-form modular-addition XOR differential probability,
    chained along each sample's realized round-by-round trajectory under the
    declared Markov/independence assumption, interpreted as a SINGLE-TRAIL
    quantity.

It is NOT the exact differential probability of the cipher, NOT the full
differential effect, and NOT "what the network learned".

CE2 is OBSERVATIONAL. It cannot establish causal use of the target.
Uncertainty is reported at the RUN level: samples within a run are not
independent experimental replicates of the model.
"""

from __future__ import annotations

import numpy as np

ANALYTICAL_TARGET = (
    "Lipmaa-Moriai closed-form modular-addition XOR differential probability, chained "
    "along each sample's realized round-by-round trajectory under the declared "
    "Markov/independence assumption; a SINGLE-TRAIL quantity - not the cipher's exact "
    "differential probability, not the full differential effect, and not 'what the "
    "network learned'"
)


def run_level_association(runs, *, alpha: float = 0.05) -> dict:
    """
    Aggregate independent runs.

    `runs` is an iterable of dicts with 'run_id', 'rho', 'n'. The primary
    quantity is the run-level rho; the sample-level p-value inside a run
    describes association within that run only, and concatenating samples
    across runs would not create additional model-level replicates.
    """
    from audit.cryptography.statistics import replicate_level_summary

    runs = list(runs)
    if not runs:
        raise ValueError("CE2 requires at least one run")
    rhos = [float(r["rho"]) for r in runs]
    out = {
        "analytical_target": ANALYTICAL_TARGET,
        "design": "observational",
        "causal_interpretation_permitted": False,
        "n_runs": len(runs),
        "per_run": [{"run_id": r["run_id"], "rho": float(r["rho"]), "n": int(r["n"])}
                    for r in runs],
        "primary_effect_size": "run-level Spearman rho",
    }
    if len(rhos) >= 2:
        summary = replicate_level_summary(rhos, alpha=alpha)
        summary["statistical_unit"] = "independent run (one Spearman rho per run)"
        summary["n_runs"] = summary.pop("n_replicates")
        out["run_level_inference"] = summary
    else:
        out["run_level_inference"] = {
            "statistical_unit": "independent run",
            "n_runs": 1,
            "note": "a single run supports no run-level uncertainty estimate",
        }
    return out
