#!/usr/bin/env python3
"""
Prospective power planning for the EV primary family.

Plans BOTH primary hypotheses together and reports one COMMON production
replicate count K = max over four requirements:

    1. H-EV-SHUFFLE        difference detection
    2. H-EV-REPRESENTATION difference detection
    3. H-EV-SHUFFLE        TOST equivalence at true effect 0
    4. H-EV-REPRESENTATION TOST equivalence at true effect 0

Uses the EXISTING framework/power.py simulation, so the procedure
simulated is the procedure actually run.

MULTIPLICITY: the frozen plan applies Holm to the two difference-detection
p-values. Prospective planning uses alpha/2 per hypothesis as a
CONSERVATIVE BONFERRONI-STYLE APPROXIMATION. This is deliberately NOT
exact Holm power: Holm's step-down only tests at alpha/2 for the smaller
p-value and may test the larger at alpha, so planning at alpha/2 for both
is an upper bound on the required n, never an exact figure.

VARIANCE: sizing uses each hypothesis's one-sided 95% UPPER confidence
bound for sigma, not the point estimate; a ~10-pair pilot leaves sigma
uncertain enough that point-estimate sizing under-powers about half the
time.

Search is a deterministic sweep n = 2..MAX_N (no sparse candidate list),
each simulation independently and reproducibly seeded. Runtime/compute is
never an input: it cannot alter the statistically required K.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit.common.provenance import sha256_file, utc_timestamp  # noqa: E402
from audit.common.strict_json import dumps_strict  # noqa: E402
from framework.power import simulate_power  # noqa: E402
from framework.seeds import statistics_rng  # noqa: E402

PRIMARY_FAMILY = ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")

#: FROZEN design parameters. The planner refuses to produce a plan for any
#: other values: a power artifact computed under different assumptions than
#: the frozen design would silently authorise the wrong replicate count.
FROZEN = {
    "epsilon": 0.01,
    "target_effect": 0.01,
    "target_effect_source": "predeclared",
    "alpha": 0.05,
    "target_power": 0.80,
    "simulations": 50_000,
    "sigma_source": "upper95",
    "calibration_pairs_per_hypothesis": 10,
    "calibration_artifact": "ev-calibration-variance-v2",
}
DEFAULT_SIMULATIONS = FROZEN["simulations"]
DEFAULT_MAX_N = 200


class FrozenDesignViolation(RuntimeError):
    pass


def _require_frozen(args, art) -> None:
    """Fail closed on every frozen design parameter and on the artifact itself."""
    bad = []
    for name, expected in (("epsilon", FROZEN["epsilon"]),
                           ("target_effect", FROZEN["target_effect"]),
                           ("target_effect_source", FROZEN["target_effect_source"]),
                           ("alpha", FROZEN["alpha"]),
                           ("target_power", FROZEN["target_power"]),
                           ("simulations", FROZEN["simulations"])):
        actual = getattr(args, name)
        if actual != expected:
            bad.append(f"{name}={actual!r} (frozen {expected!r})")
    if args.use_sigma != FROZEN["sigma_source"]:
        bad.append(f"sigma_source={args.use_sigma!r} (frozen {FROZEN['sigma_source']!r})")

    if art.get("artifact") != FROZEN["calibration_artifact"]:
        bad.append(f"calibration artifact schema={art.get('artifact')!r} "
                   f"(required {FROZEN['calibration_artifact']!r})")
    if art.get("non_evidentiary") is not True:
        bad.append("calibration artifact is not marked non_evidentiary=true")

    ucl = art.get("sigma_Delta_upper_95") or {}
    inputs = {e.get("experiment_id"): e for e in art.get("inputs", [])}
    for hyp in PRIMARY_FAMILY:
        if hyp not in ucl:
            bad.append(f"no sigma_Delta_upper_95 for {hyp}")
        entry = inputs.get(hyp)
        if entry is None:
            bad.append(f"no calibration input recorded for {hyp}")
            continue
        if entry.get("n_valid") != FROZEN["calibration_pairs_per_hypothesis"]:
            bad.append(f"{hyp}: calibration n_valid={entry.get('n_valid')!r} "
                       f"(frozen {FROZEN['calibration_pairs_per_hypothesis']})")
        digest = entry.get("certificate_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            bad.append(f"{hyp}: missing/invalid calibration certificate hash")
    extra = set(inputs) - set(PRIMARY_FAMILY)
    if extra:
        bad.append(f"calibration artifact contains unexpected hypotheses {sorted(extra)}")
    if bad:
        raise FrozenDesignViolation(
            "refusing to plan power - frozen design violated: " + "; ".join(bad))
#: deterministic, distinct seed base per (hypothesis, procedure)
_SEED_BASE = {("H-EV-SHUFFLE", "difference"): 11_000_000,
              ("H-EV-SHUFFLE", "equivalence"): 12_000_000,
              ("H-EV-REPRESENTATION", "difference"): 13_000_000,
              ("H-EV-REPRESENTATION", "equivalence"): 14_000_000}


def _required_n(hypothesis, procedure, *, sigma, epsilon, target_effect, target_effect_source,
                alpha, target_power, simulations, max_n):
    """Smallest n in 2..max_n reaching target power. None if unreachable."""
    curve = []
    for n in range(2, max_n + 1):
        rng = statistics_rng(_SEED_BASE[(hypothesis, procedure)] + n)
        if procedure == "difference":
            res = simulate_power(
                procedure="difference_ttest", target_effect=target_effect,
                target_effect_source=target_effect_source, noise_sd=sigma,
                n_replicates=n, alpha=alpha, n_simulations=simulations, rng=rng)
        else:
            res = simulate_power(
                procedure="equivalence_tost", target_effect=0.0,
                target_effect_source="predeclared", noise_sd=sigma,
                n_replicates=n, alpha=alpha, epsilon=epsilon,
                n_simulations=simulations, rng=rng)
        curve.append({"n": n, "power": res.empirical_power})
        if res.empirical_power >= target_power:
            return n, res.empirical_power, curve
    return None, None, curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variance-artifact", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--target-effect", type=float, required=True)
    ap.add_argument("--target-effect-source", default="predeclared")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-power", type=float, default=0.80)
    ap.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    ap.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    # sigma_source is frozen at upper95; the option exists only so the
    # recorded plan states it explicitly. There is no "point" mode: sizing
    # on a point estimate from a 10-pair pilot under-powers roughly half
    # the time.
    ap.add_argument("--use-sigma", choices=("upper95",), default="upper95")
    ap.add_argument("--output", type=Path, default=Path("results/calibration/ev_power_plan.json"))
    args = ap.parse_args()

    if not args.variance_artifact.is_file():
        print(f"REFUSED: variance artifact not found: {args.variance_artifact}")
        return 1
    art = json.loads(args.variance_artifact.read_text())
    try:
        _require_frozen(args, art)
    except FrozenDesignViolation as exc:
        print(f"REFUSED: {exc}")
        return 1
    plan = build_power_plan(
        art, variance_artifact=args.variance_artifact, epsilon=args.epsilon,
        target_effect=args.target_effect, target_effect_source=args.target_effect_source,
        alpha=args.alpha, target_power=args.target_power, simulations=args.simulations,
        max_n=args.max_n, sigma_source=args.use_sigma,
    )
    requirements = plan["requirements"]
    common_n = plan["common_design"]["required_n"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps_strict(plan, indent=2, sort_keys=True))

    print(f"Wrote {args.output}")
    print(f"{'hypothesis':24} {'procedure':12} {'alpha':>7} {'sigma':>11} {'required_n':>11} {'power':>7}")
    for r in requirements:
        power_text = "-" if r["achieved_power"] is None else f"{r['achieved_power']:.3f}"
        n_text = "unreachable" if r["required_n"] is None else str(r["required_n"])
        print(f"{r['hypothesis']:24} {r['procedure']:12} {r['alpha_used']:>7.4f} "
              f"{r['sigma_used']:>11.4g} {n_text:>11} {power_text:>7}")
    print(f"COMMON K = {common_n if common_n is not None else 'UNREACHABLE within max_n'}")
    return 0


def build_power_plan(
    art: dict, *, variance_artifact: Path, epsilon: float, target_effect: float,
    target_effect_source: str, alpha: float, target_power: float, simulations: int,
    max_n: int, sigma_source: str = "upper95",
) -> dict:
    """
    Build the power plan from an already-validated variance artifact.

    SEPARATION OF CONCERNS: the FROZEN-parameter gate lives in main()
    (`_require_frozen`), which still demands exactly 50,000 simulations,
    epsilon=0.01, alpha=0.05, power=0.80, sigma_source=upper95 and the
    exact primary family for any PRODUCTION plan. This function contains
    only the planning LOGIC and takes the simulation budget as an
    argument, so unit tests can exercise the logic (common-K = max, both
    families at alpha/2, upper-95% sigma) at a small budget WITHOUT
    weakening the production gate. Calling it directly never produces a
    production-valid plan: main() is the only path that validates the
    frozen design.
    """
    sigmas = art["sigma_Delta_upper_95"]
    points = art.get("sigma_Delta_estimates", {})

    # BOTH families are planned at alpha/2 as a conservative
    # Bonferroni-style approximation to Holm. Difference detection is
    # Holm-corrected across the two primary hypotheses, and NOT_SUPPORTED
    # is likewise asserted across two hypotheses, so the equivalence
    # family carries its own Holm correction at analysis time and must be
    # powered for it. Neither figure is exact Holm power.
    alpha_diff = alpha_tost = alpha / 2
    requirements, curves = [], {}
    for hyp in PRIMARY_FAMILY:
        sigma = float(sigmas[hyp])
        for procedure, alpha_used in (("difference", alpha_diff), ("equivalence", alpha_tost)):
            n, power, curve = _required_n(
                hyp, procedure, sigma=sigma, epsilon=epsilon,
                target_effect=target_effect, target_effect_source=target_effect_source,
                alpha=alpha_used, target_power=target_power,
                simulations=simulations, max_n=max_n)
            requirements.append({
                "hypothesis": hyp, "procedure": procedure, "alpha_used": alpha_used,
                "sigma_used": sigma, "sigma_source": sigma_source,
                "sigma_point_estimate": points.get(hyp),
                "required_n": n, "achieved_power": power,
                "true_effect_assumed": (target_effect if procedure == "difference" else 0.0),
            })
            curves[f"{hyp}:{procedure}"] = curve

    unmet = [r for r in requirements if r["required_n"] is None]
    common_n = None if unmet else max(r["required_n"] for r in requirements)

    plan = {
        "artifact": "ev-power-plan-v2",
        "non_evidentiary": True,
        "generated_at_utc": utc_timestamp(),
        "primary_family": list(PRIMARY_FAMILY),
        "inputs": {
            "variance_artifact": str(variance_artifact),
            "variance_artifact_sha256": sha256_file(variance_artifact),
            "pilot_certificate_hashes": {e["experiment_id"]: e["certificate_sha256"]
                                         for e in art.get("inputs", [])},
            "calibration_sample_sizes": {e["experiment_id"]: e["n_valid"]
                                         for e in art.get("inputs", [])},
            "sigma_source": sigma_source,
            "sigma_upper_95": art.get("sigma_Delta_upper_95"),
            "sigma_point_estimates": points,
            "epsilon": epsilon,
            "target_effect": target_effect,
            "target_effect_source": target_effect_source,
            "alpha": alpha, "target_power": target_power,
            "n_simulations": simulations, "max_n_searched": max_n,
            "calibration_artifact_sha256": sha256_file(variance_artifact),
            "frozen_design": dict(FROZEN),
        },
        "multiplicity": {
            "method_at_analysis": "Holm across the two difference-detection p-values",
            "method_at_planning": "alpha/2 per hypothesis, for BOTH families",
            "equivalence_family": ("Holm across the two overall TOST p-values; no additional "
                                   "correction inside each TOST (intersection-union)"),
            "planning_caveat": ("CONSERVATIVE BONFERRONI-STYLE APPROXIMATION - this is NOT exact "
                                "Holm power. Holm tests the larger p-value at alpha, so planning "
                                "at alpha/2 for both yields an upper bound on required n."),
            "tost_excluded_from_holm": "TOST carries its own intersection-union error control",
        },
        "requirements": requirements,
        "common_design": ({"required_n": common_n,
                           "rule": "max over the four requirements",
                           "binding_requirement": max(
                               (r for r in requirements), key=lambda r: r["required_n"])
                           if common_n is not None else None}
                          if common_n is not None else
                          {"required_n": None,
                           "rule": "max over the four requirements",
                           "unmet": [{"hypothesis": r["hypothesis"], "procedure": r["procedure"]}
                                     for r in unmet]}),
        "power_curves": curves,
        "assumptions": [
            "paired design; experimental unit = matched pair",
            "sigma_Delta from the CALIBRATION tier only; no confirmatory data used",
            "epsilon is predeclared and was NOT derived or validated from pilot noise",
            "assumed true effect is predeclared, not observed",
            "runtime/compute is not an input and cannot alter the required K",
            ("prospective simulation assumes INDEPENDENT paired differences that are NORMALLY "
             "distributed with mean = target effect (0 for the equivalence scenario) and "
             "sigma = sigma_Delta_upper_95; departures from normality or independence change "
             "the required n"),
        ],
    }
    return plan


if __name__ == "__main__":
    sys.exit(main())
