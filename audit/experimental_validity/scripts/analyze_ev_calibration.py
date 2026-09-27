#!/usr/bin/env python3
"""
Analyse CALIBRATION (pilot) certificates into a variance artifact for
prospective power planning.

Emits NUISANCE VARIANCE ONLY. No mean difference, no p-value, no effect
size, no conclusion: the pilot characterises noise, and letting an
observed pilot effect escape into planning is how a confirmatory design
becomes contaminated. epsilon is never derived here.

Reports, per hypothesis: n_valid, sigma_Delta (SD of the paired
differences), variance, and a ONE-SIDED 95% UPPER CONFIDENCE BOUND for
sigma from the chi-square result:

    sigma_UCL = s * sqrt(df / chi2_{0.05, df})

The upper bound, not the point estimate, sizes the production design. A
~10-pair pilot leaves sigma materially uncertain, and sizing on the point
estimate under-powers the experiment roughly half the time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scipy import stats  # noqa: E402

from audit.common.provenance import sha256_file, utc_timestamp  # noqa: E402
from audit.common.strict_json import dumps_strict  # noqa: E402

EXPECTED_EXPERIMENTS = ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")
SIGMA_UCL_LEVEL = 0.95
#: Frozen calibration design: exactly this many valid pairs per hypothesis.
REQUIRED_CALIBRATION_PAIRS = 10
#: Config each calibration certificate must have been produced from.
CALIBRATION_CONFIGS = {
    "H-EV-SHUFFLE": "configs/gohr_ev_shuffle_calibration.yaml",
    "H-EV-REPRESENTATION": "configs/gohr_ev_representation_calibration.yaml",
}


def _expected_config_hash(experiment_id: str):
    """config_hash of the frozen calibration config for this hypothesis."""
    import yaml
    from framework.provenance import config_hash
    path = REPO / CALIBRATION_CONFIGS[experiment_id]
    if not path.is_file():
        raise CalibrationInputError(f"frozen calibration config missing: {path}")
    return config_hash(yaml.safe_load(path.read_text())), yaml.safe_load(path.read_text())


class CalibrationInputError(RuntimeError):
    pass


def sigma_upper_confidence_bound(sd: float, n: int, level: float = SIGMA_UCL_LEVEL) -> float:
    """One-sided upper (1-alpha) confidence bound for sigma (chi-square)."""
    df = n - 1
    if df < 1:
        raise CalibrationInputError("need n >= 2 for a variance bound.")
    return float(sd * math.sqrt(df / stats.chi2.ppf(1 - level, df)))


def _require_calibration(cert: dict, path: Path) -> None:
    required = {"run_mode": "calibration", "calibration": True, "non_evidentiary": True}
    for key, expected in required.items():
        if key not in cert:
            raise CalibrationInputError(
                f"{path}: missing required status field {key!r}; refusing (fail-closed).")
        if cert[key] != expected:
            raise CalibrationInputError(
                f"{path}: {key}={cert[key]!r}, required {expected!r}. Variance for power "
                "planning must come from the calibration tier, and confirmatory data must "
                "never be consumed here.")


def _paired_differences(cert: dict, path: Path) -> list:
    raw = cert.get("raw_paired_values")
    if not raw:
        raise CalibrationInputError(
            f"{path}: no raw_paired_values. sigma_Delta is defined only for the PAIRED primary "
            "hypotheses; single-arm pilots cannot supply it.")
    a, b = raw.get("condition_a"), raw.get("condition_b")
    if not isinstance(a, list) or not isinstance(b, list):
        raise CalibrationInputError(f"{path}: paired values are malformed.")
    if len(a) != len(b):
        raise CalibrationInputError(
            f"{path}: paired arrays differ in length ({len(a)} vs {len(b)}).")
    for name, arr in (("condition_a", a), ("condition_b", b)):
        for v in arr:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise CalibrationInputError(f"{path}: {name} contains a non-finite value {v!r}.")
    return [float(y) - float(x) for x, y in zip(a, b)]


def summarise(path: Path) -> dict:
    if not path.is_file():
        raise CalibrationInputError(f"certificate not found: {path}")
    cert = json.loads(path.read_text())
    _require_calibration(cert, path)
    experiment = (cert.get("audit") or {}).get("id") or cert.get("experiment_id")
    if experiment not in EXPECTED_EXPERIMENTS:
        raise CalibrationInputError(
            f"{path}: experiment_id {experiment!r} is not one of the primary family "
            f"{EXPECTED_EXPERIMENTS}.")
    # --- provenance: the certificate must come from the FROZEN calibration
    # --- configuration, not from a hand-constructed minimal object.
    expected_hash, cfg = _expected_config_hash(experiment)
    recorded_hash = ((cert.get("provenance") or {}).get("config_hash")
                     or cert.get("config_hash"))
    if recorded_hash is None:
        raise CalibrationInputError(
            f"{path}: no config_hash recorded. Only certificates produced by the frozen "
            "calibration configurations are accepted; a hand-constructed minimal certificate "
            "cannot supply variance for a confirmatory design.")
    if recorded_hash != expected_hash:
        raise CalibrationInputError(
            f"{path}: config_hash {recorded_hash} does not match the frozen calibration config "
            f"{CALIBRATION_CONFIGS[experiment]} ({expected_hash}).")

    prov = cert.get("provenance") or {}
    plan_hash = prov.get("statistical_plan_sha256")
    actual_plan = sha256_file(REPO / "docs" / "statistical_plan.md")
    if plan_hash is not None and plan_hash != actual_plan:
        raise CalibrationInputError(
            f"{path}: statistical_plan_sha256 {plan_hash} does not match the current plan "
            f"{actual_plan}; the pilot was run under a different statistical plan.")
    if not prov.get("source_manifest_sha256") and not prov.get("protocol_manifest_sha256"):
        raise CalibrationInputError(
            f"{path}: no source/protocol manifest hash recorded; the code that produced this "
            "pilot cannot be identified.")
    if experiment == "H-EV-REPRESENTATION":
        perm = prov.get("candidate1_permutation_sha256") or cert.get("permutation_hash")
        expected_perm = cfg.get("expected_candidate1_permutation_sha256")
        if perm != expected_perm:
            raise CalibrationInputError(
                f"{path}: Candidate-1 permutation {perm} does not match the frozen "
                f"{expected_perm}; the pilot characterised a different intervention.")

    diffs = _paired_differences(cert, path)
    n = len(diffs)
    if n != REQUIRED_CALIBRATION_PAIRS:
        raise CalibrationInputError(
            f"{path}: the frozen calibration design is exactly "
            f"{REQUIRED_CALIBRATION_PAIRS} valid pairs per hypothesis; got {n}. A short pilot "
            "changes the sigma precision the production sizing depends on.")
    sd = statistics.stdev(diffs)
    return {
        "experiment_id": experiment,
        "certificate_path": str(path),
        "certificate_sha256": sha256_file(path),
        "run_mode": cert["run_mode"],
        "n_valid": n,
        "sigma_Delta": sd,
        "variance": sd ** 2,
        "sigma_Delta_upper_95": sigma_upper_confidence_bound(sd, n),
        "quantity": "sigma_Delta: SD of the paired difference; powers the paired primary test",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("certificates", nargs="+", type=Path)
    ap.add_argument("--output", type=Path,
                    default=Path("results/calibration/ev_calibration_variance.json"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit fewer than both primary hypotheses (planning will then refuse)")
    args = ap.parse_args()

    try:
        entries = [summarise(p) for p in args.certificates]
        seen = [e["experiment_id"] for e in entries]
        dupes = {x for x in seen if seen.count(x) > 1}
        if dupes:
            raise CalibrationInputError(f"duplicate calibration input for {sorted(dupes)}.")
        missing = [x for x in EXPECTED_EXPERIMENTS if x not in seen]
        if missing and not args.allow_partial:
            raise CalibrationInputError(
                f"missing calibration input for {missing}. Both primary hypotheses need their "
                "own sigma_Delta; pass --allow-partial only for inspection.")
    except CalibrationInputError as exc:
        print(f"REFUSED: {exc}")
        return 1

    artifact = {
        "artifact": "ev-calibration-variance-v2",
        "non_evidentiary": True,
        "generated_at_utc": utc_timestamp(),
        "purpose": ("nuisance-variance estimates for prospective power analysis only; "
                    "contains no effect estimate, no p-value and no conclusion"),
        "sigma_ucl_level": SIGMA_UCL_LEVEL,
        "inputs": entries,
        "sigma_Delta_estimates": {e["experiment_id"]: e["sigma_Delta"] for e in entries},
        "sigma_Delta_upper_95": {e["experiment_id"]: e["sigma_Delta_upper_95"] for e in entries},
        "prohibition": ("These values may size an experiment. They may NOT be used to choose or "
                        "validate epsilon, to justify an assumed effect, or as evidence."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps_strict(artifact, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    for e in entries:
        print(f"  {e['experiment_id']:22} n={e['n_valid']:<3} "
              f"sigma_Delta={e['sigma_Delta']:.6g}  UCL95={e['sigma_Delta_upper_95']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
