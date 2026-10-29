"""
AutoND EV calibration: the paired-difference SD per primary hypothesis and its upper confidence bound.

Production is sized on the one-sided 95% upper confidence bound for sigma_Delta (chi-square):

    sigma_UCL = s * sqrt(df / chi2.ppf(1 - level, df)),   df = n - 1,   level = 0.95

The variance artifact carries no mean, interval, p-value or decision: pilot data size the production
design and nothing else.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from autond.baseline import PRIMARY_FAMILY, design_hash
from framework.provenance import sha256_file, utc_timestamp

VARIANCE_ARTIFACT = "autond-ev-calibration-variance-1"
SIGMA_UCL_LEVEL = 0.95


def sigma_upper_confidence_bound(sd: float, n: int, level: float = SIGMA_UCL_LEVEL) -> float:
    """One-sided upper `level` confidence bound for sigma from a sample SD of n values (chi-square)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"need an integer n >= 2 for a variance bound, got {n!r}")
    if isinstance(sd, bool) or not isinstance(sd, (int, float, np.floating)) or not math.isfinite(sd) or sd < 0:
        raise ValueError(f"sd must be a finite non-negative number, got {sd!r}")
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    df = int(n) - 1
    return float(sd * math.sqrt(df / stats.chi2.ppf(1 - level, df)))


def calibration_variance(paths: list[Path], config: dict) -> dict:
    """sigma_Delta and its one-sided 95% upper bound per paired hypothesis. No mean, CI, p or decision."""
    entries = {}
    for path in map(Path, paths):
        cert = json.loads(path.read_text())
        if (cert.get("run_mode"), cert.get("calibration"), cert.get("non_evidentiary")) != ("calibration", True, True):
            raise ValueError(f"{path}: not a calibration certificate; confirmatory data never enter planning")
        h = cert.get("experiment_id")
        if h not in PRIMARY_FAMILY or h in entries:
            raise ValueError(f"{path}: unexpected or duplicate hypothesis {h!r}")
        if cert.get("design_hash") != design_hash(config):
            raise ValueError(f"{path}: produced under a different design (protocol, decisions or calibration plan)")
        a, b = (np.asarray(cert["raw_paired_values"][k], dtype=float) for k in ("condition_a", "condition_b"))
        n = len(a)
        if n != config["calibration"][h]["units"] or len(b) != n:
            raise ValueError(f"{path}: {n} valid pairs; the calibration plan is exactly "
                             f"{config['calibration'][h]['units']}")
        sd = float(np.std(b - a, ddof=1))
        if not (np.isfinite(sd) and sd > 0):
            raise ValueError(f"{path}: degenerate paired-difference SD {sd}")
        entries[h] = {"certificate": str(path), "certificate_sha256": sha256_file(path), "n_valid": n,
                      "sigma_Delta": sd, "sigma_Delta_upper_95": sigma_upper_confidence_bound(sd, n)}
    if set(entries) != set(PRIMARY_FAMILY):
        raise ValueError(f"calibration certificates are needed for both of {PRIMARY_FAMILY}")
    return {"artifact": VARIANCE_ARTIFACT, "non_evidentiary": True, "design_hash": design_hash(config),
            "generated_at_utc": utc_timestamp(), "inputs": entries,
            "sigma_Delta_upper_95": {h: e["sigma_Delta_upper_95"] for h, e in entries.items()},
            "prohibition": "sizes the production design only; never chooses or validates epsilon or the target effect"}
