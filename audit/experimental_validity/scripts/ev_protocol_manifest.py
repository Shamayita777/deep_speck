#!/usr/bin/env python3
"""
Emit the EV protocol manifest binding a calibration or production run to
the exact frozen scientific design and to the code that implements it.

Recorded: config hash, statistical-plan hash, source manifest hash,
baseline hash, Candidate-1 permutation hash (representation only),
epsilon, target effect, alpha, target power, calibration sample size and
power-simulation count. Everything a reviewer needs to establish WHICH
design and WHICH code produced a result, without trusting prose.
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

import yaml  # noqa: E402

from audit.common.provenance import sha256_bytes, sha256_file, utc_timestamp  # noqa: E402
from audit.common.strict_json import dumps_strict  # noqa: E402
from framework.provenance import config_hash  # noqa: E402
from gohr.baseline import BASELINE, CONFIRMATORY_EVALUATION_PROTOCOL  # noqa: E402

#: Source files whose content determines EV behaviour.
#: Frozen EV design constants (see docs/statistical_plan.md).
FROZEN_ALPHA = 0.05
FROZEN_TARGET_POWER = 0.80
FROZEN_TARGET_EFFECT = 0.01
FROZEN_EPSILON = 0.01
FROZEN_CALIBRATION_PAIRS = 10

EV_SOURCES = [
    "framework/statistics.py", "framework/certificate.py", "framework/power.py",
    "framework/firewall.py", "framework/validation.py", "framework/multiplicity.py",
    "gohr/experiments.py", "gohr/adapter.py", "gohr/model.py", "gohr/train.py",
    "gohr/dataset.py", "gohr/evaluate.py", "gohr/baseline.py", "gohr/representation.py",
    "scripts/run_ev.py", "scripts/analyze_ev_calibration.py", "scripts/plan_ev_power.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--power-artifact", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sources = {}
    for rel in EV_SOURCES:
        path = REPO / rel
        if not path.is_file():
            print(f"REFUSED: EV source missing: {rel}")
            return 1
        sources[rel] = sha256_file(path)

    plan_path = REPO / "docs" / "statistical_plan.md"
    perm_hash = cfg.get("expected_candidate1_permutation_sha256")

    power = {}
    if args.power_artifact and args.power_artifact.is_file():
        plan = json.loads(args.power_artifact.read_text())
        power = {
            "artifact_path": str(args.power_artifact),
            "artifact_sha256": sha256_file(args.power_artifact),
            "required_n": (plan.get("common_design") or {}).get("required_n"),
            "n_simulations": (plan.get("inputs") or {}).get("n_simulations"),
            "calibration_sample_sizes": (plan.get("inputs") or {}).get("calibration_sample_sizes"),
        }

    manifest = {
        "artifact": "ev-protocol-manifest-v1",
        "generated_at_utc": utc_timestamp(),
        "experiment_id": cfg.get("experiment_id"),
        "run_mode": cfg.get("run_mode"),
        "non_evidentiary": cfg.get("run_mode") != "production",
        "config": {"path": str(args.config), "config_hash": config_hash(cfg),
                   "file_sha256": sha256_file(args.config)},
        "statistical_plan_sha256": sha256_file(plan_path) if plan_path.is_file() else None,
        "source_manifest": sources,
        "source_manifest_sha256": sha256_bytes(
            dumps_strict(sources, sort_keys=True).encode("utf-8")),
        "baseline": {"hash": config_hash(BASELINE.to_dict()),
                     "confirmatory_evaluation_protocol": CONFIRMATORY_EVALUATION_PROTOCOL,
                     "reg_param": BASELINE.reg_param, "depth": BASELINE.depth},
        "candidate1_permutation_sha256": perm_hash,
        "practical_significance": {
            "epsilon": cfg.get("practical_threshold"),
            "predeclared": bool(cfg.get("practical_threshold_predeclared", False)),
            "justification": cfg.get("practical_threshold_justification"),
            "note": "epsilon is a consequence-based margin; never derived from pilot noise.",
        },
        "statistical_design": {
            "alpha": FROZEN_ALPHA, "target_power": FROZEN_TARGET_POWER,
            # FROZEN design values; a config may restate but not weaken them.
            # (`.get(..., default)` would silently accept an explicit null.)
            "target_effect": ((cfg.get("power_analysis") or {}).get("target_effect")
                              or FROZEN_TARGET_EFFECT),
            "target_effect_source": "predeclared",
            "primary_family": ["H-EV-SHUFFLE", "H-EV-REPRESENTATION"],
            "multiplicity_at_analysis": "Holm on difference-detection p-values",
            "multiplicity_at_planning": "alpha/2 (conservative Bonferroni-style, NOT exact Holm)",
        },
        "calibration": {"pairs_per_paired_primary": FROZEN_CALIBRATION_PAIRS,
                        "note": "operational predeclared pilot size, not a universal minimum"},
        "power_analysis": power,
    }
    out = args.output or Path(cfg.get("output_dir", ".")) / "ev_protocol_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_strict(manifest, indent=2, sort_keys=True))
    print(f"Wrote {out}")
    print(f"  config_hash={manifest['config']['config_hash']}")
    print(f"  source_manifest_sha256={manifest['source_manifest_sha256']}")
    print(f"  baseline_hash={manifest['baseline']['hash']}")
    print(f"  permutation={perm_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
