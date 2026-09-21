#!/usr/bin/env python3
"""
Production EV runner.

REVISION NOTE (firewall lifecycle defect fix): the previous version
called firewall.freeze() with frozen_permutation_hash=None and only
generated the Candidate-1 permutation afterward - backwards from the
required lifecycle. It also never called seal_confirmatory_dataset()
at all, so H-EV-REPRESENTATION would have crashed on its first pair,
and H-EV-SHUFFLE had no firewall integration whatsoever. This version:

    1. generates the permutation (H-EV-REPRESENTATION only);
    2. validates it;
    3. persists it and computes its hash;
    4. freezes the firewall WITH that hash already known;
    5. hands the frozen firewall to gohr.experiments, which seals each
       pair's confirmatory dataset (one hash per replicate_id) as it is
       generated, before consuming the single-use evaluation slot.

Fails closed (refuses to run, exits non-zero) if:
    - any required config value is still a placeholder
      ("UNSPECIFIED_..."), or
    - the config's frozen-baseline fields do not match gohr.baseline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from framework.experiment import PracticalSignificance
from framework.firewall import FreezeRecord, TestSetFirewall
from framework.power import UnderpoweredReplicatePlanError, validate_replicate_plan_against_power
from framework.provenance import config_hash, utc_timestamp
from gohr.baseline import BASELINE
from gohr.experiments import (
    apply_primary_family_correction,
    run_ev_baseline,
    run_ev_noise,
    run_h_ev_representation,
    run_h_ev_shuffle,
)
from gohr.representation import generate_candidate1_permutation, run_full_validation
from gohr import speck


class ConfigNotFrozenError(RuntimeError):
    pass


def _check_no_placeholders(config: dict, path_prefix: str = "") -> None:
    for key, value in config.items():
        full_key = f"{path_prefix}.{key}" if path_prefix else key
        if isinstance(value, str) and value.startswith("UNSPECIFIED"):
            raise ConfigNotFrozenError(
                f"Config field {full_key!r} is still a placeholder ({value!r}). "
                "Refusing to run a production experiment with an unresolved value."
            )
        if isinstance(value, dict):
            _check_no_placeholders(value, full_key)


def _check_baseline_consistency(config: dict) -> None:
    mismatches = []
    for field_name, baseline_value in [
        ("rounds", BASELINE.rounds), ("depth", BASELINE.depth),
        ("epochs", BASELINE.epochs), ("batch_size", BASELINE.batch_size),
        ("train_size", BASELINE.train_size), ("val_size", BASELINE.val_size),
    ]:
        if field_name in config and config[field_name] != baseline_value:
            mismatches.append((field_name, config[field_name], baseline_value))
    if mismatches:
        raise ConfigNotFrozenError(
            f"Config values diverge from the frozen baseline (gohr.baseline.BASELINE): "
            f"{mismatches}. Refusing to run - if this divergence is intentional, it must "
            "be represented as an explicit EV factor/condition, not a silent config edit."
        )


def _practical_significance_from_config(config: dict) -> PracticalSignificance:
    threshold = config.get("practical_threshold")
    if threshold is None:
        return PracticalSignificance(threshold=None, predeclared=False, justification=None)
    return PracticalSignificance(
        threshold=float(threshold), predeclared=True,
        justification=config.get(
            "practical_threshold_justification",
            "epsilon=0.01 absolute confirmatory accuracy, predeclared per the Round-5 "
            "statistical plan; gated on confirmation against pilot replicate-to-replicate "
            "spread (see docs/statistical_plan.md) before this config is used for a "
            "confirmatory production run.",
        ),
    )


def _load_or_freeze_firewall(
    experiment_id: str, output_dir: Path, config: dict, *, permutation_hash: Optional[str],
) -> TestSetFirewall:
    """
    ISSUE 2 FIX (Round 6): restore a previously persisted firewall on
    resume instead of always constructing a fresh, empty one. A fresh
    firewall would have no record of which (condition, replicate) keys
    were already consumed, allowing a resumed run to re-evaluate a
    pair's confirmatory data that was already used - silently
    weakening the single-use guarantee across a restart.

    If output_dir/firewall.json exists, it is loaded and its
    frozen_config_hash / frozen_permutation_hash are verified against
    what THIS invocation would freeze - a mismatch (e.g. a config edit
    between runs) raises rather than silently proceeding with either
    the old or the new state. If no persisted firewall exists, a new
    one is created and frozen exactly as before.
    """
    firewall_path = output_dir / "firewall.json"
    expected_config_hash = config_hash(config)

    if firewall_path.exists():
        restored = TestSetFirewall.load(firewall_path)
        if restored.freeze_record is None:
            raise RuntimeError(
                f"Persisted firewall at {firewall_path} exists but was never frozen; "
                "refusing to resume from an inconsistent firewall state."
            )
        if restored.freeze_record.frozen_config_hash != expected_config_hash:
            raise RuntimeError(
                f"Refusing to resume {experiment_id}: persisted firewall was frozen with "
                f"config_hash={restored.freeze_record.frozen_config_hash}, but the current "
                f"config hashes to {expected_config_hash}. The configuration must not change "
                "between an interrupted run and its resume."
            )
        if restored.freeze_record.frozen_permutation_hash != permutation_hash:
            raise RuntimeError(
                f"Refusing to resume {experiment_id}: persisted firewall was frozen with "
                f"permutation_hash={restored.freeze_record.frozen_permutation_hash}, but this "
                f"invocation supplies permutation_hash={permutation_hash}. The permutation must "
                "not change between an interrupted run and its resume."
            )
        print(f"Restored firewall state from {firewall_path} "
              f"({len(restored.sealed_dataset_hashes)} sealed replicate(s), "
              f"{len(restored._consumed_keys)} consumed evaluation key(s)).")
        return restored

    firewall = TestSetFirewall(experiment_id=experiment_id)
    firewall.freeze(FreezeRecord(
        frozen_config_hash=expected_config_hash, frozen_permutation_hash=permutation_hash,
        frozen_replicate_plan_hash=config_hash({
            "requested_pairs": config["requested_pairs"], "minimum_valid_pairs": config["minimum_valid_pairs"],
        }),
        frozen_statistical_plan_hash=config_hash({
            "alpha": config["alpha"], "multiplicity_family": config["multiplicity_family"],
            "practical_threshold": config.get("practical_threshold"),
        }),
        frozen_at_utc=utc_timestamp(),
    ))
    return firewall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a YAML config in configs/")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    experiment_id = config["experiment_id"]

    print(f"Loaded config for {experiment_id} from {args.config}")
    print(f"config_hash={config_hash(config)}")

    try:
        _check_no_placeholders(config)
        if config.get("run_mode") == "production":
            _check_baseline_consistency(config)
        if experiment_id in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION"):
            power_analysis = config.get("power_analysis") or {}
            validate_replicate_plan_against_power(
                minimum_valid_replicates=config["minimum_valid_pairs"],
                power_analysis_required_n=power_analysis.get("required_n"),
                underpowered_justification=power_analysis.get("underpowered_justification"),
            )
    except (ConfigNotFrozenError, UnderpoweredReplicatePlanError) as exc:
        print(f"\nREFUSING TO RUN (fail-closed): {exc}\n")
        return 1

    output_dir = Path(config["output_dir"])
    run_mode = config["run_mode"]
    practical_significance = _practical_significance_from_config(config)

    if experiment_id == "EV-BASELINE":
        cert = run_ev_baseline(
            run_mode=run_mode, requested_replicates=config["requested_replicates"],
            minimum_valid_replicates=config["minimum_valid_replicates"],
            output_dir=output_dir, base_model_seed=config["base_model_seed"],
        )

    elif experiment_id == "EV-NOISE":
        cert = run_ev_noise(
            run_mode=run_mode, requested_reruns=config["requested_reruns"],
            minimum_valid_reruns=config["minimum_valid_reruns"],
            output_dir=output_dir, model_seed=config["model_seed"],
        )

    elif experiment_id == "H-EV-SHUFFLE":
        # Step 1: freeze (no permutation involved in this factor).
        firewall = _load_or_freeze_firewall(experiment_id, output_dir, config, permutation_hash=None)
        cert = run_h_ev_shuffle(
            run_mode=run_mode, requested_pairs=config["requested_pairs"],
            minimum_valid_pairs=config["minimum_valid_pairs"], output_dir=output_dir,
            firewall=firewall, base_model_seed=config["base_model_seed"],
            practical_significance=practical_significance,
        )
        firewall.save(output_dir / "firewall.json")

    elif experiment_id == "H-EV-REPRESENTATION":
        # Step 1: generate the permutation.
        permutation = generate_candidate1_permutation(
            np.random.default_rng(config["permutation_generation_seed"])
        )
        # Step 2: validate it BEFORE it is trusted for anything downstream.
        X_probe, Y_probe = speck.make_train_data(2000, BASELINE.rounds, diff=BASELINE.differential)
        validation = run_full_validation(X_probe, Y_probe, permutation)
        if not validation["all_passed"]:
            print(f"\nREFUSING TO RUN: Candidate-1 permutation failed validation: {validation}\n")
            return 1
        # Step 3: persist it (hash is already computed as part of the object).
        output_dir.mkdir(parents=True, exist_ok=True)
        permutation.save(output_dir / "candidate1_permutation.json")
        print(f"Candidate-1 permutation generated, validated, and persisted. hash={permutation.hash}")

        # Step 4: restore a prior firewall, or freeze a new one WITH the now-known permutation hash.
        firewall = _load_or_freeze_firewall(experiment_id, output_dir, config, permutation_hash=permutation.hash)
        cert = run_h_ev_representation(
            run_mode=run_mode, requested_pairs=config["requested_pairs"],
            minimum_valid_pairs=config["minimum_valid_pairs"], output_dir=output_dir,
            permutation=permutation, firewall=firewall, base_model_seed=config["base_model_seed"],
            practical_significance=practical_significance,
        )
        firewall.save(output_dir / "firewall.json")

    else:
        print(f"Unknown experiment_id: {experiment_id}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    cert_path = output_dir / f"{experiment_id}_certificate.json"
    cert_path.write_text(json.dumps(cert, indent=2, sort_keys=True, default=str))
    print(f"Certificate written to {cert_path}")
    print(f"decision={cert['decision']} non_evidentiary={cert.get('non_evidentiary')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
