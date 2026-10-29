"""
Frozen AutoND EV protocol and the configuration that instantiates it.

The training protocol is AutoND's DBitNet at the pinned commit, trained for one stage at
round 5 (upstream `train_nets`: dbitnet.make_model(64), Adam(amsgrad=True), MSE, "acc",
EPOCHS=10, BATCHSIZE=5000, NUM_SAMPLES=10**7, NUM_VAL_SAMPLES=10**6, model.fit default
shuffle=True). EV-specific changes are listed in EV_SPECIFIC_CHANGES and recorded in
every certificate; none of them is inherited from, or approved by, D5.

configs/autond_ev.yaml is hashed in three scopes, so each tier binds exactly what was
frozen before it ran:
    design_hash      schema + protocol + representation + decisions + calibration   (calibration binds this)
    production_hash  everything except the non-evidentiary smoke section (production binds this)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from framework.provenance import config_hash

SCHEMA = "autond-ev-config-1"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "autond_ev.yaml"
PLACEHOLDER = "UNSPECIFIED_REQUIRES_APPROVAL"
CONFIRMATORY_EVALUATION_PROTOCOL = "final_epoch_weights"

EXPERIMENT_IDS = ("EV-AUTOND-BASELINE", "EV-AUTOND-NOISE", "H-EV-AUTOND-TRAIN-ORDER", "H-EV-AUTOND-REPRESENTATION")
PRIMARY_FAMILY = ("H-EV-AUTOND-TRAIN-ORDER", "H-EV-AUTOND-REPRESENTATION")

#: Must equal the config's `protocol` section exactly.
FROZEN_PROTOCOL = {
    "cipher": "speck3264", "scenario": "single-key", "delta_plain_hex": "0x400000", "rounds": 5,
    "upstream_commit": "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779",
    "architecture": "dbitnet", "builder": "dbitnet.make_model(64)", "optimizer": "Adam(amsgrad=True)",
    "loss": "mse", "metrics": ["acc"], "epochs": 10, "batch_size": 5000, "shuffle": True,
    "train_size": 10_000_000, "val_size": 1_000_000, "confirmatory_test_size": 1_000_000,
    "evaluation": CONFIRMATORY_EVALUATION_PROTOCOL,
}

#: EV statistical-plan values reused unchanged (docs/statistical_plan.md), not new decisions.
FRAMEWORK_INHERITED = {"alpha": 0.05, "target_power": 0.80, "power_simulations": 50_000, "sigma_bound": "upper95"}

EV_SPECIFIC_CHANGES = (
    "one training stage at round 5 from a fresh initialisation; upstream's staged multi-round loop "
    "(weight transfer between rounds, stopping rule) is not run",
    "outcome is accuracy of the final-epoch model on an independent 10**6-row confirmatory test set; "
    "upstream reports max validation accuracy and keeps the min-val_loss checkpoint per round",
    "TensorFlow is seeded per model with tf.keras.utils.set_random_seed (upstream is unseeded); GPU "
    "kernels remain nondeterministic and no op-determinism flag is set",
    "datasets are persisted once per replicate/pair and reloaded hash-verified, so both arms of a pair "
    "and any resumed unit use the identical data",
    "shuffle is passed explicitly; the baseline value is upstream's model.fit default (True)",
)


class ConfigError(ValueError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ConfigError(f"{path}: not a {SCHEMA} configuration")
    return config


def design_hash(config: dict) -> str:
    return config_hash({k: config.get(k) for k in ("schema", "protocol", "representation", "decisions", "calibration")})


def production_hash(config: dict) -> str:
    return config_hash({k: v for k, v in config.items() if k != "smoke"})


def _placeholders(value: Any, prefix: str) -> list[str]:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _placeholders(v, f"{prefix}.{k}")]
    return [prefix] if isinstance(value, str) and value.startswith("UNSPECIFIED") else []


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _seeds(config: dict, mode: str, experiment: str) -> set[int]:
    plan = config[mode][experiment]
    if experiment == "EV-AUTOND-NOISE":
        return {int(plan["seed"])}
    return set(range(int(plan["base_seed"]), int(plan["base_seed"]) + int(plan["units"])))


def check_config(config: dict, mode: str) -> list[str]:
    """Every reason `mode` may not run under `config` (empty list = allowed)."""
    problems = []
    if config.get("protocol") != FROZEN_PROTOCOL:
        differing = sorted(k for k in set(FROZEN_PROTOCOL) | set(config.get("protocol") or {})
                           if (config.get("protocol") or {}).get(k) != FROZEN_PROTOCOL.get(k))
        problems.append(f"protocol differs from the frozen AutoND protocol in {differing}")
    representation = config.get("representation") or {}
    if not (isinstance(representation.get("generation_seed"), int)
            and isinstance(representation.get("expected_sha256"), str)):
        problems.append("representation needs an integer generation_seed and an expected_sha256")
    decisions = config.get("decisions") or {}
    for key, value in FRAMEWORK_INHERITED.items():
        if decisions.get(key) != value:
            problems.append(f"decisions.{key}={decisions.get(key)!r}; the EV statistical plan fixes {value!r}")
    if mode == "smoke":
        return problems

    sections = ["representation", "decisions", "calibration"] + (["production"] if mode == "production" else [])
    pending = [p for s in sections for p in _placeholders(config.get(s), s)]
    if pending:
        return problems + [f"unresolved decisions: {pending}"]

    epsilon, target = decisions.get("practical_threshold"), decisions.get("target_effect")
    if not (_number(epsilon) and 0 < epsilon < 0.5):
        problems.append("decisions.practical_threshold must be an accuracy difference in (0, 0.5)")
    elif not (_number(target) and target > epsilon):
        # At a true effect equal to epsilon, SUPPORTED (CI entirely outside +/-epsilon) has probability
        # below 1/2 at every n, so a target at the margin can never be powered for SUPPORTED.
        problems.append("decisions.target_effect must exceed practical_threshold (SUPPORTED needs the "
                        "CI entirely outside the margin)")
    for key in ("practical_threshold_justification", "target_effect_justification"):
        if not str(decisions.get(key) or "").strip():
            problems.append(f"decisions.{key} must be a non-empty justification")
    for hyp in PRIMARY_FAMILY:
        if not (isinstance(config["calibration"][hyp].get("units"), int) and config["calibration"][hyp]["units"] >= 2):
            problems.append(f"calibration.{hyp}.units must be an integer >= 2")
    if mode == "production":
        production = config["production"]
        for experiment in EXPERIMENT_IDS:
            plan = production[experiment]
            if not (isinstance(plan.get("units"), int) and isinstance(plan.get("minimum_valid"), int)
                    and 2 <= plan["minimum_valid"] <= plan["units"]):
                problems.append(f"production.{experiment} needs integers 2 <= minimum_valid <= units")
        used: dict[int, str] = {}
        for tier in ("calibration", "production"):
            for experiment in (PRIMARY_FAMILY if tier == "calibration" else EXPERIMENT_IDS):
                try:
                    seeds = _seeds(config, tier, experiment)
                except (KeyError, TypeError, ValueError):
                    continue
                clash = sorted(s for s in seeds if s in used)
                if clash:
                    problems.append(f"{tier}.{experiment} reuses model seeds {clash[:3]} of {used[clash[0]]}")
                used.update({s: f"{tier}.{experiment}" for s in seeds})
    return problems


def run_plan(config: dict, experiment: str, mode: str) -> dict:
    """Sizes, epochs, replicate counts and seeds for one experiment in one tier."""
    if experiment not in EXPERIMENT_IDS:
        raise ConfigError(f"unknown experiment {experiment!r}")
    if mode == "calibration" and experiment not in PRIMARY_FAMILY:
        raise ConfigError(f"{experiment} has no calibration tier: it does not estimate a paired-difference variance")
    protocol = config["protocol"] if mode != "smoke" else {**config["protocol"], **config["smoke"]["protocol"]}
    plan = config[mode][experiment]
    units = int(plan["units"])
    return {
        "sizes": {"train": int(protocol["train_size"]), "validation": int(protocol["val_size"]),
                  "test": int(protocol["confirmatory_test_size"])},
        "epochs": int(protocol["epochs"]), "batch_size": int(protocol["batch_size"]),
        "units": units, "minimum_valid": int(plan.get("minimum_valid", units)),
        "seed": int(plan.get("seed", plan.get("base_seed", 0))),
    }
