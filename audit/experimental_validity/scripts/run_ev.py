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
import os
import sys
from pathlib import Path
from typing import Optional


def _pin_gpu_before_tensorflow_import() -> Optional[str]:
    """
    Honour --gpu by setting CUDA_VISIBLE_DEVICES BEFORE TensorFlow is
    imported transitively (gohr.* pull it in at module import time).
    TensorFlow reads that variable once at initialisation, so a later
    assignment has no effect - hence this runs at module top, ahead of
    every other import.

    Parsing argv by hand here is deliberate: argparse runs inside main(),
    which is far too late. GPU choice is a PROCESS-level concern, never a
    config field: putting it in the YAML would change the config hash and
    invalidate the resume state of an in-flight calibration.
    """
    argv = sys.argv
    value = None
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--gpu="):
            value = arg.split("=", 1)[1]
    if value is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = value
    return value


_PINNED_GPU = _pin_gpu_before_tensorflow_import()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from framework.experiment import PracticalSignificance
from framework.firewall import FreezeRecord, TestSetFirewall
from framework.power import UnderpoweredReplicatePlanError, validate_replicate_plan_against_power
from framework.validation import ConfigValidationError, require_run_mode
from framework.provenance import config_hash, sha256_file, utc_timestamp
from gohr.baseline import BASELINE, CONFIRMATORY_EVALUATION_PROTOCOL
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


class RunModeConflictError(RuntimeError):
    pass


#: Every EV experiment's output root must be unique. Two concurrent
#: hypothesis-level jobs (e.g. H-EV-SHUFFLE on GPU 0 and
#: H-EV-REPRESENTATION on GPU 1) share nothing: separate output dirs,
#: separate ledgers, separate pair sidecars, separate firewalls, separate
#: seed ranges. This table is asserted so a copy-paste edit cannot point
#: two jobs at one directory.
EXPECTED_OUTPUT_ROOTS = {
    ("H-EV-SHUFFLE", "calibration"): "results/calibration/ev_shuffle",
    ("H-EV-REPRESENTATION", "calibration"): "results/calibration/ev_representation",
    ("EV-BASELINE", "calibration"): "results/calibration/ev_baseline",
    ("EV-NOISE", "calibration"): "results/calibration/ev_noise",
}


class ConcurrentRunConflictError(RuntimeError):
    pass


def _check_output_isolation(config: dict, experiment_id: str) -> None:
    """
    Refuse an output directory that belongs to a different experiment.

    Concurrency safety rests entirely on state separation: each job owns
    its own resume ledger, pair sidecar, dataset store and firewall file
    under its own output root. If two jobs shared a root they would
    interleave ledger writes and could consume each other's firewall
    seals. Nothing else about the design changes - the two hypotheses are
    independent experiments, not two arms of one pair.
    """
    declared = str(config.get("output_dir", "")).rstrip("/")
    for (exp, mode), root in EXPECTED_OUTPUT_ROOTS.items():
        if declared.endswith(root) and exp != experiment_id:
            raise ConfigNotFrozenError(
                f"{experiment_id}: output_dir {declared!r} is the reserved output root of "
                f"{exp}. Two experiments must never share an output root - they would "
                "interleave ledger, sidecar and firewall state.")


def _acquire_output_lock(output_dir: Path, experiment_id: str) -> Optional[Path]:
    """
    Advisory single-writer lock per output directory.

    Two processes writing one output root would corrupt the resume ledger
    and the firewall. A stale lock from a crashed process is reported, not
    silently cleared: deciding whether a previous run really died is the
    operator's call, not the framework's.
    """
    lock = Path(output_dir) / "RUN_LOCK"
    if lock.exists():
        raise ConcurrentRunConflictError(
            f"{output_dir} is locked by a previous or concurrent run "
            f"({lock.read_text().strip()}). If that process is definitely gone, delete "
            f"{lock} manually after confirming no other job is writing here. Refusing to "
            "write concurrently: it would corrupt the resume ledger and firewall state.")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"pid={os.getpid()} experiment={experiment_id} "
                    f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    # Released on normal exit AND on an unhandled exception. A hard kill
    # (SIGKILL / OOM) deliberately leaves the lock behind so the operator
    # sees that a run died mid-write rather than silently resuming over it.
    import atexit
    atexit.register(lambda: lock.unlink(missing_ok=True))
    return lock


def _lock_output_dir_to_mode(output_dir: Path, run_mode: str) -> None:
    """
    Bind an output directory to exactly one run mode, permanently.

    Calibration and production use the IDENTICAL protocol, so their
    artifacts are indistinguishable by inspection. If both wrote into one
    directory, a resumed or re-analysed run could silently mix pilot and
    confirmatory replicates into a single denominator. The marker is
    written on first use and checked on every subsequent use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "RUN_MODE"
    if marker.exists():
        recorded = marker.read_text().strip()
        if recorded != run_mode:
            raise RunModeConflictError(
                f"output_dir {output_dir} was already used for run_mode={recorded!r}; refusing "
                f"to write {run_mode!r} results into it. Calibration and production must never "
                "share an output directory.")
    else:
        marker.write_text(run_mode)


#: Frozen design values a production power artifact must have been computed under.
FROZEN_DESIGN = {
    "epsilon": 0.01, "target_effect": 0.01, "target_effect_source": "predeclared",
    "alpha": 0.05, "target_power": 0.80, "n_simulations": 50_000,
    "sigma_source": "upper95", "plan_artifact": "ev-power-plan-v2",
    "primary_family": ["H-EV-SHUFFLE", "H-EV-REPRESENTATION"],
}


def _require_power_artifact(config: dict, experiment_id: str) -> None:
    """
    Production of a paired primary requires a prospective power artifact
    that was computed under EXACTLY the frozen design, and enough valid
    pairs to meet its required_n.

    Every field is checked rather than trusted: an artifact produced under
    a different epsilon, alpha, effect, simulation count or sigma source
    would authorise a replicate count that does not correspond to this
    experiment's decision procedure. The artifact hash must be declared in
    the config and must match, so the plan cannot be swapped after the
    config was frozen.
    """
    if experiment_id not in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION"):
        return
    pa = config.get("power_analysis") or {}
    artifact = pa.get("artifact_path")
    if not artifact:
        raise ConfigNotFrozenError(
            f"{experiment_id}: production requires power_analysis.artifact_path pointing at a "
            "power plan produced by scripts/plan_ev_power.py.")
    path = Path(artifact)
    if not path.is_file():
        raise ConfigNotFrozenError(f"{experiment_id}: power artifact {artifact} does not exist.")

    declared_hash = pa.get("artifact_sha256")
    actual_hash = sha256_file(path)
    if not declared_hash:
        raise ConfigNotFrozenError(
            f"{experiment_id}: power_analysis.artifact_sha256 must be declared so the plan "
            "cannot be substituted after the config was frozen.")
    if declared_hash != actual_hash:
        raise ConfigNotFrozenError(
            f"{experiment_id}: power artifact hash mismatch (config {declared_hash}, "
            f"actual {actual_hash}).")

    plan = json.loads(path.read_text())
    inputs = plan.get("inputs") or {}
    problems = []
    if plan.get("artifact") != FROZEN_DESIGN["plan_artifact"]:
        problems.append(f"plan schema={plan.get('artifact')!r} "
                        f"(required {FROZEN_DESIGN['plan_artifact']!r})")
    if plan.get("non_evidentiary") is not True:
        problems.append("power plan is not marked non_evidentiary=true")
    if list(plan.get("primary_family") or []) != FROZEN_DESIGN["primary_family"]:
        problems.append(f"primary_family={plan.get('primary_family')!r}")
    for key, expected in (("epsilon", FROZEN_DESIGN["epsilon"]),
                          ("target_effect", FROZEN_DESIGN["target_effect"]),
                          ("target_effect_source", FROZEN_DESIGN["target_effect_source"]),
                          ("alpha", FROZEN_DESIGN["alpha"]),
                          ("target_power", FROZEN_DESIGN["target_power"]),
                          ("n_simulations", FROZEN_DESIGN["n_simulations"]),
                          ("sigma_source", FROZEN_DESIGN["sigma_source"])):
        if inputs.get(key) != expected:
            problems.append(f"plan {key}={inputs.get(key)!r} (frozen {expected!r})")

    # the config's own declared parameters must agree with the artifact
    for key in ("epsilon", "target_effect", "alpha", "target_power"):
        if key in pa and pa[key] != inputs.get(key):
            problems.append(f"config power_analysis.{key}={pa[key]!r} disagrees with plan "
                            f"{inputs.get(key)!r}")
    if config.get("practical_threshold") != inputs.get("epsilon"):
        problems.append(f"config practical_threshold={config.get('practical_threshold')!r} "
                        f"disagrees with plan epsilon={inputs.get('epsilon')!r}")

    required_n = pa.get("required_n")
    plan_n = (plan.get("common_design") or {}).get("required_n")
    if required_n is None:
        problems.append("power_analysis.required_n is not declared")
    elif plan_n is None:
        problems.append("power plan records no common_design.required_n "
                        "(no design reached target power)")
    elif int(required_n) != int(plan_n):
        problems.append(f"power_analysis.required_n={required_n} disagrees with the power "
                        f"artifact ({plan_n})")
    elif int(config["minimum_valid_pairs"]) < int(required_n):
        problems.append(f"minimum_valid_pairs={config['minimum_valid_pairs']} is below the "
                        f"power-required n={required_n}")
    if problems:
        raise ConfigNotFrozenError(f"{experiment_id}: " + "; ".join(problems))


def _require_permutation_binding(config: dict, experiment_id: str) -> None:
    """
    Verify the Candidate-1 permutation binding during PRE-FLIGHT validation.

    Calibration and production must use the identical permutation, or the
    pilot characterises a different intervention and its variance cannot
    size the production design. The permutation is generated
    deterministically from permutation_generation_seed, so the realized
    hash can be checked before anything is generated or trained - which is
    where it belongs: a validation-only invocation must be able to detect
    a broken binding without launching an experiment.
    """
    if experiment_id != "H-EV-REPRESENTATION":
        return
    expected = config.get("expected_candidate1_permutation_sha256")
    if not expected:
        raise ConfigNotFrozenError(
            f"{experiment_id}: requires 'expected_candidate1_permutation_sha256' so the "
            "Candidate-1 permutation is bound across calibration and production.")
    seed = config.get("permutation_generation_seed")
    if seed is None:
        raise ConfigNotFrozenError(
            f"{experiment_id}: 'permutation_generation_seed' is not declared.")
    realized = generate_candidate1_permutation(np.random.default_rng(seed)).hash
    if realized != expected:
        raise ConfigNotFrozenError(
            f"{experiment_id}: Candidate-1 permutation hash mismatch - expected {expected}, "
            f"seed {seed} realizes {realized}. Calibration and production must use the "
            "identical permutation.")


def _require_epsilon_for_confirmatory(config: dict, experiment_id: str) -> None:
    """
    Production runs of the two PAIRED primary experiments must carry a
    predeclared epsilon. Without it no equivalence conclusion is possible
    and a non-significant result can only ever be INCONCLUSIVE - which is
    a design defect, not a finding, if it was foreseeable before running.
    """
    if experiment_id not in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION"):
        return
    threshold = config.get("practical_threshold")
    if threshold is None:
        raise ConfigNotFrozenError(
            f"{experiment_id}: production requires a PREDECLARED practical-significance margin "
            "('practical_threshold'). Without epsilon no equivalence conclusion is possible and "
            "a non-significant result can only ever be INCONCLUSIVE. epsilon must be justified "
            "from scientific consequence, never derived from an observed effect.")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ConfigNotFrozenError(
            f"{experiment_id}: practical_threshold must be numeric, got {threshold!r}.")
    if threshold <= 0:
        raise ConfigNotFrozenError(
            f"{experiment_id}: practical_threshold must be positive, got {threshold}.")
    if not str(config.get("practical_threshold_justification") or "").strip():
        raise ConfigNotFrozenError(
            f"{experiment_id}: 'practical_threshold_justification' must be a non-empty, "
            "consequence-based justification. A margin without a recorded justification is "
            "indistinguishable from a fabricated one.")
    if config.get("practical_threshold_predeclared") is not True:
        raise ConfigNotFrozenError(
            f"{experiment_id}: set 'practical_threshold_predeclared: true' to affirm that "
            "epsilon was frozen BEFORE any confirmatory data existed. The framework cannot "
            "verify this from the value alone, so it must be asserted explicitly.")


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


#: Every frozen baseline parameter a config may express, with the
#: accessor that reads the authoritative value from gohr.baseline.BASELINE.
#: BASELINE is the single source of truth; the config merely restates it
#: and MUST agree. Any parameter present here and absent from the config
#: is reported, so a duplicated field can never silently diverge and a
#: required field can never be silently dropped.
_BASELINE_CHECKS: dict = {
    "rounds": lambda b: b.rounds,
    "depth": lambda b: b.depth,
    "epochs": lambda b: b.epochs,
    "batch_size": lambda b: b.batch_size,
    "train_size": lambda b: b.train_size,
    "val_size": lambda b: b.val_size,
    "confirmatory_test_size": lambda b: b.test_size,
    "optimizer": lambda b: b.optimizer,
    "shuffle": lambda b: b.shuffle,
    "reg_param": lambda b: b.reg_param,
    "checkpoint_monitor": lambda b: b.checkpoint_monitor,
    "checkpoint_save_best_only": lambda b: b.checkpoint_save_best_only,
    "confirmatory_evaluation_protocol": lambda b: CONFIRMATORY_EVALUATION_PROTOCOL,
}

#: Nested lr_schedule keys -> BASELINE accessor.
_LR_CHECKS: dict = {
    "high": lambda b: b.lr_schedule_high,
    "low": lambda b: b.lr_schedule_low,
    "period": lambda b: b.lr_schedule_period,
}

#: Parameters every evidentiary (production/calibration) config must state
#: explicitly. Silence is not agreement.
_REQUIRED_IN_EVIDENTIARY_CONFIG = (
    "rounds", "depth", "epochs", "batch_size", "train_size", "val_size",
    "confirmatory_test_size", "optimizer", "shuffle", "lr_schedule",
)


#: The factor each experiment DELIBERATELY varies. A manipulated factor is
#: exempt from the baseline-equality check: pinning the factor under test to
#: the baseline would defeat the experiment. H-EV-SHUFFLE varies `shuffle`
#: across its two arms and therefore must NOT declare a single baseline value
#: for it; H-EV-REPRESENTATION varies the representation, which is not a
#: baseline field at all.
_MANIPULATED_FACTORS: dict = {
    "H-EV-SHUFFLE": ("shuffle",),
    "H-EV-REPRESENTATION": (),
    "EV-BASELINE": (),
    "EV-NOISE": (),
}


def _check_baseline_consistency(config: dict, *, require_complete: bool = False,
                                experiment_id: str = "") -> None:
    """
    Verify the config against EVERY frozen baseline parameter it expresses.

    Previously only six fields were checked, so a config could silently
    diverge from the baseline on optimizer, shuffle, reg_param, the LR
    schedule, the checkpoint rule or the evaluation protocol - exactly the
    class of undeclared-parameter defect that produced the depth-5 and
    reg_param findings.
    """
    manipulated = set(_MANIPULATED_FACTORS.get(experiment_id, ()))
    mismatches, missing = [], []
    for field_name, accessor in _BASELINE_CHECKS.items():
        if field_name in manipulated:
            if field_name in config:
                mismatches.append(
                    (field_name, config[field_name],
                     f"<manipulated factor of {experiment_id}: must not be pinned in config>"))
            continue
        baseline_value = accessor(BASELINE)
        if field_name in config:
            if config[field_name] != baseline_value:
                mismatches.append((field_name, config[field_name], baseline_value))
        elif require_complete and field_name in _REQUIRED_IN_EVIDENTIARY_CONFIG:
            missing.append(field_name)

    lr = config.get("lr_schedule")
    if isinstance(lr, dict):
        for key, accessor in _LR_CHECKS.items():
            baseline_value = accessor(BASELINE)
            if key in lr:
                if lr[key] != baseline_value:
                    mismatches.append((f"lr_schedule.{key}", lr[key], baseline_value))
            elif require_complete:
                missing.append(f"lr_schedule.{key}")
    elif require_complete and "lr_schedule" in _REQUIRED_IN_EVIDENTIARY_CONFIG:
        missing.append("lr_schedule")

    problems = []
    if mismatches:
        problems.append(
            f"values diverge from the frozen baseline (gohr.baseline.BASELINE): {mismatches}")
    if missing:
        problems.append(
            f"required frozen parameters are absent from the config: {sorted(missing)}")
    if problems:
        raise ConfigNotFrozenError(
            "; ".join(problems) + ". Refusing to run - if a divergence is intentional it must "
            "be represented as an explicit EV factor/condition, not a silent config edit.")


def _practical_significance_from_config(config: dict) -> PracticalSignificance:
    threshold = config.get("practical_threshold")
    if threshold is None:
        return PracticalSignificance(threshold=None, predeclared=False, justification=None)
    # Predeclaration is an ASSERTION the config must make; it can never be
    # inferred from the mere presence of a number. A threshold added after
    # results existed would otherwise be indistinguishable from one frozen
    # beforehand.
    predeclared = bool(config.get("practical_threshold_predeclared", False))
    return PracticalSignificance(
        threshold=float(threshold), predeclared=predeclared,
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
    parser.add_argument(
        "--gpu", default=None,
        help=("CUDA device index this job is pinned to (e.g. 0 or 1). Applied by setting "
              "CUDA_VISIBLE_DEVICES before TensorFlow is imported. Process-level only: it is "
              "never a config field and does not affect the config hash or resume state."))
    parser.add_argument(
        "--confirm-execute", action="store_true",
        help="actually run a calibration/production experiment (validation-only without it)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    experiment_id = config["experiment_id"]

    print(f"Loaded config for {experiment_id} from {args.config}")
    print(f"config_hash={config_hash(config)}")

    run_mode_declared = config.get("run_mode")
    try:
        require_run_mode(run_mode_declared)
        # Placeholders are refused in EVERY mode: a calibration run is also a
        # predeclared experiment, not an exploratory sweep.
        _check_no_placeholders(config)
        if run_mode_declared in ("production", "calibration"):
            _check_baseline_consistency(config, require_complete=True,
                                        experiment_id=experiment_id)
            _require_permutation_binding(config, experiment_id)
            _check_output_isolation(config, experiment_id)
        if run_mode_declared == "production":
            _require_epsilon_for_confirmatory(config, experiment_id)
            _require_power_artifact(config, experiment_id)
        if run_mode_declared == "production" and experiment_id in ("H-EV-SHUFFLE", "H-EV-REPRESENTATION"):
            power_analysis = config.get("power_analysis") or {}
            validate_replicate_plan_against_power(
                minimum_valid_replicates=config["minimum_valid_pairs"],
                power_analysis_required_n=power_analysis.get("required_n"),
                underpowered_justification=power_analysis.get("underpowered_justification"),
            )
    except (ConfigNotFrozenError, UnderpoweredReplicatePlanError, RunModeConflictError,
            ConfigValidationError) as exc:
        print(f"\nREFUSING TO RUN (fail-closed): {exc}\n")
        return 1

    output_dir = Path(config["output_dir"])
    run_mode = config["run_mode"]

    # EXECUTION CONFIRMATION. Validation must never start an experiment.
    # Calibration uses the full production protocol (10M samples, 200
    # epochs), so an accidental launch is as costly as a production one and
    # writes real data. Both evidentiary-protocol tiers therefore require an
    # explicit flag; without it this is a validation-only dry run.
    if run_mode in ("calibration", "production") and not args.confirm_execute:
        print(f"\nVALIDATION ONLY - all pre-flight checks passed for {experiment_id} "
              f"({run_mode}).\nNothing was generated or trained. Re-run with "
              f"--confirm-execute to actually launch this {run_mode} experiment.\n")
        return 0
    try:
        _lock_output_dir_to_mode(output_dir, run_mode)
    except (RunModeConflictError, ConcurrentRunConflictError) as exc:
        print(f"\nREFUSING TO RUN (fail-closed): {exc}\n")
        return 1
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"[execution] experiment={experiment_id} output_dir={output_dir} "
          f"CUDA_VISIBLE_DEVICES={gpu if gpu is not None else 'unset (all visible)'}")
    try:
        _run_lock = _acquire_output_lock(output_dir, experiment_id)
    except ConcurrentRunConflictError as exc:
        print(f"\nREFUSING TO RUN (fail-closed): {exc}\n")
        return 1
    if run_mode != "production":
        print(f"*** {run_mode.upper()} RUN - NON-EVIDENTIARY. Results may not be used as "
              f"confirmatory evidence. ***")
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
            firewall=firewall, firewall_path=output_dir / "firewall.json",
            base_model_seed=config["base_model_seed"],
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
        # Step 2b: bind the realized permutation to the expected hash. The
        # calibration pilot and the production run MUST use the identical
        # Candidate-1 permutation, or the pilot characterises a different
        # intervention and its variance cannot size the production design.
        # Re-checked here as well: pre-flight already verified the binding,
        # but the object actually used downstream must be the bound one.
        expected_perm = config.get("expected_candidate1_permutation_sha256")
        if permutation.hash != expected_perm:
            print(f"\nREFUSING TO RUN: Candidate-1 permutation hash mismatch.\n"
                  f"  expected {expected_perm}\n  realized {permutation.hash}\n"
                  "Calibration and production must use the identical permutation.\n")
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
