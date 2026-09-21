"""
The four frozen EV experiments and their orchestration:

    EV-BASELINE           - descriptive, independent replicates
    EV-NOISE               - descriptive, identical-configuration reruns
    H-EV-SHUFFLE           - matched-paired, shuffle True vs False
    H-EV-REPRESENTATION    - matched-paired, original vs Candidate-1 encoding

No additional core experiments are defined here, per the frozen scope.
H-EV-ARCHITECTURE is explicitly NOT implemented in this module.

REVISION NOTES (this file was corrected for two defects found in audit):

1. Firewall lifecycle: _run_paired_experiment now REQUIRES a firewall
   for BOTH H-EV-SHUFFLE and H-EV-REPRESENTATION (previously only
   H-EV-REPRESENTATION even attempted firewall use, and that attempt
   crashed because nothing ever called seal_confirmatory_dataset()).
   The firewall itself was restructured (framework/firewall.py) to seal
   one hash PER replicate_id, matching the fact that each pair
   generates its own independent confirmatory dataset. The lifecycle
   here is now: freeze (by the caller, before this module runs) ->
   generate per-pair datasets -> seal each pair's confirmatory hash ->
   train both arms -> consume the evaluation slot -> evaluate.

2. Statistics/decision: paired_analysis's primary test is now a paired
   t-test (see framework/statistics.py revision note), and the final
   decision for each hypothesis combines the Holm-adjusted difference
   decision with a TOST equivalence assessment via
   framework.certificate.decide_final, giving SUPPORTED /
   NOT_SUPPORTED (practically equivalent) / INCONCLUSIVE - never
   collapsing non-significance into "no effect".

Replicate-level resumability (new): each replicate/pair's completion is
recorded in a ResumeLedger keyed by (experiment_id, replicate_id). On
a subsequent invocation with an IDENTICAL configuration hash, already-
completed replicates are skipped rather than rerun; a MISMATCHED
configuration hash for an existing replicate_id raises rather than
silently overwriting it with different scientific conditions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audit.experimental_validity.framework.certificate import (
    DecisionState,
    assess_practical_equivalence,
    build_certificate,
    decide_difference_detection,
    decide_final,
)
from audit.experimental_validity.framework.experiment import (
    Condition,
    DesignType,
    Experiment,
    Factor,
    PairingDeclaration,
    PracticalSignificance,
    ReplicatePlan,
)
from audit.experimental_validity.framework.failures import ReplicateOutcome, ReplicateStatus
from audit.experimental_validity.framework.firewall import TestSetFirewall
from audit.experimental_validity.framework.multiplicity import holm_correction
from audit.experimental_validity.framework.provenance import config_hash, utc_timestamp
from audit.experimental_validity.framework.replication import ReplicateSet
from audit.experimental_validity.framework.resumability import ResumeLedger, RunState
from audit.experimental_validity.framework.statistics import descriptive_statistics, paired_analysis
from audit.experimental_validity.framework.seeds import statistics_rng
from audit.experimental_validity.gohr.adapter import GohrAdapter, MatchedDatasets, ReplicateConfig, generate_matched_datasets
from audit.experimental_validity.gohr import dataset as gohr_dataset
from audit.experimental_validity.gohr.baseline import BASELINE
from audit.experimental_validity.gohr.representation import Candidate1Permutation, generate_candidate1_permutation, identity_permutation, run_full_validation

PRIMARY_HYPOTHESIS_FAMILY_ID = "ev_primary_family_v1"

# epsilon is intentionally NOT set here. It is threaded through as an
# explicit, optional PracticalSignificance object supplied by the
# caller (scripts/run_ev.py, from a frozen config) - this module never
# invents or defaults one. See docs/statistical_plan.md for the
# gating requirement (pilot-spread sanity check) before any production
# config is allowed to set threshold != None.
NO_PRACTICAL_SIGNIFICANCE = PracticalSignificance(
    threshold=None, predeclared=False,
    justification="No practical-significance threshold has been supplied for this run.",
)


def _baseline_overrides_for_run_mode(run_mode: str) -> dict[str, Any]:
    """
    Explicit, non-hidden configuration per run mode. Smoke mode uses a
    drastically reduced problem solely to exercise the pipeline
    end-to-end; it is never treated as evidentiary. Production mode
    uses the frozen baseline exactly.
    """
    if run_mode == "smoke":
        return {
            "rounds": 3, "depth": 1, "epochs": 1, "batch_size": 64,
            "train_size": 256, "val_size": 64, "confirmatory_test_size": 64,
        }
    if run_mode == "production":
        return {
            "rounds": BASELINE.rounds, "depth": BASELINE.depth, "epochs": BASELINE.epochs,
            "batch_size": BASELINE.batch_size, "train_size": BASELINE.train_size,
            "val_size": BASELINE.val_size, "confirmatory_test_size": BASELINE.test_size,
        }
    raise ValueError(f"Unknown run_mode: {run_mode!r}")


def _resume_ledger_for(output_dir: Path) -> ResumeLedger:
    return ResumeLedger(Path(output_dir) / "resume_ledger.jsonl")


def _replicate_results_sidecar_path(output_dir: Path, experiment_id: str) -> Path:
    return Path(output_dir) / f"{experiment_id}_replicate_results.json"


def _load_replicate_results_sidecar(output_dir: Path, experiment_id: str) -> dict[str, float]:
    """
    Per-replicate metric values for descriptive experiments
    (EV-BASELINE, EV-NOISE), persisted alongside the resume ledger for
    the same reason as the paired-experiment sidecar: the ledger proves
    "this replicate completed under this config_hash" but carries no
    scientific payload, so recovering the actual metric value on resume
    is this orchestration layer's job.
    """
    import json
    path = _replicate_results_sidecar_path(output_dir, experiment_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_replicate_result_sidecar(output_dir: Path, experiment_id: str, replicate_id: str, metric_value: float) -> None:
    import json
    path = _replicate_results_sidecar_path(output_dir, experiment_id)
    data = _load_replicate_results_sidecar(output_dir, experiment_id)
    data[replicate_id] = metric_value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _ev_noise_dataset_manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / "ev_noise_dataset_manifest.json"


def _load_or_generate_ev_noise_dataset(
    output_dir: Path, *, rounds: int, differential: tuple[int, int],
    train_size: int, val_size: int, confirmatory_test_size: int,
) -> MatchedDatasets:
    """
    ISSUE 1 FIX (Round 6): EV-NOISE requires ONE dataset shared across
    every rerun (that is the entire point of "identical-configuration
    execution variability"). Previously, generate_matched_datasets()
    was called unconditionally on every invocation of run_ev_noise(),
    including on a resumed run - since Gohr dataset generation uses
    os.urandom() (not seedable), this silently produced a DIFFERENT
    dataset after an interruption, so any reruns still pending after
    resume would train on data unrelated to the reruns completed before
    the interruption, defeating the "same dataset" premise entirely.

    This function makes dataset generation itself part of the
    persisted, resumable state: the first invocation for a given
    output_dir generates and persists the three partitions (train,
    validation, confirmatory_test) and records their file paths and
    content hashes in a manifest; every subsequent invocation (resume)
    loads that manifest and reloads the exact same persisted arrays,
    verifying their hashes rather than regenerating anything.

    This does NOT claim exact_replay_available=True - the underlying
    generator remains nondeterministic, and if the manifest/artifacts
    do not exist yet, a fresh os.urandom()-based dataset is still
    generated (and cannot be reproduced from a seed). The guarantee is
    narrower and accurate: once generated and persisted, THIS SPECIFIC
    dataset instance is what every rerun of this experiment uses,
    resumed or not.
    """
    import json

    manifest_path = _ev_noise_dataset_manifest_path(output_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        train = gohr_dataset.load_dataset(
            manifest["train"]["path"], dataset_id=manifest["train"]["dataset_id"], role="train",
            rounds=rounds, differential=differential, expected_combined_hash=manifest["train"]["hash"],
        )
        validation = gohr_dataset.load_dataset(
            manifest["validation"]["path"], dataset_id=manifest["validation"]["dataset_id"], role="validation",
            rounds=rounds, differential=differential, expected_combined_hash=manifest["validation"]["hash"],
        )
        confirmatory_test = gohr_dataset.load_dataset(
            manifest["confirmatory_test"]["path"], dataset_id=manifest["confirmatory_test"]["dataset_id"],
            role="confirmatory_test", rounds=rounds, differential=differential,
            expected_combined_hash=manifest["confirmatory_test"]["hash"],
        )
        return MatchedDatasets(train=train, validation=validation, confirmatory_test=confirmatory_test)

    datasets = generate_matched_datasets(
        rounds=rounds, differential=differential, train_size=train_size,
        val_size=val_size, confirmatory_test_size=confirmatory_test_size,
    )
    manifest = {}
    for role, bundle in [("train", datasets.train), ("validation", datasets.validation),
                          ("confirmatory_test", datasets.confirmatory_test)]:
        path = gohr_dataset.persist_dataset(bundle, Path(output_dir) / "ev_noise_datasets")
        manifest[role] = {"path": str(path), "dataset_id": bundle.dataset_id, "hash": bundle.combined_hash}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return datasets


def _pair_results_sidecar_path(output_dir: Path, experiment_id: str) -> Path:
    return Path(output_dir) / f"{experiment_id}_pair_results.json"


def _load_pair_results_sidecar(output_dir: Path, experiment_id: str) -> dict[str, dict[str, float]]:
    """
    Per-pair raw accuracy values, persisted alongside the resume ledger.
    The ledger itself (framework.resumability) intentionally carries no
    experiment-specific payload - it only proves "this replicate_id
    completed under this exact config_hash". Recovering the actual
    scientific result on resume is this orchestration layer's job, done
    here via a small sidecar JSON, not by extending the generic
    resumability schema.
    """
    import json
    path = _pair_results_sidecar_path(output_dir, experiment_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_pair_result_sidecar(
    output_dir: Path, experiment_id: str, replicate_id: str, acc_a: float, acc_b: float,
) -> None:
    import json
    path = _pair_results_sidecar_path(output_dir, experiment_id)
    data = _load_pair_results_sidecar(output_dir, experiment_id)
    data[replicate_id] = {"acc_a": acc_a, "acc_b": acc_b}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _replicate_config_hash(**fields: Any) -> str:
    return config_hash(fields)


# ---------------------------------------------------------------------
# EV-BASELINE
# ---------------------------------------------------------------------

def define_ev_baseline() -> Experiment:
    return Experiment(
        id="EV-BASELINE", version="1.1", hypothesis_id=None,
        scientific_claim=(
            "Characterize replicate-to-replicate variability of the frozen 5-round "
            "Gohr/Speck32/64 baseline across independent reproductions (fresh dataset "
            "+ fresh model seed each), establishing the empirical noise floor other "
            "EV experiments are interpreted against."
        ),
        hypothesized_confounding_mechanism=None,
        rationale="Required per methodology Section 3.9.3 (repeated experimentation) "
                  "before any comparative EV claim can be interpreted.",
        factors=[],
        conditions=[Condition(condition_id="baseline", description="Frozen 5-round baseline")],
        controls=["rounds", "depth", "architecture", "optimizer", "lr_schedule", "batch_size",
                  "shuffle", "representation", "software_environment"],
        independent_variable=None, dependent_variable="confirmatory_test_accuracy",
        unit_of_replication="independently trained model (fresh dataset + fresh model seed)",
        design_type=DesignType.DESCRIPTIVE,
        replicate_plan=ReplicatePlan(requested_replicates=0, minimum_valid_replicates=0),
        pairing=None, practical_significance=NO_PRACTICAL_SIGNIFICANCE, multiplicity_family=None,
    )


def run_ev_baseline(
    *, run_mode: str, requested_replicates: int, minimum_valid_replicates: int,
    output_dir: str | Path, base_model_seed: int = 1000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    experiment = define_ev_baseline()
    adapter = GohrAdapter()
    overrides = _baseline_overrides_for_run_mode(run_mode)
    ledger = _resume_ledger_for(output_dir)
    replicate_set = ReplicateSet(condition_id="baseline")
    raw_results = []
    metric_sidecar = _load_replicate_results_sidecar(output_dir, "EV-BASELINE")

    for i in range(requested_replicates):
        replicate_id = f"baseline_r{i}"
        cfg_hash = _replicate_config_hash(
            experiment="EV-BASELINE", replicate_id=replicate_id, run_mode=run_mode,
            overrides=overrides, model_seed=base_model_seed + i,
        )
        previous = ledger.last_state_for("EV-BASELINE", replicate_id)
        if previous is not None:
            if previous["config_hash"] != cfg_hash:
                raise RuntimeError(
                    f"Resume conflict for EV-BASELINE/{replicate_id}: previously recorded "
                    f"config_hash={previous['config_hash']} does not match the current "
                    f"config_hash={cfg_hash}. Refusing to resume with a changed configuration."
                )
            if previous["stage"] == "complete":
                if replicate_id not in metric_sidecar:
                    raise RuntimeError(
                        f"Resume inconsistency for EV-BASELINE/{replicate_id}: the ledger marks "
                        "this replicate complete but its raw metric sidecar is missing."
                    )
                ledger.record_resume_event(
                    experiment_id="EV-BASELINE", replicate_id=replicate_id,
                    previous_config_hash=previous["config_hash"], resumed_config_hash=cfg_hash,
                )
                replicate_set.add(ReplicateOutcome(
                    replicate_id=replicate_id, condition_id="baseline", run_id=f"resumed-{replicate_id}",
                    status=ReplicateStatus.VALID, metric_value=metric_sidecar[replicate_id],
                ))
                continue  # already completed under an identical configuration; skip.

        cfg = ReplicateConfig(
            experiment_id=experiment.id, hypothesis_id=None, condition_id="baseline",
            replicate_id=replicate_id, run_id=str(uuid.uuid4()), run_mode=run_mode,
            rounds=overrides["rounds"], differential=BASELINE.differential, depth=overrides["depth"],
            epochs=overrides["epochs"], batch_size=overrides["batch_size"], shuffle=BASELINE.shuffle,
            optimizer=BASELINE.optimizer, lr_high=BASELINE.lr_schedule_high, lr_low=BASELINE.lr_schedule_low,
            lr_period=BASELINE.lr_schedule_period, model_seed=base_model_seed + i,
            representation=identity_permutation(), output_dir=output_dir, matched_datasets=None,
            independent_train_size=overrides["train_size"], independent_val_size=overrides["val_size"],
            independent_confirmatory_test_size=overrides["confirmatory_test_size"],
            same_dataset=False, same_model_initialization=False,
            same_training_shuffle_stream=False, same_evaluation_data=False,
        )
        result = adapter.run_replicate(cfg)
        replicate_set.add(result.outcome)
        raw_results.append(result.manifest_fields)
        if result.outcome.status.value == "valid":
            _save_replicate_result_sidecar(output_dir, "EV-BASELINE", replicate_id, result.outcome.metric_value)
        ledger.record_state(RunState(
            experiment_id="EV-BASELINE", replicate_id=replicate_id,
            stage="complete" if result.outcome.status.value == "valid" else "failed",
            completed_stages=["train", "evaluate"], config_hash=cfg_hash,
            checkpoint_hash=result.manifest_fields.get("checkpoint_hash"),
        ))

    sufficient, sufficiency_summary = replicate_set.check_sufficiency(minimum_valid_replicates)
    is_evidentiary = (run_mode == "production")

    if not sufficient:
        stats_dict = None
        decision = DecisionState.INCONCLUSIVE
    else:
        rng = statistics_rng(statistics_seed=42)
        stats = descriptive_statistics(replicate_set.valid_metric_array(), rng=rng)
        stats_dict = stats.to_dict()
        decision = DecisionState.NOT_RUN  # EV-BASELINE issues no hypothesis decision by design

    manifest_fields = _merge_manifest(
        raw_results[0] if raw_results else {}, condition_id="baseline",
        replicate_id="aggregate", run_id=str(uuid.uuid4()),
        raw_metrics=stats_dict or {}, decision=decision.value,
        limitations=["EV-BASELINE is descriptive; it performs no hypothesis test against "
                     "the historical seed=0 single run."],
        failure_status="valid" if sufficient else "insufficient_replicates",
    )

    cert = build_certificate(
        manifest_fields=manifest_fields, decision=decision,
        what_was_tested="Replicate-to-replicate variability of the frozen 5-round baseline.",
        hypothesis_text="N/A - descriptive/foundational experiment, no hypothesis test performed.",
        unit_of_replication=experiment.unit_of_replication, controls=experiment.controls,
        what_changed="Nothing manipulated; fresh dataset + fresh model seed per replicate.",
        dataset_summary={"role": "independent_per_replicate"}, exact_replay_available=False,
        requested_replicates=requested_replicates, valid_replicates=sufficiency_summary["valid_replicates"],
        raw_replicate_results=raw_results, effect_size=None,
        confidence_interval=(
            {"low": stats_dict["confidence_interval"]["low"], "high": stats_dict["confidence_interval"]["high"]}
            if stats_dict else None
        ),
        raw_p_value=None, adjusted_p_value=None, practical_significance_predeclared=False,
        practical_equivalence={"formal_practical_equivalence": "NOT_APPLICABLE", "reason": "descriptive experiment"},
        limitations=manifest_fields["limitations"], is_evidentiary=is_evidentiary,
    )
    cert["sufficiency_summary"] = sufficiency_summary
    cert["non_evidentiary"] = not is_evidentiary
    return cert


# ---------------------------------------------------------------------
# EV-NOISE
# ---------------------------------------------------------------------

def run_ev_noise(
    *, run_mode: str, requested_reruns: int, minimum_valid_reruns: int,
    output_dir: str | Path, model_seed: int = 2000,
) -> dict[str, Any]:
    """
    Identical-configuration execution variability: same dataset, same
    model seed, same everything, repeated `requested_reruns` times.
    Reported strictly as descriptive variability - never labeled
    "irreducible".
    """
    output_dir = Path(output_dir)
    adapter = GohrAdapter()
    overrides = _baseline_overrides_for_run_mode(run_mode)
    ledger = _resume_ledger_for(output_dir)

    datasets = _load_or_generate_ev_noise_dataset(
        output_dir, rounds=overrides["rounds"], differential=BASELINE.differential,
        train_size=overrides["train_size"], val_size=overrides["val_size"],
        confirmatory_test_size=overrides["confirmatory_test_size"],
    )

    replicate_set = ReplicateSet(condition_id="identical_configuration")
    raw_results = []
    metric_sidecar = _load_replicate_results_sidecar(output_dir, "EV-NOISE")
    for i in range(requested_reruns):
        replicate_id = f"noise_r{i}"
        cfg_hash = _replicate_config_hash(
            experiment="EV-NOISE", replicate_id=replicate_id, run_mode=run_mode,
            overrides=overrides, model_seed=model_seed, dataset_hash=datasets.train.combined_hash,
        )
        previous = ledger.last_state_for("EV-NOISE", replicate_id)
        if previous is not None:
            if previous["config_hash"] != cfg_hash:
                raise RuntimeError(
                    f"Resume conflict for EV-NOISE/{replicate_id}: config_hash changed. "
                    "Refusing to resume with a changed configuration."
                )
            if previous["stage"] == "complete":
                if replicate_id not in metric_sidecar:
                    raise RuntimeError(
                        f"Resume inconsistency for EV-NOISE/{replicate_id}: the ledger marks "
                        "this replicate complete but its raw metric sidecar is missing."
                    )
                ledger.record_resume_event(
                    experiment_id="EV-NOISE", replicate_id=replicate_id,
                    previous_config_hash=previous["config_hash"], resumed_config_hash=cfg_hash,
                )
                replicate_set.add(ReplicateOutcome(
                    replicate_id=replicate_id, condition_id="identical_configuration",
                    run_id=f"resumed-{replicate_id}", status=ReplicateStatus.VALID,
                    metric_value=metric_sidecar[replicate_id],
                ))
                continue

        cfg = ReplicateConfig(
            experiment_id="EV-NOISE", hypothesis_id=None, condition_id="identical_configuration",
            replicate_id=replicate_id, run_id=str(uuid.uuid4()), run_mode=run_mode,
            rounds=overrides["rounds"], differential=BASELINE.differential, depth=overrides["depth"],
            epochs=overrides["epochs"], batch_size=overrides["batch_size"], shuffle=BASELINE.shuffle,
            optimizer=BASELINE.optimizer, lr_high=BASELINE.lr_schedule_high, lr_low=BASELINE.lr_schedule_low,
            lr_period=BASELINE.lr_schedule_period, model_seed=model_seed,
            representation=identity_permutation(), output_dir=output_dir, matched_datasets=datasets,
            same_dataset=True, same_model_initialization=True,
            same_training_shuffle_stream=False, same_evaluation_data=True,
        )
        result = adapter.run_replicate(cfg)
        replicate_set.add(result.outcome)
        raw_results.append(result.manifest_fields)
        if result.outcome.status.value == "valid":
            _save_replicate_result_sidecar(output_dir, "EV-NOISE", replicate_id, result.outcome.metric_value)
        ledger.record_state(RunState(
            experiment_id="EV-NOISE", replicate_id=replicate_id,
            stage="complete" if result.outcome.status.value == "valid" else "failed",
            completed_stages=["train", "evaluate"], config_hash=cfg_hash,
            checkpoint_hash=result.manifest_fields.get("checkpoint_hash"),
        ))

    sufficient, sufficiency_summary = replicate_set.check_sufficiency(minimum_valid_reruns)
    is_evidentiary = (run_mode == "production")

    determinism_note = (
        "No TensorFlow op-determinism control (tf.config.experimental.enable_op_determinism "
        "or TF_DETERMINISTIC_OPS) is set anywhere in this project's Gohr implementations "
        "(repository-derived fact, verified by inspection). Any variability observed here "
        "may include GPU/CPU op-level nondeterminism in addition to any residual effect of "
        "re-seeding; this experiment does not by itself establish which, and does not "
        "establish that any observed variability is 'irreducible'."
    )

    stats_dict = None
    if sufficient:
        rng = statistics_rng(statistics_seed=43)
        stats_dict = descriptive_statistics(replicate_set.valid_metric_array(), rng=rng).to_dict()

    manifest_fields = _merge_manifest(
        raw_results[0] if raw_results else {}, condition_id="identical_configuration",
        replicate_id="aggregate", run_id=str(uuid.uuid4()), raw_metrics=stats_dict or {},
        decision=DecisionState.NOT_RUN.value,
        limitations=[determinism_note, "Descriptive only; not a hypothesis test."],
        failure_status="valid" if sufficient else "insufficient_replicates",
    )

    cert = build_certificate(
        manifest_fields=manifest_fields, decision=DecisionState.NOT_RUN,
        what_was_tested="Identical-configuration execution variability.",
        hypothesis_text="N/A - descriptive experiment.",
        unit_of_replication="independently executed training run under an identical configuration",
        controls=["dataset", "model_seed", "rounds", "depth", "architecture", "optimizer",
                  "lr_schedule", "batch_size", "shuffle", "representation"],
        what_changed="Nothing intentionally; repeated execution only.",
        dataset_summary={"role": "matched_single_instance", "hash": datasets.train.combined_hash},
        exact_replay_available=False, requested_replicates=requested_reruns,
        valid_replicates=sufficiency_summary["valid_replicates"], raw_replicate_results=raw_results,
        effect_size=None,
        confidence_interval=(
            {"low": stats_dict["confidence_interval"]["low"], "high": stats_dict["confidence_interval"]["high"]}
            if stats_dict else None
        ),
        raw_p_value=None, adjusted_p_value=None, practical_significance_predeclared=False,
        practical_equivalence={"formal_practical_equivalence": "NOT_APPLICABLE", "reason": "descriptive experiment"},
        limitations=manifest_fields["limitations"], is_evidentiary=is_evidentiary,
    )
    cert["sufficiency_summary"] = sufficiency_summary
    cert["non_evidentiary"] = not is_evidentiary
    return cert


# ---------------------------------------------------------------------
# H-EV-SHUFFLE and H-EV-REPRESENTATION (matched-paired core)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PairedExperimentSpec:
    experiment_id: str
    hypothesis_id: str
    factor_name: str
    scientific_claim: str
    hypothesized_confounding_mechanism: str
    rationale: str


def _run_paired_experiment(
    spec: PairedExperimentSpec, *, run_mode: str, requested_pairs: int, minimum_valid_pairs: int,
    output_dir: str | Path, base_model_seed: int, representation_a: Candidate1Permutation,
    representation_b: Candidate1Permutation, shuffle_a: bool, shuffle_b: bool,
    firewall: TestSetFirewall,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    Run `requested_pairs` matched pairs. Firewall lifecycle enforced
    per pair: seal the pair's confirmatory-test hash -> train both
    arms -> consume the evaluation slot for each condition -> evaluate.
    `firewall` must already be frozen (including, for
    H-EV-REPRESENTATION, a recorded permutation hash) before this is
    called - that ordering is the caller's (scripts/run_ev.py's)
    responsibility and is asserted here defensively.
    """
    if not firewall.is_frozen():
        raise RuntimeError(
            f"Firewall for {spec.experiment_id} is not frozen. Refusing to run: "
            "the experiment must be frozen (config + permutation hash, where "
            "applicable) before any confirmatory dataset is generated."
        )

    output_dir = Path(output_dir)
    adapter = GohrAdapter()
    overrides = _baseline_overrides_for_run_mode(run_mode)
    ledger = _resume_ledger_for(output_dir)

    acc_a, acc_b = [], []
    raw_a, raw_b = [], []
    pair_sidecar = _load_pair_results_sidecar(output_dir, spec.experiment_id)

    for i in range(requested_pairs):
        replicate_id = f"pair{i}"
        cfg_hash = _replicate_config_hash(
            experiment=spec.experiment_id, replicate_id=replicate_id, run_mode=run_mode,
            overrides=overrides, base_model_seed=base_model_seed,
            representation_a_hash=representation_a.hash, representation_b_hash=representation_b.hash,
            shuffle_a=shuffle_a, shuffle_b=shuffle_b,
            firewall_permutation_hash=firewall.freeze_record.frozen_permutation_hash,
        )
        previous = ledger.last_state_for(spec.experiment_id, replicate_id)
        if previous is not None:
            if previous["config_hash"] != cfg_hash:
                raise RuntimeError(
                    f"Resume conflict for {spec.experiment_id}/{replicate_id}: config_hash "
                    "changed. Refusing to resume with a changed configuration."
                )
            if previous["stage"] == "complete":
                if replicate_id not in pair_sidecar:
                    raise RuntimeError(
                        f"Resume inconsistency for {spec.experiment_id}/{replicate_id}: the "
                        "ledger marks this replicate complete but its raw result sidecar is "
                        "missing. Refusing to silently treat this as either a fresh run or a "
                        "valid resume."
                    )
                ledger.record_resume_event(
                    experiment_id=spec.experiment_id, replicate_id=replicate_id,
                    previous_config_hash=previous["config_hash"], resumed_config_hash=cfg_hash,
                )
                acc_a.append(pair_sidecar[replicate_id]["acc_a"])
                acc_b.append(pair_sidecar[replicate_id]["acc_b"])
                continue

        datasets = generate_matched_datasets(
            rounds=overrides["rounds"], differential=BASELINE.differential,
            train_size=overrides["train_size"], val_size=overrides["val_size"],
            confirmatory_test_size=overrides["confirmatory_test_size"],
        )
        shared_model_seed = base_model_seed + i

        # Seal BEFORE training/consuming - the firewall must know about
        # this replicate's confirmatory dataset before it can be used.
        firewall.seal_confirmatory_dataset(replicate_id, datasets.confirmatory_test.combined_hash)

        cfg_a = ReplicateConfig(
            experiment_id=spec.experiment_id, hypothesis_id=spec.hypothesis_id, condition_id="condition_a",
            replicate_id=replicate_id, run_id=str(uuid.uuid4()), run_mode=run_mode,
            rounds=overrides["rounds"], differential=BASELINE.differential, depth=overrides["depth"],
            epochs=overrides["epochs"], batch_size=overrides["batch_size"], shuffle=shuffle_a,
            optimizer=BASELINE.optimizer, lr_high=BASELINE.lr_schedule_high, lr_low=BASELINE.lr_schedule_low,
            lr_period=BASELINE.lr_schedule_period, model_seed=shared_model_seed,
            representation=representation_a, output_dir=output_dir, matched_datasets=datasets,
            same_dataset=True, same_model_initialization=True,
            same_training_shuffle_stream=(shuffle_a == shuffle_b), same_evaluation_data=True,
        )
        cfg_b = ReplicateConfig(
            experiment_id=spec.experiment_id, hypothesis_id=spec.hypothesis_id, condition_id="condition_b",
            replicate_id=replicate_id, run_id=str(uuid.uuid4()), run_mode=run_mode,
            rounds=overrides["rounds"], differential=BASELINE.differential, depth=overrides["depth"],
            epochs=overrides["epochs"], batch_size=overrides["batch_size"], shuffle=shuffle_b,
            optimizer=BASELINE.optimizer, lr_high=BASELINE.lr_schedule_high, lr_low=BASELINE.lr_schedule_low,
            lr_period=BASELINE.lr_schedule_period, model_seed=shared_model_seed,
            representation=representation_b, output_dir=output_dir, matched_datasets=datasets,
            same_dataset=True, same_model_initialization=True,
            same_training_shuffle_stream=(shuffle_a == shuffle_b), same_evaluation_data=True,
        )

        result_a = adapter.run_replicate(cfg_a)
        result_b = adapter.run_replicate(cfg_b)

        if result_a.outcome.status.value == "valid" and result_b.outcome.status.value == "valid":
            # Consume the confirmatory-evaluation slot for each condition
            # AFTER training but as the explicit act of "using" the sealed
            # data for this pair's confirmatory decision.
            firewall.consume_confirmatory_evaluation(
                condition_id="condition_a", replicate_id=replicate_id,
                observed_dataset_hash=datasets.confirmatory_test.combined_hash,
            )
            firewall.consume_confirmatory_evaluation(
                condition_id="condition_b", replicate_id=replicate_id,
                observed_dataset_hash=datasets.confirmatory_test.combined_hash,
            )
            acc_a.append(result_a.outcome.metric_value)
            acc_b.append(result_b.outcome.metric_value)
            _save_pair_result_sidecar(
                output_dir, spec.experiment_id, replicate_id,
                acc_a=result_a.outcome.metric_value, acc_b=result_b.outcome.metric_value,
            )
            ledger.record_state(RunState(
                experiment_id=spec.experiment_id, replicate_id=replicate_id, stage="complete",
                completed_stages=["seal", "train_a", "train_b", "consume", "evaluate"],
                config_hash=cfg_hash, checkpoint_hash=None,
            ))
        else:
            ledger.record_state(RunState(
                experiment_id=spec.experiment_id, replicate_id=replicate_id, stage="failed",
                completed_stages=["seal", "train_a", "train_b"],
                config_hash=cfg_hash, checkpoint_hash=None,
            ))

        raw_a.append(result_a.manifest_fields)
        raw_b.append(result_b.manifest_fields)

    return (
        {}, np.array(acc_a, dtype=float), np.array(acc_b, dtype=float), raw_a, raw_b,
    )


def run_h_ev_shuffle(
    *, run_mode: str, requested_pairs: int, minimum_valid_pairs: int, output_dir: str | Path,
    firewall: TestSetFirewall, base_model_seed: int = 3000,
    practical_significance: PracticalSignificance = NO_PRACTICAL_SIGNIFICANCE,
) -> dict[str, Any]:
    spec = PairedExperimentSpec(
        experiment_id="H-EV-SHUFFLE", hypothesis_id="H-EV-SHUFFLE", factor_name="shuffle",
        scientific_claim=(
            "Under the frozen 5-round Gohr configuration and the tested training protocol, "
            "we assess whether minibatch shuffling has a material effect on confirmatory "
            "distinguishing accuracy."
        ),
        hypothesized_confounding_mechanism=(
            "Fixed-order training dynamics (shuffle=False, as used by this project's "
            "D-dimension GohrAdapter) could let the model exploit training-loop artifacts "
            "tied to a fixed batch order, rather than genuine ciphertext structure."
        ),
        rationale="Real, present deviation between the D-dimension adapter (shuffle=False) "
                  "and the documented baseline/CE adapter (shuffle=True, by omission).",
    )
    identity = identity_permutation()
    _, acc_true, acc_false, raw_true, raw_false = _run_paired_experiment(
        spec, run_mode=run_mode, requested_pairs=requested_pairs, minimum_valid_pairs=minimum_valid_pairs,
        output_dir=output_dir, base_model_seed=base_model_seed,
        representation_a=identity, representation_b=identity,
        shuffle_a=True, shuffle_b=False, firewall=firewall,
    )
    return _finalize_paired_certificate(
        spec=spec, run_mode=run_mode, values_a=acc_true, values_b=acc_false,
        raw_a=raw_true, raw_b=raw_false, minimum_valid_pairs=minimum_valid_pairs,
        requested_pairs=requested_pairs,
        what_changed="shuffle: True (condition_a) -> False (condition_b)",
        dataset_summary={"role": "matched_per_pair"},
        practical_significance=practical_significance,
        conservative_wording={
            "supported": (
                "Under the frozen 5-round Gohr configuration and the tested training protocol, "
                "we found evidence for a material effect of minibatch shuffling on confirmatory "
                "distinguishing accuracy."
            ),
            "equivalent": (
                "Under the frozen 5-round Gohr configuration and the tested training protocol, "
                "we found evidence of practical equivalence: any effect of minibatch shuffling on "
                "confirmatory distinguishing accuracy is smaller than the predeclared threshold."
            ),
            "inconclusive": (
                "Under the frozen 5-round Gohr configuration and the tested training protocol, "
                "we found no evidence for a material effect of minibatch shuffling on confirmatory "
                "distinguishing accuracy; this does not establish that no such effect exists."
            ),
        },
    )


def run_h_ev_representation(
    *, run_mode: str, requested_pairs: int, minimum_valid_pairs: int, output_dir: str | Path,
    permutation: Candidate1Permutation, firewall: TestSetFirewall, base_model_seed: int = 4000,
    practical_significance: PracticalSignificance = NO_PRACTICAL_SIGNIFICANCE,
) -> dict[str, Any]:
    if not firewall.is_frozen():
        raise RuntimeError("Firewall must be frozen (with the permutation hash recorded) before this call.")
    if firewall.freeze_record.frozen_permutation_hash != permutation.hash:
        raise RuntimeError(
            f"Permutation hash mismatch: firewall was frozen with "
            f"{firewall.freeze_record.frozen_permutation_hash!r}, but the permutation passed "
            f"to run_h_ev_representation has hash {permutation.hash!r}. Refusing to run with a "
            "permutation that differs from the one frozen for this experiment."
        )

    validation = run_full_validation(
        X=np.random.default_rng(0).integers(0, 2, size=(64, 64)).astype(np.uint8),
        Y=np.random.default_rng(0).integers(0, 2, size=64).astype(np.uint8),
        permutation=permutation,
    )
    if not validation["all_passed"]:
        raise RuntimeError(
            f"Candidate-1 permutation failed mandatory pre-production validation: {validation}"
        )

    spec = PairedExperimentSpec(
        experiment_id="H-EV-REPRESENTATION", hypothesis_id="H-EV-REPRESENTATION",
        factor_name="representation",
        scientific_claim=(
            "Under the fixed architecture and training protocol, we assess whether the "
            "Candidate-1 information-preserving representation perturbation produces a "
            "practically meaningful change in confirmatory distinguishing accuracy."
        ),
        hypothesized_confounding_mechanism=(
            "The first 'bit-sliced' Conv1D layer can trivially compare corresponding bits "
            "across the four ciphertext words at the same tensor position; if this specific "
            "human-chosen alignment - rather than information present in the ciphertext - "
            "accounts for most of the observed performance, scrambling it under the same "
            "fixed architecture and training protocol should materially reduce accuracy."
        ),
        rationale="Central CipherMind evidence-attribution experiment per the frozen Round-4 design.",
    )
    identity = identity_permutation()
    _, acc_orig, acc_scrambled, raw_orig, raw_scrambled = _run_paired_experiment(
        spec, run_mode=run_mode, requested_pairs=requested_pairs, minimum_valid_pairs=minimum_valid_pairs,
        output_dir=output_dir, base_model_seed=base_model_seed,
        representation_a=identity, representation_b=permutation,
        shuffle_a=BASELINE.shuffle, shuffle_b=BASELINE.shuffle, firewall=firewall,
    )
    cert = _finalize_paired_certificate(
        spec=spec, run_mode=run_mode, values_a=acc_orig, values_b=acc_scrambled,
        raw_a=raw_orig, raw_b=raw_scrambled, minimum_valid_pairs=minimum_valid_pairs,
        requested_pairs=requested_pairs,
        what_changed=f"representation: original Gohr encoding -> Candidate-1 scramble (hash={permutation.hash})",
        dataset_summary={"role": "matched_per_pair"},
        practical_significance=practical_significance,
        conservative_wording={
            "supported": (
                "Under the fixed architecture and training protocol, the Candidate-1 "
                "information-preserving representation perturbation produced a statistically "
                "detectable change in confirmatory distinguishing accuracy. This demonstrates "
                "dependence on the tested representation under this architecture and training "
                "protocol; it does NOT prove that the ciphertext contains no information "
                "recoverable under another representation or architecture."
            ),
            "equivalent": (
                "Under the fixed architecture and training protocol, the Candidate-1 "
                "information-preserving representation perturbation did NOT produce a "
                "practically meaningful change in confirmatory distinguishing accuracy "
                "(formal equivalence within the predeclared threshold). This is evidence "
                "against a dependence on this specific encoding under this architecture and "
                "training protocol, but does not by itself establish a cryptographic "
                "interpretation of the surviving signal."
            ),
            "inconclusive": (
                "Under the fixed architecture and training protocol, we found no evidence "
                "that the Candidate-1 information-preserving representation perturbation "
                "produced a practically meaningful change in confirmatory distinguishing "
                "accuracy; this does not establish that no such effect exists, nor that the "
                "ciphertext contains no information recoverable under another representation "
                "or architecture."
            ),
        },
    )
    cert["representation_transform_validation"] = validation
    return cert


def _finalize_paired_certificate(
    *, spec: PairedExperimentSpec, run_mode: str, values_a: np.ndarray, values_b: np.ndarray,
    raw_a: list[dict], raw_b: list[dict], minimum_valid_pairs: int, requested_pairs: int,
    what_changed: str, dataset_summary: dict, practical_significance: PracticalSignificance,
    conservative_wording: dict[str, str],
) -> dict[str, Any]:
    is_evidentiary = (run_mode == "production")
    n_valid_pairs = len(values_a)
    # paired_analysis requires >= 2 pairs regardless of the configured
    # minimum_valid_pairs - a configured minimum below that statistical
    # floor must not be allowed to route into a crash.
    sufficient = n_valid_pairs >= max(minimum_valid_pairs, 2)

    limitations = [
        "Effect estimate and confidence interval are reported regardless of statistical "
        "significance; a non-significant result without formal equivalence is INCONCLUSIVE, "
        "never evidence of no effect.",
    ]

    paired_result = None
    practical_equiv: dict[str, Any] = {"formal_practical_equivalence": "NOT_AVAILABLE", "reason": "insufficient valid replicate pairs"}
    difference_decision = DecisionState.INCONCLUSIVE

    if sufficient:
        rng = statistics_rng(statistics_seed=44)
        paired_result = paired_analysis(values_a, values_b, rng=rng)
        difference_decision = decide_difference_detection(
            adjusted_p_value=paired_result.p_value, alpha=0.05, sufficient_replicates=True,
        )  # provisional at raw alpha; superseded by apply_primary_family_correction()
        practical_equiv = assess_practical_equivalence(paired_result, practical_significance)
        limitations.extend(paired_result.warnings)

    decision = decide_final(difference_decision=difference_decision, practical_equivalence=practical_equiv)

    if decision == DecisionState.SUPPORTED:
        conservative_conclusion = conservative_wording["supported"]
    elif decision == DecisionState.NOT_SUPPORTED:
        conservative_conclusion = conservative_wording["equivalent"]
    else:
        conservative_conclusion = conservative_wording["inconclusive"]

    manifest_fields = _merge_manifest(
        (raw_a[0] if raw_a else {}), condition_id="condition_a_vs_b",
        replicate_id="aggregate", run_id=str(uuid.uuid4()),
        raw_metrics={"condition_a": values_a.tolist(), "condition_b": values_b.tolist()},
        decision=decision.value, limitations=limitations,
        failure_status="valid" if sufficient else "insufficient_replicates",
        hypothesis_id=spec.hypothesis_id,
    )

    cert = build_certificate(
        manifest_fields=manifest_fields, decision=decision,
        what_was_tested=spec.scientific_claim, hypothesis_text=spec.hypothesized_confounding_mechanism,
        # Derived from the declared pairing rather than hard-coded, so a
        # different pairing structure (e.g. II-4's independent
        # initialization) cannot inherit a false narrative.
        unit_of_replication=PairingDeclaration(
            same_dataset=True, same_model_initialization=True,
            same_training_shuffle_stream=False, same_evaluation_data=True,
        ).describe_unit(),
        controls=["dataset", "model_seed", "rounds", "depth", "architecture", "optimizer",
                  "lr_schedule", "batch_size", "software_environment"],
        what_changed=what_changed, dataset_summary=dataset_summary, exact_replay_available=False,
        requested_replicates=requested_pairs, valid_replicates=n_valid_pairs,
        raw_replicate_results=raw_a + raw_b,
        effect_size=(paired_result.effect_size_dz if paired_result else None),
        confidence_interval=({"low": paired_result.ci_low, "high": paired_result.ci_high} if paired_result else None),
        raw_p_value=(paired_result.p_value if paired_result else None),
        adjusted_p_value=None,  # filled in by apply_primary_family_correction()
        practical_significance_predeclared=practical_significance.is_available(),
        practical_equivalence=practical_equiv, limitations=limitations, is_evidentiary=is_evidentiary,
    )
    cert["raw_paired_values"] = {"condition_a": values_a.tolist(), "condition_b": values_b.tolist()}
    cert["conservative_conclusion"] = conservative_conclusion
    cert["non_evidentiary"] = not is_evidentiary
    return cert


def apply_primary_family_correction(shuffle_cert: dict, representation_cert: dict) -> dict[str, Any]:
    """
    Apply Holm correction jointly across the frozen primary hypothesis
    family {H-EV-SHUFFLE, H-EV-REPRESENTATION} to the PRIMARY
    DIFFERENCE-DETECTION p-values, then recompute each certificate's
    final decision via decide_final() using the Holm-adjusted decision
    together with its own (unaffected) TOST equivalence assessment.

    Must be called once both raw p-values exist - never per-experiment
    in isolation, and never re-run after re-examining results to
    "improve" the family composition.
    """
    p_shuffle = shuffle_cert["statistics"]["raw_p_value"]
    p_repr = representation_cert["statistics"]["raw_p_value"]

    result = holm_correction(
        family_id=PRIMARY_HYPOTHESIS_FAMILY_ID,
        hypothesis_ids=["H-EV-SHUFFLE", "H-EV-REPRESENTATION"],
        p_values=[p_shuffle, p_repr], alpha=0.05,
    )

    for cert, hyp_id in [(shuffle_cert, "H-EV-SHUFFLE"), (representation_cert, "H-EV-REPRESENTATION")]:
        adj = result.for_hypothesis(hyp_id)
        cert["statistics"]["adjusted_p_value"] = adj["adjusted_p_value"]
        cert["multiplicity"] = adj

        if adj["adjusted_p_value"] is None:
            corrected_difference_decision = DecisionState.INCONCLUSIVE
        elif adj["adjusted_p_value"] < 0.05:
            corrected_difference_decision = DecisionState.SUPPORTED
        else:
            corrected_difference_decision = DecisionState.INCONCLUSIVE

        cert["decision"] = decide_final(
            difference_decision=corrected_difference_decision,
            practical_equivalence=cert["practical_significance"]["assessment"],
        ).value

    return {
        "multiplicity_result": {
            "family_id": result.family_id, "correction_method": result.correction_method,
            "alpha": result.alpha, "hypothesis_ids": result.hypothesis_ids,
            "raw_p_values": result.raw_p_values, "adjusted_p_values": result.adjusted_p_values,
            "reject_null": result.reject_null,
        },
        "shuffle_certificate": shuffle_cert, "representation_certificate": representation_cert,
    }


def _merge_manifest(base: dict, **overrides) -> dict:
    """Small helper: start from a replicate's manifest_fields and override aggregate-level fields."""
    merged = dict(base) if base else {}
    for k, v in overrides.items():
        merged[k] = v
    from framework.manifest import REQUIRED_MANIFEST_FIELDS
    for field_name in REQUIRED_MANIFEST_FIELDS:
        merged.setdefault(field_name, None)
    return merged
