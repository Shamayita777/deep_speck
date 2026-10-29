"""
The four AutoND EV experiments, run by one orchestration loop over framework primitives.

    EV-AUTOND-BASELINE          descriptive; fresh dataset and fresh model seed per replicate
    EV-AUTOND-NOISE             descriptive; one persisted dataset, one seed, identical reruns
    H-EV-AUTOND-TRAIN-ORDER     paired; shuffle=True (A) vs shuffle=False (B): example ORDER only,
                                not label shuffling and not D4
    H-EV-AUTOND-REPRESENTATION  paired; original encoding (A) vs the frozen Candidate-1 within-word
                                bit permutation (B) on train, validation and test features alike

Paired differences are B - A. Within a pair both arms use the same persisted dataset and the same
seed, and the realised initial-weight hashes must be identical or the pair is refused. Per unit:
    seal the confirmatory test hash -> train every arm -> verify pairing -> consume the single-use
    evaluation slot -> evaluate -> record
An interrupted unit is re-run from scratch on its persisted data with the same seed; a completed or
failed unit is never re-run; a unit whose confirmatory data were consumed but not recorded is refused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from autond import adapter as A
from autond.baseline import (
    EV_SPECIFIC_CHANGES, PRIMARY_FAMILY, ConfigError, design_hash, production_hash, run_plan,
)
from framework.certificate import (
    DecisionState, assess_practical_equivalence, build_certificate, decide_difference_detection, decide_final,
    require_confirmatory_certificate,
)
from framework.experiment import (
    Condition, DesignType, Experiment, Factor, PairingDeclaration, PracticalSignificance, ReplicatePlan,
)
from framework.failures import FailureReason, ReplicateOutcome, ReplicateStatus
from framework.firewall import FirewallViolation, FreezeRecord, TestSetFirewall
from framework.manifest import REQUIRED_MANIFEST_FIELDS
from framework.multiplicity import holm_correction
from framework.power import simulate_joint_primary_family, validate_replicate_plan_against_power
from framework.provenance import config_hash, sha256_file, software_provenance, utc_timestamp
from framework.replication import ReplicateSet
from framework.resumability import ResumeLedger, ResumeViolation, RunState
from framework.seeds import statistics_rng
from framework.statistics import descriptive_statistics, paired_analysis
from gohr.representation import Candidate1Permutation, apply_candidate1, generate_candidate1_permutation

FAMILY_ID = "ev_autond_primary_family_v1"
EQUIVALENCE_FAMILY_ID = "ev_autond_equivalence_family_v1"
VARIANCE_ARTIFACT = "autond-ev-calibration-variance-1"
POWER_PLAN_ARTIFACT = "autond-ev-power-plan-1"


@dataclass(frozen=True)
class Arm:
    condition_id: str
    shuffle: bool
    permuted: bool


@dataclass(frozen=True)
class Spec:
    experiment_id: str
    unit: str                      # replicate | rerun | pair
    shared_data: bool              # one dataset for every unit (EV-NOISE)
    shared_seed: bool              # one seed for every unit (EV-NOISE)
    arms: tuple[Arm, ...]
    statistics_seed: int
    claim: str
    mechanism: Optional[str] = None
    what_changed: str = "nothing manipulated"
    wording: Optional[dict] = None

    @property
    def paired(self) -> bool:
        return len(self.arms) == 2


ORIGINAL = Arm("original", shuffle=True, permuted=False)
SPECS = {s.experiment_id: s for s in (
    Spec("EV-AUTOND-BASELINE", "replicate", False, False, (ORIGINAL,), 51,
         "Replicate-to-replicate variability of the round-5 AutoND DBitNet protocol across independent "
         "training runs (fresh dataset and fresh model seed per replicate)."),
    Spec("EV-AUTOND-NOISE", "rerun", True, True, (ORIGINAL,), 52,
         "Execution variability of the round-5 AutoND DBitNet protocol: identical persisted data, identical "
         "seed (verified identical initial weights) and identical configuration, rerun."),
    Spec("H-EV-AUTOND-TRAIN-ORDER", "pair", False, False,
         (Arm("condition_a", shuffle=True, permuted=False), Arm("condition_b", shuffle=False, permuted=False)), 53,
         "Under the round-5 AutoND DBitNet protocol, whether training-example ordering (Keras per-epoch "
         "shuffling vs a fixed order) has a material effect on confirmatory distinguishing accuracy.",
         "Training in a fixed example order could let DBitNet fit order-specific training dynamics rather than "
         "ciphertext structure; upstream relies on model.fit's default shuffle=True.",
         "shuffle: True (condition_a) -> False (condition_b)",
         {"supported": "Training-example ordering had a material effect on confirmatory accuracy under this "
                       "protocol.",
          "equivalent": "Any effect of training-example ordering on confirmatory accuracy is smaller than the "
                        "predeclared margin under this protocol.",
          "inconclusive": "No material effect of training-example ordering was established under this protocol; "
                          "this does not establish that none exists."}),
    Spec("H-EV-AUTOND-REPRESENTATION", "pair", False, False,
         (Arm("condition_a", shuffle=True, permuted=False), Arm("condition_b", shuffle=True, permuted=True)), 54,
         "Under the fixed DBitNet architecture and round-5 protocol, whether a fixed, information-preserving "
         "within-word bit permutation of the input produces a practically meaningful change in confirmatory "
         "distinguishing accuracy.",
         "DBitNet's dilated convolutions relate input positions at fixed distances; if the natural alignment of "
         "corresponding bits across the four ciphertext words, rather than information in the ciphertext pair, "
         "drives performance, scrambling bit positions within each word should materially reduce accuracy.",
         "representation: original encoding -> Candidate-1 within-word bit permutation",
         {"supported": "The information-preserving permutation changed confirmatory accuracy materially: the result "
                       "depends on the tested encoding under this architecture and protocol. It does not show that "
                       "the ciphertext lacks information recoverable another way.",
          "equivalent": "The information-preserving permutation did not change confirmatory accuracy by more than "
                        "the predeclared margin under this architecture and protocol.",
          "inconclusive": "No practically meaningful effect of the permutation was established under this "
                          "architecture and protocol; this does not establish that none exists."}),
)}


def frozen_permutation(config: dict) -> Candidate1Permutation:
    """The Candidate-1 permutation, regenerated from its declared seed and bound to its declared hash."""
    rep = config["representation"]
    permutation = generate_candidate1_permutation(np.random.default_rng(rep["generation_seed"]))
    if permutation.hash != rep["expected_sha256"]:
        raise ConfigError(f"permutation from seed {rep['generation_seed']} has hash {permutation.hash}, "
                          f"not the declared {rep['expected_sha256']}")
    return permutation


def practical_significance(config: dict) -> PracticalSignificance:
    d = config["decisions"]
    eps = d.get("practical_threshold")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        return PracticalSignificance(threshold=None, predeclared=False, justification="not yet declared")
    return PracticalSignificance(threshold=float(eps), predeclared=True,
                                 justification=d.get("practical_threshold_justification"))


def tier_hash(config: dict, mode: str) -> str:
    if mode == "production":
        return production_hash(config)
    if mode == "calibration":
        return design_hash(config)
    return config_hash({"protocol": config["protocol"], "representation": config["representation"],
                        "smoke": config["smoke"]})


def definition(spec: Spec, plan: dict, config: dict) -> Experiment:
    factors = []
    if spec.paired:
        factor = "shuffle" if spec.experiment_id.endswith("TRAIN-ORDER") else "representation"
        factors = [Factor(factor, essential=False, rationale=spec.mechanism or "")]
    return Experiment(
        id=spec.experiment_id, version="1", hypothesis_id=spec.experiment_id if spec.paired else None,
        scientific_claim=spec.claim, hypothesized_confounding_mechanism=spec.mechanism,
        rationale="AutoND counterpart of the EV experiment of the same role.", factors=factors,
        conditions=[Condition(a.condition_id, f"shuffle={a.shuffle}, permuted={a.permuted}",
                              {"shuffle": a.shuffle, "permuted": a.permuted}) for a in spec.arms],
        controls=["dataset (per unit)", "model seed (per unit)", "round 5", "DBitNet", "Adam(amsgrad=True)",
                  "MSE", "epochs", "batch size", "software environment"],
        independent_variable=factors[0].name if factors else None, dependent_variable="confirmatory_test_accuracy",
        unit_of_replication=(_pairing(spec).describe_unit() if spec.paired
                             else f"one independently trained model ({spec.unit})"),
        design_type=DesignType.MATCHED_PAIRED if spec.paired else DesignType.DESCRIPTIVE,
        replicate_plan=ReplicatePlan(plan["units"], plan["minimum_valid"]),
        pairing=_pairing(spec) if spec.paired else None,
        practical_significance=practical_significance(config) if spec.paired else PracticalSignificance(None, False),
        multiplicity_family=FAMILY_ID if spec.paired else None)


def _pairing(spec: Spec) -> PairingDeclaration:
    # Same seed, but the shuffle stream is not verified identical, so it is not declared shared.
    return PairingDeclaration(same_dataset=True, same_model_initialization=True,
                              same_training_shuffle_stream=False, same_evaluation_data=True)


# ============================================================ orchestration


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _arm_partitions(parts: dict, arm: Arm, permutation: Optional[Candidate1Permutation]) -> dict:
    if not arm.permuted:
        return parts
    return {role: A.Partition(apply_candidate1(p.X, permutation), p.Y) for role, p in parts.items()}


def _load_firewall(path: Path, spec: Spec, unit_hash: str, plan: dict, config: dict,
                   permutation: Optional[Candidate1Permutation]) -> TestSetFirewall:
    permutation_hash = permutation.hash if any(a.permuted for a in spec.arms) else None
    if path.is_file():
        firewall = TestSetFirewall.load(path)
        if (firewall.freeze_record is None or firewall.freeze_record.frozen_config_hash != unit_hash
                or firewall.freeze_record.frozen_permutation_hash != permutation_hash):
            raise ResumeViolation(f"{path}: persisted firewall was frozen under a different configuration")
        return firewall
    firewall = TestSetFirewall(experiment_id=spec.experiment_id)
    firewall.freeze(FreezeRecord(
        frozen_config_hash=unit_hash, frozen_permutation_hash=permutation_hash,
        frozen_replicate_plan_hash=config_hash(plan),
        frozen_statistical_plan_hash=config_hash(config["decisions"]), frozen_at_utc=utc_timestamp()))
    firewall.save(path)
    return firewall


def run_experiment(experiment_id: str, *, config: dict, mode: str, output_dir: Path, binding,
                   permutation: Optional[Candidate1Permutation] = None, provenance: Optional[dict] = None) -> dict:
    """Run (or resume) every unit of one experiment in one tier and return its certificate."""
    spec = SPECS[experiment_id]
    plan = run_plan(config, experiment_id, mode)
    if any(a.permuted for a in spec.arms) and (permutation is None
                                               or permutation.hash != frozen_permutation(config).hash):
        raise ConfigError(f"{experiment_id} requires the frozen Candidate-1 permutation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_hash = config_hash({"tier": tier_hash(config, mode), "experiment": experiment_id, "mode": mode})
    ledger = ResumeLedger(output_dir / "resume_ledger.jsonl")
    results_path = output_dir / "unit_results.json"
    results: dict = json.loads(results_path.read_text()) if results_path.is_file() else {}
    firewall_path = output_dir / "firewall.json"
    firewall = _load_firewall(firewall_path, spec, experiment_hash, plan, config, permutation) if spec.paired else None

    for i in range(plan["units"]):
        unit_id = f"{spec.unit}{i:03d}"
        seed = plan["seed"] if spec.shared_seed else plan["seed"] + i
        unit_hash = config_hash({"experiment": experiment_hash, "unit": unit_id, "seed": seed})
        previous = ledger.last_state_for(experiment_id, unit_id)
        if previous is not None:
            if previous["config_hash"] != unit_hash:
                raise ResumeViolation(f"{experiment_id}/{unit_id}: configuration changed since it was started")
            if previous["stage"] in ("complete", "failed"):
                if unit_id not in results:
                    raise ResumeViolation(f"{experiment_id}/{unit_id}: recorded {previous['stage']} but its "
                                          "result is missing")
                ledger.record_resume_event(experiment_id=experiment_id, replicate_id=unit_id,
                                           previous_config_hash=previous["config_hash"], resumed_config_hash=unit_hash)
                continue
            if previous["stage"] == "consumed":
                raise FirewallViolation(f"{experiment_id}/{unit_id}: its confirmatory data were consumed but the "
                                        "result was never recorded; refusing to evaluate them again")
        ledger.record_state(RunState(experiment_id, unit_id, "started", [], unit_hash, None))
        data_file = output_dir / "datasets" / ("shared.npz" if spec.shared_data else f"{unit_id}.npz")
        parts = A.load_or_generate(binding, data_file, plan["sizes"])
        if firewall is not None:
            firewall.seal_confirmatory_dataset(unit_id, parts["test"].sha256)
            firewall.save(firewall_path)

        arms = []
        for arm in spec.arms:
            data = _arm_partitions(parts, arm, permutation)
            model = A.build_model(binding, seed)
            initial = A.weights_sha256(model)
            training = A.train(model, data["train"], data["validation"], epochs=plan["epochs"],
                               batch_size=plan["batch_size"], shuffle=arm.shuffle)
            arms.append((arm, model, data, {
                "condition_id": arm.condition_id, "model_seed": seed, "shuffle": arm.shuffle,
                "permutation_sha256": permutation.hash if arm.permuted else None,
                "dataset_sha256": {r: p.sha256 for r, p in parts.items()},
                "input_sha256": {r: p.sha256 for r, p in data.items()},
                "initial_weight_sha256": initial, "history": training.history,
                "failure": training.failure.value if training.failure else None}))

        initial_hashes = {rec["initial_weight_sha256"] for *_, rec in arms}
        reference = results.get("_reference_initial_weight_sha256") if spec.shared_seed else None
        if len(initial_hashes) != 1 or (reference is not None and reference not in initial_hashes):
            raise RuntimeError(f"{experiment_id}/{unit_id}: arms declared to share their initialisation have different "
                               f"realised initial weights {sorted(initial_hashes)} (reference {reference}); refusing")
        if spec.shared_seed:
            results["_reference_initial_weight_sha256"] = initial_hashes.pop()

        valid = all(rec["failure"] is None for *_, rec in arms)
        if valid:
            if firewall is not None:
                for arm, *_ in arms:
                    firewall.consume_confirmatory_evaluation(condition_id=arm.condition_id, replicate_id=unit_id,
                                                             observed_dataset_hash=parts["test"].sha256)
                firewall.save(firewall_path)
                ledger.record_state(RunState(experiment_id, unit_id, "consumed", ["train", "consume"], unit_hash, None))
            for arm, model, data, rec in arms:
                rec["confirmatory_accuracy"] = A.accuracy(model, data["test"])
                model_path = output_dir / "models" / f"{unit_id}_{arm.condition_id}.keras"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(model_path)
                rec["final_model_sha256"] = sha256_file(model_path)
        results[unit_id] = {"status": "valid" if valid else "failed", "arms": [rec for *_, rec in arms]}
        _write_json(results_path, results)
        ledger.record_state(RunState(experiment_id, unit_id, "complete" if valid else "failed",
                                     ["train", "evaluate"] if valid else ["train"], unit_hash, None))

    return _certificate(spec, plan, config, mode, results, output_dir, experiment_hash,
                        firewall.to_dict() if firewall else None, permutation, provenance or {})


# ============================================================ certificates


def _outcomes(spec: Spec, results: dict, plan: dict) -> dict[str, ReplicateSet]:
    sets = {arm.condition_id: ReplicateSet(arm.condition_id) for arm in spec.arms}
    for i in range(plan["units"]):
        unit_id = f"{spec.unit}{i:03d}"
        entry = results.get(unit_id)
        for k, arm in enumerate(spec.arms):
            if entry is None:
                outcome = ReplicateOutcome(unit_id, arm.condition_id, unit_id, ReplicateStatus.NOT_RUN)
            elif entry["status"] == "valid":
                outcome = ReplicateOutcome(unit_id, arm.condition_id, unit_id, ReplicateStatus.VALID,
                                           metric_value=entry["arms"][k]["confirmatory_accuracy"])
            else:
                reason = entry["arms"][k]["failure"] or "invalid_metric"
                outcome = ReplicateOutcome(unit_id, arm.condition_id, unit_id, ReplicateStatus.FAILED,
                                           failure_reason=FailureReason(reason),
                                           failure_detail="this arm failed" if entry["arms"][k]["failure"]
                                           else "the other arm of the pair failed")
            sets[arm.condition_id].add(outcome)
    return sets


def _certificate(spec, plan, config, mode, results, output_dir, experiment_hash, firewall, permutation, provenance):
    sets = _outcomes(spec, results, plan)
    first = sets[spec.arms[0].condition_id]
    sufficient, summary = first.check_sufficiency(max(plan["minimum_valid"], 2))
    rng = statistics_rng(spec.statistics_seed)
    ps = practical_significance(config) if spec.paired else PracticalSignificance(None, False)
    paired = None
    assessment: dict = {"formal_practical_equivalence": "NOT_APPLICABLE", "reason": "descriptive experiment"}
    decision = DecisionState.NOT_RUN
    statistics_block: dict = {}
    if spec.paired:
        a, b = (sets[arm.condition_id].valid_metric_array() for arm in spec.arms)
        assessment = {"formal_practical_equivalence": "NOT_AVAILABLE", "reason": "insufficient valid pairs"}
        decision = DecisionState.INCONCLUSIVE
        if sufficient:
            paired = paired_analysis(a, b, rng=rng)
            assessment = assess_practical_equivalence(paired, ps)
            provisional = decide_difference_detection(adjusted_p_value=paired.p_value, alpha=0.05,
                                                      sufficient_replicates=True)
            decision = decide_final(difference_decision=provisional, practical_equivalence=assessment,
                                    confidence_interval={"low": paired.ci_low, "high": paired.ci_high},
                                    epsilon=ps.threshold)
            statistics_block = paired.to_dict()
        values = {"condition_a": a.tolist(), "condition_b": b.tolist()}
    elif sufficient:
        statistics_block = descriptive_statistics(first.valid_metric_array(), rng=rng).to_dict()
        values = {"accuracy": first.valid_metric_array().tolist()}
    else:
        values = {"accuracy": first.valid_metric_array().tolist()}

    evidentiary = mode == "production"
    limitations = list(EV_SPECIFIC_CHANGES) + [
        "a non-significant difference without formal equivalence is INCONCLUSIVE, never evidence of no effect",
        "decisions in a per-experiment certificate are provisional (raw alpha); the final paired decisions come "
        "from the AutoND primary-family Holm correction",
        "exact replay is unavailable: datasets come from os.urandom; the persisted arrays are the reference",
    ] + (paired.warnings if paired else [])
    manifest = {f: None for f in REQUIRED_MANIFEST_FIELDS} | {
        "experiment_id": spec.experiment_id, "hypothesis_id": spec.experiment_id if spec.paired else None,
        "experiment_version": "1", "methodology_version": "ciphermind-methodology-v1", "cipher": A.CIPHER,
        "task": "neural_ciphertext_distinguisher", "rounds": A.ROUNDS, "differential": hex(A.DELTA_PLAIN),
        "architecture": "dbitnet", "depth": None, "dataset_id": "autond-speck3264", "dataset_role":
        "shared" if spec.shared_data else ("matched_per_pair" if spec.paired else "independent_per_replicate"),
        "dataset_size": plan["sizes"], "dataset_hash": {u: r["arms"][0]["dataset_sha256"] for u, r in results.items()
                                                        if not u.startswith("_")},
        "dataset_generation_method": "os.urandom via main.make_train_data", "exact_replay_available": False,
        "model_seed": {"base": plan["seed"], "shared_across_units": spec.shared_seed}, "dataset_seed": None,
        "shuffle_seed_if_available": None, "evaluation_seed_if_available": None,
        "statistics_seed": spec.statistics_seed, "software_versions": software_provenance(),
        "condition": [a.condition_id for a in spec.arms],
        "independent_variable": definition(spec, plan, config).independent_variable,
        "dependent_variable": "confirmatory_test_accuracy", "controls": definition(spec, plan, config).controls,
        "replicate_id": "aggregate", "run_id": experiment_hash, "raw_metrics": values,
        "alpha": 0.05 if spec.paired else None, "practical_threshold": ps.threshold,
        "practical_threshold_status": "PREDECLARED" if ps.is_available() else "NOT_AVAILABLE",
        "decision": decision.value, "warnings": paired.warnings if paired else [], "limitations": limitations,
        "failure_status": "valid" if sufficient else "insufficient_replicates",
        "timestamps": {"certified": utc_timestamp()}, "config_hash": experiment_hash}
    unit_records = {u: r for u, r in results.items() if not u.startswith("_")}
    cert = build_certificate(
        manifest_fields=manifest, decision=decision, what_was_tested=spec.claim,
        hypothesis_text=spec.mechanism or "N/A - descriptive experiment, no hypothesis test.",
        unit_of_replication=definition(spec, plan, config).unit_of_replication,
        controls=manifest["controls"], what_changed=spec.what_changed,
        dataset_summary={"role": manifest["dataset_role"], "sizes": plan["sizes"]}, exact_replay_available=False,
        requested_replicates=plan["units"], valid_replicates=summary["valid_replicates"],
        raw_replicate_results=[unit_records[u] for u in sorted(unit_records)],
        effect_size=paired.effect_size_dz if paired else None,
        confidence_interval=({"low": paired.ci_low, "high": paired.ci_high} if paired else
                             (statistics_block.get("confidence_interval") if statistics_block else None)),
        raw_p_value=paired.p_value if paired else None, adjusted_p_value=None,
        practical_significance_predeclared=ps.is_available(), practical_equivalence=assessment,
        limitations=limitations, is_evidentiary=evidentiary)
    cert.update({
        "experiment_id": spec.experiment_id, "run_mode": mode, "non_evidentiary": not evidentiary,
        "calibration": mode == "calibration", "sufficient": sufficient, "sufficiency_summary": summary,
        "experiment_definition": definition(spec, plan, config).to_dict(), "statistics_detail": statistics_block,
        "design_hash": design_hash(config),
        "production_hash": production_hash(config) if mode == "production" else None,
        "experiment_hash": experiment_hash, "plan": plan, "firewall": firewall,
        "resume_ledger_sha256": sha256_file(output_dir / "resume_ledger.jsonl"),
        "permutation": (permutation.to_dict() if permutation is not None and any(a.permuted for a in spec.arms)
                        else None),
        "provenance": provenance, "conservative_wording": spec.wording})
    if spec.paired:
        cert["raw_paired_values"] = values
        cert["initial_weight_hashes"] = {u: [a["initial_weight_sha256"] for a in r["arms"]]
                                         for u, r in unit_records.items()}
    return cert


# ============================================================ primary family


def holm_family(certificates: dict[str, dict]) -> dict:
    """Both Holm families across the AutoND primary family, then the framework's final decision rule.

    A hypothesis without a p-value (too few valid pairs) enters its family at p = 1, so the other
    hypothesis is never tested at a laxer level than the full two-member Holm procedure allows.
    """
    if set(certificates) != set(PRIMARY_FAMILY):
        raise ValueError(f"the AutoND primary family is exactly {PRIMARY_FAMILY}")
    hyps = list(PRIMARY_FAMILY)

    def p_or_one(value):
        return 1.0 if value is None else float(value)

    raw_diff = [certificates[h]["statistics"]["raw_p_value"] for h in hyps]
    raw_tost = [(certificates[h]["practical_significance"]["assessment"].get("tost") or {}).get("p_tost") for h in hyps]
    diff = holm_correction(family_id=FAMILY_ID, hypothesis_ids=hyps, p_values=[p_or_one(p) for p in raw_diff])
    equivalence = holm_correction(family_id=EQUIVALENCE_FAMILY_ID, hypothesis_ids=hyps,
                                  p_values=[p_or_one(p) for p in raw_tost])
    out = {"family_id": FAMILY_ID, "hypotheses": hyps, "raw_difference_p_values": raw_diff,
           "raw_tost_p_values": raw_tost, "missing_p_values_entered_as": 1.0, "certificates": {}}
    for h in hyps:
        cert = json.loads(json.dumps(certificates[h]))
        adjusted = diff.for_hypothesis(h)
        difference = decide_difference_detection(adjusted_p_value=adjusted["adjusted_p_value"], alpha=0.05,
                                                 sufficient_replicates=bool(cert["sufficient"]))
        assessment = dict(cert["practical_significance"]["assessment"])
        adjusted_tost = equivalence.for_hypothesis(h)
        if assessment.get("formal_practical_equivalence") == "EQUIVALENT_WITHIN_THRESHOLD" and not (
                adjusted_tost["adjusted_p_value"] < 0.05):
            assessment["formal_practical_equivalence"] = "INCONCLUSIVE"
        assessment["multiplicity"] = adjusted_tost
        decision = decide_final(difference_decision=difference, practical_equivalence=assessment,
                                confidence_interval=cert["statistics"]["confidence_interval"],
                                epsilon=assessment.get("threshold"))
        cert["statistics"]["adjusted_p_value"] = adjusted["adjusted_p_value"]
        cert["multiplicity"] = adjusted
        cert["practical_significance"]["assessment"] = assessment
        cert["decision"] = decision.value
        wording = cert.get("conservative_wording") or {}
        cert["conservative_conclusion"] = wording.get(
            {"SUPPORTED": "supported", "NOT_SUPPORTED": "equivalent"}.get(decision.value, "inconclusive"))
        out["certificates"][h] = cert
    out["decisions"] = {h: out["certificates"][h]["decision"] for h in hyps}
    return out


def apply_family_correction(certificates: dict[str, dict]) -> dict:
    """The confirmatory family correction: production certificates only."""
    for h, cert in certificates.items():
        require_confirmatory_certificate(cert, context="AutoND primary-family Holm correction")
    if len({c["production_hash"] for c in certificates.values()}) != 1:
        raise ValueError("the two production certificates were produced under different configurations")
    return holm_family(certificates) | {"non_evidentiary": False, "run_mode": "production"}


# ============================================================ calibration variance and power


def calibration_variance(paths: list[Path], config: dict) -> dict:
    """sigma_Delta and its one-sided 95% upper bound per paired hypothesis. No mean, CI, p or decision."""
    from scripts.analyze_ev_calibration import sigma_upper_confidence_bound

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


#: (true effect of TRAIN-ORDER, true effect of REPRESENTATION), in units of the target effect.
SCENARIOS = ((1, 1), (1, 0), (0, 1), (0, 0))


def power_plan(variance: dict, config: dict, *, variance_sha256: str, simulations: Optional[int] = None,
               max_n: int = 200) -> dict:
    """Smallest common number of pairs at which every final decision reaches the target power.

    For each hypothesis, P(SUPPORTED | true effect = target) and P(NOT_SUPPORTED | true effect = 0) are
    evaluated by the framework's joint simulation of the complete rule (paired t -> difference Holm ->
    TOST -> equivalence Holm -> decide_final), under every state of the OTHER hypothesis, and the worst
    case must reach the target power. Sizing on "difference significant" alone, with the target at the
    margin, would not power the SUPPORTED decision at all.
    """
    if variance.get("artifact") != VARIANCE_ARTIFACT or variance.get("design_hash") != design_hash(config):
        raise ValueError("variance artifact is not this design's calibration output")
    d = config["decisions"]
    epsilon, target, alpha, power = d["practical_threshold"], d["target_effect"], d["alpha"], d["target_power"]
    if not target > epsilon:
        raise ValueError("target_effect must exceed practical_threshold to power SUPPORTED")
    simulations = int(simulations or d["power_simulations"])
    sigmas = variance["sigma_Delta_upper_95"]
    hyps = PRIMARY_FAMILY
    curve, required = [], None
    for n in range(2, max_n + 1):
        worst = {f"{h}:{k}": 1.0 for h in hyps for k in ("supported", "not_supported")}
        for s, (ea, eb) in enumerate(SCENARIOS):
            effects = {hyps[0]: ea * target, hyps[1]: eb * target}
            res = simulate_joint_primary_family(
                hypotheses=hyps, sigmas=sigmas, true_effects=effects, n_pairs=n, alpha=alpha, epsilon=epsilon,
                n_simulations=simulations, rng=statistics_rng(70_000_000 + 1000 * n + s),
                target_effect_source="predeclared")
            for h in hyps:
                key = "supported" if effects[h] else "not_supported"
                value = res.power_supported[h] if effects[h] else res.power_not_supported[h]
                worst[f"{h}:{key}"] = min(worst[f"{h}:{key}"], value)
        curve.append({"n": n, **worst})
        if all(v >= power for v in worst.values()):
            required = n
            break
    return {
        "artifact": POWER_PLAN_ARTIFACT, "non_evidentiary": True, "generated_at_utc": utc_timestamp(),
        "design_hash": design_hash(config), "variance_artifact_sha256": variance_sha256,
        "inputs": {"epsilon": epsilon, "target_effect": target, "target_effect_source": "predeclared", "alpha": alpha,
                   "target_power": power, "n_simulations": simulations, "sigma": "sigma_Delta_upper_95",
                   "sigma_values": sigmas, "max_n": max_n},
        "required_n": required,
        "decision_rule_simulated": "paired t -> difference Holm -> TOST -> equivalence Holm -> decide_final",
        "requirement": ("for each hypothesis, min over the other hypothesis's state of P(SUPPORTED | effect = "
                        "target) and of P(NOT_SUPPORTED | effect = 0) >= target_power"),
        "assumptions": ["paired differences independent and normal with SD = sigma_Delta_upper_95",
                        "the same target effect for both hypotheses; the sign does not matter (symmetric rule)"],
        "power_curve": curve,
    }


def check_power_plan(config: dict) -> list[str]:
    """Production gate: the declared plan must be this design's, unchanged, and met by every paired plan."""
    pp = config["production"]["power_plan"]
    path = Path(__file__).resolve().parents[1] / pp["path"]
    if not path.is_file():
        return [f"power plan {path} does not exist"]
    if sha256_file(path) != pp["sha256"]:
        return ["power plan file differs from the SHA-256 declared in production.power_plan"]
    plan = json.loads(path.read_text())
    problems = []
    if plan.get("artifact") != POWER_PLAN_ARTIFACT or plan.get("design_hash") != design_hash(config):
        problems.append("power plan was not produced for this design")
    d = config["decisions"]
    for key, value in (("epsilon", d["practical_threshold"]), ("target_effect", d["target_effect"]),
                       ("alpha", d["alpha"]), ("target_power", d["target_power"]),
                       ("n_simulations", d["power_simulations"])):
        if (plan.get("inputs") or {}).get(key) != value:
            problems.append(f"power plan {key}={(plan.get('inputs') or {}).get(key)!r} differs from "
                            f"decisions ({value!r})")
    if plan.get("required_n") is None or plan.get("required_n") != pp["required_n"]:
        problems.append(f"declared required_n={pp['required_n']!r} differs from the plan's {plan.get('required_n')!r}")
    for h in PRIMARY_FAMILY:
        try:
            validate_replicate_plan_against_power(minimum_valid_replicates=config["production"][h]["minimum_valid"],
                                                  power_analysis_required_n=pp["required_n"],
                                                  underpowered_justification=None)
        except ValueError as exc:
            problems.append(f"{h}: {exc}")
    return problems
