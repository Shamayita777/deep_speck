"""
AutoND D4 - label-shuffle controlled perturbation of the FULL staged AutoND
pipeline: SPECK32/64, single-key, input difference 0x400000, DBitNet trained
round by round from the starting round until upstream's stopping rule ends it.

Target-specific orchestration only.  The statistics, adapter contract,
perturbation provenance, decision and certificate are the shared engine
(``audit.dataset.d4_controlled_perturbation``, rule
``near_chance_baseline_gated_v4``); hashing, exclusive writes, partition
digests and the overlap audit are ``audit.dataset.integrity``; the pinned
binding and generator are ``audit.dataset.adapters.autond``; the staged
lifecycle and stopping rule are ``audit.dataset.adapters.autond_d4``.  NO
Gohr-specific module is imported, directly or indirectly (tested).

WHAT IS TESTED (from the pinned code and the 12 October bundle; see the RATIONALE)
---------------------------------------------------------------------------------
Each fit runs ``train_nets.train_neural_distinguisher``'s loop: per round, 10
epochs on that round's own 10^7/10^6 partitions, best-val_loss checkpoint
carried to the next round, stop as soon as a round's max val_acc <= 0.505
(``ABORT_TRAINING_BELOW_ACC``).  Intact-label baselines are expected to pass
rounds 5-8 and stop at the round-9 sentinel; a label-shuffled pipeline is
expected to stop at its first round.  The two arms are compared at the SHARED
round every fit trains whatever the stopping rule does: the starting round.
The pipeline progression (rounds trained, per-round max val_acc, stop round,
best_round) is a registered, independently recomputed, reported endpoint.

STAGES (each refuses, and writes nothing it would replace, on any failure)
--------------------------------------------------------------------------
protocol-check  Read-only; exit 0 if well-formed, 2 if malformed;
                --require-approved exits 3 while anything is pending.
preflight       HUMAN-AUTHORISED.  Approved + committed + anchored
                preregistration; clean sources; the pinned checkout verified
                by commit and digests BEFORE import; generates every round's
                train/validation partition (starting round .. cap) and the
                comparison-round test partition ONCE, validates the encoding,
                runs the overlap audit, applies the policy, writes atomically.
run             HUMAN-AUTHORISED PRODUCTION.  Re-verifies everything; trains;
                writes run-dir/result/ once.  Resumable.
verify          Independent, read-only.  Exit 0 = integrity verified (the
                printed overall D4 outcome may still be non-SUPPORTED, never a
                pass); 5 = no result (NOT_COMPLETED); 1 = refused.
                --rerun-models reloads every selected checkpoint.

USAGE (from the repository root)
    PY=~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python
    $PY -B -m audit.dataset.experiments.d4_autond_label_shuffle protocol-check \\
        --preregistration audit/targets/autond/preregistrations/d4_autond_speck3264_full_pipeline_label_shuffle.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.dataset import d4_controlled_perturbation as d4
from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN
from audit.dataset.integrity import (
    REPOSITORY_ROOT,
    IntegrityError,
    audit_partition_overlap,
    canonical_json_bytes,
    labelled_partition_digest,
    partition_digest,
    read_json,
    sha256_bytes,
    sha256_file,
    source_identity,
    stale_staging_directories,
    verify_anchored_committed_record,
    write_directory_exclusive,
)

# ============================================================
# Target definition (fixed design; must equal the preregistration)
# ============================================================

CIPHER = "speck3264"
SCENARIO = "single-key"
DELTA_PLAIN = 0x400000
DELTA_KEY = 0
PLAIN_BITS = 32
STOP_THRESHOLD = 0.505  # train_nets.ABORT_TRAINING_BELOW_ACC at the pinned commit (checked at bind time)
BATCH_SIZE = 5000
AUDIT_SEED = 0
BOOTSTRAP_REPLICATES = 5000
CONFIDENCE_LEVEL = 0.95
FEATURE_WIDTH = 2 * PLAIN_BITS
INPUT_ENCODING = "raw_bits_uint8_0_1_width64_hstack_C0_C1_no_scaling"
CHANCE_LEVEL = 0.5
DATASET_ID = "autond-speck3264"
DATASET_VERSION = "pinned-8d66e0a4-main.make_train_data"
SAMPLE_SIZE_KEYS = ("train_per_round", "validation_per_round", "test")
DEFAULT_PREREGISTRATION = (REPOSITORY_ROOT / "audit/targets/autond/preregistrations/"
                           "d4_autond_speck3264_full_pipeline_label_shuffle.json")
DEFAULT_AUTOND_ROOT = os.environ.get("AUTOND_ROOT", "~/Documents/AutoND-upstream")

SOURCE_FILES = tuple(REPOSITORY_ROOT / p for p in (
    "audit/dataset/d4_controlled_perturbation.py",
    "audit/dataset/integrity.py",
    "audit/dataset/adapters/autond.py",
    "audit/dataset/adapters/autond_d4.py",
    "audit/dataset/perturbations/label_shuffle.py",
    "audit/dataset/experiments/d0_autond_canonical.py",
    "audit/dataset/experiments/d4_autond_label_shuffle.py",
    "audit/dataset/experiments/d4_autond_two_gpu.py",
))

PREFLIGHT_SCHEMA = "d4-autond-preflight-2"
PROGRESSION_ROLES = ("reported_non_gating", "co_primary")

#: Run-directory locks.  The scheduler lock is held by the two-GPU coordinator for its whole life; its
#: workers and the final ``run`` it launches carry its token in AUTOND_D4_SCHEDULER_TOKEN.  An arm lock
#: is held by the one worker training that arm.  The run lock is held by the single ``run`` stage.
SCHEDULER_LOCK = ".d4_autond_scheduler.lock"
RUN_LOCK = ".d4_autond_run.lock"
TOKEN_ENV = "AUTOND_D4_SCHEDULER_TOKEN"


def arm_lock_path(run_dir: Path, condition: str) -> Path:
    return Path(run_dir) / "fits" / f".arm_{condition}.lock"
OVERLAP_POLICIES = ("refuse_on_any_cross_partition_feature_row", "record_only")
DUPLICATE_POLICIES = ("record_only", "refuse_on_any_within_partition_duplicate")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def registered_rounds(values: dict) -> list[int]:
    """Every round a fit may train: starting_round .. max_round."""
    return list(range(values["starting_round"], values["max_round"] + 1))


def expected_model(values: dict) -> dict:
    """The model description every fit's identity must carry (AutoNDD4Adapter.model_description)."""
    rounds = registered_rounds(values)
    return {"architecture": "dbitnet", "builder": "dbitnet.make_model(2 * plain_bits)",
            "input_bits": FEATURE_WIDTH, "optimizer": "Adam(amsgrad=True)", "loss": "mse", "metrics": ["acc"],
            "batch_size": BATCH_SIZE, "shuffle": True, "rounds": rounds,
            "round_rows": [values["sample_sizes"]["train_per_round"]] * len(rounds),
            "epochs_per_round": values["epochs_per_round"], "stop_threshold": STOP_THRESHOLD,
            "comparison_round": values["starting_round"], "upstream_commit": AUTOND_SOURCE_PIN["upstream_commit"],
            "cipher": CIPHER, "delta_plain_hex": hex(DELTA_PLAIN)}


# ============================================================
# AutoND-local helpers (no Gohr code is imported)
# ============================================================


def npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    return buffer.getvalue()


def refused_directory(run_dir: Path) -> Path:
    return Path(run_dir).with_name(Path(run_dir).name + ".preflight_refused")


def overlap_violations(report: dict, values: dict) -> list[str]:
    """Violations of the approved overlap and duplicate policies in an integrity overlap report."""
    violations = []
    if values["partition_overlap_policy"] == "refuse_on_any_cross_partition_feature_row":
        violations += [f"cross {pair}: {c['rows']} shared rows" for pair, c in report["cross"].items() if c["rows"]]
    if values["within_partition_duplicates_policy"] == "refuse_on_any_within_partition_duplicate":
        violations += [f"within {role}: {w['duplicate_rows']} duplicate rows"
                       for role, w in report["within"].items() if w["duplicate_rows"]]
    return violations


def recount_from_predictions(run_dir: Path, certificate: dict, test_labels) -> None:
    """Recompute every fit's counts from its persisted predictions and the test labels."""
    directory = run_dir / "fits" / "predictions"
    rows = certificate["findings"]["replicates"]
    expected_files = {row[f"{c}_predictions"].get("file") for row in rows for c in ("baseline", "perturbed")}
    present = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
    if present != expected_files:
        raise IntegrityError(f"Prediction files differ from the certificate: missing {sorted(expected_files - present)}, "
                             f"unexpected {sorted(present - expected_files)}.")
    for row in rows:
        r, correct = row["replicate"], {}
        for condition in ("baseline", "perturbed"):
            entry = row[f"{condition}_predictions"]
            path = d4.prediction_path(run_dir / "fits", r, condition)
            if entry.get("file") != path.name or not path.is_file() or sha256_file(path) != entry.get("sha256"):
                raise IntegrityError(f"Replicate {r} {condition} predictions are missing or do not match their "
                                     "recorded digest.")
            predictions = np.load(path)
            if predictions.dtype.str != entry.get("dtype") or list(predictions.shape) != entry.get("shape"):
                raise IntegrityError(f"Replicate {r} {condition} predictions have the wrong dtype or shape.")
            correct[condition] = d4.binary_predictions_to_correctness(predictions, test_labels)
            if entry.get("class_counts") != d4.class_diagnostics(correct[condition], test_labels):
                raise IntegrityError(f"Replicate {r} {condition}: certificate class counts differ from those the "
                                     "persisted predictions give.")
        b, c, _ = d4.mcnemar_exact_pvalue(correct["baseline"], correct["perturbed"])
        recounted = {"test_examples": int(len(test_labels)), "baseline_correct": int(correct["baseline"].sum()),
                     "perturbed_correct": int(correct["perturbed"].sum()), "mcnemar_b": b, "mcnemar_c": c}
        for key, value in recounted.items():
            if row.get(key) != value:
                raise IntegrityError(f"Replicate {r}: certificate {key}={row.get(key)!r} but the persisted "
                                     f"predictions give {value!r}.")


def load_json_strict(path: Path) -> dict:
    """Parse JSON refusing duplicate keys at any depth (json.load silently keeps the last one)."""
    def pairs(items):
        keys = [k for k, _ in items]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise IntegrityError(f"{path} has duplicate JSON keys {duplicates}.")
        return dict(items)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"{path} is truncated or malformed ({exc}).") from None
    if not isinstance(value, dict):
        raise IntegrityError(f"{path} is not a JSON object.")
    return value


def acquire_lock(path: Path, payload: dict) -> str:
    """Create ``path`` exclusively with a fresh token; refuse if it exists (never broken automatically)."""
    token = uuid.uuid4().hex
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise IntegrityError(f"Lock {path} exists: another scheduler, worker or run owns this run directory, or a "
                             "previous one crashed. Inspect it and remove it by hand only when no owner remains.") from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({**payload, "token": token, "pid": os.getpid(), "created_utc": _utc()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return token


def release_lock(path: Path, token: str) -> None:
    """Remove ``path`` only if it still carries ``token``."""
    try:
        if json.loads(Path(path).read_text()).get("token") == token:
            Path(path).unlink()
    except (OSError, ValueError):
        pass


def check_scheduler_permission(run_dir: Path) -> None:
    """Refuse when a scheduler owns the run directory and this process is not one of its children."""
    lock = Path(run_dir) / SCHEDULER_LOCK
    if lock.exists():
        try:
            owner = json.loads(lock.read_text()).get("token")
        except (OSError, ValueError):
            owner = None
        if not owner or os.environ.get(TOKEN_ENV) != owner:
            raise IntegrityError(f"Scheduler lock {lock} is held by another process; refusing to touch this run.")


def progression_criterion(section: dict, starting_round: int) -> dict:
    """The registered progression criterion (used only when pipeline_progression_role = co_primary)."""
    fits = section["fits"]
    perturbed = all(f["best_round"] is None and f["stop_round"] == starting_round
                    for f in fits if f["condition"] == "perturbed")
    baseline = all(f["best_round"] is not None for f in fits if f["condition"] == "baseline")
    return {"rule": "every perturbed fit stops at the starting round with no best_round, and every baseline fit "
                    "passes at least its starting round (best_round is not None)",
            "perturbed_stop_at_starting_round_without_best_round": perturbed,
            "baseline_pass_the_starting_round": baseline, "met": perturbed and baseline}


def autond_decision(certificate: dict, role: str, starting_round: int) -> dict:
    """The AutoND-level outcome: the engine outcome, gated by progression only under co_primary.

    Under co_primary the outcome is SUPPORTED only when the engine outcome is SUPPORTED AND the
    progression criterion is met; any other engine outcome passes through unchanged.  Requiring an
    extra component can only shrink the SUPPORTED event, so the engine's level alpha for SUPPORTED
    is kept; the progression component itself has no error control and its failure is descriptive.
    """
    if role not in PROGRESSION_ROLES:
        raise IntegrityError(f"pipeline_progression_role {role!r} is not one of {PROGRESSION_ROLES}.")
    engine = certificate["decision"]["overall_d4_outcome"]
    criterion = progression_criterion(certificate["autond_pipeline_progression"], starting_round)
    outcome = engine
    if role == "co_primary" and engine == "SUPPORTED" and not criterion["met"]:
        outcome = "PROGRESSION_NOT_CONFIRMED"
    return {"pipeline_progression_role": role, "engine_overall_d4_outcome": engine,
            "progression_criterion": criterion, "progression_gates_outcome": role == "co_primary",
            "autond_overall_outcome": outcome, "overall_supported": outcome == "SUPPORTED"}


def pipeline_progression(certificate: dict, role: str) -> dict:
    """Registered endpoint: each fit's stopping progression; gating only under co_primary."""
    fits = []
    for row in certificate["findings"]["replicates"]:
        for condition in ("baseline", "perturbed"):
            prog = row[f"{condition}_selection"]["progression"]
            fits.append({"replicate": row["replicate"], "condition": condition, "status": prog["status"],
                         "rounds_trained": prog["rounds_trained"], "rounds_max_val_acc": prog["rounds_max_val_acc"],
                         "stop_round": prog["stop_round"], "best_round": prog["best_round"],
                         "best_val_acc": prog["best_val_acc"]})
    def count(condition, test):
        return sum(1 for f in fits if f["condition"] == condition and test(f))
    first = certificate["findings"]["replicates"][0]["baseline_selection"]["comparison_round"]
    return {
        "role": role, "stop_threshold": STOP_THRESHOLD,
        "rule": "upstream train_neural_distinguisher: stop when a round's max val_acc <= 0.505; best_round is the "
                "last round above it (None if the first round stops)",
        "fits": fits,
        "summary": {
            "perturbed_stopped_at_first_round": count("perturbed", lambda f: f["stop_round"] == first),
            "perturbed_with_no_best_round": count("perturbed", lambda f: f["best_round"] is None),
            "baseline_best_rounds": [f["best_round"] for f in fits if f["condition"] == "baseline"],
            "baseline_stop_rounds": [f["stop_round"] for f in fits if f["condition"] == "baseline"],
            "cap_reached": [f"{f['condition']}#{f['replicate']}" for f in fits if f["status"] == "cap_reached"],
        },
    }


# ============================================================
# Preregistration
# ============================================================


def check_target_design(record: dict) -> list[str]:
    """Problems if the preregistration's fixed design differs from this runner."""
    from audit.dataset.adapters import autond_d4

    problems = []
    fixed = record.get("fixed_design", {})
    expected = {
        ("target", "cipher"): CIPHER, ("target", "scenario"): SCENARIO,
        ("target", "delta_plain_hex"): hex(DELTA_PLAIN), ("target", "delta_key_hex"): hex(DELTA_KEY),
        ("training_procedure", "stop_threshold"): STOP_THRESHOLD,
        ("training_procedure", "comparison_round"): "starting_round",
        ("upstream", "commit"): AUTOND_SOURCE_PIN["upstream_commit"],
        ("upstream", "source_digests"): AUTOND_SOURCE_PIN["source_digests"],
        ("model", "architecture"): "dbitnet", ("model", "optimizer"): "Adam(amsgrad=True)",
        ("model", "loss"): "mse", ("model", "batch_size"): BATCH_SIZE, ("model", "shuffle"): True,
        ("model_selection", "rule"): autond_d4.MODEL_SELECTION_RULE,
        ("seeds", "audit_seed"): AUDIT_SEED,
        ("bootstrap", "replicates"): BOOTSTRAP_REPLICATES, ("bootstrap", "confidence_level"): CONFIDENCE_LEVEL,
        ("dataset", "input_encoding"): INPUT_ENCODING,
    }
    for (section, key), value in expected.items():
        actual = fixed.get(section, {}).get(key)
        if actual != value:
            problems.append(f"fixed_design.{section}.{key} is {actual!r}; this runner implements {value!r}")
    for section, procedure in (("primary_endpoint", d4.NEAR_CHANCE_PROCEDURE),
                               ("baseline_adequacy_endpoint", d4.BASELINE_ADEQUACY_PROCEDURE),
                               ("supporting_endpoint", d4.PAIRED_PROCEDURE)):
        if fixed.get(section, {}).get("procedure") != procedure:
            problems.append(f"fixed_design.{section}.procedure differs from the implemented "
                            f"{d4.NEAR_CHANCE_RULE} procedure (estimand, sidedness, aggregation or level)")
    if fixed.get("overall_rule") != d4.OVERALL_RULE:
        problems.append("fixed_design.overall_rule differs from the implemented composite outcome rule")
    if record.get("perturbation", {}).get("name") != "label_shuffle":
        problems.append("perturbation.name must be 'label_shuffle'")
    return problems


def check_decision_values(values: dict) -> list[str]:
    problems = []
    sizes = values.get("sample_sizes")
    if not (isinstance(sizes, dict) and set(sizes) == set(SAMPLE_SIZE_KEYS)
            and all(isinstance(sizes[k], int) and not isinstance(sizes[k], bool) and sizes[k] >= 1
                    for k in SAMPLE_SIZE_KEYS)):
        problems.append(f"decisions.sample_sizes must give positive integers for {list(SAMPLE_SIZE_KEYS)}")
    start, cap = values.get("starting_round"), values.get("max_round")
    if not (isinstance(start, int) and not isinstance(start, bool) and start >= 1):
        problems.append("decisions.starting_round must be a positive integer")
    if not (isinstance(cap, int) and not isinstance(cap, bool) and isinstance(start, int) and cap > start):
        problems.append("decisions.max_round must be an integer greater than starting_round")
    if not (isinstance(values.get("epochs_per_round"), int) and values["epochs_per_round"] >= 1):
        problems.append("decisions.epochs_per_round must be a positive integer")
    if values.get("pipeline_progression_role") not in PROGRESSION_ROLES:
        problems.append(f"decisions.pipeline_progression_role must be one of {PROGRESSION_ROLES}")
    if not (isinstance(values.get("replicates"), int) and values["replicates"] >= 2):
        problems.append("decisions.replicates must be an integer >= 2")
    if not (isinstance(values.get("alpha"), float) and 0 < values["alpha"] < 0.5):
        problems.append("decisions.alpha must be a float in (0, 0.5)")
    if not (isinstance(values.get("effect_threshold"), float) and values["effect_threshold"] >= 0):
        problems.append("decisions.effect_threshold must be a non-negative float")
    if values.get("decision_rule") != d4.NEAR_CHANCE_RULE:
        problems.append(f"decisions.decision_rule must be {d4.NEAR_CHANCE_RULE}")
    margin = values.get("chance_margin")
    if not (isinstance(margin, float) and 0.0 < margin < CHANCE_LEVEL):
        problems.append("decisions.chance_margin must be a float in (0, 0.5), a fraction of accuracy")
    floor = values.get("baseline_adequacy_threshold")
    if not (isinstance(floor, float) and 0.5 < floor < 1.0):
        problems.append("decisions.baseline_adequacy_threshold must be a float in (0.5, 1), a fraction of accuracy")
    if values.get("paired_endpoint_role") not in d4.PAIRED_ENDPOINT_ROLES:
        problems.append(f"decisions.paired_endpoint_role must be one of {d4.PAIRED_ENDPOINT_ROLES}")
    if values.get("partition_overlap_policy") not in OVERLAP_POLICIES:
        problems.append("decisions.partition_overlap_policy is not implemented")
    if values.get("within_partition_duplicates_policy") not in DUPLICATE_POLICIES:
        problems.append("decisions.within_partition_duplicates_policy is not implemented")
    return problems


def load_approved_preregistration(path: Path, anchor: str) -> tuple[dict, dict, str]:
    """Anchored + committed + approved + matching this runner; returns (record, values, sha)."""
    load_json_strict(path)
    record = verify_anchored_committed_record(path, expected_sha256=anchor, what="AutoND D4 preregistration")
    commit = record.pop("_commit")
    values = d4.require_approved_preregistration(record)
    problems = check_target_design(record) + check_decision_values(values)
    if problems:
        raise IntegrityError(f"Preregistration does not match this runner: {problems}")
    approved = datetime.strptime(record["approved_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if commit["committed_utc"] and approved > datetime.fromisoformat(commit["committed_utc"]):
        raise IntegrityError("approved_utc is after the commit that records the preregistration.")
    return record, dict(values, _commit=commit), anchor


def design_of(values: dict, anchor: str) -> dict:
    """The complete design a run executes; its hash is the run_config_hash."""
    from audit.dataset.adapters import autond_d4

    return {
        "audit": "D4", "target": "autond-speck3264-single-key-0x400000-staged-dbitnet-full-pipeline",
        "perturbation": "label_shuffle", "preregistration_sha256": anchor,
        "cipher": CIPHER, "scenario": SCENARIO, "delta_plain_hex": hex(DELTA_PLAIN), "stop_threshold": STOP_THRESHOLD,
        "upstream_commit": AUTOND_SOURCE_PIN["upstream_commit"],
        "upstream_source_digests": AUTOND_SOURCE_PIN["source_digests"],
        "batch_size": BATCH_SIZE, "audit_seed": AUDIT_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL, "input_encoding": INPUT_ENCODING,
        "model_selection_rule": autond_d4.MODEL_SELECTION_RULE,
        "chance_level": CHANCE_LEVEL, "primary_procedure": d4.NEAR_CHANCE_PROCEDURE,
        "baseline_procedure": d4.BASELINE_ADEQUACY_PROCEDURE, "supporting_procedure": d4.PAIRED_PROCEDURE,
        "overall_rule": d4.OVERALL_RULE,
        **{k: v for k, v in values.items() if not k.startswith("_")},
    }


def run_config_hash(design: dict) -> str:
    return sha256_bytes(canonical_json_bytes(design))


# ============================================================
# Pinned binding and data
# ============================================================


def make_binding(autond_root, num_rounds: int):
    """Pin-verified AutoND binding for one round (commit and digests checked BEFORE import)."""
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter

    return AutoNDDatasetAdapter(
        autond_root=autond_root, cipher_name=CIPHER, delta_plain_int=DELTA_PLAIN, delta_key_int=DELTA_KEY,
        num_rounds=int(num_rounds), scenario=SCENARIO,
        expected_source_digests=AUTOND_SOURCE_PIN["source_digests"],
        expected_upstream_commit=AUTOND_SOURCE_PIN["upstream_commit"])


def upstream_identity(binding) -> dict:
    """What was actually bound; refuses if it is not the pinned revision or its stop rule differs."""
    upstream_threshold = getattr(sys.modules.get("train_nets"), "ABORT_TRAINING_BELOW_ACC", None)
    if upstream_threshold != STOP_THRESHOLD:
        raise IntegrityError(f"Pinned train_nets.ABORT_TRAINING_BELOW_ACC is {upstream_threshold}; this runner "
                             f"implements {STOP_THRESHOLD}.")
    digests = dict(binding.source_digests)
    pinned = AUTOND_SOURCE_PIN["source_digests"]
    if {k: digests.get(k) for k in pinned} != pinned:
        raise IntegrityError("The bound AutoND sources do not match AUTOND_SOURCE_PIN.")
    return {"commit": AUTOND_SOURCE_PIN["upstream_commit"], "source_digests": {k: digests[k] for k in sorted(pinned)}}


def round_role(kind: str, stage_round: int) -> str:
    return f"{kind}_r{int(stage_round):02d}"


def generate_autond_partitions(values: dict, autond_root) -> tuple[dict, dict]:
    """Every registered round's train/validation partition and the comparison-round test partition."""
    sizes, partitions, identity = values["sample_sizes"], {}, None
    for stage_round in registered_rounds(values):
        binding = make_binding(autond_root, stage_round)
        identity = identity or upstream_identity(binding)
        for kind, key in (("train", "train_per_round"), ("validation", "validation_per_round")):
            print(f"Generating {round_role(kind, stage_round)} ({sizes[key]} rows, main.make_train_data, os.urandom)...")
            partitions[round_role(kind, stage_round)] = binding.generate_partition(int(sizes[key]))
    print(f"Generating test (comparison round {values['starting_round']}, {sizes['test']} rows)...")
    partitions["test"] = make_binding(autond_root, values["starting_round"]).generate_partition(int(sizes["test"]))
    return partitions, identity


def validate_input_encoding(partitions: dict) -> None:
    for role, (x, y) in partitions.items():
        x, y = np.asarray(x), np.asarray(y).reshape(-1)
        if x.dtype != np.uint8 or x.ndim != 2 or x.shape[1] != FEATURE_WIDTH:
            raise IntegrityError(f"{role} features have dtype {x.dtype} and shape {x.shape}; the frozen encoding "
                                 f"is uint8 with width {FEATURE_WIDTH}.")
        if x.size and int(x.max()) > 1:
            raise IntegrityError(f"{role} features contain values outside {{0, 1}}; inputs must not be rescaled.")
        if y.dtype != np.uint8 or len(y) != len(x):
            raise IntegrityError(f"{role} labels must be uint8 with one label per row.")
        if y.size and int(y.max()) > 1:
            raise IntegrityError(f"{role} labels contain values outside {{0, 1}}.")


def _all_digest(arrays: dict) -> str:
    return partition_digest({f"{role}_{k}": a for role, (x, y) in arrays.items() for k, a in (("x", x), ("y", y))})


def load_preflight(run_dir: Path) -> tuple[dict, dict]:
    """Read preflight.json and re-verify every partition file from its bytes.

    Returns (preflight, data) with data = {"train": (x, y) stage-major,
    "validation": {round: (x, y)}, "test": (x, y)}.
    """
    run_dir = Path(run_dir)
    stale = stale_staging_directories(run_dir)
    if stale:
        raise IntegrityError(f"Interrupted write {[p.name for p in stale]} exists beside {run_dir}; refusing.")
    preflight = read_json(run_dir / "preflight.json", "preflight record")
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA or preflight.get("outcome") != "PASS":
        raise IntegrityError(f"{run_dir / 'preflight.json'} is not a passing {PREFLIGHT_SCHEMA} record.")
    entries = preflight["partitions"]
    recorded = {entries[r][f"{k}_file"] for r in entries for k in ("x", "y")}
    present = {p.relative_to(run_dir).as_posix() for p in (run_dir / "dataset").rglob("*") if p.is_file()} \
        if (run_dir / "dataset").is_dir() else set()
    if present != recorded:
        raise IntegrityError(f"Dataset directory differs from preflight.json: missing {sorted(recorded - present)}, "
                             f"unexpected {sorted(present - recorded)}.")
    arrays = {}
    for role, entry in entries.items():
        loaded = []
        for kind in ("x", "y"):
            path = run_dir / entry[f"{kind}_file"]
            if not path.is_file() or sha256_file(path) != entry[f"{kind}_sha256"]:
                raise IntegrityError(f"Partition file {path} is missing or does not match preflight.json.")
            loaded.append(np.load(path, mmap_mode="r"))
        if labelled_partition_digest(*loaded) != entry["digest"]:
            raise IntegrityError(f"{role} partition content does not reproduce its recorded digest.")
        arrays[role] = tuple(loaded)
    validate_input_encoding(arrays)
    if _all_digest(arrays) != preflight["all_partitions_digest"]:
        raise IntegrityError("The combined partition digest does not match preflight.json.")
    rounds = registered_rounds(preflight["design"])
    train_x, train_y = arrays["train"]
    round_rows = entries["train"]["round_rows"]
    if entries["train"]["rounds"] != rounds or len(round_rows) != len(rounds) or sum(round_rows) != len(train_x):
        raise IntegrityError("Recorded round rows do not match the training partition.")
    start = 0
    for stage_round, rows, digest in zip(rounds, round_rows, entries["train"]["round_digests"]):
        if labelled_partition_digest(train_x[start:start + rows], train_y[start:start + rows]) != digest:
            raise IntegrityError(f"Training round {stage_round} does not reproduce its recorded digest.")
        start += rows
    data = {"train": arrays["train"], "test": arrays["test"],
            "validation": {r: arrays[round_role("validation", r)] for r in rounds}}
    return preflight, data


# ============================================================
# Stages
# ============================================================


def stage_protocol_check(args) -> int:
    record = load_json_strict(args.preregistration)
    review = d4.review_preregistration(record)
    problems = review["problems"] + check_target_design(record)
    if review["values"]:
        problems += [p for p in check_decision_values(review["values"])
                     if p.split()[0].split(".")[-1] in review["values"]]
    print(json.dumps({"sha256": sha256_file(args.preregistration), "problems": problems,
                      "pending": review["pending"]}, indent=2))
    if problems:
        print("PROTOCOL MALFORMED")
        return 2
    if review["pending"]:
        print(f"PROTOCOL WELL-FORMED; {len(review['pending'])} field(s) PENDING human approval.")
        return 3 if args.require_approved else 0
    print("PROTOCOL WELL-FORMED AND APPROVED (anchoring and commit are checked at preflight).")
    return 0


def stage_preflight(args) -> int:
    run_dir = Path(args.run_dir)
    refused_dir = refused_directory(run_dir)
    for existing in (run_dir, refused_dir):
        if existing.exists():
            raise IntegrityError(f"Refusing: {existing} exists. Preflight never reuses or replaces a run directory.")
    stale = stale_staging_directories(run_dir) + stale_staging_directories(refused_dir)
    if stale:
        raise IntegrityError(f"Interrupted preflight output {[p.name for p in stale]} exists; refusing.")
    record, values, anchor = load_approved_preregistration(args.preregistration, args.preregistration_sha256)
    identity = source_identity(SOURCE_FILES)
    dirty = [f for f, e in identity["files"].items() if not e["clean_at_head"]]
    if dirty:
        raise IntegrityError(f"AutoND D4 sources are not clean at HEAD: {dirty}")

    partitions, upstream = generate_autond_partitions(values, args.autond_root)
    validate_input_encoding(partitions)
    report = audit_partition_overlap(partitions)
    print(f"Partition overlap audit: {json.dumps(report, sort_keys=True)}")
    violations = overlap_violations(report, values)

    rounds = registered_rounds(values)
    stages = [partitions[round_role("train", r)] for r in rounds]
    train_x = np.concatenate([x for x, _ in stages])
    train_y = np.concatenate([np.asarray(y).reshape(-1) for _, y in stages])
    persisted = {"train": (train_x, train_y), "test": partitions["test"],
                 **{round_role("validation", r): partitions[round_role("validation", r)] for r in rounds}}
    files, entries = {}, {}
    for role, (x, y) in persisted.items():
        x_bytes, y_bytes = npy_bytes(x), npy_bytes(y)
        files[f"dataset/{role}_x.npy"], files[f"dataset/{role}_y.npy"] = x_bytes, y_bytes
        entries[role] = {"x_file": f"dataset/{role}_x.npy", "x_sha256": sha256_bytes(x_bytes),
                         "y_file": f"dataset/{role}_y.npy", "y_sha256": sha256_bytes(y_bytes),
                         "rows": int(len(x)), "digest": labelled_partition_digest(x, y)}
    entries["train"].update(rounds=list(rounds), round_rows=[int(len(x)) for x, _ in stages],
                            round_digests=[labelled_partition_digest(x, y) for x, y in stages],
                            layout="round-major concatenation of the train_rRR partitions")
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA, "outcome": "REFUSED" if violations else "PASS",
        "violations": violations, "created_utc": _utc(),
        "preregistration": {"path": values["_commit"]["path"], "sha256": anchor, "commit": values["_commit"]["sha"]},
        "design": design_of(values, anchor), "run_config_hash": run_config_hash(design_of(values, anchor)),
        "source_identity": identity, "upstream_identity": upstream,
        "autond_root": str(Path(args.autond_root).expanduser().resolve()),
        "input_encoding": INPUT_ENCODING, "partitions": entries,
        "all_partitions_digest": _all_digest(persisted), "overlap_audit": report,
    }
    files["preflight.json"] = json.dumps(preflight, indent=2, sort_keys=True).encode() + b"\n"
    if violations:
        write_directory_exclusive(refused_dir, files)
        print(f"PREFLIGHT REFUSED by the approved overlap policy: {violations}\nMeasurements and data: {refused_dir}. "
              "The protocol stops here; a new dataset needs a documented protocol amendment.")
        return 4
    write_directory_exclusive(run_dir, files)
    print(f"PREFLIGHT PASSED. Wrote {run_dir}. No model was trained and no test statistic was computed.")
    return 0


def run_binding_of(preflight: dict, run_dir: Path) -> dict:
    """The preflight/run identity bound into every fit."""
    return {"preflight_sha256": sha256_file(Path(run_dir) / "preflight.json"),
            "all_partitions_digest": preflight["all_partitions_digest"],
            "run_config_hash": preflight["run_config_hash"]}


def replicate_plan(design: dict) -> list[tuple[int, int, object]]:
    """(replicate, model seed, SeedSequence child) exactly as the engine's run_d4 derives them."""
    children = np.random.SeedSequence(design["audit_seed"]).spawn(design["replicates"])
    return [(r, int(child.generate_state(1, dtype=np.uint32)[0]), child) for r, child in enumerate(children, start=1)]


def condition_labels(train_y, condition: str, child) -> np.ndarray:
    """Training labels of one condition: intact, or permuted by the engine's planned label shuffle."""
    from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

    train_y = np.asarray(train_y)
    if condition == "baseline":
        return train_y
    indices = LabelShufflePerturbation().permutation_indices(len(train_y), np.random.default_rng(child))
    return train_y[indices]


def expected_identity(preflight: dict, data: dict, run_dir: Path, design: dict, *, replicate: int, seed: int,
                      condition: str, labels) -> dict:
    """The complete identity a fit must carry (read-only; no TensorFlow)."""
    from audit.dataset.adapters import autond_d4

    train_x = data["train"][0]
    labels = np.asarray(labels)
    offsets = np.cumsum([0, *preflight["partitions"]["train"]["round_rows"]])
    return {
        "seed": seed, "condition": condition, "replicate": replicate, "config_hash": preflight["run_config_hash"],
        "train_digest": labelled_partition_digest(train_x, labels),
        "round_train_digests": [labelled_partition_digest(train_x[a:b], labels[a:b])
                                for a, b in zip(offsets[:-1], offsets[1:])],
        "validation_digests": [labelled_partition_digest(*data["validation"][r]) for r in registered_rounds(design)],
        "evaluation_digest": preflight["partitions"]["test"]["digest"], "model": expected_model(design),
        "run_binding": run_binding_of(preflight, run_dir), "adapter": autond_d4.ADAPTER_IDENTITY,
    }


def fit_status(run_dir: Path, preflight: dict, data: dict, design: dict, *, replicate: int, seed: int, child,
               condition: str) -> dict:
    """Verified, read-only status of one fit: absent, partial, finished or invalid.  Writes nothing."""
    from audit.dataset.adapters import autond_d4

    fit_dir, state_path = d4.checkpoint_paths(Path(run_dir) / "fits", replicate, condition)
    status = {"replicate": replicate, "condition": condition, "fit_dir": str(fit_dir)}
    if not state_path.exists():
        return {**status, "status": "absent" if not fit_dir.exists() or not any(fit_dir.rglob("*.keras"))
                else "invalid", "completed_steps": 0,
                **({} if not fit_dir.exists() or not any(fit_dir.rglob("*.keras"))
                   else {"error": "checkpoints exist without a committed state.json"})}
    try:
        identity = expected_identity(preflight, data, run_dir, design, replicate=replicate, seed=seed,
                                     condition=condition, labels=condition_labels(data["train"][1], condition, child))
        plan = autond_d4.step_plan(registered_rounds(design), design["epochs_per_round"])
        state = autond_d4.load_committed_state(state_path, fit_dir, identity=identity, plan=plan)
        records, _ = autond_d4.verify_epoch_chain(fit_dir, plan=plan, steps=state["completed_steps"],
                                                  identity=identity)
        prog = autond_d4.progression(records, rounds=registered_rounds(design),
                                     epochs_per_round=design["epochs_per_round"], threshold=STOP_THRESHOLD)
    except IntegrityError as exc:
        return {**status, "status": "invalid", "error": str(exc)}
    return {**status, "status": "partial" if prog["status"] == "in_progress" else "finished",
            "completed_steps": state["completed_steps"], "progression_status": prog["status"],
            "rounds_trained": prog["rounds_trained"], "retries": len(state["retries"]),
            "selected_checkpoint_recorded": (fit_dir / "selected_checkpoint.json").is_file()}


def make_adapter_factory(design: dict, data: dict, evaluation_digest: str, binding, run_binding: dict):
    from audit.dataset.adapters.autond_d4 import AutoNDD4Adapter

    rounds = registered_rounds(design)
    rows = [design["sample_sizes"]["train_per_round"]] * len(rounds)

    def factory(seed: int):
        return AutoNDD4Adapter(binding=binding, rounds=rounds, round_rows=rows, validation=data["validation"],
                               evaluation_digest=evaluation_digest, epochs_per_round=design["epochs_per_round"],
                               batch_size=BATCH_SIZE, seed=seed, upstream_commit=AUTOND_SOURCE_PIN["upstream_commit"],
                               stop_threshold=STOP_THRESHOLD, run_binding=run_binding)

    return factory


def stage_run(args) -> int:
    run_dir = Path(args.run_dir)
    result_dir = run_dir / "result"
    if result_dir.exists():
        raise IntegrityError(f"Refusing: {result_dir} exists. A completed D4 result is never replaced.")
    if not (run_dir / "preflight.json").is_file():
        raise IntegrityError(f"{run_dir} has no preflight.json; run preflight first.")
    check_scheduler_permission(run_dir)
    busy = [c for c in ("baseline", "perturbed") if arm_lock_path(run_dir, c).exists()]
    if busy:
        raise IntegrityError(f"Arm worker lock(s) {busy} exist; a worker is training. Refusing to run concurrently.")
    token = acquire_lock(run_dir / RUN_LOCK, {"stage": "run"})
    try:
        return _stage_run(args, run_dir, result_dir)
    finally:
        release_lock(run_dir / RUN_LOCK, token)


def _stage_run(args, run_dir: Path, result_dir: Path) -> int:
    record, values, anchor = load_approved_preregistration(args.preregistration, args.preregistration_sha256)
    preflight, data = load_preflight(run_dir)
    design = design_of(values, anchor)
    if preflight["design"] != design or preflight["preregistration"]["sha256"] != anchor:
        raise IntegrityError("The approved preregistration differs from the one preflight was run under.")
    identity = source_identity(SOURCE_FILES)
    if identity["files"] != preflight["source_identity"]["files"]:
        raise IntegrityError("AutoND D4 sources differ from those recorded at preflight. Refusing to run.")
    binding = make_binding(args.autond_root, design["starting_round"])
    if upstream_identity(binding) != preflight["upstream_identity"]:
        raise IntegrityError("The bound AutoND checkout differs from the one recorded at preflight.")
    from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

    run_id, started = uuid.uuid4().hex, _utc()
    test_x, test_y = data["test"]
    config_hash = preflight["run_config_hash"]
    result = d4.run_d4(
        perturbation=LabelShufflePerturbation(), train_features=data["train"][0], train_labels=data["train"][1],
        test_features=test_x, test_labels=test_y,
        adapter_factory=make_adapter_factory(design, data, preflight["partitions"]["test"]["digest"], binding,
                                             run_binding_of(preflight, run_dir)),
        checkpoint_root=run_dir / "fits", replicates=design["replicates"], audit_seed=AUDIT_SEED,
        effect_threshold=design["effect_threshold"], bootstrap_replicates=BOOTSTRAP_REPLICATES,
        confidence_level=CONFIDENCE_LEVEL, alpha=design["alpha"], run_config_hash=config_hash,
        decision_rule=design["decision_rule"], chance_margin=design["chance_margin"],
        paired_endpoint_role=design["paired_endpoint_role"], chance_level=CHANCE_LEVEL,
        baseline_adequacy_threshold=design["baseline_adequacy_threshold"],
    )
    preflight_sha = sha256_file(run_dir / "preflight.json")
    certificate = d4.generate_certificate(
        result, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
        generation_procedure=("main.make_train_data at the pinned commit: one call per registered round for train "
                              "and validation, one comparison-round call for test, persisted at preflight"),
        generation_parameters={"sample_sizes": design["sample_sizes"], "rounds": registered_rounds(design),
                               "randomness_source": "os.urandom", "cipher": CIPHER, "scenario": SCENARIO,
                               "delta_plain_hex": hex(DELTA_PLAIN), "delta_key_hex": hex(DELTA_KEY)},
        audit_seed=AUDIT_SEED, confidence_level=CONFIDENCE_LEVEL, alpha=design["alpha"],
        provenance={"preregistration": preflight["preregistration"], "preflight_sha256": preflight_sha,
                    "run_config_hash": config_hash, "partitions": preflight["partitions"],
                    "test_label_counts": {"zeros": int(np.sum(np.asarray(test_y) == 0)),
                                          "ones": int(np.sum(np.asarray(test_y) == 1))},
                    "overlap_audit": preflight["overlap_audit"], "source_identity": identity,
                    "upstream_identity": preflight["upstream_identity"], "run_id": run_id},
    )
    d4.recompute_certificate_decision(certificate)
    certificate["autond_pipeline_progression"] = pipeline_progression(certificate, design["pipeline_progression_role"])
    certificate["autond_decision"] = autond_decision(certificate, design["pipeline_progression_role"],
                                                     design["starting_round"])
    certificate_bytes = json.dumps(certificate, indent=2, sort_keys=True, allow_nan=True).encode() + b"\n"
    manifest = {"schema_version": "d4-run-manifest-1", "run_id": run_id, "started_utc": started,
                "finished_utc": _utc(), "run_status": "COMPLETE", "preflight_sha256": preflight_sha,
                "preregistration_sha256": anchor, "source_identity": identity,
                "files": {"d4_certificate.json": sha256_bytes(certificate_bytes)}}
    write_directory_exclusive(result_dir, {
        "d4_certificate.json": certificate_bytes,
        "d4_run_manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
    })
    d4.print_report(result)
    outcome = certificate["autond_decision"]["autond_overall_outcome"]
    print(f"Pipeline progression ({design['pipeline_progression_role']}): "
          f"{json.dumps(certificate['autond_pipeline_progression']['summary'])}")
    print(f"D4 RUN COMPLETE. Engine outcome: {certificate['decision']['outcome']}; AutoND overall outcome: {outcome}"
          f"{'' if outcome == 'SUPPORTED' else ' (NOT a scientific pass)'}  ->  {result_dir}")
    return 0


def verify_run(run_dir: Path) -> dict:
    """Independent post-run verification; read-only. Raises IntegrityError on any defect."""
    from audit.dataset.adapters import autond_d4
    from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "result" / "d4_run_manifest.json", "run manifest")
    if manifest.get("run_status") != "COMPLETE" or set(manifest.get("files", {})) != {"d4_certificate.json"}:
        raise IntegrityError("Run manifest is not a COMPLETE manifest of exactly the certificate.")
    certificate_path = run_dir / "result" / "d4_certificate.json"
    if not certificate_path.is_file() or sha256_file(certificate_path) != manifest["files"]["d4_certificate.json"]:
        raise IntegrityError("Certificate is missing or does not match the run manifest.")
    certificate = json.loads(certificate_path.read_text())
    preflight, data = load_preflight(run_dir)
    provenance = certificate["provenance"]
    preflight_sha = sha256_file(run_dir / "preflight.json")
    if provenance["preflight_sha256"] != preflight_sha or manifest["preflight_sha256"] != preflight_sha:
        raise IntegrityError("Certificate or manifest is bound to a different preflight record.")
    for key in ("preregistration", "upstream_identity"):
        if provenance.get(key) != preflight[key]:
            raise IntegrityError(f"Certificate {key} differs from preflight.")
    if provenance["source_identity"]["files"] != preflight["source_identity"]["files"]:
        raise IntegrityError("The run used sources that differ from those recorded at preflight.")
    config_hash = preflight["run_config_hash"]
    if provenance["run_config_hash"] != config_hash or certificate["settings"]["run_config_hash"] != config_hash:
        raise IntegrityError("Certificate run_config_hash differs from preflight.")
    design, settings = preflight["design"], certificate["settings"]
    for key in ("alpha", "decision_rule", "replicates", "audit_seed", "bootstrap_replicates", "confidence_level",
                "chance_level", "chance_margin", "baseline_adequacy_threshold", "paired_endpoint_role",
                "effect_threshold"):
        if settings.get(key) != design[key]:
            raise IntegrityError(f"Certificate setting {key} differs from the preregistered design.")
    test_x, test_y = data["test"]
    if settings.get("test_examples") != design["sample_sizes"]["test"] or settings.get("test_examples") != len(test_y):
        raise IntegrityError("Certificate test sample size differs from the preregistered design.")
    methodology = certificate.get("methodology", {})
    if any(methodology.get(key) != design[key] for key in
           ("primary_procedure", "baseline_procedure", "supporting_procedure", "overall_rule")):
        raise IntegrityError("Certificate endpoint procedure (sidedness, aggregation or level) differs from the "
                             "preregistered design.")
    train_x, train_y = data["train"]
    source_digest = labelled_partition_digest(train_x, train_y)
    if settings["source_train_digest"] != source_digest:
        raise IntegrityError("Certificate training digest differs from the preflight training partition.")
    test_y_arr = np.asarray(test_y)
    if provenance.get("test_label_counts") != {"zeros": int(np.sum(test_y_arr == 0)), "ones": int(np.sum(test_y_arr == 1))}:
        raise IntegrityError("Certificate test class counts differ from the test partition.")
    recount_from_predictions(run_dir, certificate, test_y_arr)
    decision = d4.recompute_certificate_decision(certificate)

    rounds = registered_rounds(design)
    plan = autond_d4.step_plan(rounds, design["epochs_per_round"])
    children =np.random.SeedSequence(design["audit_seed"]).spawn(design["replicates"])
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in children]
    shuffle = LabelShufflePerturbation()
    for row, seed, child in zip(certificate["findings"]["replicates"], seeds, children):
        r = row["replicate"]
        if row["seed"] != seed:
            raise IntegrityError(f"Replicate {r} seed differs from the audit-seed plan.")
        record = row["perturbation_record"]
        if record.get("replicate") != r or record.get("source_train_digest") != source_digest:
            raise IntegrityError(f"Replicate {r} perturbation record is not bound to this replicate/data.")
        on_disk = read_json(run_dir / "fits" / "perturbations" / f"replicate_{r:02d}.perturbation.json",
                            f"replicate {r} perturbation record")
        if on_disk != record:
            raise IntegrityError(f"Replicate {r} perturbation sidecar differs from the certificate.")
        planned = shuffle.permutation_indices(len(train_y), np.random.default_rng(child))
        _, shuffled_y, _ = d4.load_verified_perturbation(
            run_dir / "fits" / "perturbations", replicate=r, seed=seed, perturbation=shuffle,
            source_train_digest=source_digest, source_label_digest=partition_digest({"labels": train_y}),
            audit_seed=design["audit_seed"], train_features=train_x, train_labels=train_y, expected_indices=planned)
        for condition, labels in (("baseline", np.asarray(train_y)), ("perturbed", np.asarray(shuffled_y))):
            selection = row[f"{condition}_selection"]
            identity = {key: selection[key] for key in autond_d4.IDENTITY_FIELDS}
            expected = expected_identity(preflight, data, run_dir, design, replicate=r, seed=seed,
                                         condition=condition, labels=labels)
            for key, value in expected.items():
                if identity[key] != value:
                    raise IntegrityError(f"Replicate {r} {condition} selection has {key}={identity[key]!r}, "
                                         f"expected {value!r}.")
            fit_dir, state_path = d4.checkpoint_paths(run_dir / "fits", r, condition)
            state = autond_d4.load_committed_state(state_path, fit_dir, identity=identity, plan=plan)
            autond_d4.classify_uncommitted(fit_dir, plan=plan, committed=state["completed_steps"],
                                           chain_head=state["record_chain_head"], identity=identity, finished=True)
            recomputed = autond_d4.select_checkpoints(
                fit_dir, plan=plan, rounds=rounds, epochs_per_round=design["epochs_per_round"],
                threshold=STOP_THRESHOLD, identity=identity, chain_head=state["record_chain_head"],
                completed=state["completed_steps"])
            recomputed["retries"] = state["retries"]
            if recomputed != selection or read_json(fit_dir / "selected_checkpoint.json",
                                                    "selection record") != selection:
                raise IntegrityError(f"Replicate {r} {condition}: selection does not match the verified "
                                     "staged epoch history on disk.")
    if certificate.get("autond_pipeline_progression") != pipeline_progression(certificate,
                                                                              design["pipeline_progression_role"]):
        raise IntegrityError("Certificate pipeline progression does not match the verified fit histories.")
    autond = autond_decision(certificate, design["pipeline_progression_role"], design["starting_round"])
    if certificate.get("autond_decision") != autond:
        raise IntegrityError("Certificate AutoND decision does not match the recomputed outcome.")
    return {"integrity": "VERIFIED", "outcome": autond["autond_overall_outcome"],
            "overall_d4_outcome": decision["outcome"], "autond_overall_outcome": autond["autond_overall_outcome"],
            "pipeline_progression_role": autond["pipeline_progression_role"],
            "progression_criterion_met": autond["progression_criterion"]["met"],
            "overall_supported": autond["overall_supported"],
            "shuffled_near_chance_endpoint": decision["primary"]["outcome"],
            "baseline_adequacy_gate": decision["baseline"]["outcome"],
            "paired_degradation_endpoint": {"role": "secondary_non_gating",
                                            "established": decision["supporting"]["established"]},
            "pipeline_progression": certificate["autond_pipeline_progression"]["summary"],
            "replicates": len(seeds), "run_id": manifest["run_id"]}


def _load_selected_model(path: Path):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def rerun_selected_models(run_dir: Path) -> list[dict]:
    """Reload every fit's selected comparison-round checkpoint, re-predict the test partition, compare.

    Call only after verify_run has passed.  Thresholded predictions must equal
    the persisted ones exactly; the maximum raw-score difference is reported.
    """
    from audit.dataset.adapters.autond_d4 import PREDICT_BATCH_SIZE

    run_dir = Path(run_dir)
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    _, data = load_preflight(run_dir)
    test_x = data["test"][0]
    report = []
    for row in certificate["findings"]["replicates"]:
        r = row["replicate"]
        for condition in ("baseline", "perturbed"):
            selection = row[f"{condition}_selection"]
            fit_dir, _ = d4.checkpoint_paths(run_dir / "fits", r, condition)
            checkpoint = fit_dir / selection["checkpoint"]
            if not checkpoint.is_file() or sha256_file(checkpoint) != selection["checkpoint_sha256"]:
                raise IntegrityError(f"Replicate {r} {condition}: selected checkpoint is missing or does not match "
                                     "its selection record.")
            fresh = np.asarray(_load_selected_model(checkpoint).predict(
                test_x, batch_size=PREDICT_BATCH_SIZE, verbose=0)).reshape(-1)
            stored = np.load(d4.prediction_path(run_dir / "fits", r, condition))
            disagreements = int(np.sum((fresh >= 0.5) != (stored >= 0.5))) if fresh.shape == stored.shape else -1
            if disagreements != 0:
                raise IntegrityError(f"Replicate {r} {condition}: the reloaded selected checkpoint (round "
                                     f"{selection['selected_stage_round']}, epoch {selection['selected_epoch']}) does "
                                     f"not reproduce the persisted predictions ({disagreements} label disagreements).")
            report.append({"replicate": r, "condition": condition,
                           "selected_stage_round": selection["selected_stage_round"],
                           "selected_epoch": selection["selected_epoch"],
                           "checkpoint_sha256": selection["checkpoint_sha256"], "label_disagreements": 0,
                           "max_abs_score_difference": float(np.max(np.abs(fresh - stored))) if fresh.size else 0.0})
    return report


def stage_verify(args) -> int:
    if not (Path(args.run_dir) / "result").exists():
        print(json.dumps({"overall_d4_outcome": "NOT_COMPLETED", "overall_supported": False}, indent=2))
        print(f"NO RESULT: {Path(args.run_dir) / 'result'} does not exist; the run is NOT_COMPLETED.")
        return 5
    summary = verify_run(Path(args.run_dir))
    if args.rerun_models:
        summary["reloaded_checkpoints_reproduce_predictions"] = rerun_selected_models(Path(args.run_dir))
    print(json.dumps(summary, indent=2))
    print("INTEGRITY CHECKS PASSED (post-run verification; read-only). "
          f"AutoND overall outcome: {summary['outcome']} (engine: {summary['overall_d4_outcome']})"
          f"{'' if summary['overall_supported'] else ' (NOT a scientific pass)'}.")
    return 0


# ============================================================
# CLI
# ============================================================


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoND D4 label-shuffle controlled perturbation audit.")
    sub = parser.add_subparsers(dest="stage", required=True)
    check = sub.add_parser("protocol-check", help="structural review of the preregistration (read-only)")
    check.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    check.add_argument("--require-approved", action="store_true")
    for name, text in (("preflight", "HUMAN-AUTHORISED: generate and audit partitions"),
                       ("run", "HUMAN-AUTHORISED PRODUCTION: train, evaluate, certify")):
        stage = sub.add_parser(name, help=text)
        stage.add_argument("--preregistration", type=Path, required=True)
        stage.add_argument("--preregistration-sha256", required=True)
        stage.add_argument("--run-dir", type=Path, required=True)
        stage.add_argument("--autond-root", default=DEFAULT_AUTOND_ROOT)
    verify = sub.add_parser("verify", help="independent post-run verification (read-only)")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--rerun-models", action="store_true",
                        help="also reload every selected checkpoint and re-predict the test partition (needs TF)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = {"protocol-check": stage_protocol_check, "preflight": stage_preflight,
              "run": stage_run, "verify": stage_verify}
    try:
        return stages[args.stage](args)
    except IntegrityError as exc:
        print(f"REFUSED ({args.stage}): {exc}", file=sys.stderr)
        if args.stage == "run":
            print("D4 OUTCOME: NOT_COMPLETED (no result written; not a scientific result).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
