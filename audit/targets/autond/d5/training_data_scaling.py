"""
AutoND D5 -- fixed round-5, fixed-budget training-data scaling (target-specific driver).

Question: under a fixed AutoND round-5 training protocol (pinned DBitNet, Adam(amsgrad), MSE, batch 5000,
a fixed epoch budget, final-epoch model), how does training-set size affect held-out accuracy?

Reuse, unchanged:
    audit.dataset.d5_training_data_scaling   run_d5 (independent replicate datasets, nested prefixes, same
                                             model seed across sizes within a replicate, one fixed test set,
                                             paired t vs the smallest size, Holm, bootstrap, slopes) and
                                             generate_certificate
    audit.dataset.adapters.autond            AutoNDDatasetAdapter: pinned source binding (commit + digests
                                             checked BEFORE import), main.make_train_data
    audit.dataset.experiments.d0_autond_canonical.AUTOND_SOURCE_PIN   the pin, re-checked against the files
    audit.dataset.integrity                  anchored/committed preregistration, source identity, overlap
                                             audit, exclusive writes

What this driver adds, without touching any of those modules:
    * gates before anything is created, enforced inside execute() itself (not only by the `run` stage): SciPy
      present, preregistration approved + anchored + committed, fixed design equals this driver, run values
      equal the approved values, every source file clean at HEAD and equal to the recorded identity, pinned
      binding;
    * a NO-RESUME policy: the run directory must be new or empty and is claimed exclusively; an interrupted
      run is preserved and never resumed (the engine's resume path is therefore never exercised);
    * data checks on every generated partition (shape, dtype, value set, row count, class counts) and an
      exact train/test overlap audit before training, under the registered policy;
    * a target evaluation log binding each condition's final checkpoint SHA-256 at evaluation time;
    * independent verification and a certificate (see .verification).

NOT reproduced, by design (see the preregistration): AutoND's staged multi-round pipeline, its validation
data, its min-val_loss checkpoint selection and its stopping rule.  The engine fits without validation data
and evaluates the final-epoch model.  This is a fixed round-5 scaling experiment, not full-pipeline scaling.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audit.dataset.integrity import (
    REPOSITORY_ROOT,
    IntegrityError,
    audit_partition_overlap,
    sha256_file,
    source_identity,
    verify_anchored_committed_record,
    write_json_exclusive,
)
from audit.targets.autond.d5 import verification as V

# ============================================================
# Fixed design (must equal the preregistration's fixed_design)
# ============================================================

CIPHER = "speck3264"
SCENARIO = "single-key"
DELTA_PLAIN = 0x400000
DELTA_KEY = 0x0
NUM_ROUNDS = 5
FEATURE_WIDTH = 64
ENGINE_MODULE = "audit.dataset.d5_training_data_scaling"
ENGINE_SCHEMA = "2.1"
MODEL = {"architecture": "dbitnet", "builder": "dbitnet.make_model(64)", "optimizer": "Adam(amsgrad=True)",
         "loss": "mse", "metrics": ["acc"], "shuffle": True, "learning_rate_schedule": None}
TRAINING = {"validation_data": None, "model_selection": "final_epoch_model", "stopping_rule": None,
            "initialization": "tf.keras.utils.set_random_seed(replicate_seed) immediately before building"}
EXPERIMENT_NAME = "AutoND SPECK32/64 r5 DBitNet training-data scaling with nested independent replicates"
PREREGISTRATION_SCHEMA = "d5-autond-preregistration-1"
PENDING = "PENDING_HUMAN_APPROVAL"
CONFIRM_RUN = "RUN_AUTOND_D5_PRODUCTION"
DEFAULT_PREREGISTRATION = (REPOSITORY_ROOT / "audit/targets/autond/preregistrations/"
                           "d5_autond_speck3264_r5_training_data_scaling.json")
DEFAULT_AUTOND_ROOT = os.environ.get("AUTOND_ROOT", "~/Documents/AutoND-upstream")

#: Repository-relative; resolved against V.repository_root() at call time.
SOURCE_PATHS = (
    "audit/dataset/d5_training_data_scaling.py",
    "audit/dataset/integrity.py",
    "audit/dataset/adapters/autond.py",
    "audit/dataset/experiments/d0_autond_canonical.py",
    "audit/targets/autond/d5/__init__.py",
    "audit/targets/autond/d5/training_data_scaling.py",
    "audit/targets/autond/d5/verification.py",
)

DATASET_REPLAY_REASON = (
    "AutoND dataset generation is main.make_train_data at the pinned commit (via AutoNDDatasetAdapter."
    "generate_partition), which draws keys, plaintexts, labels and the random-class substitution from "
    "os.urandom. The integer audit/replicate seed does not control dataset content and cannot regenerate a "
    "bit-identical dataset. The persisted dataset files and their file and logical SHA-256 hashes (including "
    "every nested prefix) are the reproducibility object.")
REPLICATE_SEED_ROLE = (
    "model_initialization_and_training_RNG_stream (tf.keras.utils.set_random_seed before building the model; "
    "also seeds Keras' shuffle order) and the audit's own RNG stream; NOT dataset generation (os.urandom).")

CLAIM_SCOPE = {
    "estimand": ("Per training size n: mean over independent replicates of the final-epoch held-out accuracy of "
                 "the pinned DBitNet trained for the registered epoch budget on the first n rows of a replicate's "
                 "independent round-5 dataset; contrasts are within-replicate differences against the smallest size."),
    "training_policy": ("Fixed-EPOCH policy: every size is trained for the same number of epochs at batch 5000, so "
                        "a size-n condition receives epochs * ceil(n / 5000) optimizer updates (2.5M: 5,000; 10M: "
                        "20,000 at 10 epochs). The estimand is performance under this policy; dataset size and "
                        "training compute vary together and are not separated."),
    "conditional_on": ["this fixed test set (inference is conditional on it)", "round 5, delta 0x400000, single-key",
                       "the registered epoch budget, batch size, optimizer and final-epoch selection",
                       "the pinned upstream commit and this software environment"],
    "does_not_establish": ["a universal or extrapolable scaling law", "cryptographic learning",
                           "behaviour of AutoND's staged full pipeline, its validation-based selection or stopping rule",
                           "equivalence between sizes from a non-significant contrast",
                           "an effect of dataset size independent of training compute (updates scale with n)",
                           "statistical independence from the absence of exact duplicates"],
}

DECISIONS = ("training_sizes", "replicates", "epochs", "batch_size", "test_samples", "alpha", "bootstrap_replicates",
             "audit_seed", "primary_analysis", "practical_effect_threshold", "practical_effect_rule",
             "partition_overlap_policy", "within_partition_duplicates_policy", "failure_handling",
             "verification_tolerance")
OVERLAP_POLICIES = ("refuse_on_any_train_test_feature_row", "record_only")
DUPLICATE_POLICIES = ("record_only", "refuse_on_any_within_partition_duplicate")
PRACTICAL_RULES = ("point_estimate_at_least_threshold", "t_interval_excludes_threshold_band")
ENGINE_HOLM_ALPHA = 0.05      # hard-coded in the engine's _aggregate; the registered alpha must equal it

EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_NOT_VERIFIED, EXIT_NOT_COMPLETED = 0, 1, 2, 3, 4


# ============================================================
# Preregistration
# ============================================================


def load_json_strict(path: Path) -> dict:
    """Parse JSON refusing duplicate keys (a later duplicate would silently override an approved value)."""
    def no_duplicates(pairs):
        keys = [k for k, _ in pairs]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise IntegrityError(f"{path}: duplicate JSON keys {dupes}")
        return dict(pairs)

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise IntegrityError(f"{path} is not a JSON object.")
    return value


def review_preregistration(record: dict) -> dict:
    problems, pending, values = [], [], {}
    if record.get("schema_version") != PREREGISTRATION_SCHEMA:
        problems.append(f"schema_version is {record.get('schema_version')!r}, expected {PREREGISTRATION_SCHEMA!r}")
    decisions = record.get("decisions") or {}
    for name in DECISIONS:
        entry = decisions.get(name)
        if not isinstance(entry, dict) or "status" not in entry or "value" not in entry:
            problems.append(f"decision {name!r} is missing or lacks status/value")
            continue
        if not entry.get("rationale"):
            problems.append(f"decision {name!r} has no rationale")
        if entry["status"] == PENDING or entry["value"] is None:
            pending.append(f"decisions.{name}")
            continue
        if entry["status"] != "APPROVED":
            problems.append(f"decision {name!r} has status {entry['status']!r}")
        if entry.get("options") is not None and entry["value"] not in entry["options"]:
            problems.append(f"decision {name!r} value {entry['value']!r} is not one of {entry['options']}")
        values[name] = entry["value"]
    extra = sorted(set(decisions) - set(DECISIONS))
    if extra:
        problems.append(f"unregistered decisions {extra}")
    if record.get("status") != "APPROVED":
        pending.append("status")
    for field in ("approved", "approved_by", "approved_utc", "approval_statement", "data_examined_before_approval"):
        value = record.get(field)
        if value in (None, False, PENDING):
            pending.append(field)
    return {"problems": problems, "pending": pending, "values": values}


def check_fixed_design(record: dict) -> list[str]:
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    fixed = record.get("fixed_design", {})
    expected = {
        ("target", "cipher"): CIPHER, ("target", "scenario"): SCENARIO,
        ("target", "delta_plain_hex"): hex(DELTA_PLAIN), ("target", "delta_key_hex"): hex(DELTA_KEY),
        ("target", "num_rounds"): NUM_ROUNDS, ("target", "feature_width"): FEATURE_WIDTH,
        ("upstream", "commit"): AUTOND_SOURCE_PIN["upstream_commit"],
        ("upstream", "source_digests"): AUTOND_SOURCE_PIN["source_digests"],
        ("engine", "module"): ENGINE_MODULE, ("engine", "schema_version"): ENGINE_SCHEMA,
        ("model", None): MODEL, ("training", None): TRAINING,
    }
    problems = []
    for (section, key), want in expected.items():
        have = fixed.get(section) if key is None else fixed.get(section, {}).get(key)
        if have != want:
            problems.append(f"fixed_design.{section}{'' if key is None else '.' + key} is {have!r}; "
                            f"this driver implements {want!r}")
    return problems


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_decision_values(values: dict) -> list[str]:
    problems = []
    sizes = values.get("training_sizes")
    if sizes is not None:
        if not (isinstance(sizes, list) and len(sizes) >= 2 and all(_positive_int(n) for n in sizes)
                and sizes == sorted(set(sizes))):
            problems.append("decisions.training_sizes must be >= 2 distinct positive integers in ascending order")
    for name, minimum in (("replicates", 2), ("epochs", 1), ("batch_size", 1), ("test_samples", 1),
                          ("bootstrap_replicates", 1000)):
        if name in values and not (_positive_int(values[name]) and values[name] >= minimum):
            problems.append(f"decisions.{name} must be an integer >= {minimum}")
    if "audit_seed" in values and not (isinstance(values["audit_seed"], int) and values["audit_seed"] >= 0):
        problems.append("decisions.audit_seed must be a non-negative integer")
    if "alpha" in values and values["alpha"] != ENGINE_HOLM_ALPHA:
        problems.append(f"decisions.alpha must equal the engine's fixed Holm level {ENGINE_HOLM_ALPHA}")
    if "practical_effect_threshold" in values:
        t = values["practical_effect_threshold"]
        if not (isinstance(t, float) and 0 < t < 0.5):
            problems.append("decisions.practical_effect_threshold must be a float in (0, 0.5), accuracy fraction")
    if "practical_effect_rule" in values and values["practical_effect_rule"] not in PRACTICAL_RULES:
        problems.append(f"decisions.practical_effect_rule must be one of {PRACTICAL_RULES}")
    if "partition_overlap_policy" in values and values["partition_overlap_policy"] not in OVERLAP_POLICIES:
        problems.append(f"decisions.partition_overlap_policy must be one of {OVERLAP_POLICIES}")
    if ("within_partition_duplicates_policy" in values
            and values["within_partition_duplicates_policy"] not in DUPLICATE_POLICIES):
        problems.append(f"decisions.within_partition_duplicates_policy must be one of {DUPLICATE_POLICIES}")
    tol = values.get("verification_tolerance")
    if tol is not None and not (isinstance(tol, dict) and set(tol) == {"accuracy_abs", "loss_abs"}
                                and all(isinstance(v, float) and v > 0 for v in tol.values())):
        problems.append("decisions.verification_tolerance must give positive floats accuracy_abs and loss_abs")
    return problems


def load_approved_preregistration(path: Path, anchor: str) -> tuple[dict, dict, dict]:
    """Anchored + committed + approved + matching this driver; returns (record, values, commit)."""
    record = verify_anchored_committed_record(path, expected_sha256=anchor, what="AutoND D5 preregistration")
    load_json_strict(path)
    commit = record.pop("_commit")
    review = review_preregistration(record)
    problems = review["problems"] + check_fixed_design(record) + check_decision_values(review["values"])
    if review["pending"]:
        problems.append(f"not approved; pending: {review['pending']}")
    if record.get("approved") is not True:
        problems.append("approved is not true")
    if problems:
        raise IntegrityError(f"Preregistration refused: {problems}")
    approved = datetime.strptime(record["approved_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if commit["committed_utc"] and approved > datetime.fromisoformat(commit["committed_utc"]):
        raise IntegrityError("approved_utc is later than the commit that recorded the approval.")
    return record, review["values"], commit


# ============================================================
# Environment and source gates
# ============================================================


def require_clean_sources() -> dict:
    root = V.repository_root()
    identity = source_identity([root / p for p in SOURCE_PATHS], root=root)
    dirty = sorted(rel for rel, entry in identity["files"].items() if not entry["clean_at_head"])
    if dirty:
        raise IntegrityError(f"Source files are not clean at HEAD: {dirty}. Commit them before a run.")
    return identity


def authorize_execution(values: dict, *, prereg: dict, sources: dict, binding) -> None:
    """Every gate of the `run` stage, re-derived on the execution path itself; nothing is taken from the caller.

    The preregistration is reloaded from its recorded path and anchor and must be approved, committed and match
    this driver; the caller's values, commit and protocol id must equal what was approved; the caller's source
    identity must equal the files clean at HEAD now; and the binding must be the verified pin.
    """
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    rel = prereg.get("path") if isinstance(prereg, dict) else None
    if not (isinstance(rel, str) and rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts):
        raise IntegrityError(f"Execution refused: preregistration path {rel!r} is not repository-relative.")
    record, approved, commit = load_approved_preregistration(V.repository_root() / rel, prereg.get("sha256"))
    if values != approved:
        changed = sorted(k for k in set(values) | set(approved) if values.get(k) != approved.get(k))
        raise IntegrityError(f"Execution refused: values differ from the approved preregistration: {changed}")
    if prereg.get("commit") != commit or prereg.get("protocol_id") != record.get("protocol_id"):
        raise IntegrityError("Execution refused: the preregistration commit or protocol id supplied by the caller "
                             "is not the approved record's.")
    if sources != require_clean_sources():
        raise IntegrityError("Execution refused: the source identity supplied by the caller differs from the "
                             "source files clean at HEAD.")
    if not (getattr(binding, "pin_verified_preimport", False)
            and getattr(binding, "expected_upstream_commit", None) == AUTOND_SOURCE_PIN["upstream_commit"]
            and all((getattr(binding, "source_digests", None) or {}).get(name) == digest
                    for name, digest in AUTOND_SOURCE_PIN["source_digests"].items())):
        raise IntegrityError("Execution refused: the AutoND binding is not the verified pin.")


def environment_record() -> dict:
    import keras
    import scipy
    import tensorflow as tf

    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "tensorflow": tf.__version__, "keras": keras.__version__,
            "physical_gpus": [d.name for d in tf.config.list_physical_devices("GPU")]}


def bind_autond(autond_root) -> Any:
    """The pinned AutoND binding for this design; refuses before import on any pin mismatch."""
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    binding = AutoNDDatasetAdapter(
        autond_root=autond_root, cipher_name=CIPHER, delta_plain_int=DELTA_PLAIN, num_rounds=NUM_ROUNDS,
        delta_key_int=DELTA_KEY, scenario=SCENARIO,
        expected_source_digests=AUTOND_SOURCE_PIN["source_digests"],
        expected_upstream_commit=AUTOND_SOURCE_PIN["upstream_commit"])
    if not binding.pin_verified_preimport:
        raise IntegrityError("The AutoND pin was not verified before import.")
    if binding.feature_bits != FEATURE_WIDTH or binding.num_rounds != NUM_ROUNDS:
        raise IntegrityError(f"Binding has feature_bits={binding.feature_bits}, rounds={binding.num_rounds}.")
    return binding


def pinned_module(binding, name: str):
    """A module bound by the pinned import graph, refused unless it resolves inside the checkout."""
    from audit.dataset.adapters.autond import _is_within, _module_locations

    module = sys.modules.get(name)
    if module is None:
        raise IntegrityError(f"Pinned module {name!r} is not bound.")
    locations = _module_locations(module)
    if not locations or not all(_is_within(loc, binding.autond_root) for loc in locations):
        raise IntegrityError(f"Module {name!r} resolves to {locations}, outside {binding.autond_root}.")
    return module


def make_model_factory(binding) -> Callable[[int], Any]:
    dbitnet = pinned_module(binding, "dbitnet")

    def model_factory(seed: int):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(int(seed))
        model = dbitnet.make_model(2 * int(binding.plain_bits))
        model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss="mse", metrics=["acc"])
        return model

    return model_factory


# ============================================================
# Data checks
# ============================================================


def validate_partition(x, y, samples: int, *, width: int = FEATURE_WIDTH) -> dict:
    """Refuse a partition that does not have the verified AutoND representation; return its class counts."""
    x = np.asarray(x)
    y = np.asarray(y)
    problems = []
    if x.ndim != 2 or x.shape != (int(samples), int(width)):
        problems.append(f"features have shape {x.shape}, expected {(int(samples), int(width))}")
    if y.ndim != 1 or len(y) != int(samples):
        problems.append(f"labels have shape {y.shape}, expected ({int(samples)},)")
    if x.dtype != np.uint8 or y.dtype != np.uint8:
        problems.append(f"dtypes are {x.dtype}/{y.dtype}, expected uint8/uint8")
    if not problems:
        if int(x.max(initial=0)) > 1:
            problems.append("features contain values outside {0, 1}")
        if int(y.max(initial=0)) > 1:
            problems.append("labels contain values outside {0, 1}")
    if problems:
        raise IntegrityError(f"Generated partition refused: {problems}")
    ones = int(y.sum(dtype=np.int64))
    from scipy.stats import binomtest

    return {"rows": int(samples), "label_ones": ones, "label_zeros": int(samples) - ones,
            "label_ones_fraction": ones / int(samples),
            "class_balance_binomial_p_two_sided": float(binomtest(ones, int(samples), 0.5).pvalue),
            "class_balance_role": "descriptive; labels are independent fair bits by construction; not gating"}


def logical_hashes(x, y) -> dict:
    """The engine's own logical content hash of the audited arrays, so verification can bind each data-audit
    record to the dataset the engine persisted (and trained/evaluated on)."""
    from audit.dataset.d5_training_data_scaling import _array_logical_sha256

    return {"x_logical_sha256": _array_logical_sha256(x), "y_logical_sha256": _array_logical_sha256(y)}


class DataGate:
    """Wraps the engine's dataset factories: validate, overlap-audit against the test set, record, then return."""

    def __init__(self, binding, run_dir: Path, values: dict) -> None:
        self.binding = binding
        self.audit_dir = Path(run_dir) / "data_audit"
        self.values = values
        self.test: tuple[np.ndarray, np.ndarray] | None = None
        self.replicate = 0

    def test_factory(self, samples: int, seed: int):
        del seed      # os.urandom; the seed does not control content
        if self.test is not None:
            raise IntegrityError("The fixed test set was requested twice.")
        x, y = self.binding.generate_partition(int(samples))
        record = validate_partition(x, y, samples)
        record.update(logical_hashes(x, y))
        overlap = audit_partition_overlap({"test": (x, y)})
        record["within_duplicates"] = overlap["within"]["test"]
        self._apply_duplicate_policy("test", overlap["within"]["test"])
        write_json_exclusive(self.audit_dir / "test.json", record)
        self.test = (x, y)
        return x, y

    def train_factory(self, samples: int, seed: int):
        del seed
        if self.test is None:
            raise IntegrityError("A training partition was requested before the fixed test set.")
        self.replicate += 1
        x, y = self.binding.generate_partition(int(samples))
        record = validate_partition(x, y, samples)
        record.update(logical_hashes(x, y))
        overlap = audit_partition_overlap({"test": self.test, "train": (x, y)})
        record["within_duplicates"] = overlap["within"]["train"]
        record["train_test_overlap"] = overlap["cross"]["test__train"]
        record["overlap_method"] = overlap["method"]
        write_json_exclusive(self.audit_dir / f"replicate_{self.replicate:02d}.json", record)
        if (self.values["partition_overlap_policy"] == "refuse_on_any_train_test_feature_row"
                and overlap["cross"]["test__train"]["rows"] > 0):
            raise IntegrityError(f"Replicate {self.replicate}: {overlap['cross']['test__train']['rows']} training "
                                 "rows equal a test row; the registered policy refuses. Data are kept, not regenerated.")
        self._apply_duplicate_policy(f"replicate {self.replicate}", overlap["within"]["train"])
        return x, y

    def _apply_duplicate_policy(self, what: str, within: dict) -> None:
        if (self.values["within_partition_duplicates_policy"] == "refuse_on_any_within_partition_duplicate"
                and within["duplicate_rows"] > 0):
            raise IntegrityError(f"{what}: {within['duplicate_rows']} within-partition duplicate rows; refused.")


class EvaluationRecorder:
    """The engine's evaluate_model hook: evaluates, then binds the condition's final checkpoint digest."""

    def __init__(self, run_dir: Path, values: dict) -> None:
        from audit.dataset import d5_training_data_scaling as engine

        self.engine = engine
        self.root = Path(run_dir) / V.ENGINE_DIR
        self.log = Path(run_dir) / V.TARGET_LOG
        self.values = values
        self.order = V.expected_conditions(values["training_sizes"], values["replicates"])
        self.index = 0

    def evaluate(self, model, test_x, test_y):
        loss, accuracy = model.evaluate(test_x, test_y, batch_size=int(self.values["batch_size"]), verbose=0)
        if self.index >= len(self.order):
            raise IntegrityError("The engine evaluated more conditions than are registered.")
        replicate, size = self.order[self.index]
        self.index += 1
        paths = self.engine.condition_paths(self.root, replicate, size)
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        if (state.get("replicate"), state.get("training_samples"), state.get("completed_epochs")) != (
                replicate, size, int(self.values["epochs"])):
            raise IntegrityError(f"Evaluation order does not match condition {V.condition_name(replicate, size)}: "
                                 f"state reports {state.get('replicate')}/{state.get('training_samples')}.")
        V.append_log_record(self.log, {
            "condition": V.condition_name(replicate, size), "replicate": replicate, "training_samples": size,
            "checkpoint": str(paths["checkpoint"].relative_to(self.root.parent)),
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "engine_test_accuracy": float(accuracy), "engine_test_loss": float(loss),
            "evaluated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        return float(loss), float(accuracy)


# ============================================================
# Run directory (no resume)
# ============================================================


def new_run_id() -> str:
    return f"d5-autond-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def claim_run_directory(run_dir: Path, claim: dict) -> None:
    """Use only a new or empty directory, claimed by an exclusive run_claim.json.  Never resumes."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        if not run_dir.is_dir():
            raise IntegrityError(f"{run_dir} exists and is not a directory.")
        entries = sorted(p.name for p in run_dir.iterdir())
        if entries:
            raise IntegrityError(f"{run_dir} is not empty ({entries[:5]}). D5 runs are never resumed or reused: "
                                 "preserve it and start a new run directory.")
    else:
        run_dir.mkdir(parents=True)
    write_json_exclusive(run_dir / V.RUN_CLAIM, claim)


# ============================================================
# Execution
# ============================================================


def engine_manifest(values: dict, run_id: str, binding, prereg: dict) -> dict:
    from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

    return {
        "dataset_id": binding.dataset_id, "run_id": run_id, "preregistration": prereg,
        "target": {"cipher": CIPHER, "scenario": SCENARIO, "delta_plain_hex": hex(DELTA_PLAIN),
                   "delta_key_hex": hex(DELTA_KEY), "num_rounds": NUM_ROUNDS, "feature_width": FEATURE_WIDTH},
        "model": MODEL, "training": TRAINING,
        "upstream": {"commit": AUTOND_SOURCE_PIN["upstream_commit"], "bound_source_digests": binding.source_digests},
        "generation": {"procedure": "main.make_train_data(encrypt, 32, 64, n, 5, delta_plain_bits, 0)",
                       "randomness_source": "os.urandom", "deterministic_seed": None},
    }


def execute(values: dict, *, run_dir: Path, binding, prereg: dict, sources: dict, environment: dict) -> int:
    """Gate, claim the directory, run the unmodified engine, then certify.  Trusts nothing from its caller."""
    # On the execution path itself, before the directory is claimed: without SciPy the engine would compute
    # its p-values with an anti-conservative normal fallback; without approval and clean sources nothing runs.
    V.require_statistics_dependencies()
    authorize_execution(values, prereg=prereg, sources=sources, binding=binding)
    from audit.dataset import d5_training_data_scaling as engine

    run_id = new_run_id()
    claim = {"schema": "autond-d5-run-claim-1", "run_id": run_id,
             "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "preregistration": prereg, "values": values, "source_identity": sources, "environment": environment,
             "upstream": {"commit": binding.expected_upstream_commit, "bound_source_digests": binding.source_digests,
                          "autond_root": str(binding.autond_root)},
             "claim_scope": CLAIM_SCOPE, "policy": "no resume; an interrupted run is preserved and a new run started"}
    claim_run_directory(run_dir, claim)
    gate = DataGate(binding, run_dir, values)
    recorder = EvaluationRecorder(run_dir, values)
    try:
        result = engine.run_d5(
            root=Path(run_dir) / V.ENGINE_DIR, training_sizes=values["training_sizes"],
            test_samples=int(values["test_samples"]), replicates=int(values["replicates"]),
            audit_seed=int(values["audit_seed"]), total_epochs=int(values["epochs"]),
            batch_size=int(values["batch_size"]), train_dataset_factory=gate.train_factory,
            test_dataset_factory=gate.test_factory, model_factory=make_model_factory(binding),
            training_callbacks_factory=None, evaluate_model=recorder.evaluate, experiment_name=EXPERIMENT_NAME,
            bootstrap_replicates=int(values["bootstrap_replicates"]),
            manifest=engine_manifest(values, run_id, binding, prereg),
            dataset_replay_reason=DATASET_REPLAY_REASON, replicate_seed_role=REPLICATE_SEED_ROLE)
    except Exception as exc:  # noqa: BLE001 - every failure leaves the directory intact and recorded
        write_json_exclusive(Path(run_dir) / "run_failure.json", {
            "run_id": run_id, "outcome": "NOT_COMPLETED", "error_type": type(exc).__name__, "error": str(exc),
            "policy": "this run is not resumed; start a new run directory"})
        print(f"NOT_COMPLETED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_NOT_COMPLETED
    certificate = engine.generate_certificate(
        result, dataset_id=binding.dataset_id, dataset_version=f"pinned-{binding.expected_upstream_commit[:12]}-r5",
        generation_procedure="AutoNDDatasetAdapter.generate_partition -> main.make_train_data",
        generation_parameters={**binding.dataset_provenance(train_samples=max(values["training_sizes"]),
                                                            validation_samples=0,
                                                            test_samples=int(values["test_samples"]))[
                                   "generation_parameters"], "replicates": int(values["replicates"])},
        audit_seed=int(values["audit_seed"]))
    V.write_result_json(Path(run_dir) / V.RESULT_DIR / V.ENGINE_CERTIFICATE, certificate)
    return certify(run_dir)


def certify(run_dir: Path) -> int:
    report = V.verify_run(run_dir)
    V.write_result_json(Path(run_dir) / V.RESULT_DIR / V.TARGET_VERIFICATION, report)
    certificate = V.build_certificate(run_dir, report)
    V.write_result_json(Path(run_dir) / V.RESULT_DIR / V.TARGET_CERTIFICATE, certificate)
    print(f"D5 target certificate: {certificate['status']}"
          + ("" if certificate["status"] == "VERIFIED" else f" -- {len(certificate['problems'])} problem(s)"))
    return EXIT_OK if certificate["status"] == "VERIFIED" else EXIT_NOT_VERIFIED


# ============================================================
# Stages
# ============================================================


def stage_protocol_check(args) -> int:
    try:
        record = load_json_strict(args.preregistration)
    except (IntegrityError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_ERROR
    review = review_preregistration(record)
    problems = review["problems"] + check_fixed_design(record) + check_decision_values(review["values"])
    print(json.dumps({"sha256": sha256_file(args.preregistration), "problems": problems,
                      "pending": review["pending"]}, indent=2))
    if problems:
        return EXIT_ERROR
    if args.require_approved and review["pending"]:
        return EXIT_NOT_VERIFIED
    return EXIT_OK


def stage_run(args) -> int:
    if args.confirm != CONFIRM_RUN:
        print(f"REFUSED: production D5 needs --confirm {CONFIRM_RUN}.", file=sys.stderr)
        return EXIT_REFUSED
    try:
        stats = V.require_statistics_dependencies()
        record, values, commit = load_approved_preregistration(args.preregistration, args.preregistration_sha256)
        sources = require_clean_sources()
        binding = bind_autond(args.autond_root)
        environment = {**environment_record(), **stats}
        prereg = {"path": Path(args.preregistration).resolve().relative_to(V.repository_root().resolve()).as_posix(),
                  "sha256": args.preregistration_sha256, "commit": commit, "protocol_id": record.get("protocol_id")}
    except Exception as exc:  # noqa: BLE001 - nothing has been created yet
        print(f"REFUSED (before any file was created): {exc}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return execute(values, run_dir=args.run_dir, binding=binding, prereg=prereg, sources=sources,
                       environment=environment)
    except IntegrityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def stage_verify(args) -> int:
    """Read-only re-verification; compares checkpoint digests with an existing certificate if one exists."""
    V.require_statistics_dependencies()
    report = V.verify_run(args.run_dir, reevaluate=not args.no_reevaluate)
    certificate_path = Path(args.run_dir) / V.RESULT_DIR / V.TARGET_CERTIFICATE
    if certificate_path.is_file():
        certified = V.checkpoint_digests_from_certificate(json.loads(certificate_path.read_text(encoding="utf-8")))
        now = {name: entry.get("checkpoint_sha256") for name, entry in report.get("conditions", {}).items()}
        changed = sorted(name for name in certified if now.get(name) != certified[name])
        if changed:
            report["problems"].append(f"checkpoints changed since certification: {changed}")
            report["status"] = "NOT_VERIFIED"
    print(json.dumps({"status": report["status"], "problems": report.get("problems", [])}, indent=2))
    # --no-reevaluate can at most report INTEGRITY_CONSISTENT_NOT_REEVALUATED, never VERIFIED.
    return EXIT_NOT_VERIFIED if report["status"] == "NOT_VERIFIED" else EXIT_OK


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    check = sub.add_parser("protocol-check", help="structural review of the preregistration (read-only)")
    check.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    check.add_argument("--require-approved", action="store_true")
    run = sub.add_parser("run", help="gated production run in a NEW run directory")
    run.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    run.add_argument("--preregistration-sha256", required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--autond-root", default=DEFAULT_AUTOND_ROOT)
    run.add_argument("--confirm", default="")
    verify = sub.add_parser("verify", help="independent re-verification of a completed run (read-only)")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--no-reevaluate", action="store_true", help="skip checkpoint reload (hashes only)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"protocol-check": stage_protocol_check, "run": stage_run, "verify": stage_verify}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
