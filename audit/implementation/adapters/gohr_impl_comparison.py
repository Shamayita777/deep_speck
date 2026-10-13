#!/usr/bin/env python3
"""
Gohr/Speck32/64 adapter for the II-4 IMPLEMENTATION comparison.

ALL case-specific knowledge for this experiment lives here. The generic
engine (comparison.py, ii4_comparison_design.py, ii4_comparison_runner.py,
ii4_comparison_certificate.py, ii4_analysis.py) contains none of it.

THE TWO ARMS
------------
reference
    The ORIGINAL Gohr source at audit/implementation/reference/, loaded
    by path and NEVER modified: `make_resnet` and `cyclic_lr` are taken
    from train_nets.py as published, and `set_seed` is the reference's
    own. The reference's `train_speck_distinguisher` cannot be called
    directly because it generates its own data internally, and a paired
    block design requires both arms to train on the SAME data. A thin
    driver therefore reproduces that function's fit call exactly - same
    optimizer, loss, metric, schedule, batch size, epochs, shuffle - on
    data supplied from outside. The driver adds no modelling decision.

independent
    The II-2 independent reimplementation at
    audit/implementation/ii2_independent/, likewise unmodified:
    `build_gohr_model` from model_independent.py and
    `make_learning_rate_schedule` from train_independent.py. Its
    `train()` entry point is bypassed for the same reason: it generates
    its own data.

WHAT IS COMPARED, AND WHAT IS NOT
---------------------------------
COMPARED: model construction, compilation, learning-rate schedule and
the fit invocation, as each implementation writes them.

NOT COMPARED (shared): dataset generation. Both arms train on one block
dataset produced by the REFERENCE generator. This is a deliberate
design choice, not an oversight - if each arm generated its own data the
design would not be paired and the block would not be a block. Its
evidential basis is the II-2 M1 deterministic verification, which
established exact agreement between the independent dataset builder and
the reference one on the published test vector, the key schedule over
1000 keys, round traces over 10000 word states, the binary encoding and
dataset construction. The limitation is recorded in the preregistration
and propagates into the certificate.

ALSO NOT COMPARED: the Keras metric name. The reference compiles with
metrics=['acc'] and the independent with metrics=['accuracy']. Both
resolve to binary accuracy, and in any case neither determines the
endpoint: `evaluate_terminal` below computes the sealed-test accuracy
for BOTH arms with one function, so the measurement instrument is
identical by construction rather than by coincidence.

READ-ONLY. Nothing under reference/ or ii2_independent/ is written,
moved or modified. The reference directory is placed on sys.path only
for the duration of an import, because reference/train_nets.py begins
with `import speck as sp`.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

import audit as _audit_pkg

from audit.implementation.architecture_signature import (
    architecture_signature,
    diff_signatures,
    verify_signature,
)
from audit.implementation.arm_recovery import (
    ARM_TRAINED,
    ArmRecoveryRecord,
    EpochLog,
    InjectedInterruption,
    build_callbacks,
    reseed_for_session,
    resume_point,
)
from audit.implementation.durable_state import (
    CommittedCheckpointStore,
    sha256_file as _durable_sha256,
)

REPO_ROOT = Path(_audit_pkg.__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "audit" / "implementation" / "reference"
INDEPENDENT_DIR = REPO_ROOT / "audit" / "implementation" / "ii2_independent"

ARM_REFERENCE = "reference"
ARM_INDEPENDENT = "independent"

#: The declared Gohr 5-round protocol, transcribed from the reference
#: source in II-1 and hash-verified there. Both arms run exactly this.
GOHR_PROTOCOL = {
    "rounds": 5,
    "differential": (0x0040, 0x0000),
    "depth": 10,
    "reg_param": 1e-5,
    "epochs": 200,
    "batch_size": 5000,
    "optimizer": "adam",
    "loss": "mse",
    "lr_schedule": "cyclic_lr(10, 0.002, 0.0001)",
    "shuffle": True,
    "train_samples": 10 ** 7,
    "validation_samples": 10 ** 6,
    "sealed_test_samples": 10 ** 6,
}

#: The DECLARED architecture, captured once as a full ordered signature
#: and committed beside this adapter. Both arms are checked against it,
#: and against each other, before any production training.
#:
#: WHY A FILE AND NOT A CONSTANT. A digest alone tells an operator that
#: something differs, not what. The committed file holds the complete
#: signature, so a mismatch is reported as a field-level diff - "layer 3
#: kernel_regularizer l2=1e-05 vs 1e-04" - which distinguishes a
#: material change from a cosmetic serialization change introduced by a
#: different Keras version. Regenerate with
#:
#:     python3 -m audit.implementation.run_ii4_impl --capture-architecture
#:
#: A regenerated file whose diff against the committed one is non-empty
#: is a CHANGE OF DECLARED ARCHITECTURE and requires a new freeze, not a
#: quiet overwrite.
DECLARED_ARCHITECTURE_PATH = (
    REPO_ROOT / "audit" / "implementation" / "adapters" / "gohr_declared_architecture.json")

#: Coarse counts, retained ONLY as a human-readable cross-check in
#: reports. They are not the structural test; the ordered signature is.
#: Counts cannot distinguish an inline Conv1D activation from a separate
#: Activation layer, nor an L2 of 1e-5 from 1e-4.
EXPECTED_COUNTS = {
    "layers": 84,
    "conv1d": 21,
    "dense": 3,
    "batchnorm": 23,
    "activation": 23,
    "add": 10,
    "parameters": 102_497,
    "trainable_parameters": 100_897,
}


def load_declared_architecture() -> "Optional[dict]":
    """The committed declared signature payload, or None if not captured."""
    if not DECLARED_ARCHITECTURE_PATH.is_file():
        return None
    payload = json.loads(DECLARED_ARCHITECTURE_PATH.read_text())
    verify_signature(payload["signature"])
    return payload

#: Files whose content can change a result, hashed into the manifest.
ADAPTER_SOURCES = {
    "adapter_gohr_impl": "audit/implementation/adapters/gohr_impl_comparison.py",
    "reference_speck": "audit/implementation/reference/speck.py",
    "reference_train_nets": "audit/implementation/reference/train_nets.py",
    "reference_train_5_rounds": "audit/implementation/reference/train_5_rounds.py",
    "independent_speck": "audit/implementation/ii2_independent/speck_independent.py",
    "independent_dataset": "audit/implementation/ii2_independent/dataset_independent.py",
    "independent_model": "audit/implementation/ii2_independent/model_independent.py",
    "independent_train": "audit/implementation/ii2_independent/train_independent.py",
}

SHARED_COMPONENTS = {
    "dataset_generation": (
        "Both arms train on one block dataset produced by the REFERENCE generator "
        "(reference/speck.py make_train_data). Required by the paired block design: if each "
        "arm generated its own data the block would not be a shared unit. Evidential basis "
        "for treating the generators as equivalent: the II-2 M1 deterministic verification "
        "(evidence/ii2/m1_deterministic_verification.json), which established exact agreement "
        "on the published test vector, the key schedule over 1000 keys, round traces over "
        "10000 word states, the binary encoding and dataset construction. Data generation is "
        "therefore NOT a compared component of this experiment."),
    "sealed_test_set": (
        "One sealed test set, generated once by the reference generator and reused by both "
        "arms and every block, as the design requires."),
    "accuracy_measurement": (
        "Sealed-test accuracy is computed for both arms by one function in this adapter "
        "(strict prediction > 0.5, matching Keras binary_accuracy and the reference's "
        "reported metric), so the measurement instrument is identical by construction. The "
        "arms' own compiled metric names ('acc' vs 'accuracy') do not enter the endpoint."),
}


# =====================================================================
# read-only loading of the two implementations
# =====================================================================

@contextlib.contextmanager
def _sys_path(*paths: Path):
    """Temporarily extend sys.path. Nothing on disk is touched."""
    added = [str(p) for p in paths if str(p) not in sys.path]
    sys.path[:0] = added
    try:
        yield
    finally:
        for p in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(p)


def _load_module_by_path(name: str, path: Path, extra_paths: tuple[Path, ...] = ()):
    """Import a module from an explicit file path without modifying it."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:                       # pragma: no cover
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    with _sys_path(*extra_paths):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def load_reference():
    """reference/train_nets.py and reference/speck.py, unmodified."""
    speck = _load_module_by_path("speck", REFERENCE_DIR / "speck.py", (REFERENCE_DIR,))
    train_nets = _load_module_by_path(
        "reference_train_nets", REFERENCE_DIR / "train_nets.py", (REFERENCE_DIR,))
    return speck, train_nets


def load_independent():
    """The II-2 independent model builder and LR schedule, unmodified."""
    from audit.implementation.ii2_independent.model_independent import build_gohr_model
    from audit.implementation.ii2_independent.train_independent import (
        make_learning_rate_schedule,
    )
    return build_gohr_model, make_learning_rate_schedule


# =====================================================================
# training result
# =====================================================================

@dataclass
class ArmTrainResult:
    """
    `model` is the model the arm ACTUALLY finished training.

    On a resume the adapter discards the freshly built model it was
    handed and loads the resume checkpoint instead, so the runner must
    persist the model named here rather than the one it passed in -
    otherwise the terminal-epoch artifact would be an untrained model
    while every log said the arm completed 200 epochs.
    """
    n_epochs_completed: int
    final_val_acc: Optional[float]
    max_val_acc: Optional[float]
    checkpoint_hash: Optional[str]
    wall_clock_seconds: float
    failure_reason: Optional[str] = None
    model: Optional[Any] = None
    epoch_log: Optional[dict] = None
    recovery: Optional[dict] = None
    resumed_from_epoch: int = 0
    was_interrupted: bool = False


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# =====================================================================
# the adapter
# =====================================================================

class GohrImplementationComparisonAdapter:
    """
    Implements the generic ComparisonAdapter protocol for the Gohr
    reference-vs-independent comparison.

    `scale` exists ONLY to make a non-evidentiary smoke run possible on
    a laptop. A production plan must use scale=1.0, and the runner's
    production validation rejects anything else because the scaled
    protocol no longer equals GOHR_PROTOCOL.
    """

    def __init__(self, *, scale: float = 1.0, epochs: Optional[int] = None,
                 verbose: int = 0, boundary_hook=None):
        if not 0 < scale <= 1.0:
            raise ValueError("scale must lie in (0, 1].")
        self.scale = scale
        self.verbose = verbose
        #: Test-only: called at each named persistence boundary so a
        #: process-kill test can terminate at an exact point.
        self._boundary_hook = boundary_hook
        self.epochs = epochs if epochs is not None else GOHR_PROTOCOL["epochs"]
        self.n_train = max(int(GOHR_PROTOCOL["train_samples"] * scale), 1000)
        self.n_val = max(int(GOHR_PROTOCOL["validation_samples"] * scale), 500)
        self.n_test = max(int(GOHR_PROTOCOL["sealed_test_samples"] * scale), 500)
        self._speck = None
        self._train_nets = None

    @property
    def is_production_scale(self) -> bool:
        return (self.scale == 1.0 and self.epochs == GOHR_PROTOCOL["epochs"])

    # --- provenance ---------------------------------------------------
    def provenance_sources(self) -> dict[str, str]:
        return dict(ADAPTER_SOURCES)

    def shared_components(self) -> dict[str, str]:
        return dict(SHARED_COMPONENTS)

    def environment_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {"scale": self.scale, "epochs": self.epochs,
                                   "production_scale": self.is_production_scale}
        try:
            import tensorflow as tf
            details["tensorflow"] = tf.__version__
            details["keras"] = getattr(tf.keras, "__version__", "unknown")
            details["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        except Exception as exc:                                  # noqa: BLE001
            details["tensorflow"] = f"UNAVAILABLE: {type(exc).__name__}"
            details["gpus"] = []
        return details

    # --- shared data generation --------------------------------------
    def _reference(self):
        if self._speck is None:
            self._speck, self._train_nets = load_reference()
        return self._speck, self._train_nets

    def _make_data(self, n: int):
        speck, _ = self._reference()
        return speck.make_train_data(n, GOHR_PROTOCOL["rounds"],
                                     diff=GOHR_PROTOCOL["differential"])

    def generate_sealed_test_set(self, directory) -> tuple:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        X, Y = self._make_data(self.n_test)
        path = directory / "sealed_test.npz"
        np.savez_compressed(path, X=X, Y=Y)
        return str(path), _sha256_file(path), int(len(Y))

    def load_sealed_test_set(self, path: str, expected_hash: str):
        actual = _sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"sealed test set {path}: sha256 {actual} != sealed {expected_hash}. The "
                "confirmatory data have been substituted; refusing to evaluate.")
        d = np.load(path)
        return (d["X"], d["Y"])

    def generate_block_datasets(self, block_id: str, directory) -> dict:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        Xt, Yt = self._make_data(self.n_train)
        Xv, Yv = self._make_data(self.n_val)
        train_path, val_path = directory / "train.npz", directory / "validation.npz"
        np.savez_compressed(train_path, X=Xt, Y=Yt)
        np.savez_compressed(val_path, X=Xv, Y=Yv)
        return {
            "train": {"path": str(train_path), "hash": _sha256_file(train_path),
                      "n": int(len(Yt)), "content_hash": _sha256_arrays(Xt, Yt)},
            "validation": {"path": str(val_path), "hash": _sha256_file(val_path),
                           "n": int(len(Yv)), "content_hash": _sha256_arrays(Xv, Yv)},
            "generator": "reference/speck.py make_train_data (SHARED by both arms)",
        }

    def load_block_datasets(self, records: dict):
        for key in ("train", "validation"):
            rec = records[key]
            if _sha256_file(rec["path"]) != rec["hash"]:
                raise RuntimeError(
                    f"block {key} dataset {rec['path']} does not match its recorded hash; "
                    "refusing to train on substituted data.")
        tr, va = np.load(records["train"]["path"]), np.load(records["validation"]["path"])
        return {"X": tr["X"], "Y": tr["Y"], "Xv": va["X"], "Yv": va["Y"]}

    # --- per-arm model construction -----------------------------------
    def _seed_reference(self, seed: int) -> None:
        """The reference's OWN seeding function, not a reimplementation."""
        _, train_nets = self._reference()
        train_nets.set_seed(seed)

    def _seed_independent(self, seed: int) -> None:
        """
        Reproduces the seeding block of ii2_independent/train_independent.py
        train(). That function cannot be called directly here because it
        generates its own data, so its seeding is reproduced rather than
        invoked - recorded as a limitation in the preregistration.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        with contextlib.suppress(Exception):
            import keras
            keras.utils.set_random_seed(seed)

    def build_arm_model(self, arm_id: str, *, seed: int):
        if arm_id == ARM_REFERENCE:
            self._seed_reference(seed)
            _, train_nets = self._reference()
            model = train_nets.make_resnet(depth=GOHR_PROTOCOL["depth"],
                                           reg_param=GOHR_PROTOCOL["reg_param"])
            model.compile(optimizer=GOHR_PROTOCOL["optimizer"], loss=GOHR_PROTOCOL["loss"],
                          metrics=["acc"])
            return model
        if arm_id == ARM_INDEPENDENT:
            self._seed_independent(seed)
            build_gohr_model, _ = load_independent()
            model = build_gohr_model(
                depth=GOHR_PROTOCOL["depth"], filters=32, dense_1=64, dense_2=64,
                kernel_size=3, word_size=16, input_words=4,
                l2_strength=GOHR_PROTOCOL["reg_param"])
            model.compile(optimizer=GOHR_PROTOCOL["optimizer"], loss=GOHR_PROTOCOL["loss"],
                          metrics=["accuracy"])
            return model
        raise ValueError(f"unknown arm {arm_id!r}")

    def probe_arm_model(self, arm_id: str, model) -> dict[str, Any]:
        """
        FULL ORDERED architecture signature, computed identically for
        both arms: per layer, in order, the class, output shape,
        geometry (filters, units, kernel size, strides, padding,
        dilation), activation, bias, BatchNormalization settings,
        regularizer configuration, and the ordered weight shapes and
        dtypes - plus the model's input and output shapes and parameter
        counts.

        This replaces a count-by-class probe, which could not distinguish
        an inline Conv1D activation from a separate Activation layer, a
        'same' padding from a 'causal' one, or an L2 of 1e-5 from 1e-4 -
        and this project has already shipped the 1e-4 version of that
        mistake once, in the EV adapter.

        Invariant to weight VALUES, so the same digest must appear
        before training and on the artifact reloaded afterwards.
        """
        signature = architecture_signature(model)
        signature["coarse_counts_cross_check"] = self._coarse_counts(model)
        return signature

    @staticmethod
    def _coarse_counts(model) -> dict[str, int]:
        """Human-readable cross-check only; never the structural test."""
        counts = {k: 0 for k in ("conv1d", "dense", "batchnorm", "activation", "add")}
        alias = {"Conv1D": "conv1d", "Dense": "dense", "BatchNormalization": "batchnorm",
                 "Activation": "activation", "Add": "add"}
        for layer in model.layers:
            key = alias.get(type(layer).__name__)
            if key:
                counts[key] += 1
        counts["layers"] = len(model.layers)
        counts["parameters"] = int(model.count_params())
        return counts

    def expected_arm_structure(self, arm_id: str) -> Optional[str]:
        """
        The DECLARED architecture digest both arms must realize, or None
        if it has not been captured yet.

        Returning the digest rather than a structure dict keeps the
        generic runner's comparison a single equality while leaving the
        field-level diagnosis to `explain_structure_mismatch`, which has
        the committed signature to diff against.
        """
        if arm_id not in (ARM_REFERENCE, ARM_INDEPENDENT):
            raise ValueError(f"unknown arm {arm_id!r}")
        payload = load_declared_architecture()
        return None if payload is None else payload["signature"]["digest"]

    def explain_structure_mismatch(self, arm_id: str, observed: dict[str, Any]) -> list[str]:
        """Field-level differences from the committed declared signature."""
        payload = load_declared_architecture()
        if payload is None:
            return ["no declared architecture has been captured; run "
                    "--capture-architecture and re-freeze"]
        problems = diff_signatures(payload["signature"], observed,
                                   label_a="declared", label_b=arm_id)
        env = payload.get("captured_with", {})
        problems.append(
            f"declared signature was captured with tensorflow={env.get('tensorflow')} "
            f"keras={env.get('keras')}; this environment reports "
            f"tensorflow={self.environment_details().get('tensorflow')} "
            f"keras={self.environment_details().get('keras')}. If the differences above are "
            "only serialization details of a different Keras version, re-capture and "
            "RE-FREEZE; if any difference is a real configuration change, the architecture "
            "has changed and the comparison premise must be re-established.")
        return problems

    # --- per-arm training ---------------------------------------------
    def _lr_callback(self, arm_id: str):
        """Each arm's OWN schedule, taken from its own source."""
        if arm_id == ARM_REFERENCE:
            from keras.callbacks import LearningRateScheduler
            _, train_nets = self._reference()
            return LearningRateScheduler(train_nets.cyclic_lr(10, 0.002, 0.0001))
        if arm_id == ARM_INDEPENDENT:
            _, make_schedule = load_independent()
            return make_schedule()
        raise ValueError(f"unknown arm {arm_id!r}")

    def train_arm(self, arm_id: str, model, block_data, *, arm_dir, seed: int,
                  session_id: str = "s0",
                  interrupt_after_epoch: Optional[int] = None) -> ArmTrainResult:
        """
        Each arm's own fit recipe, on the SHARED block data, with
        epoch-level logging and transactional, epoch-granular recovery
        through the two-slot committed-checkpoint store.

        RECIPES ARE NOT INVENTED HERE. The reference arm's schedule is
        the reference module's own `cyclic_lr(10, 0.002, 0.0001)`; the
        independent arm's is its own `make_learning_rate_schedule()`.
        Only the data and the seed come from outside.

        RESUME. The COMMIT MANIFEST decides where to resume - not the
        epoch log, which is a derived record and can lag the manifest by
        one epoch or end in a torn line after a hard kill. If the
        manifest commits any epoch, the freshly built `model` argument is
        DISCARDED and the committed slot - a full model save carrying
        weights and optimizer state - is loaded instead. Its architecture
        signature is re-verified against the declared digest before a
        single further epoch runs, so a checkpoint from a different
        architecture cannot be trained into this arm.
        `fit(initial_epoch=...)` then continues the epoch numbering,
        which is also what makes the learning-rate schedule resume
        correctly: the schedule is a pure function of the absolute epoch
        index.

        If the committed slot fails verification, the previous slot is
        PROMOTED so the arm can continue, one epoch is lost and retrained,
        and the loss is recorded in the manifest's `repairs` and in an
        append-only corruption log. If BOTH slots fail, recovery refuses
        rather than restarting the arm while reporting a resume.

        WHAT IS RECOVERED is measured, not assumed: see
        arm_recovery.RECOVERY_EXACTNESS. On the stack that claim was
        re-measured on the stack that runs the experiment: weights, Adam's
        step counter and both moment estimates, the epoch index, the
        learning-rate schedule, the best-checkpoint record AND the
        per-epoch shuffle permutation all survive, so a resumed arm is a
        bit-exact continuation. An earlier version of this docstring
        claimed the shuffle stream could not be restored; measurement
        showed otherwise and the claim was corrected upwards. Only the
        WITHIN-EPOCH position is not recovered: a mid-epoch interruption
        redoes that epoch from its start. Every interruption is still
        appended to the arm's record and reaches the certificate, with
        the TensorFlow and Keras versions the run actually used.
        """
        import keras

        arm_dir = Path(arm_dir)
        arm_dir.mkdir(parents=True, exist_ok=True)
        log = EpochLog(arm_dir / "epochs.jsonl")
        record_path = arm_dir / "recovery.json"
        best_path = arm_dir / "best_val_loss.keras"
        declared = self.expected_arm_structure(arm_id)

        store = CommittedCheckpointStore(
            arm_dir=arm_dir, arm_id=arm_id, block_id=arm_dir.parent.name, seed=seed,
            epochs_total=self.epochs, architecture_digest=declared,
            boundary_hook=self._boundary_hook)

        record = ArmRecoveryRecord.load(record_path) or ArmRecoveryRecord(
            arm_id=arm_id, block_id=arm_dir.parent.name, seed=seed,
            epochs_total=self.epochs, architecture_digest=declared)

        # STARTED is persisted BEFORE training, so a process killed during
        # the first epoch leaves a record saying so rather than no trace.
        if record.status == "NOT_STARTED":
            record.status = "STARTED"
            record.started_at_utc = record.started_at_utc or _utc_now()
        record.sessions.append({"session_id": session_id, "began_at_utc": _utc_now()})
        record.save(record_path)

        resume_from, decision, notes = resume_point(store, log, record=record)
        record.save(record_path)

        if resume_from > 0 and decision.source != "fresh":
            model, _ = store.load_committed(
                lambda p: keras.models.load_model(p, compile=True))
            observed = architecture_signature(model)
            if declared and observed["digest"] != declared:
                raise RuntimeError(
                    f"{record.block_id}/{arm_id}: the committed checkpoint realizes "
                    f"architecture {observed['digest']}, but the plan declares {declared}.\n  "
                    + "\n  ".join(self.explain_structure_mismatch(arm_id, observed)))
            if model.optimizer is None:
                raise RuntimeError(
                    f"{record.block_id}/{arm_id}: the committed checkpoint carries no "
                    "optimizer, so Adam's step counter and moment estimates are lost. "
                    "Refusing to resume with a reinitialized optimizer.")
            record.note_interruption(
                reason=("resumed_degraded" if decision.degraded else "resumed"),
                detail=f"resumed at epoch {resume_from} from the {decision.source} slot "
                       f"{decision.slot}" + (f"; {'; '.join(decision.notes)}"
                                             if decision.notes else ""),
                at_epoch=resume_from, session_id=session_id)
            record.save(record_path)

        if resume_from >= self.epochs:
            record.status = ARM_TRAINED
            record.save(record_path)
            return self._finish(arm_id, model, log, record, best_path, store, 0.0,
                                resumed_from=resume_from)

        # Reseed HERE, on every session including a resume, so the shuffle
        # permutation is reproducible across a process boundary (see
        # arm_recovery.RECOVERY_EXACTNESS conditions). Relying on the
        # runner having just called build_arm_model would make the
        # guarantee depend on call order.
        reseed_for_session(
            self._seed_reference if arm_id == ARM_REFERENCE else self._seed_independent, seed)

        callbacks = [self._lr_callback(arm_id)] + build_callbacks(
            store=store, log=log, record=record, record_path=record_path,
            best_path=best_path, session_id=session_id,
            interrupt_after_epoch=interrupt_after_epoch)

        started = time.time()
        try:
            model.fit(
                block_data["X"], block_data["Y"],
                initial_epoch=resume_from, epochs=self.epochs,
                batch_size=GOHR_PROTOCOL["batch_size"],
                validation_data=(block_data["Xv"], block_data["Yv"]),
                shuffle=GOHR_PROTOCOL["shuffle"], callbacks=callbacks,
                verbose=self.verbose)
        except BaseException as exc:
            manifest = store.manifest()
            record.note_interruption(
                reason=type(exc).__name__, detail=str(exc),
                at_epoch=(manifest.committed_epoch + 1 if manifest else 0),
                session_id=session_id)
            record.save(record_path)
            raise
        elapsed = time.time() - started
        return self._finish(arm_id, model, log, record, best_path, store, elapsed,
                            resumed_from=resume_from)

    def _finish(self, arm_id, model, log, record, best_path, store, elapsed,
                *, resumed_from: int) -> ArmTrainResult:
        """Assemble the result from the LOG, not from an in-memory history.

        A resumed arm's `history` covers only the epochs of the final
        session, so reading `len(history.history['val_acc'])` would
        report 10 completed epochs for an arm that completed 200. The
        persisted epoch log is the only complete record, so every
        reported quantity is read from it.
        """
        manifest = store.manifest()
        committed = manifest.committed_epoch if manifest else None
        summary = log.summary(expected_total=self.epochs, committed_epoch=committed)
        records = log.read()
        val_accs = [r["val_accuracy"] for r in records
                    if isinstance(r.get("val_accuracy"), (int, float))]
        # The MANIFEST decides how many epochs completed. The log is a
        # derived metrics record and may legitimately lag it by one.
        n_completed = (committed + 1) if committed is not None else 0
        summary["manifest_committed_epoch"] = committed
        if not val_accs:
            return ArmTrainResult(
                n_epochs_completed=n_completed, final_val_acc=None,
                max_val_acc=None, checkpoint_hash=None, wall_clock_seconds=elapsed,
                failure_reason="no_validation_accuracy_recorded", model=model,
                epoch_log=summary, recovery=record.to_dict(),
                resumed_from_epoch=resumed_from, was_interrupted=record.was_interrupted)
        record.status = ARM_TRAINED
        return ArmTrainResult(
            n_epochs_completed=n_completed,
            final_val_acc=float(val_accs[-1]),
            max_val_acc=float(max(val_accs)),
            checkpoint_hash=(_durable_sha256(best_path)
                             if Path(best_path).is_file() else None),
            wall_clock_seconds=summary["total_training_seconds"],
            model=model, epoch_log=summary, recovery=record.to_dict(),
            resumed_from_epoch=resumed_from, was_interrupted=record.was_interrupted)

    # --- persistence and evaluation -----------------------------------
    def save_terminal_model(self, arm_id: str, model, path) -> dict:
        """
        Persist the TERMINAL-epoch model and verify the reload is
        weight-identical. The best-val-loss checkpoint written during
        training is provenance only and never the evaluated object.
        """
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        before = [np.asarray(w) for w in model.get_weights()]
        import keras
        reloaded = keras.models.load_model(path, compile=False)
        after = [np.asarray(w) for w in reloaded.get_weights()]
        identical = (len(before) == len(after)
                     and all(a.shape == b.shape and np.array_equal(a, b)
                             for a, b in zip(before, after)))
        reloaded_signature = architecture_signature(reloaded)
        declared = self.expected_arm_structure(arm_id)
        signature_ok = (declared is None or reloaded_signature["digest"] == declared)
        return {"path": str(path), "hash": _sha256_file(path),
                "reload_weight_identical": bool(identical),
                "weight_fingerprint": _sha256_arrays(*before),
                "architecture_digest": reloaded_signature["digest"],
                "architecture_matches_declared": bool(signature_ok),
                "architecture_mismatch": (
                    None if signature_ok
                    else self.explain_structure_mismatch(arm_id, reloaded_signature))}

    def load_terminal_model(self, arm_id: str, path: str, expected_hash: str):
        actual = _sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"terminal model {path}: sha256 {actual} != recorded {expected_hash}.")
        import keras
        return keras.models.load_model(path, compile=False)

    def evaluate_terminal(self, arm_id: str, model, sealed_test) -> float:
        """
        ONE accuracy function for both arms: strict prediction > 0.5,
        which is Keras binary_accuracy and the metric the reference
        reports. Because the same code measures both arms, the endpoint
        cannot differ because of how an arm compiled its metric.
        """
        X, Y = sealed_test
        preds = model.predict(X, batch_size=GOHR_PROTOCOL["batch_size"], verbose=0)
        return float(np.mean((np.asarray(preds).ravel() > 0.5).astype(np.int64)
                             == np.asarray(Y).ravel().astype(np.int64)))
