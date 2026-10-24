"""
Gohr adapter for D4: model-fit lifecycle with verifiable checkpoint selection.

Why a separate module
---------------------
``audit/dataset/adapters/gohr.py`` is pinned byte-for-byte (SHA-256
3c0b7ac5...) by the sealed Gohr D0 build attestation and the D3 certificate.
Changing it would break re-verification of that evidence, so the D4
checkpoint lifecycle lives here as a subclass.  Model construction
(``build_model``), dataset generation and the learning-rate schedule are
inherited unchanged.

Contract (see ``audit.dataset.d4_controlled_perturbation.D4Adapter``)
--------------------------------------------------------------------
``train`` fits on exactly the arrays it is given, selects the epoch with
minimum validation loss over the complete committed history, reloads that
checkpoint, and exposes ``selection``.  ``predict`` uses the reloaded model.
The adapter never receives test data: the test partition is bound into the
fit identity only as an opaque digest (``evaluation_digest``).

Checkpoint lifecycle and crash windows
--------------------------------------
Per epoch, in this order: (1) Keras writes ``checkpoint_epoch_NNN.keras``;
(2) an epoch record ``checkpoint_epoch_NNN.metrics.json`` is written
atomically, holding the checkpoint SHA-256, ``val_loss``, the fit identity
and the SHA-256 of the previous record; (3) ``state.json`` is replaced
atomically, committing the epoch and the new chain head.

Only epochs committed by (3) are ever trusted.  On start:

==  =============================================  =========================
#   Situation                                      Action
==  =============================================  =========================
1   committed epoch, record + checkpoint verify    used
2   incomplete write (``*.tmp``)                   quarantined, epoch rerun
3   checkpoint without record                      quarantined, epoch rerun
4   complete uncommitted orphan (record+ckpt)      quarantined, epoch rerun;
                                                   NEVER adopted, because its
                                                   val_loss is not anchored
5   checkpoint hash mismatch (committed or orphan) refused, nothing moved
6   altered record / broken chain / bad head       refused, nothing moved
7   wrong condition/replicate/seed/config/data     refused, nothing moved
8   truncated or malformed state.json or record    refused, nothing moved
-   record without checkpoint, or uncommitted      refused (cannot arise from
    artifacts beyond the next epoch                the write order above)
==  =============================================  =========================

Quarantine moves files into ``quarantine/attempt_NN/`` with a note; nothing
is deleted.  Every rerun is listed in ``state.json["retries"]`` and in the
selection record, so retries are visible in the evidence.  A rerun epoch is
retrained from the last committed checkpoint; TensorFlow does not guarantee
bitwise-identical retraining, which is why a retry is recorded rather than
hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model

import train_nets as tn
from audit.dataset.adapters.gohr import GohrAdapter
from audit.dataset.integrity import (
    IntegrityError,
    labelled_partition_digest,
    read_json,
    sha256_file,
    write_json_atomic,
)

#: Predeclared checkpoint-selection rule (Gohr's ``make_checkpoint``
#: criterion, applied to the complete committed history and reloaded).
MODEL_SELECTION_RULE = "argmin_val_loss_over_all_committed_epochs_reloaded_ties_earliest"

GENESIS_RECORD_SHA256 = "0" * 64
RECORD_SCHEMA = "d4-epoch-record-3"
STATE_SCHEMA = "d4-fit-state-3"
SELECTION_SCHEMA = "d4-selection-3"
IDENTITY_FIELDS = ("seed", "condition", "replicate", "config_hash",
                   "train_digest", "validation_digest", "evaluation_digest", "model")
RECORD_FIELDS = frozenset({"schema_version", "epoch", "checkpoint", "checkpoint_sha256", "val_loss",
                           "logs", "previous_record_sha256", *IDENTITY_FIELDS})


def epoch_checkpoint_path(directory: Path, epoch: int) -> Path:
    return Path(directory) / f"checkpoint_epoch_{int(epoch):03d}.keras"


def epoch_record_path(directory: Path, epoch: int) -> Path:
    return Path(directory) / f"checkpoint_epoch_{int(epoch):03d}.metrics.json"


def _epoch_of(path: Path) -> int | None:
    stem = path.name.split(".", 1)[0]
    if not stem.startswith("checkpoint_epoch_"):
        return None
    try:
        return int(stem.rsplit("_", 1)[1])
    except ValueError:
        return None


# ============================================================
# Record verification
# ============================================================


def verify_epoch_record(path: Path, epoch: int, identity: dict, previous_sha256: str) -> dict:
    """Return the epoch record if its schema, identity, chain link, metric
    and checkpoint digest all verify; raise IntegrityError otherwise."""
    record = read_json(path, f"epoch {epoch} record")
    if record.get("schema_version") != RECORD_SCHEMA or set(record) != RECORD_FIELDS:
        raise IntegrityError(f"Epoch record {path} does not have the {RECORD_SCHEMA} schema.")
    for key in IDENTITY_FIELDS:
        if record[key] != identity[key]:
            raise IntegrityError(
                f"Epoch record {path} has {key}={record[key]!r}, expected "
                f"{identity[key]!r}. Refusing to mix fits."
            )
    if record["epoch"] != epoch or record["checkpoint"] != epoch_checkpoint_path(".", epoch).name:
        raise IntegrityError(f"Epoch record {path} does not describe epoch {epoch}.")
    if record["previous_record_sha256"] != previous_sha256:
        raise IntegrityError(
            f"Epoch record {path} does not chain to the record of epoch {epoch - 1}; "
            "a record was altered or replaced."
        )
    val_loss = record["val_loss"]
    if not isinstance(val_loss, float) or not np.isfinite(val_loss):
        raise IntegrityError(f"Epoch record {path} has a non-finite val_loss.")
    if record["logs"].get("val_loss") != val_loss:
        raise IntegrityError(f"Epoch record {path}: val_loss disagrees with its own logs.")
    checkpoint = Path(path).parent / record["checkpoint"]
    if not checkpoint.is_file():
        raise IntegrityError(f"Checkpoint for epoch {epoch} is missing: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != record["checkpoint_sha256"]:
        raise IntegrityError(
            f"Checkpoint {checkpoint} hashes to {actual}, but its epoch record "
            f"says {record['checkpoint_sha256']}. Refusing to use it."
        )
    return record


def verify_epoch_chain(directory: Path, *, epochs: int, identity: dict) -> tuple[list, str]:
    """Verify records and checkpoints 1..epochs; return them and the chain head."""
    records, head = [], GENESIS_RECORD_SHA256
    for epoch in range(1, int(epochs) + 1):
        path = epoch_record_path(directory, epoch)
        records.append(verify_epoch_record(path, epoch, identity, head))
        head = sha256_file(path)
    return records, head


def select_checkpoint(directory: Path, *, total_epochs: int, identity: dict, chain_head: str) -> dict:
    """Apply MODEL_SELECTION_RULE to the complete, verified, committed history."""
    records, head = verify_epoch_chain(directory, epochs=total_epochs, identity=identity)
    if head != chain_head:
        raise IntegrityError(
            "The epoch-record chain does not end at the head committed in state.json; "
            "a record was altered, added or replaced."
        )
    best = min(records, key=lambda r: (r["val_loss"], r["epoch"]))
    return {
        "schema_version": SELECTION_SCHEMA,
        "rule": MODEL_SELECTION_RULE,
        "selection_partition": "validation",
        "selected_epoch": best["epoch"],
        "selected_val_loss": best["val_loss"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "epoch_checkpoint_sha256": [r["checkpoint_sha256"] for r in records],
        "record_chain_head": head,
        "total_epochs": int(total_epochs),
        **{key: identity[key] for key in IDENTITY_FIELDS},
    }


# ============================================================
# State
# ============================================================


def write_state(state_path: Path, *, identity: dict, total_epochs: int, completed: int,
                chain_head: str, retries: list) -> None:
    write_json_atomic(state_path, {
        "schema_version": STATE_SCHEMA,
        "identity": identity,
        "total_epochs": int(total_epochs),
        "completed_epochs": int(completed),
        "record_chain_head": chain_head,
        "retries": list(retries),
        "status": "complete" if completed >= total_epochs else "in_progress",
    })


def load_committed_state(state_path: Path, directory: Path, *, identity: dict, total_epochs: int) -> dict:
    """Read and fully verify state.json (cases 1, 5, 6, 7, 8)."""
    state = read_json(state_path, "training state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise IntegrityError(f"Training state {state_path} does not have the {STATE_SCHEMA} schema.")
    if state.get("identity") != identity or state.get("total_epochs") != int(total_epochs):
        raise IntegrityError(
            f"Training state {state_path} belongs to a different fit (condition, replicate, "
            "seed, configuration, data or model). Refusing to resume."
        )
    completed = state.get("completed_epochs")
    if not isinstance(completed, int) or not 0 <= completed <= total_epochs:
        raise IntegrityError(f"Training state {state_path} has an invalid completed_epochs.")
    if not isinstance(state.get("retries"), list):
        raise IntegrityError(f"Training state {state_path} has no retries list.")
    _, head = verify_epoch_chain(directory, epochs=completed, identity=identity)
    if head != state.get("record_chain_head"):
        raise IntegrityError("Committed epoch records do not end at the chain head in state.json.")
    return state


def classify_uncommitted(directory: Path, *, committed: int, chain_head: str, identity: dict) -> dict:
    """Classify artifacts past the committed epoch, without moving anything.

    Returns {"files": [...], "reason": str} for artifacts that are safe to
    quarantine (cases 2, 3, 4), or raises IntegrityError for suspicious or
    ambiguous ones (cases 5, 6, 7, 8 and impossible orderings).
    """
    directory = Path(directory)
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and (_epoch_of(p) or 0) > committed
    )
    if not files:
        return {"files": [], "reason": None}
    epochs = {_epoch_of(p) for p in files}
    nxt = committed + 1
    if epochs != {nxt}:
        raise IntegrityError(
            f"Uncommitted artifacts exist for epochs {sorted(epochs)}; only epoch {nxt} can "
            "be uncommitted after a crash. Refusing; nothing was moved."
        )
    checkpoint, record = epoch_checkpoint_path(directory, nxt), epoch_record_path(directory, nxt)
    finals = {p for p in files if not p.name.endswith(".tmp")}
    if finals - {checkpoint, record}:
        raise IntegrityError(f"Unexpected uncommitted files {sorted(map(str, finals))}; refusing.")
    if record in finals and checkpoint not in finals:
        raise IntegrityError(
            f"Epoch {nxt} has a record but no checkpoint, which the write order cannot "
            "produce. Refusing; nothing was moved."
        )
    if record in finals:
        # Case 4 if it verifies; cases 5-8 raise.  Even a verifying orphan is
        # not adopted: nothing anchors its val_loss.
        verify_epoch_record(record, nxt, identity, chain_head)
        reason = "complete_uncommitted_orphan"
    elif checkpoint in finals:
        reason = "checkpoint_without_record"
    else:
        reason = "incomplete_write"
    return {"files": [str(p) for p in files], "reason": reason, "epoch": nxt}


def quarantine(directory: Path, classification: dict) -> dict:
    """Move classified uncommitted files into quarantine/attempt_NN; delete nothing."""
    base = Path(directory) / "quarantine"
    attempt = 1
    while (base / f"attempt_{attempt:02d}").exists():
        attempt += 1
    target = base / f"attempt_{attempt:02d}"
    target.mkdir(parents=True)
    moved = []
    for name in classification["files"]:
        source = Path(name)
        source.rename(target / source.name)
        moved.append(source.name)
    entry = {"epoch": classification["epoch"], "reason": classification["reason"],
             "quarantine": f"quarantine/{target.name}", "files": moved}
    write_json_atomic(target / "note.json", entry)
    return entry


# ============================================================
# Callback
# ============================================================


class CommittedEpochCallback(Callback):
    """After Keras saves an epoch checkpoint: write its record, then commit state."""

    def __init__(self, *, directory: Path, state_path: Path, identity: dict, total_epochs: int,
                 chain_head: str, retries: list) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.state_path = Path(state_path)
        self.identity = identity
        self.total_epochs = int(total_epochs)
        self.chain_head = chain_head
        self.retries = retries

    def on_epoch_end(self, epoch, logs=None):
        completed = int(epoch + 1)
        checkpoint = epoch_checkpoint_path(self.directory, completed)
        if not checkpoint.is_file():
            raise IntegrityError(f"Epoch {completed} checkpoint was not written: {checkpoint}")
        epoch_logs = {k: float(v) for k, v in (logs or {}).items() if np.isscalar(v)}
        if not np.isfinite(epoch_logs.get("val_loss", float("nan"))):
            raise IntegrityError(f"Epoch {completed}: val_loss is missing or non-finite.")
        record_path = epoch_record_path(self.directory, completed)
        if record_path.exists():
            raise IntegrityError(f"Refusing to overwrite existing epoch record {record_path}.")
        write_json_atomic(record_path, {
            "schema_version": RECORD_SCHEMA,
            "epoch": completed,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "val_loss": epoch_logs["val_loss"],
            "logs": epoch_logs,
            "previous_record_sha256": self.chain_head,
            **self.identity,
        })
        self.chain_head = sha256_file(record_path)
        write_state(self.state_path, identity=self.identity, total_epochs=self.total_epochs,
                    completed=completed, chain_head=self.chain_head, retries=self.retries)


# ============================================================
# Adapter
# ============================================================


class GohrD4Adapter(GohrAdapter):
    """Gohr distinguisher under the D4 adapter contract."""

    def __init__(self, *, validation_x, validation_y, evaluation_digest: str, **kwargs) -> None:
        if kwargs.get("test_x") is not None or kwargs.get("test_y") is not None:
            raise ValueError("The D4 adapter must not receive test data; pass evaluation_digest.")
        kwargs.pop("test_x", None)
        kwargs.pop("test_y", None)
        super().__init__(validation_x=validation_x, validation_y=validation_y, **kwargs)
        self.evaluation_digest = str(evaluation_digest)
        self.selection = None

    def fit_identity(self, train_x, train_y, *, condition, replicate, run_config_hash) -> dict:
        return {
            "seed": int(self.seed),
            "condition": str(condition),
            "replicate": int(replicate),
            "config_hash": str(run_config_hash),
            "train_digest": labelled_partition_digest(train_x, train_y),
            "validation_digest": labelled_partition_digest(self.validation_x, self.validation_y),
            "evaluation_digest": self.evaluation_digest,
            "model": {"architecture": "gohr_resnet", "num_rounds": int(self.num_rounds),
                      "depth": int(self.depth), "epochs": int(self.epochs),
                      "batch_size": int(self.batch_size)},
        }

    def train(self, train_x, train_y, *, checkpoint_path, state_path, replicate, condition,
              run_config_hash):
        if not run_config_hash:
            raise ValueError("run_config_hash is required.")
        directory, state_path = Path(checkpoint_path), Path(state_path)
        directory.mkdir(parents=True, exist_ok=True)
        tn.set_seed(self.seed)
        identity = self.fit_identity(train_x, train_y, condition=condition, replicate=replicate,
                                     run_config_hash=run_config_hash)

        completed, head, retries = 0, GENESIS_RECORD_SHA256, []
        if state_path.exists():
            state = load_committed_state(state_path, directory, identity=identity,
                                         total_epochs=self.epochs)
            completed, head, retries = (state["completed_epochs"], state["record_chain_head"],
                                        list(state["retries"]))
        classification = classify_uncommitted(directory, committed=completed, chain_head=head,
                                              identity=identity)
        if classification["files"]:
            retries.append(quarantine(directory, classification))
            write_state(state_path, identity=identity, total_epochs=self.epochs,
                        completed=completed, chain_head=head, retries=retries)
            print(f"Quarantined uncommitted epoch {classification['epoch']} "
                  f"({classification['reason']}); it will be retrained.")

        if completed < self.epochs:
            if completed:
                print(f"Resuming {condition} replicate {replicate} from committed epoch "
                      f"{completed}/{self.epochs}.")
                self.model = load_model(epoch_checkpoint_path(directory, completed), compile=True)
            else:
                self.model = self.build_model()
            callbacks = [
                LearningRateScheduler(tn.cyclic_lr(10, 0.002, 0.0001)),
                ModelCheckpoint(filepath=str(directory / "checkpoint_epoch_{epoch:03d}.keras"),
                                save_weights_only=False, save_best_only=False,
                                save_freq="epoch", verbose=0),
                CommittedEpochCallback(directory=directory, state_path=state_path,
                                       identity=identity, total_epochs=self.epochs,
                                       chain_head=head, retries=retries),
            ]
            self.history = self.model.fit(
                train_x, train_y,
                validation_data=(self.validation_x, self.validation_y),
                epochs=self.epochs, initial_epoch=completed, batch_size=self.batch_size,
                shuffle=False, callbacks=callbacks, verbose=1,
            )
            state = load_committed_state(state_path, directory, identity=identity,
                                         total_epochs=self.epochs)
            head = state["record_chain_head"]
            retries = state["retries"]

        selection = select_checkpoint(directory, total_epochs=self.epochs, identity=identity,
                                      chain_head=head)
        selection["retries"] = retries
        selection_path = directory / "selected_checkpoint.json"
        if selection_path.exists():
            if read_json(selection_path, "selection record") != selection:
                raise IntegrityError(f"Existing selection record {selection_path} disagrees with "
                                     "the recomputed selection. Refusing to overwrite it.")
        else:
            write_json_atomic(selection_path, selection)
        self.model = load_model(directory / selection["checkpoint"], compile=True)
        self.selection = selection
        return self.model

    def predict(self, features):
        if self.selection is None:
            raise RuntimeError("predict() before train(): no selected checkpoint is loaded.")
        return super().predict(features)
