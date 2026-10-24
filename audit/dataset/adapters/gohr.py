"""
Gohr dataset/model adapter.

Dataset generation is delegated directly to speck.make_train_data().
The adapter also provides resumable Keras training for controlled
audit experiments.

Important:
    Dataset generation uses os.urandom() internally. Therefore a
    generated dataset must be persisted by the experiment driver if
    an experiment is intended to resume across processes/sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from keras.callbacks import (
    Callback,
    LearningRateScheduler,
    ModelCheckpoint,
)
from keras.models import load_model
import json
import os
import numpy as np
import speck as sp
import train_nets as tn
#: Predeclared checkpoint-selection rule for D4: the epoch with minimum
#: validation loss (Gohr's original ``make_checkpoint`` criterion), chosen
#: over the complete epoch history and reloaded from its immutable file.
MODEL_SELECTION_RULE = "argmin_val_loss_over_all_epochs_reloaded"

#: ``previous_record_sha256`` of the first epoch record.
GENESIS_RECORD_SHA256 = "0" * 64


def epoch_metrics_path(checkpoint_directory: Path, epoch: int) -> Path:
    return Path(checkpoint_directory) / f"checkpoint_epoch_{int(epoch):03d}.metrics.json"


def epoch_checkpoint_path(checkpoint_directory: Path, epoch: int) -> Path:
    return Path(checkpoint_directory) / f"checkpoint_epoch_{int(epoch):03d}.keras"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_digest(partitions: dict, chunk_rows: int = 1 << 18) -> str:
    """SHA-256 over the names, dtypes, shapes and bytes of the partitions.

    Hashed in row chunks so memory-mapped arrays are never copied whole.
    """
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(partitions):
        array = np.asarray(partitions[name])
        digest.update(f"{name}|{array.dtype.str}|{array.shape}\n".encode())
        for start in range(0, len(array), chunk_rows):
            digest.update(np.ascontiguousarray(array[start:start + chunk_rows]).tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _verify_epoch_record(path: Path, epoch: int, identity: dict, previous_sha256: str) -> dict:
    """Return the record for ``epoch`` if it is intact and links to its predecessor."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Epoch record is malformed: {path}") from exc
    for key, value in identity.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"Epoch record {path} has {key}={record.get(key)!r}, "
                f"expected {value!r}. Refusing to mix fits."
            )
    if record.get("epoch") != epoch or record.get("checkpoint") != epoch_checkpoint_path(".", epoch).name:
        raise RuntimeError(f"Epoch record {path} does not describe epoch {epoch}.")
    if record.get("previous_record_sha256") != previous_sha256:
        raise RuntimeError(
            f"Epoch record {path} does not chain to the record of epoch "
            f"{epoch - 1}; an earlier record was altered or replaced."
        )
    val_loss = record.get("val_loss")
    if not isinstance(val_loss, float) or not np.isfinite(val_loss):
        raise RuntimeError(f"Epoch record {path} has a non-finite val_loss.")
    checkpoint = Path(path).parent / record["checkpoint"]
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint for epoch {epoch} is missing: {checkpoint}")
    actual = _sha256_file(checkpoint)
    if actual != record.get("checkpoint_sha256"):
        raise RuntimeError(
            f"Checkpoint {checkpoint} hashes to {actual}, but its epoch "
            f"record says {record.get('checkpoint_sha256')}. Refusing to use it."
        )
    return record


def verify_epoch_chain(checkpoint_directory: Path, *, epochs: int, identity: dict) -> tuple[list, str]:
    """Verify epoch records 1..epochs and return them with the chain head."""
    records = []
    head = GENESIS_RECORD_SHA256
    for epoch in range(1, int(epochs) + 1):
        path = epoch_metrics_path(checkpoint_directory, epoch)
        if not path.exists():
            raise RuntimeError(f"Epoch record missing: {path}. Refusing to select a checkpoint.")
        records.append(_verify_epoch_record(path, epoch, identity, head))
        head = _sha256_file(path)
    return records, head


def select_checkpoint(
    checkpoint_directory: Path,
    *,
    total_epochs: int,
    seed: int,
    condition: str,
    replicate: int,
    config_hash: str,
    data_digest: str,
    chain_head: str,
) -> dict:
    """Select the minimum-val_loss epoch from the immutable epoch records.

    Every epoch 1..total_epochs must have a record whose identity fields
    (including the partition digest) match this fit and whose checkpoint
    still hashes to the recorded digest.  The records form a hash chain
    whose head is committed in state.json, so altering any val_loss (or
    any other record field) is refused.  Ties resolve to the earliest epoch.
    """
    identity = {
        "seed": int(seed),
        "condition": str(condition),
        "replicate": int(replicate),
        "config_hash": str(config_hash),
        "data_digest": str(data_digest),
    }
    records, head = verify_epoch_chain(checkpoint_directory, epochs=total_epochs, identity=identity)
    if head != chain_head:
        raise RuntimeError(
            "The epoch-record chain does not end at the head committed in "
            "state.json; a record was altered, added or replaced."
        )
    best = min(records, key=lambda r: (r["val_loss"], r["epoch"]))
    return {
        "schema_version": "2.0",
        "rule": MODEL_SELECTION_RULE,
        "selected_epoch": best["epoch"],
        "selected_val_loss": best["val_loss"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "record_chain_head": head,
        "total_epochs": int(total_epochs),
        **identity,
    }


def record_selection(path: Path, selection: dict) -> None:
    """Write the selection record once; an existing record must agree."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError(
                f"Existing selection record {path} disagrees with the "
                "recomputed selection. Refusing to overwrite it."
            )
        return
    _write_json_atomic(path, selection)


def write_training_state(
    state_path: Path,
    *,
    identity: dict,
    total_epochs: int,
    completed: int,
    chain_head: str,
    logs: dict | None = None,
) -> None:
    _write_json_atomic(
        state_path,
        {
            "schema_version": "3.0",
            **identity,
            "total_epochs": int(total_epochs),
            "completed_epochs": int(completed),
            "latest_checkpoint": epoch_checkpoint_path(".", completed).name,
            "record_chain_head": chain_head,
            "status": "complete" if completed >= total_epochs else "in_progress",
            "last_epoch_logs": dict(logs or {}),
        },
    )


def reconcile_uncommitted_epochs(
    checkpoint_directory: Path,
    state_path: Path,
    *,
    identity: dict,
    total_epochs: int,
    committed: int,
    chain_head: str,
) -> tuple[int, str]:
    """Recover or set aside work written after the last committed state.

    A crash can land after an epoch's checkpoint and record are written
    but before state.json advances.  If that record is intact and chains
    to the committed head, the epoch is complete and is adopted (state is
    advanced).  Anything else past the committed epoch is renamed with an
    ``.uncommitted-N`` suffix, never deleted, so rerunning the epoch cannot
    silently overwrite it.
    """
    checkpoint_directory = Path(checkpoint_directory)
    epoch = committed + 1
    record_path = epoch_metrics_path(checkpoint_directory, epoch)
    if epoch <= total_epochs and record_path.exists():
        try:
            record = _verify_epoch_record(record_path, epoch, identity, chain_head)
        except RuntimeError as exc:
            print(f"Epoch {epoch} record is not adoptable ({exc}); setting it aside.")
        else:
            committed, chain_head = epoch, _sha256_file(record_path)
            write_training_state(
                state_path, identity=identity, total_epochs=total_epochs,
                completed=committed, chain_head=chain_head, logs=record.get("logs"),
            )
            print(f"Adopted epoch {epoch}, completed before an interrupted state update.")
            return reconcile_uncommitted_epochs(
                checkpoint_directory, state_path, identity=identity,
                total_epochs=total_epochs, committed=committed, chain_head=chain_head,
            )
    for path in sorted(checkpoint_directory.glob("checkpoint_epoch_*")):
        stem = path.name.split(".")[0]
        try:
            number = int(stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if number > committed and ".uncommitted-" not in path.name:
            suffix = 1
            while path.with_name(f"{path.name}.uncommitted-{suffix}").exists():
                suffix += 1
            path.rename(path.with_name(f"{path.name}.uncommitted-{suffix}"))
            print(f"Set aside uncommitted file {path.name}.")
    return committed, chain_head


class EpochStateCallback(Callback):
    """Record each completed epoch, then atomically advance the state.

    Registered after ModelCheckpoint, so an epoch record is written only
    once its checkpoint exists, and state.json only after the record.  A
    crash between the two is repaired on the next start by
    ``reconcile_uncommitted_epochs``.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        seed: int,
        condition: str,
        replicate: int,
        total_epochs: int,
        config_hash: str,
        checkpoint_directory: Path,
        data_digest: str,
        chain_head: str,
    ) -> None:
        super().__init__()
        self.state_path = Path(state_path)
        self.total_epochs = int(total_epochs)
        self.checkpoint_directory = Path(checkpoint_directory)
        self.identity = {
            "seed": int(seed),
            "condition": str(condition),
            "replicate": int(replicate),
            "config_hash": str(config_hash),
            "data_digest": str(data_digest),
        }
        self.chain_head = str(chain_head)

    def on_epoch_end(self, epoch, logs=None):
        completed = int(epoch + 1)
        checkpoint = epoch_checkpoint_path(self.checkpoint_directory, completed)
        if not checkpoint.exists():
            raise RuntimeError(
                "Epoch state cannot advance because the expected epoch "
                f"checkpoint does not exist: {checkpoint}"
            )
        epoch_logs = {
            key: float(value)
            for key, value in (logs or {}).items()
            if np.isscalar(value)
        }
        if "val_loss" not in epoch_logs:
            raise RuntimeError(
                "val_loss is missing from the epoch logs; model selection "
                "cannot be recorded."
            )
        metrics_path = epoch_metrics_path(self.checkpoint_directory, completed)
        if metrics_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing epoch record {metrics_path}."
            )
        _write_json_atomic(
            metrics_path,
            {
                "schema_version": "2.0",
                "epoch": completed,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": _sha256_file(checkpoint),
                "val_loss": epoch_logs["val_loss"],
                "logs": epoch_logs,
                "previous_record_sha256": self.chain_head,
                **self.identity,
            },
        )
        self.chain_head = _sha256_file(metrics_path)
        write_training_state(
            self.state_path, identity=self.identity, total_epochs=self.total_epochs,
            completed=completed, chain_head=self.chain_head, logs=epoch_logs,
        )


class GohrAdapter:
    """Adapter for Gohr's neural distinguisher and dataset generator."""

    DATASET_ID = "gohr-speck"
    DATASET_VERSION = "original-make-train-data"
    GENERATOR_NAME = "speck.make_train_data"

    def __init__(
        self,
        *,
        validation_x=None,
        validation_y=None,
        test_x=None,
        test_y=None,
        num_rounds: int = 5,
        depth: int = 10,
        epochs: int = 200,
        batch_size: int = 5000,
        seed: int = 0,
        model_directory: str = "./freshly_trained_nets",
    ) -> None:

        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1.")

        if depth < 1:
            raise ValueError("depth must be >= 1.")

        if epochs < 1:
            raise ValueError("epochs must be >= 1.")

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.validation_x = validation_x
        self.validation_y = validation_y

        self.test_x = test_x
        self.test_y = test_y

        self.num_rounds = int(num_rounds)
        self.depth = int(depth)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = None
        self.history = None
        self.selection = None

    # ==========================================================
    # Dataset generation
    # ==========================================================

    @property
    def feature_bits(self) -> int:
        return 64

    def generate_partition(
        self,
        samples: int,
        *,
        num_rounds: int | None = None,
    ) -> Tuple[Any, Any]:

        if samples < 1:
            raise ValueError("samples must be >= 1.")

        rounds = (
            self.num_rounds
            if num_rounds is None
            else int(num_rounds)
        )

        if rounds < 1:
            raise ValueError("num_rounds must be >= 1.")

        return sp.make_train_data(
            int(samples),
            rounds,
        )

    def dataset_provenance(
        self,
        *,
        train_samples: int,
        validation_samples: int,
        test_samples: int,
    ) -> dict[str, Any]:

        return {
            "dataset_id": self.DATASET_ID,
            "dataset_version": self.DATASET_VERSION,
            "generation_procedure": (
                f"{self.GENERATOR_NAME}(n, nr)"
            ),
            "generation_parameters": {
                "generator": self.GENERATOR_NAME,
                "num_rounds": self.num_rounds,
                "train_samples": int(train_samples),
                "validation_samples": int(
                    validation_samples
                ),
                "test_samples": int(test_samples),
                "input_difference": {
                    "left_word": "0x0040",
                    "right_word": "0x0000",
                },
                "randomness_source": "os.urandom",
                "deterministic_seed": None,
            },
            "generation_random_seed": None,
        }

    # ==========================================================
    # Model
    # ==========================================================

    def build_model(self):

        model = tn.make_resnet(
            depth=self.depth,
            reg_param=1e-5,
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["accuracy"],
        )

        return model

    def training_callbacks(self):
        """
        Return Gohr's original training callbacks.

        Checkpointing is deliberately NOT included here.
        The generic D5 audit owns checkpointing so that the
        checkpoint/resume mechanism remains dataset-agnostic.
        """

        scheduler = LearningRateScheduler(
            tn.cyclic_lr(
                10,
                0.002,
                0.0001,
            )
        )

        return [
            scheduler,
        ]
    
    # ==========================================================
    # Training
    # ==========================================================

    def train(
        self,
        train_x,
        train_y,
        *,
        checkpoint_path: Path | None = None,
        state_path: Path | None = None,
        replicate: int = 0,
        condition: str = "unknown",
        run_config_hash: str | None = None,
    ):
        """Train with crash-safe epoch checkpoints and strict resume validation.

        ``checkpoint_path`` is interpreted as the persistent condition
        directory supplied by the D4 driver.  One immutable Keras model file
        is written after every completed epoch.  ``state.json`` is advanced
        only after that epoch checkpoint exists.

        On restart, the state file and referenced checkpoint must agree with
        the current immutable configuration.  A mismatch is a hard error;
        the code never silently mixes runs.
        """
        tn.set_seed(self.seed)
        self.model_directory.mkdir(parents=True, exist_ok=True)

        if checkpoint_path is None:
            checkpoint_directory = self.model_directory / "checkpoints"
        else:
            checkpoint_directory = Path(checkpoint_path)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        if state_path is None:
            state_path = checkpoint_directory / "state.json"
        else:
            state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        if run_config_hash is None:
            raise ValueError("run_config_hash is required for resumable D4 training.")

        # Bind the fit to the exact partition contents, not only the config
        # hash: resuming against different data under the same config must
        # be refused.
        partitions = {"train_x": train_x, "train_y": train_y,
                      "validation_x": self.validation_x, "validation_y": self.validation_y}
        if self.test_x is not None:
            partitions.update(test_x=self.test_x, test_y=self.test_y)
        data_digest = partition_digest(partitions)
        identity = {
            "seed": int(self.seed),
            "condition": str(condition),
            "replicate": int(replicate),
            "config_hash": str(run_config_hash),
            "data_digest": data_digest,
        }

        completed_epochs = 0
        chain_head = GENESIS_RECORD_SHA256

        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)

            for key, expected_value in {**identity, "total_epochs": int(self.epochs)}.items():
                actual = state.get(key)
                if actual != expected_value:
                    raise RuntimeError(
                        f"Refusing to resume {condition} replicate {replicate}: "
                        f"state field {key!r} is {actual!r}, expected {expected_value!r}. "
                        "The experiment configuration and data must remain "
                        "immutable across sessions."
                    )

            completed_epochs = int(state.get("completed_epochs", 0))
            if completed_epochs < 0 or completed_epochs > self.epochs:
                raise RuntimeError("Invalid completed_epochs in D4 state.json.")
            if state.get("latest_checkpoint") != epoch_checkpoint_path(".", completed_epochs).name:
                raise RuntimeError(
                    "D4 state/checkpoint mismatch: latest_checkpoint does not "
                    "match completed_epochs."
                )
            # Every committed checkpoint and record is re-verified before reuse.
            _, chain_head = verify_epoch_chain(
                checkpoint_directory, epochs=completed_epochs, identity=identity
            )
            if chain_head != state.get("record_chain_head"):
                raise RuntimeError(
                    "Committed epoch records do not match the chain head in "
                    "state.json. Refusing to resume."
                )

        completed_epochs, chain_head = reconcile_uncommitted_epochs(
            checkpoint_directory, state_path, identity=identity,
            total_epochs=self.epochs, committed=completed_epochs, chain_head=chain_head,
        )

        if completed_epochs > 0:
            print(
                f"Resuming {condition} replicate {replicate} from "
                f"epoch {completed_epochs}/{self.epochs}..."
            )
            if completed_epochs >= self.epochs:
                return self._load_selected(checkpoint_directory, identity, chain_head)
            self.model = load_model(
                epoch_checkpoint_path(checkpoint_directory, completed_epochs), compile=True
            )
        else:
            self.model = self.build_model()

        # Keep the original Gohr learning-rate schedule.  The only training
        # change here is crash-safe checkpointing/resumption.
        scheduler = LearningRateScheduler(
            tn.cyclic_lr(10, 0.002, 0.0001)
        )

        # Model selection is done after training from the immutable epoch
        # records (see select_checkpoint), not by a save_best_only file: such
        # a file restarts its "best" on resume and, keyed only by seed, was
        # shared by the baseline and perturbed fits of a replicate.
        callbacks = [scheduler]

        # Immutable per-epoch recovery checkpoint.  The epoch number is in
        # the filename so a successful epoch is never overwritten.
        epoch_pattern = str(
            checkpoint_directory / "checkpoint_epoch_{epoch:03d}.keras"
        )
        rolling_checkpoint = ModelCheckpoint(
            filepath=epoch_pattern,
            save_weights_only=False,
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        )
        callbacks.append(rolling_checkpoint)

        callbacks.append(
            EpochStateCallback(
                state_path=state_path,
                seed=self.seed,
                condition=condition,
                replicate=replicate,
                total_epochs=self.epochs,
                config_hash=run_config_hash,
                checkpoint_directory=checkpoint_directory,
                data_digest=data_digest,
                chain_head=chain_head,
            )
        )

        self.history = self.model.fit(
            train_x,
            train_y,
            validation_data=(self.validation_x, self.validation_y),
            epochs=self.epochs,
            initial_epoch=completed_epochs,
            batch_size=self.batch_size,
            # Fixed order makes epoch-boundary resume independent of a
            # process-local shuffle RNG.  The training protocol therefore
            # remains identical across sessions once this is frozen.
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )

        state = json.loads(state_path.read_text(encoding="utf-8"))
        return self._load_selected(
            checkpoint_directory, identity, state.get("record_chain_head")
        )

    def _load_selected(self, checkpoint_directory, identity, chain_head):
        """Select by validation loss, record the choice, reload those weights.

        Downstream prediction uses ``self.model``, so after this call it is
        the selected checkpoint rather than the last epoch held in memory.
        """
        selection = select_checkpoint(
            checkpoint_directory,
            total_epochs=self.epochs,
            chain_head=chain_head,
            **identity,
        )
        record_selection(Path(checkpoint_directory) / "selected_checkpoint.json", selection)
        self.model = load_model(
            Path(checkpoint_directory) / selection["checkpoint"], compile=True
        )
        self.selection = selection
        return self.model

    # ==========================================================
    # Evaluation
    # ==========================================================

    def evaluate(
        self,
        model,
    ):

        if self.test_x is None or self.test_y is None:
            raise ValueError(
                "test_x and test_y are required for evaluation."
            )

        _, score = model.evaluate(
            self.test_x,
            self.test_y,
            verbose=0,
        )

        return float(score)

    # ==========================================================
    # Prediction
    # ==========================================================

    def predict(
        self,
        features,
    ):

        if self.model is None:
            raise RuntimeError(
                "No model is loaded or trained."
            )

        return self.model.predict(
            features,
            verbose=0,
        )

    # ==========================================================
    # Save / Load
    # ==========================================================

    def save(
        self,
        path,
    ):

        if self.model is None:
            raise RuntimeError(
                "No model is available to save."
            )

        self.model.save(path)

    def load(
        self,
        path,
    ):

        self.model = load_model(path)

        return self.model