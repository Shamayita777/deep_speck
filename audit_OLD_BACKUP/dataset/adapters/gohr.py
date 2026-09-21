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
class EpochStateCallback(Callback):
    """Atomically persist the completed epoch after its checkpoint exists.

    The callback is deliberately registered after ModelCheckpoint.  Thus a
    state file can only advance after the corresponding immutable epoch
    checkpoint has been written successfully.  If a process dies between
    those two operations, the state safely points to the previous epoch and
    that epoch is rerun rather than silently skipped.
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
    ) -> None:
        super().__init__()
        self.state_path = Path(state_path)
        self.seed = int(seed)
        self.condition = str(condition)
        self.replicate = int(replicate)
        self.total_epochs = int(total_epochs)
        self.config_hash = str(config_hash)
        self.checkpoint_directory = Path(checkpoint_directory)

    def on_epoch_end(self, epoch, logs=None):
        completed = int(epoch + 1)
        checkpoint = self.checkpoint_directory / f"checkpoint_epoch_{completed:03d}.keras"
        if not checkpoint.exists():
            raise RuntimeError(
                "Epoch state cannot advance because the expected epoch "
                f"checkpoint does not exist: {checkpoint}"
            )
        state = {
            "schema_version": "2.0",
            "seed": self.seed,
            "condition": self.condition,
            "replicate": self.replicate,
            "total_epochs": self.total_epochs,
            "completed_epochs": completed,
            "latest_checkpoint": checkpoint.name,
            "config_hash": self.config_hash,
            "status": "complete" if completed >= self.total_epochs else "in_progress",
            "last_epoch_logs": {
                key: float(value)
                for key, value in (logs or {}).items()
                if np.isscalar(value)
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(self.state_path if False else temporary, self.state_path)

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

        completed_epochs = 0
        resume_checkpoint = None

        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)

            expected = {
                "seed": int(self.seed),
                "condition": str(condition),
                "replicate": int(replicate),
                "total_epochs": int(self.epochs),
                "config_hash": str(run_config_hash),
            }
            for key, expected_value in expected.items():
                actual = state.get(key)
                if actual != expected_value:
                    raise RuntimeError(
                        f"Refusing to resume {condition} replicate {replicate}: "
                        f"state field {key!r} is {actual!r}, expected {expected_value!r}. "
                        "The experiment configuration must remain immutable across sessions."
                    )

            completed_epochs = int(state.get("completed_epochs", 0))
            if completed_epochs < 0 or completed_epochs > self.epochs:
                raise RuntimeError("Invalid completed_epochs in D4 state.json.")

            checkpoint_name = state.get("latest_checkpoint")
            if completed_epochs > 0:
                expected_name = f"checkpoint_epoch_{completed_epochs:03d}.keras"
                if checkpoint_name != expected_name:
                    raise RuntimeError(
                        "D4 state/checkpoint mismatch: latest_checkpoint does not "
                        "match completed_epochs."
                    )
                resume_checkpoint = checkpoint_directory / expected_name
                if not resume_checkpoint.exists():
                    raise RuntimeError(
                        f"D4 state says epoch {completed_epochs} is complete, but "
                        f"{resume_checkpoint} is missing. Refusing to guess."
                    )

            print(
                f"Resuming {condition} replicate {replicate} from "
                f"epoch {completed_epochs}/{self.epochs}..."
            )
            if completed_epochs >= self.epochs:
                self.model = load_model(resume_checkpoint, compile=True)
                return self.model

            self.model = load_model(resume_checkpoint, compile=True)
        else:
            self.model = self.build_model()

        # Keep the original Gohr learning-rate schedule.  The only training
        # change here is crash-safe checkpointing/resumption.
        scheduler = LearningRateScheduler(
            tn.cyclic_lr(10, 0.002, 0.0001)
        )

        callbacks = [scheduler]

        # Preserve the original best-model artifact separately.
        best_checkpoint = tn.make_checkpoint(
            str(
                self.model_directory
                / f"best_{self.num_rounds}r_depth{self.depth}.keras"
            )
        )
        callbacks.insert(0, best_checkpoint)

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