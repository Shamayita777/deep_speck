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
import speck as sp
import train_nets as tn
class EpochStateCallback(Callback):
    """
    Persist the completed epoch after each successful epoch.

    The checkpoint itself is written by ModelCheckpoint before
    this callback records the epoch state.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        seed: int,
        condition: str,
        replicate: int,
        total_epochs: int,
    ) -> None:
        super().__init__()

        self.state_path = state_path
        self.seed = int(seed)
        self.condition = condition
        self.replicate = int(replicate)
        self.total_epochs = int(total_epochs)

    def on_epoch_end(self, epoch, logs=None):

        state = {
            "schema_version": "1.0",
            "seed": self.seed,
            "condition": self.condition,
            "replicate": self.replicate,
            "total_epochs": self.total_epochs,
            "completed_epochs": int(epoch + 1),
            "status": (
                "complete"
                if epoch + 1 >= self.total_epochs
                else "in_progress"
            ),
        }

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self.state_path.with_suffix(
            ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                state,
                handle,
                indent=2,
                sort_keys=True,
            )

        os.replace(
            temporary,
            self.state_path
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
    ):

        tn.set_seed(self.seed)

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        completed_epochs = 0

        # ----------------------------------------------------------
        # Resume from checkpoint if one exists
        # ----------------------------------------------------------

        if (
            checkpoint_path is not None
            and state_path is not None
            and checkpoint_path.exists()
            and state_path.exists()
        ):
            with state_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                state = json.load(handle)

            completed_epochs = int(
                state.get(
                    "completed_epochs",
                    0,
                )
            )

            print(
                f"Resuming {condition} replicate "
                f"{replicate} from epoch "
                f"{completed_epochs}/{self.epochs}..."
            )

            self.model = load_model(
                checkpoint_path,
                compile=True,
            )

            # Already finished.
            if completed_epochs >= self.epochs:
                print(
                    f"{condition} replicate {replicate} "
                    f"already completed."
                )
                return self.model

        else:
            self.model = self.build_model()

        # ----------------------------------------------------------
        # Best-model checkpoint used by the original training code
        # ----------------------------------------------------------

        best_checkpoint = tn.make_checkpoint(
            str(
                self.model_directory
                / f"best_{self.num_rounds}r_depth{self.depth}.keras"
            )
        )

        # ----------------------------------------------------------
        # Rolling recovery checkpoint
        #
        # This overwrites latest.keras every epoch.
        # ----------------------------------------------------------

        scheduler = LearningRateScheduler(
            tn.cyclic_lr(
                10,
                0.002,
                0.0001,
            )
            )

        callbacks = [
            best_checkpoint,
            scheduler,
        ]

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            rolling_checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_weights_only=False,
                save_best_only=False,
                save_freq="epoch",
                verbose=0,
            )

            callbacks.append(
                rolling_checkpoint
            )

        if state_path is not None:
            callbacks.append(
                EpochStateCallback(
                    state_path=state_path,
                    seed=self.seed,
                    condition=condition,
                    replicate=replicate,
                    total_epochs=self.epochs,
                )
            )

        # ----------------------------------------------------------
        # Continue training
        # ----------------------------------------------------------

        self.history = self.model.fit(
            train_x,
            train_y,
            validation_data=(
                self.validation_x,
                self.validation_y,
            ),
            epochs=self.epochs,
            initial_epoch=completed_epochs,
            batch_size=self.batch_size,
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