"""
Gohr dataset/model adapter.

This module is the dataset-specific boundary for Gohr/Speck.

Generic dataset-audit modules must not import the Gohr generator
directly. They receive arrays through this adapter.

The adapter retains the existing Gohr model/training/evaluation
interface and adds the canonical dataset-generation method used by
Dataset Integrity experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from keras.callbacks import LearningRateScheduler
from keras.models import load_model

import speck as sp
import train_nets as tn


class GohrAdapter:
    """Adapter for the Gohr neural distinguisher and dataset generator."""

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

    # ----------------------------------------------------------
    # Dataset generation
    # ----------------------------------------------------------

    @property
    def feature_bits(self) -> int:
        """Number of binary features emitted by Gohr's generator."""

        return 64

    def generate_partition(
        self,
        samples: int,
        *,
        num_rounds: int | None = None,
    ) -> Tuple[Any, Any]:
        """
        Generate one Gohr/Speck dataset partition.

        This is the single canonical adapter boundary for dataset
        generation used by Dataset Integrity experiments.

        Gohr's make_train_data() uses its own randomness source;
        this method does not claim deterministic generation from
        self.seed.
        """

        if samples < 1:
            raise ValueError("samples must be >= 1.")

        rounds = self.num_rounds if num_rounds is None else int(num_rounds)

        if rounds < 1:
            raise ValueError("num_rounds must be >= 1.")

        features, labels = sp.make_train_data(
            int(samples),
            rounds,
        )

        return features, labels

    def dataset_provenance(
        self,
        *,
        train_samples: int,
        validation_samples: int,
        test_samples: int,
    ) -> dict[str, Any]:
        """Return Gohr-specific generation metadata for certificates."""

        return {
            "dataset_id": self.DATASET_ID,
            "dataset_version": self.DATASET_VERSION,
            "generation_procedure": (
                f"{self.GENERATOR_NAME}(n, nr)"
            ),
            "generation_parameters": {
                "generator": self.GENERATOR_NAME,
                "num_rounds": self.num_rounds,
                "input_difference": {
                    "left_word": "0x0040",
                    "right_word": "0x0000",
                },
                "randomness_source": "os.urandom",
                "deterministic_seed": None,
                "train_samples": int(train_samples),
                "validation_samples": int(validation_samples),
                "test_samples": int(test_samples),
            },
            "generation_random_seed": None,
        }

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train(
        self,
        train_x,
        train_y,
    ):

        tn.set_seed(self.seed)

        self.model = self.build_model()

        checkpoint = tn.make_checkpoint(
            str(
                self.model_directory
                / f"best_{self.num_rounds}r_depth{self.depth}.keras"
            )
        )

        scheduler = LearningRateScheduler(
            tn.cyclic_lr(
                10,
                0.002,
                0.0001,
            )
        )

        self.history = self.model.fit(
            train_x,
            train_y,
            validation_data=(
                self.validation_x,
                self.validation_y,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                checkpoint,
                scheduler,
            ],
            verbose=1,
        )

        return self.model

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    def evaluate(
        self,
        model,
    ):

        _, score = model.evaluate(
            self.test_x,
            self.test_y,
            verbose=0,
        )

        return float(score)

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def best_validation_score(self):

        if self.history is None:
            return None

        history = self.history.history

        if "val_accuracy" in history:
            return max(history["val_accuracy"])

        if "val_acc" in history:
            return max(history["val_acc"])

        return None
