"""
Gohr Adapter

Dataset-specific adapter for Gohr's neural distinguisher.

The adapter encapsulates the original Gohr training and
evaluation pipeline while exposing a generic interface
required by the Controlled Perturbation Audit framework.

Only the training dataset is supplied by the framework.
Validation and test datasets remain internal to the adapter.
"""

from pathlib import Path

from keras.callbacks import LearningRateScheduler
from keras.models import load_model

import train_nets as tn


class GohrAdapter:

    def __init__(
        self,
        *,
        validation_x,
        validation_y,
        test_x,
        test_y,
        num_rounds=5,
        depth=10,
        epochs=200,
        batch_size=5000,
        seed=0,
        model_directory="./freshly_trained_nets",
    ):

        self.validation_x = validation_x
        self.validation_y = validation_y

        self.test_x = test_x
        self.test_y = test_y

        self.num_rounds = num_rounds
        self.depth = depth
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = None
        self.history = None

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
                /
                f"best_{self.num_rounds}r_depth{self.depth}.keras"
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