"""
Gohr Model Definition
=====================

Neural network architecture used by the Gohr neural distinguisher.

This module is responsible only for constructing and compiling the
network architecture. It performs no dataset generation, training,
evaluation, or persistence.

The implementation follows the residual convolutional architecture
introduced by Gohr while remaining independent of the Cryptographic
Evidence framework.
"""

from __future__ import annotations

from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    Input,
    Permute,
    Reshape,
)
from keras.models import Model
from keras.regularizers import l2


class GohrModel:
    """
    Factory for constructing the Gohr neural distinguisher.
    """

    def __init__(
        self,
        *,
        num_blocks: int = 2,
        word_size: int = 16,
        num_filters: int = 32,
        depth: int = 5,
        dense_units_1: int = 64,
        dense_units_2: int = 64,
        kernel_size: int = 3,
        regularization: float = 1e-5,
        output_units: int = 1,
        output_activation: str = "sigmoid",
        optimizer: str = "adam",
        loss: str = "mse",
        metrics: list[str] | None = None,
    ) -> None:

        self.num_blocks = num_blocks
        self.word_size = word_size
        self.num_filters = num_filters
        self.depth = depth
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.kernel_size = kernel_size
        self.regularization = regularization
        self.output_units = output_units
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or ["acc"]

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def build(self) -> Model:
        """
        Construct and compile the Gohr residual network.
        """

        inp = Input(
            shape=(
                self.num_blocks * self.word_size * 2,
            )
        )

        x = Reshape(
            (
                2 * self.num_blocks,
                self.word_size,
            )
        )(inp)

        x = Permute((2, 1))(x)

        x = Conv1D(
            self.num_filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=l2(self.regularization),
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        shortcut = x

        for _ in range(self.depth):

            residual = Conv1D(
                self.num_filters,
                kernel_size=self.kernel_size,
                padding="same",
                kernel_regularizer=l2(
                    self.regularization
                ),
            )(shortcut)

            residual = BatchNormalization()(residual)
            residual = Activation("relu")(residual)

            residual = Conv1D(
                self.num_filters,
                kernel_size=self.kernel_size,
                padding="same",
                kernel_regularizer=l2(
                    self.regularization
                ),
            )(residual)

            residual = BatchNormalization()(residual)
            residual = Activation("relu")(residual)

            shortcut = Add()(
                [
                    shortcut,
                    residual,
                ]
            )

        x = Flatten()(shortcut)

        x = Dense(
            self.dense_units_1,
            kernel_regularizer=l2(
                self.regularization
            ),
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Dense(
            self.dense_units_2,
            kernel_regularizer=l2(
                self.regularization
            ),
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        output = Dense(
            self.output_units,
            activation=self.output_activation,
            kernel_regularizer=l2(
                self.regularization
            ),
        )(x)

        model = Model(
            inputs=inp,
            outputs=output,
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        return model

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "Gohr Residual Network"

    @property
    def description(self) -> str:
        return (
            "Residual convolutional neural distinguisher "
            "architecture proposed by Gohr."
        )