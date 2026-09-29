"""
Independent reconstruction of Gohr's neural distinguisher architecture.

The architecture follows the public supplementary implementation
`train_nets.py`, with the published paper used as the architectural
description.

Important source discrepancy:
    The paper's prose states that the second 64-unit dense hidden layer
    does not use batch normalization.

    The public supplementary `train_nets.py` DOES apply BatchNormalization
    to this second dense layer.

This implementation follows the actual public supplementary code path,
while the discrepancy is recorded by the M1 verifier.

For the 5-round reference network:
    depth = 10 residual blocks
    filters = 32
    kernel size = 3
    input = 64 bits
    dense widths = 64, 64
    output = sigmoid
    L2 regularization = 1e-5
"""

from __future__ import annotations
from warnings import filters

from keras import Input, Model
from keras.layers import (
    Input,
    Reshape,
    Permute,
    Conv1D,
    BatchNormalization,
    Activation,
    Add,
    Flatten,
    Dense,
)
from keras.regularizers import l2
from keras.models import Model

DEFAULT_DEPTH = 10
DEFAULT_FILTERS = 32
DEFAULT_DENSE_1 = 64
DEFAULT_DENSE_2 = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_WORD_SIZE = 16
DEFAULT_INPUT_WORDS = 4
DEFAULT_L2 = 1e-5


def build_gohr_model(
    *,
    depth: int = 10,
    filters: int = 32,
    dense_1: int = 64,
    dense_2: int = 64,
    word_size: int = 16,
    kernel_size: int = 3,
    l2_reg: float = 1e-5,
) -> Model:
    """
    Independent reconstruction of Gohr's ResNet architecture for
    the canonical Speck32/64 distinguisher.

    Based directly on the public Gohr supplementary implementation.

    Canonical values:
        num_blocks = 2
        word_size = 16
        depth = 10
        filters = 32
        d1 = 64
        d2 = 64
        kernel_size = 3
        l2_reg = 1e-5

    The network input represents two 32-bit ciphertext blocks:

        2 ciphertexts × 2 words/ciphertext × 16 bits/word
        = 64 input bits.
    """

    # Gohr:
    # inp = Input(shape=(num_blocks * word_size * 2,))
    input_width = 2 * word_size * 2

    inp = Input(shape=(input_width,))

    rs = Reshape(
        (4, word_size)
    )(inp)

    perm = Permute((2, 1))(rs)

    conv0 = Conv1D(
        filters,
        kernel_size=1,
        padding="same",
        kernel_regularizer=l2(l2_reg),
    )(perm)

    conv0 = BatchNormalization()(conv0)
    conv0 = Activation("relu")(conv0)

    shortcut = conv0

    # Gohr's residual tower.
    for _ in range(depth):
        conv1 = Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )(shortcut)

        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)

        conv2 = Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )(conv1)

        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)

        shortcut = Add()(
            [shortcut, conv2]
        )

    # Prediction head.
    flat1 = Flatten()(shortcut)

    dense1 = Dense(
        dense_1,
        kernel_regularizer=l2(l2_reg),
    )(flat1)

    dense1 = BatchNormalization()(dense1)
    dense1 = Activation("relu")(dense1)

    dense2 = Dense(
        dense_2,
        kernel_regularizer=l2(l2_reg),
    )(dense1)

    dense2 = BatchNormalization()(dense2)
    dense2 = Activation("relu")(dense2)

    out = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=l2(l2_reg),
    )(dense2)

    return Model(
        inputs=inp,
        outputs=out,
    )