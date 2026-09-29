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
    num_blocks: int = 2,
    num_filters: int = 32,
    d1: int = 64,
    d2: int = 64,
    word_size: int = 16,
    kernel_size: int = 3,
    reg_param: float = 1e-5,
    filters: int | None = None,
    input_words: int | None = None,
    dense_1: int | None = None,
    dense_2: int | None = None,
    l2_reg: float | None = None,
):
    """
    Independently reconstructed Gohr ResNet architecture.

    Canonical configuration used for the audited 5-round Speck32/64
    distinguisher:

        depth       = 10
        filters     = 32
        input       = 64 bits
        dense_1     = 64
        dense_2     = 64
        kernel size = 3
        L2          = 1e-5

    The compatibility aliases below exist because the M1 verifier uses
    the descriptive names:

        filters
        dense_1
        dense_2
        l2_reg

    They map directly onto the canonical parameters and do not change
    the scientific architecture.
    """

    # ---------------------------------------------------------
    # Compatibility aliases
    # ---------------------------------------------------------

    if filters is not None:
        filters = int(filters)

        if num_filters != 32 and num_filters != filters:
            raise ValueError(
                "Conflicting values supplied for num_filters and filters."
            )

        num_filters = filters

    if input_words is not None:
        input_words = int(input_words)

        if input_words <= 0:
            raise ValueError(
                "input_words must be positive."
            )

        if input_words % 2 != 0:
            raise ValueError(
                "input_words must be even because each ciphertext "
                "contains two 16-bit words."
            )

        implied_num_blocks = input_words // 2

        if num_blocks != 2 and num_blocks != implied_num_blocks:
            raise ValueError(
                "Conflicting values supplied for num_blocks and "
                "input_words."
            )

        num_blocks = implied_num_blocks

    if dense_1 is not None:
        dense_1 = int(dense_1)

        if d1 != 64 and d1 != dense_1:
            raise ValueError(
                "Conflicting values supplied for d1 and dense_1."
            )

        d1 = dense_1

    if dense_2 is not None:
        dense_2 = int(dense_2)

        if d2 != 64 and d2 != dense_2:
            raise ValueError(
                "Conflicting values supplied for d2 and dense_2."
            )

        d2 = dense_2

    if l2_reg is not None:
        l2_reg = float(l2_reg)

        if reg_param != 1e-5 and reg_param != l2_reg:
            raise ValueError(
                "Conflicting values supplied for reg_param and l2_reg."
            )

        reg_param = l2_reg

    # ---------------------------------------------------------
    # Validate parameters
    # ---------------------------------------------------------

    if depth < 0:
        raise ValueError(
            "depth must be non-negative."
        )

    if num_blocks <= 0:
        raise ValueError(
            "num_blocks must be positive."
        )

    if num_filters <= 0:
        raise ValueError(
            "num_filters must be positive."
        )

    if word_size <= 0:
        raise ValueError(
            "word_size must be positive."
        )

    if d1 <= 0 or d2 <= 0:
        raise ValueError(
            "Dense-layer widths d1 and d2 must be positive."
        )

    if kernel_size <= 0:
        raise ValueError(
            "kernel_size must be positive."
        )

    if reg_param < 0:
        raise ValueError(
            "reg_param must be non-negative."
        )

    # ---------------------------------------------------------
    # Keras imports
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------

    input_width = num_blocks * word_size * 2

    inp = Input(
        shape=(input_width,)
    )

    # Gohr:
    #
    #   Reshape((2 * num_blocks, word_size))
    #   Permute((2, 1))
    #
    # Canonical case:
    #
    #   64 -> (4, 16) -> (16, 4)

    x = Reshape(
        (2 * num_blocks, word_size)
    )(inp)

    x = Permute(
        (2, 1)
    )(x)

    # ---------------------------------------------------------
    # Initial bit-sliced convolution
    # ---------------------------------------------------------

    x = Conv1D(
        num_filters,
        kernel_size=1,
        padding="same",
        kernel_regularizer=l2(reg_param),
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut = x

    # ---------------------------------------------------------
    # Residual tower
    # ---------------------------------------------------------

    for _ in range(depth):

        residual = Conv1D(
            num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=l2(reg_param),
        )(shortcut)

        residual = BatchNormalization()(residual)
        residual = Activation("relu")(residual)

        residual = Conv1D(
            num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=l2(reg_param),
        )(residual)

        residual = BatchNormalization()(residual)
        residual = Activation("relu")(residual)

        shortcut = Add()(
            [
                shortcut,
                residual,
            ]
        )

    # ---------------------------------------------------------
    # Prediction head
    # ---------------------------------------------------------

    x = Flatten()(shortcut)

    x = Dense(
        d1,
        kernel_regularizer=l2(reg_param),
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(
        d2,
        kernel_regularizer=l2(reg_param),
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    out = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=l2(reg_param),
    )(x)

    return Model(
        inputs=inp,
        outputs=out,
    )