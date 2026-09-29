"""
Independent reconstruction of Gohr's neural distinguisher architecture.

SOURCE BASIS
------------
The architecture is reconstructed from Gohr's public supplementary
implementation `train_nets.py`, with the published paper used as the
scientific description of the model.

Canonical 5-round Speck32/64 configuration:

    num_blocks   = 2
    word_size    = 16
    input        = 64 bits
    num_filters  = 32
    depth        = 10
    kernel_size  = 3
    dense_1      = 64
    dense_2      = 64
    output       = 1 sigmoid
    L2           = 1e-5

IMPORTANT
---------
The public Gohr implementation applies BatchNormalization after BOTH
64-unit dense layers. This reconstruction follows the public executable
code path rather than silently substituting a prose-only interpretation.

The aliases `filters` and `l2_strength` exist solely for compatibility
with the M1 verification/training drivers. They map directly onto the
canonical `num_filters` and `reg_param` parameters and do not define a
different architecture.
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


DEFAULT_DEPTH = 10
DEFAULT_FILTERS = 32
DEFAULT_DENSE_1 = 64
DEFAULT_DENSE_2 = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_WORD_SIZE = 16
DEFAULT_NUM_BLOCKS = 2
DEFAULT_L2 = 1e-5


def build_gohr_model(
    *,
    depth: int = DEFAULT_DEPTH,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    num_filters: int = DEFAULT_FILTERS,
    d1: int = DEFAULT_DENSE_1,
    d2: int = DEFAULT_DENSE_2,
    word_size: int = DEFAULT_WORD_SIZE,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    reg_param: float = DEFAULT_L2,
    filters: int | None = None,
    input_words: int | None = None,
    l2_strength: float | None = None,
) -> Model:
    """
    Build the independently reconstructed Gohr ResNet.

    Canonical input representation:

        two 32-bit ciphertexts
        = four 16-bit words
        = 64 binary input features.

    Compatibility aliases
    ---------------------
    filters:
        Alias for `num_filters`.

    l2_strength:
        Alias for `reg_param`.

    input_words:
        Alias-based way of specifying the number of 16-bit words in the
        four-word input representation.

    These aliases are compatibility conveniences only. The scientific
    configuration remains the Gohr configuration documented above.
    """

    # ==============================================================
    # Resolve compatibility aliases
    # ==============================================================

    if filters is not None:
        filters = int(filters)

        if num_filters != DEFAULT_FILTERS and num_filters != filters:
            raise ValueError(
                "Conflicting values supplied for num_filters and filters."
            )

        num_filters = filters

    if l2_strength is not None:
        l2_strength = float(l2_strength)

        if reg_param != DEFAULT_L2 and reg_param != l2_strength:
            raise ValueError(
                "Conflicting values supplied for reg_param and l2_strength."
            )

        reg_param = l2_strength

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

        if (
            num_blocks != DEFAULT_NUM_BLOCKS
            and num_blocks != implied_num_blocks
        ):
            raise ValueError(
                "Conflicting values supplied for num_blocks and "
                "input_words."
            )

        num_blocks = implied_num_blocks

    # ==============================================================
    # Validate parameters
    # ==============================================================

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
            "d1 and d2 must be positive."
        )

    if kernel_size <= 0:
        raise ValueError(
            "kernel_size must be positive."
        )

    if reg_param < 0:
        raise ValueError(
            "reg_param must be non-negative."
        )

    # ==============================================================
    # Input
    # ==============================================================

    # Gohr:
    #
    #     inp = Input(shape=(num_blocks * word_size * 2,))
    #
    input_width = (
        num_blocks
        * word_size
        * 2
    )

    inp = Input(
        shape=(input_width,)
    )

    # Gohr:
    #
    #     rs = Reshape((2 * num_blocks, word_size))(inp)
    #
    x = Reshape(
        (
            2 * num_blocks,
            word_size,
        )
    )(inp)

    # Gohr:
    #
    #     perm = Permute((2,1))(rs)
    #
    x = Permute(
        (2, 1)
    )(x)

    # ==============================================================
    # Initial bit-sliced convolution
    # ==============================================================

    x = Conv1D(
        num_filters,
        kernel_size=1,
        padding="same",
        kernel_regularizer=l2(reg_param),
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut = x

    # ==============================================================
    # Residual tower
    # ==============================================================

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

    # ==============================================================
    # Prediction head
    # ==============================================================

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