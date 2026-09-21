"""
Gohr residual Conv1D architecture.

Ported verbatim from archive/train_nets.py's make_resnet(). This
architecture is a held-constant control across every EV experiment in
the frozen core (H-EV-SHUFFLE, H-EV-REPRESENTATION) - it is never
modified. H-EV-ARCHITECTURE (comparing against a structurally different
network) is explicitly excluded from the EV core per the frozen
scientific scope and is not implemented here.
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


def make_resnet(
    *,
    depth: int,
    num_blocks=2,
    num_filters=32,
    num_outputs=1,
    d1=64,
    d2=64,
    word_size=16,
    ks=3,
    reg_param=0.0001,
    final_activation="sigmoid",
):
    """
    Identical in computation to archive/train_nets.py::make_resnet.

    DELIBERATE DEVIATION FROM THE REFERENCE SIGNATURE (II-4 remediation):
    `depth` is REQUIRED and keyword-only. The reference defaults it to a
    value, and the CE pillar's independent reimplementation defaulted it
    to 5 while the reference protocol declares 10; because no CE driver
    overrode the default, every historical CE experiment silently ran a
    depth-5 model (finding II-FINDING-DEPTH-V1). Removing the default
    makes that class of silent nonconformance impossible to repeat: a
    caller must now state the depth it intends.

    The layer computation below is unchanged, so models built with an
    explicit depth are identical to what the reference produces at the
    same depth. Historical artifacts are unaffected - this changes only
    how future models are constructed.
    """
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError(f"depth must be a positive integer, got {depth!r}")
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding="same", kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation("relu")(conv0)
    # add residual blocks
    shortcut = conv0
    for _ in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding="same", kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding="same", kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        shortcut = Add()([shortcut, conv2])
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation("relu")(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation("relu")(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model
