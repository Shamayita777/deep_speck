"""
CipherMind II-2: Independent Reimplementation.

This package contains an independently written reconstruction of the
computational path used by Gohr's 5-round Speck32/64 neural distinguisher.

The implementation is deliberately isolated from the project's reference
implementation. It does not import the reference Speck implementation,
reference dataset generator, or reference neural model.

Source basis:
    Gohr, A. (2019), "Improving Attacks on Round-Reduced Speck32/64
    Using Deep Learning", CRYPTO 2019.
    Supplementary repository:
    https://github.com/agohr/deep_speck
"""

from .speck_independent import (
    WORD_SIZE,
    ALPHA,
    BETA,
    MASK,
    rotate_left,
    rotate_right,
    round_encrypt,
    expand_key,
    encrypt,
    decrypt,
    convert_to_binary,
    CHECK_VECTOR,
)

from .dataset_independent import (
    GOHR_DIFFERENTIAL,
    generate_dataset,
    construct_from_material,
)

from .model_independent import build_gohr_model

__all__ = [
    "WORD_SIZE",
    "ALPHA",
    "BETA",
    "MASK",
    "rotate_left",
    "rotate_right",
    "round_encrypt",
    "expand_key",
    "encrypt",
    "decrypt",
    "convert_to_binary",
    "CHECK_VECTOR",
    "GOHR_DIFFERENTIAL",
    "generate_dataset",
    "construct_from_material",
    "build_gohr_model",
]