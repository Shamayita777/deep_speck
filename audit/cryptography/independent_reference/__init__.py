"""
Independent reference implementations for CipherMind audit verification.

This package is deliberately isolated from the production cryptographic
theory implementation. Its purpose is to provide independently written
reference calculations that can be compared against the production path.

IMPORTANT:
    Nothing in this package imports the production xdp_plus() implementation.
"""

from .lipmaa_moriai_independent import (
    independent_xdp_plus,
    independent_xdp_plus_array,
    chain_transition_probabilities,
)

__all__ = [
    "independent_xdp_plus",
    "independent_xdp_plus_array",
    "chain_transition_probabilities",
]