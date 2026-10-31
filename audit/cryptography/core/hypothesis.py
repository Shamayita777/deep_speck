"""
Hypothesis direction and association status.

The legacy ``test.base.EvidenceDirection`` has two members, HIGHER_IS_BETTER and LOWER_IS_BETTER.
It describes how a *score* is ranked, not what a hypothesis predicts, and it cannot express a
two-sided or non-directional hypothesis. ``HypothesisDirection`` states the prediction itself;
``AssociationStatus`` keeps an undefined association distinct from a zero one.

Compatibility: ``from_legacy`` maps the two legacy members to the directional hypotheses they
were used for. There is deliberately no mapping from TWO_SIDED back to a legacy member.
"""

from __future__ import annotations

from enum import Enum


class HypothesisDirection(Enum):
    POSITIVE = "POSITIVE"            # H1: association > 0
    NEGATIVE = "NEGATIVE"            # H1: association < 0
    TWO_SIDED = "TWO_SIDED"          # H1: association != 0, no direction predicted
    NO_DIRECTIONAL_CLAIM = "NO_DIRECTIONAL_CLAIM"   # descriptive quantity, no test of direction

    @property
    def tested(self) -> bool:
        return self is not HypothesisDirection.NO_DIRECTIONAL_CLAIM

    def alternative(self) -> str:
        return {"POSITIVE": "greater", "NEGATIVE": "less", "TWO_SIDED": "two-sided",
                "NO_DIRECTIONAL_CLAIM": "none"}[self.value]

    def supports(self, statistic: float) -> bool:
        """Whether the sign of a defined statistic is in the direction the hypothesis predicts.
        Two-sided hypotheses are supported by either nonzero sign."""
        if self is HypothesisDirection.POSITIVE:
            return statistic > 0
        if self is HypothesisDirection.NEGATIVE:
            return statistic < 0
        if self is HypothesisDirection.TWO_SIDED:
            return statistic != 0
        raise ValueError("a non-directional quantity supports no direction")


class AssociationStatus(Enum):
    DEFINED = "DEFINED"                          # finite coefficient computed
    UNDEFINED = "UNDEFINED"                      # e.g. constant input: no coefficient exists
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    INVALID_INPUT = "INVALID_INPUT"              # shape mismatch, non-finite values


def from_legacy(direction) -> HypothesisDirection:
    """Map ``test.base.EvidenceDirection`` to the directional hypothesis it encoded."""
    value = getattr(direction, "value", direction)
    if value == "HIGHER_IS_BETTER":
        return HypothesisDirection.POSITIVE
    if value == "LOWER_IS_BETTER":
        return HypothesisDirection.NEGATIVE
    raise ValueError(f"no legacy mapping for {direction!r}")
