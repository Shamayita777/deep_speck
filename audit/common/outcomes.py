"""
Shared outcome vocabularies for the audit framework.

The methodology defines SEVERAL distinct vocabularies that must never
be conflated with one another:

    - DimensionOutcome: the per-pillar finding vocabulary used by
      Implementation Integrity, Dataset Integrity, and Experimental
      Validity (Sections 3.5.8, 3.6.8, 3.7.8): PASS / CONDITIONAL_PASS
      / FAIL / INCONCLUSIVE.

    - CryptographicOutcome: Cryptographic Evidence's OWN outcome
      vocabulary (Section 3.8.9), which is deliberately different
      because the question it answers is different (evidential support
      for an interpretation, not reproducibility/robustness):
      SUPPORTED / PARTIALLY_SUPPORTED / INCONCLUSIVE / NOT_SUPPORTED.

    - EvidenceLevel: the cryptographic evidence hierarchy (Section
      3.8.4), LEVEL_0 through LEVEL_4.

    - FinalAuditOutcome: the claim-level outcome produced ONLY by the
      integration decision engine (Section 3.10.6), which is again a
      different vocabulary because it integrates across all four
      dimensions: SUPPORTED / SUPPORTED_WITH_LIMITATIONS /
      PARTIALLY_SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE.

    - ReproducibilityLevel (Section 3.11.4): LEVEL_I through LEVEL_IV.

A concrete pillar implementation is free to report a domain-specific
status string that fits none of these (e.g. this project's own D3
audit reports "DESCRIPTIVE_ONLY" and D4 reports "EFFECT_DETECTED" -
both legitimate, precise, narrower statements than the generic
DimensionOutcome vocabulary would allow). This module does NOT force
every pillar output into one of these enums; audit.integration is
responsible for explicitly interpreting a non-standard pillar status
into (or explicitly declining to map it into) one of these vocabularies,
and must record that interpretation as a visible step, never a silent
coercion.
"""

from __future__ import annotations

from enum import Enum


class DimensionOutcome(str, Enum):
    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class CryptographicOutcome(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class EvidenceLevel(str, Enum):
    LEVEL_0_UNSUPPORTED = "LEVEL_0_UNSUPPORTED"
    LEVEL_1_PREDICTIVE = "LEVEL_1_PREDICTIVE"
    LEVEL_2_ROBUST_EXPERIMENTAL = "LEVEL_2_ROBUST_EXPERIMENTAL"
    LEVEL_3_CRYPTOGRAPHIC = "LEVEL_3_CRYPTOGRAPHIC"
    LEVEL_4_STRONG_CRYPTOGRAPHIC = "LEVEL_4_STRONG_CRYPTOGRAPHIC"


class FinalAuditOutcome(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_LIMITATIONS = "SUPPORTED_WITH_LIMITATIONS"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ReproducibilityLevel(str, Enum):
    LEVEL_I_COMPLETE = "LEVEL_I_COMPLETE"
    LEVEL_II_SUBSTANTIAL = "LEVEL_II_SUBSTANTIAL"
    LEVEL_III_PARTIAL = "LEVEL_III_PARTIAL"
    LEVEL_IV_NON_REPRODUCIBLE = "LEVEL_IV_NON_REPRODUCIBLE"


class ExecutionMode(str, Enum):
    SMOKE = "SMOKE"
    PRODUCTION = "PRODUCTION"


class RunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    INVALID = "INVALID"
    SKIPPED = "SKIPPED"


class EvidenceTag(str, Enum):
    """Confirmatory (preregistered) vs. exploratory (post-hoc) evidence - never presented interchangeably."""
    CONFIRMATORY = "CONFIRMATORY"
    EXPLORATORY = "EXPLORATORY"


class AssessmentTier(str, Enum):
    """
    Distinguishes CORE assessment (mandatory for a pillar finding to be
    issued at all) from ENHANCED assessment (an additional, optional,
    higher-strength component that may be performed when scientifically
    justified and practically feasible, but whose absence does not
    block a core finding).

    Introduced for Implementation Integrity per explicit methodology
    revision: independent reimplementation is an ENHANCED component,
    not a prerequisite for CORE Implementation Integrity (baseline
    reconstruction, controlled verification, comparative evaluation,
    statistical assessment). An ENHANCED component that was not
    performed is recorded as NOT_ASSESSED (see
    audit.implementation.stages), never silently omitted and never
    described as having been performed at reduced rigor.
    """
    CORE = "CORE"
    ENHANCED = "ENHANCED"


class EnhancedAssessmentStatus(str, Enum):
    NOT_ASSESSED = "NOT_ASSESSED"
    ASSESSED = "ASSESSED"
