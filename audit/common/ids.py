"""
Identifier validation for the audit framework.

Generic: knows the four pillar names as an enum (per the methodology,
these are the four audit dimensions), but nothing about any specific
experiment, cipher, or claim naming convention used inside a pillar.
Individual pillars are free to name their own experiments/hypotheses
however suits their domain (e.g. "H-EV-SHUFFLE", "CE01", "D1") - this
module only enforces that whatever string is used is a well-formed,
stable, collision-resistant identifier, and provides the ONE shared
pattern the methodology explicitly requires: immutable version suffixes
("NAME-V1", "NAME-V2", ...) so evidence is never silently overwritten.
"""

from __future__ import annotations

import re
from enum import Enum


class Dimension(str, Enum):
    """
    The four audit pillars defined by the governing methodology
    (Sections 3.5-3.8). Integration (audit.integration) is deliberately
    NOT a member of this enum - it is a cross-pillar decision layer,
    not a fifth dimension, per the methodology's own scope and per
    explicit project instruction.
    """
    IMPLEMENTATION = "IMPLEMENTATION_INTEGRITY"
    DATASET = "DATASET_INTEGRITY"
    EXPERIMENTAL = "EXPERIMENTAL_VALIDITY"
    CRYPTOGRAPHIC = "CRYPTOGRAPHIC_EVIDENCE"


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,127}$")
_VERSIONED_PATTERN = re.compile(r"^(?P<base>[A-Za-z][A-Za-z0-9_.\-]*)-[Vv](?P<version>[1-9][0-9]*)$")


class InvalidIdentifierError(ValueError):
    pass


def validate_identifier(identifier: str, *, context: str = "identifier") -> str:
    """
    Validate a generic stable identifier: starts with a letter, contains
    only letters/digits/underscore/dot/hyphen, no whitespace, bounded
    length. This is intentionally permissive about naming CONVENTION
    (a claim ID, hypothesis ID, experiment ID, and evidence ID can each
    look completely different) while still rejecting identifiers that
    would be unsafe as filenames/dict keys or that hide whitespace-based
    typos.
    """
    if not isinstance(identifier, str) or not _IDENTIFIER_PATTERN.match(identifier):
        raise InvalidIdentifierError(
            f"Invalid {context}: {identifier!r}. Must start with a letter and contain only "
            "letters, digits, '_', '.', '-', with no whitespace."
        )
    return identifier


def validate_versioned_identifier(identifier: str, *, context: str = "versioned identifier") -> tuple[str, int]:
    """
    Validate an immutable versioned artifact identifier of the form
    "NAME-V<n>" (e.g. "DATASET-AUDIT-V1", "implementation-certificate-v1").
    Returns (base_name, version_number). Required wherever the
    methodology mandates that evidence never be silently overwritten -
    see audit.common.versioning.
    """
    match = _VERSIONED_PATTERN.match(identifier)
    if not match:
        raise InvalidIdentifierError(
            f"Invalid {context}: {identifier!r}. Must match 'NAME-V<n>' "
            "(e.g. 'DATASET-AUDIT-V1'), with n a positive integer."
        )
    return match.group("base"), int(match.group("version"))
