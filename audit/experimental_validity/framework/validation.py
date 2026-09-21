"""
Generic fail-closed validation helpers for Experimental Validity.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


class ConfigValidationError(ValueError):
    pass


def require_fields(d: Mapping[str, Any], fields: Iterable[str], *, context: str) -> None:
    """Raise ConfigValidationError (fail closed) if any field is missing."""
    missing = [f for f in fields if f not in d]
    if missing:
        raise ConfigValidationError(f"{context}: missing required fields {missing}")


def require_positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigValidationError(f"{name} must be a positive integer, got {value!r}")
    return value


def require_run_mode(mode: str) -> str:
    if mode not in ("smoke", "production"):
        raise ConfigValidationError(f"run_mode must be 'smoke' or 'production', got {mode!r}")
    return mode


def is_non_evidentiary(run_mode: str) -> bool:
    """Smoke results are never evidentiary, by construction."""
    return run_mode == "smoke"
