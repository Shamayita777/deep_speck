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


RUN_MODES = ("smoke", "calibration", "production")

#: Modes whose output may NEVER be used as confirmatory evidence.
NON_EVIDENTIARY_MODES = ("smoke", "calibration")


def require_run_mode(mode: str) -> str:
    if mode not in RUN_MODES:
        raise ConfigValidationError(
            f"run_mode must be one of {RUN_MODES}, got {mode!r}")
    return mode


def is_non_evidentiary(run_mode: str) -> bool:
    """
    Smoke AND calibration results are never evidentiary, by construction.

    CALIBRATION is a distinct tier from SMOKE: it runs the REAL frozen
    protocol (5 rounds, depth 10, 200 epochs, reference batch size /
    optimizer / LR schedule / reg_param, full dataset sizes), so its
    numbers are physically comparable to production. That is exactly why
    it needs a hard, separate guard: a calibration certificate looks like
    a production certificate in every respect except this flag. Its sole
    legitimate use is estimating nuisance variance for power planning; it
    must never enter a confirmatory hypothesis test, a practical-margin
    choice, or the primary-family multiplicity correction.
    """
    return run_mode in NON_EVIDENTIARY_MODES


def is_calibration(run_mode: str) -> bool:
    return run_mode == "calibration"
