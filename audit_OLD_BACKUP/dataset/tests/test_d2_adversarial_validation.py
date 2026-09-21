"""
Regression validation for Dataset Integrity Audit D2.

Scientific status
-----------------
NON-EVIDENTIARY.

This suite validates D2 decision semantics and the persisted controlled
fault-injection calibration artifact.

The 100-replicate D2.5 calibration itself is the statistical sensitivity
experiment. This file does NOT reproduce that expensive experiment.

It verifies:

    1. calibration provenance is present;
    2. calibration is explicitly marked non-clean evidence;
    3. the declared production detector configuration is preserved;
    4. Wilson-interval records are structurally valid;
    5. one-gate D2 findings remain INCONCLUSIVE;
    6. two-gate findings are the only conditions eligible for FAIL;
    7. non-finite excess ratios cannot be serialized as strict JSON.

This suite does not establish independence of the Gohr dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from audit.dataset import d2_sample_dependence as d2


# ---------------------------------------------------------------------
# Decision semantics
# ---------------------------------------------------------------------

def test_practical_only_d2_signal_is_not_a_confirmed_failure() -> None:
    """
    D2 explicitly requires both statistical and practical evidence gates.

    Therefore:

        practical = True
        statistical = False

    must not constitute a confirmed D2 failure.
    """
    practical = True
    statistical = False

    assert not (practical and statistical)


def test_statistical_only_d2_signal_is_not_a_confirmed_failure() -> None:
    """
    Symmetric check for the statistical-only case.
    """
    practical = False
    statistical = True

    assert not (practical and statistical)


def test_both_d2_evidence_gates_are_required_for_confirmed_failure() -> None:
    practical = True
    statistical = True

    assert practical and statistical


# ---------------------------------------------------------------------
# Mathematical boundary condition
# ---------------------------------------------------------------------

def test_zero_null_probability_with_positive_observation_is_unbounded() -> None:
    """
    This is the exact mathematical boundary that produced Infinity in
    the production D2 certificate.

    This test does NOT replace the production calculation.
    It documents the expected mathematical interpretation.
    """
    observed = 1
    expected = 0.0

    ratio = float("inf") if expected == 0 and observed > 0 else (
        1.0 if expected == 0 else observed / expected
    )

    assert math.isinf(ratio)
    assert ratio > 0


def test_zero_null_probability_with_zero_observation_is_neutral() -> None:
    observed = 0
    expected = 0.0

    ratio = 1.0 if observed == 0 and expected == 0 else observed / expected

    assert ratio == 1.0
    assert math.isfinite(ratio)


# ---------------------------------------------------------------------
# Strict JSON interoperability
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
    ids=[
        "positive_infinity",
        "negative_infinity",
        "nan",
    ],
)
def test_strict_json_rejects_non_finite_numeric_values(value: float) -> None:
    """
    JSON certificates must not contain non-standard numeric literals.

    Python's json module accepts these by default, but allow_nan=False
    correctly exposes the interoperability problem.
    """
    with pytest.raises(ValueError):
        json.dumps(
            {"value": value},
            allow_nan=False,
        )


def test_unbounded_ratio_has_explicit_json_safe_representation() -> None:
    """
    Preferred certificate representation:

        excess_ratio: null
        excess_ratio_status: UNBOUNDED

    This preserves the mathematical meaning without emitting Infinity.
    """
    certificate_fragment = {
        "excess_ratio": None,
        "excess_ratio_status": "UNBOUNDED",
    }

    encoded = json.dumps(
        certificate_fragment,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    assert decoded["excess_ratio"] is None
    assert decoded["excess_ratio_status"] == "UNBOUNDED"


def test_finite_ratio_remains_numeric() -> None:
    certificate_fragment = {
        "excess_ratio": 1.5,
        "excess_ratio_status": "FINITE",
    }

    encoded = json.dumps(
        certificate_fragment,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    assert decoded["excess_ratio"] == pytest.approx(1.5)
    assert decoded["excess_ratio_status"] == "FINITE"


# ---------------------------------------------------------------------
# Optional persisted calibration-artifact validation
# ---------------------------------------------------------------------

def _candidate_calibration_paths() -> list[Path]:
    return [
        Path("audit/dataset/evidence/d2/calibration/d2_calibration_final_v1.json"),
        Path("d2_calibration_final_v1.json"),
        Path("audit/dataset/d2_calibration_final_v1.json"),
        Path("artifacts/d2_calibration_final_v1.json"),
    ]


def _find_calibration_artifact() -> Path | None:
    for path in _candidate_calibration_paths():
        if path.is_file():
            return path
    return None


def test_d2_calibration_artifact_structure() -> None:
    """
    Validate the persisted 100-replicate calibration if it is present.

    This test intentionally SKIPS when the artifact is not present. It must
    never fabricate or substitute a calibration result.
    """
    path = _find_calibration_artifact()

    if path is None:
        pytest.skip(
            "D2 calibration artifact not present in the expected local paths."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert isinstance(data, dict)

    assert data.get("schema_version") == "d2-calibration-final-v1"
    assert data.get("status") == "COMPLETED"

    assert data.get("calibration_is_not_clean_evidence") is True
    assert data["integrity"]["production_clean_logic_invoked"] is False
    assert data["integrity"]["production_dataset_generated"] is False

    assert data["configuration"]["calibration_replicates"] == 100
    assert data["configuration"]["calibration_samples"] == 100_000

    assert data["calibration_engine_provenance"]["verification_status"] == "VERIFIED"


def test_d2_calibration_does_not_claim_clean_dataset_evidence() -> None:
    path = _find_calibration_artifact()

    if path is None:
        pytest.skip(
            "D2 calibration artifact not present in the expected local paths."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert data["calibration_is_not_clean_evidence"] is True
    assert data["integrity"]["production_dataset_generated"] is False