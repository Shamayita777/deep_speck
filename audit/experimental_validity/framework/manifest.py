"""
Machine-readable manifest construction for Experimental Validity.

Fails closed: build_manifest() raises if any required field is missing
rather than silently emitting a partial manifest. This operationalizes
"the certificate must fail closed if required provenance is missing".
"""

from __future__ import annotations

from typing import Any

REQUIRED_MANIFEST_FIELDS = [
    "experiment_id",
    "hypothesis_id",
    "experiment_version",
    "methodology_version",
    "cipher",
    "task",
    "rounds",
    "differential",
    "architecture",
    "depth",
    "dataset_id",
    "dataset_role",
    "dataset_size",
    "dataset_hash",
    "dataset_generation_method",
    "exact_replay_available",
    "model_seed",
    "dataset_seed",
    "shuffle_seed_if_available",
    "evaluation_seed_if_available",
    "statistics_seed",
    "software_versions",
    "condition",
    "independent_variable",
    "dependent_variable",
    "controls",
    "replicate_id",
    "run_id",
    "raw_metrics",
    "alpha",
    "practical_threshold",
    "practical_threshold_status",
    "decision",
    "warnings",
    "limitations",
    "failure_status",
    "timestamps",
]

# Fields that are legitimately optional / present only in some manifests
OPTIONAL_MANIFEST_FIELDS = [
    "git_commit",
    "config_hash",
    "effect_size",
    "confidence_interval",
    "p_value",
    "adjusted_p_value",
    "failure_reason",
]


class ManifestValidationError(ValueError):
    pass


def build_manifest(fields: dict[str, Any]) -> dict[str, Any]:
    """
    Construct and validate a manifest. Raises ManifestValidationError
    (fail closed) if any required field is absent. `None` is an
    acceptable *value* for many fields (e.g. dataset_seed=None for
    os.urandom-based generation) - the check is for key presence, not
    truthiness, so that legitimate "explicitly recorded as unavailable"
    values are never confused with "field omitted".
    """
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in fields]
    if missing:
        raise ManifestValidationError(
            f"Manifest is missing required provenance fields: {missing}. "
            "Refusing to emit an incomplete manifest."
        )
    manifest = {f: fields[f] for f in REQUIRED_MANIFEST_FIELDS}
    for f in OPTIONAL_MANIFEST_FIELDS:
        if f in fields:
            manifest[f] = fields[f]
    return manifest
