"""
Provenance for cryptographic-audit runs.

Records what can be established and nothing more. Unavailable items are
null, never invented. The probe/CV seed and the dataset-generation
entropy are reported SEPARATELY because only the former is reproducible.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone

from audit.cryptography.audit_config import REFERENCE

#: Identifies the methodology that produced a result. Versioning lives
#: here - in metadata - not in module or file names.
EXPERIMENT_DESIGN_VERSION = "CE-cryptographic-audit-2026"
AUDIT_SCHEMA_VERSION = "ce-certificate-1"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_provenance(*, checkpoint_info: dict | None = None, config_id: str,
                     probe_seed=None, dataset_randomness: str = "os.urandom") -> dict:
    """
    Honest provenance. Unavailable items are recorded as null, never invented.

    The probe/CV seed and the dataset-generation randomness are reported
    SEPARATELY: the former is reproducible, the latter is not. Data drawn
    from os.urandom cannot be replayed byte-for-byte unless persisted.
    """
    return {
        "experiment_design_version": EXPERIMENT_DESIGN_VERSION,
        "config_id": config_id,
        "git_commit": _git_commit(),
        "checkpoint": checkpoint_info,
        "reference_configuration": REFERENCE.to_dict(),
        "seed_policy": {
            "probe_cv_seed": probe_seed,
            "dataset_generation_randomness": dataset_randomness,
            "exact_dataset_replay_available": False,
            "note": ("dataset generation draws from os.urandom and is NOT seed-replayable; "
                     "byte-for-byte replay requires the generated data to be persisted. "
                     "The probe/CV seed is separate and IS reproducible."),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or None,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


