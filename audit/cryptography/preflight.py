"""
Production preflight gate.

The final check before expensive GPU execution. Every scientifically
important property is verified HERE, before anything trains, so a
misconfigured run fails in seconds rather than after hours.

Depth is not a decorative config field: the checkpoint's hash is checked,
its architecture is read from the file, and the realized depth must be 10.
The historical artifact whose filename says `depth10` but which contains a
depth-5 network is rejected explicitly.
"""

from __future__ import annotations

from pathlib import Path

from audit.cryptography.audit_config import REFERENCE, require_reference_config
from audit.cryptography.integrity import verify_reference_checkpoint
from audit.cryptography.output_policy import assert_audit_output_path


class PreflightError(RuntimeError):
    pass


class FrozenParameterMissing(PreflightError):
    """A scientifically necessary parameter has not been frozen."""


def require_frozen(name: str, value, *, why: str):
    """
    Refuse to enter production without a predeclared parameter.

    These values must be chosen BEFORE results exist. Defaulting them
    silently would let the design be selected after seeing the data.
    """
    if value is None:
        raise FrozenParameterMissing(
            f"'{name}' is not frozen. {why} It must be predeclared before production "
            "execution; the experiment refuses to guess.")
    return value


def preflight(*, experiment_id: str, rounds: int, differential, depth: int,
              l2_reg: float | None = None, checkpoint: str | Path | None = None,
              output_path: str | Path, repo_root=None,
              frozen: dict | None = None, production: bool = True) -> dict:
    """
    Run every gate. Returns a report; raises on the first inconsistency.
    """
    report: dict = {"experiment_id": experiment_id, "production": production}

    # 1-3. reference configuration (rounds / differential / depth / L2)
    require_reference_config(rounds=rounds, differential=differential, depth=depth,
                             l2_reg=l2_reg)
    report["reference_configuration"] = REFERENCE.to_dict()

    # 4. checkpoint: hash + realized structure (never the filename)
    if checkpoint is not None:
        report["checkpoint"] = verify_reference_checkpoint(checkpoint)
    else:
        report["checkpoint"] = None    # CE1 trains its own models

    # 5. output destination is writable and cannot touch historical evidence
    assert_audit_output_path(output_path, repo_root=repo_root)
    report["output_path"] = str(output_path)

    # 6. frozen design parameters
    missing = [k for k, v in (frozen or {}).items() if v is None]
    if production and missing:
        raise FrozenParameterMissing(
            f"{experiment_id}: production requires frozen design parameters {missing}. "
            "They must be predeclared, not chosen after seeing results.")
    report["frozen_parameters"] = dict(frozen or {})

    # 7. environment
    import platform
    import sys
    report["environment"] = {"python": sys.version.split()[0],
                             "platform": platform.platform()}
    report["status"] = "PREFLIGHT_OK"
    return report
