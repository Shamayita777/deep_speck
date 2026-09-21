"""
Dataset Audit Provenance Infrastructure.

D0 provenance records identify what dataset representation was
audited, how it was obtained, and under which configuration the
audit was executed.

This module does not make any scientific decision.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def sha256_bytes(data: bytes) -> str:
    """
    Compute a SHA-256 digest for arbitrary bytes.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """
    Compute the SHA-256 digest of a file.

    The file is read incrementally to avoid loading the entire
    file into memory.
    """

    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def array_representation_metadata(
    array: np.ndarray,
) -> dict[str, Any]:
    """
    Return metadata describing the exact NumPy representation
    supplied to the audit.

    The representation includes dtype and shape in addition to
    raw values because these affect how the data are represented
    computationally.
    """

    array = np.asarray(array)

    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "ndim": int(array.ndim),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
    }


def array_sha256(
    array: np.ndarray,
) -> str:
    """
    Compute a SHA-256 digest over the exact array representation.

    The digest commits to:
        - dtype
        - shape
        - raw C-contiguous bytes

    This avoids treating arrays with different computational
    representations as identical merely because their raw byte
    payload happens to match.
    """

    array = np.asarray(array)

    contiguous = np.ascontiguousarray(array)

    digest = hashlib.sha256()

    dtype_bytes = str(contiguous.dtype).encode("utf-8")
    shape_bytes = json.dumps(
        list(contiguous.shape),
        separators=(",", ":"),
    ).encode("utf-8")

    digest.update(
        len(dtype_bytes).to_bytes(8, "big")
    )
    digest.update(dtype_bytes)

    digest.update(
        len(shape_bytes).to_bytes(8, "big")
    )
    digest.update(shape_bytes)

    digest.update(contiguous.tobytes())

    return digest.hexdigest()


def build_provenance(
    *,
    dataset_id: str,
    dataset_version: Optional[str],
    generation_procedure: Optional[str],
    generation_parameters: Optional[Mapping[str, Any]],
    random_seed: Optional[int],
    partitions: Mapping[str, Mapping[str, Any]],
    audit_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a standardized provenance record.

    Parameters
    ----------
    dataset_id:
        Stable identifier for the dataset or generation procedure.

    dataset_version:
        Version identifier if available.

    generation_procedure:
        Human-readable description of how the dataset was obtained.

    generation_parameters:
        Parameters controlling generation.

    random_seed:
        Seed if deterministic seeded generation was used.
        None is valid and explicitly records that no fixed seed
        was supplied.

    partitions:
        Per-partition provenance metadata.

    audit_configuration:
        Configuration specific to the audit execution.

    Returns
    -------
    dict
        JSON-serializable provenance record.
    """

    return {
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_procedure": generation_procedure,
            "generation_parameters": (
                dict(generation_parameters)
                if generation_parameters is not None
                else None
            ),
            "random_seed": random_seed,
        },
        "partitions": {
            name: dict(metadata)
            for name, metadata in partitions.items()
        },
        "audit_configuration": dict(audit_configuration),
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    }