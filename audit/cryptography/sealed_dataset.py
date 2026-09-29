"""
Sealed evaluation set: generated ONCE, persisted, hashed, reused by every block.

The frozen CE1 design requires ONE sealed 10^6 evaluation set shared across
all blocks and both arms. Regenerating per block -- even with a fixed seed --
does not satisfy it: each block would score against a different sample, so
the per-block differences would carry evaluation-sampling noise that the
design explicitly conditions away, and "conditional on the fixed sealed test
set" would be false.

This module makes the shared set an immutable on-disk artifact:
    generate once -> persist -> sha256 -> every block loads THAT file
A hash mismatch on reload is a hard error, so a resumed run cannot silently
attach to a different evaluation set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class SealedDatasetError(RuntimeError):
    pass


def _content_hash(X: np.ndarray, Y: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in (X, Y):
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def manifest_path(directory) -> Path:
    return Path(directory) / "sealed_evaluation_manifest.json"


def data_path(directory) -> Path:
    return Path(directory) / "sealed_evaluation_set.npz"


def prepare_sealed_evaluation_set(directory, *, generate_fn, rounds: int,
                                  differential, n_samples: int) -> dict:
    """
    Return the sealed set, generating it only if it does not already exist.

    `generate_fn(n_samples)` must return (X, Y). On every call after the
    first the persisted arrays are reloaded and their content hash is
    re-verified, so neither a restart nor a crash mid-run can substitute a
    replacement evaluation set.
    """
    directory = Path(directory)
    mpath, dpath = manifest_path(directory), data_path(directory)

    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        if not dpath.exists():
            raise SealedDatasetError(
                f"sealed evaluation manifest exists at {mpath} but its data file {dpath} is "
                "missing; refusing to regenerate - a replacement set would not be the sealed "
                "set the completed blocks were scored against.")
        with np.load(dpath) as z:
            X, Y = z["X"], z["Y"]
        actual = _content_hash(X, Y)
        if actual != manifest["sha256"]:
            raise SealedDatasetError(
                f"sealed evaluation set at {dpath} has hash {actual}, but the manifest "
                f"records {manifest['sha256']}. The evaluation artifact has changed; "
                "refusing to continue.")
        for key, expected in (("rounds", rounds), ("n_samples", n_samples)):
            if manifest[key] != expected:
                raise SealedDatasetError(
                    f"sealed evaluation set was built with {key}={manifest[key]}, but this "
                    f"run requires {expected}.")
        if tuple(manifest["differential"]) != tuple(differential):
            raise SealedDatasetError(
                f"sealed evaluation set differential {manifest['differential']} != "
                f"{list(differential)}.")
        manifest["reused"] = True
        return {"X": X, "Y": Y, **manifest}

    X, Y = generate_fn(n_samples)
    X, Y = np.asarray(X), np.asarray(Y)
    if X.shape[0] != n_samples or Y.shape[0] != n_samples:
        raise SealedDatasetError(
            f"generator returned {X.shape[0]} samples, expected {n_samples}.")
    digest = _content_hash(X, Y)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(dpath, X=X, Y=Y)
    manifest = {
        "sha256": digest, "n_samples": int(n_samples), "rounds": int(rounds),
        "differential": list(differential), "path": str(dpath),
        "role": "sealed_evaluation_set",
        "policy": ("generated exactly once; shared by every block and both arms; never "
                   "used for model or epoch selection"),
        "reused": False,
    }
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"X": X, "Y": Y, **manifest}


def verify_shared_invariant(block_records) -> dict:
    """
    Assert every block referenced the SAME evaluation artifact.

    Called before analysis: if blocks disagree, the paired differences are
    not comparable and the run must not be certified.
    """
    hashes = {r.get("sealed_evaluation_sha256") for r in block_records}
    if len(hashes) != 1 or None in hashes:
        raise SealedDatasetError(
            f"blocks referenced {len(hashes)} distinct evaluation artifacts {hashes}; the "
            "frozen design requires exactly one sealed set shared across all blocks.")
    return {"shared_evaluation_sha256": hashes.pop(), "n_blocks_checked": len(block_records)}
