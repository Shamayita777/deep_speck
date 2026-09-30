"""
Per-block dataset persistence for CE1 (EV-adapted).

CE1 training data are generated with os.urandom and are NOT replayable
from a nominal seed, so resuming an arm requires the ACTUAL arrays it was
trained on. Following EV's `persist_dataset`, files are named by their
CONTENT HASH, which makes an accidental overwrite with different content
impossible, and reload re-verifies identity before use.

Scope: engineering only. Nothing here alters CE1's frozen design; the two
arms of a block continue to share one training-input set and one sealed
evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


class DatasetStateError(RuntimeError):
    pass


def array_hash(*arrays) -> str:
    """
    Content hash over (shape, dtype, bytes) of each array.

    Digest is byte-for-byte identical to the earlier `a.tobytes()` form (and
    to sealed_dataset._content_hash); the buffer is hashed in place so a
    10^7 x 64 training matrix is not duplicated in memory.
    """
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        if a.size:
            h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" unless the name already ends in it, so the
    # temp file is named accordingly and the placeholder is removed first -
    # otherwise the empty placeholder, not the data, gets renamed into place.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    Path(tmp).unlink(missing_ok=True)
    try:
        np.savez(tmp, **arrays)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def persist_block_dataset(directory, *, block_id: str, role: str, X, Y,
                          rounds: int, differential) -> dict:
    """
    Persist one partition under its content hash and return its manifest.

    Re-persisting identical content is a no-op; persisting DIFFERENT
    content for the same block/role fails closed rather than overwriting.
    """
    directory = Path(directory)
    X, Y = np.asarray(X), np.asarray(Y)
    digest = array_hash(X, Y)
    data_path = directory / f"{block_id}_{role}_{digest[:16]}.npz"
    manifest_path = directory / f"{block_id}_{role}_manifest.json"

    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["sha256"] != digest:
            raise DatasetStateError(
                f"{block_id}/{role}: a dataset with hash {prior['sha256']} is already "
                f"persisted, but different content (hash {digest}) was supplied. Refusing "
                "to overwrite - the arms of a block must train on ONE dataset.")
        return prior

    if not data_path.exists():
        _atomic_savez(data_path, X=X, Y=Y)
    manifest = {
        "block_id": block_id, "role": role, "sha256": digest,
        "filename": data_path.name, "shape_X": list(X.shape), "shape_Y": list(Y.shape),
        "dtype_X": str(X.dtype), "dtype_Y": str(Y.dtype),
        "rounds": int(rounds), "differential": list(differential),
        "provenance": "generated with os.urandom; NOT seed-replayable - this artifact is "
                      "authoritative for any resume",
    }
    fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".json.tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, manifest_path)
    _fsync_dir(directory)
    return manifest


class PathContainmentError(DatasetStateError):
    """A recorded path is absolute, traverses upward, or resolves outside its root."""


def contained_path(root, recorded) -> Path:
    """
    Resolve `recorded` strictly INSIDE `root`, failing closed on:
      * absolute paths (POSIX or Windows-style),
      * any '..' component (even one that would resolve back inside),
      * a resolved target (symlinks followed) outside the resolved root.
    Validation happens on every LOAD, not only when the path was stored.
    """
    root = Path(root)
    if recorded is None or str(recorded) == "":
        raise PathContainmentError("empty path")
    rec = str(recorded)
    if Path(rec).is_absolute() or rec.startswith(("/", "\\")) or (len(rec) > 1 and rec[1] == ":"):
        raise PathContainmentError(f"absolute path refused: {rec!r}")
    parts = Path(rec.replace("\\", "/")).parts
    if ".." in parts:
        raise PathContainmentError(f"parent traversal refused: {rec!r}")
    root_r = root.resolve()
    target = (root / rec).resolve()
    if target != root_r and root_r not in target.parents:
        raise PathContainmentError(
            f"{rec!r} resolves to {target}, outside {root_r} (symlink escape?)")
    return root / rec


def _resolve_data_path(directory: Path, manifest: dict) -> Path:
    """
    Manifests store a bare RELATIVE filename so a run directory can move between
    hosts. The name must be a single component inside `directory`; anything else
    (absolute, traversal, nested, symlink escape, legacy absolute `path`) fails closed.
    """
    name = manifest.get("filename")
    if not name:
        raise PathContainmentError(
            "manifest has no relative 'filename' (absolute 'path' manifests are refused)")
    if len(Path(name).parts) != 1:
        raise PathContainmentError(f"dataset filename must be a single component: {name!r}")
    return contained_path(directory, name)


def load_block_dataset(directory, *, block_id: str, role: str,
                       expected_sha256: str | None = None,
                       rounds: int | None = None, differential=None):
    """Reload a persisted partition and RE-VERIFY it before use."""
    directory = Path(directory)
    manifest_path = directory / f"{block_id}_{role}_manifest.json"
    if not manifest_path.exists():
        raise DatasetStateError(
            f"{block_id}/{role}: no persisted dataset. The training data are not "
            "seed-replayable, so this arm cannot be resumed and must be re-run.")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetStateError(f"{block_id}/{role}: manifest unreadable: {exc}") from exc
    path = _resolve_data_path(directory, manifest)
    if not path.exists():
        raise DatasetStateError(f"{block_id}/{role}: dataset file missing: {path}")
    with np.load(path) as z:
        X, Y = z["X"], z["Y"]
    if list(X.shape) != manifest["shape_X"] or list(Y.shape) != manifest["shape_Y"] \
            or str(X.dtype) != manifest["dtype_X"] or str(Y.dtype) != manifest["dtype_Y"]:
        raise DatasetStateError(
            f"{block_id}/{role}: shape/dtype {X.shape}/{X.dtype}, {Y.shape}/{Y.dtype} do not "
            f"match the manifest")
    actual = array_hash(X, Y)
    if actual != manifest["sha256"]:
        raise DatasetStateError(
            f"{block_id}/{role}: dataset content hash {actual} != recorded "
            f"{manifest['sha256']}; the artifact has been modified.")
    if expected_sha256 is not None and actual != expected_sha256:
        raise DatasetStateError(
            f"{block_id}/{role}: dataset hash {actual} != the hash recorded in the arm "
            f"state ({expected_sha256}); refusing to resume on different data.")
    if rounds is not None and manifest["rounds"] != rounds:
        raise DatasetStateError(f"{block_id}/{role}: rounds {manifest['rounds']} != {rounds}")
    if differential is not None and tuple(manifest["differential"]) != tuple(differential):
        raise DatasetStateError(f"{block_id}/{role}: differential mismatch")
    return X, Y, manifest


# ---------------------------------------------------------------------------
# Label-only partition (the destroyed arm's permuted training labels).
#
# The destroyed arm shares X_train with the baseline arm BY CONSTRUCTION, so
# persisting X twice (640 MB per block) would only create a second copy that
# could drift. Only the permuted label vector is stored; the arm's training
# pair is (the ONE persisted X_train, these labels).
# ---------------------------------------------------------------------------

def persist_block_labels(directory, *, block_id: str, role: str, Y,
                         rounds: int, differential) -> dict:
    directory = Path(directory)
    Y = np.asarray(Y)
    digest = array_hash(Y)
    data_path = directory / f"{block_id}_{role}_{digest[:16]}.npz"
    manifest_path = directory / f"{block_id}_{role}_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["sha256"] != digest:
            raise DatasetStateError(
                f"{block_id}/{role}: labels with hash {prior['sha256']} already persisted; "
                f"refusing to overwrite with {digest}.")
        return prior
    if not data_path.exists():
        _atomic_savez(data_path, Y=Y)
    manifest = {"block_id": block_id, "role": role, "sha256": digest,
                "filename": data_path.name, "shape_Y": list(Y.shape), "dtype_Y": str(Y.dtype),
                "rounds": int(rounds), "differential": list(differential)}
    fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".json.tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, manifest_path)
    _fsync_dir(directory)
    return manifest


def load_block_labels(directory, *, block_id: str, role: str,
                      expected_sha256: str | None = None):
    directory = Path(directory)
    manifest_path = directory / f"{block_id}_{role}_manifest.json"
    if not manifest_path.exists():
        raise DatasetStateError(f"{block_id}/{role}: no persisted labels")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetStateError(f"{block_id}/{role}: manifest unreadable: {exc}") from exc
    path = _resolve_data_path(directory, manifest)
    if not path.exists():
        raise DatasetStateError(f"{block_id}/{role}: label file missing: {path}")
    with np.load(path) as z:
        Y = z["Y"]
    if list(Y.shape) != manifest["shape_Y"] or str(Y.dtype) != manifest["dtype_Y"]:
        raise DatasetStateError(f"{block_id}/{role}: shape/dtype mismatch")
    actual = array_hash(Y)
    if actual != manifest["sha256"] or (expected_sha256 and actual != expected_sha256):
        raise DatasetStateError(
            f"{block_id}/{role}: label hash {actual} != recorded "
            f"{expected_sha256 or manifest['sha256']}")
    return Y, manifest
