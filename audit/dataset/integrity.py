"""
Shared, target-agnostic integrity utilities for dataset audits.

Nothing here knows about a cipher, a model or a target.  The functions are
used by the D4 controlled-perturbation engine, its adapters and its
experiment drivers:

* SHA-256 of files and bytes; atomic JSON writes; exclusive (never
  overwrite) writes; an exclusive whole-directory claim.
* A canonical digest of named array partitions that covers each array's
  role, dtype (with byte order), shape and C-order content bytes.
* A read-only, memory-conscious exact-row overlap audit across partitions,
  reporting feature-row and label conflicts separately.
* Verification that a JSON record is committed to git, unmodified, and
  matches a SHA-256 anchor supplied by a human outside the file.
* A source-identity record (git HEAD, cleanliness, per-file SHA-256).

Limitations
-----------
Hash chains and digests detect accidental corruption and substitution of a
file relative to a digest recorded elsewhere.  They are not signatures: a
party able to rewrite both an artifact and every record of its digest is
not detected.  Evidence is protected only as far as at least one digest is
held somewhere that party cannot write (the approver's anchor, a commit, a
copy of the certificate).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DIGEST_SCHEME = "sha256(role|dtype.str|shape\\n + C-order bytes, roles sorted)-v1"


class IntegrityError(RuntimeError):
    """An artifact is missing, malformed, altered, foreign or ambiguous."""


# ============================================================
# Hashing and writing
# ============================================================


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json_atomic(path: Path, payload: Mapping) -> None:
    """Replace ``path`` atomically (for mutable state such as state.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Write a new file atomically; refuse if ``path`` exists or appears."""
    path = Path(path)
    if path.exists():
        raise IntegrityError(f"Refusing to overwrite existing file {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)  # fails if the destination exists, unlike os.replace
    except FileExistsError:
        raise IntegrityError(f"Refusing to overwrite {path}, which appeared during the write.") from None
    finally:
        temporary.unlink()


def write_json_exclusive(path: Path, payload: Mapping) -> None:
    write_bytes_exclusive(
        path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    )


def read_json(path: Path, what: str) -> dict:
    """Read a JSON object or raise IntegrityError naming the artifact."""
    path = Path(path)
    if not path.is_file():
        raise IntegrityError(f"{what} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"{what} is truncated or malformed: {path} ({exc})") from None
    if not isinstance(value, dict):
        raise IntegrityError(f"{what} is not a JSON object: {path}")
    return value


def stale_staging_directories(out_dir: Path) -> list[Path]:
    """Staging directories an interrupted write_directory_exclusive left behind."""
    out_dir = Path(out_dir)
    if not out_dir.parent.is_dir():
        return []
    return sorted(out_dir.parent.glob(f".{out_dir.name}.staging_*"))


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_directory_exclusive(out_dir: Path, files: Mapping[str, bytes]) -> None:
    """Write a complete set of files into a new directory, or nothing.

    Files are written and fsynced in a hidden staging directory beside the
    target; the target is claimed with an exclusive mkdir and the staging
    directory renamed onto it.  A failure inside the process removes the
    staging directory; a hard crash can leave one behind, which is never
    read as a result and makes every later write to the same target refuse
    until a human has looked at it.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise IntegrityError(f"Refusing to write into existing directory {out_dir}.")
    stale = stale_staging_directories(out_dir)
    if stale:
        raise IntegrityError(
            f"An interrupted write left {[p.name for p in stale]} beside {out_dir}. It was never "
            "accepted as complete; it is kept for investigation. Refusing until it is resolved."
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging_", dir=str(out_dir.parent)))
    claimed = False
    try:
        for name, data in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            raise IntegrityError(f"{out_dir} appeared during the write; nothing was written.") from None
        claimed = True
        os.rename(staging, out_dir)
        claimed = False
        _fsync_directory(out_dir.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if claimed and out_dir.is_dir() and not any(out_dir.iterdir()):
            out_dir.rmdir()


# ============================================================
# Canonical partition digests
# ============================================================


def partition_digest(partitions: Mapping[str, Any], chunk_rows: int = 1 << 18) -> str:
    """SHA-256 over every named array's role, dtype, shape and bytes.

    Roles are hashed in sorted order.  ``dtype.str`` carries byte order,
    so the same values stored with a different dtype or endianness give a
    different digest.  Arrays are hashed in row chunks, so memory-mapped
    arrays are never copied whole.
    """
    digest = hashlib.sha256()
    for role in sorted(partitions):
        array = np.asarray(partitions[role])
        digest.update(f"{role}|{array.dtype.str}|{tuple(array.shape)}\n".encode())
        if array.ndim == 0:
            digest.update(np.ascontiguousarray(array).tobytes())
            continue
        for start in range(0, len(array), chunk_rows):
            digest.update(np.ascontiguousarray(array[start:start + chunk_rows]).tobytes())
    return digest.hexdigest()


def labelled_partition_digest(features: Any, labels: Any) -> str:
    """Digest of one labelled partition (features and labels together)."""
    return partition_digest({"features": features, "labels": labels})


# ============================================================
# Exact-row overlap audit
# ============================================================


def row_keys(features: Any, chunk_rows: int = 1 << 20) -> np.ndarray:
    """One sortable key per row, equal iff the rows are byte-identical.

    Binary rows (all values 0/1) are bit-packed first.  Keys of at most
    8 bytes become big-endian-ordered uint64 (left zero padded; equality
    is exact because the row width is fixed); wider keys stay as fixed
    width void scalars.  Peak memory is the key array plus one chunk.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    n, width = features.shape
    binary = features.dtype == np.uint8 and n > 0 and all(
        int(np.asarray(features[s:s + chunk_rows]).max(initial=0)) <= 1 for s in range(0, n, chunk_rows)
    )
    row_bytes = (width + 7) // 8 if binary else width * features.dtype.itemsize
    as_int = row_bytes <= 8
    keys = np.empty(n, dtype=np.uint64 if as_int else np.dtype((np.void, row_bytes)))
    for start in range(0, n, chunk_rows):
        block = np.ascontiguousarray(features[start:start + chunk_rows])
        raw = np.packbits(block, axis=1) if binary else block.view(np.uint8).reshape(len(block), -1)
        if as_int:
            padded = np.zeros((len(raw), 8), dtype=np.uint8)
            padded[:, 8 - row_bytes:] = raw
            keys[start:start + len(raw)] = padded.view(">u8").reshape(-1)
        else:
            keys[start:start + len(raw)] = np.ascontiguousarray(raw).view(keys.dtype).reshape(-1)
    return keys


def _label_signature(keys: np.ndarray, labels: np.ndarray):
    """Sorted unique keys, each with its label (or -1 if its rows disagree)."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_labels = np.asarray(labels).reshape(-1)[order].astype(np.int64)
    if len(sorted_keys) == 0:
        return sorted_keys, sorted_labels, 0, 0
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    unique = sorted_keys[starts]
    lo = np.minimum.reduceat(sorted_labels, starts)
    hi = np.maximum.reduceat(sorted_labels, starts)
    signature = np.where(lo == hi, lo, -1)
    return unique, signature, int(len(sorted_keys) - len(unique)), int(np.sum(signature == -1))


def audit_partition_overlap(partitions: Mapping[str, tuple[Any, Any]]) -> dict:
    """Count exact duplicate and shared feature rows; read-only.

    ``partitions`` maps a role (e.g. "train") to (features, labels).

    within[role]:
        duplicate_rows   rows whose feature vector occurs earlier in the
                         same partition (n - distinct);
        label_conflicts  distinct feature vectors that occur with more than
                         one label inside the partition.
    cross["a__b"] (each unordered pair, in the given role order):
        rows             rows of b whose feature vector also occurs in a;
        distinct         distinct shared feature vectors;
        label_conflicts  shared feature vectors whose labels differ between
                         a and b, or are already inconsistent in either.
    """
    roles = list(partitions)
    signatures = {}
    report: dict = {
        "method": "exact_row_equality",
        "key_scheme": "bit-packed if binary else raw bytes; uint64 when <= 8 bytes else void",
        "rows": {},
        "within": {},
        "cross": {},
    }
    raw_keys = {}
    for role in roles:
        features, labels = partitions[role]
        features = np.asarray(features)
        if len(np.asarray(labels).reshape(-1)) != len(features):
            raise ValueError(f"{role}: labels and features differ in length")
        raw_keys[role] = row_keys(features)
        signatures[role] = _label_signature(raw_keys[role], labels)
        report["rows"][role] = int(len(features))
        report["within"][role] = {
            "duplicate_rows": signatures[role][2],
            "label_conflicts": signatures[role][3],
        }
    for i, first in enumerate(roles):
        for second in roles[i + 1:]:
            unique_a, label_a = signatures[first][:2]
            probe = raw_keys[second]
            if len(unique_a) == 0 or len(probe) == 0:
                report["cross"][f"{first}__{second}"] = {"rows": 0, "distinct": 0, "label_conflicts": 0}
                continue
            if unique_a.dtype != probe.dtype:
                raise ValueError(f"{first} and {second} have incompatible row encodings")
            position = np.searchsorted(unique_a, probe)
            position[position == len(unique_a)] = 0
            hit = unique_a[position] == probe
            unique_b, label_b = signatures[second][:2]
            shared_b = np.isin(unique_b, unique_a[position[hit]])
            pos_b = np.searchsorted(unique_a, unique_b[shared_b])
            conflicts = (label_a[pos_b] != label_b[shared_b]) | (label_a[pos_b] == -1) | (label_b[shared_b] == -1)
            report["cross"][f"{first}__{second}"] = {
                "rows": int(hit.sum()),
                "distinct": int(shared_b.sum()),
                "label_conflicts": int(conflicts.sum()),
            }
    return report


# ============================================================
# Git binding and source identity
# ============================================================


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, timeout=60)


def verify_anchored_committed_record(path: Path, *, expected_sha256: str, what: str) -> dict:
    """Return the parsed record if it is anchored, tracked, committed and clean.

    Refuses unless the file's SHA-256 equals ``expected_sha256`` (supplied
    by the approver outside the file), the file is tracked by git with no
    staged or unstaged modification, and its blob at HEAD is byte-identical.
    The commit that last touched it is returned in ``_commit``.
    """
    path = Path(path).resolve()
    if not (isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_sha256)):
        raise IntegrityError(f"{what}: the anchor must be a 64-hex SHA-256 supplied by the approver.")
    if not path.is_file():
        raise IntegrityError(f"{what} is missing: {path}")
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise IntegrityError(f"{what}: SHA-256 differs from the approver's anchor.")
    top = _git(["rev-parse", "--show-toplevel"], path.parent)
    if top.returncode != 0:
        raise IntegrityError(f"{what}: not inside a git repository.")
    root = Path(top.stdout.decode().strip()).resolve()
    rel = path.relative_to(root).as_posix()
    if _git(["ls-files", "--error-unmatch", rel], root).returncode != 0:
        raise IntegrityError(f"{what} is not tracked by git.")
    if _git(["status", "--porcelain", "--", rel], root).stdout.strip():
        raise IntegrityError(f"{what} has uncommitted modifications.")
    blob = _git(["show", f"HEAD:{rel}"], root)
    if blob.returncode != 0 or blob.stdout != data:
        raise IntegrityError(f"{what}: committed content differs from the file.")
    log = _git(["log", "-1", "--format=%H%x00%cI", "--", rel], root).stdout.decode().strip().split("\x00")
    try:
        record = json.loads(data)
    except ValueError as exc:
        raise IntegrityError(f"{what} is not JSON: {exc}") from None
    if not isinstance(record, dict):
        raise IntegrityError(f"{what} is not a JSON object.")
    record["_commit"] = {"sha": log[0], "committed_utc": log[1] if len(log) > 1 else None, "path": rel}
    return record


def source_identity(paths: Iterable[Path], root: Path = REPOSITORY_ROOT) -> dict:
    """Git HEAD plus, per source file, its SHA-256 and whether it is clean at HEAD."""
    root = Path(root).resolve()
    head = _git(["rev-parse", "HEAD"], root)
    files = {}
    for path in paths:
        path = Path(path).resolve()
        rel = path.relative_to(root).as_posix()
        status = _git(["status", "--porcelain", "--", rel], root)
        files[rel] = {
            "sha256": sha256_file(path),
            "clean_at_head": status.returncode == 0 and not status.stdout.strip(),
        }
    return {"git_head": head.stdout.decode().strip() if head.returncode == 0 else None, "files": files}


# ============================================================
# Permutations
# ============================================================

PERMUTATION_INDEX_DTYPE = np.dtype("<i8")


def verify_permutation_indices(indices: Any, n: int) -> np.ndarray:
    """Return ``indices`` if it is a bijection of range(n) stored as little-endian int64.

    Checks dtype, dimensionality, length, bounds and that every index occurs
    exactly once (bincount over n, so O(n) time and 8n bytes of memory).
    """
    indices = np.asarray(indices)
    if indices.dtype != PERMUTATION_INDEX_DTYPE:
        raise IntegrityError(f"Permutation indices have dtype {indices.dtype.str}, expected "
                             f"{PERMUTATION_INDEX_DTYPE.str}.")
    if indices.ndim != 1 or len(indices) != int(n):
        raise IntegrityError(f"Permutation indices have shape {indices.shape}, expected ({int(n)},).")
    if n and (int(indices.min()) < 0 or int(indices.max()) >= n):
        raise IntegrityError("Permutation indices fall outside range(n).")
    if n and not np.all(np.bincount(indices, minlength=int(n)) == 1):
        raise IntegrityError("Permutation indices repeat or omit an index; not a bijection of range(n).")
    return indices
