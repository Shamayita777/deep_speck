"""
D0 - CANONICAL DATASET INSTANCE.

THE PROBLEM THIS SOLVES
-----------------------
Every D-dimension so far generated its own dataset instance. D1's certificate,
D3's two certificates and D4's certificate each record

    "randomness_source": "os.urandom",  "random_seed": null

and `speck.make_train_data` draws fresh entropy on every call, so D1's
zero-duplicate census, D3's PASS and D4's effect are each true of a DIFFERENT
12,000,000-row dataset. The cryptographic-evidence pillar used different draws
again. No single instance has been audited across the pillars, so the D-results
cannot currently be composed into one statement about one dataset.

This module creates ONE canonical instance, persists it, hashes it, and gives
every dimension a loader that refuses to proceed unless the bytes it loads hash
to the value recorded in the manifest. After this, a D-certificate can name the
instance it audited, and an integration engine can check that two certificates
refer to the same one.

WHAT IT DOES NOT CLAIM
----------------------
Gohr's generator is not seed-replayable (os.urandom), so the canonical instance
is reproducible by DISTRIBUTION but not by REPLAY. Reproducibility here means:
the persisted bytes are fixed and hash-bound, and anyone with the artifact
recomputes identical results. It does not mean a third party can regenerate the
same 12M rows from a seed. That limitation is recorded in the manifest rather
than hidden, and it is the same honest position `adversarial/sealed_sets.py`
takes for its sealed sets.

Binding an existing certificate retroactively is NOT possible for D1/D3/D4,
because the instances they used were never persisted. Those remain per-instance
results and must be reported as such; `instance_binding_status` records that
explicitly instead of implying a binding that does not exist.

    python -m audit.dataset.canonical build  --out audit/dataset/evidence/d0
    python -m audit.dataset.canonical verify --out audit/dataset/evidence/d0
    python -m audit.dataset.canonical show   --out audit/dataset/evidence/d0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audit.dataset.common.provenance import (
    array_representation_metadata,
    array_sha256,
    sha256_file,
)

MANIFEST_NAME = "d0_canonical_manifest.json"

#: Schema 2 adds `specification_sha256` and nothing else. The identifier
#: semantics, hash algorithm and partition ordering of schema 1 are UNCHANGED,
#: so every schema-1 instance keeps its `canonical_instance_id` and still
#: verifies. See `compute_instance_id` for why the identifier was not
#: redefined, and `validate_manifest_structure` for why the version had to be
#: bumped rather than the field simply added.
SCHEMA_VERSION = "d0-canonical-instance-2"
LEGACY_SCHEMA_VERSION = "d0-canonical-instance-1"
SUPPORTED_SCHEMA_VERSIONS = (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)

#: Manifest fields without which integrity cannot be checked at all. A
#: manifest missing any of them is refused rather than partially verified.
REQUIRED_MANIFEST_FIELDS = (
    "schema_version", "canonical_instance_id", "specification", "partitions")

#: Per-partition fields the integrity check consumes.
REQUIRED_PARTITION_FIELDS = (
    "file", "samples", "features_sha256", "labels_sha256", "file_sha256")

#: Frozen specification of the canonical instance. Changing any of these
#: produces a DIFFERENT instance; the loader refuses a manifest that does not
#: match the specification it is asked for, so a silent change cannot pass.
CANONICAL_SPEC: dict[str, Any] = {
    "dataset_id": "gohr-speck",
    "dataset_version": "original-make-train-data",
    "generator": "speck.make_train_data",
    "generation_procedure": "GohrAdapter.generate_partition() -> speck.make_train_data(n, nr)",
    "num_rounds": 5,
    "input_difference": {"left_word": "0x0040", "right_word": "0x0000"},
    "train_samples": 10_000_000,
    "validation_samples": 1_000_000,
    "test_samples": 1_000_000,
    "randomness_source": "os.urandom",
    "exact_replay_available": False,
}

PARTITIONS = ("train", "validation", "test")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Canonical identity — ONE definition, used by both build and verify
# ---------------------------------------------------------------------------
# Schema 1 computed the instance id inline inside build(), and verify() never
# recomputed it. That meant a manifest whose `canonical_instance_id` had been
# edited still reported VERIFIED, and verify() RETURNED the edited value, which
# binding_record() then copied into every dimension's certificate. The identity
# that bound all the evidence together was therefore never checked against the
# data it claimed to identify.
#
# The computation below is byte-for-byte the schema-1 computation, lifted out
# of build() unchanged so that both sides call one function. The identifier
# semantics are deliberately NOT redefined: every existing instance keeps its
# id and still verifies. See compute_specification_digest for how the
# specification is covered instead.

def _is_sha256_hex(value: Any) -> bool:
    """True only for a 64-character hexadecimal string.

    A length-only check is not enough: a 64-character non-hex value such as
    'g' * 64 is malformed, and letting it through meant verification reported
    an id MISMATCH where it should have refused a malformed manifest.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def compute_instance_id(partitions: dict) -> str:
    """The canonical instance id, from the ordered partition identities.

    Identical to the schema-1 computation: the SHA-256 of the concatenation,
    in PARTITIONS order, of each partition's ``features_sha256`` followed by
    its ``labels_sha256``.

    ORDER IS PART OF THE IDENTITY. The concatenation follows PARTITIONS
    (train, validation, test), so the same three partitions presented in a
    different role assignment produce a DIFFERENT id. That is intended: a
    dataset whose train and test halves are swapped is a different dataset.
    """
    missing = [p for p in PARTITIONS if p not in partitions]
    if missing:
        raise SystemExit(
            f"REFUSED: cannot compute a canonical instance id without "
            f"partitions {missing}. An id over a subset would not identify "
            f"this instance.")
    for p in PARTITIONS:
        for field in ("features_sha256", "labels_sha256"):
            value = partitions[p].get(field)
            if not isinstance(value, str) or not value:
                raise SystemExit(
                    f"REFUSED: partition {p!r} has no usable {field!r}; the "
                    f"instance id is defined over those digests.")
    return array_sha256(np.frombuffer(
        "".join(partitions[p]["features_sha256"] + partitions[p]["labels_sha256"]
                for p in PARTITIONS).encode(), dtype=np.uint8))


def compute_specification_digest(spec: dict) -> str:
    """SHA-256 over the canonical specification, key-order independent.

    WHY THIS IS SEPARATE FROM THE INSTANCE ID
    -----------------------------------------
    `canonical_instance_id` is defined over the DATA only, so two instances
    with identical bytes and different specifications share an id. Folding the
    specification into the id would fix that but would also change every
    existing instance's identifier and invalidate the certificates that embed
    it. So the specification is covered by its own digest, recorded alongside,
    and the identifier semantics stay as they were.

    `sort_keys` makes the digest independent of JSON key order, and
    `default=str` matches how the manifest itself is serialised, so a value
    that only survives serialisation as a string digests the same on both
    sides.
    """
    import hashlib
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_manifest_structure(man: dict) -> None:
    """Refuse a manifest that cannot be integrity-checked.

    Fails closed, with the module's existing SystemExit/REFUSED convention,
    before any digest is compared — so a malformed manifest is a clean refusal
    rather than a KeyError from somewhere deep in the check.
    """
    if not isinstance(man, dict):
        raise SystemExit(
            f"REFUSED: the canonical manifest is {type(man).__name__}, not an "
            f"object.")
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in man]
    if missing:
        raise SystemExit(
            f"REFUSED: the canonical manifest is missing required field(s) "
            f"{missing}. Integrity cannot be established from it.")

    version = man["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SystemExit(
            f"REFUSED: unsupported canonical manifest schema_version "
            f"{version!r}; this build understands "
            f"{list(SUPPORTED_SCHEMA_VERSIONS)}. Old-schema instances are not "
            f"migrated silently.")

    if not isinstance(man["specification"], dict):
        raise SystemExit("REFUSED: `specification` is not an object.")
    if not isinstance(man["partitions"], dict):
        raise SystemExit("REFUSED: `partitions` is not an object.")

    for p in PARTITIONS:
        if p not in man["partitions"]:
            raise SystemExit(
                f"REFUSED: the canonical manifest has no {p!r} partition. All "
                f"of {list(PARTITIONS)} are required.")
        rec = man["partitions"][p]
        if not isinstance(rec, dict):
            raise SystemExit(f"REFUSED: partition {p!r} is not an object.")
        absent = [f for f in REQUIRED_PARTITION_FIELDS if f not in rec]
        if absent:
            raise SystemExit(
                f"REFUSED: partition {p!r} is missing {absent}.")

    cid = man["canonical_instance_id"]
    if not _is_sha256_hex(cid):
        raise SystemExit(
            f"REFUSED: canonical_instance_id is not a 64-character SHA-256 "
            f"hex digest (got {cid!r}).")

    # The recorded partition digests are what the identifier is computed over,
    # so a malformed one is a malformed manifest, not merely a mismatch.
    for p in PARTITIONS:
        rec = man["partitions"][p]
        for field in ("features_sha256", "labels_sha256", "file_sha256"):
            if not _is_sha256_hex(rec[field]):
                raise SystemExit(
                    f"REFUSED: partition {p!r} field {field!r} is not a "
                    f"64-character SHA-256 hex digest (got {rec[field]!r}).")

    # Schema 2 records the specification digest. Its ABSENCE in a schema-2
    # manifest is refused rather than treated as "legacy", because otherwise
    # deleting one field would be enough to skip the specification check.
    if version == SCHEMA_VERSION and "specification_sha256" not in man:
        raise SystemExit(
            f"REFUSED: schema {SCHEMA_VERSION} requires "
            f"`specification_sha256`; it is absent. Removing the field is not "
            f"a way to opt out of the specification check.")


#: Representation fields `array_representation_metadata` produces, and that a
#: partition record must therefore carry for both arrays.
REQUIRED_REPRESENTATION_FIELDS = (
    "dtype", "shape", "ndim", "c_contiguous", "f_contiguous")


def describe_partition(X: np.ndarray, Y: np.ndarray, *, file_name: str,
                       file_sha256: str) -> dict:
    """The partition record, computed ENTIRELY from the arrays themselves.

    One function builds this record, and `check_partition_metadata` checks a
    stored record against a recomputation of it. Keeping construction and
    checking adjacent is what stops the two drifting into disagreement, which
    is how a field ends up recorded but never verified.

    Every value is derived from `X` and `Y`. Nothing is taken from a caller's
    description of what the arrays are supposed to be.
    """
    pos = int((Y == 1).sum())
    return {
        "file": file_name,
        "samples": int(Y.size),
        "features_sha256": array_sha256(X),
        "labels_sha256": array_sha256(Y),
        "file_sha256": file_sha256,
        "features": array_representation_metadata(X),
        "labels": array_representation_metadata(Y),
        # The established definition: positive == count of label value 1,
        # negative == everything else. Unchanged.
        "label_counts": {"positive": pos, "negative": int(Y.size - pos)},
    }


def check_partition_metadata(rec: dict, X: np.ndarray, Y: np.ndarray) -> list:
    """Compare a stored partition record against the arrays on disk.

    WHY THIS EXISTS
    ---------------
    Hash checks alone verify the BYTES but not the DESCRIPTION of the bytes.
    Every one of these edits previously passed verification, because the array
    digests were untouched and nothing compared the rest of the record with the
    data:

        samples: 200000 -> 999999
        features.shape: [200000, 64] -> [1, 1]
        features.dtype: uint8 -> float64
        label_counts: {positive: 100017, ...} -> {positive: 99999, negative: -5}
        features: (the whole representation block deleted)

    A reader of such a manifest is told the dataset has a different size, a
    different shape, a different dtype or a different class balance from the
    data it names, and verification called it VERIFIED. Returns a list of
    human-readable problems; empty means consistent.
    """
    problems: list = []

    # (7) Mandatory metadata present and structurally valid.
    for key in ("features", "labels"):
        block = rec.get(key)
        if not isinstance(block, dict):
            problems.append(f"{key!r} representation metadata is missing "
                            f"or not an object")
            continue
        for field in REQUIRED_REPRESENTATION_FIELDS:
            if field not in block:
                problems.append(f"{key!r} representation metadata has no "
                                f"{field!r}")
    if not isinstance(rec.get("samples"), int) or isinstance(
            rec.get("samples"), bool):
        problems.append("`samples` is missing or not an integer")
    counts = rec.get("label_counts")
    if not isinstance(counts, dict):
        problems.append("`label_counts` is missing or not an object")
    else:
        for side in ("positive", "negative"):
            v = counts.get(side)
            if not isinstance(v, int) or isinstance(v, bool):
                problems.append(f"`label_counts.{side}` is missing or not an "
                                f"integer")
            elif v < 0:
                problems.append(f"`label_counts.{side}` is negative ({v})")
    if problems:
        # Structure is broken; comparing values would produce noise on top.
        return problems

    actual = describe_partition(X, Y, file_name=rec.get("file", ""),
                                file_sha256=rec.get("file_sha256", ""))

    # (3) Shape and dtype agree with the arrays.
    for key in ("features", "labels"):
        for field in REQUIRED_REPRESENTATION_FIELDS:
            want, got = rec[key][field], actual[key][field]
            # shape arrives from JSON as a list; normalise before comparing.
            if field == "shape":
                want, got = list(want), list(got)
            if want != got:
                problems.append(
                    f"{key}.{field} recorded as {want!r} but the persisted "
                    f"array has {got!r}")

    # (4) Features and labels describe the same rows.
    if X.shape[0] != Y.shape[0]:
        problems.append(
            f"the persisted arrays disagree on row count: features have "
            f"{X.shape[0]} rows, labels have {Y.shape[0]}")

    # (5) The recorded sample count is the actual row count.
    if rec["samples"] != actual["samples"]:
        problems.append(
            f"`samples` recorded as {rec['samples']} but the persisted labels "
            f"have {actual['samples']} rows")
    if rec["samples"] != int(X.shape[0]):
        problems.append(
            f"`samples` recorded as {rec['samples']} but the persisted "
            f"features have {int(X.shape[0])} rows")

    # (6) Label counts match the actual labels, by the existing definition.
    for side in ("positive", "negative"):
        if rec["label_counts"][side] != actual["label_counts"][side]:
            problems.append(
                f"`label_counts.{side}` recorded as "
                f"{rec['label_counts'][side]} but the persisted labels give "
                f"{actual['label_counts'][side]}")
    total = (rec["label_counts"]["positive"] + rec["label_counts"]["negative"])
    if total != rec["samples"]:
        problems.append(
            f"`label_counts` sums to {total}, which is not the recorded "
            f"`samples` of {rec['samples']}")
    return problems


def check_partition_against_specification(name: str, X: np.ndarray,
                                          Y: np.ndarray, spec: dict) -> list:
    """Compare the persisted arrays against the DECLARED SPECIFICATION.

    `check_partition_metadata` compares the arrays with the partition RECORD.
    This compares them with the SPECIFICATION, which is a different claim and
    was previously unchecked: a manifest could declare
    `train_samples: 10_000_000` and `representation.feature_bits: 64` over a
    persisted 400-row, 8-column array, leave every partition digest and the
    partition record itself perfectly consistent, and still verify.

    Everything here reads shapes, dtypes and at most one reduction over `Y`.
    No full-size array is copied.

    Returns a list of problems; empty means the arrays conform to the spec.
    """
    problems: list = []

    # Declared per-partition sample count.
    key = f"{name}_samples"
    declared = spec.get(key)
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int):
            problems.append(
                f"the specification's {key} is {declared!r}, not an integer")
        elif int(declared) != int(Y.size):
            problems.append(
                f"the specification declares {key}={int(declared)} but the "
                f"persisted {name} partition has {int(Y.size)} rows")

    # Features and labels must describe the same rows.
    if X.ndim == 2 and Y.ndim == 1 and X.shape[0] != Y.shape[0]:
        problems.append(
            f"{name}: persisted features have {X.shape[0]} rows but labels "
            f"have {Y.shape[0]}")

    # Declared representation.
    rep = spec.get("representation")
    if isinstance(rep, dict):
        want_bits = rep.get("feature_bits")
        if want_bits is not None:
            if X.ndim != 2:
                problems.append(
                    f"{name}: the specification declares a 2-D feature "
                    f"representation of {want_bits} columns, but the persisted "
                    f"features are {X.ndim}-dimensional")
            elif int(X.shape[1]) != int(want_bits):
                problems.append(
                    f"{name}: the specification declares "
                    f"representation.feature_bits={want_bits} but the "
                    f"persisted features are {int(X.shape[1])} columns wide")
        for field, expected_ndim in (("feature_ndim", X.ndim),):
            declared_ndim = rep.get(field)
            if declared_ndim is not None and int(declared_ndim) != int(
                    expected_ndim):
                problems.append(
                    f"{name}: the specification declares {field}="
                    f"{declared_ndim} but the persisted features are "
                    f"{expected_ndim}-dimensional")

    want_dtype = spec.get("feature_dtype")
    if want_dtype is not None and str(X.dtype) != str(want_dtype):
        problems.append(
            f"{name}: the specification declares feature_dtype="
            f"{want_dtype!r} but the persisted features are {X.dtype}")
    want_label_dtype = spec.get("label_dtype")
    if want_label_dtype is not None and str(Y.dtype) != str(want_label_dtype):
        problems.append(
            f"{name}: the specification declares label_dtype="
            f"{want_label_dtype!r} but the persisted labels are {Y.dtype}")

    # Declared label alphabet.
    label_values = spec.get("label_values")
    if label_values is not None:
        try:
            declared_set = set(int(v) for v in label_values)
        except (TypeError, ValueError):
            problems.append(
                f"the specification's label_values is {label_values!r}, which "
                f"is not a list of integers")
        else:
            actual = set(int(v) for v in np.unique(Y).tolist())
            stray = sorted(actual - declared_set)
            if stray:
                problems.append(
                    f"{name}: the persisted labels contain {stray}, outside "
                    f"the specification's declared label_values "
                    f"{sorted(declared_set)}")
    return problems


def validate_generated_partition(name: str, X: np.ndarray, Y: np.ndarray, *,
                                 requested: int, spec: dict) -> list:
    """Check freshly generated arrays against the declared specification.

    Runs BEFORE persistence, so a non-conforming partition is never written.
    Operates on the arrays already in memory and takes no copies: the checks
    are shape/dtype reads and two reductions over `Y`, so nothing here
    duplicates a full-size array or re-hashes it.

    `representation` in the spec is optional; when a target declares one, the
    generated arrays must match it. A target that declares none is checked for
    internal consistency only.
    """
    problems: list = []

    if X.ndim != 2:
        problems.append(f"features are {X.ndim}-dimensional, expected 2")
    if Y.ndim != 1:
        problems.append(f"labels are {Y.ndim}-dimensional, expected 1")
    if problems:
        return problems

    if X.shape[0] != Y.shape[0]:
        problems.append(
            f"features have {X.shape[0]} rows but labels have {Y.shape[0]}")
    if int(Y.size) != int(requested):
        problems.append(
            f"{int(requested)} samples were requested but the generator "
            f"returned {int(Y.size)}")

    rep = spec.get("representation")
    if isinstance(rep, dict):
        want_bits = rep.get("feature_bits")
        if want_bits is not None and int(X.shape[1]) != int(want_bits):
            problems.append(
                f"the declared representation has feature_bits={want_bits} "
                f"but the generated features are {X.shape[1]} columns wide")

    want_dtype = spec.get("feature_dtype")
    if want_dtype is not None and str(X.dtype) != str(want_dtype):
        problems.append(
            f"features are {X.dtype}, but the specification declares "
            f"{want_dtype}")

    label_values = spec.get("label_values")
    if label_values is not None:
        actual_values = set(int(v) for v in np.unique(Y).tolist())
        declared = set(int(v) for v in label_values)
        if not actual_values <= declared:
            problems.append(
                f"labels contain values {sorted(actual_values - declared)} "
                f"outside the declared label_values {sorted(declared)}")

    if Y.size and int(np.unique(Y).size) < 2:
        problems.append(
            f"the generated labels are a single class "
            f"({int(np.unique(Y)[0])}); every downstream class statistic over "
            f"this partition would be meaningless")
    return problems


def check_manifest_identity(man: dict) -> dict:
    """Recompute the identity fields and compare them with what is recorded.

    Pure manifest arithmetic: reads no partition files, so it is cheap enough
    for every consumer of a manifest to call.
    """
    validate_manifest_structure(man)
    recomputed_id = compute_instance_id(man["partitions"])
    recorded_id = man["canonical_instance_id"]
    out = {
        "canonical_instance_id_recorded": recorded_id,
        "canonical_instance_id_recomputed": recomputed_id,
        "canonical_instance_id_consistent": bool(recomputed_id == recorded_id),
    }
    if man["schema_version"] == SCHEMA_VERSION:
        recomputed_spec = compute_specification_digest(man["specification"])
        recorded_spec = man["specification_sha256"]
        out["specification_sha256_recorded"] = recorded_spec
        out["specification_sha256_recomputed"] = recomputed_spec
        out["specification_consistent"] = bool(recomputed_spec == recorded_spec)
    else:
        out["specification_sha256_recorded"] = None
        out["specification_sha256_recomputed"] = None
        out["specification_consistent"] = None
        out["specification_note"] = (
            f"schema {man['schema_version']} predates `specification_sha256`; "
            f"the specification is NOT covered by a digest for this instance.")
    out["consistent"] = (out["canonical_instance_id_consistent"]
                         and out["specification_consistent"] is not False)
    return out


def _git(repo_root: Path) -> dict:
    """Git state. A failed call is never reported as a clean tree."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                              capture_output=True, text=True, timeout=20)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                                capture_output=True, text=True, timeout=20)
        if head.returncode != 0 or status.returncode != 0:
            return {"commit": "UNAVAILABLE", "working_tree": "UNAVAILABLE",
                    "repo_root": str(repo_root)}
        return {"commit": head.stdout.strip() or "UNAVAILABLE",
                "working_tree": "dirty" if status.stdout.strip() else "clean",
                "repo_root": str(repo_root)}
    except Exception:                                           # noqa: BLE001
        return {"commit": "UNAVAILABLE", "working_tree": "UNAVAILABLE",
                "repo_root": str(repo_root)}


def _atomic_savez(path: Path, **arrays) -> None:
    """Write a .npz atomically, leaving no temporary file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        written = tmp
        if os.path.exists(tmp + ".npz"):          # numpy appends .npz
            written = tmp + ".npz"
            os.unlink(tmp)
        os.replace(written, path)
    except BaseException:
        for leftover in (tmp, tmp + ".npz"):
            if os.path.exists(leftover):
                os.unlink(leftover)
        raise


#: Per-target history of which dataset dimensions previously audited their own
#: unpersisted instance. Baked into schema 1 as Gohr-specific prose; made
#: target-aware here because the same literal was factually WRONG in an AutoND
#: manifest, which has no prior D-stage runs to be unbound from.
PRIOR_UNBOUND_DIMENSIONS = {
    "gohr-speck": ("D1", "D2", "D3", "D4"),
}


def _binding_status(spec: dict) -> dict:
    """Which dimensions are bound to this instance, and why not.

    A fresh target has simply NOT_RUN every dimension. Only a target with
    pre-existing certificates over unpersisted data has dimensions that are
    NOT_BOUND for the stronger reason that they audited something else.
    """
    target = str(spec.get("target", DEFAULT_TARGET))
    prior = set(PRIOR_UNBOUND_DIMENSIONS.get(target, ()))
    status: dict = {}
    for dim in ("D1", "D2", "D3", "D4", "D5"):
        if dim in prior:
            status[dim] = (f"NOT_BOUND - {dim} audited its own unpersisted "
                           f"instance")
        else:
            status[dim] = "NOT_RUN"
    status["note"] = (
        "A dimension becomes bound only by being re-run against this instance, "
        "recording canonical_instance_id in its certificate. Where a dimension "
        "is NOT_BOUND rather than NOT_RUN, retroactive binding is impossible: "
        "that instance was generated from an unseeded source and never "
        "persisted.")
    return status


#: Default target. A spec with no "target" key resolves to Gohr, so every
#: existing caller, manifest and evidence artifact behaves exactly as before.
DEFAULT_TARGET = "gohr-speck"


def _resolve_gohr(spec: dict):
    """Gohr/Speck generator. Behaviour unchanged from schema 1.

    GohrAdapter.generate_partition() is a thin wrapper that calls
    speck.make_train_data(n, nr) and nothing else, but importing the adapter
    pulls in keras and train_nets. D0 needs neither, and an artifact evaluator
    should be able to rebuild and verify the canonical instance on a plain CPU
    box with numpy alone. So the adapter is preferred when it imports cleanly,
    and speck.make_train_data is called directly when it does not. Both paths
    invoke the same generator with the same arguments; the manifest records
    which one was used.
    """
    num_rounds = int(spec["num_rounds"])
    try:
        from audit.dataset.adapters.gohr import GohrAdapter
        adapter = GohrAdapter(num_rounds=num_rounds)
        return (lambda n: adapter.generate_partition(n),
                "audit.dataset.adapters.gohr.GohrAdapter.generate_partition")
    except Exception as exc:                                    # noqa: BLE001
        try:
            import speck as sp
        except Exception:                                       # noqa: BLE001
            raise SystemExit(
                "REFUSED: neither GohrAdapter nor speck could be imported, so the "
                f"canonical instance cannot be generated. Adapter error: {exc}")
        return (lambda n: sp.make_train_data(int(n), int(num_rounds)),
                "speck.make_train_data (direct; adapter unavailable in this "
                f"environment: {type(exc).__name__})")


def _resolve_autond(spec: dict):
    """AutoND generator, via the AutoND dataset adapter.

    AutoND's generator is parameterised by the input difference, so the spec
    must name the cipher, the difference and the round count. There is no
    fallback path: unlike Speck there is no module-level generator to call
    directly, and guessing one would not be the audited subject.
    """
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter

    required = ("autond_root", "cipher", "delta_plain_hex", "num_rounds")
    missing = [k for k in required if spec.get(k) in (None, "")]
    if missing:
        raise SystemExit(
            f"REFUSED: the autond target needs {missing} in the spec. An "
            f"AutoND dataset instance is identified by (cipher, scenario, "
            f"delta_plain, delta_key, num_rounds); omitting any of them would "
            f"let two different instances share one provenance record.")

    adapter = AutoNDDatasetAdapter(
        autond_root=spec["autond_root"],
        cipher_name=spec["cipher"],
        delta_plain_int=int(str(spec["delta_plain_hex"]), 16),
        delta_key_int=int(str(spec.get("delta_key_hex", "0x0")), 16),
        num_rounds=int(spec["num_rounds"]),
        scenario=spec.get("scenario", "single-key"),
    )
    return (lambda n: adapter.generate_partition(n),
            "audit.dataset.adapters.autond.AutoNDDatasetAdapter."
            "generate_partition")


#: Target registry. A new audited target registers one resolver here and needs
#: no change to build(), load_partition(), verify() or the manifest schema,
#: all of which already take the spec as data.
GENERATOR_RESOLVERS = {
    "gohr-speck": _resolve_gohr,
    "autond": _resolve_autond,
}


def _resolve_generator(spec: dict):
    """
    Return (callable, description) producing one partition, for the spec's
    target.

    WHY THIS IS SPEC-DRIVEN (minimal generic change, 2026-10-19)
    ------------------------------------------------------------
    Schema 1 hardcoded the Gohr/Speck generator: the signature was
    `_resolve_generator(num_rounds)` and the body tried GohrAdapter then
    `speck.make_train_data`. Every other part of D0 — build(), load_manifest(),
    load_partition(), binding_record(), verify() and the manifest itself —
    already treats the spec as data and accepts a caller-supplied override, so
    this one function was the sole obstacle to auditing a second target.

    The change is a registry keyed by the spec's "target", defaulting to
    DEFAULT_TARGET. A spec without a "target" key takes the Gohr path with the
    identical adapter-then-direct fallback, so existing behaviour, manifests
    and evidence are unaffected.
    """
    target = str(spec.get("target", DEFAULT_TARGET))
    resolver = GENERATOR_RESOLVERS.get(target)
    if resolver is None:
        raise SystemExit(
            f"REFUSED: unknown target {target!r}. Registered targets: "
            f"{sorted(GENERATOR_RESOLVERS)}. A target must register a resolver "
            f"rather than be inferred, so an unrecognised spec cannot silently "
            f"generate the wrong dataset.")
    return resolver(spec)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(out_dir: Path, *, repo_root: Path, spec: dict | None = None,
          log=print) -> dict:
    """
    Generate the canonical instance once and persist it with a manifest.

    Refuses to overwrite an existing manifest: a canonical instance that can be
    silently replaced is not canonical.
    """
    spec = dict(spec or CANONICAL_SPEC)
    out_dir = Path(out_dir)
    man_path = out_dir / MANIFEST_NAME
    if man_path.exists():
        raise SystemExit(
            f"REFUSED: {man_path} already exists. The canonical instance is built "
            "once and never silently replaced. To create a different instance, use "
            "a new directory; to inspect this one, use `verify` or `show`.")

    generate, generator_path = _resolve_generator(spec)
    log(f"generator: {generator_path}")

    partitions: dict[str, Any] = {}
    for name in PARTITIONS:
        n = int(spec[f"{name}_samples"])
        log(f"generating {name}: {n:,} samples ...")
        X, Y = generate(n)
        X = np.ascontiguousarray(X)
        Y = np.ascontiguousarray(Y)
        # Validate BEFORE persisting. Writing a canonical instance that does
        # not match its own specification creates an artifact that `verify`
        # will later reject, after the generation cost has been paid; worse, a
        # requested/actual size mismatch would otherwise be silently recorded
        # as whatever was produced.
        problems = validate_generated_partition(
            name, X, Y, requested=n, spec=spec)
        if problems:
            raise SystemExit(
                f"REFUSED: generated {name!r} partition does not conform to "
                f"the declared specification, so it was NOT persisted:\n  - "
                + "\n  - ".join(problems))

        path = out_dir / f"{name}.npz"
        _atomic_savez(path, X=X, Y=Y)
        # Built by the same function `check_partition_metadata` checks against,
        # and computed from the arrays rather than from the caller's
        # description of them.
        partitions[name] = describe_partition(
            X, Y, file_name=path.name, file_sha256=sha256_file(path))
        log(f"  {name}: features {partitions[name]['features_sha256'][:16]} "
            f"labels {partitions[name]['labels_sha256'][:16]}")
        del X, Y

    # Same function verify() uses, so the two can never drift apart.
    instance_id = compute_instance_id(partitions)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "specification_sha256": compute_specification_digest(spec),
        "audit": {"id": "D0", "name": "Canonical Dataset Instance",
                  "claim": "A single persisted dataset instance that every "
                           "dataset-integrity dimension can be bound to by hash."},
        "canonical_instance_id": instance_id,
        "specification": spec,
        "generator_path": generator_path,
        "partitions": partitions,
        "instance_binding_status": _binding_status(spec),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "git": _git(repo_root),
        "created_utc": _utc(),
        "limitations": [
            f"Reproducible by persisted bytes, not by replay: "
            f"{spec.get('generator', 'the generator')} draws from "
            f"{spec.get('randomness_source', 'an unseeded source')}, so this "
            f"instance cannot be regenerated from a seed. Reproducibility rests "
            f"on distributing and hashing the arrays.",
            "This manifest establishes instance identity only. It makes no claim "
            "about duplicates (D1), dependence (D2), distribution (D3), "
            "perturbation response (D4) or scaling (D5).",
            "Binding is forward-looking: a dimension is bound only by being "
            "re-run against this instance. Certificates produced before this "
            "instance existed audited different data and are not retroactively "
            "bound.",
        ],
    }
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log(f"canonical instance id: {instance_id}")
    log(f"manifest: {man_path}")
    return manifest


# ---------------------------------------------------------------------------
# load / verify
# ---------------------------------------------------------------------------

def load_manifest(out_dir: Path) -> dict:
    man_path = Path(out_dir) / MANIFEST_NAME
    if not man_path.exists():
        raise SystemExit(
            f"REFUSED: no canonical manifest at {man_path}. Build it first with "
            "`python -m audit.dataset.canonical build`.")
    return json.loads(man_path.read_text())


def load_partition(out_dir: Path, name: str, *, spec: dict | None = None,
                   verify_file_hash: bool = False) -> tuple:
    """
    Load one partition of the canonical instance, verifying it against the
    manifest. Any mismatch is a hard failure: a dimension must never audit
    bytes that differ from the ones the manifest names.
    """
    if name not in PARTITIONS:
        raise ValueError(f"unknown partition {name!r}; expected one of {PARTITIONS}")
    out_dir = Path(out_dir)
    man = load_manifest(out_dir)

    spec = dict(spec or CANONICAL_SPEC)
    mismatch = {k: {"requested": v, "manifest": man["specification"].get(k)}
                for k, v in spec.items() if man["specification"].get(k) != v}
    if mismatch:
        raise SystemExit(
            f"REFUSED: the canonical instance does not match the requested "
            f"specification: {mismatch}. It is canonical; it is not regenerated "
            "and not reinterpreted.")

    rec = man["partitions"][name]
    path = out_dir / rec["file"]
    if not path.exists():
        raise SystemExit(
            f"REFUSED: {path} is missing. The canonical instance is never "
            "regenerated on demand; restore the artifact.")
    if verify_file_hash and sha256_file(path) != rec["file_sha256"]:
        raise SystemExit(f"REFUSED: {path} file hash differs from the manifest.")
    with np.load(path) as z:
        X, Y = np.ascontiguousarray(z["X"]), np.ascontiguousarray(z["Y"])
    if array_sha256(X) != rec["features_sha256"]:
        raise SystemExit(f"REFUSED: {name} features hash differs from the manifest.")
    if array_sha256(Y) != rec["labels_sha256"]:
        raise SystemExit(f"REFUSED: {name} labels hash differs from the manifest.")
    return X, Y


def binding_record(out_dir: Path) -> dict:
    """
    The block a dimension's certificate embeds to declare which instance it
    audited. Presence of this block is what lets an integration engine check
    that two certificates describe the same data.
    """
    man = load_manifest(out_dir)
    # This is the propagation path: whatever id this returns is copied into a
    # dimension's certificate as the thing it audited. So the identity is
    # recomputed here too, and an inconsistent manifest is refused rather than
    # quietly exported as evidence. Manifest arithmetic only; no file reads.
    identity = check_manifest_identity(man)
    if not identity["canonical_instance_id_consistent"]:
        raise SystemExit(
            f"REFUSED: the canonical manifest at {Path(out_dir) / MANIFEST_NAME} "
            f"records canonical_instance_id "
            f"{identity['canonical_instance_id_recorded']} but its partition "
            f"digests give {identity['canonical_instance_id_recomputed']}. A "
            f"certificate must not carry an identity the manifest contradicts.")
    if identity["specification_consistent"] is False:
        raise SystemExit(
            f"REFUSED: the canonical manifest's specification_sha256 does not "
            f"match its recorded specification. The instance identity cannot "
            f"be exported into a certificate until that is reconciled.")
    return {
        "canonical_instance_id": identity["canonical_instance_id_recomputed"],
        "manifest_schema_version": man["schema_version"],
        "specification_sha256": identity["specification_sha256_recorded"],
        "identity_recomputed": True,
        "manifest_path": str(Path(out_dir) / MANIFEST_NAME),
        "partition_hashes": {p: {"features_sha256": man["partitions"][p]["features_sha256"],
                                 "labels_sha256": man["partitions"][p]["labels_sha256"]}
                             for p in PARTITIONS},
        "binding_semantics": "this certificate's findings describe exactly the "
                             "dataset instance identified above",
    }


def verify(out_dir: Path, *, log=print) -> dict:
    """
    Re-hash every persisted partition against the manifest.

    Verification asks "are these the bytes the manifest names", so it uses the
    manifest's OWN recorded specification. Checking against the module default
    instead would make `verify` fail for any instance built under a different
    specification, which is a question about configuration, not integrity.
    """
    man = load_manifest(out_dir)

    # Structure first: a malformed manifest is refused before any digest is
    # compared, so the failure names the missing field instead of surfacing as
    # a KeyError from inside a hash check.
    validate_manifest_structure(man)

    spec = man["specification"]
    results = {}
    ok = True
    for name in PARTITIONS:
        try:
            # verify_file_hash=True re-reads the .npz and re-hashes the file,
            # and load_partition then re-hashes the decoded arrays, so both the
            # stored file and the decoded bytes are compared with the manifest.
            X, Y = load_partition(out_dir, name, spec=spec, verify_file_hash=True)
            # Hashes prove the bytes; this proves the manifest's DESCRIPTION of
            # those bytes. Both are required: an edit to `samples`, a shape, a
            # dtype or a label count leaves every digest intact.
            # Two independent comparisons: the arrays against the partition
            # RECORD, and the arrays against the declared SPECIFICATION. A
            # manifest can be internally consistent and still contradict the
            # specification it claims to implement, so both are required.
            problems = check_partition_metadata(man["partitions"][name], X, Y)
            problems += check_partition_against_specification(name, X, Y, spec)
            if problems:
                ok = False
                results[name] = {"verified": False,
                                 "reason": "manifest metadata disagrees with "
                                           "the persisted arrays",
                                 "metadata_problems": problems}
                log(f"  {name:11s} FAILED    metadata: {problems[0]}")
                for extra in problems[1:]:
                    log(f"  {'':11s}           metadata: {extra}")
            else:
                results[name] = {"verified": True, "samples": int(Y.size)}
                log(f"  {name:11s} VERIFIED  n={Y.size:,}")
            del X, Y
        except SystemExit as exc:
            ok = False
            results[name] = {"verified": False, "reason": str(exc)}
            log(f"  {name:11s} FAILED    {exc}")

    # Recompute the identity. Without this, editing `canonical_instance_id`
    # left verification reporting VERIFIED and handed the edited value back to
    # every caller.
    identity = check_manifest_identity(man)
    if not identity["canonical_instance_id_consistent"]:
        ok = False
        log(f"  identity    FAILED    canonical_instance_id recorded as "
            f"{identity['canonical_instance_id_recorded'][:16]} but the "
            f"partition digests give "
            f"{identity['canonical_instance_id_recomputed'][:16]}")
    if identity["specification_consistent"] is False:
        ok = False
        log(f"  spec digest FAILED    specification_sha256 recorded as "
            f"{identity['specification_sha256_recorded'][:16]} but the "
            f"recorded specification digests to "
            f"{identity['specification_sha256_recomputed'][:16]}")
    elif identity["specification_consistent"] is None:
        log(f"  spec digest NOT_RECORDED  (schema {man['schema_version']})")

    log(f"canonical instance {identity['canonical_instance_id_recomputed'][:16]}: "
        f"{'VERIFIED' if ok else 'FAILED'}")
    return {
        # The RECOMPUTED id, never the recorded one: a caller must not receive
        # an identifier that this function has just shown to be inconsistent.
        "canonical_instance_id": identity["canonical_instance_id_recomputed"],
        "schema_version": man["schema_version"],
        "verified": ok,
        "ok": ok,
        "identity": identity,
        "partitions": results,
        "checked_utc": _utc(),
        "verification_scope": (
            "Re-hashes every persisted partition and recomputes the instance "
            "identity from the manifest's own recorded digests. This detects "
            "corruption and piecemeal edits. It is NOT independent "
            "authentication: an attacker able to rewrite the data AND the "
            "whole manifest can produce a self-consistent instance that "
            "verifies. Detecting that requires anchoring this id to a digest "
            "or signature held outside the instance directory."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D0 canonical dataset instance")
    ap.add_argument("action", choices=("build", "verify", "show"))
    ap.add_argument("--out", type=Path, default=Path("audit/dataset/evidence/d0"))
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    a = ap.parse_args(argv)

    if a.action == "build":
        build(a.out, repo_root=a.repo_root)
    elif a.action == "verify":
        rep = verify(a.out)
        return 0 if rep["verified"] else 1
    else:
        man = load_manifest(a.out)
        print(json.dumps({k: v for k, v in man.items() if k != "partitions"},
                         indent=2, sort_keys=True))
        print(json.dumps(binding_record(a.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
