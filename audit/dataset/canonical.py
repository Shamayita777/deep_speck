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
SCHEMA_VERSION = "d0-canonical-instance-1"

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


def _resolve_generator(num_rounds: int):
    """
    Return (callable, description) producing one partition.

    GohrAdapter.generate_partition() is a thin wrapper that calls
    speck.make_train_data(n, nr) and nothing else, but importing the adapter
    pulls in keras and train_nets. D0 needs neither, and an artifact evaluator
    should be able to rebuild and verify the canonical instance on a plain CPU
    box with numpy alone. So the adapter is preferred when it imports cleanly,
    and speck.make_train_data is called directly when it does not. Both paths
    invoke the same generator with the same arguments; the manifest records
    which one was used.
    """
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

    generate, generator_path = _resolve_generator(spec["num_rounds"])
    log(f"generator: {generator_path}")

    partitions: dict[str, Any] = {}
    for name in PARTITIONS:
        n = int(spec[f"{name}_samples"])
        log(f"generating {name}: {n:,} samples ...")
        X, Y = generate(n)
        X = np.ascontiguousarray(X)
        Y = np.ascontiguousarray(Y)
        path = out_dir / f"{name}.npz"
        _atomic_savez(path, X=X, Y=Y)
        pos = int((Y == 1).sum())
        partitions[name] = {
            "file": path.name,
            "samples": int(Y.size),
            "features_sha256": array_sha256(X),
            "labels_sha256": array_sha256(Y),
            "file_sha256": sha256_file(path),
            "features": array_representation_metadata(X),
            "labels": array_representation_metadata(Y),
            "label_counts": {"positive": pos, "negative": int(Y.size - pos)},
        }
        log(f"  {name}: features {partitions[name]['features_sha256'][:16]} "
            f"labels {partitions[name]['labels_sha256'][:16]}")
        del X, Y

    instance_id = array_sha256(np.frombuffer(
        "".join(partitions[p]["features_sha256"] + partitions[p]["labels_sha256"]
                for p in PARTITIONS).encode(), dtype=np.uint8))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "audit": {"id": "D0", "name": "Canonical Dataset Instance",
                  "claim": "A single persisted dataset instance that every "
                           "dataset-integrity dimension can be bound to by hash."},
        "canonical_instance_id": instance_id,
        "specification": spec,
        "generator_path": generator_path,
        "partitions": partitions,
        "instance_binding_status": {
            "D1": "NOT_BOUND - D1 audited its own unpersisted instance",
            "D2": "NOT_BOUND - D2 audited its own unpersisted instance",
            "D3": "NOT_BOUND - D3 audited its own unpersisted instance",
            "D4": "NOT_BOUND - D4 audited its own unpersisted instance",
            "D5": "NOT_RUN",
            "note": "Retroactive binding is impossible: those instances were "
                    "generated with os.urandom and never persisted. A dimension "
                    "becomes bound only by being re-run against this instance, "
                    "which records canonical_instance_id in its certificate.",
        },
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "git": _git(repo_root),
        "created_utc": _utc(),
        "limitations": [
            "Reproducible by persisted bytes, not by replay: speck.make_train_data "
            "draws from os.urandom, so this instance cannot be regenerated from a "
            "seed. Reproducibility rests on distributing and hashing the arrays.",
            "This manifest establishes instance identity only. It makes no claim "
            "about duplicates (D1), dependence (D2), distribution (D3), "
            "perturbation response (D4) or scaling (D5).",
            "Binding is forward-looking: existing D1-D4 certificates audited "
            "different instances and are not retroactively bound.",
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
    return {
        "canonical_instance_id": man["canonical_instance_id"],
        "manifest_schema_version": man["schema_version"],
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
    spec = man["specification"]
    results = {}
    ok = True
    for name in PARTITIONS:
        try:
            X, Y = load_partition(out_dir, name, spec=spec, verify_file_hash=True)
            results[name] = {"verified": True, "samples": int(Y.size)}
            log(f"  {name:11s} VERIFIED  n={Y.size:,}")
            del X, Y
        except SystemExit as exc:
            ok = False
            results[name] = {"verified": False, "reason": str(exc)}
            log(f"  {name:11s} FAILED    {exc}")
    log(f"canonical instance {man['canonical_instance_id'][:16]}: "
        f"{'VERIFIED' if ok else 'FAILED'}")
    return {"canonical_instance_id": man["canonical_instance_id"],
            "verified": ok, "partitions": results, "checked_utc": _utc()}


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
