"""
Immutable, hash-bound datasets for the adversarial battery.

Fixes the v1 defect in which `distinguisher_accuracy()` generated fresh data
on every call and ignored its own `seed` argument, so nothing it reported was
reproducible and the "sealed" set was sealed in name only.

Two disjoint artifacts are produced and persisted:

    calibration  - the ONLY data a threshold may be chosen on
    sealed       - the data a final number is reported on; never used to
                   select anything, never regenerated during scoring

Entropy: Gohr's generator draws from os.urandom, so exact replay is NOT
available. That is recorded honestly in the manifest rather than papered over
with a seed argument that does nothing; reproducibility is provided by
persisting and hashing the arrays themselves.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_array(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        if a.size:
            h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    os.close(fd)
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)


def build_or_load(directory, name: str, n: int, *, rounds: int, differential) -> dict:
    """
    Return a persisted, hash-verified mixed real/random set. Built once; every
    later call reloads and re-verifies. Never silently regenerated.
    """
    directory = Path(directory)
    man_path = directory / f"{name}_manifest.json"
    data_path = directory / f"{name}.npz"

    if man_path.exists():
        man = json.loads(man_path.read_text())
        # The persisted set must match the SPECIFICATION it is being requested
        # under. A silent mismatch would mean a certificate reporting one
        # configuration while the numbers came from another.
        spec = {"n": int(n), "rounds": int(rounds),
                "differential": [int(d) for d in differential],
                "generator": "audit.cryptography.gohr.speck.make_train_data",
                "composition": "mixed real/random as emitted by speck.make_train_data"}
        mismatch = {k: {"requested": v, "persisted": man.get(k)}
                    for k, v in spec.items() if man.get(k) != v}
        if mismatch:
            raise ValueError(
                f"{name}: the persisted set does not match the requested "
                f"specification: {mismatch}. The set is sealed; it is NOT "
                "regenerated and NOT reinterpreted. Use a new run directory.")
        if not data_path.exists():
            raise FileNotFoundError(
                f"{name}: manifest exists but {data_path} is missing. The set is "
                "sealed; it is NOT regenerated. Restore the file or start a new run "
                "directory.")
        with np.load(data_path) as z:
            X, Y = z["X"], z["Y"]
        if sha256_array(X, Y) != man["content_sha256"]:
            raise ValueError(f"{name}: content hash mismatch; the sealed set has been "
                             "altered")
        man["loaded_from_disk"] = True
        return {**man, "X": X, "Y": Y}

    from audit.cryptography.gohr import speck as sp
    X, Y = sp.make_train_data(n, rounds, diff=tuple(differential))
    _atomic_savez(data_path, X=X, Y=Y)
    man = {
        "name": name, "n": int(n), "rounds": int(rounds),
        "differential": [int(d) for d in differential],
        "composition": "mixed real/random as emitted by speck.make_train_data",
        "label_counts": {"real": int((Y == 1).sum()), "random": int((Y == 0).sum())},
        "content_sha256": sha256_array(X, Y),
        "file_sha256": sha256_file(data_path),
        "file": data_path.name,
        "entropy_source": "os.urandom via speck.make_train_data",
        "exact_replay_available": False,
        "replay_note": "the generator is not seed-replayable; reproducibility is "
                       "provided by persisting and hashing the arrays",
        "generator": "audit.cryptography.gohr.speck.make_train_data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "loaded_from_disk": False,
    }
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True))
    return {**man, "X": X, "Y": Y}
