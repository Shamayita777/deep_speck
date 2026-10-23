"""D1 — exact duplicate and overlap census for the persisted canonical Gohr instance.

Orchestration only, following the instance-bound AutoND D1 runner
(`d1_autond_exact_census.py`). Every check is performed by the UNCHANGED generic
engine `audit.dataset.d1_duplicate_detection`. The legacy Gohr D1 runner
(`d1_gohr_exact_census.py`), which regenerates its own data, and its evidence
are neither used nor changed; that evidence concerns its own unpersisted
instance only.

ORDER (a refusal at any step writes nothing and never reaches the engine)
    1. refuse an existing output directory;
    2. require the frozen canonical Gohr specification and production sizes;
    3. verify the generator and build-code pins and the imported `speck` module;
    4. verify the D0 build attestation against the manifest and current pins;
    5. canonical.verify: every file and array hash, instance id, spec digest;
    6. load the partitions (file hash re-checked, arrays re-hashed) -- never regenerate;
    7. run the generic engine; bind the certificate to the instance id, the
       manifest SHA-256 and the attestation SHA-256; write atomically.

SCOPE: a D1 PASS covers exact duplicates within partitions, exact feature
overlap across partitions and exact-feature label conflicts in THIS instance.
It does not establish i.i.d. sampling, absence of near duplicates or of other
dependence, or anything about distribution (D3).

USAGE
    python -B -m audit.dataset.experiments.d1_gohr_canonical_census \\
        --instance-dir audit/dataset/evidence/d0_gohr_speck_r5_production \\
        --out-dir audit/dataset/evidence/d1_gohr_speck_r5_canonical_<id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.dataset import canonical as C
from audit.dataset import d1_duplicate_detection as D1
from audit.dataset.common.provenance import array_sha256
from audit.dataset.tools import d0_gohr_attested_build as ATT

RUNNER_SCHEMA_VERSION = "d1-gohr-canonical-certificate-1"
CERTIFICATE_NAME = "d1_gohr_certificate.json"
RUN_MANIFEST_NAME = "d1_gohr_run_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_SPEC = dict(C.CANONICAL_SPEC)
PRODUCTION_SIZES = {n: int(EXPECTED_SPEC[f"{n}_samples"]) for n in C.PARTITIONS}
#: Schemas the engine must find (verified from speck.make_train_data / convert_to_binary).
FEATURE_SCHEMA = {"ndim": 2, "columns": 64, "dtype": "uint8", "value_set": [0, 1]}
LABEL_SCHEMA = {"ndim": 1, "dtype": "uint8", "value_set": [0, 1]}
REPRESENTATION_CONVENTION = ("speck.convert_to_binary([c0l, c0r, c1l, c1r]): 16-bit words, MSB first; "
                             "columns 0-31 the first ciphertext, 32-63 the second")
FEATURE_ENCODING = "binary-bits-uint8"

#: Engine entry point behind one name so a test can prove a refused run never reaches it.
D1_ENGINE = D1.audit_duplicates

SCOPE_LIMITATIONS = [
    "Applies to ONE persisted instance (canonical Gohr specification: Speck32/64, 5 rounds, difference "
    "(0x0040, 0x0000)). The legacy Gohr D1 certificate concerns a different, unpersisted instance.",
    "A PASS covers exact duplicates within partitions, exact feature overlap across partitions and exact-feature "
    "label conflicts only. It does NOT establish i.i.d. sampling, absence of near duplicates or of other "
    "dependence, or anything about the feature distribution.",
    "exact_replay_available is False: the generator draws from os.urandom, so the instance can be re-read, "
    "not regenerated.",
    "D0 verification and the build attestation establish identity and integrity, not randomness quality.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def preflight(instance_dir: Path, *, log=print) -> dict:
    """Steps 2-5. Raises SystemExit (nothing written) on any failure."""
    instance_dir = Path(instance_dir)
    manifest = C.load_manifest(instance_dir)
    expected = dict(EXPECTED_SPEC, **{f"{n}_samples": PRODUCTION_SIZES[n] for n in C.PARTITIONS})
    spec = manifest.get("specification") or {}
    if spec != expected:
        diff = sorted(k for k in set(spec) | set(expected) if spec.get(k) != expected.get(k))
        raise SystemExit(f"REFUSED: the instance is not the frozen Gohr D0 configuration (differs in {diff}); "
                         "D1 was not run.")
    fp = ATT.source_fingerprint(REPO_ROOT)
    problems = ATT.check_pins(fp) + ATT._module_problems(ATT.generator_module_record(REPO_ROOT))
    if problems:
        raise SystemExit("REFUSED: the Gohr generator or build code is not the pinned source; D1 was not run:\n  - "
                         + "\n  - ".join(problems))
    try:
        att = ATT.verify_build_attestation(instance_dir, expected_spec=expected)
    except ATT.AttestationRejected as exc:
        raise SystemExit(f"REFUSED: {exc}") from None
    log("D1 preflight — verifying the canonical Gohr D0 instance")
    verification = C.verify(instance_dir, log=log)
    if not verification.get("verified"):
        raise SystemExit("REFUSED: the canonical D0 instance did not verify; D1 was not run.")
    return {"manifest": manifest, "manifest_sha256": _sha256(instance_dir / C.MANIFEST_NAME),
            "verification": verification, "attestation": att}


def load_partitions(instance_dir: Path, manifest: dict) -> tuple[dict, dict]:
    """Step 6: re-read (never regenerate) and re-confirm every array against the verified manifest."""
    X, Y = {}, {}
    for n in C.PARTITIONS:
        x, y = C.load_partition(instance_dir, n, spec=manifest["specification"], verify_file_hash=True)
        rec = manifest["partitions"][n]
        if array_sha256(np.ascontiguousarray(x)) != rec["features_sha256"] or \
                array_sha256(np.ascontiguousarray(y)) != rec["labels_sha256"]:
            raise SystemExit(f"REFUSED: loaded {n} arrays do not match the verified manifest.")
        X[n], Y[n] = x, y
    return X, Y


def _code_identity() -> dict:
    files = {"runner": Path(__file__), "engine": Path(D1.__file__), "canonical": Path(C.__file__),
             "attestation_tool": Path(ATT.__file__), "generator": REPO_ROOT / ATT.GENERATOR_PIN["path"]}
    return {k: {"path": _rel(p), "sha256": _sha256(p)} for k, p in files.items()}


def _write_atomically(out_dir: Path, files: dict, manifest_extra: dict) -> None:
    """Write the complete set or nothing; claim the directory exclusively; never replace."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d1_gohr_staging_", dir=str(out_dir.parent)))
    claimed = False
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        man = dict(manifest_extra, run_status="COMPLETE",
                   files={n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
        (staging / RUN_MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {out_dir} appeared during the run. Nothing was written.") from None
        claimed = True
        os.rename(staging, out_dir)
        claimed = False
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if claimed and out_dir.is_dir() and not any(out_dir.iterdir()):
            out_dir.rmdir()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="D1 census of the persisted canonical Gohr instance.")
    p.add_argument("--instance-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)
    run_id, started = uuid.uuid4().hex, datetime.now(timezone.utc).isoformat()
    if a.out_dir.exists():
        raise SystemExit(f"REFUSED: --out-dir {a.out_dir} exists. D1 evidence is never overwritten; nothing was "
                         "deleted.")
    pre = preflight(a.instance_dir)
    manifest = pre["manifest"]
    identity_before = _code_identity()
    X, Y = load_partitions(a.instance_dir, manifest)
    results = D1_ENGINE(X["train"], X["validation"], X["test"], Y["train"], Y["validation"], Y["test"],
                        expected_feature_schema=FEATURE_SCHEMA, expected_label_schema=LABEL_SCHEMA,
                        representation_convention=REPRESENTATION_CONVENTION, feature_encoding=FEATURE_ENCODING)
    del X, Y
    verdict = D1.evaluate_d1(results)
    spec = manifest["specification"]
    cert = D1.build_d1_certificate(results, dataset_id=spec["dataset_id"], dataset_version=spec.get("dataset_version"),
                                   generation_procedure=spec.get("generation_procedure"),
                                   generation_parameters={k: spec[k] for k in
                                                          ("num_rounds", "input_difference", "train_samples",
                                                           "validation_samples", "test_samples", "randomness_source")},
                                   random_seed=None)
    if _code_identity() != identity_before or \
            ATT.verify_build_attestation(a.instance_dir)["sha256"] != pre["attestation"]["sha256"]:
        raise SystemExit("REFUSED: code or the build attestation changed during the run. Nothing was written.")
    cert["runner"] = {"schema_version": RUNNER_SCHEMA_VERSION, "run_id": run_id, "started_utc": started,
                      "finished_utc": datetime.now(timezone.utc).isoformat()}
    cert["canonical_instance"] = {
        "canonical_instance_id": manifest["canonical_instance_id"],
        "manifest_path": _rel(Path(a.instance_dir) / C.MANIFEST_NAME), "manifest_sha256": pre["manifest_sha256"],
        "specification_sha256": manifest["specification_sha256"],
        "partition_hashes": {n: {k: manifest["partitions"][n][k] for k in
                                 ("file_sha256", "features_sha256", "labels_sha256", "samples")}
                             for n in C.PARTITIONS}}
    cert["build_attestation"] = {"path": _rel(Path(a.instance_dir) / ATT.ATTESTATION_NAME),
                                 "sha256": pre["attestation"]["sha256"]}
    cert["d0_verification"] = {"verified": True, "checked_utc": pre["verification"].get("checked_utc"),
                               "ordering": "instance, attestation and pins verified BEFORE the D1 engine ran"}
    cert["generator_pin"] = dict(ATT.GENERATOR_PIN)
    cert["code_identity"] = identity_before
    cert["environment_runner"] = {"python": platform.python_version(), "numpy": np.__version__,
                                  "platform": platform.platform()}
    cert["scope_limitations"] = SCOPE_LIMITATIONS
    cert["verdict"] = verdict
    files = {CERTIFICATE_NAME: json.dumps(cert, indent=2, sort_keys=True, default=str).encode()}
    _write_atomically(a.out_dir, files, {"run_id": run_id, "canonical_instance_id": manifest["canonical_instance_id"],
                                         "d0_manifest_sha256": pre["manifest_sha256"],
                                         "build_attestation_sha256": pre["attestation"]["sha256"]})
    print(f"  D1 verdict: {verdict.get('outcome', verdict)}")
    print(f"  certificate: {a.out_dir / CERTIFICATE_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
