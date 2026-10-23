"""D0 build attestation for the canonical Gohr instance (sidecar; canonical formats unchanged).

WHY THIS EXISTS
---------------
`canonical.build()` records WHICH generator path produced a Gohr instance
("GohrAdapter.generate_partition" or "speck.make_train_data (direct)") but no
digest of the source that ran. A source hash taken after the fact proves only
what the file is NOW. This wrapper closes that gap without editing
`canonical.py`, the Gohr adapter or `speck.py`: it calls the unmodified
`canonical.build()` IN THIS PROCESS and, around that one call, records

  * the pinned generator (repo-root `speck.py`): SHA-256, git blob at HEAD,
    clean status, the imported module's resolved file, and the file that the
    `make_train_data` code object was compiled from;
  * the build code (`canonical.py`, the Gohr adapter, `train_nets.py` which the
    adapter imports, and this file): SHA-256 and git blob, before AND after;
  * that the module object `speck` in `sys.modules` is the same object, from
    the same file with the same digest, before and after generation (so the
    generation that canonical.build() performed ran that module);
  * repository HEAD and a digest of `git status --porcelain` before and after;
  * the build configuration, the command, the captured build log, the
    canonical instance id, the manifest SHA-256 and every partition hash;
  * the result of `canonical.verify()` on the new instance.

The attestation is written as `d0_gohr_build_attestation.json` inside the
instance directory (canonical.verify ignores extra files), created exclusively
(never overwritten). If any check fails AFTER generation, no attestation is
written: the instance exists but is unattested, and the D3 runner refuses it.

WHAT IT PROVES AND WHAT IT DOES NOT
-----------------------------------
It binds identity and integrity: these arrays, this manifest, produced in a
process whose imported generator module was the pinned file, from a repository
whose pinned sources were unchanged across the build. It does NOT prove i.i.d.
sampling, the quality of os.urandom, the cryptographic security of Speck, or
the absence of in-process tampering (e.g. monkeypatching by other code in the
same interpreter); hashes establish identity, not randomness.

USAGE (production; the canonical specification only, no overrides)
    python -B -m audit.dataset.tools.d0_gohr_attested_build build \\
        --out-dir audit/dataset/evidence/d0_gohr_speck_r5_production
    python -B -m audit.dataset.tools.d0_gohr_attested_build verify \\
        --out-dir audit/dataset/evidence/d0_gohr_speck_r5_production
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audit.dataset import canonical as C

ATTESTATION_SCHEMA_VERSION = "d0-gohr-build-attestation-1"
ATTESTATION_NAME = "d0_gohr_build_attestation.json"
ATTESTED = "ATTESTED"
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The Gohr generator the canonical specification names ("speck.make_train_data"):
#: the repository-root speck.py, unchanged since Gohr's 2019 upload (commit fba2fd7).
GENERATOR_PIN = {
    "module": "speck",
    "path": "speck.py",
    "function": "make_train_data",
    "sha256": "59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce",
    "git_blob": "28b2f836c2642b44a7bb3a445adfec2e99b36521",
}
#: Build code whose behaviour the instance depends on (D0 builder and the
#: adapter canonical.build() prefers), pinned at the values audited for D3.
BUILD_CODE_PINS = {
    "audit/dataset/canonical.py": {
        "sha256": "7fc650f5b451850cd461761b0ac117d7052707dd9fec1c045cab4d754b73ea1d",
        "git_blob": "5f919a294d1f168640e3beafcea52e153523be9b"},
    "audit/dataset/adapters/gohr.py": {
        "sha256": "3c0b7ac5327c68233ac792062f3559232cfc965b8a403312e552bcb2b516b488",
        "git_blob": "e8019392accbdd5989edd50d0e95ed99f8941b00"},
}
#: Recorded (not gated): imported by the Gohr adapter, not used to generate data.
RECORDED_SOURCES = ("train_nets.py",)
#: canonical.build()'s two Gohr generator paths (canonical.py `_resolve_gohr`).
ALLOWED_GENERATOR_PATHS = ("audit.dataset.adapters.gohr.GohrAdapter.generate_partition",
                           "speck.make_train_data (direct; adapter unavailable in this environment: ")
THIS_FILE = "audit/dataset/tools/d0_gohr_attested_build.py"


class AttestationRejected(ValueError):
    """A build attestation is missing, malformed, or does not match its instance."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout


def _source_record(repo: Path, rel: str) -> dict:
    path = repo / rel
    rc_blob, blob = _git(repo, "rev-parse", f"HEAD:{rel}")
    rc_st, st = _git(repo, "status", "--porcelain", "--", rel)
    return {"path": rel, "sha256": _sha256_file(path) if path.is_file() else None,
            "git_blob_at_head": blob.strip() if rc_blob == 0 else None,
            "clean_at_head": rc_blob == 0 and rc_st == 0 and not st.strip()}


def source_fingerprint(repo: Path) -> dict:
    """SHA-256, HEAD blob and cleanliness of every pinned or recorded source, plus repository state."""
    rels = [GENERATOR_PIN["path"], *BUILD_CODE_PINS, *RECORDED_SOURCES, THIS_FILE]
    rc_head, head = _git(repo, "rev-parse", "HEAD")
    rc_st, status = _git(repo, "status", "--porcelain")
    return {"sources": {r: _source_record(repo, r) for r in rels},
            "git_head": head.strip() if rc_head == 0 else None,
            "working_tree": None if rc_st != 0 else ("dirty" if status.strip() else "clean"),
            "git_status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest() if rc_st == 0 else None}


def check_pins(fp: dict) -> list:
    """Problems with the fingerprint against the generator and build-code pins (empty when all hold)."""
    problems = []
    g = fp["sources"][GENERATOR_PIN["path"]]
    if g["sha256"] != GENERATOR_PIN["sha256"]:
        problems.append(f"speck.py SHA-256 {g['sha256']} != pin {GENERATOR_PIN['sha256']}")
    if g["git_blob_at_head"] != GENERATOR_PIN["git_blob"]:
        problems.append(f"speck.py HEAD blob {g['git_blob_at_head']} != pin {GENERATOR_PIN['git_blob']}")
    if not g["clean_at_head"]:
        problems.append("speck.py has uncommitted modifications or is untracked")
    for rel, pin in BUILD_CODE_PINS.items():
        s = fp["sources"][rel]
        if s["sha256"] != pin["sha256"] or s["git_blob_at_head"] != pin["git_blob"] or not s["clean_at_head"]:
            problems.append(f"{rel} differs from its pin or is not clean at HEAD")
    return problems


def generator_module_record(repo: Path) -> dict:
    """The `speck` module this interpreter uses: resolved file, digest, and code-object origin."""
    mod = importlib.import_module(GENERATOR_PIN["module"])
    fn = getattr(mod, GENERATOR_PIN["function"])
    file_ = Path(mod.__file__).resolve()
    code_file = Path(fn.__code__.co_filename).resolve()
    expected = (repo / GENERATOR_PIN["path"]).resolve()
    return {"module_id": id(mod), "resolved_file_is_pinned_path": file_ == expected,
            "code_object_file_is_pinned_path": code_file == expected,
            "file_sha256": _sha256_file(file_), "relative_path": GENERATOR_PIN["path"] if file_ == expected else None}


def _module_problems(rec: dict) -> list:
    problems = []
    if not rec["resolved_file_is_pinned_path"]:
        problems.append("the imported `speck` module is not the repository-root speck.py")
    if not rec["code_object_file_is_pinned_path"]:
        problems.append("speck.make_train_data was not compiled from the repository-root speck.py")
    if rec["file_sha256"] != GENERATOR_PIN["sha256"]:
        problems.append("the imported speck module's file does not have the pinned digest")
    return problems


def _write_exclusive(path: Path, data: bytes) -> None:
    """Create `path` with `data`; never replace an existing file."""
    fd, tmp = tempfile.mkstemp(prefix=".attestation_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)                      # fails if `path` exists
    finally:
        os.unlink(tmp)


def attested_build(out_dir: Path, *, repo_root: Path = REPO_ROOT, spec: dict | None = None,
                   command: list | None = None, log: Callable[[str], Any] = print) -> dict:
    """Build the canonical Gohr instance through canonical.build() and write its attestation.

    Refuses BEFORE generation (nothing created) when the output directory exists, a
    pin does not hold, or the imported generator is not the pinned file. Refuses
    AFTER generation (instance left unattested) when any binding check fails.
    """
    out_dir = Path(out_dir)
    repo = Path(repo_root).resolve()
    spec = dict(spec if spec is not None else C.CANONICAL_SPEC)
    if out_dir.exists():
        raise SystemExit(f"REFUSED: {out_dir} exists. A canonical instance is built once into a new "
                         "directory; nothing was generated or changed.")
    if str(spec.get("target", C.DEFAULT_TARGET)) != "gohr-speck":
        raise SystemExit("REFUSED: this attestation wrapper builds the Gohr target only.")
    before = source_fingerprint(repo)
    problems = check_pins(before)
    module_before = generator_module_record(repo)
    problems += _module_problems(module_before)
    if problems:
        raise SystemExit("REFUSED before generation (nothing created):\n  - " + "\n  - ".join(problems))

    lines: list = []

    def tee(msg="", *a, **k):
        lines.append(str(msg))
        log(msg)

    started = datetime.now(timezone.utc).isoformat()
    # canonical.build() writes str(repo_root) into the manifest's git record, and a
    # manifest can never be edited afterwards; pass a path relative to the working
    # directory so no absolute home path is persisted.
    manifest = C.build(out_dir, repo_root=Path(os.path.relpath(repo)), spec=spec, log=tee)
    finished = datetime.now(timezone.utc).isoformat()

    module_after = generator_module_record(repo)
    after = source_fingerprint(repo)
    post = _module_problems(module_after)
    if module_after["module_id"] != module_before["module_id"]:
        post.append("the `speck` module object changed during generation")
    if after["sources"] != before["sources"] or after["git_head"] != before["git_head"]:
        post.append("a pinned or recorded source, or HEAD, changed during generation")
    gp = str(manifest.get("generator_path", ""))
    if not (gp == ALLOWED_GENERATOR_PATHS[0] or gp.startswith(ALLOWED_GENERATOR_PATHS[1])):
        post.append(f"canonical.build() used an unexpected generator path {gp!r}")
    verification = C.verify(out_dir, log=tee)
    if not verification.get("verified"):
        post.append("canonical.verify() did not verify the new instance")
    if post:
        raise SystemExit("REFUSED after generation: the instance at "
                         f"{out_dir} is left UNATTESTED (the D3 runner will refuse it):\n  - " + "\n  - ".join(post))

    man_path = out_dir / C.MANIFEST_NAME
    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "status": ATTESTED,
        "canonical_instance_id": manifest["canonical_instance_id"],
        "manifest_file": C.MANIFEST_NAME,
        "manifest_sha256": _sha256_file(man_path),
        "specification": spec,
        "specification_sha256": manifest["specification_sha256"],
        "specification_is_canonical": spec == dict(C.CANONICAL_SPEC),
        "partitions": {n: {k: manifest["partitions"][n][k] for k in
                           ("file", "samples", "file_sha256", "features_sha256", "labels_sha256")}
                       for n in C.PARTITIONS},
        "generator_pin": dict(GENERATOR_PIN),
        "build_code_pins": {k: dict(v) for k, v in BUILD_CODE_PINS.items()},
        "generator_path_recorded_by_canonical": gp,
        "imported_generator": {k: v for k, v in module_after.items() if k != "module_id"},
        "generator_module_identity_unchanged": True,
        "sources_before": before, "sources_after": after,
        "canonical_verify": {"verified": True},
        "command": list(command) if command is not None else None,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "started_utc": started, "finished_utc": finished,
        "build_log": lines,
        "proves": ("identity and integrity: these arrays and this manifest were produced by canonical.build() in "
                   "a process whose imported `speck` module was the pinned file, with the pinned build code "
                   "unchanged across the build"),
        "does_not_prove": ["i.i.d. sampling or the quality of os.urandom",
                           "the cryptographic security of Speck",
                           "absence of in-process tampering by other code in the same interpreter",
                           "anything about duplicates (D1), dependence (D2) or distribution (D3)"],
    }
    _write_exclusive(out_dir / ATTESTATION_NAME,
                     json.dumps(attestation, indent=2, sort_keys=True, allow_nan=False).encode())
    log(f"attestation: {out_dir / ATTESTATION_NAME}")
    return attestation


def verify_build_attestation(instance_dir: Path, *, expected_spec: dict | None = None,
                             repo_root: Path = REPO_ROOT) -> dict:
    """Check the attestation against its instance and the CURRENT pins. Raises AttestationRejected.

    Content checks only: the caller (the D3 runner) also runs canonical.verify()
    on the arrays. Returns {"sha256": ..., "attestation": ...}.
    """
    d = Path(instance_dir)
    path = d / ATTESTATION_NAME
    if not path.is_file():
        raise AttestationRejected(f"no build attestation at {path}; the instance's generator is unattested.")
    raw = path.read_bytes()
    try:
        att = json.loads(raw)
    except ValueError as exc:
        raise AttestationRejected(f"attestation is not JSON: {exc}") from None
    man_path = d / C.MANIFEST_NAME
    if not man_path.is_file():
        raise AttestationRejected("the instance has no canonical manifest.")
    man = json.loads(man_path.read_text())
    probs = []
    if att.get("schema_version") != ATTESTATION_SCHEMA_VERSION or att.get("status") != ATTESTED:
        probs.append("schema or status is not an ATTESTED d0-gohr-build-attestation-1")
    if att.get("manifest_sha256") != _sha256_file(man_path):
        probs.append("manifest SHA-256 differs from the attested one")
    if att.get("canonical_instance_id") != man.get("canonical_instance_id"):
        probs.append("canonical instance id differs from the manifest's")
    for n in C.PARTITIONS:
        rec = (att.get("partitions") or {}).get(n) or {}
        mrec = (man.get("partitions") or {}).get(n) or {}
        if any(rec.get(k) != mrec.get(k) for k in ("file", "samples", "file_sha256", "features_sha256",
                                                    "labels_sha256")):
            probs.append(f"partition {n} hashes differ from the manifest")
    if att.get("specification") != man.get("specification"):
        probs.append("attested specification differs from the manifest's")
    if expected_spec is not None and att.get("specification") != expected_spec:
        probs.append("attested specification is not the expected specification")
    if att.get("generator_pin") != GENERATOR_PIN:
        probs.append("attested generator pin differs from the current pin")
    if att.get("build_code_pins") != BUILD_CODE_PINS:
        probs.append("attested build-code pins differ from the current pins")
    imp = att.get("imported_generator") or {}
    if not (imp.get("resolved_file_is_pinned_path") and imp.get("code_object_file_is_pinned_path")
            and imp.get("file_sha256") == GENERATOR_PIN["sha256"]):
        probs.append("the attested imported generator is not the pinned file")
    if att.get("generator_module_identity_unchanged") is not True:
        probs.append("generator module identity not attested as unchanged")
    sb, sa = att.get("sources_before") or {}, att.get("sources_after") or {}
    if sb.get("sources") != sa.get("sources") or sb.get("git_head") != sa.get("git_head") or not sb:
        probs.append("sources or HEAD changed during the attested build")
    elif check_pins(sb):
        probs.append("the attested sources did not satisfy the pins")
    gp = str(att.get("generator_path_recorded_by_canonical", ""))
    if gp != man.get("generator_path") or not (gp == ALLOWED_GENERATOR_PATHS[0]
                                                or gp.startswith(ALLOWED_GENERATOR_PATHS[1])):
        probs.append("generator path differs from the manifest's or is not a canonical Gohr path")
    if (att.get("canonical_verify") or {}).get("verified") is not True:
        probs.append("canonical.verify() was not attested")
    tool_now = _sha256_file(Path(repo_root) / THIS_FILE)
    if ((sb.get("sources") or {}).get(THIS_FILE) or {}).get("sha256") != tool_now:
        probs.append("this attestation was produced by a different version of the attestation tool")
    if probs:
        raise AttestationRejected("build attestation rejected: " + "; ".join(probs))
    return {"sha256": hashlib.sha256(raw).hexdigest(), "attestation": att}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Attested build of the canonical Gohr D0 instance")
    ap.add_argument("action", choices=("build", "verify"))
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.action == "build":
        attested_build(a.out_dir, command=["python", "-m", "audit.dataset.tools.d0_gohr_attested_build",
                                           "build", "--out-dir", str(a.out_dir)])
        return 0
    try:
        res = verify_build_attestation(a.out_dir, expected_spec=dict(C.CANONICAL_SPEC))
    except AttestationRejected as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(f"attestation verified: sha256 {res['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
