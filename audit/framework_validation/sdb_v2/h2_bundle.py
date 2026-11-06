"""
Build the H2 evidence bundle: the 341 allowlisted non-model A-D4 members, copied once from the
archive into sdb_v2/evidence/h2_ad4/, with every path and hash verified before anything is copied.

  python -I -m ... is not used: run as  python -B audit/framework_validation/sdb_v2/h2_bundle.py

Nothing in the archive is loaded, unpickled, imported or executed: member bytes are streamed into
SHA-256 and written out unchanged. The 330 `.keras` members are hashed only (never copied); their
hashes are recorded against the `checkpoint_sha256` each epoch record names. The bundle directory
is refused if it already exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ALLOWLIST = HERE.parent / "SDB_V2_H2_ALLOWLIST_DRAFT.json"
ALLOWLIST_SHA256 = "f7c2bde87c3a645128f90d7afdc2e7c014a8be8a2c467051368d1ce13ef802af"
CERTIFICATE = REPO / "audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/result/d4_certificate.json"
CERTIFICATE_SHA256 = "b4527041ff788e53f9781095897841aadb6bfaa4e0287aa3627c3bfbffbbd244"
PREFIX = "deep_speck/audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/"
BUNDLE = HERE / "evidence" / "h2_ad4"


def sha_stream(f) -> str:
    h = hashlib.sha256()
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
    return h.hexdigest()


def main() -> int:
    if BUNDLE.exists():
        print(f"REFUSED: {BUNDLE} exists; the bundle is written once")
        return 1
    al_bytes = ALLOWLIST.read_bytes()
    if hashlib.sha256(al_bytes).hexdigest() != ALLOWLIST_SHA256:
        print("REFUSED: allowlist hash differs")
        return 1
    if hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest() != CERTIFICATE_SHA256:
        print("REFUSED: A-D4 certificate hash differs")
        return 1
    al = json.loads(al_bytes)
    archive = Path(al["archive"]["path"])
    with open(archive, "rb") as f:
        arch_sha = sha_stream(f)
    if arch_sha != al["archive"]["sha256"]:
        print(f"REFUSED: archive sha256 {arch_sha} != allowlist {al['archive']['sha256']}")
        return 1
    z = zipfile.ZipFile(archive)
    names = z.namelist()
    if len(names) != al["archive"]["member_count"] or len(set(names)) != len(names):
        print("REFUSED: archive member list differs or has duplicates")
        return 1
    problems, members = [], {m["member"]: m for m in al["members"]}
    for name, m in members.items():
        info = z.getinfo(name)
        mode = (info.external_attr >> 16) & 0xFFFF
        if (name.startswith("/") or ".." in name.split("/") or "\\" in name or stat.S_ISLNK(mode)
                or not name.startswith(PREFIX)):
            problems.append(f"unsafe path {name}")
            continue
        with z.open(name) as f:
            h = sha_stream(f)
        if h != m["sha256"] or info.file_size != m["size"]:
            problems.append(f"hash or size differs: {name}")
    if problems:
        print("REFUSED before copying:", problems[:5])
        return 1

    copy = sorted(n for n in members if not n.endswith(".keras"))
    keras = sorted(n for n in members if n.endswith(".keras"))
    if len(copy) != 341 or len(keras) != 330:
        print(f"REFUSED: {len(copy)} non-model and {len(keras)} model members, expected 341 and 330")
        return 1
    BUNDLE.mkdir(parents=True)
    files = []
    for name in copy:
        rel = name[len(PREFIX):]
        out = BUNDLE / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, open(out, "xb") as dst:
            h = hashlib.sha256()
            for b in iter(lambda: src.read(1 << 20), b""):
                h.update(b)
                dst.write(b)
        if h.hexdigest() != members[name]["sha256"]:
            print(f"ABORTED: copied bytes differ for {name}")
            return 1
        os.chmod(out, 0o444)
        files.append({"path": rel, "archive_member": name, "sha256": members[name]["sha256"],
                      "bytes": members[name]["size"]})
    ckpt = []
    for name in sorted(n for n in copy if n.endswith(".metrics.json")):
        rec = json.loads((BUNDLE / name[len(PREFIX):]).read_text())
        k = name.rsplit("/stage_r", 1)[0] + "/" + rec["checkpoint"]
        ckpt.append({"record": name[len(PREFIX):], "checkpoint_member": k,
                     "record_checkpoint_sha256": rec["checkpoint_sha256"],
                     "archive_member_sha256": members[k]["sha256"],
                     "match": members[k]["sha256"] == rec["checkpoint_sha256"]})
    manifest = {
        "artifact": "sdb-v2-h2-evidence-bundle-1",
        "source_archive": {"path": str(archive), "sha256": arch_sha, "bytes": archive.stat().st_size,
                           "member_count": len(names)},
        "allowlist": {"path": str(ALLOWLIST.relative_to(REPO)), "sha256": ALLOWLIST_SHA256},
        "certificate": {"path": str(CERTIFICATE.relative_to(REPO)), "sha256": CERTIFICATE_SHA256},
        "verified_before_copy": {"members": len(members), "paths_safe": True, "hashes_and_sizes_match": True},
        "files": files,
        "model_checkpoints_hashed_not_copied": len(keras),
        "checkpoint_hash_records": ckpt,
        "checkpoint_hash_records_all_match": all(c["match"] for c in ckpt),
        "nothing_loaded_or_executed": True,
    }
    (BUNDLE / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.chmod(BUNDLE / "BUNDLE_MANIFEST.json", 0o444)
    print(f"bundle {BUNDLE}: {len(files)} files, {len(ckpt)} checkpoint hash records, all match "
          f"{manifest['checkpoint_hash_records_all_match']}; manifest "
          f"{hashlib.sha256((BUNDLE / 'BUNDLE_MANIFEST.json').read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
