#!/usr/bin/env python3
"""
Restore the 11 H2 prediction arrays (excluded from anchor commit 39c2581) from the supplementary
archive, at their exact registered paths, verified against manifest_sdb_v2.json. Standard library only.

  python3 -I audit/framework_validation/sdb_v2/post_run/restore_h2_predictions.py \
      --archive sdb_v2_h2_predictions_supplement.tar.gz --repo-root .

Refuses: an archive whose SHA-256 differs from --archive-sha256 (when given); any member that is not
one of the 11 registered `.npy` paths; links, absolute or `..` paths; a member whose bytes do not hash
to the manifest value. An existing file is left untouched if identical and refused otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

MANIFEST = "audit/framework_validation/sdb_v2/manifest_sdb_v2.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--archive-sha256")
    ap.add_argument("--repo-root", default=".", type=Path)
    a = ap.parse_args(argv)
    repo = a.repo_root.resolve()
    if a.archive_sha256 and hashlib.sha256(a.archive.read_bytes()).hexdigest() != a.archive_sha256:
        print("REFUSED: archive SHA-256 differs")
        return 1
    man = json.loads((repo / MANIFEST).read_text())
    want = {p: h for p, h in man["evidence"].items() if p.endswith(".npy")}
    if len(want) != 11:
        print(f"REFUSED: manifest lists {len(want)} .npy evidence files, not 11")
        return 1
    done = {}
    with tarfile.open(a.archive, "r:gz") as t:
        members = t.getmembers()
        names = [m.name for m in members]
        if sorted(names) != sorted(want) or any(not m.isfile() for m in members):
            print("REFUSED: archive members are not exactly the 11 registered files")
            return 1
        for m in members:
            data = t.extractfile(m).read()
            h = hashlib.sha256(data).hexdigest()
            if h != want[m.name]:
                print(f"REFUSED: {m.name} hashes to {h}, manifest {want[m.name]}")
                return 1
            dest = repo / m.name
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest() != h:
                    print(f"REFUSED: {m.name} exists with different content")
                    return 1
                done[m.name] = "already present, identical"
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "xb") as f:
                f.write(data)
            done[m.name] = "restored"
    print(json.dumps({"restored": done, "all_hashes_match_manifest": True}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
