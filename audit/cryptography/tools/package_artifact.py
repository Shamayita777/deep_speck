"""
Build a clean, checksummed archive of the cryptography audit framework.

    python -m audit.cryptography.tools.package_artifact --out <new.zip> [--include <path> ...]

Contents: every git-tracked file under ``audit/cryptography`` plus explicitly ``--include``d
paths (for example the untracked core/, targets/, experiments/autond/). Excluded always:
bytecode and caches, ``.DS_Store``, ``._*``, ``__MACOSX``, staging files. The build FAILS if any
selected path is an iCloud placeholder (``.<name>.icloud``) or a missing file, because packaging a
placeholder would silently drop evidence. ``SHA256SUMS`` and ``PACKAGE_INFO.json`` (commit, dirty
state, file count) are written into the archive. The repository is only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CRYPTO = Path(__file__).resolve().parents[1]
REPO = CRYPTO.parents[1]
EXCLUDED_PARTS = {"__pycache__", "__MACOSX", ".pytest_cache", ".staging"}


def excluded(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return (bool(EXCLUDED_PARTS & set(parts)) or name.endswith((".pyc", ".pyo")) or name == ".DS_Store"
            or name.startswith("._") or ".tmp-" in name)


def placeholder(rel: str) -> bool:
    name = rel.split("/")[-1]
    return name.startswith(".") and name.endswith(".icloud")


def select(includes: list[str]) -> tuple[list[str], list[str]]:
    p = subprocess.run(["git", "ls-files", "-z", "audit/cryptography"], cwd=REPO, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"refusing to package: {REPO} is not a git working tree, so tracked files cannot be "
                         f"determined ({p.stderr.decode().strip()[:200]})")
    tracked = p.stdout.decode().split("\0")
    chosen = {t for t in tracked if t}
    for inc in includes:
        p = (REPO / inc).resolve()
        if p.is_dir():
            chosen |= {q.relative_to(REPO).as_posix() for q in p.rglob("*") if q.is_file()}
        else:
            chosen.add(p.relative_to(REPO).as_posix())
    problems = [r for r in sorted(chosen) if placeholder(r)]
    problems += [r for r in sorted(chosen) if not excluded(r) and not placeholder(r) and not (REPO / r).is_file()]
    return sorted(r for r in chosen if not excluded(r) and not placeholder(r)), problems


def build(out: Path, includes: list[str]) -> dict:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    files, problems = select(includes)
    if problems:
        raise SystemExit("refusing to package: missing files or iCloud placeholders:\n  " + "\n  ".join(problems[:50]))
    sums = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            data = (REPO / rel).read_bytes()
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
            z.writestr(zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0)), data)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "audit/cryptography"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip())
        info = {"git_head": head, "working_tree_dirty": dirty, "n_files": len(files), "includes": includes,
                "built_utc": datetime.now(timezone.utc).isoformat()}
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
        z.writestr("PACKAGE_INFO.json", json.dumps(info, indent=1) + "\n")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="clean artifact package")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include", action="append", default=[])
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, a.include), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
