"""Packaging excludes caches and metadata and refuses iCloud placeholders."""

from __future__ import annotations

import zipfile

import pytest

from audit.cryptography.tools import package_artifact as PK


@pytest.mark.parametrize("rel,excl", [
    ("audit/cryptography/core/__pycache__/x.cpython-311.pyc", True), ("audit/cryptography/a.pyc", True),
    ("audit/cryptography/.DS_Store", True), ("__MACOSX/audit/x", True), ("audit/cryptography/._x.py", True),
    ("audit/cryptography/runs/r/.staging/data.npz.t.part", True), ("audit/cryptography/core/evidence.py", False),
])
def test_exclusions(rel, excl):
    assert PK.excluded(rel) is excl


def test_placeholders_are_detected():
    assert PK.placeholder("audit/cryptography/evidence_current/ce4/production_20261004/raw/.ce4_rep00.npz.icloud")
    assert not PK.placeholder("audit/cryptography/core/evidence.py")


def _in_git_tree() -> bool:
    import subprocess
    return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=PK.REPO,
                          capture_output=True).returncode == 0


def test_outside_a_git_tree_packaging_refuses_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(PK, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="not a git working tree"):
        PK.select([])


@pytest.mark.skipif(not _in_git_tree(), reason="packaging needs a git working tree (e.g. not an isolated copy)")
def test_build_contains_checksums_and_no_caches(tmp_path):
    out = tmp_path / "pkg.zip"
    info = PK.build(out, ["audit/cryptography/core"])
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        sums = z.read("SHA256SUMS").decode().splitlines()
    assert "audit/cryptography/core/evidence.py" in names and info["n_files"] == len(sums)
    assert not any(PK.excluded(n) for n in names)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        PK.build(out, [])
