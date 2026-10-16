"""
The packaging contract, enforced as a test.

Revision 2's archive placed `SHA256SUMS.txt` and a `run_evidence/`
directory at its ROOT, so `unzip -d .` from the repository root created
two paths that have nothing to do with the AutoND target, and an operator
who already had a `SHA256SUMS.txt` would have had it overwritten.

The contract this file enforces:

  * every path the archive carries lies under `audit/targets/autond/`;
  * the three Phase 1 initializers are not in it, so the operator's
    copies are never touched;
  * the manifest, the logs and the code have one documented location
    each, and the manifest's inventory matches the archive's exactly.

The first two checks run from the shipped manifest and need nothing
external. The last needs the archive, so it runs when
`AUTOND_PACKAGE_ZIP` points at it and skips otherwise - but
`test_manifest_is_self_consistent` still verifies every digest in the
manifest against the files on disk, so a mismatch cannot pass silently.
"""

from __future__ import annotations

import hashlib
import os
import unittest
import zipfile
from pathlib import Path

#: Everything the archive may contain lives under the AutoND target...
PACKAGE_PREFIX = "audit/targets/autond/"

#: ...with ONE deliberate exception, named by exact path rather than by a
#: widened prefix. Revision 3c adds a generic equivalence-testing module
#: to the shared framework, because the TOST / Holm / power machinery is
#: not target-specific and keeping a private copy in the adapter would be
#: duplicated generic logic. It is a pure ADDITION: no pre-existing file
#: under audit/common/ or audit/implementation/ is modified or shipped.
#:
#: Listing the exact path, not "audit/common/", keeps the contract strict
#: - any OTHER file outside the AutoND target still fails these tests.
ALLOWED_OUTSIDE_TARGET = (
    "audit/common/equivalence.py",
)


def _is_allowed(path: str) -> bool:
    return path.startswith(PACKAGE_PREFIX) or path in ALLOWED_OUTSIDE_TARGET

#: Files that belong to the operator's Phase 1 install and must never be
#: shipped, because shipping them replaces their contents.
PHASE1_FILES = (
    "audit/targets/__init__.py",
    "audit/targets/autond/__init__.py",
    "audit/targets/autond/tests/__init__.py",
)

#: Documented locations, relative to the repository root.
MANIFEST_PATH = "audit/targets/autond/SHA256SUMS.txt"
EVIDENCE_DIR = "audit/targets/autond/run_evidence"


def _target_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _target_dir().parents[2]


def _read_manifest() -> dict[str, str]:
    m = _repo_root() / MANIFEST_PATH
    if not m.is_file():
        return {}
    out: dict[str, str] = {}
    for line in m.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0]
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class TestManifestContract(unittest.TestCase):
    def setUp(self):
        self.manifest = _read_manifest()
        if not self.manifest:
            self.skipTest(f"no manifest at {MANIFEST_PATH}; this test applies "
                          "to an installed package")

    def test_every_manifest_path_is_allowed(self):
        outside = [p for p in self.manifest if not _is_allowed(p)]
        self.assertEqual(
            outside, [],
            f"these paths are neither under {PACKAGE_PREFIX} nor in the "
            f"explicit allowlist {ALLOWED_OUTSIDE_TARGET}: {outside}")

    def test_the_allowlist_is_exactly_what_is_expected(self):
        """
        Guards the exception itself: if something else is ever added
        outside the AutoND target, this fails and the reason has to be
        written down rather than absorbed silently.
        """
        self.assertEqual(ALLOWED_OUTSIDE_TARGET,
                         ("audit/common/equivalence.py",))
        shipped_outside = sorted(p for p in self.manifest
                                 if not p.startswith(PACKAGE_PREFIX))
        self.assertEqual(shipped_outside, list(ALLOWED_OUTSIDE_TARGET))

    def test_no_preexisting_framework_file_is_shipped(self):
        """
        The addition must not become a vehicle for replacing framework
        files. Only the one new module may appear outside the target.
        """
        for p in self.manifest:
            if p.startswith(("audit/common/", "audit/implementation/")):
                self.assertIn(p, ALLOWED_OUTSIDE_TARGET,
                              f"{p} is a pre-existing framework path")

    def test_no_phase1_initializer_is_shipped(self):
        for f in PHASE1_FILES:
            self.assertNotIn(
                f, self.manifest,
                f"{f} belongs to the Phase 1 install; shipping it replaces the "
                "operator's copy")

    def test_no_path_escapes_or_is_absolute(self):
        for p in self.manifest:
            self.assertFalse(Path(p).is_absolute(), p)
            self.assertNotIn("..", Path(p).parts, p)

    def test_manifest_is_self_consistent(self):
        root = _repo_root()
        wrong, missing = [], []
        for rel, digest in sorted(self.manifest.items()):
            f = root / rel
            if not f.is_file():
                missing.append(rel)
            elif _sha256(f) != digest:
                wrong.append(rel)
        self.assertEqual(missing, [], f"manifest names missing files: {missing}")
        self.assertEqual(wrong, [], f"digest mismatch: {wrong}")

    def test_documented_locations_exist(self):
        root = _repo_root()
        self.assertTrue((root / MANIFEST_PATH).is_file())
        self.assertTrue((root / EVIDENCE_DIR).is_dir(),
                        f"the logs' documented location {EVIDENCE_DIR} is "
                        "missing")

    def test_logs_are_inside_the_target(self):
        for rel in self.manifest:
            if rel.endswith(".log") or "/run_evidence/" in rel:
                self.assertTrue(rel.startswith(EVIDENCE_DIR + "/"), rel)


def _archive_pointer_mismatch(archive_manifest):
    """
    True when AUTOND_PACKAGE_ZIP names an archive whose manifest differs
    from the installed one. Returns False when either side is absent:
    the pointer can only be judged stale against something to compare.
    """
    installed = _repo_root() / MANIFEST_PATH
    if archive_manifest is None or not installed.is_file():
        return False
    return archive_manifest != installed.read_text()


class TestArchivePointerIsCurrent(unittest.TestCase):
    """
    The environment variable outlives the archive it was set for. Having
    verified one revision with AUTOND_PACKAGE_ZIP exported, the next
    revision's suite checks the new contract against the old ZIP, and the
    mismatch surfaces as a packaging defect in files the operator just
    installed correctly. It is not one. This test names the real cause so
    the archive checks never have to be read that way.
    """

    def test_the_pointer_matches_the_installed_revision(self):
        z = os.environ.get("AUTOND_PACKAGE_ZIP")
        if not z or not Path(z).is_file():
            self.skipTest("AUTOND_PACKAGE_ZIP is unset; nothing to check")
        with zipfile.ZipFile(z) as zf:
            archive_manifest = (zf.read(MANIFEST_PATH).decode()
                                if MANIFEST_PATH in zf.namelist() else None)
        if _archive_pointer_mismatch(archive_manifest):
            self.fail(
                f"AUTOND_PACKAGE_ZIP={z}\n"
                f"  That archive's {MANIFEST_PATH} differs from the one "
                f"installed in this checkout, so it is a different "
                f"revision of this package.\n"
                f"  Your installed files are not implicated. This is a "
                f"stale environment variable.\n"
                f"  Fix either way:\n"
                f"    unset AUTOND_PACKAGE_ZIP        # skip archive "
                f"checks; they only matter when building a release\n"
                f"    export AUTOND_PACKAGE_ZIP=/path/to/the/archive/you/"
                f"just/installed.zip")


class TestArchiveContract(unittest.TestCase):
    """
    Checks the archive itself. Point AUTOND_PACKAGE_ZIP at it to run.
    """

    def setUp(self):
        z = os.environ.get("AUTOND_PACKAGE_ZIP")
        if not z or not Path(z).is_file():
            self.skipTest("set AUTOND_PACKAGE_ZIP=/path/to/the.zip to check "
                          "the archive itself")
        self.zip_path = Path(z)
        with zipfile.ZipFile(self.zip_path) as zf:
            self.names = [n for n in zf.namelist() if not n.endswith("/")]
            archive_manifest = (zf.read(MANIFEST_PATH).decode()
                                if MANIFEST_PATH in zf.namelist() else None)

        # A stale AUTOND_PACKAGE_ZIP points at a previous revision's archive,
        # so this revision's contract would be checked against the wrong
        # files. TestArchivePointerIsCurrent reports that as a single named
        # failure; these checks skip rather than each restating it.
        if _archive_pointer_mismatch(archive_manifest):
            self.skipTest("AUTOND_PACKAGE_ZIP is not this revision's archive "
                          "- see TestArchivePointerIsCurrent")

    def test_every_archive_entry_is_allowed(self):
        outside = [n for n in self.names if not _is_allowed(n)]
        self.assertEqual(
            outside, [],
            f"the archive would create these outside the allowed set: "
            f"{outside}")

    def test_the_archive_ships_only_the_one_allowed_outside_file(self):
        shipped_outside = sorted(n for n in self.names
                                 if not n.startswith(PACKAGE_PREFIX))
        self.assertEqual(shipped_outside, list(ALLOWED_OUTSIDE_TARGET))

    def test_archive_ships_no_phase1_initializer(self):
        for f in PHASE1_FILES:
            self.assertNotIn(f, self.names)

    def test_archive_ships_no_freeze_record(self):
        """
        A freeze record is an act performed by the operator in their own
        repository. Shipping one would mean the package arrived claiming a
        preregistration had been frozen by someone else, at a commit that
        does not exist in their history.
        """
        frozen = [n for n in self.names if n.endswith(".frozen.json")]
        self.assertEqual(frozen, [],
                         f"the archive carries a freeze record: {frozen}")

    def test_archive_has_no_absolute_or_escaping_entries(self):
        for n in self.names:
            self.assertFalse(n.startswith("/"), n)
            self.assertNotIn("..", Path(n).parts, n)

    def test_archive_carries_no_caches(self):
        bad = [n for n in self.names
               if "__pycache__" in n or n.endswith((".pyc", ".pyo"))
               or Path(n).name in (".DS_Store",) or Path(n).name.startswith("._")]
        self.assertEqual(bad, [], f"the archive carries junk: {bad}")

    def test_archive_inventory_matches_its_manifest(self):
        with zipfile.ZipFile(self.zip_path) as zf:
            if MANIFEST_PATH not in zf.namelist():
                self.fail(f"the archive has no {MANIFEST_PATH}")
            text = zf.read(MANIFEST_PATH).decode()
        declared = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                declared.add(parts[1].strip())
        in_zip = set(self.names) - {MANIFEST_PATH}
        self.assertEqual(
            in_zip, declared,
            f"archive and manifest disagree. Only in archive: "
            f"{sorted(in_zip - declared)}. Only in manifest: "
            f"{sorted(declared - in_zip)}")

    def test_archive_digests_match_its_manifest(self):
        with zipfile.ZipFile(self.zip_path) as zf:
            text = zf.read(MANIFEST_PATH).decode()
            declared = {}
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        declared[parts[1].strip()] = parts[0]
            wrong = []
            for rel, digest in sorted(declared.items()):
                got = hashlib.sha256(zf.read(rel)).hexdigest()
                if got != digest:
                    wrong.append(rel)
        self.assertEqual(wrong, [], f"digest mismatch inside the archive: {wrong}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
