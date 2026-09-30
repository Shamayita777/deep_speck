"""
Regression tests for the Kaggle preflight failure (2026-09-30):

    --recover-legacy-block0 --run-dir <explicit> ...
    HistoricalWriteError: refusing to write run_manifest.json:
    resolved target <repo>/run_manifest.json

Root cause: nothing validated the run directory. argparse(type=Path) turns
an empty value into Path(".") (the working directory), so "", "." and "./"
reached open_run() as the repository root; output_policy caught it late.
The same gap let --run-dir point INTO the preserved legacy run, which the
output policy permits (it is inside evidence_current) - recovery would then
write run_manifest.json / ledger / blocks into production_20260929/.

Every test here runs the REAL CLI in a subprocess from the root of a COPY
of the package (the Kaggle layout: cwd = repo root, relative paths), so the
real source tree is never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]                 # audit/cryptography
REAL_LEGACY = ROOT / "evidence_current" / "ce1" / "production_20260929"
CE1_REL = "audit/cryptography/evidence_current/ce1"
LEGACY_REL = f"{CE1_REL}/production_20260929"
NEW_REL = f"{CE1_REL}/production_20260930"
MODULE = "audit.cryptography.experiments.ce1.gohr_signal_destruction"

pytestmark = pytest.mark.skipif(not (REAL_LEGACY / "checkpoints").exists(),
                                reason="legacy run absent")


def _sha_tree(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture()
def repo(tmp_path):
    """A fresh repository root containing a copy of this package (no audit/__init__.py)."""
    r = tmp_path / "deep_speck"
    shutil.copytree(ROOT, r / "audit" / "cryptography",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    return r


def _cli(repo: Path, *args, timeout=600):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)                    # import the COPY, not the real tree
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"             # so ANY new file is a real write
    return subprocess.run([sys.executable, "-m", MODULE, *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _recover(repo, run_dir):
    return _cli(repo, "--recover-legacy-block0", "--run-dir", run_dir,
                "--legacy-run-dir", LEGACY_REL, "--sealed-source", f"{LEGACY_REL}/sealed")


def _all_manifests(repo):
    return sorted(str(p.relative_to(repo)) for p in repo.rglob("run_manifest.json"))


# ---------------------------------------------------------------- (a)(b)(c)

def test_explicit_versioned_run_dir_is_honoured_exactly(repo):
    """The exact Kaggle command: manifest lands in production_20260930, nowhere else."""
    legacy_before = _sha_tree(repo / LEGACY_REL)
    p = _recover(repo, NEW_REL)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    new = repo / NEW_REL
    # (a) honoured exactly - and reported
    assert f"run directory: {new.resolve()}" in p.stdout
    # (b) run_manifest.json created under that directory
    assert (new / "run_manifest.json").is_file()
    assert _all_manifests(repo) == [f"{NEW_REL}/run_manifest.json"]
    # (c) no repo-root manifest, and nothing written at the repo root or ce1/ root
    assert not (repo / "run_manifest.json").exists()
    assert sorted(x.name for x in repo.iterdir()) == ["audit"]
    assert sorted(x.name for x in (repo / CE1_REL).iterdir()) == \
        sorted(["local_preflight", "production_20260929", "production_20260930"])
    # recovery semantics intact: block0 seeded, not evaluated, decision recorded
    m = json.loads((new / "run_manifest.json").read_text())
    assert m["legacy_block0_decision"] == "RECOVER"
    for arm in ("baseline", "destroyed"):
        st = json.loads((new / "blocks/block0" / arm / "state.json").read_text())
        assert st["status"] == "TRAINING_COMPLETE" and st["evaluation"] is None
    ledger = [json.loads(l)["event"] for l in (new / "ledger.jsonl").read_text().splitlines()]
    for forbidden in ("TRAIN_START", "CHECKPOINT", "EVALUATION", "EVALUATED"):
        assert forbidden not in ledger                         # nothing trained or scored
    # the legacy run is byte-for-byte unchanged, with no files added
    assert _sha_tree(repo / LEGACY_REL) == legacy_before

    # --status and --resume-dry-run resolve the SAME explicit directory, even with a
    # decoy manifest planted at the repo root
    (repo / "run_manifest.json").write_text('{"decoy": true}')
    for flags in (["--status", "--run-dir", NEW_REL], ["--resume-dry-run", NEW_REL]):
        s = _cli(repo, *flags)
        assert s.returncode == 0, s.stdout[-2000:] + s.stderr[-2000:]
        assert f"run directory: {new.resolve()}" in s.stdout
        assert "LEGACY_RECOVERED_PENDING_EVALUATION" in s.stdout
        assert "decoy" not in s.stdout
    (repo / "run_manifest.json").unlink()


# ---------------------------------------------------------------- (d)

@pytest.mark.parametrize("value", ["", ".", "./", "   "])
def test_degenerate_run_dir_never_falls_back_to_cwd(repo, value):
    before = _sha_tree(repo)
    p = _recover(repo, value)
    assert p.returncode == 2, p.stdout + p.stderr                 # argparse usage error
    assert "explicit directory" in p.stderr and "working directory" in p.stderr
    assert "HistoricalWriteError" not in p.stderr                 # rejected BEFORE any write
    assert _all_manifests(repo) == [] and _sha_tree(repo) == before
    for flags in (["--status", "--run-dir", value], ["--resume-dry-run", value]):
        s = _cli(repo, *flags)
        assert s.returncode == 2 and "explicit directory" in s.stderr


def test_controller_refuses_degenerate_run_dir_directly(tmp_path, monkeypatch):
    """The same guarantee without the CLI: open_run / load_run_config never use '.'."""
    from audit.cryptography.experiments.ce1 import controller as C
    monkeypatch.chdir(tmp_path)
    cfg = C.CE1RunConfig.production_config()
    for bad in ("", ".", "./", Path(""), Path("."), None):
        with pytest.raises(C.RunDirectoryError, match="Refusing to fall back"):
            C.open_run(bad, cfg, repo_root=tmp_path)
        with pytest.raises(C.RunDirectoryError):
            C.load_run_config(bad)
    assert list(tmp_path.iterdir()) == []                        # nothing written anywhere


# ------------------------------------------------ misplaced run directories

def test_run_dir_pointing_at_legacy_run_is_refused_without_writing(repo):
    legacy_before = _sha_tree(repo / LEGACY_REL)
    for target in (LEGACY_REL, f"{LEGACY_REL}/nested_run"):
        p = _recover(repo, target)
        assert p.returncode == 1, p.stdout + p.stderr
        assert "RECOVERY REFUSED" in p.stdout and "legacy run" in p.stdout
    assert _sha_tree(repo / LEGACY_REL) == legacy_before         # no file added or changed
    assert _all_manifests(repo) == []


@pytest.mark.parametrize("target", [
    "audit/cryptography/evidence_current",                       # evidence root
    CE1_REL,                                                     # ce1 root (contains legacy)
    f"{CE1_REL}/runs/production_20260930",                       # nested one level too deep
    "production_20260930",                                       # repo root child
])
def test_production_run_dir_must_be_direct_child_of_ce1(repo, target):
    before = _sha_tree(repo)
    p = _recover(repo, target)
    assert p.returncode == 1 and "RECOVERY REFUSED" in p.stdout, p.stdout + p.stderr
    assert _all_manifests(repo) == [] and _sha_tree(repo) == before


def test_controller_refuses_non_empty_directory_without_manifest(tmp_path):
    from audit.cryptography.experiments.ce1 import controller as C
    cfg = C.CE1RunConfig.production_config()
    d = tmp_path / CE1_REL / "production_20260930"
    d.mkdir(parents=True)
    (d / "something.bin").write_bytes(b"x")
    with pytest.raises(C.RunDirectoryError, match="may be a preserved legacy run"):
        C.open_run(d, cfg, repo_root=tmp_path)
    assert not (d / "run_manifest.json").exists()
