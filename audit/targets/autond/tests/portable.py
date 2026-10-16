"""
Portable test fixtures for the AutoND implementation-audit tests.

NO TEST IN THIS PACKAGE MAY HARDCODE A MACHINE PATH. An earlier revision
pinned the build container's own checkout directory into a test module as
a literal absolute path, so the suite passed only on that container and
reported failures everywhere else that looked like defects in the code.
`test_corrections.TestRunManifestBindsGenericHelpers` enforces this
across every module in the package, including this one, which is why no
example of such a path is written out here.

Two kinds of fixture live here.

  SYNTHETIC (always available). `fake_upstream()` builds a git
  repository with the directory shape of the target but with file
  contents of its own. Every negative test - wrong digest, wrong commit,
  absent file, torn manifest - runs against this, on any machine, with no
  external dependency. A negative test must never be skipped: those are
  the tests that catch a gate failing open.

  REAL (opt-in). Tests that need digests to actually match need the
  pinned checkout, which this package cannot carry. `real_upstream()`
  returns it when the operator points at it, via the AUTOND_ROOT
  environment variable or an `--autond-root`-style path file, and returns
  None otherwise, so those tests SKIP with a message naming what to set
  rather than failing.
"""

from __future__ import annotations

import json
import shutil
import hashlib
import os
import subprocess
import unittest
from pathlib import Path
from typing import Optional

#: Environment variables the operator may set to point the suite at real
#: artifacts. Checked in order.
UPSTREAM_ENV = ("AUTOND_ROOT", "AUTOND_UPSTREAM", "AUTOND_SRC")
REPO_ENV = ("CIPHERMIND_REPO_ROOT", "AUDIT_REPO_ROOT")


def repo_root() -> Path:
    """
    The audit repository root, derived from this file's own location.

    tests/ -> autond/ -> targets/ -> audit/ -> <repo root>
    """
    for var in REPO_ENV:
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def real_upstream() -> Optional[Path]:
    """The pinned checkout if the operator pointed at one, else None."""
    for var in UPSTREAM_ENV:
        v = os.environ.get(var)
        if not v:
            continue
        p = Path(v).expanduser()
        if p.is_dir():
            return p.resolve()
    return None


def require_real_upstream(test: unittest.TestCase) -> Path:
    """Return the pinned checkout or skip the test explaining how to supply it."""
    up = real_upstream()
    if up is None:
        test.skipTest(
            "needs the pinned AutoND checkout; set AUTOND_ROOT=/path/to/AutoND "
            "to run this test. Skipped rather than failed because the checkout "
            "cannot be shipped with the audit.")
    return up


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("GIT_AUTHOR_NAME", "audit-test")
    env.setdefault("GIT_AUTHOR_EMAIL", "audit-test@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "audit-test")
    env.setdefault("GIT_COMMITTER_EMAIL", "audit-test@example.invalid")
    return subprocess.run(["git", "-C", str(cwd)] + args,
                          capture_output=True, text=True, env=env, timeout=60)


def git_available() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:                                               # noqa: BLE001
        return False


def fake_upstream(root: Path, *, commit: bool = True) -> dict:
    """
    Build a synthetic 'upstream' checkout.

    It has the shape of the target - the source files and the
    model_weights directory the plan names - with contents of its own, so
    every recorded digest MISMATCHES by construction. Returns
    {"path", "head"} where head is the real commit of this fake
    repository (never the pinned commit).
    """
    root = Path(root)
    (root / "model_weights").mkdir(parents=True, exist_ok=True)
    for name in ("eval_nets.py", "dbitnet.py", "main.py", "optimizer.py"):
        (root / name).write_text(f"# synthetic stand-in for {name}\n")
    for name in ("speck3264_single-key_0x400000_round8_polished.h5",
                 "simon3264_single-key_0x400_round11_polished.h5"):
        (root / "model_weights" / name).write_bytes(b"not a real checkpoint\n")

    head = ""
    if commit and git_available():
        _git(["init", "-q"], root)
        _git(["add", "-A"], root)
        _git(["commit", "-q", "-m", "synthetic upstream"], root)
        res = _git(["rev-parse", "HEAD"], root)
        head = res.stdout.strip()
    return {"path": root, "head": head}


def fake_search_bundle(root: Path, *, status: str = "completed",
                       differences: Optional[list] = None,
                       torn_log: bool = False,
                       wrong_commit: bool = False,
                       behaviour_changing_patch: bool = False,
                       drop_manifest_entry: bool = False,
                       highest_non_random_round: int = 5) -> Path:
    """
    Build a synthetic R1 search output directory in the shape the search
    harness validates, with switches for each way it can be malformed.

    Used to prove that a malformed bundle is NOT read as a valid search
    result. The content is synthetic; no search is ever run.
    """
    import hashlib

    root = Path(root)
    run = root / "run_000"
    run.mkdir(parents=True, exist_ok=True)

    pinned = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"
    result = {
        "status": status,
        "elapsed_seconds": 12.5,
        "highest_non_random_round": highest_non_random_round,
        "returned_differences_hex": (differences if differences is not None
                                     else ["4194304"]),
        "config": {"mode": "r1_readme_demo", "num_generations": 5,
                   "num_samples": 1000, "population_init": "random"},
        "source": {"head_commit": ("0" * 40 if wrong_commit else pinned)},
        "tensorflow_observed": {"tensorflow": False},
        "reproducibility": {"bit_reproducible": False},
        "patches": ([{"patch_id": "P-BEHAVIOUR", "changes_behaviour": True}]
                    if behaviour_changing_patch else []),
        "scoring_calls_logged": 3,
    }
    (run / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    seqs = [0, 1, 3] if torn_log else [0, 1, 2]
    lines = [json.dumps({"seq": s, "event": "evo_enter"}) for s in seqs]
    (run / "generations.jsonl").write_text("\n".join(lines) + "\n")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(root))] = {"sha256": sha(p)}
    if drop_manifest_entry and files:
        files.pop(sorted(files)[0])
    (root / "MANIFEST.json").write_text(
        json.dumps({"files": files}, indent=2, sort_keys=True))
    return root


def simulate_frozen(plan_mod, repo_root: Path, *, plan_hash: Optional[str] = None,
                    frozen_at: str = "2026-01-01T00:00:00Z",
                    commit_sidecar: bool = True,
                    tamper_sidecar_after_commit: bool = False) -> dict:
    """
    Create a GENUINELY frozen state in a temporary git repository.

    This does not stub out the gate. It performs the documented freeze
    procedure for real - commit the plan file, read back its commit,
    write the sidecar, commit the sidecar - inside a temp repo, and
    repoints only `FROZEN_SIDECAR` at the sidecar it wrote. `is_frozen()`
    then runs its full set of checks against real git objects, so a test
    built on this exercises the gate rather than bypassing it.

    The plan file copied in is byte-identical to the live module, so the
    `plan_file_sha256` the sidecar records matches both the committed blob
    and the file on disk.

    The caller must restore `plan_mod.FROZEN_SIDECAR` afterwards; the
    returned dict carries the original.
    """
    rel = "audit/targets/autond/preregistrations/autond_ii_v2.py"
    side_rel = "audit/targets/autond/preregistrations/autond_ii_v2.frozen.json"
    root = Path(repo_root)
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(Path(plan_mod.__file__).read_bytes())

    _git(["init", "-q"], root)
    _git(["add", rel], root)
    _git(["commit", "-q", "-m", "freeze plan"], root)
    head = _git(["rev-parse", "HEAD"], root).stdout.strip()

    rec = {
        "preregistration": plan_mod.PREREGISTRATION_ID,
        "plan_hash": plan_hash or plan_mod.plan_hash(),
        "frozen_at_utc": frozen_at,
        "plan_file_commit": head,
        "plan_file_path": rel,
        "plan_file_sha256": plan_mod.plan_file_sha256(),
        "approved_by": "portable-test-fixture",
    }
    sidecar = root / side_rel
    sidecar.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    if commit_sidecar:
        _git(["add", side_rel], root)
        _git(["commit", "-q", "-m", "record freeze"], root)
    if tamper_sidecar_after_commit:
        rec2 = dict(rec)
        rec2["approved_by"] = "someone else"
        sidecar.write_text(json.dumps(rec2, indent=2, sort_keys=True) + "\n")

    original = plan_mod.FROZEN_SIDECAR
    plan_mod.FROZEN_SIDECAR = sidecar
    return {"repo_root": root, "sidecar": sidecar, "record": rec,
            "plan_file_commit": head, "original_sidecar": original}


def write_npz(path: Path, accs, tprs=None, tnrs=None) -> Path:
    """
    Write a real NPZ in the shape the target writes.

    The numbers are invented, which is what makes this a fixture and not
    evidence. What is real is the FILE: the assembler and the validator
    both load it with numpy, so a test using this exercises the actual
    read path rather than a stubbed one.
    """
    import numpy as np

    n = len(accs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, accs=np.asarray(accs, dtype=float),
             tprs=np.asarray(tprs if tprs is not None else [0.47] * n, dtype=float),
             tnrs=np.asarray(tnrs if tnrs is not None else [0.55] * n, dtype=float))
    return Path(path)


def staged_run_dir(root: Path, upstream: Path, plan_mod, *,
                   n_sets: Optional[int] = None,
                   invocations: Optional[int] = None,
                   suffixes: Optional[list] = None,
                   returncode: int = 0,
                   attestation: str = "tracked-clean;untracked-benign=0",
                   commit: Optional[str] = None,
                   started: str = "2026-02-01T00:00:00Z",
                   finished: str = "2026-02-01T01:00:00Z",
                   claimed_accuracies: Optional[dict] = None,
                   identical_npzs: bool = False,
                   only: Optional[tuple] = None) -> Path:
    """
    Build a run directory shaped exactly as the gated launcher leaves one:
    real checkpoint copies taken from the pinned checkout, real NPZs, and
    one journal entry per INVOCATION.

    `evaluation_assemble.assemble()` then derives RESULT.json from it, so
    the result record is never written by hand in any test.

    The defaults follow the REGISTERED design, read from `plan_mod` rather
    than hardcoded: `invocations` defaults to
    plan.INVOCATIONS_PER_CHECKPOINT and `n_sets` - the sets within one
    invocation - to plan.SETS_PER_INVOCATION. A caller that wants the
    pre-r3d single-invocation shape, or any other malformed shape, asks
    for it explicitly; that way raising the registered sample size does
    not silently turn every success-path fixture into a short sample.

    Deliberately malformable, for the tests that prove assembly refuses:
      invocations=1     one surviving invocation (half the design)
      n_sets=3          an invocation whose NPZ is short
      identical_npzs    both invocations carrying the same accuracies,
                        which os.urandom draws would never produce
      suffixes=[..]     wrong or colliding --add_str values
    """
    root = Path(root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "journal").mkdir(parents=True, exist_ok=True)

    per_inv = plan_mod.SETS_PER_INVOCATION if n_sets is None else n_sets
    n_inv = (plan_mod.INVOCATIONS_PER_CHECKPOINT if invocations is None
             else invocations)
    sufs = list(plan_mod.INVOCATION_SUFFIXES) if suffixes is None else list(suffixes)
    if len(sufs) < n_inv:                      # pad so a caller can ask for more
        sufs = sufs + [f"_x{i}" for i in range(len(sufs), n_inv)]

    base = {"speck3264": 0.5144, "simon3264": 0.5179}
    env = {"python": "3.11.17", "tensorflow": "2.15.1", "numpy": "1.26.4",
           "keras": "2.15.0"}

    for cid, rec in sorted(plan_mod.TARGET["checkpoints"].items()):
        if only is not None and cid not in only:
            continue
        src = Path(upstream) / rec["path"]
        dst = root / "checkpoints" / Path(rec["path"]).name
        shutil.copy2(src, dst)
        h = hashlib.sha256(dst.read_bytes()).hexdigest()

        for idx in range(n_inv):
            suf = sufs[idx]
            # Offsets shifted per invocation so the two NPZs differ, as
            # independent os.urandom draws would. identical_npzs=True
            # suppresses the shift to exercise the duplicate check.
            shift = 0 if identical_npzs else idx * per_inv
            accs = [base[cid] + (i + shift - 2) * 1e-5 for i in range(per_inv)]
            npz = write_npz(
                Path(str(dst).replace(".h5", f"_eval{suf}.npz")), accs)
            inv_id = f"{cid}{suf}"
            (root / "logs" / f"{inv_id}.log").write_text("synthetic log\n")

            journal = {
                "checkpoint_id": cid,
                "invocation_id": inv_id,
                "invocation_index": idx,
                "invocation_suffix": suf,
                "sets_this_invocation": per_inv,
                "command": ["python", "eval_nets.py", "--model_path", str(dst),
                            "--add_str", suf],
                "cwd": str(upstream),
                "started_utc": started,
                "finished_utc": finished,
                "returncode": returncode,
                "model_copy": str(dst.relative_to(root)),
                "model_copy_sha256": h,
                "npz": str(npz.relative_to(root)),
                "log": f"logs/{inv_id}.log",
                "upstream_commit": commit or plan_mod.TARGET["pinned_commit"],
                "upstream_write_check": attestation,
                "environment": env,
            }
            if claimed_accuracies and cid in claimed_accuracies:
                journal["observed_accuracies"] = claimed_accuracies[cid]
            (root / "journal" / f"exec_{inv_id}.json").write_text(
                json.dumps(journal, indent=2, sort_keys=True))
    return root


class force_unfrozen:
    """
    Context manager that makes the preregistration read as unfrozen.

    Several tests exist to prove the gate REFUSES while the plan is
    unfrozen. Written against the module's real state they passed only
    while the operator had not yet frozen anything - so performing the
    documented freeze, a correct act, broke them. That made the suite a
    hazard rather than a check.

    This repoints FROZEN_SIDECAR at a path that does not exist, so the
    unfrozen branch is exercised deterministically whatever the operator
    has done in their own repository. The real state is restored on exit.
    """

    def __init__(self, plan_mod):
        self.plan = plan_mod
        self._orig = None
        self._tmp = None

    def __enter__(self):
        import tempfile

        self._orig = self.plan.FROZEN_SIDECAR
        self._tmp = tempfile.mkdtemp(prefix="autond-unfrozen-")
        self.plan.FROZEN_SIDECAR = Path(self._tmp) / "absent.frozen.json"
        return self.plan

    def __exit__(self, *exc):
        self.plan.FROZEN_SIDECAR = self._orig
        return False
