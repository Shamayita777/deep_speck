"""Fail-closed behaviour of the Gohr D3/D1 runbook (audit/dataset/tools/d3_gohr_runbook.zsh).

The script is executed in subprocesses WITHOUT a terminal, with a temporary log
directory. No stage here can stage, commit or run production work: every case
asserts that the git index stays empty and HEAD is unchanged.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "audit" / "dataset" / "tools" / "d3_gohr_runbook.zsh"
ZSH = shutil.which("zsh")
pytestmark = pytest.mark.skipif(ZSH is None, reason="zsh not installed (SKIPPED IS NOT PASSED)")
GATED = {"commit": "COMMIT GOHR D3 PREREGISTRATION", "calibrate": "RUN GOHR CALIBRATION $PC",
         "d0": "BUILD GOHR D0 $PC", "d1": "RUN GOHR D1 $PC", "d3": "RUN GOHR D3 PRODUCTION $PC"}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo_unchanged():
    head, index = _git("rev-parse", "HEAD"), _git("diff", "--cached", "--name-only")
    assert index == "", "precondition: the index is empty"
    yield
    assert _git("rev-parse", "HEAD") == head and _git("diff", "--cached", "--name-only") == ""


def _run(*args, runlog: Path, env_over=None, script=SCRIPT, stdin=""):
    env = dict(os.environ, GOHR_RUNLOG=str(runlog), **(env_over or {}))
    return subprocess.run([ZSH, str(script), *args], cwd=REPO, env=env, input=stdin, capture_output=True,
                          text=True, timeout=300)


def test_script_syntax() -> None:
    assert subprocess.run([ZSH, "-n", str(SCRIPT)], capture_output=True).returncode == 0


@pytest.mark.parametrize("args, env_over, message", [
    ((), None, "usage"),
    (("no_such_stage",), None, "unknown stage"),
    (("content",), {"GOHR_PY": "/nonexistent/python"}, "interpreter /nonexistent/python not found"),
])
def test_setup_failures_exit_immediately(tmp_path, repo_unchanged, args, env_over, message) -> None:
    r = _run(*args, runlog=tmp_path / "logs", env_over=env_over)
    assert r.returncode == 1 and f"STOP: " in r.stderr and message in r.stderr
    assert "STAGE" not in r.stdout


def test_unwritable_log_directory_exits_immediately(tmp_path, repo_unchanged) -> None:
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    r = _run("content", runlog=blocker / "logs")
    assert r.returncode == 1 and "cannot create log directory" in r.stderr


def test_a_copy_outside_the_repository_refuses(tmp_path, repo_unchanged) -> None:
    copy = tmp_path / "a" / "b" / "c" / "d3_gohr_runbook.zsh"
    copy.parent.mkdir(parents=True)
    shutil.copy(SCRIPT, copy)
    r = _run("content", runlog=tmp_path / "logs", script=copy)
    assert r.returncode == 1 and "is not the repository root" in r.stderr


def test_content_refuses_the_unapproved_draft_and_writes_no_marker(tmp_path, repo_unchanged) -> None:
    import json
    draft = json.loads((REPO / "audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json").read_text())
    if draft["approved"] is True:
        assert draft["approved"] is True
        return
    logs = tmp_path / "logs"
    r = _run("content", runlog=logs)
    assert r.returncode == 1 and "content failed (exit 1)" in r.stderr
    assert not (logs / "content.ok").exists()


def test_commit_without_prerequisite_markers_is_refused(tmp_path, repo_unchanged) -> None:
    r = _run("commit", runlog=tmp_path / "logs")
    assert r.returncode == 1 and "prerequisite stage 'content' has not passed" in r.stderr


def _files_state() -> str:
    files = re.search(r"FILES=\((.*?)\)", SCRIPT.read_text(), re.S).group(1).split()
    out = subprocess.run(f"shasum -a 256 {' '.join(files)} | shasum -a 256 | cut -d' ' -f1", shell=True, cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    return out


def test_commit_gate_denies_without_an_interactive_terminal_even_with_the_phrase(tmp_path, repo_unchanged) -> None:
    """Valid prerequisite markers, the exact phrase piped on stdin, no terminal: default deny, nothing staged."""
    logs = tmp_path / "logs"
    logs.mkdir()
    prereg_sha = subprocess.run(["shasum", "-a", "256", "audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json"],
                                cwd=REPO, capture_output=True, text=True, check=True).stdout.split()[0]
    (logs / "content.ok").write_text(prereg_sha + "\n")
    (logs / "tests.ok").write_text(_files_state() + "\n")
    (logs / "check.ok").write_text(_files_state() + "\n")
    r = _run("commit", runlog=logs, stdin="COMMIT GOHR D3 PREREGISTRATION\n")
    assert r.returncode == 1 and "authorisation requires an interactive terminal; nothing was done" in r.stderr
    assert "Will stage exactly:" in r.stdout and "AUTHORISED" not in r.stdout


def test_a_stale_marker_does_not_unlock_the_commit(tmp_path, repo_unchanged) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "content.ok").write_text("0" * 64 + "\n")
    r = _run("commit", runlog=logs)
    assert r.returncode == 1 and "prerequisite stage 'content' has not passed for the current state" in r.stderr


@pytest.mark.parametrize("stage", ["anchor", "calibrate", "calverify", "d0", "d1", "preflight", "d3", "verify"])
def test_post_commit_stages_refuse_before_the_preregistration_is_committed(tmp_path, repo_unchanged, stage) -> None:
    if _git("log", "-1", "--format=%h", "--", "audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json"):
        pytest.skip("the preregistration is committed; this pre-commit refusal no longer applies")
    r = _run(stage, runlog=tmp_path / "logs")
    assert r.returncode == 1 and "the preregistration is not committed" in r.stderr


def test_every_gated_stage_authorises_before_running_anything() -> None:
    text = SCRIPT.read_text()
    for stage, phrase in GATED.items():
        block = re.search(rf"\n{stage}\)\n(.*?)\n  ;;", text, re.S).group(1)
        lines = [ln.strip() for ln in block.splitlines()]
        auth = next(i for i, ln in enumerate(lines) if ln.startswith(f'authorize "{phrase}"'))
        first_effect = next(i for i, ln in enumerate(lines) if ln.startswith(("run ", "git add")))
        assert auth < first_effect, stage
    for stage in ("content", "tests", "check", "anchor", "calverify", "preflight", "verify"):
        block = re.search(rf"\n{stage}\)\n(.*?)\n  ;;", text, re.S).group(1)
        assert "authorize" not in block and "git add" not in block and "git commit" not in block, stage


def test_production_stages_pass_the_d1_evidence_to_the_runner() -> None:
    text = SCRIPT.read_text()
    for stage in ("preflight", "d3"):
        block = re.search(rf"\n{stage}\)\n(.*?)\n  ;;", text, re.S).group(1)
        assert "--d1-dir $D1OUT" in block, stage
    assert "git add -- $FILES" in text and "git add ." not in text and "git add -A" not in text
