#!/usr/bin/env python3
"""
The gated launcher for the shipped-checkpoint evaluation.

This is the ONLY module in this package that would start the
preregistered evaluation. It refuses unless every one of the following
holds, each established by computation rather than by a recorded claim:

  1. the successor preregistration is frozen, with a sidecar binding the
     live plan hash AND the plan file's own bytes, committed and matching
     the repository version;
  2. every source digest the plan records is recomputed and matches;
  3. every checkpoint digest the plan records is recomputed and matches;
  4. the checkout's HEAD is the pinned commit;
  5. the target's own command-line interface is read from its source and
     the commands this module would run are validated against it;
  6. the required interpreter and library versions are present;
  7. the output directory lies OUTSIDE the checkout and is empty or absent;
  8. the checkout has no modification to a tracked file.

NOTHING RUNS BY DEFAULT. `preflight()` only reads. `launch()` executes
only when preflight returns no blockers AND the caller passes
execute=True; the CLI additionally requires
`--i-have-reviewed-the-preregistration`.

THE UPSTREAM CHECKOUT IS READ-ONLY, AND STAYS THAT WAY
    `eval_nets.py` derives its output path from the model path it is
    handed (`model_path.replace('.h5', '_eval.npz')`), so the checkpoints
    are COPIED into the run directory and the copies are evaluated. The
    target's cwd must be the checkout, because its cipher discovery uses
    a relative glob, but every path it is given to write points outside.

    This module never deletes, moves or modifies anything in the
    checkout. See `checkout_state()` for how an untracked file in the
    checkout is handled without touching it.

WHY THE JOURNAL
    Each execution writes a journal entry before and after it runs,
    holding the command, cwd, timestamps, exit status, the digest of the
    copy evaluated, the NPZ path, the commit and the environment of the
    process that actually ran. `evaluation_assemble` derives RESULT.json
    from those entries plus the artifacts, so no field of the result
    record is ever written by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from audit.targets.autond import upstream_interface as ui
from audit.targets.autond.preregistrations import autond_ii_v2 as plan

GATE_VERSION = "autond-evaluation-gate-v2"

#: Untracked paths in the checkout that are operating-system or
#: interpreter debris rather than content. Their presence is RECORDED and
#: does not block; they are never deleted. See checkout_state().
BENIGN_UNTRACKED = (
    ".DS_Store", "._.DS_Store", "Thumbs.db", "desktop.ini",
    "__pycache__/", ".ipynb_checkpoints/", ".pytest_cache/",
    ".idea/", ".vscode/", ".mypy_cache/", ".ruff_cache/",
)
BENIGN_SUFFIXES = (".pyc", ".pyo", ".swp", ".orig")

#: The exact attestation grammar the result schema accepts. Written by
#: this module, validated by evaluation_results.validate_bundle.
#:   tracked-clean;untracked-benign=0
#:   tracked-clean;untracked-benign=2:.DS_Store,__pycache__/
ATTESTATION_CLEAN = "tracked-clean;untracked-benign="


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_benign(path: str) -> bool:
    p = path.strip().strip('"')
    base = p.rstrip("/").split("/")[-1]
    if p in BENIGN_UNTRACKED or base in (b.rstrip("/") for b in BENIGN_UNTRACKED):
        return True
    if any(seg.rstrip("/") in (b.rstrip("/") for b in BENIGN_UNTRACKED)
           for seg in p.split("/") if seg):
        return True
    return any(p.endswith(s) for s in BENIGN_SUFFIXES)


def checkout_state(upstream: Path) -> dict:
    """
    Classify the checkout's working-tree state WITHOUT changing it.

    The pinned target is read-only, so a stray file in it is a fact to be
    recorded and judged, never something to clean up. Three categories:

      tracked_dirty     a tracked file is modified, staged, deleted or
                        renamed. BLOCKING: the target is no longer the
                        pinned revision, and every digest the plan
                        records describes a file that may have changed.
      untracked_benign  operating-system or interpreter debris such as
                        .DS_Store or __pycache__/. NOT blocking. It
                        cannot change what the target computes, it is
                        recorded in the attestation, and it is left
                        exactly where it is. A `.DS_Store` beside the
                        checkpoints does not make the checkpoints
                        different files.
      untracked_other   anything else. BLOCKING, because an untracked
                        file in the checkout CAN change behaviour: a
                        stray `optimizer.py`, `ciphers/speck3264.py` or
                        `sitecustomize.py` shadows an import, and the
                        cipher list is built by globbing `ciphers/*.py`,
                        so an extra file there is read by the target.

    Returns the lists and the attestation string. Nothing is removed.
    """
    out: dict = {"tracked_dirty": [], "untracked_benign": [],
                 "untracked_other": [], "attestation": None, "error": None}
    try:
        res = subprocess.run(
            ["git", "-C", str(upstream), "status", "--porcelain",
             "--untracked-files=normal"],
            capture_output=True, text=True, timeout=60)
    except Exception as exc:                                        # noqa: BLE001
        out["error"] = f"git could not be run: {type(exc).__name__}: {exc}"
        return out
    if res.returncode != 0:
        out["error"] = f"git status failed: {res.stderr.strip()}"
        return out

    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        if code == "??":
            (out["untracked_benign"] if _is_benign(path)
             else out["untracked_other"]).append(path)
        else:
            out["tracked_dirty"].append(f"{code.strip()} {path}")

    if not out["tracked_dirty"]:
        n = len(out["untracked_benign"])
        names = ",".join(sorted(out["untracked_benign"]))
        out["attestation"] = (f"{ATTESTATION_CLEAN}{n}"
                              + (f":{names}" if n else ""))
    return out


def _versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}."
                  f"{sys.version_info.micro}"}
    for mod in ("tensorflow", "numpy", "keras"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", None)
        except Exception:                                           # noqa: BLE001
            out[mod] = None
    return out


def run_dir_is_usable(output_dir: Path) -> tuple[bool, str]:
    """
    A run directory must be absent or empty.

    Revision 2 would have written straight into an existing directory,
    overwriting the checkpoints, logs and NPZs of a previous run. Prior
    evidence is never overwritten: a non-empty directory is refused and
    the operator chooses a new one.
    """
    d = Path(output_dir)
    if not d.exists():
        return True, "absent; will be created"
    if not d.is_dir():
        return False, f"{d} exists and is not a directory"
    entries = [p.name for p in d.iterdir()]
    if entries:
        return False, (f"{d} already exists and is not empty "
                       f"({len(entries)} entries, e.g. {sorted(entries)[:4]}). "
                       "Refusing to write into it: a previous run's "
                       "checkpoints, logs, NPZs or RESULT.json would be "
                       "overwritten. Choose a new run directory; prior "
                       "evidence is never replaced.")
    return True, "exists and is empty"


def planned_commands(upstream: Path, output_dir: Path, *,
                     argspec: Optional[dict] = None,
                     python: Optional[str] = None) -> list[dict]:
    """
    The commands a permitted run WOULD execute, built against the pinned
    target's own argument definitions and validated before being returned.

    Exposed so a reviewer can read the exact command without the gate
    opening, and so the plan's audit trail records what was authorised.
    """
    up, out = Path(upstream), Path(output_dir)
    spec = argspec
    if spec is None:
        try:
            spec = ui.extract_argspec(up)
        except FileNotFoundError:
            spec = ui.EVAL_NETS_ARGSPEC
    exe = python or sys.executable

    cmds: list[dict] = []
    for cid, rec in sorted(plan.TARGET["checkpoints"].items()):
        copy_to = out / "checkpoints" / Path(rec["path"]).name

        # The registered design is n = 10 per checkpoint. The target's
        # nEval = 5 is a module constant, not an argument, so the only way
        # to reach ten sets without editing the pinned repository is to
        # invoke it once per registered suffix. --add_str changes nothing
        # but the output filename (eval_nets.py:152), so the two
        # invocations are the same experiment twice.
        for idx, suffix in enumerate(plan.INVOCATION_SUFFIXES):
            cmd = ui.build_eval_command(
                python=exe,
                model_path=copy_to,
                cipher=rec["cipher"],
                model_type=rec["model_type"],
                scenario=rec["scenario"],
                input_difference=rec["input_difference"],
                round_number=int(rec["round_number"]),
                add_str=suffix,
                argspec=spec,
            )
            cmds.append({
                "checkpoint": cid,
                "invocation_index": idx,
                "invocation_suffix": suffix,
                "invocation_id": f"{cid}{suffix}",
                "sets_this_invocation": plan.SETS_PER_INVOCATION,
                "copy_from": str(up / rec["path"]),
                "copy_to": str(copy_to),
                "expected_sha256": rec["sha256"],
                "command": cmd,
                "cwd": str(up),
                "npz": str(ui.npz_name_for(copy_to, suffix)),
                "note": ("cwd is the checkout because the target's cipher "
                         "discovery globs 'ciphers/*.py' relatively; every path "
                         "it is given to write points outside the checkout"),
            })
    return cmds


def preflight(upstream: Path, output_dir: Path, *,
              repo_root: Optional[Path] = None,
              check_versions: bool = True) -> dict:
    """Read-only. Returns every blocker at once rather than the first."""
    up = Path(upstream).expanduser().resolve()
    out_d = Path(output_dir).expanduser().resolve()
    blockers: list[str] = []
    detail: dict = {"gate": GATE_VERSION, "upstream": str(up),
                    "output_dir": str(out_d), "plan_hash": plan.plan_hash()}

    permitted, plan_blockers = plan.confirmatory_run_permitted(up, repo_root)
    blockers.extend(plan_blockers)
    detail["preregistration_permitted"] = permitted

    iface = ui.verify_argspec(up if up.is_dir() else None)
    detail["interface"] = {"verified": iface["verified"],
                           "source_sha256": iface["source_sha256"],
                           "problems": iface["problems"]}
    if not iface["verified"]:
        blockers.extend(f"target interface: {p}" for p in iface["problems"])

    try:
        cmds = planned_commands(up, out_d,
                                argspec=iface.get("observed") or None)
        detail["planned_commands"] = cmds
    except Exception as exc:                                        # noqa: BLE001
        cmds = []
        detail["planned_commands"] = []
        blockers.append(f"the evaluation command could not be built against the "
                        f"target's interface: {type(exc).__name__}: {exc}")

    versions = _versions()
    detail["versions"] = versions
    if check_versions:
        want = plan.DESIGN["environment"]
        py = versions["python"] or ""
        if not py.startswith("3.11"):
            blockers.append(
                f"python {py} is not {want['python']}; the plan fixes the "
                "interpreter because the target is incompatible with newer Keras")
        for lib in ("tensorflow", "numpy"):
            got, wanted = versions.get(lib), want.get(lib)
            if got is None:
                blockers.append(f"{lib} is not importable; the plan requires {wanted}")
            elif wanted and got != wanted:
                blockers.append(f"{lib} {got} != the plan's required {wanted}")

    try:
        out_d.relative_to(up)
        blockers.append(
            f"output directory {out_d} is inside the upstream checkout {up}. "
            "The target writes its NPZ beside the model path it is given, so "
            "this would write into the checkout.")
    except ValueError:
        pass

    usable, why = run_dir_is_usable(out_d)
    detail["run_dir"] = why
    if not usable:
        blockers.append(why)

    if up.is_dir():
        state = checkout_state(up)
        detail["checkout_state"] = state
        if state["error"]:
            blockers.append(f"upstream checkout: {state['error']}")
        if state["tracked_dirty"]:
            blockers.append(
                f"upstream checkout has modified tracked file(s): "
                f"{state['tracked_dirty'][:5]}. The target is no longer the "
                "pinned revision. Nothing here will modify or restore it; "
                "resolve it yourself.")
        if state["untracked_other"]:
            blockers.append(
                f"upstream checkout has untracked file(s) that are not OS "
                f"debris: {state['untracked_other'][:5]}. An untracked file in "
                "the checkout can change what the target computes - a stray "
                "module shadows an import, and the cipher list is built by "
                "globbing ciphers/*.py. Nothing here will delete them; move "
                "them out yourself, or confirm they are inert.")
        if state["untracked_benign"]:
            detail["benign_untracked_note"] = (
                f"{len(state['untracked_benign'])} untracked OS/interpreter "
                f"file(s) present and LEFT IN PLACE: "
                f"{sorted(state['untracked_benign'])}. These do not block and "
                "are recorded in the attestation.")
        detail["upstream_write_check"] = state["attestation"]
    else:
        detail["checkout_state"] = None
        detail["upstream_write_check"] = None
        blockers.append(f"no checkout at {up}")

    detail["permitted"] = not blockers
    detail["blockers"] = blockers
    return detail


def launch(upstream: Path, output_dir: Path, *, execute: bool = False,
           repo_root: Optional[Path] = None) -> int:
    """Launch the preregistered evaluation, or refuse with reasons."""
    pf = preflight(upstream, output_dir, repo_root=repo_root)
    print(f"{GATE_VERSION}")
    print(f"  plan hash      {pf['plan_hash']}")
    print(f"  upstream       {pf['upstream']}")
    print(f"  output dir     {pf['output_dir']}  ({pf.get('run_dir')})")
    print(f"  interface      verified={pf['interface']['verified']}")
    print(f"  versions       {pf['versions']}")
    print(f"  write check    {pf.get('upstream_write_check')}")
    if pf.get("benign_untracked_note"):
        print(f"  note           {pf['benign_untracked_note']}")
    print(f"  permitted      {pf['permitted']}")

    for c in pf.get("planned_commands") or []:
        print(f"    {c.get('invocation_id', c['checkpoint'])}: "
              f"{' '.join(c['command'])}")

    if pf["blockers"]:
        print(f"\n  REFUSING TO LAUNCH - {len(pf['blockers'])} blocker(s):")
        for b in pf["blockers"]:
            print(f"    - {b}")
        print("\n  No evaluation was started. Nothing was written. The "
              "checkout was not modified.")
        return 1

    if not execute:
        print("\n  execute=False: the gate is open but nothing was run.")
        return 0

    out = Path(output_dir).expanduser().resolve()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "journal").mkdir(parents=True, exist_ok=True)
    env_record = _versions()

    for c in pf["planned_commands"]:
        cid = c["checkpoint"]
        inv = c.get("invocation_id", cid)
        src, dst = Path(c["copy_from"]), Path(c["copy_to"])
        if not dst.exists():
            shutil.copy2(src, dst)
        # Re-hashed before EVERY invocation, not once per checkpoint: the
        # pooling assumption requires that both invocations evaluated the
        # same weights, so each one carries its own verification.
        got = _sha256_file(dst)
        if got != c["expected_sha256"]:
            print(f"  ABORT: copy of {cid} hashes to {got}, not "
                  f"{c['expected_sha256']}")
            return 1

        started = utc_now()
        log = out / "logs" / f"{inv}.log"
        print(f"  running {inv}...")
        with log.open("wb") as fh:
            res = subprocess.run(c["command"], cwd=c["cwd"], stdout=fh,
                                 stderr=subprocess.STDOUT)
        finished = utc_now()

        state = checkout_state(Path(upstream))
        journal = {
            "checkpoint_id": cid,
            "invocation_id": inv,
            "invocation_index": c.get("invocation_index", 0),
            "invocation_suffix": c.get("invocation_suffix", ""),
            "sets_this_invocation": c.get("sets_this_invocation",
                                          plan.SETS_PER_INVOCATION),
            "command": c["command"],
            "cwd": c["cwd"],
            "started_utc": started,
            "finished_utc": finished,
            "returncode": res.returncode,
            "model_copy": str(dst.relative_to(out)),
            "model_copy_sha256": got,
            "npz": str(Path(c["npz"]).relative_to(out)),
            "log": str(log.relative_to(out)),
            "upstream_commit": plan.TARGET["pinned_commit"],
            "upstream_write_check": state["attestation"],
            "environment": env_record,
            "gate": GATE_VERSION,
        }
        (out / "journal" / f"exec_{inv}.json").write_text(
            json.dumps(journal, indent=2, sort_keys=True))

        if res.returncode != 0:
            print(f"  {inv} exited {res.returncode}; see {log}")
            return 1
        if state["tracked_dirty"]:
            print(f"  ABORT: the checkout has modified tracked files after "
                  f"{inv}: {state['tracked_dirty'][:5]}")
            return 1

    print(f"\n  complete. Assemble the record, then validate it:\n"
          f"    python3 -m audit.targets.autond.evaluation_assemble {out}\n"
          f"    python3 -m audit.targets.autond.evaluation_results {out} "
          f"--plan-hash {plan.plan_hash()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gated launcher for the preregistered AutoND checkpoint "
                    "evaluation. Refuses unless the plan is frozen, every "
                    "recorded digest verifies, and the run directory is empty.")
    ap.add_argument("--autond-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--execute", action="store_true",
                    help="actually run, if and only if every gate opens")
    ap.add_argument("--i-have-reviewed-the-preregistration", action="store_true",
                    dest="reviewed", help="required alongside --execute")
    args = ap.parse_args(argv)

    if args.execute and not args.reviewed:
        print("REFUSED: --execute requires "
              "--i-have-reviewed-the-preregistration.")
        return 1
    return launch(args.autond_root, args.output_dir, execute=args.execute,
                  repo_root=args.repo_root)


if __name__ == "__main__":
    sys.exit(main())
