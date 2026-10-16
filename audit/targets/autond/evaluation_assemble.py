#!/usr/bin/env python3
"""
Assemble an evaluation result bundle from what an authorised run produced.

WHY THIS EXISTS
    Revision 2 defined the result schema and a validator for it, but
    nothing that could actually produce a conforming bundle. The only way
    to get one was to write RESULT.json by hand, which is exactly the
    fabricated-record path the schema was supposed to close. This module
    is the missing step: it reads the artifacts a run left behind and
    derives the record from them.

WHAT IT READS, AND WHAT IT REFUSES TO INVENT
    Every field in RESULT.json comes from an artifact or from the
    execution journal the launcher wrote at run time:

      accuracies, tprs, tnrs   loaded from the NPZ the target wrote
      npz_sha256               computed from that NPZ
      sha256                   computed from the checkpoint copy evaluated
      upstream_sha256          read from the plan, then CHECKED against
                               the copy's own digest
      command, cwd, timestamps recorded by the launcher before and after
                               each execution
      environment              recorded by the launcher, in the process
                               that actually ran the evaluation
      upstream_commit          read from the checkout at run time
      upstream_write_check     the launcher's cleanliness attestation

    Nothing is defaulted into existence. A missing journal, a missing
    NPZ, an NPZ whose contents disagree with its journal, or a checkpoint
    copy whose digest is not the plan's, makes assembly FAIL and write
    nothing. An absent field is never filled with a plausible value.

    This module runs no evaluation. It cannot: it has no launcher and no
    model code. It reads files and writes two of them, RESULT.json and
    SHA256SUMS.txt, into the bundle directory the run already owns.

CROSS-CHECKING
    The journal records what the launcher believed it produced; the NPZ
    is what the target actually wrote. Assembly compares them and refuses
    on disagreement, so a journal that drifted from the artifacts cannot
    become a record. `evaluation_results.validate_bundle` then re-derives
    the same comparison independently from the written bundle, which is
    the point of keeping the two modules separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from audit.targets.autond import evaluation_results as evres
from audit.targets.autond.preregistrations import autond_ii_v2 as plan

ASSEMBLER_VERSION = "autond-evaluation-assemble-v1"

#: The launcher writes one of these per checkpoint before and after its
#: execution. Assembly requires every field.
JOURNAL_REQUIRED = (
    "checkpoint_id", "command", "cwd", "started_utc", "finished_utc",
    "returncode", "model_copy", "model_copy_sha256", "npz",
    "upstream_commit", "upstream_write_check", "environment",
)

_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: Path, root: Path) -> str:
    """Bundle-relative path, refusing anything outside the bundle."""
    p = Path(path).resolve()
    r = Path(root).resolve()
    try:
        return str(p.relative_to(r))
    except ValueError as exc:
        raise ValueError(
            f"{p} is outside the bundle directory {r}; a bundle may not "
            "reference a file it does not contain") from exc


def read_journals(bundle_dir: Path) -> tuple[list[dict], list[str]]:
    """Load every execution journal the launcher wrote."""
    d = Path(bundle_dir)
    problems: list[str] = []
    journals: list[dict] = []
    jdir = d / "journal"
    if not jdir.is_dir():
        return [], [f"no journal directory at {jdir}; the launcher writes one "
                    "entry per execution, and assembly cannot infer what ran "
                    "without them"]
    for f in sorted(jdir.glob("exec_*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception as exc:                                    # noqa: BLE001
            problems.append(f"{f.name}: unreadable JSON "
                            f"({type(exc).__name__}: {exc})")
            continue
        missing = [k for k in JOURNAL_REQUIRED if k not in rec]
        if missing:
            problems.append(f"{f.name}: missing fields {missing}")
            continue
        journals.append(rec)
    if not journals and not problems:
        problems.append(f"{jdir} holds no exec_*.json entries")
    return journals, problems


def _load_npz_metrics(path: Path) -> tuple[Optional[dict], list[str]]:
    """Read accs/tprs/tnrs out of the NPZ the target wrote."""
    problems: list[str] = []
    try:
        import numpy as np
    except Exception as exc:                                        # noqa: BLE001
        return None, [f"numpy is not importable ({type(exc).__name__}: {exc}), "
                      "so the NPZ could not be read. Assembly does not guess "
                      "metrics it cannot load."]
    try:
        with np.load(path) as z:
            keys = list(z.files)
            out = {}
            for name in ("accs", "tprs", "tnrs"):
                if name not in z.files:
                    problems.append(f"{path.name}: no '{name}' array "
                                    f"(found {keys})")
                    continue
                out[name] = [float(v) for v in np.asarray(z[name]).ravel()]
    except Exception as exc:                                        # noqa: BLE001
        return None, [f"{path.name}: could not be loaded as an NPZ "
                      f"({type(exc).__name__}: {exc})"]
    if problems:
        return None, problems
    return out, []


def assemble(bundle_dir: Path, *, upstream: Optional[Path] = None,
             repo_root: Optional[Path] = None,
             dry_run: bool = False) -> dict:
    """
    Build RESULT.json and SHA256SUMS.txt from the bundle's own artifacts.

    Returns {"ok", "problems", "record", "written"}. With ok False
    nothing is written, including on a dry run.
    """
    d = Path(bundle_dir).expanduser().resolve()
    problems: list[str] = []
    out: dict = {"ok": False, "problems": problems, "record": None,
                 "written": [], "bundle_dir": str(d)}

    if not d.is_dir():
        problems.append(f"no bundle directory at {d}")
        return out

    journals, jproblems = read_journals(d)
    problems.extend(jproblems)
    if not journals:
        return out

    # The validator's design constants duplicate the plan's. Check they
    # agree BEFORE reading any artifact: if they have diverged, every
    # count enforced below would be the wrong one.
    try:
        evres.assert_matches_preregistration()
    except ValueError as exc:
        problems.append(str(exc))
        return out

    expected = plan.TARGET["checkpoints"]
    #: Keyed by INVOCATION, not checkpoint: the registered design runs
    #: eval_nets.py twice per checkpoint, so two journals per checkpoint
    #: is the expected shape rather than a duplicate to reject.
    seen: dict[str, dict] = {}
    per_invocation: dict[str, list[dict]] = {}
    checkpoints: dict[str, dict] = {}
    commands: list[str] = []
    starts: list[str] = []
    finishes: list[str] = []
    commits: set[str] = set()
    writechecks: set[str] = set()
    envs: list[dict] = []

    for rec in journals:
        cid = rec["checkpoint_id"]
        # A journal written before the two-invocation design carries no
        # invocation_id. Treat its checkpoint id as the invocation id so
        # such a bundle is reported as having one invocation short of the
        # registered design, rather than crashing here.
        inv = str(rec.get("invocation_id") or cid)
        if inv in seen:
            problems.append(f"two journal entries for invocation {inv}; "
                            "assembly will not choose between them")
            continue
        seen[inv] = rec

        if cid not in expected:
            problems.append(f"journal names checkpoint {cid!r}, which the plan "
                            f"does not list ({sorted(expected)})")
            continue
        want = expected[cid]["sha256"]

        if rec["returncode"] != 0:
            problems.append(f"{cid}: the evaluation exited "
                            f"{rec['returncode']}; a failed run is not a result")
            continue

        for field in ("started_utc", "finished_utc"):
            if not _ISO_UTC.match(str(rec[field])):
                problems.append(f"{cid}.{field} {rec[field]!r} is not an "
                                "ISO-8601 UTC instant")
        if _ISO_UTC.match(str(rec["started_utc"])) and \
                _ISO_UTC.match(str(rec["finished_utc"])) and \
                rec["finished_utc"] < rec["started_utc"]:
            problems.append(f"{cid}: finished_utc precedes started_utc")

        # --- the checkpoint copy that was actually evaluated -----------
        copy_path = Path(rec["model_copy"])
        if not copy_path.is_absolute():
            copy_path = d / copy_path
        if not copy_path.is_file():
            problems.append(f"{cid}: the checkpoint copy {rec['model_copy']} "
                            "is missing from the bundle")
            continue
        got = sha256_file(copy_path)
        if got != rec["model_copy_sha256"]:
            problems.append(
                f"{cid}: the copy on disk hashes to {got} but the journal "
                f"recorded {rec['model_copy_sha256']}; the artifact changed "
                "after the run")
            continue
        if got != want:
            problems.append(
                f"{cid}: the copy evaluated hashes to {got}, not the plan's "
                f"recorded {want}. The run did not evaluate the checkpoint the "
                "plan authorised.")
            continue

        # --- the NPZ the target wrote ----------------------------------
        npz_path = Path(rec["npz"])
        if not npz_path.is_absolute():
            npz_path = d / npz_path
        if not npz_path.is_file():
            problems.append(f"{cid}: the NPZ {rec['npz']} is missing; the "
                            "target writes it beside the model path it was "
                            "given")
            continue
        metrics, mproblems = _load_npz_metrics(npz_path)
        problems.extend(f"{cid}: {p}" for p in mproblems)
        if metrics is None:
            continue

        accs = metrics["accs"]
        if len(accs) != evres.EXPECTED_N_SETS:
            problems.append(
                f"{inv}: the NPZ holds {len(accs)} accuracies, not the "
                f"{evres.EXPECTED_SETS_PER_INVOCATION} that upstream "
                f"eval_nets.nEval fixes for one invocation. An invocation that "
                "produced fewer sets is INCONCLUSIVE by the plan's "
                "missing-data rule, not a result to be assembled.")
            continue
        bad = [a for a in accs if not 0.0 <= a <= 1.0]
        if bad:
            problems.append(f"{cid}: accuracies outside [0,1]: {bad}")
            continue
        for name in ("tprs", "tnrs"):
            if len(metrics[name]) != len(accs):
                problems.append(f"{cid}: {name} has {len(metrics[name])} "
                                f"entries, accuracies has {len(accs)}")

        # --- cross-check against the journal, if it recorded metrics ---
        claimed = rec.get("observed_accuracies")
        if claimed is not None:
            if [round(float(x), 12) for x in claimed] != \
                    [round(float(x), 12) for x in accs]:
                problems.append(
                    f"{cid}: the journal recorded accuracies {claimed} but the "
                    f"NPZ holds {accs}. The record is derived from the NPZ, and "
                    "a journal that disagrees with the artifact is a reason to "
                    "stop, not to pick one.")
                continue

        try:
            per_invocation.setdefault(cid, []).append({
                "invocation_id": inv,
                "invocation_index": int(rec.get("invocation_index", 0)),
                "invocation_suffix": str(rec.get("invocation_suffix", "")),
                "path": _rel(copy_path, d),
                "sha256": got,
                "upstream_sha256": want,
                "round_number": int(expected[cid]["round_number"]),
                "input_difference": expected[cid]["input_difference"],
                "accuracies": accs,
                "tprs": metrics["tprs"],
                "tnrs": metrics["tnrs"],
                "npz": _rel(npz_path, d),
                "npz_sha256": sha256_file(npz_path),
                "n_samples_per_set": evres.EXPECTED_SAMPLES_PER_SET,
            })
        except ValueError as exc:
            problems.append(f"{cid}: {exc}")
            continue

        commands.append(" ".join(rec["command"]) if isinstance(rec["command"], list)
                        else str(rec["command"]))
        starts.append(rec["started_utc"])
        finishes.append(rec["finished_utc"])
        commits.add(str(rec["upstream_commit"]))
        writechecks.add(str(rec["upstream_write_check"]))
        envs.append(rec["environment"])

    # --- pool the invocations into the registered n = 10 sample -------
    #
    # The analysis is a TOST on ONE sample of ten per checkpoint. Pooling
    # is therefore done here, once, from the per-invocation artifacts,
    # and the per-invocation accuracies are kept alongside so a reader can
    # inspect the two groups the pooled sample came from. Nothing is
    # averaged twice and no invocation is weighted.
    want_inv = evres.EXPECTED_INVOCATIONS_PER_CHECKPOINT
    for cid, invs in sorted(per_invocation.items()):
        if len(invs) != want_inv:
            got_ids = sorted(i["invocation_id"] for i in invs)
            problems.append(
                f"{cid}: {len(invs)} usable invocation(s) {got_ids}, not the "
                f"registered {want_inv}. The plan's missing-data rule forbids "
                f"analysing a single surviving invocation as an "
                f"n = {evres.EXPECTED_SETS_PER_INVOCATION} result, because that "
                f"would be an unregistered fallback design chosen after seeing "
                f"which run failed. Rerun the whole checkpoint into a fresh run "
                f"directory.")
            continue

        suffixes = [i["invocation_suffix"] for i in invs]
        if len(set(suffixes)) != len(suffixes):
            problems.append(
                f"{cid}: invocations share the suffix(es) {sorted(suffixes)}. "
                "eval_nets.py derives its NPZ name from --add_str, so equal "
                "suffixes mean the second invocation overwrote the first's "
                "artifact and the two are not independent draws.")
            continue
        if sorted(suffixes) != sorted(plan.INVOCATION_SUFFIXES):
            problems.append(
                f"{cid}: invocation suffixes {sorted(suffixes)} are not the "
                f"registered {sorted(plan.INVOCATION_SUFFIXES)}")
            continue

        npzs = [i["npz_sha256"] for i in invs]
        if len(set(npzs)) != len(npzs):
            problems.append(
                f"{cid}: two invocations produced byte-identical NPZs "
                f"({npzs[0]}). eval_nets.py draws from os.urandom, so identical "
                "artifacts mean one file was read twice rather than two "
                "independent datasets being generated.")
            continue

        weights = {i["sha256"] for i in invs}
        if len(weights) != 1:
            problems.append(
                f"{cid}: invocations evaluated different checkpoint bytes "
                f"{sorted(weights)}. Pooling assumes both measured the same "
                "weights; it is not valid across two different models.")
            continue

        ordered = sorted(invs, key=lambda i: (i["invocation_index"],
                                              i["invocation_suffix"]))
        pooled = [a for i in ordered for a in i["accuracies"]]
        if len(pooled) != evres.EXPECTED_N_POOLED:
            problems.append(
                f"{cid}: pooled {len(pooled)} accuracies, not the registered "
                f"{evres.EXPECTED_N_POOLED}")
            continue

        first = ordered[0]
        checkpoints[cid] = {
            "path": first["path"],
            "sha256": first["sha256"],
            "upstream_sha256": first["upstream_sha256"],
            "round_number": first["round_number"],
            "input_difference": first["input_difference"],
            "accuracies": pooled,
            "tprs": [x for i in ordered for x in i["tprs"]],
            "tnrs": [x for i in ordered for x in i["tnrs"]],
            "npz": [i["npz"] for i in ordered],
            "npz_sha256": [i["npz_sha256"] for i in ordered],
            "n_samples_per_set": evres.EXPECTED_SAMPLES_PER_SET,
            "n_pooled": len(pooled),
            "invocations": [
                {"invocation_id": i["invocation_id"],
                 "invocation_index": i["invocation_index"],
                 "invocation_suffix": i["invocation_suffix"],
                 "accuracies": i["accuracies"],
                 "npz": i["npz"], "npz_sha256": i["npz_sha256"]}
                for i in ordered
            ],
            "pooling": (
                f"one sample of {len(pooled)}: the "
                f"{evres.EXPECTED_SETS_PER_INVOCATION} sets of each invocation "
                f"concatenated in registered suffix order "
                f"{list(plan.INVOCATION_SUFFIXES)}. Per-invocation accuracies "
                f"are retained above; they are reported for inspection and are "
                f"not compared to each other as a test."),
        }

    # --- the plan requires BOTH checkpoints ---------------------------
    absent = sorted(set(expected) - set(checkpoints))
    if absent:
        problems.append(
            f"no usable result for {absent}; the plan's hypothesis family "
            "covers both checkpoints and a partial bundle is not assembled")

    if len(commits) > 1:
        problems.append(f"journals disagree about the upstream commit: "
                        f"{sorted(commits)}")
    if len(writechecks) > 1:
        problems.append(f"journals disagree about the upstream write check: "
                        f"{sorted(writechecks)}")
    if commits and next(iter(commits)) != plan.TARGET["pinned_commit"]:
        problems.append(
            f"the run executed against commit {next(iter(commits))}, not the "
            f"pinned {plan.TARGET['pinned_commit']}")

    # Environments must agree, or the ten sets were not drawn under one
    # stack and the record cannot name a single environment.
    if envs:
        first = json.dumps(envs[0], sort_keys=True)
        if any(json.dumps(e, sort_keys=True) != first for e in envs[1:]):
            problems.append("journals record different environments; a single "
                            "result record cannot describe two stacks")

    if problems:
        return out

    frozen, freeze_detail = plan.is_frozen(repo_root)
    if not frozen:
        problems.append(
            f"the plan is not frozen ({freeze_detail}), so a result assembled "
            "under it could not be confirmatory evidence. Assembly refuses "
            "rather than writing a bundle that validate_bundle would reject.")
        return out

    rec0 = plan.read_freeze_record() or {}
    record = {
        "schema_version": evres.SCHEMA_VERSION,
        "assembler": ASSEMBLER_VERSION,
        "preregistration": {
            "id": plan.PREREGISTRATION_ID,
            "plan_hash": plan.plan_hash(),
            "frozen_at_utc": rec0.get("frozen_at_utc"),
        },
        "execution": {
            "started_utc": min(starts),
            "finished_utc": max(finishes),
            "command": " && ".join(commands),
            "upstream_commit": next(iter(commits)),
            "upstream_write_check": next(iter(writechecks)),
        },
        "environment": envs[0],
        "checkpoints": checkpoints,
    }

    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if dry_run:
        out["ok"] = True
        out["record"] = record
        out["written"] = []
        return out

    (d / "RESULT.json").write_text(payload, encoding="utf-8")

    lines = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{sha256_file(p)}  {p.relative_to(d)}")
    (d / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out["ok"] = True
    out["record"] = record
    out["written"] = ["RESULT.json", "SHA256SUMS.txt"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Assemble RESULT.json and SHA256SUMS.txt from an "
                    "authorised evaluation run. Runs no evaluation.")
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("--autond-root", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="repository root, for verifying the freeze record")
    ap.add_argument("--dry-run", action="store_true",
                    help="derive the record and report it; write nothing")
    args = ap.parse_args(argv)

    res = assemble(args.bundle_dir, upstream=args.autond_root,
                   repo_root=args.repo_root, dry_run=args.dry_run)
    print(f"{ASSEMBLER_VERSION}")
    print(f"  bundle: {res['bundle_dir']}")
    if res["record"]:
        for cid, c in sorted(res["record"]["checkpoints"].items()):
            stats = evres.mean_and_t_interval(c["accuracies"])
            print(f"  {cid}: n={stats['n']} mean={stats['mean']:.6f} "
                  f"sd={stats['sample_sd']:.7f}")
    if res["problems"]:
        print(f"\n  {len(res['problems'])} PROBLEM(S) - nothing written:")
        for p in res["problems"]:
            print(f"    - {p}")
        return 1
    if res["written"]:
        print(f"\n  wrote: {', '.join(res['written'])}")
        print(f"\n  validate it with:\n    python3 -m "
              f"audit.targets.autond.evaluation_results {res['bundle_dir']} "
              f"--plan-hash {plan.plan_hash()}")
    else:
        print("\n  dry run: record derived, nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
