#!/usr/bin/env python3
"""
Verify a completed search output directory.

Checks structure, recomputes every hash in MANIFEST.json, confirms each
run's record is internally consistent, and prints a summary. It makes no
scientific claim about the results; it establishes that the evidence
bundle is intact and says what it says.

Exits 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


#: Operating-system and interpreter debris. Kept byte-identical in intent
#: to evaluation_gate.BENIGN_UNTRACKED / BENIGN_SUFFIXES so the two
#: validators cannot disagree about what counts as evidence.
BENIGN_NAMES = (
    ".DS_Store", "._.DS_Store", "Thumbs.db", "desktop.ini",
)
BENIGN_DIRS = (
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
)
BENIGN_SUFFIXES = (".pyc", ".pyo", ".swp", ".orig")


def _is_benign(rel: str) -> bool:
    """
    True for a bundle-relative path that is OS/interpreter debris.

    Matched on path PARTS, not substrings, so a legitimate evidence file
    whose name merely contains one of these strings is not excused.
    """
    parts = Path(rel).parts
    if any(d in parts for d in BENIGN_DIRS):
        return True
    name = parts[-1] if parts else rel
    if name in BENIGN_NAMES:
        return True
    # AppleDouble sidecars: ._<something>
    if name.startswith("._"):
        return True
    return name.endswith(BENIGN_SUFFIXES)


def verify(out_dir: Path) -> tuple[bool, list[str], dict]:
    problems: list[str] = []
    summary: dict = {"runs": [], "output_dir": str(out_dir)}

    mpath = out_dir / "MANIFEST.json"
    if not mpath.is_file():
        return False, [f"no MANIFEST.json in {out_dir}; the run did not finish "
                       "writing its evidence bundle."], summary
    manifest = json.loads(mpath.read_text())

    # 1. every file named in the manifest still hashes to its recorded value
    for rel, meta in sorted(manifest.get("files", {}).items()):
        p = out_dir / rel
        if not p.is_file():
            problems.append(f"manifest names {rel}, which is missing.")
            continue
        got = sha256_file(p)
        if got != meta["sha256"]:
            problems.append(f"{rel}: sha256 {got} != manifest {meta['sha256']}.")

    # 2. every file present is named in the manifest
    #
    # EXCEPT operating-system and interpreter debris. A Finder visit drops
    # a .DS_Store into the directory; an import writes __pycache__. Neither
    # is evidence, neither was there when the manifest was written, and
    # neither can alter a result. Treating them as validation failures made
    # a correct bundle read as corrupt - which is exactly what happened:
    # browsing the evidence folder in Finder flipped II1-SEARCH-R1 from
    # PARTIAL to FAILED and discarded a real scientific finding.
    #
    # They are RECORDED and LEFT IN PLACE, never deleted. This mirrors
    # evaluation_gate.BENIGN_UNTRACKED, which already had this policy; the
    # two now agree instead of contradicting each other.
    benign: list[str] = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            rel = str(p.relative_to(out_dir))
            if rel in manifest.get("files", {}):
                continue
            if _is_benign(rel):
                benign.append(rel)
                continue
            problems.append(f"{rel} is present but not named in the manifest.")
    summary["unmanifested_benign"] = benign
    if benign:
        summary["unmanifested_benign_note"] = (
            f"{len(benign)} operating-system/interpreter file(s) present and "
            f"not named in the manifest: {benign}. These are RECORDED and "
            "LEFT IN PLACE. They are not evidence, were not present when the "
            "manifest was written, and cannot affect a result, so they do not "
            "invalidate the bundle.")

    # 3. per-run consistency
    for run_dir in sorted(out_dir.glob("run_*")):
        rp = run_dir / "result.json"
        if not rp.is_file():
            problems.append(f"{run_dir.name}: no result.json.")
            continue
        rec = json.loads(rp.read_text())
        row = {"run": run_dir.name, "status": rec.get("status"),
               "elapsed_seconds": rec.get("elapsed_seconds"),
               "highest_non_random_round": rec.get("highest_non_random_round"),
               "differences": rec.get("returned_differences_hex"),
               "mode": rec.get("config", {}).get("mode"),
               "generations": rec.get("config", {}).get("num_generations"),
               "samples": rec.get("config", {}).get("num_samples"),
               "population_init": rec.get("config", {}).get("population_init"),
               "bit_reproducible": rec.get("reproducibility", {}).get("bit_reproducible")}
        summary["runs"].append(row)

        src = rec.get("source", {})
        if src.get("head_commit") != PINNED_COMMIT:
            problems.append(
                f"{run_dir.name}: ran against commit {src.get('head_commit')}, "
                f"not the pinned {PINNED_COMMIT}.")
        if rec.get("tensorflow_observed", {}).get("tensorflow"):
            problems.append(f"{run_dir.name}: TensorFlow was imported during the run.")

        gl = run_dir / "generations.jsonl"
        if not gl.is_file():
            problems.append(f"{run_dir.name}: no generations.jsonl.")
        else:
            rows = [json.loads(l) for l in gl.read_text().splitlines() if l.strip()]
            seqs = [r["seq"] for r in rows]
            if seqs != list(range(len(rows))):
                problems.append(f"{run_dir.name}: generations.jsonl seq is not "
                                "contiguous from 0; the log is incomplete or torn.")
            if rec.get("scoring_calls_logged") != len(rows):
                problems.append(
                    f"{run_dir.name}: result.json says {rec.get('scoring_calls_logged')} "
                    f"logged calls, the log holds {len(rows)}.")
            row["log_rows"] = len(rows)
            row["evo_rounds"] = sum(1 for r in rows if r.get("event") == "evo_enter")

        # patches that changed behaviour must be visible in the summary
        row["behaviour_changing_patches"] = [
            p["patch_id"] for p in rec.get("patches", []) if p.get("changes_behaviour")]

    modes = Counter(r["mode"] for r in summary["runs"])
    summary["modes"] = dict(modes)
    if len(modes) > 1:
        problems.append(
            f"this directory mixes modes {dict(modes)}. Benchmark timings and a "
            "paper-configured reproduction must not share an output directory.")
    inits = {r["population_init"] for r in summary["runs"]}
    if len(inits) > 1:
        problems.append(
            f"this directory mixes population_init values {inits}. Those are two "
            "different procedures and their results are not pooled.")
    return not problems, problems, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify a search output directory.")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the summary as JSON")
    args = ap.parse_args(argv)

    out = args.output_dir.expanduser().resolve()
    ok, problems, summary = verify(out)

    print(f"Verifying {out}\n")
    for r in summary["runs"]:
        print(f"  {r['run']}  {r.get('status')}  "
              f"{(r.get('elapsed_seconds') or 0):.1f}s  "
              f"mode={r.get('mode')}  gens={r.get('generations')}  "
              f"samples={r.get('samples')}  init={r.get('population_init')}")
        print(f"      highest_non_random_round = {r.get('highest_non_random_round')}")
        print(f"      differences              = {r.get('differences')}")
        print(f"      log rows / evo rounds    = {r.get('log_rows')} / {r.get('evo_rounds')}")
        print(f"      bit reproducible         = {r.get('bit_reproducible')}")
        if r.get("behaviour_changing_patches"):
            print(f"      behaviour-changing       = {r['behaviour_changing_patches']}")
    if summary["runs"]:
        found = Counter()
        for r in summary["runs"]:
            for d in (r.get("differences") or []):
                found[str(d)] += 1
        print(f"\n  differences returned, across {len(summary['runs'])} run(s):")
        for d, n in found.most_common():
            print(f"      {d}  returned by {n} run(s)")
    print()
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  all checks passed: manifest complete, hashes match, logs contiguous, "
              "pinned commit confirmed, no TensorFlow imported.")
    print("\n  This script verifies the INTEGRITY of the bundle. It makes no claim "
          "about\n  whether the search reproduced the publication; that is a "
          "scientific judgement\n  to be made against the frozen protocol.")

    if args.json:
        args.json.write_text(json.dumps(
            {"ok": ok, "problems": problems, "summary": summary},
            indent=2, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
