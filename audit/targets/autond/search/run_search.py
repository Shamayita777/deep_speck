#!/usr/bin/env python3
"""
Command-line entry point for the AutoND evolutionary-search reproduction.

Modes, deliberately named so that a benchmark can never be mistaken for
the reproduction in a filename or a log:

    --mode benchmark   small settings, for timing only. The result record
                       is tagged BENCHMARK and the summary refuses to
                       describe it as a reproduction of anything.
    --mode paper       the configuration the publication describes.

Each run writes its own directory; a crashed or interrupted run leaves
its partial record behind and `--resume` skips the runs that completed.

Nothing in the upstream checkout is written. Bytecode generation in the
checkout is suppressed by default (`sys.dont_write_bytecode`), so a
read-only clone stays byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True                     # keep the target pristine

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from search_runner import (                         # noqa: E402
    PINNED_COMMIT, RUNNER_VERSION, SearchRunnerError, bind_source, run_once,
    utc_now, write_manifest,
)

REQUIRED_CONFIG_KEYS = ("mode", "cipher", "scenario", "num_generations",
                        "num_samples", "epsilon", "population_init", "runs")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise SearchRunnerError(f"{path}: configuration is missing {missing}.")
    if cfg["mode"] not in ("r1_readme_demo", "r2_paper_parameters"):
        raise SearchRunnerError(
            f"{path}: mode must be 'r1_readme_demo' or 'r2_paper_parameters', not "
            f"{cfg['mode']!r}.")
    if cfg["population_init"] not in ("previous", "fresh"):
        raise SearchRunnerError(
            f"{path}: population_init must be 'previous' (shipped code) or 'fresh' "
            f"(as the paper describes), not {cfg['population_init']!r}.")
    if cfg["mode"] == "r1_readme_demo":
        if cfg["num_generations"] is not None or cfg["num_samples"] is not None:
            raise SearchRunnerError(
                f"{path}: R1 reproduces the released code AS DOCUMENTED, so it must "
                "override nothing. num_generations and num_samples must both be null; "
                f"got {cfg['num_generations']!r} and {cfg['num_samples']!r}.")
    if cfg["mode"] == "r2_paper_parameters":
        if cfg["num_generations"] is None or cfg["num_generations"] < 50:
            raise SearchRunnerError(
                f"{path}: R2 declares num_generations={cfg['num_generations']}, below "
                "the publication's 50. A reduced run must not carry the R2 label.")
        if cfg["num_samples"] is None or cfg["num_samples"] < 10**4:
            raise SearchRunnerError(
                f"{path}: R2 declares num_samples={cfg['num_samples']}, below the "
                "publication's 10**4.")
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run AutoND's evolutionary input-difference search against the "
                    "pinned upstream source, without modifying it.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--autond-root", type=Path, required=True,
                    help="the pinned AutoND checkout (contains optimizer.py)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="must be OUTSIDE the AutoND checkout")
    ap.add_argument("--runs", type=int, default=None,
                    help="override the configuration's run count")
    ap.add_argument("--resume", action="store_true",
                    help="skip run indices that already hold a COMPLETED result.json")
    ap.add_argument("--protocol", type=Path, default=None,
                    help="a frozen protocol file; its sha256 is bound into every "
                         "result. Required for --mode paper.")
    ap.add_argument("--allow-bytecode", action="store_true",
                    help="permit Python to write __pycache__ into the checkout")
    args = ap.parse_args(argv)

    if args.allow_bytecode:
        sys.dont_write_bytecode = False

    cfg = load_config(args.config)
    if args.runs is not None:
        cfg["runs"] = args.runs

    out = args.output_dir.expanduser().resolve()
    root = args.autond_root.expanduser().resolve()
    if root == out or root in out.parents:
        raise SearchRunnerError(
            f"--output-dir {out} is inside the AutoND checkout {root}. The target must "
            "stay pristine; choose a directory outside it.")
    out.mkdir(parents=True, exist_ok=True)

    protocol_sha = None
    if args.protocol is not None:
        from search_runner import sha256_file
        protocol_sha = sha256_file(args.protocol.expanduser().resolve())
    elif cfg["mode"] == "r2_paper_parameters":
        raise SearchRunnerError(
            "R2 requires --protocol pointing at the frozen protocol file. "
            "A confirmatory run whose plan was not fixed beforehand is not "
            "confirmatory, and this harness will not produce one.")

    cfg["protocol_sha256"] = protocol_sha
    cfg["config_path"] = str(args.config.resolve())
    from search_runner import sha256_file as _sha
    cfg["config_sha256"] = _sha(args.config.resolve())

    optimizer, cipher, binding = bind_source(root, cfg["cipher"])

    print(f"{RUNNER_VERSION}")
    print(f"  variant              {cfg['mode']}")
    print("  claim                " + (
        "reproduces the README's documented demo output (0x400000 returned)"
        if cfg["mode"] == "r1_readme_demo" else
        "paper parameters for what the code exposes; NOT fully paper-faithful"))
    print(f"  cipher / scenario    {cfg['cipher']} / {cfg['scenario']}")
    print(f"  generations          " + (
        "released default (unpatched)" if cfg["num_generations"] is None
        else f"{cfg['num_generations']}  (P1-GENERATIONS applied)"))
    print(f"  samples per round    " + (
        "optimize() default (not passed, as upstream does not pass it)"
        if cfg["num_samples"] is None else f"{cfg['num_samples']}  (passed explicitly)"))
    print(f"  bias threshold T_b   0.05  (hardcoded at optimizer.py:138; "
          "the paper states 0.01)")
    print(f"  population init      {cfg['population_init']}"
          + ("  (as the paper describes)" if cfg["population_init"] == "fresh"
             else "  (as the shipped code does)"))
    print(f"  epsilon              {cfg['epsilon']}")
    print(f"  runs                 {cfg['runs']}")
    print(f"  upstream commit      {binding.head_commit}")
    print(f"  optimizer.py sha256  {binding.source_hashes.get('optimizer.py')}")
    print(f"  cipher sha256        {binding.cipher_hash}")
    print(f"  protocol sha256      {protocol_sha}")
    print(f"  output               {out}")
    print(f"  tensorflow imported  {bool(sys.modules.get('tensorflow'))}  (search is CPU-only)")
    print()

    completed, failed, skipped = [], [], []
    for i in range(cfg["runs"]):
        rd = out / f"run_{i:03d}" / "result.json"
        if args.resume and rd.is_file():
            try:
                if json.loads(rd.read_text()).get("status") == "COMPLETED":
                    print(f"  run {i:03d}  SKIPPED (already completed)")
                    skipped.append(i)
                    continue
            except json.JSONDecodeError:
                pass                                 # torn file: rerun this index
        print(f"  run {i:03d}  starting at {utc_now()}")
        try:
            rec = run_once(cfg, optimizer, cipher, binding, out, i)
            print(f"  run {i:03d}  {rec['status']}  "
                  f"{rec['elapsed_seconds']:.1f}s  "
                  f"highest_non_random_round={rec.get('highest_non_random_round')}  "
                  f"differences={rec.get('returned_differences_hex')}")
            completed.append(i)
        except KeyboardInterrupt:
            print(f"\n  run {i:03d}  INTERRUPTED by the operator. Its partial record is "
                  f"in {out / f'run_{i:03d}'}; rerun with --resume to continue.")
            failed.append(i)
            break
        except Exception as exc:                                    # noqa: BLE001
            print(f"  run {i:03d}  FAILED: {type(exc).__name__}: {exc}")
            failed.append(i)

    manifest = write_manifest(out, {
        "config": cfg, "source": binding.to_dict(),
        "runs_completed": completed, "runs_failed": failed, "runs_skipped": skipped,
    })
    print(f"\n  completed {len(completed)}   failed {len(failed)}   skipped {len(skipped)}")
    print(f"  manifest  {manifest}")
    if cfg["mode"] == "r2_paper_parameters":
        print("\n  REMINDER: R2 is NOT a full paper-faithful reproduction. T_b remains "
              "0.05 against\n  the paper's 0.01, selection remains elitist, and "
              "mutation remains 99/100. See\n  _deviations in the config and "
              "PAPER_CODE_README_COMPARISON.md.")
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SearchRunnerError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
