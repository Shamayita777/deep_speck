"""
AutoND EV entry point (separate from Gohr's scripts/run_ev.py).

    python -m autond.runner preflight --mode {calibration,production}
    python -m autond.runner run --mode smoke|calibration|production --experiment ID|paired|all [--confirm-execute]
    python -m autond.runner analyze-calibration
    python -m autond.runner plan-power
    python -m autond.runner family

Run from audit/experimental_validity. Calibration and production refuse unless every gate passes:
no unresolved decision for that tier, the production power plan matches the design, every EV/AutoND
source file is committed and clean, the AutoND pin verifies before import, and the frozen
Candidate-1 permutation validates on real AutoND data. Without --confirm-execute a calibration or
production `run` is a dry run. Each (mode, experiment) owns its own output directory, bound to its
mode and held by a single-writer lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EV_ROOT = Path(__file__).resolve().parents[1]
if str(EV_ROOT) not in sys.path:
    sys.path.insert(0, str(EV_ROOT))

from autond import adapter as A  # noqa: E402
from autond import experiments as E  # noqa: E402
from autond.baseline import EXPERIMENT_IDS, PRIMARY_FAMILY, check_config, load_config  # noqa: E402

RESULTS = EV_ROOT / "results" / "autond"
SOURCES = (
    *(f"audit/experimental_validity/autond/{f}" for f in ("__init__.py", "adapter.py", "baseline.py", "experiments.py",
                                                          "runner.py")),
    *(f"audit/experimental_validity/framework/{f}" for f in (
        "certificate.py", "confidence_intervals.py", "effect_sizes.py", "experiment.py", "failures.py", "firewall.py",
        "manifest.py", "multiplicity.py", "power.py", "provenance.py", "replication.py", "resumability.py", "seeds.py",
        "statistics.py", "validation.py")),
    "audit/experimental_validity/gohr/representation.py",
    "audit/experimental_validity/scripts/analyze_ev_calibration.py",
    "audit/experimental_validity/configs/autond_ev.yaml",
    "audit/dataset/adapters/autond.py", "audit/dataset/experiments/d0_autond_canonical.py",
    "audit/dataset/integrity.py",
)


class Refused(RuntimeError):
    pass


def output_dir(mode: str, experiment: str, root: Path = RESULTS) -> Path:
    return Path(root) / mode / experiment


def lock_mode(directory: Path, mode: str) -> None:
    """Bind a directory to one run mode for good: calibration and production never share data or state."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "RUN_MODE"
    if marker.exists() and marker.read_text().strip() != mode:
        raise Refused(f"{directory} belongs to run_mode={marker.read_text().strip()!r}, not {mode!r}")
    marker.write_text(mode)


def acquire_lock(directory: Path) -> Path:
    """Single writer per directory. A lock left by a killed run is reported, never silently cleared."""
    lock = directory / "RUN_LOCK"
    try:
        with open(lock, "x") as handle:
            handle.write(f"pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    except FileExistsError:
        raise Refused(f"{lock} exists ({lock.read_text().strip()}): another run is writing here, or one died. "
                      "Remove it only after confirming no process is using this directory.") from None
    return lock


def source_identity() -> dict:
    from audit.dataset.integrity import source_identity as identity

    return identity([A.REPOSITORY_ROOT / p for p in SOURCES], root=A.REPOSITORY_ROOT)


def gates(config: dict, mode: str, binding=None) -> dict:
    """Every pre-run check for `mode`; returns the provenance to record, raises Refused otherwise.

    Configuration and source checks come first, so nothing is imported from AutoND for a run that
    could not proceed anyway; the binding (pin verified before import) is checked when supplied."""
    problems = check_config(config, mode)
    if mode == "production" and not problems:
        problems += E.check_power_plan(config)
    if problems:
        raise Refused("; ".join(problems))
    provenance: dict = {"software": E.software_provenance()}
    if mode != "smoke":
        identity = source_identity()
        dirty = sorted(rel for rel, entry in identity["files"].items() if not entry["clean_at_head"])
        if dirty:
            raise Refused(f"sources are not committed and clean at HEAD: {dirty}")
        provenance["source_identity"] = identity
    if binding is not None:
        from gohr.representation import run_full_validation

        X, Y = binding.generate_partition(2000)
        validation = run_full_validation(X, Y, E.frozen_permutation(config))
        if not validation["all_passed"]:
            raise Refused(f"Candidate-1 permutation failed validation on AutoND data: {validation}")
        provenance |= {"representation_validation": validation,
                       "upstream": {"commit": binding.expected_upstream_commit,
                                    "bound_source_digests": binding.source_digests}}
    return provenance


def _experiments(selection: str, mode: str) -> list[str]:
    if selection == "all":
        return list(PRIMARY_FAMILY if mode == "calibration" else EXPERIMENT_IDS)
    if selection == "paired":
        return list(PRIMARY_FAMILY)
    return [selection]


def cmd_run(args, config) -> int:
    gates(config, args.mode)
    binding = A.bind(args.autond_root)
    provenance = gates(config, args.mode, binding)
    if args.mode != "smoke" and not args.confirm_execute:
        print(f"VALIDATION ONLY: every {args.mode} gate passed; nothing generated or trained. "
              "Re-run with --confirm-execute.")
        return 0
    root = Path(args.output_root) if (args.mode == "smoke" and args.output_root) else RESULTS
    permutation = E.frozen_permutation(config)
    for experiment in _experiments(args.experiment, args.mode):
        directory = output_dir(args.mode, experiment, root)
        lock_mode(directory, args.mode)
        lock = acquire_lock(directory)
        try:
            cert = E.run_experiment(experiment, config=config, mode=args.mode, output_dir=directory,
                                    binding=binding, permutation=permutation, provenance=provenance)
            (directory / "certificate.json").write_text(json.dumps(cert, indent=2, sort_keys=True, default=str))
        finally:
            lock.unlink(missing_ok=True)
        summary = cert["sufficiency_summary"]
        print(f"{experiment} [{args.mode}]: decision={cert['decision']} valid={summary['valid_replicates']}"
              f"/{summary['requested_replicates']} non_evidentiary={cert['non_evidentiary']}")
    if args.mode == "smoke" and args.experiment == "all":
        family = E.holm_family({h: json.loads((output_dir("smoke", h, root) / "certificate.json").read_text())
                                for h in PRIMARY_FAMILY}) | {"non_evidentiary": True, "run_mode": "smoke"}
        (Path(root) / "smoke" / "family_certificate.json").write_text(json.dumps(family, indent=2, default=str))
        print(f"smoke family (non-evidentiary): {family['decisions']}")
    return 0


def cmd_analyze(args, config) -> int:
    paths = args.certificates or [output_dir("calibration", h) / "certificate.json" for h in PRIMARY_FAMILY]
    artifact = E.calibration_variance(paths, config)
    out = Path(args.output or RESULTS / "calibration" / "variance.json")
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"wrote {out}: sigma_Delta_upper_95={artifact['sigma_Delta_upper_95']}")
    return 0


def cmd_plan(args, config) -> int:
    problems = check_config(config, "calibration")
    if problems:
        raise Refused("; ".join(problems))
    path = Path(args.variance or RESULTS / "calibration" / "variance.json")
    plan = E.power_plan(json.loads(path.read_text()), config, variance_sha256=E.sha256_file(path), max_n=args.max_n)
    out = Path(args.output or RESULTS / "calibration" / "power_plan.json")
    out.write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(f"wrote {out} (sha256 {E.sha256_file(out)}): required_n={plan['required_n']}")
    return 0 if plan["required_n"] is not None else 3


def cmd_family(args, config) -> int:
    paths = args.certificates or [output_dir("production", h) / "certificate.json" for h in PRIMARY_FAMILY]
    certs = {}
    for path in paths:
        cert = json.loads(Path(path).read_text())
        certs[cert["experiment_id"]] = cert
    family = E.apply_family_correction(certs)
    out = RESULTS / "production" / "family_certificate.json"
    out.write_text(json.dumps(family, indent=2, sort_keys=True, default=str))
    print(f"wrote {out}: {family['decisions']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AutoND Experimental Validity")
    parser.add_argument("--config", default=None)
    parser.add_argument("--gpu", default=None, help="CUDA device for this process (set before TensorFlow loads)")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--mode", choices=("calibration", "production"), required=True)
    pre.add_argument("--autond-root", default=os.environ.get("AUTOND_ROOT", "~/Documents/AutoND-upstream"))
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("smoke", "calibration", "production"), required=True)
    run.add_argument("--experiment", default="all", choices=(*EXPERIMENT_IDS, "paired", "all"))
    run.add_argument("--autond-root", default=os.environ.get("AUTOND_ROOT", "~/Documents/AutoND-upstream"))
    run.add_argument("--confirm-execute", action="store_true")
    run.add_argument("--output-root", default=None, help="smoke only")
    for name in ("analyze-calibration", "family"):
        p = sub.add_parser(name)
        p.add_argument("certificates", nargs="*")
        p.add_argument("--output", default=None)
    plan = sub.add_parser("plan-power")
    plan.add_argument("--variance", default=None)
    plan.add_argument("--output", default=None)
    plan.add_argument("--max-n", type=int, default=200)
    args = parser.parse_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_config(Path(args.config)) if args.config else load_config()
    try:
        if args.command == "preflight":
            gates(config, args.mode)
            provenance = gates(config, args.mode, A.bind(args.autond_root))
            print(f"PREFLIGHT PASSED for {args.mode}: {len(provenance['source_identity']['files'])} sources clean at "
                  f"{provenance['source_identity']['git_head']}; permutation validated on AutoND data")
            return 0
        return {"run": cmd_run, "analyze-calibration": cmd_analyze, "plan-power": cmd_plan,
                "family": cmd_family}[args.command](args, config)
    except (RuntimeError, ValueError) as exc:     # Refused, binding, firewall and resume violations included
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
