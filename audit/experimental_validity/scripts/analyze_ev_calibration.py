#!/usr/bin/env python3
"""
Analyse CALIBRATION (pilot) certificates into a variance artifact for
prospective power planning.

Emits NUISANCE VARIANCE ONLY. No mean difference, no p-value, no effect
size, no conclusion: the pilot characterises noise, and letting an
observed pilot effect escape into planning is how a confirmatory design
becomes contaminated. epsilon is never derived here.

Reports, per hypothesis: n_valid, sigma_Delta (SD of the paired
differences), variance, and a ONE-SIDED 95% UPPER CONFIDENCE BOUND for
sigma from the chi-square result:

    sigma_UCL = s * sqrt(df / chi2_{0.05, df})

The upper bound, not the point estimate, sizes the production design. A
~10-pair pilot leaves sigma materially uncertain, and sizing on the point
estimate under-powers the experiment roughly half the time.

PROVENANCE RECOVERY (v3)
------------------------
Certificates written by run_ev.py before the provenance block existed do
not carry `provenance.config_hash`, `provenance.source_manifest_sha256`
or `provenance.candidate1_permutation_sha256`, even though the runs that
produced them did establish all three. This module therefore reads each
binding from wherever the WRITER actually put it, and, for bindings that
exist only outside the certificate, accepts an explicit attestation file.

Three rules govern that recovery, and none of them may be relaxed:

  1. A binding is never invented. Every value is either read out of the
     certificate, or supplied by an attestation that names its external
     evidence.
  2. Every recovered binding is still CHECKED against the frozen
     configuration. Recovery changes where a value is read from, never
     whether it has to match.
  3. Every recovery is recorded in the output artifact, and any artifact
     containing one is marked `binding_degraded: true` so the weakened
     provenance travels downstream into the power plan and the write-up
     instead of being laundered into a clean-looking result.

Default behaviour is unchanged: a certificate that supplies nothing and
has no attestation is still refused.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scipy import stats  # noqa: E402

from audit.common.provenance import sha256_file, utc_timestamp  # noqa: E402
from audit.common.strict_json import dumps_strict  # noqa: E402

EXPECTED_EXPERIMENTS = ("H-EV-SHUFFLE", "H-EV-REPRESENTATION")
SIGMA_UCL_LEVEL = 0.95
#: Frozen calibration design: exactly this many valid pairs per hypothesis.
REQUIRED_CALIBRATION_PAIRS = 10
#: Config each calibration certificate must have been produced from.
CALIBRATION_CONFIGS = {
    "H-EV-SHUFFLE": "configs/gohr_ev_shuffle_calibration.yaml",
    "H-EV-REPRESENTATION": "configs/gohr_ev_representation_calibration.yaml",
}
#: Attestation file format this reader understands.
ATTESTATION_VERSION = 1
#: Fields an attestation may supply. Nothing outside this set is read, so
#: an attestation can never introduce a value the checks below do not
#: independently verify against the frozen configuration.
ATTESTABLE_FIELDS = ("config_hash", "source_identity")


class CalibrationInputError(RuntimeError):
    pass


def _expected_config_hash(experiment_id: str):
    """config_hash of the frozen calibration config for this hypothesis."""
    import yaml
    from framework.provenance import config_hash
    path = REPO / CALIBRATION_CONFIGS[experiment_id]
    if not path.is_file():
        raise CalibrationInputError(f"frozen calibration config missing: {path}")
    return config_hash(yaml.safe_load(path.read_text())), yaml.safe_load(path.read_text())


def sigma_upper_confidence_bound(sd: float, n: int, level: float = SIGMA_UCL_LEVEL) -> float:
    """One-sided upper (1-alpha) confidence bound for sigma (chi-square)."""
    df = n - 1
    if df < 1:
        raise CalibrationInputError("need n >= 2 for a variance bound.")
    return float(sd * math.sqrt(df / stats.chi2.ppf(1 - level, df)))


def load_attestation(path: Path | None) -> dict:
    """
    Load and structurally validate a provenance attestation.

    An attestation is a signed-by-hand statement that a binding the
    certificate failed to record was nonetheless established by the run,
    together with the external evidence for it (a run log line, a commit
    id). It is deliberately a separate file: it is a human claim, it is
    reviewable on its own, and it must never be confused with something
    the pipeline produced.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise CalibrationInputError(f"attestation file not found: {path}")
    try:
        att = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationInputError(f"{path}: not valid JSON ({exc}).") from exc
    if att.get("attestation_version") != ATTESTATION_VERSION:
        raise CalibrationInputError(
            f"{path}: attestation_version {att.get('attestation_version')!r}; this reader "
            f"understands version {ATTESTATION_VERSION}.")
    for key in ("reason", "attested_by", "attested_at_utc"):
        if not str(att.get(key) or "").strip():
            raise CalibrationInputError(
                f"{path}: {key!r} is required and must be non-empty. An attestation without "
                "a recorded reason and author is an unsourced assertion.")
    experiments = att.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise CalibrationInputError(f"{path}: 'experiments' must be a non-empty object.")
    for exp, fields in experiments.items():
        if exp not in EXPECTED_EXPERIMENTS:
            raise CalibrationInputError(
                f"{path}: attests to {exp!r}, which is not in {EXPECTED_EXPERIMENTS}.")
        if not isinstance(fields, dict):
            raise CalibrationInputError(f"{path}: {exp}: expected an object of fields.")
        for name, entry in fields.items():
            if name not in ATTESTABLE_FIELDS:
                raise CalibrationInputError(
                    f"{path}: {exp}: field {name!r} is not attestable. Only "
                    f"{ATTESTABLE_FIELDS} may be supplied out-of-band; everything else "
                    "must come from the certificate itself.")
            if not isinstance(entry, dict):
                raise CalibrationInputError(
                    f"{path}: {exp}.{name}: expected {{'value': ..., 'source': ...}}.")
            if not str(entry.get("value") or "").strip():
                raise CalibrationInputError(f"{path}: {exp}.{name}: empty 'value'.")
            if not str(entry.get("source") or "").strip():
                raise CalibrationInputError(
                    f"{path}: {exp}.{name}: empty 'source'. Every attested value must name "
                    "the external evidence it was recovered from.")
    return att


def _attested(att: dict, experiment: str, field: str):
    return ((att.get("experiments") or {}).get(experiment) or {}).get(field)


def _resolve_config_hash(cert: dict, path: Path, experiment: str, expected: str,
                         att: dict, binding: dict) -> None:
    """
    Establish which configuration produced this certificate.

    Preference order: the certificate's own record, then an attestation.
    Either way the value must equal the frozen config's hash - recovery
    relocates the evidence, it does not waive the check.
    """
    recorded = ((cert.get("provenance") or {}).get("config_hash")
                or cert.get("config_hash"))
    if recorded is not None:
        method, source = "certificate", "provenance.config_hash"
    else:
        entry = _attested(att, experiment, "config_hash")
        if entry is None:
            raise CalibrationInputError(
                f"{path}: no config_hash recorded. Only certificates produced by the frozen "
                "calibration configurations are accepted; a hand-constructed minimal "
                "certificate cannot supply variance for a confirmatory design. If this "
                "certificate predates the provenance block, supply the hash the run itself "
                "printed via --provenance-attestation.")
        recorded, method, source = entry["value"], "attestation", entry["source"]

    if recorded != expected:
        raise CalibrationInputError(
            f"{path}: config_hash {recorded} does not match the frozen calibration config "
            f"{CALIBRATION_CONFIGS[experiment]} ({expected}).")
    binding["config_hash"] = {"value": recorded, "method": method, "source": source}


def _resolve_source_identity(cert: dict, path: Path, experiment: str,
                             att: dict, binding: dict) -> None:
    """
    Establish which code produced this certificate.

    A source/protocol manifest hash is the strong form. A recorded git
    commit is the weak form: it identifies the tree but does not prove the
    working tree was clean, so it is accepted only via attestation and
    always marked degraded.
    """
    prov = cert.get("provenance") or {}
    strong = prov.get("source_manifest_sha256") or prov.get("protocol_manifest_sha256")
    if strong:
        binding["source_identity"] = {
            "value": strong, "method": "certificate",
            "source": ("provenance.source_manifest_sha256"
                       if prov.get("source_manifest_sha256")
                       else "provenance.protocol_manifest_sha256")}
        return

    entry = _attested(att, experiment, "source_identity")
    if entry is None:
        raise CalibrationInputError(
            f"{path}: no source/protocol manifest hash recorded; the code that produced this "
            "pilot cannot be identified. If this certificate predates the provenance block, "
            "attest the recorded git commit via --provenance-attestation.")

    commit = (((cert.get("manifest") or {}).get("software_versions") or {}).get("git_commit"))
    if commit and entry["value"] != commit:
        raise CalibrationInputError(
            f"{path}: attested source_identity {entry['value']} disagrees with the commit the "
            f"certificate itself records ({commit}).")
    binding["source_identity"] = {
        "value": entry["value"], "method": "attestation", "source": entry["source"],
        "weak": True,
        "note": ("commit identity, not a source manifest hash: it identifies the tree but "
                 "does not establish that the working tree was clean at run time")}


def _resolve_permutation(cert: dict, path: Path, cfg: dict, binding: dict) -> None:
    """
    Establish that the frozen Candidate-1 permutation was the intervention.

    The certificate-level field is checked first. Failing that, the
    per-replicate records are used: run_ev.py writes
    representation_permutation_hash into every replicate, which is a
    stronger record than one certificate-level claim because it shows the
    permutation on each individual run.

    Two conditions must hold. The treatment arm must carry the frozen
    hash, and the control arm must carry a DIFFERENT one - if every
    replicate shares a permutation then the two arms are indistinguishable
    and nothing shows the intervention was applied at all.
    """
    expected = cfg.get("expected_candidate1_permutation_sha256")
    prov = cert.get("provenance") or {}
    recorded = prov.get("candidate1_permutation_sha256") or cert.get("permutation_hash")
    if recorded is not None:
        if recorded != expected:
            raise CalibrationInputError(
                f"{path}: Candidate-1 permutation {recorded} does not match the frozen "
                f"{expected}; the pilot characterised a different intervention.")
        binding["candidate1_permutation"] = {
            "value": recorded, "method": "certificate",
            "source": "provenance.candidate1_permutation_sha256"}
        return

    reps = ((cert.get("replication") or {}).get("raw_replicate_results")) or []
    observed = [r.get("representation_permutation_hash") for r in reps
                if isinstance(r, dict) and r.get("representation_permutation_hash")]
    if not observed:
        raise CalibrationInputError(
            f"{path}: no Candidate-1 permutation recorded at certificate level and no "
            "representation_permutation_hash in any replicate record; the intervention "
            "this pilot applied cannot be identified.")

    n_treatment = sum(1 for h in observed if h == expected)
    control = sorted({h for h in observed if h != expected})
    if n_treatment == 0:
        raise CalibrationInputError(
            f"{path}: no replicate carries the frozen Candidate-1 permutation {expected}; "
            f"observed {control}. The pilot characterised a different intervention.")
    if not control:
        raise CalibrationInputError(
            f"{path}: every replicate carries the same permutation hash {expected}; the "
            "control and treatment arms are indistinguishable, so the certificate does not "
            "show that the intervention was applied to one arm only.")
    if n_treatment != REQUIRED_CALIBRATION_PAIRS:
        raise CalibrationInputError(
            f"{path}: {n_treatment} replicate(s) carry the frozen permutation, expected "
            f"{REQUIRED_CALIBRATION_PAIRS} (one per pair).")

    binding["candidate1_permutation"] = {
        "value": expected, "method": "replicate_records",
        "source": (f"replication.raw_replicate_results[*]."
                   f"representation_permutation_hash: {n_treatment} treatment replicates "
                   f"carry the frozen hash; control arm carries {control}")}


def _require_calibration(cert: dict, path: Path) -> None:
    required = {"run_mode": "calibration", "calibration": True, "non_evidentiary": True}
    for key, expected in required.items():
        if key not in cert:
            raise CalibrationInputError(
                f"{path}: missing required status field {key!r}; refusing (fail-closed).")
        if cert[key] != expected:
            raise CalibrationInputError(
                f"{path}: {key}={cert[key]!r}, required {expected!r}. Variance for power "
                "planning must come from the calibration tier, and confirmatory data must "
                "never be consumed here.")


def _paired_differences(cert: dict, path: Path) -> list:
    raw = cert.get("raw_paired_values")
    if not raw:
        raise CalibrationInputError(
            f"{path}: no raw_paired_values. sigma_Delta is defined only for the PAIRED primary "
            "hypotheses; single-arm pilots cannot supply it.")
    a, b = raw.get("condition_a"), raw.get("condition_b")
    if not isinstance(a, list) or not isinstance(b, list):
        raise CalibrationInputError(f"{path}: paired values are malformed.")
    if len(a) != len(b):
        raise CalibrationInputError(
            f"{path}: paired arrays differ in length ({len(a)} vs {len(b)}).")
    for name, arr in (("condition_a", a), ("condition_b", b)):
        for v in arr:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise CalibrationInputError(f"{path}: {name} contains a non-finite value {v!r}.")
    return [float(y) - float(x) for x, y in zip(a, b)]


def summarise(path: Path, attestation: dict | None = None) -> dict:
    attestation = attestation or {}
    if not path.is_file():
        raise CalibrationInputError(f"certificate not found: {path}")
    cert = json.loads(path.read_text())
    _require_calibration(cert, path)
    experiment = ((cert.get("audit") or {}).get("id")
                  or cert.get("experiment_id")
                  or (cert.get("manifest") or {}).get("experiment_id"))
    if experiment not in EXPECTED_EXPERIMENTS:
        raise CalibrationInputError(
            f"{path}: experiment_id {experiment!r} is not one of the primary family "
            f"{EXPECTED_EXPERIMENTS}.")

    # --- provenance: the certificate must come from the FROZEN calibration
    # --- configuration, not from a hand-constructed minimal object.
    expected_hash, cfg = _expected_config_hash(experiment)
    binding: dict = {}
    _resolve_config_hash(cert, path, experiment, expected_hash, attestation, binding)
    _resolve_source_identity(cert, path, experiment, attestation, binding)

    prov = cert.get("provenance") or {}
    plan_hash = prov.get("statistical_plan_sha256")
    actual_plan = sha256_file(REPO / "docs" / "statistical_plan.md")
    if plan_hash is not None and plan_hash != actual_plan:
        raise CalibrationInputError(
            f"{path}: statistical_plan_sha256 {plan_hash} does not match the current plan "
            f"{actual_plan}; the pilot was run under a different statistical plan.")
    binding["statistical_plan"] = (
        {"value": plan_hash, "method": "certificate", "source": "provenance.statistical_plan_sha256"}
        if plan_hash is not None else
        {"value": None, "method": "absent", "weak": True,
         "source": ("the certificate records no statistical_plan_sha256, so it cannot be "
                    "shown which revision of the statistical plan governed this pilot")})

    if experiment == "H-EV-REPRESENTATION":
        _resolve_permutation(cert, path, cfg, binding)

    diffs = _paired_differences(cert, path)
    n = len(diffs)
    if n != REQUIRED_CALIBRATION_PAIRS:
        raise CalibrationInputError(
            f"{path}: the frozen calibration design is exactly "
            f"{REQUIRED_CALIBRATION_PAIRS} valid pairs per hypothesis; got {n}. A short pilot "
            "changes the sigma precision the production sizing depends on.")
    sd = statistics.stdev(diffs)

    degraded = sorted(k for k, v in binding.items()
                      if v.get("method") in ("attestation", "replicate_records", "absent")
                      or v.get("weak"))
    return {
        "experiment_id": experiment,
        "certificate_path": str(path),
        "certificate_sha256": sha256_file(path),
        "run_mode": cert["run_mode"],
        "n_valid": n,
        "sigma_Delta": sd,
        "variance": sd ** 2,
        "sigma_Delta_upper_95": sigma_upper_confidence_bound(sd, n),
        "quantity": "sigma_Delta: SD of the paired difference; powers the paired primary test",
        "provenance_binding": binding,
        "binding_degraded": bool(degraded),
        "binding_degraded_fields": degraded,
        "git_commit": (((cert.get("manifest") or {}).get("software_versions") or {})
                       .get("git_commit")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("certificates", nargs="+", type=Path)
    ap.add_argument("--output", type=Path,
                    default=Path("results/calibration/ev_calibration_variance.json"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit fewer than both primary hypotheses (planning will then refuse)")
    ap.add_argument("--provenance-attestation", type=Path, default=None,
                    help=("JSON file supplying provenance bindings that a certificate written "
                          "before the provenance block existed does not record. Each attested "
                          "value is still checked against the frozen configuration, is recorded "
                          "in the output with its external evidence, and marks the artifact "
                          "binding_degraded=true."))
    args = ap.parse_args()

    try:
        attestation = load_attestation(args.provenance_attestation)
        entries = [summarise(p, attestation) for p in args.certificates]
        seen = [e["experiment_id"] for e in entries]
        dupes = {x for x in seen if seen.count(x) > 1}
        if dupes:
            raise CalibrationInputError(f"duplicate calibration input for {sorted(dupes)}.")
        missing = [x for x in EXPECTED_EXPERIMENTS if x not in seen]
        if missing and not args.allow_partial:
            raise CalibrationInputError(
                f"missing calibration input for {missing}. Both primary hypotheses need their "
                "own sigma_Delta; pass --allow-partial only for inspection.")
    except CalibrationInputError as exc:
        print(f"REFUSED: {exc}")
        return 1

    degraded = [e for e in entries if e["binding_degraded"]]
    commits = sorted({e["git_commit"] for e in entries if e["git_commit"]})
    warnings = []
    for e in degraded:
        warnings.append(
            f"{e['experiment_id']}: provenance binding recovered rather than certificate-"
            f"recorded for {e['binding_degraded_fields']}.")
    if len(commits) > 1:
        warnings.append(
            f"the pooled hypotheses were produced at different commits {commits}; the "
            "variance artifact pools pilots run under non-identical source trees.")

    artifact = {
        "artifact": "ev-calibration-variance-v3",
        "non_evidentiary": True,
        "generated_at_utc": utc_timestamp(),
        "purpose": ("nuisance-variance estimates for prospective power analysis only; "
                    "contains no effect estimate, no p-value and no conclusion"),
        "sigma_ucl_level": SIGMA_UCL_LEVEL,
        "inputs": entries,
        "sigma_Delta_estimates": {e["experiment_id"]: e["sigma_Delta"] for e in entries},
        "sigma_Delta_upper_95": {e["experiment_id"]: e["sigma_Delta_upper_95"] for e in entries},
        "binding_degraded": bool(degraded),
        "provenance_warnings": warnings,
        "source_commits": commits,
        "attestation": ({"reason": attestation.get("reason"),
                         "attested_by": attestation.get("attested_by"),
                         "attested_at_utc": attestation.get("attested_at_utc"),
                         "file_sha256": sha256_file(args.provenance_attestation)}
                        if attestation else None),
        "prohibition": ("These values may size an experiment. They may NOT be used to choose or "
                        "validate epsilon, to justify an assumed effect, or as evidence."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps_strict(artifact, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    for e in entries:
        print(f"  {e['experiment_id']:22} n={e['n_valid']:<3} "
              f"sigma_Delta={e['sigma_Delta']:.6g}  UCL95={e['sigma_Delta_upper_95']:.6g}"
              f"{'  [BINDING DEGRADED]' if e['binding_degraded'] else ''}")
    for w in warnings:
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())