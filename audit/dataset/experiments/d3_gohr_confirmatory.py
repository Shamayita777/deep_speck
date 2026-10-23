"""D3 — confirmatory marginal and low-order distributional consistency audit, Gohr Speck32/64, 5 rounds.

Orchestration and target declarations only. Every statistical method, family,
bound, test and decision is the generic confirmatory engine
`audit.dataset.d3_distribution_statistics` (class_stratified_audit,
verify_preregistration, verify_calibration_evidence,
evaluate_confirmatory_decision). The legacy Gohr D3 runner
(`d3_gohr_distribution_statistics.py`) and its legacy engine path are neither
imported nor changed; its evidence is untouched.

THE AUDIT IS BOUND TO A PERSISTED, ATTESTED D0 INSTANCE
-------------------------------------------------------
`--instance-dir` names a canonical Gohr instance built by
`audit.dataset.tools.d0_gohr_attested_build` (unmodified canonical.build()
plus a build-attestation sidecar). It is checked against the frozen canonical
specification, its attestation is checked against the manifest and the current
generator/build-code pins, every file and array hash is re-verified
(canonical.verify), and the partitions are LOADED, never regenerated.

GATES (all checked before any partition is loaded; a refusal writes nothing)
---------------------------------------------------------------------------
* the preregistration: approved, committed, anchored by the approver's SHA-256
  (engine verify_preregistration), with EXACTLY this protocol's roles, the
  equal six-family split of the G1/G2 budget, global alpha 0.01, and the
  tolerances derived from the frozen II-4 margin;
* the calibration: accepted by the engine for THIS runner's Gohr calibration
  tool and the bound test suite, and recording exactly this design (roles,
  allocation, global and exploratory alpha, partition sizes, dimension, class
  fraction range) -- checked here because the engine does not check roles;
* the instance: specification, attestation, generator pin, hashes, class
  fractions inside the calibrated range;
* the frozen D1 gate: the instance-bound D1 census (`--d1-dir`, written by
  d1_gohr_canonical_census) is complete, its hashes verify, it is bound to this
  exact instance, manifest and build attestation, and its outcome is PASS;
* the output directory does not exist.

ROLES: R1, R2, A, B, C, D confirmatory; E exploratory and non-gating. R1 is
confirmatory because the Gohr instance did not exist when the protocol was
fixed, so its sufficient statistic (the label counts) could not have been seen.

USAGE
    python -B -m audit.dataset.experiments.d3_gohr_confirmatory \\
        --instance-dir audit/dataset/evidence/d0_gohr_speck_r5_production \\
        --d1-dir audit/dataset/evidence/d1_gohr_speck_r5_canonical_<id> \\
        --out-dir audit/dataset/evidence/d3_gohr_speck3264_r5_confirmatory_<id> \\
        --preregistration audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json \\
        --preregistration-sha256 <anchor> \\
        --calibration-dir audit/dataset/evidence/d3_gohr_method_validity_<id> [--preflight-only]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audit.dataset import canonical as C
from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.common.provenance import array_sha256
from audit.dataset.experiments import d1_gohr_canonical_census as D1R
from audit.dataset.tools import d0_gohr_attested_build as ATT

RUNNER_SCHEMA_VERSION = "d3-gohr-certificate-1"
CERTIFICATE_NAME = "d3_gohr_certificate.json"
AUDIT_FILE_NAME = "d3_gohr_sealed_audit.json"
RUN_MANIFEST_NAME = "d3_gohr_run_manifest.json"
MC_SEED = 20261023

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_TOOL_PATH = PACKAGE_DIR / "tools" / "d3_gohr_method_validity_calibration.py"
CALIBRATION_SUITE_PATH = PACKAGE_DIR / "tests" / "test_d3_method_validity.py"

#: Frozen canonical Gohr specification (canonical.CANONICAL_SPEC: gohr-speck,
#: speck.make_train_data, 5 rounds, input difference (0x0040, 0x0000), os.urandom).
EXPECTED_SPEC = dict(C.CANONICAL_SPEC)
#: Production partition sizes (the canonical specification's).
PRODUCTION_SIZES = {n: int(EXPECTED_SPEC[f"{n}_samples"]) for n in C.PARTITIONS}
FEATURE_DIMENSION = 64
#: speck.convert_to_binary([c0l, c0r, c1l, c1r]) with 16-bit words, MSB first:
#: columns 0-31 are the first ciphertext, 32-63 the second.
FEATURE_GROUPS = {"C0": list(range(0, 32)), "C1": list(range(32, 64))}

GLOBAL_ALPHA = 0.01
#: Exploratory families are tested at the global alpha within each family, never gating.
EXPLORATORY_ALPHA = GLOBAL_ALPHA
REQUIRED_ROLES = {"R1": "confirmatory", "R2": "confirmatory", "A": "confirmatory", "B": "confirmatory",
                  "C": "confirmatory", "D": "confirmatory", "E": "exploratory"}
CONFIRMATORY_FAMILIES = tuple(f for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES
                              if REQUIRED_ROLES[f] == "confirmatory")
#: Equal split of the per-guarantee budget GLOBAL_ALPHA / GUARANTEE_COUNT over the six
#: confirmatory families; each family's alpha serves both its bounds (G1) and its tests (G2).
EXPECTED_ALLOCATION = d3.allocate_familywise_alpha(GLOBAL_ALPHA / d3.GUARANTEE_COUNT, CONFIRMATORY_FAMILIES)
SUPPORTED_CLASS_FRACTION = [0.45, 0.55]

#: Engine entry point behind one name so a test can prove a refused run never reaches it.
D3_ENGINE = d3.class_stratified_audit

#: Source of the operational margin m: II-4-GOHR-IMPL-V1, frozen (commit d89590f, 2026-10-13).
MARGIN_SOURCE = {
    "experiment": "II-4-GOHR-IMPL-V1",
    "frozen_record": "audit/implementation/preregistrations/ii-4-gohr-impl-v1.frozen.json",
    "preregistration": "audit/implementation/preregistrations/ii4_impl_gohr_v1.py",
    "field": "DESIGN = ComparisonDesign(..., delta_materiality=...)",
    "meaning": ("equivalence margin on the absolute difference in sealed test accuracy between the reference "
                "and an independent Gohr implementation (reference 5-round accuracy 0.9291)"),
}

#: Lines of speck.py the declarations rest on, verbatim.
GENERATOR_SOURCE_LINES = (
    "def make_train_data(n, nr, diff=(0x0040,0)):",
    "Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;",
    "keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);",
    "plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);",
    "plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);",
    "plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];",
    "plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);",
    "plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);",
    "ks = expand_key(keys, nr);",
    "ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);",
    "ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);",
    "X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);",
    "offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;",
)

REFERENCE = {
    "expected_class_ratio": 0.5,
    "expected_bit_probability": {0: 0.5, 1: 0.5},
    "exactness": "exact under the generator with an ideal os.urandom, not approximate",
    "justification": {
        "class_ratio": "Y = (uniform byte) & 1 is Bernoulli(1/2), independently per row.",
        "key": "Every row draws its own uniform 64-bit key (four uniform 16-bit words).",
        "Y=1 bits": ("For a fixed key, Speck encryption is a permutation of {0,1}^32. P uniform and independent "
                     "of the key makes C0 = E_k(P) uniform, and P ^ (0x0040, 0) is uniform too, so "
                     "C1 = E_k(P ^ delta) is uniform. Each bit has P(1) = 1/2 and each within-half pair is "
                     "uniform; C0 and C1 are NOT independent (the differential)."),
        "Y=0 bits": ("Both plaintexts are independent and uniform, encrypted under the same uniform key. Given k, "
                     "E_k(P0) and E_k(P1) are independent uniform 32-bit words (independent inputs through a "
                     "permutation), with a law not depending on k, so the 64-bit row is exactly uniform."),
        "pooled": "A mixture of two laws with bit marginals exactly 1/2 has marginals 1/2.",
    },
    "not_claimed": [
        "the 64-bit joint law of a Y=1 row is uniform",
        "C0 and C1 are independent given Y=1",
        "cross-half feature pairs are uniform given Y=1",
        "rows are independent beyond the ideal-os.urandom assumption",
    ],
}

EXPECTED_FAMILY_BEHAVIOUR = {
    "R1": "null exactly true (class ratio 1/2); confirmatory",
    "R2": "null exactly true (every class-conditional bit 1/2)",
    "A": "null exactly true (one generator, independent rows); rejection => split defect",
    "B": "null exactly true within Y=1; rejection => class-1 split defect",
    "C": "null exactly true within Y=0; rejection => class-0 split defect",
    "D": "null exactly true; a rejection would be a first-order label shortcut",
    "E": "EXPLORATORY: cross-half pairs expected to differ between classes; never gates",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


# ---------------------------------------------------------------------------
# Protocol declarations
# ---------------------------------------------------------------------------

def materiality_margin(repo_root: Path = REPO_ROOT) -> float:
    """delta_materiality of II-4-GOHR-IMPL-V1, read from the frozen-bound source without importing it.

    The frozen record pins the preregistration file's SHA-256; the literal is
    then parsed from that exact file. Raises SystemExit on any mismatch.
    """
    rec = json.loads((repo_root / MARGIN_SOURCE["frozen_record"]).read_text())
    src = repo_root / MARGIN_SOURCE["preregistration"]
    pinned = ((rec.get("source_manifest") or {}).get("preregistration") or {})
    if pinned.get("path") != MARGIN_SOURCE["preregistration"] or pinned.get("sha256") != _sha256(src):
        raise SystemExit("REFUSED: the II-4 preregistration no longer matches its frozen record; the margin "
                         "source cannot be trusted.")
    values = []
    for node in ast.walk(ast.parse(src.read_text())):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DESIGN" for t in node.targets)
                and isinstance(node.value, ast.Call)):
            values += [kw.value.value for kw in node.value.keywords
                       if kw.arg == "delta_materiality" and isinstance(kw.value, ast.Constant)]
    if len(values) != 1 or isinstance(values[0], bool) or not isinstance(values[0], (int, float)):
        raise SystemExit("REFUSED: DESIGN.delta_materiality is not a single numeric literal in the II-4 source.")
    return float(values[0])


def derived_tolerances(repo_root: Path = REPO_ROOT) -> dict:
    """The operational FAIL/PASS thresholds: m for class ratio and pair TVD, 2m for single-bit effects."""
    m = materiality_margin(repo_root)
    return {"class_ratio": m, "class_conditional_bit": 2 * m, "pairwise_tvd": m, "cross_class_bit": 2 * m}


def verify_generator_source(repo_root: Path = REPO_ROOT) -> dict:
    """The pinned generator: digest, HEAD blob, clean status, resolved import path and cited lines."""
    fp = ATT.source_fingerprint(Path(repo_root))
    problems = ATT.check_pins(fp)
    text = (Path(repo_root) / ATT.GENERATOR_PIN["path"]).read_text()
    absent = [ln for ln in GENERATOR_SOURCE_LINES if ln not in text]
    if absent:
        problems.append(f"speck.py no longer contains {absent}")
    mod = ATT.generator_module_record(Path(repo_root))
    problems += ATT._module_problems(mod)
    if problems:
        raise SystemExit("REFUSED: the Gohr generator is not the pinned source, so D3 was not run:\n  - "
                         + "\n  - ".join(problems))
    return {"generator_pin": dict(ATT.GENERATOR_PIN), "build_code_pins": {k: dict(v) for k, v in ATT.BUILD_CODE_PINS.items()},
            "imported_generator": {k: v for k, v in mod.items() if k != "module_id"},
            "cited_generator_lines": list(GENERATOR_SOURCE_LINES), "sources": fp["sources"]}


def check_protocol(content: dict, calibration_design: dict) -> None:
    """Refuse, before any data is loaded, a preregistration or calibration that is not exactly this protocol.

    The engine checks the joint-alpha rule, allocation equality and sizes at
    decision time but not roles; a configuration error is refused here instead.
    """
    problems = []
    if content["family_roles"] != REQUIRED_ROLES:
        problems.append(f"family roles must be exactly {REQUIRED_ROLES}")
    if content["family_alpha_allocation"] != EXPECTED_ALLOCATION:
        problems.append(f"family_alpha_allocation must be the equal six-family split {EXPECTED_ALLOCATION}")
    if content["global_familywise_alpha"] != GLOBAL_ALPHA:
        problems.append(f"global alpha must be {GLOBAL_ALPHA}")
    joint = d3.joint_alpha_problem(content["global_familywise_alpha"], content["family_alpha_allocation"])
    if joint:
        problems.append(joint)
    if content["tolerances"] != derived_tolerances():
        problems.append(f"tolerances must be the II-4-derived values {derived_tolerances()}")
    cd = calibration_design
    expect = {"global_alpha": GLOBAL_ALPHA, "family_alpha_allocation": EXPECTED_ALLOCATION,
              "family_roles": {f: REQUIRED_ROLES[f] for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES},
              "exploratory_alpha": EXPLORATORY_ALPHA, "partition_sizes": PRODUCTION_SIZES,
              "feature_dimension": FEATURE_DIMENSION, "supported_class_fraction": SUPPORTED_CLASS_FRACTION}
    for k, v in expect.items():
        if cd.get(k) != v:
            problems.append(f"calibration design {k} is {cd.get(k)!r}, expected {v!r}")
    if problems:
        raise SystemExit("REFUSED: the preregistration or calibration is not this protocol, so D3 was not run:\n  - "
                         + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Instance preflight (no statistics; hashing only)
# ---------------------------------------------------------------------------

def verify_d1_evidence(d1_dir: Path, instance_dir: Path) -> dict:
    """The frozen D1 gate: complete instance-bound D1 evidence for THIS instance, outcome PASS.

    Raises SystemExit unless the directory holds exactly the D1 certificate and
    run manifest, the manifest is COMPLETE and hashes the certificate, and both
    name this instance's id, manifest SHA-256 and build-attestation SHA-256.
    """
    d1_dir, instance_dir = Path(d1_dir), Path(instance_dir)
    problems = []
    if not d1_dir.is_dir():
        raise SystemExit(f"REFUSED: D1 evidence directory {d1_dir} is missing; the frozen D1 gate is not met.")
    names = sorted(p.name for p in d1_dir.iterdir())
    if names != sorted([D1R.CERTIFICATE_NAME, D1R.RUN_MANIFEST_NAME]):
        raise SystemExit(f"REFUSED: D1 evidence directory must hold exactly the D1 certificate and run manifest; "
                         f"found {names}.")
    cert_sha = _sha256(d1_dir / D1R.CERTIFICATE_NAME)
    man_sha = _sha256(d1_dir / D1R.RUN_MANIFEST_NAME)
    cert = json.loads((d1_dir / D1R.CERTIFICATE_NAME).read_text())
    rm = json.loads((d1_dir / D1R.RUN_MANIFEST_NAME).read_text())
    inst_id = C.load_manifest(instance_dir)["canonical_instance_id"]
    d0_man_sha = _sha256(instance_dir / C.MANIFEST_NAME)
    att_sha = _sha256(instance_dir / ATT.ATTESTATION_NAME)
    if rm.get("run_status") != "COMPLETE" or (rm.get("files") or {}).get(D1R.CERTIFICATE_NAME) != cert_sha:
        problems.append("the D1 run manifest is not COMPLETE or does not hash the certificate")
    if (cert.get("runner") or {}).get("schema_version") != D1R.RUNNER_SCHEMA_VERSION:
        problems.append("the certificate was not written by the instance-bound Gohr D1 runner")
    ci = cert.get("canonical_instance") or {}
    if not (ci.get("canonical_instance_id") == rm.get("canonical_instance_id") == inst_id):
        problems.append("D1 evidence names a different canonical instance")
    if not (ci.get("manifest_sha256") == rm.get("d0_manifest_sha256") == d0_man_sha):
        problems.append("D1 evidence names a different D0 manifest")
    if not ((cert.get("build_attestation") or {}).get("sha256") == rm.get("build_attestation_sha256") == att_sha):
        problems.append("D1 evidence names a different build attestation")
    if problems:
        raise SystemExit("REFUSED: the D1 evidence is not bound to this instance:\n  - " + "\n  - ".join(problems))
    outcome = (cert.get("verdict") or {}).get("outcome")
    if outcome != "PASS":
        raise SystemExit(f"REFUSED: the frozen D1 gate requires D1 outcome PASS; this instance's D1 outcome is "
                         f"{outcome!r}. Stop, preserve all data and evidence, and investigate.")
    return {"path": _rel(d1_dir), "certificate_sha256": cert_sha, "run_manifest_sha256": man_sha,
            "run_id": rm.get("run_id"), "outcome": outcome}


def preflight(instance_dir: Path, d1_dir: Path, *, log=print) -> dict:
    """Verify the instance's specification, attestation, generator, hashes and the D1 gate. Raises SystemExit."""
    instance_dir = Path(instance_dir)
    manifest = C.load_manifest(instance_dir)
    spec = manifest.get("specification") or {}
    expected = dict(EXPECTED_SPEC, **{f"{n}_samples": PRODUCTION_SIZES[n] for n in C.PARTITIONS})
    problems = []
    if spec != expected:
        diff = sorted(k for k in set(spec) | set(expected) if spec.get(k) != expected.get(k))
        problems.append(f"instance specification differs from the frozen Gohr specification in {diff}")
    for n in C.PARTITIONS:
        if (manifest.get("partitions") or {}).get(n, {}).get("samples") != PRODUCTION_SIZES[n]:
            problems.append(f"{n} partition size is not {PRODUCTION_SIZES[n]}")
    if problems:
        raise SystemExit("REFUSED: the instance is not the frozen Gohr D0 configuration, so D3 was not run:\n  - "
                         + "\n  - ".join(problems))
    source = verify_generator_source()
    try:
        att = ATT.verify_build_attestation(instance_dir, expected_spec=expected)
    except ATT.AttestationRejected as exc:
        raise SystemExit(f"REFUSED: {exc}") from None
    d1 = verify_d1_evidence(d1_dir, instance_dir)
    log("D3 preflight — verifying the canonical Gohr D0 instance")
    verification = C.verify(instance_dir, log=log)
    if not verification.get("verified"):
        raise SystemExit("REFUSED: the canonical D0 instance did not verify, so D3 was not run.")
    lo, hi = SUPPORTED_CLASS_FRACTION
    out_of_range = [n for n in C.PARTITIONS
                    if not lo <= manifest["partitions"][n]["label_counts"]["positive"] / PRODUCTION_SIZES[n] <= hi]
    if out_of_range:
        raise SystemExit(f"REFUSED: class fraction outside the calibrated range [{lo}, {hi}] in {out_of_range}.")
    log(f"  preflight PASSED — canonical instance {manifest['canonical_instance_id'][:16]}…")
    return {"manifest": manifest, "manifest_sha256": _sha256(instance_dir / C.MANIFEST_NAME),
            "verification": verification, "specification": spec, "source": source, "attestation": att, "d1": d1}


def load_partitions(instance_dir: Path, spec: dict, manifest: dict) -> tuple[dict, dict, dict]:
    """Load through C.load_partition (file + array hashes) and re-confirm against the manifest."""
    X, Y, hashes = {}, {}, {}
    for name in C.PARTITIONS:
        x, y = C.load_partition(instance_dir, name, spec=spec, verify_file_hash=True)
        X[name] = np.ascontiguousarray(x)
        Y[name] = np.ascontiguousarray(y)
        hashes[name] = {"features_sha256": array_sha256(X[name]), "labels_sha256": array_sha256(Y[name])}
        rec = manifest["partitions"][name]
        if (hashes[name]["features_sha256"] != rec["features_sha256"]
                or hashes[name]["labels_sha256"] != rec["labels_sha256"]):
            raise SystemExit(f"REFUSED: loaded {name} arrays do not match the verified manifest.")
    return X, Y, hashes


# ---------------------------------------------------------------------------
# Output handling (fail-closed, never overwrite)
# ---------------------------------------------------------------------------

def _refuse_existing(out_dir: Path) -> None:
    if out_dir.exists():
        raise SystemExit(f"REFUSED: --out-dir {out_dir} exists. D3 evidence is never overwritten or merged; "
                         "choose a new directory. Nothing was deleted.")


def _git_state(repo: Path) -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=str(repo), capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"head": run("rev-parse", "HEAD"), "working_tree": None if status is None else ("dirty" if status else "clean")}


def _write_atomically(out_dir: Path, files: dict, manifest_extra: dict) -> None:
    """Write the complete set or nothing, and never replace an existing directory.

    The output directory is claimed with an exclusive mkdir and the fully
    written staging directory is renamed onto it.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d3_gohr_staging_", dir=str(out_dir.parent)))
    claimed = False
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        man = dict(manifest_extra, run_status="COMPLETE",
                   files={n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
        (staging / RUN_MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {out_dir} appeared during the run. Nothing was written.") from None
        claimed = True
        os.rename(staging, out_dir)          # onto the empty directory claimed above
        claimed = False
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if claimed and out_dir.is_dir() and not any(out_dir.iterdir()):
            out_dir.rmdir()


def _code_identity() -> dict:
    files = {"engine": Path(d3.__file__), "runner": Path(__file__), "canonical": Path(C.__file__),
             "attestation_tool": Path(ATT.__file__), "calibration_tool": CALIBRATION_TOOL_PATH,
             "calibration_suite": CALIBRATION_SUITE_PATH, "generator": REPO_ROOT / ATT.GENERATOR_PIN["path"]}
    return {k: {"path": _rel(p), "sha256": _sha256(p)} for k, p in files.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Confirmatory D3 audit of the canonical Gohr instance.")
    p.add_argument("--instance-dir", type=Path, required=True)
    p.add_argument("--d1-dir", type=Path, required=True,
                   help="instance-bound D1 evidence for --instance-dir (the frozen D1 gate requires outcome PASS)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--preregistration", type=Path, required=True)
    p.add_argument("--preregistration-sha256", required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true",
                   help="run every gate and the instance verification, compute no statistic, write nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    run_id, started = uuid.uuid4().hex, _utc()
    t0 = time.time()
    _refuse_existing(a.out_dir)
    try:
        prereg = d3.verify_preregistration(a.preregistration, expected_sha256=a.preregistration_sha256)
    except d3.PreregistrationRejected as exc:
        raise SystemExit(f"REFUSED: preregistration rejected, so D3 was not run: {exc}") from None
    try:
        cal = d3.verify_calibration_evidence(a.calibration_dir, tool_path=CALIBRATION_TOOL_PATH,
                                             test_suite_path=CALIBRATION_SUITE_PATH)
    except d3.CalibrationRejected as exc:
        raise SystemExit(f"REFUSED: calibration rejected, so D3 was not run: {exc}") from None
    content = d3.validate_preregistration_content(prereg.record())
    check_protocol(content, cal.design())
    pre = preflight(a.instance_dir, a.d1_dir)
    if a.preflight_only:
        print("\n  PREFLIGHT PASSED: every gate holds; no statistic was computed and nothing was written.")
        return 0

    manifest, spec = pre["manifest"], pre["specification"]
    identity_before = _code_identity()
    X, Y, hashes = load_partitions(a.instance_dir, spec, manifest)
    print(f"\nD3 — Gohr, confirmatory, run {run_id}")
    for n in C.PARTITIONS:
        print(f"  {n:<12} n={X[n].shape[0]:>10,}  verified against manifest")
    audit = D3_ENGINE({n: (X[n], Y[n]) for n in C.PARTITIONS},
                      expected_class_ratio=REFERENCE["expected_class_ratio"],
                      expected_bit_probability=REFERENCE["expected_bit_probability"],
                      tolerances=content["tolerances"], family_alpha_allocation=content["family_alpha_allocation"],
                      mc_seed=MC_SEED, feature_groups=FEATURE_GROUPS, exploratory_alpha=EXPLORATORY_ALPHA,
                      family_roles=dict(REQUIRED_ROLES))
    del X, Y
    expected_prov = {n: {"features_sha256": manifest["partitions"][n]["features_sha256"],
                         "labels_sha256": manifest["partitions"][n]["labels_sha256"]} for n in C.PARTITIONS}
    decision = d3.evaluate_confirmatory_decision(audit, calibration=cal, preregistration=prereg,
                                                 expected_input_provenance=expected_prov)
    payload = d3.verify_audit_result(audit)
    if payload["input_provenance"] != expected_prov:
        raise SystemExit("REFUSED: the engine analysed arrays other than the D0 manifest's. Nothing was written.")
    if (_code_identity() != identity_before
            or ATT.verify_build_attestation(a.instance_dir)["sha256"] != pre["attestation"]["sha256"]
            or verify_d1_evidence(a.d1_dir, a.instance_dir) != pre["d1"]):
        raise SystemExit("REFUSED: code, the build attestation or the D1 evidence changed during the run. "
                         "Nothing was written.")

    cert = {
        "certificate_schema_version": RUNNER_SCHEMA_VERSION, "mode": "confirmatory",
        "audit": {"id": "D3", "name": "Marginal and Low-Order Distributional Consistency Audit",
                  "target": "gohr-speck", "num_rounds": EXPECTED_SPEC["num_rounds"],
                  "method_version": d3.D3_METHOD_VERSION, "decision_rule_version": d3.DECISION_RULE_VERSION,
                  "engine_entry_point": "audit.dataset.d3_distribution_statistics.class_stratified_audit"},
        "chain": {
            "preregistration": {"path": _rel(Path(prereg.path)), "sha256": prereg.record_sha256,
                                "commit": prereg.commit, "commit_author": prereg.commit_author,
                                "commit_utc": prereg.commit_utc},
            "calibration": dict(cal.as_record(), tool_path=_rel(CALIBRATION_TOOL_PATH),
                                suite_path=_rel(CALIBRATION_SUITE_PATH)),
            "d0_instance": {"canonical_instance_id": manifest["canonical_instance_id"],
                            "manifest_path": _rel(Path(a.instance_dir) / C.MANIFEST_NAME),
                            "manifest_sha256": pre["manifest_sha256"],
                            "specification_sha256": manifest["specification_sha256"],
                            "partition_hashes": {n: {k: manifest["partitions"][n][k] for k in
                                                     ("file_sha256", "features_sha256", "labels_sha256", "samples")}
                                                 for n in C.PARTITIONS}},
            "d1_census": pre["d1"],
            "build_attestation": {"path": _rel(Path(a.instance_dir) / ATT.ATTESTATION_NAME),
                                  "sha256": pre["attestation"]["sha256"],
                                  "schema_version": pre["attestation"]["attestation"]["schema_version"]},
            "audit_seal_sha256": audit.seal_sha256,
        },
        "d0_verification": {"verified": True, "checked_utc": pre["verification"].get("checked_utc"),
                            "canonical_instance_id": pre["verification"].get("canonical_instance_id")},
        "loaded_partition_hashes": hashes,
        "audited_input_provenance": payload["input_provenance"],
        "provenance_link": "audited_input_provenance == loaded_partition_hashes == D0 manifest (checked)",
        "generator_source": pre["source"],
        "declarations": {"reference": {**REFERENCE, "expected_bit_probability": {"0": 0.5, "1": 0.5}},
                         "expected_family_behaviour": EXPECTED_FAMILY_BEHAVIOUR, "feature_groups": FEATURE_GROUPS,
                         "margin_source": MARGIN_SOURCE},
        "design": payload["design"], "configuration": payload["configuration"],
        "families": [{"family_id": f["family_id"], "role": f["role"], "hypothesis_kind": f["hypothesis_kind"],
                      "hypothesis_count": f["hypothesis_count"], "familywise_alpha": f["familywise_alpha"],
                      "per_hypothesis_alpha": f["per_hypothesis_alpha"]} for f in payload["families"]],
        "decision": decision,
        "run_identity": {"run_id": run_id, "started_utc": started, "finished_utc": _utc(),
                         "elapsed_seconds": round(time.time() - t0, 1)},
        "code_identity": dict(identity_before, git=_git_state(REPO_ROOT)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__, "platform": platform.platform()},
        "scope_exclusions": {"exact_duplicates": "D1", "sample_dependence": "D2",
                             "full_joint_distribution": "not claimed"},
        "interpretation": ("confirmatory D3 bound report under preregistered OPERATIONAL thresholds derived by "
                           "convention from the frozen II-4 accuracy margin; they are not a mathematical conversion "
                           "from accuracy to TVD and not validated practical-equivalence margins. A family PASS means "
                           "only that every simultaneous upper bound is at or below its threshold. Every guarantee is "
                           "conditional on i.i.d. rows within each partition (and, for the exact tests, independent "
                           "partitions); single-bit and bit-pair effects only, not the 64-bit model or the joint "
                           "law; E is exploratory and non-gating; PRACTICAL_UNSUPPORTED is not a deviation"),
    }
    files = {AUDIT_FILE_NAME: audit.sealed_json,
             CERTIFICATE_NAME: json.dumps(cert, indent=2, sort_keys=True, allow_nan=False, default=str).encode()}
    _write_atomically(a.out_dir, files, {"run_id": run_id, "mode": "confirmatory", "started_utc": started,
                                         "canonical_instance_id": manifest["canonical_instance_id"],
                                         "audit_seal_sha256": audit.seal_sha256,
                                         "preregistration_sha256": prereg.record_sha256,
                                         "calibration_run_id": cal.run_id,
                                         "build_attestation_sha256": pre["attestation"]["sha256"],
                                         "d1_certificate_sha256": pre["d1"]["certificate_sha256"]})
    print(f"\n  certificate: {a.out_dir / CERTIFICATE_NAME}")
    print(f"  outcome: {decision['outcome']}")
    for r in decision["outcome_overridden_because"]:
        print(f"    gate: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
