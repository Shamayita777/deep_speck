"""D3 — marginal and low-order distributional consistency, AutoND.

Orchestration and target declarations only. Every statistical method,
family, bound, gate and decision is the generic confirmatory engine in
`audit.dataset.d3_distribution_statistics` (method version 5). The legacy
D3 path (used by the historical Gohr runner) is never imported here.

THE AUDIT IS BOUND TO A PERSISTED D0 INSTANCE
---------------------------------------------
`--instance-dir` names an existing canonical AutoND D0 instance. It is
verified in full (C.verify), checked against the frozen case-study
configuration and source pin, and its partitions are LOADED with file and
array hashes re-checked -- never regenerated. The generic engine records the
array hashes of what it analysed; the decision requires them to equal the D0
manifest's.

TWO MODES (explicit; there is no default)
-----------------------------------------
blinded-integration
    Runs the complete path on the real instance with NO tolerances and the
    production roles and joint-alpha allocation, verifies every integrity and provenance link, and
    writes a certificate that deliberately contains NO statistical result
    (no p-value, effect, bound, category or outcome). It exists to prove the
    path works on the production data without exposing outcomes before the
    tolerances are preregistered. It is not evidence about the dataset.
confirmatory
    Requires an approved, committed preregistration (path + SHA-256 anchor)
    and a calibration run. Both are verified BEFORE any data is loaded; a
    rejection stops the run with nothing written. Tolerances, global alpha,
    family allocation and roles come only from the preregistration, which
    must use REQUIRED_ROLES and satisfy the engine's joint-alpha rule.

ROLES ON THIS INSTANCE
----------------------
R1 is exploratory (non-gating): its complete sufficient statistic, the
per-partition label counts, was examined in the D0 manifest before any
preregistration was approved. R2, A, B, C and D are confirmatory; E is
exploratory. The global alpha is joint over bounds (G1) and detections (G2).

TARGET DECLARATIONS (verified against the pinned source, not assumed)
---------------------------------------------------------------------
`main.make_train_data` at upstream 8d66e0a4, main.py lines 15-25:

    keys0, pt0 uniform (urandom); keys1 = keys0 ^ delta_key; pt1 = pt0 ^ delta_state
    C0 = E(pt0, keys0, nr); C1 = E(pt1, keys1, nr); C = hstack([C0, C1])
    Y = urandom byte & 1
    C[Y == 0] = urandom bits reshaped to (num_rand_samples, 2 * plain_bits)

The last line replaces the WHOLE 64-bit row of every Y == 0 sample. Hence:
Y ~ Bernoulli(1/2) exactly; for Y == 1, each Speck round and the bit/word
conversions are bijections for a fixed key, so C0 and C1 are each exactly
uniform on {0,1}^32 (every bit 1/2, every within-half pair uniform); for
Y == 0 all 64 bits are independent uniform. NOT implied: a uniform 64-bit
joint law for Y == 1, independence of C0 and C1, or anything about
cross-half pairs given Y == 1 (the differential the distinguisher learns).
`verify_generator_source()` re-reads the cited lines so this cannot drift.

USAGE
    python -m audit.dataset.experiments.d3_autond_distribution --mode blinded-integration \\
        --instance-dir audit/dataset/evidence/d0_autond_speck3264_r8_production \\
        --out-dir audit/dataset/evidence/d3_autond_speck3264_r8_blinded_<id>
    python -m audit.dataset.experiments.d3_autond_distribution --mode confirmatory \\
        --instance-dir ... --out-dir ... --preregistration PATH --preregistration-sha256 HEX \\
        --calibration-dir audit/dataset/evidence/d3_method_validity_<id>
"""
from __future__ import annotations

import argparse
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
from audit.dataset.adapters.autond import AutoNDDatasetAdapter
from audit.dataset.common.provenance import array_sha256
from audit.dataset.experiments.d0_autond_canonical import AUTOND_SOURCE_PIN

RUNNER_SCHEMA_VERSION = "d3-autond-certificate-2"
CERTIFICATE_NAME = "d3_autond_certificate.json"
AUDIT_FILE_NAME = "d3_autond_sealed_audit.json"
RUN_MANIFEST_NAME = "d3_autond_run_manifest.json"
MC_SEED = 20261022

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_TOOL_PATH = PACKAGE_DIR / "tools" / "d3_method_validity_calibration.py"
CALIBRATION_SUITE_PATH = PACKAGE_DIR / "tests" / "test_d3_method_validity.py"

#: The frozen case-study configuration the declarations below are argued for.
EXPECTED = {"target": "autond", "dataset_id": "autond-speck3264", "cipher": "speck3264",
            "scenario": "single-key", "num_rounds": 8, "feature_dtype": "uint8"}
EXPECTED_DELTA_PLAIN = 0x400000
EXPECTED_DELTA_KEY = 0x0
EXPECTED_FEATURE_BITS = 64
EXPECTED_PLAIN_BITS = 32
EXPECTED_LAYOUT = "np.hstack([C0, C1])"

#: Engine entry point behind one name so a test can prove a failed preflight
#: never reaches it (the D2 runner's convention).
D3_ENGINE = d3.class_stratified_audit
#: Adapter constructor behind one name for the same reason.
ADAPTER_FACTORY = AutoNDDatasetAdapter

#: Lines of the pinned sources the declarations rest on, verbatim.
GENERATOR_SOURCE_LINES = (
    "keys0 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1)",
    "pt0 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);",
    "keys1 = keys0^delta_key",
    "pt1 = pt0^delta_state",
    "C0 = encryption_function(pt0, keys0, nr)",
    "C1 = encryption_function(pt1, keys1, nr)",
    "C = np.hstack([C0, C1])",
    "Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;",
    "C[Y==0] = (np.frombuffer(urandom(num_rand_samples*C0.shape[1]*2),dtype=np.uint8)&1)"
    ".reshape(num_rand_samples, -1)",
)
CIPHER_SOURCE_LINES = (
    "def enc_one_round(p, k):", "c0 = ror(c0, ALPHA());", "c0 = (c0 + c1) & MASK_VAL;",
    "c0 = c0 ^ k;", "c1 = rol(c1, BETA());", "c1 = c1 ^ c0;",
)

REFERENCE = {
    "expected_class_ratio": 0.5,
    "expected_bit_probability": {0: 0.5, 1: 0.5},
    "exactness": "exact under the generator, not approximate",
    "justification": {
        "class_ratio": "Y = (uniform byte) & 1 is Bernoulli(1/2), independently per row.",
        "Y=1 bits": ("For fixed k every round and the bit/word conversions are bijections, so E_k permutes "
                     "{0,1}^32; P uniform and independent of k makes C0 = E_k(P) and C1 = E_k(P ^ delta) "
                     "uniform. Each bit has P(1) = 1/2 and each within-half pair is uniform."),
        "Y=0 bits": "Both halves are replaced by independent uniform bits.",
        "pooled": "A mixture of two laws with bit marginals exactly 1/2 has marginals 1/2.",
    },
    "not_claimed": [
        "the 64-bit joint law of a Y=1 row is uniform",
        "C0 and C1 are independent given Y=1",
        "cross-half feature pairs are uniform given Y=1",
        "rows are independent beyond the ideal-os.urandom assumption (D2 tests dependence)",
    ],
}

EXPECTED_FAMILY_BEHAVIOUR = {
    "R1": "null exactly true (class ratio 1/2); NON-GATING here: its sufficient statistic was examined in D0",
    "R2": "null exactly true (every class-conditional bit 1/2)",
    "A": "null exactly true (one generator, independent rows); rejection => split defect",
    "B": "null exactly true within Y=1; rejection => class-1 split defect",
    "C": "null exactly true within Y=0; rejection => class-0 split defect",
    "D": "null exactly true; a rejection would be a first-order label shortcut",
    "E": "EXPLORATORY: cross-half pairs expected to differ between classes; never gates",
}

#: Roles on the canonical instance. R1 is non-gating because its sufficient
#: statistic (the per-partition label counts) was examined in the D0 manifest
#: before the preregistration was approved. E, if preregistered, is exploratory.
REQUIRED_ROLES = {"R1": "exploratory", "R2": "confirmatory", "A": "confirmatory", "B": "confirmatory",
                  "C": "confirmatory", "D": "confirmatory"}
CONFIRMATORY_FAMILIES = tuple(f for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES
                              if REQUIRED_ROLES[f] == "confirmatory")
#: Global alpha of the blinded integration path (joint; see d3.joint_alpha_problem).
BLINDED_GLOBAL_ALPHA = 0.01

#: Published value the CANDIDATE tolerances derive from (never a default).
PUBLISHED_KEY = ("speck3264_round8", "paper_table_6")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Declarations and their verification
# ---------------------------------------------------------------------------

def verify_generator_source(autond_root: Path | str) -> dict[str, Any]:
    """Re-hash the pinned files and confirm the lines the declarations cite (bytes only)."""
    root = Path(autond_root).expanduser().resolve()
    pins = AUTOND_SOURCE_PIN["source_digests"]
    checked = {}
    for rel in ("main.py", "ciphers/speck3264.py"):
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"REFUSED: {path} is missing; the D3 declaration cannot be checked.")
        digest = _sha256(path)
        if digest != pins.get(rel):
            raise SystemExit(f"REFUSED: {rel} digest {digest} differs from the frozen pin {pins.get(rel)}.")
        checked[rel] = digest
    for rel, lines in (("main.py", GENERATOR_SOURCE_LINES), ("ciphers/speck3264.py", CIPHER_SOURCE_LINES)):
        text = (root / rel).read_text()
        absent = [ln for ln in lines if ln not in text]
        if absent:
            raise SystemExit(f"REFUSED: {rel} no longer contains {absent}; the D3 declaration is stale.")
    return {"autond_root": str(root), "upstream_commit_pinned": AUTOND_SOURCE_PIN["upstream_commit"],
            "source_digests_verified": checked, "cited_generator_lines": list(GENERATOR_SOURCE_LINES),
            "cited_cipher_lines": list(CIPHER_SOURCE_LINES)}


def published_accuracy(published_values: Any) -> float:
    """The published round-8 accuracy, read strictly; no fallback."""
    case, field = PUBLISHED_KEY
    rec = published_values.get(case) if isinstance(published_values, dict) else None
    if not isinstance(rec, dict) or field not in rec:
        raise ValueError(f"published value {case}.{field} is missing from the authoritative source.")
    v = rec[field]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.5 < float(v) < 1.0:
        raise ValueError(f"published value {case}.{field} = {v!r} is not an accuracy in (0.5, 1).")
    if rec.get("agree") is not True:
        raise ValueError(f"published values for {case} are not recorded as agreeing; refusing to choose one.")
    return float(v)


def margin_derived_tolerance_candidates() -> dict[str, Any]:
    """CANDIDATE tolerances from the operator-approved accuracy margin. Never used as defaults.

    m = MARGIN_FRACTION * (published - 0.5). Each tolerance is the largest
    effect that cannot by itself move a classifier restricted to the tested
    feature(s) by more than m: class ratio m; bit vs 1/2 in one class 2m;
    cross-class bit 2m; pair TVD m (|E_P f - E_Q f| <= TVD). A full-width
    classifier can exploit differences D3 does not measure.
    """
    from audit.targets.autond.preregistrations.autond_ii_v2 import (
        MARGIN_FRACTION, PUBLISHED_VALUES, equivalence_margin)
    m = equivalence_margin(published_accuracy(PUBLISHED_VALUES))
    return {"source": "audit.targets.autond.preregistrations.autond_ii_v2.equivalence_margin",
            "margin_fraction": MARGIN_FRACTION, "accuracy_margin": m,
            "tolerances": {"class_ratio": m, "class_conditional_bit": 2 * m, "pairwise_tvd": m,
                           "cross_class_bit": 2 * m},
            "status": "CANDIDATE: adopting it is a preregistration decision"}


def feature_groups_from(spec: dict, adapter: Any) -> dict[str, list[int]]:
    """C0/C1 column groups, from the D0 specification and cross-checked against the adapter."""
    rep = spec["representation"]
    c0, c1 = rep["c0_columns"], rep["c1_columns"]
    groups = {"C0": list(range(int(c0[0]), int(c0[1]))), "C1": list(range(int(c1[0]), int(c1[1])))}
    half = int(adapter.half_width)
    if groups != {"C0": list(range(0, half)), "C1": list(range(half, 2 * half))}:
        raise SystemExit(f"REFUSED: D0 column groups {c0}/{c1} disagree with the adapter's half width {half}.")
    return groups


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(instance_dir: Path, *, autond_root: str | None = None, log=print) -> dict:
    """Verify the D0 instance, the frozen configuration, the pin and the adapter. Raises SystemExit."""
    instance_dir = Path(instance_dir)
    manifest = C.load_manifest(instance_dir)
    log("D3 preflight — verifying the canonical D0 instance")
    verification = C.verify(instance_dir, log=log)
    if not verification.get("verified"):
        raise SystemExit("REFUSED: the canonical D0 instance did not verify, so D3 was not run. "
                         "No certificate is produced for an unverified instance.")
    spec = manifest["specification"]
    problems = []
    for field, want in EXPECTED.items():
        if spec.get(field) != want:
            problems.append(f"{field} is {spec.get(field)!r}, expected {want!r}")
    for field in ("verified_configuration", "delta_plain_verified"):
        if spec.get(field) is not True:
            problems.append(f"{field} is {spec.get(field)!r}, expected True")
    for field, want in (("delta_plain_hex", EXPECTED_DELTA_PLAIN), ("delta_key_hex", EXPECTED_DELTA_KEY)):
        try:
            if int(str(spec.get(field)), 16) != want:
                problems.append(f"{field} is {spec.get(field)!r}, expected {hex(want)}")
        except ValueError:
            problems.append(f"{field} is {spec.get(field)!r}, not a hex integer")
    rep = spec.get("representation") or {}
    if (rep.get("feature_bits") != EXPECTED_FEATURE_BITS or rep.get("plain_bits") != EXPECTED_PLAIN_BITS
            or rep.get("layout") != EXPECTED_LAYOUT):
        problems.append(f"representation is {rep!r}")
    for part in C.PARTITIONS:
        v = spec.get(f"{part}_samples")
        if isinstance(v, bool) or not isinstance(v, int) or v < 1 or manifest["partitions"][part]["samples"] != v:
            problems.append(f"{part}_samples is {v!r} or disagrees with the partition record")
    if problems:
        raise SystemExit("REFUSED: the canonical instance is outside the frozen case-study configuration, "
                         "so D3 was not run:\n  - " + "\n  - ".join(problems))
    digests = (spec.get("adapter_provenance") or {}).get("source_digests") or spec.get("source_digests")
    frozen = AUTOND_SOURCE_PIN["source_digests"]
    bad = {k: v for k, v in frozen.items() if (digests or {}).get(k) != v}
    if not digests or bad:
        raise SystemExit(f"REFUSED: the instance's recorded AutoND source does not match the frozen pin: {bad}.")
    if spec.get("source_commit_observed") != AUTOND_SOURCE_PIN["upstream_commit"]:
        raise SystemExit(f"REFUSED: the instance records AutoND revision {spec.get('source_commit_observed')!r}, "
                         f"but the frozen pin is {AUTOND_SOURCE_PIN['upstream_commit']}.")
    root = autond_root or spec["autond_root"]
    source = verify_generator_source(root)
    adapter = ADAPTER_FACTORY(
        autond_root=root, cipher_name=spec["cipher"], delta_plain_int=int(spec["delta_plain_hex"], 16),
        delta_key_int=int(spec["delta_key_hex"], 16), num_rounds=int(spec["num_rounds"]),
        scenario=spec["scenario"], expected_source_digests=frozen,
        expected_upstream_commit=AUTOND_SOURCE_PIN["upstream_commit"])
    mism = []
    if adapter.dataset_id != spec["dataset_id"]:
        mism.append(f"adapter dataset_id {adapter.dataset_id!r} != {spec['dataset_id']!r}")
    if int(adapter.feature_bits) != EXPECTED_FEATURE_BITS:
        mism.append(f"adapter feature_bits {adapter.feature_bits}")
    if dict(adapter.source_digests).get("main.py") != frozen["main.py"]:
        mism.append("adapter imported a main.py other than the pinned one")
    if mism:
        raise SystemExit("REFUSED: the AutoND adapter disagrees with the D0 instance: " + "; ".join(mism))
    groups = feature_groups_from(spec, adapter)
    log(f"  preflight PASSED — canonical instance {manifest['canonical_instance_id'][:16]}…")
    return {"manifest": manifest, "verification": verification, "specification": spec,
            "source": source, "adapter": adapter, "feature_groups": groups}


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


def _write_atomically(out_dir: Path, files: dict[str, bytes], manifest_extra: dict) -> None:
    """Write the complete set or nothing, and never replace an existing directory.

    The output directory is CLAIMED with an exclusive mkdir (fails if anything
    exists at that path), and the fully written staging directory is then
    renamed onto that empty directory this process created. A check-then-rename
    alone would leave a window in which a directory created by someone else
    could be replaced.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d3_autond_staging_", dir=str(out_dir.parent)))
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


def check_gate_compatibility(content: dict, calibration_design: dict) -> None:
    """Refuse, before any data is loaded, a preregistration the engine or the calibration cannot accept.

    The engine would otherwise discover these only after the audit (as an
    INCONCLUSIVE decision or a ValueError); a configuration error is refused
    up front instead.
    """
    confirm = set(CONFIRMATORY_FAMILIES)
    problems = []
    if set(content["tolerances"]) != set(d3.TOLERANCE_KEYS):
        problems.append(f"tolerances must declare exactly {d3.TOLERANCE_KEYS}")
    if set(content["family_alpha_allocation"]) != confirm:
        problems.append(f"family_alpha_allocation must cover exactly {sorted(confirm)}")
    roles = content["family_roles"]
    if {f: roles.get(f) for f in REQUIRED_ROLES} != REQUIRED_ROLES:
        problems.append(f"family roles must be {REQUIRED_ROLES} (R1 non-gating: its sufficient statistic "
                        "was examined in the D0 manifest)")
    if set(roles) - set(REQUIRED_ROLES) - {"E"} or roles.get("E", "exploratory") != "exploratory":
        problems.append("the only optional family is E, and it must be exploratory")
    joint = d3.joint_alpha_problem(content["global_familywise_alpha"], content["family_alpha_allocation"])
    if joint:
        problems.append(joint)
    if content["global_familywise_alpha"] != calibration_design.get("global_alpha"):
        problems.append(f"preregistered global alpha {content['global_familywise_alpha']} != calibrated "
                        f"{calibration_design.get('global_alpha')}")
    if content["family_alpha_allocation"] != calibration_design.get("family_alpha_allocation"):
        problems.append("preregistered allocation differs from the calibrated allocation")
    if problems:
        raise SystemExit("REFUSED: the preregistration is incompatible with the engine or the calibration, "
                         "so D3 was not run:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="D3 distributional consistency audit for the canonical AutoND instance.")
    p.add_argument("--mode", required=True, choices=("blinded-integration", "confirmatory"))
    p.add_argument("--instance-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--autond-root", default=None, help="pinned AutoND checkout (default: as recorded in D0)")
    p.add_argument("--preregistration", type=Path, default=None)
    p.add_argument("--preregistration-sha256", default=None)
    p.add_argument("--calibration-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    run_id, started = uuid.uuid4().hex, _utc()
    t0 = time.time()
    _refuse_existing(a.out_dir)
    confirmatory = a.mode == "confirmatory"
    prereg = cal = None
    if confirmatory:
        if not (a.preregistration and a.preregistration_sha256 and a.calibration_dir):
            raise SystemExit("REFUSED: confirmatory mode requires --preregistration, --preregistration-sha256 "
                             "and --calibration-dir. Nothing was run.")
        try:
            prereg = d3.verify_preregistration(a.preregistration, expected_sha256=a.preregistration_sha256)
        except d3.PreregistrationRejected as exc:
            raise SystemExit(f"REFUSED: preregistration rejected, so D3 was not run: {exc}") from None
        try:
            cal = d3.verify_calibration_evidence(a.calibration_dir, tool_path=CALIBRATION_TOOL_PATH,
                                                 test_suite_path=CALIBRATION_SUITE_PATH)
        except d3.CalibrationRejected as exc:
            raise SystemExit(f"REFUSED: calibration rejected, so D3 was not run: {exc}") from None
        check_gate_compatibility(d3.validate_preregistration_content(prereg.record()), cal.design())
    elif a.preregistration or a.preregistration_sha256 or a.calibration_dir:
        raise SystemExit("REFUSED: blinded-integration mode takes no preregistration or calibration; "
                         "a confirmatory run must say so with --mode confirmatory.")

    pre = preflight(a.instance_dir, autond_root=a.autond_root)
    manifest, spec = pre["manifest"], pre["specification"]
    X, Y, hashes = load_partitions(a.instance_dir, spec, manifest)

    if confirmatory:
        content = d3.validate_preregistration_content(prereg.record())
        tolerances = content["tolerances"]
        alloc = content["family_alpha_allocation"]
        # Exploratory families (R1, and E when preregistered) are tested at the
        # global alpha WITHIN each family. They are outside the confirmatory
        # budget and can never change the outcome (the engine marks them NOT_GATED).
        exploratory_alpha = content["global_familywise_alpha"]
        roles = {**REQUIRED_ROLES, **({"E": "exploratory"} if "E" in content["family_roles"] else {})}
    else:
        tolerances = {k: None for k in d3.TOLERANCE_KEYS}
        alloc = d3.allocate_familywise_alpha(BLINDED_GLOBAL_ALPHA / d3.GUARANTEE_COUNT, CONFIRMATORY_FAMILIES)
        exploratory_alpha = BLINDED_GLOBAL_ALPHA
        roles = {**REQUIRED_ROLES, "E": "exploratory"}
    if set(tolerances) != set(d3.TOLERANCE_KEYS):
        raise SystemExit(f"REFUSED: the preregistration must declare exactly {d3.TOLERANCE_KEYS}.")

    print(f"\nD3 — AutoND, mode {a.mode}, run {run_id}")
    for n in C.PARTITIONS:
        print(f"  {n:<12} n={X[n].shape[0]:>10,}  verified against manifest")
    audit = D3_ENGINE({n: (X[n], Y[n]) for n in C.PARTITIONS},
                      expected_class_ratio=REFERENCE["expected_class_ratio"],
                      expected_bit_probability=REFERENCE["expected_bit_probability"],
                      tolerances=tolerances, family_alpha_allocation=alloc, mc_seed=MC_SEED,
                      feature_groups=pre["feature_groups"], exploratory_alpha=exploratory_alpha,
                      family_roles=roles)
    del X, Y
    expected_prov = {n: {"features_sha256": manifest["partitions"][n]["features_sha256"],
                         "labels_sha256": manifest["partitions"][n]["labels_sha256"]} for n in C.PARTITIONS}
    decision = d3.evaluate_confirmatory_decision(audit, calibration=cal, preregistration=prereg,
                                                 expected_input_provenance=expected_prov)
    payload = d3.verify_audit_result(audit)
    if payload["input_provenance"] != expected_prov:
        raise SystemExit("REFUSED: the engine analysed arrays other than the D0 manifest's. Nothing was written.")

    common = {
        "certificate_schema_version": RUNNER_SCHEMA_VERSION, "mode": a.mode,
        "audit": {"id": "D3", "name": "Marginal and Low-Order Distributional Consistency Audit",
                  "target": "autond", "method_version": d3.D3_METHOD_VERSION,
                  "decision_rule_version": d3.DECISION_RULE_VERSION,
                  "engine_entry_point": "audit.dataset.d3_distribution_statistics.class_stratified_audit"},
        "canonical_instance": C.binding_record(a.instance_dir),
        "d0_verification": {k: pre["verification"][k] for k in ("verified", "checked_utc", "canonical_instance_id",
                                                                "schema_version")},
        "loaded_partition_hashes": hashes,
        "audited_input_provenance": payload["input_provenance"],
        "provenance_link": "audited_input_provenance == loaded_partition_hashes == D0 manifest (checked)",
        "autond_source": pre["source"],
        "adapter": {"class": f"{type(pre['adapter']).__module__}.{type(pre['adapter']).__name__}",
                    "dataset_id": pre["adapter"].dataset_id, "feature_bits": int(pre["adapter"].feature_bits),
                    "half_width": int(pre["adapter"].half_width)},
        "declarations": {"reference": {**REFERENCE, "expected_bit_probability": {"0": 0.5, "1": 0.5}},
                         "expected_family_behaviour": EXPECTED_FAMILY_BEHAVIOUR,
                         "feature_groups": pre["feature_groups"]},
        "design": payload["design"],
        "families": [{"family_id": f["family_id"], "role": f["role"], "hypothesis_kind": f["hypothesis_kind"],
                      "hypothesis_count": f["hypothesis_count"], "familywise_alpha": f["familywise_alpha"],
                      "per_hypothesis_alpha": f["per_hypothesis_alpha"]} for f in payload["families"]],
        "audit_seal_sha256": audit.seal_sha256,
        "gates": {"outcome_overridden_because": decision["outcome_overridden_because"]},
        "run_identity": {"run_id": run_id, "started_utc": started, "finished_utc": _utc(),
                         "elapsed_seconds": round(time.time() - t0, 1)},
        "code_identity": {"engine_sha256": d3.engine_sha256(), "runner_sha256": _sha256(Path(__file__)),
                          "adapter_sha256": _sha256(Path(sys.modules[AutoNDDatasetAdapter.__module__].__file__)),
                          "git": _git_state(PACKAGE_DIR.parents[1])},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__, "platform": platform.platform(),
                        "executable": sys.executable},
        "scope_exclusions": {"exact_duplicates": "D1", "sample_dependence": "D2",
                             "full_joint_distribution": "not claimed"},
    }
    files: dict[str, bytes] = {}
    if confirmatory:
        cert = dict(common, configuration=payload["configuration"], decision=decision,
                    preregistration={"path": prereg.path, "sha256": prereg.record_sha256, "commit": prereg.commit,
                                     "commit_author": prereg.commit_author},
                    calibration=cal.as_record(), audit_file=AUDIT_FILE_NAME,
                    interpretation=("confirmatory D3 bound report and FAIL screen under the preregistered "
                                    "operational FAIL thresholds (not validated equivalence margins); a PASS "
                                    "or equivalence claim is not made at these thresholds; R1 and E are "
                                    "non-gating; every guarantee is conditional on i.i.d. rows (D2 "
                                    "INCONCLUSIVE); single-bit and bit-pair effects only, not the 64-bit "
                                    "model or the joint law"))
        files[AUDIT_FILE_NAME] = audit.sealed_json
    else:
        cert = dict(common, blinded=True,
                    configuration={"tolerances": tolerances, "family_alpha_allocation": alloc,
                                   "exploratory_alpha": exploratory_alpha, "mc_seed": MC_SEED,
                                   "status": "integration-only; not preregistered; no claim"},
                    interpretation=("BLINDED INTEGRATION RUN: the complete path ran and every integrity and "
                                    "provenance link was verified. No statistical result is recorded or reported. "
                                    "This is not evidence about the dataset and supports no D3 claim."))
    files[CERTIFICATE_NAME] = json.dumps(cert, indent=2, sort_keys=True, allow_nan=False, default=str).encode()
    _write_atomically(a.out_dir, files, {"run_id": run_id, "mode": a.mode, "started_utc": started,
                                         "canonical_instance_id": manifest["canonical_instance_id"],
                                         "audit_seal_sha256": audit.seal_sha256})
    print(f"\n  certificate: {a.out_dir / CERTIFICATE_NAME}")
    if confirmatory:
        print(f"  outcome: {decision['outcome']}")
        for r in decision["outcome_overridden_because"]:
            print(f"    gate: {r}")
    else:
        print("  blinded integration run: no statistical result recorded; this supports no D3 claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
