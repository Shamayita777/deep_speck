"""
Gohr D4 - label-shuffle controlled perturbation, Speck32/64, 5 rounds.

Target-specific experiment definition and stage orchestration only.  The
statistics, adapter contract, perturbation provenance, decision and
certificate are the shared engine (``audit.dataset.d4_controlled_perturbation``);
hashing, exclusive writes, partition digests and the overlap audit are
``audit.dataset.integrity``; the model-fit lifecycle is
``audit.dataset.adapters.gohr_d4``.

STAGES (each refuses, and writes nothing it would replace, on any failure)
--------------------------------------------------------------------------
protocol-check  Structural review of the preregistration; lists every field
                still pending.  Read-only.  Exit 0 if well-formed (pending
                fields are reported, not an error), 2 if malformed;
                --require-approved also exits 3 while anything is pending.
preflight       HUMAN-AUTHORISED.  Verifies the preregistration is approved,
                committed and matches the approver's SHA-256 anchor; the D4
                sources are clean at HEAD; --run-dir does not exist.  Then
                generates the three partitions ONCE, validates the input
                encoding from the arrays, runs the read-only overlap audit,
                applies the approved overlap policy, and writes run-dir/
                {dataset/*.npy, preflight.json} as one exclusive directory.
                A policy refusal writes only <run-dir>.preflight_refused/
                (measurements and the generated partitions, never
                regenerated or altered).  Every output directory is written
                in a hidden staging directory and renamed into place, so a
                partial write is never accepted; a leftover staging
                directory makes later stages refuse.
run             HUMAN-AUTHORISED PRODUCTION.  Re-verifies the anchored
                preregistration, preflight.json, every partition file and the
                source identity recorded at preflight; trains and evaluates;
                writes run-dir/result/ {d4_certificate.json,
                d4_run_manifest.json} once, exclusively.  Resumable: fits
                resume through the adapter's verified recovery.
verify          Independent post-run verification.  Read-only.  Recounts every
                fit's correct count and McNemar (b, c) from the persisted
                predictions and the preflight-bound test labels, recomputes
                every endpoint statistic and the outcome, and checks every
                setting against the approved design.  Exit 0 = integrity
                verified (the printed primary outcome may still be
                INCONCLUSIVE or NOT_SUPPORTED, which is never a pass);
                exit 5 = no result (NOT_COMPLETED); exit 1 = refused.
                --rerun-models additionally reloads every selected
                checkpoint (digest-checked) and requires its thresholded
                test predictions to equal the persisted ones.

Decision rule: near_chance_tost_v3 only.  The primary endpoint is near-chance
equivalence of every label-shuffled fit; the paired comparison with the
baseline is a separate endpoint (see the engine and the RATIONALE).

USAGE (from the repository root)
    PY=~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python
    $PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle protocol-check \\
        --preregistration audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json
    $PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle preflight \\
        --preregistration <P> --preregistration-sha256 <anchor> --run-dir <D>
    $PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle run \\
        --preregistration <P> --preregistration-sha256 <anchor> --run-dir <D>
    $PY -B -m audit.dataset.experiments.d4_gohr_label_shuffle verify --run-dir <D>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.dataset import d4_controlled_perturbation as d4
from audit.dataset.integrity import (
    REPOSITORY_ROOT,
    IntegrityError,
    audit_partition_overlap,
    canonical_json_bytes,
    labelled_partition_digest,
    partition_digest,
    read_json,
    sha256_bytes,
    sha256_file,
    source_identity,
    verify_anchored_committed_record,
    stale_staging_directories,
    write_directory_exclusive,
)

# ============================================================
# Target definition (fixed design; must equal the preregistration)
# ============================================================

NUM_ROUNDS = 5
DEPTH = 10
BATCH_SIZE = 5000
AUDIT_SEED = 0
BOOTSTRAP_REPLICATES = 5000
CONFIDENCE_LEVEL = 0.95
FEATURE_WIDTH = 64
INPUT_ENCODING = "raw_bits_uint8_0_1_width64_no_scaling"
CHANCE_LEVEL = 0.5
DATASET_ID = "gohr-speck"
DATASET_VERSION = "original-make-train-data"
PARTITIONS = ("train", "validation", "test")
DEFAULT_PREREGISTRATION = REPOSITORY_ROOT / "audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "audit" / "dataset" / "evidence"

SOURCE_FILES = tuple(REPOSITORY_ROOT / p for p in (
    "audit/dataset/d4_controlled_perturbation.py",
    "audit/dataset/integrity.py",
    "audit/dataset/adapters/gohr.py",
    "audit/dataset/adapters/gohr_d4.py",
    "audit/dataset/perturbations/label_shuffle.py",
    "audit/dataset/experiments/d4_gohr_label_shuffle.py",
    "speck.py",
    "train_nets.py",
))

PREFLIGHT_SCHEMA = "d4-gohr-preflight-1"
OVERLAP_POLICIES = ("refuse_on_any_cross_partition_feature_row", "record_only")
DUPLICATE_POLICIES = ("record_only", "refuse_on_any_within_partition_duplicate")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# Preregistration
# ============================================================


def check_target_design(record: dict) -> list[str]:
    """Problems if the preregistration's fixed design differs from this runner."""
    problems = []
    fixed = record.get("fixed_design", {})
    expected = {
        ("target", "num_rounds"): NUM_ROUNDS,
        ("model", "depth"): DEPTH,
        ("model", "batch_size"): BATCH_SIZE,
        ("seeds", "audit_seed"): AUDIT_SEED,
        ("bootstrap", "replicates"): BOOTSTRAP_REPLICATES,
        ("bootstrap", "confidence_level"): CONFIDENCE_LEVEL,
        ("dataset", "input_encoding"): INPUT_ENCODING,
    }
    from audit.dataset.adapters import gohr_d4
    expected[("model_selection", "rule")] = gohr_d4.MODEL_SELECTION_RULE
    for (section, key), value in expected.items():
        actual = fixed.get(section, {}).get(key)
        if actual != value:
            problems.append(f"fixed_design.{section}.{key} is {actual!r}; this runner implements {value!r}")
    for section, procedure in (("primary_endpoint", d4.NEAR_CHANCE_PROCEDURE),
                               ("supporting_endpoint", d4.PAIRED_PROCEDURE)):
        registered = fixed.get(section, {}).get("procedure")
        if registered != procedure:
            problems.append(f"fixed_design.{section}.procedure differs from the implemented "
                            f"{d4.NEAR_CHANCE_RULE} procedure (estimand, sidedness, aggregation or level)")
    if fixed.get("primary_endpoint", {}).get("procedure", {}).get("chance_reference") != CHANCE_LEVEL:
        problems.append(f"fixed_design.primary_endpoint chance reference must be {CHANCE_LEVEL}")
    if record.get("perturbation", {}).get("name") != "label_shuffle":
        problems.append("perturbation.name must be 'label_shuffle'")
    return problems


def check_decision_values(values: dict) -> list[str]:
    problems = []
    sizes = values.get("sample_sizes")
    if not (isinstance(sizes, dict) and set(sizes) == set(PARTITIONS)
            and all(isinstance(sizes[p], int) and sizes[p] >= 1 for p in PARTITIONS)):
        problems.append("decisions.sample_sizes must give positive integers for train/validation/test")
    if not (isinstance(values.get("epochs"), int) and values["epochs"] >= 1):
        problems.append("decisions.epochs must be a positive integer")
    if not (isinstance(values.get("replicates"), int) and values["replicates"] >= 2):
        problems.append("decisions.replicates must be an integer >= 2")
    if not (isinstance(values.get("alpha"), float) and 0 < values["alpha"] < 1):
        problems.append("decisions.alpha must be a float in (0, 1)")
    if not (isinstance(values.get("effect_threshold"), float) and values["effect_threshold"] >= 0):
        problems.append("decisions.effect_threshold must be a non-negative float")
    if values.get("partition_overlap_policy") not in OVERLAP_POLICIES:
        problems.append("decisions.partition_overlap_policy is not implemented")
    if values.get("within_partition_duplicates_policy") not in DUPLICATE_POLICIES:
        problems.append("decisions.within_partition_duplicates_policy is not implemented")
    if values.get("decision_rule") != d4.NEAR_CHANCE_RULE:
        problems.append(f"decisions.decision_rule must be {d4.NEAR_CHANCE_RULE}: the registered primary "
                        "endpoint is near-chance equivalence, which no paired-only rule establishes")
    margin = values.get("chance_margin")
    if not (isinstance(margin, float) and 0.0 < margin < CHANCE_LEVEL):
        problems.append("decisions.chance_margin must be a float in (0, 0.5), a fraction of accuracy")
    if values.get("paired_endpoint_role") not in d4.PAIRED_ENDPOINT_ROLES:
        problems.append(f"decisions.paired_endpoint_role must be one of {d4.PAIRED_ENDPOINT_ROLES}")
    return problems


def load_approved_preregistration(path: Path, anchor: str) -> tuple[dict, dict, str]:
    """Anchored + committed + approved + matching this runner; returns (record, values, sha)."""
    record = verify_anchored_committed_record(path, expected_sha256=anchor, what="D4 preregistration")
    commit = record.pop("_commit")
    values = d4.require_approved_preregistration(record)
    problems = check_target_design(record) + check_decision_values(values)
    if problems:
        raise IntegrityError(f"Preregistration does not match this runner: {problems}")
    approved = datetime.strptime(record["approved_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if commit["committed_utc"] and approved > datetime.fromisoformat(commit["committed_utc"]):
        raise IntegrityError("approved_utc is after the commit that records the preregistration.")
    return record, dict(values, _commit=commit), anchor


def design_of(values: dict, anchor: str) -> dict:
    """The complete design a run executes; its hash is the run_config_hash."""
    return {
        "audit": "D4", "target": "gohr-speck3264-r5", "perturbation": "label_shuffle",
        "preregistration_sha256": anchor,
        "num_rounds": NUM_ROUNDS, "depth": DEPTH, "batch_size": BATCH_SIZE,
        "audit_seed": AUDIT_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL, "input_encoding": INPUT_ENCODING,
        "chance_level": CHANCE_LEVEL, "primary_procedure": d4.NEAR_CHANCE_PROCEDURE,
        "supporting_procedure": d4.PAIRED_PROCEDURE,
        **{k: v for k, v in values.items() if not k.startswith("_")},
    }


def run_config_hash(design: dict) -> str:
    return sha256_bytes(canonical_json_bytes(design))


# ============================================================
# Data
# ============================================================


def generate_gohr_partitions(sizes: dict) -> dict:
    """Three independent speck.make_train_data calls; returns {role: (x, y)}."""
    from audit.dataset.adapters.gohr import GohrAdapter

    generator = GohrAdapter(num_rounds=NUM_ROUNDS, depth=DEPTH)
    partitions = {}
    for role in PARTITIONS:
        print(f"Generating {role} partition ({sizes[role]} rows, speck.make_train_data, os.urandom)...")
        partitions[role] = generator.generate_partition(int(sizes[role]))
    return partitions


def validate_input_encoding(partitions: dict) -> None:
    """Refuse any partition that departs from the frozen input encoding."""
    for role, (x, y) in partitions.items():
        x, y = np.asarray(x), np.asarray(y).reshape(-1)
        if x.dtype != np.uint8 or x.ndim != 2 or x.shape[1] != FEATURE_WIDTH:
            raise IntegrityError(f"{role} features have dtype {x.dtype} and shape {x.shape}; the frozen "
                                 f"encoding is uint8 with width {FEATURE_WIDTH}.")
        if x.size and int(x.max()) > 1:
            raise IntegrityError(f"{role} features contain values outside {{0, 1}}; inputs must not be rescaled.")
        if y.dtype != np.uint8 or len(y) != len(x):
            raise IntegrityError(f"{role} labels must be uint8 with one label per row.")
        if y.size and int(y.max()) > 1:
            raise IntegrityError(f"{role} labels contain values outside {{0, 1}}.")


def overlap_violations(report: dict, values: dict) -> list[str]:
    violations = []
    if values["partition_overlap_policy"] == "refuse_on_any_cross_partition_feature_row":
        violations += [f"cross {pair}: {c['rows']} shared rows" for pair, c in report["cross"].items() if c["rows"]]
    if values["within_partition_duplicates_policy"] == "refuse_on_any_within_partition_duplicate":
        violations += [f"within {role}: {w['duplicate_rows']} duplicate rows"
                       for role, w in report["within"].items() if w["duplicate_rows"]]
    return violations


def _npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    return buffer.getvalue()


def refused_directory(run_dir: Path) -> Path:
    return Path(run_dir).with_name(Path(run_dir).name + ".preflight_refused")


def load_preflight(run_dir: Path) -> tuple[dict, dict]:
    """Read preflight.json and re-verify every partition file from its bytes.

    The dataset directory must hold exactly the recorded files.
    """
    run_dir = Path(run_dir)
    stale = stale_staging_directories(run_dir)
    if stale:
        raise IntegrityError(f"Interrupted write {[p.name for p in stale]} exists beside {run_dir}; refusing.")
    preflight_path = run_dir / "preflight.json"
    preflight = read_json(preflight_path, "preflight record")
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA or preflight.get("outcome") != "PASS":
        raise IntegrityError(f"{preflight_path} is not a passing {PREFLIGHT_SCHEMA} record.")
    recorded = {preflight["partitions"][r][f"{k}_file"] for r in PARTITIONS for k in ("x", "y")}
    present = {p.relative_to(run_dir).as_posix() for p in (run_dir / "dataset").rglob("*") if p.is_file()} \
        if (run_dir / "dataset").is_dir() else set()
    if present != recorded:
        raise IntegrityError(f"Dataset directory differs from preflight.json: missing "
                             f"{sorted(recorded - present)}, unexpected {sorted(present - recorded)}.")
    partitions = {}
    for role in PARTITIONS:
        entry = preflight["partitions"][role]
        arrays = []
        for kind in ("x", "y"):
            path = run_dir / entry[f"{kind}_file"]
            if not path.is_file() or sha256_file(path) != entry[f"{kind}_sha256"]:
                raise IntegrityError(f"Partition file {path} is missing or does not match preflight.json.")
            arrays.append(np.load(path, mmap_mode="r"))
        if labelled_partition_digest(*arrays) != entry["digest"]:
            raise IntegrityError(f"{role} partition content does not reproduce its recorded digest.")
        partitions[role] = tuple(arrays)
    validate_input_encoding(partitions)
    if partition_digest({f"{r}_{k}": a for r, (x, y) in partitions.items()
                         for k, a in (("x", x), ("y", y))}) != preflight["all_partitions_digest"]:
        raise IntegrityError("The combined partition digest does not match preflight.json.")
    return preflight, partitions


# ============================================================
# Stages
# ============================================================


def stage_protocol_check(args) -> int:
    record = read_json(args.preregistration, "D4 preregistration")
    review = d4.review_preregistration(record)
    problems = review["problems"] + check_target_design(record)
    if review["values"]:
        problems += [p for p in check_decision_values(review["values"])
                     if p.split()[0].split(".")[-1] in review["values"]]
    print(json.dumps({"sha256": sha256_file(args.preregistration), "problems": problems,
                      "pending": review["pending"]}, indent=2))
    if problems:
        print("PROTOCOL MALFORMED")
        return 2
    if review["pending"]:
        print(f"PROTOCOL WELL-FORMED; {len(review['pending'])} field(s) PENDING human approval.")
        return 3 if args.require_approved else 0
    print("PROTOCOL WELL-FORMED AND APPROVED (anchoring and commit are checked at preflight).")
    return 0


def stage_preflight(args) -> int:
    run_dir = Path(args.run_dir)
    refused_dir = refused_directory(run_dir)
    for existing in (run_dir, refused_dir):
        if existing.exists():
            raise IntegrityError(f"Refusing: {existing} exists. Preflight never reuses or replaces a run directory.")
    stale = stale_staging_directories(run_dir) + stale_staging_directories(refused_dir)
    if stale:
        raise IntegrityError(f"Interrupted preflight output {[p.name for p in stale]} exists; it was never "
                             "accepted and is kept for investigation. Refusing.")
    record, values, anchor = load_approved_preregistration(args.preregistration, args.preregistration_sha256)
    identity = source_identity(SOURCE_FILES)
    dirty = [f for f, e in identity["files"].items() if not e["clean_at_head"]]
    if dirty:
        raise IntegrityError(f"D4 sources are not clean at HEAD: {dirty}")

    partitions = generate_gohr_partitions(values["sample_sizes"])
    validate_input_encoding(partitions)
    report = audit_partition_overlap(partitions)
    print(f"Partition overlap audit: {json.dumps(report, sort_keys=True)}")
    violations = overlap_violations(report, values)

    files, entries = {}, {}
    for role, (x, y) in partitions.items():
        x_bytes, y_bytes = _npy_bytes(x), _npy_bytes(y)
        files[f"dataset/{role}_x.npy"], files[f"dataset/{role}_y.npy"] = x_bytes, y_bytes
        entries[role] = {"x_file": f"dataset/{role}_x.npy", "x_sha256": sha256_bytes(x_bytes),
                         "y_file": f"dataset/{role}_y.npy", "y_sha256": sha256_bytes(y_bytes),
                         "rows": int(len(x)), "digest": labelled_partition_digest(x, y)}
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA,
        "outcome": "REFUSED" if violations else "PASS",
        "violations": violations,
        "created_utc": _utc(),
        "preregistration": {"path": values["_commit"]["path"], "sha256": anchor,
                            "commit": values["_commit"]["sha"]},
        "design": design_of(values, anchor),
        "run_config_hash": run_config_hash(design_of(values, anchor)),
        "source_identity": identity,
        "input_encoding": INPUT_ENCODING,
        "partitions": entries,
        "all_partitions_digest": partition_digest({f"{r}_{k}": a for r, (x, y) in partitions.items()
                                                   for k, a in (("x", x), ("y", y))}),
        "overlap_audit": report,
    }
    files["preflight.json"] = json.dumps(preflight, indent=2, sort_keys=True).encode() + b"\n"
    if violations:
        # The generated data and the measurements are kept together, written
        # atomically; the data is never regenerated, deduplicated or altered.
        write_directory_exclusive(refused_dir, files)
        print(f"PREFLIGHT REFUSED by the approved overlap policy: {violations}\n"
              f"Measurements and data: {refused_dir}. The protocol stops here; a new dataset needs a "
              "documented protocol amendment.")
        return 4
    write_directory_exclusive(run_dir, files)
    print(f"PREFLIGHT PASSED. Wrote {run_dir}. No model was trained and no test statistic was computed.")
    return 0


def make_adapter_factory(run_dir: Path, design: dict, partitions: dict, evaluation_digest: str):
    from audit.dataset.adapters.gohr_d4 import GohrD4Adapter

    validation_x, validation_y = partitions["validation"]

    def factory(seed: int):
        return GohrD4Adapter(
            validation_x=validation_x, validation_y=validation_y,
            evaluation_digest=evaluation_digest,
            num_rounds=NUM_ROUNDS, depth=DEPTH, epochs=design["epochs"],
            batch_size=BATCH_SIZE, seed=seed, model_directory=str(run_dir / "fits" / "models"),
        )

    return factory


def stage_run(args) -> int:
    run_dir = Path(args.run_dir)
    result_dir = run_dir / "result"
    if result_dir.exists():
        raise IntegrityError(f"Refusing: {result_dir} exists. A completed D4 result is never replaced.")
    record, values, anchor = load_approved_preregistration(args.preregistration, args.preregistration_sha256)
    preflight, partitions = load_preflight(run_dir)
    design = design_of(values, anchor)
    if preflight["design"] != design or preflight["preregistration"]["sha256"] != anchor:
        raise IntegrityError("The approved preregistration differs from the one preflight was run under.")
    identity = source_identity(SOURCE_FILES)
    if identity["files"] != preflight["source_identity"]["files"]:
        raise IntegrityError("D4 sources differ from those recorded at preflight. Refusing to run.")
    from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

    run_id, started = uuid.uuid4().hex, _utc()
    test_x, test_y = partitions["test"]
    config_hash = preflight["run_config_hash"]
    result = d4.run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=partitions["train"][0], train_labels=partitions["train"][1],
        test_features=test_x, test_labels=test_y,
        adapter_factory=make_adapter_factory(run_dir, design, partitions, preflight["partitions"]["test"]["digest"]),
        checkpoint_root=run_dir / "fits", replicates=design["replicates"], audit_seed=AUDIT_SEED,
        effect_threshold=design["effect_threshold"], bootstrap_replicates=BOOTSTRAP_REPLICATES,
        confidence_level=CONFIDENCE_LEVEL, alpha=design["alpha"], run_config_hash=config_hash,
        decision_rule=design["decision_rule"], chance_margin=design["chance_margin"],
        paired_endpoint_role=design["paired_endpoint_role"], chance_level=CHANCE_LEVEL,
    )
    preflight_sha = sha256_file(run_dir / "preflight.json")
    certificate = d4.generate_certificate(
        result, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
        generation_procedure="speck.make_train_data(n, 5), three independent calls, persisted at preflight",
        generation_parameters={"sample_sizes": design["sample_sizes"], "randomness_source": "os.urandom",
                               "input_difference": {"left_word": "0x0040", "right_word": "0x0000"}},
        audit_seed=AUDIT_SEED, confidence_level=CONFIDENCE_LEVEL, alpha=design["alpha"],
        provenance={"preregistration": preflight["preregistration"], "preflight_sha256": preflight_sha,
                    "run_config_hash": config_hash, "partitions": preflight["partitions"],
                    "test_label_counts": {"zeros": int(np.sum(np.asarray(test_y) == 0)),
                                          "ones": int(np.sum(np.asarray(test_y) == 1))},
                    "overlap_audit": preflight["overlap_audit"], "source_identity": identity,
                    "run_id": run_id},
    )
    d4.recompute_certificate_decision(certificate)
    certificate_bytes = json.dumps(certificate, indent=2, sort_keys=True, allow_nan=True).encode() + b"\n"
    manifest = {"schema_version": "d4-run-manifest-1", "run_id": run_id, "started_utc": started,
                "finished_utc": _utc(), "run_status": "COMPLETE", "preflight_sha256": preflight_sha,
                "preregistration_sha256": anchor, "source_identity": identity,
                "files": {"d4_certificate.json": sha256_bytes(certificate_bytes)}}
    write_directory_exclusive(result_dir, {
        "d4_certificate.json": certificate_bytes,
        "d4_run_manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
    })
    d4.print_report(result)
    outcome = certificate["decision"]["outcome"]
    print(f"D4 RUN COMPLETE. Primary outcome: {outcome}"
          f"{'' if outcome == 'SUPPORTED' else ' (NOT a scientific pass)'}  ->  {result_dir}")
    return 0


def verify_run(run_dir: Path) -> dict:
    """Independent post-run verification; read-only. Raises IntegrityError on any defect."""
    from audit.dataset.adapters import gohr_d4

    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "result" / "d4_run_manifest.json", "run manifest")
    if manifest.get("run_status") != "COMPLETE" or set(manifest.get("files", {})) != {"d4_certificate.json"}:
        raise IntegrityError("Run manifest is not a COMPLETE manifest of exactly the certificate.")
    certificate_path = run_dir / "result" / "d4_certificate.json"
    if not certificate_path.is_file() or sha256_file(certificate_path) != manifest["files"]["d4_certificate.json"]:
        raise IntegrityError("Certificate is missing or does not match the run manifest.")
    certificate = json.loads(certificate_path.read_text())
    preflight, partitions = load_preflight(run_dir)
    provenance = certificate["provenance"]
    preflight_sha = sha256_file(run_dir / "preflight.json")
    if provenance["preflight_sha256"] != preflight_sha or manifest["preflight_sha256"] != preflight_sha:
        raise IntegrityError("Certificate or manifest is bound to a different preflight record.")
    if provenance["preregistration"] != preflight["preregistration"]:
        raise IntegrityError("Certificate preregistration binding differs from preflight.")
    if provenance["source_identity"]["files"] != preflight["source_identity"]["files"]:
        raise IntegrityError("The run used sources that differ from those recorded at preflight.")
    config_hash = preflight["run_config_hash"]
    if provenance["run_config_hash"] != config_hash or certificate["settings"]["run_config_hash"] != config_hash:
        raise IntegrityError("Certificate run_config_hash differs from preflight.")
    design = preflight["design"]
    settings = certificate["settings"]
    for key in ("alpha", "decision_rule", "replicates", "audit_seed", "bootstrap_replicates", "confidence_level",
                "chance_level", "chance_margin", "paired_endpoint_role", "effect_threshold"):
        if settings.get(key) != design[key]:
            raise IntegrityError(f"Certificate setting {key} differs from the preregistered design.")
    if settings.get("test_examples") != design["sample_sizes"]["test"] or \
            settings.get("test_examples") != len(partitions["test"][1]):
        raise IntegrityError("Certificate test sample size differs from the preregistered design.")
    methodology = certificate.get("methodology", {})
    if methodology.get("primary_procedure") != design["primary_procedure"] or \
            methodology.get("supporting_procedure") != design["supporting_procedure"]:
        raise IntegrityError("Certificate endpoint procedure (sidedness, aggregation or level) differs from "
                             "the preregistered design.")
    source_digest = labelled_partition_digest(*partitions["train"])
    if settings["source_train_digest"] != source_digest:
        raise IntegrityError("Certificate training digest differs from the preflight training partition.")
    test_y = np.asarray(partitions["test"][1])
    if provenance.get("test_label_counts") != {"zeros": int(np.sum(test_y == 0)), "ones": int(np.sum(test_y == 1))}:
        raise IntegrityError("Certificate test class counts differ from the test partition.")
    recount_from_predictions(run_dir, certificate, test_y)
    decision = d4.recompute_certificate_decision(certificate)

    validation_digest = labelled_partition_digest(*partitions["validation"])
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
             for s in np.random.SeedSequence(design["audit_seed"]).spawn(design["replicates"])]
    for row, seed in zip(certificate["findings"]["replicates"], seeds):
        r = row["replicate"]
        if row["seed"] != seed:
            raise IntegrityError(f"Replicate {r} seed differs from the audit-seed plan.")
        record = row["perturbation_record"]
        if record.get("replicate") != r or record.get("source_train_digest") != source_digest:
            raise IntegrityError(f"Replicate {r} perturbation record is not bound to this replicate/data.")
        on_disk = read_json(run_dir / "fits" / "perturbations" / f"replicate_{r:02d}.perturbation.json",
                            f"replicate {r} perturbation record")
        if on_disk != record:
            raise IntegrityError(f"Replicate {r} perturbation sidecar differs from the certificate.")
        from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation
        shuffle = LabelShufflePerturbation()
        child = np.random.SeedSequence(design["audit_seed"]).spawn(design["replicates"])[r - 1]
        planned = shuffle.permutation_indices(len(partitions["train"][1]), np.random.default_rng(child))
        d4.load_verified_perturbation(run_dir / "fits" / "perturbations", replicate=r, seed=seed,
                                      perturbation=shuffle, source_train_digest=source_digest,
                                      source_label_digest=partition_digest({"labels": partitions["train"][1]}),
                                      audit_seed=design["audit_seed"], train_features=partitions["train"][0],
                                      train_labels=partitions["train"][1], expected_indices=planned)
        for condition, train_digest in (("baseline", source_digest), ("perturbed", record["perturbed_train_digest"])):
            selection = row[f"{condition}_selection"]
            identity = {key: selection[key] for key in gohr_d4.IDENTITY_FIELDS}
            expected = {"condition": condition, "replicate": r, "seed": seed, "config_hash": config_hash,
                        "train_digest": train_digest, "validation_digest": validation_digest,
                        "evaluation_digest": preflight["partitions"]["test"]["digest"]}
            for key, value in expected.items():
                if identity[key] != value:
                    raise IntegrityError(f"Replicate {r} {condition} selection has {key}={identity[key]!r}, "
                                         f"expected {value!r}.")
            directory, state_path = d4.checkpoint_paths(run_dir / "fits", r, condition)
            state = gohr_d4.load_committed_state(state_path, directory, identity=identity,
                                                 total_epochs=design["epochs"])
            recomputed = gohr_d4.select_checkpoint(directory, total_epochs=design["epochs"], identity=identity,
                                                   chain_head=state["record_chain_head"])
            recomputed["retries"] = state["retries"]
            if recomputed != selection or read_json(directory / "selected_checkpoint.json",
                                                    "selection record") != selection:
                raise IntegrityError(f"Replicate {r} {condition}: selection does not match the verified "
                                     "epoch history on disk.")
    return {"integrity": "VERIFIED", "outcome": decision["outcome"],
            "primary_claim_supported": decision["outcome"] == "SUPPORTED",
            "primary_outcome": decision["primary"]["outcome"],
            "supporting_degradation_established": decision["supporting"]["established"],
            "replicates": len(seeds), "run_id": manifest["run_id"]}


def recount_from_predictions(run_dir: Path, certificate: dict, test_labels) -> None:
    """Recompute every fit's counts from its persisted predictions and the test labels."""
    directory = run_dir / "fits" / "predictions"
    expected_files = {row[f"{c}_predictions"].get("file") for row in certificate["findings"]["replicates"]
                      for c in ("baseline", "perturbed")}
    present = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
    if present != expected_files:
        raise IntegrityError(f"Prediction files differ from the certificate: missing "
                             f"{sorted(expected_files - present)}, unexpected {sorted(present - expected_files)}.")
    for row in certificate["findings"]["replicates"]:
        r = row["replicate"]
        correct = {}
        for condition in ("baseline", "perturbed"):
            entry = row[f"{condition}_predictions"]
            path = d4.prediction_path(run_dir / "fits", r, condition)
            if entry.get("file") != path.name or not path.is_file() or sha256_file(path) != entry.get("sha256"):
                raise IntegrityError(f"Replicate {r} {condition} predictions are missing or do not match "
                                     "their recorded digest.")
            predictions = np.load(path)
            if predictions.dtype.str != entry.get("dtype") or list(predictions.shape) != entry.get("shape"):
                raise IntegrityError(f"Replicate {r} {condition} predictions have the wrong dtype or shape.")
            correct[condition] = d4.binary_predictions_to_correctness(predictions, test_labels)
        b, c, _ = d4.mcnemar_exact_pvalue(correct["baseline"], correct["perturbed"])
        recounted = {"test_examples": int(len(test_labels)), "baseline_correct": int(correct["baseline"].sum()),
                     "perturbed_correct": int(correct["perturbed"].sum()), "mcnemar_b": b, "mcnemar_c": c}
        for key, value in recounted.items():
            if row.get(key) != value:
                raise IntegrityError(f"Replicate {r}: certificate {key}={row.get(key)!r} but the persisted "
                                     f"predictions give {value!r}.")


def _load_selected_model(path: Path):
    from keras.models import load_model

    return load_model(path, compile=False)


def rerun_selected_models(run_dir: Path) -> list[dict]:
    """Reload every fit's selected checkpoint, re-predict the test partition, compare.

    Call only after verify_run has passed.  The checkpoint must match the
    SHA-256 in its selection record; its thresholded test predictions must
    equal those of the persisted prediction file exactly (raw scores may
    differ in the last bits across processes, so the maximum absolute
    difference is reported, not gated).  Read-only; needs TensorFlow.
    """
    run_dir = Path(run_dir)
    certificate = json.loads((run_dir / "result" / "d4_certificate.json").read_text())
    _, partitions = load_preflight(run_dir)
    test_x = partitions["test"][0]
    report = []
    for row in certificate["findings"]["replicates"]:
        r = row["replicate"]
        for condition in ("baseline", "perturbed"):
            selection = row[f"{condition}_selection"]
            directory, _ = d4.checkpoint_paths(run_dir / "fits", r, condition)
            checkpoint = directory / selection["checkpoint"]
            if not checkpoint.is_file() or sha256_file(checkpoint) != selection["checkpoint_sha256"]:
                raise IntegrityError(f"Replicate {r} {condition}: selected checkpoint is missing or does not "
                                     "match its selection record.")
            fresh = np.asarray(_load_selected_model(checkpoint).predict(test_x, verbose=0)).reshape(-1)
            stored = np.load(d4.prediction_path(run_dir / "fits", r, condition))
            disagreements = int(np.sum((fresh >= 0.5) != (stored >= 0.5))) if fresh.shape == stored.shape else -1
            if disagreements != 0:
                raise IntegrityError(f"Replicate {r} {condition}: the reloaded selected checkpoint (epoch "
                                     f"{selection['selected_epoch']}) does not reproduce the persisted predictions "
                                     f"({disagreements} label disagreements).")
            report.append({"replicate": r, "condition": condition, "selected_epoch": selection["selected_epoch"],
                           "checkpoint_sha256": selection["checkpoint_sha256"], "label_disagreements": 0,
                           "max_abs_score_difference": float(np.max(np.abs(fresh - stored))) if fresh.size else 0.0})
    return report


def stage_verify(args) -> int:
    if not (Path(args.run_dir) / "result").exists():
        print(json.dumps({"outcome": "NOT_COMPLETED", "primary_claim_supported": False}, indent=2))
        print(f"NO RESULT: {Path(args.run_dir) / 'result'} does not exist; the run is NOT_COMPLETED.")
        return 5
    summary = verify_run(Path(args.run_dir))
    if args.rerun_models:
        summary["reloaded_checkpoints_reproduce_predictions"] = rerun_selected_models(Path(args.run_dir))
    print(json.dumps(summary, indent=2))
    print("INTEGRITY CHECKS PASSED (post-run verification; read-only). "
          f"Primary outcome: {summary['outcome']}"
          f"{'' if summary['primary_claim_supported'] else ' (NOT a scientific pass)'}.")
    return 0


# ============================================================
# CLI
# ============================================================


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gohr D4 label-shuffle controlled perturbation audit.")
    sub = parser.add_subparsers(dest="stage", required=True)
    check = sub.add_parser("protocol-check", help="structural review of the preregistration (read-only)")
    check.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    check.add_argument("--require-approved", action="store_true")
    for name, text in (("preflight", "HUMAN-AUTHORISED: generate and audit partitions"),
                       ("run", "HUMAN-AUTHORISED PRODUCTION: train, evaluate, certify")):
        stage = sub.add_parser(name, help=text)
        stage.add_argument("--preregistration", type=Path, required=True)
        stage.add_argument("--preregistration-sha256", required=True)
        stage.add_argument("--run-dir", type=Path, required=True)
    verify = sub.add_parser("verify", help="independent post-run verification (read-only)")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--rerun-models", action="store_true",
                        help="also reload every selected checkpoint and re-predict the test partition (needs TF)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = {"protocol-check": stage_protocol_check, "preflight": stage_preflight,
              "run": stage_run, "verify": stage_verify}
    try:
        return stages[args.stage](args)
    except IntegrityError as exc:
        print(f"REFUSED ({args.stage}): {exc}", file=sys.stderr)
        if args.stage == "run":
            print("D4 OUTCOME: NOT_COMPLETED (no result written; not a scientific result).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
