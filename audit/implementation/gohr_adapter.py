"""
Gohr/Speck32/64-specific Implementation Integrity adapter.

ALL Gohr-specific knowledge for Pillar 1 lives here. The generic
framework (audit/common, audit/implementation/stages.py,
audit/implementation/equivalence.py) contains none of it.

REFERENCE PROTOCOL
------------------
The authoritative reference is the original Gohr source supplied with
this audit and stored read-only under audit/implementation/reference/:

    speck.py          sha256 59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce
    train_nets.py     sha256 b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669
    train_5_rounds.py sha256 63b8bec9779a6b9a95cda4abdbd763b2b8aef384df45a40109c926ff8864ff24

Those files are never modified. The declared baseline below is read
FROM them, not from documentation or recollection:

    train_5_rounds.py : train_speck_distinguisher(200, num_rounds=5, depth=10)
    train_nets.py     : bs = 5000; optimizer 'adam'; loss 'mse';
                        cyclic_lr(10, 0.002, 0.0001);
                        make_checkpoint monitors 'val_loss', save_best_only=True;
                        make_train_data(10**7) train / make_train_data(10**6) eval
    speck.py          : diff default (0x0040, 0), urandom-based generation

test_key_recovery.py concerns the separate 11/12-round key-recovery
endpoint and is deliberately NOT part of this 5-round distinguisher
audit's declared baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REFERENCE_FILE_HASHES = {
    "speck.py": "59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce",
    "train_nets.py": "b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669",
    "train_5_rounds.py": "63b8bec9779a6b9a95cda4abdbd763b2b8aef384df45a40109c926ff8864ff24",
}

# Declared reference protocol, transcribed from the reference source.
DECLARED_BASELINE: dict[str, Any] = {
    "cipher": "speck32_64",
    "rounds": 5,
    "differential": (0x0040, 0x0000),
    "depth": 10,
    "num_filters": 32,
    "kernel_size": 3,
    "word_size": 16,
    "num_blocks": 2,
    "epochs": 200,
    "batch_size": 5000,
    "optimizer": "adam",
    "loss": "mse",
    "lr_schedule": "cyclic(period=10, high=0.002, low=0.0001)",
    "checkpoint_monitor": "val_loss",
    "checkpoint_save_best_only": True,
    "train_samples": 10 ** 7,
    "validation_samples": 10 ** 6,
    "reported_best_validation_accuracy": 0.9291,
}

# Conv1D layer count implied by the reference architecture:
# one initial bit-sliced 1x1 conv, then 2 convs per residual block.
def implied_conv1d_count(depth: int) -> int:
    return 1 + 2 * depth


def depth_from_conv1d_count(n_conv1d: int) -> Optional[int]:
    """Inverse of implied_conv1d_count; None if the count is not of the expected form."""
    if n_conv1d < 1 or (n_conv1d - 1) % 2 != 0:
        return None
    return (n_conv1d - 1) // 2


@dataclass(frozen=True)
class CheckpointArchitectureProbe:
    """
    Result of reading an actual Keras .h5 checkpoint and deriving the
    architecture it REALLY contains - as opposed to what its filename
    or the surrounding documentation asserts.
    """
    path: str
    sha256: str
    n_conv1d_layers: int
    n_dense_layers: int
    n_batchnorm_layers: int
    derived_depth: Optional[int]
    # Residual `Add` merges counted from the embedded model config; None
    # when the artifact carries no readable config.
    n_add_layers: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "sha256": self.sha256,
            "n_conv1d_layers": self.n_conv1d_layers, "n_dense_layers": self.n_dense_layers,
            "n_batchnorm_layers": self.n_batchnorm_layers, "derived_depth": self.derived_depth,
            "n_add_layers": self.n_add_layers,
        }

    @property
    def is_consistent(self) -> bool:
        """Conv1D-derived depth agrees with the residual-merge count (when known)."""
        return self.derived_depth is not None and (
            self.n_add_layers is None or self.n_add_layers == self.derived_depth)


def _config_layer_counts(model_config: dict) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(layers):
        for layer in layers:
            counts[layer.get("class_name")] = counts.get(layer.get("class_name"), 0) + 1
            inner = layer.get("config", {}).get("layers")
            if isinstance(inner, list):
                walk(inner)

    walk(model_config.get("config", {}).get("layers", []))
    return counts


def probe_checkpoint_architecture(path: str | Path) -> CheckpointArchitectureProbe:
    """
    Read a Keras HDF5 checkpoint and count its layer groups to derive
    the realized residual depth. Uses h5py only - does not require
    TensorFlow, and does not execute or deserialize the model.

    This is the II-3 Controlled Verification primitive that established
    the finding recorded in audit/implementation/findings.py: the
    artifact's realized architecture is read from the artifact itself,
    never inferred from its filename.

    Keras 3 `.keras` files (ZIP containers; also the II-2 model stored
    as `.zip`) are read from their `config.json`. For HDF5 files the
    Conv1D/Dense/BN counts come from the weight groups as before, and the
    residual-merge count from the embedded `model_config` attribute.
    """
    import json
    import re
    import zipfile

    import h5py

    from audit.common.provenance import sha256_file

    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            counts = _config_layer_counts(json.loads(archive.read("config.json")))
        n_conv = counts.get("Conv1D", 0)
        return CheckpointArchitectureProbe(
            path=str(path), sha256=sha256_file(path),
            n_conv1d_layers=n_conv, n_dense_layers=counts.get("Dense", 0),
            n_batchnorm_layers=counts.get("BatchNormalization", 0),
            derived_depth=depth_from_conv1d_count(n_conv), n_add_layers=counts.get("Add", 0),
        )

    n_add: Optional[int] = None
    with h5py.File(path, "r") as handle:
        if "model_weights" in handle:
            keys = list(handle["model_weights"].keys())
        else:
            keys = list(handle.keys())
        raw_config = handle.attrs.get("model_config")
        if raw_config is not None:
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            n_add = _config_layer_counts(json.loads(raw_config)).get("Add", 0)

    conv = [k for k in keys if re.match(r"^conv1d(_\d+)?$", k)]
    dense = [k for k in keys if re.match(r"^dense(_\d+)?$", k)]
    bn = [k for k in keys if re.match(r"^batch_normalization(_\d+)?$", k)]

    return CheckpointArchitectureProbe(
        path=str(path), sha256=sha256_file(path),
        n_conv1d_layers=len(conv), n_dense_layers=len(dense), n_batchnorm_layers=len(bn),
        derived_depth=depth_from_conv1d_count(len(conv)), n_add_layers=n_add,
    )


# ---------------------------------------------------------------------
# Evidence locations for the five-stage certificate (repository-relative).
# ---------------------------------------------------------------------

# II-2: the independent implementation, its evidence, and its model.
II2_EVIDENCE = {
    "m1_deterministic_verification": "audit/implementation/evidence/ii2/m1_deterministic_verification.json",
    "m1_independent_training_report": "audit/implementation/evidence/ii2/m1_independent_training_report.json",
    "m1_independent_model": "audit/implementation/evidence/ii2/m1_independent_best.zip",
    "m7_theory_verification": "audit/implementation/evidence/m7/m7_theory_verification.json",
}
II2_SOURCES = {
    "independent_speck": "audit/implementation/ii2_independent/speck_independent.py",
    "independent_dataset": "audit/implementation/ii2_independent/dataset_independent.py",
    "independent_model": "audit/implementation/ii2_independent/model_independent.py",
    "independent_train": "audit/implementation/ii2_independent/train_independent.py",
    "independent_theory": "audit/implementation/ii2_independent/theory_independent.py",
    "verify_m1": "audit/implementation/ii2_independent/verify_m1.py",
    "verify_m7": "audit/implementation/ii2_independent/verify_m7.py",
}

# Reference-implementation depth-10 artifact: produced by the upstream
# code (with TF 2.20 / Keras 3 compatibility edits) at commit 2b54425,
# docs/reproduction_report.md. The same bytes are the CE2-CE4 checkpoint.
REFERENCE_DEPTH10_CHECKPOINT = "audit/cryptography/Archive/best5depth10.h5"

# The historical depth-5 artifacts on which II-FINDING-DEPTH-V1 rests.
HISTORICAL_DEPTH5_CHECKPOINTS = {
    "best5depth10 (10).h5": "audit/cryptography/evidence/ce1/best5depth10 (10).h5",
    "signal_destroyed.h5": "audit/cryptography/evidence/ce1/signal_destroyed.h5",
}

# The frozen II-4-GOHR-DEPTH-V1 artifacts, as bound next to certificate V4/V5.
DEPTH_EXPERIMENT_ID = "II-4-GOHR-DEPTH-V1"
DEPTH_EXPERIMENT_BOUND_DIR = "audit/implementation/evidence/ii4_bound/7893611dffa86656"
DEPTH_EXPERIMENT_ORIGINAL_DIR = "audit/implementation/evidence/ii4"

# Historical downstream references (as written in DEPTH_FINDING_V1) ->
# the historical certificate files, and the current certificate that
# re-executed each one at the declared depth.
HISTORICAL_CE_CERTIFICATES = {
    "evidence/ce1/ce1_certificate.json": ["audit/cryptography/evidence/ce1/ce1_certificate.json"],
    "evidence/ce2/ce2_certificate_{1..5}.json": [
        f"audit/cryptography/evidence/ce2/{run}_run/ce2_certificate_{i}.json"
        for i, run in enumerate(("1st", "2nd", "3rd", "4th", "5th"), start=1)
    ],
    "evidence/ce3/ce3_certificate.json": ["audit/cryptography/evidence/ce3/ce3_certificate.json"],
    "evidence/ce4/ce4_certificate.json": ["audit/cryptography/evidence/ce4/ce4_certificate.json"],
}
CURRENT_CE_CERTIFICATES = {
    "evidence/ce1/ce1_certificate.json":
        "audit/cryptography/evidence_current/ce1/production_20260930/certificatece1.json",
    "evidence/ce2/ce2_certificate_{1..5}.json":
        "audit/cryptography/evidence_current/ce2/production_20261003/certificatece2.json",
    "evidence/ce3/ce3_certificate.json":
        "audit/cryptography/evidence_current/ce3/certificate.json",
    "evidence/ce4/ce4_certificate.json":
        "audit/cryptography/evidence_current/ce4/production_20261004/certificatece4.json",
}


def _ce_recorded_checkpoints(cert: dict, cert_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    """
    The checkpoints a current CE certificate says it used, with what it
    records about them. CE2-CE4 record one `reference_model`; CE1 trains
    its own models and records a `model_sha256` per arm per block, stored
    at blocks/<block>/<arm>/FINAL_EPOCH.keras next to the certificate.
    """
    if "reference_model" in cert:
        ref = cert["reference_model"]
        props = ref.get("properties", {})
        return [{
            "label": "reference_model", "path": ref["path"], "recorded_sha256": ref["sha256"],
            "recorded_depth": ref.get("realized_depth"),
            "recorded_conv1d": props.get("conv1d_layers"),
            "recorded_add": props.get("residual_merges"),
            "counted": True,
        }]
    depth = cert.get("reference_configuration", {}).get("depth")
    out = []
    for block in cert["results"]["blocks"]:
        for arm, info in block["arms"].items():
            rel = cert_path.parent.relative_to(repo_root) / "blocks" / block["block_id"] / arm / "FINAL_EPOCH.keras"
            out.append({
                "label": f"{block['block_id']}/{arm}", "path": str(rel),
                "recorded_sha256": info["model_sha256"], "recorded_depth": depth,
                "recorded_conv1d": None, "recorded_add": None,
                "counted": block.get("status") != "NOT_COUNTED",
            })
    return out


def verify_ce_certificate(repo_root: Path, cert_rel: str) -> dict[str, Any]:
    """
    Verify one current CE certificate against its actual checkpoints:
    each recorded hash must equal the file's real hash, and the realized
    architecture (read from the file) must match both what the
    certificate records and the declared depth. `verified` is True only
    if every recorded checkpoint passes.
    """
    import json

    from audit.common.provenance import sha256_file

    cert_path = repo_root / cert_rel
    cert = json.loads(cert_path.read_text())
    checks, problems = [], []
    for rec in _ce_recorded_checkpoints(cert, cert_path, repo_root):
        path = repo_root / rec["path"]
        if not path.is_file():
            problems.append(f"{rec['label']}: checkpoint {rec['path']} not found")
            continue
        probe = probe_checkpoint_architecture(path)
        mismatch = []
        if probe.sha256 != rec["recorded_sha256"]:
            mismatch.append(f"sha256 {probe.sha256} != recorded {rec['recorded_sha256']}")
        if not probe.is_consistent:
            mismatch.append("Conv1D-derived depth disagrees with residual-merge count")
        for key, observed in (("recorded_depth", probe.derived_depth),
                              ("recorded_conv1d", probe.n_conv1d_layers),
                              ("recorded_add", probe.n_add_layers)):
            if rec[key] is not None and rec[key] != observed:
                mismatch.append(f"{key}={rec[key]} but artifact gives {observed}")
        if probe.derived_depth != DECLARED_BASELINE["depth"]:
            mismatch.append(f"realized depth {probe.derived_depth} != declared {DECLARED_BASELINE['depth']}")
        problems.extend(f"{rec['label']}: {m}" for m in mismatch)
        checks.append({
            "label": rec["label"], "path": rec["path"], "sha256": probe.sha256,
            "counted": rec["counted"], "n_conv1d_layers": probe.n_conv1d_layers,
            "n_add_layers": probe.n_add_layers, "realized_depth": probe.derived_depth,
            "ok": not mismatch,
        })
    return {
        "certificate": cert_rel, "certificate_sha256": sha256_file(cert_path),
        "experiment_id": cert.get("experiment_id"), "run_id": cert.get("run_id"),
        # CE1 records estimates and an exact test, not a categorical decision.
        "decision": cert.get("results", {}).get("decision"),
        "primary_test_p_value": cert.get("results", {}).get("paired_difference", {})
                                    .get("primary_test", {}).get("p_value"),
        "n_checkpoints": len(checks),
        "distinct_checkpoint_hashes": sorted({c["sha256"] for c in checks}),
        "realized_depths": sorted({c["realized_depth"] for c in checks if c["realized_depth"] is not None}),
        "checkpoints": checks, "problems": problems,
        "verified": bool(checks) and not problems,
    }


# ---------------------------------------------------------------------
# Outstanding II-4/II-5 work. These are PROPOSALS, not preregistrations:
# nothing here is frozen, and no run may cite them as its design.
# ---------------------------------------------------------------------

FIXED_CHECKPOINT_COMPARISON_PROPOSAL: dict[str, Any] = {
    "status": "PROPOSED_NOT_PREREGISTERED",
    "proposed_experiment_id": "II-4-GOHR-IMPL-FIXED-CKPT-V1",
    "question": (
        "Do the two existing depth-10 checkpoints - one trained by the reference implementation, "
        "one by the independent implementation - perform equivalently on identical data?"),
    "artifacts": {"reference": REFERENCE_DEPTH10_CHECKPOINT,
                  "independent": II2_EVIDENCE["m1_independent_model"]},
    "feasibility": (
        "Feasible without retraining. Both artifacts are present, hash-bound and depth 10 "
        "(21 Conv1D / 10 Add, 102,497 parameters). Each architecture rebuilds from its own source "
        "and loads its weights in the local Keras 3.6 environment (non-evidentiary 20,000-sample "
        "smoke check only, not recorded as evidence). Full-model deserialization needs Keras >= "
        "3.13 (the training environment)."),
    "common_data": "One fresh 10^6-example set from reference speck.make_train_data(10**6, 5), "
                   "persisted and hashed before either model is evaluated.",
    "experimental_unit": "test example (inference is about these two fixed checkpoints only)",
    "primary_endpoint": "paired accuracy difference, reference minus independent",
    "statistical_method": "exact McNemar test on discordant pairs; TOST equivalence on the paired "
                          "per-example difference with the margin preregistered below",
    "margin": "0.01 proposed, for parity with II-4-GOHR-DEPTH-V1; the owner must confirm it",
    "secondary": ["decision agreement rate", "Pearson correlation of scores", "MSE difference"],
    "resource_estimate": "minutes of CPU inference; no training",
    "limitations": [
        "One training run per implementation: says nothing about variation across training runs "
        "and is NOT evidence of training reproducibility.",
        "The reference checkpoint was produced by the upstream code with TF 2.20 / Keras 3 "
        "compatibility edits (commit 2b54425); it is not a byte-identical run of the 2019 code.",
        "The II-2 training run recorded no seed.",
    ],
}

FORMAL_COMPARISON_PROPOSAL: dict[str, Any] = {
    "status": "PROPOSED_NOT_PREREGISTERED",
    "proposed_experiment_id": "II-4-GOHR-IMPL-V1",
    "question": (
        "Under identical experimental protocols, do the reference and independent implementations "
        "reach the same sealed-test accuracy when training is replicated?"),
    "arms": {
        "reference": "audit/implementation/reference/train_nets.py make_resnet(depth=10, "
                     "reg_param=1e-5) with its fit recipe, driven by a wrapper that supplies the "
                     "block data and a recorded seed (the reference files are not modified)",
        "independent": "audit/implementation/ii2_independent model_independent + train_independent",
    },
    "protocol": "declared baseline: 200 epochs, batch 5000, Adam, MSE, cyclic_lr(10, 0.002, "
                "0.0001), L2 1e-5, 10^7 train / 10^6 validation",
    "experimental_unit": "dataset block: one train/validation set generated once per block and "
                         "shared by both arms; each arm trained once per block from its own "
                         "recorded initialization seed",
    "primary_endpoint": "accuracy on one global sealed 10^6-example test set at the terminal "
                        "epoch (as in II4-SPEC-V1)",
    "secondary_endpoints": ["accuracy of the minimum-validation-loss checkpoint (the reference "
                            "selection rule)", "decision agreement rate on the sealed set"],
    "statistical_method": "the frozen II4-SPEC-V1 analysis: mean signed paired difference, "
                          "signed 95% t-CI materiality, TOST equivalence (alpha 0.05, margin "
                          "0.01), exact sign-flip test as secondary",
    "replicates": {
        "minimum_defensible": {"n_blocks": 7, "min_valid_blocks": 6, "rationale": (
            "6 valid blocks is the smallest K at which the exact sign-flip secondary can reach "
            "two-sided p < 0.05 (2/2^6 = 0.031); TOST power at margin 0.01 is ~1 at K = 6 even "
            "if the between-block SD of differences is 10x the 3.2e-4 seen in "
            "II-4-GOHR-DEPTH-V1; one block of failure tolerance.")},
        "parity_option": {"n_blocks": 10, "min_valid_blocks": 9,
                          "rationale": "matches II-4-GOHR-DEPTH-V1"},
    },
    "resource_estimate": (
        "~9 GPU-hours per block on 2x Tesla T4 (measured: depth-10 arms in II-4-GOHR-DEPTH-V1 took "
        "3.7-4.6 h; the II-2 run took 4.5 h). K=7: ~63 h; K=10: ~90 h."),
    "limitations": [
        "Both arms follow the public-code architecture (BatchNormalization after the second dense "
        "layer); the paper-prose variant is not tested.",
        "Data generation is shared, so it is not a compared component; its equivalence rests on "
        "the II-2 deterministic checks.",
        "Inference is conditional on the single sealed test set.",
    ],
}
