"""
Deterministic recovery of legacy block0 (production_20260929).

Legacy block0 trained BOTH arms to the terminal epoch and saved each
terminal model (`block0_{arm}_FINAL_EPOCH.keras`), but the legacy path
persisted no evaluation result. The block is therefore TRAINING_COMPLETE,
not EVALUATED. Recovery:

    verify the two terminal models + the sealed set (read-only)
    byte-copy each terminal model into the new run (originals untouched)
    seed the arm state as TRAINING_COMPLETE, provenance = legacy_recovery
    -> the controller's ordinary EVALUATE path scores each terminal model on
       the sealed set and binds the result to model hash + sealed hash.

No retraining. No validation-based reselection (the *_bestval_DEBUG_ONLY
artifacts are never read). The accuracy is NOT computed by this module.

EVIDENCE USED (all outcome-independent - never the models' accuracy):
  * sha256 of each terminal artifact;
  * realized architecture read from the file (depth, input width, L2);
  * compiled optimizer / loss read from the file;
  * optimizer iteration counter == terminal_epoch x steps_per_epoch, i.e.
    the model really took every step of all 200 epochs;
  * arm binding: the legacy code wrote f"{block}_{arm}_FINAL_EPOCH.keras";
    independently, Keras names models in build order and the legacy driver
    built the baseline model before the destroyed model within a block, so
    the baseline file must carry the EARLIER functional index.

WHAT CANNOT BE VERIFIED and is recorded as such: the training data (never
persisted - os.urandom), the per-epoch LR trajectory, and the seed actually
used (the legacy code path used seed_manifest(), identical to this run's).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from audit.cryptography.experiments.ce1 import controller as C
from audit.cryptography.experiments.ce1 import resume as R

LEGACY_TAG = "legacy_recovery:production_20260929"
ARMS = ("baseline", "destroyed")


class LegacyRecoveryError(RuntimeError):
    pass


def _functional_index(name: str) -> int:
    m = re.fullmatch(r"functional(?:_(\d+))?", name or "")
    if not m:
        raise LegacyRecoveryError(f"unexpected model name {name!r}")
    return int(m.group(1) or 0)


def _inspect_file(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        cfg = json.loads(z.read("config.json"))
        meta = json.loads(z.read("metadata.json")) if "metadata.json" in z.namelist() else {}
    layers = cfg["config"]["layers"]
    l2s = sorted({float(l["config"]["kernel_regularizer"]["config"]["l2"])
                  for l in layers
                  if l["class_name"] in ("Conv1D", "Dense")
                  and l["config"].get("kernel_regularizer")})
    comp = cfg.get("compile_config") or {}
    return {
        "model_name": cfg["config"].get("name"),
        "n_add": sum(1 for l in layers if l["class_name"] == "Add"),
        "n_conv1d": sum(1 for l in layers if l["class_name"] == "Conv1D"),
        "input_shape": layers[0]["config"].get("batch_shape"),
        "l2_values": l2s,
        "optimizer": (comp.get("optimizer") or {}).get("class_name"),
        "loss": comp.get("loss"),
        "final_learning_rate": (comp.get("optimizer") or {}).get("config", {}).get(
            "learning_rate"),
        "keras_version_saved": meta.get("keras_version"),
    }


def verify_legacy_block0(legacy_run_dir, config: "C.CE1RunConfig", *,
                         load_models: bool = True, block_id: str = "block0") -> dict:
    """
    READ-ONLY verification. Computes no accuracy. Returns a report whose
    `verified` flag gates recovery.
    """
    legacy = Path(legacy_run_dir)
    ck = legacy / "checkpoints"
    checks, problems = [], []

    from audit.cryptography.sealed_dataset import _content_hash, data_path, manifest_path
    sm = json.loads(manifest_path(legacy / "sealed").read_text())
    with np.load(data_path(legacy / "sealed")) as z:
        sealed_hash = _content_hash(z["X"], z["Y"])
    if sealed_hash != sm["sha256"]:
        problems.append("legacy sealed set does not match its own manifest")
    if config.expected_sealed_sha256 and sealed_hash != config.expected_sealed_sha256:
        problems.append(f"legacy sealed set {sealed_hash} != pinned "
                        f"{config.expected_sealed_sha256}")
    checks.append({"check": "sealed_set_content_hash", "value": sealed_hash})

    want_iters = config.terminal_epoch * config.steps_per_epoch()
    arms = {}
    for arm in ARMS:
        final = ck / f"{block_id}_{arm}_FINAL_EPOCH.keras"
        if not final.exists():
            problems.append(f"{arm}: terminal artifact missing: {final}")
            continue
        info = _inspect_file(final)
        info["path"] = str(final)
        info["sha256"] = R.sha256_file(final)
        if info["n_add"] != config.depth or info["n_conv1d"] != 1 + 2 * config.depth:
            problems.append(f"{arm}: realized depth {info['n_add']} != {config.depth}")
        if info["input_shape"] != [None, 64]:
            problems.append(f"{arm}: input shape {info['input_shape']}")
        if not info["l2_values"] or any(abs(v - config.l2_reg) > 1e-9
                                        for v in info["l2_values"]):
            problems.append(f"{arm}: L2 {info['l2_values']} != {config.l2_reg}")
        if (info["optimizer"] or "").lower() != config.optimizer.lower():
            problems.append(f"{arm}: optimizer {info['optimizer']}")
        if info["loss"] != config.loss:
            problems.append(f"{arm}: loss {info['loss']}")
        info["optimizer_iterations_from_file"] = C.keras_file_optimizer_iterations(final)
        if info["optimizer_iterations_from_file"] != want_iters:
            problems.append(f"{arm}: file optimizer counter "
                            f"{info['optimizer_iterations_from_file']} != {want_iters}")
        if load_models:
            import keras
            model = keras.models.load_model(final)
            info["optimizer_iterations"] = int(model.optimizer.iterations.numpy())
            if info["optimizer_iterations"] != want_iters:
                problems.append(f"{arm}: optimizer iterations {info['optimizer_iterations']} "
                                f"!= {config.terminal_epoch} epochs x "
                                f"{config.steps_per_epoch()} steps = {want_iters}")
            out = model.predict(np.zeros((2, 64), dtype=np.uint8), verbose=0)
            info["forward_pass_shape"] = list(out.shape)      # executability only
        arms[arm] = info
    if len(arms) == 2:
        bi = _functional_index(arms["baseline"]["model_name"])
        di = _functional_index(arms["destroyed"]["model_name"])
        ok = bi < di
        checks.append({"check": "arm_binding_by_build_order",
                       "baseline_model_name": arms["baseline"]["model_name"],
                       "destroyed_model_name": arms["destroyed"]["model_name"],
                       "consistent": ok})
        if not ok:
            problems.append("build-order evidence contradicts the filename arm binding")
        if arms["baseline"]["sha256"] == arms["destroyed"]["sha256"]:
            problems.append("the two arms are the same file")
    return {
        "legacy_run_dir": str(legacy), "block_id": block_id, "arms": arms,
        "checks": checks, "problems": problems, "verified": not problems,
        "accuracy_computed": False,
        "unverifiable": ["training data (never persisted; os.urandom)",
                         "per-epoch learning-rate trajectory",
                         "seed actually used at training time"],
        "excluded_artifacts": ["*_bestval_DEBUG_ONLY.keras (never read: no validation-"
                               "based reselection)"],
    }


def recover_legacy_block0(legacy_run_dir, run_dir, config: "C.CE1RunConfig", *,
                          block_id: str = "block0") -> dict:
    """
    Seed block0 of `run_dir` from the verified legacy terminal models.
    Refuses if block0 already has ANY state, data or result in the new run
    (recovery must be the first and only thing that happens to block0).
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / C.RUN_MANIFEST).read_text())
    if manifest.get("legacy_block0_decision") != "RECOVER":
        raise LegacyRecoveryError("run manifest does not record legacy_block0_decision=RECOVER")
    bdir = run_dir / "blocks" / block_id
    if bdir.exists() and any(bdir.iterdir()):
        raise LegacyRecoveryError(f"{bdir} is not empty; recovery would mix artifacts")
    sealed = C.ensure_sealed(run_dir, config)
    report = verify_legacy_block0(legacy_run_dir, config, block_id=block_id)
    if not report["verified"]:
        raise LegacyRecoveryError("legacy verification failed: " + "; ".join(report["problems"]))
    seeds = C.seeds_for(config)["seeds"][block_id]
    ledger = C.ledger_for(run_dir)
    states = {}
    for arm in ARMS:
        src = Path(report["arms"][arm]["path"])
        adir = R.arm_dir(run_dir, block_id, arm)
        adir.mkdir(parents=True, exist_ok=True)
        dst = adir / "FINAL_EPOCH.keras"
        fd, tmp = tempfile.mkstemp(dir=str(adir), suffix=".partial.keras")
        os.close(fd)
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        sha = R.sha256_file(dst)
        if sha != report["arms"][arm]["sha256"] or R.sha256_file(src) != sha:
            raise LegacyRecoveryError(f"{arm}: copy is not byte-identical to the legacy file")
        rel = R.rel_to_run(run_dir, dst)
        st = R.ArmState(
            experiment_id=C.EXPERIMENT_ID, block_id=block_id, arm=arm,
            status=R.TRAINING_COMPLETE, last_completed_epoch=config.terminal_epoch,
            terminal_epoch=config.terminal_epoch, model_seed=seeds[arm],
            assignment_seed=config.assignment_seed, assignment_flip=seeds["assignment_flip"],
            permutation_seed=seeds["permutation"], config_fingerprint=config.fingerprint(),
            design_version=config.design_version, sealed_evaluation_sha256=sealed["sha256"],
            model_artifact=rel, model_sha256=sha, optimizer_state_included=True,
            terminal_model_artifact=rel, terminal_model_sha256=sha, provenance=LEGACY_TAG,
            terminal_optimizer_iterations=report["arms"][arm]["optimizer_iterations_from_file"])
        R.atomic_write_json(adir / "legacy_recovery.json", {
            "status": "LEGACY_RECOVERED - pending evaluation and final validity resolution",
            "source_path": str(src), "source_sha256": sha,
            "sealed_evaluation_sha256": sealed["sha256"],
            "legacy_sealed_manifest_sha256": report["checks"][0]["value"],
            "verification": report["arms"][arm], "checks": report["checks"],
            "unverifiable": report["unverifiable"],
            "rule": "byte copy of the legacy FINAL_EPOCH artifact; original untouched; "
                    "no retraining; no validation-based reselection"})
        R.save_arm_state(run_dir, st)
        ledger.record("LEGACY_RECOVERY", block_id=block_id, arm=arm,
                      config_hash=config.fingerprint(), source=str(src), model_sha256=sha,
                      optimizer_iterations=report["arms"][arm].get("optimizer_iterations"))
        states[arm] = st
    return {"block_id": block_id, "status": R.TRAINING_COMPLETE, "report": report}
