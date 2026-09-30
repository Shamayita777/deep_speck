"""
Evidence-based classification of LEGACY CE1 production artifacts.

These are artifacts produced BEFORE the resume wiring existed. They are a
different category from future resumable artifacts and must not be
blurred with them.

Classification is from ARTIFACT CONTENT ONLY. Directory existence, log
files, filenames and timestamps are explicitly NOT evidence of completion.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPLETE = "COMPLETE"        # terminal model + evaluation + metadata, all verified
PARTIAL = "PARTIAL"          # real work done, but not through the terminal epoch
INVALID = "INVALID"          # artifacts present but internally inconsistent
UNUSABLE = "UNUSABLE"        # nothing usable found

ARMS = ("baseline", "destroyed")



#: Why no legacy arm can be RESUMED. Keras 3 stores optimizer variables inside
#: model.weights.h5, so the legacy terminal artifacts DO carry optimizer state
#: (see `_keras_depth`); what is missing is the data and per-epoch record.
LEGACY_NON_RESUMABLE_REASON = (
    "optimizer state may exist inside the saved Keras artifact, but the legacy run did "
    "not preserve the original training/validation arrays or per-epoch state, so exact "
    "continuation/reconstruction of the original training run is not provable")

def _reasons_for_arm(arm_dir: Path, *, terminal_epoch: int, expected: dict) -> tuple:
    if not arm_dir.exists():
        return UNUSABLE, ["no arm directory"]
    meta_path = arm_dir / "metadata.json"
    if not meta_path.exists():
        return UNUSABLE, ["no metadata.json: completion cannot be established from "
                          "filenames, logs or timestamps"]
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:                       # noqa: BLE001
        return INVALID, [f"metadata.json unreadable: {exc}"]

    problems = []
    for key in ("block_id", "arm", "model_seed", "assignment_flip",
                "sealed_evaluation_sha256", "design_version"):
        if key not in meta:
            problems.append(f"metadata missing required field {key!r}")
    for key, want in expected.items():
        if key in meta and want is not None and meta[key] != want:
            problems.append(f"{key}: artifact {meta[key]!r} != expected {want!r}")

    epoch = meta.get("epochs_completed")
    model_rel = meta.get("model_artifact")
    model_ok = False
    if model_rel:
        mp = Path(model_rel)
        if not mp.is_absolute():
            mp = arm_dir / model_rel
        if not mp.exists():
            problems.append(f"model artifact missing: {mp}")
        elif mp.stat().st_size == 0:
            problems.append(f"model artifact is empty: {mp}")
        else:
            recorded = meta.get("model_sha256")
            if recorded:
                import hashlib
                actual = hashlib.sha256(mp.read_bytes()).hexdigest()
                if actual != recorded:
                    problems.append("model artifact hash mismatch (corrupted)")
                else:
                    model_ok = True
            else:
                problems.append("no model_sha256 recorded; integrity unverifiable")
    else:
        problems.append("no model artifact recorded")

    has_eval = meta.get("evaluation_accuracy") is not None

    if problems:
        return INVALID, problems
    if epoch is None:
        return INVALID, ["no epochs_completed recorded"]
    if epoch >= terminal_epoch and model_ok and has_eval:
        return COMPLETE, []
    reasons = []
    if epoch < terminal_epoch:
        reasons.append(f"reached epoch {epoch} of {terminal_epoch}")
    if not has_eval:
        reasons.append("no evaluation result recorded")
    reasons.append(LEGACY_NON_RESUMABLE_REASON + "; this arm must be re-run")
    return PARTIAL, reasons


def validate_legacy_run(run_dir, *, n_blocks: int, terminal_epoch: int,
                        expected: dict | None = None) -> dict:
    """Classify every block/arm of a legacy run and say what may be reused."""
    run_dir = Path(run_dir)
    expected = expected or {}
    blocks = {}
    for i in range(n_blocks):
        bid = f"block_{i:02d}"
        bdir = run_dir / "blocks" / bid
        arms = {}
        for arm in ARMS:
            status, reasons = _reasons_for_arm(bdir / arm, terminal_epoch=terminal_epoch,
                                               expected=expected)
            arms[arm] = {"status": status, "reasons": reasons}
        statuses = {a["status"] for a in arms.values()}
        if statuses == {COMPLETE}:
            block_status, action = COMPLETE, "SKIP (reuse as completed CE1 block)"
        elif INVALID in statuses:
            block_status, action = INVALID, "RERUN block"
        elif statuses == {UNUSABLE}:
            block_status, action = UNUSABLE, "RUN block (nothing usable)"
        else:
            block_status, action = PARTIAL, "RERUN incomplete arm(s) from scratch"
        blocks[bid] = {"block_id": bid, "block_status": block_status,
                       "action": action, "arms": arms}
    complete = [b for b, e in blocks.items() if e["block_status"] == COMPLETE]
    return {
        "run_dir": str(run_dir), "n_blocks": n_blocks, "terminal_epoch": terminal_epoch,
        "blocks": blocks,
        "complete_blocks": complete,
        "n_complete": len(complete),
        "reusable": complete,
        "must_rerun": [b for b, e in blocks.items() if e["block_status"] != COMPLETE],
        "category": "legacy_pre_resume_artifacts",
        "exact_resume_supported": False,
        "exact_resume_reason": LEGACY_NON_RESUMABLE_REASON,
    }


# =====================================================================
# ACTUAL legacy layout: production_YYYYMMDD/checkpoints/blockN_arm_*.keras
# =====================================================================

LEGACY_ARMS = ("baseline", "destroyed")


def _keras_depth(path):
    """Read realized depth from the saved architecture - never the filename."""
    import json as _json
    import zipfile
    with zipfile.ZipFile(path) as z:
        cfg = _json.loads(z.read("config.json"))
    layers = cfg["config"]["layers"]
    adds = sum(1 for l in layers if l["class_name"] == "Add")
    convs = sum(1 for l in layers if l["class_name"] == "Conv1D")
    # Keras 3 stores optimizer variables INSIDE model.weights.h5 (group
    # "optimizer"), not as a separate zip member; the earlier namelist check
    # therefore always reported False for Keras-3 files.
    has_opt = False
    with zipfile.ZipFile(path) as z:
        if "model.weights.h5" in z.namelist():
            import io
            import h5py
            with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as h:
                has_opt = "optimizer" in h and len(h["optimizer"].get("vars", {})) > 0
        else:
            has_opt = any("optimizer" in n for n in z.namelist())
    return adds, convs, has_opt


def validate_legacy_checkpoint_run(run_dir, *, n_blocks: int, expected_depth: int = 10,
                                   sealed_sha256: str | None = None) -> dict:
    """
    Classify the real legacy run from ARTIFACT CONTENT.

    Evidence rules:
      * a *_FINAL_EPOCH.keras artifact is written only after training
        returns, so its presence evidences the terminal epoch;
      * a *_bestval_DEBUG_ONLY.keras artifact alone evidences only that
        training started - it is NOT terminal evidence;
      * filenames and timestamps are never proof: every artifact's
        architecture is read from the file;
      * no evaluation result is persisted by the legacy path, so an arm
        that trained to the terminal epoch is TRAINING_COMPLETE, not
        EVALUATED, and the block is not yet a valid CE1 block.
    """
    import hashlib
    run_dir = Path(run_dir)
    ck = run_dir / "checkpoints"
    sealed_manifest = run_dir / "sealed" / "sealed_evaluation_manifest.json"

    sealed = {"present": sealed_manifest.exists()}
    if sealed["present"]:
        sm = json.loads(sealed_manifest.read_text())
        sealed.update({"recorded_sha256": sm.get("sha256"), "rounds": sm.get("rounds"),
                       "n_samples": sm.get("n_samples")})
        if sealed_sha256:
            sealed["matches_expected"] = sm.get("sha256") == sealed_sha256

    blocks = {}
    for i in range(n_blocks):
        arms = {}
        for arm in LEGACY_ARMS:
            final = ck / f"block{i}_{arm}_FINAL_EPOCH.keras"
            best = ck / f"block{i}_{arm}_bestval_DEBUG_ONLY.keras"
            if final.exists():
                try:
                    adds, convs, has_opt = _keras_depth(final)
                except Exception as exc:                      # noqa: BLE001
                    arms[arm] = {"status": INVALID,
                                 "reasons": [f"FINAL_EPOCH artifact unreadable: {exc}"]}
                    continue
                problems = []
                if adds != expected_depth or convs != 1 + 2 * expected_depth:
                    problems.append(f"architecture is depth {adds} (conv {convs}), "
                                    f"expected {expected_depth}")
                if problems:
                    arms[arm] = {"status": INVALID, "reasons": problems}
                else:
                    arms[arm] = {
                        "status": "TRAINING_COMPLETE",
                        "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                        "artifact": str(final), "realized_depth": adds,
                        "optimizer_state": has_opt,
                        "reasons": ["terminal training evidenced by the FINAL_EPOCH "
                                    "artifact; no persisted evaluation result, so this arm "
                                    "is not yet EVALUATED"],
                    }
            elif best.exists():
                arms[arm] = {"status": PARTIAL, "artifact": str(best),
                             "reasons": ["only a best-val debug artifact exists: training "
                                         "started but no terminal artifact was written",
                                         LEGACY_NON_RESUMABLE_REASON
                                         + " - this arm must be re-run"]}
            else:
                arms[arm] = {"status": UNUSABLE, "reasons": ["no artifact"]}

        statuses = {a["status"] for a in arms.values()}
        if statuses == {"TRAINING_COMPLETE"}:
            bs = "TRAINING_COMPLETE"
            action = ("RE-EVALUATE both terminal models against the sealed set "
                      "(deterministic recovery; no retraining) before counting the block")
        elif statuses == {UNUSABLE}:
            bs, action = UNUSABLE, "RUN block"
        elif INVALID in statuses:
            bs, action = INVALID, "RERUN block"
        else:
            bs, action = PARTIAL, "RERUN incomplete arm(s); completed arms preserved"
        blocks[f"block{i}"] = {"block_status": bs, "action": action, "arms": arms}

    return {
        "run_dir": str(run_dir), "layout": "legacy_checkpoints_dir",
        "sealed_evaluation": sealed, "blocks": blocks,
        "n_training_complete": sum(1 for b in blocks.values()
                                   if b["block_status"] == "TRAINING_COMPLETE"),
        "complete_ce1_blocks": [],
        "exact_resume_supported": False,
        "exact_resume_reason": LEGACY_NON_RESUMABLE_REASON,
        "note": ("epoch 200 means TRAINING_COMPLETE, not EVALUATED; only a pair of "
                 "EVALUATED arms forms a complete CE1 block"),
    }
