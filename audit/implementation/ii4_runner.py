"""
Generic, resumable II-4 execution runner.

Implements the frozen execution sequence against ANY ConformanceAdapter.
It contains no Gohr/Speck/Keras knowledge: everything case-specific is
delegated to the adapter. The same runner will execute II-4 for another
audited system given a different ConformanceFactor and adapter.

FROZEN SEQUENCE
---------------
  preregister (fingerprint persisted)
  -> freeze firewall (mode SHARED)
  -> generate ONE global confirmatory test set -> persist -> hash
  -> seal that ONE hash under every block id
  -> for each block:
        generate fresh train/validation data -> persist -> hash
        for arm in (declared, realized):
            build model with that arm's predetermined seed
            probe realized value IN MEMORY  == requested  (else abort)
            train (fixed epoch count; sealed set never touched)
            verify terminal-epoch semantics (all epochs completed)
            save terminal model; verify reload is weight-identical
            probe realized value FROM THE SAVED ARTIFACT == requested (else abort)
        evaluate BOTH terminal models on the SAME sealed global set
        consume firewall slot exactly once per (arm, block)
        record accuracies and Delta_k
        any failed arm -> whole block invalid
  -> analyse valid Delta_k
  -> persist analysis + provenance

RESUMABILITY
------------
State is written after every irreversible step. On resume nothing
already produced is regenerated: the sealed test set, a completed
block's data, and a completed arm's terminal model are each reloaded and
HASH-VERIFIED before reuse. A changed configuration (fingerprint) or a
substituted sealed test set fails loudly.

Crash window, handled fail-closed: if a crash occurs after a firewall
slot is consumed but before that arm's accuracy is persisted, the slot
cannot be consumed again (that would be a second look at the sealed
set). The arm is then recorded FAILED with reason
'interrupted_after_consumption', and its block becomes invalid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import sys

import audit as _audit_pkg

# The generic EV framework lives under audit/experimental_validity/.
# Its modules intentionally use top-level imports such as:
#     from framework import firewall
#     from framework.provenance import ...
# Keep those imports working after the repository reorganization.
EV_ROOT = Path(_audit_pkg.__file__).resolve().parent / "experimental_validity"
if str(EV_ROOT) not in sys.path:
    sys.path.insert(0, str(EV_ROOT))
from audit.common.outcomes import ExecutionMode
from audit.common.provenance import sha256_bytes, sha256_file, utc_timestamp
from audit.common.run_manifest import capture_environment
from audit.common.strict_json import dumps_strict
from audit.implementation.conformance import BlockOutcome
from audit.implementation.ii4_analysis import analyse_ii4
from audit.implementation.ii4_experiment import II4Plan

ARMS = ("declared", "realized")
RUNNER_VERSION = "ii4-runner-v2"

REPO_ROOT = Path(_audit_pkg.__file__).resolve().parents[1]

# Generic II-4 machinery whose content determines how the experiment runs.
# Case-specific files are contributed by the adapter (provenance_sources()).
GENERIC_II4_SOURCES = {
    "conformance": "audit/implementation/conformance.py",
    "ii4_design": "audit/implementation/ii4_design.py",
    "ii4_analysis": "audit/implementation/ii4_analysis.py",
    "ii4_experiment": "audit/implementation/ii4_experiment.py",
    "ii4_runner": "audit/implementation/ii4_runner.py",
    "ii4_certificate": "audit/implementation/ii4_certificate.py",
    "ii4_launcher": "audit/implementation/run_ii4.py",
    "common_provenance": "audit/common/provenance.py",
    "common_strict_json": "audit/common/strict_json.py",
    "common_outcomes": "audit/common/outcomes.py",
    "common_run_manifest": "audit/common/run_manifest.py",
    "framework_firewall": "audit/experimental_validity/framework/firewall.py",
}

# Path components under which II-4 must never write. Historical and
# frozen evidence lives under these names throughout the repository.
PROTECTED_COMPONENTS = frozenset({"evidence", "evidence_bundle", "frozen"})
PROTECTED_REPO_SUBTREES = (
    "audit/evidence_bundle", "audit/integration", "audit/implementation/evidence",
    "audit/implementation/reference", "audit/dataset", "audit/cryptography",
)


def check_output_dir(output_dir: Path, *, repo_root: Path = REPO_ROOT) -> list[str]:
    """
    Refuse output locations that could overwrite or pollute historical
    evidence. Returns problems; empty means acceptable.
    """
    problems: list[str] = []
    target = Path(output_dir).resolve()
    hit = PROTECTED_COMPONENTS.intersection(target.parts)
    if hit:
        problems.append(f"output_dir {target} contains protected path component(s) {sorted(hit)}.")
    for sub in PROTECTED_REPO_SUBTREES:
        protected = (repo_root / sub).resolve()
        if target == protected or protected in target.parents:
            problems.append(f"output_dir {target} lies inside protected subtree {sub}.")
    if target == repo_root.resolve():
        problems.append("output_dir must not be the repository root (reference files live there).")
    if target.exists() and any(target.iterdir()) and not (target / "ii4_state.json").exists():
        problems.append(
            f"output_dir {target} exists, is non-empty and is not an II-4 run directory; "
            "refusing to write into an unrelated location.")
    return problems


def build_source_manifest(adapter, *, repo_root: Path = REPO_ROOT) -> dict[str, dict]:
    """
    sha256 of every II-4-relevant source file (generic + adapter-declared).
    A declared file that does not exist is a hard error - a manifest that
    silently omits a file would certify code it never saw.
    """
    sources = dict(GENERIC_II4_SOURCES)
    for label, rel in adapter.provenance_sources().items():
        if label in sources and sources[label] != rel:
            raise II4ExecutionError(f"source label {label!r} declared twice with different paths.")
        sources[label] = rel
    manifest = {}
    for label, rel in sorted(sources.items()):
        path = repo_root / rel
        if not path.is_file():
            raise II4ExecutionError(f"provenance source {label!r} missing: {rel}")
        manifest[label] = {"path": rel, "sha256": sha256_file(path)}
    return manifest


def software_stack(env: dict) -> dict:
    """
    The software identity bound into the experiment fingerprint. Hardware
    (GPU model, driver) is deliberately NOT part of it: a multi-session run
    may legitimately resume on a different GPU of the same software image,
    so hardware is recorded per session as provenance, while a change of
    SOFTWARE between sessions aborts the run.
    """
    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": env.get("numpy_version"),
        "scipy": env.get("scipy_version"),
        "tensorflow": env.get("tensorflow_version"),
        "keras": env.get("keras_version"),
    }


class II4ExecutionError(RuntimeError):
    """A condition that invalidates the whole experiment (not one block)."""


def _firewall_module():
    try:
        from framework import firewall  # generic framework, shared with EV
    except ImportError as exc:  # pragma: no cover - environment problem
        raise II4ExecutionError(
            "The generic firewall (framework.firewall) is not importable. Put the "
            "experimental_validity root on sys.path.") from exc
    return firewall


@dataclass
class II4Runner:
    plan: II4Plan
    adapter: Any
    state: dict[str, Any] = field(default_factory=dict)
    _provenance: Optional[dict[str, Any]] = None

    def runtime_provenance(self) -> dict[str, Any]:
        """
        Source manifest + environment, computed once per session. The
        experiment fingerprint binds the preregistered configuration, the
        exact source code and the software stack: resuming with ANY changed
        source file or software version fails loudly.
        """
        if self._provenance is None:
            env = capture_environment(str(REPO_ROOT))
            details = self.adapter.environment_details()
            env.update({f"adapter_{k}": v for k, v in details.items()})
            sources = build_source_manifest(self.adapter)
            stack = software_stack(env)
            manifest_hash = sha256_bytes(dumps_strict(sources, sort_keys=True).encode("utf-8"))
            fingerprint = sha256_bytes(dumps_strict({
                "config_fingerprint": self.plan.config_fingerprint(),
                "source_manifest_sha256": manifest_hash,
                "software_stack": stack,
            }, sort_keys=True).encode("utf-8"))
            self._provenance = {
                "config_fingerprint": self.plan.config_fingerprint(),
                "source_manifest": sources,
                "source_manifest_sha256": manifest_hash,
                "software_stack": stack,
                "environment": env,
                "experiment_fingerprint": fingerprint,
            }
        return self._provenance

    # ------------------------------------------------------------------
    # paths / persistence
    # ------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.plan.output_dir)

    @property
    def state_path(self) -> Path:
        return self.root / "ii4_state.json"

    @property
    def firewall_path(self) -> Path:
        return self.root / "ii4_firewall.json"

    def _save_state(self) -> None:
        self.state["updated_at_utc"] = utc_timestamp()
        self.state_path.write_text(dumps_strict(self.state, indent=2, sort_keys=True))

    def _write_json(self, name: str, obj: Any) -> Path:
        path = self.root / name
        path.write_text(dumps_strict(obj, indent=2, sort_keys=True, default=str))
        return path

    # ------------------------------------------------------------------
    # stage 1: preregister + firewall
    # ------------------------------------------------------------------
    def _preregister_or_resume(self):
        fw_mod = _firewall_module()
        prov = self.runtime_provenance()
        fingerprint = prov["experiment_fingerprint"]
        self.root.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("config_fingerprint") != fingerprint:
                raise II4ExecutionError(
                    "Resume refused: the experiment fingerprint differs from the recorded run "
                    f"({self.state.get('config_fingerprint')} vs {fingerprint}). The configuration, "
                    "a source file, or the software stack has changed; that is a different "
                    "experiment.")
            firewall = fw_mod.TestSetFirewall.load(self.firewall_path)
            if firewall.confirmatory_data_mode is not fw_mod.ConfirmatoryDataMode.SHARED:
                raise II4ExecutionError("Resume refused: persisted firewall is not in SHARED mode.")
            if firewall.freeze_record.frozen_config_hash != fingerprint:
                raise II4ExecutionError("Resume refused: firewall was frozen for a different config.")
            self.state.setdefault("resume_events", []).append({
                "at_utc": utc_timestamp(),
                "environment": prov["environment"],
            })
            first_hw = self.state.get("session_environments", [{}])[0]
            if first_hw.get("adapter_gpus") != prov["environment"].get("adapter_gpus"):
                self.state.setdefault("recorded_deviations", []).append({
                    "at_utc": utc_timestamp(), "kind": "hardware_changed_on_resume",
                    "before": first_hw.get("adapter_gpus"),
                    "after": prov["environment"].get("adapter_gpus"),
                })
            self.state.setdefault("session_environments", []).append(prov["environment"])
            return firewall

        prereg = self.plan.preregistration()
        prereg["config_fingerprint"] = prov["config_fingerprint"]
        prereg["experiment_fingerprint"] = fingerprint
        prereg["source_manifest"] = prov["source_manifest"]
        prereg["source_manifest_sha256"] = prov["source_manifest_sha256"]
        prereg["software_stack"] = prov["software_stack"]
        prereg["environment"] = prov["environment"]
        prereg["runner_version"] = RUNNER_VERSION
        self._write_json("ii4_preregistration.json", prereg)

        firewall = fw_mod.TestSetFirewall(
            experiment_id=self.plan.experiment_id,
            confirmatory_data_mode=fw_mod.ConfirmatoryDataMode.SHARED,
        )
        firewall.freeze(fw_mod.FreezeRecord(
            frozen_config_hash=fingerprint,
            frozen_permutation_hash=None,
            frozen_replicate_plan_hash=fingerprint,
            frozen_statistical_plan_hash=self.plan.design.preregistration_hash(),
            frozen_at_utc=utc_timestamp(),
        ))
        firewall.save(self.firewall_path)
        self.state = {
            "experiment_id": self.plan.experiment_id,
            "execution_mode": self.plan.execution_mode.value,
            "config_fingerprint": fingerprint,
            "plan_config_fingerprint": prov["config_fingerprint"],
            "source_manifest_sha256": prov["source_manifest_sha256"],
            "software_stack": prov["software_stack"],
            "session_environments": [prov["environment"]],
            "runner_version": RUNNER_VERSION,
            "created_at_utc": utc_timestamp(),
            "test_set": None,
            "blocks": {},
        }
        self._save_state()
        return firewall

    # ------------------------------------------------------------------
    # stage 2: ONE global sealed test set
    # ------------------------------------------------------------------
    def _sealed_test_set(self, firewall):
        block_ids = [f"block{i}" for i in range(self.plan.design.n_blocks)]
        rec = self.state.get("test_set")
        if rec is None:
            path, digest, n = self.adapter.generate_sealed_test_set(self.root / "sealed_test")
            rec = {"path": str(path), "hash": digest, "n": n,
                   "generated_at_utc": utc_timestamp(), "scope": "GLOBAL_SHARED"}
            self.state["test_set"] = rec
            self._save_state()
            for bid in block_ids:                     # ONE hash sealed under every block
                firewall.seal_confirmatory_dataset(bid, digest)
            firewall.save(self.firewall_path)
        else:
            sealed = set(firewall.sealed_dataset_hashes.values())
            if sealed != {rec["hash"]}:
                raise II4ExecutionError(
                    f"Sealed-set integrity failure: firewall seals {sorted(sealed)} but the run "
                    f"recorded {rec['hash']}. The sealed test set may not be substituted.")
        # Always hash-verify on load; a substituted file fails here.
        return rec, self.adapter.load_sealed_test_set(rec["path"], rec["hash"])

    # ------------------------------------------------------------------
    # stage 3: one arm
    # ------------------------------------------------------------------
    def _train_arm(self, block_id: str, block_index: int, arm: str, block_data) -> dict:
        requested = self.plan.factor.arm_values[arm]
        seed = self.plan.seeds.seed_for(block_index, arm)
        rec: dict[str, Any] = {"arm": arm, "requested_value": requested, "seed": seed,
                               "status": "STARTED", "started_at_utc": utc_timestamp()}
        arm_dir = self.root / "blocks" / block_id / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        model = self.adapter.build_model(requested, seed=seed)
        probed_mem = self.adapter.probe_model(model)
        rec["probe_in_memory"] = probed_mem
        if probed_mem != requested:
            raise II4ExecutionError(
                f"{block_id}/{arm}: requested {requested!r} but the constructed model realizes "
                f"{probed_mem!r}. Aborting: the experiment would not measure what it claims.")

        try:
            result = self.adapter.train(
                model, block_data,
                checkpoint_path=str(arm_dir / "best_val_loss_checkpoint.weights.h5"))
        except Exception as exc:  # training crash -> block invalid, experiment continues
            rec.update(status="FAILED", failure={"stage": "train", "reason": type(exc).__name__,
                                                 "detail": str(exc)})
            return rec

        rec["secondary"] = {
            "final_val_acc": getattr(result, "final_val_acc", None),
            "max_val_acc_historical_descriptive_only": getattr(result, "max_val_acc", None),
            "best_val_loss_checkpoint_hash_provenance_only": getattr(result, "checkpoint_hash", None),
        }
        reason = getattr(result, "failure_reason", None)
        if reason is not None:
            rec.update(status="FAILED", failure={"stage": "train",
                                                 "reason": getattr(reason, "value", str(reason)),
                                                 "detail": "reported by training"})
            return rec

        completed = getattr(result, "n_epochs_completed", None)
        if completed != self.plan.design.epochs:
            rec.update(status="FAILED", failure={
                "stage": "terminal_epoch_check", "reason": "epochs_incomplete",
                "detail": f"completed {completed} of {self.plan.design.epochs} epochs"})
            return rec
        rec["terminal_epoch"] = completed

        saved = self.adapter.save_terminal_model(model, arm_dir / "terminal_model.keras")
        rec["terminal_model"] = saved
        if not saved.get("reload_weight_identical"):
            raise II4ExecutionError(
                f"{block_id}/{arm}: saved terminal model is not weight-identical to the "
                "in-memory terminal model; evaluating it would not be terminal-epoch evaluation.")

        probed_art = self.adapter.probe_realized_value(saved["path"])
        rec["probe_artifact"] = probed_art
        if probed_art != requested:
            raise II4ExecutionError(
                f"{block_id}/{arm}: persisted artifact realizes {probed_art!r}, requested "
                f"{requested!r}. Aborting.")
        rec.update(status="TRAINED", trained_at_utc=utc_timestamp(),
                   evaluation_rule="terminal_epoch")
        return rec

    # ------------------------------------------------------------------
    # stage 4: one block
    # ------------------------------------------------------------------
    def _run_block(self, block_index: int, firewall, test_rec: dict, sealed_test) -> BlockOutcome:
        block_id = f"block{block_index}"
        blk = self.state["blocks"].setdefault(block_id, {"arms": {}})

        if "dataset" not in blk:
            blk["dataset"] = self.adapter.generate_block_datasets(
                block_id, self.root / "blocks" / block_id / "data")
            self._save_state()
        block_data = None   # loaded lazily, hash-verified

        for arm in ARMS:
            prev = blk["arms"].get(arm)
            if prev and prev["status"] in ("TRAINED", "EVALUATED", "FAILED"):
                if prev["status"] in ("TRAINED", "EVALUATED"):
                    # hash-verify the completed arm's artifact before reuse
                    self.adapter.load_terminal_model(prev["terminal_model"]["path"],
                                                     prev["terminal_model"]["hash"])
                continue
            if block_data is None:
                block_data = self.adapter.load_block_datasets(blk["dataset"])
            blk["arms"][arm] = self._train_arm(block_id, block_index, arm, block_data)
            self._save_state()

        # evaluate only if BOTH arms trained - otherwise the block is invalid
        if all(blk["arms"][a]["status"] in ("TRAINED", "EVALUATED") for a in ARMS):
            for arm in ARMS:
                rec = blk["arms"][arm]
                if rec["status"] == "EVALUATED":
                    continue
                model = self.adapter.load_terminal_model(rec["terminal_model"]["path"],
                                                         rec["terminal_model"]["hash"])
                accuracy = self.adapter.evaluate_terminal(model, sealed_test)
                try:
                    firewall.consume_confirmatory_evaluation(
                        condition_id=arm, replicate_id=block_id,
                        observed_dataset_hash=test_rec["hash"])
                except Exception as exc:
                    rec.update(status="FAILED", failure={
                        "stage": "evaluate", "reason": "interrupted_after_consumption",
                        "detail": str(exc)})
                    self._save_state()
                    continue
                firewall.save(self.firewall_path)
                rec.update(status="EVALUATED", sealed_test_accuracy=accuracy,
                           sealed_test_hash=test_rec["hash"], evaluated_at_utc=utc_timestamp())
                self._save_state()

        return self._block_outcome(block_index)

    def _block_outcome(self, block_index: int) -> BlockOutcome:
        block_id = f"block{block_index}"
        blk = self.state["blocks"].get(block_id, {})
        ds = blk.get("dataset", {})
        outcome = BlockOutcome(
            block_id=block_id,
            dataset_hash=ds.get("train", {}).get("hash", ""),
            declared_seed=self.plan.seeds.seed_for(block_index, "declared"),
            realized_seed=self.plan.seeds.seed_for(block_index, "realized"),
        )
        for arm in ARMS:
            rec = blk.get("arms", {}).get(arm)
            if rec is None:
                outcome.record_failure(arm=arm, reason="not_run", detail="", seed=None)
                continue
            if rec["status"] == "EVALUATED":
                setattr(outcome, f"{arm}_arm_value", rec["sealed_test_accuracy"])
                outcome.secondary[arm] = rec.get("secondary", {})
            else:
                f = rec.get("failure", {"reason": rec["status"]})
                outcome.record_failure(arm=arm, reason=f.get("reason", "unknown"),
                                       detail=f.get("detail", ""), seed=rec.get("seed"))
        return outcome

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def run(self, *, execute: bool = False, allow_production: bool = False) -> dict[str, Any]:
        """
        execute=False -> validation only, touches nothing on disk.
        A PRODUCTION plan additionally requires allow_production=True.
        """
        problems = (self.plan.validate_production()
                    if self.plan.execution_mode is ExecutionMode.PRODUCTION
                    else self.plan.validate())
        if problems:
            raise II4ExecutionError(f"Plan validation failed: {problems}")
        dir_problems = check_output_dir(self.plan.output_dir)
        if dir_problems:
            raise II4ExecutionError(f"Output directory refused: {dir_problems}")
        if not execute:
            return {"executed": False, "validation": "CLEAN",
                    "config_fingerprint": self.plan.config_fingerprint()}
        if self.plan.execution_mode is ExecutionMode.PRODUCTION and not allow_production:
            raise II4ExecutionError(
                "Refusing to execute a PRODUCTION II-4 plan without allow_production=True.")

        firewall = self._preregister_or_resume()
        test_rec, sealed_test = self._sealed_test_set(firewall)
        outcomes = [self._run_block(i, firewall, test_rec, sealed_test)
                    for i in range(self.plan.design.n_blocks)]

        diffs = [o.signed_difference for o in outcomes if o.is_valid]
        analysis = analyse_ii4(
            diffs, n_blocks_requested=self.plan.design.n_blocks,
            min_valid_blocks=self.plan.design.min_valid_blocks,
            delta=self.plan.design.delta_materiality, alpha=self.plan.design.alpha,
        )
        non_evidentiary = self.plan.execution_mode is not ExecutionMode.PRODUCTION
        report = {
            "experiment_id": self.plan.experiment_id,
            "execution_mode": self.plan.execution_mode.value,
            "non_evidentiary": non_evidentiary,
            "config_fingerprint": self.state["config_fingerprint"],
            "plan_config_fingerprint": self.runtime_provenance()["config_fingerprint"],
            "source_manifest": self.runtime_provenance()["source_manifest"],
            "source_manifest_sha256": self.runtime_provenance()["source_manifest_sha256"],
            "software_stack": self.runtime_provenance()["software_stack"],
            "session_environments": self.state.get("session_environments", []),
            "recorded_deviations": self.state.get("recorded_deviations", []),
            "conformance_factor": self.plan.factor.to_dict(),
            "sealed_test_set": test_rec,
            "blocks": [o.to_dict() for o in outcomes],
            "analysis": analysis.to_dict(),
            "firewall": firewall.to_dict(),
            "generated_at_utc": utc_timestamp(),
        }
        self._write_json("ii4_analysis.json", report)
        self.state["completed_at_utc"] = utc_timestamp()
        self._save_state()
        return report
