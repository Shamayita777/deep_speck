#!/usr/bin/env python3
"""
Paired-block II-4 comparison runner - GENERIC.

Executes a ComparisonPlan along any ComparisonAxis. Contains no
knowledge of Gohr, Speck, Keras, TensorFlow or residual depth; every
case-specific operation is delegated to a ComparisonAdapter.

RELATIONSHIP TO ii4_runner.py. That runner is frozen: it produced the
completed conformance experiment and its source hash is bound into that
experiment's evidence. It is not modified, not imported and not
re-run. This runner is its generalization along three axes, and the test
suite demonstrates the generalization by reconstructing the frozen
experiment's arm semantics through the generic ComparisonAxis and
asserting agreement, rather than by asserting parity in prose:

  1. ARMS ARE NAMED BY THE AXIS. ii4_runner hardcodes ARMS =
     ("declared", "realized"). Those names are meaningless for an
     implementation comparison, so arms here come from the axis.

  2. ARMS MAY BE DIFFERENT CODE. ii4_runner calls one
     adapter.build_model(value) and one adapter.train(). Here both are
     arm-aware, so an implementation axis dispatches to a different
     implementation per arm.

  3. BLOCKS MAY BE REPLACED. ii4_runner has a fixed block list. Here a
     technically failed block may be replaced from a frozen budget,
     under the rule in ii4_comparison_design.REPLACEMENT_BLOCK_RULE.

WHAT THIS RUNNER REFUSES TO DO
------------------------------
  * train before verifying that both arms realize the structure they
    declared, and - on an implementation axis - that they realize the
    SAME structure as each other (the comparison's premise);
  * evaluate a block unless BOTH arms trained to the terminal epoch;
  * replace a block that already reached sealed-test evaluation;
  * exceed the frozen replacement budget;
  * resume a run whose configuration or source code has changed;
  * write anywhere that could disturb historical evidence;
  * emit an analysis for a run with fewer than min_valid_blocks valid
    blocks (the frozen analysis returns INCONCLUSIVE by rule).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import audit as _audit_pkg

from audit.common.outcomes import ExecutionMode
from audit.common.provenance import sha256_bytes, sha256_file, utc_timestamp
from audit.common.run_manifest import capture_environment
from audit.common.strict_json import dumps_strict
from audit.implementation.architecture_signature import (
    SignatureMismatch,
    require_identical,
)
from audit.implementation.arm_recovery import (
    RECOVERY_EXACTNESS,
    EpochLog,
    InjectedInterruption,
)
from audit.implementation.durable_state import (
    WHAT_SIGKILL_ESTABLISHES,
    EVAL_CONSUMED,
    EVAL_MEASURED,
    EVAL_RECORDED,
    CommitManifest,
    DurableStateError,
    EvalJournal,
    atomic_write_text,
    save_firewall_atomically,
)
from audit.implementation.comparison import (
    AxisKind,
    ComparisonBlockOutcome,
    collect_valid_differences,
)
from audit.implementation.ii4_analysis import analyse_ii4
from audit.implementation.ii4_comparison_design import (
    ComparisonPlan,
    IMPLEMENTATION_INTERPRETATION_RULE,
    apply_implementation_interpretation_rule,
)
from audit.implementation.ii4_runner import (     # reuse, do not duplicate
    PROTECTED_COMPONENTS,
    PROTECTED_REPO_SUBTREES,
    REPO_ROOT,
    software_stack,
)

RUNNER_VERSION = "ii4-comparison-runner-v1"

EV_ROOT = Path(_audit_pkg.__file__).resolve().parent / "experimental_validity"
if str(EV_ROOT) not in sys.path:
    sys.path.insert(0, str(EV_ROOT))

#: Generic machinery whose content determines how a comparison runs.
#: Case-specific files are contributed by adapter.provenance_sources().
GENERIC_COMPARISON_SOURCES = {
    "comparison": "audit/implementation/comparison.py",
    # Added after review: these were NOT bound, so the entire recovery
    # and signature layer could change between review and launch while
    # --execute proceeded. test_freeze_manifest_coverage.py asserts that
    # mutating each of them makes execution refuse.
    "arm_recovery": "audit/implementation/arm_recovery.py",
    "durable_state": "audit/implementation/durable_state.py",
    "architecture_signature": "audit/implementation/architecture_signature.py",
    "preflight": "audit/implementation/preflight.py",
    "launcher": "audit/implementation/run_ii4_impl.py",
    "preregistration": "audit/implementation/preregistrations/ii4_impl_gohr_v1.py",
    "declared_architecture": "audit/implementation/adapters/gohr_declared_architecture.json",
    "ii4_comparison_design": "audit/implementation/ii4_comparison_design.py",
    "ii4_comparison_runner": "audit/implementation/ii4_comparison_runner.py",
    "ii4_comparison_power": "audit/implementation/ii4_comparison_power.py",
    "ii4_comparison_certificate": "audit/implementation/ii4_comparison_certificate.py",
    "ii4_analysis": "audit/implementation/ii4_analysis.py",
    "ii4_runner": "audit/implementation/ii4_runner.py",
    "stages": "audit/implementation/stages.py",
    "common_provenance": "audit/common/provenance.py",
    "common_strict_json": "audit/common/strict_json.py",
    "common_outcomes": "audit/common/outcomes.py",
    "common_run_manifest": "audit/common/run_manifest.py",
    "framework_firewall": "audit/experimental_validity/framework/firewall.py",
    # Added after an independent derivation of the import graph from the
    # real entry points found these five reachable but UNBOUND. Each can
    # change what the experiment does, verifies or records:
    #
    #   audit/__init__.py     executes on every import of the package.
    #                         It is empty today, which is precisely why
    #                         it is bound: an empty file that nobody
    #                         watches is the cheapest place to put code.
    #   common/ids.py         validate_identifier() gates run_id and
    #                         experiment_id in run_manifest.py, so it
    #                         decides which identities the provenance
    #                         layer will accept.
    #   conformance.py        ConformanceFactor and BlockOutcome back
    #                         comparison.conformance_axis_from_factor(),
    #                         the bridge from the conformance axis to the
    #                         generic ComparisonAxis - the framework's
    #                         own abstraction, not Gohr-specific.
    #   ii4_design.py         the historical depth experiment's design
    #   ii4_experiment.py     and execution. Reachable through
    #                         ii4_runner. They produced the V1-V6
    #                         evidence this run must not disturb, so an
    #                         edit to either must make --execute refuse.
    "package_init": "audit/__init__.py",
    "common_ids": "audit/common/ids.py",
    "conformance": "audit/implementation/conformance.py",
    "ii4_design": "audit/implementation/ii4_design.py",
    "ii4_experiment": "audit/implementation/ii4_experiment.py",
}

#: The test modules whose PASSING authorizes a launch. Bound for a
#: different reason from the sources above: a weakened test cannot change
#: what the experiment does, but it changes what the clearance MEANS. The
#: readiness gate runs these fresh at launch, so an edited test would
#: simply pass and the gate would clear - the freeze is what notices.
#: Binding them means a test edit after the freeze forces a new freeze,
#: which is the correct behaviour for a run whose authorization rests on
#: those tests.
VERIFICATION_SOURCES = {
    "test_durable_state": "audit/tests/test_durable_state.py",
    "test_arm_recovery": "audit/tests/test_arm_recovery.py",
    "test_arm_recovery_keras": "audit/tests/test_arm_recovery_keras.py",
    "test_firewall_real_integration": "audit/tests/test_firewall_real_integration.py",
    "test_firewall_crash_recovery": "audit/tests/test_firewall_crash_recovery.py",
    "test_freeze_manifest_coverage": "audit/tests/test_freeze_manifest_coverage.py",
    "test_ii4_interruption": "audit/tests/test_ii4_interruption.py",
    "test_ii4_comparison": "audit/tests/test_ii4_comparison.py",
    "test_preflight_filesystem": "audit/tests/test_preflight_filesystem.py",
    "test_execute_gate": "audit/tests/test_execute_gate.py",
    "test_unrecoverable_state": "audit/tests/test_unrecoverable_state.py",
    "test_journal_exactly_once": "audit/tests/test_journal_exactly_once.py",
    "test_repeated_interruption": "audit/tests/test_repeated_interruption.py",
    "test_unrecoverable_state": "audit/tests/test_unrecoverable_state.py",
    "test_suite_integrity": "audit/tests/test_suite_integrity.py",
    "kill_worker": "audit/tests/kill_worker.py",
}


class ComparisonExecutionError(RuntimeError):
    """A condition that invalidates the whole experiment (not one block)."""


class BlockTechnicalFailure(RuntimeError):
    """A per-block technical fault. Invalidates one block, not the run."""


def _firewall_module():
    try:
        from framework import firewall
    except ImportError as exc:                              # pragma: no cover
        raise ComparisonExecutionError(
            "The generic firewall (framework.firewall) is not importable. Put the "
            "experimental_validity root on sys.path.") from exc
    return firewall


def firewall_confirms_consumption(firewall_path, *, condition_id: str, replicate_id: str,
                                  sealed_hash: str) -> tuple[bool, str]:
    """
    Does the PERSISTED TestSetFirewall record a confirmatory consumption
    for exactly this (condition, replicate, sealed hash)?

    Written against the REAL `framework.firewall.TestSetFirewall` schema,
    which `to_dict()` serialises as:

        experiment_id             str
        frozen                    bool
        freeze_record             FreezeRecord fields, or None
        confirmatory_data_mode    "PER_REPLICATE" | "SHARED"
        sealed_dataset_hashes     {replicate_id: hash}
        consumed_evaluation_keys  sorted ["<condition_id>:<replicate_id>", ...]

    An earlier version of this function was written against a mock whose
    field was called `consumed` and held pairs. Against the real firewall
    that lookup finds nothing, so EVERY exception would have been
    reported as unconfirmed and every crash-after-consumption would have
    lost its block. The key is now the exact string the firewall itself
    builds, `f"{condition_id}:{replicate_id}"` - not a substring search,
    which would let "reference:block1" satisfy a query about
    "reference:block10".

    WHY THIS EXISTS AT ALL. The runner must not treat an arbitrary
    exception from `consume_confirmatory_evaluation` as proof the permit
    was already spent. The real firewall raises `FirewallViolation` for
    two quite different things: a genuine duplicate, and an
    OBSERVED-vs-SEALED HASH MISMATCH, which means the confirmatory data
    have been substituted. Recording an accuracy in the second case would
    be the most serious failure this framework can produce.

    FAILS CLOSED. An unloadable file, an unfrozen firewall, a
    non-SHARED mode, a sealed-hash mismatch, a missing or differently
    shaped key list, or no matching key all return False with the
    reason. False means "a prior consumption has NOT been established" -
    never "probably fine".
    """
    path = Path(firewall_path)
    if not path.is_file():
        return False, (f"no persisted firewall at {path}: a prior consumption cannot be "
                       "confirmed from memory alone")
    fw_mod = _firewall_module()
    try:
        firewall = fw_mod.TestSetFirewall.load(path)
        record = firewall.to_dict()
    except Exception as exc:                                    # noqa: BLE001
        return False, (f"the persisted firewall at {path} could not be read "
                       f"({type(exc).__name__}: {exc}), so no prior consumption can be "
                       "confirmed")
    if not isinstance(record, dict):
        return False, f"the firewall serialised to {type(record).__name__}, not a mapping"

    if record.get("frozen") is not True or not record.get("freeze_record"):
        return False, ("the persisted firewall carries no freeze record, so nothing it "
                       "contains can be treated as a sealed, consumed confirmatory "
                       "evaluation")
    expected_mode = fw_mod.ConfirmatoryDataMode.SHARED.value
    if record.get("confirmatory_data_mode") != expected_mode:
        return False, (f"the persisted firewall is in mode "
                       f"{record.get('confirmatory_data_mode')!r}, not {expected_mode!r}; "
                       "this experiment's design declares one shared confirmatory set")

    sealed = record.get("sealed_dataset_hashes")
    if not isinstance(sealed, dict):
        return False, ("the firewall record carries no 'sealed_dataset_hashes' mapping, so "
                       "this run's sealed set cannot be matched against it")
    actual = sealed.get(replicate_id)
    if actual != sealed_hash:
        return False, (f"the firewall seals {actual!r} for {replicate_id} but this run's "
                       f"sealed test set is {sealed_hash!r}. The confirmatory data do not "
                       "match; this is NOT a duplicate consumption and must not be treated "
                       "as one")

    keys = record.get("consumed_evaluation_keys")
    if keys is None:
        return False, ("the firewall record carries no 'consumed_evaluation_keys' list, so a "
                       "prior consumption cannot be confirmed")
    if not isinstance(keys, (list, tuple, set)):
        return False, (f"'consumed_evaluation_keys' is a {type(keys).__name__}, which this "
                       "version cannot interpret")
    wanted = f"{condition_id}:{replicate_id}"
    if wanted in {str(k) for k in keys}:
        return True, (f"the persisted firewall records consumed_evaluation_key {wanted!r} "
                      f"against sealed set {sealed_hash}")
    return False, (f"the persisted firewall records no consumed_evaluation_key {wanted!r} "
                   f"(it holds {len(keys)} key(s)); the failure has some other cause")


def check_output_dir(output_dir: Path, *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Refuse output locations that could overwrite or pollute evidence."""
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
    # A directory is recognised as OURS if it holds any artifact this
    # runner writes - not only state.json. A process killed between
    # generating the sealed test set and the first state write leaves
    # sealed_test/ and firewall.json but no state.json, and requiring
    # state.json alone made that run directory permanently unusable:
    # the resume was refused as "an unrelated location" and the sealed
    # set could not be reused, which is precisely the data that must
    # never be regenerated.
    OURS = ("state.json", "firewall.json", "preregistration.json", "sealed_test",
            "blocks", "analysis.json")
    if target.exists() and any(target.iterdir()) and not any(
            (target / name).exists() for name in OURS):
        problems.append(
            f"output_dir {target} exists, is non-empty and contains none of {list(OURS)}, so "
            "it is not a comparison run directory; refusing to write into an unrelated "
            "location.")
    return problems


def build_source_manifest(adapter, *, repo_root: Path = REPO_ROOT) -> dict[str, dict]:
    """
    sha256 of every source file that can influence the result.

    A declared file that does not exist is a HARD ERROR. A manifest that
    silently omits a missing file would let the experiment fingerprint
    stay stable while the code that produced the result changed.
    """
    manifest: dict[str, dict] = {}
    declared = {**GENERIC_COMPARISON_SOURCES, **VERIFICATION_SOURCES,
                **dict(adapter.provenance_sources())}
    missing = []
    for label, rel in sorted(declared.items()):
        path = (repo_root / rel)
        if not path.is_file():
            missing.append(rel)
            continue
        manifest[label] = {"path": rel, "sha256": sha256_file(path),
                           "bytes": path.stat().st_size}
    if missing:
        raise ComparisonExecutionError(
            f"declared source file(s) absent, refusing to build an incomplete source "
            f"manifest: {missing}")
    return manifest


@dataclass
class ComparisonRunner:
    plan: ComparisonPlan
    adapter: Any
    state: dict[str, Any] = field(default_factory=dict)
    #: Distinguishes one execution session from another in the epoch log,
    #: so a resumed arm's history shows WHERE it was resumed.
    session_id: str = field(default_factory=lambda: utc_timestamp().replace(":", ""))
    #: Fault injection for the interruption test ONLY. Production
    #: validation refuses a plan with this set (see run()).
    interrupt_after_epoch: Optional[int] = None
    #: Test-only hook called at each named runner-level persistence
    #: boundary, so a process-kill test can terminate at an exact point
    #: in the sealed-set and firewall lifecycle. Never set in production.
    boundary_hook: Optional[Any] = None
    _provenance: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # provenance / paths
    # ------------------------------------------------------------------
    def runtime_provenance(self) -> dict[str, Any]:
        if self._provenance is None:
            env = capture_environment(str(REPO_ROOT))
            env.update({f"adapter_{k}": v for k, v in self.adapter.environment_details().items()})
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
                "source_manifest": sources, "source_manifest_sha256": manifest_hash,
                "software_stack": stack, "environment": env,
                "experiment_fingerprint": fingerprint,
            }
        return self._provenance

    @property
    def root(self) -> Path:
        return Path(self.plan.output_dir)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def firewall_path(self) -> Path:
        return self.root / "firewall.json"

    @property
    def arm_ids(self) -> tuple[str, str]:
        return self.plan.axis.arm_ids

    def _boundary(self, name: str) -> None:
        if self.boundary_hook is not None:
            self.boundary_hook(name)

    def _save_state(self) -> None:
        """
        Atomically. `write_text` truncates first, so a process killed
        mid-write would leave an empty or half-written state file where
        the run's entire progress record used to be - and the next
        resume could not distinguish that from a fresh run. The state
        file is rewritten after every arm and every epoch-completing
        callback, so the window is not theoretical.
        """
        self.state["updated_at_utc"] = utc_timestamp()
        atomic_write_text(self.state_path,
                          dumps_strict(self.state, indent=2, sort_keys=True, default=str))

    def _write_json(self, name: str, obj: Any) -> Path:
        path = self.root / name
        atomic_write_text(path, dumps_strict(obj, indent=2, sort_keys=True, default=str))
        return path

    # ------------------------------------------------------------------
    # stage 1: preregister or resume
    # ------------------------------------------------------------------
    def _preregister_or_resume(self):
        fw_mod = _firewall_module()
        prov = self.runtime_provenance()
        fingerprint = prov["experiment_fingerprint"]
        self.root.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("experiment_fingerprint") != fingerprint:
                raise ComparisonExecutionError(
                    "Refusing to resume: the experiment fingerprint has changed since this run "
                    "started. The preregistered configuration, a source file or the software "
                    "stack differs, so continuing would mix two experiments in one artifact.\n"
                    f"  recorded : {self.state.get('experiment_fingerprint')}\n"
                    f"  current  : {fingerprint}\n"
                    "Start a new run directory, or restore the recorded source state.")
            firewall = fw_mod.TestSetFirewall.load(self.firewall_path)
            if firewall.confirmatory_data_mode is not fw_mod.ConfirmatoryDataMode.SHARED:
                raise ComparisonExecutionError(
                    "Resume refused: the persisted firewall is not in SHARED mode, but this "
                    "design declares ONE shared confirmatory test set.")
            if not firewall.is_frozen():
                raise ComparisonExecutionError(
                    "Resume refused: the persisted firewall carries no freeze record.")
            if firewall.freeze_record.frozen_config_hash != fingerprint:
                raise ComparisonExecutionError(
                    "Resume refused: the firewall was frozen for a different configuration "
                    f"({firewall.freeze_record.frozen_config_hash} vs {fingerprint}).")
            self.state.setdefault("session_environments", []).append(
                {"resumed_at_utc": utc_timestamp(), "software_stack": prov["software_stack"]})
        else:
            prereg = self.plan.preregistration()
            prereg["config_fingerprint"] = prov["config_fingerprint"]
            prereg["experiment_fingerprint"] = fingerprint
            prereg["source_manifest"] = prov["source_manifest"]
            prereg["source_manifest_sha256"] = prov["source_manifest_sha256"]
            prereg["software_stack"] = prov["software_stack"]
            prereg["runner_version"] = RUNNER_VERSION
            self._write_json("preregistration.json", prereg)

            # The real firewall REFUSES to seal anything before freeze(),
            # and the freeze record must come from this run's actual
            # preregistration and provenance - not from placeholders
            # invented to get past the check.
            firewall = fw_mod.TestSetFirewall(
                experiment_id=self.plan.experiment_id,
                confirmatory_data_mode=fw_mod.ConfirmatoryDataMode.SHARED,
            )
            firewall.freeze(self._freeze_record(fw_mod, fingerprint))
            self._boundary("firewall_frozen_in_memory")
            self.state = {
                "runner_version": RUNNER_VERSION,
                "experiment_id": self.plan.experiment_id,
                "experiment_fingerprint": fingerprint,
                "config_fingerprint": prov["config_fingerprint"],
                "preregistration_sha256": self.plan.preregistration_hash(),
                "axis_kind": self.plan.axis.kind.value,
                "arm_ids": list(self.arm_ids),
                "started_at_utc": utc_timestamp(),
                "blocks": {},
                "block_order": [],
                "replacements": [],
                "session_environments": [
                    {"started_at_utc": utc_timestamp(), "software_stack": prov["software_stack"]}],
                "recorded_deviations": [],
                "firewall_experiment_id": self.plan.experiment_id,
                "firewall_mode": fw_mod.ConfirmatoryDataMode.SHARED.value,
            }
            save_firewall_atomically(firewall, self.firewall_path)
            self._boundary("firewall_saved_after_freeze")
        self._save_state()
        return firewall

    def _freeze_record(self, fw_mod, fingerprint: str):
        """
        A real `FreezeRecord` built from this run's own frozen artifacts.

        Each hash is the hash of the thing it names, so the record is
        traceable rather than decorative:

          frozen_config_hash           the experiment fingerprint, which
                                       binds the preregistered
                                       configuration, every source file
                                       and the software stack. The resume
                                       path compares against it.
          frozen_permutation_hash      None, and deliberately so. The
                                       real firewall's own docstring
                                       requires this field only for an
                                       experiment that manipulates
                                       representation (H-EV-REPRESENTATION).
                                       This comparison manipulates NO
                                       representation and generates no
                                       permutation, so inventing a hash
                                       here would record a frozen object
                                       that does not exist.
          frozen_replicate_plan_hash   the block and seed plan: n_blocks,
                                       min_valid_blocks, the replacement
                                       budget and every predetermined
                                       (block, arm) seed.
          frozen_statistical_plan_hash the design's own preregistration
                                       hash: endpoint, evaluation rule,
                                       alpha, margin, pairing semantics.
        """
        design = self.plan.design
        replicate_plan = {
            "n_blocks": design.n_blocks,
            "min_valid_blocks": design.min_valid_blocks,
            "max_replacement_blocks": design.max_replacement_blocks,
            "replacement_rule_version": "II4-REPLACEMENT-V1",
            "seed_manifest": self.plan.seeds.to_dict(),
        }
        return fw_mod.FreezeRecord(
            frozen_config_hash=fingerprint,
            frozen_permutation_hash=None,
            frozen_replicate_plan_hash=sha256_bytes(
                dumps_strict(replicate_plan, sort_keys=True, default=str).encode("utf-8")),
            frozen_statistical_plan_hash=design.preregistration_hash(),
            frozen_at_utc=utc_timestamp(),
        )

    # ------------------------------------------------------------------
    # stage 2: the ONE sealed test set
    # ------------------------------------------------------------------
    def _sealed_test_set(self, firewall):
        """
        Generate, record and seal the ONE confirmatory test set, so that
        every persistence boundary is recoverable WITHOUT regenerating or
        substituting it.

        The windows this replaces, each of which bricked a run:

          * generated, recorded in state, crash BEFORE sealing - the old
            code took the "already recorded" branch and compared the
            firewall's (empty) seal set against the recorded hash, which
            raised "Sealed-set integrity failure" permanently;
          * sealed some block ids, crash before the firewall was saved -
            same failure on the next session;
          * generated the file, crash BEFORE the state record - the old
            code regenerated, which is safe only because nothing was
            sealed, and it never checked whether anything HAD been.

        The repair rests on a property of the real firewall:
        `seal_confirmatory_dataset` accepts resealing a replicate with
        the SAME hash and refuses a different one. Sealing is therefore
        IDEMPOTENT, and re-running it every session completes a partial
        seal without ever substituting data.

        A recorded set is never regenerated. If its file is missing or
        its hash has moved, the run REFUSES: the recorded hash may
        already be sealed, and generating a replacement would substitute
        the confirmatory data.
        """
        all_block_ids = [f"block{i}" for i in range(self.plan.design.max_total_blocks)]
        rec = self.state.get("test_set")
        already_sealed = dict(getattr(firewall, "sealed_dataset_hashes", {}) or {})

        if rec is None:
            if already_sealed:
                raise ComparisonExecutionError(
                    "Sealed-set state inconsistency: the firewall already seals "
                    f"{sorted(set(already_sealed.values()))} but the run state records no "
                    "confirmatory test set. Generating one now would substitute data that "
                    "has already been sealed, and possibly already consumed. Refusing: this "
                    "needs a human decision, not a fresh test set.")
            path, digest, n = self.adapter.generate_sealed_test_set(self.root / "sealed_test")
            if not Path(path).is_file():
                raise ComparisonExecutionError(
                    f"the adapter reported generating {path} but no such file exists.")
            actual = sha256_file(path)
            if actual != digest:
                raise ComparisonExecutionError(
                    f"the adapter reported hash {digest} for {path} but it hashes to "
                    f"{actual}; refusing to seal a set whose identity is already unclear.")
            self._boundary("sealed_set_generated")
            rec = {"path": str(path), "hash": digest, "n": n,
                   "generated_at_utc": utc_timestamp(), "scope": "GLOBAL_SHARED",
                   "sealed_under": list(all_block_ids)}
            self.state["test_set"] = rec
            self._save_state()
            self._boundary("sealed_set_recorded")
        else:
            # NEVER regenerate. Verify the recorded file is still itself.
            if not Path(rec["path"]).is_file():
                raise ComparisonExecutionError(
                    f"the recorded confirmatory test set {rec['path']} is missing. It may "
                    "already be sealed and consumed, so it cannot be regenerated: a fresh "
                    "set would be different data under the same recorded hash. Restore the "
                    "file or abandon this run directory.")
            actual = sha256_file(rec["path"])
            if actual != rec["hash"]:
                raise ComparisonExecutionError(
                    f"the confirmatory test set {rec['path']} hashes to {actual} but the run "
                    f"recorded {rec['hash']}. The sealed set has been substituted or "
                    "corrupted; refusing to evaluate against it.")

        # Idempotent (re)sealing: completes a partial seal, and the real
        # firewall refuses a DIFFERENT hash for an already-sealed
        # replicate, so substitution cannot pass through here.
        newly = [bid for bid in all_block_ids if bid not in already_sealed]
        try:
            for bid in all_block_ids:
                firewall.seal_confirmatory_dataset(bid, rec["hash"])
        except Exception as exc:                                 # noqa: BLE001
            raise ComparisonExecutionError(
                f"the firewall refused to seal the confirmatory test set: {exc}") from exc
        self._boundary("sealed_set_sealed_in_memory")
        if newly:
            save_firewall_atomically(firewall, self.firewall_path)
            self._boundary("firewall_saved_after_sealing")
            self.state.setdefault("sealing_events", []).append({
                "session_id": self.session_id, "at_utc": utc_timestamp(),
                "newly_sealed_block_ids": newly, "hash": rec["hash"],
                "note": ("sealing is idempotent for the same hash, so this completes any "
                         "partial seal left by an interrupted session"),
            })
            self._save_state()

        sealed = set(firewall.sealed_dataset_hashes.values())
        if sealed != {rec["hash"]}:
            raise ComparisonExecutionError(
                f"Sealed-set integrity failure: the firewall seals {sorted(sealed)} but the "
                f"run recorded {rec['hash']}. The sealed test set may not be substituted.")
        missing = [bid for bid in all_block_ids
                   if bid not in firewall.sealed_dataset_hashes]
        if missing:
            raise ComparisonExecutionError(
                f"the confirmatory set is not sealed under block id(s) {missing}; a "
                "replacement block could not be evaluated.")
        return rec, self.adapter.load_sealed_test_set(rec["path"], rec["hash"])

    # ------------------------------------------------------------------
    # stage 3: build both arms and check the comparison's premise
    # ------------------------------------------------------------------
    def _build_and_verify_arms(self, block_id: str, block_index: int) -> dict[str, Any]:
        """
        Construct both arms' models and verify structure BEFORE training.

        Two checks, in order:
          (a) each arm realizes the structure IT declared;
          (b) on an IMPLEMENTATION axis, the arms realize the SAME
              structure as each other.

        (b) is the premise of the whole comparison. If two
        implementations build different models, the measured difference
        confounds implementation with architecture and the experiment
        answers no question. Catching it here costs seconds; catching it
        after training costs the whole run.
        """
        built: dict[str, Any] = {}
        probes: dict[str, Any] = {}
        for arm in self.arm_ids:
            seed = self.plan.seeds.seed_for(block_index, arm)
            try:
                model = self.adapter.build_arm_model(arm, seed=seed)
            except Exception as exc:                            # noqa: BLE001
                raise BlockTechnicalFailure(
                    f"{block_id}/{arm}: model construction failed: "
                    f"{type(exc).__name__}: {exc}") from exc
            probe = self.adapter.probe_arm_model(arm, model)
            expected = self.adapter.expected_arm_structure(arm)
            observed_digest = probe.get("digest") if isinstance(probe, dict) else probe
            if expected is None:
                if self.plan.execution_mode is ExecutionMode.PRODUCTION:
                    raise ComparisonExecutionError(
                        f"{block_id}/{arm}: no DECLARED architecture is available to check the "
                        "constructed model against. A production run may not train against an "
                        "undeclared architecture; capture it and re-freeze.")
                self.state.setdefault("undeclared_architecture_warning", []).append(
                    f"{block_id}/{arm}: observed {observed_digest} with nothing declared")
            elif observed_digest != expected:
                detail = ""
                explain = getattr(self.adapter, "explain_structure_mismatch", None)
                if explain is not None:
                    detail = "\n  " + "\n  ".join(explain(arm, probe))
                raise ComparisonExecutionError(
                    f"{block_id}/{arm}: the constructed model realizes architecture "
                    f"{observed_digest}, but the arm declares {expected}. Aborting: the "
                    f"experiment would not measure what it claims to measure.{detail}")
            built[arm] = model
            probes[arm] = probe

        if self.plan.axis.kind is AxisKind.IMPLEMENTATION:
            a, b = self.arm_ids
            try:
                if isinstance(probes[a], dict) and "layers" in probes[a]:
                    require_identical(probes[a], probes[b], label_a=a, label_b=b,
                                      context=f"{block_id}: ")
                elif probes[a] != probes[b]:
                    raise SignatureMismatch(
                        f"{block_id}: {a}={probes[a]!r} vs {b}={probes[b]!r}")
            except SignatureMismatch as exc:
                raise ComparisonExecutionError(
                    f"{block_id}: the two implementations realize DIFFERENT architectures. An "
                    "implementation comparison presupposes a common architecture; otherwise "
                    "the measured difference confounds implementation with architecture. "
                    f"Aborting rather than recording an uninterpretable result.\n{exc}"
                ) from exc
        return {"models": built, "probes": probes}

    # ------------------------------------------------------------------
    # stage 4: one arm
    # ------------------------------------------------------------------
    def _train_arm(self, block_id: str, block_index: int, arm: str, model, probe,
                   block_data) -> dict[str, Any]:
        """
        Train one arm, with its STARTED state persisted BEFORE training.

        ORDER MATTERS. The arm record is written to disk before fit() is
        entered, so a process killed during the first epoch leaves a
        record that says STARTED rather than no trace at all - which
        would be indistinguishable from an arm that was never attempted
        and would silently restart from zero on resume.

        The model persisted afterwards is the one the ADAPTER returns,
        not the one passed in: on a resume the adapter discards the
        freshly built model and loads the resume checkpoint, so using
        the argument would save an untrained model while every log said
        the arm completed its full schedule.
        """
        seed = self.plan.seeds.seed_for(block_index, arm)
        arm_dir = self.root / "blocks" / block_id / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        rec: dict[str, Any] = {
            "arm": arm, "arm_identity": self.plan.axis.arms[arm].identity, "seed": seed,
            "architecture_digest": (probe.get("digest") if isinstance(probe, dict) else None),
            "status": "STARTED", "started_at_utc": utc_timestamp(),
            "session_id": self.session_id, "arm_dir": str(arm_dir),
        }
        self.state["blocks"][block_id]["arms"][arm] = rec
        self._save_state()

        try:
            result = self.adapter.train_arm(
                arm, model, block_data, arm_dir=arm_dir, seed=seed,
                session_id=self.session_id,
                interrupt_after_epoch=self.interrupt_after_epoch)
        except InjectedInterruption:
            # Test-only, and deliberately NOT converted into a FAILED arm:
            # it stands in for a process kill, which never reaches an
            # except clause at all. Converting it would test a code path
            # that a real interruption does not take.
            self._save_state()
            raise
        except Exception as exc:                                # noqa: BLE001
            rec.update(status="FAILED", failure={
                "stage": "train", "reason": type(exc).__name__, "detail": str(exc)[:1000],
                "technical": True,
                "recoverable": (
                    "the arm's epoch log and resume checkpoint are retained; re-running the "
                    "same command resumes this arm from its last completed epoch rather than "
                    "restarting its schedule")})
            return rec

        rec["epoch_log"] = getattr(result, "epoch_log", None)
        rec["recovery"] = getattr(result, "recovery", None)
        rec["resumed_from_epoch"] = getattr(result, "resumed_from_epoch", 0)
        rec["was_interrupted"] = bool(getattr(result, "was_interrupted", False))
        rec["secondary"] = {
            "final_val_acc": getattr(result, "final_val_acc", None),
            "max_val_acc_historical_descriptive_only": getattr(result, "max_val_acc", None),
            "best_val_loss_checkpoint_hash_provenance_only":
                getattr(result, "checkpoint_hash", None),
            "training_seconds": getattr(result, "wall_clock_seconds", None),
        }
        reason = getattr(result, "failure_reason", None)
        if reason is not None:
            rec.update(status="FAILED", failure={
                "stage": "train", "reason": getattr(reason, "value", str(reason)),
                "detail": "reported by training", "technical": True})
            return rec

        completed = getattr(result, "n_epochs_completed", None)
        if completed != self.plan.design.epochs:
            rec.update(status="FAILED", failure={
                "stage": "terminal_epoch_check", "reason": "epochs_incomplete",
                "detail": (f"the epoch log records {completed} completed epoch(s) of "
                           f"{self.plan.design.epochs}"),
                "technical": True})
            return rec
        rec["terminal_epoch"] = completed

        trained = getattr(result, "model", None) or model
        saved = self.adapter.save_terminal_model(arm, trained,
                                                 arm_dir / "terminal_model.keras")
        rec["terminal_model"] = saved
        if not saved.get("reload_weight_identical"):
            raise ComparisonExecutionError(
                f"{block_id}/{arm}: the saved terminal model is not weight-identical to the "
                "in-memory terminal model; evaluating it would not be terminal-epoch "
                "evaluation.")
        if saved.get("architecture_matches_declared") is False:
            raise ComparisonExecutionError(
                f"{block_id}/{arm}: the persisted terminal model realizes architecture "
                f"{saved.get('architecture_digest')}, not the declared architecture.\n  "
                + "\n  ".join(saved.get("architecture_mismatch") or []))
        rec.update(status="TRAINED", trained_at_utc=utc_timestamp(),
                   evaluation_rule="terminal_epoch")
        return rec

    # ------------------------------------------------------------------
    # stage 5: one block
    # ------------------------------------------------------------------
    def _run_block(self, block_id: str, block_index: int, firewall, test_rec: dict,
                   sealed_test, replaces: Optional[str] = None) -> ComparisonBlockOutcome:
        blk = self.state["blocks"].setdefault(
            block_id, {"arms": {}, "block_index": block_index, "replaces_block_id": replaces})
        if block_id not in self.state["block_order"]:
            self.state["block_order"].append(block_id)

        if "dataset" not in blk:
            blk["dataset"] = self.adapter.generate_block_datasets(
                block_id, self.root / "blocks" / block_id / "data")
            self._save_state()

        pending = [a for a in self.arm_ids if self._arm_needs_training(block_id, a, blk)]
        if pending:
            try:
                built = self._build_and_verify_arms(block_id, block_index)
            except BlockTechnicalFailure as exc:
                for arm in pending:
                    blk["arms"][arm] = {
                        "arm": arm, "status": "FAILED",
                        "seed": self.plan.seeds.seed_for(block_index, arm),
                        "failure": {"stage": "build", "reason": "construction_failed",
                                    "detail": str(exc), "technical": True}}
                self._save_state()
                return self._block_outcome(block_id)

            block_data = self.adapter.load_block_datasets(blk["dataset"])
            for arm in pending:
                blk["arms"][arm] = self._train_arm(
                    block_id, block_index, arm, built["models"][arm], built["probes"][arm],
                    block_data)
                self._save_state()

        # evaluate ONLY if every arm trained; otherwise the block is invalid
        if all(blk["arms"].get(a, {}).get("status") in ("TRAINED", "EVALUATED")
               for a in self.arm_ids):
            for arm in self.arm_ids:
                rec = blk["arms"][arm]
                arm_dir = self.root / "blocks" / block_id / arm
                if rec["status"] == "EVALUATED":
                    # A kill between the state write and the final
                    # journal advance leaves the journal at CONSUMED
                    # forever. The result is safe and the permit was
                    # spent once, but the evidence trail would carry a
                    # non-terminal journal with no explanation, so it is
                    # reconciled here rather than left stranded.
                    try:
                        journal = EvalJournal(arm_dir / "eval_journal.json")
                        if journal.read() is not None:
                            journal.advance_at_least(EVAL_RECORDED)
                    except DurableStateError as exc:
                        self.state.setdefault("journal_reconciliation_problems", []).append(
                            {"block_id": block_id, "arm": arm, "detail": str(exc)[:300]})
                        self._save_state()
                    continue
                journal = EvalJournal(arm_dir / "eval_journal.json")
                tm = rec["terminal_model"]

                # A journal means the measurement ALREADY happened. Adopt
                # it; never re-run inference on a sealed set. The journal
                # is bound to the sealed-set and terminal-model hashes, so
                # a journal belonging to other artifacts is refused rather
                # than adopted.
                try:
                    existing = journal.adopt(sealed_test_hash=test_rec["hash"],
                                             terminal_model_hash=tm["hash"],
                                             arm_id=arm, block_id=block_id)
                except DurableStateError as exc:
                    rec.update(status="FAILED", failure={
                        "stage": "evaluate", "reason": "eval_journal_unusable",
                        "detail": str(exc)[:600], "technical": False})
                    self._save_state()
                    continue

                if existing is None:
                    model = self.adapter.load_terminal_model(arm, tm["path"], tm["hash"])
                    probe_art = self.adapter.probe_arm_model(arm, model)
                    art_digest = (probe_art.get("digest") if isinstance(probe_art, dict)
                                  else probe_art)
                    declared = self.adapter.expected_arm_structure(arm)
                    if declared is not None and art_digest != declared:
                        detail = ""
                        explain = getattr(self.adapter, "explain_structure_mismatch", None)
                        if explain is not None:
                            detail = "\n  " + "\n  ".join(explain(arm, probe_art))
                        raise ComparisonExecutionError(
                            f"{block_id}/{arm}: the persisted artifact realizes architecture "
                            f"{art_digest}, not the declared {declared}. The object about to "
                            f"be evaluated is not the architecture the plan declares.{detail}")
                    rec["probe_artifact_digest"] = art_digest
                    accuracy = float(self.adapter.evaluate_terminal(arm, model, sealed_test))
                    # JOURNAL BEFORE THE PERMIT IS SPENT. A crash from here
                    # on can no longer lose the measurement.
                    journal.write(state=EVAL_MEASURED, accuracy=accuracy,
                                  sealed_test_hash=test_rec["hash"],
                                  terminal_model_hash=tm["hash"], arm_id=arm,
                                  block_id=block_id, session_id=self.session_id)
                    rec["evaluation_adopted_from_journal"] = False
                    self._boundary("journal_measured")
                else:
                    accuracy = float(existing["accuracy"])
                    rec["evaluation_adopted_from_journal"] = True
                    rec["evaluation_journal_state_on_resume"] = existing["state"]
                    self.state.setdefault("adopted_evaluations", []).append(
                        {"block_id": block_id, "arm": arm, "accuracy": accuracy,
                         "journal_state": existing["state"], "session_id": self.session_id,
                         "note": ("a previous session measured this arm; the value was "
                                  "adopted from the journal and inference was NOT repeated")})

                # Spend the permit, unless a previous session already did.
                # Whether it did is decided by the firewall's own durable
                # record, not by our bookkeeping.
                try:
                    firewall.consume_confirmatory_evaluation(
                        condition_id=arm, replicate_id=block_id,
                        observed_dataset_hash=test_rec["hash"])
                    save_firewall_atomically(firewall, self.firewall_path)
                    rec["firewall_consumed_this_session"] = True
                    self._boundary("firewall_consumed")
                except Exception as exc:                        # noqa: BLE001
                    # An exception is NOT evidence that the permit was
                    # already spent. Verify the persisted firewall for
                    # this exact arm, block and sealed-set hash, and
                    # distinguish a CONFIRMED duplicate from every other
                    # failure. Anything unconfirmed fails closed.
                    confirmed, why = firewall_confirms_consumption(
                        self.firewall_path, condition_id=arm, replicate_id=block_id,
                        sealed_hash=test_rec["hash"])
                    rec["firewall_verification"] = {
                        "exception": f"{type(exc).__name__}: {str(exc)[:400]}",
                        "confirmed_prior_consumption": confirmed, "basis": why}
                    if not confirmed:
                        rec.update(status="FAILED", failure={
                            "stage": "evaluate",
                            "reason": "firewall_error_not_a_confirmed_duplicate",
                            "detail": (
                                f"{type(exc).__name__}: {exc}. Verification of the PERSISTED "
                                f"firewall did not confirm a prior consumption for "
                                f"{arm}/{block_id} against this sealed test set: {why}. "
                                "Refusing to record an evaluation whose permit cannot be "
                                "accounted for."),
                            "technical": False})
                        self._save_state()
                        continue
                    if existing is None:
                        rec.update(status="FAILED", failure={
                            "stage": "evaluate", "reason": "permit_spent_without_a_journal",
                            "detail": (
                                f"A prior consumption is CONFIRMED ({why}) but no evaluation "
                                "journal exists, so the measurement cannot be recovered. This "
                                "block is not replaceable: a sealed-test evaluation has "
                                "occurred."),
                            "technical": False})
                        self._save_state()
                        continue
                    rec["firewall_consumed_this_session"] = False
                    rec["firewall_note"] = (
                        f"a prior consumption is CONFIRMED against the persisted firewall "
                        f"({why}); the journal state was {existing['state']} and consumption "
                        "was not repeated")

                journal.advance_at_least(EVAL_CONSUMED)
                rec.update(status="EVALUATED", sealed_test_accuracy=accuracy,
                           sealed_test_hash=test_rec["hash"],
                           evaluated_at_utc=utc_timestamp())
                self._save_state()
                self._boundary("eval_recorded_in_state")
                journal.advance_at_least(EVAL_RECORDED)
        return self._block_outcome(block_id)

    def _completed_epochs(self, block_id: str, arm: str) -> int:
        """
        Committed epochs for this arm, from the COMMIT MANIFEST.

        Deliberately not from the epoch log. The log is a derived record
        and can lag the manifest by one epoch or end in a torn line after
        a hard kill; an earlier version read the log here and therefore
        abandoned arms that had 50 good committed epochs and one
        half-written log line. Genuine corruption still raises, and is
        reported rather than treated as "no progress".
        """
        arm_dir = self.root / "blocks" / block_id / arm
        try:
            manifest = CommitManifest.load(arm_dir / "commit.json")
        except DurableStateError as exc:
            self.state.setdefault("corrupt_arm_state", []).append(
                {"block_id": block_id, "arm": arm, "detail": str(exc)[:400],
                 "session_id": self.session_id})
            return 0
        if manifest is None:
            return 0
        return manifest.committed_epoch + 1

    def _arm_needs_training(self, block_id: str, arm: str, blk: dict) -> bool:
        """
        Whether this arm must be entered for training in this session.

        A TRAINED or EVALUATED arm is done. An arm that has never
        started needs training. The case that matters is an arm recorded
        FAILED: if its epoch log carries completed epochs and its
        failure was technical, it is RESUMABLE, and skipping it would
        throw away up to 200 epochs of finished work and invalidate the
        block - which is precisely the loss this recovery mechanism
        exists to prevent. A FAILED arm with no completed epochs is a
        genuine failure (construction, out-of-memory at the first step)
        and is left failed, so the block is invalidated and may be
        replaced under the frozen rule.
        """
        rec = blk.get("arms", {}).get(arm, {})
        status = rec.get("status")
        if status in ("TRAINED", "EVALUATED"):
            return False
        if status != "FAILED":
            return True
        if not (rec.get("failure") or {}).get("technical"):
            return False
        done = self._completed_epochs(block_id, arm)
        if done <= 0:
            return False
        self.state.setdefault("resumed_failed_arms", []).append({
            "block_id": block_id, "arm": arm, "completed_epochs": done,
            "previous_failure": rec.get("failure"), "session_id": self.session_id,
            "noted_at_utc": utc_timestamp()})
        return True

    def _block_outcome(self, block_id: str) -> ComparisonBlockOutcome:
        blk = self.state["blocks"].get(block_id, {})
        index = blk.get("block_index", 0)
        ds = blk.get("dataset", {})
        outcome = ComparisonBlockOutcome(
            block_id=block_id, arm_ids=self.arm_ids,
            dataset_hash=ds.get("train", {}).get("hash", ""),
            seeds={a: self.plan.seeds.seed_for(index, a) for a in self.arm_ids},
            replaces_block_id=blk.get("replaces_block_id"),
        )
        for arm in self.arm_ids:
            rec = blk.get("arms", {}).get(arm)
            if rec is None:
                outcome.record_failure(arm=arm, reason="not_run", detail="")
                continue
            if rec["status"] == "EVALUATED":
                outcome.values[arm] = rec["sealed_test_accuracy"]
                outcome.secondary[arm] = rec.get("secondary", {})
            else:
                f = rec.get("failure", {"reason": rec["status"], "detail": ""})
                outcome.record_failure(arm=arm, reason=f.get("reason", "unknown"),
                                       detail=f.get("detail", ""), seed=rec.get("seed"))
        return outcome

    # ------------------------------------------------------------------
    # replacement policy
    # ------------------------------------------------------------------
    def _is_replaceable(self, block_id: str) -> tuple[bool, str]:
        """
        A block may be replaced only if it failed technically BEFORE any
        arm reached sealed-test evaluation. A block with an evaluated arm
        has an endpoint value in hand, so replacing it could depend on
        that value; it is therefore never replaceable, regardless of what
        happened to the other arm.
        """
        blk = self.state["blocks"].get(block_id, {})
        arms = blk.get("arms", {})
        if any(r.get("status") == "EVALUATED" for r in arms.values()):
            return False, ("a sealed-test evaluation was already consumed for this block; "
                           "replacing it could be outcome-dependent")
        failures = [r.get("failure", {}) for r in arms.values() if r.get("status") == "FAILED"]
        if not failures:
            return False, "the block did not fail"
        if not all(f.get("technical") for f in failures):
            return False, "at least one failure is not classified as technical"
        return True, "technical failure before any evaluation"

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def check_frozen(self, frozen: dict[str, Any]) -> list[str]:
        """
        Bind this execution to the REVIEWED configuration AND the
        REVIEWED SOURCE CODE.

        A configuration fingerprint alone is not enough: every design
        parameter can be identical while the adapter, the runner, the
        analysis or the reference loader has changed since review, and
        the run would then execute unreviewed code under a reviewed
        plan. The freeze therefore records the whole source manifest,
        and this check names each file whose hash has moved rather than
        reporting one unhelpful aggregate mismatch.

        Returns problems; empty means this execution is the reviewed one.
        """
        problems: list[str] = []
        prov = self.runtime_provenance()

        if frozen.get("config_fingerprint") != prov["config_fingerprint"]:
            problems.append(
                f"configuration changed since the freeze: frozen "
                f"{frozen.get('config_fingerprint')}, current {prov['config_fingerprint']}.")
        if frozen.get("preregistration_sha256") != self.plan.preregistration_hash():
            problems.append(
                f"preregistration changed since the freeze: frozen "
                f"{frozen.get('preregistration_sha256')}, current "
                f"{self.plan.preregistration_hash()}.")

        frozen_sources = {rec["path"]: rec["sha256"]
                          for rec in (frozen.get("source_manifest") or {}).values()}
        current_sources = {rec["path"]: rec["sha256"]
                           for rec in prov["source_manifest"].values()}
        for path in sorted(set(frozen_sources) | set(current_sources)):
            before, after = frozen_sources.get(path), current_sources.get(path)
            if before is None:
                problems.append(f"source file ADDED since the freeze: {path} ({after}).")
            elif after is None:
                problems.append(f"source file REMOVED since the freeze: {path}.")
            elif before != after:
                problems.append(
                    f"source file CHANGED since the freeze: {path}\n"
                    f"      frozen  {before}\n      current {after}")

        declared = frozen.get("declared_architecture_digest")
        if declared is not None:
            live = None
            try:
                live = self.adapter.expected_arm_structure(self.arm_ids[0])
            except Exception as exc:                            # noqa: BLE001
                problems.append(f"declared architecture could not be read: {exc}")
            if live is not None and live != declared:
                problems.append(
                    f"declared architecture changed since the freeze: frozen {declared}, "
                    f"current {live}. The architecture both arms are checked against is not "
                    "the one that was reviewed.")
        if problems:
            problems.append(
                "A change after review requires a NEW FREEZE: re-run --freeze, review the "
                "diff, and launch against the new frozen record. Production is deliberately "
                "not permitted to proceed on unreviewed code.")
        return problems

    def run(self, *, execute: bool = False, allow_production: bool = False,
            frozen_hash: Optional[str] = None,
            frozen: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        execute=False -> validation only; touches nothing on disk.
        A PRODUCTION plan additionally requires allow_production=True.
        """
        problems = (self.plan.validate_production(frozen_hash=frozen_hash)
                    if self.plan.execution_mode is ExecutionMode.PRODUCTION
                    else self.plan.validate())
        if problems:
            raise ComparisonExecutionError(f"Plan validation failed: {problems}")
        dir_problems = check_output_dir(self.plan.output_dir)
        if dir_problems:
            raise ComparisonExecutionError(f"Output directory refused: {dir_problems}")
        if not execute:
            return {"executed": False, "validation": "CLEAN",
                    "experiment_id": self.plan.experiment_id,
                    "config_fingerprint": self.plan.config_fingerprint(),
                    "preregistration_sha256": self.plan.preregistration_hash(),
                    "source_manifest_sha256":
                        self.runtime_provenance()["source_manifest_sha256"],
                    "experiment_fingerprint":
                        self.runtime_provenance()["experiment_fingerprint"]}
        if (self.interrupt_after_epoch is not None
                and self.plan.execution_mode is ExecutionMode.PRODUCTION):
            raise ComparisonExecutionError(
                "interrupt_after_epoch is set on a PRODUCTION plan. Fault injection exists "
                "only for the interruption test and must never run against evidence.")
        if self.boundary_hook is not None and self.plan.execution_mode is ExecutionMode.PRODUCTION:
            raise ComparisonExecutionError(
                "boundary_hook is set on a PRODUCTION plan. The persistence-boundary hook "
                "exists only for process-kill tests and must never run against evidence.")
        if self.plan.execution_mode is ExecutionMode.PRODUCTION and not allow_production:
            raise ComparisonExecutionError(
                "Refusing to execute a PRODUCTION comparison plan without "
                "allow_production=True.")

        if frozen is not None:
            frozen_problems = self.check_frozen(frozen)
            if frozen_problems:
                raise ComparisonExecutionError(
                    "Refusing to execute: this is not the reviewed experiment.\n  "
                    + "\n  ".join(frozen_problems))
        elif (self.plan.execution_mode is ExecutionMode.PRODUCTION and execute):
            raise ComparisonExecutionError(
                "Refusing to execute a PRODUCTION plan without the frozen record. Pass the "
                "frozen manifest written by --freeze so that the configuration AND the source "
                "code can be bound to what was reviewed.")

        firewall = self._preregister_or_resume()
        test_rec, sealed_test = self._sealed_test_set(firewall)
        design = self.plan.design

        outcomes: list[ComparisonBlockOutcome] = []
        for i in range(design.n_blocks):
            outcomes.append(self._run_block(f"block{i}", i, firewall, test_rec, sealed_test))

        # Replacement, under the frozen budget and the frozen rule.
        next_index = design.n_blocks
        while True:
            valid = sum(1 for o in outcomes if o.is_valid)
            used = len(self.state["replacements"])
            # Replace toward the PREREGISTERED block count, not merely to the
            # minimum. Stopping at min_valid_blocks would accept a design with
            # less power than the one that was preregistered, while leaving an
            # unused replacement budget that exists for exactly this purpose.
            if valid >= design.n_blocks or used >= design.max_replacement_blocks:
                break
            candidate = next(
                (o.block_id for o in outcomes
                 if not o.is_valid and self._is_replaceable(o.block_id)[0]
                 and o.block_id not in {r["replaced"] for r in self.state["replacements"]}),
                None)
            if candidate is None:
                break
            new_id = f"block{next_index}"
            self.state["replacements"].append({
                "replaced": candidate, "replacement": new_id,
                "reason": self._is_replaceable(candidate)[1],
                "budget_used": used + 1, "budget_total": design.max_replacement_blocks,
                "decided_at_utc": utc_timestamp()})
            self._save_state()
            outcomes.append(self._run_block(new_id, next_index, firewall, test_rec,
                                            sealed_test, replaces=candidate))
            next_index += 1

        diffs = collect_valid_differences(outcomes)
        analysis = analyse_ii4(
            diffs, n_blocks_requested=design.n_blocks,
            min_valid_blocks=design.min_valid_blocks,
            delta=design.delta_materiality, alpha=design.alpha)

        interpretation = None
        if self.plan.axis.kind is AxisKind.IMPLEMENTATION:
            interpretation = apply_implementation_interpretation_rule(
                IMPLEMENTATION_INTERPRETATION_RULE,
                analysis.materiality.value, analysis.equivalence.value)

        prov = self.runtime_provenance()
        report = {
            "artifact": RUNNER_VERSION,
            "experiment_id": self.plan.experiment_id,
            "execution_mode": self.plan.execution_mode.value,
            "non_evidentiary": self.plan.execution_mode is not ExecutionMode.PRODUCTION,
            "preregistration_sha256": self.state["preregistration_sha256"],
            "frozen_record_verified": frozen is not None,
            "frozen_record": ({k: v for k, v in frozen.items() if k != "source_manifest"}
                              if frozen else None),
            "config_fingerprint": self.state["config_fingerprint"],
            "experiment_fingerprint": self.state["experiment_fingerprint"],
            "source_manifest": prov["source_manifest"],
            "source_manifest_sha256": prov["source_manifest_sha256"],
            "software_stack": prov["software_stack"],
            "session_environments": self.state.get("session_environments", []),
            "recorded_deviations": self.state.get("recorded_deviations", []),
            "comparison_axis": self.plan.axis.to_dict(),
            "shared_components": dict(self.plan.shared_components),
            "sealed_test_set": test_rec,
            "blocks": [o.to_dict() for o in outcomes],
            "block_order": list(self.state["block_order"]),
            "replacements": list(self.state["replacements"]),
            "n_blocks_valid": len(diffs),
            "interrupted_arms": [
                {"block_id": bid, "arm": arm,
                 "resumed_from_epoch": rec.get("resumed_from_epoch"),
                 "interruptions": (rec.get("recovery") or {}).get("interruptions", [])}
                for bid, blk in sorted(self.state["blocks"].items())
                for arm, rec in sorted(blk.get("arms", {}).items())
                if rec.get("was_interrupted")],
            "recovery_exactness": RECOVERY_EXACTNESS,
            "sigkill_test_scope": WHAT_SIGKILL_ESTABLISHES,
            "degraded_recoveries": [
                {"block_id": bid, "arm": arm,
                 "decisions": [d for d in (rec.get("recovery") or {}).get(
                     "recovery_decisions", []) if d.get("degraded")]}
                for bid, blk in sorted(self.state["blocks"].items())
                for arm, rec in sorted(blk.get("arms", {}).items())
                if (rec.get("recovery") or {}).get("was_degraded")],
            "adopted_evaluations": list(self.state.get("adopted_evaluations", [])),
            "corrupt_arm_state": list(self.state.get("corrupt_arm_state", [])),
            "epoch_logs": {
                f"{bid}/{arm}": rec.get("epoch_log")
                for bid, blk in sorted(self.state["blocks"].items())
                for arm, rec in sorted(blk.get("arms", {}).items())
                if rec.get("epoch_log")},
            "analysis": analysis.to_dict(),
            "interpretation": interpretation,
            "firewall": firewall.to_dict(),
            "generated_at_utc": utc_timestamp(),
        }
        self._write_json("analysis.json", report)
        self.state["completed_at_utc"] = utc_timestamp()
        self._save_state()
        return report
