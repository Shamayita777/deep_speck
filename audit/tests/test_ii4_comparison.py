#!/usr/bin/env python3
"""
Tests for the generic II-4 comparison machinery.

Three things are tested, in increasing order of what they protect:

  1. THE ABSTRACTION. That a ComparisonAxis cannot be constructed in a
     state that would make its experiment uninterpretable, and that the
     axis kind - not the runner, not a reviewer - decides whether a
     completed run can satisfy methodology Stage II-4.

  2. THE GENERALITY CLAIM. That the generic axis reproduces the frozen
     conformance experiment's arm semantics exactly. The frozen
     experiment is never re-run; its own frozen ConformanceFactor is
     reconstructed through the generic abstraction and compared. If the
     generic machinery could not express the experiment the project has
     already completed, the claim that it generalizes would be false.

  3. THE RUNNER'S REFUSALS. Each refusal in the runner is tested by
     constructing the condition it refuses and asserting that it does.
     A guard that has never been executed is a comment.

Everything here runs without TensorFlow, without a GPU and without the
target implementations, using a deterministic fake adapter. That is the
point: the generic engine is testable in seconds, so the only thing a
GPU is needed for is the science.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from audit.common.outcomes import ExecutionMode
from audit.implementation.comparison import (
    ArmSeedManifest,
    AxisKind,
    ComparisonArm,
    ComparisonAxis,
    ComparisonBlockOutcome,
    collect_valid_differences,
    conformance_axis_from_factor,
)
from audit.implementation.ii4_comparison_design import (
    ComparisonDesign,
    ComparisonPlan,
    IMPLEMENTATION_INTERPRETATION_RULE,
    apply_implementation_interpretation_rule,
)
from audit.implementation.arm_recovery import (
    ArmRecoveryRecord,
    EpochLog,
    InjectedInterruption,
    resume_point,
)
from audit.implementation.durable_state import (
    CommittedCheckpointStore,
    atomic_write_json,
)
from audit.implementation.ii4_comparison_runner import (
    ComparisonExecutionError,
    ComparisonRunner,
    check_output_dir,
)
from audit.implementation.ii4_design import GOHR_DEPTH_FACTOR, II4_DESIGN
from audit.implementation.stages import STAGE_TIERS, ComponentKind, StageId


# =====================================================================
# fake adapter - deterministic, no TF, no GPU
# =====================================================================

class _TrainResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeAdapter:
    """
    A two-arm adapter with injectable faults, driving the REAL epoch log
    and recovery record.

    Its "training" is a loop over simulated epochs that appends to an
    `EpochLog` and rewrites an `ArmRecoveryRecord` exactly as the Keras
    callbacks do, so the resume arithmetic, the gap refusal, the
    best-value restoration and the STARTED-before-training ordering are
    all exercised here - in milliseconds, without TensorFlow. The real
    adapter is then tested separately against real Keras.

    `accuracy_fn(arm, block_id, seed)` fully determines the endpoint, so
    every statistical outcome in these tests is exact, not sampled.
    """

    EPOCHS = 3

    def __init__(self, *, arm_ids, structures=None, fail_train=(), fail_build=(),
                 accuracy_fn=None, bad_save=(), declared=True, fail_at_epoch=None):
        self.arm_ids = arm_ids
        self.structures = structures or {
            a: {"signature_version": "test", "layer_count": 3,
                "layers": [{"class_name": "Dense", "units": 4, "weights": []}],
                "input_shape": [None, 8], "output_shape": [None, 1],
                "total_parameters": 102497, "trainable_parameters": 100897,
                "non_trainable_parameters": 1600}
            for a in arm_ids}
        for a in arm_ids:                       # give each its own digest
            self.structures[a] = dict(self.structures[a])
            self.structures[a]["digest"] = _sig_digest(self.structures[a])
        self.fail_train = set(fail_train)       # (block_id, arm)
        self.fail_build = set(fail_build)
        self.bad_save = set(bad_save)
        self.declared = declared
        self.fail_at_epoch = dict(fail_at_epoch or {})   # (block,arm) -> epoch
        self.accuracy_fn = accuracy_fn or (
            lambda arm, block_id, seed: 0.9290 + (0.0001 if arm == arm_ids[0] else 0.0)
            + 0.00005 * (hash((block_id, arm)) % 7))
        self.calls: list[tuple] = []

    # --- provenance ---------------------------------------------------
    def provenance_sources(self):
        return {"fake_adapter": "audit/implementation/ii4_analysis.py"}

    def environment_details(self):
        return {"backend": "fake", "gpus": []}

    def shared_components(self):
        return {"data_generation": "shared fake generator; basis: test design"}

    # --- data ---------------------------------------------------------
    def generate_sealed_test_set(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        p = directory / "sealed.json"
        p.write_text(json.dumps({"n": 1000, "kind": "sealed"}))
        import hashlib
        return str(p), hashlib.sha256(p.read_bytes()).hexdigest(), 1000

    def load_sealed_test_set(self, path, expected_hash):
        import hashlib
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected_hash:
            raise RuntimeError(f"sealed hash mismatch: {actual} != {expected_hash}")
        return {"n": 1000}

    def generate_block_datasets(self, block_id, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        p = directory / "train.json"
        p.write_text(json.dumps({"block": block_id}))
        import hashlib
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        return {"train": {"path": str(p), "hash": h, "n": 100},
                "validation": {"path": str(p), "hash": h, "n": 10}}

    def load_block_datasets(self, records):
        return {"block": json.loads(Path(records["train"]["path"]).read_text())["block"]}

    # --- per-arm ------------------------------------------------------
    def build_arm_model(self, arm_id, *, seed):
        self.calls.append(("build", arm_id, seed))
        if arm_id in self.fail_build:
            raise RuntimeError(f"injected construction fault in {arm_id}")
        return {"arm": arm_id, "seed": seed, "signature": self.structures[arm_id],
                "epochs_done": 0}

    def probe_arm_model(self, arm_id, model):
        if isinstance(model, dict) and "signature" in model:
            return dict(model["signature"])
        return dict(self.structures[arm_id])

    def expected_arm_structure(self, arm_id):
        return self.structures[arm_id]["digest"] if self.declared else None

    def explain_structure_mismatch(self, arm_id, observed):
        return [f"fake adapter: observed digest {observed.get('digest')}"]

    def train_arm(self, arm_id, model, block_data, *, arm_dir, seed, session_id="s0",
                  interrupt_after_epoch=None):
        """Simulated epoch loop over the REAL two-slot store and epoch log."""
        block_id = block_data["block"]
        self.calls.append(("train", arm_id, block_id, seed))
        arm_dir = Path(arm_dir); arm_dir.mkdir(parents=True, exist_ok=True)
        log = EpochLog(arm_dir / "epochs.jsonl")
        record_path = arm_dir / "recovery.json"
        declared = self.expected_arm_structure(arm_id)

        store = CommittedCheckpointStore(
            arm_dir=arm_dir, arm_id=arm_id, block_id=block_id, seed=seed,
            epochs_total=self.EPOCHS, suffix=".json", architecture_digest=declared)

        record = ArmRecoveryRecord.load(record_path) or ArmRecoveryRecord(
            arm_id=arm_id, block_id=block_id, seed=seed, epochs_total=self.EPOCHS,
            architecture_digest=declared)
        if record.status == "NOT_STARTED":
            record.status = "STARTED"
        record.sessions.append({"session_id": session_id})
        record.save(record_path)

        if (block_id, arm_id) in self.fail_train:
            record.note_interruption(reason="injected", detail="injected training fault",
                                     at_epoch=0, session_id=session_id)
            record.save(record_path)
            raise RuntimeError(f"injected training fault in {block_id}/{arm_id}")

        resume_from, decision, notes = resume_point(store, log, record=record)
        if resume_from > 0:
            loaded, _ = store.load_committed(lambda p: json.loads(Path(p).read_text()))
            model = {**model, **loaded}
            record.note_interruption(reason="resumed", detail=f"at epoch {resume_from}",
                                     at_epoch=resume_from, session_id=session_id)
        record.save(record_path)
        self.calls.append(("resume_from", arm_id, block_id, resume_from))

        crash_at = self.fail_at_epoch.get((block_id, arm_id), interrupt_after_epoch)
        for epoch in range(resume_from, self.EPOCHS):
            model = {**model, "epochs_done": epoch + 1}
            val_loss = 0.24 - 0.01 * epoch
            best = min([v for v in (record.best_monitor_value, val_loss) if v is not None])
            # COMMIT first (alternate slot -> fsync -> verify -> manifest),
            # then the derived log, exactly as the real callbacks do.
            store.commit_epoch(
                epoch=epoch,
                save_fn=lambda p, m=model: Path(p).write_text(json.dumps(m, sort_keys=True)),
                best_monitor_value=best, best_checkpoint_epoch=epoch,
                session_id=session_id)
            log.append({"epoch": epoch, "status": "COMPLETED",
                        "loss": 0.25 - 0.01 * epoch, "accuracy": 0.5 + 0.01 * epoch,
                        "val_loss": val_loss,
                        "val_accuracy": self.accuracy_fn(arm_id, block_id, seed) - 0.001 * (
                            self.EPOCHS - 1 - epoch),
                        "learning_rate": 0.002 - 0.0001 * (epoch % 10),
                        "epoch_seconds": 0.001, "checkpoint_seconds": 0.0005,
                        "session_id": session_id,
                        "resumed_at_this_epoch": epoch == resume_from and epoch != 0})
            record.best_monitor_value = best
            record.epochs_completed = epoch + 1
            record.status = "TRAINING"
            record.save(record_path)
            if crash_at is not None and epoch == crash_at:
                record.note_interruption(reason="InjectedInterruption",
                                         detail=f"after epoch {epoch}",
                                         at_epoch=epoch + 1, session_id=session_id)
                record.save(record_path)
                raise InjectedInterruption(f"injected interruption after epoch {epoch}")

        manifest = store.manifest()
        summary = log.summary(expected_total=self.EPOCHS,
                              committed_epoch=manifest.committed_epoch if manifest else None)
        record.status = "TRAINED"; record.save(record_path)
        vals = [r["val_accuracy"] for r in log.read()]
        return _TrainResult(
            n_epochs_completed=(manifest.committed_epoch + 1) if manifest else 0,
            final_val_acc=vals[-1], max_val_acc=max(vals),
            checkpoint_hash="be" * 32, wall_clock_seconds=0.01, failure_reason=None,
            model=model, epoch_log=summary, recovery=record.to_dict(),
            resumed_from_epoch=resume_from, was_interrupted=record.was_interrupted)

    def save_terminal_model(self, arm_id, model, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(model, sort_keys=True)
        path.write_text(payload)
        import hashlib
        return {"path": str(path), "hash": hashlib.sha256(payload.encode()).hexdigest(),
                "reload_weight_identical": arm_id not in self.bad_save,
                "architecture_digest": self.structures[arm_id]["digest"],
                "architecture_matches_declared": True, "architecture_mismatch": None}

    def load_terminal_model(self, arm_id, path, expected_hash):
        import hashlib
        data = Path(path).read_text()
        if hashlib.sha256(data.encode()).hexdigest() != expected_hash:
            raise RuntimeError("terminal model hash mismatch")
        return json.loads(data)

    def evaluate_terminal(self, arm_id, model, sealed_test):
        return self.accuracy_fn(arm_id, "eval", model["seed"])


def _sig_digest(sig):
    from audit.implementation.architecture_signature import signature_digest
    return signature_digest(sig)


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =====================================================================
# fixtures
# =====================================================================

ARM_A, ARM_B = "reference", "independent"


def impl_axis(**kw) -> ComparisonAxis:
    base = dict(
        kind=AxisKind.IMPLEMENTATION, name="implementation",
        arm_a=ComparisonArm(arm_id=ARM_A, label="reference implementation",
                            identity={"impl": "reference"},
                            source_paths=("a/ref_model.py", "a/ref_train.py")),
        arm_b=ComparisonArm(arm_id=ARM_B, label="independent implementation",
                            identity={"impl": "independent"},
                            source_paths=("b/ind_model.py", "b/ind_train.py")),
        scientific_question="Do the two implementations agree under identical protocols?",
        verification_method="source hashes pinned; deterministic components verified in II-2",
    )
    base.update(kw)
    return ComparisonAxis(**base)


def design(**kw) -> ComparisonDesign:
    base = dict(
        n_blocks=4, min_valid_blocks=3, max_replacement_blocks=1,
        primary_endpoint="sealed_confirmatory_test_accuracy",
        evaluation_rule="terminal_epoch", epochs=FakeAdapter.EPOCHS,
        same_dataset=True, same_model_initialization=False,
        same_training_shuffle_stream=False, same_evaluation_data=True,
        alpha=0.05, delta_materiality=0.01, delta_is_equivalence_margin=True)
    base.update(kw)
    return ComparisonDesign(**base)


def plan(tmp, *, d=None, axis=None, mode=ExecutionMode.SMOKE, **kw) -> ComparisonPlan:
    d = d or design()
    axis = axis or impl_axis()
    base = dict(
        experiment_id="TEST-COMPARISON-SMOKE", design=d, axis=axis,
        fixed_protocol={"epochs": d.epochs, "shuffle": True, "batch_size": 10},
        seeds=ArmSeedManifest(base_seed=900_000, n_blocks=d.max_total_blocks,
                              arm_ids=axis.arm_ids),
        output_dir=Path(tmp) / "run",
        shared_components={"data_generation": "shared; justified by II-2 deterministic checks"},
        execution_mode=mode)
    base.update(kw)
    return ComparisonPlan(**base)


# =====================================================================
# 1. the abstraction
# =====================================================================

class TestComparisonAxis(unittest.TestCase):

    def test_implementation_axis_satisfies_stage_ii4(self):
        axis = impl_axis()
        self.assertTrue(axis.satisfies_stage_ii4)
        self.assertEqual(axis.component_kind,
                         ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON.value)

    def test_conformance_axis_cannot_satisfy_stage_ii4(self):
        axis = conformance_axis_from_factor(GOHR_DEPTH_FACTOR)
        self.assertFalse(axis.satisfies_stage_ii4)
        self.assertEqual(axis.component_kind,
                         ComponentKind.CONFORMANCE_IMPACT_EXPERIMENT.value)
        self.assertIn("never satisfy Stage II-4", axis.stage_consequence())

    def test_implementation_arms_must_be_pinned_by_source(self):
        with self.assertRaisesRegex(ValueError, "declares no source_paths"):
            impl_axis(arm_b=ComparisonArm(arm_id=ARM_B, label="independent",
                                          identity={"impl": "independent"}))

    def test_implementation_arms_may_not_share_source_files(self):
        """Arms that execute the same code are not independent implementations."""
        with self.assertRaisesRegex(ValueError, "share source file"):
            impl_axis(arm_b=ComparisonArm(
                arm_id=ARM_B, label="independent", identity={"impl": "independent"},
                source_paths=("a/ref_model.py", "b/ind_train.py")))

    def test_axis_requires_a_stated_question_and_verification(self):
        with self.assertRaisesRegex(ValueError, "scientific_question"):
            impl_axis(scientific_question="   ")
        with self.assertRaisesRegex(ValueError, "verification_method"):
            impl_axis(verification_method="")

    def test_sign_convention_is_explicit_and_matches_difference(self):
        axis = impl_axis()
        self.assertEqual(axis.sign_convention, f"{ARM_A} minus {ARM_B}")
        blk = ComparisonBlockOutcome(block_id="block0", arm_ids=axis.arm_ids,
                                     dataset_hash="h", values={ARM_A: 0.93, ARM_B: 0.92})
        self.assertAlmostEqual(blk.signed_difference, 0.01, places=12)


class TestGeneralityAgainstTheFrozenExperiment(unittest.TestCase):
    """
    The frozen conformance experiment is NOT re-run. Its own frozen
    factor is expressed through the generic axis and checked, which is
    what makes "the engine generalizes" a tested claim rather than a
    docstring.
    """

    def test_generic_axis_reproduces_the_frozen_factor_arms(self):
        axis = conformance_axis_from_factor(GOHR_DEPTH_FACTOR)
        self.assertEqual(axis.arm_ids, ("declared", "realized"))
        self.assertEqual(axis.arm_identities,
                         {"declared": GOHR_DEPTH_FACTOR.declared_value,
                          "realized": GOHR_DEPTH_FACTOR.realized_value})
        self.assertEqual(axis.verification_method, GOHR_DEPTH_FACTOR.verification_method)

    def test_generic_seed_policy_matches_the_frozen_seed_policy(self):
        """
        The frozen manifest derives seeds as base + 2*block + armIndex with
        declared=0, realized=1. The generic manifest must agree exactly, or
        a re-expressed experiment would use different seeds.
        """
        from audit.implementation.ii4_experiment import II4SeedManifest
        frozen = II4SeedManifest(base_seed=700_000, n_blocks=10)
        generic = ArmSeedManifest(base_seed=700_000, n_blocks=10,
                                  arm_ids=("declared", "realized"))
        for i in range(10):
            for arm in ("declared", "realized"):
                self.assertEqual(generic.seed_for(i, arm), frozen.seed_for(i, arm),
                                 f"seed disagreement at block{i}/{arm}")

    def test_generic_design_agrees_field_for_field_with_the_frozen_design(self):
        """Same quantities, same meanings - parity asserted, not described."""
        g = design(n_blocks=II4_DESIGN.n_blocks,
                   min_valid_blocks=II4_DESIGN.min_valid_blocks,
                   max_replacement_blocks=0, epochs=II4_DESIGN.epochs).to_dict()
        f = II4_DESIGN.to_dict()
        for key in ("n_blocks", "min_valid_blocks", "primary_endpoint", "evaluation_rule",
                    "epochs", "pairing", "alpha", "delta_materiality",
                    "delta_is_equivalence_margin", "confirmatory_test_set",
                    "confirmatory_data_mode"):
            self.assertEqual(g[key], f[key], f"design field {key} diverges from the frozen design")

    def test_stage_registry_still_marks_ii2_core(self):
        """The axis machinery is meaningless if II-2 slips back to ENHANCED."""
        from audit.common.outcomes import AssessmentTier
        self.assertIs(STAGE_TIERS[StageId.II_2_INDEPENDENT_REIMPLEMENTATION],
                      AssessmentTier.CORE)


class TestSeedsAndBlocks(unittest.TestCase):

    def test_seeds_are_pairwise_distinct(self):
        m = ArmSeedManifest(base_seed=1_000, n_blocks=8, arm_ids=(ARM_A, ARM_B))
        self.assertEqual(m.problems(), [])
        self.assertEqual(len({m.seed_for(i, a) for i in range(8) for a in (ARM_A, ARM_B)}), 16)

    def test_unknown_arm_and_out_of_range_block_are_refused(self):
        m = ArmSeedManifest(base_seed=1, n_blocks=2, arm_ids=(ARM_A, ARM_B))
        with self.assertRaises(ValueError):
            m.seed_for(0, "nonexistent")
        with self.assertRaises(ValueError):
            m.seed_for(5, ARM_A)

    def test_any_arm_failure_invalidates_the_whole_block(self):
        blk = ComparisonBlockOutcome(block_id="b", arm_ids=(ARM_A, ARM_B), dataset_hash="h",
                                     values={ARM_A: 0.93})
        blk.record_failure(arm=ARM_B, reason="crash", detail="OOM")
        self.assertFalse(blk.is_valid)
        self.assertIsNone(blk.signed_difference)
        self.assertEqual(collect_valid_differences([blk]), [])


class TestDesignAndInterpretation(unittest.TestCase):

    def test_paired_initialization_cannot_be_declared(self):
        with self.assertRaisesRegex(ValueError, "same_model_initialization"):
            design(same_model_initialization=True)

    def test_sign_flip_feasibility_is_stated_at_min_valid_not_n_blocks(self):
        self.assertTrue(design(n_blocks=7, min_valid_blocks=6).sign_flip_feasible_at_min_valid)
        self.assertFalse(design(n_blocks=7, min_valid_blocks=5).sign_flip_feasible_at_min_valid)

    def test_production_refuses_a_design_whose_secondary_cannot_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp, d=design(n_blocks=5, min_valid_blocks=4, max_replacement_blocks=0),
                     mode=ExecutionMode.PRODUCTION, experiment_id="TEST-PROD")
            self.assertTrue(any("sign-flip" in x for x in p.validate_production()))

    def test_equivalence_requires_the_conjunction(self):
        eq = apply_implementation_interpretation_rule(
            IMPLEMENTATION_INTERPRETATION_RULE, "NO_DIRECTIONAL_CLAIM",
            "EQUIVALENT_WITHIN_MARGIN")
        self.assertEqual(eq["conclusion"], "IMPLEMENTATIONS_EQUIVALENT")
        for materiality, equivalence in (("NO_DIRECTIONAL_CLAIM", "INCONCLUSIVE"),
                                         ("INSUFFICIENT_BLOCKS", "NOT_ASSESSED"),
                                         ("NO_DIRECTIONAL_CLAIM", "NOT_EQUIVALENT")):
            self.assertEqual(
                apply_implementation_interpretation_rule(
                    IMPLEMENTATION_INTERPRETATION_RULE, materiality, equivalence)["conclusion"],
                "INCONCLUSIVE", f"{materiality}/{equivalence} must not conclude equivalence")

    def test_unmapped_combination_is_inconclusive_and_recorded(self):
        out = apply_implementation_interpretation_rule(
            IMPLEMENTATION_INTERPRETATION_RULE, "MATERIALLY_HIGHER",
            "EQUIVALENT_WITHIN_MARGIN")
        self.assertEqual(out["conclusion"], "INCONCLUSIVE")
        self.assertEqual(out["unmapped_pair"], ["MATERIALLY_HIGHER", "EQUIVALENT_WITHIN_MARGIN"])

    def test_implementation_plan_must_declare_shared_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp, shared_components={})
            self.assertTrue(any("shared_components" in x for x in p.validate()))

    def test_seed_manifest_must_cover_replacement_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = design(n_blocks=4, min_valid_blocks=3, max_replacement_blocks=2)
            axis = impl_axis()
            p = plan(tmp, d=d, axis=axis,
                     seeds=ArmSeedManifest(base_seed=1, n_blocks=4, arm_ids=axis.arm_ids))
            self.assertTrue(any("replacement blocks would have no predetermined seeds" in x
                                for x in p.validate()))

    def test_preregistration_hash_is_stable_and_excludes_wall_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = plan(tmp, created_at_utc="2026-01-01T00:00:00Z")
            b = plan(tmp, created_at_utc="2027-09-09T09:09:09Z")
            self.assertEqual(a.config_fingerprint(), b.config_fingerprint())
            self.assertNotEqual(a.preregistration_hash(), b.preregistration_hash())


# =====================================================================
# 3. the runner's refusals and its happy path
# =====================================================================

class TestRunnerEndToEnd(unittest.TestCase):

    def _run(self, tmp, adapter, p=None, **kw):
        p = p or plan(tmp)
        return ComparisonRunner(plan=p, adapter=adapter).run(execute=True, **kw)

    def test_all_blocks_succeed_and_the_analysis_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = self._run(tmp, FakeAdapter(arm_ids=(ARM_A, ARM_B)))
            self.assertEqual(rep["n_blocks_valid"], 4)
            self.assertEqual(len(rep["blocks"]), 4)
            self.assertTrue(rep["non_evidentiary"])
            a = rep["analysis"]
            for key in ("replication", "warnings"):
                self.assertIn(key, a)
            self.assertIsNotNone(rep["interpretation"])
            self.assertIn(rep["interpretation"]["conclusion"],
                          ("IMPLEMENTATIONS_EQUIVALENT", "IMPLEMENTATIONS_DIFFER",
                           "INCONCLUSIVE"))
            # everything needed for II-4/II-5 is in the artifact already
            for key in ("comparison_axis", "shared_components", "sealed_test_set", "blocks",
                        "analysis", "interpretation", "firewall", "source_manifest",
                        "source_manifest_sha256", "preregistration_sha256",
                        "experiment_fingerprint", "replacements"):
                self.assertIn(key, rep, f"report is missing {key}")
            self.assertTrue((Path(tmp) / "run" / "analysis.json").is_file())
            self.assertTrue((Path(tmp) / "run" / "preregistration.json").is_file())

    def test_one_arm_failure_invalidates_the_block_and_triggers_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            ad = FakeAdapter(arm_ids=(ARM_A, ARM_B), fail_train=[("block1", ARM_B)])
            rep = self._run(tmp, ad)
            self.assertEqual(len(rep["replacements"]), 1)
            self.assertEqual(rep["replacements"][0]["replaced"], "block1")
            self.assertEqual(rep["replacements"][0]["replacement"], "block4")
            self.assertEqual(rep["n_blocks_valid"], 4)
            failed = [b for b in rep["blocks"] if b["block_id"] == "block1"][0]
            self.assertFalse(failed["is_valid"])          # retained, not deleted
            replacement = [b for b in rep["blocks"] if b["block_id"] == "block4"][0]
            self.assertEqual(replacement["replaces_block_id"], "block1")

    def test_replacement_budget_is_not_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ad = FakeAdapter(arm_ids=(ARM_A, ARM_B),
                             fail_train=[("block0", ARM_A), ("block1", ARM_B),
                                         ("block2", ARM_A)])
            rep = self._run(tmp, ad)
            self.assertLessEqual(len(rep["replacements"]), 1)   # budget is 1
            self.assertLess(rep["n_blocks_valid"], 3)
            self.assertEqual(rep["analysis"]["decision"]["materiality"]
                             if "decision" in rep["analysis"] else
                             rep["analysis"].get("materiality", "INSUFFICIENT_BLOCKS"),
                             "INSUFFICIENT_BLOCKS")
            self.assertEqual(rep["interpretation"]["conclusion"], "INCONCLUSIVE")

    def test_an_evaluated_block_is_never_replaceable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp)
            runner = ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B)))
            runner.run(execute=True)
            for bid in ("block0", "block1", "block2", "block3"):
                ok, why = runner._is_replaceable(bid)
                self.assertFalse(ok)
                self.assertIn("already consumed", why)

    def test_divergent_arm_structures_abort_before_training(self):
        """The premise of an implementation comparison is a common architecture."""
        with tempfile.TemporaryDirectory() as tmp:
            ad = FakeAdapter(arm_ids=(ARM_A, ARM_B))
            ad.structures[ARM_B] = {**ad.structures[ARM_B], "total_parameters": 70177}
            ad.structures[ARM_B]["digest"] = _sig_digest(ad.structures[ARM_B])
            with self.assertRaisesRegex(ComparisonExecutionError, "DIFFERENT architectures"):
                self._run(tmp, ad)
            self.assertFalse(any(c[0] == "train" for c in ad.calls),
                             "training started despite a structure mismatch")

    def test_non_identical_reload_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ad = FakeAdapter(arm_ids=(ARM_A, ARM_B), bad_save=[ARM_A])
            with self.assertRaisesRegex(ComparisonExecutionError, "weight-identical"):
                self._run(tmp, ad)

    def test_sealed_test_set_is_sealed_under_every_admissible_block_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = self._run(tmp, FakeAdapter(arm_ids=(ARM_A, ARM_B)))
            sealed = rep["firewall"]["sealed_dataset_hashes"]
            self.assertEqual(set(sealed), {f"block{i}" for i in range(5)})  # 4 + 1 replacement
            self.assertEqual(len(set(sealed.values())), 1)

    def test_each_arm_block_pair_is_evaluated_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = self._run(tmp, FakeAdapter(arm_ids=(ARM_A, ARM_B)))
            # the real firewall records "<condition_id>:<replicate_id>" keys
            consumed = list(rep["firewall"]["consumed_evaluation_keys"])
            self.assertEqual(len(consumed), len(set(consumed)))
            self.assertEqual(len(consumed), 8)             # 4 blocks x 2 arms
            self.assertEqual(set(consumed),
                             {f"{a}:block{b}" for a in (ARM_A, ARM_B) for b in range(4)})
            self.assertTrue(rep["firewall"]["frozen"])
            self.assertEqual(rep["firewall"]["confirmatory_data_mode"], "SHARED")

    def test_resume_refuses_a_changed_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp)
            ad = FakeAdapter(arm_ids=(ARM_A, ARM_B), fail_train=[("block0", ARM_A)])
            ComparisonRunner(plan=p, adapter=ad).run(execute=True)
            changed = plan(tmp, d=design(n_blocks=4, min_valid_blocks=2,
                                         max_replacement_blocks=1))
            with self.assertRaisesRegex(ComparisonExecutionError, "fingerprint has changed"):
                ComparisonRunner(plan=changed,
                                 adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B))).run(execute=True)

    def test_resume_continues_an_unchanged_experiment_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp)
            first = FakeAdapter(arm_ids=(ARM_A, ARM_B))
            ComparisonRunner(plan=p, adapter=first).run(execute=True)
            second = FakeAdapter(arm_ids=(ARM_A, ARM_B))
            rep = ComparisonRunner(plan=plan(tmp), adapter=second).run(execute=True)
            self.assertEqual(rep["n_blocks_valid"], 4)
            self.assertFalse(any(c[0] == "train" for c in second.calls),
                             "a resumed run retrained completed arms")

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp)
            out = ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B))).run(
                execute=False)
            self.assertFalse(out["executed"])
            self.assertEqual(out["validation"], "CLEAN")
            self.assertFalse(Path(p.output_dir).exists())
            for key in ("preregistration_sha256", "config_fingerprint",
                        "source_manifest_sha256", "experiment_fingerprint"):
                self.assertEqual(len(out[key]), 64)

    def test_production_requires_explicit_authorisation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = design(n_blocks=7, min_valid_blocks=6, max_replacement_blocks=2)
            axis = impl_axis()
            p = plan(tmp, d=d, axis=axis, mode=ExecutionMode.PRODUCTION,
                     experiment_id="TEST-PROD-V1",
                     seeds=ArmSeedManifest(base_seed=1, n_blocks=d.max_total_blocks,
                                           arm_ids=axis.arm_ids))
            with self.assertRaisesRegex(ComparisonExecutionError, "allow_production"):
                ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B))).run(
                    execute=True)

    def test_production_refuses_a_plan_that_is_not_the_frozen_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = design(n_blocks=7, min_valid_blocks=6, max_replacement_blocks=2)
            axis = impl_axis()
            p = plan(tmp, d=d, axis=axis, mode=ExecutionMode.PRODUCTION,
                     experiment_id="TEST-PROD-V1",
                     seeds=ArmSeedManifest(base_seed=1, n_blocks=d.max_total_blocks,
                                           arm_ids=axis.arm_ids))
            problems = p.validate_production(frozen_hash="0" * 64)
            self.assertTrue(any("does not match the frozen preregistration hash" in x
                                for x in problems))


class TestOutputDirectoryProtection(unittest.TestCase):

    def test_protected_components_are_refused(self):
        self.assertTrue(check_output_dir(Path("/tmp/x/evidence/run")))
        self.assertTrue(check_output_dir(Path("/tmp/x/frozen/run")))

    def test_a_clean_new_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check_output_dir(Path(tmp) / "fresh_run"), [])


class TestPowerModule(unittest.TestCase):

    def test_refuses_to_power_a_reimplementation_of_the_analysis(self):
        from audit.implementation.ii4_comparison_power import joint_rule_power
        with self.assertRaisesRegex(ValueError, "frozen `analyse` function"):
            joint_rule_power(n_blocks=6, sd=1e-4, delta_margin=0.01, simulations=10)

    def test_zero_failures_does_not_imply_a_zero_failure_rate(self):
        from audit.implementation.ii4_comparison_power import failure_rate_upper_bound
        ucl = failure_rate_upper_bound(observed_successes=21, observed_failures=0)
        self.assertGreater(ucl, 0.10)
        self.assertLess(ucl, 0.15)

    def test_survival_probability_falls_as_the_failure_rate_rises(self):
        from audit.implementation.ii4_comparison_power import valid_block_probability
        p = [valid_block_probability(n_blocks=7, min_valid_blocks=6,
                                     per_arm_failure_probability=r)
             for r in (0.0, 0.02, 0.05, 0.10)]
        self.assertEqual(p[0], 1.0)
        self.assertTrue(all(a > b for a, b in zip(p, p[1:])))

    def test_extra_blocks_improve_survival_at_the_same_minimum(self):
        from audit.implementation.ii4_comparison_power import valid_block_probability
        k7 = valid_block_probability(n_blocks=7, min_valid_blocks=6,
                                     per_arm_failure_probability=0.10)
        k8 = valid_block_probability(n_blocks=8, min_valid_blocks=6,
                                     per_arm_failure_probability=0.10)
        self.assertGreater(k8, k7)




# =====================================================================
# 4. certificate assembly - the stage records the run licenses
# =====================================================================

class TestCertificateAssembly(unittest.TestCase):
    """
    Assembles stage records from runs this suite actually produced, then
    hands them to the project's own `validate_stage_record`. That is the
    gate which refuses an overclaiming record, so a record that passes
    here is one certificate.py will accept and one a reviewer can read.
    """

    def _run_dir(self, tmp, **kw):
        p = plan(tmp, **kw)
        ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=p.axis.arm_ids)).run(execute=True)
        return Path(p.output_dir)

    def test_a_completed_implementation_run_licenses_ii4_completed(self):
        from audit.implementation.ii4_comparison_certificate import build_stage_records
        from audit.implementation.stages import StageOutcome, validate_stage_record
        with tempfile.TemporaryDirectory() as tmp:
            ii4, ii5 = build_stage_records(self._run_dir(tmp), allow_non_evidentiary=True)
            self.assertIs(ii4.status, StageOutcome.COMPLETED)
            self.assertIs(ii5.status, StageOutcome.COMPLETED)
            self.assertEqual(ii4.components[-1].kind,
                             ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON)
            self.assertEqual(validate_stage_record(ii4), [])
            self.assertEqual(validate_stage_record(ii5), [])

    def test_every_component_binds_real_sha256_evidence(self):
        from audit.implementation.ii4_comparison_certificate import build_components
        with tempfile.TemporaryDirectory() as tmp:
            ii4, _ = build_components(self._run_dir(tmp), allow_non_evidentiary=True)
            self.assertTrue(ii4.evidence)
            for path, digest in ii4.evidence.items():
                self.assertEqual(len(digest), 64, f"{path} has no sha256")

    def test_a_conformance_axis_run_can_never_complete_ii4(self):
        from audit.implementation.ii4_comparison_certificate import build_stage_records
        from audit.implementation.stages import StageOutcome, validate_stage_record
        with tempfile.TemporaryDirectory() as tmp:
            axis = conformance_axis_from_factor(GOHR_DEPTH_FACTOR)
            run = self._run_dir(tmp, axis=axis,
                                seeds=ArmSeedManifest(base_seed=5_000, n_blocks=5,
                                                      arm_ids=axis.arm_ids))
            ii4, ii5 = build_stage_records(run, allow_non_evidentiary=True)
            self.assertIs(ii4.status, StageOutcome.PARTIAL)
            self.assertIs(ii5.status, StageOutcome.PARTIAL)
            self.assertEqual(ii4.components[-1].kind,
                             ComponentKind.CONFORMANCE_IMPACT_EXPERIMENT)
            self.assertEqual(validate_stage_record(ii4), [])
            # and forcing it to COMPLETED is refused by the project's own gate
            import dataclasses
            forced = dataclasses.replace(ii4, status=StageOutcome.COMPLETED)
            self.assertTrue(any("FORMAL_IMPLEMENTATION_COMPARISON" in p
                                for p in validate_stage_record(forced)))

    def test_an_insufficient_run_is_partial_and_says_why(self):
        from audit.implementation.ii4_comparison_certificate import build_stage_records
        from audit.implementation.stages import StageOutcome
        with tempfile.TemporaryDirectory() as tmp:
            p = plan(tmp)
            ad = FakeAdapter(arm_ids=(ARM_A, ARM_B),
                             fail_train=[("block0", ARM_A), ("block1", ARM_B),
                                         ("block2", ARM_A)])
            ComparisonRunner(plan=p, adapter=ad).run(execute=True)
            ii4, _ = build_stage_records(Path(p.output_dir), allow_non_evidentiary=True)
            self.assertIs(ii4.status, StageOutcome.PARTIAL)
            self.assertTrue(any("valid blocks were obtained" in x
                                for x in ii4.components[-1].limitations))

    def test_shared_components_become_stated_limitations(self):
        from audit.implementation.ii4_comparison_certificate import build_components
        with tempfile.TemporaryDirectory() as tmp:
            ii4, _ = build_components(self._run_dir(tmp), allow_non_evidentiary=True)
            self.assertTrue(any("SHARED by both arms" in x for x in ii4.limitations))

    def test_a_foreign_analysis_schema_is_refused(self):
        from audit.implementation.ii4_comparison_certificate import (
            ComparisonCertificateError, read_analysis,
        )
        with self.assertRaisesRegex(ComparisonCertificateError, "not the frozen"):
            read_analysis({"materiality": "NO_DIRECTIONAL_CLAIM"})

    def test_accessor_matches_the_real_frozen_analysis_schema(self):
        """
        Pins read_analysis against the schema ii4_analysis actually emits.
        Reading these keys off the top level yields None silently, which
        would blank a certificate's verdict.
        """
        from audit.implementation.ii4_analysis import analyse_ii4
        from audit.implementation.ii4_comparison_certificate import read_analysis
        result = analyse_ii4([0.001, -0.002, 0.0005, 0.0011, -0.0007, 0.0003],
                             n_blocks_requested=7, min_valid_blocks=6, delta=0.01, alpha=0.05)
        flat = read_analysis(json.loads(json.dumps(result.to_dict())))
        self.assertEqual(flat["materiality"], result.materiality.value)
        self.assertEqual(flat["equivalence"], result.equivalence.value)
        self.assertAlmostEqual(flat["mean_difference"], result.mean_difference, places=15)
        self.assertAlmostEqual(flat["ci95_low"], result.ci95_low, places=15)
        self.assertEqual(flat["n_blocks_valid"], 6)
        self.assertTrue(flat["sufficient"])

    def test_a_run_without_a_preregistration_is_refused(self):
        from audit.implementation.ii4_comparison_certificate import (
            ComparisonCertificateError, build_stage_records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(tmp)
            (run / "preregistration.json").unlink()
            with self.assertRaisesRegex(ComparisonCertificateError, "preregistration"):
                build_stage_records(run, allow_non_evidentiary=True)

    def test_a_non_evidentiary_run_may_not_license_a_stage(self):
        """A SMOKE artifact has the same SHAPE as a production one."""
        from audit.implementation.ii4_comparison_certificate import (
            ComparisonCertificateError, build_stage_records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(tmp)       # plan() defaults to ExecutionMode.SMOKE
            with self.assertRaisesRegex(ComparisonCertificateError, "may not license"):
                build_stage_records(run)
            ii4, _ = build_stage_records(run, allow_non_evidentiary=True)
            self.assertTrue(ii4.components[-1].limitations[0].startswith("NOT EVIDENCE"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
