#!/usr/bin/env python3
"""
Integration tests against the REAL `framework.firewall.TestSetFirewall`.

NOT A MOCK. An earlier version of this suite exercised a substitute whose
persisted field was called `consumed` and held pairs. The real firewall
serialises `consumed_evaluation_keys` as sorted `"<condition_id>:<replicate_id>"`
strings, so every verification against it found nothing - which meant
every firewall exception would have been reported as unconfirmed and
every crash-after-consumption would have lost its block. The mock passed
16 tests while the integration was broken. These tests import the real
class; if it is not importable they FAIL rather than skip, because a
skipped required test proves nothing and this one gates a 53-84
GPU-hour run.

Covered, as the review specified: initialization, freeze-before-seal,
save/load, successful consumption, duplicate consumption, mismatched
hash, and crash recovery after a persisted consumption.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import audit.implementation.ii4_comparison_runner as runner_mod
from audit.implementation.ii4_comparison_runner import (
    ComparisonRunner,
    firewall_confirms_consumption,
)
from audit.tests.test_ii4_comparison import ARM_A, ARM_B, FakeAdapter, design, plan

# A hard import, deliberately. No try/except: if the real framework is
# absent this module fails to load and the required suite reports it.
FW = runner_mod._firewall_module()
TestSetFirewall = FW.TestSetFirewall
FreezeRecord = FW.FreezeRecord
ConfirmatoryDataMode = FW.ConfirmatoryDataMode
FirewallViolation = FW.FirewallViolation

SEALED = "a" * 64
OTHER = "b" * 64
EXPERIMENT = "II-4-GOHR-IMPL-V1-TEST"


def a_freeze_record(**kw) -> "FreezeRecord":
    base = dict(frozen_config_hash="c" * 64, frozen_permutation_hash=None,
                frozen_replicate_plan_hash="r" * 64,
                frozen_statistical_plan_hash="s" * 64,
                frozen_at_utc="2026-10-13T00:00:00Z")
    base.update(kw)
    return FreezeRecord(**base)


def a_frozen_firewall(**kw) -> "TestSetFirewall":
    fw = TestSetFirewall(experiment_id=EXPERIMENT,
                         confirmatory_data_mode=ConfirmatoryDataMode.SHARED)
    fw.freeze(a_freeze_record(**kw))
    return fw


class TestRealFirewallLifecycle(unittest.TestCase):
    """The API as it actually is, not as a mock imagined it."""

    def test_the_class_is_TestSetFirewall_and_takes_an_experiment_id(self):
        fw = TestSetFirewall(experiment_id=EXPERIMENT)
        self.assertEqual(fw.experiment_id, EXPERIMENT)
        self.assertFalse(fw.is_frozen())
        self.assertIs(fw.confirmatory_data_mode, ConfirmatoryDataMode.PER_REPLICATE)
        self.assertFalse(hasattr(FW, "ConfirmatoryFirewall"),
                         "the module exposes no ConfirmatoryFirewall; the old mock's name")

    def test_sealing_before_freeze_is_refused(self):
        fw = TestSetFirewall(experiment_id=EXPERIMENT)
        with self.assertRaises(FirewallViolation) as cm:
            fw.seal_confirmatory_dataset("block0", SEALED)
        self.assertIn("before the experiment is frozen", str(cm.exception))

    def test_freeze_then_seal_succeeds(self):
        fw = a_frozen_firewall()
        fw.seal_confirmatory_dataset("block0", SEALED)
        self.assertEqual(fw.sealed_dataset_hashes["block0"], SEALED)

    def test_refreezing_is_refused(self):
        fw = a_frozen_firewall()
        with self.assertRaisesRegex(FirewallViolation, "already frozen"):
            fw.freeze(a_freeze_record())

    def test_shared_mode_requires_one_hash_across_replicates(self):
        fw = a_frozen_firewall()
        fw.seal_confirmatory_dataset("block0", SEALED)
        with self.assertRaisesRegex(FirewallViolation, "SHARED"):
            fw.seal_confirmatory_dataset("block1", OTHER)

    def test_resealing_a_replicate_with_a_different_hash_is_refused(self):
        fw = a_frozen_firewall()
        fw.seal_confirmatory_dataset("block0", SEALED)
        with self.assertRaisesRegex(FirewallViolation, "refusing to reseal"):
            fw.seal_confirmatory_dataset("block0", OTHER)

    def test_successful_consumption_records_the_expected_key(self):
        fw = a_frozen_firewall()
        fw.seal_confirmatory_dataset("block0", SEALED)
        fw.consume_confirmatory_evaluation(condition_id=ARM_A, replicate_id="block0",
                                           observed_dataset_hash=SEALED)
        self.assertEqual(fw.to_dict()["consumed_evaluation_keys"], [f"{ARM_A}:block0"])

    def test_duplicate_consumption_is_refused(self):
        fw = a_frozen_firewall()
        fw.seal_confirmatory_dataset("block0", SEALED)
        kw = dict(condition_id=ARM_A, replicate_id="block0", observed_dataset_hash=SEALED)
        fw.consume_confirmatory_evaluation(**kw)
        with self.assertRaisesRegex(FirewallViolation, "already evaluated"):
            fw.consume_confirmatory_evaluation(**kw)

    def test_a_mismatched_observed_hash_is_refused(self):
        fw = a_frozen_firewall()
        fw.seal_confirmatory_dataset("block0", SEALED)
        with self.assertRaisesRegex(FirewallViolation, "hash mismatch"):
            fw.consume_confirmatory_evaluation(condition_id=ARM_A, replicate_id="block0",
                                               observed_dataset_hash=OTHER)
        self.assertEqual(fw.to_dict()["consumed_evaluation_keys"], [],
                         "a refused consumption must not record a key")

    def test_consuming_an_unsealed_replicate_is_refused(self):
        fw = a_frozen_firewall()
        with self.assertRaisesRegex(FirewallViolation, "No confirmatory dataset"):
            fw.consume_confirmatory_evaluation(condition_id=ARM_A, replicate_id="block9",
                                               observed_dataset_hash=SEALED)

    def test_save_and_load_round_trip_preserves_everything(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "firewall.json"
            fw = a_frozen_firewall()
            for b in range(3):
                fw.seal_confirmatory_dataset(f"block{b}", SEALED)
            fw.consume_confirmatory_evaluation(condition_id=ARM_A, replicate_id="block0",
                                               observed_dataset_hash=SEALED)
            fw.save(path)

            back = TestSetFirewall.load(path)
            self.assertEqual(back.experiment_id, EXPERIMENT)
            self.assertTrue(back.is_frozen())
            self.assertIs(back.confirmatory_data_mode, ConfirmatoryDataMode.SHARED)
            self.assertEqual(back.sealed_dataset_hashes, fw.sealed_dataset_hashes)
            self.assertEqual(back.to_dict()["consumed_evaluation_keys"],
                             [f"{ARM_A}:block0"])
            # the reloaded firewall still enforces exactly-once
            with self.assertRaises(FirewallViolation):
                back.consume_confirmatory_evaluation(
                    condition_id=ARM_A, replicate_id="block0", observed_dataset_hash=SEALED)

    def test_the_persisted_schema_is_what_the_verifier_expects(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "firewall.json"
            fw = a_frozen_firewall()
            fw.seal_confirmatory_dataset("block0", SEALED)
            fw.consume_confirmatory_evaluation(condition_id=ARM_A, replicate_id="block0",
                                               observed_dataset_hash=SEALED)
            fw.save(path)
            raw = json.loads(path.read_text())
            self.assertEqual(sorted(raw), ["confirmatory_data_mode",
                                           "consumed_evaluation_keys", "experiment_id",
                                           "freeze_record", "frozen",
                                           "sealed_dataset_hashes"])
            self.assertNotIn("consumed", raw, "the mock's field name must not appear")
            self.assertEqual(raw["consumed_evaluation_keys"], [f"{ARM_A}:block0"])


class TestVerifierAgainstTheRealFirewall(unittest.TestCase):

    def _persist(self, tmp, *, consume=True, sealed=SEALED, frozen=True,
                 mode=ConfirmatoryDataMode.SHARED, condition=ARM_A, replicate="block0"):
        path = Path(tmp) / "firewall.json"
        fw = TestSetFirewall(experiment_id=EXPERIMENT, confirmatory_data_mode=mode)
        if frozen:
            fw.freeze(a_freeze_record())
            fw.seal_confirmatory_dataset(replicate, sealed)
            if consume:
                fw.consume_confirmatory_evaluation(
                    condition_id=condition, replicate_id=replicate,
                    observed_dataset_hash=sealed)
        fw.save(path)
        return path

    def _check(self, path, *, condition=ARM_A, replicate="block0", sealed=SEALED):
        return firewall_confirms_consumption(path, condition_id=condition,
                                             replicate_id=replicate, sealed_hash=sealed)

    def test_a_genuine_prior_consumption_is_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(self._persist(t))
            self.assertTrue(ok, why)
            self.assertIn(f"{ARM_A}:block0", why)

    def test_no_consumption_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(self._persist(t, consume=False))
            self.assertFalse(ok)
            self.assertIn("records no consumed_evaluation_key", why)

    def test_a_sealed_hash_mismatch_is_NOT_a_duplicate(self):
        """The substituted-confirmatory-data case. Must never pass."""
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(self._persist(t, sealed=OTHER))
            self.assertFalse(ok)
            self.assertIn("do not match", why)
            self.assertIn("must not be treated", why)

    def test_a_consumption_for_another_arm_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(self._persist(t, condition=ARM_B))
            self.assertFalse(ok)

    def test_a_consumption_for_another_block_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            ok, _ = self._check(self._persist(t, replicate="block1"), replicate="block0")
            self.assertFalse(ok)

    def test_key_matching_is_exact_not_a_prefix(self):
        """"reference:block1" must not satisfy a query about "reference:block10"."""
        with tempfile.TemporaryDirectory() as t:
            path = self._persist(t, replicate="block1")
            ok, _ = self._check(path, replicate="block10")
            self.assertFalse(ok)

    def test_an_unfrozen_firewall_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(self._persist(t, frozen=False))
            self.assertFalse(ok)
            self.assertIn("no freeze record", why)

    def test_a_per_replicate_firewall_is_not_confirmed(self):
        """This design declares one SHARED set; another mode is not ours."""
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(self._persist(t, mode=ConfirmatoryDataMode.PER_REPLICATE))
            self.assertFalse(ok)
            self.assertIn("not 'SHARED'", why)

    def test_a_missing_file_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            ok, why = self._check(Path(t) / "absent.json")
            self.assertFalse(ok)
            self.assertIn("no persisted firewall", why)

    def test_a_torn_file_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as t:
            path = self._persist(t)
            path.write_text('{"experiment_id": "X", "consumed_evaluation_ke')
            ok, why = self._check(path)
            self.assertFalse(ok)
            self.assertIn("could not be read", why)

    def test_the_verifier_reads_disk_not_memory(self):
        with tempfile.TemporaryDirectory() as t:
            path = self._persist(t, consume=False)          # disk: not consumed
            live = TestSetFirewall.load(path)
            live.consume_confirmatory_evaluation(condition_id=ARM_A, replicate_id="block0",
                                                 observed_dataset_hash=SEALED)
            self.assertIn(f"{ARM_A}:block0",
                          live.to_dict()["consumed_evaluation_keys"])   # memory says yes
            self.assertFalse(self._check(path)[0])                      # disk says no


class TestRunnerWithTheRealFirewall(unittest.TestCase):
    """End-to-end through the runner, with no firewall substitute at all."""

    def _run(self, tmp, **kw):
        d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
        p = plan(tmp, d=d)
        report = ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=p.axis.arm_ids),
                                  **kw).run(execute=True)
        return p, report

    def test_the_runner_freezes_before_sealing_and_records_a_real_record(self):
        with tempfile.TemporaryDirectory() as t:
            p, report = self._run(t)
            fw = report["firewall"]
            self.assertTrue(fw["frozen"])
            self.assertEqual(fw["confirmatory_data_mode"], "SHARED")
            self.assertEqual(fw["experiment_id"], p.experiment_id)
            record = fw["freeze_record"]
            # every hash present and traceable; the permutation hash is
            # legitimately None for an experiment with no permutation
            self.assertEqual(len(record["frozen_config_hash"]), 64)
            self.assertEqual(len(record["frozen_replicate_plan_hash"]), 64)
            self.assertEqual(record["frozen_statistical_plan_hash"],
                             p.design.preregistration_hash())
            self.assertIsNone(record["frozen_permutation_hash"])
            self.assertTrue(record["frozen_at_utc"])

    def test_the_freeze_record_binds_this_run_not_a_placeholder(self):
        with tempfile.TemporaryDirectory() as t:
            p, report = self._run(t)
            runner = ComparisonRunner(plan=p, adapter=FakeAdapter(arm_ids=p.axis.arm_ids))
            expected = runner.runtime_provenance()["experiment_fingerprint"]
            self.assertEqual(report["firewall"]["freeze_record"]["frozen_config_hash"],
                             expected)
            for placeholder in ("c" * 64, "0" * 64, "", None):
                self.assertNotEqual(
                    report["firewall"]["freeze_record"]["frozen_config_hash"], placeholder)

    def test_one_shared_hash_is_sealed_under_every_admissible_block(self):
        with tempfile.TemporaryDirectory() as t:
            _, report = self._run(t)
            sealed = report["firewall"]["sealed_dataset_hashes"]
            self.assertEqual(set(sealed), {"block0", "block1"})
            self.assertEqual(len(set(sealed.values())), 1)

    def test_exactly_once_per_arm_and_block(self):
        with tempfile.TemporaryDirectory() as t:
            _, report = self._run(t)
            keys = report["firewall"]["consumed_evaluation_keys"]
            self.assertEqual(sorted(keys),
                             sorted(f"{a}:block{b}" for a in (ARM_A, ARM_B)
                                    for b in range(2)))
            self.assertEqual(len(keys), len(set(keys)))

    def test_resume_refuses_a_firewall_frozen_for_another_configuration(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = self._run(t)
            path = Path(p.output_dir) / "firewall.json"
            raw = json.loads(path.read_text())
            raw["freeze_record"]["frozen_config_hash"] = "d" * 64
            path.write_text(json.dumps(raw))
            d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
            with self.assertRaisesRegex(Exception, "frozen for a different configuration"):
                ComparisonRunner(plan=plan(t, d=d),
                                 adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B))).run(execute=True)

    def test_resume_refuses_a_per_replicate_firewall(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = self._run(t)
            path = Path(p.output_dir) / "firewall.json"
            raw = json.loads(path.read_text())
            raw["confirmatory_data_mode"] = "PER_REPLICATE"
            path.write_text(json.dumps(raw))
            d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
            with self.assertRaisesRegex(Exception, "not in SHARED mode"):
                ComparisonRunner(plan=plan(t, d=d),
                                 adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B))).run(execute=True)

    def test_crash_recovery_after_a_persisted_consumption(self):
        """
        The scenario the whole journal exists for, against the real
        firewall: the permit is spent on disk and the run state was never
        updated. The resumed session must adopt the journalled accuracy,
        NOT re-evaluate, and NOT lose the block.
        """
        with tempfile.TemporaryDirectory() as t:
            p, first = self._run(t)
            self.assertEqual(first["n_blocks_valid"], 2)
            fw_before = json.loads((Path(p.output_dir) / "firewall.json").read_text())
            self.assertIn(f"{ARM_A}:block0", fw_before["consumed_evaluation_keys"])

            # rewind ONLY the bookkeeping, as a crash between the
            # consumption and the state write would have left it
            state_path = Path(p.output_dir) / "state.json"
            state = json.loads(state_path.read_text())
            rec = state["blocks"]["block0"]["arms"][ARM_A]
            rec["status"] = "TRAINED"
            rec.pop("sealed_test_accuracy", None)
            state_path.write_text(json.dumps(state))

            d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
            report = ComparisonRunner(plan=plan(t, d=d),
                                      adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B)),
                                      session_id="s2").run(execute=True)

            arms = json.loads(state_path.read_text())["blocks"]["block0"]["arms"]
            self.assertEqual(arms[ARM_A]["status"], "EVALUATED")
            self.assertTrue(arms[ARM_A]["evaluation_adopted_from_journal"])
            self.assertTrue(
                arms[ARM_A]["firewall_verification"]["confirmed_prior_consumption"])
            self.assertFalse(arms[ARM_A]["firewall_consumed_this_session"])
            self.assertIsNotNone(arms[ARM_A]["sealed_test_accuracy"])
            self.assertEqual(report["n_blocks_valid"], 2)
            # and the permit was not spent twice
            fw_after = json.loads((Path(p.output_dir) / "firewall.json").read_text())
            self.assertEqual(sorted(fw_after["consumed_evaluation_keys"]),
                             sorted(fw_before["consumed_evaluation_keys"]))

    def test_a_substituted_sealed_set_aborts_the_run_at_the_earliest_gate(self):
        """
        Defence in depth, three layers deep, and the inner two turn out
        to be unreachable from disk tampering:

          1. the REAL firewall refuses to reseal a replicate with a
             different hash - this is what fires;
          2. the runner's sealed-set integrity comparison would catch it
             if sealing somehow succeeded;
          3. the consumption-level hash check in the firewall would
             catch it at evaluation time.

        So the consumption-level path is exercised by the verifier's own
        unit test (`test_a_sealed_hash_mismatch_is_NOT_a_duplicate`)
        rather than through the runner. Nothing is evaluated and no
        accuracy is recorded.
        """
        with tempfile.TemporaryDirectory() as t:
            p, _ = self._run(t)
            state_path = Path(p.output_dir) / "state.json"
            state = json.loads(state_path.read_text())
            rec = state["blocks"]["block0"]["arms"][ARM_A]
            rec["status"] = "TRAINED"
            rec.pop("sealed_test_accuracy", None)
            state_path.write_text(json.dumps(state))

            fw_path = Path(p.output_dir) / "firewall.json"
            raw = json.loads(fw_path.read_text())
            raw["sealed_dataset_hashes"]["block0"] = OTHER
            fw_path.write_text(json.dumps(raw))

            d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
            # The REAL firewall's own reseal guard fires first and names the
            # substitution precisely, which is stronger than the runner's
            # downstream integrity comparison. Both are retained; this
            # asserts the one that actually triggers.
            with self.assertRaisesRegex(
                    Exception, "refusing to reseal with a different dataset"):
                ComparisonRunner(plan=plan(t, d=d),
                                 adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B)),
                                 session_id="s3").run(execute=True)
            arms = json.loads(state_path.read_text())["blocks"]["block0"]["arms"]
            self.assertIsNone(arms[ARM_A].get("sealed_test_accuracy"),
                              "an accuracy was recorded despite a substituted sealed set")

    def test_an_unrelated_firewall_error_fails_closed(self):
        """
        An arbitrary exception from the REAL firewall's consume method
        must not be read as a duplicate. Injected by patching the real
        class's method, so the firewall under test is still the real one.
        """
        original = TestSetFirewall.consume_confirmatory_evaluation

        def boom(self, **kw):
            raise OSError("Input/output error writing firewall state")

        with tempfile.TemporaryDirectory() as t:
            TestSetFirewall.consume_confirmatory_evaluation = boom
            try:
                p, report = self._run(t)
            finally:
                TestSetFirewall.consume_confirmatory_evaluation = original
            arms = json.loads(
                (Path(p.output_dir) / "state.json").read_text())["blocks"]["block0"]["arms"]
            for arm, rec in arms.items():
                self.assertEqual(rec["status"], "FAILED", f"{arm} was recorded as evaluated")
                self.assertEqual(rec["failure"]["reason"],
                                 "firewall_error_not_a_confirmed_duplicate")
                self.assertIsNone(rec.get("sealed_test_accuracy"))
                self.assertIn("OSError", rec["firewall_verification"]["exception"])
                self.assertFalse(rec["firewall_verification"]["confirmed_prior_consumption"])
            self.assertEqual(report["n_blocks_valid"], 0)

    def test_a_confirmed_duplicate_without_a_journal_still_fails(self):
        """Provably spent, measurement gone: fail, and NOT replaceable."""
        with tempfile.TemporaryDirectory() as t:
            p, _ = self._run(t)
            state_path = Path(p.output_dir) / "state.json"
            state = json.loads(state_path.read_text())
            rec = state["blocks"]["block0"]["arms"][ARM_A]
            rec["status"] = "TRAINED"
            rec.pop("sealed_test_accuracy", None)
            state_path.write_text(json.dumps(state))
            (Path(p.output_dir) / "blocks" / "block0" / ARM_A
             / "eval_journal.json").unlink()

            d = design(n_blocks=2, min_valid_blocks=2, max_replacement_blocks=0)
            runner = ComparisonRunner(plan=plan(t, d=d),
                                      adapter=FakeAdapter(arm_ids=(ARM_A, ARM_B)),
                                      session_id="s4")
            runner.run(execute=True)
            arms = json.loads(state_path.read_text())["blocks"]["block0"]["arms"]
            self.assertEqual(arms[ARM_A]["status"], "FAILED")
            self.assertEqual(arms[ARM_A]["failure"]["reason"],
                             "permit_spent_without_a_journal")
            self.assertTrue(
                arms[ARM_A]["firewall_verification"]["confirmed_prior_consumption"])
            replaceable, why = runner._is_replaceable("block0")
            self.assertFalse(replaceable, f"an evaluated block was replaceable: {why}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
