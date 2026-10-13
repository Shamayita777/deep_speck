#!/usr/bin/env python3
"""
Exactly-once sealed-test evaluation: an audit of the invariant, not of
the happy path.

WHAT MUST HOLD. The sealed test set is the confirmatory measurement.
Evaluating it twice and choosing between the answers, or losing a
measurement and re-deriving it, would make the confirmatory result
adaptive - which is the thing the firewall exists to prevent. So:

  1. A permit may never be spent without a DURABLE journal entry for
     that measurement existing first.
  2. A recovered run may never count one measurement twice, re-evaluate
     a sealed set it has already measured, or attach a measurement to a
     different model, sealed set, arm or block.
  3. Any state the code cannot account for must FAIL CLOSED and leave
     the evidence in place, rather than be repaired into something
     plausible.

HOW THIS SUITE IS ORGANISED. The per-boundary SIGKILL evidence lives in
test_firewall_crash_recovery.py and the real-firewall API behaviour in
test_firewall_real_integration.py. This suite audits the two things
neither of those asks:

  - the ORDERING, as a structural property of the source: the journal
    write must precede the only permit-spending call site, and there
    must be exactly one such call site.
  - the CONFLICT MATRIX between the journal and the firewall. Each of
    those two durable records can be ahead of the other after a crash,
    and each combination has to resolve to one measurement and one
    spent permit.

ONE WINDOW IS DOCUMENTED, NOT CLOSED, and is stated here rather than in
a comment: a kill between running inference and writing the journal
loses the measurement, and the resumed session re-runs inference on the
sealed set. No permit was spent and no result was recorded, so the
exactly-once invariant on the EVIDENCE holds; the sealed set is read
twice computationally. That is harmless only because the terminal model
and the sealed set are both frozen and the evaluation is deterministic,
so the second pass necessarily yields the same number and cannot be
used to choose between answers. If the evaluation were ever made
stochastic, this window would become a real adaptivity hole.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from audit.implementation import ii4_comparison_runner as runner_mod
from audit.implementation.durable_state import (
    EVAL_CONSUMED,
    EVAL_MEASURED,
    EVAL_RECORDED,
    DurableStateError,
    EvalJournal,
)

SEALED = "s" * 64
MODEL = "m" * 64


def journal_at(path: Path, state: str, **kw) -> EvalJournal:
    journal = EvalJournal(path)
    fields = dict(accuracy=0.812345, sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                  arm_id="reference", block_id="block0", session_id="s1")
    fields.update(kw)
    journal.write(state=EVAL_MEASURED, **fields)
    if state != EVAL_MEASURED:
        journal.advance_at_least(state)
    return journal


class TestThePermitIsNeverSpentBeforeTheJournalIsDurable(unittest.TestCase):
    """
    Invariant 1, audited as a property of the source rather than of one
    execution. A reordering here would be invisible to every behavioural
    test that did not happen to crash in the new window.
    """

    def setUp(self):
        self.source = inspect.getsource(runner_mod.ComparisonRunner._run_block)

    def test_there_is_exactly_one_permit_spending_call_site(self):
        self.assertEqual(self.source.count("consume_confirmatory_evaluation("), 1,
                         "more than one place spends a permit; the ordering argument "
                         "below only covers one of them")

    def test_the_whole_module_has_exactly_one_permit_spending_call_site(self):
        whole = Path(runner_mod.__file__).read_text(encoding="utf-8")
        calls = [ln for ln in whole.splitlines()
                 if "consume_confirmatory_evaluation(" in ln
                 and not ln.strip().startswith("#")
                 and "exception from" not in ln]
        self.assertEqual(len(calls), 1, f"permit-spending call sites: {calls}")

    def test_the_journal_write_precedes_the_permit_in_the_source(self):
        measured = self.source.index("journal.write(state=EVAL_MEASURED")
        spent = self.source.index("consume_confirmatory_evaluation(")
        self.assertLess(measured, spent,
                        "the permit is spent before the measurement is journalled; a "
                        "crash in between would spend a permit with no recoverable "
                        "measurement")

    def test_the_journal_is_advanced_to_consumed_only_after_the_permit(self):
        spent = self.source.index("consume_confirmatory_evaluation(")
        advanced = self.source.index("advance_at_least(EVAL_CONSUMED)")
        self.assertLess(spent, advanced)

    def test_the_result_is_recorded_before_the_journal_closes(self):
        """
        RECORDED must mean "the run state carries this result", so the
        state write has to come first.
        """
        state_write = self.source.index('rec.update(status="EVALUATED"')
        # rindex, not index: the FIRST advance to RECORDED is the
        # reconciliation branch for an arm a previous session already
        # evaluated. The closing advance is the last one.
        closed = self.source.rindex("advance_at_least(EVAL_RECORDED)")
        self.assertLess(state_write, closed)
        self.assertEqual(self.source.count("advance_at_least(EVAL_RECORDED)"), 2,
                         "expected two advances to RECORDED: the reconciliation of an "
                         "already-evaluated arm, and the close of a fresh evaluation")

    def test_a_confirmed_duplicate_without_a_journal_is_not_replaceable(self):
        """
        The one unrecoverable combination: the firewall says the permit
        was spent and no journal exists. The measurement cannot be
        recovered and the sealed set has been used, so the block must
        not be quietly replaced by a fresh one.
        """
        self.assertIn("permit_spent_without_a_journal", self.source)
        # the sentence spans a line break in the source
        self.assertIn("block is not replaceable", self.source)


class TestTheJournalIsBoundToItsMeasurement(unittest.TestCase):
    """
    Invariant 2. A journal is only adoptable for the exact artifacts,
    arm and block it was written for.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "eval_journal.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_matching_journal_is_adopted(self):
        journal = journal_at(self.path, EVAL_MEASURED)
        record = journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                               arm_id="reference", block_id="block0")
        self.assertEqual(record["accuracy"], 0.812345)

    def test_a_journal_for_another_arm_is_refused(self):
        journal = journal_at(self.path, EVAL_MEASURED, arm_id="independent")
        with self.assertRaisesRegex(DurableStateError, "different arm or block"):
            journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                          arm_id="reference", block_id="block0")

    def test_a_journal_for_another_block_is_refused(self):
        journal = journal_at(self.path, EVAL_MEASURED, block_id="block5")
        with self.assertRaisesRegex(DurableStateError, "different arm or block"):
            journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                          arm_id="reference", block_id="block0")

    def test_a_journal_for_another_sealed_set_is_refused(self):
        journal = journal_at(self.path, EVAL_MEASURED, sealed_test_hash="x" * 64)
        with self.assertRaisesRegex(DurableStateError, "different data"):
            journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                          arm_id="reference", block_id="block0")

    def test_a_journal_for_another_terminal_model_is_refused(self):
        journal = journal_at(self.path, EVAL_MEASURED, terminal_model_hash="y" * 64)
        with self.assertRaisesRegex(DurableStateError, "different model"):
            journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                          arm_id="reference", block_id="block0")

    def test_the_runner_passes_the_identity_it_is_evaluating(self):
        """
        The checks above are worthless if the caller omits the fields.
        """
        source = inspect.getsource(runner_mod.ComparisonRunner._run_block)
        call = source[source.index("journal.adopt("):]
        call = call[:call.index(")") + 1]
        for field in ("sealed_test_hash=", "terminal_model_hash=", "arm_id=", "block_id="):
            self.assertIn(field, call, f"the runner's adopt() call omits {field}")

    def test_a_stale_journal_left_by_an_earlier_run_is_refused(self):
        """
        The realistic accident: an arm directory reused, or a journal
        copied while debugging. It carries a different model hash AND a
        different identity, and either is enough.
        """
        journal = journal_at(self.path, EVAL_RECORDED, arm_id="independent",
                             block_id="block3", terminal_model_hash="z" * 64)
        with self.assertRaises(DurableStateError):
            journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                          arm_id="reference", block_id="block0")


class TestAmbiguousJournalStatesFailClosed(unittest.TestCase):
    """Invariant 3: refuse and preserve, never repair into plausibility."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "eval_journal.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_torn_journal_refuses_and_is_preserved(self):
        torn = '{"state": "MEASURED", "accuracy": 0.81'
        self.path.write_text(torn)
        journal = EvalJournal(self.path)
        with self.assertRaises(DurableStateError):
            journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL)
        self.assertEqual(self.path.read_text(), torn,
                         "the torn journal was rewritten instead of refused")

    def test_a_journal_missing_its_state_refuses(self):
        self.path.write_text(json.dumps({"accuracy": 0.8, "sealed_test_hash": SEALED,
                                         "terminal_model_hash": MODEL}))
        with self.assertRaises((DurableStateError, KeyError)):
            EvalJournal(self.path).adopt(sealed_test_hash=SEALED,
                                         terminal_model_hash=MODEL)

    def test_an_unknown_state_refuses(self):
        self.path.write_text(json.dumps({
            "state": "PROBABLY_FINE", "accuracy": 0.8, "sealed_test_hash": SEALED,
            "terminal_model_hash": MODEL, "arm_id": "reference", "block_id": "block0"}))
        with self.assertRaises((DurableStateError, ValueError)):
            EvalJournal(self.path).advance_at_least(EVAL_RECORDED)

    def test_the_journal_cannot_move_backwards(self):
        journal = journal_at(self.path, EVAL_RECORDED)
        with self.assertRaisesRegex(DurableStateError, "backwards"):
            journal.advance(EVAL_MEASURED)

    def test_advancing_an_absent_journal_refuses_rather_than_creating_one(self):
        journal = EvalJournal(self.path)
        with self.assertRaisesRegex(DurableStateError, "does not exist"):
            journal.advance_at_least(EVAL_CONSUMED)
        self.assertFalse(self.path.exists(),
                         "a journal was fabricated for a measurement that never happened")

    def test_advance_at_least_is_idempotent_at_every_state(self):
        """
        The resume path walks the same code with the journal already
        ahead. Each call must be a no-op, not a backwards move and not
        a second transition.
        """
        for state in (EVAL_MEASURED, EVAL_CONSUMED, EVAL_RECORDED):
            with self.subTest(state=state):
                journal = journal_at(self.path, state)
                before = self.path.read_text()
                journal.advance_at_least(state)
                after = json.loads(self.path.read_text())
                self.assertEqual(after["state"], json.loads(before)["state"])
                self.assertEqual(after["accuracy"], 0.812345)


class TestTheJournalFirewallConflictMatrix(unittest.TestCase):
    """
    Either durable record can be ahead of the other after a crash. Each
    combination must resolve to ONE measurement and ONE spent permit.
    This enumerates them against the real firewall.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "eval_journal.json"
        self.fw_path = self.dir / "firewall.json"
        import importlib
        try:
            self.fw_mod = importlib.import_module(
                "audit.experimental_validity.framework.firewall")
        except ImportError:                                       # pragma: no cover
            self.fw_mod = importlib.import_module("framework.firewall")

    def tearDown(self):
        self.tmp.cleanup()

    def _firewall(self, *, consume=False, sealed_hash=SEALED):
        fw = self.fw_mod.TestSetFirewall(
            experiment_id="II-4-GOHR-IMPL-V1-TEST",
            confirmatory_data_mode=self.fw_mod.ConfirmatoryDataMode.SHARED)
        fw.freeze(self.fw_mod.FreezeRecord(
            frozen_config_hash="c" * 64, frozen_permutation_hash=None,
            frozen_replicate_plan_hash="r" * 64, frozen_statistical_plan_hash="p" * 64,
            frozen_at_utc="2026-10-13T00:00:00Z"))
        fw.seal_confirmatory_dataset(replicate_id="block0", dataset_hash=sealed_hash)
        if consume:
            fw.consume_confirmatory_evaluation(
                condition_id="reference", replicate_id="block0",
                observed_dataset_hash=sealed_hash)
        from audit.implementation.durable_state import save_firewall_atomically
        save_firewall_atomically(fw, self.fw_path)
        return fw

    def _confirms(self, sealed_hash=SEALED):
        return runner_mod.firewall_confirms_consumption(
            self.fw_path, condition_id="reference", replicate_id="block0",
            sealed_hash=sealed_hash)

    def test_neither_record_advanced_is_a_clean_first_evaluation(self):
        self._firewall(consume=False)
        confirmed, why = self._confirms()
        self.assertFalse(confirmed, why)
        self.assertIsNone(EvalJournal(self.path).read())

    def test_journal_measured_and_permit_unspent_resolves_to_one_consumption(self):
        """
        A kill between the journal write and the permit. The resumed run
        adopts the measurement and spends the permit once.
        """
        journal_at(self.path, EVAL_MEASURED)
        fw = self._firewall(consume=False)
        self.assertFalse(self._confirms()[0])

        fw.consume_confirmatory_evaluation(
            condition_id="reference", replicate_id="block0",
            observed_dataset_hash=SEALED)
        from audit.implementation.durable_state import save_firewall_atomically
        save_firewall_atomically(fw, self.fw_path)

        confirmed, _ = self._confirms()
        self.assertTrue(confirmed)
        self.assertEqual(json.loads(self.fw_path.read_text())["consumed_evaluation_keys"],
                         ["reference:block0"])

    def test_permit_spent_and_journal_measured_does_not_re_evaluate(self):
        """
        A kill after the permit, before the journal advance. The
        measurement is adopted; a second consumption is refused by the
        firewall and CONFIRMED as a duplicate, so the run continues.
        """
        journal = journal_at(self.path, EVAL_MEASURED)
        fw = self._firewall(consume=True)
        confirmed, why = self._confirms()
        self.assertTrue(confirmed, why)

        with self.assertRaises(Exception):
            fw.consume_confirmatory_evaluation(
                condition_id="reference", replicate_id="block0",
                observed_dataset_hash=SEALED)
        adopted = journal.adopt(sealed_test_hash=SEALED, terminal_model_hash=MODEL,
                                arm_id="reference", block_id="block0")
        self.assertEqual(adopted["accuracy"], 0.812345)
        journal.advance_at_least(EVAL_CONSUMED)
        self.assertEqual(json.loads(self.fw_path.read_text())["consumed_evaluation_keys"],
                         ["reference:block0"])

    def test_an_unpersisted_consumption_is_not_counted(self):
        """
        A kill between the in-memory consumption and the atomic save.
        The persisted firewall shows nothing, so the resumed run spends
        the permit - and the journal stops it re-measuring. One permit
        in the record, one measurement, which is the invariant.
        """
        journal_at(self.path, EVAL_MEASURED)
        fw = self._firewall(consume=False)
        fw.consume_confirmatory_evaluation(
            condition_id="reference", replicate_id="block0",
            observed_dataset_hash=SEALED)
        # ... killed here: self.fw_path still holds the pre-consumption state
        self.assertFalse(self._confirms()[0],
                         "an unpersisted consumption was treated as durable")
        persisted = json.loads(self.fw_path.read_text())
        self.assertEqual(persisted["consumed_evaluation_keys"], [])

    def test_a_consumption_against_a_different_sealed_hash_is_not_a_duplicate(self):
        """
        The dangerous near-miss: a permit spent for this arm and block,
        but against other data. Treating it as a duplicate would record
        a measurement of the wrong dataset.
        """
        self._firewall(consume=True, sealed_hash="d" * 64)
        confirmed, why = self._confirms(sealed_hash=SEALED)
        self.assertFalse(confirmed)
        self.assertTrue(why)

    def test_a_journal_recorded_with_no_consumption_at_all_is_detected(self):
        """
        The journal claims the result is recorded while the firewall has
        no permit spent. The firewall is the authority on permits, so
        this must not be read as "already consumed".
        """
        journal_at(self.path, EVAL_RECORDED)
        self._firewall(consume=False)
        self.assertFalse(self._confirms()[0])

    def test_an_absent_firewall_file_is_never_a_confirmation(self):
        journal_at(self.path, EVAL_RECORDED)
        confirmed, why = runner_mod.firewall_confirms_consumption(
            self.dir / "no_firewall_here.json", condition_id="reference",
            replicate_id="block0", sealed_hash=SEALED)
        self.assertFalse(confirmed)
        self.assertTrue(why)

    def test_a_torn_firewall_file_is_never_a_confirmation(self):
        self._firewall(consume=True)
        self.fw_path.write_text('{"frozen": true, "consumed_evaluation_ke')
        confirmed, why = self._confirms()
        self.assertFalse(confirmed, "a torn firewall file confirmed a consumption")
        self.assertTrue(why)


class TestTheDocumentedWindowIsStated(unittest.TestCase):
    """
    The one window left open is documented rather than closed, so the
    documentation is tested: an undocumented known gap is the problem.
    """

    def test_the_module_states_the_inference_before_journal_window(self):
        doc = " ".join(__doc__.split())
        self.assertIn("kill between running inference and writing the journal", doc)
        self.assertIn("No permit was spent and no result was recorded", doc)
        self.assertIn("deterministic", doc)
        self.assertIn("stochastic", doc)

    def test_the_runner_comments_the_ordering_at_the_call_site(self):
        source = inspect.getsource(runner_mod.ComparisonRunner._run_block)
        self.assertIn("JOURNAL BEFORE THE PERMIT IS SPENT", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
