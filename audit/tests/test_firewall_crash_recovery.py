#!/usr/bin/env python3
"""
Firewall and sealed-test-set lifecycles under real process kills.

Everything here drives the REAL `TestSetFirewall` through the REAL
runner in a child process that SIGKILLs itself at a named persistence
boundary. The firewall is the one artifact whose loss cannot be repaired
by recomputation - a spent confirmatory permit is a fact about the past -
and the framework's own `save()` is a non-atomic, non-fsynced
`write_text`, so every write here goes through
`durable_state.save_firewall_atomically` instead.

The windows covered:

    firewall_frozen_in_memory      frozen, not yet persisted
    firewall_saved_after_freeze    persisted, state not yet written
    sealed_set_generated           file on disk, not yet recorded
    sealed_set_recorded            recorded, not yet sealed   <- used to brick
    sealed_set_sealed_in_memory    sealed in memory, not yet persisted
    firewall_saved_after_sealing   persisted, state not yet written
    journal_measured               accuracy journalled, permit unspent
    firewall_consumed              permit spent, state not yet written
    eval_recorded_in_state         state written, journal not yet advanced

WHAT THESE TESTS ESTABLISH, AND WHAT THEY DO NOT: see
`durable_state.WHAT_SIGKILL_ESTABLISHES`, which is asserted below. In
short they establish ORDERING - whatever a later process finds is
interpretable and recoverable - and they do NOT establish durability
against power loss, because SIGKILL leaves the page cache intact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import audit.implementation.ii4_comparison_runner as runner_mod
from audit.implementation.durable_state import (
    WHAT_SIGKILL_ESTABLISHES,
    EvalJournal,
    save_firewall_atomically,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FW = runner_mod._firewall_module()


def run_worker(run_dir: Path, payload: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "TF_CPP_MIN_LOG_LEVEL": "3"}
    return subprocess.run(
        [sys.executable, "-m", "audit.tests.kill_worker", "runner", str(run_dir),
         json.dumps(payload)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


class TestAtomicFirewallPersistence(unittest.TestCase):

    def test_the_wrapper_writes_the_frameworks_own_serialisation(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "firewall.json"
            fw = FW.TestSetFirewall(experiment_id="X",
                                    confirmatory_data_mode=FW.ConfirmatoryDataMode.SHARED)
            fw.freeze(FW.FreezeRecord(frozen_config_hash="c" * 64,
                                      frozen_permutation_hash=None,
                                      frozen_replicate_plan_hash="r" * 64,
                                      frozen_statistical_plan_hash="s" * 64,
                                      frozen_at_utc="2026-10-13T00:00:00Z"))
            fw.seal_confirmatory_dataset("block0", "h" * 64)
            save_firewall_atomically(fw, path)
            self.assertEqual(json.loads(path.read_text()), fw.to_dict())
            back = FW.TestSetFirewall.load(path)
            self.assertTrue(back.is_frozen())
            self.assertEqual(back.sealed_dataset_hashes, {"block0": "h" * 64})

    def test_the_wrapper_replaces_rather_than_truncating(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "firewall.json"
            fw = FW.TestSetFirewall(experiment_id="X")
            save_firewall_atomically(fw, path)
            fw2 = FW.TestSetFirewall(experiment_id="Y")
            save_firewall_atomically(fw2, path)
            self.assertEqual(json.loads(path.read_text())["experiment_id"], "Y")
            self.assertEqual(sorted(p.name for p in Path(t).iterdir()), ["firewall.json"])

    def test_the_frameworks_own_save_is_not_atomic(self):
        """
        Documents WHY the wrapper exists, against the real source. If a
        future framework version makes save() durable this test fails and
        the wrapper can be reconsidered - it does not fail silently.
        """
        import inspect
        src = inspect.getsource(FW.TestSetFirewall.save)
        self.assertIn("write_text", src)
        self.assertNotIn("fsync", src)
        self.assertNotIn("replace", src)


class TestFirewallLifecycleUnderKill(unittest.TestCase):

    def _kill_then_resume(self, boundary: str, tmp: Path, **kw):
        run = tmp / "run"
        first = run_worker(run, {"boundary": boundary, "session": "s1", **kw})
        self.assertEqual(first.returncode, -9,
                         f"child did not die by SIGKILL (rc={first.returncode}); "
                         f"stderr={first.stderr[-500:]}")
        second = run_worker(run, {"session": "s2", **kw})
        return first, second, run

    def _loadable(self, run: Path):
        path = run / "firewall.json"
        self.assertTrue(path.is_file(), "no persisted firewall survived the kill")
        return FW.TestSetFirewall.load(path)        # raises on a torn file

    # --- freeze -------------------------------------------------------
    def test_kill_after_freeze_before_persist(self):
        with tempfile.TemporaryDirectory() as t:
            first, second, run = self._kill_then_resume(
                "firewall_frozen_in_memory", Path(t))
            # nothing was persisted, so the next session starts clean
            self.assertEqual(second.returncode, 0, second.stderr[-500:])
            fw = self._loadable(run)
            self.assertTrue(fw.is_frozen())
            self.assertEqual(json.loads(second.stdout.strip().splitlines()[-1])
                             ["n_blocks_valid"], 2)

    def test_kill_after_firewall_persisted_before_state(self):
        with tempfile.TemporaryDirectory() as t:
            first, second, run = self._kill_then_resume(
                "firewall_saved_after_freeze", Path(t))
            fw = self._loadable(run)
            self.assertTrue(fw.is_frozen(), "the persisted firewall lost its freeze record")
            self.assertEqual(second.returncode, 0, second.stderr[-500:])

    # --- sealing ------------------------------------------------------
    def test_kill_after_generating_before_recording(self):
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume("sealed_set_generated", Path(t))
            self.assertEqual(second.returncode, 0, second.stderr[-500:])
            fw = self._loadable(run)
            self.assertEqual(len(set(fw.sealed_dataset_hashes.values())), 1)

    def test_kill_after_recording_before_sealing(self):
        """
        The window that used to brick the run: the test set was recorded
        in state and the firewall sealed nothing, so the next session's
        integrity comparison found an empty seal set against a recorded
        hash and refused permanently. Idempotent resealing repairs it.
        """
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume("sealed_set_recorded", Path(t))
            self.assertEqual(second.returncode, 0,
                             f"the run did not recover: {second.stdout[-400:]} "
                             f"{second.stderr[-600:]}")
            fw = self._loadable(run)
            self.assertEqual(sorted(fw.sealed_dataset_hashes), ["block0", "block1"])
            self.assertEqual(len(set(fw.sealed_dataset_hashes.values())), 1)
            state = json.loads((run / "state.json").read_text())
            self.assertTrue(state.get("sealing_events"),
                            "the repair was not recorded as a sealing event")
            self.assertEqual(json.loads(second.stdout.strip().splitlines()[-1])
                             ["n_blocks_valid"], 2)

    def test_kill_after_sealing_in_memory_before_persisting(self):
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume(
                "sealed_set_sealed_in_memory", Path(t))
            self.assertEqual(second.returncode, 0, second.stderr[-600:])
            fw = self._loadable(run)
            self.assertEqual(sorted(fw.sealed_dataset_hashes), ["block0", "block1"])

    def test_kill_after_persisting_the_seals(self):
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume(
                "firewall_saved_after_sealing", Path(t))
            self.assertEqual(second.returncode, 0, second.stderr[-600:])
            fw = self._loadable(run)
            self.assertEqual(len(set(fw.sealed_dataset_hashes.values())), 1)

    def test_the_sealed_set_is_never_regenerated_across_a_kill(self):
        """The recorded hash must be the SAME object after recovery."""
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume("sealed_set_recorded", Path(t))
            state = json.loads((run / "state.json").read_text())
            recorded = state["test_set"]["hash"]
            fw = self._loadable(run)
            self.assertEqual(set(fw.sealed_dataset_hashes.values()), {recorded})
            from audit.common.provenance import sha256_file
            self.assertEqual(sha256_file(state["test_set"]["path"]), recorded,
                             "the sealed test set on disk is not the one recorded")

    # --- evaluation ---------------------------------------------------
    def test_kill_after_journalling_the_measurement(self):
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume("journal_measured", Path(t))
            self.assertEqual(second.returncode, 0, second.stderr[-600:])
            out = json.loads(second.stdout.strip().splitlines()[-1])
            self.assertEqual(out["n_blocks_valid"], 2)
            self.assertGreaterEqual(out["adopted"], 1,
                                    "the journalled measurement was not adopted")
            self.assertEqual(len(out["consumed"]), len(set(out["consumed"])))

    def test_kill_after_the_permit_is_spent(self):
        """
        The permit is spent on disk and the run state was never updated.
        The resumed session must adopt the journalled accuracy, not
        re-evaluate, and not spend the permit twice.
        """
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume("firewall_consumed", Path(t))
            self.assertEqual(second.returncode, 0, second.stderr[-600:])
            out = json.loads(second.stdout.strip().splitlines()[-1])
            self.assertEqual(out["n_blocks_valid"], 2)
            self.assertGreaterEqual(out["adopted"], 1)
            self.assertEqual(sorted(out["consumed"]), sorted(set(out["consumed"])),
                             "a permit was spent twice")
            self.assertEqual(len(out["consumed"]), 4)

    def test_kill_after_the_state_write_before_the_journal_advance(self):
        with tempfile.TemporaryDirectory() as t:
            _, second, run = self._kill_then_resume("eval_recorded_in_state", Path(t))
            self.assertEqual(second.returncode, 0, second.stderr[-600:])
            out = json.loads(second.stdout.strip().splitlines()[-1])
            self.assertEqual(out["n_blocks_valid"], 2)
            self.assertEqual(len(out["consumed"]), 4)

    def test_the_journal_is_recoverable_after_every_eval_boundary(self):
        for boundary in ("journal_measured", "firewall_consumed",
                         "eval_recorded_in_state"):
            with self.subTest(boundary=boundary):
                with tempfile.TemporaryDirectory() as t:
                    _, second, run = self._kill_then_resume(boundary, Path(t))
                    self.assertEqual(second.returncode, 0, second.stderr[-400:])
                    journals = list(run.glob("blocks/*/*/eval_journal.json"))
                    self.assertTrue(journals, "no evaluation journal survived")
                    for j in journals:
                        record = EvalJournal(j).read()
                        self.assertEqual(record["state"], "RECORDED")
                        self.assertIsInstance(record["accuracy"], float)

    def test_every_covered_boundary_is_a_real_boundary(self):
        """A typo in a boundary name would silently disable a test."""
        import inspect
        src = inspect.getsource(runner_mod)
        for name in ("firewall_frozen_in_memory", "firewall_saved_after_freeze",
                     "sealed_set_generated", "sealed_set_recorded",
                     "sealed_set_sealed_in_memory", "firewall_saved_after_sealing",
                     "journal_measured", "firewall_consumed",
                     "eval_recorded_in_state"):
            self.assertIn(f'_boundary("{name}")', src,
                          f"{name} is not a boundary the runner actually reaches")


class TestSigkillScopeIsStated(unittest.TestCase):
    """The honesty requirement for item 6, enforced."""

    def test_the_scope_names_what_is_established(self):
        est = WHAT_SIGKILL_ESTABLISHES["establishes"]
        self.assertTrue(any("ORDERING" in x for x in est))
        self.assertTrue(any("atexit" in x for x in est))
        self.assertTrue(any("correspond to that epoch" in x for x in est))

    def test_the_scope_names_what_is_NOT_established(self):
        not_est = WHAT_SIGKILL_ESTABLISHES["does_not_establish"]
        self.assertTrue(any("page cache" in x for x in not_est),
                        "the page-cache limitation is not disclosed")
        self.assertTrue(any("power loss" in x for x in not_est))
        self.assertTrue(any("network filesystem" in x for x in not_est))
        self.assertTrue(any("third-party write" in x for x in not_est))
        self.assertTrue(any("concurrent" in x for x in not_est))

    def test_the_scope_reaches_the_run_artifact(self):
        self.assertIn("sigkill_test_scope",
                      __import__("inspect").getsource(runner_mod))


class TestTheRealSharedModulesAreInstalled(unittest.TestCase):
    """
    A stub must never be able to come back unnoticed. These hashes are
    the ones certificate V6 records for the shared modules, so a
    substituted module fails here rather than in a 60-GPU-hour run.
    """

    V6_HASHES = {
        "audit/common/outcomes.py":
            "edd70f8e38f7b2c4d53d992115bcd3e35ab0ea337c0586ad129fbdb9bc1b005e",
        "audit/common/provenance.py":
            "7cf33f8ed32f999cb58345f20e01b442f8b0c4e12c732e2bb95a08ce4c6b0eb2",
        "audit/common/run_manifest.py":
            "3c40f96987de7edae337ff199ac918f0bbb9635c1d60503949bfbbeac99e1ab9",
        "audit/common/strict_json.py":
            "cb02f0113caf88cd86b2a3f6624d866afaf74b358ea443ebaf111f7311a2bbcf",
        "audit/experimental_validity/framework/firewall.py":
            "6e40ad7c8f6119a6c50eb12155db980a76e6afb0a73a9ee2b93b97b2b17a6b62",
    }

    #: Shared modules that have legitimately changed since V6, each with
    #: the reason and the V6 SEMANTIC digest - the AST with every
    #: docstring stripped.
    #:
    #: A certified hash that no longer matches is a real provenance
    #: event and must not be silently re-baselined. But not every change
    #: invalidates a result: a corrected docstring changes the bytes and
    #: nothing the experiment computes. The distinction is made by
    #: MEASUREMENT rather than by assertion - the divergence is accepted
    #: only if the code, with prose removed, is bit-identical to what V6
    #: certified. Anything else fails exactly as a stub would.
    POST_V6_DOCUMENTATION_CORRECTIONS = {
        "audit/common/outcomes.py": {
            "v6_semantic_sha256":
                "fa568ed7ea6ebc2f7e41213bbd9d6cc5d10059adb3f1d274d3af2ec7fd812d57",
            "changed": "the AssessmentTier docstring, and nothing else",
            "reason": (
                "V6 was certified while that docstring stated that independent "
                "reimplementation (II-2) is an ENHANCED component and 'not a "
                "prerequisite for CORE Implementation Integrity'. Under the five-stage "
                "methodology (Section 3.5.4) all five II stages including II-2 are CORE, "
                "and the earlier wording is the claim that let the II-2 gap go unnoticed. "
                "The text was corrected afterwards. Reverting it to restore the V6 byte "
                "hash would reinstate the error, so the file is kept and the divergence "
                "is recorded here."),
            "effect_on_v6": (
                "NONE. Every class and every enum member is unchanged, so no value V6 "
                "computed or recorded could differ. V6 is not re-issued for a docstring."),
        },
    }

    @staticmethod
    def semantic_digest(path) -> str:
        """
        sha256 of the module's AST with all docstrings removed.

        Two files with the same semantic digest define exactly the same
        code and differ only in prose.
        """
        import ast
        import hashlib

        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
        return hashlib.sha256(
            ast.dump(ast.fix_missing_locations(tree)).encode()).hexdigest()

    def test_every_shared_module_matches_its_V6_hash(self):
        from audit.common.provenance import sha256_file
        for rel, expected in self.V6_HASHES.items():
            path = REPO_ROOT / rel
            with self.subTest(module=rel):
                self.assertTrue(path.is_file(), f"{rel} is missing")
                actual = sha256_file(path)
                if actual == expected:
                    continue

                correction = self.POST_V6_DOCUMENTATION_CORRECTIONS.get(rel)
                self.assertIsNotNone(
                    correction,
                    f"{rel} is NOT the module certificate V6 records, and no documented "
                    f"correction accounts for it. A stub or a modified copy is in place; "
                    f"every hash and every claim derived from it is void.\n"
                    f"  V6      {expected}\n  current {actual}")
                self.assertEqual(
                    self.semantic_digest(path), correction["v6_semantic_sha256"],
                    f"{rel} diverges from V6 by more than documentation. The code itself "
                    f"has changed, so the recorded correction does not account for it and "
                    f"results derived from the V6 hash are void.")

    def test_each_documented_correction_states_its_effect_on_V6(self):
        """
        A recorded divergence is only acceptable if someone wrote down
        why it is harmless. An entry with no reason is a silent
        re-baseline wearing a comment.
        """
        for rel, rec in self.POST_V6_DOCUMENTATION_CORRECTIONS.items():
            with self.subTest(module=rel):
                self.assertIn(rel, self.V6_HASHES)
                for field in ("v6_semantic_sha256", "changed", "reason", "effect_on_v6"):
                    self.assertTrue(rec.get(field), f"{rel} has no {field}")
                self.assertEqual(len(rec["v6_semantic_sha256"]), 64)
                self.assertGreater(len(rec["reason"]), 120,
                                   "the reason is too short to be an explanation")

    def test_the_semantic_digest_is_not_a_rubber_stamp(self):
        """
        The whole exemption rests on the semantic digest noticing a real
        code change. Proven, not assumed: an added enum member and a
        changed enum VALUE must both move the digest, while a rewritten
        docstring must not.
        """
        import tempfile

        original = (REPO_ROOT / "audit/common/outcomes.py").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.py"
            base.write_text(original)
            baseline = self.semantic_digest(base)

            prose = Path(tmp) / "prose.py"
            prose.write_text(original.replace(
                '"""\nShared outcome vocabularies for the audit framework.',
                '"""\nCompletely rewritten prose that says something else entirely.'))
            self.assertEqual(self.semantic_digest(prose), baseline,
                             "a docstring change moved the semantic digest")

            added = Path(tmp) / "added.py"
            added.write_text(original.replace(
                '    FAIL = "FAIL"', '    FAIL = "FAIL"\n    SMUGGLED = "SMUGGLED"', 1))
            self.assertNotEqual(self.semantic_digest(added), baseline,
                                "an ADDED enum member did not move the semantic digest")

            changed = Path(tmp) / "changed.py"
            changed.write_text(original.replace('    FAIL = "FAIL"',
                                                '    FAIL = "PASS"', 1))
            self.assertNotEqual(self.semantic_digest(changed), baseline,
                                "a CHANGED enum value did not move the semantic digest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
