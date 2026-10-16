#!/usr/bin/env python3
"""
Tests for the search harness.

THESE ARE SOFTWARE TESTS, NOT THE RESEARCH EXPERIMENT. Every test that
executes a search does so against `toy_cipher`, an 8-bit fixture defined
in this directory, at one or two generations. No test searches Speck32/64
or Simon32/64 at any setting, so no test artifact can be mistaken for
research evidence.

The tests that read the pinned upstream source need the checkout:

    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream

They FAIL rather than skip when it is absent, for the same reason as the
rest of this audit's suites: a skipped test is indistinguishable from a
passing one in a summary line.

Requires numpy 1.26.x (the harness refuses otherwise, and one test
asserts that refusal).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                                  # noqa: E402

import run_search                                                   # noqa: E402
import search_runner as sr                                          # noqa: E402
import toy_cipher                                                   # noqa: E402


def _toy_optimizer():
    """Import the pinned optimizer module fresh, without a cipher."""
    root = os.environ.get("AUTOND_UPSTREAM")
    if not root:
        raise AssertionError(
            "AUTOND_UPSTREAM is not set, so the pinned optimizer.py cannot be "
            "imported and these tests did not run. This is a FAILURE, not a skip.\n"
            "    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream")
    root = Path(root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    return importlib.import_module("optimizer"), root


class TestEnvironmentFacts(unittest.TestCase):
    """The four source-derived facts, re-checked where they are checkable."""

    def test_numpy_is_the_pinned_major_version(self):
        self.assertTrue(np.__version__.startswith("1.26"),
                        f"numpy {np.__version__}: AutoND's cipher module cannot be "
                        "imported under NumPy 2 (NEP 50). Install numpy==1.26.4.")

    def test_search_path_does_not_import_tensorflow(self):
        _opt, _root = _toy_optimizer()
        observed = sr.verify_no_tensorflow()
        self.assertFalse(observed["tensorflow"])
        self.assertFalse(observed["keras"])

    def test_evo_generation_default_is_bound_at_import(self):
        """
        The reproducibility trap: assigning the module constant does not
        change the already-bound default, so a programmatic override is
        silently ignored.
        """
        opt, _ = _toy_optimizer()
        before = opt.evo.__defaults__[0]
        opt.NUM_GENERATIONS = 999
        self.assertEqual(opt.evo.__defaults__[0], before,
                         "if this ever passes, upstream changed how evo binds n and "
                         "patch P1-GENERATIONS must be revisited.")
        opt.NUM_GENERATIONS = before

    def test_optimize_accepts_nb_samples_as_a_parameter(self):
        import inspect
        opt, _ = _toy_optimizer()
        self.assertIn("nb_samples", inspect.signature(opt.optimize).parameters)


class TestPatches(unittest.TestCase):
    """Patches must be declared, effective, and fully reversible."""

    def setUp(self):
        self.opt, _ = _toy_optimizer()
        self.tmp = Path(tempfile.mkdtemp())
        self.log = sr.GenerationLog(self.tmp / "g.jsonl")

    def tearDown(self):
        self.log.close()

    def test_generation_patch_sets_the_default_and_is_declared(self):
        patches, restore = sr.install_patches(
            self.opt, num_generations=37, population_init="previous",
            log=self.log, urandom_seed=None)
        try:
            # P2 wraps evo, so read the function that actually runs.
            self.assertEqual(self.opt.evo.__wrapped__.__defaults__[0], 37)
            p1 = [p for p in patches if p.patch_id == "P1-GENERATIONS"]
            self.assertEqual(len(p1), 1)
            self.assertTrue(p1[0].changes_behaviour)
            self.assertEqual(p1[0].after, "37")
        finally:
            restore()

    def test_restore_returns_the_module_to_its_original_state(self):
        before_evo = self.opt.evo
        before_defaults = self.opt.evo.__defaults__
        before_eval = self.opt.evaluate_multiple_differences
        before_urandom = self.opt.urandom
        _patches, restore = sr.install_patches(
            self.opt, num_generations=11, population_init="fresh",
            log=self.log, urandom_seed=7)
        restore()
        self.assertIs(self.opt.evo, before_evo)
        self.assertEqual(self.opt.evo.__defaults__, before_defaults)
        self.assertIs(self.opt.evaluate_multiple_differences, before_eval)
        self.assertIs(self.opt.urandom, before_urandom)

    def test_scoring_observer_is_declared_behaviour_neutral(self):
        patches, restore = sr.install_patches(
            self.opt, num_generations=2, population_init="previous",
            log=self.log, urandom_seed=None)
        try:
            p3 = [p for p in patches if p.patch_id == "P3-SCORING-OBSERVE"][0]
            self.assertFalse(p3.changes_behaviour)
        finally:
            restore()

    def test_fresh_population_patch_is_declared_behaviour_changing(self):
        patches, restore = sr.install_patches(
            self.opt, num_generations=2, population_init="fresh",
            log=self.log, urandom_seed=None)
        try:
            p2 = [p for p in patches if p.patch_id.startswith("P2-")][0]
            self.assertEqual(p2.patch_id, "P2-POPULATION-FRESH")
            self.assertTrue(p2.changes_behaviour)
        finally:
            restore()

    def test_previous_population_patch_is_observation_only(self):
        patches, restore = sr.install_patches(
            self.opt, num_generations=2, population_init="previous",
            log=self.log, urandom_seed=None)
        try:
            p2 = [p for p in patches if p.patch_id.startswith("P2-")][0]
            self.assertEqual(p2.patch_id, "P2-POPULATION-OBSERVE")
            self.assertFalse(p2.changes_behaviour)
        finally:
            restore()

    def test_urandom_patch_is_absent_unless_requested(self):
        patches, restore = sr.install_patches(
            self.opt, num_generations=2, population_init="previous",
            log=self.log, urandom_seed=None)
        try:
            self.assertEqual([p for p in patches if p.patch_id == "P4-URANDOM"], [])
        finally:
            restore()

    def test_urandom_patch_is_declared_a_deviation_when_requested(self):
        patches, restore = sr.install_patches(
            self.opt, num_generations=2, population_init="previous",
            log=self.log, urandom_seed=42)
        try:
            p4 = [p for p in patches if p.patch_id == "P4-URANDOM"][0]
            self.assertTrue(p4.changes_behaviour)
            self.assertIn("DEVIATION", p4.reason)
            self.assertEqual(len(self.opt.urandom(16)), 16)
        finally:
            restore()


class TestToySearchRun(unittest.TestCase):
    """
    A complete run of the harness against the 8-bit toy cipher.
    NOT the research experiment.
    """

    def setUp(self):
        self.opt, self.root = _toy_optimizer()
        self.tmp = Path(tempfile.mkdtemp())
        self.binding = sr.SourceBinding(
            autond_root=str(self.root), head_commit=sr.PINNED_COMMIT,
            worktree_dirty=None,
            source_hashes={"optimizer.py": sr.sha256_file(self.root / "optimizer.py")},
            cipher_module="tests.toy_cipher", cipher_hash="toy-fixture-not-a-cipher-file")
        self.cfg = {"mode": "r1_readme_demo", "cipher": "toy", "_cipher_module_supplied": True, "scenario": "single-key",
                    "num_generations": 1, "num_samples": 1000, "epsilon": 0.1,
                    "population_init": "previous", "runs": 1,
                    "numpy_seed": 1234, "urandom_seed": 99}

    def test_a_toy_run_completes_and_writes_its_record(self):
        rec = sr.run_once(self.cfg, self.opt, toy_cipher, self.binding, self.tmp, 0)
        self.assertEqual(rec["status"], "COMPLETED")
        self.assertIsInstance(rec["highest_non_random_round"], int)
        self.assertGreater(rec["elapsed_seconds"], 0.0)
        self.assertTrue((self.tmp / "run_000" / "result.json").is_file())

    def test_the_generation_log_is_contiguous_and_counted(self):
        rec = sr.run_once(self.cfg, self.opt, toy_cipher, self.binding, self.tmp, 0)
        rows = [json.loads(l) for l in
                (self.tmp / "run_000" / "generations.jsonl").read_text().splitlines()
                if l.strip()]
        self.assertEqual([r["seq"] for r in rows], list(range(len(rows))))
        self.assertEqual(rec["scoring_calls_logged"], len(rows))
        self.assertTrue(any(r["event"] == "evo_enter" for r in rows))
        self.assertTrue(any(r["event"] == "score_batch" for r in rows))

    def test_the_record_declares_its_reproducibility_honestly(self):
        rec = sr.run_once(self.cfg, self.opt, toy_cipher, self.binding, self.tmp, 0)
        self.assertTrue(rec["reproducibility"]["bit_reproducible"])
        unseeded = dict(self.cfg, urandom_seed=None)
        rec2 = sr.run_once(unseeded, self.opt, toy_cipher, self.binding, self.tmp, 1)
        self.assertFalse(rec2["reproducibility"]["bit_reproducible"])
        self.assertIn("NOT reproducible", rec2["reproducibility"]["os_urandom"])

    def test_the_module_is_unchanged_after_a_run(self):
        before = (self.opt.evo, self.opt.evo.__defaults__,
                  self.opt.evaluate_multiple_differences, self.opt.urandom)
        sr.run_once(self.cfg, self.opt, toy_cipher, self.binding, self.tmp, 0)
        self.assertIs(self.opt.evo, before[0])
        self.assertEqual(self.opt.evo.__defaults__, before[1])
        self.assertIs(self.opt.evaluate_multiple_differences, before[2])
        self.assertIs(self.opt.urandom, before[3])

    def test_manifest_hashes_every_produced_file(self):
        sr.run_once(self.cfg, self.opt, toy_cipher, self.binding, self.tmp, 0)
        sr.write_manifest(self.tmp, {"note": "toy"})
        manifest = json.loads((self.tmp / "MANIFEST.json").read_text())
        for rel, meta in manifest["files"].items():
            self.assertEqual(sr.sha256_file(self.tmp / rel), meta["sha256"], rel)
        self.assertIn("run_000/result.json", manifest["files"])


class TestVerifyOutputs(unittest.TestCase):
    """The verification script must catch a tampered or torn bundle."""

    def setUp(self):
        self.opt, self.root = _toy_optimizer()
        self.tmp = Path(tempfile.mkdtemp())
        binding = sr.SourceBinding(
            autond_root=str(self.root), head_commit=sr.PINNED_COMMIT,
            worktree_dirty=None, source_hashes={}, cipher_module="tests.toy_cipher",
            cipher_hash="toy")
        cfg = {"mode": "r1_readme_demo", "cipher": "toy", "_cipher_module_supplied": True, "scenario": "single-key",
               "num_generations": 1, "num_samples": 1000, "epsilon": 0.1,
               "population_init": "previous", "runs": 1,
               "numpy_seed": 5, "urandom_seed": 5}
        sr.run_once(cfg, self.opt, toy_cipher, binding, self.tmp, 0)
        sr.write_manifest(self.tmp, {"config": cfg})
        import verify_outputs
        self.verify = verify_outputs.verify

    def test_an_intact_bundle_verifies(self):
        ok, problems, _summary = self.verify(self.tmp)
        self.assertTrue(ok, problems)

    def test_a_modified_file_is_detected(self):
        p = self.tmp / "run_000" / "result.json"
        p.write_text(p.read_text() + "\n")
        ok, problems, _ = self.verify(self.tmp)
        self.assertFalse(ok)
        self.assertTrue(any("sha256" in x for x in problems))

    def test_an_added_file_is_detected(self):
        (self.tmp / "run_000" / "extra.txt").write_text("not in the manifest")
        ok, problems, _ = self.verify(self.tmp)
        self.assertFalse(ok)
        self.assertTrue(any("not named in the manifest" in x for x in problems))

    def test_os_debris_does_not_invalidate_the_bundle(self):
        """
        REGRESSION. Browsing the evidence folder in Finder drops a
        .DS_Store into it. Rule 2 counted that as an unmanifested file and
        failed the bundle, which made implementation_stages report
        II1-SEARCH-R1 as FAILED rather than PARTIAL - discarding a real
        scientific finding (2 epsilon-close differences vs the README's
        one) because of an OS artifact. The file is recorded and left in
        place; it is never deleted.
        """
        for rel in (".DS_Store",
                    "run_000/.DS_Store",
                    "run_000/._result.json",
                    "run_000/__pycache__/x.pyc",
                    "run_000/.ipynb_checkpoints/result-checkpoint.json",
                    "Thumbs.db"):
            p = self.tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"os debris\n")

        ok, problems, summary = self.verify(self.tmp)
        self.assertTrue(ok, problems)
        self.assertFalse(
            any("not named in the manifest" in x for x in problems), problems)

        recorded = summary.get("unmanifested_benign") or []
        self.assertIn(".DS_Store", recorded)
        self.assertIn("Thumbs.db", recorded)
        self.assertIn("unmanifested_benign_note", summary)
        self.assertIn("LEFT IN PLACE", summary["unmanifested_benign_note"])

        # and nothing was deleted
        for rel in (".DS_Store", "run_000/.DS_Store", "Thumbs.db"):
            self.assertTrue((self.tmp / rel).is_file(), rel)

    def test_a_real_added_file_still_fails_even_beside_debris(self):
        """The exemption must not become a hole: debris is excused, content is not."""
        (self.tmp / ".DS_Store").write_bytes(b"debris\n")
        (self.tmp / "run_000" / "smuggled.json").write_text('{"result": 1}')
        ok, problems, _ = self.verify(self.tmp)
        self.assertFalse(ok)
        self.assertTrue(any("smuggled.json" in x for x in problems), problems)

    def test_a_name_merely_containing_a_benign_string_is_not_excused(self):
        """Matched on path parts, not substrings."""
        for rel in ("run_000/my.DS_Store.json",
                    "run_000/__pycache__notes.txt",
                    "run_000/results.pyc.json"):
            p = self.tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("content pretending to be debris")
            ok, problems, _ = self.verify(self.tmp)
            self.assertFalse(ok, rel)
            self.assertTrue(any(Path(rel).name in x for x in problems),
                            f"{rel} was wrongly excused: {problems}")
            p.unlink()

    def test_a_manifested_file_is_still_hash_checked_even_if_benign_named(self):
        """
        A file the manifest DOES name is verified normally, whatever it is
        called. The exemption applies only to unmanifested paths.
        """
        ok, problems, _ = self.verify(self.tmp)
        self.assertTrue(ok, problems)
        p = self.tmp / "run_000" / "result.json"
        p.write_text(p.read_text() + "\n")
        ok, problems, _ = self.verify(self.tmp)
        self.assertFalse(ok)
        self.assertTrue(any("sha256" in x for x in problems))

    def test_a_missing_manifest_is_detected(self):
        (self.tmp / "MANIFEST.json").unlink()
        ok, problems, _ = self.verify(self.tmp)
        self.assertFalse(ok)
        self.assertIn("MANIFEST.json", problems[0])


class TestConfigurationRefusals(unittest.TestCase):
    """The CLI must refuse configurations that would mislabel a result."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _cfg(self, **over):
        base = {"mode": "r2_paper_parameters", "cipher": "speck3264",
                "scenario": "single-key", "num_generations": 50,
                "num_samples": 10000, "epsilon": 0.1,
                "population_init": "fresh", "runs": 1}
        base.update(over)
        p = self.tmp / "c.json"
        p.write_text(json.dumps(base))
        return p

    def test_r2_below_published_settings_is_refused(self):
        with self.assertRaises(sr.SearchRunnerError) as cm:
            run_search.load_config(self._cfg(num_generations=5))
        self.assertIn("must not carry the R2 label", str(cm.exception))

    def test_r1_with_any_override_is_refused(self):
        with self.assertRaises(sr.SearchRunnerError) as cm:
            run_search.load_config(self._cfg(mode="r1_readme_demo",
                                             num_generations=50, num_samples=10000))
        self.assertIn("must override nothing", str(cm.exception))

    def test_r2_with_too_few_samples_is_refused(self):
        with self.assertRaises(sr.SearchRunnerError):
            run_search.load_config(self._cfg(num_samples=1000))

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(sr.SearchRunnerError):
            run_search.load_config(self._cfg(mode="quick"))

    def test_unknown_population_init_is_refused(self):
        with self.assertRaises(sr.SearchRunnerError):
            run_search.load_config(self._cfg(population_init="random"))

    def test_missing_keys_are_refused(self):
        p = self.tmp / "bad.json"
        p.write_text(json.dumps({"mode": "benchmark"}))
        with self.assertRaises(sr.SearchRunnerError) as cm:
            run_search.load_config(p)
        self.assertIn("missing", str(cm.exception))

    def test_a_valid_benchmark_config_loads(self):
        cfg = run_search.load_config(PKG / "config" / "r1_readme_demo.json")
        self.assertEqual(cfg["mode"], "r1_readme_demo")
        self.assertIsNone(cfg["num_generations"])
        self.assertIsNone(cfg["num_samples"])

    def test_the_shipped_r2_config_is_valid_and_declares_its_deviations(self):
        import json
        path = PKG / "config" / "r2_paper_parameters.json"
        cfg = run_search.load_config(path)
        self.assertEqual(cfg["mode"], "r2_paper_parameters")
        self.assertGreaterEqual(cfg["num_generations"], 50)
        self.assertGreaterEqual(cfg["num_samples"], 10000)
        raw = json.loads(path.read_text())
        self.assertIn("_deviations", raw)
        joined = " ".join(raw["_deviations"]).lower()
        self.assertIn("0.01", joined)
        self.assertIn("elitist", joined)


class TestSourceBindingRefusals(unittest.TestCase):

    def test_a_root_without_optimizer_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        with self.assertRaises(sr.SearchRunnerError) as cm:
            sr.bind_source(tmp, "speck3264")
        self.assertIn("optimizer.py", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
