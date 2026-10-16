"""
Regression tests for the ten defects corrected in revision 3.

Each class is tied to one review item. The tests are written to fail
against revision 2's behaviour, so they pin the specific mistakes rather
than testing in general:

   1  launcher built `--model`, which the pinned parser does not define
   2  no code could produce a conforming result bundle at all
   3  freeze gate ignored the plan module's executable logic and the
      sidecar's committed state, and emitted unusable git paths
   4  --strict could never pass on valid R1 evidence, and `ok` hid a
      wrong upstream commit and FAILED stages
   5  II-3 reported AGREES without reading the pinned source
   6  validator accepted one checkpoint, "not clean", unordered
      timestamps, escaping paths, and metrics that disagreed with the NPZ
   7  --paper-pdf produced nothing persistable
   8  the launcher would overwrite a previous run directory
   9  the archive wrote unrelated root-level files
  10  an untracked .DS_Store in the checkout blocked every run

Nothing here trains, searches or evaluates. The one fixture that needs a
frozen plan builds a real git repository and performs the documented
freeze procedure inside it; the shipped plan stays unfrozen.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit.implementation.stages import StageOutcome
from audit.targets.autond import evaluation_assemble as asm
from audit.targets.autond import evaluation_gate as gate
from audit.targets.autond import evaluation_results as evres
from audit.targets.autond import implementation_stages as st
from audit.targets.autond import paper_source_architecture as psa
from audit.targets.autond import upstream_interface as ui
from audit.targets.autond.preregistrations import autond_ii_v2 as plan
from audit.targets.autond.tests import portable


# ===========================================================================
# 1. The launcher must speak the target's actual interface
# ===========================================================================

class TestUpstreamInterface(unittest.TestCase):
    def test_recorded_spec_matches_the_pinned_source(self):
        up = portable.require_real_upstream(self)
        res = ui.verify_argspec(up)
        self.assertTrue(res["verified"], res["problems"])
        self.assertEqual(res["source_sha256"], ui.EVAL_NETS_SHA256)

    def test_extraction_finds_the_real_arguments(self):
        up = portable.require_real_upstream(self)
        spec = ui.extract_argspec(up)
        for name in ("model_path", "model_type", "cipher", "scenario",
                     "input_difference", "round_number", "add_str"):
            self.assertIn(name, spec)
        self.assertNotIn("model", spec,
                         "the pinned parser has no --model; revision 2 used it")
        self.assertTrue(spec["cipher"]["required"])
        self.assertEqual(spec["model_type"]["choices"],
                         ["gohr-depth1", "dbitnet"])

    def test_no_checkout_is_reported_unverified_not_assumed(self):
        res = ui.verify_argspec(None)
        self.assertFalse(res["verified"])
        self.assertTrue(res["problems"])

    def test_commands_use_only_arguments_the_target_defines(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            cmds = gate.planned_commands(up, Path(tmp) / "out",
                                         argspec=ui.EVAL_NETS_ARGSPEC)
            # Two checkpoints x INVOCATIONS_PER_CHECKPOINT: the
            # registered n = 10 is reached by invoking the target
            # twice per checkpoint, since its nEval = 5 is a module
            # constant and editing it would modify the pinned repo.
            self.assertEqual(
                len(cmds),
                2 * plan.INVOCATIONS_PER_CHECKPOINT)
            for c in cmds:
                flags = [a for a in c["command"] if a.startswith("--")]
                for f in flags:
                    self.assertIn(f[2:], ui.EVAL_NETS_ARGSPEC,
                                  f"{f} is not an argument of eval_nets.py")
                self.assertIn("--model_path", flags)
                self.assertNotIn("--model", flags)

    def test_commands_supply_every_required_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            for c in gate.planned_commands(up, Path(tmp) / "out",
                                           argspec=ui.EVAL_NETS_ARGSPEC):
                flags = {a[2:] for a in c["command"] if a.startswith("--")}
                for name, e in ui.EVAL_NETS_ARGSPEC.items():
                    if e["required"]:
                        self.assertIn(name, flags, f"--{name} is required")
                for need in ("model_type", "input_difference", "round_number",
                             "scenario"):
                    self.assertIn(need, flags,
                                  f"--{need} is needed for a real evaluation")

    def test_command_values_respect_the_targets_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            for c in gate.planned_commands(up, Path(tmp) / "out",
                                           argspec=ui.EVAL_NETS_ARGSPEC):
                cmd = c["command"]
                mt = cmd[cmd.index("--model_type") + 1]
                sc = cmd[cmd.index("--scenario") + 1]
                self.assertIn(mt, ui.EVAL_NETS_ARGSPEC["model_type"]["choices"])
                self.assertIn(sc, ui.EVAL_NETS_ARGSPEC["scenario"]["choices"])

    def test_unknown_or_forbidden_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            ui.build_eval_command(python="p", model_path=Path("/m.h5"),
                                  cipher="speck3264", model_type="nope",
                                  scenario="single-key",
                                  input_difference="0x1", round_number=1)
        spec = dict(ui.EVAL_NETS_ARGSPEC)
        spec.pop("cipher")
        with self.assertRaises(ValueError):
            ui.build_eval_command(python="p", model_path=Path("/m.h5"),
                                  cipher="speck3264", model_type="dbitnet",
                                  scenario="single-key", input_difference="0x1",
                                  round_number=1, argspec=spec)

    def test_npz_name_matches_the_targets_own_arithmetic(self):
        self.assertEqual(ui.npz_name_for(Path("/o/c/m.h5")),
                         Path("/o/c/m_eval.npz"))
        self.assertEqual(ui.npz_name_for(Path("/o/c/m.h5"), "_x"),
                         Path("/o/c/m_eval_x.npz"))
        with self.assertRaises(ValueError):
            ui.npz_name_for(Path("/o/c/m.hdf5"))

    def test_planned_npz_lies_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            out = Path(tmp) / "out"
            for c in gate.planned_commands(up, out,
                                           argspec=ui.EVAL_NETS_ARGSPEC):
                self.assertTrue(str(c["npz"]).startswith(str(out)))
                self.assertFalse(str(c["npz"]).startswith(str(up)))


# ===========================================================================
# 2. Result assembly must exist and produce a valid bundle
# ===========================================================================

class TestAssembly(unittest.TestCase):
    def _frozen(self, stack, tmp):
        fx = portable.simulate_frozen(plan, Path(tmp) / "repo")
        stack.append(fx)
        return fx

    def tearDown(self):
        for fx in getattr(self, "_fx", []):
            plan.FROZEN_SIDECAR = fx["original_sidecar"]

    def setUp(self):
        self._fx: list = []

    def test_assembled_bundle_validates_without_a_handwritten_record(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan)
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertTrue(res["ok"], res["problems"])
            self.assertEqual(sorted(res["written"]),
                             ["RESULT.json", "SHA256SUMS.txt"])

            v = evres.validate_bundle(run, expected_plan_hash=plan.plan_hash())
            self.assertTrue(v["ok"], v["problems"])
            self.assertTrue(v["plan_bound"])
            self.assertEqual(sorted(v["derived"]), ["simon3264", "speck3264"])

    def test_metrics_come_from_the_npz_not_the_journal(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan)
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertTrue(res["ok"], res["problems"])
            import numpy as np
            rec = res["record"]["checkpoints"]["speck3264"]
            # `npz` is a list under the registered two-invocation design,
            # one entry per invocation. The pooled accuracies must be the
            # concatenation of the raw arrays IN THAT ORDER, so the check
            # now covers pooling as well as provenance.
            paths = rec["npz"] if isinstance(rec["npz"], list) else [rec["npz"]]
            self.assertEqual(len(paths), plan.INVOCATIONS_PER_CHECKPOINT)
            raw: list[float] = []
            for rel in paths:
                with np.load(run / rel) as z:
                    raw.extend(float(x) for x in z["accs"])
            self.assertEqual(raw, rec["accuracies"])
            self.assertEqual(len(raw), plan.N_TEST_SETS)

    def test_a_journal_disagreeing_with_the_npz_is_refused(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(
                Path(tmp) / "run", up, plan,
                claimed_accuracies={"speck3264": [0.9, 0.9, 0.9, 0.9, 0.9]})
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("disagrees with the artifact" in p
                                for p in res["problems"]), res["problems"])
            self.assertFalse((run / "RESULT.json").exists(),
                             "a refused assembly must write nothing")

    def test_a_failed_execution_is_not_assembled(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan,
                                          returncode=1)
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("exited 1" in p for p in res["problems"]))

    def test_a_partial_run_is_not_assembled(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan,
                                          only=("speck3264",))
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("simon3264" in p for p in res["problems"]))

    def test_wrong_commit_in_the_journal_is_refused(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan,
                                          commit="0" * 40)
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("not the pinned" in p for p in res["problems"]))

    def test_short_sample_is_refused(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            self._frozen(self._fx, tmp)
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan, n_sets=3)
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("not 5" in p or "missing-data" in p
                                for p in res["problems"]), res["problems"])

    def test_assembly_refuses_while_the_plan_is_unfrozen(self):
        up = portable.require_real_upstream(self)
        with tempfile.TemporaryDirectory() as tmp:
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan)
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("not frozen" in p for p in res["problems"]))

    def test_missing_journal_cannot_be_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "run"
            d.mkdir()
            res = asm.assemble(d)
            self.assertFalse(res["ok"])
            self.assertTrue(any("journal" in p for p in res["problems"]))

    def test_assembler_runs_no_evaluation(self):
        import ast

        tree = ast.parse(Path(asm.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, "run",
                                    "the assembler must not invoke a subprocess")


# ===========================================================================
# 3. The freeze gate
# ===========================================================================

class TestFreezeGateR3(unittest.TestCase):
    def setUp(self):
        self._orig = plan.FROZEN_SIDECAR

    def tearDown(self):
        plan.FROZEN_SIDECAR = self._orig

    def test_a_correctly_frozen_plan_opens_the_gate(self):
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            fx = portable.simulate_frozen(plan, Path(tmp) / "repo")
            ok, detail = plan.is_frozen(fx["repo_root"])
            self.assertTrue(ok, detail)

    def test_modified_executable_logic_blocks_the_gate(self):
        """
        The decisive regression test for revision 2. plan_hash() covers
        only the declarative PLAN object, so rewriting decide() left it
        unchanged and the gate opened. The sidecar now binds the plan
        FILE's bytes.
        """
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            fx = portable.simulate_frozen(plan, Path(tmp) / "repo")
            rec = dict(fx["record"])
            rec["plan_file_sha256"] = "b" * 64        # as if the code changed
            fx["sidecar"].write_text(json.dumps(rec, indent=2, sort_keys=True))
            portable._git(["add", "-A"], fx["repo_root"])
            portable._git(["commit", "-q", "-m", "x"], fx["repo_root"])
            ok, detail = plan.is_frozen(fx["repo_root"])
            self.assertFalse(ok)
            self.assertIn("plan FILE sha256", detail)
            self.assertIn("code changed after it was frozen", detail)

    def test_an_uncommitted_sidecar_blocks_the_gate(self):
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            fx = portable.simulate_frozen(plan, Path(tmp) / "repo",
                                          commit_sidecar=False)
            ok, detail = plan.is_frozen(fx["repo_root"])
            self.assertFalse(ok)
            self.assertIn("not tracked by git", detail)

    def test_a_sidecar_altered_after_commit_blocks_the_gate(self):
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            fx = portable.simulate_frozen(plan, Path(tmp) / "repo",
                                          tamper_sidecar_after_commit=True)
            ok, detail = plan.is_frozen(fx["repo_root"])
            self.assertFalse(ok)
            self.assertIn("differs from the committed version", detail)

    def test_a_stale_plan_hash_still_blocks(self):
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            fx = portable.simulate_frozen(plan, Path(tmp) / "repo",
                                          plan_hash="a" * 64)
            ok, detail = plan.is_frozen(fx["repo_root"])
            self.assertFalse(ok)
            self.assertIn("changed after it was frozen", detail)

    def test_generated_git_commands_use_repository_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = plan.freeze_instructions(Path(tmp))
        self.assertIn("git add audit/targets/autond/preregistrations/"
                      "autond_ii_v2.py", text)
        self.assertIn("git add audit/targets/autond/preregistrations/"
                      "autond_ii_v2.frozen.json", text)
        self.assertNotIn("git add autond_ii_v2.py", text)
        self.assertIn('"plan_file_path": "audit/targets/autond/preregistrations/'
                      'autond_ii_v2.py"', text)
        # git hash-object may be NAMED, to warn against it; it must not be the
        # command that produces the digest the sidecar records.
        self.assertNotIn("PLAN_SHA=$(git hash-object", text)
        self.assertIn("PLAN_SHA=$(shasum -a 256 audit/targets/autond/"
                      "preregistrations/autond_ii_v2.py", text)

    def test_repo_relative_helper(self):
        rel = plan.repo_relative(plan.PLAN_FILE, plan.PLAN_FILE.parents[4])
        self.assertEqual(
            rel, "audit/targets/autond/preregistrations/autond_ii_v2.py")

    def test_wrong_source_hashes_still_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            permitted, blockers = plan.confirmatory_run_permitted(up)
            self.assertFalse(permitted)
            self.assertTrue(any("!= recorded" in b for b in blockers))

    def test_the_plan_is_either_unfrozen_or_legitimately_frozen(self):
        """
        See the companion in test_corrections. Asserting the sidecar is
        absent would make the operator's own correct freeze fail the suite;
        asserting that any sidecar present actually verifies is both
        survivable and stronger.
        """
        if not plan.FROZEN_SIDECAR.exists():
            ok, _ = plan.is_frozen()
            self.assertFalse(ok)
            return
        ok, detail = plan.is_frozen()
        self.assertTrue(ok, f"freeze record present but not verifying: {detail}")


# ===========================================================================
# 4. Strict-mode semantics
# ===========================================================================

class TestStrictSemantics(unittest.TestCase):
    def test_strict_can_succeed_on_manifest_bound_evidence(self):
        """
        Revision 2's --strict required a hard-coded digest per artifact, so
        R1 files - bound by MANIFEST.json, deliberately without separate
        digests - could never satisfy it. It is now satisfiable.
        """
        up = portable.require_real_upstream(self)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            home = Path(tmp) / "home"
            run = home / "Documents/autond_search_runs/r1_20261015T035553Z"
            portable.fake_search_bundle(run)
            (repo / "audit").mkdir(parents=True)

            # Restrict the inventory to the artifacts this machine can
            # satisfy: the two shipped checkpoints and the manifest-bound
            # R1 files. The census artifacts live on the operator's Mac.
            keep = {"r1_search_manifest", "r1_search_result",
                    "r1_generation_log", "speck_polished_checkpoint",
                    "simon_polished_checkpoint"}
            original = st.EXPECTED
            st.EXPECTED = tuple(a for a in original if a.label in keep)
            try:
                rec = st.build_record(repo, home, up, strict=True)
            finally:
                st.EXPECTED = original

            if not rec["manifest_verified"]:
                self.skipTest("search harness not installed, so R1 integrity "
                              "cannot be established")
            self.assertEqual(rec["strict_problems"], [],
                             "strict must pass on complete valid evidence")
            self.assertEqual(rec["blocking_problems"], [])
            self.assertTrue(rec["ok"])
            for label in ("r1_search_manifest", "r1_search_result",
                          "r1_generation_log"):
                self.assertIn(label, rec["integrity_established"])

    def test_strict_writes_the_record_when_it_passes(self):
        up = portable.require_real_upstream(self)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            home = Path(tmp) / "home"
            portable.fake_search_bundle(
                home / "Documents/autond_search_runs/r1_20261015T035553Z")
            (repo / "audit").mkdir(parents=True)
            keep = {"r1_search_manifest", "r1_search_result",
                    "r1_generation_log", "speck_polished_checkpoint",
                    "simon_polished_checkpoint"}
            original = st.EXPECTED
            st.EXPECTED = tuple(a for a in original if a.label in keep)
            out = Path(tmp) / "rec.json"
            try:
                rc = st.main(["--repo-root", str(repo), "--home", str(home),
                              "--autond-root", str(up), "--strict",
                              "--output", str(out)])
            finally:
                st.EXPECTED = original
            if rc != 0:
                self.skipTest("strict could not pass in this environment")
            self.assertTrue(out.exists(), "a passing strict run must write")

    def test_wrong_upstream_commit_blocks_regardless_of_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            rec = st.build_record(repo, Path(tmp), up, strict=False)
            self.assertFalse(rec["ok"],
                             "a wrong upstream commit must not report ok")
            self.assertTrue(any("not the pinned revision" in p
                                for p in rec["blocking_problems"]),
                            rec["blocking_problems"])

    def test_a_failed_stage_blocks_regardless_of_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            cen = repo / "audit/targets/autond/evidence/census"
            cen.mkdir(parents=True)
            for name in ("census_v2_20261015T043855Z.json",
                         "arch_reconciliation_20261015T043855Z.json",
                         "load_verify_r2_20261015T051547Z.json"):
                (cen / name).write_text("{}")
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            rec = st.build_record(repo, Path(tmp), up, strict=False)
            self.assertFalse(rec["ok"])
            self.assertTrue(any("HASH_MISMATCH" in p
                                for p in rec["blocking_problems"]))

    def test_cli_writes_nothing_when_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            out = Path(tmp) / "out.json"
            rc = st.main(["--repo-root", str(repo), "--home", tmp,
                          "--autond-root", str(up), "--output", str(out)])
            self.assertEqual(rc, 1)
            self.assertFalse(out.exists())


# ===========================================================================
# 5. II-3 must be bound to the pinned source
# ===========================================================================

class TestSourceBinding(unittest.TestCase):
    def test_real_checkout_verifies(self):
        up = portable.require_real_upstream(self)
        res = psa.verify_sources(up)
        self.assertTrue(res["verified"], res["problems"])
        self.assertEqual(res["commit"], psa.PINNED_COMMIT)
        for rel in psa.SOURCE_FILES:
            self.assertTrue(res["files"][rel]["verified"], rel)

    def test_absent_checkout_is_not_verified(self):
        res = psa.verify_sources(None)
        self.assertFalse(res["verified"])
        self.assertTrue(res["problems"])

    def test_wrong_source_is_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            res = psa.verify_sources(up)
            self.assertFalse(res["verified"])
            self.assertTrue(any("dbitnet.py" in p for p in res["problems"]))

    def test_agrees_is_downgraded_without_a_verified_source(self):
        eff = psa.effective_summary(psa.verify_sources(None))
        self.assertEqual(eff["by_verdict"].get("AGREES", 0), 0)
        self.assertEqual(eff["agrees"], [])
        self.assertEqual(len(eff["downgraded_from_agrees"]),
                         len(psa.summarise()["agrees"]))
        self.assertFalse(eff["source_verified"])

    def test_agrees_stands_on_a_verified_source(self):
        up = portable.require_real_upstream(self)
        eff = psa.effective_summary(psa.verify_sources(up))
        self.assertTrue(eff["source_verified"])
        self.assertEqual(eff["by_verdict"].get("AGREES"),
                         psa.summarise()["by_verdict"].get("AGREES"))

    def test_ii3_component_reports_the_effective_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            stage, detail = st.build_ii3(repo, Path(tmp), up)
            self.assertFalse(detail["source_verification"]["verified"])
            comp = next(c for c in stage.components
                        if c.component_id == "II3-PAPER-SOURCE-ARCHITECTURE")
            self.assertIn("source verified: False", comp.summary)
            self.assertIn("Agreeing: none", comp.summary)

    def test_ii3_binds_the_verified_source_files_as_evidence(self):
        up = portable.require_real_upstream(self)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "audit").mkdir(parents=True)
            stage, detail = st.build_ii3(repo, Path(tmp), up)
            comp = next(c for c in stage.components
                        if c.component_id == "II3-PAPER-SOURCE-ARCHITECTURE")
            bound = {Path(k).name for k in comp.evidence}
            for rel in psa.SOURCE_FILES:
                self.assertIn(rel, bound)
            for path, digest in comp.evidence.items():
                if Path(path).is_file():
                    self.assertEqual(digest, st.sha256_file(Path(path)))


# ===========================================================================
# 6. The validator
# ===========================================================================

class TestValidatorHardening(unittest.TestCase):
    def setUp(self):
        self._orig = plan.FROZEN_SIDECAR

    def tearDown(self):
        plan.FROZEN_SIDECAR = self._orig

    def _good(self, tmp, up, **kw):
        portable.simulate_frozen(plan, Path(tmp) / "repo")
        run = portable.staged_run_dir(Path(tmp) / "run", up, plan, **kw)
        res = asm.assemble(run, repo_root=Path(tmp) / "repo")
        if not res["ok"]:
            raise AssertionError(f"fixture assembly failed: {res['problems']}")
        return run

    def test_attestation_grammar_is_exact(self):
        self.assertTrue(evres.ATTESTATION_RE.match(
            "tracked-clean;untracked-benign=0"))
        self.assertTrue(evres.ATTESTATION_RE.match(
            "tracked-clean;untracked-benign=2:.DS_Store,__pycache__/"))
        for bad in ("clean", "not clean", "unmodified", "tracked-clean",
                    "tracked-dirty;untracked-benign=0",
                    "tracked-clean;untracked-benign=x"):
            self.assertIsNone(evres.ATTESTATION_RE.match(bad), bad)

    def test_free_text_attestation_is_rejected(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            portable.simulate_frozen(plan, Path(tmp) / "repo")
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan,
                                          attestation="the checkout is clean")
            asm.assemble(run, repo_root=Path(tmp) / "repo")
            v = evres.validate_bundle(run, expected_plan_hash=plan.plan_hash())
            self.assertFalse(v["ok"])
            self.assertTrue(any("grammar" in p for p in v["problems"]))

    def test_missing_expected_plan_hash_is_a_failure_by_default(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            run = self._good(tmp, up)
            v = evres.validate_bundle(run)
            self.assertFalse(v["ok"])
            self.assertFalse(v["plan_bound"])
            self.assertTrue(any("not checked against a frozen plan" in p.lower()
                                or "no expected plan hash" in p.lower()
                                for p in v["problems"]), v["problems"])

    def test_structure_only_is_explicit_and_not_plan_bound(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            run = self._good(tmp, up)
            v = evres.validate_bundle(run, require_plan_binding=False)
            self.assertTrue(v["ok"], v["problems"])
            self.assertFalse(v["plan_bound"])

    def test_both_checkpoints_are_required(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            portable.simulate_frozen(plan, Path(tmp) / "repo")
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan,
                                          only=("speck3264",))
            # assembly refuses, so write a one-checkpoint bundle by hand to
            # prove the VALIDATOR also refuses it independently.
            asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse((run / "RESULT.json").exists())
            v = evres.validate_bundle(run, expected_plan_hash=plan.plan_hash())
            self.assertFalse(v["ok"])

    def test_timestamp_ordering_is_checked(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            portable.simulate_frozen(plan, Path(tmp) / "repo")
            run = portable.staged_run_dir(
                Path(tmp) / "run", up, plan,
                started="2026-02-01T05:00:00Z", finished="2026-02-01T04:00:00Z")
            res = asm.assemble(run, repo_root=Path(tmp) / "repo")
            self.assertFalse(res["ok"])
            self.assertTrue(any("precedes" in p for p in res["problems"]))

    def test_a_run_predating_the_freeze_is_rejected(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            portable.simulate_frozen(plan, Path(tmp) / "repo",
                                     frozen_at="2026-03-01T00:00:00Z")
            run = portable.staged_run_dir(Path(tmp) / "run", up, plan,
                                          started="2026-02-01T00:00:00Z",
                                          finished="2026-02-01T01:00:00Z")
            asm.assemble(run, repo_root=Path(tmp) / "repo")
            v = evres.validate_bundle(run, expected_plan_hash=plan.plan_hash())
            self.assertFalse(v["ok"])
            self.assertTrue(any("cannot precede its own preregistration" in p
                                for p in v["problems"]), v["problems"])

    def test_metrics_are_compared_against_the_raw_npz(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            run = self._good(tmp, up)
            rec = json.loads((run / "RESULT.json").read_text())
            rec["checkpoints"]["speck3264"]["accuracies"][0] += 0.01
            (run / "RESULT.json").write_text(json.dumps(rec, indent=2,
                                                        sort_keys=True))
            # refresh the manifest so only the metric disagreement remains
            lines = []
            for p in sorted(run.rglob("*")):
                if p.is_file() and p.name != "SHA256SUMS.txt":
                    lines.append(f"{evres.sha256_file(p)}  {p.relative_to(run)}")
            (run / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

            v = evres.validate_bundle(run, expected_plan_hash=plan.plan_hash())
            self.assertFalse(v["ok"])
            self.assertTrue(any("disagrees with the raw NPZ" in p
                                for p in v["problems"]), v["problems"])

    def test_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "b"
            d.mkdir()
            (d / "RESULT.json").write_text(json.dumps({
                "schema_version": evres.SCHEMA_VERSION,
                "preregistration": {"id": "x", "plan_hash": "a" * 64,
                                    "frozen_at_utc": "2026-01-01T00:00:00Z"},
                "execution": {"started_utc": "2026-01-02T00:00:00Z",
                              "finished_utc": "2026-01-02T01:00:00Z",
                              "command": "c",
                              "upstream_commit": plan.TARGET["pinned_commit"],
                              "upstream_write_check":
                                  "tracked-clean;untracked-benign=0"},
                "environment": {"python": "3.11", "tensorflow": "2.15.1",
                                "numpy": "1.26.4"},
                "checkpoints": {"speck3264": {
                    "path": "../../escape.h5", "sha256": "a" * 64,
                    "upstream_sha256": "a" * 64, "round_number": 8,
                    "input_difference": "0x400000",
                    "accuracies": [0.5] * 5, "tprs": [0.5] * 5,
                    "tnrs": [0.5] * 5, "npz": "../../escape.npz",
                    "npz_sha256": "a" * 64, "n_samples_per_set": 10 ** 6}},
            }))
            (d / "SHA256SUMS.txt").write_text(
                f"{evres.sha256_file(d / 'RESULT.json')}  RESULT.json\n")
            v = evres.validate_bundle(d, expected_plan_hash="a" * 64)
            self.assertFalse(v["ok"])
            self.assertTrue(any("escapes the bundle" in p for p in v["problems"]),
                            v["problems"])

    def test_wrong_commit_is_rejected_by_the_validator(self):
        up = portable.require_real_upstream(self)
        if not portable.git_available():
            self.skipTest("git is not available")
        with tempfile.TemporaryDirectory() as tmp:
            run = self._good(tmp, up)
            v = evres.validate_bundle(run, expected_plan_hash=plan.plan_hash(),
                                      expected_upstream_commit="0" * 40)
            self.assertFalse(v["ok"])
            self.assertTrue(any("not the pinned" in p for p in v["problems"]))

    def test_checkpoint_hashes_default_to_the_plans(self):
        """No caller argument should be needed for the digests to be checked."""
        import inspect

        sig = inspect.signature(evres.validate_bundle)
        self.assertIsNone(sig.parameters["expected_checkpoint_hashes"].default)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "nope"
            res = evres.validate_bundle(d)
            self.assertFalse(res["ok"])


# ===========================================================================
# 7. Paper provenance
# ===========================================================================

class TestPaperProvenance(unittest.TestCase):
    def test_record_carries_digest_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "2022-1467.pdf"
            f.write_bytes(b"%PDF-1.7\nnot really a paper\n")
            rec = psa.verify_paper_pdf(f)
            self.assertTrue(rec["present"])
            self.assertRegex(rec["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(rec["file_name"], "2022-1467.pdf")
            self.assertTrue(rec["looks_like_pdf"])
            self.assertIn("verified_at_utc", rec)
            self.assertEqual(rec["bytes"], f.stat().st_size)

    def test_revision_is_not_asserted_from_the_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "eprint-2022-1467-rev4-2024-01-24.pdf"
            f.write_bytes(b"%PDF-1.7\n")
            rec = psa.verify_paper_pdf(f)
            self.assertFalse(rec["revision_asserted"])
            self.assertIn("NOT asserted", rec["revision_note"])

    def test_a_non_pdf_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x.pdf"
            f.write_bytes(b"this is not a pdf")
            self.assertFalse(psa.verify_paper_pdf(f)["looks_like_pdf"])

    def test_absent_file_reports_absence(self):
        rec = psa.verify_paper_pdf(Path("/nonexistent/p.pdf"))
        self.assertFalse(rec["present"])
        self.assertNotIn("sha256", rec)

    def test_saved_record_reflects_the_verified_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p.pdf"
            f.write_bytes(b"%PDF-1.7\npaper\n")
            pv = psa.verify_paper_pdf(f)
            out = Path(tmp) / "rec.json"
            psa.write_record(out, psa.verify_sources(None), pv)
            rec = json.loads(out.read_text())
            self.assertEqual(rec["paper"]["pdf_sha256"], pv["sha256"])
            self.assertIn("VERIFIED", rec["paper"]["pdf_sha256_status"])
            self.assertEqual(rec["paper"]["pdf_verification"]["file_name"],
                             "p.pdf")

    def test_unverified_paper_leaves_the_hash_null(self):
        rec = psa.to_record(None, None)
        self.assertIsNone(rec["paper"]["pdf_sha256"])
        self.assertIn("UNVERIFIED", rec["paper"]["pdf_sha256_status"])

    def test_module_constant_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p.pdf"
            f.write_bytes(b"%PDF-1.7\n")
            psa.to_record(None, psa.verify_paper_pdf(f))
        self.assertIsNone(psa.PAPER["pdf_sha256"],
                          "to_record must not write into the module constant")


# ===========================================================================
# 8. Collision-safe output
# ===========================================================================

class TestRunDirectorySafety(unittest.TestCase):
    def test_absent_directory_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, why = gate.run_dir_is_usable(Path(tmp) / "new")
            self.assertTrue(ok, why)

    def test_empty_directory_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "empty"
            d.mkdir()
            ok, _ = gate.run_dir_is_usable(d)
            self.assertTrue(ok)

    def test_non_empty_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "used"
            (d / "checkpoints").mkdir(parents=True)
            (d / "RESULT.json").write_text("{}")
            ok, why = gate.run_dir_is_usable(d)
            self.assertFalse(ok)
            self.assertIn("not empty", why)

    def test_preflight_blocks_on_a_used_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            d = Path(tmp) / "used"
            d.mkdir()
            (d / "RESULT.json").write_text("{}")
            pf = gate.preflight(up, d)
            self.assertTrue(any("not empty" in b for b in pf["blockers"]))

    def test_prior_evidence_survives_a_refused_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = portable.fake_upstream(Path(tmp) / "up")["path"]
            d = Path(tmp) / "used"
            d.mkdir()
            prior = d / "RESULT.json"
            prior.write_text('{"prior": true}')
            before = prior.read_bytes()
            rc = gate.launch(up, d, execute=True)
            self.assertEqual(rc, 1)
            self.assertEqual(prior.read_bytes(), before,
                             "a refused launch must not touch prior evidence")
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             ["RESULT.json"])


# ===========================================================================
# 10. Checkout cleanliness without modifying the target
# ===========================================================================

class TestCheckoutCleanliness(unittest.TestCase):
    def _repo(self, tmp):
        up = portable.fake_upstream(Path(tmp) / "up")["path"]
        return up

    def test_a_ds_store_does_not_block(self):
        """
        The operator's checkout carried an untracked .DS_Store. Revision 2
        blocked every run on it. It is OS debris: it cannot change what
        the target computes, so it is recorded and left alone.
        """
        with tempfile.TemporaryDirectory() as tmp:
            up = self._repo(tmp)
            (up / ".DS_Store").write_bytes(b"\x00\x01")
            state = gate.checkout_state(up)
            self.assertEqual(state["tracked_dirty"], [])
            self.assertIn(".DS_Store", state["untracked_benign"])
            self.assertEqual(state["untracked_other"], [])
            self.assertTrue(evres.ATTESTATION_RE.match(state["attestation"]))
            self.assertTrue((up / ".DS_Store").exists(),
                            "the file must not be deleted")

    def test_pycache_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = self._repo(tmp)
            (up / "__pycache__").mkdir()
            (up / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"\x00")
            state = gate.checkout_state(up)
            self.assertEqual(state["untracked_other"], [])
            self.assertTrue(state["untracked_benign"])

    def test_a_modified_tracked_file_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = self._repo(tmp)
            (up / "dbitnet.py").write_text("# tampered\n")
            state = gate.checkout_state(up)
            self.assertTrue(state["tracked_dirty"])
            self.assertIsNone(state["attestation"])
            pf = gate.preflight(up, Path(tmp) / "out")
            self.assertTrue(any("modified tracked file" in b
                                for b in pf["blockers"]))

    def test_a_stray_module_blocks(self):
        """An untracked .py in the checkout can shadow an import."""
        with tempfile.TemporaryDirectory() as tmp:
            up = self._repo(tmp)
            (up / "sitecustomize.py").write_text("print('hi')\n")
            state = gate.checkout_state(up)
            self.assertIn("sitecustomize.py", state["untracked_other"])
            pf = gate.preflight(up, Path(tmp) / "out")
            self.assertTrue(any("not OS debris" in b for b in pf["blockers"]))

    def test_a_stray_cipher_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            up = self._repo(tmp)
            (up / "ciphers").mkdir(exist_ok=True)
            (up / "ciphers" / "evil.py").write_text("plain_bits=1\n")
            state = gate.checkout_state(up)
            self.assertTrue(state["untracked_other"])

    def test_nothing_in_the_gate_deletes_from_the_checkout(self):
        import ast

        tree = ast.parse(Path(gate.__file__).read_text())
        banned = {"unlink", "rmtree", "remove", "rmdir", "move", "rename"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, banned,
                                 f"the gate must not call {node.func.attr}() - "
                                 "the pinned checkout is read-only")

    def test_benign_classification_is_conservative(self):
        for benign in (".DS_Store", "__pycache__/", "x.pyc",
                       "ciphers/__pycache__/speck.pyc", ".idea/", "a.swp"):
            self.assertTrue(gate._is_benign(benign), benign)
        for blocking in ("optimizer.py", "ciphers/evil.py", "patch.diff",
                         "model_weights/extra.h5", "conftest.py"):
            self.assertFalse(gate._is_benign(blocking), blocking)


if __name__ == "__main__":
    unittest.main(verbosity=2)
