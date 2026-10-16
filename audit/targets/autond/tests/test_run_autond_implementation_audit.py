"""
Tests for the AutoND Implementation Integrity runner.

These tests assert the properties that make the certificate trustworthy:

  * the runner uses the GENERIC framework entry point and the generic
    stage vocabulary; it does not invent its own;
  * all five CORE stages appear in the certificate, and the three that
    were not performed are recorded as NOT_PERFORMED with a reason
    rather than omitted;
  * missing evidence cannot become a satisfied stage;
  * the evidence tag is EXPLORATORY while the successor preregistration
    is unfrozen, and cannot be forced to CONFIRMATORY;
  * the runner trains nothing, searches nothing and writes nothing in
    dry-run mode;
  * the write path refuses to overwrite existing evidence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit.common.outcomes import EvidenceTag
from audit.implementation.stages import (
    AssessmentTier,
    STAGE_TIERS,
    StageId,
    StageOutcome,
    validate_stage_record,
)
from audit.targets.autond import run_autond_implementation_audit as runner
from audit.targets.autond.preregistrations import autond_ii_v2 as plan

from audit.targets.autond.tests import portable

REPO_ROOT = portable.repo_root()


def _roots():
    """
    Resolve the checkout portably.

    Uses the operator's real pinned checkout when AUTOND_ROOT points at
    one, and otherwise a synthetic stand-in created per test session, so
    these tests run on any machine. No path is hardcoded.
    """
    real = portable.real_upstream()
    if real is not None:
        return real, Path.home()
    global _FAKE_DIR
    if _FAKE_DIR is None:
        import tempfile
        _FAKE_DIR = tempfile.mkdtemp(prefix="autond-fake-upstream-")
        portable.fake_upstream(Path(_FAKE_DIR) / "up")
    return Path(_FAKE_DIR) / "up", Path(_FAKE_DIR)


_FAKE_DIR = None


def _assemble():
    upstream, home = _roots()
    return runner.assemble(REPO_ROOT, home, upstream)


class TestAllFiveStagesPresent(unittest.TestCase):
    def test_every_stage_id_appears_exactly_once(self):
        _, ctx = _assemble()
        ids = [s.stage_id for s in ctx["stages"]]
        self.assertEqual(len(ids), 5)
        self.assertEqual(set(ids), set(StageId))
        self.assertEqual(len(set(ids)), 5, "a stage id is duplicated")

    def test_all_five_stages_are_core(self):
        """If the framework ever downgrades a stage, this test must fail loudly."""
        _, ctx = _assemble()
        for stage in ctx["stages"]:
            self.assertIs(
                stage.tier,
                AssessmentTier.CORE,
                f"{stage.stage_id.value} is not CORE; tiering came from STAGE_TIERS",
            )
            self.assertIs(stage.tier, STAGE_TIERS[stage.stage_id])

    def test_unperformed_stages_are_not_performed_not_omitted(self):
        _, ctx = _assemble()
        by_id = {s.stage_id: s for s in ctx["stages"]}
        for stage_id in runner.NOT_PERFORMED:
            stage = by_id[stage_id]
            self.assertIs(stage.status, StageOutcome.NOT_PERFORMED)
            self.assertFalse(stage.is_satisfied)

    def test_unperformed_stages_carry_a_reason(self):
        _, ctx = _assemble()
        by_id = {s.stage_id: s for s in ctx["stages"]}
        for stage_id, reason in runner.NOT_PERFORMED.items():
            stage = by_id[stage_id]
            self.assertEqual(stage.summary, reason)
            self.assertTrue(stage.findings, "a NOT_PERFORMED stage must say why")
            self.assertIn(stage_id.value, stage.findings[0])

    def test_unperformed_stages_have_no_components(self):
        """A stage that was not performed must not cite evidence."""
        _, ctx = _assemble()
        by_id = {s.stage_id: s for s in ctx["stages"]}
        for stage_id in runner.NOT_PERFORMED:
            self.assertEqual(by_id[stage_id].components, ())


class TestStageRecordsAreFrameworkValid(unittest.TestCase):
    def test_generic_validator_accepts_every_stage(self):
        _, ctx = _assemble()
        for stage in ctx["stages"]:
            self.assertEqual(
                validate_stage_record(stage), [], f"{stage.stage_id.value} rejected"
            )

    def test_assemble_raises_if_a_stage_record_is_invalid(self):
        """The runner must refuse to continue on an invalid stage record."""
        import audit.targets.autond.implementation_stages as st

        original = st.build_ii1

        def broken(repo, home, upstream):
            stage, detail = original(repo, home, upstream)
            bad = type(stage)(
                stage_id=stage.stage_id,
                tier=stage.tier,
                status=stage.status,
                summary=stage.summary,
                findings=stage.findings,
                components=tuple(
                    type(c)(
                        component_id=c.component_id,
                        kind=c.kind,
                        status=c.status,
                        question=c.question,
                        summary=c.summary,
                        evidence={"some/path": "not-a-sha256"},
                    )
                    for c in stage.components[:1]
                ),
            )
            return bad, detail

        st.build_ii1 = broken
        try:
            with self.assertRaises(RuntimeError) as cm:
                _assemble()
            self.assertIn("rejected by the generic validator", str(cm.exception))
        finally:
            st.build_ii1 = original


class TestMissingEvidenceCannotPass(unittest.TestCase):
    def test_no_stage_is_satisfied_while_required_artifacts_are_missing(self):
        _, ctx = _assemble()
        missing_required = [
            r for r in ctx["artifact_report"]
            if r["required"] and r["verdict"] != "VERIFIED"
        ]
        if not missing_required:
            self.skipTest("all required artifacts are present on this machine")
        for stage in ctx["stages"]:
            self.assertNotIn(
                stage.status,
                (StageOutcome.COMPLETED, StageOutcome.CONFORMANCE_VERIFIED),
                f"{stage.stage_id.value} claims completion with required evidence missing",
            )

    def test_missing_artifacts_are_reported_as_unavailable(self):
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        unavailable = " ".join(evidence.limitations or [])
        for rec in ctx["artifact_report"]:
            if rec["verdict"] == "MISSING":
                self.assertIn(rec["label"], unavailable)

    def test_only_verified_artifacts_are_bound_as_hashes(self):
        _, ctx = _assemble()
        verified = {
            r["label"] for r in ctx["artifact_report"] if r["verdict"] == "VERIFIED"
        }
        self.assertEqual(set(ctx["artifact_hashes"]), verified)
        for label, digest in ctx["artifact_hashes"].items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class TestEvidenceTagAndDecision(unittest.TestCase):
    def test_tag_is_exploratory_because_no_preregistration_is_passed(self):
        """
        The tag is EXPLORATORY because this runner passes
        preregistration_reference=None, which is a property of the runner
        and NOT of whether the operator has frozen the plan. The previous
        version asserted the plan was unfrozen, so a correct freeze broke
        it - and worse, it implied the tag would flip on freezing, which
        it does not: the pillar assembly itself was never preregistered.
        """
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        self.assertIs(evidence.tag, EvidenceTag.EXPLORATORY)
        self.assertIn("preregistration_reference=None",
                      Path(runner.__file__).read_text())

    def test_runner_passes_no_preregistration_reference(self):
        """Confirmatory status must not be manufactured after the fact."""
        src = Path(runner.__file__).read_text()
        self.assertIn("preregistration_reference=None", src)

    def test_decision_is_not_a_pass_while_core_stages_unsatisfied(self):
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        unsatisfied = [s for s in ctx["stages"] if not s.is_satisfied]
        self.assertTrue(unsatisfied, "fixture assumes some stage is unsatisfied")
        self.assertNotIn(str(evidence.decision).upper(), ("PASS", "CONDITIONAL_PASS"))

    def test_independent_reimplementation_recorded_not_assessed(self):
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        payload = json.dumps(evidence.to_dict(), sort_keys=True)
        self.assertIn("NOT_ASSESSED", payload)


class TestPreregistrationGateIsReported(unittest.TestCase):
    def test_gate_state_is_read_not_asserted(self):
        """
        The context must carry what the plan actually reports. Compared
        with the SAME arguments assemble() uses - it passes the checkout
        and the repo root, and a bare call omits both, which made this
        test disagree with itself once the plan was frozen.
        """
        upstream, home = _roots()
        _, ctx = _assemble()
        self.assertEqual(ctx["prereg_frozen"],
                         plan.is_frozen(REPO_ROOT)[0])
        self.assertEqual(ctx["prereg_permitted"],
                         plan.confirmatory_run_permitted(upstream, REPO_ROOT)[0])

    def test_certificate_records_the_plan_hash_and_unverified_fields(self):
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        gate = evidence.observation["preregistration_gate"]
        self.assertEqual(gate["plan_hash"], plan.plan_hash())
        # The certificate must REPORT the gate state faithfully, not assert
        # a particular one: the operator may legitimately have frozen the
        # plan. What matters is that the recorded values are the live ones.
        self.assertEqual(gate["frozen"], ctx["prereg_frozen"])
        self.assertEqual(gate["confirmatory_run_permitted"],
                         ctx["prereg_permitted"])
        self.assertIsInstance(gate["unverified_fields"], list)
        if not gate["frozen"]:
            self.assertFalse(gate["confirmatory_run_permitted"],
                             "an unfrozen plan can never permit a run")


class TestPublishedDiscrepancyPreserved(unittest.TestCase):
    def test_simon_discrepancy_is_carried_into_the_certificate(self):
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        obs = evidence.observation["published_value_discrepancy"]
        self.assertEqual(obs["values"], plan.PUBLISHED_VALUES)

    def test_architecture_comparison_digest_is_bound(self):
        from audit.targets.autond import paper_source_architecture as arch

        _, ctx = _assemble()
        self.assertEqual(ctx["arch_digest"], arch.record_digest())
        self.assertRegex(ctx["arch_digest"], r"^[0-9a-f]{64}$")


class TestRunnerExecutesNoExperiment(unittest.TestCase):
    """
    The runner must not train, search or shell out. Checked against the
    parsed AST, so prose in docstrings that merely NAMES these things
    (for example when describing outstanding work) does not trip it.
    """

    # Names that would mean "an experiment ran here". Checked as calls,
    # with functions this module defines itself excluded, so the runner's
    # own run() is not mistaken for subprocess.run().
    FORBIDDEN_CALLS = frozenset({
        "fit", "evo", "optimize", "train_distinguisher", "train_nets",
        "system", "run", "Popen", "call", "check_output", "check_call",
        "eval", "exec", "compile",
    })
    FORBIDDEN_IMPORTS = frozenset({
        "subprocess", "tensorflow", "keras", "torch", "os.system",
    })

    def _tree(self):
        import ast

        return ast.parse(Path(runner.__file__).read_text())

    def test_runner_makes_no_training_or_search_call(self):
        import ast

        tree = self._tree()
        own = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name, qualified = func.attr, True
            elif isinstance(func, ast.Name):
                name, qualified = func.id, False
            else:
                continue
            if name not in self.FORBIDDEN_CALLS:
                continue
            # A bare call to a function this module defines is its own code.
            if not qualified and name in own:
                continue
            offenders.append(f"{name}() at line {node.lineno}")
        self.assertEqual(
            offenders, [], f"runner must execute no experiment; found {offenders}"
        )

    def test_runner_imports_no_training_stack(self):
        import ast

        offenders = []
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.FORBIDDEN_IMPORTS:
                        offenders.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in self.FORBIDDEN_IMPORTS:
                    offenders.append(node.module)
        self.assertEqual(
            offenders, [], f"runner must not import a training stack; found {offenders}"
        )

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence"
            rc = runner.run(REPO_ROOT, _roots()[1], _roots()[0], out, write=False)
            self.assertEqual(rc, 0)
            self.assertFalse(out.exists(), "dry run created the output directory")


class TestWritePath(unittest.TestCase):
    def test_write_emits_manifest_registry_and_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence"
            rc = runner.run(REPO_ROOT, _roots()[1], _roots()[0], out, write=True)
            self.assertEqual(rc, 0)
            names = sorted(p.name for p in out.iterdir())
            self.assertIn(f"run_manifest_{runner.RUN_ID}.json", names)
            self.assertIn("version_registry.json", names)
            certs = [n for n in names if n.startswith("autond-implementation-certificate")]
            self.assertEqual(len(certs), 1, f"expected one certificate, got {names}")

            cert = json.loads((out / certs[0]).read_text())
            stage_ids = [s["stage_id"] for s in cert["observation"]["stages"]]
            self.assertEqual(set(stage_ids), {s.value for s in StageId})

    def test_write_refuses_to_overwrite_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence"
            self.assertEqual(runner.run(REPO_ROOT, _roots()[1], _roots()[0], out, write=True), 0)
            before = (out / f"run_manifest_{runner.RUN_ID}.json").read_bytes()
            rc = runner.run(REPO_ROOT, _roots()[1], _roots()[0], out, write=True)
            self.assertEqual(rc, 1, "second write must refuse, not overwrite")
            after = (out / f"run_manifest_{runner.RUN_ID}.json").read_bytes()
            self.assertEqual(before, after, "existing manifest was modified")


class TestProvenanceDeclaresFrameworkUnmodified(unittest.TestCase):
    def test_provenance_names_the_generic_entry_point(self):
        _, ctx = _assemble()
        manifest_ref = {
            "run_id": runner.RUN_ID,
            "artifact_path": "x.json",
            "content_hash": "0" * 64,
        }
        evidence = runner.build_evidence(manifest_ref, ctx)
        prov = evidence.provenance
        self.assertEqual(
            prov["framework_entry_point"],
            "audit.implementation.certificate.build_implementation_evidence",
        )
        self.assertFalse(prov["framework_modified_for_this_target"])
        self.assertEqual(prov["target"], "AutoND")

    def test_manifest_binds_framework_and_adapter_code_hashes(self):
        manifest, _ = _assemble()
        code = manifest.to_dict()["code_hashes"]
        self.assertIn("audit/implementation/stages.py", code)
        self.assertIn("audit/implementation/certificate.py", code)
        self.assertIn("audit/targets/autond/implementation_stages.py", code)
        self.assertIn(
            "audit/targets/autond/run_autond_implementation_audit.py", code
        )
        for digest in code.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_assemble_refuses_when_bound_code_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as cm:
                runner.assemble(Path(tmp), _roots()[1], _roots()[0])
            self.assertIn("missing", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
