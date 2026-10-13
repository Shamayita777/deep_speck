#!/usr/bin/env python3
"""
Freeze-manifest coverage: every critical file is bound, and mutating any
one of them makes execution refuse.

WHY THIS SUITE EXISTS. The freeze mechanism worked and named drifted
files precisely - but its manifest listed 21 files and omitted
`arm_recovery.py`, `architecture_signature.py`, `run_ii4_impl.py`,
`preflight.py`, the preregistration module and the declared-architecture
record. The whole recovery and structural-verification layer could
therefore be rewritten between review and launch while `--execute`
proceeded against a manifest that still matched. A test that asserts the
manifest's CONTENTS is not enough; this suite mutates each file and
requires a refusal, so coverage is demonstrated rather than listed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import audit as _audit_pkg

from audit.implementation.ii4_comparison_runner import (
    GENERIC_COMPARISON_SOURCES,
    build_source_manifest,
)

REPO_ROOT = Path(_audit_pkg.__file__).resolve().parents[1]

#: Files whose content can change what the experiment DOES or what it
#: VERIFIES. Each must be bound, and each is mutated below.
CRITICAL = (
    "audit/implementation/arm_recovery.py",
    "audit/implementation/durable_state.py",
    "audit/implementation/architecture_signature.py",
    "audit/implementation/preflight.py",
    "audit/implementation/run_ii4_impl.py",
    "audit/implementation/preregistrations/ii4_impl_gohr_v1.py",
    "audit/implementation/adapters/gohr_declared_architecture.json",
    "audit/implementation/adapters/gohr_impl_comparison.py",
    "audit/implementation/comparison.py",
    "audit/implementation/ii4_comparison_design.py",
    "audit/implementation/ii4_comparison_runner.py",
    "audit/implementation/ii4_comparison_certificate.py",
    "audit/implementation/ii4_analysis.py",
    "audit/implementation/stages.py",
    # The real firewall: its API and its persisted schema are what the
    # consumption verifier is written against, so a change to it can
    # silently invalidate the exactly-once guarantee.
    "audit/experimental_validity/framework/firewall.py",
)


class _Adapter:
    """Minimal adapter: only `provenance_sources` matters here."""

    def provenance_sources(self):
        from audit.implementation.adapters.gohr_impl_comparison import ADAPTER_SOURCES
        return dict(ADAPTER_SOURCES)

    def environment_details(self):
        return {}


class TestManifestCoverage(unittest.TestCase):

    def setUp(self):
        self.manifest = build_source_manifest(_Adapter(), repo_root=REPO_ROOT)
        self.paths = {rec["path"] for rec in self.manifest.values()}

    def test_every_critical_file_is_bound(self):
        missing = [p for p in CRITICAL if p not in self.paths]
        self.assertEqual(missing, [], f"critical file(s) not bound by the freeze: {missing}")

    def test_the_recovery_and_signature_layer_is_bound(self):
        """Named explicitly because its omission was the review finding."""
        for path in ("audit/implementation/arm_recovery.py",
                     "audit/implementation/durable_state.py",
                     "audit/implementation/architecture_signature.py"):
            self.assertIn(path, self.paths)

    def test_the_declared_architecture_record_is_bound(self):
        """
        The JSON both arms are checked against. If it were unbound, the
        declared architecture could be swapped after review and every
        structural check would still 'pass'.
        """
        self.assertIn("audit/implementation/adapters/gohr_declared_architecture.json",
                      self.paths)

    EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_every_bound_file_carries_a_sha256_and_a_size(self):
        for label, rec in self.manifest.items():
            with self.subTest(label=label):
                self.assertEqual(len(rec["sha256"]), 64, f"{label} has no sha256")
                if rec["path"].endswith("__init__.py"):
                    # A package marker may legitimately be empty. The
                    # protection is the hash, not the size: binding the
                    # empty digest means adding a single line to it is
                    # drift. An __init__ that executes on every import is
                    # worth watching precisely because it looks inert.
                    if rec["bytes"] == 0:
                        self.assertEqual(rec["sha256"], self.EMPTY_SHA256)
                else:
                    self.assertGreater(rec["bytes"], 0, f"{label} is empty")

    def test_the_package_init_is_bound_to_its_actual_content(self):
        """
        The package `__init__` executes on every import of `audit`, so a
        change to it is a change to the experiment and must register as
        drift - whether the file is empty today or carries a docstring.

        This deliberately does NOT assert what the file contains. An
        earlier version compared the bound hash against the empty-file
        digest, which encoded one particular checkout's state into a
        test about the mechanism, and failed on any repository whose
        package header was not empty.
        """
        import hashlib
        bound = self.manifest["package_init"]["sha256"]
        on_disk = hashlib.sha256((REPO_ROOT / "audit" / "__init__.py").read_bytes())
        self.assertEqual(bound, on_disk.hexdigest(),
                         "the freeze manifest does not describe the file on disk")
        self.assertEqual(self.manifest["package_init"]["path"], "audit/__init__.py")

    def test_changing_the_package_init_would_be_drift(self):
        """
        The reason binding it matters, empty or not: any edit moves the
        hash, so the frozen record stops matching.
        """
        import hashlib
        current = (REPO_ROOT / "audit" / "__init__.py").read_bytes()
        bound = self.manifest["package_init"]["sha256"]
        # Only genuine edits: b"" is not an edit to a file that is
        # already empty, and comparing it would assert that a no-op
        # counts as drift.
        edits = [current + b"\nimport os\n", current + b" ", b"# header\n" + current]
        for edit in edits:
            self.assertNotEqual(current, edit, "this case is not actually an edit")
            with self.subTest(edit=edit[:24]):
                self.assertNotEqual(bound, hashlib.sha256(edit).hexdigest())

    def test_a_declared_file_that_is_absent_is_a_hard_error(self):
        from audit.implementation.ii4_comparison_runner import ComparisonExecutionError

        class Broken(_Adapter):
            def provenance_sources(self):
                return {"nonexistent": "audit/implementation/does_not_exist.py"}

        with self.assertRaisesRegex(ComparisonExecutionError, "refusing to build an incomplete"):
            build_source_manifest(Broken(), repo_root=REPO_ROOT)


class TestMutatingEachFileCausesRefusal(unittest.TestCase):
    """
    The real test: mutate each critical file, recompute the manifest, and
    require `check_frozen` to name that file. Mutation is done on a COPY
    of the tree, so the repository is never modified.
    """

    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "repo"
        shutil.copytree(REPO_ROOT, cls.root, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "runs", ".git"))
        cls.frozen = build_source_manifest(_Adapter(), repo_root=cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _drift(self, path: str) -> list[str]:
        """Mutate one file, recompute, and return the drifted paths."""
        target = self.root / path
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# mutated after review\n")
            current = build_source_manifest(_Adapter(), repo_root=self.root)
            before = {r["path"]: r["sha256"] for r in self.frozen.values()}
            after = {r["path"]: r["sha256"] for r in current.values()}
            return [p for p in before if before[p] != after.get(p)]
        finally:
            target.write_bytes(original)

    def test_mutating_each_critical_file_is_detected_and_named(self):
        for path in CRITICAL:
            with self.subTest(file=path):
                drifted = self._drift(path)
                self.assertIn(path, drifted,
                              f"a change to {path} was NOT detected by the freeze manifest")
                self.assertEqual(len(drifted), 1,
                                 f"mutating {path} reported unrelated drift: {drifted}")

    def test_the_runner_refuses_execution_on_each_mutation(self):
        """
        End-to-end: `check_frozen` must produce a refusal that names the
        file and demands a new freeze.
        """
        from audit.implementation.ii4_comparison_runner import ComparisonRunner

        class _Plan:
            class _Axis:
                arm_ids = ("reference", "independent")
            axis = _Axis()
            def preregistration_hash(self):
                return "p" * 64
            def config_fingerprint(self):
                return "c" * 64

        frozen_record = {
            "config_fingerprint": "c" * 64,
            "preregistration_sha256": "p" * 64,
            "declared_architecture_digest": None,
            "source_manifest": self.frozen,
        }

        class _Runner(ComparisonRunner):
            def runtime_provenance(self):
                return {"config_fingerprint": "c" * 64,
                        "source_manifest": self._manifest_override}

        for path in ("audit/implementation/arm_recovery.py",
                     "audit/implementation/architecture_signature.py",
                     "audit/implementation/run_ii4_impl.py",
                     "audit/implementation/preflight.py",
                     "audit/implementation/preregistrations/ii4_impl_gohr_v1.py",
                     "audit/implementation/adapters/gohr_declared_architecture.json"):
            with self.subTest(file=path):
                target = self.root / path
                original = target.read_bytes()
                try:
                    target.write_bytes(original + b"\n# mutated\n")
                    runner = _Runner(plan=_Plan(), adapter=_Adapter())
                    runner._manifest_override = build_source_manifest(
                        _Adapter(), repo_root=self.root)
                    problems = runner.check_frozen(frozen_record)
                finally:
                    target.write_bytes(original)
                self.assertTrue(any(path in p for p in problems),
                                f"check_frozen did not name {path}: {problems}")
                self.assertTrue(any("NEW FREEZE" in p for p in problems),
                                "the refusal does not demand a new freeze")


# =====================================================================
# INDEPENDENT derivation: do not trust the list, re-derive it
# =====================================================================

#: The real entry points. Everything the experiment can execute is
#: reachable from one of these.
ENTRY_POINTS = (
    "audit/implementation/run_ii4_impl.py",
    "audit/implementation/ii4_comparison_runner.py",
    "audit/implementation/adapters/gohr_impl_comparison.py",
    "audit/implementation/preregistrations/ii4_impl_gohr_v1.py",
    "audit/implementation/ii4_comparison_certificate.py",
)


def import_closure(root: Path, entries=ENTRY_POINTS) -> set[str]:
    """
    Every module inside the `audit` package reachable from `entries`,
    derived by parsing imports rather than by reading any manifest.

    This is deliberately independent of GENERIC_COMPARISON_SOURCES: its
    whole purpose is to disagree with it if the manifest is incomplete.
    """
    import ast as _ast

    def resolve(module: str):
        direct = root / (module.replace(".", "/") + ".py")
        if direct.is_file():
            return direct
        pkg = root / module.replace(".", "/") / "__init__.py"
        return pkg if pkg.is_file() else None

    seen: set[str] = set()
    queue = list(entries)
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        path = root / rel
        if not path.is_file():
            continue
        seen.add(rel)
        for node in _ast.walk(_ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, _ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, _ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                if not name.startswith("audit"):
                    continue
                found = resolve(name)
                if found is not None:
                    queue.append(str(found.relative_to(root)))
    return seen


class TestTheManifestMatchesAnIndependentDerivation(unittest.TestCase):
    """
    The review question was: is the manifest complete, or does it merely
    pass its own tests? So the required set is DERIVED from the import
    graph here and compared with the manifest. When this was first run
    it found five reachable files unbound - audit/__init__.py,
    common/ids.py, conformance.py, ii4_design.py and ii4_experiment.py -
    which is why it exists as a standing test rather than a one-off
    check.
    """

    def setUp(self):
        self.bound = {rec["path"] for rec in
                      build_source_manifest(_Adapter(), repo_root=REPO_ROOT).values()}
        self.derived = import_closure(REPO_ROOT)

    def test_the_derivation_reaches_a_plausible_number_of_modules(self):
        """A broken walker that finds nothing would make this suite vacuous."""
        self.assertGreater(len(self.derived), 15,
                           f"the import walker found only {len(self.derived)} modules; "
                           f"it is probably broken, which would make the comparison "
                           f"below meaningless")
        self.assertIn("audit/implementation/ii4_comparison_runner.py", self.derived)
        self.assertIn("audit/implementation/durable_state.py", self.derived)

    def test_every_module_in_the_import_graph_is_bound(self):
        unbound = sorted(self.derived - self.bound)
        self.assertEqual(
            unbound, [],
            f"{len(unbound)} file(s) are reachable from the entry points but NOT bound "
            f"by the freeze manifest, so they could change between review and launch "
            f"without drift detection noticing: {unbound}")

    def test_the_five_files_found_by_the_audit_are_bound(self):
        """Named, so a future edit cannot quietly drop them again."""
        for path in ("audit/__init__.py",
                     "audit/common/ids.py",
                     "audit/implementation/conformance.py",
                     "audit/implementation/ii4_design.py",
                     "audit/implementation/ii4_experiment.py"):
            self.assertIn(path, self.bound)

    def test_the_package_init_is_bound_even_though_it_is_empty(self):
        """
        It executes on every import. An empty file nobody watches is the
        cheapest place to put code.
        """
        self.assertIn("audit/__init__.py", self.bound)

    def test_every_required_test_module_is_bound(self):
        """
        A weakened test cannot change what the experiment DOES, but it
        changes what the clearance means - and the gate runs the tests
        fresh, so only the freeze can notice an edit.
        """
        from audit.implementation.run_ii4_impl import REQUIRED_SUITES
        for suite in REQUIRED_SUITES:
            rel = suite.replace(".", "/") + ".py"
            self.assertIn(rel, self.bound,
                          f"{rel} authorizes the launch but is not bound by the freeze")

    def test_the_files_bound_beyond_the_graph_are_accounted_for(self):
        """
        Extra files are fine - dynamically imported modules, data files
        and the reference sources never appear in a static import graph -
        but each must be explainable, not accidental.
        """
        allowed_extra_prefixes = (
            "audit/tests/",                                   # verification sources
            "audit/implementation/reference/",                # Gohr's own sources
            "audit/implementation/ii2_independent/",          # the independent arm
            "audit/experimental_validity/framework/",         # imported via importlib
        )
        allowed_exact = {
            "audit/implementation/adapters/gohr_declared_architecture.json",
            "audit/implementation/ii4_comparison_power.py",
        }
        unexplained = [p for p in sorted(self.bound - self.derived)
                       if p not in allowed_exact
                       and not p.startswith(allowed_extra_prefixes)]
        self.assertEqual(unexplained, [],
                         f"bound files with no stated reason: {unexplained}")


class TestDriftDetectionCannotBeBypassed(unittest.TestCase):

    def setUp(self):
        self.manifest = build_source_manifest(_Adapter(), repo_root=REPO_ROOT)

    def _runner(self):
        from audit.implementation.ii4_comparison_runner import ComparisonRunner

        class _Plan:
            class _Axis:
                arm_ids = ("reference", "independent")
            axis = _Axis()
            def preregistration_hash(self):
                return "p" * 64
            def config_fingerprint(self):
                return "c" * 64

        class _Runner(ComparisonRunner):
            def runtime_provenance(self):
                return {"config_fingerprint": "c" * 64,
                        "source_manifest": self._manifest_override}

        runner = _Runner(plan=_Plan(), adapter=_Adapter())
        return runner

    def _frozen(self, manifest):
        return {"config_fingerprint": "c" * 64, "preregistration_sha256": "p" * 64,
                "declared_architecture_digest": None, "source_manifest": manifest}

    def test_a_changed_file_is_reported_with_frozen_and_current_hashes(self):
        label = "durable_state"
        tampered = {k: dict(v) for k, v in self.manifest.items()}
        original = tampered[label]["sha256"]
        tampered[label]["sha256"] = "f" * 64
        runner = self._runner()
        runner._manifest_override = self.manifest
        problems = runner.check_frozen(self._frozen(tampered))
        joined = "\n".join(problems)
        self.assertIn("CHANGED since the freeze", joined)
        self.assertIn("f" * 64, joined, "the frozen hash is not reported")
        self.assertIn(original, joined, "the current hash is not reported")
        self.assertIn("audit/implementation/durable_state.py", joined)

    def test_a_file_REMOVED_from_the_manifest_is_detected(self):
        """
        A rename shows up as one removal plus one addition, so this is
        also the rename case.
        """
        tampered = {k: v for k, v in self.manifest.items() if k != "arm_recovery"}
        runner = self._runner()
        runner._manifest_override = self.manifest
        problems = runner.check_frozen(self._frozen(tampered))
        self.assertTrue(any("ADDED since the freeze" in p and "arm_recovery.py" in p
                            for p in problems), problems)

    def test_a_file_ADDED_since_the_freeze_is_detected(self):
        runner = self._runner()
        extra = {**self.manifest,
                 "smuggled": {"path": "audit/implementation/smuggled.py",
                              "sha256": "a" * 64, "bytes": 10}}
        runner._manifest_override = extra
        problems = runner.check_frozen(self._frozen(self.manifest))
        self.assertTrue(any("ADDED since the freeze" in p and "smuggled.py" in p
                            for p in problems), problems)

    def test_a_declared_file_that_vanishes_refuses_rather_than_omitting_it(self):
        """
        The dangerous failure: a manifest that silently drops a missing
        file would keep the experiment fingerprint stable while the code
        that produced the result was gone.
        """
        from audit.implementation.ii4_comparison_runner import ComparisonExecutionError

        class Vanished(_Adapter):
            def provenance_sources(self):
                return {"gone": "audit/implementation/this_was_deleted.py"}

        with self.assertRaisesRegex(ComparisonExecutionError,
                                    "refusing to build an incomplete"):
            build_source_manifest(Vanished(), repo_root=REPO_ROOT)

    def test_every_refusal_demands_a_new_freeze(self):
        tampered = {k: dict(v) for k, v in self.manifest.items()}
        tampered["stages"]["sha256"] = "e" * 64
        runner = self._runner()
        runner._manifest_override = self.manifest
        problems = runner.check_frozen(self._frozen(tampered))
        self.assertTrue(any("NEW FREEZE" in p for p in problems),
                        f"no refusal demanded a new freeze: {problems}")


if __name__ == "__main__":
    unittest.main(verbosity=2)