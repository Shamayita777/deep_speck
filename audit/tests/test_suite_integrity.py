#!/usr/bin/env python3
"""
Meta-tests on the test suite itself.

WHY THIS SUITE EXISTS. Twice during this work, tests were appended to a
module AFTER its `if __name__ == "__main__": unittest.main()` block. The
code still executes on import, so `python -m unittest <module>` collects
it - but running the file directly calls `unittest.main()` before those
classes exist, and the stranded tests are silently skipped. In one file
this hid ten crash-continuation tests and in another nine certificate
tests, and in both cases the run reported OK. A green run that quietly
omits a fifth of its assertions is worse than a red one.

The second check is the mirror image: a suite can exist, pass, and never
be consulted by the launch gate. `REQUIRED_SUITES` is what `--execute`
actually verifies, so a test module absent from it does not block
anything. Every module under `audit/tests` must either be listed or be
named here as a deliberate exclusion, with a reason.

Neither check tests the experiment. They test whether the evidence for
the experiment is real.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from audit.implementation.run_ii4_impl import REQUIRED_SUITES

TESTS_DIR = Path(__file__).resolve().parent

#: Modules deliberately outside the launch gate, each with its reason.
EXCLUDED: dict[str, str] = {
    "__init__": "package marker, holds no tests",
    "kill_worker": (
        "a subprocess ENTRY POINT, not a suite: it is launched by the "
        "crash tests and kills itself with SIGKILL, so collecting it as "
        "a test module would terminate the runner"
    ),
    # --- pre-existing suites, excluded from the II-4-GOHR-IMPL-V1 gate -----
    #
    # These cover other pillars and earlier phases of the framework. They
    # predate this experiment and were NOT part of its audited readiness
    # gate, so they must not be able to authorize - or block - a 53-84
    # GPU-hour confirmatory run whose evidence they do not bear on.
    #
    # Excluded is not "ignored": they should still pass, and they are the
    # check that installing this experiment did not break anything else.
    # Run them separately, not as part of --execute.
    "conftest": (
        "pytest configuration, not a suite: it places the "
        "experimental_validity root on sys.path for pytest runs"
    ),
    "phase5_integrity_gate": (
        "a standalone verification SCRIPT with no TestCase - it re-derives "
        "the Phase-5 documents from artifacts on disk and exits non-zero; "
        "run it directly, not through the launch gate"
    ),
    "test_common": "pre-existing: the shared audit/common modules",
    "test_ii4": "pre-existing: the II-4 conformance axis (the depth experiment)",
    "test_ii4_gohr_smoke": "pre-existing: the Gohr conformance smoke run",
    "test_ii4_provenance": "pre-existing: provenance for the conformance II-4",
    "test_ii4_runner": "pre-existing: the frozen conformance runner",
    "test_implementation": "pre-existing: Implementation Integrity stages overall",
    "test_implementation_five_stage": (
        "pre-existing: the five-stage Implementation Integrity pillar "
        "(II-1 to II-5) as a whole"
    ),
    "test_m1_independent": "pre-existing: the II-2 independent reimplementation",
    "test_phase4_integration": "pre-existing: Phase-4 integration",
    "test_phase5_derivation": "pre-existing: Phase-5 derivation",
    "test_phase5_ledger": "pre-existing: Phase-5 ledger",
    "test_phase6_adversarial": "pre-existing: Phase-6 adversarial checks",
    "test_replication_consistency": "pre-existing: replication consistency",
}


#: Suites that belong to the host repository rather than to this
#: package. Absent from the standalone package, present after the merge.
PRE_EXISTING = frozenset({
    "conftest", "phase5_integrity_gate", "test_common", "test_ii4",
    "test_ii4_gohr_smoke", "test_ii4_provenance", "test_ii4_runner",
    "test_implementation", "test_implementation_five_stage",
    "test_m1_independent", "test_phase4_integration",
    "test_phase5_derivation", "test_phase5_ledger", "test_phase6_adversarial",
    "test_replication_consistency",
})


def _modules() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("*.py"))


class TestNoStrandedTests(unittest.TestCase):
    """
    The `__main__` guard must be the LAST module-level statement, so that
    direct execution and `-m unittest` collect exactly the same tests.
    """

    def _guard_index(self, tree: ast.Module):
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
                return i
        return None

    def test_the_entry_point_is_the_last_statement_in_every_module(self):
        for path in _modules():
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                guard = self._guard_index(tree)
                if guard is None:
                    continue                      # no entry point, nothing to strand
                after = tree.body[guard + 1:]
                described = [getattr(n, "name", type(n).__name__) for n in after]
                self.assertEqual(
                    after, [],
                    f"{path.name} has module-level code after its __main__ guard: "
                    f"{described}. Running the file directly would skip it silently. "
                    f"Move `if __name__ == \"__main__\"` to the end of the file.")

    def test_no_module_defines_the_same_test_class_twice(self):
        """
        A re-defined class replaces the first silently, so its tests
        vanish from the run without any error.
        """
        for path in _modules():
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
                duplicates = sorted({n for n in names if names.count(n) > 1})
                self.assertEqual(duplicates, [],
                                 f"{path.name} defines {duplicates} more than once; "
                                 f"the earlier definition's tests are discarded")

    def test_no_class_defines_the_same_test_method_twice(self):
        for path in _modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                methods = [m.name for m in cls.body
                           if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and m.name.startswith("test")]
                duplicates = sorted({m for m in methods if methods.count(m) > 1})
                with self.subTest(module=path.name, cls=cls.name):
                    self.assertEqual(duplicates, [],
                                     f"{path.name}:{cls.name} defines {duplicates} twice")


class TestEveryModuleIsGated(unittest.TestCase):

    def test_every_test_module_is_required_or_excluded_with_a_reason(self):
        listed = {name.rsplit(".", 1)[-1] for name in REQUIRED_SUITES}
        ungated = []
        for path in _modules():
            stem = path.stem
            if stem in listed or stem in EXCLUDED:
                continue
            ungated.append(stem)
        self.assertEqual(
            ungated, [],
            f"test module(s) {ungated} are neither in REQUIRED_SUITES nor listed in "
            f"EXCLUDED. They can pass or rot without blocking a launch; add them to "
            f"the gate, or to EXCLUDED with the reason they do not belong there.")

    def test_every_required_suite_exists_and_is_importable(self):
        import importlib
        for name in REQUIRED_SUITES:
            with self.subTest(suite=name):
                module = importlib.import_module(name)
                classes = [v for v in vars(module).values()
                           if isinstance(v, type) and issubclass(v, unittest.TestCase)]
                self.assertTrue(classes, f"{name} declares no TestCase")

    def test_every_required_suite_contributes_at_least_one_test(self):
        loader = unittest.TestLoader()
        for name in REQUIRED_SUITES:
            with self.subTest(suite=name):
                count = loader.loadTestsFromName(name).countTestCases()
                self.assertGreater(count, 0, f"{name} contributes no tests to the gate")

    def test_the_crash_recovery_suites_are_gated(self):
        """
        Named explicitly: these three carry the SIGKILL evidence for the
        recovery, firewall-persistence and sealed-set claims. If any one
        of them is dropped from the gate, `--execute` would proceed with
        those claims unmeasured.
        """
        for name in ("audit.tests.test_durable_state",
                     "audit.tests.test_firewall_crash_recovery",
                     "audit.tests.test_ii4_interruption"):
            self.assertIn(name, REQUIRED_SUITES)

    def test_the_launcher_gate_suites_are_gated(self):
        """
        Named explicitly because they are the gates ON the launcher.
        `--execute` enforces readiness by calling `readiness_gate`; if
        the suites that prove it refuses correctly were themselves
        outside the gate, the enforcement could be removed and every
        remaining test would still pass.
        """
        for name in ("audit.tests.test_preflight_filesystem",
                     "audit.tests.test_execute_gate"):
            self.assertIn(name, REQUIRED_SUITES)

    def test_the_launcher_still_enforces_both_gates(self):
        """
        Structural check on the launcher rather than on a test file: the
        execute branch must call the readiness gate, and the gate must
        consult both the test runner and the preflight. This is the
        defect that was found in review - a gate that existed only in
        `--dry-run` - so it is asserted where it cannot be edited away
        quietly.
        """
        import inspect
        from audit.implementation import run_ii4_impl as cli

        gate = inspect.getsource(cli.readiness_gate)
        self.assertIn("verify_required_suites()", gate)
        self.assertIn("run_preflight(", gate)
        self.assertIn("check_output_dir(", gate)

        main = inspect.getsource(cli.main)
        execute_branch = main[main.index("# --execute"):]
        self.assertIn("readiness_gate(", execute_branch)
        self.assertIn("return 1", execute_branch)

    def test_the_filesystem_policy_tables_are_disjoint(self):
        """
        A type in two tables would make its verdict depend on check
        order rather than on policy.
        """
        from audit.implementation.preflight import (
            DURABLE_FILESYSTEMS, EPHEMERAL_FILESYSTEMS, REJECTED_FILESYSTEMS,
        )
        durable = set(DURABLE_FILESYSTEMS)
        rejected = set(REJECTED_FILESYSTEMS)
        ephemeral = set(EPHEMERAL_FILESYSTEMS)
        self.assertEqual(durable & rejected, set())
        self.assertEqual(durable & ephemeral, set())
        self.assertEqual(rejected & ephemeral, set())

    def test_the_exclusion_list_only_names_modules_that_exist(self):
        """
        An exclusion for a module that is not present is dead policy. The
        pre-existing suites are exempt from this check: they live in the
        repository this experiment is installed INTO, so they are absent
        from the standalone package and present after the merge, and both
        states are correct.
        """
        present = {p.stem for p in _modules()}
        stale = sorted(set(EXCLUDED) - present - PRE_EXISTING)
        self.assertEqual(stale, [],
                         f"EXCLUDED names modules that no longer exist: {stale}")

    def test_no_pre_existing_suite_was_overwritten_by_this_package(self):
        """
        The merge must be additive. If a module this package ships ever
        collided with a pre-existing name, the older suite would be
        silently replaced - so the two name sets must stay disjoint.
        """
        shipped = {name.rsplit(".", 1)[-1] for name in REQUIRED_SUITES} | {"kill_worker"}
        self.assertEqual(shipped & PRE_EXISTING, set(),
                         "a shipped test module has the same name as a pre-existing one")


if __name__ == "__main__":
    unittest.main(verbosity=2)