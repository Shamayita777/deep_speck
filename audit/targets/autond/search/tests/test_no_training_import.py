#!/usr/bin/env python3
"""
Proof that the search-only entry point reaches no training code.

TWO INDEPENDENT PROOFS, BOTH AGAINST THE REAL PINNED SOURCE. No fixture
cipher appears in this file. Nothing here is offered as evidence that
AutoND was reproduced; it is evidence about which modules an import
graph contains, which is a different and much smaller claim.

  STATIC   walk the import graph of optimizer.py and ciphers/speck3264.py
           with `ast`, transitively through the checkout, and assert that
           no training module is reachable. This holds without executing
           anything.

  DYNAMIC  in a FRESH SUBPROCESS, import the search-only entry point and
           bind the real Speck32/64 cipher, then assert sys.modules. A
           subprocess is used because this test session has already
           imported other things; a claim about what the search path
           loads can only be made in a process that loaded nothing else.

  CONTROL  the same static walk from main.py MUST reach tensorflow. Without
           this, the two proofs above would be consistent with the
           analyser simply never finding anything, and would establish
           nothing.

Needs the checkout:

    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream

Fails rather than skips when it is absent.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import search_only                                                  # noqa: E402

TRAINING = {"tensorflow", "keras", "train_nets", "polish_nets",
            "gohrnet", "dbitnet", "eval_nets", "main"}


def _root() -> Path:
    raw = os.environ.get("AUTOND_UPSTREAM")
    if not raw:
        raise AssertionError(
            "AUTOND_UPSTREAM is not set, so the pinned source cannot be analysed "
            "and these proofs did not run. This is a FAILURE, not a skip.\n"
            "    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream")
    root = Path(raw).expanduser().resolve()
    if not (root / "optimizer.py").is_file():
        raise AssertionError(f"{root} does not contain optimizer.py.")
    return root


def imported_names(path: Path) -> set[str]:
    """Top-level module names imported by one file, from its AST."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
            elif node.module:
                names.add(node.module.split(".")[0])
    return names


def transitive_imports(root: Path, start: str) -> tuple[set[str], set[str]]:
    """
    Walk the import graph inside the checkout from one module.

    Returns (all names seen, files visited). A name that does not
    correspond to a file in the checkout is a third-party or stdlib
    module and is recorded but not descended into.
    """
    seen: set[str] = set()
    visited: set[str] = set()
    queue = [start]
    while queue:
        mod = queue.pop()
        if mod in visited:
            continue
        candidates = [root / f"{mod}.py", root / mod.replace(".", "/") / "__init__.py"]
        if "." in mod:
            candidates.insert(0, root / (mod.replace(".", "/") + ".py"))
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            continue
        visited.add(mod)
        for name in imported_names(path):
            seen.add(name)
            if name not in visited:
                queue.append(name)
    return seen, visited


class StaticImportGraph(unittest.TestCase):
    """Proof 1: the search path cannot reach training code."""

    def setUp(self):
        self.root = _root()

    def test_optimizer_reaches_no_training_module(self):
        seen, visited = transitive_imports(self.root, "optimizer")
        offenders = sorted(seen & TRAINING)
        self.assertEqual(offenders, [],
                         f"optimizer.py transitively imports {offenders}; "
                         f"files walked: {sorted(visited)}")

    def test_the_cipher_module_reaches_no_training_module(self):
        seen, _ = transitive_imports(self.root, "ciphers.speck3264")
        self.assertEqual(sorted(seen & TRAINING), [])

    def test_optimizer_imports_only_numpy_and_pandas_beyond_the_stdlib(self):
        seen, _ = transitive_imports(self.root, "optimizer")
        heavy = {"tensorflow", "keras", "torch", "jax", "ray", "sklearn"}
        self.assertEqual(sorted(seen & heavy), [])
        self.assertIn("numpy", seen)
        self.assertIn("pandas", seen)

    def test_CONTROL_main_py_does_reach_tensorflow(self):
        """
        Without this the analyser could be finding nothing at all and the
        two tests above would prove nothing.
        """
        seen, visited = transitive_imports(self.root, "main")
        self.assertIn("train_nets", seen,
                      "main.py should import train_nets; if not, the pinned source "
                      "changed and these proofs must be revisited.")
        self.assertIn("tensorflow", seen,
                      f"main.py should transitively reach tensorflow via train_nets; "
                      f"files walked: {sorted(visited)}")

    def test_CONTROL_train_nets_imports_tensorflow_directly(self):
        self.assertIn("tensorflow", imported_names(self.root / "train_nets.py"))


SUBPROCESS_PROBE = """
import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, {pkg!r})
sys.path.insert(0, {root!r})
import search_only
args = search_only.resolve_search_arguments(
    cipher_name="speck3264", scenario="single-key", output_dir="/tmp/unused")
import optimizer  # the real search module, as the entry point would load it
obs = search_only.assert_no_training_imports("after binding the real cipher")
print(json.dumps({{
    "loaded": sorted(m for m in sys.modules
                     if m.split(".")[0] in {training!r}),
    "observation": obs,
    "plain_bits": args.plain_bits,
    "key_bits": args.key_bits,
    "log_file": args.log_file,
    "nb_samples_passed": args.nb_samples,
    "call_kwargs": sorted(args.as_call_kwargs()),
}}))
"""


class DynamicImportCheck(unittest.TestCase):
    """
    Proof 2: in a clean process, binding the REAL Speck32/64 cipher and
    the REAL optimizer loads no training module. No search is executed.
    """

    def setUp(self):
        self.root = _root()

    def _probe(self) -> dict:
        import json
        code = SUBPROCESS_PROBE.format(
            pkg=str(PKG), root=str(self.root), training=sorted(TRAINING))
        proc = subprocess.run([sys.executable, "-I", "-c", textwrap.dedent(code)],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         f"probe failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_no_training_module_is_loaded_in_a_clean_process(self):
        out = self._probe()
        self.assertEqual(out["loaded"], [], f"training modules loaded: {out['loaded']}")
        self.assertEqual(out["observation"]["training_modules_present"], [])

    def test_the_real_cipher_parameters_are_read(self):
        out = self._probe()
        self.assertEqual(out["plain_bits"], 32)
        self.assertEqual(out["key_bits"], 64)

    def test_the_log_prefix_matches_upstreams_own_expression(self):
        """
        main.findGoodInputDifferences builds it as
        cipher.__name__[8:] + "_" + scenario, giving the README's
        results/speck3264_single-key.
        """
        out = self._probe()
        self.assertTrue(out["log_file"].endswith("/speck3264_single-key"),
                        out["log_file"])

    def test_r1_passes_no_nb_samples_as_upstream_does_not(self):
        out = self._probe()
        self.assertIsNone(out["nb_samples_passed"])
        self.assertEqual(out["call_kwargs"], ["epsilon", "log_file", "scenario"])


class BoundaryRefusal(unittest.TestCase):
    """The runtime assertion must actually fire."""

    def test_a_loaded_training_module_is_refused(self):
        import types
        sentinel = types.ModuleType("train_nets")
        sys.modules["train_nets"] = sentinel
        try:
            with self.assertRaises(search_only.TrainingBoundaryViolation) as cm:
                search_only.assert_no_training_imports("unit test")
            self.assertIn("train_nets", str(cm.exception))
        finally:
            del sys.modules["train_nets"]

    def test_a_clean_process_passes_the_assertion(self):
        obs = search_only.assert_no_training_imports("unit test")
        self.assertEqual(obs["training_modules_present"], [])
        self.assertIn("tensorflow", obs["checked"])


if __name__ == "__main__":
    unittest.main()
