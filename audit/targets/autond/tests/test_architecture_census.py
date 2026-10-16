#!/usr/bin/env python3
"""
Architecture-census tests that need no TensorFlow and no target files.

SCOPE. Every test here runs against synthetic fixtures, so the suite is
green on any machine with h5py and numpy and never touches a research
artifact. The real shipped checkpoints are exercised by
test_architecture_census_real.py, which fails loudly rather than skipping
when the target checkout is absent - a skipped audit test is a failed
one, and this project's launch gate already treats it that way.

WHAT IS AND IS NOT ASSERTED. The expected AutoND figures - 34 layers, 68
tensors, 298,113 parameters, dilations [31, 15, 7, 3] - are NOT asserted
here. Baking a target's measurements into the instrument's own tests
would make the instrument agree with them by construction. These tests
assert the MECHANISM: that arithmetic over an inventory is correct, that
incomplete evidence is refused, and that two different architectures
cannot collide on one digest. The target's numbers are checked against
the pinned source by arch_derivation.py, where a mismatch is a finding
rather than a test failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit.targets.autond import arch_derivation as deriv
from audit.targets.autond import architecture_census_v2 as census
from audit.targets.autond.tests import fixtures as fx

#: The collector as a standalone script, which is how the runbook runs it.
SCRIPT = Path(census.__file__).resolve()


class TestValidCensus(unittest.TestCase):
    """A readable, complete file produces a complete, correct census."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_weights_only_layout_is_recognised_and_counted(self):
        p = fx.write_weights_only(self.tmp / "wo.h5")
        rec = census.census(p)
        self.assertEqual(rec["layout"], "weights_only")
        total, non_trainable = fx.spec_parameters(fx.SPEC_A)
        d = rec["derived"]
        self.assertEqual(d["layer_count"], len(fx.SPEC_A))
        self.assertEqual(d["tensor_count"],
                         sum(len(w) for _, w in fx.SPEC_A))
        self.assertEqual(d["total_parameters"], total)
        self.assertEqual(d["non_trainable_parameters"], non_trainable)
        self.assertEqual(d["trainable_parameters"], total - non_trainable)
        self.assertEqual(len(d["structural_digest"]), 64)

    def test_weightless_layers_are_kept_in_the_inventory(self):
        """A layer with no weights is real structure, not an absence."""
        p = fx.write_weights_only(self.tmp / "wo.h5")
        rec = census.census(p)
        names = [l["layer"] for l in rec["layers"]]
        self.assertIn("input_1", names)
        self.assertIn("add", names)
        self.assertEqual([l["parameters"] for l in rec["layers"] if l["layer"] == "add"], [0])

    def test_full_model_layout_records_optimizer_state(self):
        p = fx.write_full_model(self.tmp / "fm.h5")
        rec = census.census(p)
        self.assertEqual(rec["layout"], "full_model")
        self.assertTrue(rec["observed"]["has_model_config"])
        self.assertTrue(rec["observed"]["has_optimizer_state"])
        self.assertIn("optimizer_slot_tensors", rec["derived"])

    def test_weights_only_declares_what_it_cannot_establish(self):
        p = fx.write_weights_only(self.tmp / "wo.h5")
        rec = census.census(p)
        self.assertIn("dilation_rate", rec["not_in_file"]["fields"])
        self.assertIn("residual_wiring", rec["not_in_file"]["fields"])

    def test_full_model_does_not_claim_graph_fields_are_missing(self):
        p = fx.write_full_model(self.tmp / "fm.h5")
        rec = census.census(p)
        self.assertEqual(rec["not_in_file"]["fields"], [])

    def test_file_hash_is_recorded_and_correct(self):
        import hashlib
        p = fx.write_weights_only(self.tmp / "wo.h5")
        rec = census.census(p)
        self.assertEqual(rec["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())


class TestFailClosed(unittest.TestCase):
    """Incomplete or unreadable evidence must refuse, never return a result."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_layer_declared_but_group_absent_is_refused(self):
        """
        The v1 defect in its quieter form. A census that recorded the gap
        as a flag and carried on would under-count layers, tensors and
        parameters while reporting success.
        """
        p = fx.write_weights_only(self.tmp / "gap.h5", drop_groups=("conv1d",))
        with self.assertRaises(census.CensusRefusal) as cm:
            census.census(p)
        self.assertIn("layer inventory incomplete", str(cm.exception))
        self.assertIn("conv1d", str(cm.exception))

    def test_unrecognised_layout_is_refused_not_guessed(self):
        p = fx.write_unrecognised_hdf5(self.tmp / "other.h5")
        with self.assertRaises(census.CensusRefusal) as cm:
            census.census(p)
        self.assertIn("unrecognised HDF5 layout", str(cm.exception))

    def test_zero_parameter_file_is_refused(self):
        """The exact v1 failure: a successful-looking empty census."""
        p = fx.write_zero_parameter(self.tmp / "empty.h5")
        with self.assertRaises(census.CensusRefusal) as cm:
            census.census(p)
        self.assertIn("zero weight parameters", str(cm.exception))

    def test_non_hdf5_file_is_refused(self):
        p = fx.write_not_hdf5(self.tmp / "archive.keras")
        with self.assertRaises(census.CensusRefusal) as cm:
            census.census(p)
        self.assertIn("not an HDF5 file", str(cm.exception))

    def test_missing_file_is_refused(self):
        with self.assertRaises(census.CensusRefusal):
            census.census(self.tmp / "does_not_exist.h5")

    def test_empty_census_can_never_reach_the_digest(self):
        with self.assertRaises(census.CensusRefusal) as cm:
            census.structural_digest([], "weights_only")
        self.assertIn("empty census", str(cm.exception))

    def test_digest_of_empty_string_is_never_emitted(self):
        """
        v1 returned sha256('') for unreadable files. No path may produce
        it, because it is identical for every such file.
        """
        import hashlib
        forbidden = hashlib.sha256(b"").hexdigest()
        p = fx.write_weights_only(self.tmp / "wo.h5")
        self.assertNotEqual(census.census(p)["derived"]["structural_digest"], forbidden)


class TestDigestDiscrimination(unittest.TestCase):
    """The regression test for the v1 collision."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_different_architectures_get_different_digests(self):
        a = census.census(fx.write_weights_only(self.tmp / "a.h5", fx.SPEC_A))
        b = census.census(fx.write_weights_only(self.tmp / "b.h5", fx.SPEC_B))
        self.assertNotEqual(a["derived"]["structural_digest"],
                            b["derived"]["structural_digest"])

    def test_same_architecture_in_both_layouts_gets_one_digest(self):
        """
        The format control. Layout is deliberately not hashed, so a
        weights-only file and a full-model file holding the same
        architecture must agree.
        """
        wo = census.census(fx.write_weights_only(self.tmp / "wo.h5", fx.SPEC_A, seed=1))
        fm = census.census(fx.write_full_model(self.tmp / "fm.h5", fx.SPEC_A, seed=2))
        self.assertEqual(wo["derived"]["structural_digest"],
                         fm["derived"]["structural_digest"])

    def test_digest_is_invariant_to_weight_values(self):
        one = census.census(fx.write_weights_only(self.tmp / "s1.h5", fx.SPEC_A, seed=11))
        two = census.census(fx.write_weights_only(self.tmp / "s2.h5", fx.SPEC_A, seed=22))
        self.assertEqual(one["derived"]["structural_digest"],
                         two["derived"]["structural_digest"])

    def test_a_single_changed_tensor_shape_changes_the_digest(self):
        spec = [(n, list(w)) for n, w in fx.SPEC_A]
        spec[1] = ("conv1d", [("conv1d/kernel:0", (2, 1, 9)), ("conv1d/bias:0", (9,))])
        base = census.census(fx.write_weights_only(self.tmp / "base.h5", fx.SPEC_A))
        bent = census.census(fx.write_weights_only(self.tmp / "bent.h5", spec))
        self.assertNotEqual(base["derived"]["structural_digest"],
                            bent["derived"]["structural_digest"])


class TestSourceDerivation(unittest.TestCase):
    """
    The derivation reproduces dbitnet.make_model's shape algebra in pure
    Python. These tests check the algebra and the reconciliation logic,
    not AutoND's measured numbers.
    """

    def test_dilation_rates_follow_the_source_rule(self):
        self.assertEqual(deriv.dilation_rates(64), [31, 15, 7, 3])
        self.assertEqual(deriv.dilation_rates(32), [15, 7, 3])
        self.assertEqual(deriv.dilation_rates(8), [3])
        self.assertEqual(deriv.dilation_rates(4), [])

    def test_sequence_length_shrinks_by_the_dilation_each_block(self):
        d = deriv.derive(input_size=64)
        self.assertEqual(d["sequence_length_chain"], [64, 33, 18, 11, 8])
        self.assertEqual(d["flatten_width"], 8 * 80)

    def test_parameter_total_equals_the_sum_of_its_tensors(self):
        d = deriv.derive(input_size=64)
        recomputed = 0
        for layer in d["layers"]:
            for _name, shape in layer["tensors"]:
                recomputed += int(np.prod(shape)) if shape else 1
        self.assertEqual(recomputed, d["total_parameters"])
        self.assertEqual(d["trainable_parameters"] + d["non_trainable_parameters"],
                         d["total_parameters"])

    def test_batchnorm_statistics_are_counted_non_trainable(self):
        d = deriv.derive(input_size=64)
        bn = [l for l in d["layers"] if l["class"] == "BatchNormalization"]
        self.assertTrue(bn)
        expected_non_trainable = sum(
            int(np.prod(shape)) for l in bn for name, shape in l["tensors"]
            if name.split(":")[0] in ("moving_mean", "moving_variance"))
        self.assertEqual(d["non_trainable_parameters"], expected_non_trainable)

    def test_head_has_four_dense_layers_three_hidden_plus_output(self):
        d = deriv.derive(input_size=64)
        dense = [l for l in d["layers"] if l["class"] == "Dense"]
        self.assertEqual(len(dense), 4)
        self.assertEqual(d["dense_layers"]["hidden"], [256, 256, 64])

    def test_a_different_input_size_gives_a_different_architecture(self):
        self.assertNotEqual(deriv.derive(input_size=64)["total_parameters"],
                            deriv.derive(input_size=32)["total_parameters"])

    def test_derivation_reconciles_with_a_census_built_from_itself(self):
        """
        Round trip: turn the derived inventory into a fixture, census it,
        reconcile. This exercises the comparison on data whose correct
        answer is known without reference to any AutoND artifact.
        """
        d = deriv.derive(input_size=64)
        spec = [(l["layer"], [(n, tuple(s)) for n, s in l["tensors"]]) for l in d["layers"]]
        tmp = Path(tempfile.mkdtemp())
        p = fx.write_weights_only(tmp / "derived.h5", spec)
        rec = census.census(p)
        out = {"models": [rec]}
        (tmp / "c.json").write_text(json.dumps(out, default=str))
        result = deriv.reconcile(d, tmp / "c.json")
        self.assertTrue(result["all_models_reconcile"], result["models"][0]["checks"])

    def test_reconciliation_detects_a_shape_mismatch(self):
        d = deriv.derive(input_size=64)
        spec = [(l["layer"], [(n, tuple(s)) for n, s in l["tensors"]]) for l in d["layers"]]
        for i, (name, weights) in enumerate(spec):
            if name == "dense_3":
                spec[i] = (name, [("dense_3/kernel:0", (64, 2)),
                                  ("dense_3/bias:0", (2,))])
        tmp = Path(tempfile.mkdtemp())
        p = fx.write_weights_only(tmp / "bent.h5", spec)
        (tmp / "c.json").write_text(json.dumps({"models": [census.census(p)]}, default=str))
        result = deriv.reconcile(d, tmp / "c.json")
        self.assertFalse(result["all_models_reconcile"])
        self.assertFalse(result["models"][0]["checks"]["tensor_names_and_shapes"])
        self.assertTrue(result["models"][0]["tensor_mismatch"]["only_in_file"])

    def test_reconciliation_detects_a_missing_layer(self):
        d = deriv.derive(input_size=64)
        spec = [(l["layer"], [(n, tuple(s)) for n, s in l["tensors"]])
                for l in d["layers"] if l["layer"] != "dense_2"]
        tmp = Path(tempfile.mkdtemp())
        p = fx.write_weights_only(tmp / "short.h5", spec)
        (tmp / "c.json").write_text(json.dumps({"models": [census.census(p)]}, default=str))
        result = deriv.reconcile(d, tmp / "c.json")
        self.assertFalse(result["all_models_reconcile"])
        self.assertFalse(result["models"][0]["checks"]["layer_names_and_order"])

    def test_derivation_states_what_it_cannot_establish(self):
        d = deriv.derive(input_size=64)
        joined = " ".join(d["not_established_here"]).lower()
        self.assertIn("paper", joined)
        self.assertIn("weight values", joined)


class TestCommandLine(unittest.TestCase):
    """
    The tool must exit non-zero on a refusal, so a script cannot miss it.

    Invoked exactly as the runbook invokes it: as a script path under
    `python3 -I`. The collector imports only h5py and numpy, so it needs
    nothing of this package on sys.path, and isolated mode is what keeps
    a stray module in the working directory out of an audit instrument.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-I", str(SCRIPT), *map(str, args)],
            capture_output=True, text=True)

    def test_exit_zero_on_a_clean_census(self):
        p = fx.write_weights_only(self.tmp / "ok.h5")
        out = self.tmp / "out.json"
        r = self._run(p, "--output", out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(out.is_file())
        self.assertEqual(json.loads(out.read_text())["refusals"], [])

    def test_exit_non_zero_when_any_input_is_refused(self):
        good = fx.write_weights_only(self.tmp / "ok.h5")
        bad = fx.write_zero_parameter(self.tmp / "empty.h5")
        out = self.tmp / "out.json"
        r = self._run(good, bad, "--output", out)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        payload = json.loads(out.read_text())
        self.assertEqual(len(payload["refusals"]), 1)
        self.assertIn("REFUSED", r.stdout)

    def test_refusals_are_recorded_in_the_artifact_not_only_printed(self):
        bad = fx.write_unrecognised_hdf5(self.tmp / "x.h5")
        out = self.tmp / "out.json"
        self._run(bad, "--output", out)
        payload = json.loads(out.read_text())
        self.assertEqual(payload["models"], [])
        self.assertIn("unrecognised", payload["refusals"][0]["refusal"])

    def test_format_control_flag_reports_a_mismatch(self):
        a = fx.write_weights_only(self.tmp / "a.h5", fx.SPEC_A)
        b = fx.write_weights_only(self.tmp / "b.h5", fx.SPEC_B)
        out = self.tmp / "out.json"
        self._run(a, b, "--output", out, "--expect-digest-match")
        payload = json.loads(out.read_text())
        self.assertFalse(payload["digest_match"])
        self.assertEqual(payload["structural_digests_distinct"], 2)


if __name__ == "__main__":
    unittest.main()
