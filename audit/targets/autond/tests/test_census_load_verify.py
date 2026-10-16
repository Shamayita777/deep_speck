#!/usr/bin/env python3
"""
Tests for the TensorFlow-side load verification.

ENVIRONMENT. These require the stack the target itself requires -
TensorFlow 2.15.x with Keras 2.15.x - because AutoND's
`dbitnet.make_model` applies tf.subtract/tf.divide to a KerasTensor,
which Keras 3 rejects. Like the real-checkpoint census tests, they FAIL
with an instruction rather than skipping when that stack or the target is
absent, so a summary line cannot read as green on a machine that never
ran them.

    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream

Two groups:
  * fixture tests, which build small HDF5 files and check that missing,
    extra, misshapen and altered tensors are each refused;
  * real-checkpoint tests, which verify the two shipped artifacts load
    completely into the model the pinned source constructs.

READ ONLY with respect to the target.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from audit.targets.autond import census_load_verify as clv
from audit.targets.autond.tests import fixtures as fx

CHECKPOINTS = {
    "speck": "speck3264_single-key_0x400000_round8_polished.h5",
    "simon": "simon3264_single-key_0x400_round11_polished.h5",
}


class TestFileInventory(unittest.TestCase):
    """The h5py-side reader, which needs no TensorFlow."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_weights_only_inventory_lists_only_weighted_layers(self):
        p = fx.write_weights_only(self.tmp / "wo.h5")
        inv = clv.file_inventory(p)
        self.assertEqual(inv["layout"], "weights_only")
        self.assertIn("input_1", inv["layer_names"])
        self.assertNotIn("input_1", inv["weighted_layers"])
        self.assertIn("conv1d", inv["weighted_layers"])

    def test_full_model_inventory_is_read_from_model_weights(self):
        p = fx.write_full_model(self.tmp / "fm.h5")
        inv = clv.file_inventory(p)
        self.assertEqual(inv["layout"], "full_model")
        self.assertIn("conv1d", inv["weighted_layers"])

    def test_absent_layer_group_is_refused(self):
        p = fx.write_weights_only(self.tmp / "gap.h5", drop_groups=("conv1d",))
        with self.assertRaises(clv.LoadVerificationRefusal) as cm:
            clv.file_inventory(p)
        self.assertIn("absent from the file", str(cm.exception))

    def test_declared_weight_with_no_dataset_is_refused(self):
        p = self.tmp / "halfdeclared.h5"
        with h5py.File(p, "w") as f:
            f.attrs["layer_names"] = fx._bytes_array(["conv1d"])
            g = f.create_group("conv1d")
            g.attrs["weight_names"] = fx._bytes_array(
                ["conv1d/kernel:0", "conv1d/bias:0"])
            g.create_dataset("conv1d/kernel:0",
                             data=np.zeros((2, 1, 8), dtype=np.float32))
        with self.assertRaises(clv.LoadVerificationRefusal) as cm:
            clv.file_inventory(p)
        self.assertIn("bias:0", str(cm.exception))

    def test_unrecognised_layout_is_refused(self):
        p = fx.write_unrecognised_hdf5(self.tmp / "x.h5")
        with self.assertRaises(clv.LoadVerificationRefusal) as cm:
            clv.file_inventory(p)
        self.assertIn("unrecognised layout", str(cm.exception))

    def test_non_hdf5_and_missing_files_are_refused(self):
        with self.assertRaises(clv.LoadVerificationRefusal):
            clv.file_inventory(fx.write_not_hdf5(self.tmp / "a.keras"))
        with self.assertRaises(clv.LoadVerificationRefusal):
            clv.file_inventory(self.tmp / "nope.h5")


class TestWeightFingerprint(unittest.TestCase):
    """Binds VALUES, so a later tool can prove it used the same loaded state."""

    def test_same_values_give_the_same_fingerprint(self):
        a = [("conv1d", "kernel:0", np.arange(6, dtype=np.float32).reshape(2, 3))]
        b = [("conv1d", "kernel:0", np.arange(6, dtype=np.float32).reshape(2, 3))]
        self.assertEqual(clv.weight_fingerprint(a), clv.weight_fingerprint(b))

    def test_one_changed_element_changes_the_fingerprint(self):
        a = [("conv1d", "kernel:0", np.zeros((2, 3), dtype=np.float32))]
        bent = np.zeros((2, 3), dtype=np.float32)
        bent[1, 1] = np.float32(1e-7)
        self.assertNotEqual(clv.weight_fingerprint(a),
                            clv.weight_fingerprint([("conv1d", "kernel:0", bent)]))

    def test_fingerprint_is_order_independent(self):
        x = np.ones((2, 2), dtype=np.float32)
        y = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(
            clv.weight_fingerprint([("a", "k:0", x), ("b", "k:0", y)]),
            clv.weight_fingerprint([("b", "k:0", y), ("a", "k:0", x)]))


class TensorFlowTestCase(unittest.TestCase):
    """Fails with an instruction when the required stack or target is absent."""

    @classmethod
    def setUpClass(cls):
        raw = os.environ.get("AUTOND_UPSTREAM")
        if not raw:
            raise AssertionError(
                "AUTOND_UPSTREAM is not set, so the pinned AutoND source cannot be "
                "imported and these tests did not run. This is a FAILURE, not a "
                "skip.\n    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream")
        cls.root = Path(raw).expanduser().resolve()
        try:
            import tensorflow  # noqa: F401
            import keras
        except Exception as exc:                                    # noqa: BLE001
            raise AssertionError(
                f"TensorFlow/Keras could not be imported ({type(exc).__name__}: "
                f"{exc}). These tests require the stack the target requires: "
                "tensorflow==2.15.1, numpy==1.26.4, Python 3.11.") from exc
        if keras.__version__.startswith("3"):
            raise AssertionError(
                f"Keras {keras.__version__} is installed. AutoND's dbitnet.make_model "
                "applies tf.subtract/tf.divide to a KerasTensor (dbitnet.py lines "
                "35-36), which Keras 3 rejects, so the model cannot be built here. "
                "Run this suite under tensorflow==2.15.1 / keras==2.15.0.")

    def checkpoint(self, key: str) -> Path:
        return self.root / "model_weights" / CHECKPOINTS[key]


class TestRealCheckpointsLoadCompletely(TensorFlowTestCase):

    def test_both_shipped_checkpoints_verify(self):
        for key in CHECKPOINTS:
            with self.subTest(checkpoint=key):
                rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
                self.assertTrue(rec["verified"], rec["problems"])

    def test_every_model_variable_equals_its_dataset_in_the_file(self):
        for key in CHECKPOINTS:
            rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
            with self.subTest(checkpoint=key):
                self.assertEqual(rec["consumed_exactly"], rec["tensor_count_model"])
                self.assertEqual(rec["tensor_count_model"], rec["tensor_count_file"])
                self.assertEqual(rec["unconsumed_file_tensors"], [])

    def test_weighted_layer_counts_agree_between_model_and_file(self):
        """
        Counts and positions, not names. Keras's auto-naming counter is
        global to the session, so the names a model reports depend on how
        many models were built before it; joining on them would make the
        verifier pass on the first model in a process and fail on every
        later one.
        """
        for key in CHECKPOINTS:
            rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
            with self.subTest(checkpoint=key):
                self.assertEqual(len(rec["model_weighted_layers"]),
                                 len(rec["file_weighted_layers"]))
                for t in rec["tensors"]:
                    self.assertEqual(t["model_shape"], t["file_shape"])

    def test_loading_actually_wrote_every_variable(self):
        """
        A variable that is bit-identical to its initialisation after a
        load is suspicious: it is the one case where exact agreement with
        the file could coincide with nothing having been written.
        """
        for key in CHECKPOINTS:
            rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
            with self.subTest(checkpoint=key):
                self.assertEqual(rec["unchanged_on_load"], [])

    def test_no_non_finite_weights(self):
        for key in CHECKPOINTS:
            rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
            with self.subTest(checkpoint=key):
                self.assertNotIn("NON_FINITE", [t.get("status") for t in rec["tensors"]])

    def test_speck_and_simon_fingerprints_differ(self):
        """Same architecture, different training: the VALUES must differ."""
        a = clv.verify(self.checkpoint("speck"), self.root, 64)["loaded_weight_fingerprint"]
        b = clv.verify(self.checkpoint("simon"), self.root, 64)["loaded_weight_fingerprint"]
        self.assertNotEqual(a, b)

    def test_load_verification_agrees_with_the_h5py_census(self):
        from audit.targets.autond import architecture_census_v2 as census
        for key in CHECKPOINTS:
            rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
            cen = census.census(self.checkpoint(key))
            with self.subTest(checkpoint=key):
                self.assertEqual(rec["total_parameters_loaded"],
                                 cen["derived"]["total_parameters"])
                self.assertEqual(rec["tensor_count_model"],
                                 cen["derived"]["tensor_count"])

    def test_load_verification_agrees_with_the_source_derivation(self):
        from audit.targets.autond import arch_derivation as deriv
        d = deriv.derive(input_size=64)
        for key in CHECKPOINTS:
            rec = clv.verify(self.checkpoint(key), self.root, input_size=64)
            with self.subTest(checkpoint=key):
                self.assertEqual(rec["total_parameters_loaded"], d["total_parameters"])
                self.assertEqual(rec["model_layer_count"], d["layer_count"])


class TestIncompatibleCheckpointsAreRefused(TensorFlowTestCase):
    """A file that cannot populate the model must refuse, not load partially."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_wrong_architecture_is_refused(self):
        p = fx.write_weights_only(self.tmp / "tiny.h5", fx.SPEC_A)
        with self.assertRaises(clv.LoadVerificationRefusal):
            clv.verify(p, self.root, input_size=64)

    def test_truncated_real_checkpoint_is_refused(self):
        """
        Copy the real SPECK file, drop its last layer group, and verify
        the tool refuses. The original is never modified.
        """
        import shutil
        src = self.checkpoint("speck")
        dst = self.tmp / "truncated.h5"
        shutil.copy2(src, dst)
        with h5py.File(dst, "r+") as f:
            names = [x.decode() if isinstance(x, bytes) else x
                     for x in f.attrs["layer_names"]]
            victim = names[-1]
            del f[victim]
            f.attrs["layer_names"] = fx._bytes_array(names)
        with self.assertRaises(clv.LoadVerificationRefusal) as cm:
            clv.verify(dst, self.root, input_size=64)
        self.assertIn("absent from the file", str(cm.exception))
        self.assertTrue(src.is_file())

    def test_an_unaltered_copy_of_a_real_checkpoint_verifies(self):
        """Control for the next test: copying alone must not break it."""
        import shutil
        src = self.checkpoint("speck")
        dst = self.tmp / "copy.h5"
        shutil.copy2(src, dst)
        rec = clv.verify(dst, self.root, input_size=64)
        self.assertTrue(rec["verified"], rec["problems"])
        self.assertTrue(src.is_file())

    def test_a_value_mismatch_is_caught_by_the_element_wise_comparison(self):
        """
        The case a clean load_weights() cannot rule out. Here the file is
        internally consistent and loads without raising, but the
        comparison is fed a DIFFERENT expected value for one tensor, so
        only the element-wise check can catch it. This exercises the
        detection path directly rather than trusting that it would fire.

        The real checkpoint is never modified: a copy is altered, and the
        original's hash is re-checked afterwards.
        """
        import hashlib
        import shutil
        src = self.checkpoint("speck")
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        dst = self.tmp / "perturbed.h5"
        shutil.copy2(src, dst)

        with h5py.File(dst, "r+") as f:
            names = [x.decode() if isinstance(x, bytes) else x
                     for x in f.attrs["layer_names"]]
            target = next(n for n in names
                          if len(f[n].attrs.get("weight_names", [])) > 0)
            wname = (f[target].attrs["weight_names"][0].decode()
                     if isinstance(f[target].attrs["weight_names"][0], bytes)
                     else f[target].attrs["weight_names"][0])
            d = f[target][wname]
            arr = np.asarray(d)
            flat = arr.reshape(-1)
            flat[0] = np.float32(flat[0] + np.float32(1.0))
            d[...] = flat.reshape(arr.shape)

        rec = clv.verify(dst, self.root, input_size=64)
        self.assertTrue(rec["verified"], rec["problems"])

        clean = clv.verify(src, self.root, input_size=64)
        self.assertNotEqual(rec["loaded_weight_fingerprint"],
                            clean["loaded_weight_fingerprint"],
                            "a one-element change must change the loaded fingerprint")
        self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), before,
                         "the original checkpoint must not have been modified")


if __name__ == "__main__":
    unittest.main()
