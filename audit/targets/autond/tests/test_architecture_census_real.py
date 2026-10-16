#!/usr/bin/env python3
"""
Architecture-census tests against the REAL shipped AutoND checkpoints.

WHY THIS IS A SEPARATE MODULE, AND WHY IT FAILS RATHER THAN SKIPS.
These tests need the pinned upstream checkout. A machine without it
cannot run them, and the honest response to that is a failure with an
instruction, not a skip: a skipped audit test reads as a passing one in
any summary line, and this project's launch gate already counts a skip as
a failure. test_architecture_census.py holds everything that runs
anywhere; this module holds what needs the target.

Point AUTOND_UPSTREAM at the pinned checkout:

    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream

READ ONLY. Nothing here writes into the checkout.

ON THE RECORDED FIGURES. The values in PREVIOUSLY_RECORDED were measured
on 14 October 2026 from the two shipped checkpoints at the hashes below.
They are a DRIFT DETECTOR, not a specification: a mismatch means the file
or the reading of it changed and the difference must be explained in the
audit record. It does not mean the test should be edited to agree.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from audit.targets.autond import arch_derivation as deriv
from audit.targets.autond import architecture_census_v2 as census

CHECKPOINTS = {
    "speck": ("speck3264_single-key_0x400000_round8_polished.h5",
              "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26"),
    "simon": ("simon3264_single-key_0x400_round11_polished.h5",
              "769becfe854acd238f45ed4ade2810bb1e454166db3f5c6d557aa817275d25af"),
}

#: Measured 2026-10-14 from the checkpoints at the hashes above.
PREVIOUSLY_RECORDED = {
    "layer_count": 34,
    "tensor_count": 68,
    "total_parameters": 298113,
    "trainable_parameters": 296065,
    "non_trainable_parameters": 2048,
    "keras_version": "2.7.0",
    "layout": "weights_only",
    "measured_on": "2026-10-14",
    "note": ("A mismatch is a finding about the artifact or the instrument, to be "
             "explained in the audit record. It is not licence to edit these values."),
}

DRIFT = ("{key}: observed {got!r}, previously recorded {want!r} on "
         + PREVIOUSLY_RECORDED["measured_on"] +
         ". Explain the difference in the audit record before changing this test: "
         "either the checkpoint, the collector, or the reading of it has changed.")


class RealCheckpointTestCase(unittest.TestCase):
    """Base class that fails loudly when the target is not available."""

    @classmethod
    def setUpClass(cls):
        raw = os.environ.get("AUTOND_UPSTREAM")
        if not raw:
            raise AssertionError(
                "AUTOND_UPSTREAM is not set, so the shipped checkpoints cannot be "
                "read and these tests did not run. This is a FAILURE, not a skip: a "
                "skipped audit test is indistinguishable from a passing one in a "
                "summary line. Set it to the pinned checkout, e.g.\n"
                "    export AUTOND_UPSTREAM=~/Documents/AutoND-upstream")
        cls.root = Path(raw).expanduser().resolve()
        cls.weights = cls.root / "model_weights"
        missing = [n for n, _ in CHECKPOINTS.values() if not (cls.weights / n).is_file()]
        if missing:
            raise AssertionError(
                f"AUTOND_UPSTREAM={cls.root} does not contain {missing}. Point it at "
                "the pinned AutoND checkout that ships model_weights/.")

    def path(self, key: str) -> Path:
        return self.weights / CHECKPOINTS[key][0]


class TestShippedCheckpointIdentity(RealCheckpointTestCase):

    def test_checkpoint_hashes_match_the_recorded_values(self):
        for key, (name, want) in CHECKPOINTS.items():
            with self.subTest(checkpoint=key):
                got = hashlib.sha256(self.path(key).read_bytes()).hexdigest()
                self.assertEqual(got, want, DRIFT.format(
                    key=f"{name} sha256", got=got, want=want))


class TestShippedCheckpointCensus(RealCheckpointTestCase):

    def test_both_checkpoints_census_without_refusal(self):
        for key in CHECKPOINTS:
            with self.subTest(checkpoint=key):
                rec = census.census(self.path(key))
                self.assertIn("derived", rec)
                self.assertGreater(rec["derived"]["total_parameters"], 0)

    def test_observed_figures_match_what_was_previously_recorded(self):
        for key in CHECKPOINTS:
            rec = census.census(self.path(key))
            d = rec["derived"]
            with self.subTest(checkpoint=key):
                for field in ("layer_count", "tensor_count", "total_parameters",
                              "trainable_parameters", "non_trainable_parameters"):
                    self.assertEqual(d[field], PREVIOUSLY_RECORDED[field],
                                     DRIFT.format(key=f"{key}.{field}", got=d[field],
                                                  want=PREVIOUSLY_RECORDED[field]))
                self.assertEqual(rec["layout"], PREVIOUSLY_RECORDED["layout"],
                                 DRIFT.format(key=f"{key}.layout", got=rec["layout"],
                                              want=PREVIOUSLY_RECORDED["layout"]))
                self.assertEqual(rec["observed"]["keras_version"],
                                 PREVIOUSLY_RECORDED["keras_version"],
                                 DRIFT.format(key=f"{key}.keras_version",
                                              got=rec["observed"]["keras_version"],
                                              want=PREVIOUSLY_RECORDED["keras_version"]))

    def test_trainable_and_non_trainable_sum_to_the_total(self):
        for key in CHECKPOINTS:
            with self.subTest(checkpoint=key):
                d = census.census(self.path(key))["derived"]
                self.assertEqual(d["trainable_parameters"] + d["non_trainable_parameters"],
                                 d["total_parameters"])

    def test_weights_only_files_declare_the_graph_fields_they_lack(self):
        for key in CHECKPOINTS:
            with self.subTest(checkpoint=key):
                rec = census.census(self.path(key))
                self.assertIn("dilation_rate", rec["not_in_file"]["fields"])

    def test_shared_digest_across_speck_and_simon_is_expected_not_a_collision(self):
        """
        Both targets are 32-bit-block ciphers, so DBitNet sizes itself
        identically for each and one digest is the correct result. The
        collision the v1 defect produced is excluded separately, by
        test_architecture_census.TestDigestDiscrimination, on fixtures
        that really are different.
        """
        digests = {k: census.census(self.path(k))["derived"]["structural_digest"]
                   for k in CHECKPOINTS}
        self.assertEqual(len(set(digests.values())), 1, digests)
        import hashlib as _h
        self.assertNotEqual(set(digests.values()), {_h.sha256(b"").hexdigest()})


class TestCensusReconcilesWithPinnedSource(RealCheckpointTestCase):
    """h5py inventory versus the source-derived expectation."""

    def test_both_checkpoints_reconcile_with_the_source_derivation(self):
        tmp = Path(tempfile.mkdtemp())
        records = [census.census(self.path(k)) for k in CHECKPOINTS]
        (tmp / "c.json").write_text(json.dumps({"models": records}, default=str))
        derived = deriv.derive(input_size=64)
        result = deriv.reconcile(derived, tmp / "c.json")
        for r in result["models"]:
            with self.subTest(checkpoint=r["path"]):
                failed = [k for k, v in r["checks"].items() if not v]
                self.assertTrue(
                    r["all_checks_pass"],
                    f"{r['path']} disagrees with the pinned source on {failed}. "
                    f"Mismatch detail: {r.get('tensor_mismatch')}")
        self.assertTrue(result["all_models_reconcile"])

    def test_derivation_total_equals_the_census_total(self):
        derived = deriv.derive(input_size=64)
        for key in CHECKPOINTS:
            with self.subTest(checkpoint=key):
                self.assertEqual(census.census(self.path(key))["derived"]["total_parameters"],
                                 derived["total_parameters"])

    def test_pinned_commit_matches_the_package_record(self):
        from audit.targets.autond import PINNED_UPSTREAM_COMMIT
        head = self.root / ".git" / "HEAD"
        if not head.is_file():
            self.fail(f"{self.root} has no .git/HEAD, so the revision cannot be "
                      "verified. The target must be a git checkout pinned to "
                      f"{PINNED_UPSTREAM_COMMIT}.")
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            commit = (self.root / ".git" / ref.split(" ", 1)[1]).read_text().strip()
        else:
            commit = ref
        self.assertEqual(commit, PINNED_UPSTREAM_COMMIT,
                         DRIFT.format(key="upstream HEAD", got=commit,
                                      want=PINNED_UPSTREAM_COMMIT))


if __name__ == "__main__":
    unittest.main()
