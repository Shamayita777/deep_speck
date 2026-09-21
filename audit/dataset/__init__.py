"""
Dataset Integrity (Pillar 2 of 4) - D1-D5.

This package IS the Dataset Integrity pillar: d1_duplicate_detection,
d2_sample_dependence, d3_distribution_statistics,
d4_controlled_perturbation and d5_training_data_scaling live here, with
their evidence under audit/dataset/evidence/ and their tests under
audit/dataset/tests/.

The cross-pillar integration layer does not import this package. It
consumes byte-identical, hash-bound copies of the historical D evidence
from audit/evidence_bundle/ (see BUNDLE_MANIFEST.json).

(This file previously claimed that the D1-D5 code lived elsewhere and
referred to an audit/dataset/snapshot.py that never existed; both claims
were false and have been corrected. Only this docstring file changed -
no D1-D5 code or evidence was touched.)
"""
