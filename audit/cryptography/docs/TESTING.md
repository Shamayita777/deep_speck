# Test suite — scope and exact reproduction

**Test root:** `audit/cryptography/tests/` (inside this package). It is the only
pytest suite. `audit/cryptography/test/` is a runtime package of CE test
*definitions* (`base.py`, `ce*/…`) and contains no pytest tests.

**Nothing outside `audit/cryptography/` is required.** `audit` is imported as an
implicit namespace package; no `audit/__init__.py` is needed. No `conftest.py`,
`pytest.ini` or other file outside this tree is used.

**Exact reproduction (clean room):**

```bash
mkdir clean && cd clean
unzip /path/to/audit_cryptography_complete.zip       # creates ./audit/cryptography/
sha256sum -c --quiet audit/cryptography/PACKAGE_MANIFEST.sha256
python -m pytest audit/cryptography/tests -q -p no:cacheprovider -W ignore::DeprecationWarning
```

Run from the directory that CONTAINS `audit/` (the repository root). Environment
used for the recorded run: Python 3.12.3, TensorFlow 2.21.0 (CPU), Keras 3.15.1,
numpy 2.4.4, pytest 9.1.1, h5py 3.14.0, scipy 1.17.1. No GPU.

**Scope of what the suite establishes:** correctness of the control path at toy
scale on CPU (real Keras models, real `model.fit(initial_epoch=…)`, real SIGKILL,
real subprocess workers). It does NOT establish physical dual-GPU execution or
production-scale (10^7-sample) GPU training.

The real-artifact tests (legacy block0 read-only verification, CLI recovery into a
scratch run, validator wording) require
`evidence_current/ce1/production_20260929/`, which ships in this package; they
are skipped only if it is absent.
