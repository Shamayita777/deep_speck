"""
Validation test suite for the AutoND dataset adapter
(audit.dataset.adapters.autond.AutoNDDatasetAdapter).

Scientific status
-----------------
NON-EVIDENTIARY.

This suite validates that the adapter presents the pinned AutoND generator to
the generic D0-D3 engines without altering it, and that it refuses
configurations that would silently produce a mislabelled dataset instance. It
does NOT establish, and must never be cited as establishing:

    - that any AutoND dataset is clean under D1, D2, D3 or any other audit;
    - that the AutoND generator is correct;
    - cryptographic soundness of any dataset it produces;
    - any property of a round count, cipher or difference other than the one
      a given test constructs.

The production AutoND dataset-integrity results are produced separately, on a
persisted canonical instance, by
    audit/dataset/experiments/d0_autond_canonical.py   (build the instance)
    audit/dataset/experiments/d1_autond_exact_census.py (audit it)
This suite is implementation-level validation intended to run *before* those
executions, not a substitute for either.

Expected values are derived from the pinned upstream source
(commit 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779) and from the adapter's
documented contract -- never by calling the adapter and trusting its own
output as the expected result. Where a value was read off the pinned source by
execution, the test re-derives it from that source at run time instead of
hardcoding it, so a change of pin fails the test rather than passing silently.

The pinned checkout is NOT modified by this suite. The one test that needs to
observe the call into upstream replaces the attribute on a single adapter
INSTANCE, never on the upstream module.

Scope note
----------
This suite deliberately does not test duplicate detection, sample dependence,
distributional statistics, perturbation response, scaling behaviour, model
training, or cryptographic validity. Those belong to D1-D5 and to their own
suites. Here the subject is the adapter boundary only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from audit.dataset.adapters import autond as A

# ---------------------------------------------------------------------
# Locating the pinned checkout
# ---------------------------------------------------------------------
# The adapter imports the audited subject, so these tests cannot run without
# it. They SKIP rather than fail when it is absent: a missing checkout is an
# environment fact, not an adapter defect. A skip is visible in the pytest
# summary and must not be read as a pass.

_CANDIDATE_ROOTS = (
    os.environ.get("AUTOND_ROOT"),
    "~/Documents/AutoND-upstream",
)


def _pinned_root() -> Path:
    for cand in _CANDIDATE_ROOTS:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if (p / "main.py").is_file():
            return p.resolve()
    pytest.skip(
        "pinned AutoND checkout not found; set AUTOND_ROOT to the pinned "
        "checkout to run the adapter suite. SKIPPED IS NOT PASSED.")


@pytest.fixture(scope="module")
def root() -> Path:
    return _pinned_root()


#: The case-study configuration. Chosen once here so every test below talks
#: about the same dataset instance; tests that need a different configuration
#: construct it explicitly and say why.
DELTA = 0x400000
ROUNDS = 8
CIPHER = "speck3264"


@pytest.fixture(scope="module")
def adapter(root):
    return A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name=CIPHER,
        delta_plain_int=DELTA, num_rounds=ROUNDS, scenario="single-key")


@pytest.fixture(scope="module")
def upstream(root):
    """The pinned `main` and cipher modules, imported independently.

    Imported here through the adapter's own import helper so the test does not
    maintain a second, possibly divergent import path. The point is to obtain
    the upstream values the adapter's contract is checked AGAINST, so they come
    from the source rather than from this file.
    """
    probe = A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name=CIPHER,
        delta_plain_int=DELTA, num_rounds=ROUNDS)
    return probe._import_pinned()


# ---------------------------------------------------------------------
# 1. Structural absence of the training half
# ---------------------------------------------------------------------
# The adapter documents that build_model/train/evaluate/predict/save/load are
# deliberately NOT implemented, so that a caller cannot believe the training
# path exists. These tests lock that decision: if someone later adds a stub --
# including a stub that raises -- this fails and forces an explicit decision
# instead of letting D4/D5 appear reachable for AutoND when they are not.

TRAINING_SURFACE = ("build_model", "train", "evaluate", "predict",
                    "save", "load")


@pytest.mark.parametrize("name", TRAINING_SURFACE)
def test_training_surface_is_absent_not_stubbed(adapter, name):
    assert not hasattr(adapter, name), (
        f"{name!r} is present on AutoNDDatasetAdapter. The adapter's contract "
        f"is that the model/training half is structurally absent, so that "
        f"D4/D5 for AutoND cannot be entered by accident. Adding it -- even "
        f"as a stub that raises -- requires establishing the interface fit "
        f"first and updating the adapter docstring.")


def test_dataset_surface_is_present(adapter):
    for name in ("feature_bits", "half_width", "generate_partition",
                 "dataset_provenance"):
        assert hasattr(adapter, name), f"missing dataset-side member {name!r}"


# ---------------------------------------------------------------------
# 2. No generic audit logic inside the target-specific adapter
# ---------------------------------------------------------------------
# audit/dataset/ is the authoritative home of the D-pillar engines. A
# reimplementation inside a target adapter would let the two targets be
# audited by different code while reporting the same dimension name.

GENERIC_ENGINES = ("d1_duplicate_detection", "d2_sample_dependence",
                   "d3_distribution_statistics",
                   "d4_controlled_perturbation",
                   "d5_training_data_scaling")


def test_adapter_does_not_import_or_reimplement_audit_engines():
    src = Path(A.__file__).read_text()
    for engine in GENERIC_ENGINES:
        assert engine not in src, (
            f"the AutoND adapter references {engine!r}. Audit logic and audit "
            f"verdicts belong in audit/dataset/, not in a target adapter; the "
            f"adapter supplies data and provenance only.")
    for verdict in ("PASS", "FAIL", "verdict"):
        assert verdict not in src, (
            f"the AutoND adapter contains {verdict!r}. Deciding an audit "
            f"outcome is the generic engine's job.")


def test_dataset_id_cannot_be_confused_with_gohr():
    """Read Gohr's id from source rather than importing it.

    Importing the Gohr adapter pulls in the Gohr checkout (`speck`), which is
    an unrelated dependency for this question. A skip here would quietly
    retire the one test that keeps the two targets' evidence distinguishable,
    so the constant is read textually instead and a missing file is a failure.
    """
    import re
    gohr_src = Path(A.__file__).parent / "gohr.py"
    assert gohr_src.is_file(), f"{gohr_src} is missing"
    found = re.findall(r"""DATASET_ID\s*=\s*["']([^"']+)["']""",
                       gohr_src.read_text())
    assert found, f"no DATASET_ID literal found in {gohr_src}"
    assert A.DATASET_ID not in found, (
        f"the AutoND dataset_id {A.DATASET_ID!r} collides with a Gohr "
        f"adapter id {found!r}; evidence for the two targets must never be "
        f"confusable by dataset_id alone.")
    assert A.DATASET_ID == "autond-speck3264"


# ---------------------------------------------------------------------
# 3. Representation contract, derived from the pinned source
# ---------------------------------------------------------------------

def test_bit_widths_come_from_the_cipher_module(adapter, upstream):
    _, cipher = upstream
    assert adapter.plain_bits == int(cipher.plain_bits)
    assert adapter.key_bits == int(cipher.key_bits)
    assert adapter.word_size == int(cipher.word_size)


def test_feature_width_is_twice_plain_bits(adapter, upstream):
    _, cipher = upstream
    expected = 2 * int(cipher.plain_bits)
    assert adapter.feature_bits == expected
    assert adapter.half_width == int(cipher.plain_bits)
    # The halves must tile the feature vector exactly, with no gap or overlap:
    # D1's column semantics and D3's per-half statistics both rely on it.
    assert adapter.half_width * 2 == adapter.feature_bits


def test_delta_bit_vector_is_the_targets_own_encoding(adapter, upstream):
    """The adapter must not re-encode the difference itself.

    `main.integer_to_binary_array` is MSB-first and returns a (1, num_bits)
    row. An adapter that built its own little-endian flat vector would still
    have popcount 1 for this delta, so the test compares the whole array
    against the upstream helper's output rather than just the popcount.
    """
    main, cipher = upstream
    expected = main.integer_to_binary_array(DELTA, int(cipher.plain_bits))
    got = adapter._delta_plain
    assert got.shape == expected.shape, (
        f"delta shape {got.shape} != upstream {expected.shape}; the adapter "
        f"has reshaped the difference the generator expects.")
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    # Re-derived, not hardcoded: 0x400000 == 2**22, MSB-first in 32 bits puts
    # the set bit at index 32 - 1 - 22.
    assert int(expected.sum()) == 1
    assert np.flatnonzero(expected.ravel()).tolist() == [
        int(cipher.plain_bits) - 1 - 22]


def test_single_key_scenario_passes_scalar_zero_key_difference(adapter):
    """Upstream's single-key path receives 0, not a zero vector.

    `make_train_data` branches on the key difference, so substituting a
    zero-filled array here would silently exercise the related-key path with a
    trivial difference. That would be a different dataset instance wearing the
    single-key label.
    """
    assert adapter.scenario == "single-key"
    assert adapter._delta_key == 0
    assert not isinstance(adapter._delta_key, np.ndarray)


# ---------------------------------------------------------------------
# 4. generate_partition: shape, dtype, value set, and pass-through
# ---------------------------------------------------------------------

N_SMALL = 2000


def test_partition_shape_dtype_and_value_set(adapter):
    X, Y = adapter.generate_partition(N_SMALL)
    assert X.shape == (N_SMALL, adapter.feature_bits)
    assert Y.shape == (N_SMALL,)
    assert X.dtype == np.uint8
    assert Y.dtype == np.uint8
    assert set(np.unique(X).tolist()) <= {0, 1}
    assert set(np.unique(Y).tolist()) <= {0, 1}
    # Both labels must actually occur, or a "dataset" is one class and every
    # downstream balance statistic is meaningless. This is a weak sanity
    # bound, not a distributional claim: the test asserts presence, not rate.
    assert set(np.unique(Y).tolist()) == {0, 1}


def test_generate_partition_passes_arguments_through_unaltered(adapter,
                                                               upstream):
    """The adapter is a thin wrapper; this pins what "thin" means.

    Recorded on the adapter INSTANCE only. The pinned module is never
    patched: it is the audited subject.
    """
    main, cipher = upstream
    seen = {}

    class Recorder:
        def __init__(self, real):
            self._real = real

        def make_train_data(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return self._real.make_train_data(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    original = adapter._main
    adapter._main = Recorder(original)
    try:
        X, Y = adapter.generate_partition(64, num_rounds=3)
    finally:
        adapter._main = original

    assert not seen["kwargs"], (
        "the adapter passed keyword arguments; upstream's signature is "
        "positional and the order is part of the audited contract.")
    args = seen["args"]
    assert len(args) == 7, f"expected 7 positional arguments, got {len(args)}"
    enc, pb, kb, n, nr, dstate, dkey = args
    assert enc is cipher.encrypt
    assert pb == int(cipher.plain_bits)
    assert kb == int(cipher.key_bits)
    assert n == 64
    assert nr == 3, (
        "the num_rounds override was not forwarded; a partition would then be "
        "generated at a round count other than the one recorded in the "
        "manifest.")
    np.testing.assert_array_equal(dstate, adapter._delta_plain)
    assert dkey == 0
    # And the arrays come back untouched.
    assert X.dtype == np.uint8 and Y.dtype == np.uint8
    assert X.shape == (64, adapter.feature_bits)


def test_num_rounds_default_is_the_constructed_value(adapter, upstream):
    seen = {}
    original = adapter._main

    class Recorder:
        def make_train_data(self, *args, **kwargs):
            seen["nr"] = args[4]
            return original.make_train_data(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(original, name)

    adapter._main = Recorder()
    try:
        adapter.generate_partition(64)
    finally:
        adapter._main = original
    assert seen["nr"] == ROUNDS


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_partition_refuses_empty_partition(adapter, bad):
    with pytest.raises(ValueError):
        adapter.generate_partition(bad)


# `bool` is a subclass of `int`, so a plain `int(x) >= 1` test accepts True.
# Without these, `samples=True` silently generated a ONE-ROW partition and
# `num_rounds=True` silently ran at ONE round while the manifest recorded
# `True` -- a mislabelled instance rather than an error. Both were real
# defects in this adapter, found by an independent review, and these tests
# exist so they cannot return.

def test_boolean_sample_count_is_refused_not_coerced(adapter):
    with pytest.raises(TypeError, match="bool"):
        adapter.generate_partition(True)


def test_boolean_round_override_is_refused_not_coerced(adapter):
    with pytest.raises(TypeError, match="bool"):
        adapter.generate_partition(64, num_rounds=True)


def test_boolean_round_count_is_refused_at_construction(root):
    with pytest.raises(TypeError, match="bool"):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=DELTA, num_rounds=True)


def test_generate_partition_refuses_nonpositive_round_override(adapter):
    with pytest.raises(ValueError):
        adapter.generate_partition(64, num_rounds=0)


# ---------------------------------------------------------------------
# 5. Non-replayability, recorded rather than worked around
# ---------------------------------------------------------------------
# AutoND draws from os.urandom. This is an UNFAVOURABLE property for an audit
# -- a dataset cannot be regenerated, only re-read -- and these tests exist to
# keep it true and declared, not to be "fixed". If someone later adds a seed
# parameter, this fails and the manifest's exact_replay_available claim must be
# revisited at the same time.

def test_identical_calls_produce_different_data(adapter):
    X1, Y1 = adapter.generate_partition(N_SMALL)
    X2, Y2 = adapter.generate_partition(N_SMALL)
    assert not np.array_equal(X1, X2), (
        "two identical calls produced identical features. Either the "
        "randomness source changed or something is seeding it; either way the "
        "exact_replay_available=False claim in the manifest is now wrong.")


def test_provenance_declares_non_replayability_honestly(adapter):
    prov = adapter.dataset_provenance(
        train_samples=10, validation_samples=20, test_samples=30)
    assert prov["exact_replay_available"] is False
    assert prov["generation_random_seed"] is None
    assert prov["generation_parameters"]["deterministic_seed"] is None
    assert prov["generation_parameters"]["randomness_source"] == "os.urandom"
    assert "os.urandom" in prov["exact_replay_note"]


def test_provenance_records_the_source_digests_it_imported(adapter, root):
    """The manifest must name the source BYTES, not just the configuration.

    Two checkouts can share a configuration and differ in code. Recorded
    rather than enforced: the adapter does not refuse a checkout whose digests
    differ from a declared pin, so this test also pins the fact that it is
    observational, to stop the note and the behaviour drifting apart.
    """
    import hashlib
    prov = adapter.dataset_provenance(
        train_samples=1, validation_samples=1, test_samples=1)
    digests = prov["source_digests"]
    # The whole import graph is recorded, not just the entry point and the
    # cipher: `main` imports `optimizer` and `train_nets`, and `train_nets`
    # imports `dbitnet` and `gohrnet`, so all of their code executes.
    assert set(digests) >= {"main.py", f"ciphers/{CIPHER}.py",
                            "optimizer.py", "train_nets.py",
                            "dbitnet.py", "gohrnet.py"}
    for rel, recorded in digests.items():
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert recorded == actual, (
            f"the recorded digest for {rel} does not match the file that was "
            f"imported; the manifest would misname the source.")
    assert "not enforced" in prov["source_digest_note"]


def test_randomness_source_claim_matches_the_pinned_source(upstream):
    """The manifest says os.urandom; check the source actually says so.

    A provenance field that is merely asserted by the adapter is the adapter's
    opinion. This reads the pinned generator.
    """
    main, _ = upstream
    src = Path(main.__file__).read_text()
    assert "urandom" in src, (
        "the pinned main.py contains no reference to urandom, but the adapter "
        "records randomness_source='os.urandom'. One of the two is wrong.")


# ---------------------------------------------------------------------
# 6. Provenance record: completeness and agreement with the instance
# ---------------------------------------------------------------------

def test_provenance_records_the_identifying_tuple(adapter):
    prov = adapter.dataset_provenance(
        train_samples=1, validation_samples=2, test_samples=3)
    gp = prov["generation_parameters"]
    assert prov["dataset_id"] == A.DATASET_ID
    assert gp["generator"] == A.GENERATOR_NAME == "main.make_train_data"
    assert gp["cipher"] == CIPHER
    assert gp["scenario"] == "single-key"
    assert gp["num_rounds"] == ROUNDS
    assert gp["train_samples"] == 1
    assert gp["validation_samples"] == 2
    assert gp["test_samples"] == 3
    # A dataset instance is identified by (cipher, scenario, delta_plain,
    # delta_key, num_rounds). All five must be recoverable from the record, or
    # two different instances could share a provenance block.
    diff = gp["input_difference"]
    assert diff["delta_plain_hex"] == hex(DELTA)
    assert diff["delta_key_hex"] == hex(0)
    assert diff["delta_plain_popcount"] == 1
    assert str(ROUNDS) in prov["dataset_version"]


def test_provenance_representation_matches_generated_data(adapter):
    prov = adapter.dataset_provenance(
        train_samples=1, validation_samples=1, test_samples=1)
    rep = prov["generation_parameters"]["representation"]
    X, _ = adapter.generate_partition(128)
    assert rep["feature_bits"] == X.shape[1], (
        "the manifest's declared feature width disagrees with the data the "
        "same adapter produces.")
    assert rep["c0_columns"] == [0, adapter.half_width]
    assert rep["c1_columns"] == [adapter.half_width, 2 * adapter.half_width]
    assert rep["c1_columns"][1] == rep["feature_bits"]
    assert rep["plain_bits"] == adapter.plain_bits
    assert rep["key_bits"] == adapter.key_bits
    assert rep["word_size"] == adapter.word_size


def test_provenance_is_a_plain_serialisable_record(adapter):
    import json
    prov = adapter.dataset_provenance(
        train_samples=1, validation_samples=1, test_samples=1)
    # Must round-trip without a custom encoder: a manifest that needs
    # default=str to serialise is a manifest whose values are not what they
    # appear to be on re-read.
    assert json.loads(json.dumps(prov)) == prov


# ---------------------------------------------------------------------
# 7. Configurations the adapter must refuse
# ---------------------------------------------------------------------
# Each of these would otherwise produce a dataset instance whose manifest
# misdescribes it. Refusing at construction time is what keeps a mislabelled
# instance from ever being persisted and audited.

def test_zero_plaintext_difference_is_refused(root):
    with pytest.raises(ValueError, match="identical plaintext pairs"):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=0, num_rounds=ROUNDS)


def test_difference_wider_than_the_block_is_refused(root, upstream):
    _, cipher = upstream
    too_wide = 1 << int(cipher.plain_bits)
    with pytest.raises(ValueError, match="plaintext bits"):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=too_wide, num_rounds=ROUNDS)


def test_negative_difference_is_refused(root):
    with pytest.raises(ValueError):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=-1, num_rounds=ROUNDS)


def test_nonzero_key_difference_in_single_key_is_refused(root):
    with pytest.raises(ValueError, match="related-key"):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=DELTA, num_rounds=ROUNDS,
                               delta_key_int=1, scenario="single-key")


def test_unknown_scenario_is_refused(root):
    with pytest.raises(ValueError, match="scenario"):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=DELTA, num_rounds=ROUNDS,
                               scenario="chosen-ciphertext")


@pytest.mark.parametrize("bad", [0, -3])
def test_nonpositive_round_count_is_refused(root, bad):
    with pytest.raises(ValueError, match="num_rounds"):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=DELTA, num_rounds=bad)


def test_num_rounds_has_no_default(root):
    """An omitted round count must be an error, not a silent default.

    Two AutoND datasets at different round counts are different instances. A
    default would let one provenance record describe either.
    """
    with pytest.raises(TypeError):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=DELTA)


def test_missing_checkout_is_refused_with_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="main.py"):
        A.AutoNDDatasetAdapter(autond_root=tmp_path, cipher_name=CIPHER,
                               delta_plain_int=DELTA, num_rounds=ROUNDS)


def test_unknown_cipher_is_refused(root):
    with pytest.raises(FileNotFoundError):
        A.AutoNDDatasetAdapter(autond_root=root, cipher_name="no_such_cipher",
                               delta_plain_int=DELTA, num_rounds=ROUNDS)


# ---------------------------------------------------------------------
# 8. The audited subject is left as it was found
# ---------------------------------------------------------------------

def test_import_does_not_leave_the_checkout_on_sys_path(root):
    """An audit process must not leave the audited subject importable.

    Otherwise a later `import main` anywhere in the process silently resolves
    to the audited code, and an unrelated module name could be shadowed by the
    checkout for the rest of the run.
    """
    import sys
    before = list(sys.path)
    A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                           delta_plain_int=DELTA, num_rounds=ROUNDS)
    assert str(root) not in [p for p in sys.path
                             if p not in before], (
        "the adapter left the pinned checkout on sys.path")


def test_related_key_scenario_builds_a_key_difference_vector(root, upstream):
    """Declared for completeness; NOT part of the current case study.

    The related-key path is reachable through the adapter, so its encoding is
    checked here rather than left untested. No related-key dataset instance
    has been built or audited, and this test is not evidence that one has.
    """
    main, cipher = upstream
    a = A.AutoNDDatasetAdapter(autond_root=root, cipher_name=CIPHER,
                               delta_plain_int=DELTA, num_rounds=ROUNDS,
                               delta_key_int=0x1, scenario="related-key")
    expected = main.integer_to_binary_array(0x1, int(cipher.key_bits))
    np.testing.assert_array_equal(a._delta_key, expected)
    assert a._delta_key.shape == (1, int(cipher.key_bits))
