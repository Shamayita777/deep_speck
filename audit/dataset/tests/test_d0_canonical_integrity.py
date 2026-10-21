"""
Integrity tests for the D0 canonical dataset instance
(audit.dataset.canonical).

Scientific status
-----------------
NON-EVIDENTIARY. This suite validates that canonical-instance construction and
verification agree, and that verification FAILS on tampering. It does not
establish that any real dataset is clean under D0 or any other dimension, and
it is not a substitute for building and verifying a real instance.

What this suite exists to prevent
---------------------------------
Schema 1 computed `canonical_instance_id` inline in `build()` and `verify()`
never recomputed it. A manifest whose id had been edited still verified, and
`verify()` RETURNED the edited id, which `binding_record()` then copied into
every dimension's certificate as the instance that was audited. The identity
binding all the evidence together was never checked against the data it named.

Method
------
Every fixture is a REAL canonical instance built through the real
`canonical.build()`, using a deterministic synthetic generator registered into
`GENERATOR_RESOLVERS` for the duration of a test. Nothing is hand-assembled,
so the tests exercise the production construction path rather than a
reimplementation of it. Partitions are a few rows: this suite tests manifest
integrity, which is size-independent, and running a production-size dataset to
test it would be waste.

SECURITY QUALIFICATION
----------------------
These checks detect corruption and piecemeal edits: changed array bytes, a
changed recorded digest, a changed identifier, a removed field. They are NOT
independent authentication. An attacker who can rewrite the partition files
AND the whole manifest can produce a self-consistent instance that verifies
successfully, because every value verification compares against lives inside
the instance directory. Detecting whole-manifest replacement requires
anchoring `canonical_instance_id` to a digest or signature kept outside that
directory — a trusted ledger, a signed release, or a commit. No test here
claims otherwise, and `verify()` states the same limitation in its
`verification_scope` field.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audit.dataset import canonical as C

TEST_TARGET = "test-synthetic-deterministic"


def _synthetic_resolver(spec: dict):
    """A deterministic generator, so a fixture is reproducible from its seed.

    Deliberately NOT the AutoND or Gohr generator: this suite is about
    manifest integrity, and binding it to a real generator would make it slow
    and would couple it to a pinned third-party checkout it does not need.
    """
    seed = int(spec.get("seed", 12345))
    width = int(spec.get("feature_bits", 8))

    def generate(n: int):
        rng = np.random.default_rng(seed + n)
        X = rng.integers(0, 2, size=(n, width), dtype=np.uint8)
        Y = rng.integers(0, 2, size=n, dtype=np.uint8)
        return X, Y

    return generate, f"{TEST_TARGET}.generate(seed={seed})"


def _spec(**over) -> dict:
    spec = {
        "target": TEST_TARGET,
        "dataset_id": "test-synthetic",
        "generator": "test.synthetic",
        "randomness_source": "numpy.default_rng(seed)",
        "seed": 12345,
        "feature_bits": 8,
        "train_samples": 40,
        "validation_samples": 10,
        "test_samples": 10,
    }
    spec.update(over)
    return spec


@pytest.fixture
def instance(tmp_path, monkeypatch):
    """A real, freshly built canonical instance in a temp directory."""
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _synthetic_resolver)
    out = tmp_path / "d0"
    man = C.build(out, repo_root=tmp_path, spec=_spec(), log=lambda *a, **k: None)
    return out, man


def _read(out: Path) -> dict:
    return json.loads((out / C.MANIFEST_NAME).read_text())


def _write(out: Path, man: dict) -> None:
    (out / C.MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))


def _verify(out: Path) -> dict:
    return C.verify(out, log=lambda *a, **k: None)


# ---------------------------------------------------------------------
# 1. The control: an untouched instance verifies
# ---------------------------------------------------------------------

def test_untouched_instance_verifies(instance):
    out, man = instance
    report = _verify(out)
    assert report["verified"] is True
    assert report["identity"]["canonical_instance_id_consistent"] is True
    assert report["identity"]["specification_consistent"] is True
    assert report["canonical_instance_id"] == man["canonical_instance_id"]
    for p in C.PARTITIONS:
        assert report["partitions"][p]["verified"] is True


def test_build_and_verify_agree_on_the_identifier(instance):
    """One function computes the id; both sides must call it."""
    out, man = instance
    assert C.compute_instance_id(man["partitions"]) == man["canonical_instance_id"]
    assert (C.compute_specification_digest(man["specification"])
            == man["specification_sha256"])


# ---------------------------------------------------------------------
# 2. Tampering with the persisted DATA
# ---------------------------------------------------------------------

def test_modified_partition_data_is_detected(instance):
    out, man = instance
    path = out / man["partitions"]["train"]["file"]
    with np.load(path) as z:
        X, Y = z["X"].copy(), z["Y"].copy()
    X[0, 0] ^= 1                      # flip exactly one bit
    np.savez(path, X=X, Y=Y)
    report = _verify(out)
    assert report["verified"] is False
    assert report["partitions"]["train"]["verified"] is False
    assert "hash differs" in report["partitions"]["train"]["reason"]


def test_modified_labels_are_detected(instance):
    out, man = instance
    path = out / man["partitions"]["test"]["file"]
    with np.load(path) as z:
        X, Y = z["X"].copy(), z["Y"].copy()
    Y[0] ^= 1
    np.savez(path, X=X, Y=Y)
    report = _verify(out)
    assert report["verified"] is False
    assert report["partitions"]["test"]["verified"] is False


def test_missing_partition_file_is_detected(instance):
    out, man = instance
    (out / man["partitions"]["validation"]["file"]).unlink()
    report = _verify(out)
    assert report["verified"] is False
    assert report["partitions"]["validation"]["verified"] is False


# ---------------------------------------------------------------------
# 3. Tampering with a RECORDED partition hash
# ---------------------------------------------------------------------

def test_modified_recorded_partition_hash_is_detected(instance):
    """Editing the digest, not the data.

    This also changes the instance id derived from it, so BOTH the partition
    check and the identity check must fire. Catching only one would mean a
    matched pair of edits could pass.
    """
    out, man = instance
    man = _read(out)
    man["partitions"]["train"]["features_sha256"] = "0" * 64
    _write(out, man)
    report = _verify(out)
    assert report["verified"] is False
    assert report["partitions"]["train"]["verified"] is False
    assert report["identity"]["canonical_instance_id_consistent"] is False


def test_modified_recorded_file_hash_is_detected(instance):
    out, man = instance
    man = _read(out)
    man["partitions"]["validation"]["file_sha256"] = "1" * 64
    _write(out, man)
    report = _verify(out)
    assert report["verified"] is False
    assert "file hash differs" in report["partitions"]["validation"]["reason"]


# ---------------------------------------------------------------------
# 4. Tampering with the IDENTIFIER — the gap this suite was written for
# ---------------------------------------------------------------------

def test_modified_canonical_instance_id_is_detected(instance):
    out, _ = instance
    man = _read(out)
    original = man["canonical_instance_id"]
    man["canonical_instance_id"] = "f" * 64
    _write(out, man)
    report = _verify(out)
    assert report["verified"] is False, (
        "an edited canonical_instance_id verified successfully; this is the "
        "exact schema-1 defect this check exists to close.")
    assert report["identity"]["canonical_instance_id_consistent"] is False
    assert report["identity"]["canonical_instance_id_recorded"] == "f" * 64
    assert report["identity"]["canonical_instance_id_recomputed"] == original


def test_verify_returns_the_recomputed_id_not_the_recorded_one(instance):
    """A caller must never be handed an id just shown to be inconsistent."""
    out, _ = instance
    man = _read(out)
    original = man["canonical_instance_id"]
    man["canonical_instance_id"] = "f" * 64
    _write(out, man)
    report = _verify(out)
    assert report["canonical_instance_id"] == original
    assert report["canonical_instance_id"] != "f" * 64


def test_binding_record_refuses_an_inconsistent_identity(instance):
    """Close the propagation path into certificates.

    binding_record() is what a dimension embeds to declare which instance it
    audited. If it exports a contradicted id, every downstream certificate
    carries it.
    """
    out, _ = instance
    man = _read(out)
    man["canonical_instance_id"] = "e" * 64
    _write(out, man)
    with pytest.raises(SystemExit, match="canonical_instance_id"):
        C.binding_record(out)


def test_restoring_the_correct_id_restores_consistency(instance):
    """Recomputing from the correct manifest values makes it verify again."""
    out, _ = instance
    man = _read(out)
    good = man["canonical_instance_id"]
    man["canonical_instance_id"] = "a" * 64
    _write(out, man)
    assert _verify(out)["verified"] is False

    man = _read(out)
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    _write(out, man)
    report = _verify(out)
    assert report["verified"] is True
    assert report["canonical_instance_id"] == good
    assert C.binding_record(out)["canonical_instance_id"] == good


# ---------------------------------------------------------------------
# 4b. Manifest METADATA must agree with the persisted arrays
# ---------------------------------------------------------------------
# Hash checks verify the bytes, not the manifest's description of them. Each
# edit below leaves every array digest untouched, and every one of them passed
# verification before `check_partition_metadata` existed. The identifier and
# specification digest are recomputed after each edit, so the attack is
# self-consistent everywhere except in the metadata itself.

def _tamper_metadata(out: Path, mutate) -> dict:
    man = _read(out)
    mutate(man)
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(
        man["specification"])
    _write(out, man)
    return _verify(out)


def test_wrong_recorded_sample_count_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out, lambda m: m["partitions"]["train"].__setitem__("samples", 999999))
    assert report["verified"] is False
    assert report["partitions"]["train"]["verified"] is False
    assert any("samples" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_wrong_recorded_feature_shape_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["train"]["features"].__setitem__(
            "shape", [1, 1]))
    assert report["verified"] is False
    assert any("features.shape" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_wrong_recorded_feature_dtype_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["train"]["features"].__setitem__(
            "dtype", "float64"))
    assert report["verified"] is False
    assert any("features.dtype" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_wrong_recorded_label_dtype_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["validation"]["labels"].__setitem__(
            "dtype", "int64"))
    assert report["verified"] is False
    assert report["partitions"]["validation"]["verified"] is False


def test_wrong_recorded_ndim_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["test"]["features"].__setitem__("ndim", 3))
    assert report["verified"] is False


def test_wrong_contiguity_claim_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["train"]["features"].__setitem__(
            "c_contiguous", False))
    assert report["verified"] is False


def test_wrong_label_counts_are_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["train"].__setitem__(
            "label_counts", {"positive": 99999, "negative": 1}))
    assert report["verified"] is False
    assert any("label_counts" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_swapped_label_counts_are_detected(instance):
    """A plausible edit: positive and negative exchanged.

    The pair still sums to `samples`, so a sum check alone would miss it.
    """
    out, _ = instance
    def swap(m):
        lc = m["partitions"]["train"]["label_counts"]
        m["partitions"]["train"]["label_counts"] = {
            "positive": lc["negative"], "negative": lc["positive"]}
    report = _tamper_metadata(out, swap)
    assert report["verified"] is False


def test_negative_label_count_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["train"].__setitem__(
            "label_counts", {"positive": -1, "negative": 41}))
    assert report["verified"] is False


def test_label_counts_not_summing_to_samples_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out,
        lambda m: m["partitions"]["train"].__setitem__(
            "label_counts", {"positive": 1, "negative": 1}))
    assert report["verified"] is False


@pytest.mark.parametrize("block", ["features", "labels"])
def test_deleted_representation_block_is_detected(instance, block):
    out, _ = instance
    report = _tamper_metadata(
        out, lambda m: m["partitions"]["train"].pop(block))
    assert report["verified"] is False
    assert any("representation metadata" in p
               for p in report["partitions"]["train"]["metadata_problems"])


@pytest.mark.parametrize("field", C.REQUIRED_REPRESENTATION_FIELDS)
def test_deleted_representation_field_is_detected(instance, field):
    out, _ = instance
    report = _tamper_metadata(
        out, lambda m: m["partitions"]["train"]["features"].pop(field))
    assert report["verified"] is False


def test_non_integer_sample_count_is_detected(instance):
    out, _ = instance
    report = _tamper_metadata(
        out, lambda m: m["partitions"]["train"].__setitem__("samples", "40"))
    assert report["verified"] is False


def test_inconsistent_feature_and_label_row_counts_are_detected(instance):
    """Truncate the labels on disk and rebuild every digest and count.

    The manifest is then entirely self-consistent for each array taken alone;
    what is wrong is that the two arrays no longer describe the same rows.
    """
    out, man = instance
    from audit.dataset.common.provenance import array_sha256, sha256_file
    rec = man["partitions"]["train"]
    path = out / rec["file"]
    with np.load(path) as z:
        X, Y = z["X"].copy(), z["Y"].copy()
    Y = np.ascontiguousarray(Y[:-5])              # fewer label rows than X
    np.savez(path, X=X, Y=Y)

    m = _read(out)
    r = m["partitions"]["train"]
    r["features_sha256"] = array_sha256(np.ascontiguousarray(X))
    r["labels_sha256"] = array_sha256(Y)
    r["file_sha256"] = sha256_file(path)
    r["features"] = {"dtype": str(X.dtype), "shape": list(X.shape),
                     "ndim": int(X.ndim), "c_contiguous": True,
                     "f_contiguous": bool(X.flags.f_contiguous)}
    r["labels"] = {"dtype": str(Y.dtype), "shape": list(Y.shape),
                   "ndim": int(Y.ndim), "c_contiguous": True,
                   "f_contiguous": True}
    pos = int((Y == 1).sum())
    r["samples"] = int(Y.size)
    r["label_counts"] = {"positive": pos, "negative": int(Y.size - pos)}
    m["canonical_instance_id"] = C.compute_instance_id(m["partitions"])
    m["specification_sha256"] = C.compute_specification_digest(
        m["specification"])
    _write(out, m)

    report = _verify(out)
    assert report["verified"] is False
    probs = report["partitions"]["train"]["metadata_problems"]
    assert any("row count" in p or "rows" in p for p in probs), probs


def test_metadata_check_passes_on_an_untouched_instance(instance):
    """The control: the checker must not fire on good data."""
    out, man = instance
    for name in C.PARTITIONS:
        X, Y = C.load_partition(out, name, spec=man["specification"])
        assert C.check_partition_metadata(man["partitions"][name], X, Y) == []


def test_describe_partition_is_what_build_recorded(instance):
    """Construction and checking share one definition."""
    out, man = instance
    from audit.dataset.common.provenance import sha256_file
    for name in C.PARTITIONS:
        X, Y = C.load_partition(out, name, spec=man["specification"])
        rec = man["partitions"][name]
        rebuilt = C.describe_partition(
            X, Y, file_name=rec["file"],
            file_sha256=sha256_file(out / rec["file"]))
        assert rebuilt == rec


# ---------------------------------------------------------------------
# 4b-ii. The SPECIFICATION must agree with the persisted arrays
# ---------------------------------------------------------------------
# `check_partition_metadata` compares the arrays with the partition RECORD.
# These tests tamper only with the SPECIFICATION, leaving every array, every
# partition record and every digest untouched, then reseal the identifier and
# the specification digest. Before `check_partition_against_specification`
# existed, all of them verified: a manifest could declare 10,000,000 training
# samples over a persisted 40-row array and be called VERIFIED.


def _tamper_spec(out: Path, mutate) -> dict:
    man = _read(out)
    mutate(man["specification"])
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(
        man["specification"])
    _write(out, man)
    return _verify(out)


@pytest.fixture
def typed_instance(tmp_path, monkeypatch):
    """An instance whose specification declares dtype, labels and width.

    The base fixture's spec declares only sample counts, so the
    representation and label-alphabet checks would have nothing to compare
    against.
    """
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _synthetic_resolver)
    out = tmp_path / "d0typed"
    spec = _spec(representation={"feature_bits": 8},
                 feature_dtype="uint8", label_dtype="uint8",
                 label_values=[0, 1])
    man = C.build(out, repo_root=tmp_path, spec=spec, log=lambda *a, **k: None)
    return out, man


def test_typed_instance_verifies_untouched(typed_instance):
    out, _ = typed_instance
    assert _verify(out)["verified"] is True


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_declared_sample_count_contradicting_the_arrays_is_detected(
        typed_instance, partition):
    out, _ = typed_instance
    report = _tamper_spec(
        out, lambda s: s.__setitem__(f"{partition}_samples", 10_000_000))
    assert report["verified"] is False
    assert report["partitions"][partition]["verified"] is False
    assert any("specification declares" in p
               for p in report["partitions"][partition]["metadata_problems"])


def test_declared_feature_bits_contradicting_the_arrays_is_detected(
        typed_instance):
    out, _ = typed_instance
    report = _tamper_spec(
        out, lambda s: s["representation"].__setitem__("feature_bits", 64))
    assert report["verified"] is False
    assert any("feature_bits" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_declared_feature_dtype_contradicting_the_arrays_is_detected(
        typed_instance):
    out, _ = typed_instance
    report = _tamper_spec(
        out, lambda s: s.__setitem__("feature_dtype", "float32"))
    assert report["verified"] is False
    assert any("feature_dtype" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_declared_label_dtype_contradicting_the_arrays_is_detected(
        typed_instance):
    out, _ = typed_instance
    report = _tamper_spec(out, lambda s: s.__setitem__("label_dtype", "int64"))
    assert report["verified"] is False


def test_declared_label_values_excluding_an_actual_label_is_detected(
        typed_instance):
    """The arrays contain both classes; the spec now claims only one."""
    out, _ = typed_instance
    report = _tamper_spec(out, lambda s: s.__setitem__("label_values", [0]))
    assert report["verified"] is False
    assert any("label_values" in p
               for p in report["partitions"]["train"]["metadata_problems"])


def test_declared_label_values_of_the_wrong_alphabet_is_detected(
        typed_instance):
    out, _ = typed_instance
    report = _tamper_spec(out, lambda s: s.__setitem__("label_values", [7, 8]))
    assert report["verified"] is False


def test_non_integer_declared_sample_count_is_detected(typed_instance):
    out, _ = typed_instance
    report = _tamper_spec(out, lambda s: s.__setitem__("train_samples", "40"))
    assert report["verified"] is False


def test_malformed_declared_label_values_is_detected(typed_instance):
    out, _ = typed_instance
    report = _tamper_spec(
        out, lambda s: s.__setitem__("label_values", "zero-or-one"))
    assert report["verified"] is False


def test_specification_check_passes_on_an_untouched_instance(typed_instance):
    """The control: the spec checker must not fire on a conforming instance."""
    out, man = typed_instance
    for name in C.PARTITIONS:
        X, Y = C.load_partition(out, name, spec=man["specification"])
        assert C.check_partition_against_specification(
            name, X, Y, man["specification"]) == []


def test_specification_without_declarations_is_checked_for_consistency_only(
        instance):
    """A spec that declares no representation must not fail spuriously.

    The base fixture declares only sample counts. Absent declarations are
    skipped rather than treated as mismatches, so a target that does not
    declare a representation still verifies.
    """
    out, man = instance
    for name in C.PARTITIONS:
        X, Y = C.load_partition(out, name, spec=man["specification"])
        assert C.check_partition_against_specification(
            name, X, Y, man["specification"]) == []
    assert _verify(out)["verified"] is True


# ---------------------------------------------------------------------
# 4c. The builder refuses non-conforming data BEFORE persisting it
# ---------------------------------------------------------------------

def _bad_resolver(make):
    def resolver(spec):
        return make, "bad.generator"
    return resolver


def test_builder_refuses_a_wrong_sized_partition(tmp_path, monkeypatch):
    """A generator returning the wrong count must not be persisted."""
    def gen(n):
        rng = np.random.default_rng(3)
        m = n - 1                                  # one row short
        return (rng.integers(0, 2, size=(m, 8), dtype=np.uint8),
                rng.integers(0, 2, size=m, dtype=np.uint8))
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _bad_resolver(gen))
    out = tmp_path / "d0"
    with pytest.raises(SystemExit) as exc:
        C.build(out, repo_root=tmp_path, spec=_spec(), log=lambda *a, **k: None)
    assert "REFUSED" in str(exc.value)
    assert "requested" in str(exc.value)
    assert not (out / C.MANIFEST_NAME).exists(), (
        "a manifest was written for a non-conforming partition")


def test_builder_refuses_mismatched_row_counts(tmp_path, monkeypatch):
    def gen(n):
        rng = np.random.default_rng(3)
        return (rng.integers(0, 2, size=(n, 8), dtype=np.uint8),
                rng.integers(0, 2, size=n - 2, dtype=np.uint8))
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _bad_resolver(gen))
    with pytest.raises(SystemExit, match="REFUSED"):
        C.build(tmp_path / "d0", repo_root=tmp_path, spec=_spec(),
                log=lambda *a, **k: None)


def test_builder_refuses_a_declared_representation_mismatch(tmp_path,
                                                            monkeypatch):
    """Declared feature_bits must match what the generator produced."""
    def gen(n):
        rng = np.random.default_rng(3)
        return (rng.integers(0, 2, size=(n, 8), dtype=np.uint8),
                rng.integers(0, 2, size=n, dtype=np.uint8))
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _bad_resolver(gen))
    spec = _spec(representation={"feature_bits": 64})      # but 8 produced
    with pytest.raises(SystemExit) as exc:
        C.build(tmp_path / "d0", repo_root=tmp_path, spec=spec,
                log=lambda *a, **k: None)
    assert "feature_bits" in str(exc.value)


def test_builder_refuses_a_declared_dtype_mismatch(tmp_path, monkeypatch):
    def gen(n):
        rng = np.random.default_rng(3)
        return (rng.integers(0, 2, size=(n, 8)).astype(np.uint16),
                rng.integers(0, 2, size=n, dtype=np.uint8))
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _bad_resolver(gen))
    with pytest.raises(SystemExit, match="uint8"):
        C.build(tmp_path / "d0", repo_root=tmp_path,
                spec=_spec(feature_dtype="uint8"), log=lambda *a, **k: None)


def test_builder_refuses_out_of_range_label_values(tmp_path, monkeypatch):
    def gen(n):
        rng = np.random.default_rng(3)
        Y = rng.integers(0, 2, size=n, dtype=np.uint8)
        Y[0] = 7
        return rng.integers(0, 2, size=(n, 8), dtype=np.uint8), Y
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _bad_resolver(gen))
    with pytest.raises(SystemExit, match="label_values"):
        C.build(tmp_path / "d0", repo_root=tmp_path,
                spec=_spec(label_values=[0, 1]), log=lambda *a, **k: None)


def test_builder_refuses_a_single_class_partition(tmp_path, monkeypatch):
    def gen(n):
        rng = np.random.default_rng(3)
        return (rng.integers(0, 2, size=(n, 8), dtype=np.uint8),
                np.ones(n, dtype=np.uint8))
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _bad_resolver(gen))
    with pytest.raises(SystemExit, match="single class"):
        C.build(tmp_path / "d0", repo_root=tmp_path, spec=_spec(),
                log=lambda *a, **k: None)


def test_builder_accepts_conforming_data_and_the_result_verifies(instance):
    """The control for the builder checks."""
    out, _ = instance
    assert _verify(out)["verified"] is True


# ---------------------------------------------------------------------
# 5. Identity behaviour of ordering and specification — defined explicitly
# ---------------------------------------------------------------------

def test_partition_order_is_part_of_the_identity(instance):
    """Swapping which partition is train and which is test changes the id.

    `compute_instance_id` concatenates in PARTITIONS order, so a dataset whose
    train and test roles are exchanged is a DIFFERENT instance. Asserted here
    so the property is pinned rather than incidental.
    """
    _, man = instance
    parts = man["partitions"]
    straight = C.compute_instance_id(parts)
    swapped = dict(parts)
    swapped["train"], swapped["test"] = parts["test"], parts["train"]
    assert C.compute_instance_id(swapped) != straight


def test_instance_id_is_independent_of_manifest_key_order(instance):
    """Re-serialising a manifest must not change its identity."""
    _, man = instance
    parts = man["partitions"]
    reordered = {k: parts[k] for k in reversed(list(parts))}
    assert C.compute_instance_id(reordered) == C.compute_instance_id(parts)


def test_specification_change_does_not_alter_the_instance_id(instance):
    """Stated behaviour, not an accident.

    The id is defined over the DATA. A changed specification therefore leaves
    it unchanged, which is why the specification needs its own digest; folding
    the specification into the id would have invalidated every existing
    instance's identifier.
    """
    out, man = instance
    before = C.compute_instance_id(man["partitions"])
    altered = dict(man["specification"])
    altered["num_rounds"] = 999
    assert C.compute_instance_id(man["partitions"]) == before
    assert (C.compute_specification_digest(altered)
            != C.compute_specification_digest(man["specification"]))


def test_modified_specification_is_detected_by_its_digest(instance):
    out, _ = instance
    man = _read(out)
    man["specification"]["train_samples"] = 999999
    _write(out, man)
    report = _verify(out)
    assert report["verified"] is False
    assert report["identity"]["specification_consistent"] is False


def test_specification_digest_is_key_order_independent(instance):
    _, man = instance
    spec = man["specification"]
    reordered = {k: spec[k] for k in reversed(list(spec))}
    assert (C.compute_specification_digest(reordered)
            == C.compute_specification_digest(spec))


# ---------------------------------------------------------------------
# 6. Malformed and incomplete manifests fail safely
# ---------------------------------------------------------------------

@pytest.mark.parametrize("field", C.REQUIRED_MANIFEST_FIELDS)
def test_missing_required_manifest_field_fails_safely(instance, field):
    out, _ = instance
    man = _read(out)
    del man[field]
    _write(out, man)
    with pytest.raises(SystemExit) as exc:
        _verify(out)
    assert "REFUSED" in str(exc.value)


@pytest.mark.parametrize("field", C.REQUIRED_PARTITION_FIELDS)
def test_missing_required_partition_field_fails_safely(instance, field):
    out, _ = instance
    man = _read(out)
    del man["partitions"]["train"][field]
    _write(out, man)
    with pytest.raises(SystemExit) as exc:
        _verify(out)
    assert "REFUSED" in str(exc.value)


def test_missing_partition_entry_fails_safely(instance):
    out, _ = instance
    man = _read(out)
    del man["partitions"]["test"]
    _write(out, man)
    with pytest.raises(SystemExit, match="REFUSED"):
        _verify(out)


def test_unsupported_schema_version_fails_safely(instance):
    out, _ = instance
    man = _read(out)
    man["schema_version"] = "d0-canonical-instance-99"
    _write(out, man)
    with pytest.raises(SystemExit, match="schema_version"):
        _verify(out)


def test_deleting_the_specification_digest_is_refused_not_skipped(instance):
    """Removing a field must not be a way to opt out of a check."""
    out, _ = instance
    man = _read(out)
    del man["specification_sha256"]
    _write(out, man)
    with pytest.raises(SystemExit, match="specification_sha256"):
        _verify(out)


@pytest.mark.parametrize("bad", ["", "short", "g" * 64, 12345, None])
def test_malformed_instance_id_fails_safely(instance, bad):
    out, _ = instance
    man = _read(out)
    man["canonical_instance_id"] = bad
    _write(out, man)
    with pytest.raises(SystemExit, match="REFUSED"):
        _verify(out)


def test_non_object_partitions_fails_safely(instance):
    out, _ = instance
    man = _read(out)
    man["partitions"] = ["train", "validation", "test"]
    _write(out, man)
    with pytest.raises(SystemExit, match="REFUSED"):
        _verify(out)


# ---------------------------------------------------------------------
# 7. Backward compatibility with schema 1
# ---------------------------------------------------------------------

def test_legacy_schema_1_instance_still_verifies(instance):
    """An instance built before `specification_sha256` existed must still pass.

    Its specification is simply not covered by a digest, which verification
    reports rather than silently treating as verified.
    """
    out, _ = instance
    man = _read(out)
    man["schema_version"] = C.LEGACY_SCHEMA_VERSION
    del man["specification_sha256"]
    _write(out, man)
    report = _verify(out)
    assert report["verified"] is True
    assert report["identity"]["canonical_instance_id_consistent"] is True
    assert report["identity"]["specification_consistent"] is None
    assert "NOT covered" in report["identity"]["specification_note"]


def test_legacy_instance_still_detects_an_edited_identifier(instance):
    """The identity check is not conditional on the new schema."""
    out, _ = instance
    man = _read(out)
    man["schema_version"] = C.LEGACY_SCHEMA_VERSION
    del man["specification_sha256"]
    man["canonical_instance_id"] = "b" * 64
    _write(out, man)
    assert _verify(out)["verified"] is False


# ---------------------------------------------------------------------
# 8. The limitation, asserted so it stays documented
# ---------------------------------------------------------------------

def test_verify_states_that_it_is_not_independent_authentication(instance):
    out, _ = instance
    scope = _verify(out)["verification_scope"]
    assert "NOT independent authentication" in scope
    assert "outside the instance directory" in scope


def test_a_fully_rewritten_manifest_and_data_verifies(instance, tmp_path,
                                                      monkeypatch):
    """Demonstrates the limitation instead of overclaiming.

    An attacker who rewrites the data AND recomputes every digest and the
    identifier produces a self-consistent instance that verifies. This test
    asserts that it DOES verify, so nobody reads the tamper tests above as
    protection against whole-manifest replacement.
    """
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _synthetic_resolver)
    out, original = instance
    original_id = original["canonical_instance_id"]

    # Rewrite every partition with different data, then rebuild the ENTIRE
    # partition record from the new bytes -- digests, shapes, dtypes, sample
    # counts and label counts -- exactly as build() does, via the same
    # function. An attacker who rewrites only the digests is caught by
    # check_partition_metadata; this test is about one who does the whole job.
    from audit.dataset.common.provenance import sha256_file
    man = _read(out)
    rng = np.random.default_rng(999)
    for name in C.PARTITIONS:
        rec = man["partitions"][name]
        n = rec["samples"]
        X = np.ascontiguousarray(rng.integers(0, 2, size=(n, 8), dtype=np.uint8))
        Y = np.ascontiguousarray(rng.integers(0, 2, size=n, dtype=np.uint8))
        path = out / rec["file"]
        np.savez(path, X=X, Y=Y)
        man["partitions"][name] = C.describe_partition(
            X, Y, file_name=rec["file"], file_sha256=sha256_file(path))
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(
        man["specification"])
    _write(out, man)

    report = _verify(out)
    assert report["verified"] is True, (
        "a self-consistently rewritten instance should verify; if it does "
        "not, this test's premise has changed and the documented limitation "
        "needs revisiting.")
    # ...but it is a DIFFERENT instance, and that is the only thing an
    # outside observer could notice — which is why the id must be anchored
    # somewhere outside this directory.
    assert report["canonical_instance_id"] != original_id
