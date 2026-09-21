"""
Validation test suite for Dataset Integrity Audit D1
(audit.dataset.d1_duplicate_detection).

Scientific status
------------------
This suite is NON-EVIDENTIARY. It validates that the production D1
implementation behaves correctly on independently constructed known-clean
and known-violation synthetic fixtures (see d1_test_fixtures.py). It does
NOT establish, and must never be cited as establishing:

    - that the Gohr/Speck dataset is clean under D1 or any other audit;
    - statistical independence of any dataset;
    - absence of near duplicates;
    - any property outside D1's explicitly documented scope (see the
      module docstring of audit.dataset.d1_duplicate_detection).

The actual Gohr D1 result is produced separately, on the real generated
dataset, by audit/dataset/d1_gohr_exact_census.py. This suite is
implementation-level validation that is intended to run *before* that
production execution, not a substitute for it.

Every expected value below is derived from the fixture's construction (or
from D1's own documented k-1 multiplicity formula and set-based overlap
definition) -- never by calling the production function under test and
trusting its own output as the expected result. See d1_test_fixtures.py for
the fixture definitions and the reasoning behind each expected value.

This suite deliberately does not test near-duplicate detection, statistical
independence, generation-order effects, metadata leakage, distributional
equivalence, model/cryptographic validity, or any other audit component
(D2-D5, Experimental Validity, Cryptographic Evidence). Those are out of
scope for D1 and out of scope for this suite.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from audit.dataset import d1_duplicate_detection as d1
from audit.dataset.tests import d1_test_fixtures as fx


# ============================================================
# Category A / D -- known-clean null / known-no-overlap null.
# ============================================================

def test_clean_partitions_have_zero_internal_duplicates_and_overlap() -> None:
    """The known-clean fixture must show a valid schema, zero internal
    duplicates in every partition, zero cross-partition overlap for every
    compatible pair, and an overall PASS decision."""
    partitions = fx.clean_partitions()
    results = d1.audit_duplicates(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        representation_convention="synthetic fixture",
        feature_encoding="synthetic fixture",
    )

    assert results["schema"]["valid"] is True

    for name, stats in results["partitions"].items():
        assert stats["duplicate_groups"] == 0, name
        assert stats["duplicate_samples"] == 0, name
        assert stats["unique_samples"] == stats["total_samples"], name

    assert results["partition_overlap"]["train_validation"] == 0
    assert results["partition_overlap"]["train_test"] == 0
    assert results["partition_overlap"]["validation_test"] == 0

    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")
    assert certificate["decision"]["outcome"] == "PASS"


# ============================================================
# Category B -- internal exact duplicates.
# ============================================================

@pytest.mark.parametrize(
    "partitions_factory, duplicated_partition",
    [
        (fx.train_with_internal_duplicate, "train"),
        (fx.validation_with_internal_duplicate, "validation"),
        (fx.test_with_internal_duplicate, "test"),
    ],
    ids=["train", "validation", "test"],
)
def test_within_partition_exact_duplicate_is_detected(
    partitions_factory, duplicated_partition: str
) -> None:
    """Exactly one duplicated representation, isolated to one named
    partition, with no cross-partition overlap, must produce FAIL with
    duplicate_samples == 1 attributed to exactly that partition."""
    partitions = partitions_factory()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")

    assert certificate["decision"]["outcome"] == "FAIL"
    assert certificate["findings"]["duplicate_samples"] == 1
    assert certificate["findings"]["exact_partition_overlap"] == 0

    for name, stats in results["partitions"].items():
        expected = 1 if name == duplicated_partition else 0
        assert stats["duplicate_samples"] == expected, (
            f"unexpected duplicate_samples in partition {name!r}"
        )


def test_multiplicity_three_contributes_two_duplicate_samples() -> None:
    """A, A, A, B: independently known from D1's documented k-1 rule to
    yield duplicate_groups=1, duplicate_samples=2, maximum_multiplicity=3.
    This also serves as a smallest-partition-with-repetition edge case."""
    dataset = fx.multiplicity_three_fixture()
    stats = d1.partition_duplicate_statistics(dataset)

    assert stats["total_samples"] == 4
    assert stats["unique_samples"] == 2
    assert stats["duplicate_groups"] == 1
    assert stats["duplicate_samples"] == 2
    assert stats["maximum_multiplicity"] == 3


# ============================================================
# Category C -- cross-partition exact overlap.
# ============================================================

@pytest.mark.parametrize(
    "partitions_factory, overlap_key",
    [
        (fx.train_validation_overlap_fixture, "train_validation"),
        (fx.train_test_overlap_fixture, "train_test"),
        (fx.validation_test_overlap_fixture, "validation_test"),
    ],
    ids=["train_validation", "train_test", "validation_test"],
)
def test_exact_cross_partition_overlap_is_detected(
    partitions_factory, overlap_key: str
) -> None:
    """Exactly one shared exact representation between one named pair of
    partitions, with no internal duplicates anywhere, must produce FAIL
    with duplicate_samples == 0 (an overlap is not an internal duplicate)
    and overlap == 1 attributed to exactly that pair."""
    partitions = partitions_factory()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")

    assert certificate["decision"]["outcome"] == "FAIL"
    assert certificate["findings"]["duplicate_samples"] == 0

    for key, value in results["partition_overlap"].items():
        expected = 1 if key == overlap_key else 0
        assert value == expected, f"unexpected overlap for pair {key!r}"


def test_overlap_counts_unique_representations_not_occurrences() -> None:
    """B repeated within both train and validation, shared between them:
    overlap must be 1 (one unique shared representation), not the number of
    pairwise occurrence-matches (4) or the raw occurrence count (2). The
    internal-duplicate signal each partition separately carries for B must
    also be correct and independent of the overlap count."""
    partitions = fx.repeated_shared_representation_overlap_fixture()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )

    assert results["partitions"]["train"]["duplicate_samples"] == 1
    assert results["partitions"]["validation"]["duplicate_samples"] == 1
    assert results["partitions"]["test"]["duplicate_samples"] == 0

    assert results["partition_overlap"]["train_validation"] == 1
    assert results["partition_overlap"]["train_test"] == 0
    assert results["partition_overlap"]["validation_test"] == 0


# ============================================================
# Category E -- partition-size edge cases.
# ============================================================

def test_singleton_partitions_are_valid_and_clean() -> None:
    """One sample per partition, all mutually distinct: valid schema, one
    unique sample per partition, zero duplicates, overall PASS."""
    partitions = fx.singleton_partitions()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")

    assert results["schema"]["valid"] is True
    for name, stats in results["partitions"].items():
        assert stats["unique_samples"] == 1, name
        assert stats["duplicate_samples"] == 0, name
    assert certificate["decision"]["outcome"] == "PASS"


def test_empty_partition_is_rejected() -> None:
    """A partition with a sample dimension of length zero must be rejected
    with a ValueError before any duplicate/overlap computation -- never
    silently treated as PASS or INCONCLUSIVE."""
    clean = fx.clean_partitions()
    with pytest.raises(ValueError):
        d1.audit_duplicates(
            train=fx.EMPTY_PARTITION, validation=clean["validation"], test=clean["test"],
        )


def test_scalar_input_is_rejected() -> None:
    """A zero-dimensional (scalar) array has no sample dimension and must
    be rejected with a ValueError, never silently accepted."""
    clean = fx.clean_partitions()
    with pytest.raises(ValueError):
        d1.audit_duplicates(
            train=fx.SCALAR_INPUT, validation=clean["validation"], test=clean["test"],
        )


# ============================================================
# Category F -- schema integrity.
# ============================================================

def test_incompatible_partition_schema_is_invalid_and_overlap_is_not_fabricated() -> None:
    """validation uses a different per-sample shape than train/test. The
    overall schema must be invalid, and the two pairs touching validation
    must report overlap as None (not a fabricated 0) -- while train/test,
    which remain mutually schema-compatible, still get a real computed
    overlap value. The final decision must be FAIL, not PASS, regardless of
    that real 0."""
    partitions = fx.incompatible_schema_fixture()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )

    assert results["schema"]["valid"] is False
    assert results["partition_overlap"]["train_validation"] is None
    assert results["partition_overlap"]["validation_test"] is None
    assert results["partition_overlap"]["train_test"] == 0

    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")
    assert certificate["decision"]["outcome"] == "FAIL"


def test_expected_feature_schema_mismatch_is_detected() -> None:
    """An explicitly supplied expected_feature_schema that the actual data
    does not match must invalidate the schema and drive FAIL, even though
    the partitions are mutually consistent with each other."""
    partitions = fx.clean_partitions()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
        expected_feature_schema={"dtype": "uint16", "ndim": 2, "sample_shape": [4]},
    )

    assert results["schema"]["valid"] is False
    assert results["schema"]["expected_feature_mismatches"]

    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")
    assert certificate["decision"]["outcome"] == "FAIL"


def test_label_sample_count_mismatch_is_rejected() -> None:
    """A label array whose length disagrees with its partition's sample
    count must raise ValueError, never be silently truncated/padded."""
    partitions = fx.clean_partitions()
    wrong_length_train_labels = np.array([0, 1], dtype=np.int64)  # train has 3 samples
    validation_labels = np.zeros(len(partitions["validation"]), dtype=np.int64)
    test_labels = np.zeros(len(partitions["test"]), dtype=np.int64)

    with pytest.raises(ValueError):
        d1.audit_duplicates(
            train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
            train_labels=wrong_length_train_labels,
            validation_labels=validation_labels,
            test_labels=test_labels,
        )

def test_missing_labels_for_partition_are_rejected() -> None:
    """If labels are supplied for an audit, every audited partition must
    have a corresponding label array."""
    partitions = fx.clean_partitions()

    validation_labels = np.zeros(len(partitions["validation"]), dtype=np.int64)
    test_labels = np.zeros(len(partitions["test"]), dtype=np.int64)

    with pytest.raises(
        ValueError,
        match=r"Labels must be supplied for every audited partition; missing 'train'\.",
    ):
        d1.audit_duplicates(
            train=partitions["train"],
            validation=partitions["validation"],
            test=partitions["test"],
            validation_labels=validation_labels,
            test_labels=test_labels,
        )


def test_partially_supplied_labels_are_rejected() -> None:
    """Supplying labels for only some partitions must fail closed rather
    than silently performing a partially labelled audit."""
    partitions = fx.clean_partitions()

    train_labels = np.zeros(len(partitions["train"]), dtype=np.int64)

    with pytest.raises(ValueError, match="Labels must be supplied for every audited partition"):
        d1.audit_duplicates(
            train=partitions["train"],
            validation=partitions["validation"],
            test=partitions["test"],
            train_labels=train_labels,
        )


def test_cross_partition_label_schema_mismatch_invalidates_schema() -> None:
    """Feature partitions may be compatible while label representations
    differ across partitions. D1 must report that label-schema mismatch
    and mark the overall schema invalid."""
    partitions = fx.clean_partitions()

    train_labels = np.zeros(len(partitions["train"]), dtype=np.int64)
    validation_labels = np.zeros(len(partitions["validation"]), dtype=np.int32)
    test_labels = np.zeros(len(partitions["test"]), dtype=np.int64)

    results = d1.audit_duplicates(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        train_labels=train_labels,
        validation_labels=validation_labels,
        test_labels=test_labels,
    )

    assert results["schema"]["valid"] is False
    assert "validation" in results["schema"]["label_mismatches"]

    certificate = d1.build_d1_certificate(
        results,
        dataset_id="fixture-dataset",
    )
    assert certificate["decision"]["outcome"] == "FAIL"


def test_expected_label_schema_mismatch_invalidates_schema() -> None:
    """An explicitly supplied expected_label_schema that does not match
    the observed labels must invalidate the schema and drive FAIL."""
    partitions = fx.clean_partitions()

    train_labels = np.zeros(len(partitions["train"]), dtype=np.int64)
    validation_labels = np.zeros(len(partitions["validation"]), dtype=np.int64)
    test_labels = np.zeros(len(partitions["test"]), dtype=np.int64)

    results = d1.audit_duplicates(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        train_labels=train_labels,
        validation_labels=validation_labels,
        test_labels=test_labels,
        expected_label_schema={
            "dtype": "int32",
            "ndim": 1,
            "sample_shape": [],
        },
    )

    assert results["schema"]["valid"] is False
    assert results["schema"]["label_mismatches"]

    certificate = d1.build_d1_certificate(
        results,
        dataset_id="fixture-dataset",
    )
    assert certificate["decision"]["outcome"] == "FAIL"


def test_optional_test_partition_can_be_omitted() -> None:
    """D1's generic audit API permits test=None. A clean train/validation
    audit without a test partition must remain valid and PASS."""
    partitions = fx.clean_partitions()

    results = d1.audit_duplicates(
        train=partitions["train"],
        validation=partitions["validation"],
        test=None,
        representation_convention="synthetic fixture",
        feature_encoding="synthetic fixture",
    )

    assert results["schema"]["valid"] is True
    assert results["partitions"]["train"]["duplicate_samples"] == 0
    assert results["partitions"]["validation"]["duplicate_samples"] == 0
    assert results["partition_overlap"]["train_validation"] == 0

    decision = d1.evaluate_d1(results)
    assert decision["outcome"] == "PASS"
# ============================================================
# Category G -- label-conflict diagnostic.
# ============================================================

def test_exact_feature_label_conflict_is_reported() -> None:
    """Feature A occurring with two different labels must be reported as
    exactly one label-conflict group, with consistent == False."""
    features, labels = fx.label_conflict_features_and_labels()
    stats = d1.label_conflict_statistics(features, labels)

    assert stats["label_conflict_groups"] == 1
    assert stats["consistent"] is False


def test_label_diagnostic_does_not_alter_feature_duplicate_or_overlap_semantics() -> None:
    """Supplying labels must not change any feature-level duplicate/overlap
    number; it must only add a label_consistency sub-report."""
    partitions = fx.clean_partitions()
    features, labels = fx.label_conflict_features_and_labels()

    train = features
    validation = partitions["validation"]
    test = partitions["test"]
    validation_labels = np.zeros(len(validation), dtype=np.int64)
    test_labels = np.zeros(len(test), dtype=np.int64)

    results_without_labels = d1.audit_duplicates(train=train, validation=validation, test=test)
    results_with_labels = d1.audit_duplicates(
        train=train, validation=validation, test=test,
        train_labels=labels, validation_labels=validation_labels, test_labels=test_labels,
    )

    for name in ("train", "validation", "test"):
        for key in (
            "total_samples", "unique_samples", "duplicate_groups",
            "duplicate_samples", "maximum_multiplicity",
        ):
            assert (
                results_without_labels["partitions"][name][key]
                == results_with_labels["partitions"][name][key]
            ), f"{name}.{key} differed when labels were supplied"

    assert results_without_labels["partition_overlap"] == results_with_labels["partition_overlap"]

    assert "label_consistency" not in results_without_labels["partitions"]["train"]
    assert results_with_labels["partitions"]["train"]["label_consistency"]["label_conflict_groups"] == 1
    # This fixture's label conflict unavoidably co-occurs with a feature
    # duplicate (the same exact feature A must recur to carry two labels):
    assert results_with_labels["partitions"]["train"]["duplicate_samples"] == 1


def test_label_conflict_alone_does_not_drive_fail_outcome() -> None:
    """evaluate_d1's literal decision rule reads schema validity, per-
    partition duplicate_samples, and partition_overlap -- not
    label_conflict_groups. A hand-built minimal results object containing
    exactly the keys evaluate_d1 documents reading (with duplicate_samples
    and overlap held at zero) isolates a label conflict from the feature-
    duplicate signal it normally co-occurs with in a real audit_duplicates
    call, and shows directly that evaluate_d1 still returns PASS."""
    results = {
        "schema": {"valid": True},
        "partitions": {
            "train": {
                "duplicate_samples": 0,
                "label_consistency": {"label_conflict_groups": 1, "consistent": False},
            },
            "validation": {"duplicate_samples": 0},
        },
        "partition_overlap": {"train_validation": 0},
    }
    decision = d1.evaluate_d1(results)

    assert decision["outcome"] == "PASS"
    assert decision["label_conflict_groups"] == 1


# ============================================================
# Category H -- fingerprint semantics.
# ============================================================

def test_fingerprint_identical_samples_match() -> None:
    assert d1.sample_hash(fx.SAMPLE_A) == d1.sample_hash(fx.SAMPLE_A.copy())


def test_fingerprint_changed_value_differs() -> None:
    changed = fx.SAMPLE_A.copy()
    changed[-1] = 99
    assert d1.sample_hash(fx.SAMPLE_A) != d1.sample_hash(changed)


def test_fingerprint_commits_to_dtype() -> None:
    same_values_uint8 = np.array([0, 1, 2, 3], dtype=np.uint8)
    same_values_int32 = np.array([0, 1, 2, 3], dtype=np.int32)
    assert d1.sample_hash(same_values_uint8) != d1.sample_hash(same_values_int32)


def test_fingerprint_commits_to_shape() -> None:
    flat = np.array([0, 1, 2, 3], dtype=np.uint8)
    reshaped = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert d1.sample_hash(flat) != d1.sample_hash(reshaped)


def test_partition_hash_set_collapses_repeated_representations() -> None:
    dataset = fx.stack(fx.SAMPLE_A, fx.SAMPLE_A, fx.SAMPLE_B)
    assert len(d1.partition_hash_set(dataset)) == 2


# ============================================================
# Category I -- decision rule.
# ============================================================

@pytest.mark.parametrize(
    "partitions_factory, expected_outcome",
    [
        (fx.clean_partitions, "PASS"),
        (fx.train_with_internal_duplicate, "FAIL"),
        (fx.train_validation_overlap_fixture, "FAIL"),
    ],
    ids=["clean", "internal_duplicate", "cross_partition_overlap"],
)
def test_decision_rule_outcome_matches_fixture_construction(
    partitions_factory, expected_outcome: str
) -> None:
    partitions = partitions_factory()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    decision = d1.evaluate_d1(results)
    assert decision["outcome"] == expected_outcome


def test_decision_rule_fails_on_invalid_schema() -> None:
    partitions = fx.incompatible_schema_fixture()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    decision = d1.evaluate_d1(results)
    assert decision["outcome"] == "FAIL"
    assert decision["schema_valid"] is False


# ============================================================
# Category J -- certificate determinism, canonicalization, round-trip.
# ============================================================

def test_certificate_is_deterministic_when_inputs_and_provenance_are_fixed(monkeypatch) -> None:
    """Two certificates built from identical D1 findings, with the
    provenance boundary pinned to a fixed value, must be identical. This
    validates deterministic CERTIFICATE CONSTRUCTION given fixed inputs --
    it is not, and must never be read as, a claim about deterministic Gohr
    DATASET generation (Gohr's generator uses os.urandom() and is not
    deterministically replayable from a numerical seed; see
    d1_gohr_exact_census.py). build_provenance is monkeypatched to a fixed
    stand-in specifically so this test's correctness does not depend on
    whether the real build_provenance embeds any non-deterministic field
    (e.g. a timestamp) -- this isolates the deterministic portion under
    test, per the requested approach, without touching production code.
    """
    fixed_provenance = {"dataset_id": "fixture-dataset", "note": "fixed-for-determinism-test"}
    monkeypatch.setattr(d1, "build_provenance", lambda **kwargs: dict(fixed_provenance))

    partitions = fx.clean_partitions()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
        representation_convention="synthetic fixture", feature_encoding="synthetic fixture",
    )
    cert_kwargs = dict(
        dataset_id="fixture-dataset",
        dataset_version="fixture-v1",
        generation_procedure="deterministic synthetic fixture (see d1_test_fixtures.clean_partitions)",
        generation_parameters={"note": "fixed synthetic fixture, not Gohr/Speck generation"},
        random_seed=0,
    )

    cert1 = d1.build_d1_certificate(results, **cert_kwargs)
    cert2 = d1.build_d1_certificate(results, **cert_kwargs)

    assert cert1["generated_at_utc"] != cert2["generated_at_utc"]

    cert1_without_timestamp = dict(cert1)
    cert2_without_timestamp = dict(cert2)

    cert1_without_timestamp.pop("generated_at_utc", None)
    cert2_without_timestamp.pop("generated_at_utc", None)

    assert cert1_without_timestamp == cert2_without_timestamp


def _canonical_hash(value: object) -> str:
    """Test-only canonicalization helper (not production code): a
    deterministic-serialization integrity check, not a cryptographic
    security proof."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_canonical_findings_hash_changes_when_an_exact_overlap_is_injected() -> None:
    """Same findings -> same canonical hash; a single injected exact
    train/test overlap -> a different canonical hash. This is a
    serialization/regression integrity check on the machine-readable
    findings, not a cryptographic security proof."""
    clean = fx.clean_partitions()
    overlap = fx.train_test_overlap_fixture()

    results_clean = d1.audit_duplicates(
        train=clean["train"], validation=clean["validation"], test=clean["test"],
    )
    results_overlap = d1.audit_duplicates(
        train=overlap["train"], validation=overlap["validation"], test=overlap["test"],
    )

    findings_clean_a = d1.build_d1_certificate(results_clean, dataset_id="fixture-dataset")["findings"]
    findings_clean_b = d1.build_d1_certificate(results_clean, dataset_id="fixture-dataset")["findings"]
    findings_overlap = d1.build_d1_certificate(results_overlap, dataset_id="fixture-dataset")["findings"]

    assert _canonical_hash(findings_clean_a) == _canonical_hash(findings_clean_b)
    assert _canonical_hash(findings_clean_a) != _canonical_hash(findings_overlap)


def test_certificate_round_trip_via_tmp_path(tmp_path) -> None:
    """A certificate written to disk and read back must equal the returned
    in-memory certificate (up to JSON's own type normalization), and must
    preserve audit_id, the decision outcome, and the key findings."""
    partitions = fx.clean_partitions()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    output_path = tmp_path / "d1_certificate.json"

    certificate = d1.build_d1_certificate(
        results, dataset_id="fixture-dataset", output_path=str(output_path),
    )

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    normalized_in_memory = json.loads(json.dumps(certificate, default=str))

    assert loaded == normalized_in_memory
    assert loaded["audit"]["id"] == "D1"
    assert loaded["decision"]["outcome"] == "PASS"
    assert loaded["findings"]["duplicate_samples"] == 0
    assert loaded["findings"]["exact_partition_overlap"] == 0


# ============================================================
# Category 18 -- non-scope / scientific-scope regression.
# ============================================================

def test_certificate_preserves_exact_equality_scope_limitations() -> None:
    """The certificate must keep stating, in its limitations, that D1
    addresses exact equality only and does not establish absence of near
    duplicates or statistical dependence. This guards against a future
    refactor silently broadening D1's scientific claim."""
    partitions = fx.clean_partitions()
    results = d1.audit_duplicates(
        train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
    )
    certificate = d1.build_d1_certificate(results, dataset_id="fixture-dataset")

    limitations_text = " ".join(certificate["limitations"]).lower()
    assert "exact equality" in limitations_text
    assert "near duplicate" in limitations_text
    assert "statistical dependence" in limitations_text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
