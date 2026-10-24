"""Tests for the shared, target-agnostic integrity utilities (audit.dataset.integrity)."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from audit.dataset import integrity as I


# ---------------------------------------------------------------- digests


def test_partition_digest_binds_role_dtype_shape_and_bytes():
    a = np.zeros((4, 8), dtype=np.uint8)
    base = I.partition_digest({"train_x": a})
    assert I.partition_digest({"train_x": a.copy()}) == base
    assert I.partition_digest({"train_x": a}, chunk_rows=1) == base
    flipped = a.copy()
    flipped[3, 7] = 1
    assert I.partition_digest({"train_x": flipped}) != base
    assert I.partition_digest({"test_x": a}) != base                      # role
    assert I.partition_digest({"train_x": a.astype(np.int8)}) != base     # dtype
    assert I.partition_digest({"train_x": a.reshape(8, 4)}) != base       # shape
    wide = np.zeros(4, dtype="<u4")
    assert I.partition_digest({"v": wide}) != I.partition_digest({"v": wide.astype(">u4")})  # byte order


def test_labelled_digest_changes_with_labels_only():
    x = np.zeros((3, 2), dtype=np.uint8)
    assert I.labelled_partition_digest(x, np.array([0, 1, 0], np.uint8)) != \
        I.labelled_partition_digest(x, np.array([1, 0, 0], np.uint8))


# ---------------------------------------------------------------- writes


def test_exclusive_write_refuses_existing_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "c.json"
    I.write_json_exclusive(path, {"v": 1})
    with pytest.raises(I.IntegrityError, match="Refusing to overwrite"):
        I.write_json_exclusive(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_directory_write_is_all_or_nothing(tmp_path):
    target = tmp_path / "run"
    I.write_directory_exclusive(target, {"a.txt": b"a", "sub/b.txt": b"b"})
    assert (target / "sub" / "b.txt").read_bytes() == b"b"
    with pytest.raises(I.IntegrityError, match="existing directory"):
        I.write_directory_exclusive(target, {"a.txt": b"changed"})
    assert (target / "a.txt").read_bytes() == b"a"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_read_json_reports_truncation(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"a": 1', encoding="utf-8")
    with pytest.raises(I.IntegrityError, match="truncated or malformed"):
        I.read_json(path, "state")
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(I.IntegrityError, match="not a JSON object"):
        I.read_json(path, "state")


# ---------------------------------------------------------------- overlap audit


def bits(values, width=64):
    return np.array([[int(c) for c in f"{v:0{width}b}"] for v in values], dtype=np.uint8)


def labels(*v):
    return np.array(v, dtype=np.uint8)


def test_overlap_counts_and_label_conflicts():
    parts = {
        "train": (bits([1, 2, 2, 3, 2**63 + 5]), labels(0, 1, 0, 1, 1)),
        "validation": (bits([3, 4, 9]), labels(1, 0, 0)),
        "test": (bits([2, 2**63 + 5, 4, 7]), labels(1, 0, 0, 1)),
    }
    report = I.audit_partition_overlap(parts)
    assert report["rows"] == {"train": 5, "validation": 3, "test": 4}
    assert report["within"]["train"] == {"duplicate_rows": 1, "label_conflicts": 1}
    assert report["within"]["test"] == {"duplicate_rows": 0, "label_conflicts": 0}
    assert report["cross"]["train__validation"] == {"rows": 1, "distinct": 1, "label_conflicts": 0}
    # row 2 is label-inconsistent in train; row 2^63+5 has label 1 in train, 0 in test
    assert report["cross"]["train__test"] == {"rows": 2, "distinct": 2, "label_conflicts": 2}
    assert report["cross"]["validation__test"] == {"rows": 1, "distinct": 1, "label_conflicts": 0}


def test_overlap_keys_are_exact_for_high_bit_and_order_independent():
    # Rows differing only in the most or least significant bit stay distinct.
    report = I.audit_partition_overlap({"a": (bits([0, 2**63, 1]), labels(0, 0, 0)),
                                        "b": (bits([2**63 + 1]), labels(0))})
    assert report["cross"]["a__b"]["rows"] == 0
    keys = I.row_keys(bits([5, 2**63]))
    assert keys[0] < keys[1]  # big-endian packing preserves row order


def test_overlap_wide_and_non_binary_rows_use_raw_bytes():
    wide = np.array([[1] * 80, [0] * 80, [1] * 80], dtype=np.uint8)
    report = I.audit_partition_overlap({"a": (wide[:2], labels(0, 1)), "b": (wide[2:], labels(0))})
    assert report["cross"]["a__b"]["rows"] == 1
    values = np.array([[300, 7], [300, 7], [1, 2]], dtype="<u2")
    assert I.audit_partition_overlap({"a": (values, labels(0, 0, 0))})["within"]["a"]["duplicate_rows"] == 1
    assert I.row_keys(values).dtype.kind == "u"


def test_overlap_chunked_keys_match_and_empty_partitions_are_handled():
    x = np.random.default_rng(9).integers(0, 2, size=(1000, 64), dtype=np.uint8)
    np.testing.assert_array_equal(I.row_keys(x, chunk_rows=7), I.row_keys(x))
    empty = np.zeros((0, 64), dtype=np.uint8)
    report = I.audit_partition_overlap({"a": (x, np.zeros(1000, np.uint8)), "b": (empty, np.zeros(0, np.uint8))})
    assert report["rows"]["b"] == 0 and report["cross"]["a__b"]["rows"] == 0


def test_overlap_rejects_length_mismatch():
    with pytest.raises(ValueError, match="differ in length"):
        I.audit_partition_overlap({"a": (bits([1, 2]), labels(0))})


# ---------------------------------------------------------------- git anchoring


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for k, v in (("user.email", "t@example.invalid"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    record = tmp_path / "prereg.json"
    record.write_text('{"a": 1}\n')
    subprocess.run(["git", "-C", str(tmp_path), "add", "prereg.json"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "p"], check=True)
    return record


def test_anchored_committed_record_accepts_exact_bytes(repo):
    record = I.verify_anchored_committed_record(repo, expected_sha256=I.sha256_file(repo), what="p")
    assert record["a"] == 1 and len(record["_commit"]["sha"]) == 40


def test_anchored_record_refuses_wrong_anchor_and_uncommitted_edit(repo):
    with pytest.raises(I.IntegrityError, match="anchor"):
        I.verify_anchored_committed_record(repo, expected_sha256="0" * 64, what="p")
    repo.write_text('{"a": 2}\n')
    with pytest.raises(I.IntegrityError, match="uncommitted"):
        I.verify_anchored_committed_record(repo, expected_sha256=I.sha256_file(repo), what="p")


def test_anchored_record_refuses_untracked(repo):
    other = repo.parent / "other.json"
    other.write_text("{}")
    with pytest.raises(I.IntegrityError, match="not tracked"):
        I.verify_anchored_committed_record(other, expected_sha256=I.sha256_file(other), what="p")
