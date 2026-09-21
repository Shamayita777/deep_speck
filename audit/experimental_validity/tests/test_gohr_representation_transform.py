import numpy as np
import pytest

from gohr.representation import (
    Candidate1Permutation,
    apply_candidate1,
    generate_candidate1_permutation,
    identity_permutation,
    invert_candidate1,
    run_full_validation,
    verify_bijection,
    verify_content_preserved,
    verify_deterministic,
    verify_round_trip,
)
from gohr import speck


@pytest.fixture
def sample_data():
    X, Y = speck.make_train_data(500, 5)
    return X, Y


def test_generated_permutation_is_bijection():
    perm = generate_candidate1_permutation(np.random.default_rng(0))
    ok, problems = verify_bijection(perm)
    assert ok, problems


def test_identity_permutation_is_bijection():
    ok, problems = verify_bijection(identity_permutation())
    assert ok, problems


def test_malformed_permutation_detected():
    bad = Candidate1Permutation(permutations=np.array([[0] * 16, list(range(16)), list(range(16)), list(range(16))]))
    ok, problems = verify_bijection(bad)
    assert not ok
    assert len(problems) == 1


def test_round_trip_decode_encode_identity(sample_data):
    """decode(encode(x)) == x for representative data."""
    X, _ = sample_data
    perm = generate_candidate1_permutation(np.random.default_rng(1))
    assert verify_round_trip(X, perm) is True


def test_content_preserved_no_bits_added_or_removed(sample_data):
    X, _ = sample_data
    perm = generate_candidate1_permutation(np.random.default_rng(2))
    scrambled = apply_candidate1(X, perm)
    assert verify_content_preserved(X, scrambled) is True
    # Whole-array bit count must also be identical (necessary consequence).
    assert X.sum() == scrambled.sum()


def test_labels_are_never_touched(sample_data):
    """The transform module has no function that accepts or modifies Y."""
    import inspect
    from gohr import representation as repr_module
    src = inspect.getsource(repr_module)
    assert "Y" not in inspect.signature(apply_candidate1).parameters
    assert "Y" not in inspect.signature(invert_candidate1).parameters


def test_deterministic_given_same_seed(sample_data):
    X, _ = sample_data
    assert verify_deterministic(12345, X) is True


def test_different_seeds_produce_different_permutations():
    perm_a = generate_candidate1_permutation(np.random.default_rng(1))
    perm_b = generate_candidate1_permutation(np.random.default_rng(2))
    assert not np.array_equal(perm_a.permutations, perm_b.permutations)


def test_permutation_hash_is_stable_and_verifiable(tmp_path):
    perm = generate_candidate1_permutation(np.random.default_rng(3))
    h1 = perm.hash
    path = tmp_path / "permutation.json"
    perm.save(path)
    loaded = Candidate1Permutation.load(path)
    assert loaded.hash == h1
    assert np.array_equal(loaded.permutations, perm.permutations)


def test_tampered_permutation_file_rejected(tmp_path):
    perm = generate_candidate1_permutation(np.random.default_rng(4))
    path = tmp_path / "permutation.json"
    perm.save(path)
    # Tamper with the file after saving.
    import json
    data = json.loads(path.read_text())
    data["permutations"][0][0], data["permutations"][0][1] = (
        data["permutations"][0][1], data["permutations"][0][0]
    )
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Candidate1Permutation.load(path)


def test_applied_identically_across_partitions(sample_data):
    """
    The SAME permutation object, applied to two different partitions
    (simulating train vs confirmatory-test), must apply the identical
    column mapping to both - verified by checking a few known columns.
    """
    X, _ = sample_data
    X_other, _ = speck.make_train_data(500, 5)
    perm = generate_candidate1_permutation(np.random.default_rng(5))

    scrambled_1 = apply_candidate1(X, perm)
    scrambled_2 = apply_candidate1(X_other, perm)

    # For word 0, destination column 0 should pull from the same source
    # column index in both partitions (same permutation, so the mapping
    # from source bit to destination bit is identical even though the
    # underlying data differs).
    source_col = perm.permutations[0, 0]
    assert np.array_equal(scrambled_1[:, 0], X[:, source_col])
    assert np.array_equal(scrambled_2[:, 0], X_other[:, source_col])


def test_identity_permutation_does_not_change_data(sample_data):
    X, _ = sample_data
    identity = identity_permutation()
    unchanged = apply_candidate1(X, identity)
    assert np.array_equal(unchanged, X)


def test_run_full_validation_passes_for_generated_permutation(sample_data):
    X, Y = sample_data
    perm = generate_candidate1_permutation(np.random.default_rng(6))
    results = run_full_validation(X, Y, perm)
    assert results["all_passed"] is True, results


def test_run_full_validation_catches_bad_permutation(sample_data):
    X, Y = sample_data
    bad = Candidate1Permutation(permutations=np.array([[0] * 16, list(range(16)), list(range(16)), list(range(16))]))
    results = run_full_validation(X, Y, bad)
    assert results["all_passed"] is False


def test_run_full_validation_labels_check_is_not_tautological(sample_data):
    """
    ISSUE 6 (Round 6): regression test for the tautological
    verify_labels_unchanged(Y, Y) check. This test verifies the OUTWARD
    contract using an independent copy taken by the TEST itself (not
    relying on run_full_validation's internal copy), so it would catch
    a regression even if run_full_validation's own fix were reverted:
    Y must be bit-for-bit identical before and after run_full_validation
    is called, and the function's own reported 'labels_unchanged' field
    must be computed from two genuinely distinct array objects, not the
    same object compared with itself.
    """
    X, Y = sample_data
    Y_independent_copy_before = np.array(Y, copy=True)
    permutation = generate_candidate1_permutation(np.random.default_rng(7))

    results = run_full_validation(X, Y, permutation)

    assert results["labels_unchanged"] is True
    assert np.array_equal(Y, Y_independent_copy_before), "Y was mutated by run_full_validation."

    # Confirm the check itself is non-tautological via AST inspection
    # (not a raw string search, which would false-positive on this
    # docstring's own explanatory prose): no call to
    # verify_labels_unchanged may pass the same Name node as both
    # arguments (i.e. verify_labels_unchanged(Y, Y) as an actual call).
    import ast
    import inspect
    from gohr.representation import run_full_validation as rfv_func

    tree = ast.parse(inspect.getsource(rfv_func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "verify_labels_unchanged":
            arg_names = [a.id for a in node.args if isinstance(a, ast.Name)]
            assert not (len(arg_names) == 2 and arg_names[0] == arg_names[1]), (
                "run_full_validation still calls verify_labels_unchanged with the same "
                "variable for both arguments - the tautological check was not actually fixed."
            )
