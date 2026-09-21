import numpy as np

from gohr import speck


def test_speck_testvector():
    """Gohr's published Speck32/64 test vector must verify."""
    assert speck.check_testvector() is True


def test_convert_to_binary_shape_and_values():
    arr = [
        np.array([0b1010101010101010], dtype=np.uint16),
        np.array([0], dtype=np.uint16),
        np.array([0xFFFF], dtype=np.uint16),
        np.array([1], dtype=np.uint16),
    ]
    X = speck.convert_to_binary(arr)
    assert X.shape == (1, 64)
    assert set(np.unique(X).tolist()) <= {0, 1}
    # word 0 = 0b1010101010101010 -> alternating bits, MSB first
    assert list(X[0, 0:16]) == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    # word 2 = 0xFFFF -> all ones
    assert all(v == 1 for v in X[0, 32:48])
    # word 3 = 1 -> only LSB set
    assert list(X[0, 48:64]) == [0] * 15 + [1]


def test_make_train_data_shapes_and_labels():
    n = 200
    X, Y = speck.make_train_data(n, 5)
    assert X.shape == (n, 64)
    assert Y.shape == (n,)
    assert set(np.unique(Y).tolist()) <= {0, 1}
    assert set(np.unique(X).tolist()) <= {0, 1}


def test_make_train_data_is_not_seedable_by_design():
    """
    Two calls must differ (os.urandom-based) - this documents, at the
    test level, that exact replay is genuinely unavailable rather than
    merely undocumented.
    """
    X1, Y1 = speck.make_train_data(500, 5)
    X2, Y2 = speck.make_train_data(500, 5)
    assert not np.array_equal(X1, X2) or not np.array_equal(Y1, Y2)
