import numpy as np
from numcompute.rank import rank, percentile


def test_rank_average():
    x = np.array([10, 20, 20, 30], dtype=float)
    r = rank(x, method="average")
    assert np.allclose(r, np.array([1.0, 2.5, 2.5, 4.0]))


def test_rank_dense():
    x = np.array([10, 20, 20, 30], dtype=float)
    r = rank(x, method="dense")
    assert np.allclose(r, np.array([1.0, 2.0, 2.0, 3.0]))


def test_rank_ordinal():
    x = np.array([10, 20, 20, 30], dtype=float)
    r = rank(x, method="ordinal")
    assert np.allclose(r, np.array([1.0, 2.0, 3.0, 4.0]))


def test_percentile_linear():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    assert percentile(x, 50, interpolation="linear") == 3.0