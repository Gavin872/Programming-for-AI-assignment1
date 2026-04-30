import numpy as np
from numcompute.stats import mean, median, std, min_value, max_value, histogram, quantiles, Welford


def test_mean_axis():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(mean(X, axis=0), np.array([2.0, 3.0]))


def test_median_basic():
    x = np.array([1.0, 3.0, 2.0])
    assert median(x) == 2.0


def test_std_basic():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(std(x), np.std(x))


def test_min_max_basic():
    x = np.array([1.0, 2.0, 3.0])
    assert min_value(x) == 1.0
    assert max_value(x) == 3.0


def test_histogram_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    counts, edges = histogram(x, bins=2)
    assert counts.sum() == 4
    assert len(edges) == 3


def test_quantiles_skipna():
    x = np.array([1.0, np.nan, 3.0])
    q = quantiles(x, 0.5, skipna=True)
    assert q == 2.0


def test_welford_basic():
    w = Welford().update([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(w.mean, 2.5)
    assert np.isclose(w.variance, np.var([1.0, 2.0, 3.0, 4.0]))