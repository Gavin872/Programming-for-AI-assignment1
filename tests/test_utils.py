import numpy as np
from numcompute.utils import euclidean_distance, sigmoid, relu, softmax, logsumexp, topk_indices


def test_euclidean_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert np.isclose(euclidean_distance(a, b), 5.0)


def test_sigmoid():
    x = np.array([0.0])
    assert np.isclose(sigmoid(x)[0], 0.5)


def test_relu():
    x = np.array([-1.0, 2.0])
    assert np.array_equal(relu(x), np.array([0.0, 2.0]))


def test_softmax_sum():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(softmax(x).sum(), 1.0)


def test_logsumexp():
    x = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.sum(np.exp(x)))
    assert np.isclose(logsumexp(x), expected)


def test_topk_indices():
    x = np.array([10, 5, 8, 1])
    idx = topk_indices(x, 2)
    assert np.array_equal(idx, np.array([0, 2]))