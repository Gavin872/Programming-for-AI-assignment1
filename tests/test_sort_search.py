import numpy as np
from numcompute.sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search


def test_stable_sort_basic():
    x = np.array([3, 1, 2])
    y = stable_sort(x)
    assert np.array_equal(y, np.array([1, 2, 3]))


def test_multi_key_sort_basic():
    X = np.array([[2, 1], [1, 2], [1, 1]])
    sorted_X, idx = multi_key_sort(X, keys=[0, 1], ascending=[True, True])
    assert np.array_equal(sorted_X, np.array([[1, 1], [1, 2], [2, 1]]))
    assert idx.shape == (3,)


def test_topk_largest():
    x = np.array([10, 5, 8, 1])
    vals, idx = topk(x, 2, largest=True, return_indices=True)
    assert np.array_equal(vals, np.array([10, 8]))
    assert len(idx) == 2


def test_topk_smallest():
    x = np.array([10, 5, 8, 1])
    vals = topk(x, 2, largest=False, return_indices=False)
    assert np.array_equal(vals, np.array([1, 5]))


def test_quickselect_smallest():
    x = np.array([7, 2, 9, 1, 5])
    assert quickselect(x, 2) == 5


def test_binary_search_found():
    x = np.array([1, 3, 5, 7])
    idx, exists = binary_search(x, 5)
    assert idx == 2
    assert exists is True


def test_binary_search_not_found():
    x = np.array([1, 3, 5, 7])
    idx, exists = binary_search(x, 4)
    assert idx == 2
    assert exists is False