import numpy as np


def stable_sort(array, axis=-1):
    array = np.asarray(array)
    return np.sort(array, axis=axis, kind="stable")


def multi_key_sort(data, keys, ascending=True):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D.")
    if isinstance(keys, int):
        keys = [keys]
    if isinstance(ascending, bool):
        ascending = [ascending] * len(keys)
    if len(keys) != len(ascending):
        raise ValueError("keys and ascending must have the same length.")
    sort_keys = []
    for col, asc in zip(reversed(keys), reversed(ascending)):
        values = data[:, col]
        sort_keys.append(values if asc else -values)
    idx = np.lexsort(tuple(sort_keys))
    return data[idx], idx


def topk(values, k, largest=True, return_indices=True):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be 1D.")
    if not 1 <= k <= values.size:
        raise ValueError("k must satisfy 1 <= k <= len(values).")
    if largest:
        idx = np.argpartition(values, -k)[-k:]
        order = np.argsort(values[idx])[::-1]
    else:
        idx = np.argpartition(values, k - 1)[:k]
        order = np.argsort(values[idx])
    idx = idx[order]
    vals = values[idx]
    if return_indices:
        return vals, idx
    return vals


def quickselect(values, k, largest=False):
    arr = np.asarray(values).copy()
    if arr.ndim != 1:
        raise ValueError("values must be 1D.")
    if not 0 <= k < arr.size:
        raise ValueError("k out of range.")
    if largest:
        k = arr.size - 1 - k

    left = 0
    right = arr.size - 1

    while True:
        if left == right:
            return arr[left]

        pivot = arr[right]
        store = left

        for i in range(left, right):
            if arr[i] < pivot:
                arr[store], arr[i] = arr[i], arr[store]
                store += 1

        arr[right], arr[store] = arr[store], arr[right]

        if k == store:
            return arr[k]
        if k < store:
            right = store - 1
        else:
            left = store + 1


def binary_search(sorted_array, x):
    sorted_array = np.asarray(sorted_array)
    if sorted_array.ndim != 1:
        raise ValueError("sorted_array must be 1D.")
    idx = int(np.searchsorted(sorted_array, x, side="left"))
    exists = idx < sorted_array.size and sorted_array[idx] == x
    return idx, bool(exists)