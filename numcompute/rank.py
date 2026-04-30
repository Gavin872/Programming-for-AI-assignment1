import numpy as np


def rank(data, method="average"):
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("data must be 1D.")
    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'.")

    sorter = np.argsort(data, kind="mergesort")
    sorted_data = data[sorter]
    out = np.empty(data.size, dtype=float)

    if method == "ordinal":
        out[sorter] = np.arange(1, data.size + 1, dtype=float)
        return out

    unique_vals, first_idx, counts = np.unique(
        sorted_data, return_index=True, return_counts=True
    )

    if method == "dense":
        for dense_rank, (start, count) in enumerate(zip(first_idx, counts), start=1):
            out[sorter[start:start + count]] = float(dense_rank)
        return out

    for start, count in zip(first_idx, counts):
        positions = np.arange(start + 1, start + count + 1, dtype=float)
        avg = positions.mean()
        out[sorter[start:start + count]] = avg

    return out


def percentile(data, q, interpolation="linear"):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("data must not be empty.")
    return np.percentile(data, q, method=interpolation)