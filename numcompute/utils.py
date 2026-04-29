import numpy as np


def euclidean_distance(a, b, axis=-1):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2, axis=axis))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def relu(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x, dtype=float)
    xmax = np.max(x, axis=axis, keepdims=True)
    out = xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True))
    if axis is None:
        out = np.asarray(out).squeeze()
        return out if keepdims else float(out)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def topk_indices(values, k, largest=True):
    values = np.asarray(values)
    if largest:
        idx = np.argpartition(values, -k)[-k:]
        return idx[np.argsort(values[idx])[::-1]]
    idx = np.argpartition(values, k - 1)[:k]
    return idx[np.argsort(values[idx])]


def batched(iterable, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]