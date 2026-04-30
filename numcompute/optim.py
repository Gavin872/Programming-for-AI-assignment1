import numpy as np


def grad(f, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")

    g = np.zeros_like(x, dtype=float)

    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        if method == "central":
            g[i] = (f(x + e) - f(x - e)) / (2 * h)
        else:
            g[i] = (f(x + e) - f(x)) / h

    return g


def jacobian(F, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")

    y0 = np.asarray(F(x), dtype=float)
    if y0.ndim != 1:
        raise ValueError("F(x) must return a 1D array.")

    J = np.zeros((y0.size, x.size), dtype=float)

    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        if method == "central":
            J[:, i] = (np.asarray(F(x + e)) - np.asarray(F(x - e))) / (2 * h)
        else:
            J[:, i] = (np.asarray(F(x + e)) - y0) / h

    return J