import numpy as np


def mean(x, axis=None, skipna=False):
    x = np.asarray(x, dtype=float)
    return np.nanmean(x, axis=axis) if skipna else np.mean(x, axis=axis)


def median(x, axis=None, skipna=False):
    x = np.asarray(x, dtype=float)
    return np.nanmedian(x, axis=axis) if skipna else np.median(x, axis=axis)


def std(x, axis=None, ddof=0, skipna=False):
    x = np.asarray(x, dtype=float)
    return np.nanstd(x, axis=axis, ddof=ddof) if skipna else np.std(x, axis=axis, ddof=ddof)


def min_value(x, axis=None, skipna=False):
    x = np.asarray(x, dtype=float)
    return np.nanmin(x, axis=axis) if skipna else np.min(x, axis=axis)


def max_value(x, axis=None, skipna=False):
    x = np.asarray(x, dtype=float)
    return np.nanmax(x, axis=axis) if skipna else np.max(x, axis=axis)


def histogram(x, bins=10, range=None):
    x = np.asarray(x, dtype=float)
    return np.histogram(x, bins=bins, range=range)


def quantiles(x, q, axis=None, skipna=False):
    x = np.asarray(x, dtype=float)
    return np.nanquantile(x, q, axis=axis) if skipna else np.quantile(x, q, axis=axis)


class Welford:
    def __init__(self):
        self.n = 0
        self.mean_ = 0.0
        self.m2_ = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=float).ravel()
        for v in values:
            self.n += 1
            delta = v - self.mean_
            self.mean_ += delta / self.n
            delta2 = v - self.mean_
            self.m2_ += delta * delta2
        return self

    @property
    def mean(self):
        if self.n == 0:
            raise ValueError("No data observed.")
        return self.mean_

    @property
    def variance(self):
        if self.n == 0:
            raise ValueError("No data observed.")
        return self.m2_ / self.n

    @property
    def sample_variance(self):
        if self.n < 2:
            raise ValueError("Need at least two observations.")
        return self.m2_ / (self.n - 1)