import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.mean_ is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Input shape does not match fitted data.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.data_min_ is None:
            raise ValueError("MinMaxScaler must be fitted before transform.")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Input shape does not match fitted data.")
        low, high = self.feature_range
        X_std = (X - self.data_min_) / self.data_range_
        return X_std * (high - low) + low

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self):
        self.categories_ = None
        self._mapping = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D.")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self._mapping = [{v: i for i, v in enumerate(cats)} for cats in self.categories_]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transform.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != len(self.categories_):
            raise ValueError("Input shape does not match fitted data.")
        blocks = []
        for j in range(X.shape[1]):
            cats = self.categories_[j]
            block = (X[:, [j]] == cats.reshape(1, -1)).astype(float)
            if not np.all(block.sum(axis=1) == 1):
                raise ValueError("Unknown category encountered.")
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Imputer:
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        return np.where(np.isnan(X), self.fill_value, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)