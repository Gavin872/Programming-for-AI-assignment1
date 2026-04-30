import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, MinMaxScaler


class DummyModel:
    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        return self

    def predict(self, X):
        return (X.sum(axis=1) > 0).astype(int)


def test_pipeline_fit_transform():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("range", MinMaxScaler()),
    ])
    Xt = pipe.fit_transform(X)
    assert Xt.shape == X.shape
    assert np.allclose(np.min(Xt, axis=0), 0.0)
    assert np.allclose(np.max(Xt, axis=0), 1.0)


def test_pipeline_predict():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DummyModel()),
    ])
    pipe.fit(X)
    y = pipe.predict(X)
    assert y.shape == (2,)