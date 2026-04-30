import numpy as np
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Imputer


def test_standard_scaler_fit_transform():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xt = StandardScaler().fit_transform(X)
    assert Xt.shape == X.shape
    assert np.allclose(np.mean(Xt, axis=0), 0.0)


def test_standard_scaler_zero_variance():
    X = np.array([[1.0, 2.0], [1.0, 4.0]])
    Xt = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Xt))


def test_minmax_scaler_default_range():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xt = MinMaxScaler().fit_transform(X)
    assert np.allclose(np.min(Xt, axis=0), 0.0)
    assert np.allclose(np.max(Xt, axis=0), 1.0)


def test_minmax_scaler_custom_range():
    X = np.array([[1.0], [3.0]])
    Xt = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
    assert np.allclose(Xt.ravel(), [-1.0, 1.0])


def test_onehot_encoder_1d():
    x = np.array(["a", "b", "a"])
    Xt = OneHotEncoder().fit_transform(x)
    assert Xt.shape == (3, 2)
    assert np.array_equal(Xt[0], Xt[2])


def test_onehot_encoder_2d():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]], dtype=object)
    Xt = OneHotEncoder().fit_transform(X)
    assert Xt.shape[0] == 3


def test_imputer_constant():
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    Xt = Imputer(fill_value=9.0).fit_transform(X)
    assert np.array_equal(Xt, np.array([[1.0, 9.0], [9.0, 4.0]]))