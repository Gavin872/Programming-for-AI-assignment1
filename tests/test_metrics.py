import numpy as np
from numcompute.metrics import accuracy, precision, recall, f1, confusion_matrix, mse, roc_curve, auc


def test_accuracy():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    assert np.isclose(accuracy(y_true, y_pred), 0.75)


def test_precision():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    assert np.isclose(precision(y_true, y_pred), 2 / 3)


def test_recall():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    assert np.isclose(recall(y_true, y_pred), 2 / 3)


def test_f1():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    assert np.isclose(f1(y_true, y_pred), 2 / 3)


def test_confusion_matrix():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, np.array([[2, 0], [1, 1]]))


def test_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert np.isclose(mse(y_true, y_pred), 1 / 3)


def test_roc_curve_and_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)
    assert fpr.ndim == 1
    assert tpr.ndim == 1
    assert score >= 0.0