import numpy as np


def _validate_classification_inputs(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return y_true, y_pred


def accuracy(y_true, y_pred):
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, positive=1):
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fp = np.sum((y_true != positive) & (y_pred == positive))
    denom = tp + fp
    return 0.0 if denom == 0 else float(tp / denom)


def recall(y_true, y_pred, positive=1):
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fn = np.sum((y_true == positive) & (y_pred != positive))
    denom = tp + fn
    return 0.0 if denom == 0 else float(tp / denom)


def f1(y_true, y_pred, positive=1):
    p = precision(y_true, y_pred, positive=positive)
    r = recall(y_true, y_pred, positive=positive)
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true, y_score, positive=1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must be 1D and have the same shape.")

    desc = np.argsort(y_score)[::-1]
    y_true = y_true[desc]
    y_score = y_score[desc]

    thresholds = np.r_[np.inf, np.unique(y_score)[::-1]]
    P = np.sum(y_true == positive)
    N = np.sum(y_true != positive)

    tpr = [0.0]
    fpr = [0.0]

    for thr in thresholds[1:]:
        pred = y_score >= thr
        tp = np.sum((y_true == positive) & pred)
        fp = np.sum((y_true != positive) & pred)
        tpr.append(0.0 if P == 0 else tp / P)
        fpr.append(0.0 if N == 0 else fp / N)

    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float), thresholds


def auc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D and have the same shape.")
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))