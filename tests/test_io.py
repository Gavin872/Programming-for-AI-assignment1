import numpy as np
from numcompute.io import load_csv


def test_load_csv_keep(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("1,2\n3,\n", encoding="utf-8")
    x = load_csv(str(p))
    assert x.shape == (2, 2)
    assert np.isnan(x[1, 1])


def test_load_csv_fill(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("1,2\n3,\n", encoding="utf-8")
    x = load_csv(str(p), missing_strategy="fill", fill_value=0.0)
    assert x[1, 1] == 0.0


def test_load_csv_skip(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("1,2\n3,\n", encoding="utf-8")
    x = load_csv(str(p), missing_strategy="skip")
    assert x.shape == (1, 2)