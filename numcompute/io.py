import numpy as np


def load_csv(
    filepath,
    delimiter=",",
    skip_header=0,
    missing_strategy="keep",
    fill_value=np.nan,
    dtype=float,
    encoding="utf-8",
):
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=skip_header,
        dtype=dtype,
        filling_values=np.nan,
        encoding=encoding,
    )

    if data.ndim == 0:
        data = np.array([[data]], dtype=dtype)
    elif data.ndim == 1:
        data = data.reshape(-1, 1)

    if missing_strategy not in {"keep", "fill", "skip"}:
        raise ValueError("missing_strategy must be 'keep', 'fill', or 'skip'.")

    if missing_strategy == "fill":
        data = np.where(np.isnan(data), fill_value, data)
    elif missing_strategy == "skip":
        if np.isnan(data).ndim == 2:
            data = data[~np.isnan(data).any(axis=1)]

    return np.asarray(data, dtype=dtype)