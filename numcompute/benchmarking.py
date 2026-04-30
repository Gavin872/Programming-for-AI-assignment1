import time
import numpy as np


def benchmark(fn, *args, repeat=5, **kwargs):
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return {
        "result": result,
        "times": times,
        "mean": float(np.mean(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def python_sum(x):
    total = 0.0
    for v in x:
        total += v
    return total


def numpy_sum(x):
    return np.sum(x)


def compare_sum(n=100000, repeat=5, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    return {
        "python_sum": benchmark(python_sum, x, repeat=repeat),
        "numpy_sum": benchmark(numpy_sum, x, repeat=repeat),
    }