"""Core engine: input coercion, output wrapping, rolling/delta drivers, plotting.

Every measure module delegates its standard methods here so the
"Series in -> Series out, array in -> array out" contract and the windowing
logic live in exactly one place.
"""
import matplotlib

matplotlib.use("Agg")  # headless-safe; never opens a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .utils.windows import sliding_windows  # noqa: E402


def as_array(x):
    """Coerce input to a 1-D float ndarray, returning (array, index_or_None)."""
    if isinstance(x, pd.Series):
        index = x.index
        arr = x.to_numpy(dtype=float)
    else:
        index = None
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("input must be 1-dimensional")
    if arr.size == 0:
        raise ValueError("input is empty")
    return arr, index


def wrap(values, index):
    """Wrap a result array as a Series (if index given) or return the ndarray."""
    values = np.asarray(values, dtype=float)
    if index is not None:
        return pd.Series(values, index=index)
    return values


def rolling(x, window, kernel, **params):
    """Apply `kernel` over each full sliding window.

    Output has the same length as the input; positions before the first full
    window are NaN. Returns a Series (preserving index) if `x` was a Series.
    """
    arr, index = as_array(x)
    n = len(arr)
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > n:
        raise ValueError(f"window ({window}) is larger than series length ({n})")
    out = np.full(n, np.nan)
    for end, w in zip(range(window, n + 1), sliding_windows(arr, window)):
        out[end - 1] = kernel(w, **params)
    return wrap(out, index)


def delta(x, window, kernel, **params):
    """First difference of the rolling entropy."""
    roll = rolling(x, window, kernel, **params)
    if isinstance(roll, pd.Series):
        return roll.diff()
    out = np.full_like(roll, np.nan)
    out[1:] = np.diff(roll)
    return out


def make_plot(x, window, kernel, *, title=None, ylabel="entropy", **params):
    """Build and return a Figure of rolling entropy. Never calls plt.show()."""
    roll = rolling(x, window, kernel, **params)
    fig, ax = plt.subplots(figsize=(10, 4))
    if isinstance(roll, pd.Series):
        ax.plot(roll.index, roll.to_numpy())
    else:
        ax.plot(range(len(roll)), roll)
    ax.set_title(title or f"Rolling {ylabel} (window={window})")
    ax.set_xlabel("position")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
