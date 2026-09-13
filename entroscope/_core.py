"""Core engine: input coercion, output wrapping, rolling/delta drivers, plotting.

Every measure module delegates its standard methods here so the
"Series in -> Series out, array in -> array out" contract and the windowing
logic live in exactly one place.

This module does NOT force a matplotlib backend. Plot helpers build a Figure and
return it (never calling ``plt.show()``), so they work under whatever backend the
environment provides. Headless / Docker / CI users who want a guaranteed
non-interactive backend should set ``MPLBACKEND=Agg`` in their environment.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils.windows import sliding_windows


class _PolarsName:
    """Stand-in 'index' for polars input, which has no index — only a name."""

    def __init__(self, name):
        self.name = name


def _is_polars_series(x):
    # Checked by module name so polars stays an optional dependency.
    cls = type(x)
    return cls.__name__ == "Series" and cls.__module__.startswith("polars")


def as_array(x):
    """Coerce input to a 1-D float ndarray, returning (array, index_or_None).

    pandas input returns its index; polars input returns a `_PolarsName` so
    `wrap` can rebuild a polars Series; anything else returns None.
    """
    if isinstance(x, pd.Series):
        index = x.index
        arr = x.to_numpy(dtype=float)
    elif _is_polars_series(x):
        index = _PolarsName(x.name)
        arr = np.asarray(x.to_numpy(), dtype=float)
    else:
        index = None
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("input must be 1-dimensional")
    if arr.size == 0:
        raise ValueError("input is empty")
    return arr, index


def wrap(values, index):
    """Wrap a result array to match the input type: pandas, polars, or ndarray."""
    values = np.asarray(values, dtype=float)
    if isinstance(index, _PolarsName):
        import polars as pl

        return pl.Series(index.name, values)
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
    arr, index = as_array(x)
    return wrap(first_difference(rolling(arr, window, kernel, **params)), index)


def first_difference(values):
    """`values[t] - values[t-1]`, with NaN at position 0 (like `Series.diff`)."""
    out = np.full(len(values), np.nan)
    out[1:] = np.diff(values)
    return out


def make_plot(x, window, kernel, *, title=None, ylabel="entropy", **params):
    """Build and return a Figure of rolling entropy. Never calls plt.show()."""
    roll = rolling(x, window, kernel, **params)
    fig, ax = plt.subplots(figsize=(10, 4))
    if isinstance(roll, pd.Series):
        ax.plot(roll.index, roll.to_numpy())
    else:
        ax.plot(range(len(roll)), np.asarray(roll))
    ax.set_title(title or f"Rolling {ylabel} (window={window})")
    ax.set_xlabel("position")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
