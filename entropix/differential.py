"""Differential entropy — continuous entropy via a fitted distribution."""

import numpy as np
from scipy import stats

from . import _core


def _kernel(values, dist="normal"):
    """Differential entropy (nats). dist in {'normal', 'kde'}."""
    values = np.asarray(values, dtype=float)
    if dist == "normal":
        var = np.var(values)
        if var == 0:
            return float("-inf")  # degenerate: zero-width distribution
        return float(0.5 * np.log(2 * np.pi * np.e * var))
    if dist == "kde":
        if np.std(values) == 0:
            return float("-inf")
        kde = stats.gaussian_kde(values)
        density = kde(values)
        density = density[density > 0]
        return float(-np.mean(np.log(density)))
    raise ValueError(f"unknown dist {dist!r}; expected 'normal' or 'kde'")


def compute(series, dist="normal"):
    arr, _ = _core.as_array(series)
    return _kernel(arr, dist=dist)


def rolling(series, window=50, dist="kde"):
    return _core.rolling(series, window, _kernel, dist=dist)


def delta(series, window=50, dist="kde"):
    return _core.delta(series, window, _kernel, dist=dist)


def plot(series, window=50, dist="kde", title=None):
    return _core.make_plot(
        series, window, _kernel, dist=dist, title=title, ylabel="differential entropy"
    )
