"""Shannon entropy — classic information entropy over a binned distribution."""
import numpy as np

from . import _core
from .utils import normalize


def _kernel(values, bins=10):
    """Shannon entropy (base 2) of `values` histogrammed into `bins`."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer")
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def compute(series, bins=10):
    arr, _ = _core.as_array(series)
    return _kernel(arr, bins=bins)


def rolling(series, window=20, bins=10):
    return _core.rolling(series, window, _kernel, bins=bins)


def delta(series, window=20, bins=10):
    return _core.delta(series, window, _kernel, bins=bins)


def normalized(series, bins=10):
    """Entropy scaled to [0, 1] by the maximum possible (log2(bins))."""
    return normalize.by_max(compute(series, bins=bins), np.log2(bins))


def geographic(region_df, col="interest"):
    """Shannon entropy of a spatial distribution (e.g. Google Trends by region)."""
    values = np.asarray(region_df[col], dtype=float)
    total = values.sum()
    if total == 0:
        return 0.0
    p = values[values > 0] / total
    return float(-np.sum(p * np.log2(p)))


def plot(series, window=20, bins=10, title=None):
    return _core.make_plot(series, window, _kernel, bins=bins,
                           title=title, ylabel="Shannon entropy")
