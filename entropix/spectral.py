"""Spectral entropy — Shannon entropy of the normalized power spectrum."""
import numpy as np
from scipy import signal as _signal

from . import _core
from .utils import normalize


def _psd(values, sf):
    """Return the (positive-frequency) power spectral density of `values`."""
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    freqs, psd = _signal.periodogram(values, fs=sf)
    return freqs, psd


def _kernel(values, sf=1.0):
    """Spectral entropy (base 2) of the normalized power spectrum."""
    _, psd = _psd(values, sf)
    total = psd.sum()
    if total == 0:
        return 0.0
    p = psd[psd > 0] / total
    return float(-np.sum(p * np.log2(p)))


def compute(series, sf=1.0):
    arr, _ = _core.as_array(series)
    return _kernel(arr, sf=sf)


def rolling(series, window=50, sf=1.0):
    return _core.rolling(series, window, _kernel, sf=sf)


def delta(series, window=50, sf=1.0):
    return _core.delta(series, window, _kernel, sf=sf)


def normalized(series, sf=1.0):
    """Entropy scaled to [0, 1] by log2(number of frequency bins)."""
    arr, _ = _core.as_array(series)
    freqs, _psd_vals = _psd(arr, sf)
    n_bins = int(np.count_nonzero(_psd_vals > 0))
    if n_bins <= 1:
        return 0.0
    return normalize.by_max(_kernel(arr, sf=sf), np.log2(n_bins))


def plot(series, window=50, sf=1.0, title=None):
    return _core.make_plot(series, window, _kernel, sf=sf,
                           title=title, ylabel="spectral entropy")
