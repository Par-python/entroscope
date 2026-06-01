"""Approximate entropy (ApEn) — regularity measure, less noise-sensitive."""

import numpy as np

from . import _core


def _phi(values, m, tol):
    """Mean of log(fraction of templates within `tol`) over length-m templates."""
    n = len(values)
    templates = np.array([values[i : i + m] for i in range(n - m + 1)])
    counts = np.empty(len(templates))
    for i in range(len(templates)):
        dist = np.max(np.abs(templates - templates[i]), axis=1)
        counts[i] = np.count_nonzero(dist <= tol) / len(templates)
    return float(np.mean(np.log(counts)))


def _kernel(values, m=2, r=0.2):
    """Approximate entropy: phi(m) - phi(m+1)."""
    if r <= 0:
        raise ValueError("r must be positive")
    if m < 1:
        raise ValueError("m must be >= 1")
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= m + 1:
        raise ValueError("series too short for given m")
    tol = r * np.std(values)
    if tol == 0:
        return 0.0
    return float(_phi(values, m, tol) - _phi(values, m + 1, tol))


def compute(series, m=2, r=0.2):
    arr, _ = _core.as_array(series)
    return _kernel(arr, m=m, r=r)


def rolling(series, window=50, m=2, r=0.2):
    return _core.rolling(series, window, _kernel, m=m, r=r)


def delta(series, window=50, m=2, r=0.2):
    return _core.delta(series, window, _kernel, m=m, r=r)


def plot(series, window=50, m=2, r=0.2, title=None):
    return _core.make_plot(
        series, window, _kernel, m=m, r=r, title=title, ylabel="approximate entropy"
    )
