"""Sample entropy (SampEn) — regularity/predictability of a time series."""
import numpy as np

from . import _core


def _count_matches(values, m, tol):
    """Count template-vector pairs (length m) within Chebyshev distance `tol`."""
    n = len(values)
    templates = np.array([values[i:i + m] for i in range(n - m + 1)])
    count = 0
    for i in range(len(templates) - 1):
        dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
        count += np.count_nonzero(dist <= tol)
    return count


def _kernel(values, m=2, r=0.2):
    """Sample entropy: -ln(A/B) of length-(m+1) vs length-m matches."""
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
        return 0.0  # constant signal: perfectly regular
    b = _count_matches(values, m, tol)
    a = _count_matches(values, m + 1, tol)
    if b == 0 or a == 0:
        # no regularity detected; return a large-but-finite ceiling
        return float(np.log((n - m) * (n - m - 1)))
    return float(-np.log(a / b))


def compute(series, m=2, r=0.2):
    arr, _ = _core.as_array(series)
    return _kernel(arr, m=m, r=r)


def rolling(series, window=50, m=2, r=0.2):
    return _core.rolling(series, window, _kernel, m=m, r=r)


def delta(series, window=50, m=2, r=0.2):
    return _core.delta(series, window, _kernel, m=m, r=r)


def plot(series, window=50, m=2, r=0.2, title=None):
    return _core.make_plot(series, window, _kernel, m=m, r=r,
                           title=title, ylabel="sample entropy")
