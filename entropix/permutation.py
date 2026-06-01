"""Permutation entropy — complexity from ordinal patterns (Bandt & Pompe)."""
import math
from itertools import permutations

import numpy as np

from . import _core
from .utils import normalize


def _kernel(values, order=3, delay=1):
    """Permutation entropy (base 2) of ordinal patterns of length `order`."""
    if order < 2:
        raise ValueError("order must be >= 2")
    if delay < 1:
        raise ValueError("delay must be >= 1")
    values = np.asarray(values, dtype=float)
    n = len(values)
    span = delay * (order - 1)
    if n - span <= 0:
        raise ValueError("series too short for given order/delay")
    perm_index = {p: i for i, p in enumerate(permutations(range(order)))}
    counts = np.zeros(len(perm_index))
    for i in range(n - span):
        window = values[i:i + span + 1:delay]
        pattern = tuple(np.argsort(window, kind="stable"))
        counts[perm_index[pattern]] += 1
    total = counts.sum()
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _check_window(window, order, delay):
    """Raise ValueError if window is too small for the given order/delay."""
    span = delay * (order - 1)
    if window <= span:
        raise ValueError(
            f"window ({window}) must exceed delay*(order-1) = {span}"
        )


def compute(series, order=3, delay=1):
    arr, _ = _core.as_array(series)
    return _kernel(arr, order=order, delay=delay)


def rolling(series, window=20, order=3, delay=1):
    _check_window(window, order, delay)
    return _core.rolling(series, window, _kernel, order=order, delay=delay)


def delta(series, window=20, order=3, delay=1):
    _check_window(window, order, delay)
    return _core.delta(series, window, _kernel, order=order, delay=delay)


def normalized(series, order=3, delay=1):
    """Entropy scaled to [0, 1] by log2(order!)."""
    return normalize.by_max(compute(series, order=order, delay=delay),
                            math.log2(math.factorial(order)))


def plot(series, window=20, order=3, delay=1, title=None):
    _check_window(window, order, delay)
    return _core.make_plot(series, window, _kernel, order=order, delay=delay,
                           title=title, ylabel="permutation entropy")
