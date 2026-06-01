"""Sliding-window utilities shared by all entropy measures."""

import numpy as np


def sliding_windows(values, window):
    """Yield successive length-`window` views over `values`.

    Produces one window ending at each index from `window-1` to the end,
    i.e. ``len(values) - window + 1`` windows.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > n:
        raise ValueError(f"window ({window}) is larger than series length ({n})")
    for end in range(window, n + 1):
        yield values[end - window : end]
