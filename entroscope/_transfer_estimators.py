"""Transfer-entropy estimators (history length 1) and shared delay-embedding.

Private module: all the math lives here so the public ``transfer`` module stays
thin. Both estimators consume the SAME embedded vectors from ``embed`` — the
embedding is verified in isolation (see tests) so a bug here cannot silently fool
both estimators.

NOTE: imports for later estimators (digamma from scipy, knn from .utils) are
added in the tasks that implement the binned and KSG estimators respectively.
"""

import numpy as np


def embed(x, y, lag=1):
    """Build aligned (y_future, y_past, x_past) sample columns, history length 1.

    For each valid time t (from ``lag`` to n-1): y_future=y[t], y_past=y[t-lag],
    x_past=x[t-lag]. Returns three 1-D arrays of length ``n - lag``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    n = len(y)
    if n - lag < 1:
        raise ValueError(f"series too short ({n}) for lag {lag}")
    y_future = y[lag:]
    y_past = y[: n - lag]
    x_past = x[: n - lag]
    return y_future, y_past, x_past
