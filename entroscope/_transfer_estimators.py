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


def _hist_prob(*cols, bins):
    """Joint probability table over the given equal-width-binned columns."""
    sample = np.column_stack(cols)
    counts, _ = np.histogramdd(sample, bins=bins)
    total = counts.sum()
    return counts / total if total > 0 else counts


def te_binned(y_future, y_past, x_past, bins=6):
    """Transfer entropy X->Y via equal-width histogram probabilities (base 2)."""
    p_fpx = _hist_prob(y_future, y_past, x_past, bins=bins)   # p(yf, yp, xp)
    p_fp = p_fpx.sum(axis=2)                                  # p(yf, yp)
    p_px = p_fpx.sum(axis=0)                                  # p(yp, xp)
    p_p = p_fpx.sum(axis=(0, 2))                              # p(yp)

    te = 0.0
    nf, npq, nx = p_fpx.shape
    for i in range(nf):
        for j in range(npq):
            for k in range(nx):
                pijk = p_fpx[i, j, k]
                if pijk <= 0:
                    continue
                denom = p_fp[i, j] * p_px[j, k]
                numer = pijk * p_p[j]
                if denom <= 0 or numer <= 0:
                    continue
                te += pijk * np.log2(numer / denom)
    return float(te)
