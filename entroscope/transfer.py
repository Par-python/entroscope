"""Transfer entropy — directional information flow X->Y (Schreiber 2000).

The library's first bivariate measure: two series in, one directional scalar out.
Two estimators: ``method="ksg"`` (Kraskov k-NN, no binning, default) and
``method="binned"`` (histogram). Math lives in ``_transfer_estimators``.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _core
from . import _transfer_estimators as est

_EMBED_DIM = 3  # history-length-1 joint space (y_future, y_past, x_past)


def _coerce_pair(x, y):
    xa, _ = _core.as_array(x)
    ya, yindex = _core.as_array(y)
    if len(xa) != len(ya):
        raise ValueError(f"x and y must be equal length ({len(xa)} != {len(ya)})")
    return xa, ya, yindex


def _guard(n_samples, k, method):
    if n_samples < 10 * _EMBED_DIM:
        warnings.warn(
            f"sample size ({n_samples}) is small for embedding dimension "
            f"{_EMBED_DIM}; transfer-entropy estimate may be unreliable",
            UserWarning,
            stacklevel=3,
        )


def _estimate(xa, ya, k, lag, method, bins):
    yf, yp, xp = est.embed(xa, ya, lag=lag)
    _guard(len(yf), k, method)
    if method == "ksg":
        return est.te_ksg(yf, yp, xp, k=k)
    if method == "binned":
        return est.te_binned(yf, yp, xp, bins=bins)
    raise ValueError(f"unknown method {method!r}; use 'ksg' or 'binned'")


def compute(x, y, *, k=4, lag=1, method="ksg", bins=6):
    """Transfer entropy TE(X->Y) as a float (bits)."""
    xa, ya, _ = _coerce_pair(x, y)
    return _estimate(xa, ya, k, lag, method, bins)


def rolling(x, y, window=120, *, k=4, lag=1, method="ksg", bins=6):
    """Rolling TE(X->Y) over sliding windows. Series-in -> Series-out.

    Note: with method="ksg" this runs a k-NN search per window, so it is
    noticeably slower than the single-series rolling measures on long inputs.
    """
    xa, ya, yindex = _coerce_pair(x, y)
    n = len(xa)
    if window <= _EMBED_DIM + lag:
        raise ValueError(f"window ({window}) must exceed embedding+lag ({_EMBED_DIM + lag})")
    if window > n:
        raise ValueError(f"window ({window}) larger than series length ({n})")
    out = np.full(n, np.nan)
    for end in range(window, n + 1):
        wx = xa[end - window : end]
        wy = ya[end - window : end]
        out[end - 1] = _estimate(wx, wy, k, lag, method, bins)
    return _core.wrap(out, yindex)


def delta(x, y, window=120, *, k=4, lag=1, method="ksg", bins=6):
    """First difference of the rolling transfer entropy."""
    roll = rolling(x, y, window, k=k, lag=lag, method=method, bins=bins)
    if isinstance(roll, pd.Series):
        return roll.diff()
    out = np.full_like(roll, np.nan)
    out[1:] = np.diff(roll)
    return out


def plot(x, y, window=120, *, k=4, lag=1, method="ksg", bins=6, title=None):
    """Figure of rolling TE(X->Y). Never calls plt.show()."""
    roll = rolling(x, y, window, k=k, lag=lag, method=method, bins=bins)
    fig, ax = plt.subplots(figsize=(10, 4))
    if isinstance(roll, pd.Series):
        ax.plot(roll.index, roll.to_numpy())
    else:
        ax.plot(range(len(roll)), roll)
    ax.set_title(title or f"Rolling transfer entropy (window={window})")
    ax.set_xlabel("position")
    ax.set_ylabel("transfer entropy (bits)")
    fig.tight_layout()
    return fig
