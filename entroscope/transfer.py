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
