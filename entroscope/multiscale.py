"""Multiscale entropy — sample entropy across coarse-grained time scales."""

import matplotlib.pyplot as plt
import numpy as np

from . import _core, sample


def _coarse_grain(values, scale):
    """Average non-overlapping blocks of length `scale`."""
    n = len(values) // scale
    trimmed = values[: n * scale]
    return trimmed.reshape(n, scale).mean(axis=1)


def compute(series, scales=range(1, 10), method="sample", m=2, r=0.2):
    """Return {scale: entropy} by coarse-graining then applying `method`.

    Following Costa et al. (2002), the tolerance is fixed at ``r * std`` of the
    ORIGINAL series and reused at every scale, so white noise loses entropy as
    the scale grows while 1/f-like signals keep it.

    Scales that coarse-grain the series below sample entropy's minimum
    length are skipped (omitted from the result).
    """
    if method != "sample":
        raise ValueError("only method='sample' is supported")
    if r <= 0:
        raise ValueError("r must be positive")
    if m < 1:
        raise ValueError("m must be >= 1")
    arr, _ = _core.as_array(series)
    finite = bool(np.isfinite(arr).all())
    tol = r * np.std(arr)
    result = {}
    for scale in scales:
        if scale == 1:
            grained = arr
        else:
            grained = _coarse_grain(arr, scale)
        if len(grained) <= m + 1:  # sample entropy needs n > m+1
            continue
        result[int(scale)] = sample._sampen(grained, m, tol) if finite else float("nan")
    return result


def plot(series, scales=range(1, 10), title=None):
    """Plot the complexity profile (entropy vs. scale)."""
    profile = compute(series, scales=scales)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(profile.keys()), list(profile.values()), marker="o")
    ax.set_title(title or "Multiscale entropy")
    ax.set_xlabel("scale")
    ax.set_ylabel("sample entropy")
    fig.tight_layout()
    return fig
