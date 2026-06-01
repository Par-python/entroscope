"""Multiscale entropy — sample entropy across coarse-grained time scales."""
import matplotlib.pyplot as plt
import numpy as np

from . import _core, sample


def _coarse_grain(values, scale):
    """Average non-overlapping blocks of length `scale`."""
    n = len(values) // scale
    trimmed = values[:n * scale]
    return trimmed.reshape(n, scale).mean(axis=1)


def compute(series, scales=range(1, 10), method="sample"):
    """Return {scale: entropy} by coarse-graining then applying `method`.

    Scales that coarse-grain the series below sample entropy's minimum
    length are skipped (omitted from the result).
    """
    if method != "sample":
        raise ValueError("only method='sample' is supported")
    arr, _ = _core.as_array(series)
    result = {}
    for scale in scales:
        if scale == 1:
            grained = arr
        else:
            grained = _coarse_grain(arr, scale)
        if len(grained) < 4:  # sample entropy needs n > m+1 (m=2 default)
            continue
        result[int(scale)] = sample.compute(grained)
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
