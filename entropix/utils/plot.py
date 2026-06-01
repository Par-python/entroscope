"""Unified visualization helpers across entropy measures."""

import matplotlib.pyplot as plt
import numpy as np

from .. import shannon, permutation, spectral, sample, approximate, differential

# Registry of measures usable by the rolling-based helpers.
_REGISTRY = {
    "shannon": shannon,
    "permutation": permutation,
    "spectral": spectral,
    "sample": sample,
    "approximate": approximate,
    "differential": differential,
}


def _resolve(name):
    if name not in _REGISTRY:
        raise ValueError(f"unknown measure {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def _xy(roll):
    """Return (x, y) for plotting: use the Series index when available."""
    if hasattr(roll, "to_numpy"):  # pandas Series
        return roll.index, roll.to_numpy()
    arr = np.asarray(roll)
    return range(len(arr)), arr


def compare(series, measures=("shannon", "permutation", "spectral"), window=20):
    """Overlay the rolling entropy of several measures on one axis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in measures:
        roll = _resolve(name).rolling(series, window=window)
        x, y = _xy(roll)
        ax.plot(x, y, label=name)
    ax.set_title(f"Entropy comparison (window={window})")
    ax.set_xlabel("position")
    ax.set_ylabel("entropy")
    ax.legend()
    fig.tight_layout()
    return fig


def dashboard(
    series,
    window=20,
    measures=("shannon", "permutation", "spectral", "sample", "approximate", "differential"),
):
    """Grid of rolling entropy plots, one per measure."""
    n = len(measures)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, measures):
        roll = _resolve(name).rolling(series, window=window)
        x, y = _xy(roll)
        ax.plot(x, y)
        ax.set_title(name)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Entropy dashboard (window={window})")
    fig.tight_layout()
    return fig


def drop_events(series, measure="shannon", window=20, threshold=0.4):
    """Plot rolling entropy and mark positions where it drops > `threshold`."""
    mod = _resolve(measure)
    roll = mod.rolling(series, window=window)
    is_series = hasattr(roll, "to_numpy")
    x, y = _xy(roll)
    drops = np.diff(y, prepend=np.nan)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, label=f"{measure} entropy")
    event_pos = np.where(drops < -abs(threshold))[0]
    x_event = roll.index[event_pos] if is_series else event_pos
    ax.scatter(x_event, y[event_pos], color="red", zorder=5, label="drop event")
    ax.set_title(f"{measure} entropy drop events (threshold={threshold})")
    ax.set_xlabel("position")
    ax.set_ylabel("entropy")
    ax.legend()
    fig.tight_layout()
    return fig
