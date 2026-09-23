"""Regenerate the README banner (banner-light.png and banner-dark.png next to this file).

    python docs/assets/make_banner.py

Adapted from the original banner design in assets/make_banner.py. The curve is
entroscope's own rolling spectral entropy of a signal that starts as noise and
settles into a clean oscillation, so the banner shows the library doing its job.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from entroscope import spectral

OUT = Path(__file__).resolve().parent

THEMES = {
    "light": {
        "bg": "#ffffff", "ink": "#1b1f24", "sub": "#3a4149", "mute": "#6b7280",
        "line": "#2a78d6", "grid": "#e7e9ec", "rule": "#d7dbe0",
    },
    "dark": {
        "bg": "#0d1117", "ink": "#e6edf3", "sub": "#c9d1d9", "mute": "#8b949e",
        "line": "#4493f8", "grid": "#21262d", "rule": "#30363d",
    },
}  # fmt: skip

MEASURES = (
    "shannon · permutation · sample · approximate · spectral · "
    "differential · multiscale · transfer · divergence"
)


def _pick(names, default="DejaVu Sans"):
    available = {f.name for f in fm.fontManager.ttflist}
    return next((name for name in names if name in available), default)


SANS = _pick(["Helvetica Neue", "Helvetica", "Arial"])
MONO = _pick(["SF Mono", "Menlo", "DejaVu Sans Mono"], default="DejaVu Sans Mono")


def make_signal(n=600, seed=7):
    """Pure noise that ramps into a clean sine over the middle of the series."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24 * np.pi, n)
    emergence = np.clip((np.arange(n) - n * 0.30) / (n * 0.40), 0, 1)
    return pd.Series((1 - emergence) * 1.6 * rng.standard_normal(n) + emergence * 2.2 * np.sin(t))


def draw(entropy, c):
    fig = plt.figure(figsize=(12.8, 4.6), dpi=100)
    fig.patch.set_facecolor(c["bg"])

    fig.text(0.05, 0.84, "entroscope", color=c["ink"], fontsize=44, fontweight="bold",
             fontfamily=SANS, va="center")  # fmt: skip
    fig.text(0.05, 0.69, "the definitive entropy toolkit for time series data",
             color=c["sub"], fontsize=15, fontfamily=SANS, va="center")  # fmt: skip
    fig.text(0.05, 0.605, MEASURES, color=c["mute"], fontsize=10.5, fontfamily=SANS,
             va="center")  # fmt: skip
    fig.text(0.95, 0.84, "pip install entroscope", color=c["ink"], fontsize=13,
             fontfamily=MONO, ha="right", va="center")  # fmt: skip
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.53, 0.53], transform=fig.transFigure,
                              color=c["rule"], lw=1.0))  # fmt: skip

    ax = fig.add_axes([0.05, 0.08, 0.90, 0.38])
    ax.set_facecolor(c["bg"])
    ax.grid(axis="y", color=c["grid"], lw=1.0)
    ax.set_axisbelow(True)
    ax.plot(entropy.index, entropy.to_numpy(), color=c["line"], lw=1.8,
            solid_capstyle="round")  # fmt: skip
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
    ax.spines["bottom"].set_color(c["rule"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.0)
    ax.set_ylim(bottom=-0.06 * float(entropy.max()))  # keep the flat tail off the baseline
    ax.text(0.99, 0.95, "rolling spectral entropy falls as noise turns into a rhythm",
            transform=ax.transAxes, color=c["mute"], fontsize=9.5, fontfamily=SANS,
            ha="right", va="top")  # fmt: skip
    return fig


def main():
    entropy = spectral.rolling(make_signal(), window=100)  # two full cycles
    for name, colors in THEMES.items():
        fig = draw(entropy, colors)
        path = OUT / f"banner-{name}.png"
        fig.savefig(path, dpi=100, facecolor=colors["bg"])
        plt.close(fig)
        print("wrote", path.relative_to(OUT.parent.parent))


if __name__ == "__main__":
    main()
