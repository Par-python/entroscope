"""Regenerate the README chart (light and dark PNGs next to this file).

    python docs/assets/make_readme_chart.py

The data is synthetic and labelled as such: white noise that, at a known
change point, turns into a regular cycle at the same level. Rolling spectral
entropy is computed with entroscope itself.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from entroscope import spectral

OUT = Path(__file__).resolve().parent
WINDOW, CHANGE = 40, 180

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "signal": "#2a78d6", "entropy": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "signal": "#3987e5", "entropy": "#d95926",
    },
}  # fmt: skip


def make_signal(n=360, seed=4):
    """Noise around 50, then a clean 20-step cycle around the same level."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    noisy = 50 + 8 * rng.randn(n)
    cyclic = 50 + 12 * np.sin(2 * np.pi * t / 20) + 1.5 * rng.randn(n)
    return pd.Series(np.where(t < CHANGE, noisy, cyclic), index=t)


def style(ax, c):
    ax.set_facecolor(c["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.grid(axis="y", color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)


def draw(signal, entropy, lag, c):
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"hspace": 0.35}
    )
    fig.patch.set_facecolor(c["surface"])
    for ax in (top, bottom):
        style(ax, c)
        ax.axvline(CHANGE, color=c["muted"], linewidth=1, linestyle=(0, (3, 3)))

    top.plot(signal.index, signal.to_numpy(), color=c["signal"], linewidth=1.6)
    top.set_title("Signal (synthetic): noise, then a regular cycle", loc="left",
                  color=c["ink"], fontsize=11)  # fmt: skip

    bottom.plot(entropy.index, entropy.to_numpy(), color=c["entropy"], linewidth=1.6)
    bottom.set_ylim(0, None)
    bottom.set_title(f"Rolling spectral entropy, bits (window={WINDOW})", loc="left",
                     color=c["ink"], fontsize=11)  # fmt: skip
    bottom.text(
        CHANGE + 45, 1.6,
        f"regime change at step {CHANGE};\nentropy is halfway down {lag} steps later",
        color=c["secondary"], fontsize=9, va="center",
    )  # fmt: skip
    bottom.set_xlabel("time step", color=c["muted"], fontsize=9)

    fig.text(
        0.125, 0.975, "Rolling entropy flags when a noisy series turns structured",
        color=c["ink"], fontsize=13, fontweight="bold", va="top",
    )  # fmt: skip
    return fig


def main():
    signal = make_signal()
    entropy = spectral.rolling(signal, window=WINDOW)
    noise_level = entropy[entropy.index < CHANGE].mean()
    cycle_level = entropy[entropy.index >= CHANGE + WINDOW].mean()
    halfway = (noise_level + cycle_level) / 2
    after = entropy[entropy.index >= CHANGE]
    lag = int(after.index[(after < halfway).to_numpy()][0]) - CHANGE
    for name, colors in THEMES.items():
        fig = draw(signal, entropy, lag, colors)
        path = OUT / f"entropy-drop-{name}.png"
        fig.savefig(path, dpi=200, facecolor=colors["surface"], bbox_inches="tight")
        plt.close(fig)
        print("wrote", path.relative_to(OUT.parent.parent))
    print(f"noise-phase {noise_level:.2f} bits, cycle-phase {cycle_level:.2f} bits, "
          f"halfway down {lag} steps after the change")  # fmt: skip


if __name__ == "__main__":
    main()
