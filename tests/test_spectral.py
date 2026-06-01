import numpy as np
import pandas as pd
import matplotlib
from entropix import spectral


def test_pure_sine_lower_than_noise():
    t = np.linspace(0, 10, 1000)
    sine = np.sin(2 * np.pi * 3 * t)
    noise = np.random.RandomState(0).rand(1000)
    assert spectral.compute(sine, sf=100.0) < spectral.compute(noise, sf=100.0)


def test_compute_returns_float():
    assert isinstance(spectral.compute(np.random.RandomState(0).rand(200)), float)


def test_normalized_in_unit_interval():
    n = spectral.normalized(np.random.RandomState(1).rand(200))
    assert 0.0 <= n <= 1.0


def test_rolling_series_in_series_out_same_length():
    s = pd.Series(np.random.RandomState(2).rand(80))
    out = spectral.rolling(s, window=40, sf=1.0)
    assert isinstance(out, pd.Series)
    assert len(out) == 80


def test_plot_returns_figure():
    fig = spectral.plot(np.random.RandomState(3).rand(80), window=40)
    assert isinstance(fig, matplotlib.figure.Figure)
