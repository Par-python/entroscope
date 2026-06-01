import numpy as np
import pandas as pd
import pytest
import matplotlib
from entropix import sample


def test_regular_signal_lower_than_noise():
    t = np.linspace(0, 20, 600)
    regular = np.sin(t)
    noise = np.random.RandomState(0).rand(600)
    assert sample.compute(regular, m=2, r=0.2) < sample.compute(noise, m=2, r=0.2)


def test_compute_returns_float():
    assert isinstance(sample.compute(np.random.RandomState(0).rand(200)), float)


def test_compute_nonnegative():
    assert sample.compute(np.random.RandomState(1).rand(200)) >= 0.0


def test_rolling_series_in_series_out_same_length():
    s = pd.Series(np.random.RandomState(2).rand(120))
    out = sample.rolling(s, window=60, m=2, r=0.2)
    assert isinstance(out, pd.Series)
    assert len(out) == 120


def test_invalid_r_raises():
    with pytest.raises(ValueError):
        sample.compute(list(range(50)), r=0.0)


def test_plot_returns_figure():
    fig = sample.plot(np.random.RandomState(3).rand(120), window=60)
    assert isinstance(fig, matplotlib.figure.Figure)
