import numpy as np
import pandas as pd
import pytest
import matplotlib
from entropix import approximate


def test_regular_signal_lower_than_noise():
    t = np.linspace(0, 20, 500)
    regular = np.sin(t)
    noise = np.random.RandomState(0).rand(500)
    assert approximate.compute(regular, m=2, r=0.2) < approximate.compute(noise, m=2, r=0.2)


def test_compute_returns_float():
    assert isinstance(approximate.compute(np.random.RandomState(0).rand(200)), float)


def test_rolling_series_in_series_out_same_length():
    s = pd.Series(np.random.RandomState(2).rand(120))
    out = approximate.rolling(s, window=60, m=2, r=0.2)
    assert isinstance(out, pd.Series)
    assert len(out) == 120


def test_invalid_r_raises():
    with pytest.raises(ValueError):
        approximate.compute(list(range(50)), r=-1.0)


def test_plot_returns_figure():
    fig = approximate.plot(np.random.RandomState(3).rand(120), window=60)
    assert isinstance(fig, matplotlib.figure.Figure)
