import numpy as np
import pandas as pd
import pytest
import matplotlib
from entropix import differential


def test_normal_matches_analytic_formula():
    rng = np.random.RandomState(0)
    data = rng.normal(0, 1, 5000)
    h = differential.compute(data, dist="normal")
    expected = 0.5 * np.log(2 * np.pi * np.e * np.var(data))
    assert h == pytest.approx(expected, abs=1e-6)


def test_wider_normal_has_higher_entropy():
    rng = np.random.RandomState(1)
    narrow = differential.compute(rng.normal(0, 1, 3000), dist="normal")
    wide = differential.compute(rng.normal(0, 5, 3000), dist="normal")
    assert wide > narrow


def test_kde_returns_float():
    rng = np.random.RandomState(2)
    assert isinstance(differential.compute(rng.normal(0, 1, 500), dist="kde"), float)


def test_invalid_dist_raises():
    with pytest.raises(ValueError):
        differential.compute(list(range(50)), dist="bogus")


def test_rolling_series_in_series_out_same_length():
    s = pd.Series(np.random.RandomState(3).normal(0, 1, 120))
    out = differential.rolling(s, window=60, dist="kde")
    assert isinstance(out, pd.Series)
    assert len(out) == 120


def test_plot_returns_figure():
    fig = differential.plot(np.random.RandomState(4).normal(0, 1, 120), window=60)
    assert isinstance(fig, matplotlib.figure.Figure)
