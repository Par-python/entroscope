import numpy as np
import pandas as pd
import pytest
import matplotlib
from entropix import permutation


def test_monotonic_series_low_entropy():
    assert permutation.compute(list(range(200)), order=3) == pytest.approx(0.0, abs=1e-9)


def test_noise_higher_than_monotonic():
    rng = np.random.RandomState(0)
    noise = permutation.compute(rng.rand(500), order=3)
    mono = permutation.compute(list(range(500)), order=3)
    assert noise > mono


def test_compute_returns_float():
    assert isinstance(permutation.compute(list(range(50)), order=3), float)


def test_normalized_in_unit_interval():
    rng = np.random.RandomState(1)
    n = permutation.normalized(rng.rand(300), order=3)
    assert 0.0 <= n <= 1.0


def test_rolling_series_in_series_out_same_length():
    s = pd.Series(np.random.RandomState(2).rand(60))
    out = permutation.rolling(s, window=20, order=3)
    assert isinstance(out, pd.Series)
    assert len(out) == 60


def test_invalid_order_raises():
    with pytest.raises(ValueError):
        permutation.compute(list(range(10)), order=1)


def test_plot_returns_figure():
    fig = permutation.plot(list(range(60)), window=20, order=3)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_rolling_window_too_small_for_order_raises():
    s = pd.Series(np.random.RandomState(0).rand(50))
    with pytest.raises(ValueError):
        permutation.rolling(s, window=4, order=3, delay=2)
