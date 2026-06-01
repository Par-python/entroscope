import numpy as np
import pandas as pd
import pytest
import matplotlib
from entropix import shannon


def test_constant_series_zero_entropy():
    assert shannon.compute([5.0] * 50) == pytest.approx(0.0, abs=1e-9)


def test_uniform_data_near_max_entropy():
    # one sample per bin -> entropy near log2(bins)
    data = list(range(100))
    h = shannon.compute(data, bins=10)
    assert h == pytest.approx(np.log2(10), abs=0.1)


def test_compute_returns_float():
    assert isinstance(shannon.compute([1, 2, 3, 4, 5, 6]), float)


def test_normalized_in_unit_interval():
    n = shannon.normalized(list(range(100)), bins=10)
    assert 0.0 <= n <= 1.0


def test_rolling_same_length_and_series_in_series_out():
    s = pd.Series(np.random.RandomState(0).rand(50))
    out = shannon.rolling(s, window=10, bins=5)
    assert isinstance(out, pd.Series)
    assert len(out) == 50
    assert out.iloc[:9].isna().all()


def test_geographic_uniform_regions():
    df = pd.DataFrame({"region": ["a", "b", "c", "d"], "interest": [25, 25, 25, 25]})
    h = shannon.geographic(df, col="interest")
    assert h == pytest.approx(np.log2(4), abs=1e-9)


def test_plot_returns_figure():
    fig = shannon.plot(list(range(50)), window=10)
    assert isinstance(fig, matplotlib.figure.Figure)
