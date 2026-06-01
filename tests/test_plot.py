import numpy as np
import matplotlib
from entropix import plot


def _data():
    return np.random.RandomState(0).rand(100)


def test_compare_returns_figure():
    fig = plot.compare(_data(), measures=["shannon", "permutation", "spectral"], window=20)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_dashboard_returns_figure():
    fig = plot.dashboard(_data(), window=20)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_drop_events_returns_figure():
    fig = plot.drop_events(_data(), measure="shannon", window=20, threshold=0.4)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_compare_unknown_measure_raises():
    import pytest
    with pytest.raises(ValueError):
        plot.compare(_data(), measures=["bogus"], window=20)
