import numpy as np
import matplotlib
from entropix import multiscale


def test_compute_returns_dict_keyed_by_scale():
    rng = np.random.RandomState(0)
    data = rng.rand(600)
    result = multiscale.compute(data, scales=range(1, 5))
    assert isinstance(result, dict)
    assert set(result.keys()) == {1, 2, 3, 4}
    assert all(isinstance(v, float) for v in result.values())


def test_scale_one_matches_sample_entropy():
    from entropix import sample
    rng = np.random.RandomState(1)
    data = rng.rand(400)
    ms = multiscale.compute(data, scales=range(1, 2))
    assert ms[1] == sample.compute(data)


def test_plot_returns_figure():
    rng = np.random.RandomState(2)
    fig = multiscale.plot(rng.rand(600), scales=range(1, 5))
    assert isinstance(fig, matplotlib.figure.Figure)
