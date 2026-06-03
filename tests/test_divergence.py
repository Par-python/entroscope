import numpy as np
import pytest

from entroscope import divergence as dv


def test_binned_probs_shared_edges_align_lengths():
    # Two samples over DIFFERENT ranges must still produce equal-length vectors.
    p = np.array([0.0, 1.0, 2.0])
    q = np.array([10.0, 11.0, 12.0])
    pp, qq = dv._binned_probs(p, q, bins=5)
    assert len(pp) == len(qq) == 5
    assert pp.sum() == pytest.approx(1.0)
    assert qq.sum() == pytest.approx(1.0)
    # p occupies the low bins, q the high bins -> their mass is in different bins.
    assert pp[0] > 0 and qq[-1] > 0
    assert pp[-1] == 0 and qq[0] == 0


def test_binned_probs_degenerate_constant_input():
    p = np.array([5.0, 5.0, 5.0])
    q = np.array([5.0, 5.0, 5.0])
    pp, qq = dv._binned_probs(p, q, bins=10)
    assert pp.sum() == pytest.approx(1.0)
    assert qq.sum() == pytest.approx(1.0)
    assert len(pp) == len(qq)


def test_kl_identical_is_zero():
    rng = np.random.RandomState(0)
    x = rng.randn(5000)
    assert dv.kl(x, x.copy(), bins=20) == pytest.approx(0.0, abs=1e-6)


def test_kl_known_value():
    # Construct samples that bin into known probabilities over 2 bins on [0,1).
    # p: 75% in bin0, 25% in bin1 ; q: 50/50.
    p = np.array([0.1, 0.1, 0.1, 0.6])  # 3 in [0,0.5), 1 in [0.5,1)
    q = np.array([0.1, 0.6])  # 1 in each
    # KL = 0.75*log2(0.75/0.5) + 0.25*log2(0.25/0.5) = 0.75*0.585 - 0.25 = 0.1887
    val = dv.kl(p, q, bins=2)
    assert val == pytest.approx(0.1887, abs=0.005)


def test_kl_finite_when_q_has_empty_bin():
    # q has no mass where p does -> would be +inf without smoothing.
    p = np.array([0.0, 0.0, 1.0, 1.0])
    q = np.array([0.0, 0.0, 0.0, 0.0])
    val = dv.kl(p, q, bins=4)
    assert np.isfinite(val)


def test_kl_bad_bins_raises():
    with pytest.raises(ValueError):
        dv.kl(np.array([1.0, 2.0]), np.array([1.0, 2.0]), bins=0)


def test_js_identical_is_zero():
    rng = np.random.RandomState(1)
    x = rng.randn(5000)
    assert dv.js(x, x.copy(), bins=20) == pytest.approx(0.0, abs=1e-6)


def test_js_is_symmetric():
    rng = np.random.RandomState(2)
    a = rng.randn(3000)
    b = rng.randn(3000) + 2.0
    assert dv.js(a, b, bins=20) == pytest.approx(dv.js(b, a, bins=20), abs=1e-9)


def test_js_bounded_zero_to_one():
    rng = np.random.RandomState(3)
    a = rng.randn(3000)
    b = rng.randn(3000) + 50.0  # effectively disjoint supports
    val = dv.js(a, b, bins=20)
    assert 0.0 <= val <= 1.0
    assert val > 0.9  # near the upper bound for disjoint distributions


def test_js_known_value():
    # Two 2-bin distributions: p=[1,0], q=[0,1] (disjoint) -> JS = 1 bit exactly.
    p = np.array([0.1, 0.1])  # both in bin0 of shared range [0.1,0.9]
    q = np.array([0.9, 0.9])  # both in bin1
    val = dv.js(p, q, bins=2)
    assert val == pytest.approx(1.0, abs=1e-3)


import matplotlib


def test_plot_returns_figure():
    rng = np.random.RandomState(4)
    a = rng.randn(1000)
    b = rng.randn(1000) + 1.0
    fig = dv.plot(a, b, bins=20)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_divergence_in_public_api():
    import entroscope

    assert "divergence" in entroscope.__all__
    assert hasattr(entroscope.divergence, "kl")
    assert hasattr(entroscope.divergence, "js")
