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
    p = np.array([0.1, 0.1, 0.1, 0.6])           # 3 in [0,0.5), 1 in [0.5,1)
    q = np.array([0.1, 0.6])                       # 1 in each
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
