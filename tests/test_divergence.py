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
