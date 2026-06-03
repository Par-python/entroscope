import numpy as np
import pytest

from entroscope import _transfer_estimators as est


def test_embed_hand_computed_lag1():
    # 6 points, lag=1 -> 5 aligned samples.
    x = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    y = np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
    y_future, y_past, x_past = est.embed(x, y, lag=1)
    # t = 1..5
    np.testing.assert_array_equal(y_future, [21.0, 22.0, 23.0, 24.0, 25.0])
    np.testing.assert_array_equal(y_past,   [20.0, 21.0, 22.0, 23.0, 24.0])
    np.testing.assert_array_equal(x_past,   [10.0, 11.0, 12.0, 13.0, 14.0])


def test_embed_lag2():
    x = np.arange(6, dtype=float)
    y = np.arange(10, 16, dtype=float)
    y_future, y_past, x_past = est.embed(x, y, lag=2)
    # t = 2..5 -> 4 samples
    np.testing.assert_array_equal(y_future, [12.0, 13.0, 14.0, 15.0])
    np.testing.assert_array_equal(y_past,   [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(x_past,   [0.0, 1.0, 2.0, 3.0])


def test_embed_too_short_raises():
    with pytest.raises(ValueError):
        est.embed(np.array([1.0]), np.array([2.0]), lag=1)


from entroscope.utils import knn


def test_kth_neighbor_distance_chebyshev():
    # Points on a line, spacing 1. For point at 0 with k=1, nearest is at 1 -> dist 1.
    pts = np.array([[0.0], [1.0], [2.0], [5.0]])
    d = knn.kth_neighbor_distance(pts, k=1)
    assert d[0] == pytest.approx(1.0)
    assert d[3] == pytest.approx(3.0)  # nearest to 5 is 2


def test_count_within_radius_excludes_self_and_boundary():
    pts = np.array([[0.0], [1.0], [2.0]])
    # radius strictly greater than distance: count points with dist < r (excl self).
    counts = knn.count_within_radius(pts, radii=np.array([1.5, 1.5, 1.5]))
    # point 0: neighbor at 1 (dist1<1.5) -> 1 ; point 1: 0 and 2 -> 2 ; point2: 1 ->1
    np.testing.assert_array_equal(counts, [1, 2, 1])


def test_te_binned_independent_near_zero():
    rng = np.random.RandomState(0)
    x = rng.randn(4000)
    y = rng.randn(4000)
    yf, yp, xp = est.embed(x, y, lag=1)
    te = est.te_binned(yf, yp, xp, bins=6)
    assert abs(te) < 0.05


def test_te_binned_directional_coupling():
    rng = np.random.RandomState(1)
    x = rng.randn(4000)
    y = np.empty_like(x)
    y[0] = rng.randn()
    y[1:] = x[:-1] + 0.1 * rng.randn(3999)  # y_t driven by x_{t-1}
    yf, yp, xp = est.embed(x, y, lag=1)
    te_xy = est.te_binned(yf, yp, xp, bins=6)
    yf2, yp2, xp2 = est.embed(y, x, lag=1)  # reverse direction
    te_yx = est.te_binned(yf2, yp2, xp2, bins=6)
    assert te_xy > te_yx + 0.1


def test_te_ksg_independent_near_zero():
    rng = np.random.RandomState(2)
    x = rng.randn(2000)
    y = rng.randn(2000)
    yf, yp, xp = est.embed(x, y, lag=1)
    te = est.te_ksg(yf, yp, xp, k=4)
    assert abs(te) < 0.05


def test_te_ksg_directional_coupling():
    rng = np.random.RandomState(3)
    x = rng.randn(2000)
    y = np.empty_like(x)
    y[0] = rng.randn()
    y[1:] = x[:-1] + 0.1 * rng.randn(1999)
    yf, yp, xp = est.embed(x, y, lag=1)
    te_xy = est.te_ksg(yf, yp, xp, k=4)
    yf2, yp2, xp2 = est.embed(y, x, lag=1)
    te_yx = est.te_ksg(yf2, yp2, xp2, k=4)
    assert te_xy > te_yx + 0.1
