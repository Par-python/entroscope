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
