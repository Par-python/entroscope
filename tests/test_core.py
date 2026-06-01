import numpy as np
import pandas as pd
import pytest
from entroscope import _core
from entroscope.utils import windows


def test_as_array_from_list():
    arr, idx = _core.as_array([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float
    assert idx is None


def test_as_array_from_series_keeps_index():
    s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    arr, idx = _core.as_array(s)
    assert list(arr) == [1.0, 2.0, 3.0]
    assert list(idx) == ["a", "b", "c"]


def test_as_array_empty_raises():
    with pytest.raises(ValueError):
        _core.as_array([])


def test_wrap_with_index_returns_series():
    out = _core.wrap(np.array([1.0, 2.0]), pd.Index(["a", "b"]))
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["a", "b"]


def test_wrap_without_index_returns_ndarray():
    out = _core.wrap(np.array([1.0, 2.0]), None)
    assert isinstance(out, np.ndarray)


def test_sliding_windows_count_and_shape():
    vals = np.arange(5.0)
    wins = list(windows.sliding_windows(vals, 3))
    assert len(wins) == 3  # windows ending at idx 2,3,4
    assert list(wins[0]) == [0.0, 1.0, 2.0]
    assert list(wins[-1]) == [2.0, 3.0, 4.0]


def test_rolling_length_and_warmup_nan():
    vals = np.arange(6.0)
    out = _core.rolling(vals, 3, lambda w: float(np.mean(w)))
    assert len(out) == 6
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(1.0)  # mean of 0,1,2


def test_rolling_series_preserves_index():
    s = pd.Series(np.arange(6.0), index=list("abcdef"))
    out = _core.rolling(s, 3, lambda w: float(np.mean(w)))
    assert isinstance(out, pd.Series)
    assert list(out.index) == list("abcdef")


def test_rolling_window_too_large_raises():
    with pytest.raises(ValueError):
        _core.rolling(np.arange(3.0), 5, lambda w: 0.0)


def test_delta_is_diff_of_rolling():
    vals = np.arange(6.0)
    roll = _core.rolling(vals, 3, lambda w: float(np.mean(w)))
    dlt = _core.delta(vals, 3, lambda w: float(np.mean(w)))
    assert np.isnan(dlt[2])  # first valid rolling has no previous
    assert dlt[3] == pytest.approx(roll[3] - roll[2])


def test_rolling_window_zero_raises():
    with pytest.raises(ValueError):
        _core.rolling(np.arange(5.0), 0, lambda w: 0.0)


def test_delta_series_preserves_index():
    s = pd.Series(np.arange(6.0), index=list("abcdef"))
    out = _core.delta(s, 3, lambda w: float(np.mean(w)))
    assert isinstance(out, pd.Series)
    assert list(out.index) == list("abcdef")


from entroscope.utils import normalize


def test_normalize_by_max_basic():
    assert normalize.by_max(1.0, 2.0) == pytest.approx(0.5)


def test_normalize_by_max_zero_denominator_returns_zero():
    assert normalize.by_max(0.0, 0.0) == 0.0


def test_normalize_clips_to_unit_interval():
    assert normalize.by_max(3.0, 2.0) == pytest.approx(1.0)
    assert normalize.by_max(-1.0, 2.0) == pytest.approx(0.0)
