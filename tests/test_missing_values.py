"""Non-finite input (NaN, +/-inf) yields NaN, consistently, instead of a plausible number.

Rolling results are NaN only for the windows that contain a non-finite value.
"""

import math

import numpy as np
import pandas as pd
import pytest

from entroscope import (
    approximate,
    differential,
    divergence,
    multiscale,
    permutation,
    sample,
    shannon,
    spectral,
    transfer,
)

CLEAN = np.random.RandomState(0).randn(120)
BAD_AT = 50
WINDOW = 20


def _with(value):
    x = CLEAN.copy()
    x[BAD_AT] = value
    return x


SINGLE = [
    pytest.param(shannon.compute, id="shannon"),
    pytest.param(permutation.compute, id="permutation"),
    pytest.param(spectral.compute, id="spectral"),
    pytest.param(sample.compute, id="sample"),
    pytest.param(approximate.compute, id="approximate"),
    pytest.param(lambda x: differential.compute(x, dist="normal"), id="differential-normal"),
    pytest.param(lambda x: differential.compute(x, dist="kde"), id="differential-kde"),
    pytest.param(shannon.normalized, id="shannon-normalized"),
    pytest.param(permutation.normalized, id="permutation-normalized"),
    pytest.param(spectral.normalized, id="spectral-normalized"),
]

ROLLING = [shannon, permutation, spectral, sample, approximate, differential]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("fn", SINGLE)
def test_compute_with_non_finite_is_nan(fn, bad):
    assert math.isnan(fn(_with(bad)))


@pytest.mark.parametrize("fn", SINGLE)
def test_compute_all_nan_is_nan(fn):
    assert math.isnan(fn(np.full(64, np.nan)))


@pytest.mark.parametrize("measure", ROLLING, ids=lambda m: m.__name__.split(".")[-1])
def test_rolling_nan_only_in_windows_containing_the_gap(measure):
    clean = measure.rolling(pd.Series(CLEAN), window=WINDOW)
    gappy = measure.rolling(pd.Series(_with(np.nan)), window=WINDOW)
    # Window ending at t covers t-WINDOW+1..t, so it contains BAD_AT for these ends.
    affected = np.arange(BAD_AT, BAD_AT + WINDOW)
    assert gappy.iloc[affected].isna().all()
    unaffected = np.setdiff1d(np.arange(WINDOW - 1, len(CLEAN)), affected)
    pd.testing.assert_series_equal(gappy.iloc[unaffected], clean.iloc[unaffected])


def test_delta_nan_around_the_gap():
    d = shannon.delta(_with(np.nan), window=WINDOW)
    assert np.isnan(d[BAD_AT : BAD_AT + WINDOW + 1]).all()
    assert np.isfinite(d[WINDOW:BAD_AT]).all()


def test_multiscale_non_finite_is_nan_at_every_scale():
    clean = multiscale.compute(CLEAN, scales=range(1, 5))
    gappy = multiscale.compute(_with(np.nan), scales=range(1, 5))
    assert gappy.keys() == clean.keys()
    assert all(math.isnan(v) for v in gappy.values())


@pytest.mark.parametrize("method", ["ksg", "binned"])
@pytest.mark.parametrize("which", ["x", "y"])
def test_transfer_non_finite_is_nan(method, which):
    x, y = CLEAN, np.roll(CLEAN, 1)
    if which == "x":
        x = _with(np.nan)
    else:
        y = _with(np.nan)
    assert math.isnan(transfer.compute(x, y, method=method))


def test_transfer_rolling_nan_only_near_the_gap():
    x, y = _with(np.nan), np.roll(CLEAN, 1)
    roll = transfer.rolling(x, y, window=60)
    assert np.isnan(roll[BAD_AT : BAD_AT + 60]).all()
    assert np.isfinite(roll[BAD_AT + 60 :]).all()


@pytest.mark.parametrize("fn", [divergence.kl, divergence.js], ids=["kl", "js"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_divergence_non_finite_is_nan(fn, bad):
    assert math.isnan(fn(_with(bad), CLEAN))
    assert math.isnan(fn(CLEAN, _with(bad)))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(lambda: shannon.compute(np.ones(10)), id="shannon-constant"),
        pytest.param(lambda: permutation.compute(np.arange(10.0)), id="permutation-ordered"),
        pytest.param(lambda: shannon.geographic({"interest": [5.0]}), id="geographic-one-region"),
    ],
)
def test_zero_entropy_is_positive_zero(value):
    result = value()
    assert result == 0.0
    assert math.copysign(1.0, result) == 1.0, "got -0.0"
