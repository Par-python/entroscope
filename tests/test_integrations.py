"""polars input support and the scikit-learn EntropyFeatures transformer."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from entroscope import permutation, plot, sample, shannon, spectral, transfer

pl = pytest.importorskip("polars")

RNG = np.random.RandomState(0)
VALUES = RNG.randn(120)


# ---------------------------------------------------------------------------
# polars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("measure", [shannon, permutation, spectral, sample])
def test_polars_compute_matches_numpy(measure):
    assert measure.compute(pl.Series("x", VALUES)) == measure.compute(VALUES)


@pytest.mark.parametrize("fn", ["rolling", "delta"])
def test_polars_in_polars_out_keeps_name(fn):
    out = getattr(shannon, fn)(pl.Series("sales", VALUES), window=20)
    assert isinstance(out, pl.Series)
    assert out.name == "sales"
    np.testing.assert_array_equal(out.to_numpy(), getattr(shannon, fn)(VALUES, window=20))


def test_polars_normalized_and_integer_input():
    s = pl.Series("n", np.arange(50) % 7)  # integer dtype
    assert 0.0 <= shannon.normalized(s) <= 1.0


def test_polars_transfer_rolling_and_delta():
    x, y = pl.Series("x", VALUES), pl.Series("y", np.roll(VALUES, 1))
    roll = transfer.rolling(x, y, window=60)
    d = transfer.delta(x, y, window=60)
    assert isinstance(roll, pl.Series) and isinstance(d, pl.Series)
    assert roll.name == d.name == "y"
    np.testing.assert_allclose(d.to_numpy()[61:], np.diff(roll.to_numpy())[60:])


def test_polars_plots():
    s = pl.Series("x", VALUES)
    figs = [
        shannon.plot(s, window=20),
        transfer.plot(s, s, window=60),
        plot.compare(s, window=20),
        plot.drop_events(s, window=20, threshold=0.1),
    ]
    for fig in figs:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# scikit-learn
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from entroscope.features import EntropyFeatures

WINDOWS = RNG.randn(12, 64)


def test_features_values_match_compute():
    out = EntropyFeatures(measures=("shannon", "sample")).fit_transform(WINDOWS)
    assert out.shape == (12, 2)
    np.testing.assert_array_equal(out[:, 0], [shannon.compute(w) for w in WINDOWS])
    np.testing.assert_array_equal(out[:, 1], [sample.compute(w) for w in WINDOWS])


def test_features_params_are_forwarded():
    default = EntropyFeatures(measures=("sample",)).fit_transform(WINDOWS)
    loose = EntropyFeatures(measures=("sample",), params={"sample": {"r": 0.5}}).fit_transform(
        WINDOWS
    )
    np.testing.assert_array_equal(loose[:, 0], [sample.compute(w, r=0.5) for w in WINDOWS])
    assert not np.allclose(default, loose)


def test_features_pandas_output_and_names():
    t = EntropyFeatures(measures=("shannon", "spectral")).set_output(transform="pandas")
    df = t.fit_transform(pd.DataFrame(WINDOWS))
    assert list(df.columns) == ["shannon_entropy", "spectral_entropy"]


def test_features_in_pipeline_separates_noise_from_sine():
    t = np.arange(64)
    rng = np.random.RandomState(1)
    sines = [np.sin(2 * np.pi * t / p) + 0.05 * rng.randn(64) for p in range(8, 28)]
    noise = [rng.randn(64) for _ in range(20)]
    X, y = np.vstack(sines + noise), np.array([0] * 20 + [1] * 20)
    model = make_pipeline(
        EntropyFeatures(measures=("spectral", "permutation")), LogisticRegression()
    )
    assert model.fit(X, y).score(X, y) == 1.0
    assert isinstance(clone(model), type(model))


@pytest.mark.parametrize(
    "kwargs",
    [{"measures": ("nope",)}, {"measures": ()}, {"params": {"sample": {"m": 2}}}],
)
def test_features_bad_config_raises(kwargs):
    kwargs.setdefault("measures", ("shannon",))
    with pytest.raises(ValueError):
        EntropyFeatures(**kwargs).fit(WINDOWS)


def test_features_width_mismatch_raises():
    t = EntropyFeatures().fit(WINDOWS)
    with pytest.raises(ValueError, match="expecting 64 features"):
        t.transform(WINDOWS[:, :32])
