"""Stratified no-replacement subsampling sensitivity (spec 6.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.feature_audit.prototype.describe import describe_features
from research.feature_audit.prototype.stability import (
    STABILITY_COLUMNS,
    stability_summary,
    stratified_subsamples,
)
from research.feature_audit.prototype.validation import validate_input


def test_subsamples_are_stratified_without_replacement():
    y = np.r_[np.zeros(100, dtype=int), np.ones(50, dtype=int)]
    samples = stratified_subsamples(y, repeats=10, seed=9)
    assert len(samples) == 10
    for ids in samples:
        assert len(ids) == len(np.unique(ids)) == 120
        assert (y[ids] == 0).sum() == 80
        assert (y[ids] == 1).sum() == 40


def test_subsample_floor_and_determinism():
    y = np.r_[np.zeros(13, dtype=int), np.ones(101, dtype=int)]
    a = stratified_subsamples(y, repeats=12, seed=4)
    b = stratified_subsamples(y, repeats=12, seed=4)
    c = stratified_subsamples(y, repeats=12, seed=5)
    for s in a:
        assert (y[s] == 0).sum() == 10  # floor(0.8 * 13)
        assert (y[s] == 1).sum() == 80  # floor(0.8 * 101)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not all(np.array_equal(p, q) for p, q in zip(a, c))
    # Independent streams: draws are not all identical to each other.
    assert len({tuple(s) for s in a}) > 1


def prepare(df, types, n_subsamples=10):
    return validate_input(
        df,
        target="y",
        feature_types=types,
        assume_iid=True,
        n_permutations=19,
        n_subsamples=n_subsamples,
        random_state=0,
    )


def test_categorical_copy_is_one_bit_in_every_subsample():
    y = np.tile([0, 1], 100)
    df = pd.DataFrame({"c": np.where(y == 1, "u", "v"), "y": y})
    data = prepare(df, {"c": "categorical"})
    out = stability_summary(data, describe_features(data))
    assert list(out.columns) == STABILITY_COLUMNS
    row = out.loc["c"]
    assert row["subsample_valid_count"] == 10
    for col in ("subsample_mi_median", "subsample_mi_p10", "subsample_mi_p90"):
        assert row[col] == pytest.approx(1.0)


def test_ineligible_features_have_null_summary_and_zero_count():
    df = pd.DataFrame({"k": ["z"] * 200, "y": np.tile([0, 1], 100)})
    data = prepare(df, {"k": "categorical"})
    row = stability_summary(data, describe_features(data)).loc["k"]
    assert row["subsample_valid_count"] == 0
    assert pd.isna(row["subsample_mi_median"])


def test_support_is_rechecked_and_invalid_draws_are_not_replaced():
    # 124 observed rows: floor(0.8*124)=99 < 100 rows, so every draw is invalid.
    x = np.r_[np.random.default_rng(0).normal(size=124), [np.nan] * 76]
    df = pd.DataFrame({"x": x, "y": np.tile([0, 1], 100)})
    data = prepare(df, {"x": "continuous"})
    features = describe_features(data)
    assert features.loc[0, "value_status"] == "eligible"
    row = stability_summary(data, features).loc["x"]
    assert row["subsample_valid_count"] == 0
    assert pd.isna(row["subsample_mi_p10"]) and pd.isna(row["subsample_mi_median"])


def test_class_support_rechecked_within_subsample():
    # Class 1 has 12 observed rows: floor(0.8*12)=9 < 10 in every subsample.
    y = np.r_[np.zeros(188, dtype=int), np.ones(12, dtype=int)]
    df = pd.DataFrame({"x": np.random.default_rng(1).normal(size=200), "y": y})
    data = prepare(df, {"x": "continuous"})
    features = describe_features(data)
    assert features.loc[0, "value_status"] == "eligible"
    row = stability_summary(data, features).loc["x"]
    assert row["subsample_valid_count"] == 0


def test_valid_continuous_summary_uses_all_draws():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": (np.abs(x) > 0.7).astype(int)})
    data = prepare(df, {"x": "continuous"}, n_subsamples=12)
    row = stability_summary(data, describe_features(data)).loc["x"]
    assert row["subsample_valid_count"] == 12
    assert row["subsample_mi_p10"] <= row["subsample_mi_median"] <= row["subsample_mi_p90"]
    assert row["subsample_mi_median"] > 0.2
