"""Descriptive metadata and exclusion-policy tests (spec 6.1, 6.2, 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.feature_audit.prototype.describe import FEATURE_COLUMNS, describe_features
from research.feature_audit.prototype.validation import validate_input


def prepare(df, types, target="y"):
    return validate_input(
        df,
        target=target,
        feature_types=types,
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )


def test_balanced_category_entropy_and_unique_id_exclusion():
    df = pd.DataFrame(
        {"category": ["a", "b"] * 100, "id": [f"person-{i}" for i in range(200)], "y": [0, 1] * 100}
    )
    data = validate_input(
        df,
        target="y",
        feature_types={"category": "categorical", "id": "categorical"},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )
    rows = describe_features(data).set_index("feature")
    assert rows.loc["category", "entropy_bits"] == pytest.approx(1.0)
    assert rows.loc["id", "value_status"] == "high_cardinality"


def test_schema_order_and_every_predictor_present():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"b": rng.normal(size=200), "a": ["p", "q"] * 100, "y": [0, 1] * 100})
    out = describe_features(prepare(df, {"b": "continuous", "a": "categorical"}))
    assert list(out.columns) == FEATURE_COLUMNS
    assert list(out["feature"]) == ["b", "a"]
    assert out["subsample_mi_median"].isna().all()
    assert (out["subsample_valid_count"] == 0).all()
    row = out.set_index("feature").loc["b"]
    assert row["declared_type"] == "continuous"
    assert row["value_status"] == "eligible"
    assert pd.isna(row["entropy_bits"])  # numeric entropy not reported in v1


def _statuses(df, types):
    return describe_features(prepare(df, types)).set_index("feature")


def test_all_missing_and_constant():
    y = [0, 1] * 100
    df = pd.DataFrame(
        {
            "gone": [np.nan] * 200,
            "const": ["k"] * 200,
            "const_num": [3.0] * 150 + [np.nan] * 50,
            "y": y,
        }
    )
    rows = _statuses(df, {"gone": "continuous", "const": "categorical", "const_num": "continuous"})
    assert rows.loc["gone", "value_status"] == "all_missing"
    assert rows.loc["gone", "n_observed"] == 0
    assert rows.loc["gone", "missing_fraction"] == 1.0
    assert pd.isna(rows.loc["gone", "entropy_bits"])
    assert pd.isna(rows.loc["gone", "dominant_fraction"])
    assert "missingness_untestable" in rows.loc["gone", "warning_codes"]
    assert rows.loc["const", "value_status"] == "constant"
    assert rows.loc["const", "entropy_bits"] == 0.0
    assert rows.loc["const", "dominant_fraction"] == 1.0
    assert rows.loc["const_num", "value_status"] == "constant"
    assert rows.loc["const_num", "n_observed"] == 150


def test_ninety_nine_observed_rows_is_insufficient():
    x = np.r_[np.linspace(0, 1, 99)[::-1], [np.nan] * 101]
    df = pd.DataFrame({"x": x, "y": [0, 1] * 100})
    row = _statuses(df, {"x": "continuous"}).loc["x"]
    assert row["value_status"] == "insufficient_support"
    assert row["n_observed"] == 99
    assert row["missing_fraction"] == pytest.approx(101 / 200)


def test_class_disappearing_after_dropping_missing_x():
    # 150 class-0 rows, 50 class-1 rows; x observed on only 5 class-1 rows.
    y = np.r_[np.zeros(150, dtype=int), np.ones(50, dtype=int)]
    x = np.r_[np.linspace(0, 1, 150), np.linspace(0, 1, 5), [np.nan] * 45]
    df = pd.DataFrame({"x": x, "y": y})
    row = _statuses(df, {"x": "continuous"}).loc["x"]
    assert row["value_status"] == "insufficient_support"
    assert row["n_observed"] == 155


def test_singleton_category_is_rare():
    cats = ["a", "b"] * 99 + ["c", "a"]
    df = pd.DataFrame({"c": cats, "y": [0, 1] * 100})
    row = _statuses(df, {"c": "categorical"}).loc["c"]
    assert row["value_status"] == "rare_categories"
    assert row["n_unique_observed"] == 3


def test_twenty_percent_unique_boundary():
    at = [f"k{i % 40}" for i in range(200)]  # 40 / 200 = 0.20 -> excluded
    below = [f"k{i % 39}" for i in range(195)] + ["k0"] * 5  # 39 / 200 < 0.20
    df = pd.DataFrame({"at": at, "below": below, "y": [0, 1] * 100})
    rows = _statuses(df, {"at": "categorical", "below": "categorical"})
    assert rows.loc["at", "value_status"] == "high_cardinality"
    assert rows.loc["below", "value_status"] == "eligible"


def test_more_than_hundred_categories_is_high_cardinality_even_below_twenty_percent():
    cats = [f"k{i % 101}" for i in range(1010)]
    df = pd.DataFrame({"c": cats, "y": np.arange(1010) % 2})
    row = _statuses(df, {"c": "categorical"}).loc["c"]
    assert row["n_unique_observed"] == 101
    assert row["value_status"] == "high_cardinality"


def test_monotone_unique_numeric_id_vs_nonmonotone_measurement():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "row_id": np.arange(200) * 3.0,
            "desc_id": -np.arange(200.0),
            "measure": rng.permutation(np.arange(200) * 0.5),
            "y": [0, 1] * 100,
        }
    )
    rows = _statuses(df, {"row_id": "continuous", "desc_id": "continuous", "measure": "continuous"})
    assert rows.loc["row_id", "value_status"] == "possible_identifier"
    assert rows.loc["desc_id", "value_status"] == "possible_identifier"
    assert rows.loc["measure", "value_status"] == "eligible"


def test_low_distinct_continuous_is_ambiguous():
    df = pd.DataFrame({"score": np.tile(np.arange(9.0), 23)[:200], "y": [0, 1] * 100})
    row = _statuses(df, {"score": "continuous"}).loc["score"]
    assert row["value_status"] == "ambiguous_discrete"
    assert row["n_unique_observed"] == 9


def test_precedence_keeps_all_warning_codes():
    # Insufficient support and high cardinality both apply; precedence picks support.
    ids = [f"u{i}" for i in range(90)] + [None] * 110
    df = pd.DataFrame({"u": ids, "y": [0, 1] * 100})
    row = _statuses(df, {"u": "categorical"}).loc["u"]
    assert row["value_status"] == "insufficient_support"
    assert "high_cardinality" in row["warning_codes"]
    assert "rare_categories" in row["warning_codes"]


def test_observed_only_counts_and_dominance():
    c = ["a"] * 120 + ["b"] * 40 + [None] * 40
    df = pd.DataFrame({"c": c, "y": [0, 1] * 100})
    row = _statuses(df, {"c": "categorical"}).loc["c"]
    assert row["n_total"] == 200
    assert row["n_observed"] == 160
    assert row["dominant_fraction"] == pytest.approx(0.75)
    p = np.array([0.75, 0.25])
    assert row["entropy_bits"] == pytest.approx(-(p * np.log2(p)).sum())
