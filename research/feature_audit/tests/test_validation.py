"""Input-contract tests for the feature-audit prototype (spec section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.feature_audit.prototype.validation import validate_input


def valid_frame():
    return pd.DataFrame(
        {"x": np.linspace(-1, 1, 200), "y": np.tile([0, 1], 100)},
        index=np.arange(200)[::-1],
    )


def prepare(df, **overrides):
    params = dict(
        target="y",
        feature_types={"x": "continuous"},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )
    params.update(overrides)
    return validate_input(df, **params)


def test_copies_without_reordering():
    df = valid_frame()
    original = df.copy(deep=True)
    data = prepare(df)
    data.frame.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(df, original)
    np.testing.assert_array_equal(data.frame.index, original.index)
    np.testing.assert_array_equal(data.target, np.tile([0, 1], 100))


def test_iid_acknowledgment_required():
    with pytest.raises(ValueError, match="independent"):
        prepare(valid_frame(), assume_iid=False)


def test_target_is_excluded_from_predictors_and_config_is_recorded():
    data = prepare(valid_frame())
    assert list(data.frame.columns) == ["x"]
    assert data.feature_types == {"x": "continuous"}
    cfg = data.config
    assert cfg["n_permutations"] == 19
    assert cfg["n_subsamples"] == 10
    assert cfg["random_state"] == 0
    assert cfg["subsample_fraction"] == 0.8
    assert cfg["n_neighbors"] == 3
    assert cfg["alpha"] == 0.05
    assert cfg["max_rows"] == 5000
    assert cfg["max_predictors"] == 50


def test_feature_types_mapping_is_copied():
    types = {"x": "continuous"}
    data = prepare(valid_frame(), feature_types=types)
    types["x"] = "categorical"
    assert data.feature_types == {"x": "continuous"}


def test_reduced_permutations_warn_about_resolution():
    data = prepare(valid_frame(), n_permutations=19)
    assert any("0.05" in w and "permutation" in w for w in data.warnings)
    full = prepare(valid_frame(), n_permutations=1999)
    assert not any("resolution" in w for w in full.warnings)


def test_string_target_uses_first_seen_codes():
    df = valid_frame()
    df["y"] = np.tile(["zeta", "alpha"], 100)
    data = prepare(df)
    np.testing.assert_array_equal(data.target, np.tile([0, 1], 100))
    assert data.n_classes == 2


def test_bool_category_and_integral_float_targets_are_accepted():
    for values in (
        np.tile([True, False], 100),
        pd.Categorical(np.tile(["b", "a"], 100)),
        np.tile([1.0, 2.0], 100),
        pd.array(np.tile([3, 7], 100), dtype="Int64"),
    ):
        df = valid_frame()
        df["y"] = values
        np.testing.assert_array_equal(prepare(df).target, np.tile([0, 1], 100))


def test_nullable_numeric_and_na_categorical_predictors_are_accepted():
    df = valid_frame()
    df["i"] = pd.array([1, None] * 100, dtype="Int64")
    df["f"] = pd.array([0.5, None] * 100, dtype="Float64")
    df["c"] = pd.Series(["a", pd.NA, "b", None] * 50, dtype=object, index=df.index)
    df["s"] = pd.Series(["a", pd.NA] * 100, dtype="string", index=df.index)
    types = {
        "x": "continuous",
        "i": "continuous",
        "f": "continuous",
        "c": "categorical",
        "s": "categorical",
    }
    data = prepare(df, feature_types=types)
    assert np.isnan(data.values["i"][1])
    assert data.values["f"][0] == 0.5
    np.testing.assert_array_equal(data.values["c"][:4], [0, -1, 1, -1])
    np.testing.assert_array_equal(data.values["s"][:2], [0, -1])


def test_integer_codes_declared_categorical_are_not_treated_as_geometry():
    df = valid_frame()
    df["code"] = np.tile([30, 10, 20, 10], 50)
    data = prepare(df, feature_types={"x": "continuous", "code": "categorical"})
    np.testing.assert_array_equal(data.values["code"][:4], [0, 1, 2, 1])


def _frame_with(column, values):
    df = valid_frame()
    df[column] = values
    return df


BAD_CASES = {
    "not_a_frame": (lambda: valid_frame()["x"], {}, TypeError, "DataFrame"),
    "missing_target_column": (valid_frame, {"target": "nope"}, ValueError, "target"),
    "target_missing_value": (
        lambda: _frame_with("y", [np.nan] + [0.0, 1.0] * 99 + [1.0]),
        {},
        ValueError,
        "target.*missing",
    ),
    "one_class": (lambda: _frame_with("y", 1), {}, ValueError, "target.*two classes"),
    "nonintegral_target": (
        lambda: _frame_with("y", np.tile([0.5, 1.0], 100)),
        {},
        ValueError,
        "target.*integral",
    ),
    "too_many_classes": (
        lambda: _frame_with("y", np.tile(np.arange(21), 10)[:200]),
        {},
        ValueError,
        "target.*20",
    ),
    "small_class": (
        lambda: _frame_with("y", np.r_[np.zeros(191), np.ones(9)]),
        {},
        ValueError,
        "target.*10",
    ),
    "datetime_target": (
        lambda: _frame_with("y", pd.date_range("2020", periods=200)),
        {},
        TypeError,
        "target",
    ),
    "duplicate_columns": (
        lambda: pd.concat([valid_frame(), valid_frame()[["x"]]], axis=1),
        {},
        ValueError,
        "column names",
    ),
    "non_string_column": (
        lambda: _frame_with(3, 1.0),
        {"feature_types": {"x": "continuous", 3: "continuous"}},
        TypeError,
        "column names",
    ),
    "duplicate_index": (
        lambda: valid_frame().set_axis(np.zeros(200, dtype=int), axis=0),
        {},
        ValueError,
        "index",
    ),
    "missing_type_entry": (
        lambda: _frame_with("z", 1.0),
        {},
        ValueError,
        "feature_types.*z",
    ),
    "extra_type_entry": (
        valid_frame,
        {"feature_types": {"x": "continuous", "w": "continuous"}},
        ValueError,
        "feature_types.*w",
    ),
    "target_type_entry": (
        valid_frame,
        {"feature_types": {"x": "continuous", "y": "categorical"}},
        ValueError,
        "feature_types.*target",
    ),
    "bad_type_value": (
        valid_frame,
        {"feature_types": {"x": "numeric"}},
        ValueError,
        "feature_types.*x",
    ),
    "feature_types_not_mapping": (
        valid_frame,
        {"feature_types": ["x"]},
        TypeError,
        "feature_types",
    ),
    "bool_seed": (valid_frame, {"random_state": True}, TypeError, "random_state"),
    "negative_seed": (valid_frame, {"random_state": -1}, ValueError, "random_state"),
    "huge_seed": (valid_frame, {"random_state": 2**32}, ValueError, "random_state"),
    "float_seed": (valid_frame, {"random_state": 1.0}, TypeError, "random_state"),
    "few_permutations": (valid_frame, {"n_permutations": 18}, ValueError, "n_permutations"),
    "bool_permutations": (valid_frame, {"n_permutations": True}, TypeError, "n_permutations"),
    "few_subsamples": (valid_frame, {"n_subsamples": 9}, ValueError, "n_subsamples"),
    "bool_subsamples": (valid_frame, {"n_subsamples": False}, TypeError, "n_subsamples"),
    "too_few_rows": (lambda: valid_frame().iloc[:99], {}, ValueError, "rows"),
    "too_many_rows": (
        lambda: pd.DataFrame({"x": np.arange(5001.0), "y": np.arange(5001) % 2}),
        {},
        ValueError,
        "rows",
    ),
    "no_predictors": (
        lambda: valid_frame()[["y"]],
        {"feature_types": {}},
        ValueError,
        "predictor",
    ),
    "too_many_predictors": (
        lambda: pd.DataFrame(
            {**{f"x{i}": np.arange(200.0) for i in range(51)}, "y": np.arange(200) % 2}
        ),
        {"feature_types": {f"x{i}": "continuous" for i in range(51)}},
        ValueError,
        "predictor",
    ),
    "nested_category": (
        lambda: _frame_with("c", [[1], [2]] * 100),
        {"feature_types": {"x": "continuous", "c": "categorical"}},
        TypeError,
        "c",
    ),
    "tuple_category": (
        lambda: _frame_with("c", [(1, 2), (3, 4)] * 100),
        {"feature_types": {"x": "continuous", "c": "categorical"}},
        TypeError,
        "c",
    ),
    "mixed_category": (
        lambda: _frame_with("c", ["a", 1] * 100),
        {"feature_types": {"x": "continuous", "c": "categorical"}},
        TypeError,
        "c.*mixed",
    ),
    "datetime_predictor": (
        lambda: _frame_with("d", pd.date_range("2020", periods=200)),
        {"feature_types": {"x": "continuous", "d": "categorical"}},
        TypeError,
        "d",
    ),
    "timedelta_predictor": (
        lambda: _frame_with("d", pd.to_timedelta(np.arange(200), unit="s")),
        {"feature_types": {"x": "continuous", "d": "continuous"}},
        TypeError,
        "d",
    ),
    "object_timestamp_predictor": (
        lambda: _frame_with(
            "d",
            pd.Series(
                list(pd.date_range("2020", periods=200)), dtype=object, index=np.arange(200)[::-1]
            ),
        ),
        {"feature_types": {"x": "continuous", "d": "categorical"}},
        TypeError,
        "d",
    ),
    "infinite_predictor": (
        lambda: _frame_with("x", np.r_[np.inf, np.linspace(0, 1, 199)]),
        {},
        ValueError,
        "x.*infinit",
    ),
    "infinite_categorical_float": (
        lambda: _frame_with("c", np.tile([1.0, np.inf], 100)),
        {"feature_types": {"x": "continuous", "c": "categorical"}},
        ValueError,
        "c.*infinit",
    ),
    "complex_predictor": (
        lambda: _frame_with("z", np.ones(200) * (1 + 1j)),
        {"feature_types": {"x": "continuous", "z": "continuous"}},
        TypeError,
        "z.*complex",
    ),
    "complex_object_category": (
        lambda: _frame_with(
            "z", pd.Series([1j, 2j] * 100, dtype=object, index=np.arange(200)[::-1])
        ),
        {"feature_types": {"x": "continuous", "z": "categorical"}},
        TypeError,
        "z.*complex",
    ),
    "string_declared_continuous": (
        lambda: _frame_with("s", ["1.5", "2.5"] * 100),
        {"feature_types": {"x": "continuous", "s": "continuous"}},
        TypeError,
        "s.*continuous",
    ),
    "bool_declared_continuous": (
        lambda: _frame_with("b", [True, False] * 100),
        {"feature_types": {"x": "continuous", "b": "continuous"}},
        TypeError,
        "b.*continuous",
    ),
}


@pytest.mark.parametrize("name", sorted(BAD_CASES))
def test_rejections_name_the_field(name):
    make, overrides, exc, pattern = BAD_CASES[name]
    df = make()
    before = df.copy(deep=True) if isinstance(df, pd.DataFrame) else None
    with pytest.raises(exc, match=pattern):
        prepare(df, **overrides)
    if before is not None:
        pd.testing.assert_frame_equal(df, before)


def test_boundary_sizes_are_accepted():
    df = pd.DataFrame({"x": np.linspace(0, 1, 100), "y": np.arange(100) % 2})
    assert prepare(df).frame.shape == (100, 1)
    many = pd.DataFrame({**{f"x{i}": np.arange(100.0) for i in range(50)}, "y": np.arange(100) % 2})
    data = prepare(many, feature_types={f"x{i}": "continuous" for i in range(50)})
    assert data.frame.shape == (100, 50)


def test_numpy_integer_scalars_are_accepted():
    data = prepare(valid_frame(), random_state=np.uint32(7), n_permutations=np.int64(19))
    assert data.config["random_state"] == 7
    assert type(data.config["random_state"]) is int


def test_unused_categories_do_not_trigger_mixed_type_rejection():
    df = valid_frame()
    df["c"] = pd.Categorical(np.tile(["a", "b"], 100), categories=["a", "b", 1])
    data = prepare(df, feature_types={"x": "continuous", "c": "categorical"})
    np.testing.assert_array_equal(data.values["c"][:2], [0, 1])


def test_observed_mixed_categories_are_still_rejected():
    df = valid_frame()
    df["c"] = pd.Categorical(np.tile(np.array(["a", 1], dtype=object), 100))
    with pytest.raises(TypeError, match="c.*mixed"):
        prepare(df, feature_types={"x": "continuous", "c": "categorical"})
