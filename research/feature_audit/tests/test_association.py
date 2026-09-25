"""MI in bits, permutation inference and Holm adjustment (spec 6.3-6.5)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import mutual_info_classif

from research.feature_audit.prototype.association import (
    TEST_COLUMNS,
    association_tests,
    holm_adjust,
    mi_bits,
    permutation_summary,
    resolution_warnings,
    seed_for,
)
from research.feature_audit.prototype.describe import describe_features
from research.feature_audit.prototype.validation import validate_input


def test_discrete_information_in_bits():
    x = np.array([0, 0, 1, 1] * 50)
    independent = np.array([0, 1, 0, 1] * 50)
    assert mi_bits(x, x, kind="categorical", seed=0) == pytest.approx(1.0)
    assert mi_bits(x, independent, kind="categorical", seed=0) == pytest.approx(0.0)


def test_xor_blind_spot_is_preserved_not_hidden():
    x1 = np.array([0, 0, 1, 1] * 50)
    x2 = np.array([0, 1, 0, 1] * 50)
    y = x1 ^ x2
    assert mi_bits(x1, y, kind="categorical", seed=0) == pytest.approx(0.0)
    assert mi_bits(x2, y, kind="categorical", seed=0) == pytest.approx(0.0)


def test_holm_restores_order_and_is_monotonic():
    np.testing.assert_allclose(holm_adjust([0.03, 0.001, 0.04]), [0.06, 0.003, 0.06])


def test_holm_caps_at_one_and_handles_empty():
    np.testing.assert_allclose(holm_adjust([0.5, 0.9]), [1.0, 1.0])
    assert holm_adjust([]).shape == (0,)


def test_ties_do_not_create_zero_p_values():
    x = np.ones(100, dtype=int)
    y = np.tile([0, 1], 50)
    r = permutation_summary(x, y, kind="categorical", n_permutations=19, seed=0)
    assert r["mi_bits"] == 0
    assert r["p_value"] == 1
    assert r["excess_bits"] == 0


def test_perfect_copy_reaches_minimum_attainable_p_not_zero():
    y = np.tile([0, 1], 100)
    r = permutation_summary(y.copy(), y, kind="categorical", n_permutations=19, seed=3)
    assert r["mi_bits"] == pytest.approx(1.0)
    assert r["p_value"] == pytest.approx(1 / 20)
    assert r["n_permutations"] == 19


def test_negative_excess_is_not_clipped():
    x = np.array([0, 0, 1, 1] * 50)
    y = np.array([0, 1, 0, 1] * 50)
    r = permutation_summary(x, y, kind="categorical", n_permutations=99, seed=1)
    assert r["mi_bits"] == pytest.approx(0.0)
    assert r["null_median_bits"] > 0
    assert r["excess_bits"] < 0
    assert r["p_value"] == 1


def test_continuous_matches_installed_sklearn_in_bits():
    rng = np.random.default_rng(11)
    x = rng.normal(size=300)
    y = (np.abs(x) > 0.7).astype(int)
    expected = mutual_info_classif(
        x.reshape(-1, 1), y, discrete_features=False, n_neighbors=3, copy=True, random_state=5
    )[0]
    assert mi_bits(x, y, kind="continuous", seed=5) == pytest.approx(expected / math.log(2))


def test_categorical_label_renaming_invariance():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 4, 300)
    y = (x + rng.integers(0, 2, 300)) % 3
    renamed = np.array([7, 1, 9, 3])[x]
    assert mi_bits(renamed, y, kind="categorical", seed=0) == pytest.approx(
        mi_bits(x, y, kind="categorical", seed=0)
    )


def test_permutation_summary_is_deterministic_and_seed_sensitive():
    rng = np.random.default_rng(4)
    x = rng.normal(size=200)
    y = rng.integers(0, 2, 200)
    a = permutation_summary(x, y, kind="continuous", n_permutations=19, seed=10)
    b = permutation_summary(x, y, kind="continuous", n_permutations=19, seed=10)
    c = permutation_summary(x, y, kind="continuous", n_permutations=19, seed=11)
    assert a == b
    assert a["null_median_bits"] != c["null_median_bits"]


def test_seed_for_is_stable_distinct_and_in_range():
    s = seed_for(0, "age", "permutation:value")
    assert s == seed_for(0, "age", "permutation:value")
    assert 0 <= s < 2**32
    others = {
        seed_for(0, "age", "subsample"),
        seed_for(0, "age", "estimator:value"),
        seed_for(0, "age2", "permutation:value"),
        seed_for(1, "age", "permutation:value"),
    }
    assert s not in others and len(others) == 4
    # Hard-coded expectation guards against accidental use of Python hash().
    import hashlib
    import json

    digest = hashlib.sha256(json.dumps([0, "age", "permutation:value"]).encode()).digest()
    assert s == int.from_bytes(digest[:4], "big")


def test_resolution_warning_when_family_is_large_relative_to_b():
    assert resolution_warnings(10, 1999) == []
    msgs = resolution_warnings(10, 19)
    assert len(msgs) == 1 and "1/(B+1)" in msgs[0]


def _missingness_frame():
    y = np.tile([0, 1], 100)
    rng = np.random.default_rng(8)
    cls0, cls1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    a = pd.Series(np.where(np.arange(200) % 4 < 2, "p", "q"), dtype=object)
    a[np.r_[cls0[:50], cls1[:50]]] = None
    b = pd.Series(rng.normal(size=200))
    b[np.r_[cls0[:20], cls1[:80]]] = np.nan
    return pd.DataFrame({"a": a, "b": b, "y": y})


def _run(df, types, seed=0, b=19):
    data = validate_input(
        df,
        target="y",
        feature_types=types,
        assume_iid=True,
        n_permutations=b,
        n_subsamples=10,
        random_state=seed,
    )
    return association_tests(data, describe_features(data))


def test_missingness_is_a_separate_hypothesis_in_the_same_family():
    tests = _run(_missingness_frame(), {"a": "categorical", "b": "continuous"})
    assert list(tests.columns) == TEST_COLUMNS
    keyed = tests.set_index(["feature", "kind"])
    assert keyed.loc[("a", "value"), "n"] == 100
    assert keyed.loc[("b", "value"), "n"] == 100
    assert keyed.loc[("a", "missingness"), "n"] == 200
    assert keyed.loc[("b", "missingness"), "n"] == 200
    assert keyed.loc[("a", "missingness"), "mi_bits"] == pytest.approx(0.0, abs=1e-12)
    # Hand-computed contingency MI: missing (20 class-0, 80 class-1), observed (80, 20).
    joint = np.array([[20, 80], [80, 20]]) / 200.0
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    expected = float((joint * np.log2(joint / (px * py))).sum())
    assert keyed.loc[("b", "missingness"), "mi_bits"] == pytest.approx(expected)
    assert len(tests) == 4
    np.testing.assert_allclose(tests["p_holm"], holm_adjust(tests["p_value"].to_numpy()))
    assert set(tests["assessment"]) <= {"above_permutation_null", "not_distinguished_from_null"}


def test_no_target_selected_filtering_and_input_order():
    tests = _run(_missingness_frame(), {"a": "categorical", "b": "continuous"})
    assert list(zip(tests["feature"], tests["kind"])) == [
        ("a", "value"),
        ("a", "missingness"),
        ("b", "value"),
        ("b", "missingness"),
    ]


def test_excluded_values_still_allow_missingness_tests():
    ids = pd.Series([f"u{i}" for i in range(200)], dtype=object)
    ids[:40] = None
    df = pd.DataFrame({"id": ids, "y": np.tile([0, 1], 100)})
    tests = _run(df, {"id": "categorical"})
    assert list(tests["kind"]) == ["missingness"]


def test_column_order_does_not_change_raw_scores():
    df = _missingness_frame()
    types = {"a": "categorical", "b": "continuous"}
    first = _run(df, types).set_index(["feature", "kind"])
    second = _run(df[["b", "a", "y"]], types).set_index(["feature", "kind"])
    for col in ["n", "mi_bits", "null_median_bits", "excess_bits", "p_value"]:
        pd.testing.assert_series_equal(first[col], second.loc[first.index, col])


def test_empty_family_returns_complete_schema():
    df = pd.DataFrame({"c": ["k"] * 200, "y": np.tile([0, 1], 100)})
    tests = _run(df, {"c": "categorical"})
    assert tests.empty
    assert list(tests.columns) == TEST_COLUMNS


def test_strong_signal_is_flagged_above_null():
    y = np.tile([0, 1], 100)
    df = pd.DataFrame({"c": np.where(y == 1, "u", "v"), "y": y})
    tests = _run(df, {"c": "categorical"}, b=199)
    row = tests.iloc[0]
    assert row["p_value"] == pytest.approx(1 / 200)
    assert row["assessment"] == "above_permutation_null"
