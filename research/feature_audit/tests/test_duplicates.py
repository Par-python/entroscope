"""Exact, type-sensitive duplicate detection (spec section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.feature_audit.prototype.duplicates import exact_duplicates
from research.feature_audit.prototype.validation import validate_input


def prepare(df, types):
    return validate_input(
        df,
        target="y",
        feature_types=types,
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )


def test_duplicate_requires_same_missingness_and_dtype():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan] * 100,
            "b": [1.0, np.nan] * 100,
            "c": [np.nan, 1.0] * 100,
            "y": [0, 1] * 100,
        }
    )
    data = validate_input(
        df,
        target="y",
        feature_types={k: "continuous" for k in ("a", "b", "c")},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )
    assert exact_duplicates(data) == [("a", "b")]


def test_declared_type_and_dtype_must_match():
    base = np.tile([1, 2, 3, 4], 50)
    df = pd.DataFrame(
        {"i": base, "f": base.astype(float), "i_cat": base, "i2": base.copy(), "y": [0, 1] * 100}
    )
    types = {"i": "continuous", "f": "continuous", "i_cat": "categorical", "i2": "continuous"}
    assert exact_duplicates(prepare(df, types)) == [("i", "i2")]


def test_object_scalar_type_matters():
    df = pd.DataFrame(
        {
            "ints": pd.Series([1, 2] * 100, dtype=object),
            "floats": pd.Series([1.0, 2.0] * 100, dtype=object),
            "y": [0, 1] * 100,
        }
    )
    types = {"ints": "categorical", "floats": "categorical"}
    assert exact_duplicates(prepare(df, types)) == []


def test_all_pairs_reported_in_input_order():
    v = ["x", "y"] * 100
    df = pd.DataFrame({"p": v, "q": ["z"] * 200, "r": v, "s": v, "y": [0, 1] * 100})
    types = {k: "categorical" for k in "pqrs"}
    assert exact_duplicates(prepare(df, types)) == [("p", "r"), ("p", "s"), ("r", "s")]


def test_relabelled_categories_are_not_exact_duplicates():
    df = pd.DataFrame({"p": ["x", "y"] * 100, "q": ["u", "v"] * 100, "y": [0, 1] * 100})
    assert exact_duplicates(prepare(df, {"p": "categorical", "q": "categorical"})) == []
