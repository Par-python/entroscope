"""Validate audit input and build an internal, never-exported representation.

Nothing here performs inference. The prepared data holds raw values and must
never be retained by the report object.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
FEATURE_TYPES = (CONTINUOUS, CATEGORICAL)

MIN_ROWS = 100
MAX_ROWS = 5000
MIN_PREDICTORS = 1
MAX_PREDICTORS = 50
MIN_CLASSES = 2
MAX_CLASSES = 20
MIN_CLASS_COUNT = 10
MIN_PERMUTATIONS = 19
MIN_SUBSAMPLES = 10
DEFAULT_PERMUTATIONS = 1999
SUBSAMPLE_FRACTION = 0.8
N_NEIGHBORS = 3
ALPHA = 0.05

IID_MESSAGE = (
    "assume_iid=True is required. This audit assumes independent rows: repeated "
    "measurements of the same subject, grouped records, or time-series observations "
    "need dependence-aware methods that this prototype does not support. Passing "
    "assume_iid=True is an acknowledgment, not an automatic independence test."
)


@dataclass
class PreparedData:
    """Internal validated input. Never stored on, or exported by, a report."""

    frame: pd.DataFrame
    target: np.ndarray
    feature_types: Dict[str, str]
    config: Dict[str, Any]
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    n_classes: int = 0
    warnings: List[str] = field(default_factory=list)


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _check_int(name, value, low, high=None):
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer (not bool), got {type(value).__name__}")
    value = int(value)
    if value < low or (high is not None and value >= high):
        bound = f">= {low}" if high is None else f"in [{low}, {high})"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


def _is_temporal_dtype(dtype) -> bool:
    return (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    )


_TEMPORAL_SCALARS = (
    pd.Timestamp,
    pd.Timedelta,
    pd.Period,
    np.datetime64,
    np.timedelta64,
)


def _scalar_kind(value) -> str:
    """Classify a non-missing object scalar; raise for unsupported values."""
    import datetime as _dt

    if isinstance(value, (_dt.date, _dt.time, _dt.timedelta) + _TEMPORAL_SCALARS):
        raise TypeError("datetime/timedelta")
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError("complex")
    if isinstance(value, (tuple, frozenset)) or not isinstance(value, Hashable):
        raise TypeError("nested or unhashable")
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "integer"
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError("infinite")
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bytes):
        return "bytes"
    raise TypeError(f"unsupported scalar type {type(value).__name__}")


def _object_kinds(series: pd.Series, name: str) -> set:
    kinds = set()
    for value in series[series.notna()].tolist():
        try:
            kinds.add(_scalar_kind(value))
        except TypeError as exc:
            raise TypeError(
                f"column {name!r}: values must be simple scalars; found {exc}. "
                "Normalize the column explicitly before auditing."
            ) from None
        except ValueError:
            raise ValueError(f"column {name!r} contains infinite values") from None
    return kinds


def _check_values(series: pd.Series, name: str) -> None:
    """Reject temporal, complex, infinite, nested and mixed-type values."""
    dtype = series.dtype
    if _is_temporal_dtype(dtype):
        raise TypeError(f"column {name!r} has a datetime/timedelta dtype, which is unsupported")
    if pd.api.types.is_complex_dtype(dtype):
        raise TypeError(f"column {name!r} contains complex values, which are unsupported")
    if isinstance(dtype, pd.CategoricalDtype):
        # Only categories that occur matter; unused categories are never audited.
        used = series.cat.remove_unused_categories().cat.categories
        _check_values(pd.Series(list(used), dtype=object), name)
        return
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        numeric = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric).any():
            raise ValueError(f"column {name!r} contains infinite values")
        return
    if pd.api.types.is_object_dtype(dtype):
        kinds = _object_kinds(series, name)
        if len(kinds) > 1:
            raise TypeError(
                f"column {name!r} has mixed scalar types ({', '.join(sorted(kinds))}); "
                "normalize it explicitly before auditing"
            )


def _encode_categories(series: pd.Series) -> np.ndarray:
    """First-seen integer codes; missing values become -1. No string conversion."""
    codes, _ = pd.factorize(series, sort=False, use_na_sentinel=True)
    if isinstance(series.dtype, pd.CategoricalDtype):
        # factorize on a Categorical may follow category order; re-map to first-seen.
        seen = {}
        remapped = np.full(len(codes), -1, dtype=np.int64)
        for i, code in enumerate(codes):
            if code >= 0:
                remapped[i] = seen.setdefault(code, len(seen))
        return remapped
    return np.asarray(codes, dtype=np.int64)


def _continuous_values(series: pd.Series, name: str) -> np.ndarray:
    dtype = series.dtype
    if (
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        raise TypeError(
            f"column {name!r} is declared continuous but has non-numeric dtype {dtype}; "
            "convert it explicitly or declare it categorical"
        )
    return series.to_numpy(dtype=float, na_value=np.nan)


def _encode_target(series: pd.Series, target: str) -> np.ndarray:
    dtype = series.dtype
    if series.isna().any():
        raise ValueError(f"target {target!r} contains missing values")
    if _is_temporal_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        raise TypeError(f"target {target!r} must hold class labels, not {dtype} values")
    if pd.api.types.is_float_dtype(dtype):
        numeric = series.to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
            raise ValueError(
                f"target {target!r} has nonintegral numeric values; this audit supports "
                "classification labels only"
            )
    elif pd.api.types.is_object_dtype(dtype):
        kinds = _object_kinds(series, f"target {target}")
        if "float" in kinds:
            numeric = np.array([float(v) for v in series.tolist()])
            if not np.all(numeric == np.round(numeric)):
                raise ValueError(f"target {target!r} has nonintegral numeric values")
        if len(kinds) > 1:
            raise TypeError(f"target {target!r} has mixed scalar label types")
    codes = _encode_categories(series)
    counts = np.bincount(codes)
    if len(counts) < MIN_CLASSES:
        raise ValueError(f"target {target!r} must have at least two classes")
    if len(counts) > MAX_CLASSES:
        raise ValueError(f"target {target!r} has {len(counts)} classes; at most 20 are supported")
    if counts.min() < MIN_CLASS_COUNT:
        raise ValueError(
            f"target {target!r}: every class needs at least 10 rows (smallest has {counts.min()})"
        )
    return codes


def validate_input(
    df,
    *,
    target,
    feature_types,
    assume_iid,
    n_permutations,
    n_subsamples,
    random_state,
) -> PreparedData:
    """Validate the audit contract and return a deep-copied internal representation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if assume_iid is not True:
        raise ValueError(IID_MESSAGE)
    random_state = _check_int("random_state", random_state, 0, 2**32)
    n_permutations = _check_int("n_permutations", n_permutations, MIN_PERMUTATIONS)
    n_subsamples = _check_int("n_subsamples", n_subsamples, MIN_SUBSAMPLES)

    columns = list(df.columns)
    if not all(isinstance(c, str) for c in columns):
        raise TypeError("column names must all be strings")
    if len(set(columns)) != len(columns):
        raise ValueError("column names must be unique")
    if not df.index.is_unique:
        raise ValueError("the row index must be unique")
    if not isinstance(target, str) or target not in columns:
        raise ValueError(f"target {target!r} is not a column of df")
    if not isinstance(feature_types, Mapping):
        raise TypeError("feature_types must be a mapping of predictor name to type")

    predictors = [c for c in columns if c != target]
    if target in feature_types:
        raise ValueError(f"feature_types must not contain an entry for the target {target!r}")
    missing = [c for c in predictors if c not in feature_types]
    extra = [c for c in feature_types if c not in predictors]
    if missing:
        raise ValueError(f"feature_types is missing entries for: {missing}")
    if extra:
        raise ValueError(f"feature_types has entries for unknown columns: {extra}")
    for name in predictors:
        if feature_types[name] not in FEATURE_TYPES:
            raise ValueError(
                f"feature_types[{name!r}] must be 'continuous' or 'categorical', "
                f"got {feature_types[name]!r}"
            )

    n_rows = len(df)
    if not MIN_ROWS <= n_rows <= MAX_ROWS:
        raise ValueError(
            f"df has {n_rows} rows; between {MIN_ROWS} and {MAX_ROWS} rows are supported "
            "(larger inputs are rejected, not silently sampled)"
        )
    if not MIN_PREDICTORS <= len(predictors) <= MAX_PREDICTORS:
        raise ValueError(
            f"df has {len(predictors)} predictor columns; between {MIN_PREDICTORS} and "
            f"{MAX_PREDICTORS} are supported"
        )

    target_codes = _encode_target(df[target], target)

    values = {}
    for name in predictors:
        series = df[name]
        _check_values(series, name)
        if feature_types[name] == CONTINUOUS:
            values[name] = _continuous_values(series, name)
        else:
            values[name] = _encode_categories(series)

    config = {
        "n_permutations": n_permutations,
        "n_subsamples": n_subsamples,
        "random_state": random_state,
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "n_neighbors": N_NEIGHBORS,
        "alpha": ALPHA,
        "min_rows": MIN_ROWS,
        "max_rows": MAX_ROWS,
        "max_predictors": MAX_PREDICTORS,
        "max_classes": MAX_CLASSES,
        "min_class_count": MIN_CLASS_COUNT,
    }
    warnings = []
    if n_permutations < DEFAULT_PERMUTATIONS:
        warnings.append(
            f"Reduced permutation count ({n_permutations}): the smallest attainable raw "
            f"p-value is 1/(B+1) = {1 / (n_permutations + 1):.4g}, which limits resolution "
            "near alpha=0.05 and after Holm adjustment. The default is 1999."
        )

    return PreparedData(
        frame=df[predictors].copy(deep=True),
        target=target_codes,
        feature_types=dict(feature_types),
        config=config,
        values=values,
        n_classes=int(target_codes.max()) + 1,
        warnings=warnings,
    )
