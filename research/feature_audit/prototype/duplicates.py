"""Exact, type-sensitive duplicated predictor columns (spec section 7).

Only verified value-by-value equality is reported; nothing here claims that
non-duplicated columns are independent or that duplicates should be removed.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import pandas as pd

from .validation import MAX_PREDICTORS, PreparedData

MAX_PAIRS = MAX_PREDICTORS * (MAX_PREDICTORS - 1) // 2  # 1,225


def _scalar_types(series: pd.Series) -> frozenset:
    if series.dtype != object:
        return frozenset()
    return frozenset(type(v) for v in series[series.notna()].tolist())


def exact_duplicates(data: PreparedData) -> List[Tuple[str, str]]:
    """Return (first, second) name pairs whose values and missingness match exactly."""
    frame = data.frame
    names = list(frame.columns)
    pairs = list(combinations(names, 2))
    if len(pairs) > MAX_PAIRS:
        raise ValueError(f"at most {MAX_PAIRS} predictor pairs can be compared")
    found = []
    for a, b in pairs:
        if data.feature_types[a] != data.feature_types[b]:
            continue
        if frame[a].dtype != frame[b].dtype:
            continue
        if not frame[a].equals(frame[b]):
            continue
        if _scalar_types(frame[a]) != _scalar_types(frame[b]):
            continue
        found.append((a, b))
    return found
