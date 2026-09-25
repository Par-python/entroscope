"""Per-predictor metadata and conservative eligibility policies (spec 6.1-6.2).

The exclusion rules are fixed product policies, not theorems and not leakage
verdicts. Every predictor gets exactly one row, in input order.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .validation import CATEGORICAL, PreparedData

FEATURE_COLUMNS = [
    "feature",
    "declared_type",
    "n_total",
    "n_observed",
    "missing_fraction",
    "n_unique_observed",
    "dominant_fraction",
    "entropy_bits",
    "value_status",
    "warning_codes",
    "subsample_mi_median",
    "subsample_mi_p10",
    "subsample_mi_p90",
    "subsample_valid_count",
]

# Exclusion precedence from spec section 7; the first applicable code wins.
STATUS_PRECEDENCE = [
    "all_missing",
    "constant",
    "insufficient_support",
    "high_cardinality",
    "rare_categories",
    "ambiguous_discrete",
    "possible_identifier",
]

MIN_ANALYSIS_ROWS = 100
MIN_ROWS_PER_CLASS = 10
HIGH_CARDINALITY_FRACTION = 0.20
HIGH_CARDINALITY_COUNT = 100
RARE_CATEGORY_COUNT = 5
MIN_CONTINUOUS_DISTINCT = 10


def observed_mask(data: PreparedData, name: str) -> np.ndarray:
    values = data.values[name]
    if data.feature_types[name] == CATEGORICAL:
        return values >= 0
    return ~np.isnan(values)


def has_class_support(y: np.ndarray, n_classes: int) -> bool:
    """At least 100 rows and at least 10 rows of every original target class."""
    if len(y) < MIN_ANALYSIS_ROWS:
        return False
    counts = np.bincount(y, minlength=n_classes)
    return bool(counts.min() >= MIN_ROWS_PER_CLASS)


def _conditions(data: PreparedData, name: str, observed: np.ndarray, counts: np.ndarray):
    n_observed = int(observed.sum())
    n_unique = len(counts)
    kind = data.feature_types[name]
    codes = []
    if n_observed == 0:
        return ["all_missing"]
    if n_unique == 1:
        codes.append("constant")
    if not has_class_support(data.target[observed], data.n_classes):
        codes.append("insufficient_support")
    if kind == CATEGORICAL:
        if n_unique / n_observed >= HIGH_CARDINALITY_FRACTION or n_unique > HIGH_CARDINALITY_COUNT:
            codes.append("high_cardinality")
        if counts.min() < RARE_CATEGORY_COUNT:
            codes.append("rare_categories")
    else:
        if n_unique < MIN_CONTINUOUS_DISTINCT:
            codes.append("ambiguous_discrete")
        x = data.values[name][observed]
        if n_unique == n_observed and n_observed > 1:
            diffs = np.diff(x)
            if np.all(diffs > 0) or np.all(diffs < 0):
                codes.append("possible_identifier")
    return codes


def describe_features(data: PreparedData) -> pd.DataFrame:
    """Return one metadata row per predictor with stability fields left null."""
    rows = []
    for name in data.frame.columns:
        values = data.values[name]
        observed = observed_mask(data, name)
        n_total = len(values)
        n_observed = int(observed.sum())
        if data.feature_types[name] == CATEGORICAL:
            counts = np.bincount(values[observed]) if n_observed else np.array([], dtype=int)
            counts = counts[counts > 0]
        else:
            _, counts = np.unique(values[observed], return_counts=True)

        entropy_bits = np.nan
        if data.feature_types[name] == CATEGORICAL and n_observed:
            p = counts / counts.sum()
            entropy_bits = float(-(p * np.log2(p)).sum()) + 0.0

        warning_codes: List[str] = _conditions(data, name, observed, counts)
        status = next((c for c in STATUS_PRECEDENCE if c in warning_codes), "eligible")
        if n_observed == 0:
            warning_codes.append("missingness_untestable")

        rows.append(
            {
                "feature": name,
                "declared_type": data.feature_types[name],
                "n_total": n_total,
                "n_observed": n_observed,
                "missing_fraction": (n_total - n_observed) / n_total,
                "n_unique_observed": len(counts),
                "dominant_fraction": float(counts.max() / n_observed) if n_observed else np.nan,
                "entropy_bits": entropy_bits,
                "value_status": status,
                "warning_codes": warning_codes,
                "subsample_mi_median": np.nan,
                "subsample_mi_p10": np.nan,
                "subsample_mi_p90": np.nan,
                "subsample_valid_count": 0,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
