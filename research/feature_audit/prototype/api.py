"""Compose the audit steps. No statistical algorithms live here."""

from __future__ import annotations

import platform

import numpy as np
import pandas as pd
import scipy

from .association import association_tests, resolution_warnings
from .describe import describe_features
from .duplicates import exact_duplicates
from .report import LIMITATIONS, AuditReport
from .stability import MIN_VALID_SUBSAMPLES, STABILITY_COLUMNS, stability_summary
from .validation import validate_input

SCHEMA_VERSION = "0.1"


def _versions():
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:  # pragma: no cover
        sklearn_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn_version,
        "prototype_schema": SCHEMA_VERSION,
    }


def audit(
    df,
    *,
    target,
    feature_types,
    assume_iid=False,
    n_permutations=1999,
    n_subsamples=30,
    random_state=0,
) -> AuditReport:
    """Audit individual feature-target associations in a classification training frame.

    Returns an :class:`AuditReport` holding aggregate results only. Requires
    explicit feature types and ``assume_iid=True``; see the README for scope.
    """
    data = validate_input(
        df,
        target=target,
        feature_types=feature_types,
        assume_iid=assume_iid,
        n_permutations=n_permutations,
        n_subsamples=n_subsamples,
        random_state=random_state,
    )
    features = describe_features(data)
    tests = association_tests(data, features)
    stability = stability_summary(data, features)
    features[STABILITY_COLUMNS] = stability.loc[features["feature"], STABILITY_COLUMNS].to_numpy(
        dtype=object
    )
    features["subsample_valid_count"] = features["subsample_valid_count"].astype(int)
    for col in STABILITY_COLUMNS[:3]:
        features[col] = features[col].astype(float)
    short = (features["value_status"] == "eligible") & (
        features["subsample_valid_count"] < MIN_VALID_SUBSAMPLES
    )
    features["warning_codes"] = [
        list(codes) + (["insufficient_subsample_support"] if flag else [])
        for codes, flag in zip(features["warning_codes"], short)
    ]
    duplicates = exact_duplicates(data)

    n_tests = len(tests)
    warnings = list(data.warnings)
    warnings += resolution_warnings(n_tests, data.config["n_permutations"])
    if (tests["kind"] == "missingness").any():
        warnings.append(
            "Some features have missing values. Their value tests use only observed rows "
            "and their missingness is tested separately on all rows; do not compare "
            "these scores directly with fully observed features."
        )
    if short.any():
        warnings.append(
            "Some eligible features had fewer than 10 subsamples meeting the support "
            "rules (insufficient_subsample_support); their stability summary is left "
            "empty and their value status is unchanged."
        )
    if (features["value_status"] == "possible_identifier").any():
        warnings.append(
            "possible_identifier marks unique values that are strictly monotone in row "
            "order. This may be a row ID, or a legitimate measurement stored in sorted "
            "order; review it manually."
        )
    warnings.append(
        "Feature and target names appear in this report and may themselves be "
        "sensitive; review before sharing."
    )

    counts = np.bincount(data.target)
    p = counts / counts.sum()
    configuration = dict(data.config)
    configuration["minimum_attainable_p"] = 1.0 / (data.config["n_permutations"] + 1)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "target_name": target,
        "n_rows": int(len(data.target)),
        "target_class_count": int(data.n_classes),
        "target_entropy_bits": float(-(p * np.log2(p)).sum()),
        "n_tests": int(n_tests),
        "configuration": configuration,
        "versions": _versions(),
        "assume_iid": True,
        "limitations": list(LIMITATIONS),
    }
    return AuditReport(
        features=features.reset_index(drop=True),
        tests=tests.reset_index(drop=True),
        duplicates=duplicates,
        warnings=warnings,
        metadata=metadata,
    )
