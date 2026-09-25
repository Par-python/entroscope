"""Mutual information, label-permutation inference and Holm adjustment (spec 6.3-6.5).

Estimators are scikit-learn's; this module only fixes their inputs, converts
nats to bits, and places each estimate in the context of a shuffled-label null.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .describe import observed_mask
from .validation import CATEGORICAL, CONTINUOUS, N_NEIGHBORS, PreparedData

TEST_COLUMNS = [
    "feature",
    "kind",
    "n",
    "mi_bits",
    "null_median_bits",
    "excess_bits",
    "p_value",
    "p_holm",
    "n_permutations",
    "assessment",
]
ABOVE_NULL = "above_permutation_null"
NOT_DISTINGUISHED = "not_distinguished_from_null"
_LN2 = math.log(2.0)
_SHUFFLE_STREAM = 0x5EED


def _sklearn():
    try:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.metrics import mutual_info_score
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise ImportError(
            "The feature audit needs scikit-learn: pip install 'entroscope[sklearn]'"
        ) from exc
    return mutual_info_classif, mutual_info_score


def seed_for(master_seed: int, feature: str, purpose: str) -> int:
    """Deterministic 32-bit seed from SHA-256; never Python's salted hash()."""
    payload = json.dumps([int(master_seed), str(feature), str(purpose)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def mi_bits(x, y, *, kind: str, seed: int) -> float:
    """Empirical MI in bits. ``kind`` describes x: 'categorical' or 'continuous'."""
    mutual_info_classif, mutual_info_score = _sklearn()
    x = np.asarray(x)
    y = np.asarray(y)
    if kind == CATEGORICAL:
        nats = mutual_info_score(x, y)
    elif kind == CONTINUOUS:
        nats = mutual_info_classif(
            np.asarray(x, dtype=float).reshape(-1, 1),
            y,
            discrete_features=False,
            n_neighbors=N_NEIGHBORS,
            copy=True,
            random_state=int(seed),
        )[0]
    else:
        raise ValueError(f"kind must be 'categorical' or 'continuous', got {kind!r}")
    return float(nats) / _LN2


def permutation_summary(
    x, y, *, kind: str, n_permutations: int, seed: int, estimator_seed=None
) -> Dict[str, float]:
    """Observed MI against B label permutations with x and estimator seed held fixed.

    ``seed`` drives the label shuffles; the estimator seed (default: ``seed``)
    is reused unchanged for the observed and every permuted score.
    """
    est_seed = int(seed if estimator_seed is None else estimator_seed)
    y = np.asarray(y)
    observed = mi_bits(x, y, kind=kind, seed=est_seed)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SHUFFLE_STREAM]))
    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        null[b] = mi_bits(x, rng.permutation(y), kind=kind, seed=est_seed)
    tol = 100 * np.finfo(float).eps * np.maximum(np.maximum(abs(observed), np.abs(null)), 1.0)
    exceed = int(np.count_nonzero(null >= observed - tol))
    median = float(np.median(null))
    return {
        "mi_bits": observed,
        "null_median_bits": median,
        "excess_bits": observed - median,
        "p_value": (1 + exceed) / (n_permutations + 1),
        "n_permutations": int(n_permutations),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise adjustment, returned in the original order."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted = np.minimum(1.0, np.maximum.accumulate(sorted_p * np.arange(len(sorted_p), 0, -1)))
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def resolution_warnings(n_tests: int, n_permutations: int) -> List[str]:
    """Warn when Holm's worst-case multiplier makes 1/(B+1) too coarse."""
    if n_tests and n_tests / (n_permutations + 1) > 0.05:
        return [
            f"With {n_tests} tests and B={n_permutations} permutations, the minimum "
            f"attainable raw p-value 1/(B+1) = {1 / (n_permutations + 1):.4g} times "
            f"m={n_tests} exceeds 0.05, so some associations cannot reach significance "
            "after Holm adjustment regardless of strength."
        ]
    return []


def association_tests(data: PreparedData, features: pd.DataFrame) -> pd.DataFrame:
    """Run every eligible value and missingness hypothesis; Holm-adjust jointly."""
    master = data.config["random_state"]
    n_perm = data.config["n_permutations"]
    alpha = data.config["alpha"]
    status = dict(zip(features["feature"], features["value_status"]))
    rows = []
    for name in data.frame.columns:
        observed = observed_mask(data, name)
        if status[name] == "eligible":
            summary = permutation_summary(
                data.values[name][observed],
                data.target[observed],
                kind=data.feature_types[name],
                n_permutations=n_perm,
                seed=seed_for(master, name, "permutation:value"),
                estimator_seed=seed_for(master, name, "estimator:value"),
            )
            rows.append({"feature": name, "kind": "value", "n": int(observed.sum()), **summary})
        n_missing = int((~observed).sum())
        if 0 < n_missing < len(observed):
            summary = permutation_summary(
                (~observed).astype(np.int64),
                data.target,
                kind=CATEGORICAL,
                n_permutations=n_perm,
                seed=seed_for(master, name, "permutation:missingness"),
                estimator_seed=seed_for(master, name, "estimator:missingness"),
            )
            rows.append({"feature": name, "kind": "missingness", "n": len(observed), **summary})

    tests = pd.DataFrame(
        rows, columns=[c for c in TEST_COLUMNS if c not in ("p_holm", "assessment")]
    )
    tests["p_holm"] = holm_adjust(tests["p_value"].to_numpy(dtype=float))
    tests["assessment"] = np.where(tests["p_holm"] <= alpha, ABOVE_NULL, NOT_DISTINGUISHED)
    if tests.empty:
        tests = tests.astype({"assessment": object})
    return tests[TEST_COLUMNS].reset_index(drop=True)
