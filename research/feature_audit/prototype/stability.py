"""Stratified subsampling sensitivity of MI estimates (spec 6.5).

Draws are WITHOUT replacement: repeated points would distort the kNN estimator.
The p10/p90 summaries describe sensitivity to resampling on this dataset; they
are not confidence intervals for a population parameter.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .association import mi_bits, seed_for
from .describe import has_class_support, observed_mask
from .validation import PreparedData

STABILITY_COLUMNS = [
    "subsample_mi_median",
    "subsample_mi_p10",
    "subsample_mi_p90",
    "subsample_valid_count",
]
MIN_VALID_SUBSAMPLES = 10


def stratified_subsamples(y, *, repeats: int, seed: int, fraction: float = 0.8) -> List[np.ndarray]:
    """Positions of ``repeats`` stratified draws of floor(fraction*n_class) per class."""
    y = np.asarray(y)
    classes = np.unique(y)
    streams = np.random.SeedSequence(int(seed)).spawn(repeats)
    draws = []
    for stream in streams:
        rng = np.random.default_rng(stream)
        parts = []
        for c in classes:
            members = np.flatnonzero(y == c)
            size = int(np.floor(fraction * len(members)))
            parts.append(rng.choice(members, size=size, replace=False))
        draws.append(np.sort(np.concatenate(parts)))
    return draws


def stability_summary(data: PreparedData, features: pd.DataFrame) -> pd.DataFrame:
    """Median and 10th/90th percentiles of subsample MI for eligible value hypotheses."""
    master = data.config["random_state"]
    repeats = data.config["n_subsamples"]
    fraction = data.config["subsample_fraction"]
    status = dict(zip(features["feature"], features["value_status"]))
    records = {}
    for name in data.frame.columns:
        record = {c: np.nan for c in STABILITY_COLUMNS}
        record["subsample_valid_count"] = 0
        if status[name] == "eligible":
            observed = observed_mask(data, name)
            x = data.values[name][observed]
            y = data.target[observed]
            est_seed = seed_for(master, name, "estimator:value")
            draws = stratified_subsamples(
                y, repeats=repeats, seed=seed_for(master, name, "subsample"), fraction=fraction
            )
            scores = [
                mi_bits(x[ids], y[ids], kind=data.feature_types[name], seed=est_seed)
                for ids in draws
                if has_class_support(y[ids], data.n_classes)
            ]
            record["subsample_valid_count"] = len(scores)
            if len(scores) >= MIN_VALID_SUBSAMPLES:
                p10, median, p90 = np.quantile(scores, [0.1, 0.5, 0.9])
                record.update(
                    subsample_mi_median=float(median),
                    subsample_mi_p10=float(p10),
                    subsample_mi_p90=float(p90),
                )
        records[name] = record
    out = pd.DataFrame.from_dict(records, orient="index", columns=STABILITY_COLUMNS)
    out["subsample_valid_count"] = out["subsample_valid_count"].astype(int)
    return out
