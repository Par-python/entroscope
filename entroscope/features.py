"""scikit-learn integration — entropy values as features for ML pipelines.

Each row of ``X`` is one time-series window; each output column is one entropy
measure of that window::

    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from entroscope.features import EntropyFeatures

    model = make_pipeline(EntropyFeatures(), RandomForestClassifier())
    model.fit(windows, labels)          # windows: (n_windows, window_length)

Requires scikit-learn (``pip install "entroscope[sklearn]"``). It is imported
here rather than in ``entroscope/__init__.py`` so the core library stays free
of the dependency.
"""

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_array, check_is_fitted
except ImportError as exc:  # pragma: no cover - exercised only without sklearn
    raise ImportError(
        "entroscope.features needs scikit-learn: pip install 'entroscope[sklearn]'"
    ) from exc

from . import approximate, differential, permutation, sample, shannon, spectral

_MEASURES = {
    "shannon": shannon.compute,
    "permutation": permutation.compute,
    "spectral": spectral.compute,
    "sample": sample.compute,
    "approximate": approximate.compute,
    "differential": differential.compute,
}

DEFAULT_MEASURES = ("shannon", "permutation", "spectral", "sample", "approximate")


class EntropyFeatures(TransformerMixin, BaseEstimator):
    """Transform time-series windows into one entropy feature per measure.

    Parameters
    ----------
    measures : sequence of str
        Measures to compute, in output-column order. Any of ``shannon``,
        ``permutation``, ``spectral``, ``sample``, ``approximate``,
        ``differential``.
    params : dict, optional
        Per-measure keyword arguments, e.g. ``{"sample": {"m": 2, "r": 0.15}}``.

    Stateless: ``fit`` only validates input and records its width. Supports
    ``set_output(transform="pandas")`` for DataFrame output with named columns.
    """

    def __init__(self, measures=DEFAULT_MEASURES, params=None):
        self.measures = measures
        self.params = params

    def _check_config(self):
        unknown = [m for m in self.measures if m not in _MEASURES]
        if unknown or not len(self.measures):
            raise ValueError(
                f"unknown or empty measures {unknown or list(self.measures)!r}; "
                f"choose from {sorted(_MEASURES)}"
            )
        extra = set(self.params or {}) - set(self.measures)
        if extra:
            raise ValueError(f"params given for measures not in `measures`: {sorted(extra)}")

    def fit(self, X, y=None):
        self._check_config()
        X = check_array(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but EntropyFeatures is expecting "
                f"{self.n_features_in_} features as input."
            )
        params = self.params or {}
        out = np.empty((X.shape[0], len(self.measures)))
        for j, name in enumerate(self.measures):
            fn, kwargs = _MEASURES[name], params.get(name, {})
            out[:, j] = [fn(row, **kwargs) for row in X]
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.asarray([f"{name}_entropy" for name in self.measures], dtype=object)
