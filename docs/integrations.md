# Integrations

## polars

Every function that accepts a pandas Series also accepts a polars Series.
Rolling and delta results come back as a polars Series with the same name:

```python
import polars as pl
from entroscope import shannon, permutation

s = pl.Series("interest", [10, 20, 15, 80, 90, 85, 88, 92] * 10)

shannon.compute(s)                  # float
permutation.rolling(s, window=20)   # polars Series named "interest"
```

Install with `pip install "entroscope[polars]"` (or just have polars installed).

## scikit-learn

`EntropyFeatures` turns time-series windows into entropy features, so entropy
drops straight into a scikit-learn pipeline. Each row of `X` is one window;
each output column is one measure.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from entroscope.features import EntropyFeatures

# Cut a long series into overlapping windows of 64 samples, 16 apart.
windows = np.lib.stride_tricks.sliding_window_view(signal, 64)[::16]

model = make_pipeline(
    EntropyFeatures(measures=("spectral", "permutation", "sample")),
    RandomForestClassifier(),
)
model.fit(windows, labels)
```

Pass per-measure settings with `params`:

```python
EntropyFeatures(
    measures=("sample", "permutation"),
    params={"sample": {"m": 2, "r": 0.15}, "permutation": {"order": 4}},
)
```

For named DataFrame output, use scikit-learn's output API:

```python
EntropyFeatures().set_output(transform="pandas").fit_transform(windows)
# columns: shannon_entropy, permutation_entropy, spectral_entropy, ...
```

Available measures: `shannon`, `permutation`, `spectral`, `sample`,
`approximate`, `differential`. The transformer is stateless, so `fit` only
validates the input. Install with `pip install "entroscope[sklearn]"`.
