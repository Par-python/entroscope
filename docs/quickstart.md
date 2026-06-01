# Quickstart

## Install

```bash
pip install entropix
```

## Compute entropy

```python
import pandas as pd
from entropix import shannon, permutation, spectral

s = pd.Series([10, 20, 15, 80, 90, 85, 88, 92])

shannon.compute(s)                  # single value
shannon.rolling(s, window=20)       # rolling Series (index preserved)
shannon.delta(s, window=20)         # rate of change
shannon.normalized(s)               # 0-1 scaled
fig = shannon.plot(s, window=20)    # matplotlib Figure
```

## Compare measures

```python
from entropix import plot

fig = plot.compare(s, measures=["shannon", "permutation", "spectral"], window=20)
fig = plot.dashboard(s, window=20)
```

Every measure follows the same contract:

| Method        | Returns                              |
| ------------- | ------------------------------------ |
| `compute`     | `float`                              |
| `rolling`     | Series/ndarray, same length          |
| `delta`       | Series/ndarray (first difference)    |
| `normalized`  | `float` in [0, 1] (where defined)    |
| `plot`        | `matplotlib.figure.Figure`           |
