# Quickstart

## Install

```bash
pip install entroscope
```

## Compute entropy

A series that is pure noise for 200 steps, then turns into a clean 20-step cycle:

```python
import numpy as np
import pandas as pd
from entroscope import shannon, spectral

rng = np.random.default_rng(0)
t = np.arange(400)
noise = rng.normal(size=400)
s = pd.Series(np.where(t < 200, noise, np.sin(2 * np.pi * t / 20) + 0.2 * noise))

spectral.compute(s[:200])              # -> 6.05 bits: noise spreads power over every frequency
spectral.compute(s[200:])              # -> 0.88 bits: the cycle concentrates it
spectral.normalized(s[200:])           # -> 0.13 on a 0-1 scale
roll = spectral.rolling(s, window=40)  # -> Series, same index, NaN for the first 39 steps
spectral.delta(s, window=40)           # -> step-to-step change in the rolling entropy
fig = spectral.plot(s, window=40)      # -> matplotlib Figure
shannon.compute(s)                     # every measure has the same call shape
```

## Compare measures

```python
from entroscope import plot

fig = plot.compare(s, measures=["shannon", "permutation", "spectral"], window=40)
fig = plot.dashboard(s, window=40)
fig = plot.drop_events(s, measure="spectral", window=40, threshold=0.5)
```

Every measure follows the same contract:

| Method        | Returns                                                     |
| ------------- | ----------------------------------------------------------- |
| `compute`     | `float`                                                     |
| `rolling`     | same type and length as the input, NaN during warm-up       |
| `delta`       | first difference of `rolling`                               |
| `normalized`  | `float` in [0, 1] (shannon, permutation, spectral)          |
| `plot`        | `matplotlib.figure.Figure`                                  |

Inputs can be a pandas Series (index kept), a polars Series (name kept) or a numpy
array. Any NaN or ±inf in a window makes that window's result NaN.
