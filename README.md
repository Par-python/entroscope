# entropix

[![CI](https://github.com/entropix/entropix/actions/workflows/ci.yml/badge.svg)](https://github.com/entropix/entropix/actions/workflows/ci.yml)

**The definitive entropy toolkit for time series data.**

`pip install entropix` and get every entropy measure you'd ever need, with one
consistent interface that works directly on pandas Series and numpy arrays.

Born from [NextOnMenu](https://nextonmenu.com), where Shannon entropy of food-trend
search interest had to be computed by hand. entropix makes that a one-liner.

## Install

```bash
pip install entropix
```

## Quick start

```python
import pandas as pd
from entropix import shannon

s = pd.Series([10, 20, 15, 80, 90, 85, 88, 92])
shannon.compute(s)              # single entropy value
shannon.rolling(s, window=20)   # rolling entropy over time
shannon.delta(s, window=20)     # rate of change
shannon.plot(s, window=20)      # matplotlib Figure
```

## Measures

shannon · permutation · sample · approximate · spectral · differential · multiscale

Every measure shares the same API: `compute`, `rolling`, `delta`, `plot`
(`normalized` where a theoretical maximum exists). Series in → Series out
(index preserved); ndarray in → ndarray out.

| Method        | Returns                              |
| ------------- | ------------------------------------ |
| `compute`     | `float`                              |
| `rolling`     | Series/ndarray, same length          |
| `delta`       | Series/ndarray (first difference)    |
| `normalized`  | `float` in [0, 1] (where defined)    |
| `plot`        | `matplotlib.figure.Figure`           |

## Real-world example — food-trend analysis (NextOnMenu)

```python
import pandas as pd
from entropix import shannon

matcha_trends = pd.read_csv("matcha_trends.csv")["interest"]
shannon.plot(matcha_trends, window=20, title="Matcha — entropy over time")
# entropy drops before a trend goes mainstream
```

A sustained drop in rolling Shannon entropy means search interest is becoming
concentrated/structured rather than noisy — an early signal of a trend.

## License

MIT
