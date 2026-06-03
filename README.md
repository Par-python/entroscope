# entroscope

[![CI](https://img.shields.io/github/actions/workflow/status/Par-python/entroscope/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI&labelColor=24292e)](https://github.com/Par-python/entroscope/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/entroscope.svg?style=flat-square&logo=pypi&logoColor=white&labelColor=24292e&color=blue)](https://pypi.org/project/entroscope/)
[![Downloads](https://img.shields.io/pepy/dt/entroscope.svg?style=flat-square&logo=python&logoColor=white&labelColor=24292e&color=blue)](https://pepy.tech/project/entroscope)
[![Python](https://img.shields.io/pypi/pyversions/entroscope.svg?style=flat-square&logo=python&logoColor=white&labelColor=24292e&color=blue)](https://pypi.org/project/entroscope/)
[![Stars](https://img.shields.io/github/stars/Par-python/entroscope.svg?style=flat-square&logo=github&logoColor=white&labelColor=24292e&color=yellow)](https://github.com/Par-python/entroscope/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&logo=opensourceinitiative&logoColor=white&labelColor=24292e)](https://opensource.org/licenses/MIT)

**The definitive entropy toolkit for time series data.** Seven entropy measures,
one consistent interface, working directly on pandas Series and numpy arrays.

It started in [NextOnMenu](https://github.com/Par-python/nextonmenu): a falling
Shannon entropy of a food's regional search interest turned out to be an early
signal that it was about to trend. Computing it meant re-writing the same
histogram-and-log boilerplate every time. entroscope is that code, written once.

```bash
pip install entroscope
```

## Quick start

```python
import pandas as pd
from entroscope import shannon

s = pd.Series([10, 20, 15, 80, 90, 85, 88, 92])

shannon.compute(s)              # 0.73 (a single entropy value)
shannon.rolling(s, window=20)   # rolling entropy over time (a Series)
shannon.delta(s, window=20)     # rate of change of entropy
shannon.normalized(s)           # entropy scaled to [0, 1]
shannon.plot(s, window=20)      # a matplotlib Figure
```

Every method accepts a `pd.Series` **or** a `np.ndarray`. Pass a Series and you
get a Series back with its index preserved; pass an array and you get an array.

### Headless environments (Docker, CI)

`entroscope` does not change your matplotlib backend on import, so interactive
plotting in notebooks keeps working. In a headless environment (a Docker
container or CI runner) where you want a guaranteed non-interactive backend, set
the standard environment variable:

```bash
export MPLBACKEND=Agg        # or, in a Dockerfile:  ENV MPLBACKEND=Agg
```

## The seven measures

| Measure          | Import                    | Captures                                         |
| ---------------- | ------------------------- | ------------------------------------------------ |
| **Shannon**      | `entroscope.shannon`      | Uncertainty in a binned distribution             |
| **Permutation**  | `entroscope.permutation`  | Ordinal-pattern complexity (robust to noise)     |
| **Sample**       | `entroscope.sample`       | Regularity / predictability                      |
| **Approximate**  | `entroscope.approximate`  | Regularity (less noise-sensitive, faster)        |
| **Spectral**     | `entroscope.spectral`     | Spread of the power spectrum (frequency domain)  |
| **Differential** | `entroscope.differential` | Continuous entropy via a fitted distribution     |
| **Multiscale**   | `entroscope.multiscale`   | Sample entropy across coarse-grained time scales |

## One consistent API

Every measure exposes the same methods, so switching measures is a one-word change:

| Method                         | Input             | Returns                                               |
| ------------------------------ | ----------------- | ----------------------------------------------------- |
| `compute(x, **params)`         | Series or ndarray | `float`                                               |
| `rolling(x, window, **params)` | Series or ndarray | Series/ndarray, same length (NaN warm-up)             |
| `delta(x, window, **params)`   | Series or ndarray | Series/ndarray (first difference)                     |
| `normalized(x, **params)`      | Series or ndarray | `float` in [0, 1] (shannon/permutation/spectral only) |
| `plot(x, window, **params)`    | Series or ndarray | `matplotlib.figure.Figure`                            |

Shannon additionally provides `geographic(df, col=...)` for spatial distributions
(e.g. search interest by region). Multiscale provides `compute` and `plot`.

## Visualization

```python
from entroscope import plot

# overlay several measures on one axis
plot.compare(s, measures=["shannon", "permutation", "spectral"], window=20)

# a grid of every measure at once
plot.dashboard(s, window=20)

# highlight where entropy drops sharply (trend / regime-change detection)
plot.drop_events(s, measure="shannon", window=20, threshold=0.4)
```

All plot functions return a `matplotlib.figure.Figure` and never call
`plt.show()`, so they're safe in scripts, notebooks, and CI alike.

## Real-world examples

Runnable scripts live in [`examples/`](examples/); worked write-ups are in
[`docs/examples/`](docs/examples/):

- **[Food trends](docs/examples/food_trends.md)**: detect when search interest
  stops being random (the original NextOnMenu use case).
- **[Finance](docs/examples/finance.md)**: market uncertainty via permutation
  and spectral entropy.
- **[Medical](docs/examples/medical.md)**: HRV, EEG seizure onset, respiration,
  and continuous glucose.
- **[Business](docs/examples/business.md)**: sales demand, web-traffic anomalies,
  price volatility, and manufacturing QC.

```python
# food-trend analysis: entropy drops before a trend goes mainstream
import pandas as pd
from entroscope import shannon

matcha = pd.read_csv("matcha_trends.csv")["interest"]
shannon.plot(matcha, window=20, title="Matcha entropy over time")
```

A sustained drop in rolling entropy means a signal is becoming structured rather
than noisy, an early indicator of a forming pattern.

## Requirements

Python 3.9+, with numpy, pandas, scipy, and matplotlib (installed automatically).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the
test/lint commands, and how to add a new entropy measure.

## License

[MIT](LICENSE)
