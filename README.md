<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Par-python/entroscope/master/docs/assets/banner-dark.png">
    <img alt="entroscope: the definitive entropy toolkit for time series data" src="https://raw.githubusercontent.com/Par-python/entroscope/master/docs/assets/banner-light.png" width="100%">
  </picture>
</p>

<p align="center">
  <b>Entropy for time series, over time.</b><br>
  Nine entropy measures behind one API, rolling windows on pandas, polars and numpy,<br>
  and every result checked against antropy, EntropyHub and scipy.
</p>

<p align="center">
  <a href="https://pypi.org/project/entroscope/"><img alt="PyPI" src="https://img.shields.io/pypi/v/entroscope.svg?style=flat-square&logo=pypi&logoColor=white&labelColor=24292e&color=blue"></a>
  <a href="https://pepy.tech/project/entroscope"><img alt="Downloads" src="https://img.shields.io/pepy/dt/entroscope.svg?style=flat-square&logo=python&logoColor=white&labelColor=24292e&color=blue"></a>
  <a href="https://pypi.org/project/entroscope/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/entroscope.svg?style=flat-square&logo=python&logoColor=white&labelColor=24292e&color=blue"></a>
  <a href="https://github.com/Par-python/entroscope/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Par-python/entroscope/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI&labelColor=24292e"></a>
  <a href="https://par-python.github.io/entroscope/"><img alt="Docs" src="https://img.shields.io/badge/docs-site-blue?style=flat-square&logo=materialformkdocs&logoColor=white&labelColor=24292e"></a>
  <a href="https://github.com/Par-python/entroscope/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Par-python/entroscope.svg?style=flat-square&logo=github&logoColor=white&labelColor=24292e&color=yellow"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&labelColor=24292e"></a>
</p>

Most entropy libraries hand you one number per array. entroscope is built for
the question that comes next: **how is the entropy of my series changing, and
when did it change?** Every measure computes on a single window, rolls across a
whole series, and plots, with the same call shape. pandas and polars Series keep
their index or name.

## Installation

```bash
pip install entroscope
```

| Extra | Adds |
| --- | --- |
| `pip install "entroscope[sklearn]"` | `EntropyFeatures`, a scikit-learn transformer |
| `pip install "entroscope[polars]"` | polars Series input and output |

Requires Python 3.9+. numpy, pandas, scipy and matplotlib install automatically.

## Quickstart

A series that is pure noise for 200 steps, then turns into a clean 20-step cycle:

```python
import numpy as np
import pandas as pd
from entroscope import shannon, spectral

rng = np.random.default_rng(0)
t = np.arange(400)
noise = rng.normal(size=400)
s = pd.Series(np.where(t < 200, noise, np.sin(2 * np.pi * t / 20) + 0.2 * noise))

spectral.compute(s[:200])              # -> 6.05 bits: power spread over every frequency
spectral.compute(s[200:])              # -> 0.88 bits: power concentrated in one
spectral.normalized(s[200:])           # -> 0.13 on a 0-1 scale

roll = spectral.rolling(s, window=40)  # -> pd.Series, same index, NaN for the first 39 steps
roll[100], roll[300]                   # -> (3.64, 0.70): the drop marks the change
spectral.delta(s, window=40)           # -> step-to-step change in the rolling entropy
fig = spectral.plot(s, window=40)      # -> matplotlib Figure (never calls plt.show())

shannon.compute(s[:200]), shannon.compute(s[200:])  # -> (3.08, 3.23): can't tell them apart
```

The last line is why there are nine measures. Shannon entropy only sees the spread
of values, not their order, so it misses the rhythm that spectral entropy picks up
immediately. [The nine measures](#the-nine-measures) says which to reach for.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Par-python/entroscope/master/docs/assets/entropy-drop-dark.png">
    <img alt="Two stacked charts. Top: a synthetic signal that is random noise until step 180, then a regular cycle. Bottom: its rolling spectral entropy, high during the noise and falling sharply shortly after step 180." src="https://raw.githubusercontent.com/Par-python/entroscope/master/docs/assets/entropy-drop-light.png" width="90%">
  </picture>
</p>

### Why trust the numbers

Every measure is checked against an independent implementation on every CI run,
on four kinds of signal (white noise, a noisy sine, an AR(1) process and the chaotic
logistic map):

| Measure | Checked against | Agreement |
| --- | --- | --- |
| sample, approximate, permutation | antropy and EntropyHub | exact (relative error < 1e-10) |
| spectral | antropy | exact |
| multiscale | EntropyHub `MSEn` | exact |
| shannon, differential (normal) | scipy | exact |
| transfer | bivariate-Gaussian closed form, Kraskov et al. (2004) | within 0.05 bits |

Cross-checking also found two places where entroscope had drifted from the
published definitions (sample entropy and multiscale entropy); both are fixed.
Details are on the [validation page](https://par-python.github.io/entroscope/validation/),
and you can rerun the checks with `pip install -e ".[dev,reference]"` then
`pytest tests/test_reference.py`.

## Watching entropy over time

```python
from entroscope import plot

fig = plot.compare(s, measures=["shannon", "permutation", "spectral"], window=40)
fig = plot.dashboard(s, window=40)  # one panel per measure
fig = plot.drop_events(s, measure="spectral", window=40, threshold=0.5)  # marks sharp drops
```

`rolling` is causal: the value at step t uses only steps t−window+1 through t, so
it is safe to use for live monitoring. Missing data never turns into a made-up
number:

```python
gappy = s.copy()
gappy[250] = np.nan

spectral.compute(gappy)                          # -> nan
spectral.rolling(gappy, window=40).isna().sum()  # -> 79: the 39 warm-up steps + the 40 windows holding the gap
```

## Two signals: direction and drift

```python
from entroscope import divergence, transfer

rng = np.random.default_rng(1)
driver = rng.normal(size=1000)
follower = 0.8 * np.roll(driver, 1) + 0.6 * rng.normal(size=1000)  # follows driver, one step behind

transfer.compute(driver, follower)  # -> 0.73 bits: driver's past predicts follower
transfer.compute(follower, driver)  # -> 0.02 bits: not the other way round

train = rng.normal(0, 1, 5000)
live = rng.normal(0.5, 1.2, 5000)
divergence.js(train, live)          # -> 0.04 bits (0 = identical, 1 = no overlap)
divergence.kl(live, train)          # -> 0.23 bits (directional)
```

## Machine learning and polars

`EntropyFeatures` turns time-series windows into entropy features inside a
scikit-learn pipeline. Each row of `X` is one window:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from entroscope.features import EntropyFeatures

rng = np.random.default_rng(2)
t = np.arange(128)
cycles = [np.sin(2 * np.pi * t / rng.uniform(8, 32)) + rng.normal(size=128) for _ in range(100)]
noise = [rng.normal(size=128) for _ in range(100)]
X, y = np.vstack(cycles + noise), np.repeat([0, 1], 100)

model = make_pipeline(EntropyFeatures(measures=("spectral", "permutation")), LogisticRegression())
cross_val_score(model, X, y, cv=5).mean()  # -> 0.945 from two features per window
```

Pass a polars Series anywhere a pandas Series works:

```python
import polars as pl

spectral.rolling(pl.Series("sensor", s.to_numpy()), window=40)  # -> polars Series named "sensor"
```

## The nine measures

| Measure | Captures | Reach for it when |
| --- | --- | --- |
| `shannon` | Spread of values in a histogram | Order doesn't matter, e.g. how evenly interest spreads across regions (`shannon.geographic`) |
| `permutation` | Complexity of ordinal patterns | Noisy data where the order of ups and downs matters; robust to scale and outliers |
| `spectral` | Spread of power across frequencies | Detecting rhythms and cycles appearing or disappearing |
| `sample` | Regularity: do repeating patterns keep repeating? | Short-to-medium series where predictability is the question |
| `approximate` | Regularity, like sample, with a gentler bias | You want sample-style regularity on shorter series |
| `differential` | Entropy of a continuous distribution (normal fit or KDE) | Continuous data without natural bins |
| `multiscale` | Sample entropy across coarse-grained time scales | Structure that only shows up at some resolutions |
| `transfer` | Directional information flow X → Y | Lead–lag questions: which series drives which |
| `divergence` | KL and Jensen-Shannon distance between two samples | Data drift, e.g. training vs production distributions |

## One API

| Method | Input | Returns |
| --- | --- | --- |
| `compute(x, **params)` | pandas, polars or numpy | `float` |
| `rolling(x, window, **params)` | pandas, polars or numpy | same type and length as the input, NaN during warm-up |
| `delta(x, window, **params)` | pandas, polars or numpy | first difference of `rolling` |
| `normalized(x, **params)` | pandas, polars or numpy | `float` in [0, 1] (shannon, permutation, spectral) |
| `plot(x, window, **params)` | pandas, polars or numpy | `matplotlib.figure.Figure` |

`transfer` takes two series (`x`, `y`) and has the same four methods except
`normalized`. `divergence` takes two samples and provides `kl`, `js` and `plot`.
`multiscale` provides `compute` and `plot`.

## Honest limits

- **Entropy is not always the best detector.** In a
  [60-run study](https://github.com/Par-python/training-early-warning) of neural
  networks diverging during training, a one-line gradient-norm spike rule warned
  on every run with no false alarms and beat both entropy detectors. Compare
  against a simple baseline before trusting entropy for a new job.
- **`rolling` recomputes every window in Python.** On 3,000 points, shannon,
  permutation and spectral take about 0.1–0.4 s; sample entropy with a 100-step
  window takes about 3 s, and sample and approximate grow with the square of the
  window. Very long series or wide windows need patience.
- **Sample entropy can be undefined.** With no matching templates (short series or
  a tiny `r`), `sample.compute` returns a finite ceiling,
  `ln((n−m)(n−m−1))`, instead of infinity. Check `r` if you see identical high values.
- **Spectral entropy follows antropy's periodogram definition.** EntropyHub's
  `SpecEn` uses a different estimator, so its numbers differ by design.
- **Transfer entropy needs data.** Below about 30 samples it warns, and k-NN
  estimates on short windows are noisy.

## Resources

- **Docs:** https://par-python.github.io/entroscope/ ([quickstart](https://par-python.github.io/entroscope/quickstart/), [validation](https://par-python.github.io/entroscope/validation/), [integrations](https://par-python.github.io/entroscope/integrations/))
- **Worked examples:** [food trends](docs/examples/food_trends.md), [finance](docs/examples/finance.md),
  [business](docs/examples/business.md), [medical](docs/examples/medical.md); runnable scripts in [`examples/`](examples/)
- **Case study:** [Can entropy warn that training is about to diverge?](https://github.com/Par-python/training-early-warning)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md) · **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Headless use (Docker, CI):** entroscope never changes your matplotlib backend;
  set `MPLBACKEND=Agg` in the environment if you need a non-interactive one.

entroscope started in [NextOnMenu](https://github.com/Par-python/nextonmenu), where a
falling Shannon entropy of a food's regional search interest turned out to be an
early sign it was about to trend.

## License

[MIT](LICENSE)
