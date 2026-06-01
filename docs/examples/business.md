# Business & operational data

Four worked examples on raw business data. Runnable end-to-end in
[`examples/business.py`](https://github.com/entropix/entropix/blob/master/examples/business.py)
(uses synthetic data so it runs with no files — swap in `pd.read_csv(...)` for
your sales exports, server logs, price feeds, or sensor dumps).

## Daily sales / demand — rolling Shannon entropy

Rolling Shannon entropy **drops** as erratic launch-period demand consolidates
into a stable, repeating pattern.

```python
import pandas as pd
from entropix import shannon

sales = pd.read_csv("sales.csv")["units_sold"]
demand_entropy = shannon.rolling(sales, window=21, bins=10)
# erratic-launch mean ~3.02 -> settled-pattern mean ~2.34 : a drop
```

## Web traffic / server logs — spectral entropy

A healthy site has a strong daily cycle (low spectral entropy). When bot-driven
or anomalous bursts spread power across many frequencies, spectral entropy
**jumps**.

```python
from entropix import spectral

requests = pd.read_csv("access_log_hourly.csv")["requests"]
anomaly_score = spectral.rolling(requests, window=72, sf=24.0)  # 24 samples/day
# clean-cycle mean ~0.15 -> bot-anomaly mean ~1.82 : a large rise
```

## Price volatility — permutation entropy

Permutation entropy is a noise-robust gauge of how disordered price moves are.
It **rises** as a calm, trending market turns turbulent.

```python
from entropix import permutation

prices = pd.read_csv("prices.csv")["close"]
uncertainty = permutation.rolling(prices, window=50, order=3)
# calm-market mean ~2.32 -> turbulent-market mean ~2.42 (bounded measure: small move)
```

## Quality control — multiscale entropy

Multiscale entropy profiles complexity across time scales. When a manufacturing
process drifts out of control, its profile **diverges** from the in-control
baseline — often most visibly at coarser scales.

```python
from entropix import multiscale

readings = pd.read_csv("qc_sensor.csv")["dimension_mm"]
baseline = multiscale.compute(readings.iloc[:400], scales=range(1, 6))
current = multiscale.compute(readings.iloc[400:], scales=range(1, 6))
# at scale 5: in-control ~2.43 vs out-of-control ~3.09 — a clear divergence
```
