# entropix

The definitive entropy toolkit for time series data. Seven measures, one
consistent interface, native pandas/numpy support.

- **[Quickstart](quickstart.md)** — install and first entropy in 60 seconds.
- **Examples** — [food trends](examples/food_trends.md),
  [finance](examples/finance.md), [biomedical](examples/biomedical.md).

## Measures at a glance

| Measure       | What it captures                                  |
| ------------- | ------------------------------------------------- |
| shannon       | Uncertainty in a binned distribution              |
| permutation   | Ordinal-pattern complexity (robust to noise)      |
| sample        | Regularity / predictability                       |
| approximate   | Regularity (less noise-sensitive, faster)         |
| spectral      | Spread of the power spectrum (frequency domain)   |
| differential  | Continuous entropy via a fitted distribution      |
| multiscale    | Sample entropy across coarse-grained time scales  |
