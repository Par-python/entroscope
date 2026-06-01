# Changelog

## 0.1.0 — 2026-06-01

Initial release.

- Seven entropy measures: shannon, permutation, sample, approximate, spectral,
  differential, multiscale.
- Consistent API across measures: `compute`, `rolling`, `delta`, `plot`, plus
  `normalized` (shannon/permutation/spectral) and `geographic` (shannon).
- Works on `pd.Series` (index preserved) and `np.ndarray`.
- Visualization helpers: `plot.compare`, `plot.dashboard`, `plot.drop_events`.
