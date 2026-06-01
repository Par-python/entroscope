# Changelog

## Unreleased

- CI/CD via GitHub Actions: lint (ruff check + format), test matrix across
  Python 3.9–3.14 with a 90% coverage gate, and a build + `twine check` job.
- Tag-triggered PyPI publish workflow using trusted publishing (inert until a
  PyPI publisher is configured).
- Applied `ruff format` across the codebase; added `ruff` to the `dev` extra.

## 0.1.0 — 2026-06-01

Initial release.

- Seven entropy measures: shannon, permutation, sample, approximate, spectral,
  differential, multiscale.
- Consistent API across measures: `compute`, `rolling`, `delta`, `plot`, plus
  `normalized` (shannon/permutation/spectral) and `geographic` (shannon).
- Works on `pd.Series` (index preserved) and `np.ndarray`.
- Visualization helpers: `plot.compare`, `plot.dashboard`, `plot.drop_events`.
