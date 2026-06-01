# Changelog

## 0.1.1 — 2026-06-01

- Rewrote the README: clearer intro, working PyPI/CI/Python/license badges, and
  links to the example write-ups.
- Added PyPI trove classifiers (supported Python versions, license, topics) so
  the Python-versions badge and PyPI sidebar populate correctly.
- Runnable example scripts in `examples/`: four medical/biomedical scenarios
  (HRV/sample, EEG/permutation+spectral, respiration/approximate,
  glucose/differential) and four business scenarios (sales/shannon,
  web-traffic/spectral, prices/permutation, QC/multiscale), plus matching
  `docs/examples/medical.md` and `docs/examples/business.md` pages.
- Smoke tests (`tests/test_examples.py`) run the example scripts in CI so they
  can't silently break; CI lint now covers `examples/` too.
- CI/CD via GitHub Actions: lint (ruff check + format), test matrix across
  Python 3.9 to 3.14 with a 90% coverage gate, and a build + `twine check` job.
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
